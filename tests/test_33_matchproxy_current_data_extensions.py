"""Tests for the post-primary current-data extension audits."""

from __future__ import annotations

from math import log
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


def _frame(players: int = 16, dates_per_player: int = 10) -> pd.DataFrame:
    rows = []
    burdens = [0.0, 45.0, 60.0, 90.0, 110.0, 135.0, 150.0, 180.0, 75.0, 120.0]
    recoveries = ["0-3 days", "4-5 days", "6-7 days", "0-3 days", "4-5 days"]
    for player in range(1, players + 1):
        for index in range(dates_per_player):
            burden = burdens[index % len(burdens)]
            event = int((player + 2 * index) % 5 == 0 or (player + index) % 11 == 0)
            is_higher = (player + index) % 2 == 0
            if event and (player + index) % 4 == 0:
                injury_type = "unknown"
            elif event and (player + index) % 3 == 0:
                injury_type = "joint/ligament"
            else:
                injury_type = "muscle/tendon"
            role = "starting_lineup" if index % 3 else "substitute_list"
            rows.append(
                {
                    "tm_player_id": player,
                    "date": pd.Timestamp("2022-08-01") + pd.Timedelta(days=index * 3 + player),
                    "model_group": "fragile" if is_higher else "regular",
                    "fragility_group": "fragile" if is_higher else "regular",
                    "all_minutes_last_7d": burden,
                    "all_minutes_played": 90.0 if role == "starting_lineup" else 35.0,
                    "log_minutes_played": log(90.0 if role == "starting_lineup" else 35.0),
                    "injury_event_matchproxy": event,
                    "injury_event_matchproxy_muscle_tendon": int(event and injury_type == "muscle/tendon"),
                    "matchproxy_public_injury_type": injury_type,
                    "matchproxy_duration_days": float(7 + (index % 4) * 14) if event else np.nan,
                    "injury_event_matchproxy_same_day": int(event and index % 2 == 0),
                    "injury_event_matchproxy_lag1": int(event and index % 2 == 1),
                    "lineup_role_model": role,
                    "lineup_role": role,
                    "returned_from_recorded_injury_within_14d": bool(index == 3),
                    "week_phase_sin": float(np.sin(index)),
                    "week_phase_cos": float(np.cos(index)),
                    "halfweek_phase_sin": float(np.sin(index / 2)),
                    "halfweek_phase_cos": float(np.cos(index / 2)),
                    "recovery_interval_refined": recoveries[index % len(recoveries)],
                    "all_games_last_7d": 1 if index % 2 else 2,
                    "match_cluster_id": f"match_{index}",
                    "competition_context": "Premier League" if index % 2 else "domestic cup",
                }
            )
    return pd.DataFrame(rows)


class _FakePrimary:
    def run_prediction_bundle(self, frame, event_col, **kwargs):
        selected = pd.DataFrame(
            {
                "fragility_group": ["regular", "fragile"],
                "all_minutes_last_7d": [0.0, 0.0],
                "pred_events_per_10000_min": [1.0, 2.0],
            }
        )
        ratios = pd.DataFrame(
            {"all_minutes_last_7d": [0.0], "rate_ratio": [2.0], "rr_ci_low": [1.0], "rr_ci_high": [4.0]}
        )
        effects = pd.DataFrame(
            {
                "contrast_id": [
                    "global_spline_by_history_interaction",
                    "intermediate_history_180_vs_0",
                    "higher_history_180_vs_0",
                    "ratio_of_180_vs_0_changes",
                    "intermediate_history_180_vs_90",
                    "higher_history_180_vs_90",
                    "ratio_of_180_vs_90_changes",
                ],
                "test_statistic": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "df": [4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "p_value": [0.5] * 7,
                "estimate": [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "ci_low": [np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "ci_high": [np.nan, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            }
        )
        return {
            "selected": selected,
            "ratios": ratios,
            "effect_modification": effects,
            "predictions": selected,
            "dispersion": 1.0,
            "model_family": "poisson",
            "denominator": "observed_minutes",
            "estimator": "fake",
        }

    def summary_row(self, label, event_col, group_col, controls, frame, bundle):
        return {"model": label, "n_match_rows": len(frame), "n_events": int(frame[event_col].sum())}

    def spline_curve_shape_summary(self, predictions):
        return pd.DataFrame({"history_stratum": ["regular"], "max_minutes_last_7d": [45.0]})

    def label_effect_modification_rows(self, effects, *args, **kwargs):
        return effects.copy()


def test_extension_imports_and_basics(load_src_module, monkeypatch, tmp_path):
    module = load_src_module("33_matchproxy_current_data_extensions.py")
    assert module.BOOTSTRAP_REPLICATES == 1000
    assert module.IPW_BOOTSTRAP_REPLICATES == 200
    assert module.LINEUP_ROLES == ("starting_lineup", "substitute_list")
    assert module.BURDEN_LABELS[-1] == ">220"
    assert module._z_interval(0.0, 0.1)[0] == pytest.approx(1.0)
    assert all(np.isnan(value) for value in module._z_interval(np.nan, 0.1))
    assert module._normal_p_value(0.0, 0.1) == pytest.approx(1.0)
    assert np.isnan(module._normal_p_value(np.nan, 0.1))
    assert module._normal_p_value(1.0, 0.0) == 0.0
    with pytest.raises(KeyError, match="missing columns"):
        module._require_columns(pd.DataFrame(), ["x"], "test")
    adjusted = module.primary_p_adjust(pd.Series([0.01, np.nan, 0.03]))
    assert adjusted["holm"][0] >= 0.01
    assert np.isnan(adjusted["bh"][1])
    assert np.isnan(module.primary_p_adjust(pd.Series([np.nan]))["holm"][0])
    table = module.primary_adjusted_table(pd.DataFrame({"p_value": [0.1]}), "x")
    assert table.loc[0, "analysis_family"] == "x"
    src_dir = str(Path(module.__file__).resolve().parent)
    while src_dir in sys.path:
        sys.path.remove(src_dir)
    primary = module.load_primary_module()
    assert hasattr(primary, "run_prediction_bundle")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "missing_extension.py"))
    with pytest.raises(FileNotFoundError):
        module.load_primary_module()
    monkeypatch.undo()
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *args, **kwargs: None)
    with pytest.raises(ImportError):
        module.load_primary_module()


def test_prepare_extension_and_lineup_refits_with_fake_primary(load_src_module):
    module = load_src_module("33_matchproxy_current_data_extensions.py")
    frame = _frame()

    class PrepPrimary:
        @staticmethod
        def prepare_model_frame(panel, event_col, group_col):
            return panel.copy()

        @staticmethod
        def add_recent_prior_injury_return_flags(panel, injuries):
            return panel.copy()

        @staticmethod
        def add_lineup_start_status(panel, lineups):
            return panel.copy()

    prepared = module.prepare_extension_frame(PrepPrimary(), frame, pd.DataFrame(), None)
    assert prepared["lineup_role_model"].isin(module.LINEUP_ROLES).all()
    outputs = module.lineup_refit_outputs(_FakePrimary(), prepared)
    assert set(outputs) == {"summary", "predictions", "shape", "effects"}
    assert "starters_only" in outputs["summary"]["model"].tolist()
    no_events = prepared.copy()
    no_events["injury_event_matchproxy"] = 0
    failed = module.lineup_refit_outputs(_FakePrimary(), no_events)
    assert failed["summary"]["fit_status"].eq("not_estimable").all()


def test_direct_lineup_interaction_audit(load_src_module, monkeypatch):
    module = load_src_module("33_matchproxy_current_data_extensions.py")
    primary = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    frame = _frame(players=30)
    results = module.lineup_spline_interaction_tests(primary, frame)
    assert set(results["model"]) == {
        "pooled_history_adjusted",
        "regular_history_only",
        "fragile_history_only",
    }
    assert results["fit_status"].eq("ok").all()
    unavailable = module.lineup_spline_interaction_tests(
        primary,
        frame.assign(lineup_role_model="starting_lineup"),
    )
    assert unavailable["fit_status"].eq("not_estimable").all()

    class NoInteractionResult:
        params = pd.Series([0.0], index=["Intercept"])

    class NoInteractionFit:
        def fit(self, **kwargs):
            return NoInteractionResult()

    monkeypatch.setattr(module.smf, "glm", lambda *args, **kwargs: NoInteractionFit())
    no_interaction = module.lineup_spline_interaction_tests(primary, frame)
    assert no_interaction["fit_status"].eq("not_estimable").all()


def test_reporting_completeness_models_and_weighted_sensitivity(load_src_module):
    module = load_src_module("33_matchproxy_current_data_extensions.py")
    primary = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    frame = _frame(players=20)
    events = module.classify_reporting_completeness(frame)
    assert len(events) == int(frame["injury_event_matchproxy"].sum())
    assert set(events["proxy_timing"]) == {"same_day", "lag1"}
    context = module.reporting_completeness_by_context(events)
    assert {"overall", "history", "lineup", "timing", "history_by_timing"}.issubset(context["context"])
    with pytest.raises(KeyError):
        module.classify_reporting_completeness(pd.DataFrame())
    result, terms, weighted = module.reporting_type_model(events)
    assert result is not None
    assert terms["effect_measure"].eq("odds_ratio_type_classifiable").all()
    assert weighted["type_reporting_ipw"].ge(0).all()
    with pytest.raises(ValueError, match="requires both"):
        module.reporting_type_model(events.assign(report_type_classifiable=True))
    _, fixed_terms, _ = module.reporting_type_model(
        events.assign(returned_from_recorded_injury_within_14d=False)
    )
    assert "recent_return_indicator" not in fixed_terms["term"].tolist()
    unstable = weighted.copy()
    unstable["report_type_classifiable"] = True
    unstable["predicted_type_classifiable_probability"] = 0.05
    unstable["type_reporting_ipw"] = 20.0
    assert (
        module.reporting_type_ipw_diagnostics(unstable).loc[0, "stability_status"]
        == "unstable_positivity_or_weight_tail"
    )
    outputs = module.reporting_type_ipw_sensitivity(
        primary,
        frame,
        weighted,
        bootstrap_replicates=8,
        seed=4,
    )
    assert not outputs["summary"].empty
    assert outputs["summary"].loc[0, "event_col"] == "injury_event_matchproxy_muscle_tendon"
    assert outputs["selected"]["pred_events_per_10000_min_ci_low"].notna().all()
    assert outputs["ratios"]["rr_ci_high"].notna().all()
    assert 2 <= outputs["bootstrap_samples"]["replicate"].nunique() <= 8
    assert outputs["diagnostics"].loc[0, "stability_status"] in {
        "stable",
        "unstable_positivity_or_weight_tail",
    }
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        module.run_weighted_prediction_bundle(
            primary,
            frame.assign(type_reporting_ipw=1.0),
            "injury_event_matchproxy_muscle_tendon",
            "type_reporting_ipw",
            bootstrap_replicates=1,
        )
    with pytest.raises(ValueError, match="No events"):
        module.run_weighted_prediction_bundle(
            primary,
            frame.assign(type_reporting_ipw=1.0, injury_event_matchproxy_muscle_tendon=0),
            "injury_event_matchproxy_muscle_tendon",
            "type_reporting_ipw",
            bootstrap_replicates=2,
        )
    with pytest.raises(ValueError, match="regular and fragile"):
        module.run_weighted_prediction_bundle(
            primary,
            frame.assign(type_reporting_ipw=1.0, model_group="regular"),
            "injury_event_matchproxy_muscle_tendon",
            "type_reporting_ipw",
            bootstrap_replicates=2,
        )
    no_muscle = frame.copy()
    no_muscle["injury_event_matchproxy_muscle_tendon"] = 0
    empty = module.reporting_type_ipw_sensitivity(primary, no_muscle, weighted, bootstrap_replicates=4)
    assert empty["summary"].empty


def test_weighted_bootstrap_failure_paths(load_src_module, monkeypatch):
    module = load_src_module("33_matchproxy_current_data_extensions.py")
    primary = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    frame = _frame(players=20)
    event_col = "injury_event_matchproxy_muscle_tendon"
    result = module.smf.glm(
        primary.spline_formula(event_col, float(frame["all_minutes_last_7d"].max())),
        data=frame,
        family=module.sm.families.Poisson(),
        offset=frame["log_minutes_played"],
    ).fit()
    _, _, zero_samples = module.weighted_player_bootstrap_predictions(
        primary,
        result,
        frame.assign(ipw=0.0),
        event_col,
        "ipw",
        replicates=2,
        seed=1,
    )
    assert zero_samples.empty

    class NonFiniteFit:
        def __init__(self, value):
            self.params = np.full(len(result.params), value)

        def fit(self):
            return self

    with monkeypatch.context() as patch:
        patch.setattr(module.sm, "GLM", lambda *args, **kwargs: NonFiniteFit(np.nan))
        _, _, failed_samples = module.weighted_player_bootstrap_predictions(
            primary,
            result,
            frame.assign(ipw=1.0),
            event_col,
            "ipw",
            replicates=2,
            seed=2,
        )
    assert failed_samples.empty
    with monkeypatch.context() as patch:
        patch.setattr(module.sm, "GLM", lambda *args, **kwargs: NonFiniteFit(1000.0))
        _, _, overflow_samples = module.weighted_player_bootstrap_predictions(
            primary,
            result,
            frame.assign(ipw=1.0),
            event_col,
            "ipw",
            replicates=2,
            seed=3,
        )
    assert overflow_samples.empty


def test_reported_absence_burden_and_bootstrap(load_src_module):
    module = load_src_module("33_matchproxy_current_data_extensions.py")
    frame = _frame()
    burden = module.reported_absence_day_burden_frame(frame)
    assert burden["reported_absence_days_proxy"].ge(0).all()
    assert burden["burden_bin"].notna().all()
    rates = module.cluster_bootstrap_rate_intervals(
        burden,
        ["model_group"],
        "reported_absence_days_proxy",
        "all_minutes_played",
        replicates=8,
        seed=1,
    )
    assert rates["bootstrap_successful_replicates"].eq(8).all()
    assert rates["rate_ci_low"].notna().all()
    by_burden = module.cluster_bootstrap_rate_intervals(
        burden,
        ["model_group", "burden_bin"],
        "reported_absence_days_proxy",
        "all_minutes_played",
        replicates=4,
        seed=2,
    )
    assert not by_burden.empty
    assert module.cluster_bootstrap_rate_intervals(
        burden.iloc[0:0],
        ["model_group"],
        "reported_absence_days_proxy",
        "all_minutes_played",
        replicates=4,
    ).empty
    with pytest.raises(ValueError, match="at least 2"):
        module.cluster_bootstrap_rate_intervals(
            burden, ["model_group"], "reported_absence_days_proxy", "all_minutes_played", replicates=1
        )
    one_player = burden[burden["tm_player_id"].eq(1)]
    with pytest.raises(ValueError, match="At least two players"):
        module.cluster_bootstrap_rate_intervals(
            one_player, ["model_group"], "reported_absence_days_proxy", "all_minutes_played", replicates=4
        )
    duration = module.conditional_reported_duration_model(burden)
    assert duration["analysis_family"].eq("reported_duration").all()
    empty = module.conditional_reported_duration_model(burden.assign(injury_event_matchproxy=0))
    assert empty.empty


def test_joint_support_schedule_model_and_named_intervals(load_src_module, monkeypatch):
    module = load_src_module("33_matchproxy_current_data_extensions.py")
    primary = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    frame = _frame(players=28)
    support = module.joint_burden_recovery_support(primary, frame)
    assert {"supported_for_joint_model", "sparse_or_structurally_unavailable"}.intersection(support["support_status"])
    assert support["events_per_1000_match_hours_ci_high"].notna().all()
    schedule = module.supported_schedule_compression_model(frame)
    assert set(schedule["contrast"]) == {
        "intermediate_history_two_vs_one",
        "higher_history_two_vs_one",
        "difference_in_two_vs_one_changes",
    }
    assert schedule["analysis_family"].eq("joint_schedule_compression").all()
    no_events = module.supported_schedule_compression_model(frame.assign(injury_event_matchproxy=0))
    assert no_events.loc[0, "fit_status"] == "not_estimable"
    params = pd.Series([0.1, 0.2], index=["a", "b"])
    interval = module.named_coefficient_interval(params, np.eye(2) * 0.01, {"a": 1.0, "b": -1.0})
    assert interval["rate_ratio"] < 1.0
    with pytest.raises(KeyError):
        module.named_coefficient_interval(params, np.eye(2), {"missing": 1.0})

    class MissingTermsResult:
        params = pd.Series([0.0], index=["Intercept"])

        @staticmethod
        def cov_params():
            return np.eye(1)

    class MissingTermsFit:
        @staticmethod
        def fit(**kwargs):
            return MissingTermsResult()

    monkeypatch.setattr(module.smf, "glm", lambda *args, **kwargs: MissingTermsFit())
    missing_terms = module.supported_schedule_compression_model(frame)
    assert missing_terms["fit_status"].eq("not_estimable").all()


def test_case_crossover_and_curve_bootstrap(load_src_module, monkeypatch):
    module = load_src_module("33_matchproxy_current_data_extensions.py")
    primary = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    frame = _frame(players=24)
    crossover = module.prepare_case_crossover_frame(primary, frame)
    assert crossover["player_season_stratum"].nunique() > 1
    results = module.fit_case_crossover_models(crossover)
    assert {"previous_7day_minutes", "recovery_shortness"}.issubset(results["model"])
    assert results["analysis_family"].eq("within_player_case_crossover").all()
    assert module.fit_case_crossover_models(crossover.iloc[0:0]).empty
    no_recovery = module.fit_case_crossover_models(
        crossover.assign(recovery_interval_refined="8-14 days")
    )
    assert set(no_recovery["model"]) == {"previous_7day_minutes"}
    no_variation = module.fit_case_crossover_models(crossover.assign(injury_event_matchproxy=0))
    assert no_variation["fit_status"].eq("not_estimable").all()

    class FailingConditionalLogit:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, **kwargs):
            raise ValueError("synthetic non-convergence")

    monkeypatch.setattr(module, "ConditionalLogit", FailingConditionalLogit)
    failed_crossover = module.fit_case_crossover_models(crossover)
    assert failed_crossover["fit_status"].eq("failed_convergence").all()
    monkeypatch.undo()
    samples, summary = module.curve_feature_bootstrap(primary, frame, replicates=4, seed=3)
    assert len(samples) == 8
    assert summary["successful_replicates"].le(4).all()
    with pytest.raises(ValueError, match="replicates"):
        module.curve_feature_bootstrap(primary, frame, replicates=1)
    with pytest.raises(ValueError, match="At least two players"):
        module.curve_feature_bootstrap(primary, frame[frame["tm_player_id"].eq(1)], replicates=3)

    class FailingGLM:
        def __init__(self, *args, **kwargs):
            raise ValueError("synthetic bootstrap failure")

    monkeypatch.setattr(module.sm, "GLM", FailingGLM)
    failed_samples, failed_summary = module.curve_feature_bootstrap(primary, frame, replicates=3, seed=5)
    assert failed_samples["fit_status"].eq("failed").all()
    assert failed_summary["successful_replicates"].eq(0).all()


def test_current_match_metadata_context_and_two_way_cluster(load_src_module, tmp_path, monkeypatch):
    module = load_src_module("33_matchproxy_current_data_extensions.py")
    primary = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    frame = _frame(players=18)
    tm = Path(tmp_path)
    games = pd.DataFrame(
        {
            "game_id": [f"match_{index}" for index in range(10)],
            "competition_id": ["GB1" if index % 2 else "FAC" for index in range(10)],
            "season": [2022] * 10,
            "home_club_id": [10] * 10,
            "away_club_id": [20] * 10,
        }
    )
    games.to_csv(tm / "games.csv", index=False)
    pd.DataFrame({"competition_id": ["GB1", "FAC"], "type": ["domestic_league", "domestic_cup"]}).to_csv(
        tm / "competitions.csv", index=False
    )
    appearances = frame[["tm_player_id", "date", "all_minutes_played"]].copy()
    appearances["game_id"] = appearances["date"].map(
        {pd.Timestamp("2022-08-01") + pd.Timedelta(days=index * 3 + 1): f"match_{index}" for index in range(10)}
    ).fillna("match_0")
    appearances["player_club_id"] = 10
    appearances = appearances.rename(columns={"tm_player_id": "player_id", "all_minutes_played": "minutes_played"})
    unmatched = appearances.iloc[[1]].copy()
    appearances = appearances.drop(index=1).reset_index(drop=True)
    multiple = appearances.iloc[[0]].copy()
    multiple["game_id"] = "match_1"
    unmatched["game_id"] = np.nan
    appearances = pd.concat([appearances, multiple, unmatched], ignore_index=True)
    appearances[["game_id", "player_id", "player_club_id", "date", "minutes_played"]].to_csv(
        tm / "appearances.csv", index=False
    )
    metadata, audit = module.attach_current_match_metadata(primary, frame, tm)
    assert metadata["match_cluster_id"].notna().all()
    assert {"multiple_source_matches", "unmatched_source_match"}.issubset(metadata["match_link_status"])
    assert audit["match_rows"].sum() == len(frame)
    rates, refits = module.competition_context_outputs(primary, metadata)
    assert not rates.empty
    assert set(refits["model"]) == {"competition_context_adjusted", "premier_league_current_match_only"}
    two_way = module.two_way_cluster_sensitivity(primary, metadata)
    assert "global_spline_by_history_interaction" in two_way["contrast"].tolist()
    with monkeypatch.context() as patch:
        patch.setattr(
            primary,
            "spline_formula",
            lambda event_col, burden_max: f"{event_col} ~ all_minutes_last_7d + C(model_group)",
        )
        no_interaction = module.two_way_cluster_sensitivity(primary, metadata)
    assert (
        no_interaction.loc[
            no_interaction["contrast"].eq("global_spline_by_history_interaction"), "fit_status"
        ].iloc[0]
        == "not_estimable"
    )
    assert module.two_way_cluster_sensitivity(primary, metadata.assign(injury_event_matchproxy=0)).empty
    no_context_rates, no_context_refits = module.competition_context_outputs(
        primary,
        metadata.assign(injury_event_matchproxy=0),
    )
    assert not no_context_rates.empty
    assert no_context_refits["fit_status"].eq("not_estimable").all()
