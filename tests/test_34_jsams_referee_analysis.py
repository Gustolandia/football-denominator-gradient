"""Tests for the JSAMS reviewer-requested analysis layer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _frame(players: int = 40, rows_per_player: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(91)
    rows = []
    burdens = np.arange(rows_per_player, dtype=float) * 30.0
    for player in range(players):
        history = float((player % 9) / 2.0)
        for index, burden in enumerate(burdens):
            same_probability = 0.10 + 0.006 * (burden / 90.0) + 0.006 * history
            lag_probability = 0.14 + 0.004 * (burden / 90.0) + 0.004 * history
            same_day = int(rng.random() < min(same_probability, 0.3))
            lag1 = int(rng.random() < min(lag_probability, 0.3))
            rows.append(
                {
                    "tm_player_id": player,
                    "date": pd.Timestamp("2020-01-01")
                    + pd.Timedelta(days=7 * index + player % 4),
                    "all_minutes_last_7d": burden,
                    "all_minutes_played": float((30, 60, 90)[index % 3]),
                    "prior_injuries_per_10000min": history,
                    "prior_minutes_played": 1000.0 + index * 90.0,
                    "prior_n_spells": float(player % 5),
                    "prior_max_spell_duration_days": float(player % 30),
                    "fragility_group": ("tough", "regular", "fragile")[player % 3],
                    "injury_event_matchproxy_same_day": same_day,
                    "injury_event_matchproxy_lag1": lag1,
                    "injury_event_matchproxy": int(same_day or lag1),
                    "same_day_reported_absence_ge28d": same_day,
                    "same_day_muscle_tendon_report": same_day,
                    "week_phase_sin": np.sin(index),
                    "week_phase_cos": np.cos(index),
                    "halfweek_phase_sin": np.sin(index / 2),
                    "halfweek_phase_cos": np.cos(index / 2),
                    "lineup_role": ("starting_lineup", "substitute_list")[index % 2],
                    "lineup_role_model": ("starting_lineup", "substitute_list")[index % 2],
                    "returned_from_recorded_injury_within_14d": int(index % 7 == 0),
                    "matchproxy_injury_desc": "strain" if same_day or lag1 else "",
                    "available_for_injury_risk": True,
                    "age_years": 20.0 + player % 15,
                    "position_group": ("Attack", "Midfield", "Defender")[player % 3],
                    "club_season": f"{2020 + index % 3}_{player % 5}",
                    "competition_context": (
                        "Premier League" if index % 2 == 0 else "domestic cup"
                    ),
                    "match_cluster_id": f"match_{index}_{player // 2}",
                }
            )
    frame = pd.DataFrame(rows)
    # Guarantee events for every model on small platforms.
    frame.loc[0, "injury_event_matchproxy_same_day"] = 1
    frame.loc[1, "injury_event_matchproxy_lag1"] = 1
    frame["injury_event_matchproxy"] = frame[
        ["injury_event_matchproxy_same_day", "injury_event_matchproxy_lag1"]
    ].max(axis=1)
    return frame


class _PrepPrimary:
    @staticmethod
    def prepare_model_frame(panel, event_col, group_col, include_tough=False):
        assert event_col == "injury_event_matchproxy_same_day"
        assert group_col == "fragility_group"
        assert include_tough
        return panel.copy()

    @staticmethod
    def add_prior_history_control_columns(frame):
        return frame.copy()

    @staticmethod
    def add_recent_prior_injury_return_flags(frame, injuries):
        return frame.copy()

    @staticmethod
    def add_lineup_start_status(frame, lineups):
        return frame.copy()

    @staticmethod
    def add_player_and_club_metadata(frame, transfermarkt_dir):
        out = frame.copy()
        out["metadata_attached"] = str(transfermarkt_dir)
        return out


def test_history_scaling_preparation_and_formulas(load_src_module, tmp_path):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(12)
    scaling = module.history_scale(frame)
    assert scaling["scale_log_iqr"] > 0
    scaled = module.apply_history_scale(frame, scaling)
    assert module.HISTORY_MODEL_COL in scaled
    with pytest.raises(KeyError, match="missing columns"):
        module.history_scale(pd.DataFrame())
    with pytest.raises(ValueError, match="positive"):
        module.apply_history_scale(frame, {**scaling, "scale_log_iqr": 0.0})

    constant = pd.DataFrame({module.HISTORY_COL: [0.0, 0.0]})
    assert module.history_scale(constant)["scale_log_iqr"] == 1.0
    spread = pd.DataFrame({module.HISTORY_COL: [0.0, 0.0, 0.0, 2.0]})
    assert module.history_scale(spread)["scale_log_iqr"] > 0

    prepared, prepared_scale = module.prepare_jsams_frame(
        _PrepPrimary(), frame, pd.DataFrame(), None, tmp_path
    )
    assert prepared["metadata_attached"].eq(str(tmp_path)).all()
    assert prepared["lineup_role_model"].isin(module.LINEUP_ROLES).all()
    assert prepared_scale["median_rate"] >= 0
    no_metadata, _ = module.prepare_jsams_frame(
        _PrepPrimary(),
        frame.drop(
            columns=[module.SAME_DAY_SEVERE_COL, module.SAME_DAY_MUSCLE_COL]
        ),
        pd.DataFrame(),
        None,
    )
    assert "metadata_attached" not in no_metadata

    expression = module.spline_expression(330.0)
    assert "knots=(45.0, 90.0, 135.0)" in expression
    assert module.SAME_DAY_COL in module.continuous_formula(module.SAME_DAY_COL, 330.0)
    with pytest.raises(ValueError, match="support"):
        module.spline_expression(100.0)


def test_model_fit_predictions_tests_and_contrasts(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame()
    scaling = module.history_scale(frame)
    frame = module.apply_history_scale(frame, scaling)

    fitted = module.fit_continuous_model(
        frame, module.SAME_DAY_COL, "per_appearance"
    )
    observed = module.fit_continuous_model(
        frame, module.SAME_DAY_COL, "observed_minutes"
    )
    fixed = module.fit_continuous_model(
        frame, module.SAME_DAY_COL, "fixed_90"
    )
    assert len(fitted.params) > 5
    assert len(observed.params) == len(fixed.params)
    with pytest.raises(ValueError, match="Unknown denominator"):
        module.fit_continuous_model(frame, module.SAME_DAY_COL, "wrong")
    with pytest.raises(ValueError, match="No events"):
        module.fit_continuous_model(
            frame.assign(injury_event_matchproxy_same_day=0),
            module.SAME_DAY_COL,
        )
    with pytest.raises(KeyError, match="missing columns"):
        module.fit_continuous_model(
            frame.drop(columns=module.PLAYER_ID_COL), module.SAME_DAY_COL
        )

    tests = module.formal_model_tests(fitted, "primary")
    assert len(tests) == 3
    assert tests["df"].gt(0).all()
    assert module._joint_wald(fitted, ["not-a-term"])["df"] == 0
    assert np.isnan(module._normal_p_value(np.nan, 1.0))
    assert module._normal_p_value(0.0, 0.0) == 1.0
    assert module._normal_p_value(1.0, 0.0) == 0.0

    template = module.prediction_template(
        [0.0, 90.0], scaling["median_log"], scaling, {"extra": 1}
    )
    assert template["extra"].eq(1).all()
    per_appearance = module.prediction_intervals(
        fitted, template.drop(columns="extra"), "per_appearance"
    )
    per_hours = module.prediction_intervals(
        observed, template.drop(columns="extra"), "observed_minutes"
    )
    assert per_appearance["prediction_unit"].eq(
        "reported_events_per_1000_appearances"
    ).all()
    assert per_hours["prediction_unit"].eq(
        "reported_events_per_1000_match_hours"
    ).all()
    contrast = module.design_contrast(
        fitted,
        module.prediction_template([0.0], scaling["median_log"], scaling),
        module.prediction_template([180.0], scaling["median_log"], scaling),
        "odds_ratio",
    )
    assert contrast["estimate"] > 0

    output = module.model_outputs(
        frame,
        scaling,
        module.SAME_DAY_COL,
        "per_appearance",
        "primary",
    )
    assert set(output) == {"predictions", "tests", "contrasts", "coefficients"}
    assert len(output["contrasts"]) == 9
    output_rate = module.model_outputs(
        frame,
        scaling,
        module.SAME_DAY_COL,
        "observed_minutes",
        "rate",
    )
    assert output_rate["contrasts"]["effect_measure"].eq(
        "incidence_rate_ratio"
    ).all()


def test_symmetric_suite_support_and_specification(load_src_module, monkeypatch):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = module.apply_history_scale(_frame(12), module.history_scale(_frame(12)))
    scaling = module.history_scale(frame)

    def fake_outputs(frame, scaling, event_col, denominator, model_id):
        base = pd.DataFrame({"model_id": [model_id], "value": [1.0]})
        return {
            "predictions": base.copy(),
            "tests": base.copy(),
            "contrasts": base.copy(),
            "coefficients": base.copy(),
        }

    monkeypatch.setattr(module, "model_outputs", fake_outputs)
    suite = module.run_symmetric_model_suite(frame, scaling)
    assert all(len(value) == 9 for value in suite.values())
    monkeypatch.undo()

    support = module.exposure_support_table(
        pd.concat(
            [
                frame,
                frame.assign(fragility_group="unknown"),
            ],
            ignore_index=True,
        )
    )
    assert len(support) == 16
    assert "unknown" in support["history_stratum"].tolist()
    with pytest.raises(KeyError, match="missing columns"):
        module.exposure_support_table(pd.DataFrame())
    specification = module.model_specification_table(frame, scaling)
    assert specification.loc[0, "denominator"] == "recorded appearance"
    assert specification.loc[0, "cluster"] == "player"


def test_minute_bootstrap(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(15)
    samples, summary = module.cluster_bootstrap_minute_difference(
        frame, replicates=20, seed=4
    )
    assert len(samples) == 20
    assert summary.loc[0, "bootstrap_replicates"] == 20
    assert summary.loc[0, "difference_ci_low"] <= summary.loc[
        0, "difference_ci_high"
    ]
    with pytest.raises(ValueError, match="at least 2"):
        module.cluster_bootstrap_minute_difference(frame, replicates=1)
    with pytest.raises(ValueError, match="Both event"):
        module.cluster_bootstrap_minute_difference(
            frame.assign(injury_event_matchproxy_same_day=0), replicates=2
        )
    with pytest.raises(KeyError, match="missing columns"):
        module.cluster_bootstrap_minute_difference(pd.DataFrame(), replicates=2)


def test_selection_standardization(load_src_module, monkeypatch):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(30)
    scaling = module.history_scale(frame)
    frame = module.apply_history_scale(frame, scaling)
    curves, comparisons, tests = module.selection_standardization(frame, scaling)
    assert set(curves["model"]) == {
        "unadjusted_lineup_known",
        "standardized_lineup_and_return",
    }
    assert len(comparisons) == 12
    assert len(tests) == 6
    assert comparisons["standardized_per_1000_appearances"].gt(0).all()
    with pytest.raises(ValueError, match="Both recorded lineup"):
        module.selection_standardization(
            frame.assign(lineup_role_model="starting_lineup"), scaling
        )
    with pytest.raises(KeyError, match="missing columns"):
        module.selection_standardization(pd.DataFrame(), scaling)

    real_fit = module.fit_continuous_model

    class Marker:
        pass

    markers = iter([Marker(), Marker()])
    monkeypatch.setattr(module, "fit_continuous_model", lambda *args, **kwargs: next(markers))
    monkeypatch.setattr(
        module,
        "_weighted_standardized_prediction",
        lambda result, reference, burden, history: {
            "estimate_per_1000_appearances": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "standard_error_probability": 0.0,
        },
    )
    monkeypatch.setattr(
        module,
        "formal_model_tests",
        lambda result, model_id: pd.DataFrame(
            {"model_id": [model_id], "contrast_id": ["x"], "p_value": [1.0]}
        ),
    )
    _, zero, _ = module.selection_standardization(frame, scaling)
    assert zero["relative_change_percent"].isna().all()
    monkeypatch.setattr(module, "fit_continuous_model", real_fit)


def test_context_and_two_way_sensitivities(load_src_module, monkeypatch):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(30)
    scaling = module.history_scale(frame)
    frame = module.apply_history_scale(frame, scaling)
    result = module.fit_two_way_continuous_model(
        frame,
        module.SAME_DAY_COL,
        module.PLAYER_ID_COL,
        "match_cluster_id",
    )
    assert len(result.params) > 5
    monkeypatch.setattr(
        module,
        "fit_continuous_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "fit_two_way_continuous_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "formal_model_tests",
        lambda result, model_id: pd.DataFrame(
            {
                "model_id": [model_id] * 3,
                "contrast_id": ["a", "b", "c"],
                "p_value": [0.1, 0.2, 0.3],
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "_context_model_contrast",
        lambda result, subset: {
            "estimate": 1.2,
            "ci_low": 0.9,
            "ci_high": 1.6,
            "log_estimate": np.log(1.2),
            "standard_error": 0.1,
            "p_value": 0.2,
        },
    )
    outputs, contrasts = module.context_sensitivity_analysis(frame)
    assert set(outputs["model_id"]) == {
        "age_position_clubseason_adjusted",
        "competition_context_adjusted",
        "premier_league_current_match_only",
        "two_way_player_match_cluster",
    }
    assert len(outputs) == 12
    assert len(contrasts) == 4
    assert contrasts["estimate"].eq(1.2).all()
    monkeypatch.setattr(
        module,
        "context_sensitivity_analysis",
        lambda frame: (outputs, contrasts),
    )
    assert module.context_sensitivity_tests(frame).equals(outputs)
    monkeypatch.undo()
    with pytest.raises(KeyError, match="two-way cluster frame"):
        module.fit_two_way_continuous_model(
            pd.DataFrame(), module.SAME_DAY_COL, "a", "b"
        )
    with pytest.raises(ValueError, match="at least two"):
        module.fit_two_way_continuous_model(
            frame.assign(match_cluster_id="one"),
            module.SAME_DAY_COL,
            module.PLAYER_ID_COL,
            "match_cluster_id",
        )
    with pytest.raises(KeyError, match="context sensitivity frame"):
        module.context_sensitivity_analysis(pd.DataFrame())
    with pytest.raises(ValueError, match="No estimable rows"):
        module.context_sensitivity_analysis(
            frame.assign(
                competition_context="domestic cup",
                injury_event_matchproxy_same_day=0,
            )
        )


def test_cohort_flow_and_descriptives(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(12)
    reports = pd.DataFrame({"tm_player_id": [1, 1, 2]})
    episodes = pd.DataFrame({"tm_player_id": [1, 2]})
    flow = module.cohort_flow_table(reports, episodes, frame, frame)
    assert len(flow) == 8
    assert flow.loc[flow["row_type"].eq("analysis cohort"), "same_day_events"].iloc[0] == int(
        frame[module.SAME_DAY_COL].sum()
    )
    for bad, message in (
        ((pd.DataFrame(), episodes, frame, frame), "source reports"),
        ((reports, pd.DataFrame(), frame, frame), "injury episodes"),
        ((reports, episodes, pd.DataFrame(), frame), "raw match panel"),
        ((reports, episodes, frame, pd.DataFrame()), "match frame"),
    ):
        with pytest.raises(KeyError, match=message):
            module.cohort_flow_table(*bad)

    typed = frame.assign(matchproxy_public_injury_type="muscle/tendon")
    national = pd.DataFrame(
        {
            "tm_player_id": [0, 0, 1, 999],
            "is_senior_competitive": [True, False, True, True],
        }
    )
    descriptive = module.cohort_descriptive_table(typed, national)
    assert {
        "players",
        "appearances",
        "outcomes",
        "playing position",
    }.issubset(set(descriptive["section"]))
    reduced = frame.drop(columns=["age_years", "position_group"])
    reduced_descriptive = module.cohort_descriptive_table(reduced)
    assert "playing position" not in reduced_descriptive["section"].tolist()
    with pytest.raises(KeyError, match="national appearances"):
        module.cohort_descriptive_table(frame, pd.DataFrame())
    empty_values = frame.copy()
    empty_values["all_minutes_last_7d"] = np.nan
    missing_summary = module.cohort_descriptive_table(empty_values)
    burden = missing_summary.loc[
        missing_summary["metric"].eq("previous-7-day match minutes")
    ].iloc[0]
    assert burden["display"] == "not available"
    with pytest.raises(KeyError, match="missing columns"):
        module.cohort_descriptive_table(pd.DataFrame())


def _source_frames(module):
    return {
        "exposure": pd.DataFrame(
            {"contrast_id": ["a", "b"], "p_value": [0.01, np.nan]}
        ),
        "type_history": pd.DataFrame({"test_id": ["c"], "p_value": [0.2]}),
        "national_status": pd.DataFrame(
            {"contrast_id": ["d"], "p_value": [0.4]}
        ),
        "recovery": pd.DataFrame({"contrast": ["e"], "p_value": [0.5]}),
        "lineup": pd.DataFrame({"model": ["f"], "p_value": [0.6]}),
        "two_way_cluster": pd.DataFrame(
            {"contrast": ["g"], "p_value": [0.7]}
        ),
        "primary_contrasts": pd.DataFrame(
            {
                "model_id": ["primary_same_day_per_appearance"],
                "contrast_id": ["h"],
                "p_value": [0.8],
            }
        ),
        "selection_standardized": pd.DataFrame(
            {"contrast_id": ["i"], "p_value": [0.9]}
        ),
        "context_sensitivity": pd.DataFrame(
            {"contrast_id": ["j"], "p_value": [0.95]}
        ),
        "functional_form": pd.DataFrame(
            {"contrast_id": ["k"], "p_value": [0.51]}
        ),
        "outcome_quality": pd.DataFrame(
            {"contrast_id": ["l"], "p_value": [0.52]}
        ),
        "lineup_role": pd.DataFrame(
            {"contrast_id": ["m"], "p_value": [0.53]}
        ),
        "selection_effect": pd.DataFrame(
            {"contrast_id": ["n"], "p_value": [0.54]}
        ),
        "cohort_robustness": pd.DataFrame(
            {"contrast_id": ["o"], "p_value": [0.55]}
        ),
        "within_player": pd.DataFrame(
            {"model_id": ["within"], "p_value": [0.56]}
        ),
    }


def test_hypothesis_register_and_holm(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    primary = pd.DataFrame(
        {
            "model_id": [
                "primary_same_day_per_appearance",
                "lag1_per_appearance",
            ],
            "contrast_id": [
                "global_recent_exposure_association_at_median_history",
                "global_any_recent_exposure_term",
            ],
            "p_value": [0.03, 0.8],
        }
    )
    sources = _source_frames(module)
    sources["recovery"]["p_holm_extension_family"] = [0.42]
    register, summary = module.hypothesis_register(primary, sources)
    assert len(register) == 18
    assert register["hypothesis_id"].is_unique
    assert "primary_reference_association" in register["analysis_role"].tolist()
    assert summary["registered_tests"].sum() == 18
    assert "p_adjusted_reported" in register
    assert register["confirmatory_status"].eq("exploratory").all()
    assert not register["dated_prospective_analysis_plan_available"].any()
    assert "adjusted_rejections" in summary
    assert register.loc[
        register["family"].eq("recovery_interval_family")
        & register["contrast_id"].eq("e"),
        "p_adjusted_reported",
    ].iloc[0] == pytest.approx(0.42)
    assert module._holm_adjust(pd.Series([np.nan])).isna().all()
    adjusted = module._holm_adjust(pd.Series([0.01, 0.04]))
    assert adjusted.iloc[0] == pytest.approx(0.02)
    with pytest.raises(KeyError, match="Unknown"):
        module.hypothesis_register(
            primary, {"other": pd.DataFrame({"p_value": [0.2]})}
        )
    with pytest.raises(KeyError, match="missing columns"):
        module.hypothesis_register(pd.DataFrame(), {})
    bad_sources = _source_frames(module)
    bad_sources["lineup"] = pd.DataFrame()
    with pytest.raises(KeyError, match="lineup hypothesis"):
        module.hypothesis_register(primary, bad_sources)


def test_context_model_contrast_builds_reference_rows(
    load_src_module, monkeypatch
):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(players=2, rows_per_player=3)
    captured = {}

    def fake_contrast(result, first, second, scale):
        captured["result"] = result
        captured["first"] = first
        captured["second"] = second
        captured["scale"] = scale
        return {"estimate": 1.2, "ci_low": 0.9, "ci_high": 1.6}

    monkeypatch.setattr(module, "design_contrast", fake_contrast)
    result = object()
    output = module._context_model_contrast(result, frame)

    assert output["estimate"] == pytest.approx(1.2)
    assert captured["result"] is result
    assert captured["scale"] == "odds_ratio"
    assert captured["first"][module.BURDEN_COL].iloc[0] == 0.0
    assert captured["second"][module.BURDEN_COL].iloc[0] == 180.0
    assert captured["first"][module.HISTORY_MODEL_COL].iloc[0] == 0.0
    for term in module.CALENDAR_TERMS:
        assert captured["first"][term].iloc[0] == 0.0


def test_publication_claim_hierarchy(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    tests = pd.DataFrame(
        [
            {
                "model_id": "primary_same_day_per_appearance",
                "contrast_id": "global_recent_exposure_association_at_median_history",
                "p_value": 0.01,
            },
            {
                "model_id": "primary_same_day_per_appearance",
                "contrast_id": "global_recent_exposure_by_continuous_history_interaction",
                "p_value": 0.20,
            },
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "model_id": model,
                "history_anchor": "median",
                "burden_from": 0.0,
                "burden_to": 180.0,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
            }
            for model, estimate, low, high in (
                ("primary_same_day_per_appearance", 1.5, 1.1, 2.0),
                ("same_day_observed_minutes", 1.2, 0.9, 1.6),
            )
        ]
    )
    minute = pd.DataFrame(
        [
            {
                "event_minus_non_event_minutes": -20.0,
                "difference_ci_low": -22.0,
                "difference_ci_high": -18.0,
                "bootstrap_replicates": 1000,
            }
        ]
    )
    lineup_minutes = pd.DataFrame(
        [
            {
                "comparison": comparison,
                "event_minus_non_event_minutes": estimate,
                "difference_ci_low": estimate - 2.0,
                "difference_ci_high": estimate + 2.0,
            }
            for comparison, estimate in (
                ("starting_lineup", -30.0),
                ("substitute_list", 1.0),
                ("lineup_standardized", -24.0),
            )
        ]
    )
    selection = pd.DataFrame(
        [
            {
                "history_anchor": "median",
                "burden_to": 180.0,
                "comparison_type": "difference_between_changes",
                "estimate": 0.8,
                "ci_low": 0.5,
                "ci_high": 1.1,
                "p_value": 0.046,
            }
        ]
    )
    functional = pd.DataFrame(
        {
            "model_id": ["f1", "f2"],
            "estimate": [1.4, 1.6],
            "ci_low": [1.1, 1.2],
            "ci_high": [1.8, 2.0],
        }
    )
    quality = pd.DataFrame(
        {
            "model_id": [
                f"quality_{module.SAME_DAY_SEVERE_COL}_per_appearance",
                f"quality_{module.SAME_DAY_MUSCLE_COL}_per_appearance",
            ],
            "estimate": [1.6, 2.1],
            "ci_low": [1.1, 1.3],
            "ci_high": [2.4, 3.4],
        }
    )
    cohorts = pd.DataFrame(
        {
            "cohort_id": ["club_plus_senior_national_exposure"],
            "estimate": [1.8],
            "ci_low": [1.3],
            "ci_high": [2.4],
        }
    )
    family = pd.DataFrame(
        [
            {
                "family": "historical_exposure_response_family",
                "test_domain": "exposure_response_or_effect_modification",
                "registered_tests": 154,
                "adjusted_rejections": 0,
            }
        ]
    )
    headline = pd.DataFrame(
        [
            {
                "inference_target": (
                    "post hoc median-history 0-to-180-minute anchor contrast"
                ),
                "p_value_raw": 0.003,
                "p_value_holm_within_family": 0.20,
            }
        ]
    )
    within = pd.DataFrame(
        [
            {
                "model_id": "within_player_reference_bspline_same_day",
                "estimate": 1.6,
                "ci_low": 1.1,
                "ci_high": 2.3,
                "n_discordant_strata": 100,
            }
        ]
    )
    hierarchy = module.publication_claim_hierarchy(
        tests,
        contrasts,
        minute,
        lineup_minutes,
        selection,
        functional,
        quality,
        cohorts,
        family,
        headline,
        within,
    )
    assert hierarchy["visibility_rule_passes"].all()
    interaction = hierarchy.loc[
        hierarchy["claim_id"].eq("continuous_history_effect_modification_null")
    ].iloc[0]
    assert interaction["tier"] == 5
    assert not interaction["abstract_visible"]
    assert not interaction["abstract_recommended"]
    assert not interaction["main_display_recommended"]
    assert interaction["main_results_sentence_limit"] == 1
    assert hierarchy["tier_justification"].str.len().gt(0).all()
    assert hierarchy.loc[hierarchy["tier"].eq(2), "abstract_recommended"].all()
    assert "0/154" in hierarchy.loc[
        hierarchy["claim_id"].eq("historical_categorical_exposure_family_null"),
        "evidence",
    ].iloc[0]

    with pytest.raises(KeyError, match="claim-hierarchy model tests"):
        module.publication_claim_hierarchy(
            pd.DataFrame(),
            contrasts,
            minute,
            lineup_minutes,
            selection,
            functional,
            quality,
            cohorts,
            family,
            headline,
            within,
        )
    duplicate = pd.concat([tests, tests.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Expected one primary global exposure"):
        module.publication_claim_hierarchy(
            duplicate,
            contrasts,
            minute,
            lineup_minutes,
            selection,
            functional,
            quality,
            cohorts,
            family,
            headline,
            within,
        )

    invalid = hierarchy.copy()
    invalid.loc[invalid["tier"].eq(5), "abstract_recommended"] = True
    with pytest.raises(ValueError, match="promoted"):
        module.enforce_claim_visibility(invalid)
    unjustified = hierarchy.copy()
    unjustified.loc[0, "tier_justification"] = ""
    with pytest.raises(ValueError, match="tier justification"):
        module.enforce_claim_visibility(unjustified)
    with pytest.raises(KeyError, match="publication claim hierarchy"):
        module.enforce_claim_visibility(pd.DataFrame())


def test_quality_outcomes_seasons_and_exposure_forms(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    panel = _frame(players=2, rows_per_player=4)
    episodes = pd.DataFrame(
        {
            "tm_player_id": [0, 1],
            "start_date": [panel.loc[0, "date"], panel.loc[4, "date"]],
            "duration_days": [35.0, 5.0],
            "injury_desc": ["hamstring strain", "fracture"],
        }
    )
    panel.loc[0, module.SAME_DAY_COL] = 1
    panel.loc[4, module.SAME_DAY_COL] = 1
    classified = module.add_same_day_quality_outcomes(
        panel,
        episodes,
        lambda value: "muscle/tendon" if "strain" in str(value) else "bone/fracture",
    )
    assert classified.loc[0, module.SAME_DAY_SEVERE_COL] == 1
    assert classified.loc[0, module.SAME_DAY_MUSCLE_COL] == 1
    assert classified.loc[4, module.SAME_DAY_SEVERE_COL] == 0
    assert classified["same_day_quality_metadata_matched"].sum() == 2
    with pytest.raises(KeyError, match="same-day quality panel"):
        module.add_same_day_quality_outcomes(pd.DataFrame(), episodes, str)
    with pytest.raises(KeyError, match="same-day quality episodes"):
        module.add_same_day_quality_outcomes(panel, pd.DataFrame(), str)

    seasons = module.season_start_year(
        pd.Series([pd.Timestamp("2023-08-01"), pd.Timestamp("2024-01-01")])
    )
    assert seasons.tolist() == [2023, 2023]
    bands = module.exposure_band(pd.Series([0, 30, 60, 120, 170, 200]))
    assert list(bands.astype(str)) == list(module.EXPOSURE_BAND_ORDER)
    for exposure_spec in module.FUNCTIONAL_FORM_SPECS:
        expression = module.exposure_expression(exposure_spec, 330.0)
        assert expression
    with pytest.raises(ValueError, match="Unknown exposure"):
        module.exposure_expression("bad", 330.0)
    categorical = module.add_exposure_spec_columns(panel, "fixed_match_bands")
    assert "exposure_band_model" in categorical
    unchanged = module.add_exposure_spec_columns(panel, "linear_per_90")
    assert "exposure_band_model" not in unchanged
    with pytest.raises(ValueError, match="finite and non-negative"):
        module.prepare_jsams_frame(
            _PrepPrimary(), panel, pd.DataFrame(), None, minimum_prior_minutes=-1
        )


def test_marginal_standardization_and_response_differences(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame()
    scaling = module.history_scale(frame)
    frame = module.apply_history_scale(frame, scaling)
    fitted = module.fit_continuous_model(frame, module.SAME_DAY_COL)
    reference = module.standardization_reference(frame)
    prediction = module.marginal_prediction_interval(
        fitted, reference, 90.0, 0.0
    )
    assert prediction["estimate"] > 0
    assert prediction["standardization"] == "observed_calendar_distribution"
    first = module._marginal_prediction_components(
        fitted, reference, 0.0, 0.0
    )
    second = module._marginal_prediction_components(
        fitted, reference, 180.0, 0.0
    )
    difference = module._response_difference(fitted, first, second)
    assert difference["effect_measure"].startswith("risk_difference")

    observed = module.fit_continuous_model(
        frame, module.SAME_DAY_COL, "observed_minutes"
    )
    rate = module.marginal_prediction_interval(
        observed, reference, 90.0, 0.0, "observed_minutes"
    )
    assert rate["prediction_unit"] == "reported_events_per_1000_match_hours"
    with pytest.raises(ValueError, match="Unknown denominator"):
        module._marginal_prediction_components(
            fitted, reference, 0.0, 0.0, "bad"
        )
    with pytest.raises(KeyError, match="prediction reference"):
        module._marginal_prediction_components(
            fitted, pd.DataFrame(), 0.0, 0.0
        )
    bad_reference = reference.copy()
    bad_reference["standardization_weight"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        module._marginal_prediction_components(
            fitted, bad_reference, 0.0, 0.0
        )
    with pytest.raises(ValueError, match="positive weight"):
        module.standardization_reference(frame.iloc[0:0])
    factorized = module.factorized_standardization_reference(
        frame, frame, ["lineup_role_model"]
    )
    assert factorized["standardization_weight"].sum() > 0
    with pytest.raises(KeyError, match="composition frame"):
        module.factorized_standardization_reference(frame, pd.DataFrame(), ["x"])
    mismatched = (second[0], second[1], "other", second[3])
    with pytest.raises(ValueError, match="same unit"):
        module._response_difference(fitted, first, mismatched)


def test_new_model_sensitivity_families(load_src_module, monkeypatch):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(30)
    scaling = module.history_scale(frame)
    frame = module.apply_history_scale(frame, scaling)
    functional_tests, functional_contrasts = module.functional_form_sensitivity(
        frame, scaling
    )
    assert functional_tests["exposure_spec"].nunique() == 5
    assert len(functional_contrasts) == 5
    quality_tests, quality_contrasts, quality_summary = (
        module.outcome_quality_sensitivity(frame, scaling)
    )
    assert quality_tests["event_col"].nunique() == 3
    assert len(quality_contrasts) == 3
    assert len(quality_summary) == 3
    zero_summary = module.crude_outcome_summary(
        frame.assign(**{module.SAME_DAY_COL: 0}),
        module.SAME_DAY_COL,
        "zero",
    )
    assert zero_summary.loc[0, "hour_rate_ci_low"] == 0
    with pytest.raises(ValueError, match="positive hours"):
        module.crude_outcome_summary(
            frame.iloc[0:0], module.SAME_DAY_COL, "empty"
        )
    zero_quality = frame.assign(**{module.SAME_DAY_SEVERE_COL: 0})
    tests, contrasts, summaries = module.outcome_quality_sensitivity(
        zero_quality, scaling
    )
    assert module.SAME_DAY_SEVERE_COL not in tests["event_col"].tolist()
    assert len(contrasts) == 2
    assert len(summaries) == 3

    fake_tests = pd.DataFrame(
        {"model_id": ["x"], "contrast_id": ["x"], "p_value": [0.5]}
    )
    fake_contrast = pd.DataFrame(
        {"model_id": ["x"], "contrast_id": ["x"], "p_value": [0.5]}
    )
    monkeypatch.setattr(
        module,
        "_model_test_and_contrast_rows",
        lambda *args, **kwargs: (fake_tests.copy(), fake_contrast.copy()),
    )
    lineup_tests, lineup_contrasts = module.lineup_role_model_sensitivity(frame)
    assert len(lineup_tests) == 2
    assert len(lineup_contrasts) == 2
    with pytest.raises(KeyError, match="lineup-role frame"):
        module.lineup_role_model_sensitivity(pd.DataFrame())


def test_selection_effect_and_lineup_minute_standardization(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(40)
    scaling = module.history_scale(frame)
    frame = module.apply_history_scale(frame, scaling)
    bootstrap, effects = module.selection_effect_contrasts(
        frame, scaling, replicates=20, seed=6
    )
    assert len(bootstrap) == 20 * 18
    assert len(effects) == 18
    assert set(effects["comparison_type"]) == {
        "changing_observed_lineup_return_composition",
        "fixed_pooled_lineup_return_composition",
        "difference_between_changes",
    }
    samples, summary = module.lineup_standardized_minute_difference(
        frame, replicates=20, seed=5
    )
    assert len(samples) == 20
    assert set(summary["comparison"]) == {
        "starting_lineup",
        "substitute_list",
        "lineup_standardized",
    }
    with pytest.raises(ValueError, match="at least 2"):
        module.lineup_standardized_minute_difference(frame, replicates=1)
    with pytest.raises(ValueError, match="Both recorded lineup"):
        module.lineup_standardized_minute_difference(
            frame.assign(lineup_role_model="starting_lineup"), replicates=2
        )
    with pytest.raises(ValueError, match="Both recorded lineup"):
        module.selection_effect_contrasts(
            frame.assign(lineup_role_model="starting_lineup"),
            scaling,
            replicates=2,
        )
    with pytest.raises(ValueError, match="at least 2"):
        module.selection_effect_contrasts(frame, scaling, replicates=1)


def test_national_exposure_and_cohort_robustness(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(30)
    features = frame[["tm_player_id", "date"]].copy()
    features["senior_competitive_national_only_minutes_last_7d"] = 0.0
    features.loc[features.index[:5], "senior_competitive_national_only_minutes_last_7d"] = 90.0
    expanded, audit = module.add_senior_national_exposure(frame, features)
    assert audit.loc[0, "rows_with_added_national_minutes"] == 5
    assert expanded[module.BURDEN_COL].sum() == pytest.approx(
        frame[module.BURDEN_COL].sum() + 450.0
    )
    duplicate = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique by player-date"):
        module.add_senior_national_exposure(frame, duplicate)
    with pytest.raises(KeyError, match="national exposure features"):
        module.add_senior_national_exposure(frame, pd.DataFrame())

    tests, contrasts, cohort_audit = module.cohort_robustness_suite(
        {"reference": frame, "expanded": expanded}
    )
    assert len(tests) == 6
    assert len(contrasts) == 2
    assert len(cohort_audit) == 2
    with pytest.raises(ValueError, match="At least one"):
        module.cohort_robustness_suite({})
    with pytest.raises(ValueError, match="No estimable"):
        module.cohort_robustness_suite(
            {"zero": frame.assign(**{module.SAME_DAY_COL: 0})}
        )


def test_nonestimable_and_exclusion_branches(load_src_module, monkeypatch):
    module = load_src_module("34_jsams_referee_analysis.py")

    class BrokenWald:
        params = pd.Series([0.0], index=[module.BURDEN_COL])

        @staticmethod
        def wald_test(*args, **kwargs):
            raise ValueError("singular")

    failed = module._joint_wald(BrokenWald(), [module.BURDEN_COL])
    assert np.isnan(failed["p_value"])

    frame = _frame(20)
    scaling = module.history_scale(frame)
    scaled = module.apply_history_scale(frame, scaling)
    no_zero_support = scaled[scaled[module.BURDEN_COL].ne(0.0)]
    with pytest.raises(ValueError, match="support at 0, 45 and 180"):
        module.selection_effect_contrasts(
            no_zero_support, scaling, replicates=2
        )

    fake_tests = pd.DataFrame(
        {"model_id": ["x"], "contrast_id": ["x"], "p_value": [0.5]}
    )
    fake_contrast = pd.DataFrame(
        {"model_id": ["x"], "contrast_id": ["x"], "p_value": [0.5]}
    )
    monkeypatch.setattr(
        module,
        "_model_test_and_contrast_rows",
        lambda *args, **kwargs: (fake_tests.copy(), fake_contrast.copy()),
    )
    one_role_without_events = scaled.copy()
    one_role_without_events.loc[
        one_role_without_events["lineup_role_model"].eq("substitute_list"),
        module.SAME_DAY_COL,
    ] = 0
    tests, contrasts = module.lineup_role_model_sensitivity(
        one_role_without_events
    )
    assert len(tests) == 1
    assert len(contrasts) == 1

    reports = pd.DataFrame({"tm_player_id": [1]})
    episodes = pd.DataFrame({"tm_player_id": [1]})
    with pytest.raises(ValueError, match="do not reproduce"):
        module.cohort_flow_table(
            reports, episodes, frame, frame.iloc[:-1]
        )
    without_optional = frame.drop(
        columns=[
            module.SAME_DAY_SEVERE_COL,
            module.SAME_DAY_MUSCLE_COL,
            "lineup_role_model",
        ]
    )
    descriptive = module.cohort_descriptive_table(without_optional)
    assert "lineup ascertainment" not in descriptive["section"].tolist()


def test_lineup_bootstrap_handles_failed_draws(load_src_module, monkeypatch):
    module = load_src_module("34_jsams_referee_analysis.py")
    rows = []
    for player in (0, 1):
        for role in module.LINEUP_ROLES:
            for event, minutes in ((1, 40.0), (0, 80.0)):
                if player == 1 and role == "substitute_list" and event == 1:
                    continue
                rows.append(
                    {
                        module.PLAYER_ID_COL: player,
                        module.MINUTES_COL: minutes,
                        module.SAME_DAY_COL: event,
                        "lineup_role_model": role,
                    }
                )
    frame = pd.DataFrame(rows)

    class SomeFailedDraws:
        calls = 0

        def multinomial(self, *args, **kwargs):
            self.calls += 1
            return np.array([0, 2]) if self.calls == 1 else np.array([2, 0])

    monkeypatch.setattr(
        module.np.random, "default_rng", lambda seed: SomeFailedDraws()
    )
    samples, summary = module.lineup_standardized_minute_difference(
        frame, replicates=10
    )
    assert len(samples) == 9
    assert summary["bootstrap_replicates_estimable"].eq(9).all()

    class AllFailedDraws:
        @staticmethod
        def multinomial(*args, **kwargs):
            return np.array([0, 2])

    monkeypatch.setattr(
        module.np.random, "default_rng", lambda seed: AllFailedDraws()
    )
    with pytest.raises(ValueError, match="Too few estimable"):
        module.lineup_standardized_minute_difference(frame, replicates=10)


def test_full_selection_bootstrap_handles_failed_refits(
    load_src_module, monkeypatch
):
    module = load_src_module("34_jsams_referee_analysis.py")
    full = _frame(30)
    scaling = module.history_scale(full)
    full = module.apply_history_scale(full, scaling)
    monkeypatch.setattr(
        module,
        "_fit_selection_bootstrap_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("failed")),
    )
    with pytest.raises(ValueError, match="full selection bootstrap"):
        module.selection_effect_contrasts(
            full,
            scaling,
            replicates=10,
            seed=1,
        )


def test_lineup_completeness_scope_and_structural_missingness(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(16)
    scaling = module.history_scale(frame)
    frame = module.apply_history_scale(frame, scaling)
    frame["season_start"] = np.where(frame[module.PLAYER_ID_COL] % 2, 2021, 2020)
    frame.loc[
        frame["season_start"].eq(2021), "lineup_role_model"
    ] = "lineup_unavailable_or_other"
    audit, assessment, complete = module.lineup_completeness_audit(frame)
    assert set(complete["season_start"]) == {2020}
    assert assessment.loc[0, "zero_coverage_seasons"] == "2021"
    assert not assessment.loc[0, "inverse_probability_reweighting_performed"]
    assert {
        "overall",
        "season",
        "recent exposure",
        "continuous-history quartile",
        "same-day outcome",
        "competition",
    } == set(audit["dimension"])

    all_known = frame.assign(
        lineup_role_model=np.where(
            np.arange(len(frame)) % 2,
            "starting_lineup",
            "substitute_list",
        )
    )
    _, no_gap, complete_all = module.lineup_completeness_audit(all_known)
    assert no_gap.loc[0, "zero_coverage_seasons"] == ""
    assert len(complete_all) == len(all_known)
    with pytest.raises(ValueError, match="No season"):
        module.lineup_completeness_audit(
            frame.assign(lineup_role_model="lineup_unavailable_or_other")
        )
    with pytest.raises(KeyError, match="lineup completeness"):
        module.lineup_completeness_audit(pd.DataFrame())


def test_within_player_same_day_analysis_and_failure_modes(
    load_src_module, monkeypatch
):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(24)
    scaling = module.history_scale(frame)
    frame = module.apply_history_scale(frame, scaling)
    frame["season_start"] = module.season_start_year(frame["date"])
    fitted = module.within_player_same_day_analysis(frame)
    assert len(fitted) == 4
    assert fitted["fit_status"].eq("ok").sum() >= 2
    assert fitted["n_discordant_strata"].gt(0).all()
    assert set(fitted["stratum_definition"]) == {"player", "player-season"}

    no_events = module.within_player_same_day_analysis(
        frame.assign(**{module.SAME_DAY_COL: 0})
    )
    assert no_events["fit_status"].eq(
        "not_estimable_no_discordant_strata"
    ).all()

    class FailingConditionalLogit:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def fit(**kwargs):
            raise ValueError("synthetic failure")

    monkeypatch.setattr(module, "ConditionalLogit", FailingConditionalLogit)
    failed = module.within_player_same_day_analysis(frame)
    assert failed["fit_status"].eq("failed_convergence").all()
    with pytest.raises(KeyError, match="within-player"):
        module.within_player_same_day_analysis(pd.DataFrame())


def _daily_timing_frame(module, players=8):
    rows = []
    for player in range(players):
        for index in range(9):
            timing = index % 3
            rows.append(
                {
                    module.PLAYER_ID_COL: player,
                    "injury_event": int(index in (0, 1, 2)),
                    "available_for_injury_risk": True,
                    "prior_minutes_played": 1000.0,
                    module.MINUTES_COL: 90.0 if timing == 0 else 0.0,
                    "minutes_yesterday": 90.0 if timing == 1 else 0.0,
                    module.BURDEN_COL: 90.0,
                    module.HISTORY_COL: 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_daily_report_timing_enrichment(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _daily_timing_frame(module)
    summary, samples, contrasts = module.daily_report_timing_enrichment(
        frame, replicates=20, seed=4
    )
    assert len(summary) == 3
    assert len(samples) == 40
    assert len(contrasts) == 2
    assert contrasts["risk_ratio_ci_low"].le(
        contrasts["risk_ratio_ci_high"]
    ).all()
    with pytest.raises(ValueError, match="at least 2"):
        module.daily_report_timing_enrichment(frame, replicates=1)
    no_appearance_events = frame.copy()
    no_appearance_events.loc[
        no_appearance_events[module.MINUTES_COL].gt(0), "injury_event"
    ] = 0
    with pytest.raises(ValueError, match="Each timing class"):
        module.daily_report_timing_enrichment(
            no_appearance_events, replicates=2
        )
    with pytest.raises(KeyError, match="daily timing"):
        module.daily_report_timing_enrichment(pd.DataFrame(), replicates=2)


def test_eligibility_player_comparison(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    base = _frame(12)
    reference = base[base[module.PLAYER_ID_COL].lt(8)].copy()
    comparison = module.eligibility_player_comparison(base, reference)
    assert set(comparison["eligibility_group"]) == {
        "included established players",
        "excluded before reaching 900 prior minutes",
    }
    assert "same-day reports per 1,000 appearances" in comparison["metric"].tolist()
    with pytest.raises(KeyError, match="eligibility base"):
        module.eligibility_player_comparison(pd.DataFrame(), reference)
    with pytest.raises(KeyError, match="eligibility reference"):
        module.eligibility_player_comparison(base, pd.DataFrame())


def test_headline_inference_audit(load_src_module):
    module = load_src_module("34_jsams_referee_analysis.py")
    register = pd.DataFrame(
        [
            {
                "model_id": "primary_same_day_per_appearance",
                "contrast_id": "global_recent_exposure_association_at_median_history",
                "family": "reference",
                "p_value": 0.01,
                "p_adjusted_reported": 0.02,
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
            {
                "model_id": "primary_same_day_per_appearance",
                "contrast_id": "median_history_180_vs_0",
                "family": "anchors",
                "p_value": 0.003,
                "p_adjusted_reported": 0.21,
                "estimate": 1.59,
                "ci_low": 1.16,
                "ci_high": 2.17,
            },
        ]
    )
    audit = module.headline_inference_audit(register)
    assert len(audit) == 2
    assert audit.loc[
        audit["inference_target"].str.contains("anchor"),
        "p_value_holm_within_family",
    ].iloc[0] == pytest.approx(0.21)
    with pytest.raises(ValueError, match="Expected one headline"):
        module.headline_inference_audit(
            pd.concat([register, register.iloc[[1]]], ignore_index=True)
        )
    with pytest.raises(KeyError, match="headline inference"):
        module.headline_inference_audit(pd.DataFrame())


def test_weighted_selection_helpers_reject_invalid_weights(
    load_src_module, monkeypatch
):
    module = load_src_module("34_jsams_referee_analysis.py")
    frame = _frame(20)
    scaling = module.history_scale(frame)
    frame = module.apply_history_scale(frame, scaling)
    valid = module._weighted_pattern_reference(frame, module.CALENDAR_TERMS)
    assert valid["standardization_weight"].sum() == len(frame)
    with pytest.raises(ValueError, match="finite and non-negative"):
        module._weighted_pattern_reference(
            frame,
            module.CALENDAR_TERMS,
            pd.Series(-1.0, index=frame.index),
        )
    with pytest.raises(ValueError, match="positive weight"):
        module._weighted_pattern_reference(
            frame,
            module.CALENDAR_TERMS,
            pd.Series(0.0, index=frame.index),
        )
    fitted = module.fit_continuous_model(
        frame,
        module.SAME_DAY_COL,
        extra_controls=(
            " + C(lineup_role_model) + "
            "returned_from_recorded_injury_within_14d"
        ),
    )
    with pytest.raises(ValueError, match="align"):
        module._fit_selection_bootstrap_model(
            frame,
            np.ones(len(frame) - 1),
            fitted.params,
        )

    class NonConverged:
        converged = False

    class FakeModel:
        @staticmethod
        def fit(**kwargs):
            return NonConverged()

    monkeypatch.setattr(module.smf, "glm", lambda **kwargs: FakeModel())
    with pytest.raises(ValueError, match="did not converge"):
        module._fit_selection_bootstrap_model(
            frame,
            np.ones(len(frame)),
            fitted.params,
        )
