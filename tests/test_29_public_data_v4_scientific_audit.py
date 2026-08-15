import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def feature_frame(module):
    data = {"tm_player_id": [1, 2], "date": ["2024-01-10", "2024-01-10"]}
    for scope in module.NATIONAL_INCREMENT_SCOPES.values():
        for days in module.WINDOW_DAYS:
            data[f"{scope}_minutes_last_{days}d"] = [90, 0]
            data[f"{scope}_matches_last_{days}d"] = [1, 0]
        data[f"{scope}_days_since_previous_appearance"] = [2, np.nan]
    return pd.DataFrame(data)


def match_frame():
    return pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "date": ["2024-01-10", "2024-01-10"],
            "injury_event_matchproxy": [1, 0],
            "fragility_group": ["fragile", "regular"],
            "all_minutes_last_7d": [0, 90],
            "all_minutes_played": [60, 90],
            "week_phase_sin": [0.0, 0.0],
            "week_phase_cos": [0.0, 0.0],
            "halfweek_phase_sin": [0.0, 0.0],
            "halfweek_phase_cos": [0.0, 0.0],
        }
    )


def test_loading_requirements_and_increment_merge(load_src_module, monkeypatch, tmp_path):
    module = load_src_module("29_public_data_v4_scientific_audit.py")
    path = tmp_path / "small.py"
    path.write_text("VALUE = 3\n", encoding="utf-8")
    loaded = module.load_module(path, "small")
    assert loaded.VALUE == 3
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *args: None)
    with pytest.raises(ImportError, match="Cannot load"):
        module.load_module(path, "missing")

    features = feature_frame(module)
    merged = module.merge_national_increment(
        match_frame(), features, "senior_competitive_national_only", 7
    )
    assert merged["recent_national_duty"].tolist() == [1, 0]
    assert merged["national_minutes_in_window"].tolist() == [90, 0]
    with pytest.raises(KeyError, match="match panel"):
        module.merge_national_increment(
            match_frame().drop(columns="date"), features, "senior_competitive_national_only", 7
        )
    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row"):
        module.merge_national_increment(
            match_frame(), duplicated, "senior_competitive_national_only", 7
        )


def test_national_record_quality_audit(load_src_module):
    module = load_src_module("29_public_data_v4_scientific_audit.py")
    assert np.isnan(module._percent(1, 0))
    records = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2],
            "game_id": [10, 11, 12],
            "participation_state": ["played", "played", "bench"],
            "minutes_played": [90, np.nan, 0],
            "competition_type_id": [11, 11, 19],
            "competition_id": ["EURO", "FS", "WCQ"],
            "kickoff_time_known": [True, True, False],
            "retained_for_exposure": [True, False, False],
            "exclusion_reason": ["", "played_missing_minutes", "not_played"],
        }
    )
    retained = pd.DataFrame(
        {
            "tm_player_id": [1],
            "game_id": [10],
            "source": ["endpoint"],
            "minutes_played": [90],
        }
    )
    audit = module.national_record_quality_audit(records, retained)
    values = audit.set_index("metric")
    assert values.loc["played_records_with_missing_minutes", "count"] == 1
    assert values.loc["primary_senior_competitive_missing_minutes", "count"] == 0
    assert values.loc["retained_source_rows:endpoint", "percent"] == 100
    assert "excluded_records:not_played" in values.index
    with pytest.raises(KeyError, match="national record audit"):
        module.national_record_quality_audit(records.drop(columns="game_id"), retained)


def test_exposure_change_audit_covers_scopes_windows_and_empty_support(load_src_module):
    module = load_src_module("29_public_data_v4_scientific_audit.py")
    features = feature_frame(module)
    audit = module.exposure_change_audit(match_frame(), features)
    assert len(audit) == len(module.NATIONAL_INCREMENT_SCOPES) * len(module.WINDOW_DAYS) * 3
    primary_all = audit.loc[
        audit["country_scope"].eq("senior_competitive")
        & audit["window_days"].eq(7)
        & audit["history_stratum"].eq("all")
    ].iloc[0]
    assert primary_all["n_rows_with_country_minutes"] == 1
    assert primary_all["n_proxy_events_on_changed_rows"] == 1
    assert primary_all["zero_club_burden_rows_reclassified"] == 1
    regular = audit.loc[
        audit["country_scope"].eq("senior_competitive")
        & audit["window_days"].eq(3)
        & audit["history_stratum"].eq("intermediate prior-injury history")
    ].iloc[0]
    assert np.isnan(regular["median_country_minutes_when_positive"])
    assert np.isnan(regular["zero_club_burden_rows_reclassified"])

    recovery_panel = match_frame().assign(days_since_last_match=[5, 3])
    recovery = module.recovery_change_audit(recovery_panel, features)
    assert len(recovery) == len(module.NATIONAL_INCREMENT_SCOPES) * 3
    recovery_all = recovery.loc[
        recovery["country_scope"].eq("senior_competitive")
        & recovery["history_stratum"].eq("all")
    ].iloc[0]
    assert recovery_all["n_recovery_intervals_changed"] == 1
    assert recovery_all["n_proxy_events_on_changed_rows"] == 1
    with pytest.raises(KeyError, match="match panel"):
        module.recovery_change_audit(
            recovery_panel.drop(columns="days_since_last_match"), features
        )


class FakeBaseModel:
    @staticmethod
    def prepare_model_frame(frame, event_col, group_col):
        out = frame.copy()
        out["model_group"] = out[group_col]
        out["log_minutes_played"] = np.log(out["all_minutes_played"])
        return out

    @staticmethod
    def spline_basis_expression(maximum):
        assert maximum == 90
        return "all_minutes_last_7d"


def test_duty_frame_formula_ratio_and_contrasts(load_src_module):
    module = load_src_module("29_public_data_v4_scientific_audit.py")
    panel = pd.concat([match_frame(), match_frame().assign(date="2024-01-20")], ignore_index=True)
    features = pd.concat(
        [feature_frame(module), feature_frame(module).assign(date="2024-01-20")],
        ignore_index=True,
    )
    frame = module.prepare_duty_model_frame(
        FakeBaseModel,
        panel,
        features,
        "senior_competitive_national_only",
        7,
        "injury_event_matchproxy",
    )
    assert frame["higher_history"].tolist() == [1, 0, 1, 0]
    assert frame["duty_within"].eq(0).all()
    assert frame["national_match_equivalents"].tolist() == [1, 0, 1, 0]
    formula = module._duty_formula(
        FakeBaseModel, frame, "injury_event_matchproxy", " + age_years"
    )
    assert "duty_within * higher_history" in formula
    assert formula.endswith(" + age_years")
    continuous_formula = module._duty_formula(
        FakeBaseModel,
        frame,
        "injury_event_matchproxy",
        "",
        exposure_prefix="national_match_equivalents",
    )
    assert "national_match_equivalents_within * higher_history" in continuous_formula

    names = ["duty_within", "duty_within:higher_history", "duty_between"]
    params = pd.Series([np.log(2), np.log(1.5), np.log(0.8)], index=names)
    covariance = pd.DataFrame(np.eye(3) * 0.01, index=names, columns=names)
    interval = module.ratio_interval(params, covariance, {"duty_within": 1})
    assert interval["estimate"] == pytest.approx(2)
    with pytest.raises(KeyError, match="missing contrast"):
        module.ratio_interval(params, covariance, {"absent": 1})
    zero_covariance = covariance * 0
    zero = module.ratio_interval(params * 0, zero_covariance, {"duty_within": 1})
    assert np.isnan(zero["p_value"])
    result = SimpleNamespace(
        params=params,
        cov_params=lambda: covariance,
    )
    contrasts = module.duty_contrasts(result)
    higher = contrasts.loc[
        contrasts["contrast_id"].eq("higher_history_recent_duty_within_player")
    ].iloc[0]
    assert higher["estimate"] == pytest.approx(3)
    assert "joint_within_player_duty_terms" in set(contrasts["contrast_id"])
    continuous_names = [
        "national_match_equivalents_within",
        "national_match_equivalents_within:higher_history",
        "national_match_equivalents_between",
    ]
    continuous_result = SimpleNamespace(
        params=pd.Series([0.1, 0.2, -0.1], index=continuous_names),
        cov_params=lambda: pd.DataFrame(
            np.eye(3) * 0.01, index=continuous_names, columns=continuous_names
        ),
    )
    assert len(
        module.duty_contrasts(
            continuous_result, exposure_prefix="national_match_equivalents"
        )
    ) == 5


def test_specifications_and_multiplicity(load_src_module):
    module = load_src_module("29_public_data_v4_scientific_audit.py")
    specs = module._specifications()
    assert len(specs) == 20
    assert {spec["window_days"] for spec in specs[:5]} == set(module.WINDOW_DAYS)
    assert all("history_group_col" in spec for spec in specs)
    results = pd.DataFrame(
        {
            "contrast_id": ["a", "a", "b"],
            "p_value": [0.01, 0.04, np.nan],
        }
    )
    adjusted = module.add_exploratory_multiplicity(results)
    assert adjusted["p_holm_exploratory_family"].notna().sum() == 2
    assert adjusted["p_holm_across_specifications"].notna().sum() == 2
    empty = module.add_exploratory_multiplicity(
        pd.DataFrame({"contrast_id": ["a"], "p_value": [np.nan]})
    )
    assert empty["p_holm_exploratory_family"].isna().all()
    with pytest.raises(KeyError, match="country-duty results"):
        module.add_exploratory_multiplicity(pd.DataFrame())


def conclusion_inputs():
    scopes = ["frozen_club_all", "frozen_club_plus_senior_national"]
    total = pd.DataFrame(
        {
            "contrast_id": ["global_spline_by_history_interaction"] * 2,
            "exposure_scope": scopes,
            "p_value": [0.53, 0.63],
            "reject_holm_v4_0_05": [False, False],
        }
    )
    selected = pd.DataFrame(
        {
            "exposure_scope": scopes * 2,
            "fragility_group": ["regular", "regular", "fragile", "fragile"],
            "all_minutes_last_7d": [0, 0, 180, 180],
            "pred_events_per_10000_min": [2.5, 2.6, 4.0, 4.2],
        }
    )
    duty = pd.DataFrame(
        {
            "specification_id": [
                "window_7d_primary_observed",
                "window_7d_primary_observed",
                "continuous_national_minutes_7d",
                "international_break_calendar_control_7d",
                "history_count_only_7d",
            ],
            "specification_family": [
                "country_duty_window",
                "country_duty_window",
                "country_duty_exposure_definition",
                "calendar_context",
                "prior_history_definition",
            ],
            "contrast_id": [
                "higher_history_recent_duty_within_player",
                "higher_vs_intermediate_recent_duty_interaction",
                "higher_history_recent_duty_within_player",
                "higher_history_recent_duty_within_player",
                "higher_history_recent_duty_within_player",
            ],
            "estimate": [1.9, 0.5, 1.8, 1.7, 1.6],
            "ci_low": [1.2, 0.2, 1.3, 1.1, 1.0],
            "ci_high": [3.0, 1.1, 2.6, 2.8, 2.5],
            "p_value": [0.005, 0.07, 0.001, 0.02, 0.04],
            "p_holm_exploratory_family": [0.10, 1.0, 0.04, 0.4, 0.8],
        }
    )
    return total, selected, duty


def test_conclusion_audit_false_and_true_gates(load_src_module):
    module = load_src_module("29_public_data_v4_scientific_audit.py")
    total, selected, duty = conclusion_inputs()
    gate = pd.DataFrame({"primary_v4_exposure_allowed": [False]})
    selection = pd.DataFrame({"ipw_usable": [False]})
    travel = pd.DataFrame(
        {"metric": ["timeline_rows_with_travel_distance"], "value": [0]}
    )
    audit = module.conclusion_audit(gate, total, selected, duty, selection, travel)
    assert set(audit["decision"]) >= {
        "no_material_change",
        "sensitivity_only",
        "hypothesis_generating",
        "not_supported",
    }
    true_audit = module.conclusion_audit(
        pd.DataFrame({"primary_v4_exposure_allowed": [True]}),
        total,
        selected,
        duty,
        pd.DataFrame({"ipw_usable": [True]}),
        pd.DataFrame(
            {"metric": ["timeline_rows_with_travel_distance"], "value": [2]}
        ),
    )
    assert (true_audit["decision"] == "supported").sum() == 2
    no_diag = module.conclusion_audit(
        gate, total, selected, duty, pd.DataFrame(), pd.DataFrame()
    )
    assert no_diag["decision"].eq("not_supported").sum() == 2
    with pytest.raises(KeyError, match="coverage gate"):
        module.conclusion_audit(
            pd.DataFrame(), total, selected, duty, selection, travel
        )
