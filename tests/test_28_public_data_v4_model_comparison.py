import types

import pandas as pd
import pytest


def model_inputs():
    match = pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "date": ["2024-01-01", "2024-01-01"],
            "all_minutes_played": [90, 90],
            "all_minutes_last_7d": [45, 50],
            "all_games_last_7d": [1, 1],
            "days_since_last_match": [4, 5],
            "injury_event_matchproxy": [0, 1],
            "fragility_group": ["regular", "fragile"],
        }
    )
    features = pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "date": ["2024-01-01", "2024-01-01"],
            "club_competitive_minutes_last_7d": [45, 50],
            "club_competitive_days_since_previous_appearance": [4, 5],
            "club_competitive_matches_last_7d": [1, 1],
            "club_competitive_national_minutes_last_7d": [0, 0],
            "club_plus_senior_national_minutes_last_7d": [105, 50],
            "club_plus_senior_national_days_since_previous_appearance": [2, 5],
            "club_plus_senior_national_matches_last_7d": [2, 1],
            "club_plus_senior_national_national_minutes_last_7d": [60, 0],
            "senior_competitive_national_only_minutes_last_7d": [60, 0],
            "senior_competitive_national_only_matches_last_7d": [1, 0],
            "senior_competitive_national_only_days_since_previous_appearance": [2, float("nan")],
        }
    )
    return match, features


class FakePrimaryModel:
    def prepare_model_frame(self, panel, event_col, group_col):
        out = panel.copy()
        out["model_group"] = out[group_col]
        assert event_col == "injury_event_matchproxy"
        return out

    def run_prediction_bundle(self, prepared, event_col):
        return {
            "selected": pd.DataFrame({"fragility_group": ["regular"], "all_minutes_last_7d": [90.0]}),
            "effect_modification": pd.DataFrame({"contrast_id": ["interaction"], "p_value": [0.2]}),
            "dispersion": 1.1,
            "estimator": "fake_glm",
        }

    def recovery_interval_trend_tests(self, prepared, event_col):
        return pd.DataFrame({"contrast_id": ["recovery"], "p_value": [0.4]})


def test_scope_features_preparation_and_parity(load_src_module, monkeypatch):
    module = load_src_module("28_public_data_v4_model_comparison.py")
    fake_loader = types.SimpleNamespace(exec_module=lambda loaded: setattr(loaded, "loaded", True))
    fake_spec = types.SimpleNamespace(loader=fake_loader)
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *args: fake_spec)
    monkeypatch.setattr(module.importlib.util, "module_from_spec", lambda spec: types.SimpleNamespace())
    assert module.load_primary_model_module("fake").loaded
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *args: None)
    with pytest.raises(ImportError):
        module.load_primary_model_module("missing")
    match, features = model_inputs()
    assert module.scope_feature_columns("x")["burden"] == "x_minutes_last_7d"
    frozen = module.prepare_scope_model_panel(match, features, module.FROZEN_CLUB_SCOPE)
    assert frozen.loc[0, "all_minutes_last_7d"] == 45
    assert frozen.loc[0, "scope_national_minutes_last_7d"] == 0
    frozen_expanded = module.prepare_scope_model_panel(
        match,
        features,
        module.FROZEN_CLUB_PLUS_SENIOR_NATIONAL_SCOPE,
    )
    assert frozen_expanded.loc[0, "all_minutes_last_7d"] == 105
    assert frozen_expanded.loc[0, "days_since_last_match"] == 2
    assert frozen_expanded.loc[1, "days_since_last_match"] == 5
    prepared = module.prepare_scope_model_panel(match, features, "club_plus_senior_national")
    assert prepared.loc[0, "all_minutes_last_7d"] == 105
    assert prepared.loc[0, "days_since_last_match"] == 2
    parity = module.baseline_parity_report(match, features)
    assert parity.set_index("metric").loc["frozen_comparator_burden_mismatch_rows", "value"] == 0
    assert parity.set_index("metric").loc["refreshed_strict_club_burden_mismatch_rows", "value"] == 0
    mismatched = features.copy()
    mismatched.loc[0, "club_competitive_minutes_last_7d"] = 100
    assert module.baseline_parity_report(match, mismatched).set_index("metric").loc["refreshed_strict_club_burden_mismatch_rows", "value"] == 1
    with pytest.raises(KeyError, match="match panel"):
        module.prepare_scope_model_panel(match.drop(columns="date"), features, "club_competitive")
    duplicate = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row"):
        module.prepare_scope_model_panel(match, duplicate, "club_competitive")
    with pytest.raises(ValueError, match="one row"):
        module.prepare_scope_model_panel(
            match,
            duplicate,
            module.FROZEN_CLUB_PLUS_SENIOR_NATIONAL_SCOPE,
        )
    with pytest.raises(KeyError, match="match panel"):
        module.prepare_scope_model_panel(
            match.drop(columns="all_games_last_7d"),
            features,
            module.FROZEN_CLUB_SCOPE,
        )


def test_bundle_model_execution_and_adjustment(load_src_module):
    module = load_src_module("28_public_data_v4_model_comparison.py")
    match, features = model_inputs()
    fake = FakePrimaryModel()
    panel = module.prepare_scope_model_panel(match, features, "club_competitive")
    selected, effects, recovery, audit = module.run_scope_model(fake, panel, "club_competitive")
    assert selected.loc[0, "analysis_label"] == "unweighted"
    assert effects.loc[0, "analysis_component"] == "previous_7d_minutes"
    assert recovery.loc[0, "analysis_component"] == "recovery_interval"
    assert audit.loc[0, "n_events"] == 1
    adjusted = module.add_comparison_multiplicity(pd.DataFrame({"p_value": [0.01, float("nan")]}))
    assert adjusted.loc[0, "reject_holm_v4_0_05"]
    assert not adjusted.loc[1, "reject_bh_v4_0_05"]
    assert not module.add_comparison_multiplicity(pd.DataFrame({"p_value": [float("nan")]})).loc[0, "reject_holm_v4_0_05"]
    with pytest.raises(KeyError, match="formal comparison tests"):
        module.add_comparison_multiplicity(pd.DataFrame())
    gate = pd.DataFrame({"primary_v4_exposure_allowed": [True]})
    assert module.coverage_decision(gate) == "primary_v4"
    assert module.coverage_decision(pd.DataFrame({"primary_v4_exposure_allowed": [True, False]})) == "sensitivity_only"
    with pytest.raises(KeyError, match="coverage gate"):
        module.coverage_decision(pd.DataFrame())

    selected, tests, audits = module.run_prespecified_comparison(fake, match, features, gate)
    assert set(selected["exposure_scope"]) == set(module.PRIMARY_SCOPES)
    assert len(tests) == 4
    assert len(audits) == 2
