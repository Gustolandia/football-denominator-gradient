"""Tests for national matchday-status features and uncertainty outputs."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def records_frame():
    return pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 1, 2],
            "game_id": ["a", "b", "c", "d", "e"],
            "date": ["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-12", "2024-01-01"],
            "participation_state": ["not in squad", "in squad", "played", "injured", "mystery"],
            "competition_type_id": [11, 11, 19, 11, 99],
            "competition_id": ["FS", "Q", "Q", "Q", "X"],
        }
    )


def test_classification_and_ledger(load_src_module):
    module = load_src_module("30_national_status_analysis.py")
    assert module.classify_national_record("played", 11, "FS") == "played_senior_friendly"
    assert module.classify_national_record("in squad", 17, "U21") == "squad_only_youth_or_olympic_competitive"
    assert module.classify_national_record("absent", None, "X") == "recorded_unavailable_unknown_competitive"
    assert module.classify_national_record("not in squad", 11, "Q") == "not_in_squad_senior_competitive"
    assert module.classify_national_record("other", 99, "Q") == "other_unknown_competitive"
    appearances = pd.DataFrame(
        {
            "game_id": ["c"],
            "independent_schedule_verified": [True],
            "external_city": ["Paris"],
        }
    )
    ledger = module.build_status_ledger(records_frame(), appearances)
    assert ledger.loc[ledger.game_id.eq("c"), "external_city"].item() == "Paris"
    assert not ledger.loc[ledger.game_id.eq("a"), "independent_schedule_verified"].item()
    plain = module.build_status_ledger(records_frame(), pd.DataFrame())
    assert not plain.independent_schedule_verified.any()
    with pytest.raises(KeyError, match="national record audit"):
        module.build_status_ledger(records_frame().drop(columns="date"), appearances)


def test_numbered_module_loader(load_src_module, monkeypatch):
    module = load_src_module("30_national_status_analysis.py")
    loaded = module.load_module(Path(module.__file__), "status_module_copy")
    assert loaded.classify_national_record("played", 11, "Q") == "played_senior_competitive"
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *_: None)
    with pytest.raises(ImportError, match="Cannot load module"):
        module.load_module(Path(module.__file__), "missing_spec")
    monkeypatch.setattr(
        module.importlib.util,
        "spec_from_file_location",
        lambda *_: SimpleNamespace(loader=None),
    )
    with pytest.raises(ImportError, match="Cannot load module"):
        module.load_module(Path(module.__file__), "missing_loader")


def test_status_windows_hierarchy_and_attachment(load_src_module):
    module = load_src_module("30_national_status_analysis.py")
    ledger = module.build_status_ledger(records_frame(), pd.DataFrame())
    targets = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 1, 2, 3],
            "date": ["2024-01-04", "2024-01-08", "2024-01-11", "2024-01-13", "2024-01-03", "2024-01-03"],
        }
    )
    features = module.build_status_features(ledger, targets, windows=(7,))
    assert features.loc[0, "national_status_last_7d"] == "explicitly_not_in_squad"
    assert features.loc[1, "national_status_last_7d"] == "squad_only"
    assert features.loc[2, "national_status_last_7d"] == "played"
    assert features.loc[3, "national_status_last_7d"] == "played"
    assert features.loc[5, "national_status_last_7d"] == "no_recent_senior_record"
    attached = module.attach_status_features(
        targets.assign(minutes_played=90, injury_event_matchproxy=0, fragility_group="regular"),
        features,
    )
    assert attached["national_status_last_7d"].notna().all()
    with pytest.raises(ValueError, match="duplicate"):
        module.attach_status_features(attached, pd.concat([features, features.iloc[[0]]]))
    with pytest.raises(KeyError, match="status ledger"):
        module.build_status_features(ledger.drop(columns="date"), targets)
    with pytest.raises(KeyError, match="target matches"):
        module.build_status_features(ledger, targets.drop(columns="date"))


def test_rate_table_model_preparation_formula_and_specifications(load_src_module):
    module = load_src_module("30_national_status_analysis.py")
    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2],
            "minutes_played": [60, 30, 90],
            "injury_event_matchproxy": [1, 0, 0],
            "fragility_group": ["regular", "regular", "fragile"],
            "national_status_last_7d": ["played", "played", "no_recent_senior_record"],
        }
    )
    rates = module.status_rate_table(panel)
    played = rates.loc[rates.national_status.eq("played")].iloc[0]
    assert played.events_per_1000_match_hours == pytest.approx(1000 / 1.5)
    assert played.ci_low < played.events_per_1000_match_hours < played.ci_high
    with pytest.raises(KeyError, match="status panel"):
        module.status_rate_table(panel.drop(columns="minutes_played"))

    class Base:
        @staticmethod
        def prepare_model_frame(source, event_col, group_col):
            out = source.copy()
            out["model_group"] = out[group_col]
            out["log_minutes_played"] = np.log(out.minutes_played)
            return out

        @staticmethod
        def spline_basis_expression(_maximum):
            return "all_minutes_last_7d"

    model_panel = panel.assign(all_minutes_last_7d=0, week_phase_sin=0, week_phase_cos=0, halfweek_phase_sin=0, halfweek_phase_cos=0)
    prepared = module.prepare_status_model_frame(
        Base, model_panel, 7, "injury_event_matchproxy", "played"
    )
    assert prepared.higher_history.sum() == 1
    assert "recent_status" in module._status_formula(Base, prepared, "injury_event_matchproxy", "")
    with pytest.raises(ValueError, match="Unknown model exposure"):
        module.prepare_status_model_frame(
            Base, model_panel, 7, "injury_event_matchproxy", "training"
        )
    assert len(module.status_specifications()) == 11


def test_linear_contrasts_and_multiplicity(load_src_module):
    module = load_src_module("30_national_status_analysis.py")
    names = ["Intercept", "recent_status", "recent_status:higher_history"]
    params = pd.Series(np.linspace(0.0, 0.8, len(names)), index=names)
    covariance = pd.DataFrame(np.eye(len(names)) * 0.01, index=names, columns=names)
    ratio = module.linear_ratio(params, covariance, {names[1]: 1})
    assert ratio["estimate"] > 1
    with pytest.raises(KeyError, match="missing contrast"):
        module.linear_ratio(params, covariance, {"missing": 1})
    zero_cov = covariance * 0
    assert np.isnan(module.linear_ratio(params * 0, zero_cov, {names[1]: 1})["p_value"])
    result = SimpleNamespace(params=params, cov_params=lambda: covariance)
    contrasts = module.status_contrasts(result, "played")
    assert len(contrasts) == 3
    assert "played_history_interaction" in set(contrasts.contrast_id)
    sparse = module.sparse_contrasts("squad_only")
    assert sparse.estimate.isna().all()
    support = pd.DataFrame(
        {
            "model_group": ["regular", "fragile"],
            "n_exposed_events": [5, 6],
            "n_exposed_rows": [50, 60],
        }
    )
    assert module.support_is_adequate(support)
    assert not module.support_is_adequate(support.iloc[[0]])
    assert not module.support_is_adequate(
        support.assign(n_exposed_events=[4, 6])
    )
    frame = pd.DataFrame(
        {
            "model_group": ["regular", "regular", "fragile", "fragile"],
            "recent_status": [0, 1, 0, 1],
            "injury_event_matchproxy": [0, 1, 1, 1],
        }
    )
    counted = module.status_support(frame, "injury_event_matchproxy")
    assert counted.n_exposed_events.sum() == 2
    adjusted = module.add_status_multiplicity(contrasts)
    assert adjusted.p_holm_status_family.notna().all()
    empty = module.add_status_multiplicity(pd.DataFrame({"p_value": [np.nan]}))
    assert empty.p_holm_status_family.isna().all()
    with pytest.raises(KeyError, match="status results"):
        module.add_status_multiplicity(pd.DataFrame())
