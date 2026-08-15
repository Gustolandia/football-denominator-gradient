import numpy as np
import pandas as pd
import pytest


def test_22_imports(load_src_module):
    module = load_src_module("22_clinical_surveillance_bridge.py")
    assert module.MATCH_MINUTES_COL == "all_minutes_played"
    assert module.PLAYER_ID_COL == "tm_player_id"
    assert module.DURATION_BUCKETS[:3] == [
        "<1 week",
        "1 week to 2 months",
        "2 months to 1 year",
    ]


def test_duration_parsing_and_buckets(load_src_module):
    module = load_src_module("22_clinical_surveillance_bridge.py")
    assert module.parse_duration_days({"days": 10}) == 10.0
    assert module.parse_duration_days("{'days': 12}") == 12.0
    assert np.isnan(module.parse_duration_days("{bad"))
    assert np.isnan(module.parse_duration_days(""))
    assert np.isnan(module.parse_duration_days(None))
    assert np.isnan(module.parse_duration_days(np.nan))
    assert np.isnan(module.parse_duration_days(["not", "a", "dict"]))
    assert np.isnan(module.parse_duration_days({"days": "bad"}))

    assert module.duration_bucket(np.nan) == "unknown"
    assert module.duration_bucket(6) == "<1 week"
    assert module.duration_bucket(7) == "1 week to 2 months"
    assert module.duration_bucket(60) == "1 week to 2 months"
    assert module.duration_bucket(61) == "2 months to 1 year"
    assert module.duration_bucket(365) == "2 months to 1 year"
    assert module.duration_bucket(366) == ">1 year"


def test_group_spell_and_rate_helpers(load_src_module):
    module = load_src_module("22_clinical_surveillance_bridge.py")
    assert module.clinical_risk_group("fragile") == "higher_history"
    assert module.clinical_risk_group("regular") == "lower_intermediate_history"
    assert module.clinical_risk_group("tough") == "lower_intermediate_history"
    assert module.clinical_risk_group("low_exposure") == "other"

    assert module.split_spell_ids("1; 2.0; bad; ;3;1:2024-01-01:1") == [
        "1",
        "2",
        "3",
        "1:2024-01-01:1",
    ]
    assert module.split_spell_ids(None) == []
    assert module.split_spell_ids(np.nan) == []

    rates = module.safe_rates(2, 1000)
    assert rates["events_per_10000_min"] == pytest.approx(20.0)
    assert rates["events_per_1000_match_hours"] == pytest.approx(120.0)
    assert rates["events_per_10000_min_ci_low"] < 20.0
    assert rates["events_per_10000_min_ci_high"] > 20.0
    assert np.isnan(rates["events_per_1000_appearances"])
    app_rates = module.safe_rates(2, 1000, appearances=10)
    assert app_rates["events_per_1000_appearances"] == pytest.approx(200.0)
    assert app_rates["events_per_1000_appearances_ci_low"] < 200.0
    assert np.isnan(module.safe_rates(1, 0)["events_per_10000_min"])
    assert np.isnan(module.safe_rates(1, np.nan)["events_per_1000_match_hours"])
    zero = module.approximate_count_rate_interval(0, 10, 1000)
    assert zero[0] == 0.0
    assert zero[1] == 0.0
    assert zero[2] > 0.0


def test_injury_duration_lookup_and_spell_summaries(load_src_module):
    module = load_src_module("22_clinical_surveillance_bridge.py")
    assert module.split_spell_ids("1.5;invalid") == []
    injuries = pd.DataFrame(
        {
            "injury_spell_id": [1, 2, 3, 4],
            "durationDetails": [
                "{'days': 5}",
                "{'days': None}",
                "bad",
                "{'days': 400}",
            ],
            "start_date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
            "end_date": ["2024-01-06", "2024-02-11", None, "2025-05-06"],
            "missedGamesCount": [1, 2, np.nan, 10],
            "injury_desc": ["A", "B", "C", "D"],
        }
    )
    lookup = module.build_injury_duration_lookup(injuries)
    assert lookup.loc[lookup["injury_spell_id"] == "1", "duration_days"].iloc[0] == 5.0
    assert lookup.loc[lookup["injury_spell_id"] == "2", "duration_days"].iloc[0] == 10.0
    assert lookup.loc[lookup["injury_spell_id"] == "4", "duration_bucket"].iloc[0] == ">1 year"

    summary = module.summarize_spell_ids("1;2", lookup)
    assert summary["duration_days"] == 10.0
    assert summary["duration_bucket"] == "1 week to 2 months"
    assert summary["missed_games"] == 3.0
    assert summary["duration_known"] is True

    unknown_summary = module.summarize_spell_ids("3", lookup)
    assert np.isnan(unknown_summary["duration_days"])
    assert unknown_summary["duration_bucket"] == "unknown"
    assert np.isnan(unknown_summary["missed_games"])
    assert unknown_summary["duration_known"] is False

    assert module.summarize_spell_ids("", lookup)["duration_bucket"] == "unknown"
    assert module.summarize_spell_ids("99", lookup)["duration_known"] is False

    with pytest.raises(KeyError):
        module.build_injury_duration_lookup(pd.DataFrame({"x": [1]}))

    minimal = module.build_injury_duration_lookup(pd.DataFrame({"injury_spell_id": [8]}))
    assert minimal.loc[0, "duration_bucket"] == "unknown"
    assert np.isnan(minimal.loc[0, "missedGamesCount"])
    canonical = module.build_injury_duration_lookup(
        pd.DataFrame(
            {
                "injury_episode_id": ["1:2024-01-01:1"],
                "duration_days": [12],
                "missedGamesCount": [2],
                "injury_desc": ["strain"],
            }
        )
    )
    assert canonical.loc[0, "duration_days"] == 12


def test_attach_duration_metadata(load_src_module):
    module = load_src_module("22_clinical_surveillance_bridge.py")
    lookup = pd.DataFrame(
        {
            "injury_spell_id": ["10", "11"],
            "duration_days": [5.0, 30.0],
            "duration_bucket": ["<1 week", "1 week to 2 months"],
            "missedGamesCount": [1, 4],
            "injury_desc": ["Knock", "Hamstring"],
        }
    )
    match_panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "injury_spell_id": ["10", "", "11"],
            "injury_event_matchproxy_same_day": [1, 0, 0],
            "injury_event_matchproxy_lag1": [0, 1, 0],
        }
    )
    out = module.attach_matchproxy_duration_metadata(match_panel, lookup)
    assert out["matchproxy_spell_id"].tolist() == ["10", "11", ""]
    assert out["matchproxy_duration_bucket"].tolist() == [
        "<1 week",
        "1 week to 2 months",
        "unknown",
    ]

    with pytest.raises(KeyError):
        module.attach_matchproxy_duration_metadata(match_panel.drop(columns=["date"]), lookup)

    onset = module.attach_onset_duration_metadata(match_panel, lookup)
    assert onset.loc[0, "duration_bucket"] == "<1 week"
    with pytest.raises(KeyError):
        module.attach_onset_duration_metadata(pd.DataFrame({"x": [1]}), lookup)


def test_build_rate_and_summary_tables(load_src_module):
    module = load_src_module("22_clinical_surveillance_bridge.py")
    match_panel = pd.DataFrame(
        {
            "fragility_group": ["regular", "fragile", "tough"],
            "clinical_risk_group": [
                "lower_intermediate_history",
                "higher_history",
                "lower_intermediate_history",
            ],
            "all_minutes_played": [100.0, 50.0, 50.0],
            "injury_event_matchproxy": [1, 1, 0],
            "injury_event_matchproxy_same_day": [1, 0, 0],
            "injury_event_matchproxy_lag1": [0, 1, 0],
            "injury_event_matchproxy_specific": [1, 1, 0],
            "matchproxy_duration_bucket": ["<1 week", "2 months to 1 year", "unknown"],
        }
    )
    rates = module.build_match_hour_rates(match_panel)
    overall = rates[
        (rates["rate_scope"] == "same_day_plus_lag1")
        & (rates["group_kind"] == "overall")
    ].iloc[0]
    assert overall["events"] == 2
    assert overall["match_minutes"] == 200.0
    assert overall["events_per_1000_appearances"] == pytest.approx(666.6666667)

    duration_rates = module.build_duration_rate_table(
        match_panel,
        ["clinical_risk_group"],
    )
    fragile_long = duration_rates[
        (duration_rates["clinical_risk_group"] == "higher_history")
        & (duration_rates["duration_bucket"] == "2 months to 1 year")
    ].iloc[0]
    assert fragile_long["events"] == 1
    assert fragile_long["events_per_1000_match_hours"] == pytest.approx(1200.0)

    context = pd.DataFrame(
        {
            "fragility_group": ["regular", "regular"],
            "clinical_risk_group": [
                "lower_intermediate_history",
                "lower_intermediate_history",
            ],
            "injury_context": ["match_same_day", "training_or_other"],
            "duration_days": [7.0, np.nan],
            "missed_games": [2.0, np.nan],
        }
    )
    summary = module.build_duration_context_summary(context)
    assert set(summary["injury_context"]) == {"match_same_day", "training_or_other"}
    assert summary.loc[summary["injury_context"] == "match_same_day", "duration_known"].iloc[0] == 1

    with pytest.raises(KeyError):
        module.build_match_hour_rates(match_panel.drop(columns=["injury_event_matchproxy"]))


def test_build_clinical_outputs(load_src_module):
    module = load_src_module("22_clinical_surveillance_bridge.py")
    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2, 3],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"]),
            "fragility_group": ["regular", "regular", "fragile", "low_exposure"],
            "available_for_injury_risk": [1, 1, 1, 1],
            "all_minutes_played": [90.0, 0.0, 45.0, 90.0],
            "all_minutes_last_7d": [0.0, 90.0, 180.0, 0.0],
            "injury_event": [0, 1, 1, 0],
            "injury_context": ["none", "match_lag1_recorded_next_day", "match_same_day", "none"],
            "injury_spell_id": ["", "22", "23", ""],
            "injury_event_matchproxy": [1, 0, 1, 0],
            "injury_event_matchproxy_same_day": [0, 0, 1, 0],
            "injury_event_matchproxy_lag1": [1, 0, 0, 0],
            "injury_event_matchproxy_specific": [1, 0, 1, 0],
        }
    )
    injuries = pd.DataFrame(
        {
            "injury_spell_id": [22, 23],
            "durationDetails": ["{'days': 14}", "{'days': 70}"],
            "missedGamesCount": [2, 6],
            "injury_desc": ["Hamstring", "Knee"],
        }
    )
    outputs = module.build_clinical_outputs(panel, injuries)
    assert set(outputs) == {
        "clinical_match_hour_rates",
        "clinical_duration_context_summary",
        "clinical_matchproxy_duration_rates_by_group",
        "clinical_matchproxy_duration_rates_by_burden",
    }
    rates = outputs["clinical_match_hour_rates"]
    assert "clinical_risk_group" in set(rates["group_kind"])
    duration = outputs["clinical_matchproxy_duration_rates_by_group"]
    assert set(duration["clinical_risk_group"]) == {
        "higher_history",
        "lower_intermediate_history",
    }
    non_fragile_mid = duration[
        (duration["clinical_risk_group"] == "lower_intermediate_history")
        & (duration["duration_bucket"] == "1 week to 2 months")
    ].iloc[0]
    assert non_fragile_mid["events"] == 1
