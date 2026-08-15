import numpy as np
import pandas as pd
import pytest


def test_23_imports(load_src_module):
    module = load_src_module("23_prior_injury_duration_next_risk.py")
    assert module.MATCH_MINUTES_COL == "all_minutes_played"
    assert module.EVENT_COL == "injury_event_matchproxy"
    assert module.GROUP_ORDER == ["regular", "fragile"]
    assert module.GROUP_DISPLAY_LABELS["fragile"] == "Higher prior-injury-history"


def test_duration_helpers(load_src_module):
    module = load_src_module("23_prior_injury_duration_next_risk.py")
    assert module.parse_duration_days({"days": 6}) == 6.0
    assert module.parse_duration_days("{'days': 61}") == 61.0
    assert np.isnan(module.parse_duration_days("{bad"))
    assert np.isnan(module.parse_duration_days(""))
    assert np.isnan(module.parse_duration_days(None))
    assert np.isnan(module.parse_duration_days(np.nan))
    assert np.isnan(module.parse_duration_days([1, 2]))
    assert np.isnan(module.parse_duration_days({"days": "bad"}))

    assert module.duration_bucket(np.nan) == "unknown duration"
    assert module.duration_bucket(6) == "<1 week"
    assert module.duration_bucket(7) == "1 week to 2 months"
    assert module.duration_bucket(60) == "1 week to 2 months"
    assert module.duration_bucket(61) == "2 months to 1 year"
    assert module.duration_bucket(365) == "2 months to 1 year"
    assert module.duration_bucket(366) == ">1 year"

    assert module.classify_injury_type("Hamstring injury") == "muscle/tendon"
    assert module.classify_injury_type("ACL rupture") == "joint/ligament"
    assert module.classify_injury_type("Metatarsal fracture") == "bone/fracture"
    assert module.classify_injury_type("Concussion") == "head/concussion"
    assert module.classify_injury_type("Covid-19") == "illness/other medical"
    assert module.classify_injury_type("unknown injury") == "unknown"
    assert module.classify_injury_type("") == "unknown"
    assert module.classify_injury_type("bruise") == "other/unspecified"

    rates = module.safe_rates(1, 1000)
    assert rates["events_per_10000_min"] == pytest.approx(10.0)
    assert rates["events_per_1000_match_hours"] == pytest.approx(60.0)
    assert rates["events_per_1000_match_hours_ci_low"] < 60.0
    assert rates["events_per_1000_match_hours_ci_high"] > 60.0
    zero_event = module.safe_rates(0, 1000)
    assert zero_event["events_per_10000_min_ci_low"] == 0.0
    assert zero_event["events_per_10000_min_ci_high"] > 0.0
    assert np.isnan(module.safe_rates(1, 0)["events_per_10000_min"])
    assert np.isnan(module.safe_rates(1, np.nan)["events_per_1000_match_hours"])


def test_prepare_injury_spells(load_src_module):
    module = load_src_module("23_prior_injury_duration_next_risk.py")
    injuries = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2, None],
            "injury_spell_id": [10, 11, 20, 99],
            "start_date": ["2024-01-01", "2024-03-01", "2024-04-01", "2024-01-01"],
            "end_date": ["2024-01-06", "2024-05-15", None, "2024-01-02"],
            "durationDetails": ["{'days': 5}", "{'days': None}", "{'days': 400}", "{'days': 1}"],
            "injury_desc": ["hamstring", "knee", "fracture", "illness"],
        }
    )
    spells = module.prepare_injury_spells(injuries)
    assert spells["tm_player_id"].tolist() == [1, 1, 2]
    assert spells.loc[spells["injury_spell_id"] == 10, "prior_injury_duration_bucket"].iloc[0] == "<1 week"
    assert spells.loc[spells["injury_spell_id"] == 11, "prior_injury_duration_bucket"].iloc[0] == "2 months to 1 year"
    assert spells.loc[spells["injury_spell_id"] == 20, "prior_injury_duration_bucket"].iloc[0] == ">1 year"
    assert spells.loc[spells["injury_spell_id"] == 10, "prior_injury_type"].iloc[0] == "muscle/tendon"

    no_end = module.prepare_injury_spells(
        pd.DataFrame(
            {
                "tm_player_id": [4],
                "injury_spell_id": [40],
                "start_date": ["2024-01-01"],
                "durationDetails": ["{'days': 8}"],
            }
        )
    )
    assert no_end.loc[0, "prior_injury_end_date"] == pd.Timestamp("2024-01-09")

    canonical = module.prepare_injury_spells(
        pd.DataFrame(
            {
                "tm_player_id": [5],
                "injury_episode_id": ["5:2024-01-01:1"],
                "start_date": ["2024-01-01"],
                "end_date": ["2024-01-12"],
                "duration_days": [12],
                "injury_desc": ["calf strain"],
            }
        )
    )
    assert canonical.loc[0, "prior_injury_duration_days"] == 12
    assert canonical.loc[0, "injury_spell_id"] == "5:2024-01-01:1"

    with pytest.raises(KeyError):
        module.prepare_injury_spells(pd.DataFrame({"tm_player_id": [1]}))


def test_frequency_only_group(load_src_module):
    module = load_src_module("23_prior_injury_duration_next_risk.py")
    panel = pd.DataFrame(
        {
            "prior_minutes_played": [899, 900, 900],
            "prior_injuries_per_10000min": [0, 5, 11],
            "q3_freq": [10, 10, 10],
        }
    )
    out = module.add_frequency_only_group(panel)
    assert out["frequency_only_group"].tolist() == [
        "low_exposure",
        "regular",
        "fragile",
    ]
    with pytest.raises(KeyError):
        module.add_frequency_only_group(panel.drop(columns=["q3_freq"]))


def test_attach_most_recent_prior_injury(load_src_module):
    module = load_src_module("23_prior_injury_duration_next_risk.py")
    matches = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 2],
            "date": pd.to_datetime(["2024-01-05", "2024-01-07", "2024-06-01", "2024-02-01"]),
            "all_minutes_played": [90, 90, 90, 90],
        }
    )
    spells = pd.DataFrame(
        {
            "tm_player_id": [1, 1],
            "injury_spell_id": [10, 11],
            "prior_injury_end_date": pd.to_datetime(["2024-01-06", "2024-05-15"]),
            "prior_injury_duration_days": [5.0, 75.0],
            "prior_injury_duration_bucket": ["<1 week", "2 months to 1 year"],
            "prior_injury_type": ["muscle/tendon", "joint/ligament"],
        }
    )
    out = module.attach_most_recent_prior_injury(matches, spells)
    assert out.loc[out["date"].eq(pd.Timestamp("2024-01-05")), "prior_injury_duration_bucket"].iloc[0] == "no prior completed injury"
    assert out.loc[out["date"].eq(pd.Timestamp("2024-01-07")), "prior_injury_duration_bucket"].iloc[0] == "<1 week"
    assert out.loc[out["date"].eq(pd.Timestamp("2024-06-01")), "prior_injury_duration_bucket"].iloc[0] == "2 months to 1 year"
    assert out.loc[out["tm_player_id"].eq(2), "prior_injury_duration_bucket"].iloc[0] == "no prior completed injury"
    assert out.loc[out["tm_player_id"].eq(2), "prior_injury_type"].iloc[0] == "none"

    with pytest.raises(KeyError):
        module.attach_most_recent_prior_injury(matches.drop(columns=["date"]), spells)
    with pytest.raises(KeyError):
        module.attach_most_recent_prior_injury(matches, spells.drop(columns=["prior_injury_end_date"]))

    empty = module.attach_most_recent_prior_injury(matches.iloc[0:0], spells)
    assert empty.empty


def test_build_duration_rate_table(load_src_module):
    module = load_src_module("23_prior_injury_duration_next_risk.py")
    panel = pd.DataFrame(
        {
            "fragility_group": ["regular", "fragile", "regular"],
            "prior_injury_duration_bucket": [
                "no prior completed injury",
                "1 week to 2 months",
                "1 week to 2 months",
            ],
            "all_minutes_played": [100.0, 50.0, 50.0],
            "injury_event_matchproxy": [1, 1, 0],
        }
    )
    out = module.build_duration_rate_table(panel, "fragility_group")
    reg_no_prior = out[
        (out["group"] == "regular")
        & (out["prior_injury_duration_bucket"] == "no prior completed injury")
    ].iloc[0]
    fragile_mid = out[
        (out["group"] == "fragile")
        & (out["prior_injury_duration_bucket"] == "1 week to 2 months")
    ].iloc[0]
    assert reg_no_prior["events"] == 1
    assert reg_no_prior["publication_group"] == "Intermediate prior-injury-history"
    assert reg_no_prior["events_per_1000_match_hours"] == pytest.approx(600.0)
    assert fragile_mid["events"] == 1
    assert np.isnan(
        out[
            (out["group"] == "fragile")
            & (out["prior_injury_duration_bucket"] == ">1 year")
        ]["events_per_10000_min"].iloc[0]
    )

    with pytest.raises(KeyError):
        module.build_duration_rate_table(panel.drop(columns=["all_minutes_played"]), "fragility_group")


def test_build_prior_injury_type_mix(load_src_module):
    module = load_src_module("23_prior_injury_duration_next_risk.py")
    panel = pd.DataFrame(
        {
            "prior_injury_spell_id": [1, 1, 2, np.nan],
            "prior_injury_duration_bucket": ["<1 week", "<1 week", "1 week to 2 months", "no prior completed injury"],
            "prior_injury_type": ["muscle/tendon", "muscle/tendon", "joint/ligament", "none"],
            "all_minutes_played": [90.0, 45.0, 30.0, 90.0],
            "injury_event_matchproxy": [1, 0, 1, 0],
        }
    )
    out = module.build_prior_injury_type_mix(panel)
    muscle = out[
        (out["prior_injury_duration_bucket"] == "<1 week")
        & (out["prior_injury_type"] == "muscle/tendon")
    ].iloc[0]
    assert muscle["unique_prior_spells"] == 1
    assert muscle["match_rows"] == 2
    assert muscle["events"] == 1
    assert muscle["match_row_percent_within_duration"] == pytest.approx(100.0)

    empty = module.build_prior_injury_type_mix(panel.iloc[0:0])
    assert empty.empty
    with pytest.raises(KeyError):
        module.build_prior_injury_type_mix(panel.drop(columns=["prior_injury_type"]))


def test_build_prior_duration_outputs(load_src_module):
    module = load_src_module("23_prior_injury_duration_next_risk.py")
    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2, 3],
            "date": pd.to_datetime(["2024-01-10", "2024-03-10", "2024-02-10", "2024-02-10"]),
            "fragility_group": ["regular", "regular", "fragile", "low_exposure"],
            "available_for_injury_risk": [1, 1, 1, 1],
            "all_minutes_played": [90.0, 90.0, 45.0, 90.0],
            "injury_event_matchproxy": [0, 1, 1, 0],
            "prior_minutes_played": [900, 1000, 1000, 0],
            "prior_injuries_per_10000min": [1.0, 2.0, 12.0, 0.0],
            "q3_freq": [10.0, 10.0, 10.0, 10.0],
        }
    )
    injuries = pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "injury_spell_id": [10, 20],
            "start_date": ["2024-01-01", "2024-01-01"],
            "end_date": ["2024-01-05", "2024-01-20"],
            "durationDetails": ["{'days': 4}", "{'days': 19}"],
        }
    )
    outputs = module.build_prior_duration_outputs(panel, injuries)
    assert set(outputs) == {
        "prior_injury_duration_next_risk_canonical",
        "prior_injury_duration_next_risk_frequency_only",
        "prior_injury_duration_type_mix",
    }
    canonical = outputs["prior_injury_duration_next_risk_canonical"]
    assert set(canonical["group"]) == {"regular", "fragile"}
    freq = outputs["prior_injury_duration_next_risk_frequency_only"]
    fragile_mid = freq[
        (freq["group"] == "fragile")
        & (freq["prior_injury_duration_bucket"] == "1 week to 2 months")
    ].iloc[0]
    assert fragile_mid["events"] == 1
