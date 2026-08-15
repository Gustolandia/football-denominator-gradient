import pandas as pd
import pytest


def test_build_injury_day_table_empty_and_missing(load_src_module):
    module = load_src_module("09_build_fragility_groups.py")
    empty = module.build_injury_day_table(
        pd.DataFrame(),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
    )
    assert list(empty.columns) == [
        "tm_player_id",
        "date",
        "n_spells_today",
        "total_days_injured_today",
        "max_spell_duration_today",
    ]

    with pytest.raises(KeyError):
        module.build_injury_day_table(
            pd.DataFrame({"tm_player_id": [1]}),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-31"),
        )


def test_build_injury_day_table_aggregates_and_clips(load_src_module):
    module = load_src_module("09_build_fragility_groups.py")
    injuries = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1],
            "injury_spell_id": [10, 11, 12],
            "start_date": ["2023-12-31", "2024-01-02", "2024-01-02"],
            "end_date": ["2024-01-03", "2024-01-04", None],
        }
    )
    out = module.build_injury_day_table(
        injuries,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
    )
    row = out[out["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert row["n_spells_today"] == 1
    assert row["total_days_injured_today"] == 3
    assert row["max_spell_duration_today"] == 3

    no_end = module.build_injury_day_table(
        pd.DataFrame(
            {
                "tm_player_id": [2],
                "injury_spell_id": [20],
                "start_date": ["2024-01-05"],
            }
        ),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
    )
    assert no_end.loc[0, "total_days_injured_today"] == 1

    out_of_window = module.build_injury_day_table(
        pd.DataFrame(
            {
                "tm_player_id": [2],
                "injury_spell_id": [21],
                "start_date": ["2025-01-05"],
            }
        ),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
    )
    assert out_of_window.empty


def test_thresholds_and_labels(load_src_module):
    module = load_src_module("09_build_fragility_groups.py")
    empty_thresholds = module.estimate_thresholds(
        pd.DataFrame(
            {
                "tm_player_id": [1],
                "date": pd.to_datetime(["2024-01-01"]),
                "prior_minutes_played": [0],
                "prior_injuries_per_10000min": [0],
                "prior_max_spell_duration_days": [0],
            }
        )
    )
    assert empty_thresholds == {"q1_freq": 0.0, "q3_freq": 0.0, "q1_sev": 0.0, "q3_sev": 0.0}

    history = pd.DataFrame(
        {
            "prior_minutes_played": [0, 900, 900, 900],
            "prior_n_spells": [0, 1, 1, 3],
            "prior_injuries_per_10000min": [0.0, 0.1, 0.5, 2.0],
            "prior_max_spell_duration_days": [0.0, 1.0, 5.0, 30.0],
        }
    )
    labels = module.assign_fragility(
        history,
        {"q1_freq": 0.2, "q3_freq": 1.0, "q1_sev": 2.0, "q3_sev": 10.0},
    )
    assert labels.tolist() == ["low_exposure", "tough", "regular", "fragile"]


def test_player_day_fragility_uses_prior_history(load_src_module):
    module = load_src_module("09_build_fragility_groups.py")
    with pytest.raises(KeyError):
        module.build_player_day_fragility(pd.DataFrame({"tm_player_id": [1]}), pd.DataFrame())

    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "minutes_played": [0, 0, 0],
            "all_minutes_played": [900, 0, 0],
        }
    )
    injuries = pd.DataFrame(
        {
            "tm_player_id": [1],
            "injury_spell_id": [10],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-01-02"],
        }
    )
    out = module.build_player_day_fragility(panel, injuries)
    assert out.loc[0, "prior_n_spells"] == 0
    assert out.loc[1, "prior_n_spells"] == 1
    assert out.loc[1, "prior_minutes_played"] == 900
    latest = module.latest_player_fragility(out)
    assert latest.loc[0, "last_date"] == pd.Timestamp("2024-01-03")
    assert "total_minutes_played" in latest.columns


def test_pre_entry_history_and_all_club_minutes_are_strictly_prior(load_src_module):
    module = load_src_module("09_build_fragility_groups.py")
    appearances = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 99],
            "date": ["2024-01-01", "2024-01-05", "2024-02-01", "2024-01-03", None],
            "minutes_played": [90, 45, 10, 0, 90],
        }
    )
    history_minutes = module.build_history_minutes(
        appearances,
        pd.Series([1, 2]),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
    )
    assert history_minutes["history_minutes_played"].tolist() == [90, 45]

    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1],
            "date": pd.to_datetime(["2024-01-05", "2024-01-06"]),
            "all_minutes_played": [45, 0],
        }
    )
    episodes = pd.DataFrame(
        {
            "tm_player_id": [1, 1],
            "injury_episode_id": ["pre", "same-day"],
            "start_date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "duration_days": [3, 2],
        }
    )
    out = module.build_player_day_fragility(
        panel,
        episodes,
        history_minutes=history_minutes,
        history_start_date=pd.Timestamp("2024-01-01"),
    )
    assert out.loc[0, "prior_minutes_played"] == 90
    assert out.loc[0, "prior_n_spells"] == 1
    assert out.loc[0, "prior_total_days_injured"] == 3
    assert out.loc[1, "prior_minutes_played"] == 135
    assert out.loc[1, "prior_n_spells"] == 2
    assert out.loc[1, "prior_max_spell_duration_days"] == 3


def test_build_history_minutes_requires_columns(load_src_module):
    module = load_src_module("09_build_fragility_groups.py")
    with pytest.raises(KeyError, match="minutes_played"):
        module.build_history_minutes(
            pd.DataFrame({"player_id": [1], "date": ["2024-01-01"]}),
            pd.Series([1]),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-31"),
        )


def test_canonical_episode_fallback_duration_and_missing_player_history(load_src_module):
    module = load_src_module("09_build_fragility_groups.py")
    canonical = pd.DataFrame(
        {
            "tm_player_id": [1],
            "injury_episode_id": ["one"],
            "start_date": ["2024-01-02"],
            "end_date": ["2024-01-04"],
        }
    )
    table = module.build_injury_day_table(
        canonical,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
    )
    assert table.loc[0, "total_days_injured_today"] == 3

    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "all_minutes_played": [0, 0],
        }
    )
    history = pd.DataFrame(
        {
            "tm_player_id": [1],
            "date": pd.to_datetime(["2024-01-01"]),
            "history_minutes_played": [90],
        }
    )
    episodes = canonical.assign(duration_days=3)
    out = module.build_player_day_fragility(
        panel,
        episodes,
        history_minutes=history,
        history_start_date=pd.Timestamp("2024-01-01"),
    )
    assert out.loc[out["tm_player_id"] == 2, "prior_minutes_played"].iloc[0] == 0
    assert out.loc[out["tm_player_id"] == 2, "prior_n_spells"].iloc[0] == 0
