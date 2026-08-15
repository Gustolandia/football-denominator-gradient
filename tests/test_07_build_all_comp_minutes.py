from pathlib import Path

import pandas as pd
import pytest


def test_build_all_comp_minutes_rolls_prior_days(load_src_module, tmp_path):
    module = load_src_module("07_build_all_comp_minutes.py")
    player_day = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2],
            "date": pd.to_datetime(["2024-01-07", "2024-01-08", "2024-01-08"]),
            "minutes_last_7d": [0, 0, 0],
        }
    )
    apps = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 99],
            "game_id": [10, 11, 12, 11],
            "player_club_id": [100, 100, 999, 100],
            "date": ["2024-01-01", "2024-01-07", "2024-01-07", "2024-01-07"],
            "minutes_played": [90, 45, 90, 90],
        }
    )
    apps_path = tmp_path / "appearances.csv"
    apps.to_csv(apps_path, index=False)
    games = pd.DataFrame(
        {
            "game_id": [1, 10, 11, 12],
            "competition_id": ["GB1", "FAC", "CL", "ES1"],
            "season": [2023, 2023, 2023, 2023],
            "home_club_id": [100, 100, 100, 999],
            "away_club_id": [200, 300, 400, 888],
        }
    )
    games_path = tmp_path / "games.csv"
    games.to_csv(games_path, index=False)

    out = module.build_all_comp_minutes(player_day, Path(apps_path), games_path)
    p1_jan8 = out[(out["tm_player_id"] == 1) & (out["date"] == pd.Timestamp("2024-01-08"))]
    p2_jan8 = out[(out["tm_player_id"] == 2) & (out["date"] == pd.Timestamp("2024-01-08"))]
    assert float(p1_jan8["all_minutes_last_7d"].iloc[0]) == 135.0
    assert int(p1_jan8["all_games_last_7d"].iloc[0]) == 2
    assert float(p1_jan8["all_minutes_played"].iloc[0]) == 0.0
    assert float(p2_jan8["all_minutes_last_7d"].iloc[0]) == 0.0
    p1_jan7 = out[(out["tm_player_id"] == 1) & (out["date"] == pd.Timestamp("2024-01-07"))]
    assert float(p1_jan7["all_minutes_played"].iloc[0]) == 45.0
    assert int(p1_jan7["all_games_played"].iloc[0]) == 1


def test_build_all_comp_minutes_requires_date(load_src_module, tmp_path):
    module = load_src_module("07_build_all_comp_minutes.py")
    player_day = pd.DataFrame(
        {"tm_player_id": [1], "date": pd.to_datetime(["2024-01-01"]), "minutes_last_7d": [0]}
    )
    apps_path = tmp_path / "appearances.csv"
    games_path = tmp_path / "games.csv"
    pd.DataFrame({"player_id": [1], "game_id": [1], "player_club_id": [100], "minutes_played": [90]}).to_csv(apps_path, index=False)
    pd.DataFrame(
        {
            "game_id": [1],
            "competition_id": ["GB1"],
            "season": [2023],
            "home_club_id": [100],
            "away_club_id": [200],
        }
    ).to_csv(games_path, index=False)
    with pytest.raises(ValueError, match="date"):
        module.build_all_comp_minutes(player_day, apps_path, games_path)


def test_restrict_to_epl_club_seasons_requires_club(load_src_module, tmp_path):
    module = load_src_module("07_build_all_comp_minutes.py")
    games_path = tmp_path / "games.csv"
    pd.DataFrame(
        {
            "game_id": [1],
            "competition_id": ["GB1"],
            "season": [2023],
            "home_club_id": [100],
            "away_club_id": [200],
        }
    ).to_csv(games_path, index=False)
    with pytest.raises(ValueError, match="player_club_id"):
        module.restrict_to_epl_club_seasons(pd.DataFrame({"game_id": [1]}), games_path)


def test_load_observed_days_and_expand_risk_set(load_src_module, tmp_path):
    module = load_src_module("07_build_all_comp_minutes.py")
    apps_path = tmp_path / "appearances.csv"
    pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 99],
            "date": ["2024-01-01", "2024-01-03", "2024-02-01", "2024-01-03", None],
            "minutes_played": [30, 45, 90, 0, 10],
        }
    ).to_csv(apps_path, index=False)
    observed = module.load_observed_appearance_days(
        apps_path,
        pd.Series([1, 2]),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
    )
    assert observed["date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-03"),
    ]

    player_day = pd.DataFrame(
        {
            "fbref_player_id": [1, 1],
            "tm_player_id": [1, 1],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "minutes_played": [0, 90],
            "injury_event": [0, 0],
            "injury_unavailable": [0, 1],
            "available_for_injury_risk": [1, 0],
            "games_played": [0, 1],
            "games_last_7d": [0, 0],
            "minutes_last_7d": [0, 0],
        }
    )
    all_players = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "all_minutes_played": [30, 0, 90],
        }
    )
    injuries = pd.DataFrame(
        {
            "tm_player_id": [1],
            "injury_spell_id": [10],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-01-10"],
            "injury_desc": ["strain"],
        }
    )
    rebuilt, episodes = module.expand_player_day_risk_set(
        player_day, all_players, injuries, observed
    )
    jan1 = rebuilt[rebuilt["date"] == pd.Timestamp("2024-01-01")].iloc[0]
    jan3 = rebuilt[rebuilt["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert jan1["row_added_by_all_comp_span"] == 1
    assert jan1["injury_event"] == 1
    assert jan3["available_for_injury_risk"] == 1
    assert episodes.loc[0, "end_date"] == pd.Timestamp("2024-01-02")
    assert rebuilt["minutes_last_7d"].tolist() == [0.0, 0.0, 0.0]


def test_expand_risk_set_requires_columns_and_adds_identity(load_src_module):
    module = load_src_module("07_build_all_comp_minutes.py")
    with pytest.raises(KeyError, match="minutes_played"):
        module.expand_player_day_risk_set(
            pd.DataFrame({"tm_player_id": [1], "date": ["2024-01-01"]}),
            pd.DataFrame(
                {
                    "tm_player_id": [1],
                    "date": ["2024-01-01"],
                    "all_minutes_played": [0],
                }
            ),
            pd.DataFrame(columns=["tm_player_id", "start_date"]),
            pd.DataFrame(columns=["tm_player_id", "date"]),
        )
    with pytest.raises(KeyError, match="all_minutes_played"):
        module.expand_player_day_risk_set(
            pd.DataFrame(
                {
                    "tm_player_id": [1],
                    "date": ["2024-01-01"],
                    "minutes_played": [0],
                }
            ),
            pd.DataFrame({"tm_player_id": [1], "date": ["2024-01-01"]}),
            pd.DataFrame(columns=["tm_player_id", "start_date"]),
            pd.DataFrame(columns=["tm_player_id", "date"]),
        )

    rebuilt, episodes = module.expand_player_day_risk_set(
        pd.DataFrame(
            {
                "tm_player_id": [1],
                "date": pd.to_datetime(["2024-01-01"]),
                "minutes_played": [0],
            }
        ),
        pd.DataFrame(
            {
                "tm_player_id": [1],
                "date": pd.to_datetime(["2024-01-01"]),
                "all_minutes_played": [0],
            }
        ),
        pd.DataFrame(columns=["tm_player_id", "start_date"]),
        pd.DataFrame(columns=["tm_player_id", "date"]),
    )
    assert rebuilt.loc[0, "fbref_player_id"] == 1
    assert episodes.empty
