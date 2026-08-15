import pandas as pd
import pytest


def test_add_load_features_and_missing(load_src_module):
    module = load_src_module("13_max_daily_load_features.py")
    with pytest.raises(KeyError, match="all_minutes_played"):
        module.add_load_features(pd.DataFrame({"tm_player_id": [1]}))

    with pytest.raises(KeyError, match="required"):
        module.add_load_features(
            pd.DataFrame(
                {
                    "tm_player_id": [1],
                    "date": pd.to_datetime(["2024-01-01"]),
                    "all_minutes_played": [90],
                }
            )
        )

    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 1],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-20"]
            ),
            "all_games_played": [1, 0, 1, 1],
            "all_minutes_played": [100, 0, 45, 90],
            "all_minutes_last_7d": [0, 100, 150, 0],
            "all_games_last_7d": [0, 1, 1, 0],
        }
    )
    out = module.add_load_features(panel)
    assert out.loc[1, "minutes_yesterday"] == 100
    assert out.loc[2, "minutes_last_match"] == 100
    assert out.loc[0, "recovery_interval_bin"] == ">14 days/no prior match"
    assert pd.isna(out.loc[0, "days_since_last_match"])
    assert out.loc[1, "days_since_last_match"] == 1
    assert out.loc[1, "prior_match_within_3d"] == 1
    assert out.loc[2, "prior_match_within_5d"] == 1
    assert out.loc[3, "days_since_last_match"] == 17
    assert out.loc[3, "zero_burden_long_rest"] == 1
    assert out.loc[2, "any_day_last7_full_match"] == 1
    assert out.loc[2, "excess_minutes_last7d"] == 60
    assert "week_phase_sin" in out.columns
