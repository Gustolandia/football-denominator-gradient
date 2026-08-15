import pandas as pd
import pytest


def test_16_imports(load_src_module):
    module = load_src_module("16_build_match_proxy_events.py")
    assert module.PLAYER_ID_COL == "tm_player_id"
    assert module.MATCH_MINUTES_COL == "all_minutes_played"


def test_specific_injury_desc_filter(load_src_module):
    module = load_src_module("16_build_match_proxy_events.py")
    desc = pd.Series(["Hamstring injury", "unknown injury", "", None, "Not specified"])
    assert module.is_specific_injury_desc(desc).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_match_proxy_reconciliation(load_src_module):
    module = load_src_module("16_build_match_proxy_events.py")
    panel = pd.DataFrame(
        {
            "fragility_group": ["regular", "regular", "fragile", "fragile"],
            "injury_event": [1, 0, 1, 0],
            "injury_context": [
                "match_lag1_recorded_next_day",
                "none",
                "match_same_day",
                "none",
            ],
            "all_minutes_played": [0.0, 90.0, 45.0, 90.0],
            "injury_event_matchproxy_same_day": [0, 0, 1, 0],
            "injury_event_matchproxy_lag1": [0, 1, 0, 0],
            "injury_event_matchproxy": [0, 1, 1, 0],
        }
    )
    reconciliation = module.match_proxy_reconciliation(panel)
    regular = reconciliation[reconciliation["fragility_group"] == "regular"].iloc[0]
    fragile = reconciliation[reconciliation["fragility_group"] == "fragile"].iloc[0]
    overall = reconciliation[reconciliation["fragility_group"] == "overall"].iloc[0]
    assert regular["start_lag1_events"] == 1
    assert regular["matchrow_lag1_events"] == 1
    assert regular["unassigned_proxy_events"] == 0
    assert fragile["start_same_day_events"] == 1
    assert fragile["matchrow_same_day_events"] == 1
    assert overall["matchrow_proxy_events"] == 2

    with pytest.raises(KeyError):
        module.match_proxy_reconciliation(panel.drop(columns=["injury_context"]))
