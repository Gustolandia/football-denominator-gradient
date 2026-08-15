import pandas as pd


def test_compute_restriction_counts(load_src_module):
    module = load_src_module("21_panel_restriction_counts.py")
    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2],
            "minutes_played": [450, 450, 10],
        }
    )
    panel_all = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2],
            "minutes_played": [90, 0, 10],
            "all_minutes_played": [90, 45, 10],
            "injury_event_matchproxy": [1, 1, 1],
            "fragility_group": ["regular", "low_exposure", "fragile"],
            "available_for_injury_risk": [1, 1, 0],
        }
    )
    counts = module.compute_restriction_counts(panel, panel_all)
    assert counts["players_all"] == 2
    assert counts["players_risk"] == 1
    assert counts["days_risk"] == 1
    assert counts["days_matchproxy"] == 1
    assert counts["events_matchproxy"] == 1
