def test_20_imports(load_src_module):
    module = load_src_module("20_model_diagnostics.py")
    assert module.SPLINE_DF == 4


def test_restrict_to_daily_analysis_window(load_src_module):
    import pandas as pd
    import pytest

    module = load_src_module("20_model_diagnostics.py")
    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2, 2, 3],
            "all_minutes_played": [500.0, 500.0, 899.0, 0.0, 1000.0],
            "available_for_injury_risk": [1, 1, 1, 1, 0],
        }
    )
    out = module.restrict_to_daily_analysis_window(panel)
    assert out["tm_player_id"].unique().tolist() == [1]
    assert out["all_minutes_played"].sum() == 1000.0

    relaxed = module.restrict_to_daily_analysis_window(panel, min_minutes=800.0)
    assert relaxed["tm_player_id"].unique().tolist() == [1, 2]

    with pytest.raises(KeyError):
        module.restrict_to_daily_analysis_window(panel.drop(columns=["all_minutes_played"]))
