def test_14_imports_full_panel_daily_model(load_src_module):
    module = load_src_module("14_hazard_by_fragility_45min_weekly.py")
    assert module.MIN_EVENTS_FOR_GLM == 200
    assert module.CURRENT_MINUTES_COL == "all_minutes_played"
    assert "excess_minutes_last7d" not in module.REQUIRED_LOAD_COLS
    assert "week_phase_sin" in module.REQUIRED_LOAD_COLS
