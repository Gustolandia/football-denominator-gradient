import pandas as pd
import pytest


def test_make_5min_bins_and_empty(load_src_module):
    module = load_src_module("11_hazard_5min_all_comp_and_fragility.py")
    binned, labels = module.make_5min_bins(pd.Series([0, 1, 6]))
    assert labels == ["0-5", "5-10"]
    assert str(binned.iloc[0]) == "0-5"
    with pytest.raises(ValueError):
        module.make_5min_bins(pd.Series([], dtype=float))


def test_compute_crude_rates_and_remove_stale(load_src_module, tmp_path):
    module = load_src_module("11_hazard_5min_all_comp_and_fragility.py")
    stale = tmp_path / "glm_or_all_minutes7d_bins_5min_tough.csv"
    stale.write_text("old", encoding="utf-8")
    removed = module.remove_stale_5min_glm_outputs(tmp_path)
    assert stale in removed
    assert not stale.exists()

    panel = pd.DataFrame(
        {
            "all_minutes_last_7d": [0, 6],
            "injury_event": [0, 1],
            "all_minutes_played": [90, 45],
        }
    )
    perday, perminute = module.compute_crude_rates(panel, "demo", tmp_path)
    assert (tmp_path / "hazard_5min_demo_perday.csv").exists()
    assert perday["n_events"].sum() == 1
    assert perminute["total_minutes"].sum() == 135
