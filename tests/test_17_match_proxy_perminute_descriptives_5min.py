import pandas as pd
import pytest


def test_bins_and_perminute_table(load_src_module, tmp_path):
    module = load_src_module("17_match_proxy_perminute_descriptives_5min.py")
    binned, labels = module.make_5min_bins(pd.Series([0, 8]))
    assert labels == ["0-5", "5-10"]
    assert str(binned.iloc[1]) == "5-10"
    with pytest.raises(ValueError):
        module.make_5min_bins(pd.Series([], dtype=float))

    df = pd.DataFrame(
        {
            "all_minutes7d_bin_5min": pd.Categorical(["0-5", "0-5"]),
            "injury_event_matchproxy": [1, 0],
            "all_minutes_played": [90, 90],
        }
    )
    out = module.perminute_table(df, "demo", tmp_path)
    assert out.loc[0, "events_per_10000_min"] == pytest.approx(55.5555556)
    assert (tmp_path / "matchproxy_hazard_5min_demo_perminute.csv").exists()
