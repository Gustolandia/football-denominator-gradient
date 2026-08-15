import numpy as np
import pandas as pd


def test_daily_rates_and_fft(load_src_module, tmp_path):
    module = load_src_module("12_fft_injury_rates_fragility.py")
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "injury_event": [1, 0],
            "all_minutes_played": [100, 0],
        }
    )
    daily = module.build_daily_rates(panel, "demo")
    assert len(daily) == 3
    assert daily.loc[0, "rate_per_10000_min"] == 100.0
    assert daily.loc[1, "rate_per_10000_min"] == 0.0
    module.compute_fft_spectrum(daily, "demo", tmp_path)
    out = pd.read_csv(tmp_path / "fft_spectrum_demo.csv")
    assert np.isinf(out.loc[0, "period_days"])
