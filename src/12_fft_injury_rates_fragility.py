#!/usr/bin/env python
"""
12_fft_injury_rates_fragility.py

FFT-based temporal plausibility check for daily injury-incidence proxies,
stratified by the internal prior-injury-history labels in ``fragility_group``.

Procedure:
    1) Load player_day_panel_all_comp.csv.
    2) Merge player_day_fragility.csv and keep day-level prior-history labels
       in {tough, regular, fragile}; low_exposure days are excluded.
    3) For each fragility_group in {tough, regular, fragile}:
        - Build a daily time series from min(date) to max(date):
            * total_events          = sum(injury_event)
            * total_minutes         = sum(all_minutes_played)
            * rate_per_10000_min    = (total_events / total_minutes) * 10,000
              (set to 0 on days with total_minutes == 0)
        - Center the rate time series (subtract mean).
        - Compute FFT (rfft) and power spectrum as a calendar-periodicity check.
        - Save the full spectrum to CSV, and print the top K peaks.

Outputs:
    data/processed/results/fft_spectrum_<group>.csv

Run from repo root:

    python src/12_fft_injury_rates_fragility.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from pipeline_io import (
    merge_day_fragility,
    restrict_to_available_risk_set,
    restrict_to_fragility_risk_set,
)


TOP_K_PEAKS = 10  # how many strongest frequency peaks to print per group
CURRENT_MINUTES_COL = "all_minutes_played"


def build_daily_rates(panel: pd.DataFrame, group_label: str) -> pd.DataFrame:
    """
    Given a panel with columns:
        - date
        - injury_event
        - all_minutes_played

    Build a daily time series with columns:
        - date
        - total_events
        - total_minutes
        - rate_per_10000_min

    Ensures a complete daily grid between min(date) and max(date).
    """
    date_min = panel["date"].min()
    date_max = panel["date"].max()
    print(f"[{group_label}] Date range: {date_min.date()} .. {date_max.date()}")

    # Aggregate to daily counts and minutes
    daily = (
        panel
        .groupby("date", as_index=False)
        .agg(
            total_events=("injury_event", "sum"),
            total_minutes=(CURRENT_MINUTES_COL, "sum"),
        )
    )

    # Build full daily grid
    full_dates = pd.DataFrame(
        {"date": pd.date_range(start=date_min, end=date_max, freq="D")}
    )

    daily = full_dates.merge(daily, on="date", how="left")
    daily[["total_events", "total_minutes"]] = daily[
        ["total_events", "total_minutes"]
    ].fillna(0)

    # Injury rate per 10,000 minutes (0 if total_minutes == 0)
    daily["rate_per_10000_min"] = 0.0
    mask_minutes = daily["total_minutes"] > 0
    daily.loc[mask_minutes, "rate_per_10000_min"] = (
        daily.loc[mask_minutes, "total_events"]
        / daily.loc[mask_minutes, "total_minutes"]
        * 10000.0
    )

    return daily


def compute_fft_spectrum(
    daily: pd.DataFrame,
    group_label: str,
    out_dir: Path,
) -> None:
    """
    Compute FFT of rate_per_10000_min in the 'daily' DataFrame.

    Saves:
        fft_spectrum_<group_label>.csv

    Columns:
        - frequency_per_day
        - period_days
        - power
    """
    y = daily["rate_per_10000_min"].values.astype(float)
    # Center the series (remove mean) to focus on oscillations rather than level
    y_centered = y - np.mean(y)

    n = len(y_centered)
    dt = 1.0  # time step in days

    # Real FFT
    freq = np.fft.rfftfreq(n, d=dt)  # cycles per day
    fft_vals = np.fft.rfft(y_centered)
    power = np.abs(fft_vals) ** 2

    spectrum = pd.DataFrame(
        {
            "frequency_per_day": freq,
            "power": power,
        }
    )

    # Convert frequency to period in days (avoid division by zero at freq=0)
    spectrum["period_days"] = np.where(
        spectrum["frequency_per_day"] > 0,
        1.0 / spectrum["frequency_per_day"],
        np.inf,
    )

    # Save full spectrum
    out_path = out_dir / f"fft_spectrum_{group_label}.csv"
    spectrum.to_csv(out_path, index=False)
    print(f"[{group_label}] Saved FFT spectrum -> {out_path}")

    # Print top K non-zero-frequency peaks within a reasonable period range
    # e.g., between 2 and 90 days.
    mask_nonzero = spectrum["frequency_per_day"] > 0
    spec_nz = spectrum[mask_nonzero].copy()

    # Keep only a reasonable period band (optional but helpful for interpretation)
    spec_nz = spec_nz[
        (spec_nz["period_days"] >= 2.0) & (spec_nz["period_days"] <= 90.0)
    ]

    top_peaks = spec_nz.sort_values("power", ascending=False).head(TOP_K_PEAKS)

    print(f"\n[{group_label}] Top {TOP_K_PEAKS} FFT peaks (by power, 2–90 day periods):")
    print(
        top_peaks[["frequency_per_day", "period_days", "power"]]
        .sort_values("period_days")
        .to_string(index=False, float_format="{:.4f}".format)
    )
    print(
        f"Note: look for strong peaks near ~7 days, 3.5 days, 14 days, etc."
        " when you plot the spectrum."
    )


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    proc_dir = root / "data" / "processed"
    results_dir = proc_dir / "results"
    results_dir.mkdir(exist_ok=True)

    panel_path = proc_dir / "player_day_panel_all_comp.csv"

    print(f"Repo root: {root}")
    print(f"Loading player-day panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    print("Panel shape (all players):", panel.shape)

    print("Merging day-level prior-injury-history labels ...")
    panel = merge_day_fragility(panel, proc_dir)
    panel = restrict_to_fragility_risk_set(panel)
    panel = restrict_to_available_risk_set(panel)
    print("Panel shape after history-stratum and availability restrictions:", panel.shape)

    print("\nPrior-injury-history label counts (restricted cohort):")
    print(panel["fragility_group"].value_counts(dropna=False))

    # For each group, build daily rate series and compute FFT
    for group in ["tough", "regular", "fragile"]:
        sub = panel[panel["fragility_group"] == group].copy()
        if sub.empty:
            print(f"\nNo rows for fragility_group='{group}' after restriction; skipping.")
            continue

        print(f"\n================ FFT FOR {group.upper()} STRATUM ================")
        print(f"Sub-panel shape ({group}):", sub.shape)
        true_rate = sub["injury_event"].mean()
        print(
            f"True daily injury event rate ({group}): {true_rate:.6f} "
            "(events per player-day)"
        )

        daily = build_daily_rates(sub, group_label=group)
        compute_fft_spectrum(daily, group_label=group, out_dir=results_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
