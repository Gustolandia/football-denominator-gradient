#!/usr/bin/env python
"""
11_hazard_5min_all_comp_and_fragility.py

High-resolution descriptive injury rates by recent load (ALL competitions),
using 5-minute bins of all_minutes_last_7d, overall and stratified by
fragility_group.

Key ideas:
- Overall and group-specific tables use the dynamic prior-history risk set:
    fragility_group in {tough, regular, fragile}
  This excludes low_exposure player-days without excluding whole players once
  and forever.

- For the restricted cohort, we compute:
    1) Crude per-day injury rate by 5-min bins of all_minutes_last_7d.
    2) Crude match-day per-minute injury rate by 5-min bins
       (events per 10,000 minutes).

- Then we stratify by fragility_group:
    - tough
    - regular
    - fragile
  using dynamic, prior-history-only labels from player_day_fragility.csv.

Why this script is descriptive only:
    5-minute burden bins are intentionally high resolution, but many extreme
    bins are sparse and can contain zero events. Dummy-bin GLMs on those bins
    suffer perfect separation and produce unstable/infinite odds ratios. The
    inferential burden models therefore live in the 45-minute and spline scripts;
    this script removes stale 5-minute GLM CSVs if older runs created them.

Inputs:
    data/processed/player_day_panel_all_comp.csv
    data/processed/player_day_fragility.csv

Outputs (all under data/processed/results):

    # Descriptive 5-min crude rates
    hazard_5min_overall_perday.csv
    hazard_5min_overall_perminute.csv
    hazard_5min_<group>_perday.csv       for group in {tough, regular, fragile}
    hazard_5min_<group>_perminute.csv    for group in {tough, regular, fragile}

Run from repo root:

    python src/11_hazard_5min_all_comp_and_fragility.py
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from pipeline_io import (
    merge_day_fragility,
    restrict_to_available_risk_set,
    restrict_to_fragility_risk_set,
)


ANALYSIS_GROUPS = ["tough", "regular", "fragile"]
CURRENT_MINUTES_COL = "all_minutes_played"
STALE_5MIN_GLM_FILES = [
    f"{prefix}_all_minutes7d_bins_5min_{group}.csv"
    for group in ANALYSIS_GROUPS
    for prefix in ("glm_or", "glm_predicted_probs")
]


def make_5min_bins(series: pd.Series) -> Tuple[pd.Series, List[str]]:
    """
    Create 5-minute bins for a non-negative series of all_minutes_last_7d.

    Returns:
        binned_series (Categorical),
        labels (list of str) used.
    """
    if series.empty:
        raise ValueError("Cannot create 5-minute bins for an empty series.")

    max_val = float(series.max())
    # Round up to nearest 5
    upper = max(5, int(np.ceil(max_val / 5.0) * 5))
    edges = np.arange(0, upper + 5, 5)  # e.g. 0,5,10,...,upper
    labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]

    binned = pd.cut(
        series,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return binned, labels


def remove_stale_5min_glm_outputs(out_dir: Path) -> List[Path]:
    """Delete obsolete separated 5-minute GLM outputs from previous runs."""
    removed: List[Path] = []
    for file_name in STALE_5MIN_GLM_FILES:
        path = out_dir / file_name
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def compute_crude_rates(
    panel: pd.DataFrame, label_prefix: str, out_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a player-day panel with:
        - injury_event (0/1)
        - all_minutes_played (all-competition match minutes)
        - all_minutes_last_7d

    1) Create 5-min bins of all_minutes_last_7d -> all_minutes7d_bin_5min
    2) Compute per-day crude injury rate by bin
    3) On match days (all_minutes_played > 0), compute per-minute crude rates by bin

    Save two CSVs:
        hazard_5min_<label_prefix>_perday.csv
        hazard_5min_<label_prefix>_perminute.csv
    """
    # Drop missing load values (should be extremely rare after preprocessing)
    panel = panel.dropna(subset=["all_minutes_last_7d"]).copy()

    # 5-min bins of all_minutes_last_7d
    panel["all_minutes7d_bin_5min"], labels = make_5min_bins(
        panel["all_minutes_last_7d"]
    )

    print(f"\nBin counts for all_minutes_last_7d (5-min bins) [{label_prefix}]:")
    print(panel["all_minutes7d_bin_5min"].value_counts().sort_index())

    # --- Per-day crude rates ---
    perday = (
        panel.groupby("all_minutes7d_bin_5min", dropna=False, observed=False)
        .agg(
            n_days=("injury_event", "size"),
            n_events=("injury_event", "sum"),
        )
        .assign(injury_rate=lambda d: d["n_events"] / d["n_days"])
        .reset_index()
        .rename(columns={"all_minutes7d_bin_5min": "all_minutes7d_bin_5min"})
    )

    print(f"\nCrude per-day injury rate by 5-min load bins [{label_prefix}]:")
    print(perday.head(20))  # print first 20 bins for sanity

    # --- Per-minute crude rates on match days ---
    match_days = panel[panel[CURRENT_MINUTES_COL] > 0].copy()

    perminute = (
        match_days.groupby("all_minutes7d_bin_5min", dropna=False, observed=False)
        .agg(
            total_events=("injury_event", "sum"),
            total_minutes=(CURRENT_MINUTES_COL, "sum"),
        )
        .assign(
            events_per_minute=lambda d: d["total_events"] / d["total_minutes"],
        )
    )
    perminute["events_per_10000_min"] = perminute["events_per_minute"] * 10000.0
    perminute = perminute.reset_index().rename(
        columns={"all_minutes7d_bin_5min": "all_minutes7d_bin_5min"}
    )

    print(
        f"\nCrude match-day per-minute injury rate by 5-min load bins "
        f"[{label_prefix}]:"
    )
    print(perminute.head(20))

    # --- Save ---
    perday_out = out_dir / f"hazard_5min_{label_prefix}_perday.csv"
    perminute_out = out_dir / f"hazard_5min_{label_prefix}_perminute.csv"

    perday.to_csv(perday_out, index=False)
    perminute.to_csv(perminute_out, index=False)

    print(f"\nSaved {label_prefix} 5-min crude per-day rates -> {perday_out}")
    print(f"Saved {label_prefix} 5-min crude per-minute rates -> {perminute_out}")
    return perday, perminute


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

    print("Merging prior-history day-level fragility labels ...")
    panel = merge_day_fragility(panel, proc_dir)
    panel = restrict_to_fragility_risk_set(panel)
    panel = restrict_to_available_risk_set(panel)
    print("Panel shape after fragility and availability restrictions:", panel.shape)

    total_events_before = int((panel["injury_event"]).sum())
    print(f"Total injury events in restricted panel: {total_events_before}")

    print("\nFragility_group value counts in restricted panel:")
    print(panel["fragility_group"].value_counts(dropna=False))

    # ------------------------------------------------------------------
    # 3) Overall 5-min bin crude rates
    # ------------------------------------------------------------------
    compute_crude_rates(panel, label_prefix="overall", out_dir=results_dir)

    # ------------------------------------------------------------------
    # 4) Stratified by fragility_group (descriptive)
    # ------------------------------------------------------------------
    for group in ANALYSIS_GROUPS:
        sub = panel[panel["fragility_group"] == group].copy()
        if sub.empty:
            print(
                f"\nNo rows for fragility_group='{group}' in restricted "
                f"panel; skipping."
            )
            continue

        print(f"\n================ {group.upper()} PLAYERS (5-min bins) ================")
        print(f"Sub-panel shape ({group}):", sub.shape)

        true_rate = sub["injury_event"].mean()
        print(
            f"True daily injury event rate ({group}): {true_rate:.6f} "
            "(events per player-day)"
        )

        compute_crude_rates(sub, label_prefix=group, out_dir=results_dir)

    removed = remove_stale_5min_glm_outputs(results_dir)
    if removed:
        print("\nRemoved obsolete separated 5-min GLM outputs:")
        for path in removed:
            print(f"  - {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
