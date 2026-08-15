#!/usr/bin/env python
"""
17_match_proxy_perminute_descriptives_5min.py

Descriptive per-minute injury risk *per match appearance*, using the
match-proxy outcome from 16_build_match_proxy_events.py.

Key metric:
    events per 10,000 minutes of match exposure

Steps:
1) Load enriched panel.
2) Merge day-level prior-history fragility labels and keep the dynamic
   {tough, regular, fragile} risk set.
3) Keep match days only (all_minutes_played > 0) => match-level risk set.
4) Bin prior-7-day burden (all_minutes_last_7d) into 5-minute bins.
5) Compute crude per-minute risk by burden bin:
       injury_event_matchproxy / total_minutes_today
6) Stratify by fragility_group.
7) Optional: stratify by today's minutes bin (cameo vs starter).

Outputs under data/processed/results:
- matchproxy_hazard_5min_overall_perminute.csv
- matchproxy_hazard_5min_<group>_perminute.csv
- matchproxy_hazard_5min_<group>_perminute_by_todaybin.csv

Notes:
- These are descriptive 5-minute crude rates only; inferential per-minute
  models live in src/18_match_proxy_poisson_splines_perminute.py.
- all_minutes7d_bin_5min is created once on the match panel before subgroup
  splits so today-bin stratification uses the same burden grid.

Run from repo root:
    python src/17_match_proxy_perminute_descriptives_5min.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from pipeline_io import (
    merge_day_fragility,
    restrict_to_available_risk_set,
    restrict_to_fragility_risk_set,
)

TODAY_BINS = [0, 30, 60, 90, 130]
TODAY_LABELS = ["1-30", "31-60", "61-90", "91-120"]
MATCH_MINUTES_COL = "all_minutes_played"


def make_5min_bins(series: pd.Series):
    if series.empty:
        raise ValueError("Cannot create 5-minute bins for an empty series.")
    max_val = float(series.max())
    upper = max(5, int(np.ceil(max_val / 5.0) * 5))
    edges = np.arange(0, upper + 5, 5)
    labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True, right=True), labels


def perminute_table(df: pd.DataFrame, label_prefix: str, out_dir: Path):
    # IMPORTANT: df is assumed to already contain all_minutes7d_bin_5min
    perminute = (
        df.groupby("all_minutes7d_bin_5min", dropna=False, observed=False)
        .agg(
            total_events=("injury_event_matchproxy", "sum"),
            total_minutes=(MATCH_MINUTES_COL, "sum"),
        )
        .assign(events_per_minute=lambda d: d["total_events"] / d["total_minutes"])
        .reset_index()
    )
    perminute["events_per_10000_min"] = perminute["events_per_minute"] * 10000.0

    out_path = out_dir / f"matchproxy_hazard_5min_{label_prefix}_perminute.csv"
    perminute.to_csv(out_path, index=False)
    print(f"Saved per-minute table -> {out_path}")
    return perminute


def main():  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    proc_dir = root / "data" / "processed"
    results_dir = proc_dir / "results"
    results_dir.mkdir(exist_ok=True)

    panel_path = proc_dir / "player_day_panel_all_comp.csv"

    print(f"Repo root: {root}")
    print(f"Loading enriched panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    print("Panel shape:", panel.shape)

    panel = merge_day_fragility(panel, proc_dir)
    panel = restrict_to_fragility_risk_set(panel)
    panel = restrict_to_available_risk_set(panel)

    if "injury_event_matchproxy" not in panel.columns:
        raise RuntimeError("Missing injury_event_matchproxy. Run script 16 first.")

    # Match-appearance risk set
    match_panel = panel[panel[MATCH_MINUTES_COL] > 0].copy()
    print("Match-panel shape:", match_panel.shape)

    # Create 5-min burden bins ONCE on match_panel so they persist into subgroups
    match_panel["all_minutes7d_bin_5min"], _ = make_5min_bins(match_panel["all_minutes_last_7d"])

    print("\n=== OVERALL MATCH-PROXY PER-MINUTE RATES (5-min burden bins) ===")
    perminute_table(match_panel, "overall", results_dir)

    for g in ["tough", "regular", "fragile"]:
        sub = match_panel[match_panel["fragility_group"] == g].copy()
        print(f"\n=== {g.upper()} MATCH-PROXY PER-MINUTE RATES (5-min burden bins) ===")
        perminute_table(sub, g, results_dir)

        # Today-minutes strata (cameo vs starters)
        sub["today_bin"] = pd.cut(
            sub[MATCH_MINUTES_COL],
            bins=TODAY_BINS,
            labels=TODAY_LABELS,
            include_lowest=True,
            right=True,
        )

        by_today = (
            sub.groupby(["today_bin", "all_minutes7d_bin_5min"], dropna=False, observed=False)
            .agg(
                total_events=("injury_event_matchproxy", "sum"),
                total_minutes=(MATCH_MINUTES_COL, "sum"),
            )
            .assign(events_per_minute=lambda d: d["total_events"] / d["total_minutes"])
            .reset_index()
        )
        by_today["events_per_10000_min"] = by_today["events_per_minute"] * 10000.0
        out_path = results_dir / f"matchproxy_hazard_5min_{g}_perminute_by_todaybin.csv"
        by_today.to_csv(out_path, index=False)
        print(f"Saved today-bin stratified per-minute table -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":  # pragma: no cover
    main()
