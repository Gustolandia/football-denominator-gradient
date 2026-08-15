#!/usr/bin/env python
"""
16_build_match_proxy_events.py

Create match-level proxy outcomes for injuries, aiming to approximate
*per-minute* injury risk within matches.

Background
----------
Our raw daily outcome is:
    injury_event = 1 on the calendar day a Transfermarkt injury spell starts.

We do NOT observe the exact minute of injury within a match. However, to
focus the project on *probability per minute of match exposure*, we build
a defensible proxy:

1) If a spell starts on an all-competition match day (all_minutes_played > 0),
   we treat it as a match injury on that day.

2) If a spell starts on a non-match day but the player played a match the
   *previous* day, we treat it as a match injury that occurred in that
   previous match (lag-1 attribution).

3) All other spell starts are treated as training/other injuries.

Proxy minute within match:
- For match injuries, we assume injuries are most likely late in the
  exposure window.
- We set a proxy minute as the midpoint of the last 15 minutes played:
      injury_minute_proxy = all_minutes_played - 7.5  (if all_minutes_played >= 15)
  For very short appearances (< 15 mins), we use the midpoint:
      injury_minute_proxy = all_minutes_played / 2

Outputs (written back into panel + saved tables):
------------------------------------------------
New columns added to player_day_panel_all_comp.csv:

- match_injury_same_day (bool)
- match_injury_lag1_recorded_next_day (bool)
- injury_event_matchproxy (0/1 on match days)
- injury_event_matchproxy_same_day (0/1 same-day-only sensitivity)
- injury_event_matchproxy_lag1 (0/1 lag-1-only sensitivity)
- injury_event_matchproxy_specific (0/1 excluding ambiguous injury descriptions)
- matchproxy_source in {"same_day","lag1","none"}
- matchproxy_injury_desc (description attached to the back-attributed match row)
- injury_context in {"none","match_same_day","match_lag1_recorded_next_day","training_or_other"}
- injury_minute_proxy (float, NaN if not a match-proxy injury)
- injury_event_trainingproxy (0/1 on training/other injury days)
- player_match_panel_all_comp.csv is also written as a match-row audit table

CSV summaries under data/processed/results:
- match_proxy_counts_overall.csv
- match_proxy_counts_by_fragility.csv
- match_proxy_backattribution_reconciliation.csv

Operational note:
This script mutates data/processed/player_day_panel_all_comp.csv in place, but
it preserves all player-days. Summary CSVs are restricted to rows with eligible
prior-history fragility labels.

Run from repo root:
    python src/16_build_match_proxy_events.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from pipeline_io import (
    merge_day_fragility,
    restrict_to_available_risk_set,
    restrict_to_fragility_risk_set,
)

MATCH_MINUTES_COL = "all_minutes_played"
PLAYER_ID_COL = "tm_player_id"
AMBIGUOUS_INJURY_PATTERN = r"\b(?:unknown|unclear|not reported|not specified|other)\b"


def is_specific_injury_desc(desc: pd.Series) -> pd.Series:
    """Return True for non-empty injury descriptions that are not ambiguous."""
    clean = desc.fillna("").astype(str).str.strip().str.lower()
    return clean.ne("") & ~clean.str.contains(AMBIGUOUS_INJURY_PATTERN, regex=True)


def match_proxy_reconciliation(summary_panel: pd.DataFrame) -> pd.DataFrame:
    """Reconcile injury start-date contexts with back-attributed match-row events."""
    required = {
        "fragility_group",
        "injury_event",
        "injury_context",
        MATCH_MINUTES_COL,
        "injury_event_matchproxy_same_day",
        "injury_event_matchproxy_lag1",
        "injury_event_matchproxy",
    }
    missing = required - set(summary_panel.columns)
    if missing:
        raise KeyError(f"summary_panel missing required columns: {sorted(missing)}")

    groups = sorted(summary_panel["fragility_group"].dropna().astype(str).unique())
    rows = []
    for group in groups + ["overall"]:
        if group == "overall":
            frame = summary_panel
        else:
            frame = summary_panel[summary_panel["fragility_group"].astype(str) == group]

        starts = frame[frame["injury_event"].fillna(0).astype(int).eq(1)]
        match_rows = frame[frame[MATCH_MINUTES_COL].astype(float).gt(0)]
        start_same_day = int(starts["injury_context"].eq("match_same_day").sum())
        start_lag1 = int(starts["injury_context"].eq("match_lag1_recorded_next_day").sum())
        match_same_day = int(match_rows["injury_event_matchproxy_same_day"].sum())
        match_lag1 = int(match_rows["injury_event_matchproxy_lag1"].sum())
        rows.append(
            {
                "fragility_group": group,
                "start_same_day_events": start_same_day,
                "start_lag1_events": start_lag1,
                "start_match_proxy_events": start_same_day + start_lag1,
                "matchrow_same_day_events": match_same_day,
                "matchrow_lag1_events": match_lag1,
                "matchrow_proxy_events": int(match_rows["injury_event_matchproxy"].sum()),
                "unassigned_same_day_events": start_same_day - match_same_day,
                "unassigned_lag1_events": start_lag1 - match_lag1,
                "unassigned_proxy_events": (start_same_day + start_lag1)
                - int(match_rows["injury_event_matchproxy"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    proc_dir = root / "data" / "processed"
    results_dir = proc_dir / "results"
    results_dir.mkdir(exist_ok=True)

    panel_path = proc_dir / "player_day_panel_all_comp.csv"
    match_panel_path = proc_dir / "player_match_panel_all_comp.csv"

    print(f"Repo root: {root}")
    print(f"Loading player-day panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    print("Panel shape (all players):", panel.shape)

    print("Merging prior-history day-level fragility labels ...")
    panel = merge_day_fragility(panel, proc_dir)
    print("Panel shape after fragility merge:", panel.shape)

    required = ["date", PLAYER_ID_COL, "injury_event", MATCH_MINUTES_COL]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(f"Panel missing required columns: {missing}")

    panel = panel.sort_values([PLAYER_ID_COL, "date"]).copy()

    def per_player(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        prev_minutes = g[MATCH_MINUTES_COL].shift(1).fillna(0.0)

        g["match_injury_same_day"] = (
            (g["injury_event"] == 1) & (g[MATCH_MINUTES_COL] > 0)
        ).astype(bool)

        g["match_injury_lag1_recorded_next_day"] = (
            (g["injury_event"] == 1)
            & (g[MATCH_MINUTES_COL] == 0)
            & (prev_minutes > 0)
        ).astype(bool)

        # Back-attribute next-day detection to previous match day
        lag1_prevday = (
            g["match_injury_lag1_recorded_next_day"]
            .shift(-1, fill_value=False)
            .astype(bool)
        )

        g["injury_event_matchproxy"] = (
            (g[MATCH_MINUTES_COL] > 0) & (g["match_injury_same_day"] | lag1_prevday)
        ).astype(int)
        g["injury_event_matchproxy_same_day"] = (
            (g[MATCH_MINUTES_COL] > 0) & g["match_injury_same_day"]
        ).astype(int)
        g["injury_event_matchproxy_lag1"] = (
            (g[MATCH_MINUTES_COL] > 0) & lag1_prevday
        ).astype(int)

        g["matchproxy_source"] = np.where(
            g["match_injury_same_day"],
            "same_day",
            np.where(lag1_prevday, "lag1", "none"),
        )
        same_day_desc = g.get("injury_desc", pd.Series("", index=g.index))
        lag1_desc = same_day_desc.shift(-1).fillna("")
        g["matchproxy_injury_desc"] = np.where(
            g["injury_event_matchproxy_same_day"] == 1,
            same_day_desc.fillna(""),
            np.where(g["injury_event_matchproxy_lag1"] == 1, lag1_desc, ""),
        )
        g["injury_event_matchproxy_specific"] = (
            (g["injury_event_matchproxy"] == 1)
            & is_specific_injury_desc(g["matchproxy_injury_desc"])
        ).astype(int)

        g["injury_context"] = "none"
        g.loc[g["match_injury_same_day"], "injury_context"] = "match_same_day"
        g.loc[g["match_injury_lag1_recorded_next_day"], "injury_context"] = (
            "match_lag1_recorded_next_day"
        )
        g.loc[
            (g["injury_event"] == 1)
            & (~g["match_injury_same_day"])
            & (~g["match_injury_lag1_recorded_next_day"]),
            "injury_context",
        ] = "training_or_other"

        # Proxy minute (late-match assumption)
        m = g[MATCH_MINUTES_COL].astype(float)
        g["injury_minute_proxy"] = np.where(
            g["injury_event_matchproxy"] == 1,
            np.where(m >= 15.0, m - 7.5, m / 2.0),
            np.nan,
        )

        g["injury_event_trainingproxy"] = (
            g["injury_context"] == "training_or_other"
        ).astype(int)

        return g

    print("Building match-proxy injury outcomes ...")
    panel = pd.concat(
        [per_player(group) for _, group in panel.groupby(PLAYER_ID_COL, sort=False)],
        ignore_index=True,
    )

    summary_panel = restrict_to_fragility_risk_set(panel)
    summary_panel = restrict_to_available_risk_set(summary_panel)

    print("\nOverall proxy counts:")
    overall_counts = summary_panel.loc[
        summary_panel["injury_event"] == 1, "injury_context"
    ].value_counts()
    print(overall_counts)
    overall_counts.rename_axis("injury_context").reset_index(name="n_events").to_csv(
        results_dir / "match_proxy_counts_overall.csv", index=False
    )

    by_group = (
        summary_panel.loc[summary_panel["injury_event"] == 1]
        .groupby(["fragility_group", "injury_context"])
        .size()
        .rename("n_events")
        .reset_index()
    )
    print("\nProxy counts by fragility group:")
    print(by_group.head(20))
    by_group.to_csv(results_dir / "match_proxy_counts_by_fragility.csv", index=False)

    reconciliation = match_proxy_reconciliation(summary_panel)
    print("\nBack-attribution reconciliation:")
    print(reconciliation)
    reconciliation.to_csv(
        results_dir / "match_proxy_backattribution_reconciliation.csv",
        index=False,
    )

    print(f"\nSaving enriched panel back to: {panel_path}")
    panel.to_csv(panel_path, index=False)
    match_panel = restrict_to_available_risk_set(panel)
    match_panel = match_panel[match_panel[MATCH_MINUTES_COL] > 0].copy()
    print(f"Saving all-competition match-row audit panel to: {match_panel_path}")
    match_panel.to_csv(match_panel_path, index=False)
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
