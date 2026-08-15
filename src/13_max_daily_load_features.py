#!/usr/bin/env python
"""
13_max_daily_load_features.py

Add per-day / per-week load features to the player-day panel, plus
calendar-phase (weekly cycle) terms for later hazard models.

New columns written back into:
    data/processed/player_day_panel_all_comp.csv

Features (by row = player-day):

Load based on `LOAD_MINUTES_COL` (default: all-competition minutes played):

Rows are grouped by the stable Transfermarkt player identifier (`tm_player_id`).

- minutes_yesterday:
      Minutes played by this player on the previous calendar day.
      0 if no row yesterday or they didn't play.

- minutes_last_match:
      Minutes in the most recent *prior* match (games_played == 1),
      0 if no prior match in the panel yet.

- days_since_last_match:
      Calendar days since the most recent prior match, missing if the player
      has no previous observed match in the panel.

- recovery_interval_bin:
      Clinician-readable recovery interval before the current row:
      0-3 days, 4-5 days, 6-7 days, 8-14 days, or >14 days/no prior match.
      This is mainly interpreted on match rows.

- prior_match_within_3d / prior_match_within_5d:
      Binary short-recovery flags for the most recent prior match.

- zero_burden_long_rest:
      1 when previous-7-day minutes are zero because the most recent observed
      prior match is more than 14 days earlier or there is no prior observed
      match. This supports a cleaner zero-burden comparator sensitivity.

- max_daily_minutes_last_7d:
      Maximum minutes in any single day in the 7 days BEFORE today
      (window t-7 .. t-1).

- any_day_last7_over_90:
      **REPAIRED MEANING**
      1 if max_daily_minutes_last_7d >= FULL_MATCH_THRESHOLD (default 85),
      else 0.
      We keep the historic column name for backwards compatibility,
      but it now captures "played a near-full match in at least one day
      in the last week".

- any_day_last7_full_match:
      Alias of any_day_last7_over_90, used for clarity in downstream work / paper.

- excess_minutes_last7d:
      Continuous estimate of extra-time surplus minutes within the past 7 days:
          max(0, all_minutes_last_7d - 90 * all_games_last_7d)
      This preserves extra-time minutes on a *per-week* basis even when
      daily minutes are capped at 90 in league logs.

- any_extra_time_last7d:
      1 if excess_minutes_last7d >= 15, else 0.
      Small threshold avoids firing on rounding noise.

Calendar-phase (same for all players on a given date):

- day_index:
      Integer days since the first date in the panel.

- week_phase_sin, week_phase_cos:
      sin(2π * day_index / 7), cos(2π * day_index / 7)
      → fundamental weekly cycle.

- halfweek_phase_sin, halfweek_phase_cos:
      sin(2π * day_index / 3.5), cos(2π * day_index / 3.5)
      → 3.5-day "half-week" harmonic (weekend + midweek pattern).

Run from repo root:

    python src/13_max_daily_load_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Which minutes column to treat as "load" for these features.
LOAD_MINUTES_COL = "all_minutes_played"
LOAD_GAMES_COL = "all_games_played"
LOAD_GAMES_LAST_7D_COL = "all_games_last_7d"
PLAYER_ID_COL = "tm_player_id"

# Observable “near full match” threshold for visibility.
# EPL logs cap at 90, so >90 is unobservable; 85 cleanly captures “full match”.
FULL_MATCH_THRESHOLD = 85.0

# Minimum extra-time surplus (in minutes) to count as ET for binary flag.
EXTRA_TIME_THRESHOLD = 15.0


def add_load_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add prior-match recovery, per-day load, extra-time surplus,
    clean-comparator, and weekly/half-weekly sin/cos terms.
    """
    if LOAD_MINUTES_COL not in panel.columns:
        raise KeyError(
            f"Panel is missing '{LOAD_MINUTES_COL}'. "
            "Update LOAD_MINUTES_COL at the top of 13_max_daily_load_features.py "
            "to point at the correct minutes column."
        )

    required_cols = [
        "date",
        PLAYER_ID_COL,
        LOAD_MINUTES_COL,
        LOAD_GAMES_COL,
        "all_minutes_last_7d",
        LOAD_GAMES_LAST_7D_COL,
    ]
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise KeyError(
            f"Panel is missing required columns {missing}. "
            "Make sure player_day_panel_all_comp.csv is up to date."
        )

    panel = panel.sort_values([PLAYER_ID_COL, "date"]).copy()

    def per_player(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("date").copy()

        # 1) Minutes yesterday (simple one-day lag)
        g["minutes_yesterday"] = g[LOAD_MINUTES_COL].shift(1).fillna(0.0)

        # 2) Minutes in the most recent *prior* match
        match_minutes = g[LOAD_MINUTES_COL].where(g[LOAD_GAMES_COL] == 1, np.nan)
        # ffill to carry forward last match's minutes, then shift by 1
        g["minutes_last_match"] = match_minutes.ffill().shift(1).fillna(0.0)

        # 2b) Recovery interval since the most recent prior observed match.
        match_dates = g["date"].where(g[LOAD_GAMES_COL] == 1, pd.NaT)
        prior_match_date = match_dates.ffill().shift(1)
        g["days_since_last_match"] = (g["date"] - prior_match_date).dt.days
        g["recovery_interval_bin"] = pd.cut(
            g["days_since_last_match"],
            bins=[-np.inf, 3, 5, 7, 14, np.inf],
            labels=[
                "0-3 days",
                "4-5 days",
                "6-7 days",
                "8-14 days",
                ">14 days/no prior match",
            ],
            right=True,
        ).astype("object")
        g["recovery_interval_bin"] = g["recovery_interval_bin"].fillna(
            ">14 days/no prior match"
        )
        g["prior_match_within_3d"] = (g["days_since_last_match"] <= 3).astype(int)
        g["prior_match_within_5d"] = (g["days_since_last_match"] <= 5).astype(int)

        # 3) Max daily minutes in previous 7 days (t-7 .. t-1)
        shifted = g[LOAD_MINUTES_COL].shift(1)
        g["max_daily_minutes_last_7d"] = (
            shifted.rolling(window=7, min_periods=1).max().fillna(0.0)
        )

        # 4a) Full-match visibility within last 7 days (observable)
        # Keep old name for downstream compatibility.
        g["any_day_last7_over_90"] = (
            g["max_daily_minutes_last_7d"] >= FULL_MATCH_THRESHOLD
        ).astype(int)
        g["any_day_last7_full_match"] = g["any_day_last7_over_90"]

        # 4b) Extra-time surplus minutes within last 7 days (ALL competitions)
        normal_cap = 90.0 * g[LOAD_GAMES_LAST_7D_COL]
        g["excess_minutes_last7d"] = (g["all_minutes_last_7d"] - normal_cap).clip(
            lower=0.0
        )

        # Optional binary ET flag (robust to rounding)
        g["any_extra_time_last7d"] = (
            g["excess_minutes_last7d"] >= EXTRA_TIME_THRESHOLD
        ).astype(int)
        g["zero_burden_long_rest"] = (
            (g["all_minutes_last_7d"].astype(float) <= 0.0)
            & (
                g["days_since_last_match"].isna()
                | (g["days_since_last_match"].astype(float) > 14.0)
            )
        ).astype(int)

        return g

    print("Adding per-player load features ...")
    panel = pd.concat(
        [per_player(group) for _, group in panel.groupby(PLAYER_ID_COL, sort=False)],
        ignore_index=True,
    )

    # ----------------------------------------------------------------------
    # Calendar-based weekly cycle terms
    # ----------------------------------------------------------------------
    print("Adding weekly and half-weekly calendar phase terms ...")

    min_date = panel["date"].min()
    panel["day_index"] = (panel["date"] - min_date).dt.days.astype(int)

    two_pi = 2.0 * np.pi
    # Weekly (7-day) cycle
    panel["week_phase_sin"] = np.sin(two_pi * panel["day_index"] / 7.0)
    panel["week_phase_cos"] = np.cos(two_pi * panel["day_index"] / 7.0)
    # Half-week (3.5-day) harmonic
    panel["halfweek_phase_sin"] = np.sin(two_pi * panel["day_index"] / 3.5)
    panel["halfweek_phase_cos"] = np.cos(two_pi * panel["day_index"] / 3.5)

    return panel


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    proc_dir = root / "data" / "processed"
    panel_path = proc_dir / "player_day_panel_all_comp.csv"

    print(f"Repo root: {root}")
    print(f"Loading player-day panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    print("Original panel shape:", panel.shape)

    panel = add_load_features(panel)

    # Quick sanity check summaries
    print("\nSummary of new load features:")
    cols = [
        "minutes_yesterday",
        "minutes_last_match",
        "days_since_last_match",
        "prior_match_within_3d",
        "prior_match_within_5d",
        "zero_burden_long_rest",
        "max_daily_minutes_last_7d",
        "any_day_last7_over_90",
        "any_day_last7_full_match",
        "excess_minutes_last7d",
        "any_extra_time_last7d",
    ]
    print(panel[cols].describe())

    print("\nSaving enriched panel back to:", panel_path)
    panel.to_csv(panel_path, index=False)
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
