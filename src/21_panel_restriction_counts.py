#!/usr/bin/env python
"""
21_panel_restriction_counts.py

Summarise the main analytic denominators used by the corrected pipeline:

(1) Full EPL player-day panel
    - number of stable Transfermarkt players
    - player-days
    - match-days (EPL minutes_played > 0)

(2) Prior-history fragility risk set
    - player-days whose day-level label is tough, regular, or fragile
    - this excludes low_exposure days, not whole players

(3) Match-proxy per-minute risk set
    - availability-adjusted prior-history risk-set rows with positive
      all-competition match minutes for EPL club-seasons
    - number of back-attributed match-proxy events on those match rows

The match-proxy event count can differ slightly from the onset-day context
tables written by src/16_build_match_proxy_events.py because lag-1 injuries are
recorded on the next calendar day but modelled on the previous match day.
"""

import pandas as pd

from pipeline_io import restrict_to_available_risk_set, restrict_to_fragility_risk_set

ID_COL = "tm_player_id"
MATCH_MINUTES_COL = "all_minutes_played"


def compute_restriction_counts(
    panel: pd.DataFrame, panel_all: pd.DataFrame
) -> dict[str, int]:
    """Compute full-panel, dynamic-risk-set, and match-proxy denominators."""
    # ---------- (1) Full EPL player-day panel ----------
    players_all = panel[ID_COL].nunique()
    days_all = len(panel)
    match_days_all = (panel["minutes_played"] > 0).sum()

    # ---------- (2) Prior-history fragility risk set ----------
    risk_set = restrict_to_fragility_risk_set(panel_all)
    risk_set = restrict_to_available_risk_set(risk_set)
    players_risk = risk_set[ID_COL].nunique()
    days_risk = len(risk_set)
    match_days_risk = (risk_set[MATCH_MINUTES_COL] > 0).sum()

    # ---------- (3) Match-proxy risk set (dynamic risk set, match days only) ----------
    match_proxy_risk = risk_set[risk_set[MATCH_MINUTES_COL] > 0]

    players_matchproxy = match_proxy_risk[ID_COL].nunique()
    days_matchproxy = len(match_proxy_risk)

    # injury_event_matchproxy is 0/1 in player_day_panel_all_comp.csv
    events_matchproxy = match_proxy_risk["injury_event_matchproxy"].sum()

    return {
        "players_all": int(players_all),
        "days_all": int(days_all),
        "match_days_all": int(match_days_all),
        "players_risk": int(players_risk),
        "days_risk": int(days_risk),
        "match_days_risk": int(match_days_risk),
        "players_matchproxy": int(players_matchproxy),
        "days_matchproxy": int(days_matchproxy),
        "events_matchproxy": int(events_matchproxy),
    }


def main() -> None:  # pragma: no cover
    # Base EPL-only panel
    panel = pd.read_csv("data/processed/player_day_panel.csv", low_memory=False)

    # All-competitions panel (has injury_event_matchproxy etc.)
    panel_all = pd.read_csv(
        "data/processed/player_day_panel_all_comp.csv",
        low_memory=False,
    )

    counts = compute_restriction_counts(panel, panel_all)

    # ---------- Print nicely ----------
    print(
        "Full EPL panel      : "
        f"players = {counts['players_all']}, "
        f"player-days = {counts['days_all']}, "
        f"match-days = {counts['match_days_all']}"
    )
    print(
        "Dynamic risk set    : "
        f"players = {counts['players_risk']}, "
        f"player-days = {counts['days_risk']}, "
        f"match-days = {counts['match_days_risk']}"
    )
    print(
        "Match-proxy risk set: "
        f"players = {counts['players_matchproxy']}, "
        f"match-days = {counts['days_matchproxy']}, "
        f"match-proxy events = {counts['events_matchproxy']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
