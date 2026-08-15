#!/usr/bin/env python
"""
07_build_all_comp_minutes.py

Enrich player_day_panel.csv with all-competitions load from Transfermarkt
(appearances.csv), computing appearances for clubs that were EPL clubs in the
corresponding season:

- all_minutes_played: total match minutes in all recorded club competitions on
  the current day per tm_player_id, restricted to EPL club-seasons.
- all_games_played: 1 if the player appeared in any competition on the current day.
- all_minutes_last_7d: total EPL-club-season competition minutes in the previous
  7 days (excluding today) per tm_player_id.
- all_games_last_7d: count of EPL-club-season appearance days in the previous
  7 days (excluding today) per tm_player_id.
- non_epl_minutes_last_7d: all_minutes_last_7d - minutes_last_7d (EPL-only)

Inputs:
- data/processed/player_day_panel.csv
- external_data/transfermarkt/appearances.csv
- external_data/transfermarkt/games.csv

Output:
- data/processed/player_day_panel_all_comp.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from config import ANALYSIS_START_DATE
from pipeline_io import (
    build_injury_episodes,
    expand_injury_episode_days,
    injury_episode_start_table,
)


def epl_club_seasons(games_path: Path) -> pd.DataFrame:
    """Return (season, club_id) pairs for clubs in the EPL in each season."""
    games = pd.read_csv(
        games_path,
        usecols=["competition_id", "season", "home_club_id", "away_club_id"],
    )
    epl_games = games[games["competition_id"] == "GB1"].copy()
    home = epl_games[["season", "home_club_id"]].rename(columns={"home_club_id": "club_id"})
    away = epl_games[["season", "away_club_id"]].rename(columns={"away_club_id": "club_id"})
    return (
        pd.concat([home, away], ignore_index=True)
        .dropna(subset=["season", "club_id"])
        .drop_duplicates()
        .assign(season=lambda d: d["season"].astype(int), club_id=lambda d: d["club_id"].astype(int))
    )


def restrict_to_epl_club_seasons(apps: pd.DataFrame, games_path: Path) -> pd.DataFrame:
    """
    Keep all-competition appearances made for clubs that are EPL clubs that season.

    This keeps domestic cups and European matches for EPL clubs while excluding
    appearances for non-EPL clubs during a player's broader career window.
    """
    required = {"game_id", "player_club_id"}
    missing = required - set(apps.columns)
    if missing:
        raise ValueError(f"appearances.csv missing required columns: {sorted(missing)}")

    games = pd.read_csv(games_path, usecols=["game_id", "season"])
    apps = apps.merge(games, on="game_id", how="left")
    club_seasons = epl_club_seasons(games_path)
    before = len(apps)
    apps = apps.merge(
        club_seasons,
        left_on=["season", "player_club_id"],
        right_on=["season", "club_id"],
        how="inner",
    ).drop(columns=["club_id"])
    print(
        "Restricting all-competition appearances to EPL club-seasons: "
        f"{before} -> {len(apps)} rows"
    )
    return apps


def load_observed_appearance_days(
    apps_path: Path,
    player_ids: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return every recorded club appearance day for cohort players."""
    apps = pd.read_csv(
        apps_path,
        usecols=["player_id", "date", "minutes_played"],
        low_memory=False,
    )
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps["player_id"] = pd.to_numeric(apps["player_id"], errors="coerce")
    apps["minutes_played"] = pd.to_numeric(apps["minutes_played"], errors="coerce")
    keep = (
        apps["player_id"].isin(set(pd.Series(player_ids).astype(int)))
        & apps["date"].between(start_date, end_date)
        & apps["minutes_played"].fillna(0).gt(0)
    )
    return (
        apps.loc[keep, ["player_id", "date"]]
        .rename(columns={"player_id": "tm_player_id"})
        .drop_duplicates()
        .sort_values(["tm_player_id", "date"])
        .reset_index(drop=True)
    )


def expand_player_day_risk_set(
    player_day: pd.DataFrame,
    all_players: pd.DataFrame,
    injuries: pd.DataFrame,
    observed_appearance_days: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand each player span to every valid EPL-club-season appearance.

    Injury events and unavailable days are rebuilt from return-reconciled
    episodes so an observed appearance can never be removed from the match
    risk set by a stale public injury end date.
    """
    required_panel = {"tm_player_id", "date", "minutes_played"}
    missing_panel = required_panel - set(player_day.columns)
    if missing_panel:
        raise KeyError(f"player_day missing required columns: {sorted(missing_panel)}")
    required_all = {"tm_player_id", "date", "all_minutes_played"}
    missing_all = required_all - set(all_players.columns)
    if missing_all:
        raise KeyError(f"all_players missing required columns: {sorted(missing_all)}")

    base = player_day.copy()
    base["tm_player_id"] = base["tm_player_id"].astype(int)
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    source_days = all_players.loc[
        all_players["all_minutes_played"].fillna(0).gt(0), ["tm_player_id", "date"]
    ].copy()
    source_days["tm_player_id"] = source_days["tm_player_id"].astype(int)
    source_days["date"] = pd.to_datetime(source_days["date"], errors="coerce")

    base_spans = base.groupby("tm_player_id")["date"].agg(base_first="min", base_last="max")
    source_spans = source_days.groupby("tm_player_id")["date"].agg(
        source_first="min", source_last="max"
    )
    spans = base_spans.join(source_spans, how="left")
    spans["first"] = spans[["base_first", "source_first"]].min(axis=1)
    spans["last"] = spans[["base_last", "source_last"]].max(axis=1)

    grids = [
        pd.DataFrame(
            {
                "tm_player_id": int(player_id),
                "date": pd.date_range(row["first"], row["last"], freq="D"),
            }
        )
        for player_id, row in spans.iterrows()
    ]
    grid = pd.concat(grids, ignore_index=True)
    base["_base_row"] = 1
    rebuilt = grid.merge(base, on=["tm_player_id", "date"], how="left")
    rebuilt["row_added_by_all_comp_span"] = rebuilt["_base_row"].isna().astype(int)
    rebuilt = rebuilt.drop(columns=["_base_row"])

    if "fbref_player_id" in rebuilt:
        rebuilt["fbref_player_id"] = rebuilt["fbref_player_id"].fillna(
            rebuilt["tm_player_id"]
        )
    else:
        rebuilt["fbref_player_id"] = rebuilt["tm_player_id"]
    rebuilt["fbref_player_id"] = rebuilt["fbref_player_id"].astype(int)
    rebuilt["minutes_played"] = pd.to_numeric(
        rebuilt["minutes_played"], errors="coerce"
    ).fillna(0.0)

    reset_columns = [
        "injury_spell_id",
        "injury_desc",
        "n_injury_spells",
        "injury_event",
        "injury_unavailable",
        "available_for_injury_risk",
        "games_played",
        "games_last_7d",
        "minutes_last_7d",
    ]
    rebuilt = rebuilt.drop(columns=[column for column in reset_columns if column in rebuilt])

    episodes = build_injury_episodes(
        injuries,
        appearance_days=observed_appearance_days,
        min_date=rebuilt["date"].min(),
        max_date=rebuilt["date"].max(),
    )
    starts = injury_episode_start_table(episodes)
    rebuilt = rebuilt.merge(
        starts,
        left_on=["tm_player_id", "date"],
        right_on=["tm_player_id", "start_date"],
        how="left",
    ).drop(columns=["start_date"])
    rebuilt["n_injury_spells"] = pd.to_numeric(
        rebuilt["n_injury_spells"], errors="coerce"
    ).fillna(0).astype(int)
    rebuilt["injury_event"] = rebuilt["n_injury_spells"].gt(0).astype(int)

    unavailable = expand_injury_episode_days(
        episodes,
        rebuilt["date"].min(),
        rebuilt["date"].max(),
    )
    rebuilt = rebuilt.merge(unavailable, on=["tm_player_id", "date"], how="left")
    rebuilt["injury_unavailable"] = pd.to_numeric(
        rebuilt["injury_unavailable"], errors="coerce"
    ).fillna(0).astype(int)
    rebuilt["available_for_injury_risk"] = rebuilt["injury_unavailable"].eq(0).astype(int)

    rebuilt = rebuilt.sort_values(["tm_player_id", "date"]).reset_index(drop=True)
    rebuilt["games_played"] = rebuilt["minutes_played"].gt(0).astype(int)
    rebuilt["games_last_7d"] = rebuilt.groupby("tm_player_id")["games_played"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).sum()
    ).fillna(0.0)
    rebuilt["minutes_last_7d"] = rebuilt.groupby("tm_player_id")["minutes_played"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).sum()
    ).fillna(0.0)
    return rebuilt, episodes


def build_all_comp_minutes(player_day: pd.DataFrame,
                           apps_path: Path,
                           games_path: Path) -> pd.DataFrame:
    """
    Build per-(tm_player_id, date) all_minutes_last_7d from Transfermarkt data.

    Parameters
    ----------
    player_day : DataFrame
        Existing EPL player_day_panel with columns:
        - tm_player_id
        - date
        - minutes_last_7d (EPL-only)
    apps_path : Path
        Path to transfermarkt appearances.csv
    games_path : Path
        Path to transfermarkt games.csv, used to identify EPL club-seasons and
        prevent non-EPL-club career appearances from entering the load history.

    Returns
    -------
    all_players : DataFrame
        Columns:
        - tm_player_id
        - date
        - all_minutes_played
        - all_games_played
        - all_minutes_last_7d
        - all_games_last_7d
    """
    print("Loading Transfermarkt appearances...")
    apps = pd.read_csv(apps_path, low_memory=False)

    print("Columns in appearances.csv:", apps.columns.tolist())

    if "date" not in apps.columns:
        raise ValueError(
            "Expected a 'date' column in appearances.csv, "
            f"but found: {apps.columns.tolist()}"
        )

    # Parse date column to datetime
    apps["date"] = pd.to_datetime(apps["date"])

    # Restrict to tm_player_ids that actually appear in the EPL panel
    tm_ids = player_day["tm_player_id"].unique()
    print(f"Filtering appearances to {len(tm_ids)} tm_player_id present in EPL panel...")
    apps = apps[apps["player_id"].isin(tm_ids)].copy()
    apps = restrict_to_epl_club_seasons(apps, games_path)

    # Retain the complete configured study window. Some domestic-cup or
    # European appearances precede a player's first EPL appearance.
    start_date = pd.Timestamp(ANALYSIS_START_DATE)
    end_date = player_day["date"].max()
    start_date_grid = start_date

    print(f"Restricting appearances to [{start_date_grid.date()} .. {end_date.date()}]...")
    mask_dates = (apps["date"] >= start_date_grid) & (apps["date"] <= end_date)
    apps = apps.loc[mask_dates].copy()

    # Aggregate total minutes per (tm_player_id, date) across ALL competitions
    print("Aggregating total minutes per (tm_player_id, date) across all competitions...")
    all_minutes_day = (
        apps
        .groupby(["player_id", "date"], as_index=False)["minutes_played"]
        .sum()
        .rename(columns={
            "player_id": "tm_player_id",
            "minutes_played": "all_minutes_played"
        })
    )

    print("Building complete daily grid per tm_player_id...")
    all_players = (
        pd.MultiIndex.from_product(
            [
                pd.Index(tm_ids, name="tm_player_id"),
                pd.date_range(start=start_date_grid, end=end_date, freq="D", name="date"),
            ]
        )
        .to_frame(index=False)
        .merge(all_minutes_day, on=["tm_player_id", "date"], how="left")
    )
    all_players["all_minutes_played"] = all_players["all_minutes_played"].fillna(0.0)
    all_players["all_games_played"] = (all_players["all_minutes_played"] > 0).astype(int)

    print(f"All-competitions daily panel shape: {all_players.shape}")

    # Compute rolling 7-day total minutes (excluding today)
    print("Computing all_minutes_last_7d (rolling previous 7 days, excluding today)...")
    all_players = all_players.sort_values(["tm_player_id", "date"])

    all_players["all_minutes_last_7d"] = all_players.groupby("tm_player_id")[
        "all_minutes_played"
    ].transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()).fillna(0)
    all_players["all_games_last_7d"] = all_players.groupby("tm_player_id")[
        "all_games_played"
    ].transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()).fillna(0)

    return all_players


def main() -> None:  # pragma: no cover
    # Resolve paths relative to repo root (assumes this script lives in src/)
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    proc_dir = data_dir / "processed"
    # external_data is at repo root
    tm_dir = root / "external_data" / "transfermarkt"

    player_day_path = proc_dir / "player_day_panel.csv"
    apps_path = tm_dir / "appearances.csv"
    games_path = tm_dir / "games.csv"
    out_path = proc_dir / "player_day_panel_all_comp.csv"
    episodes_path = proc_dir / "tm_injury_episodes.csv"
    reconciliation_path = proc_dir / "results" / "risk_set_history_reconciliation.csv"

    print(f"Repo root: {root}")
    print(f"Loading EPL player_day_panel from {player_day_path} ...")
    player_day = pd.read_csv(player_day_path, parse_dates=["date"], low_memory=False)

    print("player_day_panel shape:", player_day.shape)

    injuries = pd.read_csv(
        proc_dir / "tm_injuries_clean.csv",
        parse_dates=["start_date", "end_date"],
        low_memory=False,
    )
    study_start = pd.Timestamp(ANALYSIS_START_DATE)
    study_end = player_day["date"].max()
    observed_appearance_days = load_observed_appearance_days(
        apps_path,
        player_day["tm_player_id"].unique(),
        study_start,
        study_end,
    )

    # Build all-competitions minutes panel.
    all_players = build_all_comp_minutes(player_day, apps_path, games_path)

    all_players = all_players[all_players["date"].between(study_start, study_end)]
    expanded_player_day, episodes = expand_player_day_risk_set(
        player_day,
        all_players,
        injuries,
        observed_appearance_days,
    )

    # Merge all_minutes_last_7d back into player_day_panel
    print("Merging all_minutes_last_7d into player_day_panel...")
    player_day_with_all = expanded_player_day.merge(
        all_players[
            [
                "tm_player_id",
                "date",
                "all_minutes_played",
                "all_games_played",
                "all_minutes_last_7d",
                "all_games_last_7d",
            ]
        ],
        on=["tm_player_id", "date"],
        how="left"
    )

    # If any all-competition fields are missing, set to 0.
    all_comp_cols = [
        "all_minutes_played",
        "all_games_played",
        "all_minutes_last_7d",
        "all_games_last_7d",
    ]
    for col in all_comp_cols:
        missing_all = player_day_with_all[col].isna().sum()
        if missing_all > 0:
            print(f"Warning: {missing_all} player-day rows have NaN {col}. Filling with 0.")
            player_day_with_all[col] = player_day_with_all[col].fillna(0.0)

    # Ensure numeric type
    for col in ["all_minutes_played", "all_minutes_last_7d"]:
        player_day_with_all[col] = player_day_with_all[col].astype(float)
    for col in ["all_games_played", "all_games_last_7d"]:
        player_day_with_all[col] = player_day_with_all[col].astype(int)

    unavailable_appearances = player_day_with_all[
        player_day_with_all["all_minutes_played"].gt(0)
        & player_day_with_all["available_for_injury_risk"].eq(0)
    ]
    if not unavailable_appearances.empty:
        raise ValueError(
            "Observed appearances remain excluded by injury-unavailability intervals: "
            f"{len(unavailable_appearances)} rows"
        )

    print("Computing non_epl_minutes_played = all_minutes_played - minutes_played ...")
    player_day_with_all["non_epl_minutes_played"] = (
        player_day_with_all["all_minutes_played"] - player_day_with_all["minutes_played"]
    )

    # Derive non-EPL minutes in last 7 days
    print("Computing non_epl_minutes_last_7d = all_minutes_last_7d - minutes_last_7d ...")
    player_day_with_all["non_epl_minutes_last_7d"] = (
        player_day_with_all["all_minutes_last_7d"] - player_day_with_all["minutes_last_7d"]
    )

    # Basic sanity checks / EDA
    print("\nSummary of EPL vs ALL competitions load variables:")
    summary = player_day_with_all[
        [
            "minutes_played",
            "all_minutes_played",
            "non_epl_minutes_played",
            "minutes_last_7d",
            "all_minutes_last_7d",
            "non_epl_minutes_last_7d",
        ]
    ].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    print(summary)

    print("\nCorrelation between EPL-only and all-competitions minutes_last_7d:")
    corr = player_day_with_all[["minutes_last_7d", "all_minutes_last_7d"]].corr()
    print(corr)

    # Check for any negative non-EPL minutes (should be only tiny numerical noise, if any)
    negatives = player_day_with_all[player_day_with_all["non_epl_minutes_last_7d"] < -1e-6]
    print(f"\nRows with non_epl_minutes_last_7d < 0: {negatives.shape[0]}")

    # Save updated panel
    print(f"\nSaving enriched panel with all competitions to {out_path} ...")
    player_day_with_all.to_csv(out_path, index=False)
    episodes.to_csv(episodes_path, index=False)

    source_match_rows = int(all_players["all_minutes_played"].gt(0).sum())
    represented_match_rows = int(player_day_with_all["all_minutes_played"].gt(0).sum())
    injury_start_dates = pd.to_datetime(injuries["start_date"], errors="coerce")
    in_window_injuries = injuries[
        injury_start_dates.between(study_start, study_end)
    ]
    reconciliation = pd.DataFrame(
        [
            {"metric": "source_epl_club_season_appearance_days", "value": source_match_rows},
            {"metric": "represented_match_risk_rows", "value": represented_match_rows},
            {
                "metric": "match_rows_added_by_all_comp_span",
                "value": int(
                    (
                        player_day_with_all["row_added_by_all_comp_span"].eq(1)
                        & player_day_with_all["all_minutes_played"].gt(0)
                    ).sum()
                ),
            },
            {
                "metric": "observed_appearance_rows_still_unavailable",
                "value": int(len(unavailable_appearances)),
            },
            {
                "metric": "cleaned_source_injury_reports_all_dates",
                "value": int(len(injuries)),
            },
            {
                "metric": "cleaned_source_injury_reports_in_study_window",
                "value": int(len(in_window_injuries)),
            },
            {"metric": "reconciled_injury_episodes", "value": int(len(episodes))},
            {
                "metric": "source_reports_absorbed_into_continuous_episodes",
                "value": int(episodes["n_source_spells"].sum() - len(episodes)),
            },
            {
                "metric": "episodes_truncated_by_observed_return",
                "value": int(episodes["return_truncated"].sum()),
            },
        ]
    )
    reconciliation_path.parent.mkdir(parents=True, exist_ok=True)
    reconciliation.to_csv(reconciliation_path, index=False)
    print(f"Saved reconciled injury episodes to {episodes_path}")
    print(f"Saved risk-set reconciliation to {reconciliation_path}")
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
