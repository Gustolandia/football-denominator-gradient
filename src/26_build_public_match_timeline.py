#!/usr/bin/env python
"""Build the v4 club-plus-country match chronology and exposure features.

The v4 chronology is deliberately independent of the frozen club-only panel.
It joins observed competitive club appearances with observed national-team
appearances, labels each record by the protocol's exposure scopes, and then
computes the same prior-window variables for every scope.  Missing national
minutes, kick-off times, stadiums and coordinates remain missing throughout.

Run
---
``python src/26_build_public_match_timeline.py`` after
``python src/25_public_data_v4.py``.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from public_data_sources import (
    harmonize_national_matches,
    newest_independent_snapshot,
    newest_worldcup_lineup_snapshot,
    parse_openfootball_worldcup_snapshot,
    validate_worldcup_lineups,
)


PLAYER_ID = "tm_player_id"
SCOPES = (
    "club_competitive",
    "club_plus_senior_national",
    "club_plus_senior_all",
    "club_plus_broader_international",
    "club_plus_all_public",
)
# This internal scope is not an additional scientific subgroup.  It supplies
# the national-only increment needed to add country exposure to the frozen
# club baseline without also replacing the historical club data snapshot.
SENIOR_NATIONAL_ONLY_SCOPE = "senior_competitive_national_only"
SENIOR_ALL_NATIONAL_ONLY_SCOPE = "senior_all_national_only"
BROADER_NATIONAL_ONLY_SCOPE = "broader_international_only"
NATIONAL_ONLY_SCOPES = (
    SENIOR_NATIONAL_ONLY_SCOPE,
    SENIOR_ALL_NATIONAL_ONLY_SCOPE,
    BROADER_NATIONAL_ONLY_SCOPE,
)
WINDOW_DAYS = (3, 5, 7, 14, 28)
STUDY_START_WITH_WARMUP = pd.Timestamp("2017-06-03")
STUDY_END = pd.Timestamp("2025-04-07")
SENIOR_TYPE_IDS = {11, 19}
YOUTH_TYPE_IDS = {17, 20}
TIMELINE_COLUMNS = [
    PLAYER_ID,
    "match_key",
    "game_id",
    "date",
    "kickoff_utc",
    "kickoff_time_known",
    "source",
    "team_id",
    "team_name",
    "opponent_team_id",
    "opponent_team_name",
    "team_venue",
    "competition_id",
    "competition_name",
    "competition_type",
    "competition_status",
    "team_level",
    "minutes_played",
    "is_starter",
    "stadium_id",
    "stadium_name",
    "venue_key",
    "source_url",
    "duplicate_resolution",
    "is_club_competitive",
    "is_senior_competitive",
    "is_senior_friendly",
    "is_youth_international",
    "is_club_friendly",
]
NATIONAL_OUTPUT_COLUMNS = [
    PLAYER_ID,
    "game_id",
    "match_key",
    "date",
    "kickoff_utc",
    "kickoff_time_known",
    "team_id",
    "team_name",
    "opponent_team_id",
    "opponent_team_name",
    "team_venue",
    "team_goals",
    "opponent_goals",
    "validation_team_goals",
    "validation_opponent_goals",
    "validation_score_source",
    "competition_id",
    "competition_name",
    "competition_type",
    "competition_status",
    "team_level",
    "minutes_played",
    "is_starter",
    "participation_state",
    "stadium_id",
    "stadium_name",
    "venue_key",
    "source",
    "source_url",
    "cache_file",
    "retrieved_at_utc",
    "duplicate_resolution",
    "is_club_competitive",
    "is_senior_competitive",
    "is_senior_friendly",
    "is_youth_international",
    "is_club_friendly",
    "competition_type_id",
    "competition_group_id",
    "season",
]


def _required(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def newest_snapshot_dir(raw_root: Path) -> Path:
    """Return the latest immutable Transfermarkt snapshot directory by date tag."""
    candidates = sorted(
        path for path in Path(raw_root).glob("transfermarkt_datasets_*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"No immutable Transfermarkt snapshot under {raw_root}")
    return candidates[-1]


def epl_club_seasons(games: pd.DataFrame) -> pd.DataFrame:
    """Return unique (season, club) rows for Premier League club-seasons."""
    _required(games, ["competition_id", "season", "home_club_id", "away_club_id"], "games")
    epl = games.loc[games["competition_id"].eq("GB1")]
    home = epl[["season", "home_club_id"]].rename(columns={"home_club_id": "club_id"})
    away = epl[["season", "away_club_id"]].rename(columns={"away_club_id": "club_id"})
    return pd.concat([home, away], ignore_index=True).dropna().drop_duplicates().astype({"season": int, "club_id": int})


def classify_team_level(competition_type_id: object, team_name: object = pd.NA) -> str:
    """Classify national level from the source type, using a name only as fallback."""
    type_id = pd.to_numeric(competition_type_id, errors="coerce")
    if not pd.isna(type_id) and int(type_id) in SENIOR_TYPE_IDS:
        return "senior"
    if not pd.isna(type_id) and int(type_id) in YOUTH_TYPE_IDS:
        return "youth_or_olympic"
    text = "" if pd.isna(team_name) else str(team_name).lower()
    if any(marker in text for marker in ("u17", "u18", "u19", "u20", "u21", "u23", "olympic")):
        return "youth_or_olympic"
    return "unknown"


def classify_competition_status(competition_id: object, competition_name: object = pd.NA) -> str:
    """Classify an explicitly labelled friendly without treating missing as competitive."""
    identifier = "" if pd.isna(competition_id) else str(competition_id).strip().lower()
    name = "" if pd.isna(competition_name) else str(competition_name).strip().lower()
    if identifier == "fs" or "friendly" in name:
        return "friendly"
    if identifier or name:
        return "competitive"
    return "unknown"


def _venue_key(stadium_id: object, stadium_name: object) -> object:
    """Return a source-stable venue key or missing when neither identifier is known."""
    parsed_id = pd.to_numeric(stadium_id, errors="coerce")
    if not pd.isna(parsed_id):
        return f"tm_stadium_id:{int(parsed_id)}"
    if not pd.isna(stadium_name) and str(stadium_name).strip():
        return "tm_stadium_name:" + str(stadium_name).strip().casefold()
    return pd.NA


def enrich_endpoint_national_appearances(
    appearances: pd.DataFrame,
    national_teams: pd.DataFrame,
    competitions: pd.DataFrame,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add source-table team and competition labels to endpoint appearance records."""
    _required(appearances, [PLAYER_ID, "game_id", "team_id", "opponent_team_id", "competition_id"], "national appearances")
    _required(national_teams, ["national_team_id", "name"], "national teams")
    _required(competitions, ["competition_id", "name", "type"], "competitions")
    team_lookup = national_teams[["national_team_id", "name"]].drop_duplicates("national_team_id")
    out = appearances.copy()
    out["game_id"] = out["game_id"].astype(str)
    out[PLAYER_ID] = pd.to_numeric(out[PLAYER_ID], errors="coerce").astype("Int64")
    def numeric_or_missing(column: str) -> pd.Series:
        values = out[column] if column in out else pd.Series(np.nan, index=out.index)
        return pd.to_numeric(values, errors="coerce")

    minutes = numeric_or_missing("minutes_played")
    duration = numeric_or_missing("game_duration_minutes")
    on_pitch_team = numeric_or_missing("team_goals_on_pitch")
    on_pitch_opponent = pd.to_numeric(
        numeric_or_missing("opponent_goals_on_pitch"), errors="coerce"
    )
    starters = (
        out["is_starter"].fillna(False).astype(bool)
        if "is_starter" in out
        else pd.Series(False, index=out.index)
    )
    full_match_score = (
        starters
        & minutes.notna()
        & duration.notna()
        & minutes.ge(duration)
        & on_pitch_team.notna()
        & on_pitch_opponent.notna()
    )
    out["validation_team_goals"] = np.where(
        full_match_score, on_pitch_team, np.nan
    )
    out["validation_opponent_goals"] = np.where(
        full_match_score, on_pitch_opponent, np.nan
    )
    out["validation_score_source"] = np.where(
        full_match_score, "full_match_on_pitch", "unavailable"
    )
    out = out.merge(
        team_lookup.rename(columns={"national_team_id": "team_id", "name": "team_name"}),
        on="team_id",
        how="left",
    )
    out = out.merge(
        team_lookup.rename(columns={"national_team_id": "opponent_team_id", "name": "opponent_team_name"}),
        on="opponent_team_id",
        how="left",
    )
    out = out.merge(
        competitions[["competition_id", "name", "type"]].drop_duplicates("competition_id").rename(
            columns={"name": "competition_name", "type": "competition_type"}
        ),
        on="competition_id",
        how="left",
    )
    out["stadium_name"] = pd.NA
    if games is not None:
        _required(
            games,
            [
                "game_id",
                "home_club_id",
                "away_club_id",
                "home_club_name",
                "away_club_name",
                "home_club_goals",
                "away_club_goals",
                "stadium",
            ],
            "snapshot games",
        )
        game_lookup = games[
            [
                "game_id",
                "home_club_id",
                "away_club_id",
                "home_club_name",
                "away_club_name",
                "home_club_goals",
                "away_club_goals",
                "stadium",
            ]
        ].copy()
        game_lookup["game_id"] = game_lookup["game_id"].astype(str)
        game_lookup = game_lookup.drop_duplicates("game_id")
        out = out.merge(game_lookup, on="game_id", how="left")
        source_team = pd.to_numeric(out["team_id"], errors="coerce")
        is_home = source_team.eq(pd.to_numeric(out["home_club_id"], errors="coerce"))
        is_away = source_team.eq(pd.to_numeric(out["away_club_id"], errors="coerce"))
        out["team_name"] = out["team_name"].fillna(
            pd.Series(
                np.where(is_home, out["home_club_name"], np.where(is_away, out["away_club_name"], pd.NA)),
                index=out.index,
            )
        )
        out["opponent_team_name"] = out["opponent_team_name"].fillna(
            pd.Series(
                np.where(is_home, out["away_club_name"], np.where(is_away, out["home_club_name"], pd.NA)),
                index=out.index,
            )
        )
        home_goals = pd.to_numeric(out["home_club_goals"], errors="coerce")
        away_goals = pd.to_numeric(out["away_club_goals"], errors="coerce")
        snapshot_score = (is_home | is_away) & home_goals.notna() & away_goals.notna()
        out.loc[snapshot_score, "validation_team_goals"] = np.where(
            is_home.loc[snapshot_score],
            home_goals.loc[snapshot_score],
            away_goals.loc[snapshot_score],
        )
        out.loc[snapshot_score, "validation_opponent_goals"] = np.where(
            is_home.loc[snapshot_score],
            away_goals.loc[snapshot_score],
            home_goals.loc[snapshot_score],
        )
        out.loc[snapshot_score, "validation_score_source"] = "published_snapshot_game"
        out["stadium_name"] = out["stadium"]
        out = out.drop(
            columns=[
                "home_club_id",
                "away_club_id",
                "home_club_name",
                "away_club_name",
                "home_club_goals",
                "away_club_goals",
                "stadium",
            ]
        )
    out["team_level"] = [
        classify_team_level(type_id, name)
        for type_id, name in zip(out.get("competition_type_id", pd.Series(pd.NA, index=out.index)), out["team_name"])
    ]
    out["competition_status"] = [
        classify_competition_status(identifier, name)
        for identifier, name in zip(out["competition_id"], out["competition_name"])
    ]
    out["source"] = "player_performance_endpoint"
    out["match_key"] = "national:" + out["game_id"].astype(str)
    out["venue_key"] = [
        _venue_key(stadium_id, stadium_name)
        for stadium_id, stadium_name in zip(out.get("stadium_id", pd.Series(pd.NA, index=out.index)), out["stadium_name"])
    ]
    out["duplicate_resolution"] = "single_source"
    return _with_scope_flags(out)


def published_national_appearances(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    cohort_ids: Iterable[int],
    competitions: pd.DataFrame,
) -> pd.DataFrame:
    """Extract the published national-tournament rows before endpoint fallback."""
    _required(appearances, ["player_id", "game_id", "player_club_id", "minutes_played"], "published appearances")
    _required(games, ["game_id", "competition_id", "competition_type", "date", "home_club_id", "away_club_id", "stadium", "url"], "published games")
    cohort = {int(value) for value in cohort_ids}
    game_columns = ["game_id", "competition_id", "competition_type", "date", "home_club_id", "away_club_id", "home_club_name", "away_club_name", "home_club_goals", "away_club_goals", "stadium", "url"]
    game_columns = [column for column in game_columns if column in games.columns]
    national_games = games.loc[games["competition_type"].eq("national_team_competition"), game_columns]
    out = appearances.loc[appearances["player_id"].isin(cohort)].drop(columns=["competition_id", "date"], errors="ignore").merge(
        national_games,
        on="game_id",
        how="inner",
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.loc[out["date"].between(STUDY_START_WITH_WARMUP, STUDY_END, inclusive="both")].copy()
    out = out.rename(columns={"player_id": PLAYER_ID, "player_club_id": "team_id", "stadium": "stadium_name", "url": "source_url"})
    out["game_id"] = out["game_id"].astype(str)
    out[PLAYER_ID] = pd.to_numeric(out[PLAYER_ID], errors="coerce").astype("Int64")
    is_home = out["team_id"].eq(out["home_club_id"])
    out["opponent_team_id"] = np.where(is_home, out["away_club_id"], out["home_club_id"])
    out["team_name"] = np.where(is_home, out.get("home_club_name", pd.NA), out.get("away_club_name", pd.NA))
    out["opponent_team_name"] = np.where(is_home, out.get("away_club_name", pd.NA), out.get("home_club_name", pd.NA))
    out["team_venue"] = np.where(is_home, "home", "away")
    out["team_goals"] = np.where(
        is_home, out.get("home_club_goals", np.nan), out.get("away_club_goals", np.nan)
    )
    out["opponent_goals"] = np.where(
        is_home, out.get("away_club_goals", np.nan), out.get("home_club_goals", np.nan)
    )
    out["validation_team_goals"] = out["team_goals"]
    out["validation_opponent_goals"] = out["opponent_goals"]
    out["validation_score_source"] = np.where(
        pd.to_numeric(out["team_goals"], errors="coerce").notna()
        & pd.to_numeric(out["opponent_goals"], errors="coerce").notna(),
        "published_snapshot_game",
        "unavailable",
    )
    out = out.merge(
        competitions[["competition_id", "name", "type"]].drop_duplicates("competition_id").rename(
            columns={"name": "competition_name", "type": "competition_type_name"}
        ),
        on="competition_id",
        how="left",
    )
    out["competition_type_id"] = pd.NA
    out["competition_group_id"] = pd.NA
    out["season"] = pd.NA
    out["stadium_id"] = pd.NA
    out["kickoff_utc"] = pd.NA
    out["kickoff_time_known"] = False
    out["is_starter"] = pd.NA
    out["participation_state"] = "played"
    out["competition_type"] = out["competition_type_name"]
    out["team_level"] = "senior"
    out["competition_status"] = [
        classify_competition_status(identifier, name)
        for identifier, name in zip(out["competition_id"], out["competition_name"])
    ]
    out["source"] = "published_snapshot"
    out["match_key"] = "national:" + out["game_id"].astype(str)
    out["venue_key"] = [_venue_key(pd.NA, value) for value in out["stadium_name"]]
    out["duplicate_resolution"] = "single_source"
    return _with_scope_flags(out)


def resolve_national_duplicates(
    endpoint_rows: pd.DataFrame,
    published_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use published rows first while retaining a complete duplicate decision audit."""
    combined = pd.concat([published_rows, endpoint_rows], ignore_index=True, sort=False)
    if combined.empty:
        return combined.copy(), pd.DataFrame(columns=[PLAYER_ID, "game_id", "n_rows", "sources", "minutes_consistent", "resolution"])
    _required(combined, [PLAYER_ID, "game_id", "minutes_played", "source"], "combined national appearances")
    kept: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for (player_id, game_id), group in combined.groupby([PLAYER_ID, "game_id"], dropna=False, sort=False):
        sources = sorted(group["source"].astype(str).unique())
        minutes = pd.to_numeric(group["minutes_played"], errors="coerce").dropna().unique()
        consistent = len(minutes) <= 1
        if len(group) == 1:
            resolution = "single_source"
            selected = group.iloc[[0]].copy()
        elif not consistent:
            resolution = "unresolved_minutes_conflict_excluded"
            selected = group.iloc[0:0].copy()
        elif "published_snapshot" in sources:
            resolution = "published_preferred_consistent"
            selected = group.loc[group["source"].eq("published_snapshot")].iloc[[0]].copy()
        else:
            resolution = "endpoint_duplicate_consistent"
            selected = group.iloc[[0]].copy()
        audit_rows.append(
            {
                PLAYER_ID: player_id,
                "game_id": game_id,
                "n_rows": len(group),
                "sources": ";".join(sources),
                "minutes_consistent": consistent,
                "resolution": resolution,
            }
        )
        if not selected.empty:
            selected["duplicate_resolution"] = resolution
            kept.append(selected)
    resolved = pd.concat(kept, ignore_index=True, sort=False) if kept else combined.iloc[0:0].copy()
    for column in NATIONAL_OUTPUT_COLUMNS:
        if column not in resolved:
            resolved[column] = pd.NA
    return resolved[NATIONAL_OUTPUT_COLUMNS], pd.DataFrame(audit_rows)


def _with_scope_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach mutually readable scope indicators to club or national rows."""
    out = frame.copy()
    source = out["source"].astype(str)
    level = out["team_level"].astype(str)
    status = out["competition_status"].astype(str)
    out["is_club_competitive"] = source.eq("club") & status.eq("competitive")
    out["is_senior_competitive"] = source.ne("club") & level.eq("senior") & status.eq("competitive")
    out["is_senior_friendly"] = source.ne("club") & level.eq("senior") & status.eq("friendly")
    out["is_youth_international"] = source.ne("club") & level.eq("youth_or_olympic")
    out["is_club_friendly"] = source.eq("club") & status.eq("friendly")
    return out


def build_club_timeline(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    cohort_ids: Iterable[int],
) -> pd.DataFrame:
    """Build club records using the frozen analysis' EPL-club-season restriction."""
    _required(appearances, ["player_id", "game_id", "player_club_id", "minutes_played", "date"], "club appearances")
    _required(games, ["game_id", "season", "home_club_id", "away_club_id", "competition_id", "competition_type"], "club games")
    cohort = {int(value) for value in cohort_ids}
    game_columns = ["game_id", "season", "competition_id", "competition_type", "home_club_id", "away_club_id", "home_club_name", "away_club_name", "stadium", "url"]
    game_columns = [column for column in game_columns if column in games.columns]
    out = appearances.loc[appearances["player_id"].isin(cohort)].drop(columns="competition_id", errors="ignore").copy().merge(
        games[game_columns],
        on="game_id",
        how="left",
    )
    out = out.merge(epl_club_seasons(games), left_on=["season", "player_club_id"], right_on=["season", "club_id"], how="inner")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.loc[out["date"].between(STUDY_START_WITH_WARMUP, STUDY_END, inclusive="both")].copy()
    out = out.rename(columns={"player_id": PLAYER_ID, "player_club_id": "team_id", "stadium": "stadium_name", "url": "source_url"})
    is_home = out["team_id"].eq(out["home_club_id"])
    out["opponent_team_id"] = np.where(is_home, out["away_club_id"], out["home_club_id"])
    out["team_name"] = np.where(is_home, out.get("home_club_name", pd.NA), out.get("away_club_name", pd.NA))
    out["opponent_team_name"] = np.where(is_home, out.get("away_club_name", pd.NA), out.get("home_club_name", pd.NA))
    out["team_venue"] = np.where(is_home, "home", "away")
    out["competition_name"] = out["competition_id"]
    out["competition_status"] = np.where(out.get("competition_type", "").astype(str).str.contains("friendly", case=False, na=False), "friendly", "competitive")
    out["team_level"] = "club"
    out["source"] = "club"
    out["match_key"] = "club:" + out["game_id"].astype(str)
    out["kickoff_utc"] = pd.NA
    out["kickoff_time_known"] = False
    out["is_starter"] = pd.NA
    out["stadium_id"] = pd.NA
    out["venue_key"] = [_venue_key(pd.NA, value) for value in out["stadium_name"]]
    out["duplicate_resolution"] = "single_source"
    return _with_scope_flags(out)


def build_public_match_timeline(club_rows: pd.DataFrame, national_rows: pd.DataFrame) -> pd.DataFrame:
    """Return a single ordered chronology with one retained row per player-match."""
    timeline = pd.concat([club_rows, national_rows], ignore_index=True, sort=False)
    if timeline.empty:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    _required(timeline, [PLAYER_ID, "match_key", "date", "minutes_played"], "timeline rows")
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
    timeline["minutes_played"] = pd.to_numeric(timeline["minutes_played"], errors="coerce")
    timeline = timeline.dropna(subset=[PLAYER_ID, "match_key", "date", "minutes_played"])
    if timeline.duplicated([PLAYER_ID, "match_key"]).any():
        raise ValueError("Timeline contains unresolved duplicate player-match records")
    for column in TIMELINE_COLUMNS:
        if column not in timeline:
            timeline[column] = pd.NA
    return timeline[TIMELINE_COLUMNS].sort_values([PLAYER_ID, "date", "match_key"]).reset_index(drop=True)


def scope_mask(timeline: pd.DataFrame, scope: str) -> pd.Series:
    """Return the preregistered row inclusion mask for one exposure scope."""
    if scope not in (*SCOPES, *NATIONAL_ONLY_SCOPES):
        raise ValueError(f"Unknown exposure scope: {scope}")
    club = timeline["is_club_competitive"].astype(bool)
    senior_competitive = timeline["is_senior_competitive"].astype(bool)
    senior_friendly = timeline["is_senior_friendly"].astype(bool)
    youth = timeline["is_youth_international"].astype(bool)
    club_friendly = timeline["is_club_friendly"].astype(bool)
    masks = {
        "club_competitive": club,
        "club_plus_senior_national": club | senior_competitive,
        "club_plus_senior_all": club | senior_competitive | senior_friendly,
        "club_plus_broader_international": club | senior_competitive | senior_friendly | youth,
        "club_plus_all_public": club | senior_competitive | senior_friendly | youth | club_friendly,
        SENIOR_NATIONAL_ONLY_SCOPE: senior_competitive,
        SENIOR_ALL_NATIONAL_ONLY_SCOPE: senior_competitive | senior_friendly,
        BROADER_NATIONAL_ONLY_SCOPE: senior_competitive | senior_friendly | youth,
    }
    return masks[scope]


def _window_values(event_dates: np.ndarray, values: np.ndarray, target_dates: np.ndarray, days: int) -> np.ndarray:
    """Sum source values in [target-days, target) for sorted date arrays."""
    prefix = np.concatenate(([0.0], np.cumsum(values.astype(float))))
    stop = np.searchsorted(event_dates, target_dates, side="left")
    start = np.searchsorted(event_dates, target_dates - np.timedelta64(days, "D"), side="left")
    return prefix[stop] - prefix[start]


def _prior_match_features(events: pd.DataFrame, target_dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return calendar days since previous event and prior contiguous-sequence size."""
    event_dates = events["date"].to_numpy(dtype="datetime64[ns]")
    event_counts = events["n_matches"].to_numpy(dtype=float)
    sequence = np.zeros(len(events), dtype=float)
    for index in range(len(events)):
        if index == 0 or (event_dates[index] - event_dates[index - 1]).astype("timedelta64[D]").astype(int) > 14:
            sequence[index] = event_counts[index]
        else:
            sequence[index] = sequence[index - 1] + event_counts[index]
    prior_index = np.searchsorted(event_dates, target_dates, side="left") - 1
    days_since = np.full(len(target_dates), np.nan)
    sequence_before = np.zeros(len(target_dates), dtype=float)
    available = prior_index >= 0
    if available.any():
        days_since[available] = (
            target_dates[available] - event_dates[prior_index[available]]
        ).astype("timedelta64[D]").astype(float)
        sequence_before[available] = sequence[prior_index[available]]
    return days_since, sequence_before


def build_scope_exposure_features(
    timeline: pd.DataFrame,
    target_matches: pd.DataFrame,
    scopes: Sequence[str] = SCOPES,
) -> pd.DataFrame:
    """Compute identical shifted exposure variables for all protocol scopes."""
    _required(timeline, [PLAYER_ID, "date", "match_key", "minutes_played", "source"], "timeline")
    _required(target_matches, [PLAYER_ID, "date"], "target matches")
    targets = target_matches[[PLAYER_ID, "date"]].copy()
    targets["date"] = pd.to_datetime(targets["date"], errors="coerce")
    targets["_target_row"] = np.arange(len(targets))
    output = targets.copy()
    for scope in scopes:
        source_rows = timeline.loc[scope_mask(timeline, scope)].copy()
        source_rows["date"] = pd.to_datetime(source_rows["date"], errors="coerce")
        source_rows["minutes_played"] = pd.to_numeric(source_rows["minutes_played"], errors="coerce")
        source_rows = source_rows.dropna(subset=["date", "minutes_played"])
        source_rows["national_minutes"] = np.where(source_rows["source"].eq("club"), 0.0, source_rows["minutes_played"])
        daily = (
            source_rows.groupby([PLAYER_ID, "date"], as_index=False)
            .agg(minutes_played=("minutes_played", "sum"), national_minutes=("national_minutes", "sum"), n_matches=("match_key", "nunique"))
            .sort_values([PLAYER_ID, "date"])
        )
        rows: list[pd.DataFrame] = []
        for player_id, player_targets in targets.groupby(PLAYER_ID, sort=False):
            player_targets = player_targets.sort_values("date")
            events = daily.loc[daily[PLAYER_ID].eq(player_id)]
            values = pd.DataFrame({"_target_row": player_targets["_target_row"].to_numpy()})
            target_dates = player_targets["date"].to_numpy(dtype="datetime64[ns]")
            if events.empty:
                for days in WINDOW_DAYS:
                    values[f"{scope}_minutes_last_{days}d"] = 0.0
                    values[f"{scope}_matches_last_{days}d"] = 0.0
                    values[f"{scope}_national_minutes_last_{days}d"] = 0.0
                values[f"{scope}_days_since_previous_appearance"] = np.nan
                values[f"{scope}_consecutive_match_sequence"] = 0.0
            else:
                event_dates = events["date"].to_numpy(dtype="datetime64[ns]")
                minutes = events["minutes_played"].to_numpy(dtype=float)
                matches = events["n_matches"].to_numpy(dtype=float)
                national_minutes = events["national_minutes"].to_numpy(dtype=float)
                for days in WINDOW_DAYS:
                    values[f"{scope}_minutes_last_{days}d"] = _window_values(event_dates, minutes, target_dates, days)
                    values[f"{scope}_matches_last_{days}d"] = _window_values(event_dates, matches, target_dates, days)
                    values[f"{scope}_national_minutes_last_{days}d"] = _window_values(event_dates, national_minutes, target_dates, days)
                days_since, sequence = _prior_match_features(events, target_dates)
                values[f"{scope}_days_since_previous_appearance"] = days_since
                values[f"{scope}_consecutive_match_sequence"] = sequence
            rows.append(values)
        wide = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame({"_target_row": targets["_target_row"]})
        output = output.merge(wide, on="_target_row", how="left")
    output["recovery_measure"] = "calendar_days"
    return output.drop(columns="_target_row")


def exposure_scope_comparison(features: pd.DataFrame) -> pd.DataFrame:
    """Summarise seven-day burden reclassification between primary v4 scopes."""
    baseline = "club_competitive_minutes_last_7d"
    expanded = "club_plus_senior_national_minutes_last_7d"
    _required(features, [baseline, expanded], "scope features")
    before = pd.to_numeric(features[baseline], errors="coerce")
    after = pd.to_numeric(features[expanded], errors="coerce")
    changed = after.ne(before) & before.notna() & after.notna()
    reclassified_zero = before.eq(0) & after.gt(0)
    return pd.DataFrame(
        [
            {"metric": "eligible_match_rows", "value": int(before.notna().sum())},
            {"metric": "rows_changed_previous_7d_minutes", "value": int(changed.sum())},
            {"metric": "percent_rows_changed_previous_7d_minutes", "value": float(changed.mean() * 100.0)},
            {"metric": "zero_burden_rows_reclassified", "value": int(reclassified_zero.sum())},
            {"metric": "percent_zero_burden_rows_reclassified", "value": float(reclassified_zero.mean() * 100.0)},
            {"metric": "total_added_senior_national_minutes", "value": float((after - before).sum())},
        ]
    )


def frozen_baseline_scope_comparison(
    target_matches: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify national minutes added to the unchanged frozen club burden.

    The frozen analysis includes recorded club friendlies under its
    all-competition definition.  This comparison therefore adds only observed
    senior competitive national minutes to that stored baseline instead of
    substituting a refreshed, stricter club-competition reconstruction.
    """
    national_column = f"{SENIOR_NATIONAL_ONLY_SCOPE}_minutes_last_7d"
    _required(target_matches, [PLAYER_ID, "date", "all_minutes_last_7d"], "target matches")
    _required(features, [PLAYER_ID, "date", national_column], "scope features")
    merged = target_matches[[PLAYER_ID, "date", "all_minutes_last_7d"]].copy()
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    additions = features[[PLAYER_ID, "date", national_column]].copy()
    additions["date"] = pd.to_datetime(additions["date"], errors="coerce")
    merged = merged.merge(additions, on=[PLAYER_ID, "date"], how="left", validate="one_to_one")
    before = pd.to_numeric(merged["all_minutes_last_7d"], errors="coerce")
    national = pd.to_numeric(merged[national_column], errors="coerce")
    after = before + national
    changed = national.gt(0) & before.notna() & national.notna()
    reclassified_zero = before.eq(0) & after.gt(0)
    return pd.DataFrame(
        [
            {"metric": "eligible_match_rows", "value": int(before.notna().sum())},
            {"metric": "rows_changed_previous_7d_minutes", "value": int(changed.sum())},
            {"metric": "percent_rows_changed_previous_7d_minutes", "value": float(changed.mean() * 100.0)},
            {"metric": "zero_burden_rows_reclassified", "value": int(reclassified_zero.sum())},
            {"metric": "percent_zero_burden_rows_reclassified", "value": float(reclassified_zero.mean() * 100.0)},
            {"metric": "total_added_senior_national_minutes", "value": float(national.sum())},
        ]
    )


def venue_geocode_template(timeline: pd.DataFrame, verified_geocodes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create a provenance-first coordinate template without inventing locations."""
    _required(timeline, ["venue_key", "stadium_id", "stadium_name"], "timeline")
    base = timeline[["venue_key", "stadium_id", "stadium_name"]].dropna(subset=["venue_key"]).drop_duplicates("venue_key")
    base = base.sort_values("venue_key").reset_index(drop=True)
    required_geocode = ["venue_key", "latitude", "longitude", "source_url", "match_confidence", "timezone_offset_hours"]
    if verified_geocodes is None:
        verified_geocodes = pd.DataFrame(columns=required_geocode)
    _required(verified_geocodes, required_geocode, "verified geocodes")
    out = base.merge(verified_geocodes[required_geocode].drop_duplicates("venue_key"), on="venue_key", how="left")
    out["evidence_status"] = np.where(out["latitude"].notna() & out["longitude"].notna(), "verified", "unresolved")
    return out


def great_circle_km(latitude_a: object, longitude_a: object, latitude_b: object, longitude_b: object) -> object:
    """Return the geographic distance, leaving any unknown venue pair missing."""
    values = [pd.to_numeric(value, errors="coerce") for value in (latitude_a, longitude_a, latitude_b, longitude_b)]
    if any(pd.isna(value) for value in values):
        return pd.NA
    lat_a, lon_a, lat_b, lon_b = map(radians, values)
    haversine = sin((lat_b - lat_a) / 2) ** 2 + cos(lat_a) * cos(lat_b) * sin((lon_b - lon_a) / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(haversine))


def add_geographic_travel_proxies(timeline: pd.DataFrame, geocodes: pd.DataFrame) -> pd.DataFrame:
    """Add observed-venue distance/time-zone proxies without imputing missing travel."""
    _required(timeline, [PLAYER_ID, "date", "venue_key"], "timeline")
    _required(geocodes, ["venue_key", "latitude", "longitude", "timezone_offset_hours"], "geocodes")
    out = timeline.merge(geocodes[["venue_key", "latitude", "longitude", "timezone_offset_hours"]], on="venue_key", how="left")
    out = out.sort_values([PLAYER_ID, "date", "match_key"]).copy()
    out["previous_observed_venue_key"] = out.groupby(PLAYER_ID)["venue_key"].shift(1)
    for column in ["latitude", "longitude", "timezone_offset_hours"]:
        out[f"previous_{column}"] = out.groupby(PLAYER_ID)[column].shift(1)
    out["geographic_travel_km"] = [
        great_circle_km(previous_lat, previous_lon, latitude, longitude)
        for previous_lat, previous_lon, latitude, longitude in zip(
            out["previous_latitude"], out["previous_longitude"], out["latitude"], out["longitude"]
        )
    ]
    out["geographic_timezone_change_hours"] = (
        pd.to_numeric(out["timezone_offset_hours"], errors="coerce")
        - pd.to_numeric(out["previous_timezone_offset_hours"], errors="coerce")
    ).abs()
    return out


def geographic_travel_coverage_audit(timeline: pd.DataFrame, geocodes: pd.DataFrame) -> pd.DataFrame:
    """Report whether verified venue data support geographic travel proxies."""
    _required(timeline, ["venue_key", "geographic_travel_km", "geographic_timezone_change_hours"], "timeline")
    _required(geocodes, ["venue_key", "evidence_status"], "geocodes")
    venue_count = int(geocodes["venue_key"].nunique())
    verified = int(geocodes.loc[geocodes["evidence_status"].eq("verified"), "venue_key"].nunique())
    travel_pairs = int(pd.to_numeric(timeline["geographic_travel_km"], errors="coerce").notna().sum())
    timezone_pairs = int(pd.to_numeric(timeline["geographic_timezone_change_hours"], errors="coerce").notna().sum())
    return pd.DataFrame(
        [
            {"metric": "unique_observed_venues", "value": venue_count},
            {"metric": "verified_coordinate_venues", "value": verified},
            {"metric": "verified_coordinate_venue_percent", "value": float(verified / venue_count * 100.0) if venue_count else np.nan},
            {"metric": "timeline_rows_with_travel_distance", "value": travel_pairs},
            {"metric": "timeline_rows_with_timezone_change", "value": timezone_pairs},
            {"metric": "travel_proxy_usable", "value": bool(travel_pairs > 0)},
        ]
    )


def main() -> None:  # pragma: no cover
    """Build the audit-ready v4 timeline and exposure feature outputs."""
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw" / "public_data_v4"
    snapshot = newest_snapshot_dir(raw)
    processed = root / "data" / "processed" / "public_data_v4"
    processed.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(processed / "epl_cohort_manifest.csv", low_memory=False)
    endpoint = pd.read_csv(processed / "international_appearances_raw.csv", low_memory=False)
    teams = pd.read_csv(snapshot / "national_teams.csv.gz", low_memory=False)
    competitions = pd.read_csv(snapshot / "competitions.csv.gz", low_memory=False)
    snapshot_apps = pd.read_csv(snapshot / "appearances.csv.gz", low_memory=False)
    snapshot_games = pd.read_csv(snapshot / "games.csv.gz", low_memory=False)
    endpoint_enriched = enrich_endpoint_national_appearances(
        endpoint, teams, competitions, snapshot_games
    )
    published = published_national_appearances(snapshot_apps, snapshot_games, cohort[PLAYER_ID], competitions)
    national, duplicates = resolve_national_duplicates(endpoint_enriched, published)
    independent_snapshot = newest_independent_snapshot(raw)
    independent_results = pd.read_csv(
        independent_snapshot / "results.csv", low_memory=False
    )
    shootout_path = independent_snapshot / "shootouts.csv"
    independent_shootouts = (
        pd.read_csv(shootout_path, low_memory=False)
        if shootout_path.exists()
        else None
    )
    national, independent_validation, team_crosswalk = harmonize_national_matches(
        national, independent_results, independent_shootouts
    )
    national.to_csv(processed / "international_appearances.csv", index=False)
    duplicates.to_csv(processed / "national_duplicate_audit.csv", index=False)
    independent_validation.to_csv(
        processed / "independent_schedule_validation.csv", index=False
    )
    team_crosswalk.to_csv(processed / "national_team_id_crosswalk.csv", index=False)
    record_audit = pd.read_csv(
        processed / "international_performance_record_audit.csv", low_memory=False
    )
    worldcup_lineups = parse_openfootball_worldcup_snapshot(
        newest_worldcup_lineup_snapshot(raw)
    )
    validate_worldcup_lineups(
        record_audit, cohort, team_crosswalk, worldcup_lineups
    ).to_csv(processed / "openfootball_worldcup_player_validation.csv", index=False)
    club = build_club_timeline(snapshot_apps, snapshot_games, cohort[PLAYER_ID])
    timeline = build_public_match_timeline(club, national)
    template_path = processed / "venue_geocodes_manual.csv"
    manual = pd.read_csv(template_path, low_memory=False) if template_path.exists() else None
    geocodes = venue_geocode_template(timeline, manual)
    geocodes.to_csv(processed / "venue_geocodes.csv", index=False)
    timeline = add_geographic_travel_proxies(timeline, geocodes)
    timeline.to_csv(processed / "public_match_timeline.csv", index=False)
    geographic_travel_coverage_audit(timeline, geocodes).to_csv(
        processed / "geographic_travel_coverage_audit.csv", index=False
    )
    targets = pd.read_csv(root / "data" / "processed" / "player_match_panel_all_comp.csv", low_memory=False)
    features = build_scope_exposure_features(
        timeline,
        targets,
        scopes=(*SCOPES, *NATIONAL_ONLY_SCOPES),
    )
    features.to_csv(processed / "match_exposure_scope_features.csv", index=False)
    exposure_scope_comparison(features).to_csv(processed / "exposure_scope_comparison.csv", index=False)
    frozen_baseline_scope_comparison(targets, features).to_csv(
        processed / "frozen_baseline_national_scope_comparison.csv", index=False
    )
    print(f"Timeline rows: {len(timeline)}")
    print(f"International rows retained: {len(national)}")
    print(f"Unresolved duplicate player-game records: {(duplicates['resolution'] == 'unresolved_minutes_conflict_excluded').sum()}")


if __name__ == "__main__":  # pragma: no cover
    main()
