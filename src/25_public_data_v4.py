#!/usr/bin/env python
"""Acquire and audit the immutable public-data v4 snapshot.

This module intentionally separates the existing club-only inputs from the
new public-data extension.  It downloads a dated Transfermarkt-datasets
snapshot into ``data/raw/public_data_v4/``, creates an EPL cohort manifest, and
acquires cached national-team performance histories for the existing stable
``tm_player_id`` cohort.  The public performance endpoint is undocumented, so
the script records every URL, retrieval time, HTTP result and cache path.

The resulting national appearances are *observed source records*, not a claim
of complete international coverage.  ``src/27_public_data_v4_audits.py``
applies the prespecified coverage gate before an expanded exposure can be used
as a primary model exposure.

Run
---
``python src/25_public_data_v4.py``
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from zipfile import ZipFile

import pandas as pd
import requests


SNAPSHOT_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
    "transfermarkt-datasets.zip"
)
UPSTREAM_REPOSITORY = "https://github.com/dcaribou/transfermarkt-datasets"
NATIONAL_PERFORMANCE_URL = "https://www.transfermarkt.co.uk/ceapi/performance-game/{player_id}"
SOURCE_USER_AGENT = "EPLCongestionResearch/1.0"
STUDY_START_WITH_WARMUP = pd.Timestamp("2017-06-03")
STUDY_END = pd.Timestamp("2025-04-07")
NATIONAL_COLUMNS = [
    "tm_player_id",
    "game_id",
    "date",
    "kickoff_utc",
    "kickoff_time_known",
    "team_id",
    "opponent_team_id",
    "team_venue",
    "team_goals",
    "opponent_goals",
    "game_duration_minutes",
    "team_goals_on_pitch",
    "opponent_goals_on_pitch",
    "competition_id",
    "competition_group_id",
    "competition_type_id",
    "season",
    "stadium_id",
    "minutes_played",
    "is_starter",
    "participation_state",
    "source_url",
    "cache_file",
    "retrieved_at_utc",
]
NATIONAL_RECORD_AUDIT_COLUMNS = [
    *NATIONAL_COLUMNS,
    "retained_for_exposure",
    "exclusion_reason",
]


def utc_now() -> str:
    """Return a sortable UTC timestamp without relying on local time."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    """Return the SHA-256 digest of ``path`` using bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_schema(path: Path) -> list[str]:
    """Return a CSV header exactly as supplied by the upstream snapshot."""
    return list(pd.read_csv(path, nrows=0).columns)


def file_record(path: Path, source_url: str | None = None) -> dict[str, Any]:
    """Build the immutable-manifest record for one file."""
    path = Path(path)
    record: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if source_url is not None:
        record["source_url"] = source_url
    if path.suffix.lower() == ".csv" or path.name.lower().endswith(".csv.gz"):
        record["columns"] = csv_schema(path)
    return record


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a stable, human-readable JSON audit artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def download_snapshot(
    url: str,
    target: Path,
    get: Callable[..., Any] = requests.get,
) -> Path:
    """Download a snapshot once, refusing to overwrite a dated raw artifact."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Immutable snapshot already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    response = get(url, stream=True, timeout=120, headers={"User-Agent": SOURCE_USER_AGENT})
    response.raise_for_status()
    with target.open("xb") as handle:
        for chunk in response.iter_content(chunk_size=1_048_576):
            if chunk:
                handle.write(chunk)
    return target


def extract_snapshot(archive_path: Path, target_dir: Path) -> list[Path]:
    """Extract a zip snapshot after rejecting unsafe archive member paths."""
    target_dir = Path(target_dir)
    if target_dir.exists():
        raise FileExistsError(f"Immutable snapshot directory already exists: {target_dir}")
    extracted: list[Path] = []
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in snapshot archive: {member.filename}")
        target_dir.mkdir(parents=True)
        for member in members:
            archive.extract(member, target_dir)
            if not member.is_dir():
                extracted.append(target_dir / member.filename)
    return extracted


def snapshot_manifest(
    archive_path: Path,
    extracted_paths: Iterable[Path],
    retrieval_time_utc: str,
    upstream_commit: str,
    terms_url: str,
) -> dict[str, Any]:
    """Return the complete provenance record for a dated source snapshot."""
    records = [file_record(Path(path)) for path in sorted(extracted_paths)]
    return {
        "retrieved_at_utc": retrieval_time_utc,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": upstream_commit,
        "snapshot_url": SNAPSHOT_URL,
        "redistribution_terms_url": terms_url,
        "archive": file_record(archive_path, source_url=SNAPSHOT_URL),
        "files": records,
    }


def _required_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def epl_club_seasons(games: pd.DataFrame) -> pd.DataFrame:
    """Return club/season pairs for clubs competing in the Premier League."""
    _required_columns(
        games,
        ["competition_id", "season", "home_club_id", "away_club_id"],
        "games",
    )
    epl = games.loc[games["competition_id"].eq("GB1")]
    home = epl[["season", "home_club_id"]].rename(columns={"home_club_id": "club_id"})
    away = epl[["season", "away_club_id"]].rename(columns={"away_club_id": "club_id"})
    return (
        pd.concat([home, away], ignore_index=True)
        .dropna()
        .drop_duplicates()
        .astype({"season": int, "club_id": int})
    )


def _joined_values(values: pd.Series) -> str:
    """Join non-empty metadata values in a deterministic form."""
    cleaned = sorted({str(value) for value in values.dropna() if str(value).strip()})
    return ";".join(cleaned)


def build_cohort_manifest(
    panel: pd.DataFrame,
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    transfers: pd.DataFrame,
) -> pd.DataFrame:
    """Create a stable-id cohort manifest with observed EPL club membership."""
    _required_columns(panel, ["tm_player_id"], "existing player-day panel")
    _required_columns(appearances, ["player_id", "game_id", "player_club_id", "date"], "appearances")
    _required_columns(games, ["game_id", "season"], "games")
    ids = pd.Index(pd.to_numeric(panel["tm_player_id"], errors="coerce").dropna().astype(int).unique())
    app = appearances.loc[appearances["player_id"].isin(ids)].copy()
    app["date"] = pd.to_datetime(app["date"], errors="coerce")
    app = app.merge(games[["game_id", "season"]], on="game_id", how="left")
    app = app.merge(
        epl_club_seasons(games),
        left_on=["season", "player_club_id"],
        right_on=["season", "club_id"],
        how="inner",
    )
    membership = (
        app.groupby(["player_id", "season", "player_club_id"], as_index=False)
        .agg(observed_first_appearance=("date", "min"), observed_last_appearance=("date", "max"))
        .rename(columns={"player_id": "tm_player_id"})
    )
    memberships = (
        membership.assign(
            observed_club_season=lambda d: d["season"].astype(str) + "_" + d["player_club_id"].astype(str)
        )
        .groupby("tm_player_id", as_index=False)
        .agg(
            observed_club_seasons=("observed_club_season", _joined_values),
            observed_first_epl_club_appearance=("observed_first_appearance", "min"),
            observed_last_epl_club_appearance=("observed_last_appearance", "max"),
        )
    )
    player_columns = ["player_id", "name", "country_of_citizenship", "current_national_team_id"]
    existing_player_columns = [column for column in player_columns if column in players.columns]
    player_meta = players[existing_player_columns].copy()
    if "player_id" not in player_meta:
        player_meta = pd.DataFrame({"player_id": ids})
    player_meta = player_meta.rename(columns={"player_id": "tm_player_id"}).drop_duplicates("tm_player_id")
    transfer_meta = pd.DataFrame({"tm_player_id": ids})
    if {"player_id", "transfer_date"}.issubset(transfers.columns):
        transfer_meta = (
            transfers.loc[transfers["player_id"].isin(ids), ["player_id", "transfer_date"]]
            .assign(transfer_date=lambda d: pd.to_datetime(d["transfer_date"], errors="coerce"))
            .groupby("player_id", as_index=False)["transfer_date"]
            .agg(_joined_values)
            .rename(columns={"player_id": "tm_player_id", "transfer_date": "recorded_transfer_dates"})
        )
    manifest = pd.DataFrame({"tm_player_id": ids}).merge(player_meta, on="tm_player_id", how="left")
    manifest = manifest.merge(memberships, on="tm_player_id", how="left")
    manifest = manifest.merge(transfer_meta, on="tm_player_id", how="left")
    for column in ["name", "country_of_citizenship", "current_national_team_id"]:
        if column not in manifest:
            manifest[column] = pd.NA
    for column in ["observed_club_seasons", "recorded_transfer_dates"]:
        if column not in manifest:
            manifest[column] = ""
        manifest[column] = manifest[column].fillna("")
    manifest["unresolved_cohort_id"] = manifest["observed_club_seasons"].eq("")
    return manifest.sort_values("tm_player_id").reset_index(drop=True)


def audit_snapshot_national_appearances(
    competitions: pd.DataFrame,
    games: pd.DataFrame,
    appearances: pd.DataFrame,
    cohort_player_ids: Iterable[int],
    min_date: pd.Timestamp = STUDY_START_WITH_WARMUP,
    max_date: pd.Timestamp = STUDY_END,
) -> pd.DataFrame:
    """Enumerate national-team rows available in the dated bulk snapshot.

    The upstream bulk files and the player-performance histories originate
    from the same public website, so this is a completeness/reproducibility
    audit rather than independent validation.  Exact counts make it explicit
    whether the bulk snapshot can replace any endpoint-derived exposure row.
    """
    _required_columns(
        competitions,
        ["competition_id", "type"],
        "snapshot competitions",
    )
    _required_columns(
        games,
        ["game_id", "competition_id", "date"],
        "snapshot games",
    )
    _required_columns(
        appearances,
        ["game_id", "player_id", "date"],
        "snapshot appearances",
    )
    national_competitions = set(
        competitions.loc[
            competitions["type"].eq("national_team_competition"),
            "competition_id",
        ]
    )
    national_games = games.loc[
        games["competition_id"].isin(national_competitions)
    ].copy()
    national_games["date"] = pd.to_datetime(national_games["date"], errors="coerce")
    in_window_games = national_games.loc[
        national_games["date"].between(min_date, max_date)
    ]
    national_appearances = appearances.loc[
        appearances["game_id"].isin(national_games["game_id"])
    ].copy()
    national_appearances["date"] = pd.to_datetime(
        national_appearances["date"], errors="coerce"
    )
    in_window_appearances = national_appearances.loc[
        national_appearances["game_id"].isin(in_window_games["game_id"])
    ]
    cohort_ids = {int(value) for value in cohort_player_ids}
    cohort_rows = in_window_appearances.loc[
        pd.to_numeric(in_window_appearances["player_id"], errors="coerce").isin(
            cohort_ids
        )
    ]
    dated_appearances = national_appearances["date"].dropna()
    first_date = (
        dated_appearances.min().strftime("%Y-%m-%d")
        if not dated_appearances.empty
        else ""
    )
    last_date = (
        dated_appearances.max().strftime("%Y-%m-%d")
        if not dated_appearances.empty
        else ""
    )
    rows = [
        (
            "snapshot_national_match_rows_all_dates",
            int(len(national_games)),
            "exact rows",
            "National-team competition matches in the immutable bulk snapshot.",
        ),
        (
            "snapshot_national_match_rows_in_acquisition_window",
            int(len(in_window_games)),
            "exact rows",
            "National-team competition matches dated inside the prespecified acquisition window.",
        ),
        (
            "snapshot_national_appearance_rows_all_dates",
            int(len(national_appearances)),
            "exact rows",
            "Player appearances linked to national-team competition matches at any date.",
        ),
        (
            "snapshot_national_appearance_rows_in_acquisition_window",
            int(len(in_window_appearances)),
            "exact rows",
            "Bulk-snapshot national player appearances available for the exposure window.",
        ),
        (
            "snapshot_cohort_national_appearance_rows_in_acquisition_window",
            int(len(cohort_rows)),
            "exact rows",
            "In-window bulk-snapshot national appearances belonging to the frozen EPL cohort.",
        ),
        (
            "snapshot_first_national_appearance_date",
            first_date,
            "calendar date",
            "Earliest national player-appearance date present in the bulk snapshot.",
        ),
        (
            "snapshot_last_national_appearance_date",
            last_date,
            "calendar date",
            "Latest national player-appearance date present in the bulk snapshot.",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "interpretation"])


def _timestamp_parts(value: Any) -> tuple[pd.Timestamp, object, bool]:
    """Return date, UTC timestamp and known-time flag from a source date object."""
    if not isinstance(value, Mapping):
        return pd.NaT, pd.NA, False
    utc_value = value.get("dateTimeUTC")
    local_value = value.get("dateTimeLocalized")
    parsed_utc = pd.to_datetime(utc_value, errors="coerce", utc=True)
    parsed_local = pd.to_datetime(local_value, errors="coerce")
    if not pd.isna(parsed_local) and getattr(parsed_local, "tzinfo", None) is not None:
        parsed_local = parsed_local.tz_localize(None)
    date = parsed_utc.tz_localize(None).normalize() if not pd.isna(parsed_utc) else parsed_local.normalize()
    known = bool(value.get("isTimeDefined", False)) and not pd.isna(parsed_utc)
    return date, parsed_utc if known else pd.NA, known


def national_performance_record_audit(
    payload: Mapping[str, Any],
    player_id: int,
    source_url: str,
    cache_file: str,
    retrieved_at_utc: str,
    min_date: pd.Timestamp = STUDY_START_WITH_WARMUP,
    max_date: pd.Timestamp = STUDY_END,
) -> pd.DataFrame:
    """Retain every in-window national record and its exposure decision."""
    performance = payload.get("data", {}).get("performance", [])
    if not isinstance(performance, list):
        raise ValueError("Performance payload has no list-valued data.performance")
    rows: list[dict[str, Any]] = []
    for raw in performance:
        game = raw.get("gameInformation", {})
        clubs = raw.get("clubsInformation", {})
        statistics = raw.get("statistics", {})
        if not game.get("isNationalGame", False):
            continue
        date, kickoff_utc, kickoff_time_known = _timestamp_parts(game.get("date"))
        if pd.isna(date) or date < min_date or date > max_date:
            continue
        playing = statistics.get("playingTimeStatistics", {})
        general = statistics.get("generalStatistics", {})
        goals = statistics.get("goalStatistics", {})
        minutes = pd.to_numeric(playing.get("playedMinutes"), errors="coerce")
        participation = general.get("participationState")
        retained = participation == "played" and not pd.isna(minutes)
        if participation != "played":
            exclusion_reason = "not_played"
        elif pd.isna(minutes):
            exclusion_reason = "played_missing_minutes"
        else:
            exclusion_reason = ""
        club = clubs.get("club", {})
        opponent = clubs.get("opponent", {})
        rows.append(
            {
                "tm_player_id": int(player_id),
                "game_id": str(game.get("gameId", "")),
                "date": date,
                "kickoff_utc": kickoff_utc,
                "kickoff_time_known": kickoff_time_known,
                "team_id": pd.to_numeric(club.get("clubId"), errors="coerce"),
                "opponent_team_id": pd.to_numeric(opponent.get("clubId"), errors="coerce"),
                "team_venue": club.get("venue", pd.NA),
                "team_goals": pd.to_numeric(club.get("goalsTotal"), errors="coerce"),
                "opponent_goals": pd.to_numeric(
                    club.get("opponentGoalsTotal"), errors="coerce"
                ),
                "game_duration_minutes": pd.to_numeric(
                    game.get("gameDuration"), errors="coerce"
                ),
                "team_goals_on_pitch": pd.to_numeric(
                    goals.get("teamGoalsOnThePitch"), errors="coerce"
                ),
                "opponent_goals_on_pitch": pd.to_numeric(
                    goals.get("opponentGoalsOnThePitch"), errors="coerce"
                ),
                "competition_id": game.get("competitionId", pd.NA),
                "competition_group_id": game.get("competitionGroupId", pd.NA),
                "competition_type_id": game.get("competitionTypeId", pd.NA),
                "season": game.get("seasonId", pd.NA),
                "stadium_id": game.get("stadiumId", pd.NA),
                "minutes_played": float(minutes) if not pd.isna(minutes) else pd.NA,
                "is_starter": bool(playing.get("isStarting", False)),
                "participation_state": participation,
                "source_url": source_url,
                "cache_file": cache_file,
                "retrieved_at_utc": retrieved_at_utc,
                "retained_for_exposure": retained,
                "exclusion_reason": exclusion_reason,
            }
        )
    return pd.DataFrame(rows, columns=NATIONAL_RECORD_AUDIT_COLUMNS)


def normalise_national_performance(
    payload: Mapping[str, Any],
    player_id: int,
    source_url: str,
    cache_file: str,
    retrieved_at_utc: str,
    min_date: pd.Timestamp = STUDY_START_WITH_WARMUP,
    max_date: pd.Timestamp = STUDY_END,
) -> pd.DataFrame:
    """Return played national appearances with known exposure minutes."""
    audit = national_performance_record_audit(
        payload,
        player_id,
        source_url,
        cache_file,
        retrieved_at_utc,
        min_date=min_date,
        max_date=max_date,
    )
    return audit.loc[audit["retained_for_exposure"].astype(bool), NATIONAL_COLUMNS].reset_index(drop=True)


def acquire_national_appearances(
    player_ids: Iterable[int],
    cache_dir: Path,
    request_log_path: Path,
    session: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    request_interval_seconds: float = 0.75,
    retries: int = 3,
    record_audit_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Acquire cached national histories with explicit retry and failure records."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    client.headers.update({"User-Agent": SOURCE_USER_AGENT})
    all_rows: list[pd.DataFrame] = []
    audit_rows: list[pd.DataFrame] = []
    log_rows: list[dict[str, Any]] = []
    for raw_player_id in sorted({int(value) for value in player_ids}):
        player_id = int(raw_player_id)
        url = NATIONAL_PERFORMANCE_URL.format(player_id=player_id)
        cache_path = cache_dir / f"{player_id}.json"
        retrieved_at = utc_now()
        status = "cached"
        attempts = 0
        try:
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                status = "requested"
                last_error: Exception | None = None
                payload = None
                for attempts in range(1, retries + 1):
                    try:
                        response = client.get(url, timeout=45)
                        response.raise_for_status()
                        payload = response.json()
                        cache_path.write_text(json.dumps(payload), encoding="utf-8")
                        status = "downloaded"
                        break
                    except (requests.RequestException, ValueError) as error:
                        last_error = error
                        if attempts < retries:
                            sleep(request_interval_seconds * attempts)
                if payload is None:
                    raise RuntimeError(str(last_error))
            record_audit = national_performance_record_audit(
                payload,
                player_id,
                url,
                str(cache_path),
                retrieved_at,
            )
            frame = record_audit.loc[
                record_audit["retained_for_exposure"].astype(bool), NATIONAL_COLUMNS
            ].reset_index(drop=True)
            all_rows.append(frame)
            audit_rows.append(record_audit)
            played = record_audit["participation_state"].eq("played")
            log_rows.append(
                {
                    "tm_player_id": player_id,
                    "source_url": url,
                    "cache_file": str(cache_path),
                    "status": status,
                    "attempts": attempts,
                    "n_normalised_rows": len(frame),
                    "n_national_records": len(record_audit),
                    "n_played_records": int(played.sum()),
                    "n_played_missing_minutes": int(
                        (played & record_audit["minutes_played"].isna()).sum()
                    ),
                    "n_nonplayed_records": int((~played).sum()),
                    "error": "",
                    "retrieved_at_utc": retrieved_at,
                }
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            log_rows.append(
                {
                    "tm_player_id": player_id,
                    "source_url": url,
                    "cache_file": str(cache_path),
                    "status": "error",
                    "attempts": attempts,
                    "n_normalised_rows": 0,
                    "n_national_records": 0,
                    "n_played_records": 0,
                    "n_played_missing_minutes": 0,
                    "n_nonplayed_records": 0,
                    "error": str(error),
                    "retrieved_at_utc": retrieved_at,
                }
            )
        if status != "cached" and request_interval_seconds > 0:
            sleep(request_interval_seconds)
    nonempty_rows = [frame for frame in all_rows if not frame.empty]
    result = pd.concat(nonempty_rows, ignore_index=True) if nonempty_rows else pd.DataFrame(columns=NATIONAL_COLUMNS)
    request_log = pd.DataFrame(log_rows)
    request_log_path.parent.mkdir(parents=True, exist_ok=True)
    request_log.to_csv(request_log_path, index=False)
    if record_audit_path is not None:
        record_audit_path = Path(record_audit_path)
        record_audit_path.parent.mkdir(parents=True, exist_ok=True)
        nonempty_audits = [frame for frame in audit_rows if not frame.empty]
        complete_audit = (
            pd.concat(nonempty_audits, ignore_index=True)
            if nonempty_audits
            else pd.DataFrame(columns=NATIONAL_RECORD_AUDIT_COLUMNS)
        )
        complete_audit.to_csv(record_audit_path, index=False)
    return result, request_log


def main() -> None:  # pragma: no cover
    """Download the snapshot and cached national records for the frozen cohort."""
    root = Path(__file__).resolve().parents[1]
    raw_root = root / "data" / "raw" / "public_data_v4"
    processed_root = root / "data" / "processed" / "public_data_v4"
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    snapshot_dir = raw_root / f"transfermarkt_datasets_{date_tag}"
    archive_path = raw_root / f"transfermarkt_datasets_{date_tag}.zip"
    if not archive_path.exists():
        download_snapshot(SNAPSHOT_URL, archive_path)
    if not snapshot_dir.exists():
        extracted = extract_snapshot(archive_path, snapshot_dir)
        response = requests.get(f"{UPSTREAM_REPOSITORY}/commit/master", timeout=30)
        upstream_commit = response.url.rstrip("/").split("/")[-1] if response.ok else "unavailable"
        write_json(
            snapshot_dir / "snapshot_manifest.json",
            snapshot_manifest(
                archive_path,
                extracted,
                utc_now(),
                upstream_commit,
                f"{UPSTREAM_REPOSITORY}/blob/master/LICENSE",
            ),
        )

    old_tm = root / "external_data" / "transfermarkt"
    panel = pd.read_csv(root / "data" / "processed" / "player_day_panel.csv", low_memory=False)
    manifest = build_cohort_manifest(
        panel,
        pd.read_csv(old_tm / "appearances.csv", low_memory=False),
        pd.read_csv(old_tm / "games.csv", low_memory=False),
        pd.read_csv(old_tm / "players.csv", low_memory=False),
        pd.read_csv(old_tm / "transfers.csv", low_memory=False),
    )
    processed_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(processed_root / "epl_cohort_manifest.csv", index=False)
    snapshot_national_audit = audit_snapshot_national_appearances(
        pd.read_csv(snapshot_dir / "competitions.csv.gz"),
        pd.read_csv(
            snapshot_dir / "games.csv.gz",
            usecols=["game_id", "competition_id", "date"],
        ),
        pd.read_csv(
            snapshot_dir / "appearances.csv.gz",
            usecols=["game_id", "player_id", "date"],
        ),
        manifest["tm_player_id"],
    )
    snapshot_national_audit.to_csv(
        processed_root / "snapshot_national_appearance_audit.csv", index=False
    )
    national, request_log = acquire_national_appearances(
        manifest["tm_player_id"],
        raw_root / "national_performance_cache",
        processed_root / "national_acquisition_log.csv",
        record_audit_path=processed_root / "international_performance_record_audit.csv",
    )
    national.to_csv(processed_root / "international_appearances_raw.csv", index=False)
    print(f"Cohort players: {len(manifest)}")
    print(f"National appearance rows: {len(national)}")
    print(
        "Bulk-snapshot cohort national rows in acquisition window: "
        f"{snapshot_national_audit.loc[snapshot_national_audit['metric'].eq('snapshot_cohort_national_appearance_rows_in_acquisition_window'), 'value'].iloc[0]}"
    )
    print(f"Acquisition errors: {(request_log['status'] == 'error').sum()}")


if __name__ == "__main__":  # pragma: no cover
    main()
