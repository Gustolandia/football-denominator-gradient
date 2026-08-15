"""Acquire and reconcile reusable public sources for the v4 extension."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import requests


INDEPENDENT_RESULTS_COMMIT = "65d212aac5deec5157071dcf6e9b05fce0223c84"
INDEPENDENT_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    f"{INDEPENDENT_RESULTS_COMMIT}/results.csv"
)
INDEPENDENT_SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    f"{INDEPENDENT_RESULTS_COMMIT}/shootouts.csv"
)
INDEPENDENT_LICENSE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    f"{INDEPENDENT_RESULTS_COMMIT}/LICENSE"
)
OPENFOOTBALL_WORLDCUP_COMMIT = "092f6b7a97b1b2cea4b2fe2b7706894a8866878b"
OPENFOOTBALL_WORLDCUP_BASE = (
    "https://raw.githubusercontent.com/openfootball/worldcup.more/"
    f"{OPENFOOTBALL_WORLDCUP_COMMIT}"
)
OPENFOOTBALL_WORLDCUP_FILES = {
    2018: f"{OPENFOOTBALL_WORLDCUP_BASE}/worldcup/2018_worldcup.txt",
    2022: f"{OPENFOOTBALL_WORLDCUP_BASE}/worldcup/2022_worldcup.txt",
}
OPENFOOTBALL_WORLDCUP_LICENSE_URL = (
    f"{OPENFOOTBALL_WORLDCUP_BASE}/LICENSE.md"
)

TEAM_ALIASES = {
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia and herzegovina": "bosnia and herzegovina",
    "cape verde": "cabo verde",
    "china pr": "china",
    "congo dr": "dr congo",
    "democratic republic of congo": "dr congo",
    "england amateurs": "england amateurs",
    "ivory coast": "cote d ivoire",
    "korea south": "south korea",
    "korea republic": "south korea",
    "republic of ireland": "ireland",
    "turkiye": "turkey",
    "u s a": "united states",
    "usa": "united states",
    "united states of america": "united states",
}


def sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    """Return a SHA-256 digest without loading the entire source into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, session: Any | None = None) -> Path:
    """Download one pinned source file, retaining an existing immutable copy."""
    destination = Path(destination)
    if destination.exists():
        return destination
    client = session or requests.Session()
    response = client.get(url, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination


def acquire_independent_results(raw_root: Path, retrieval_date: str | None = None) -> Path:
    """Cache the pinned CC0 senior-international schedule and its provenance."""
    date_label = retrieval_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    target = Path(raw_root) / f"independent_senior_results_{date_label}"
    results_path = download_file(INDEPENDENT_RESULTS_URL, target / "results.csv")
    shootouts_path = download_file(
        INDEPENDENT_SHOOTOUTS_URL, target / "shootouts.csv"
    )
    license_path = download_file(INDEPENDENT_LICENSE_URL, target / "LICENSE")
    results = pd.read_csv(results_path)
    shootouts = pd.read_csv(shootouts_path)
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/martj42/international_results",
        "source_commit": INDEPENDENT_RESULTS_COMMIT,
        "results_url": INDEPENDENT_RESULTS_URL,
        "shootouts_url": INDEPENDENT_SHOOTOUTS_URL,
        "license": "CC0-1.0",
        "license_url": INDEPENDENT_LICENSE_URL,
        "scope": "men's full senior internationals; excludes Olympic, B-team, U23 and league-select matches",
        "rows": int(len(results)),
        "shootout_rows": int(len(shootouts)),
        "minimum_date": str(results["date"].min()),
        "maximum_date": str(results["date"].max()),
        "files": {
            "results.csv": {"bytes": results_path.stat().st_size, "sha256": sha256_file(results_path)},
            "shootouts.csv": {"bytes": shootouts_path.stat().st_size, "sha256": sha256_file(shootouts_path)},
            "LICENSE": {"bytes": license_path.stat().st_size, "sha256": sha256_file(license_path)},
        },
    }
    (target / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def newest_independent_snapshot(raw_root: Path) -> Path:
    """Return the latest dated independent-results directory."""
    candidates = sorted(
        path
        for path in Path(raw_root).glob("independent_senior_results_*")
        if path.is_dir() and (path / "results.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No independent senior-results snapshot under {raw_root}")
    return candidates[-1]


def acquire_worldcup_lineups(
    raw_root: Path, retrieval_date: str | None = None
) -> Path:
    """Cache pinned CC0 World Cup lineups for player-level validation."""
    date_label = retrieval_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    target = Path(raw_root) / f"openfootball_worldcup_lineups_{date_label}"
    acquired: dict[str, dict[str, object]] = {}
    for year, url in OPENFOOTBALL_WORLDCUP_FILES.items():
        name = f"{year}_worldcup.txt"
        path = download_file(url, target / name)
        acquired[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "url": url,
        }
    license_path = download_file(
        OPENFOOTBALL_WORLDCUP_LICENSE_URL, target / "LICENSE.md"
    )
    acquired["LICENSE.md"] = {
        "bytes": license_path.stat().st_size,
        "sha256": sha256_file(license_path),
        "url": OPENFOOTBALL_WORLDCUP_LICENSE_URL,
    }
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/openfootball/worldcup.more",
        "source_commit": OPENFOOTBALL_WORLDCUP_COMMIT,
        "license": "CC0-1.0",
        "scope": "men's FIFA World Cup 2018 and 2022 match lineups and substitutions",
        "files": acquired,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return target


def newest_worldcup_lineup_snapshot(raw_root: Path) -> Path:
    """Return the latest immutable OpenFootball World Cup lineup snapshot."""
    candidates = sorted(
        path
        for path in Path(raw_root).glob("openfootball_worldcup_lineups_*")
        if path.is_dir()
        and (path / "2018_worldcup.txt").exists()
        and (path / "2022_worldcup.txt").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No OpenFootball World Cup lineup snapshot under {raw_root}")
    return candidates[-1]


def normalize_team_name(value: object) -> str:
    """Return an ASCII comparison key with a small, declared alias table."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return TEAM_ALIASES.get(text, text)


def normalize_person_name(value: object) -> str:
    """Return a conservative accent-insensitive person-name comparison key."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _split_lineup_entries(text: str) -> list[str]:
    entries: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            entries.append(text[start:index].strip())
            start = index + 1
    final = text[start:].strip().rstrip(",")
    if final:
        entries.append(final)
    return [entry for entry in entries if entry]


def _clock_minute(value: str) -> int:
    return sum(int(part) for part in value.split("+"))


def parse_openfootball_worldcup_file(path: Path, year: int) -> pd.DataFrame:
    """Parse CC0 World Cup lineups into conservative player-match records."""
    date_pattern = re.compile(
        r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) ([A-Z][a-z]{2})/(\d{1,2})(?: (\d{4}))? @"
    )
    score_pattern = re.compile(r"^\s{2}(.+?) v (.+?)\s{2,}(\d+)-(\d+)(.*)$")
    substitution_pattern = re.compile(
        r"^(.*?)\s+\((\d+(?:\+\d+)?)'\s+(.+?)\)$"
    )
    rows: list[dict[str, object]] = []
    match_date = pd.NaT
    home_team = ""
    away_team = ""
    duration = 90
    lineup_team = ""
    lineup_parts: list[str] = []

    def flush_lineup() -> None:
        nonlocal lineup_team, lineup_parts
        if not lineup_team or not lineup_parts or pd.isna(match_date):
            lineup_team = ""
            lineup_parts = []
            return
        content = " ".join(lineup_parts)
        for entry in _split_lineup_entries(content):
            substitution = substitution_pattern.match(entry)
            if substitution:
                starter, clock, substitute = substitution.groups()
                minute = min(_clock_minute(clock), duration)
                parsed = (
                    (starter.strip(), True, float(minute), clock),
                    (substitute.strip(), False, float(max(duration - minute, 0)), clock),
                )
            else:
                parsed = ((entry.strip(), True, float(duration), pd.NA),)
            for player_name, is_starter, approx_minutes, substitution_clock in parsed:
                rows.append(
                    {
                        "tournament_year": year,
                        "date": match_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "team_name": lineup_team,
                        "player_name": player_name,
                        "is_starter": is_starter,
                        "approx_minutes": approx_minutes,
                        "match_duration": duration,
                        "substitution_clock": substitution_clock,
                        "source_url": OPENFOOTBALL_WORLDCUP_FILES[year],
                    }
                )
        lineup_team = ""
        lineup_parts = []

    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        date_match = date_pattern.match(raw_line)
        if date_match:
            flush_lineup()
            month, day, explicit_year = date_match.groups()
            match_date = pd.to_datetime(
                f"{explicit_year or year}-{month}-{day}", errors="raise"
            ).normalize()
            continue
        score_match = score_pattern.match(raw_line)
        if score_match:
            flush_lineup()
            home_team, away_team, _home_score, _away_score, suffix = score_match.groups()
            duration = 120 if "aet" in suffix.casefold() else 90
            continue
        if home_team and away_team and ":" in raw_line and not raw_line.startswith(" "):
            candidate_team, content = raw_line.split(":", 1)
            if normalize_team_name(candidate_team) in {
                normalize_team_name(home_team),
                normalize_team_name(away_team),
            }:
                flush_lineup()
                lineup_team = candidate_team.strip()
                lineup_parts = [content.strip()]
                continue
        if lineup_team and raw_line.startswith("   ") and raw_line.strip():
            lineup_parts.append(raw_line.strip())
            continue
        if lineup_team:
            flush_lineup()
    flush_lineup()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["home_team_key"] = frame["home_team"].map(normalize_team_name)
    frame["away_team_key"] = frame["away_team"].map(normalize_team_name)
    frame["team_key"] = frame["team_name"].map(normalize_team_name)
    frame["player_key"] = frame["player_name"].map(normalize_person_name)
    return frame


def parse_openfootball_worldcup_snapshot(snapshot: Path) -> pd.DataFrame:
    """Combine the pinned 2018 and 2022 World Cup lineup files."""
    tables = [
        parse_openfootball_worldcup_file(Path(snapshot) / f"{year}_worldcup.txt", year)
        for year in sorted(OPENFOOTBALL_WORLDCUP_FILES)
    ]
    return pd.concat(tables, ignore_index=True)


def validate_worldcup_lineups(
    record_audit: pd.DataFrame,
    cohort: pd.DataFrame,
    team_crosswalk: pd.DataFrame,
    source_lineups: pd.DataFrame,
) -> pd.DataFrame:
    """Validate observed World Cup play, starts, and minutes without imputation."""
    audit_required = {
        "tm_player_id",
        "game_id",
        "date",
        "team_id",
        "opponent_team_id",
        "competition_id",
        "participation_state",
        "minutes_played",
        "is_starter",
    }
    cohort_required = {"tm_player_id", "name"}
    crosswalk_required = {"national_team_id", "normalized_team_name"}
    lineup_required = {
        "date",
        "home_team_key",
        "away_team_key",
        "team_key",
        "player_key",
        "player_name",
        "is_starter",
        "approx_minutes",
    }
    for frame, required, label in (
        (record_audit, audit_required, "national record audit"),
        (cohort, cohort_required, "cohort"),
        (team_crosswalk, crosswalk_required, "team crosswalk"),
        (source_lineups, lineup_required, "World Cup lineups"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{label} missing columns: {missing}")
    records = record_audit.loc[
        record_audit["competition_id"].eq("FIWC")
        & record_audit["participation_state"].eq("played")
    ].copy()
    records["date"] = pd.to_datetime(records["date"], errors="coerce").dt.normalize()
    records = records.merge(cohort[["tm_player_id", "name"]], on="tm_player_id", how="left")
    name_map = team_crosswalk.set_index("national_team_id")["normalized_team_name"]
    records["team_key"] = pd.to_numeric(records["team_id"], errors="coerce").astype("Int64").map(name_map)
    records["opponent_key"] = pd.to_numeric(
        records["opponent_team_id"], errors="coerce"
    ).astype("Int64").map(name_map)
    records["player_key"] = records["name"].map(normalize_person_name)

    def match_key(frame: pd.DataFrame, first: str, second: str) -> pd.Series:
        date = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        low = frame[[first, second]].fillna("").min(axis=1)
        high = frame[[first, second]].fillna("").max(axis=1)
        return date + "|" + low + "|" + high

    records["source_match_key"] = match_key(records, "team_key", "opponent_key")
    source = source_lineups.copy()
    source["source_match_key"] = match_key(source, "home_team_key", "away_team_key")
    source_matches = set(source["source_match_key"])
    source["source_duplicate_count"] = source.groupby(
        ["source_match_key", "team_key", "player_key"]
    )["player_key"].transform("size")
    source = source.loc[source["source_duplicate_count"].eq(1)].rename(
        columns={
            "player_name": "source_player_name",
            "is_starter": "source_is_starter",
            "approx_minutes": "source_approx_minutes",
        }
    )
    keep = [
        "source_match_key",
        "team_key",
        "player_key",
        "source_player_name",
        "source_is_starter",
        "source_approx_minutes",
        "match_duration",
        "source_url",
    ]
    out = records.merge(source[keep], on=["source_match_key", "team_key", "player_key"], how="left")
    out["source_match_found"] = out["source_match_key"].isin(source_matches)
    out["source_player_found"] = out["source_player_name"].notna()
    out["starter_agreement"] = pd.NA
    matched = out["source_player_found"]
    out.loc[matched, "starter_agreement"] = (
        out.loc[matched, "is_starter"].astype(bool)
        == out.loc[matched, "source_is_starter"].astype(bool)
    )
    observed_minutes = pd.to_numeric(out["minutes_played"], errors="coerce")
    source_minutes = pd.to_numeric(out["source_approx_minutes"], errors="coerce")
    out["minute_absolute_difference"] = (observed_minutes - source_minutes).abs()
    out["minutes_within_5"] = pd.NA
    comparable = matched & observed_minutes.notna() & source_minutes.notna()
    out.loc[comparable, "minutes_within_5"] = out.loc[
        comparable, "minute_absolute_difference"
    ].le(5.0)
    out["validation_method"] = "exact_date_team_pair_and_normalized_player_name"
    return out


def _seed_team_crosswalk(appearances: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    proposals: dict[int, list[str]] = defaultdict(list)
    display: dict[int, list[str]] = defaultdict(list)
    for identifier_column, name_column in (
        ("team_id", "team_name"),
        ("opponent_team_id", "opponent_team_name"),
    ):
        for identifier, name in appearances[[identifier_column, name_column]].dropna().itertuples(index=False):
            parsed = pd.to_numeric(identifier, errors="coerce")
            normal = normalize_team_name(name)
            if pd.isna(parsed) or not normal:
                continue
            proposals[int(parsed)].append(normal)
            display[int(parsed)].append(str(name))
    mapping: dict[int, str] = {}
    labels: dict[int, str] = {}
    for identifier, values in proposals.items():
        if len(set(values)) == 1:
            mapping[identifier] = values[0]
            labels[identifier] = Counter(display[identifier]).most_common(1)[0][0]
    return mapping, labels


def _prepared_results(results: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "home_team", "away_team", "tournament", "city", "country", "neutral"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise KeyError(f"independent results missing columns: {missing}")
    out = results.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["home_key"] = out["home_team"].map(normalize_team_name)
    out["away_key"] = out["away_team"].map(normalize_team_name)
    return out


def _prepared_shootouts(shootouts: pd.DataFrame | None) -> set[tuple[pd.Timestamp, str, str]]:
    """Return identity keys for independently recorded penalty shootouts."""
    if shootouts is None:
        return set()
    required = {"date", "home_team", "away_team"}
    missing = sorted(required - set(shootouts.columns))
    if missing:
        raise KeyError(f"independent shootouts missing columns: {missing}")
    out = shootouts.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["home_key"] = out["home_team"].map(normalize_team_name)
    out["away_key"] = out["away_team"].map(normalize_team_name)
    return set(out[["date", "home_key", "away_key"]].itertuples(index=False, name=None))


def _candidate_matches(
    results_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    match_date: pd.Timestamp,
    team_key: str,
    opponent_key: str,
) -> pd.DataFrame:
    dated = results_by_date.get(match_date)
    if dated is None:
        return pd.DataFrame(columns=["home_key", "away_key"])
    if team_key and opponent_key:
        return dated.loc[
            ((dated["home_key"].eq(team_key)) & (dated["away_key"].eq(opponent_key)))
            | ((dated["home_key"].eq(opponent_key)) & (dated["away_key"].eq(team_key)))
        ]
    known = team_key or opponent_key
    if known:
        return dated.loc[dated["home_key"].eq(known) | dated["away_key"].eq(known)]
    return dated.iloc[0:0]


def _assign_external_names(
    row: pd.Series,
    external: pd.Series,
    mapping: dict[int, str],
) -> list[tuple[int, str, str]]:
    team_id = pd.to_numeric(row["team_id"], errors="coerce")
    opponent_id = pd.to_numeric(row["opponent_team_id"], errors="coerce")
    if pd.isna(team_id) or pd.isna(opponent_id):
        return []
    team_id = int(team_id)
    opponent_id = int(opponent_id)
    known_team = mapping.get(team_id, normalize_team_name(row.get("team_name")))
    known_opponent = mapping.get(
        opponent_id, normalize_team_name(row.get("opponent_team_name"))
    )
    if known_team == external["home_key"]:
        return [(team_id, external["home_key"], external["home_team"]), (opponent_id, external["away_key"], external["away_team"])]
    if known_team == external["away_key"]:
        return [(team_id, external["away_key"], external["away_team"]), (opponent_id, external["home_key"], external["home_team"])]
    if known_opponent == external["home_key"]:
        return [(team_id, external["away_key"], external["away_team"]), (opponent_id, external["home_key"], external["home_team"])]
    if known_opponent == external["away_key"]:
        return [(team_id, external["home_key"], external["home_team"]), (opponent_id, external["away_key"], external["away_team"])]
    return []


def harmonize_national_matches(
    appearances: pd.DataFrame,
    independent_results: pd.DataFrame,
    independent_shootouts: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fill auditable team labels and independently validate senior match dates.

    The external archive contains full senior internationals only. Youth and
    Olympic records remain outside this reconciliation, and no national-team
    appearance is inferred from citizenship or profile metadata.
    """
    required = {
        "game_id",
        "date",
        "team_id",
        "opponent_team_id",
        "team_name",
        "opponent_team_name",
        "team_venue",
        "team_level",
        "competition_id",
    }
    missing = sorted(required - set(appearances.columns))
    if missing:
        raise KeyError(f"national appearances missing columns: {missing}")
    out = appearances.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    results = _prepared_results(independent_results)
    shootout_keys = _prepared_shootouts(independent_shootouts)
    results_by_date = {
        date: group.reset_index(drop=True)
        for date, group in results.groupby("date", sort=False)
    }
    mapping, labels = _seed_team_crosswalk(out)
    mapping_source = {identifier: "transfermarkt_snapshot" for identifier in mapping}
    senior_games = out.loc[out["team_level"].eq("senior")].drop_duplicates(
        ["game_id", "team_id", "opponent_team_id"]
    )

    for _ in range(8):
        proposals: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for _, row in senior_games.iterrows():
            team_id = pd.to_numeric(row["team_id"], errors="coerce")
            opponent_id = pd.to_numeric(row["opponent_team_id"], errors="coerce")
            team_key = mapping.get(int(team_id), "") if not pd.isna(team_id) else ""
            opponent_key = mapping.get(int(opponent_id), "") if not pd.isna(opponent_id) else ""
            candidates = _candidate_matches(
                results_by_date, row["date"], team_key, opponent_key
            )
            if len(candidates) != 1:
                continue
            for identifier, normal, display in _assign_external_names(row, candidates.iloc[0], mapping):
                proposals[identifier].append((normal, str(display)))
        added = 0
        for identifier, values in proposals.items():
            normal_values = {value[0] for value in values if value[0]}
            if len(normal_values) != 1:
                continue
            normal = next(iter(normal_values))
            if identifier in mapping and mapping[identifier] != normal:
                continue
            if identifier not in mapping:
                mapping[identifier] = normal
                labels[identifier] = Counter(value[1] for value in values).most_common(1)[0][0]
                mapping_source[identifier] = "independent_schedule_unique_date_team_match"
                added += 1
        if added == 0:
            break

    out["team_name_source"] = out["team_name"].notna().map(
        {True: "transfermarkt_snapshot", False: "missing"}
    )
    out["opponent_team_name_source"] = out["opponent_team_name"].notna().map(
        {True: "transfermarkt_snapshot", False: "missing"}
    )
    for identifier_column, name_column, source_column in (
        ("team_id", "team_name", "team_name_source"),
        ("opponent_team_id", "opponent_team_name", "opponent_team_name_source"),
    ):
        ids = pd.to_numeric(out[identifier_column], errors="coerce").astype("Int64")
        missing_name = out[name_column].isna()
        inferred = ids.map(labels)
        out.loc[missing_name, name_column] = inferred.loc[missing_name]
        filled = missing_name & inferred.notna()
        out.loc[filled, source_column] = ids.loc[filled].map(mapping_source)

    validation_rows: list[dict[str, object]] = []
    metadata_by_game: dict[str, dict[str, object]] = {}
    orientations = out.loc[out["team_level"].eq("senior")].copy()
    score_rank = {
        "published_snapshot_game": 2,
        "full_match_on_pitch": 1,
        "unavailable": 0,
    }
    orientations["_score_rank"] = (
        orientations.get(
            "validation_score_source", pd.Series("unavailable", index=orientations.index)
        )
        .map(score_rank)
        .fillna(0)
    )
    orientations = orientations.sort_values("_score_rank", ascending=False).drop_duplicates(
        ["game_id", "team_id", "opponent_team_id", "competition_id", "date"]
    )
    for _, row in orientations.iterrows():
        team_key = normalize_team_name(row["team_name"])
        opponent_key = normalize_team_name(row["opponent_team_name"])
        candidates = _candidate_matches(
            results_by_date, row["date"], team_key, opponent_key
        )
        matched = len(candidates) == 1 and bool(team_key and opponent_key)
        external = candidates.iloc[0] if matched else None
        method = "exact_date_unordered_team_pair" if matched else ("ambiguous" if len(candidates) > 1 else "unmatched")
        team_goals = pd.to_numeric(
            row.get("validation_team_goals", row.get("team_goals")), errors="coerce"
        )
        opponent_goals = pd.to_numeric(
            row.get("validation_opponent_goals", row.get("opponent_goals")),
            errors="coerce",
        )
        validation_score_source = row.get("validation_score_source", "legacy_endpoint_total")
        if matched and team_key == external["home_key"] and opponent_key == external["away_key"]:
            observed_home_score, observed_away_score = team_goals, opponent_goals
            score_orientation_method = "matched_team_names"
        elif matched and team_key == external["away_key"] and opponent_key == external["home_key"]:
            observed_home_score, observed_away_score = opponent_goals, team_goals
            score_orientation_method = "matched_team_names"
        else:
            observed_home_score, observed_away_score = np.nan, np.nan
            score_orientation_method = "not_orientable"
        external_home_score = (
            pd.to_numeric(external.get("home_score", pd.NA), errors="coerce")
            if matched
            else np.nan
        )
        external_away_score = (
            pd.to_numeric(external.get("away_score", pd.NA), errors="coerce")
            if matched
            else np.nan
        )
        score_values_available = bool(
            matched
            and pd.notna(observed_home_score)
            and pd.notna(observed_away_score)
            and pd.notna(external_home_score)
            and pd.notna(external_away_score)
        )
        shootout_match = bool(
            matched
            and (
                row["date"], external["home_key"], external["away_key"]
            )
            in shootout_keys
        )
        validation_rows.append(
            {
                "team_id": row["team_id"],
                "opponent_team_id": row["opponent_team_id"],
                "competition_id": row["competition_id"],
                "date": row["date"],
                "game_id": row["game_id"],
                "team_venue": row.get("team_venue", pd.NA),
                "verified": matched,
                "source_authority": "independent_secondary",
                "validation_method": method,
                "source_url": INDEPENDENT_RESULTS_URL,
                "external_home_team": external["home_team"] if matched else pd.NA,
                "external_away_team": external["away_team"] if matched else pd.NA,
                "observed_home_score": observed_home_score,
                "observed_away_score": observed_away_score,
                "external_home_score": external_home_score,
                "external_away_score": external_away_score,
                "score_orientation_method": score_orientation_method,
                "observed_score_source": validation_score_source,
                "score_values_available": score_values_available,
                "independent_shootout_match": shootout_match,
                "score_agreement": (
                    bool(
                        observed_home_score == external_home_score
                        and observed_away_score == external_away_score
                    )
                    if score_values_available
                    else pd.NA
                ),
                "external_tournament": external["tournament"] if matched else pd.NA,
                "external_city": external["city"] if matched else pd.NA,
                "external_country": external["country"] if matched else pd.NA,
                "external_neutral": external["neutral"] if matched else pd.NA,
            }
        )
        if matched:
            metadata_by_game[str(row["game_id"])] = {
                "independent_schedule_verified": True,
                "independent_validation_method": method,
                "external_tournament": external["tournament"],
                "external_city": external["city"],
                "external_country": external["country"],
                "external_neutral": external["neutral"],
            }
    validation = pd.DataFrame(validation_rows)
    if not validation.empty:
        validation["score_values_available"] = validation[
            "score_values_available"
        ].astype("boolean")
        validation["score_agreement"] = validation["score_agreement"].astype(
            "boolean"
        )
        validation["score_internal_consistency"] = pd.Series(
            pd.NA, index=validation.index, dtype="boolean"
        )
        for _game_id, group in validation.groupby("game_id", dropna=False):
            recorded = group["score_values_available"].astype("boolean").fillna(False)
            if not recorded.any():
                continue
            sources = set(group.loc[recorded, "observed_score_source"].astype(str))
            if sources & {"published_snapshot_game", "legacy_endpoint_total"}:
                consistent: object = True
            else:
                pairs = group.loc[
                    recorded, ["observed_home_score", "observed_away_score"]
                ].drop_duplicates()
                orientations_present = group.loc[recorded, "team_id"].nunique() >= 2
                consistent = bool(len(pairs) == 1) if orientations_present else pd.NA
            validation.loc[group.index, "score_internal_consistency"] = consistent
            eligible = consistent is True
            if not eligible:
                validation.loc[group.index, "score_values_available"] = False
                validation.loc[group.index, "score_agreement"] = pd.NA
        score_available = validation["score_values_available"].astype("boolean").fillna(False)
        internal_conflict = validation["score_internal_consistency"].eq(False).fillna(False)
        shootout = validation["independent_shootout_match"].astype("boolean").fillna(False)
        agreement = validation["score_agreement"].astype("boolean").fillna(False)
        validation["score_discrepancy_reason"] = "no_reliable_match_level_score"
        validation.loc[internal_conflict, "score_discrepancy_reason"] = (
            "conflicting_player_level_score_pairs"
        )
        validation.loc[score_available & agreement, "score_discrepancy_reason"] = (
            "exact_agreement"
        )
        validation.loc[
            score_available & ~agreement & shootout, "score_discrepancy_reason"
        ] = "known_shootout_score_convention"
        validation.loc[
            score_available & ~agreement & ~shootout, "score_discrepancy_reason"
        ] = "unresolved_cross_source_score_conflict"
    metadata = pd.DataFrame.from_dict(metadata_by_game, orient="index")
    metadata.index.name = "game_id"
    metadata = metadata.reset_index()
    out["game_id"] = out["game_id"].astype(str)
    if not metadata.empty:
        out = out.merge(metadata, on="game_id", how="left")
    else:
        for column in (
            "independent_schedule_verified",
            "independent_validation_method",
            "external_tournament",
            "external_city",
            "external_country",
            "external_neutral",
        ):
            out[column] = pd.NA
    out["independent_schedule_verified"] = (
        out["independent_schedule_verified"].astype("boolean").fillna(False).astype(bool)
    )
    crosswalk = pd.DataFrame(
        [
            {
                "national_team_id": identifier,
                "normalized_team_name": normal,
                "display_team_name": labels[identifier],
                "mapping_source": mapping_source[identifier],
            }
            for identifier, normal in sorted(mapping.items())
        ]
    )
    return out, validation, crosswalk


def public_source_catalog() -> pd.DataFrame:
    """Document candidate sources, accepted roles, and explicit rejections."""
    rows = [
        ("Transfermarkt datasets", "https://github.com/dcaribou/transfermarkt-datasets", "accepted_primary_public_proxy", "club match data and same-source national completeness audit", "dated 20260803 snapshot and hashes retained; national player appearances began after the study window and added zero cohort exposure rows"),
        ("Transfermarkt player performance endpoint", "https://www.transfermarkt.co.uk/", "accepted_primary_public_proxy", "cohort national performance and participation states", "undocumented public endpoint; raw responses cached"),
        ("Mart Jürisoo international results", "https://github.com/martj42/international_results", "accepted_independent_validation", "senior match date/team/tournament/city confirmation", "CC0; excludes youth, Olympic and B-team matches"),
        ("OpenFootball World Cup More", "https://github.com/openfootball/worldcup.more", "accepted_independent_player_validation", "World Cup 2018 and 2022 participation, starter and approximate-minute checks", "CC0; used only for independent validation and never to impute exposure"),
        ("OpenFootball internationals", "https://github.com/openfootball/internationals", "rejected_redundant", "structured senior schedules", "CC0 mirror of the accepted Mart Jürisoo source"),
        ("StatsBomb Open Data", "https://github.com/hudl/open-data", "rejected_usage_constraints", "World Cup lineups and event timing", "public user agreement restricts redistribution and requires branded publication attribution; a CC0 validation source was available"),
        ("Fjelstul World Cup Database", "https://github.com/jfjelstul/worldcup", "rejected_license_mixing", "World Cup appearances and substitutions", "CC BY-SA derivative licensing was avoided because a CC0 lineup source covered the same tournaments"),
        ("FIFA Data Centre", "https://inside.fifa.com/data-centre/teams", "manual_official_validation_only", "official match-history pages and venue context", "archive detail varies; no reusable bulk export adopted"),
        ("UEFA match services", "https://match.uefa.com/v5/matches", "rejected_automated_acquisition", "official European schedules and lineups", "UEFA terms prohibit automated collection"),
        ("FIFPRO Player Workload Monitoring", "https://footballbenchmark.com/fifpro", "external_benchmark_only", "club/country minutes, travel and recovery context", "public dashboard/reports but no cohort-level reusable bulk export"),
        ("GeoNames", "https://www.geonames.org/export/", "candidate_not_used", "city coordinates", "CC BY; city-centroid travel would not be observed player travel"),
        ("Wikidata", "https://www.wikidata.org/wiki/Wikidata:Data_access", "candidate_not_used", "venue coordinates and identifiers", "CC0; unresolved stadium entity matching would dominate uncertainty"),
        ("Football-Data", "https://www.football-data.co.uk/data.php", "rejected_scope_mismatch", "league results and betting fields", "does not add national player minutes or historical injury timing"),
        ("football-data.org API", "https://www.football-data.org/", "rejected_scope_and_access_mismatch", "fixture, score, squad and lineup cross-checks", "registered access is rate-limited and does not add a reusable historical national-player-minute census for this cohort"),
        ("FIFA Connect data-exchange standard", "https://data.fifaconnect.org/", "rejected_schema_not_dataset", "standardised match and roster field definitions", "publishes a schema rather than historical cohort records"),
        ("SoccerMon", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11139986/", "external_methodology_benchmark_only", "prospective training, wellness and injury monitoring", "clinically richer but unrelated players, seasons and competition; cannot be linked to or used to fill this cohort"),
    ]
    return pd.DataFrame(rows, columns=["source", "url", "decision", "potential_role", "reason_or_limit"])
