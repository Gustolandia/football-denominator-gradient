"""Assemble a frozen appearance snapshot for women's domestic leagues.

The denominator gradient is a property of the denominator alone: it is
estimated from appearance dates, recorded minutes and lineup role, and needs
no injury data. That is what lets the same diagnostic run in women's football,
where the injury record is a different and much weaker instrument (see
``src/41_womens_injury_record_audit.py``) but the appearance record is not.

The men's snapshot comes from a Transfermarkt dump. The women's competitions
are absent from that dump entirely, so appearances here come from FBref match
reports instead. Two sources for one comparison is a real threat to it, which
is why the same fit is repeated for a men's league on both sources in
``src/40_womens_denominator_gradient.py``: if the gradient agrees there, the
source is not carrying the result.

The fetch itself happens outside this module, because FBref refuses scripted
clients and the pages must be read through a browser. What arrives here is a
list of extracted records, and this module treats that list as untrusted: it
checks that every match carries two full starting elevens, that recorded
minutes are inside a match, and that no player appears twice in the same
match, before any of it becomes a snapshot. A silent extraction failure would
otherwise look exactly like a squad that rotates heavily, which is precisely
the signal the gradient measures.

Player identifiers are replaced with surrogates drawn at random, never derived
from the source identifier, for the reason set out in ``src/audit_identity``:
a hash of a public identifier is reversible by brute force over a few thousand
candidates, so it is not de-identification at all.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

COUNTRY_COL = "country"
COMPETITION_COL = "competition_id"
SEASON_COL = "season"
DATE_COL = "date"
MATCH_COL = "match_id"
SOURCE_PLAYER_COL = "fbref_player_id"
PLAYER_ID_COL = "player_id"
MINUTES_COL = "minutes_played"
ROLE_COL = "lineup_role"

STARTER_ROLE = "starting_lineup"
SUBSTITUTE_ROLE = "substitutes"

#: Some leagues are covered for minutes but carry no lineup box, so who
#: started cannot be read. That is a different thing from nobody starting, and
#: recording it as ``substitutes`` would put every appearance in that league on
#: the wrong side of the within-starter fit. The gradient module treats a
#: league with no starters as one whose within-starter gradient is absent,
#: which is the honest answer, so role-less leagues are marked rather than
#: guessed or dropped.
UNKNOWN_ROLE = "unknown"

RAW_RECORD_FIELDS = (
    COUNTRY_COL,
    COMPETITION_COL,
    SEASON_COL,
    DATE_COL,
    MATCH_COL,
    SOURCE_PLAYER_COL,
    MINUTES_COL,
    ROLE_COL,
)

#: European women's first tiers, keyed by the competition code of whichever
#: source supplied them. Two sources appear because they fail differently:
#: FBref numeric codes carry minutes but omit the lineup box for the Nordic
#: leagues, while Soccerdonna's alphabetic codes carry both. Registering both
#: lets the same league be measured from each and the two compared, which is
#: the only honest answer to the objection that the women's panel comes from a
#: different provider than the men's.
#:
#: The Nordic leagues run calendar-year seasons, which costs the gradient
#: nothing: its exposure windows are rolling dates, not season labels.
WOMENS_LEAGUES: Mapping[str, str] = {
    # FBref
    "189": "England, FA Women's Super League",
    "183": "Germany, Frauen-Bundesliga",
    "193": "France, Première Ligue",
    "208": "Italy, Serie A Femminile",
    "195": "Netherlands, Eredivisie Vrouwen",
    "185": "Norway, Toppserien",
    "187": "Sweden, Damallsvenskan",
    "340": "Denmark, Kvindeligaen",
    # Soccerdonna
    "ENG1": "England, FA Women's Super League",
    "BL1": "Germany, Frauen-Bundesliga",
    "IT1": "Italy, Serie A Femminile",
    "ESP1": "Spain, Primera División Femenina",
    "SWE1": "Sweden, Damallsvenskan",
    "NOR1": "Norway, Toppserien",
    "SUI1": "Switzerland, Women's Super League",
}

#: A league match starts twenty-two players, so a complete match report records
#: eleven starters a side. Anything else means the lineup box was missing or
#: only partly parsed.
STARTERS_PER_SIDE = 11
STARTERS_PER_MATCH = STARTERS_PER_SIDE * 2

#: Recorded minutes for a single league appearance. Ninety is the ceiling in
#: normal time; the allowance above it absorbs stoppage-time bookkeeping that
#: some sources carry, and anything beyond it is a parse error rather than a
#: long afternoon.
MAX_APPEARANCE_MINUTES = 120.0

#: Expected total minutes in a complete match: twenty-two players for ninety
#: minutes. Dismissals push the realised total below this, which is why the
#: match tolerance is one-sided and generous.
EXPECTED_MATCH_MINUTES = float(STARTERS_PER_MATCH) * 90.0
MATCH_MINUTES_TOLERANCE = 0.15

#: A league-season enters the snapshot only if this share of its scheduled
#: matches survived the per-match checks. Below it the season is reported as
#: excluded rather than quietly thinned, because a gradient fitted on the
#: matches that happened to parse is a gradient fitted on a biased sample.
MIN_MATCH_COVERAGE = 0.90

#: Lineup roles are used for a league-season only if this share of its usable
#: matches carry a readable lineup box. A partial box is worse than none: it
#: would fit the within-starter gradient on whichever matches happened to be
#: annotated, and those are unlikely to be a random half.
MIN_ROLE_COVERAGE = 0.90


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def normalize_records(records: Sequence[Sequence[object]]) -> pd.DataFrame:
    """Turn extracted rows into a typed appearance frame.

    The extractor runs in a browser and can only be trusted to have read the
    page it was given, so the types are imposed here rather than assumed.
    """
    if not records:
        raise ValueError("no appearance records were supplied")
    frame = pd.DataFrame(list(records), columns=list(RAW_RECORD_FIELDS))

    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce", format="ISO8601")
    if frame[DATE_COL].isna().any():
        raise ValueError("every appearance needs a readable date")

    frame[MINUTES_COL] = pd.to_numeric(frame[MINUTES_COL], errors="coerce")
    if frame[MINUTES_COL].isna().any():
        raise ValueError("every appearance needs readable minutes")

    for column in (COUNTRY_COL, COMPETITION_COL, SEASON_COL, MATCH_COL, SOURCE_PLAYER_COL):
        frame[column] = frame[column].astype(str)

    unknown = sorted(set(frame[COMPETITION_COL]) - set(WOMENS_LEAGUES))
    if unknown:
        raise ValueError(f"records from unregistered competitions: {unknown}")

    allowed = {STARTER_ROLE, SUBSTITUTE_ROLE, UNKNOWN_ROLE}
    roles = set(frame[ROLE_COL])
    if not roles <= allowed:
        raise ValueError(f"unrecognised lineup roles: {sorted(roles - allowed)}")

    if frame.duplicated([MATCH_COL, SOURCE_PLAYER_COL]).any():
        raise ValueError("a player cannot appear twice in one match")

    return frame.sort_values([COMPETITION_COL, DATE_COL, MATCH_COL]).reset_index(drop=True)


def match_integrity(frame: pd.DataFrame) -> pd.DataFrame:
    """Score every match against what a complete match report must contain.

    Three ways an extraction fails quietly: the lineup box is missing, so the
    starters cannot be told from the substitutes; a team's table is absent, so
    half the appearances vanish; or a minutes cell is misread. Each shows up
    here as a match that is kept or dropped for a stated reason.
    """
    _require(frame, RAW_RECORD_FIELDS, "match integrity input")

    grouped = frame.groupby(MATCH_COL, sort=True)
    summary = grouped.agg(
        competition_id=(COMPETITION_COL, "first"),
        season=(SEASON_COL, "first"),
        date=(DATE_COL, "first"),
        n_appearances=(MINUTES_COL, "size"),
        total_minutes=(MINUTES_COL, "sum"),
        max_minutes=(MINUTES_COL, "max"),
        min_minutes=(MINUTES_COL, "min"),
    ).reset_index()
    summary["n_starters"] = (
        grouped[ROLE_COL].apply(lambda roles: int((roles == STARTER_ROLE).sum())).values
    )

    lower = EXPECTED_MATCH_MINUTES * (1.0 - MATCH_MINUTES_TOLERANCE)
    upper = EXPECTED_MATCH_MINUTES * (1.0 + MATCH_MINUTES_TOLERANCE)
    summary["minutes_ok"] = summary["total_minutes"].between(lower, upper)
    summary["bounds_ok"] = summary["max_minutes"].le(MAX_APPEARANCE_MINUTES) & summary[
        "min_minutes"
    ].gt(0.0)
    # Usability is about the appearance record, which is all the pooled
    # gradient needs. Whether the lineup box could be read is a separate
    # question, because it limits only the within-starter fit.
    summary["usable"] = summary["minutes_ok"] & summary["bounds_ok"]
    summary["roles_known"] = summary["n_starters"] == STARTERS_PER_MATCH

    reasons = []
    for row in summary.itertuples(index=False):
        if not row.minutes_ok:
            reasons.append(f"total minutes {row.total_minutes:.0f} outside match bounds")
        elif not row.bounds_ok:
            reasons.append("an appearance fell outside plausible minutes")
        else:
            reasons.append("")
    summary["exclusion_reason"] = reasons
    return summary


def completeness_by_league_season(
    integrity: pd.DataFrame,
    scheduled: Mapping[tuple[str, str], int],
) -> pd.DataFrame:
    """Report what share of each league-season's fixtures produced usable data.

    ``scheduled`` counts the fixtures the league's own calendar lists, so a
    season that is simply missing from the source is visible as missing rather
    than as a small season.
    """
    _require(integrity, ("competition_id", "season", "usable"), "completeness input")

    rows = []
    seen = set()
    for (competition, season), group in integrity.groupby(["competition_id", "season"], sort=True):
        seen.add((competition, season))
        fixtures = int(scheduled.get((competition, season), len(group)))
        usable = int(group["usable"].sum())
        with_roles = int((group["usable"] & group["roles_known"]).sum())
        rows.append(
            {
                "competition_id": competition,
                "league": WOMENS_LEAGUES[competition],
                "season": season,
                "scheduled_matches": fixtures,
                "parsed_matches": int(len(group)),
                "usable_matches": usable,
                "match_coverage": usable / fixtures if fixtures else 0.0,
                "matches_with_roles": with_roles,
                "role_coverage": with_roles / usable if usable else 0.0,
                "appearances": int(group.loc[group["usable"], "n_appearances"].sum()),
            }
        )
    for (competition, season), fixtures in sorted(scheduled.items()):
        if (competition, season) in seen:
            continue
        rows.append(
            {
                "competition_id": competition,
                "league": WOMENS_LEAGUES[competition],
                "season": season,
                "scheduled_matches": int(fixtures),
                "parsed_matches": 0,
                "usable_matches": 0,
                "match_coverage": 0.0,
                "matches_with_roles": 0,
                "role_coverage": 0.0,
                "appearances": 0,
            }
        )

    frame = pd.DataFrame(rows).sort_values(["competition_id", "season"]).reset_index(drop=True)
    frame["admitted"] = frame["match_coverage"] >= MIN_MATCH_COVERAGE
    frame["roles_admitted"] = frame["admitted"] & (frame["role_coverage"] >= MIN_ROLE_COVERAGE)
    return frame


def harmonize_roles(
    frame: pd.DataFrame,
    integrity: pd.DataFrame,
    completeness: pd.DataFrame,
) -> pd.DataFrame:
    """Mark lineup role unknown wherever it could not be read reliably.

    Applied per league-season rather than per match. A league whose lineup
    boxes are present for half its fixtures would otherwise contribute a
    within-starter gradient fitted on that half, and the half that gets
    annotated is not a random sample of matches.
    """
    _require(frame, (COMPETITION_COL, SEASON_COL, ROLE_COL), "role harmonisation input")
    _require(completeness, ("competition_id", "season", "roles_admitted"), "completeness frame")

    trusted = {
        (row.competition_id, row.season)
        for row in completeness.loc[completeness["roles_admitted"]].itertuples(index=False)
    }
    known = pd.Series(
        [
            (competition, season) in trusted
            for competition, season in zip(frame[COMPETITION_COL], frame[SEASON_COL])
        ],
        index=frame.index,
    )
    out = frame.copy()
    out.loc[~known, ROLE_COL] = UNKNOWN_ROLE
    return out


def apply_completeness_gate(
    frame: pd.DataFrame,
    integrity: pd.DataFrame,
    completeness: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only appearances from usable matches in admitted league-seasons."""
    _require(frame, (MATCH_COL, COMPETITION_COL, SEASON_COL), "gate input")

    usable_matches = set(integrity.loc[integrity["usable"], MATCH_COL])
    admitted = {
        (row.competition_id, row.season)
        for row in completeness.loc[completeness["admitted"]].itertuples(index=False)
    }
    in_admitted_season = pd.Series(
        [
            (competition, season) in admitted
            for competition, season in zip(frame[COMPETITION_COL], frame[SEASON_COL])
        ],
        index=frame.index,
    )
    keep = frame[MATCH_COL].isin(usable_matches) & in_admitted_season
    kept = frame.loc[keep].reset_index(drop=True)
    if kept.empty:
        raise ValueError("the completeness gate admitted no appearances")
    return kept


def build_player_surrogates(
    frame: pd.DataFrame,
    key_factory: Callable[[], str] | None = None,
) -> pd.DataFrame:
    """Draw one surrogate per source player identifier.

    Drawn, not derived. The source identifiers are public and few enough that
    any function of them can be inverted by trying every candidate, so a hash
    would publish the mapping it pretends to hide.
    """
    _require(frame, (SOURCE_PLAYER_COL,), "surrogate source")
    factory = key_factory or (lambda: secrets.token_hex(6).upper())

    sources = sorted(frame[SOURCE_PLAYER_COL].astype(str).unique())
    keys: dict[str, str] = {}
    for source in sources:
        key = f"W{factory()}"
        while key in set(keys.values()):  # pragma: no cover - collision is astronomically rare
            key = f"W{factory()}"
        keys[source] = key
    return pd.DataFrame(
        {SOURCE_PLAYER_COL: sources, PLAYER_ID_COL: [keys[s] for s in sources]}
    )


def extend_player_surrogates(
    existing: pd.DataFrame,
    frame: pd.DataFrame,
    key_factory: Callable[[], str] | None = None,
) -> pd.DataFrame:
    """Add surrogates for players the map has not seen, keeping the rest.

    Adding a league, a season or a second source must not renumber the players
    already deposited: a surrogate that moves is not a pseudonym, it is a new
    identity, and any table published against the old one silently stops
    matching. Existing rows are therefore returned untouched and only genuinely
    new source identifiers are drawn for.
    """
    _require(existing, (SOURCE_PLAYER_COL, PLAYER_ID_COL), "existing surrogate map")
    _require(frame, (SOURCE_PLAYER_COL,), "surrogate source")

    known = set(existing[SOURCE_PLAYER_COL].astype(str))
    fresh = frame[~frame[SOURCE_PLAYER_COL].astype(str).isin(known)]
    if fresh.empty:
        return existing.copy()

    drawn = build_player_surrogates(fresh, key_factory=key_factory)
    collisions = set(drawn[PLAYER_ID_COL]) & set(existing[PLAYER_ID_COL].astype(str))
    if collisions:  # pragma: no cover - astronomically unlikely with 12 hex digits
        raise ValueError(f"surrogate collision with the existing map: {sorted(collisions)}")
    return pd.concat([existing, drawn], ignore_index=True)


def to_gradient_schema(frame: pd.DataFrame, surrogates: pd.DataFrame) -> pd.DataFrame:
    """Project onto the five columns the gradient module reads.

    The gradient module was written against the men's snapshot but never reads
    anything source-specific, so meeting its schema is the whole of the work
    needed to reuse it unchanged.
    """
    _require(frame, RAW_RECORD_FIELDS, "gradient projection input")
    _require(surrogates, (SOURCE_PLAYER_COL, PLAYER_ID_COL), "surrogate map")

    merged = frame.merge(surrogates, on=SOURCE_PLAYER_COL, how="left")
    if merged[PLAYER_ID_COL].isna().any():
        raise ValueError("every appearance needs a surrogate player identifier")

    projected = merged[
        [PLAYER_ID_COL, DATE_COL, MINUTES_COL, COMPETITION_COL, ROLE_COL, SEASON_COL, MATCH_COL]
    ].copy()
    return projected.sort_values([COMPETITION_COL, DATE_COL, PLAYER_ID_COL]).reset_index(drop=True)


def snapshot_summary(projected: pd.DataFrame) -> pd.DataFrame:
    """One row per league: what the deposited snapshot actually contains."""
    _require(projected, (COMPETITION_COL, PLAYER_ID_COL, DATE_COL, MINUTES_COL), "summary input")

    rows = []
    for competition, group in projected.groupby(COMPETITION_COL, sort=True):
        rows.append(
            {
                "competition_id": competition,
                "league": WOMENS_LEAGUES[competition],
                "seasons": group[SEASON_COL].nunique(),
                "matches": group[MATCH_COL].nunique(),
                "appearances": int(len(group)),
                "players": group[PLAYER_ID_COL].nunique(),
                "first_date": group[DATE_COL].min().date().isoformat(),
                "last_date": group[DATE_COL].max().date().isoformat(),
                "median_minutes": float(group[MINUTES_COL].median()),
                "roles_known": bool((group[ROLE_COL] != UNKNOWN_ROLE).all()),
                "starter_share": float((group[ROLE_COL] == STARTER_ROLE).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:  # pragma: no cover - orchestration
    """Assemble the deposited women's appearance snapshot from extracted rows."""
    import json

    root = Path(__file__).resolve().parents[1]
    staging = root / "data" / "raw" / "fbref_womens"
    private = root / "data" / "private"
    processed = root / "data" / "processed"
    results = processed / "results"
    for directory in (private, processed, results):
        directory.mkdir(parents=True, exist_ok=True)

    print("1. Reading extracted appearance records ...")
    records = json.loads((staging / "appearance_records.json").read_text(encoding="utf-8"))
    scheduled_raw = json.loads((staging / "scheduled_matches.json").read_text(encoding="utf-8"))
    scheduled = {(k.split("|")[0], k.split("|")[1]): v for k, v in scheduled_raw.items()}
    frame = normalize_records(records)
    print(f"   {len(frame):,} appearances in {frame[MATCH_COL].nunique():,} matches")

    print("2. Checking every match against a complete match report ...")
    integrity = match_integrity(frame)
    print(f"   {int(integrity['usable'].sum()):,} of {len(integrity):,} matches usable")

    print("3. Gating league-seasons on fixture coverage ...")
    completeness = completeness_by_league_season(integrity, scheduled)
    admitted = completeness.loc[completeness["admitted"]]
    print(f"   {len(admitted)} of {len(completeness)} league-seasons admitted")

    gated = apply_completeness_gate(frame, integrity, completeness)
    roleless = completeness.loc[completeness["admitted"] & ~completeness["roles_admitted"]]
    if len(roleless):
        print(f"   {len(roleless)} admitted league-seasons carry no readable lineup box")
    gated = harmonize_roles(gated, integrity, completeness)

    print("4. Drawing surrogate player identifiers ...")
    surrogate_path = private / "womens_player_surrogates.csv"
    if surrogate_path.exists():
        surrogates = extend_player_surrogates(
            pd.read_csv(surrogate_path, dtype=str), gated
        )
    else:
        surrogates = build_player_surrogates(gated)
    surrogates.to_csv(surrogate_path, index=False)
    projected = to_gradient_schema(gated, surrogates)

    print("5. Writing the snapshot and its summary ...")
    projected.to_csv(
        processed / "womens_appearances.csv.gz", index=False, compression="gzip"
    )
    integrity.to_csv(results / "jsams_womens_match_integrity.csv", index=False)
    completeness.to_csv(results / "jsams_womens_league_season_completeness.csv", index=False)
    snapshot_summary(projected).to_csv(
        results / "jsams_womens_snapshot_summary.csv", index=False
    )
    print(f"   {len(projected):,} appearances written")


if __name__ == "__main__":  # pragma: no cover
    main()
