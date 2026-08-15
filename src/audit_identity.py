"""Turn audited appearances into deposited records that name nobody.

The audit sampled real appearances, so every hand-built row began life
carrying a player's name, the provider's player identifier, their club and the
exact match date --- and, in the same-day audit, a body-part injury label
beside all four. Those rows are evidence the paper relies on and they have to
be deposited, but deposited in that form they publish special-category data
about identifiable people, which is the one thing the manuscript says the
deposit does not do.

This module is the boundary where identity is removed. Three rules shape it:

*Substitution happens after sampling, never before.* The audit sample is drawn
by hashing ``audit_id``, which is ``{tm_player_id}_{date}``. Feeding surrogates
into that draw would select a different thirty appearances and orphan every
hand-written adjudication. So the queues are built, joined and screened on real
identifiers in memory, and only the frames on their way to disk pass through
here.

*The surrogates are drawn, not derived.* A hash of a public identifier is
reversible by anyone willing to hash the same public identifiers, and there are
only a few thousand candidates. These keys come from a random source once and
are recorded in a map the deposit does not carry, so nothing in the published
data supports re-identification on its own.

*What survives must be worth its risk.* The injury label stays, because the
audit exists to check whether it is consistent, and a body-part label attached
to an opaque key describes no one. The exact date goes, coarsened to a season:
a club plus a match-day plus "hamstring" is a singleton, and the audit's
reasoning never needed the day.
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd


AUDIT_ID = "audit_id"
PLAYER_ID = "tm_player_id"
RECORD_KEY = "record_key"
PLAYER_KEY = "player_key"
SEASON = "season"
DATE = "date"
SOURCE_URL = "independent_source_url"
SOURCE_FOUND = "independent_source_found"

# Columns that name a person, a club or a provider record. None of them feeds a
# computed value: the absence screen joins on the appearance snapshot's own
# identifiers, and the club string is only ever displayed.
DIRECT_IDENTIFIER_COLUMNS = (PLAYER_ID, "player_name", "club_name")

# Free text a reviewer wrote by hand, which can name the player it describes.
FREE_TEXT_COLUMNS = ("review_note",)

WITHHELD = "[name withheld]"

IDENTITY_MAP_COLUMNS = (AUDIT_ID, PLAYER_ID, RECORD_KEY, PLAYER_KEY)


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def season_label(dates: pd.Series) -> pd.Series:
    """Return the football season a date falls in, as ``2018-19``.

    Seasons turn over in July, so a January match belongs to the season that
    started the previous year. Coarsening to this grain is what stops a match
    date from being a re-identification key.
    """
    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.isna().any():
        raise ValueError("audit dates must all parse before they can be coarsened")
    start = parsed.dt.year.where(parsed.dt.month >= 7, parsed.dt.year - 1)
    return start.astype(str) + "-" + (start + 1).astype(str).str[2:]


def _new_key(prefix: str, factory: Callable[[], str]) -> str:
    return f"{prefix}{factory()}"


def build_identity_map(
    frames: Sequence[pd.DataFrame],
    key_factory: Callable[[], str] | None = None,
) -> pd.DataFrame:
    """Draw one surrogate per audited record and one per audited player.

    Both grains are needed. The record key joins a hand-written verdict to the
    row it judges; the player key keeps the sample's clustering visible, since
    thirty sampled rows cover twenty-seven players and a reader who cannot see
    that cannot judge the audit's independence assumption.
    """
    if not frames:
        raise ValueError("an identity map needs at least one audited frame")
    factory = key_factory or (lambda: secrets.token_hex(6).upper())

    rows = []
    for frame in frames:
        _require(frame, (AUDIT_ID, PLAYER_ID), "identity map source")
        rows.append(frame[[AUDIT_ID, PLAYER_ID]])
    # Duplicates across the two audit files are expected and harmless; the same
    # record listed against two different players is not, and would silently
    # give one appearance two surrogates.
    pairs = pd.concat(rows, ignore_index=True).drop_duplicates()
    if pairs[AUDIT_ID].duplicated().any():
        raise ValueError("one audit_id cannot belong to two players")

    player_keys = {
        player: _new_key("P", factory)
        for player in sorted(pairs[PLAYER_ID].astype(str).unique())
    }
    mapped = pairs.assign(
        **{
            RECORD_KEY: [_new_key("A", factory) for _ in range(len(pairs))],
            PLAYER_KEY: pairs[PLAYER_ID].astype(str).map(player_keys),
        }
    )
    return mapped[list(IDENTITY_MAP_COLUMNS)].sort_values(AUDIT_ID).reset_index(
        drop=True
    )


def load_identity_map(path: Path) -> pd.DataFrame:
    """Read the surrogate map, which lives outside every deposited subtree."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"identity map not found at {path}. The audit stage cannot write "
            "de-identified outputs without it, and must not fall back to "
            "writing identified ones."
        )
    frame = pd.read_csv(path, dtype=str)
    _require(frame, IDENTITY_MAP_COLUMNS, "identity map")
    if frame[AUDIT_ID].duplicated().any():
        raise ValueError("identity map has duplicate audit_id values")
    if frame[RECORD_KEY].duplicated().any():
        raise ValueError("identity map has duplicate record keys")
    return frame


def scrub_names(text: str, surnames: Sequence[str]) -> str:
    """Remove any audited player's surname from a hand-written note."""
    cleaned = text
    for surname in surnames:
        if len(surname) < 3:
            continue
        cleaned = re.sub(
            rf"\b{re.escape(surname)}\b", WITHHELD, cleaned, flags=re.IGNORECASE
        )
    return cleaned


def audited_surnames(frames: Sequence[pd.DataFrame]) -> list[str]:
    """Every name part worth scrubbing from free text, longest first.

    Both parts of a name are taken, not just the last: a note reading "Ardal
    was already out" identifies as surely as one naming Kavanagh.
    """
    parts: set[str] = set()
    for frame in frames:
        if "player_name" not in frame.columns:
            continue
        for name in frame["player_name"].dropna().astype(str):
            for part in re.split(r"[\s\-']+", name.strip()):
                if len(part) >= 3:
                    parts.add(part)
    # Longest first, so a full name is scrubbed before its parts; alphabetical
    # within a length, because set iteration order is not stable between runs
    # and a pipeline must not depend on it.
    return sorted(parts, key=lambda part: (-len(part), part))


def deidentify_audit_frame(
    frame: pd.DataFrame,
    identity_map: pd.DataFrame,
    surnames: Sequence[str] = (),
) -> pd.DataFrame:
    """Return ``frame`` with every direct identifier replaced by a surrogate.

    Fails rather than redacts when a row has no surrogate: a silently dropped
    row would shrink an audit whose whole value is that it reports what it
    could not resolve.
    """
    _require(frame, (AUDIT_ID,), "audit frame")
    _require(identity_map, IDENTITY_MAP_COLUMNS, "identity map")

    keys = identity_map.set_index(AUDIT_ID)
    ids = frame[AUDIT_ID].astype(str)
    unmapped = sorted(set(ids) - set(keys.index))
    if unmapped:
        raise KeyError(f"no surrogate for audit records: {unmapped[:5]}")

    out = frame.copy()
    out[RECORD_KEY] = ids.map(keys[RECORD_KEY]).to_numpy()
    out[PLAYER_KEY] = ids.map(keys[PLAYER_KEY]).to_numpy()

    if DATE in out.columns:
        out[SEASON] = season_label(out[DATE]).to_numpy()

    if SOURCE_URL in out.columns:
        # The URL slug is itself an identifier: most of these end in the
        # player's surname, and one in a graded diagnosis. What the audit
        # actually rests on is whether a qualifying source was found at all,
        # and that survives as a flag beside the source type.
        found = out[SOURCE_URL].fillna("").astype(str).str.strip().ne("")
        out[SOURCE_FOUND] = found.to_numpy()

    for column in FREE_TEXT_COLUMNS:
        if column in out.columns:
            out[column] = (
                out[column]
                .fillna("")
                .astype(str)
                .map(lambda value: scrub_names(value, surnames))
            )

    dropped = [AUDIT_ID, DATE, SOURCE_URL, *DIRECT_IDENTIFIER_COLUMNS]
    out = out.drop(columns=[c for c in dropped if c in out.columns])

    lead = [c for c in (RECORD_KEY, PLAYER_KEY, SEASON) if c in out.columns]
    rest = [c for c in out.columns if c not in lead]
    return out[[*lead, *rest]].reset_index(drop=True)
