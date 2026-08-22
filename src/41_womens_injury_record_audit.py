"""Measure how the public women's injury record differs from the men's.

The manuscript restricts its outcome analysis to men's football. Saying so is
not enough: a reader is entitled to ask why, and "the data are worse" is an
assertion until someone measures it. This module measures it.

The design is a matched audit. The same scraper logic, the same squad-level
sampling frame and the same parser are applied to the public injury histories
of women's and men's clubs, so that any difference between the two is a
difference in the record rather than in how the record was read. Three things
are then reported, because three different failures would each be fatal to
using the women's record as an outcome:

*Presence* --- what share of sampled players have any recorded spell at all.
*Memory* --- how per-capita recording decays going back in time. A record that
only exists for recent seasons cannot support a study window that does not.
*Severity mix* --- what share of recorded spells are catastrophic injuries.
A record in which ruptured cruciates are among the most common entries is
capturing the injuries that make the news, not the injuries that happen.

Nothing deposited from here names anybody. The sampled rows carry player names
and diagnoses, which is special-category personal data about identifiable
athletes; they stay in ``data/private`` and only aggregates leave. That is the
same boundary ``src/audit_identity`` draws for the outcome audit, and it is
drawn here by the shape of the outputs rather than by remembering to be
careful.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

LEAGUE_COL = "league"
CLUB_COL = "club"
PLAYER_COL = "player"
SPELLS_COL = "n_spells"
SEASONS_COL = "seasons"
TYPES_COL = "types"

AUDIT_COLUMNS = (LEAGUE_COL, CLUB_COL, PLAYER_COL, SPELLS_COL, SEASONS_COL, TYPES_COL)

#: Columns that carry a person. None of these may appear in a deposited table.
PRIVATE_COLUMNS = (PLAYER_COL, CLUB_COL, SEASONS_COL, TYPES_COL)

#: A profile page that carries no injury-history section at all, as opposed to
#: one that carries an empty section. The scraper records the former as -1 so
#: the two can be told apart; both mean "no recorded spell".
NO_SECTION = -1

#: Patterns for season-ending injuries. Their share of all recorded spells is
#: the severity signal: in a complete record they are rare.
#:
#: Patterns rather than exact labels because the two sites use different words
#: for the same injury --- one records a cruciate ligament "rupture", the other
#: a "tear" --- and matching on exact strings silently scores one population at
#: zero, which reads as a finding when it is a vocabulary difference.
CATASTROPHIC_PATTERNS = (r"cruciate", r"achilles.*(?:rupture|tear)")


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def normalize_audit_records(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Type the sampled rows and mark which players have any recorded spell."""
    if not records:
        raise ValueError("the injury audit needs at least one sampled player")
    frame = pd.DataFrame(list(records))
    _require(frame, AUDIT_COLUMNS, "injury audit input")

    frame[SPELLS_COL] = pd.to_numeric(frame[SPELLS_COL], errors="coerce")
    if frame[SPELLS_COL].isna().any():
        raise ValueError("every sampled player needs a readable spell count")
    frame[SPELLS_COL] = frame[SPELLS_COL].astype(int)

    for column in (LEAGUE_COL, CLUB_COL, PLAYER_COL):
        frame[column] = frame[column].astype(str)
    for column in (SEASONS_COL, TYPES_COL):
        frame[column] = frame[column].fillna("").astype(str)

    frame["has_history"] = frame[SPELLS_COL] > 0
    frame["section_absent"] = frame[SPELLS_COL] == NO_SECTION
    frame["recorded_spells"] = frame[SPELLS_COL].clip(lower=0)
    return frame


def coverage_by_league(frame: pd.DataFrame) -> pd.DataFrame:
    """Presence of any recorded injury history, by league."""
    _require(frame, (LEAGUE_COL, "has_history", "recorded_spells"), "coverage input")

    rows = []
    for league, group in frame.groupby(LEAGUE_COL, sort=True):
        sampled = int(len(group))
        with_history = int(group["has_history"].sum())
        rows.append(
            {
                LEAGUE_COL: league,
                "players_sampled": sampled,
                "clubs_sampled": int(group[CLUB_COL].nunique()) if CLUB_COL in group else 0,
                "players_with_history": with_history,
                "share_with_history": with_history / sampled,
                "recorded_spells": int(group["recorded_spells"].sum()),
                "spells_per_player": float(group["recorded_spells"].sum() / sampled),
            }
        )
    return pd.DataFrame(rows)


def season_recording_profile(frame: pd.DataFrame, population: str) -> pd.DataFrame:
    """Count, per season, the sampled players carrying a spell in that season.

    Per capita rather than raw, because the two populations are sampled at
    different sizes. The point of the profile is its shape going backwards: a
    record that is genuinely complete decays gently, because the same players
    were being watched then too.
    """
    _require(frame, (SEASONS_COL,), "season profile input")
    sampled = int(len(frame))
    counts: dict[str, int] = {}
    for seasons in frame[SEASONS_COL]:
        for season in str(seasons).split(";"):
            season = season.strip()
            if season:
                counts[season] = counts.get(season, 0) + 1

    rows = [
        {
            "population": population,
            "season": season,
            "players_with_spell": count,
            "players_sampled": sampled,
            "per_capita": count / sampled,
        }
        for season, count in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def recording_decay(profile: pd.DataFrame, reference_season: str) -> pd.DataFrame:
    """Index each season's per-capita recording to a reference season.

    A value near one means the record reached that far back as densely as it
    does now. A value near zero means the record does not really exist there,
    whatever the fixture list says.
    """
    _require(profile, ("population", "season", "per_capita"), "decay input")

    rows = []
    for population, group in profile.groupby("population", sort=True):
        indexed = group.set_index("season")["per_capita"]
        if reference_season not in indexed.index:
            raise ValueError(
                f"reference season {reference_season!r} is absent for {population!r}"
            )
        base = float(indexed.loc[reference_season])
        for season, value in indexed.items():
            rows.append(
                {
                    "population": population,
                    "season": season,
                    "per_capita": float(value),
                    "relative_to_reference": float(value) / base if base > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def severity_mix(frame: pd.DataFrame, population: str) -> pd.DataFrame:
    """Share of recorded spells that are season-ending injuries."""
    _require(frame, (TYPES_COL,), "severity input")

    counts: dict[str, int] = {}
    for types in frame[TYPES_COL]:
        for entry in str(types).split(";"):
            entry = entry.strip()
            if entry:
                counts[entry.lower()] = counts.get(entry.lower(), 0) + 1

    total = sum(counts.values())
    catastrophic = sum(
        count
        for name, count in counts.items()
        if any(re.search(pattern, name) for pattern in CATASTROPHIC_PATTERNS)
    )
    unknown = sum(count for name, count in counts.items() if name.startswith("unknown"))
    return pd.DataFrame(
        [
            {
                "population": population,
                "recorded_type_entries": int(total),
                "distinct_types": int(len(counts)),
                "catastrophic_entries": int(catastrophic),
                "catastrophic_share": (catastrophic / total) if total else np.nan,
                "unknown_entries": int(unknown),
                "unknown_share": (unknown / total) if total else np.nan,
            }
        ]
    )


def audit_contrast(
    womens: pd.DataFrame,
    mens: pd.DataFrame,
    reference_season: str,
    early_season: str,
) -> pd.DataFrame:
    """One row: the numbers the manuscript's limitation paragraph cites."""
    for frame, label in ((womens, "women's audit"), (mens, "men's audit")):
        _require(frame, ("has_history", "recorded_spells", SEASONS_COL), label)

    profile = pd.concat(
        [
            season_recording_profile(womens, "women"),
            season_recording_profile(mens, "men"),
        ],
        ignore_index=True,
    )
    decay = recording_decay(profile, reference_season).set_index(["population", "season"])

    rows = []
    for label, frame in (("women", womens), ("men", mens)):
        early = decay.loc[(label, early_season), "relative_to_reference"] if (
            label, early_season
        ) in decay.index else np.nan
        rows.append(
            {
                "population": label,
                "players_sampled": int(len(frame)),
                "share_with_history": float(frame["has_history"].mean()),
                "spells_per_player": float(frame["recorded_spells"].mean()),
                "reference_season": reference_season,
                "early_season": early_season,
                "early_recording_relative_to_reference": float(early),
            }
        )
    return pd.DataFrame(rows)


def assert_no_personal_columns(frame: pd.DataFrame, label: str) -> None:
    """Refuse to deposit a table that carries a person.

    Checked by shape rather than by filename, so a new output cannot be added
    later without either passing this or failing loudly.
    """
    offending = [column for column in PRIVATE_COLUMNS if column in frame.columns]
    if offending:
        raise ValueError(f"{label} would publish personal columns: {offending}")


def main() -> None:  # pragma: no cover - orchestration
    """Turn the private sampled rows into deposited aggregates."""
    root = Path(__file__).resolve().parents[1]
    private = root / "data" / "private"
    results = root / "data" / "processed" / "results"
    results.mkdir(parents=True, exist_ok=True)

    print("1. Reading the privately held sampled rows ...")
    womens_raw = pd.read_csv(private / "womens_injury_audit_sample.csv").to_dict("records")
    mens_raw = pd.read_csv(private / "mens_injury_audit_sample.csv").to_dict("records")
    womens = normalize_audit_records(womens_raw)
    mens = normalize_audit_records(mens_raw)
    print(f"   {len(womens):,} women's and {len(mens):,} men's players sampled")

    print("2. Measuring presence, memory and severity mix ...")
    profile = pd.concat(
        [
            season_recording_profile(womens, "women"),
            season_recording_profile(mens, "men"),
        ],
        ignore_index=True,
    )
    outputs = {
        "womens_injury_record_coverage": coverage_by_league(womens),
        "injury_record_season_profile": profile,
        "injury_record_recording_decay": recording_decay(profile, "24/25"),
        "injury_record_severity_mix": pd.concat(
            [severity_mix(womens, "women"), severity_mix(mens, "men")], ignore_index=True
        ),
        "injury_record_audit_contrast": audit_contrast(womens, mens, "24/25", "19/20"),
    }

    print("3. Checking no deposited table names anybody ...")
    for name, table in outputs.items():
        assert_no_personal_columns(table, name)
        table.to_csv(results / f"jsams_{name}.csv", index=False)
    print(f"Wrote {len(outputs)} audit tables to {results}")


if __name__ == "__main__":  # pragma: no cover
    main()
