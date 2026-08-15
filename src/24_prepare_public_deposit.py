#!/usr/bin/env python
"""
Prepare neutral-terminology files for public deposition.

The internal pipeline historically used vivid labels such as ``fragile`` and
``tough``. The manuscript now uses neutral prior-injury-history strata, and the
public archive should do the same. This script copies selected generated CSVs
and figures into ``public_deposit/`` while sanitising:

- CSV column names;
- string values inside CSVs; and
- exported filenames.

It also removes identity, which is the part the name of this module used to
promise and not deliver: every exported table has person-name columns dropped
and person identifiers replaced by surrogates drawn once and reused, so the
deposit joins across tables without naming anyone.

The exported set covers every artifact the manuscript or supplement cites:
model and audit results (``data/processed/results``), the v4 acquisition and
quality tables (``data/processed/public_data_v4``), the hand-adjudicated
outcome audit (``data/manual``), and the manuscript figures. ``data/manual``
matters most because no script can regenerate it.

The source data are not modified.
"""

from pathlib import Path
import secrets
import shutil
from typing import Callable, Dict, Iterable, List, Mapping

import pandas as pd


# Columns that name a person or carry a provider's person identifier. The name
# is dropped outright; the identifier is replaced by a surrogate, because rows
# still have to join across the deposited tables.
#
# This module was called a sanitiser long before it did any of this: it renamed
# history strata and nothing else, so a reader auditing the anonymity claim
# would have found a step named for the job and not doing it.
PERSON_NAME_COLUMNS = ("player_name",)
PERSON_ID_COLUMNS = ("tm_player_id", "player_id", "fbref_player_id")

# Drawn once per player and reused, so the deposit is stable across rebuilds
# and nothing in it supports re-identification on its own. Lives outside every
# exported subtree.
PLAYER_MAP_PATH = Path("data") / "private" / "deposit_player_map.csv"
PLAYER_MAP_COLUMNS = ("source_id", "player_key")


TEXT_REPLACEMENTS: Mapping[str, str] = {
    "fragility": "prior_history",
    "Fragility": "Prior history",
    "fragile": "higher_history",
    "Fragile": "Higher history",
    "regular": "intermediate_history",
    "Regular": "Intermediate history",
    "tough": "lower_history",
    "Tough": "Lower history",
}

CSV_SOURCE_SUBDIRS = [
    Path("data") / "processed" / "results",
    # The v4 acquisition, reconciliation, status, and quality-gate tables. The
    # supplement cites selection_membership_resolution_audit.csv from here.
    Path("data") / "processed" / "public_data_v4",
    # Hand-adjudicated audit decisions and source URLs. No script regenerates
    # this input, so the deposit must carry it for the outcome audit to be
    # checkable.
    Path("data") / "manual",
]
FIGURE_SOURCE_SUBDIRS = [
    Path("manuscript") / "figures",
]


def sanitize_text(value: object, replacements: Mapping[str, str] = TEXT_REPLACEMENTS) -> object:
    """Return ``value`` with legacy history-stratum wording replaced."""
    if not isinstance(value, str):
        return value
    out = value
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def sanitize_path_name(path: Path) -> Path:
    """Return a path with every component passed through ``sanitize_text``."""
    return Path(*[str(sanitize_text(part)) for part in path.parts])


def sanitize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sanitise legacy terminology in dataframe column names and string values."""
    out = frame.copy()
    out = out.rename(columns={column: sanitize_text(str(column)) for column in out.columns})
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    for column in object_cols:
        out[column] = out[column].map(sanitize_text)
    return out


def iter_existing_files(root: Path, subdirs: Iterable[Path], suffixes: tuple[str, ...]) -> Iterable[Path]:
    """Yield files under existing source subdirectories."""
    for subdir in subdirs:
        source = root / subdir
        if not source.exists():
            continue
        for path in source.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                yield path


def load_player_surrogates(path: Path) -> Dict[str, str]:
    """Read the deposit-wide surrogate map, or start an empty one."""
    if not Path(path).exists():
        return {}
    frame = pd.read_csv(path, dtype=str)
    missing = sorted(set(PLAYER_MAP_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"player map missing columns: {missing}")
    if frame["source_id"].duplicated().any():
        raise ValueError("player map has duplicate source identifiers")
    return dict(zip(frame["source_id"], frame["player_key"]))


def save_player_surrogates(path: Path, surrogates: Mapping[str, str]) -> None:
    """Persist the map so a rebuild does not renumber the deposit."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        sorted(surrogates.items()), columns=list(PLAYER_MAP_COLUMNS)
    )
    frame.to_csv(path, index=False)


def deidentify_frame(
    frame: pd.DataFrame,
    surrogates: Dict[str, str],
    key_factory: Callable[[], str] | None = None,
) -> pd.DataFrame:
    """Drop person names and replace person identifiers with surrogates.

    New identifiers are minted on sight and remembered, so a table added later
    joins to the ones already deposited.
    """
    factory = key_factory or (lambda: f"P{secrets.token_hex(6).upper()}")
    out = frame.drop(
        columns=[c for c in PERSON_NAME_COLUMNS if c in frame.columns]
    )
    for column in PERSON_ID_COLUMNS:
        if column not in out.columns:
            continue
        values = out[column].astype("string")
        for value in values.dropna().unique():
            if value not in surrogates:
                surrogates[value] = factory()
        out[column] = values.map(surrogates).astype(object)
    return out


def write_sanitized_csv(
    source: Path,
    target: Path,
    surrogates: Dict[str, str] | None = None,
) -> None:
    """Read a CSV, remove identity, sanitise labels, and write it to ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, low_memory=False)
    if surrogates is not None:
        frame = deidentify_frame(frame, surrogates)
    sanitize_frame(frame).to_csv(target, index=False)


def copy_sanitized_binary(source: Path, target: Path) -> None:
    """Copy a generated binary artifact to a sanitised public filename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare_public_deposit(root: Path) -> pd.DataFrame:
    """Create ``public_deposit`` with neutral-terminology CSVs and figures."""
    root = Path(root)
    public_root = root / "public_deposit"
    rows: List[dict[str, str]] = []
    if public_root.exists():
        shutil.rmtree(public_root)

    map_path = root / PLAYER_MAP_PATH
    surrogates = load_player_surrogates(map_path)

    for source in iter_existing_files(root, CSV_SOURCE_SUBDIRS, (".csv",)):
        relative = source.relative_to(root)
        target = public_root / sanitize_path_name(relative)
        write_sanitized_csv(source, target, surrogates)
        rows.append(
            {
                "artifact_type": "csv",
                "source_path": str(relative).replace("\\", "/"),
                "public_path": str(target.relative_to(root)).replace("\\", "/"),
            }
        )

    for source in iter_existing_files(root, FIGURE_SOURCE_SUBDIRS, (".png", ".jpg", ".jpeg", ".pdf")):
        relative = source.relative_to(root)
        target = public_root / sanitize_path_name(relative)
        copy_sanitized_binary(source, target)
        rows.append(
            {
                "artifact_type": "figure",
                "source_path": str(relative).replace("\\", "/"),
                "public_path": str(target.relative_to(root)).replace("\\", "/"),
            }
        )

    save_player_surrogates(map_path, surrogates)

    manifest = pd.DataFrame(rows, columns=["artifact_type", "source_path", "public_path"])
    public_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(public_root / "sanitization_manifest.csv", index=False)
    return manifest


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    manifest = prepare_public_deposit(root)
    print(f"Prepared {len(manifest)} public-deposit artifacts under {root / 'public_deposit'}")


if __name__ == "__main__":  # pragma: no cover
    main()
