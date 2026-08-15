#!/usr/bin/env python
"""Rewrite the hand-built audit evidence so the deposited copy names nobody.

The two audit files under ``data/manual`` are the only hand-built evidence in
the repository and both deposit builders carry them verbatim. As written by the
reviewer they held a player's name, the provider's identifier, their club, the
exact match date, and --- in the same-day file --- a body-part injury label.

This script separates the two things that were previously one file:

``data/private/``   the reviewer's own record, identified, so the authors can
                    answer an editor's query and re-check any verdict. Tracked
                    in the working repository and carried by no deposit.
``data/manual/``    the deposited record, keyed by surrogates, which is what
                    every reader sees.

Run it after any change to the reviewer's record:

    python src/38_deidentify_audit_evidence.py

It is idempotent. The surrogate map is drawn once and then reused, so keys stay
stable across runs and a re-run never silently renumbers the deposited
evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_identity import (  # noqa: E402
    IDENTITY_MAP_COLUMNS,
    audited_surnames,
    build_identity_map,
    deidentify_audit_frame,
    load_identity_map,
)

AUDIT_FILES = (
    "independent_same_day_event_audit.csv",
    "independent_non_event_audit.csv",
)
IDENTITY_MAP_NAME = "audit_identity_map.csv"


def main() -> None:  # pragma: no cover - filesystem orchestration
    root = Path(__file__).resolve().parents[1]
    private = root / "data" / "private"
    manual = root / "data" / "manual"
    private.mkdir(parents=True, exist_ok=True)

    # First run: the reviewer's record is still the one under data/manual, so
    # move it across before anything overwrites it.
    for name in AUDIT_FILES:
        source, destination = manual / name, private / name
        if not destination.exists():
            if not source.exists():
                raise SystemExit(f"no audit evidence to de-identify: {source}")
            destination.write_bytes(source.read_bytes())
            print(f"[OK] preserved identified original -> {destination}")

    identified = {
        name: pd.read_csv(private / name, dtype=str) for name in AUDIT_FILES
    }

    map_path = private / IDENTITY_MAP_NAME
    if map_path.exists():
        identity_map = load_identity_map(map_path)
        print(f"[OK] reusing {len(identity_map)} surrogates from {map_path}")
    else:
        identity_map = build_identity_map(list(identified.values()))
        identity_map.to_csv(map_path, index=False)
        print(f"[OK] drew {len(identity_map)} surrogates -> {map_path}")

    known = set(identity_map[IDENTITY_MAP_COLUMNS[0]])
    for name, frame in identified.items():
        missing = sorted(set(frame["audit_id"]) - known)
        if missing:
            raise SystemExit(
                f"{name} has {len(missing)} records with no surrogate. Delete "
                f"{map_path} to redraw the whole map, which renumbers every "
                "deposited record, or add the new records to it by hand."
            )

    # The independence rule is a statement about the URLs, so it has to be
    # checked here, where they still exist. Downstream only a flag and a type
    # survive, and no later stage could re-derive this.
    for name, frame in identified.items():
        if "independent_source_url" not in frame.columns:
            continue
        urls = frame["independent_source_url"].fillna("").astype(str)
        offending = urls[urls.str.contains("transfermarkt", case=False, regex=False)]
        if not offending.empty:
            raise SystemExit(
                f"{name}: {len(offending)} source URLs are not independent of "
                "Transfermarkt, which the audit protocol forbids"
            )
        print(f"[OK] {name}: {int(urls.str.strip().ne('').sum())}/{len(frame)} "
              "records cite an independent source, none from Transfermarkt")

    surnames = audited_surnames(list(identified.values()))
    for name, frame in identified.items():
        public = deidentify_audit_frame(frame, identity_map, surnames)
        public.to_csv(manual / name, index=False)
        print(f"[OK] wrote de-identified evidence -> {manual / name}")

    print(
        "\nThe deposited audit files now carry surrogate keys only. The "
        f"identified originals and the map stay in {private}, which no "
        "deposit builder reads."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
