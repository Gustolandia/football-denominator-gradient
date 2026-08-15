"""
Clean Transfermarkt injury spells and restrict to the analysis season window.

Only reported spell starts on or after ``ANALYSIS_START_DATE`` are retained.

Inputs:
  data/raw/epl_matches.csv
  data/raw/tm_injuries_raw.csv

Output:
  data/processed/tm_injuries_clean.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

import pandas as pd
from config import ANALYSIS_START_DATE, DATA_RAW, DATA_PROCESSED


def main() -> None:  # pragma: no cover
    # Use the match file just to determine the overall date window
    schedule = pd.read_csv(DATA_RAW / "epl_matches.csv")
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")

    analysis_start = pd.Timestamp(ANALYSIS_START_DATE)
    min_date = max(schedule["date"].min(), analysis_start)
    max_date = schedule["date"].max()

    inj = pd.read_csv(DATA_RAW / "tm_injuries_raw.csv")

    # Normalise most common column names from transfermarkt-wrapper / tmkt.
    # We handle both:
    #   - 'startDate'/'endDate'/'injury'  (older/other wrappers)
    #   - 'start'/'end'/'name'           (tmkt injury response shape)
    rename_map = {
        "startDate": "start_date",
        "start": "start_date",
        "endDate": "end_date",
        "end": "end_date",
        "injury": "injury_desc",
        "name": "injury_desc",
    }
    inj.rename(
        columns={k: v for k, v in rename_map.items() if k in inj.columns},
        inplace=True,
    )

    # Parse dates
    for col in ["start_date", "end_date"]:
        if col in inj.columns:
            inj[col] = pd.to_datetime(inj[col], errors="coerce")

    if "start_date" not in inj.columns:
        raise SystemExit("No 'start_date' column found in tm_injuries_raw.csv")

    # Keep only spells whose reported start date is inside the analysis window.
    # Pre-2017 starts are excluded rather than used as warm-up availability data
    # because the analysis intentionally drops the lower-confidence early era.
    mask = (
        inj["start_date"].notna()
        & (inj["start_date"] >= min_date)
        & (inj["start_date"] <= max_date)
    )

    inj_clean = inj[mask].copy()
    inj_clean.reset_index(drop=True, inplace=True)
    inj_clean["injury_spell_id"] = inj_clean.index

    out_path = DATA_PROCESSED / "tm_injuries_clean.csv"
    inj_clean.to_csv(out_path, index=False)
    print(f"Clean injury spells: {len(inj_clean)} -> {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
