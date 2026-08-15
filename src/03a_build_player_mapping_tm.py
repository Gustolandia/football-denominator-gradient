"""Build ``player_mapping_tm.csv`` from local Transfermarkt appearance data.

Inputs:
    data/raw/epl_player_appearances.csv
    data/processed/player_index_fbref.csv

Output:
    data/processed/player_mapping_tm.csv

The local appearances file supplies Transfermarkt ``player_id`` values. The
upstream player index now contains one row per stable Transfermarkt player, and
the legacy ``fbref_player_id`` column equals ``tm_player_id`` for downstream
schema compatibility.
"""

import sys
from pathlib import Path

# --- Make project root importable when running as "python src/03a_build_player_mapping_tm.py"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

import pandas as pd
from config import ANALYSIS_START_SEASON, DATA_RAW, DATA_PROCESSED


def main() -> None:  # pragma: no cover
    # Load player index (built in step 2)
    idx_path = DATA_PROCESSED / "player_index_fbref.csv"
    players = pd.read_csv(idx_path)

    # Load appearances (built in step 1 from transfermarkt games/appearances)
    apps_path = DATA_RAW / "epl_player_appearances.csv"
    apps = pd.read_csv(apps_path)

    if "player" not in apps.columns or "player_id" not in apps.columns:
        raise SystemExit(
            "epl_player_appearances.csv must contain 'player' and 'player_id' columns."
        )

    apps["player_id"] = apps["player_id"].astype(int)
    if "season" in apps.columns:
        apps["season"] = pd.to_numeric(apps["season"], errors="coerce")
        apps = apps[apps["season"] >= ANALYSIS_START_SEASON].copy()
    id_lookup = (
        apps.groupby(["player_id", "player"])
        .size()
        .reset_index(name="n_appearances")
        .sort_values(["player_id", "n_appearances"], ascending=[True, False])
        .drop_duplicates("player_id")
        [["player_id", "player"]]
        .rename(columns={"player_id": "tm_player_id"})
    )

    # Merge into player index
    if "tm_player_id" in players.columns:
        players["tm_player_id"] = players["tm_player_id"].astype(int)
        mapping = players.merge(
            id_lookup,
            on="tm_player_id",
            how="left",
            suffixes=("", "_from_apps"),
        )
        if "player_from_apps" in mapping.columns:
            mapping["player"] = mapping["player"].fillna(mapping["player_from_apps"])
            mapping.drop(columns=["player_from_apps"], inplace=True)
    else:
        mapping = players.merge(id_lookup, on="player", how="left")

    out_path = DATA_PROCESSED / "player_mapping_tm.csv"
    mapping.to_csv(out_path, index=False)

    total = len(mapping)
    missing = mapping["tm_player_id"].isna().sum()

    print(f"Saved player_mapping_tm.csv with {total} rows -> {out_path}")
    print(f"Players without a mapped tm_player_id: {missing}")

    if missing > 0:
        sample_missing = mapping[mapping["tm_player_id"].isna()].head(10)
        print("\nExample unmapped players (up to 10):")
        cols = [c for c in ["fbref_player_id", "player", "teams", "first_season", "last_season"] if c in sample_missing.columns]
        print(sample_missing[cols])


if __name__ == "__main__":  # pragma: no cover
    main()
