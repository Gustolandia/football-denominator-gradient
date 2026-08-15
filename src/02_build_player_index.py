"""Build the stable player index used by the EPL congestion pipeline.

Only players with at least one EPL appearance in the configured analysis window
(``ANALYSIS_START_SEASON`` onward) are retained.

Input:
    data/raw/epl_player_appearances.csv

Output:
    data/processed/player_index_fbref.csv

The output keeps the legacy column name ``fbref_player_id`` for compatibility
with downstream scripts. In the corrected Transfermarkt-based pipeline it is
not a FBref identifier; it is the stable Transfermarkt ``player_id`` copied to
the legacy column so players are not multiplied across teams or seasons.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

import pandas as pd
from config import ANALYSIS_START_SEASON, DATA_RAW, DATA_PROCESSED


def main() -> None:  # pragma: no cover
    """
    Build the internal stable player index from EPL player appearances.

    Input:
      data/raw/epl_player_appearances.csv

    Output:
      data/processed/player_index_fbref.csv

    The output still uses the legacy column name `fbref_player_id` for
    compatibility with the rest of the pipeline. In the current implementation
    it is not a FBref identifier. It is the stable Transfermarkt `player_id`
    copied into the legacy column so downstream code can keep the same schema
    without multiplying a player across seasons.
    """
    lineups_path = DATA_RAW / "epl_player_appearances.csv"
    lineups = pd.read_csv(lineups_path)

    required = {"player_id", "player", "team", "season"}
    missing = required - set(lineups.columns)
    if missing:
        raise SystemExit(
            f"{lineups_path} is missing required columns: {sorted(missing)}"
        )

    lineups["player_id"] = lineups["player_id"].astype(int)
    lineups["season"] = pd.to_numeric(lineups["season"], errors="coerce")
    lineups = lineups[lineups["season"] >= ANALYSIS_START_SEASON].copy()
    if lineups.empty:
        raise SystemExit(
            f"No EPL player appearances at or after season {ANALYSIS_START_SEASON}."
        )

    primary_names = (
        lineups.groupby(["player_id", "player"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["player_id", "n_rows", "player"], ascending=[True, False, True])
        .drop_duplicates("player_id")
        [["player_id", "player"]]
    )

    teams = (
        lineups.groupby("player_id")["team"]
        .apply(lambda s: "; ".join(sorted(map(str, s.dropna().unique()))))
        .reset_index(name="teams")
    )

    seasons = (
        lineups.groupby("player_id")
        .agg(first_season=("season", "min"), last_season=("season", "max"))
        .reset_index()
    )

    players = (
        primary_names.merge(teams, on="player_id", how="left")
        .merge(seasons, on="player_id", how="left")
        .sort_values(["player", "player_id"])
        .reset_index(drop=True)
    )

    players["tm_player_id"] = players["player_id"].astype(int)
    players["fbref_player_id"] = players["tm_player_id"]
    players.drop(columns=["player_id"], inplace=True)

    out_path = DATA_PROCESSED / "player_index_fbref.csv"
    players.to_csv(out_path, index=False)
    print(f"Unique stable players: {len(players)} -> {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
