"""Extract EPL matches and player appearances from local Transfermarkt dumps.

Despite the legacy filename, this script does not fetch FBref data. It reads
Transfermarkt dataset CSVs from ``external_data/transfermarkt`` and writes the
two raw inputs used by the rest of the pipeline:

- ``data/raw/epl_matches.csv``
- ``data/raw/epl_player_appearances.csv``

The output schema intentionally keeps simple project-standard names such as
``player_id``, ``player``, ``team``, ``season``, and ``minutes_played``.
Downstream scripts treat ``player_id`` as a Transfermarkt player identifier.
"""

import sys
from pathlib import Path

# --- Make project root importable when running as "python src/01_fetch_fbref.py"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

from typing import List

import pandas as pd

from config import DATA_RAW

# Premier League competition code in the Transfermarkt dataset dumps.
EPL_COMP_ID = "GB1"


def _pick_col(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    """Return the first existing column from candidates or exit with a clear error."""
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"None of the candidate columns {candidates!r} found for {label}. "
        f"Available columns: {sorted(df.columns)}"
    )


def main() -> None:  # pragma: no cover
    base = Path("external_data") / "transfermarkt"

    games_path = base / "games.csv"
    apps_path = base / "appearances.csv"
    clubs_path = base / "clubs.csv"

    if not games_path.is_file():
        raise SystemExit(f"Expected {games_path} to exist.")
    if not apps_path.is_file():
        raise SystemExit(f"Expected {apps_path} to exist.")
    if not clubs_path.is_file():
        raise SystemExit(f"Expected {clubs_path} to exist.")

    # --------------------
    # Matches (EPL only)
    # --------------------
    print(f"Loading games from {games_path} ...")
    games = pd.read_csv(games_path)

    comp_col = _pick_col(games, ["competition_id", "competitionId"], "competition id")
    game_id_col = _pick_col(games, ["game_id", "id"], "game id")
    season_col = _pick_col(games, ["season"], "season")
    date_col = _pick_col(games, ["date"], "date")
    home_name_col = _pick_col(
        games, ["home_club_name", "home_team_name"], "home team name"
    )
    away_name_col = _pick_col(
        games, ["away_club_name", "away_team_name"], "away team name"
    )
    home_goals_col = _pick_col(
        games, ["home_club_goals", "home_goals"], "home goals"
    )
    away_goals_col = _pick_col(
        games, ["away_club_goals", "away_goals"], "away goals"
    )

    # Filter to Premier League only
    epl_games = games[games[comp_col] == EPL_COMP_ID].copy()
    if epl_games.empty:
        raise SystemExit(
            f"No games found for competition {EPL_COMP_ID!r} in games.csv"
        )

    epl_games.rename(
        columns={
            game_id_col: "game_id",
            season_col: "season",
            date_col: "date",
            home_name_col: "home_team",
            away_name_col: "away_team",
            home_goals_col: "home_goals",
            away_goals_col: "away_goals",
        },
        inplace=True,
    )

    epl_games["date"] = pd.to_datetime(epl_games["date"], errors="coerce")

    matches_out = DATA_RAW / "epl_matches.csv"
    epl_games[
        [
            "game_id",
            "season",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ].to_csv(matches_out, index=False)
    print(f"Saved EPL matches: {len(epl_games)} rows -> {matches_out}")

    # --------------------
    # Player appearances (EPL only)
    # --------------------
    print(f"Loading appearances from {apps_path} ...")
    apps = pd.read_csv(apps_path)

    game_id_col_a = _pick_col(apps, ["game_id"], "game id in appearances")
    player_id_col = _pick_col(apps, ["player_id"], "player id")
    player_name_col = _pick_col(apps, ["player_name"], "player name")
    minutes_col = _pick_col(
        apps, ["minutes_played", "minutes"], "minutes played"
    )
    club_id_col = _pick_col(
        apps, ["player_club_id", "club_id"], "player club id in appearances"
    )

    # Restrict to EPL games only
    epl_game_ids = set(epl_games["game_id"].unique())
    apps_epl = apps[apps[game_id_col_a].isin(epl_game_ids)].copy()
    if apps_epl.empty:
        raise SystemExit(
            "No appearances found for EPL games in appearances.csv. "
            "Check that games.csv and appearances.csv come from the same dataset."
        )

    # Attach season from games table
    season_lookup = epl_games[["game_id", "season"]]
    apps_epl = apps_epl.merge(
        season_lookup, left_on=game_id_col_a, right_on="game_id", how="left"
    )

    # --------------------
    # Join club IDs to club names
    # --------------------
    print(f"Loading clubs from {clubs_path} ...")
    clubs = pd.read_csv(clubs_path)
    clubs_id_col = _pick_col(clubs, ["club_id", "id"], "club id in clubs")
    clubs_name_col = _pick_col(clubs, ["name", "club_name"], "club name in clubs")

    clubs_small = clubs[[clubs_id_col, clubs_name_col]].rename(
        columns={
            clubs_id_col: "club_id",
            clubs_name_col: "team",
        }
    )

    apps_epl = apps_epl.rename(columns={club_id_col: "club_id"})
    apps_epl = apps_epl.merge(clubs_small, on="club_id", how="left")

    # Final rename to our standard columns
    apps_epl.rename(
        columns={
            player_id_col: "player_id",
            player_name_col: "player",
            minutes_col: "minutes_played",
        },
        inplace=True,
    )

    appearances_out = DATA_RAW / "epl_player_appearances.csv"
    apps_epl[
        [
            "game_id",
            "player_id",
            "player",
            "team",
            "season",
            "minutes_played",
        ]
    ].to_csv(appearances_out, index=False)
    print(
        f"Saved EPL player appearances: {len(apps_epl)} rows -> "
        f"{appearances_out}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
