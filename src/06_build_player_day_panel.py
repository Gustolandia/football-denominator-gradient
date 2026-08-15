"""Build the base EPL player-day panel.

Combines EPL minutes from local Transfermarkt-derived appearances, cleaned
Transfermarkt injury spell start dates, and EPL-only shifted rolling 7-day
games/minutes metrics. The generated panel starts in the configured
``ANALYSIS_START_DATE`` window; older raw appearances are ignored.

Inputs:
    data/raw/epl_matches.csv
    data/raw/epl_player_appearances.csv
    data/processed/player_mapping_tm.csv
    data/processed/tm_injuries_clean.csv

Output:
    data/processed/player_day_panel.csv

``fbref_player_id`` is a legacy column name. It is stable per Transfermarkt
player and equals ``tm_player_id`` in the generated mapping.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

import pandas as pd
from config import ANALYSIS_START_DATE, DATA_RAW, DATA_PROCESSED
from pipeline_io import (
    build_injury_episodes,
    expand_injury_episode_days,
    injury_episode_start_table,
)


def expand_unavailable_spell_days(
    injuries: pd.DataFrame,
    panel_min_date: pd.Timestamp,
    panel_max_date: pd.Timestamp,
    appearance_days: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Expand recorded injury spells into unavailable player-days.

    The injury start date itself is not marked unavailable because it is the
    event day in the onset model. Days after the start through the recorded
    end date are removed from downstream risk denominators.
    """
    episodes = build_injury_episodes(
        injuries,
        appearance_days=appearance_days,
        min_date=panel_min_date,
        max_date=panel_max_date,
    )
    return expand_injury_episode_days(episodes, panel_min_date, panel_max_date)


def main() -> None:  # pragma: no cover
    analysis_start = pd.Timestamp(ANALYSIS_START_DATE)

    schedule = pd.read_csv(DATA_RAW / "epl_matches.csv")
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    schedule = schedule[schedule["date"] >= analysis_start].copy()

    lineups = pd.read_csv(DATA_RAW / "epl_player_appearances.csv")
    lineups = lineups.merge(
        schedule[["game_id", "date"]],
        on="game_id",
        how="left",
    )
    lineups = lineups[lineups["date"] >= analysis_start].copy()
    if lineups.empty:
        raise SystemExit(f"No EPL player appearances on or after {ANALYSIS_START_DATE}.")

    mapping = pd.read_csv(DATA_PROCESSED / "player_mapping_tm.csv")
    inj = pd.read_csv(DATA_PROCESSED / "tm_injuries_clean.csv")

    required_lineup_cols = {"player_id", "date", "minutes_played"}
    missing_lineup = required_lineup_cols - set(lineups.columns)
    if missing_lineup:
        raise SystemExit(
            f"epl_player_appearances.csv missing required columns: {sorted(missing_lineup)}"
        )

    lineups["player_id"] = lineups["player_id"].astype(int)

    # Attach stable ids from the mapping. Prefer player_id/tm_player_id, never name-only.
    lineups = lineups.merge(
        mapping[["fbref_player_id", "tm_player_id"]],
        left_on="player_id",
        right_on="tm_player_id",
        how="left",
    )
    lineups = lineups[lineups["fbref_player_id"].notna()].copy()
    lineups["fbref_player_id"] = lineups["fbref_player_id"].astype(int)
    lineups["tm_player_id"] = lineups["tm_player_id"].astype(int)

    # Build per-player daily exposure
    players = lineups["fbref_player_id"].unique()
    rows = []

    for pid in players:
        df_p = lineups[lineups["fbref_player_id"] == pid]
        if df_p.empty:
            continue
        first = df_p["date"].min()
        last = df_p["date"].max()

        idx = pd.date_range(first, last, freq="D")
        tmp = pd.DataFrame(
            {
                "fbref_player_id": pid,
                "date": idx,
            }
        )

        minutes = (
            df_p.groupby("date")["minutes_played"]
            .sum()
            .rename("minutes_played")
        )

        tmp = tmp.merge(
            minutes,
            left_on="date",
            right_index=True,
            how="left",
        )

        tmp["minutes_played"] = tmp["minutes_played"].fillna(0)
        rows.append(tmp)

    panel = pd.concat(rows, ignore_index=True)

    # Attach stable Transfermarkt person id.
    panel = panel.merge(
        mapping[["fbref_player_id", "tm_player_id"]],
        on="fbref_player_id",
        how="left",
    )
    panel = panel[panel["tm_player_id"].notna()].copy()
    panel["tm_player_id"] = panel["tm_player_id"].astype(int)

    # Mark injury events on start_date
    if "start_date" not in inj.columns:
        raise SystemExit("tm_injuries_clean.csv missing 'start_date'")

    inj["start_date"] = pd.to_datetime(inj["start_date"], errors="coerce")

    appearance_days = lineups[["tm_player_id", "date"]].drop_duplicates()
    episodes = build_injury_episodes(
        inj,
        appearance_days=appearance_days,
        min_date=panel["date"].min(),
        max_date=panel["date"].max(),
    )
    inj_small = injury_episode_start_table(episodes)

    panel = panel.merge(
        inj_small,
        left_on=["tm_player_id", "date"],
        right_on=["tm_player_id", "start_date"],
        how="left",
    )

    panel["n_injury_spells"] = panel["n_injury_spells"].fillna(0).astype(int)
    # injury_event is 1 if at least one cleaned spell starts on this date.
    panel["injury_event"] = (panel["n_injury_spells"] > 0).astype(int)

    # We no longer need start_date as a separate column
    panel.drop(columns=["start_date"], inplace=True)

    unavailable_days = expand_unavailable_spell_days(
        inj,
        panel["date"].min(),
        panel["date"].max(),
        appearance_days=appearance_days,
    )
    panel = panel.merge(
        unavailable_days,
        on=["tm_player_id", "date"],
        how="left",
    )
    panel["injury_unavailable"] = panel["injury_unavailable"].fillna(0).astype(int)
    panel["available_for_injury_risk"] = (panel["injury_unavailable"] == 0).astype(int)

    # Simple congestion metrics: last 7 days (excluding today)
    panel = panel.sort_values(["fbref_player_id", "date"])
    panel["games_played"] = (panel["minutes_played"] > 0).astype(int)

    panel["games_last_7d"] = panel.groupby("fbref_player_id")["games_played"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    ).fillna(0)

    panel["minutes_last_7d"] = panel.groupby("fbref_player_id")["minutes_played"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    ).fillna(0)

    out_path = DATA_PROCESSED / "player_day_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"Player-day panel rows: {len(panel)} -> {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
