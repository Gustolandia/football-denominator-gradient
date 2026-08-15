#!/usr/bin/env python
"""
09_build_fragility_groups.py

Build prior-history fragility labels without outcome-period look-ahead.

The earlier version classified each player from all injuries observed across
the full study window. That was useful descriptively, but it leaked future
injury information into rows that occurred before those injuries. This version
constructs a player-day fragility table where every row is based only on
information available before that calendar date.

Inputs:
    data/processed/player_day_panel_all_comp.csv
    data/processed/tm_injuries_clean.csv

Outputs:
    data/processed/player_day_fragility.csv
    data/processed/player_fragility_groups.csv

Definitions:
    low_exposure:
        prior_minutes_played < 900

    tough:
        prior_minutes_played >= 900
        and prior_n_spells <= 1
        and prior_injuries_per_10000min <= Q1 frequency threshold
        and prior_max_spell_duration_days <= Q1 severity threshold

    fragile:
        prior_minutes_played >= 900
        and prior_n_spells >= 2
        and (
            prior_injuries_per_10000min >= Q3 frequency threshold
            or prior_max_spell_duration_days >= Q3 severity threshold
        )

    regular:
        all other adequately exposed player-days

The Q1/Q3 thresholds are estimated once from each player's latest prior-history
snapshot after applying the 900-minute exposure threshold. The day-level labels
then use those fixed thresholds while keeping each row's inputs strictly prior
to that row's date.

Run from repo root:
    python src/09_build_fragility_groups.py
"""

import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import ANALYSIS_START_DATE
from pipeline_io import build_injury_episodes


MIN_MINUTES_FOR_FRAGILITY = 900.0
EXPOSURE_MINUTES_COL = "all_minutes_played"


def build_injury_day_table(
    injuries: pd.DataFrame,
    panel_min_date: pd.Timestamp,
    panel_max_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return one injury-summary row per ``tm_player_id`` and spell start date."""
    inj = injuries.copy()
    if inj.empty:
        return pd.DataFrame(
            columns=[
                "tm_player_id",
                "date",
                "n_spells_today",
                "total_days_injured_today",
                "max_spell_duration_today",
            ]
        )

    required = {"tm_player_id", "start_date"}
    missing = required - set(inj.columns)
    if missing:
        raise KeyError(f"injury episodes missing required columns: {sorted(missing)}")

    if "injury_episode_id" not in inj.columns:
        inj = build_injury_episodes(
            inj,
            min_date=panel_min_date,
            max_date=panel_max_date,
        )
    inj["tm_player_id"] = inj["tm_player_id"].astype(int)
    inj["start_date"] = pd.to_datetime(inj["start_date"], errors="coerce")
    inj = inj[
        inj["start_date"].notna()
        & inj["start_date"].between(panel_min_date, panel_max_date)
    ].copy()

    if inj.empty:
        return pd.DataFrame(
            columns=[
                "tm_player_id",
                "date",
                "n_spells_today",
                "total_days_injured_today",
                "max_spell_duration_today",
            ]
        )

    if "duration_days" not in inj:
        end = pd.to_datetime(inj.get("end_date", inj["start_date"]), errors="coerce")
        duration = (end.fillna(inj["start_date"]) - inj["start_date"]).dt.days + 1
        inj["duration_days"] = duration.clip(lower=1)

    return (
        inj.groupby(["tm_player_id", "start_date"], as_index=False)
        .agg(
            n_spells_today=("start_date", "size"),
            total_days_injured_today=("duration_days", "sum"),
            max_spell_duration_today=("duration_days", "max"),
        )
        .rename(columns={"start_date": "date"})
    )


def estimate_thresholds(day_history: pd.DataFrame) -> Dict[str, float]:
    """Estimate fixed Q1/Q3 thresholds from each eligible player's latest row."""
    latest = (
        day_history.sort_values(["tm_player_id", "date"])
        .groupby("tm_player_id", as_index=False)
        .tail(1)
    )
    eligible = latest[latest["prior_minutes_played"] >= MIN_MINUTES_FOR_FRAGILITY]

    if eligible.empty:
        return {
            "q1_freq": 0.0,
            "q3_freq": 0.0,
            "q1_sev": 0.0,
            "q3_sev": 0.0,
        }

    return {
        "q1_freq": float(eligible["prior_injuries_per_10000min"].quantile(0.25)),
        "q3_freq": float(eligible["prior_injuries_per_10000min"].quantile(0.75)),
        "q1_sev": float(eligible["prior_max_spell_duration_days"].quantile(0.25)),
        "q3_sev": float(eligible["prior_max_spell_duration_days"].quantile(0.75)),
    }


def assign_fragility(day_history: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    """Assign prior-history fragility labels to a day-level history table."""
    adequate = day_history["prior_minutes_played"] >= MIN_MINUTES_FOR_FRAGILITY
    tough = (
        adequate
        & (day_history["prior_n_spells"] <= 1)
        & (day_history["prior_injuries_per_10000min"] <= thresholds["q1_freq"])
        & (day_history["prior_max_spell_duration_days"] <= thresholds["q1_sev"])
    )
    fragile = (
        adequate
        & (day_history["prior_n_spells"] >= 2)
        & (
            (day_history["prior_injuries_per_10000min"] >= thresholds["q3_freq"])
            | (day_history["prior_max_spell_duration_days"] >= thresholds["q3_sev"])
        )
    )

    labels = pd.Series("low_exposure", index=day_history.index, dtype="object")
    labels.loc[adequate] = "regular"
    labels.loc[tough] = "tough"
    labels.loc[fragile] = "fragile"
    return labels


def build_history_minutes(
    appearances: pd.DataFrame,
    player_ids: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate recorded all-club minutes used only for prior-history rates."""
    required = {"player_id", "date", "minutes_played"}
    missing = required - set(appearances.columns)
    if missing:
        raise KeyError(f"appearances missing required columns: {sorted(missing)}")
    frame = appearances[["player_id", "date", "minutes_played"]].copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["minutes_played"] = pd.to_numeric(frame["minutes_played"], errors="coerce")
    keep = (
        frame["player_id"].isin(set(pd.Series(player_ids).astype(int)))
        & frame["date"].between(start_date, end_date)
        & frame["minutes_played"].fillna(0).gt(0)
    )
    return (
        frame.loc[keep]
        .groupby(["player_id", "date"], as_index=False)["minutes_played"]
        .sum()
        .rename(
            columns={
                "player_id": "tm_player_id",
                "minutes_played": "history_minutes_played",
            }
        )
        .sort_values(["tm_player_id", "date"])
        .reset_index(drop=True)
    )


def _strict_prior_cumulative(
    panel_small: pd.DataFrame,
    history_minutes: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Attach cumulative minutes and injury episodes strictly before each date."""
    out = panel_small.copy()
    out["prior_minutes_played"] = 0.0
    out["prior_n_spells"] = 0.0
    out["prior_total_days_injured"] = 0.0
    out["prior_max_spell_duration_days"] = 0.0

    minutes_by_player = {
        int(player_id): group.sort_values("date")
        for player_id, group in history_minutes.groupby("tm_player_id", sort=False)
    }
    episodes_by_player = {
        int(player_id): group.sort_values("start_date")
        for player_id, group in episodes.groupby("tm_player_id", sort=False)
    }
    for player_id, indexes in out.groupby("tm_player_id", sort=False).groups.items():
        row_index = np.asarray(list(indexes))
        dates = out.loc[row_index, "date"].to_numpy(dtype="datetime64[ns]")

        minute_rows = minutes_by_player.get(int(player_id))
        if minute_rows is not None:
            minute_dates = minute_rows["date"].to_numpy(dtype="datetime64[ns]")
            minute_cumulative = minute_rows["history_minutes_played"].to_numpy(float).cumsum()
            positions = np.searchsorted(minute_dates, dates, side="left")
            out.loc[row_index, "prior_minutes_played"] = np.where(
                positions > 0,
                minute_cumulative[np.maximum(positions - 1, 0)],
                0.0,
            )

        player_episodes = episodes_by_player.get(int(player_id))
        if player_episodes is not None:
            episode_dates = player_episodes["start_date"].to_numpy(dtype="datetime64[ns]")
            durations = player_episodes["duration_days"].to_numpy(float)
            duration_cumulative = durations.cumsum()
            duration_maximum = np.maximum.accumulate(durations)
            positions = np.searchsorted(episode_dates, dates, side="left")
            previous = np.maximum(positions - 1, 0)
            out.loc[row_index, "prior_n_spells"] = positions.astype(float)
            out.loc[row_index, "prior_total_days_injured"] = np.where(
                positions > 0, duration_cumulative[previous], 0.0
            )
            out.loc[row_index, "prior_max_spell_duration_days"] = np.where(
                positions > 0, duration_maximum[previous], 0.0
            )
    return out


def build_player_day_fragility(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    history_minutes: pd.DataFrame | None = None,
    history_start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build strictly prior day-level injury-history metrics and labels."""
    required = {"tm_player_id", "date", EXPOSURE_MINUTES_COL}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"player_day_panel_all_comp.csv missing required columns: {sorted(missing)}")

    panel_small = panel[["tm_player_id", "date", EXPOSURE_MINUTES_COL]].copy()
    panel_small["tm_player_id"] = panel_small["tm_player_id"].astype(int)
    panel_small["date"] = pd.to_datetime(panel_small["date"], errors="coerce")
    panel_small = (
        panel_small.dropna(subset=["date"])
        .drop_duplicates(["tm_player_id", "date"])
        .sort_values(["tm_player_id", "date"])
        .reset_index(drop=True)
    )

    panel_min_date = panel_small["date"].min()
    panel_max_date = panel_small["date"].max()
    history_origin = (
        pd.Timestamp(history_start_date)
        if history_start_date is not None
        else panel_min_date
    )
    if "injury_episode_id" in injuries:
        episodes = injuries.copy()
    else:
        episodes = build_injury_episodes(
            injuries,
            min_date=history_origin,
            max_date=panel_max_date,
        )
    episodes["start_date"] = pd.to_datetime(episodes["start_date"], errors="coerce")
    episodes = episodes[
        episodes["start_date"].between(history_origin, panel_max_date)
    ].copy()

    if history_minutes is None:
        history_minutes = panel_small.rename(
            columns={EXPOSURE_MINUTES_COL: "history_minutes_played"}
        )[["tm_player_id", "date", "history_minutes_played"]]
    history_minutes = history_minutes.copy()
    history_minutes["date"] = pd.to_datetime(history_minutes["date"], errors="coerce")
    hist = _strict_prior_cumulative(panel_small, history_minutes, episodes)

    hist["prior_years_at_risk"] = (
        (hist["date"] - history_origin).dt.days / 365.25
    ).clip(lower=0.25)
    hist["prior_injuries_per_year"] = hist["prior_n_spells"] / hist["prior_years_at_risk"]
    hist["prior_injuries_per_10000min"] = np.where(
        hist["prior_minutes_played"] > 0,
        hist["prior_n_spells"] / hist["prior_minutes_played"] * 10000.0,
        0.0,
    )

    thresholds = estimate_thresholds(hist)
    hist["fragility_group"] = assign_fragility(hist, thresholds)
    for key, value in thresholds.items():
        hist[key] = value

    output_cols = [
        "tm_player_id",
        "date",
        "prior_minutes_played",
        "prior_n_spells",
        "prior_total_days_injured",
        "prior_max_spell_duration_days",
        "prior_years_at_risk",
        "prior_injuries_per_year",
        "prior_injuries_per_10000min",
        "fragility_group",
        "q1_freq",
        "q3_freq",
        "q1_sev",
        "q3_sev",
    ]
    return hist[output_cols].copy()


def latest_player_fragility(day_fragility: pd.DataFrame) -> pd.DataFrame:
    """Return one latest prior-history summary row per player."""
    latest = (
        day_fragility.sort_values(["tm_player_id", "date"])
        .groupby("tm_player_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest.rename(
        columns={
            "date": "last_date",
            "prior_minutes_played": "total_minutes_played",
            "prior_n_spells": "n_prior_spells",
            "prior_total_days_injured": "prior_total_days_injured",
            "prior_max_spell_duration_days": "max_prior_spell_duration_days",
        }
    )


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    proc_dir = root / "data" / "processed"

    panel_path = proc_dir / "player_day_panel_all_comp.csv"
    day_out_path = proc_dir / "player_day_fragility.csv"
    player_out_path = proc_dir / "player_fragility_groups.csv"

    print(f"Repo root: {root}")
    print(f"Loading player-day panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    print("Panel shape:", panel.shape)

    episodes_path = proc_dir / "tm_injury_episodes.csv"
    if not episodes_path.exists():
        raise FileNotFoundError(
            f"Missing {episodes_path}. Run src/07_build_all_comp_minutes.py first."
        )
    episodes = pd.read_csv(
        episodes_path,
        parse_dates=["start_date", "end_date", "reported_end_date", "return_date"],
        low_memory=False,
    )
    print("Reconciled injury episodes shape:", episodes.shape)

    appearances = pd.read_csv(
        root / "external_data" / "transfermarkt" / "appearances.csv",
        usecols=["player_id", "date", "minutes_played"],
        low_memory=False,
    )
    history_start = pd.Timestamp(ANALYSIS_START_DATE)
    history_minutes = build_history_minutes(
        appearances,
        panel["tm_player_id"].unique(),
        history_start,
        panel["date"].max(),
    )
    day_fragility = build_player_day_fragility(
        panel,
        episodes,
        history_minutes=history_minutes,
        history_start_date=history_start,
    )
    player_fragility = latest_player_fragility(day_fragility)

    print("\nDay-level fragility group counts:")
    print(day_fragility["fragility_group"].value_counts())
    print("\nLatest player-level fragility group counts:")
    print(player_fragility["fragility_group"].value_counts())

    day_fragility.to_csv(day_out_path, index=False)
    player_fragility.to_csv(player_out_path, index=False)

    first_risk_dates = panel.groupby("tm_player_id", as_index=False)["date"].min().rename(
        columns={"date": "first_risk_date"}
    )
    episode_entry = episodes.merge(first_risk_dates, on="tm_player_id", how="left")
    pre_entry_episodes = episode_entry[
        episode_entry["start_date"] < episode_entry["first_risk_date"]
    ]
    minute_entry = history_minutes.merge(first_risk_dates, on="tm_player_id", how="left")
    pre_entry_minutes = minute_entry[minute_entry["date"] < minute_entry["first_risk_date"]]
    audit = pd.DataFrame(
        [
            {"metric": "reconciled_injury_episodes", "value": int(len(episodes))},
            {"metric": "pre_entry_injury_episodes", "value": int(len(pre_entry_episodes))},
            {
                "metric": "players_with_pre_entry_injury_episodes",
                "value": int(pre_entry_episodes["tm_player_id"].nunique()),
            },
            {"metric": "pre_entry_history_appearance_days", "value": int(len(pre_entry_minutes))},
            {
                "metric": "pre_entry_history_minutes",
                "value": int(pre_entry_minutes["history_minutes_played"].sum()),
            },
            {"metric": "q1_frequency_per_10000min", "value": day_fragility["q1_freq"].iloc[0]},
            {"metric": "q3_frequency_per_10000min", "value": day_fragility["q3_freq"].iloc[0]},
            {"metric": "q1_max_episode_days", "value": day_fragility["q1_sev"].iloc[0]},
            {"metric": "q3_max_episode_days", "value": day_fragility["q3_sev"].iloc[0]},
        ]
    )
    audit_path = proc_dir / "results" / "history_chronology_reconciliation.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    print(f"\nSaved day-level fragility -> {day_out_path}")
    print(f"Saved latest player summaries -> {player_out_path}")
    print(f"Saved history chronology reconciliation -> {audit_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
