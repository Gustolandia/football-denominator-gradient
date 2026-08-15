"""Shared IO and modelling helpers for the corrected analysis scripts.

This module centralizes operations that must stay consistent across multiple
analysis files:

- merging canonical day-level prior-history fragility labels;
- restricting analyses to the dynamic fragility risk set;
- restricting analyses to availability-adjusted risk days;
- assigning canonical 45-minute prior-load bins;
- exposing representative bin midpoints for prediction tables; and
- screening sparse or separated bins before dummy-bin GLM fitting.
"""

from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np
import pandas as pd


FRAGILITY_COLUMNS = [
    "fragility_group",
    "prior_minutes_played",
    "prior_n_spells",
    "prior_total_days_injured",
    "prior_max_spell_duration_days",
    "prior_years_at_risk",
    "prior_injuries_per_year",
    "prior_injuries_per_10000min",
    "q1_freq",
    "q3_freq",
    "q1_sev",
    "q3_sev",
]

BINS_45 = [0, 45, 90, 135, 180, 225, 270, 300]
LABELS_45 = [
    "0-45",
    "46-90",
    "91-135",
    "136-180",
    "181-225",
    "226-270",
    "271-300",
]
REP_VALUES_45 = {
    "0-45": 22.5,
    "46-90": 67.5,
    "91-135": 112.5,
    "136-180": 157.5,
    "181-225": 202.5,
    "226-270": 247.5,
    "271-300": 285.0,
}
MIN_GLM_BIN_DAYS = 50
MIN_GLM_BIN_EVENTS = 1

PUBLICATION_HISTORY_LABELS = {
    "tough": "lower prior-injury-history",
    "regular": "intermediate prior-injury-history",
    "fragile": "higher prior-injury-history",
    "low_exposure": "low exposure",
    "joint": "joint spline-by-history interaction",
    "intermediate_history": "intermediate prior-injury-history",
    "higher_history": "higher prior-injury-history",
    "higher_vs_intermediate": "higher versus intermediate prior-injury-history",
    "lower_intermediate_history": "lower/intermediate prior-injury-history",
}

INJURY_EPISODE_COLUMNS = [
    "tm_player_id",
    "injury_episode_id",
    "start_date",
    "end_date",
    "reported_end_date",
    "duration_days",
    "missedGamesCount",
    "n_source_spells",
    "source_spell_ids",
    "injury_desc",
    "return_date",
    "return_truncated",
]


def _empty_injury_episodes() -> pd.DataFrame:
    """Return an empty table with the canonical injury-episode schema."""
    return pd.DataFrame(columns=INJURY_EPISODE_COLUMNS)


def build_injury_episodes(
    injuries: pd.DataFrame,
    appearance_days: pd.DataFrame | None = None,
    min_date: pd.Timestamp | None = None,
    max_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reconcile injury reports with observed returns and collapse continuous absences.

    A source spell ends no later than the day before the first recorded
    appearance after its start. Overlapping or touching source spells then form
    one episode, except when an observed return on the later start date proves
    that a new at-risk appearance occurred between reports.
    """
    required = {"tm_player_id", "start_date"}
    missing = required - set(injuries.columns)
    if missing:
        raise KeyError(f"injuries missing required columns: {sorted(missing)}")
    if injuries.empty:
        return _empty_injury_episodes()

    spells = injuries.copy().reset_index(drop=True)
    spells["tm_player_id"] = pd.to_numeric(spells["tm_player_id"], errors="coerce")
    spells["start_date"] = pd.to_datetime(spells["start_date"], errors="coerce")
    if "end_date" in spells:
        spells["reported_end_date"] = pd.to_datetime(spells["end_date"], errors="coerce")
    else:
        spells["reported_end_date"] = pd.NaT
    spells = spells.dropna(subset=["tm_player_id", "start_date"]).copy()
    if min_date is not None:
        spells = spells[spells["start_date"] >= pd.Timestamp(min_date)].copy()
    if max_date is not None:
        spells = spells[spells["start_date"] <= pd.Timestamp(max_date)].copy()
    if spells.empty:
        return _empty_injury_episodes()

    spells["tm_player_id"] = spells["tm_player_id"].astype(int)
    spells["reported_end_date"] = spells["reported_end_date"].fillna(spells["start_date"])
    spells["reported_end_date"] = spells[["start_date", "reported_end_date"]].max(axis=1)
    if max_date is not None:
        spells["reported_end_date"] = spells["reported_end_date"].clip(
            upper=pd.Timestamp(max_date)
        )
    if "injury_spell_id" not in spells:
        spells["injury_spell_id"] = spells.index.astype(str)
    spells["injury_spell_id"] = spells["injury_spell_id"].astype(str)
    if "injury_desc" not in spells:
        spells["injury_desc"] = ""
    spells["injury_desc"] = spells["injury_desc"].fillna("").astype(str)
    if "missedGamesCount" not in spells:
        spells["missedGamesCount"] = np.nan
    spells["missedGamesCount"] = pd.to_numeric(
        spells["missedGamesCount"], errors="coerce"
    )

    observed_by_player: dict[int, np.ndarray] = {}
    if appearance_days is not None and not appearance_days.empty:
        appearance_required = {"tm_player_id", "date"}
        missing_appearance = appearance_required - set(appearance_days.columns)
        if missing_appearance:
            raise KeyError(
                "appearance_days missing required columns: "
                f"{sorted(missing_appearance)}"
            )
        observed = appearance_days[["tm_player_id", "date"]].copy()
        observed["tm_player_id"] = pd.to_numeric(observed["tm_player_id"], errors="coerce")
        observed["date"] = pd.to_datetime(observed["date"], errors="coerce")
        observed = observed.dropna().drop_duplicates().sort_values(["tm_player_id", "date"])
        for player_id, group in observed.groupby("tm_player_id", sort=False):
            observed_by_player[int(player_id)] = group["date"].to_numpy(dtype="datetime64[ns]")

    effective_ends = []
    return_dates = []
    starts_with_observed_appearance = []
    for row in spells.itertuples(index=False):
        dates = observed_by_player.get(int(row.tm_player_id), np.array([], dtype="datetime64[ns]"))
        start64 = np.datetime64(row.start_date)
        starts_with_observed_appearance.append(
            bool(np.any(dates == start64))
        )
        position = int(np.searchsorted(dates, start64, side="right"))
        return_date = pd.NaT
        effective_end = row.reported_end_date
        if position < len(dates):
            candidate = pd.Timestamp(dates[position])
            if candidate <= row.reported_end_date:
                return_date = candidate
                effective_end = min(effective_end, candidate - pd.Timedelta(days=1))
        effective_ends.append(max(effective_end, row.start_date))
        return_dates.append(return_date)

    spells["end_date"] = pd.to_datetime(effective_ends)
    spells["return_date"] = pd.to_datetime(return_dates)
    spells["starts_with_observed_appearance"] = starts_with_observed_appearance
    spells["return_truncated"] = spells["end_date"] < spells["reported_end_date"]
    spells = spells.sort_values(
        ["tm_player_id", "start_date", "end_date", "injury_spell_id"]
    ).reset_index(drop=True)

    episodes = []
    for player_id, group in spells.groupby("tm_player_id", sort=False):
        current = None
        episode_number = 0
        for row in group.itertuples(index=False):
            source = {
                "start_date": row.start_date,
                "end_date": row.end_date,
                "reported_end_date": row.reported_end_date,
                "source_spell_ids": [row.injury_spell_id],
                "injury_desc": [row.injury_desc] if row.injury_desc else [],
                "missed_games": row.missedGamesCount,
                "return_date": row.return_date,
                "return_truncated": bool(row.return_truncated),
                "starts_with_observed_appearance": bool(
                    row.starts_with_observed_appearance
                ),
            }
            if current is None:
                current = source
                continue

            touches = row.start_date <= current["end_date"] + pd.Timedelta(days=1)
            observed_restart = source["starts_with_observed_appearance"]
            if touches and not observed_restart:
                current["end_date"] = max(current["end_date"], row.end_date)
                current["reported_end_date"] = max(
                    current["reported_end_date"], row.reported_end_date
                )
                current["source_spell_ids"].append(row.injury_spell_id)
                if row.injury_desc:
                    current["injury_desc"].append(row.injury_desc)
                missed_games = [current["missed_games"], row.missedGamesCount]
                known_missed_games = [value for value in missed_games if pd.notna(value)]
                current["missed_games"] = (
                    max(known_missed_games) if known_missed_games else np.nan
                )
                if pd.isna(current["return_date"]) or (
                    pd.notna(row.return_date) and row.return_date > current["return_date"]
                ):
                    current["return_date"] = row.return_date
                current["return_truncated"] = (
                    current["return_truncated"] or bool(row.return_truncated)
                )
                continue

            episode_number += 1
            episodes.append((int(player_id), episode_number, current))
            current = source

        episode_number += 1
        episodes.append((int(player_id), episode_number, current))

    rows = []
    for player_id, episode_number, episode in episodes:
        start = episode["start_date"]
        end = episode["end_date"]
        rows.append(
            {
                "tm_player_id": player_id,
                "injury_episode_id": f"{player_id}:{start.date()}:{episode_number}",
                "start_date": start,
                "end_date": end,
                "reported_end_date": episode["reported_end_date"],
                "duration_days": int((end - start).days + 1),
                "missedGamesCount": episode["missed_games"],
                "n_source_spells": len(set(episode["source_spell_ids"])),
                "source_spell_ids": ";".join(sorted(set(episode["source_spell_ids"]))),
                "injury_desc": "; ".join(sorted(set(episode["injury_desc"]))),
                "return_date": episode["return_date"],
                "return_truncated": bool(episode["return_truncated"]),
            }
        )
    return pd.DataFrame(rows, columns=INJURY_EPISODE_COLUMNS)


def injury_episode_start_table(episodes: pd.DataFrame) -> pd.DataFrame:
    """Return one injury-onset summary row per player and episode start date."""
    columns = [
        "tm_player_id",
        "start_date",
        "injury_spell_id",
        "injury_desc",
        "n_injury_spells",
    ]
    if episodes.empty:
        return pd.DataFrame(columns=columns)
    required = {"tm_player_id", "start_date", "injury_episode_id", "injury_desc"}
    missing = required - set(episodes.columns)
    if missing:
        raise KeyError(f"episodes missing required columns: {sorted(missing)}")
    return (
        episodes.groupby(["tm_player_id", "start_date"], as_index=False)
        .agg(
            injury_spell_id=(
                "injury_episode_id",
                lambda values: ";".join(sorted(set(values.astype(str)))),
            ),
            injury_desc=(
                "injury_desc",
                lambda values: "; ".join(sorted(set(value for value in values if value))),
            ),
            n_injury_spells=("injury_episode_id", "nunique"),
        )
    )


def expand_injury_episode_days(
    episodes: pd.DataFrame,
    panel_min_date: pd.Timestamp,
    panel_max_date: pd.Timestamp,
) -> pd.DataFrame:
    """Expand reconciled episodes into unavailable days after each onset date."""
    columns = ["tm_player_id", "date", "injury_unavailable"]
    if episodes.empty:
        return pd.DataFrame(columns=columns)
    required = {"tm_player_id", "start_date", "end_date"}
    missing = required - set(episodes.columns)
    if missing:
        raise KeyError(f"episodes missing required columns: {sorted(missing)}")

    rows = []
    for row in episodes.itertuples(index=False):
        start = max(pd.Timestamp(row.start_date) + pd.Timedelta(days=1), panel_min_date)
        end = min(pd.Timestamp(row.end_date), panel_max_date)
        if end >= start:
            rows.append(
                pd.DataFrame(
                    {
                        "tm_player_id": int(row.tm_player_id),
                        "date": pd.date_range(start, end, freq="D"),
                        "injury_unavailable": 1,
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(["tm_player_id", "date"])
        .reset_index(drop=True)
    )


def merge_day_fragility(panel: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    """Merge prior-history fragility labels onto a player-day panel."""
    frag_path = processed_dir / "player_day_fragility.csv"
    if not frag_path.exists():
        raise FileNotFoundError(
            f"Missing {frag_path}. Run src/09_build_fragility_groups.py first."
        )

    panel_out = panel.copy()
    panel_out["date"] = pd.to_datetime(panel_out["date"], errors="coerce")
    panel_out["tm_player_id"] = panel_out["tm_player_id"].astype(int)

    drop_cols = []
    for col in panel_out.columns:
        if col in FRAGILITY_COLUMNS or col.endswith("_frag"):
            drop_cols.append(col)
    if drop_cols:
        panel_out = panel_out.drop(columns=drop_cols)

    day_fragility = pd.read_csv(frag_path, parse_dates=["date"], low_memory=False)
    day_fragility["tm_player_id"] = day_fragility["tm_player_id"].astype(int)

    return panel_out.merge(day_fragility, on=["tm_player_id", "date"], how="left")


def restrict_to_fragility_risk_set(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep player-days with an eligible prior-history fragility label."""
    if "fragility_group" not in panel.columns:
        raise KeyError("panel must contain fragility_group")
    return panel[panel["fragility_group"].isin(["tough", "regular", "fragile"])].copy()


def restrict_to_available_risk_set(
    panel: pd.DataFrame,
    availability_col: str = "available_for_injury_risk",
) -> pd.DataFrame:
    """Keep rows where the player is not inside a known ongoing injury spell."""
    if availability_col not in panel.columns:
        raise KeyError(f"panel must contain {availability_col}")
    return panel[panel[availability_col].astype(bool)].copy()


def add_45min_load_bins(
    panel: pd.DataFrame,
    load_col: str = "all_minutes_last_7d",
    out_col: str = "all_minutes7d_bin",
) -> pd.DataFrame:
    """Return a copy with canonical 45-minute prior-load bins added."""
    if load_col not in panel.columns:
        raise KeyError(f"panel must contain {load_col}")

    panel_out = panel.copy()
    panel_out[out_col] = pd.cut(
        panel_out[load_col],
        bins=BINS_45,
        labels=LABELS_45,
        include_lowest=True,
        right=True,
    )
    return panel_out


def estimable_bin_labels(
    panel: pd.DataFrame,
    bin_col: str = "all_minutes7d_bin",
    event_col: str = "injury_event",
    min_days: int = MIN_GLM_BIN_DAYS,
    min_events: int = MIN_GLM_BIN_EVENTS,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Identify bins that can support dummy-bin GLM estimation.

    Crude summaries should keep every bin. For GLMs, bins with too few days,
    zero events, or all events are excluded to avoid separation-driven
    coefficients and misleading predictions.
    """
    missing = [col for col in [bin_col, event_col] if col not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing required columns {missing}")

    counts = (
        panel.dropna(subset=[bin_col])
        .groupby(bin_col, observed=False)
        .agg(
            n_days=(event_col, "size"),
            n_events=(event_col, "sum"),
        )
        .reset_index()
    )
    counts["estimable"] = (
        (counts["n_days"] >= min_days)
        & (counts["n_events"] >= min_events)
        & (counts["n_events"] < counts["n_days"])
    )
    labels = [str(value) for value in counts.loc[counts["estimable"], bin_col]]
    return labels, counts


def add_publication_history_labels(
    frame: pd.DataFrame,
    source_col: str,
    out_col: str = "publication_history_stratum",
    labels: Mapping[str, str] = PUBLICATION_HISTORY_LABELS,
) -> pd.DataFrame:
    """Add publication-facing prior-injury-history labels to an output table."""
    out = frame.copy()
    out[out_col] = out[source_col].astype(str).map(labels).fillna(out[source_col].astype(str))
    return out
