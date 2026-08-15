#!/usr/bin/env python
"""Build clinician-facing surveillance bridge outputs.

This pipeline step does not turn the public Transfermarkt data into a club
medical surveillance dataset. Instead, it translates the reconstructed public
signals into units clinicians commonly expect:

- match-associated injury-proxy incidence per 1,000 match hours;
- reported absence duration and missed-games summaries by proxy context;
- duration-specific proxy incidence for higher-history versus
  lower/intermediate-history player-days; and
- sparse descriptive exposure-category summaries for duration-specific events.

Inputs:
    data/processed/player_day_panel_all_comp.csv
    data/processed/tm_injuries_clean.csv

Outputs:
    data/processed/results/clinical_match_hour_rates.csv
    data/processed/results/clinical_duration_context_summary.csv
    data/processed/results/clinical_matchproxy_duration_rates_by_group.csv
    data/processed/results/clinical_matchproxy_duration_rates_by_burden.csv
    data/processed/results/figures/H1_clinical_bridge_rates.png

The figures are also mirrored to manuscript/figures/.
"""

from __future__ import annotations

import ast
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_io import (
    LABELS_45,
    add_45min_load_bins,
    restrict_to_available_risk_set,
    restrict_to_fragility_risk_set,
)


MATCH_MINUTES_COL = "all_minutes_played"
PLAYER_ID_COL = "tm_player_id"
DURATION_BUCKETS = [
    "<1 week",
    "1 week to 2 months",
    "2 months to 1 year",
    ">1 year",
    "unknown",
]
RATE_SCOPES = [
    ("same_day_plus_lag1", "injury_event_matchproxy"),
    ("same_day_only", "injury_event_matchproxy_same_day"),
    ("lag1_only", "injury_event_matchproxy_lag1"),
    ("specific_description_only", "injury_event_matchproxy_specific"),
]


def parse_duration_days(value) -> float:
    """Extract numeric reported injury duration days from a Transfermarkt payload."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    payload = value
    if isinstance(value, str):
        if not value.strip():
            return np.nan
        try:
            payload = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return np.nan
    if not isinstance(payload, dict):
        return np.nan
    days = payload.get("days")
    try:
        return float(days)
    except (TypeError, ValueError):
        return np.nan


def duration_bucket(days: float) -> str:
    """Map reported absence duration to broad clinician-readable buckets."""
    if pd.isna(days):
        return "unknown"
    if days < 7:
        return "<1 week"
    if days <= 60:
        return "1 week to 2 months"
    if days <= 365:
        return "2 months to 1 year"
    return ">1 year"


def clinical_risk_group(fragility_group: str) -> str:
    """Collapse dynamic labels into publication-safe prior-history strata."""
    if fragility_group == "fragile":
        return "higher_history"
    if fragility_group in {"regular", "tough"}:
        return "lower_intermediate_history"
    return "other"


def split_spell_ids(value) -> list[str]:
    """Split raw numeric or canonical episode identifiers."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    ids = []
    for part in str(value).split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            numeric = float(part)
            if numeric.is_integer():
                ids.append(str(int(numeric)))
        except ValueError:
            if ":" in part:
                ids.append(part)
    return ids


def approximate_count_rate_interval(
    events: float,
    denominator: float,
    scale: float,
) -> tuple[float, float, float]:
    """Return a log-normal approximate count-rate interval on a chosen scale."""
    if denominator <= 0 or pd.isna(denominator):
        return np.nan, np.nan, np.nan
    rate = float(events) / float(denominator) * float(scale)
    if events <= 0:
        return 0.0, 0.0, -math.log(0.025) / float(denominator) * float(scale)
    log_margin = 1.96 / math.sqrt(float(events))
    return rate, rate * math.exp(-log_margin), rate * math.exp(log_margin)


def safe_rates(
    events: float,
    minutes: float,
    appearances: float | None = None,
) -> dict[str, float]:
    """Return match-minute, match-hour, and appearance rates with safe denominators."""
    per_10000, low_10000, high_10000 = approximate_count_rate_interval(
        events,
        minutes,
        10000.0,
    )
    if appearances is None:
        app_rate = app_low = app_high = np.nan
    else:
        app_rate, app_low, app_high = approximate_count_rate_interval(
            events,
            appearances,
            1000.0,
        )
    return {
        "events_per_10000_min": per_10000,
        "events_per_1000_match_hours": per_10000 * 6.0,
        "events_per_10000_min_ci_low": low_10000,
        "events_per_10000_min_ci_high": high_10000,
        "events_per_1000_match_hours_ci_low": low_10000 * 6.0,
        "events_per_1000_match_hours_ci_high": high_10000 * 6.0,
        "events_per_1000_appearances": app_rate,
        "events_per_1000_appearances_ci_low": app_low,
        "events_per_1000_appearances_ci_high": app_high,
    }


def build_injury_duration_lookup(injuries: pd.DataFrame) -> pd.DataFrame:
    """Create one duration/missed-games lookup row per cleaned injury spell."""
    identifier = (
        "injury_episode_id"
        if "injury_episode_id" in injuries.columns
        else "injury_spell_id"
    )
    if identifier not in injuries.columns:
        raise KeyError("injuries must contain injury_spell_id or injury_episode_id")

    out = injuries.copy()
    out["injury_spell_id"] = out[identifier].astype(str)
    if identifier == "injury_episode_id" and "duration_days" in out:
        out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce")
    else:
        details = out.get("durationDetails", pd.Series(np.nan, index=out.index))
        out["duration_days"] = details.apply(parse_duration_days)

        if {"start_date", "end_date"}.issubset(out.columns):
            starts = pd.to_datetime(out["start_date"], errors="coerce")
            ends = pd.to_datetime(out["end_date"], errors="coerce")
            fallback = (ends - starts).dt.days.clip(lower=0)
            out["duration_days"] = out["duration_days"].fillna(fallback)

    if "missedGamesCount" not in out.columns:
        out["missedGamesCount"] = np.nan
    if "injury_desc" not in out.columns:
        out["injury_desc"] = ""

    out["duration_bucket"] = out["duration_days"].apply(duration_bucket)
    keep = [
        "injury_spell_id",
        "duration_days",
        "duration_bucket",
        "missedGamesCount",
        "injury_desc",
    ]
    out = out[keep].copy()
    return out.drop_duplicates("injury_spell_id")


def summarize_spell_ids(spell_ids, lookup: pd.DataFrame) -> dict[str, object]:
    """Summarize one or more spell ids as one event-level severity proxy."""
    ids = split_spell_ids(spell_ids)
    if not ids:
        return {
            "duration_days": np.nan,
            "duration_bucket": "unknown",
            "missed_games": np.nan,
            "duration_known": False,
        }

    rows = lookup[lookup["injury_spell_id"].isin(ids)]
    if rows.empty:
        return {
            "duration_days": np.nan,
            "duration_bucket": "unknown",
            "missed_games": np.nan,
            "duration_known": False,
        }

    duration_days = rows["duration_days"].dropna()
    missed_games = rows["missedGamesCount"].dropna()
    if duration_days.empty:
        duration_value = np.nan
    else:
        duration_value = float(duration_days.max())
    if missed_games.empty:
        missed_value = np.nan
    else:
        missed_value = float(missed_games.sum())

    return {
        "duration_days": duration_value,
        "duration_bucket": duration_bucket(duration_value),
        "missed_games": missed_value,
        "duration_known": not pd.isna(duration_value),
    }


def attach_matchproxy_duration_metadata(
    match_panel: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach back-attributed injury duration metadata to match-proxy rows."""
    required = {
        PLAYER_ID_COL,
        "date",
        "injury_spell_id",
        "injury_event_matchproxy_same_day",
        "injury_event_matchproxy_lag1",
    }
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    def per_player(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()
        next_spell = group["injury_spell_id"].shift(-1)
        group["matchproxy_spell_id"] = np.where(
            group["injury_event_matchproxy_same_day"].astype(int) == 1,
            group["injury_spell_id"].fillna(""),
            np.where(
                group["injury_event_matchproxy_lag1"].astype(int) == 1,
                next_spell.fillna(""),
                "",
            ),
        )
        return group

    out = pd.concat(
        [per_player(group) for _, group in match_panel.groupby(PLAYER_ID_COL, sort=False)],
        ignore_index=True,
    )
    meta = out["matchproxy_spell_id"].apply(lambda value: summarize_spell_ids(value, lookup))
    meta_df = pd.DataFrame(list(meta))
    meta_df = meta_df.add_prefix("matchproxy_")
    return pd.concat([out.reset_index(drop=True), meta_df], axis=1)


def attach_onset_duration_metadata(
    onset_panel: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach current-day injury duration metadata to injury-onset rows."""
    if "injury_spell_id" not in onset_panel.columns:
        raise KeyError("onset_panel must contain injury_spell_id")
    out = onset_panel.copy()
    meta = out["injury_spell_id"].apply(lambda value: summarize_spell_ids(value, lookup))
    meta_df = pd.DataFrame(list(meta))
    return pd.concat([out.reset_index(drop=True), meta_df], axis=1)


def build_match_hour_rates(match_panel: pd.DataFrame) -> pd.DataFrame:
    """Build match-proxy incidence rates in clinician-friendly units."""
    missing = [col for _, col in RATE_SCOPES if col not in match_panel.columns]
    if missing:
        raise KeyError(f"match_panel missing event columns: {missing}")

    group_specs = [("overall", "overall", match_panel)]
    for group in ["tough", "regular", "fragile"]:
        group_specs.append(
            (
                "fragility_group",
                group,
                match_panel[match_panel["fragility_group"] == group],
            )
        )
    for group in ["lower_intermediate_history", "higher_history"]:
        group_specs.append(
            (
                "clinical_risk_group",
                group,
                match_panel[match_panel["clinical_risk_group"] == group],
            )
        )

    rows = []
    for scope, event_col in RATE_SCOPES:
        for group_kind, group, subset in group_specs:
            minutes = float(subset[MATCH_MINUTES_COL].sum())
            events = float(subset[event_col].sum())
            appearances = int(len(subset))
            rates = safe_rates(events, minutes, appearances)
            rows.append(
                {
                    "rate_scope": scope,
                    "group_kind": group_kind,
                    "group": group,
                    "match_rows": appearances,
                    "match_minutes": minutes,
                    "match_hours": minutes / 60.0 if minutes > 0 else np.nan,
                    "events": int(events),
                    **rates,
                }
            )
    return pd.DataFrame(rows)


def build_duration_context_summary(onset_panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize reported time-loss proxies by onset-day injury context."""
    group_cols = ["fragility_group", "clinical_risk_group", "injury_context"]
    rows = []
    for keys, subset in onset_panel.groupby(group_cols, dropna=False):
        duration = subset["duration_days"].dropna()
        missed = subset["missed_games"].dropna()
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "events": int(len(subset)),
                "duration_known": int(duration.size),
                "median_duration_days": float(duration.median()) if not duration.empty else np.nan,
                "mean_duration_days": float(duration.mean()) if not duration.empty else np.nan,
                "total_duration_days": float(duration.sum()) if not duration.empty else np.nan,
                "missed_games_known": int(missed.size),
                "median_missed_games": float(missed.median()) if not missed.empty else np.nan,
                "total_missed_games": float(missed.sum()) if not missed.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_duration_rate_table(
    match_panel: pd.DataFrame,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    """Build cause-specific match-proxy rates by reported duration bucket."""
    group_cols = list(group_cols)
    rows = []
    groupby_arg = group_cols[0] if len(group_cols) == 1 else group_cols
    grouped = match_panel.groupby(groupby_arg, dropna=False, observed=False)
    for keys, subset in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        minutes = float(subset[MATCH_MINUTES_COL].sum())
        events = subset[subset["injury_event_matchproxy"].astype(int) == 1]
        bucket_counts = events["matchproxy_duration_bucket"].value_counts()
        for bucket in DURATION_BUCKETS:
            n_events = float(bucket_counts.get(bucket, 0))
            rows.append(
                {
                    **dict(zip(group_cols, keys)),
                    "duration_bucket": bucket,
                    "match_rows": int(len(subset)),
                    "match_minutes": minutes,
                    "match_hours": minutes / 60.0 if minutes > 0 else np.nan,
                    "events": int(n_events),
                    **safe_rates(n_events, minutes),
                }
            )
    return pd.DataFrame(rows)


def build_clinical_outputs(panel: pd.DataFrame, injuries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all clinical bridge output tables from the regenerated panel."""
    lookup = build_injury_duration_lookup(injuries)
    risk = restrict_to_fragility_risk_set(panel)
    risk = restrict_to_available_risk_set(risk)
    risk = risk.copy()
    risk["clinical_risk_group"] = risk["fragility_group"].apply(clinical_risk_group)

    risk_with_matchproxy_meta = attach_matchproxy_duration_metadata(risk, lookup)
    match_panel = risk_with_matchproxy_meta[
        risk_with_matchproxy_meta[MATCH_MINUTES_COL] > 0
    ].copy()
    match_panel = add_45min_load_bins(match_panel)

    onset_panel = risk[risk["injury_event"].astype(int) == 1].copy()
    onset_panel = attach_onset_duration_metadata(onset_panel, lookup)

    duration_by_burden = build_duration_rate_table(
        match_panel.dropna(subset=["all_minutes7d_bin"]).copy(),
        ["clinical_risk_group", "all_minutes7d_bin"],
    )
    duration_by_burden["all_minutes7d_bin"] = pd.Categorical(
        duration_by_burden["all_minutes7d_bin"].astype(str),
        categories=LABELS_45,
        ordered=True,
    )
    duration_by_burden = duration_by_burden.sort_values(
        ["clinical_risk_group", "all_minutes7d_bin", "duration_bucket"]
    ).reset_index(drop=True)

    return {
        "clinical_match_hour_rates": build_match_hour_rates(match_panel),
        "clinical_duration_context_summary": build_duration_context_summary(onset_panel),
        "clinical_matchproxy_duration_rates_by_group": build_duration_rate_table(
            match_panel,
            ["clinical_risk_group"],
        ),
        "clinical_matchproxy_duration_rates_by_burden": duration_by_burden,
    }


def plot_clinical_bridge(outputs: dict[str, pd.DataFrame], fig_dir: Path) -> None:  # pragma: no cover
    """Plot the compact clinician-facing bridge figure."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    rate_df = outputs["clinical_match_hour_rates"]
    duration_df = outputs["clinical_matchproxy_duration_rates_by_group"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    rates = rate_df[
        (rate_df["rate_scope"] == "same_day_plus_lag1")
        & (rate_df["group_kind"] == "fragility_group")
    ].copy()
    rates["group"] = pd.Categorical(
        rates["group"],
        categories=["tough", "regular", "fragile"],
        ordered=True,
    )
    rates = rates.sort_values("group")
    display_labels = {
        "tough": "Lower prior\ninjury history",
        "regular": "Intermediate prior\ninjury history",
        "fragile": "Higher prior\ninjury history",
    }
    x_labels = rates["group"].astype(str).map(display_labels)
    axes[0].bar(
        x_labels,
        rates["events_per_1000_match_hours"],
        color=["#4B5563", "#2563A7", "#B83B2E"],
    )
    axes[0].set_title("Match-associated proxy incidence")
    axes[0].set_ylabel("Proxy events per 1,000 match hours")
    axes[0].set_xlabel("Dynamic prior-injury-history stratum")

    duration = duration_df[
        duration_df["duration_bucket"].isin(DURATION_BUCKETS[:4])
    ].copy()
    pivot = duration.pivot(
        index="duration_bucket",
        columns="clinical_risk_group",
        values="events_per_1000_match_hours",
    ).reindex(DURATION_BUCKETS[:4])
    pivot = pivot[["lower_intermediate_history", "higher_history"]]
    x = np.arange(len(pivot.index))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        pivot["lower_intermediate_history"],
        width,
        label="Lower/intermediate history",
        color="#2563A7",
    )
    axes[1].bar(
        x + width / 2,
        pivot["higher_history"],
        width,
        label="Higher history",
        color="#B83B2E",
    )
    axes[1].set_title("Incidence by reported absence duration")
    axes[1].set_ylabel("Proxy events per 1,000 match hours")
    axes[1].set_xlabel("Reported absence duration")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pivot.index, rotation=20, ha="right")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    out = fig_dir / "H1_clinical_bridge_rates.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(outputs: dict[str, pd.DataFrame], results_dir: Path) -> None:  # pragma: no cover
    """Write clinical bridge tables to the results directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for stem, df in outputs.items():
        df.to_csv(results_dir / f"{stem}.csv", index=False)


def mirror_figures(results_fig_dir: Path, manuscript_fig_dir: Path) -> None:  # pragma: no cover
    """Copy clinical bridge figures into the manuscript figure directory."""
    manuscript_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig in results_fig_dir.glob("H*.png"):
        shutil.copy2(fig, manuscript_fig_dir / fig.name)


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    results_dir = processed_dir / "results"
    fig_dir = results_dir / "figures"
    manuscript_fig_dir = root / "manuscript" / "figures"

    panel_path = processed_dir / "player_day_panel_all_comp.csv"
    injuries_path = processed_dir / "tm_injury_episodes.csv"

    print(f"Loading panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    print(f"Loading injuries from {injuries_path} ...")
    injuries = pd.read_csv(injuries_path)

    outputs = build_clinical_outputs(panel, injuries)
    write_outputs(outputs, results_dir)
    plot_clinical_bridge(outputs, fig_dir)
    mirror_figures(fig_dir, manuscript_fig_dir)

    rates = outputs["clinical_match_hour_rates"]
    primary = rates[
        (rates["rate_scope"] == "same_day_plus_lag1")
        & (rates["group_kind"] == "clinical_risk_group")
    ]
    print("\nPrimary clinical bridge rates:")
    print(
        primary[
            [
                "group",
                "events",
                "match_hours",
                "events_per_1000_match_hours",
            ]
        ].to_string(index=False)
    )
    print("\nDone.")


if __name__ == "__main__":  # pragma: no cover
    main()
