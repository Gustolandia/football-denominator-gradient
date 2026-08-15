#!/usr/bin/env python
"""Analyze previous injury duration and subsequent match-associated proxy incidence.

This script asks a different question from the clinical bridge in step 22.
Step 22 describes the duration of the injury event itself. This step attaches
the most recent completed prior injury spell before each match row and asks
whether that prior duration is associated with the next match-proxy injury rate
per minute.

To avoid circularity, two grouping schemes are reported:

1. canonical_group: the current main ``regular`` and ``fragile`` labels;
2. frequency_only_group: a duration-independent sensitivity label that uses
   prior injury frequency but not prior maximum spell duration.

Inputs:
    data/processed/player_day_panel_all_comp.csv
    data/processed/tm_injuries_clean.csv

Outputs:
    data/processed/results/prior_injury_duration_next_risk_canonical.csv
    data/processed/results/prior_injury_duration_next_risk_frequency_only.csv
    data/processed/results/prior_injury_duration_type_mix.csv
    data/processed/results/figures/H2_prior_injury_duration_next_risk.png
    manuscript/figures/H2_prior_injury_duration_next_risk.png
"""

from __future__ import annotations

import ast
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_io import restrict_to_available_risk_set, restrict_to_fragility_risk_set


MATCH_MINUTES_COL = "all_minutes_played"
PLAYER_ID_COL = "tm_player_id"
EVENT_COL = "injury_event_matchproxy"
GROUP_ORDER = ["regular", "fragile"]
GROUP_DISPLAY_LABELS = {
    "regular": "Intermediate prior-injury-history",
    "fragile": "Higher prior-injury-history",
}
PRIOR_DURATION_BUCKETS = [
    "no prior completed injury",
    "<1 week",
    "1 week to 2 months",
    "2 months to 1 year",
    ">1 year",
    "unknown duration",
]
INJURY_TYPE_PATTERNS = [
    ("muscle/tendon", r"hamstring|muscle|calf|thigh|adductor|groin|quad|achilles|tendon"),
    ("joint/ligament", r"acl|cruciate|ligament|meniscus|knee|ankle|sprain|shoulder"),
    ("bone/fracture", r"fracture|broken|metatarsal|toe|foot|bone"),
    ("head/concussion", r"concussion|head"),
    ("illness/other medical", r"illness|virus|covid|infection|flu|sick"),
]


def parse_duration_days(value) -> float:
    """Extract numeric injury duration days from a Transfermarkt duration field."""
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
    try:
        return float(payload.get("days"))
    except (TypeError, ValueError):
        return np.nan


def duration_bucket(days: float) -> str:
    """Map previous injury duration to broad, readable buckets."""
    if pd.isna(days):
        return "unknown duration"
    if days < 7:
        return "<1 week"
    if days <= 60:
        return "1 week to 2 months"
    if days <= 365:
        return "2 months to 1 year"
    return ">1 year"


def classify_injury_type(description: str) -> str:
    """Classify a public injury description into a broad descriptive type."""
    desc = "" if pd.isna(description) else str(description).strip().lower()
    if not desc:
        return "unknown"
    for label, pattern in INJURY_TYPE_PATTERNS:
        if pd.Series([desc]).str.contains(pattern, regex=True).iloc[0]:
            return label
    if "unknown" in desc:
        return "unknown"
    return "other/unspecified"


def safe_rates(events: float, minutes: float) -> dict[str, float]:
    """Return per-minute and match-hour rates with safe zero denominators."""
    if minutes <= 0 or pd.isna(minutes):
        return {
            "events_per_10000_min": np.nan,
            "events_per_1000_match_hours": np.nan,
            "events_per_10000_min_ci_low": np.nan,
            "events_per_10000_min_ci_high": np.nan,
            "events_per_1000_match_hours_ci_low": np.nan,
            "events_per_1000_match_hours_ci_high": np.nan,
        }
    rate_10000 = float(events) / float(minutes) * 10000.0
    if events <= 0:
        low_10000 = 0.0
        high_10000 = -math.log(0.025) / float(minutes) * 10000.0
    else:
        log_margin = 1.96 / math.sqrt(float(events))
        low_10000 = rate_10000 * math.exp(-log_margin)
        high_10000 = rate_10000 * math.exp(log_margin)
    return {
        "events_per_10000_min": rate_10000,
        "events_per_1000_match_hours": rate_10000 * 6.0,
        "events_per_10000_min_ci_low": low_10000,
        "events_per_10000_min_ci_high": high_10000,
        "events_per_1000_match_hours_ci_low": low_10000 * 6.0,
        "events_per_1000_match_hours_ci_high": high_10000 * 6.0,
    }


def prepare_injury_spells(injuries: pd.DataFrame) -> pd.DataFrame:
    """Prepare completed injury spells for prior-only as-of matching."""
    identifier = (
        "injury_episode_id"
        if "injury_episode_id" in injuries.columns
        else "injury_spell_id"
    )
    required = {PLAYER_ID_COL, identifier, "start_date"}
    missing = required - set(injuries.columns)
    if missing:
        raise KeyError(f"injuries missing required columns: {sorted(missing)}")

    out = injuries.copy()
    out["injury_spell_id"] = out[identifier]
    out[PLAYER_ID_COL] = pd.to_numeric(out[PLAYER_ID_COL], errors="coerce")
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    if "end_date" in out.columns:
        out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    else:
        out["end_date"] = pd.NaT
    if "injury_desc" not in out.columns:
        out["injury_desc"] = ""

    if identifier == "injury_episode_id" and "duration_days" in out:
        out["prior_injury_duration_days"] = pd.to_numeric(
            out["duration_days"], errors="coerce"
        )
    else:
        details = out.get("durationDetails", pd.Series(np.nan, index=out.index))
        out["prior_injury_duration_days"] = details.apply(parse_duration_days)
        fallback_duration = (out["end_date"] - out["start_date"]).dt.days.clip(lower=0)
        out["prior_injury_duration_days"] = out["prior_injury_duration_days"].fillna(
            fallback_duration
        )
    fallback_end = out["start_date"] + pd.to_timedelta(
        out["prior_injury_duration_days"], unit="D"
    )
    out["prior_injury_end_date"] = out["end_date"].fillna(fallback_end)
    out["prior_injury_duration_bucket"] = out["prior_injury_duration_days"].apply(
        duration_bucket
    )
    out["prior_injury_type"] = out["injury_desc"].apply(classify_injury_type)

    keep = [
        PLAYER_ID_COL,
        "injury_spell_id",
        "prior_injury_end_date",
        "prior_injury_duration_days",
        "prior_injury_duration_bucket",
        "prior_injury_type",
    ]
    out = out.dropna(subset=[PLAYER_ID_COL, "start_date", "prior_injury_end_date"])
    out[PLAYER_ID_COL] = out[PLAYER_ID_COL].astype(int)
    return out[keep].sort_values([PLAYER_ID_COL, "prior_injury_end_date"])


def add_frequency_only_group(match_panel: pd.DataFrame) -> pd.DataFrame:
    """Add a duration-independent regular/fragile sensitivity grouping."""
    required = {
        "prior_minutes_played",
        "prior_injuries_per_10000min",
        "q3_freq",
    }
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    out = match_panel.copy()
    adequate = out["prior_minutes_played"].astype(float) >= 900.0
    high_frequency = (
        out["prior_injuries_per_10000min"].astype(float)
        >= out["q3_freq"].astype(float)
    )
    out["frequency_only_group"] = "low_exposure"
    out.loc[adequate & ~high_frequency, "frequency_only_group"] = "regular"
    out.loc[adequate & high_frequency, "frequency_only_group"] = "fragile"
    return out


def attach_most_recent_prior_injury(
    match_panel: pd.DataFrame,
    spells: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the most recent completed injury spell before each match row."""
    missing_match = {PLAYER_ID_COL, "date"} - set(match_panel.columns)
    missing_spells = {
        PLAYER_ID_COL,
        "prior_injury_end_date",
        "prior_injury_duration_days",
        "prior_injury_duration_bucket",
    } - set(spells.columns)
    if missing_match:
        raise KeyError(f"match_panel missing required columns: {sorted(missing_match)}")
    if missing_spells:
        raise KeyError(f"spells missing required columns: {sorted(missing_spells)}")

    rows = []
    for player_id, player_matches in match_panel.groupby(PLAYER_ID_COL, sort=False):
        left = player_matches.sort_values("date").copy()
        right = spells[spells[PLAYER_ID_COL] == player_id].sort_values(
            "prior_injury_end_date"
        )
        if right.empty:
            left["prior_injury_spell_id"] = np.nan
            left["prior_injury_end_date"] = pd.NaT
            left["prior_injury_duration_days"] = np.nan
            left["prior_injury_duration_bucket"] = "no prior completed injury"
            left["prior_injury_type"] = "none"
            rows.append(left)
            continue

        matched = pd.merge_asof(
            left,
            right.rename(columns={"injury_spell_id": "prior_injury_spell_id"}),
            left_on="date",
            right_on="prior_injury_end_date",
            direction="backward",
            allow_exact_matches=False,
            suffixes=("", "_prior"),
        )
        matched = matched.drop(columns=[f"{PLAYER_ID_COL}_prior"], errors="ignore")
        matched["prior_injury_duration_bucket"] = matched[
            "prior_injury_duration_bucket"
        ].fillna("no prior completed injury")
        matched["prior_injury_type"] = matched["prior_injury_type"].fillna("none")
        rows.append(matched)

    if not rows:
        return match_panel.copy()
    return pd.concat(rows, ignore_index=True)


def build_duration_rate_table(
    match_panel: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Build per-minute next-injury rates by group and prior duration bucket."""
    required = {group_col, "prior_injury_duration_bucket", MATCH_MINUTES_COL, EVENT_COL}
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    rows = []
    for group in GROUP_ORDER:
        group_df = match_panel[match_panel[group_col] == group]
        for bucket in PRIOR_DURATION_BUCKETS:
            subset = group_df[group_df["prior_injury_duration_bucket"] == bucket]
            minutes = float(subset[MATCH_MINUTES_COL].sum())
            events = float(subset[EVENT_COL].sum())
            rows.append(
                {
                    "grouping": group_col,
                    "group": group,
                    "publication_group": GROUP_DISPLAY_LABELS.get(group, group),
                    "prior_injury_duration_bucket": bucket,
                    "match_rows": int(len(subset)),
                    "match_minutes": minutes,
                    "match_hours": minutes / 60.0 if minutes > 0 else np.nan,
                    "events": int(events),
                    **safe_rates(events, minutes),
                }
            )
    return pd.DataFrame(rows)


def build_prior_injury_type_mix(match_panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize prior injury type mix by prior-duration bucket."""
    required = {
        "prior_injury_spell_id",
        "prior_injury_duration_bucket",
        "prior_injury_type",
        MATCH_MINUTES_COL,
        EVENT_COL,
    }
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    rows = []
    group_cols = ["prior_injury_duration_bucket", "prior_injury_type"]
    for keys, subset in match_panel.groupby(group_cols, dropna=False, observed=False):
        spell_ids = subset["prior_injury_spell_id"].dropna()
        rows.append(
            {
                "prior_injury_duration_bucket": keys[0],
                "prior_injury_type": keys[1],
                "unique_prior_spells": int(spell_ids.nunique()),
                "match_rows": int(len(subset)),
                "match_minutes": float(subset[MATCH_MINUTES_COL].sum()),
                "events": int(subset[EVENT_COL].sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    totals = out.groupby("prior_injury_duration_bucket")["match_rows"].transform("sum")
    out["match_row_percent_within_duration"] = np.where(
        totals > 0,
        out["match_rows"] / totals * 100.0,
        np.nan,
    )
    return out.sort_values(
        ["prior_injury_duration_bucket", "match_rows"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_prior_duration_outputs(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build canonical and duration-independent prior-duration rate tables."""
    risk = restrict_to_fragility_risk_set(panel)
    risk = restrict_to_available_risk_set(risk)
    match_panel = risk[risk[MATCH_MINUTES_COL] > 0].copy()
    match_panel = add_frequency_only_group(match_panel)

    spells = prepare_injury_spells(injuries)
    match_panel = attach_most_recent_prior_injury(match_panel, spells)

    canonical = build_duration_rate_table(
        match_panel[match_panel["fragility_group"].isin(GROUP_ORDER)].copy(),
        "fragility_group",
    )
    frequency_only = build_duration_rate_table(
        match_panel[match_panel["frequency_only_group"].isin(GROUP_ORDER)].copy(),
        "frequency_only_group",
    )
    return {
        "prior_injury_duration_next_risk_canonical": canonical,
        "prior_injury_duration_next_risk_frequency_only": frequency_only,
        "prior_injury_duration_type_mix": build_prior_injury_type_mix(match_panel),
    }


def write_outputs(outputs: dict[str, pd.DataFrame], results_dir: Path) -> None:  # pragma: no cover
    """Write prior-duration output tables."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for stem, df in outputs.items():
        df.to_csv(results_dir / f"{stem}.csv", index=False)


def plot_prior_duration(outputs: dict[str, pd.DataFrame], fig_dir: Path) -> None:  # pragma: no cover
    """Plot canonical and frequency-only prior-duration rates."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    specs = [
        (
            "prior_injury_duration_next_risk_canonical",
            "Canonical prior-injury-history strata",
            axes[0],
        ),
        (
            "prior_injury_duration_next_risk_frequency_only",
            "Frequency-only sensitivity strata",
            axes[1],
        ),
    ]

    plot_buckets = ["<1 week", "1 week to 2 months", "2 months to 1 year"]
    colors = {"regular": "#2563A7", "fragile": "#B83B2E"}
    x = np.arange(len(plot_buckets))
    width = 0.36

    legend_labels = {
        "prior_injury_duration_next_risk_canonical": {
            "regular": "Intermediate prior-injury-history",
            "fragile": "Higher prior-injury-history",
        },
        "prior_injury_duration_next_risk_frequency_only": {
            "regular": "Lower injury frequency",
            "fragile": "Higher injury frequency",
        },
    }

    for key, title, ax in specs:
        df = outputs[key]
        for offset, group in [(-width / 2, "regular"), (width / 2, "fragile")]:
            sub = (
                df[
                    (df["group"] == group)
                    & (df["prior_injury_duration_bucket"].isin(plot_buckets))
                ]
                .set_index("prior_injury_duration_bucket")
                .reindex(plot_buckets)
            )
            y = sub["events_per_1000_match_hours"].fillna(0.0)
            low = sub["events_per_1000_match_hours_ci_low"].fillna(y)
            high = sub["events_per_1000_match_hours_ci_high"].fillna(y)
            yerr = np.vstack([(y - low).clip(lower=0.0), (high - y).clip(lower=0.0)])
            ax.bar(
                x + offset,
                y,
                width,
                yerr=yerr,
                capsize=3,
                label=legend_labels[key][group],
                color=colors[group],
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_buckets, rotation=25, ha="right")
        ax.set_xlabel("Most recent completed prior injury duration")
        ax.legend(frameon=False)

    axes[0].set_ylabel("Subsequent proxy events per 1,000 match hours")
    fig.tight_layout()
    fig.savefig(fig_dir / "H2_prior_injury_duration_next_risk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mirror_figures(results_fig_dir: Path, manuscript_fig_dir: Path) -> None:  # pragma: no cover
    """Copy prior-duration figures into the manuscript figure directory."""
    manuscript_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig in results_fig_dir.glob("H2_*.png"):
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

    outputs = build_prior_duration_outputs(panel, injuries)
    write_outputs(outputs, results_dir)
    plot_prior_duration(outputs, fig_dir)
    mirror_figures(fig_dir, manuscript_fig_dir)

    print("\nCanonical prior-duration rates:")
    print(
        outputs["prior_injury_duration_next_risk_canonical"][
            [
                "group",
                "prior_injury_duration_bucket",
                "events",
                "match_hours",
                "events_per_1000_match_hours",
            ]
        ].to_string(index=False)
    )
    print("\nDone.")


if __name__ == "__main__":  # pragma: no cover
    main()
