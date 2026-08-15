#!/usr/bin/env python
"""Plot the four reviewer-aligned JSAMS manuscript displays.

Run after scripts 34 and 36. The main displays now foreground the cohort and
denominator audit, the additive same-day per-appearance estimand, temporal and
within-player support, and independent outcome/selection sensitivities.
National-exposure validation and the original history-interaction curves remain
supplementary analyses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


BLUE = "#1F5A8A"
TEAL = "#16817A"
GOLD = "#C58A12"
RED = "#B6463A"
INK = "#17212B"
MUTED = "#5A6875"
GRID = "#D9E0E5"

EXPOSURE_LABELS = {
    "prior_minutes_3d": "3-day minutes (per 90)",
    "prior_minutes_5d": "5-day minutes (per 90)",
    "prior_minutes_7d": "7-day minutes (per 90)",
    "prior_minutes_10d": "10-day minutes (per 90)",
    "prior_minutes_14d": "14-day minutes (per 90)",
    "prior_matches_7d": "7-day match count (per match)",
    "recovery_interval": "Recovery: 0-3 vs 6-7 days",
}
AUDIT_LABELS = {
    "muscle_tendon_nonsevere": "Muscle/tendon, <28 days",
    "other_nonsevere": "Other, <28 days",
    "reported_absence_ge28d": "Reported absence >=28 days",
}
EVENT_LABELS = {
    "injury_event_matchproxy_same_day": "Same day",
    "injury_event_matchproxy_lag1": "Lag 1 only",
    "injury_event_matchproxy": "Same day + lag 1",
}
DENOMINATOR_LABELS = {
    "per_appearance": "Per appearance",
    "fixed_90": "Fixed 90 min",
    "observed_minutes": "Recorded minutes",
}


def _required(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def _save(fig: plt.Figure, output_path: Path) -> None:
    """Write the figure in both the working and the journal-accepted format.

    PNG is what the LaTeX sources include and referees see; the journal's
    artwork instructions accept EPS/PDF vector drawings but not PNG at the
    revision stage, so a vector PDF twin is written beside every PNG and the
    submission package never needs figures re-exported by hand.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


# The columns the cross-league figure actually draws. The digest below covers
# these and nothing else, so reordering rows or editing an unplotted column
# does not raise a false alarm.
GRADIENT_FIGURE_COLUMNS = (
    "league",
    "gamma_pooled",
    "gamma_pooled_ci_low",
    "gamma_pooled_ci_high",
    "gamma_within_starters",
    "gamma_within_starters_ci_low",
    "gamma_within_starters_ci_high",
)


def gradient_source_digest(gradients: pd.DataFrame) -> str:
    """Fingerprint the table the cross-league figure is drawn from.

    Currency used to be checked by comparing file modification times, which
    says nothing once the repository has been through an archive: packaging
    stamps every source file with one timestamp, so a reader unpacking the
    deposit saw a figure that looked older than its own table. A digest of the
    plotted values travels with the files and answers the sharper question
    anyway --- whether the numbers changed, not whether a clock moved.
    """
    _required(gradients, list(GRADIENT_FIGURE_COLUMNS), "gradient figure digest")
    frame = gradients.loc[:, list(GRADIENT_FIGURE_COLUMNS)]
    canonical = frame.to_csv(index=False, lineterminator="\n", float_format="%.10g")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def figure_manifest(gradients: pd.DataFrame) -> pd.DataFrame:
    """Record what the figures were drawn from, so they can be gated.

    Figures were the last artifact class nothing checked: a label could go
    stale against the table it was drawn from and only a human reading the
    image would notice --- which is exactly how one league name outlived its
    own table. The manifest deposits the drawn league labels and the formats
    written, and the manuscript gates compare it to the current tables.
    """
    _required(gradients, ["league"], "figure manifest gradients")
    digest = gradient_source_digest(gradients)
    rows = [
        {
            "figure": name,
            "formats": "png+pdf",
            "league_labels": labels,
            "source_digest": source_digest,
        }
        for name, labels, source_digest in (
            ("J1_jsams_cohort_measurement", "", ""),
            ("J2_jsams_denominator_gradient",
             "|".join(str(label) for label in gradients["league"]), digest),
            ("J3_jsams_within_player_lineup_coverage", "", ""),
            ("J4_jsams_context_support", "", ""),
            ("J5_jsams_negative_control_exposure", "", ""),
            ("J6_jsams_primary_robustness", "", ""),
        )
    ]
    out = pd.DataFrame(rows)
    out["interpretation"] = (
        "what script 35 drew, deposited so the gates can compare the figures' "
        "labels, formats and plotted values against the tables they came from; "
        "a figure drawn from numbers that have since changed now fails a test "
        "instead of waiting for a reader"
    )
    return out


def _forest_point(
    axis: plt.Axes,
    estimate: float,
    low: float,
    high: float,
    y: float,
    color: str,
    marker: str = "o",
    fill: bool = True,
) -> None:
    axis.errorbar(
        estimate,
        y,
        xerr=[[estimate - low], [high - estimate]],
        fmt=marker,
        color=color,
        markerfacecolor=color if fill else "white",
        markeredgecolor=color,
        ecolor=color,
        capsize=3,
        markersize=7,
        linewidth=1.5,
    )


ROLE_DISPLAY = {
    "starting_lineup": "Starters",
    "substitute_list": "Substitutes",
    "lineup_unavailable_or_other": "Lineup unknown",
}
DENOMINATOR_DISPLAY = {
    "fixed_90": "Fixed 90 minutes",
    "observed_minutes": "Recorded minutes",
}


def _quantile_bar(
    axis: plt.Axes,
    row: pd.Series,
    y: float,
    colour: str,
) -> None:
    """Draw one p10-p90 whisker with a p25-p75 box and a median tick."""
    axis.plot(
        [float(row["p10_minutes"]), float(row["p90_minutes"])], [y, y],
        color=colour, linewidth=1.4, solid_capstyle="butt", zorder=2,
    )
    axis.add_patch(
        Rectangle(
            (float(row["p25_minutes"]), y - 0.16),
            float(row["p75_minutes"]) - float(row["p25_minutes"]),
            0.32, facecolor=colour, edgecolor=colour, alpha=0.35, zorder=3,
        )
    )
    axis.plot(
        [float(row["median_minutes"])] * 2, [y - 0.19, y + 0.19],
        color=colour, linewidth=2.6, zorder=4,
    )


def plot_cohort_and_denominator(
    flow: pd.DataFrame,
    minute_distribution: pd.DataFrame,
    denominator_roles: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the cohort, the truncation itself, and what truncation explains.

    Panel B shows the truncation directly as a distribution rather than as a
    difference in means, because only a distribution distinguishes a whole
    cohort shifting slightly from a subset leaving the field early. Panel C
    then asks whether that truncation is what makes per-minute and
    per-appearance answers differ; holding squad role fixed shows it is not.
    """
    _required(
        flow,
        [
            "stage_order",
            "stage",
            "records",
            "players",
            "same_day_events",
            "lag1_events",
            "combined_proxy_events",
        ],
        "flow",
    )
    _required(
        minute_distribution,
        [
            "lineup_role",
            "event_status",
            "p10_minutes",
            "p25_minutes",
            "median_minutes",
            "p75_minutes",
            "p90_minutes",
        ],
        "minute distribution",
    )
    _required(
        denominator_roles,
        [
            "lineup_role", "denominator", "estimate", "ci_low", "ci_high",
            "estimable", "n_events",
        ],
        "denominator by lineup role",
    )
    selected_stages = [1, 2, 3, 6, 8]
    selected = flow[flow["stage_order"].isin(selected_stages)].sort_values("stage_order")
    if len(selected) != len(selected_stages):
        raise ValueError("flow is missing a required display stage")

    fig, axes = plt.subplots(
        1, 3, figsize=(19.8, 6.9), gridspec_kw={"width_ratios": [1.12, 1.16, 1.0]}
    )
    axis = axes[0]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    by_stage = selected.set_index("stage_order")
    # A single vertical cascade with the exclusion branching right, so the
    # reader never has to follow an arrow backwards.
    main_stages = [
        f"{int(by_stage.loc[1, 'records']):,} public injury/absence reports",
        f"{int(by_stage.loc[2, 'records']):,} reconciled non-overlapping episodes",
        f"{int(by_stage.loc[3, 'records']):,} player-match appearances\n"
        f"among {int(by_stage.loc[3, 'players']):,} players",
        f"{int(by_stage.loc[8, 'players']):,} players | "
        f"{int(by_stage.loc[8, 'records']):,} eligible appearances\n"
        f"{int(by_stage.loc[8, 'same_day_events']):,} same-day | "
        f"{int(by_stage.loc[8, 'lag1_events']):,} lag-1 | "
        f"{int(by_stage.loc[8, 'combined_proxy_events']):,} combined",
    ]
    top, step, height = 0.80, 0.215, 0.125
    centre = 0.46
    for index, label in enumerate(main_stages):
        y = top - index * step
        axis.add_patch(
            FancyBboxPatch(
                (centre - 0.45, y - height / 2), 0.90, height,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                edgecolor=BLUE, facecolor="#F3F7FA", linewidth=1.4,
            )
        )
        axis.text(
            centre, y, label, ha="center", va="center",
            fontsize=9.6, color=INK, fontweight="semibold",
        )
        if index < len(main_stages) - 1:
            axis.annotate(
                "", xy=(centre, y - step + height / 2), xytext=(centre, y - height / 2),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.3},
            )
    axis.text(
        centre, top - (len(main_stages) - 1) * step - height / 2 - 0.055,
        f"excluded before the final stage: {int(by_stage.loc[6, 'records']):,} appearances\n"
        "below 900 prior club minutes",
        ha="center", va="top", fontsize=8.9, color=MUTED, style="italic",
    )
    axis.set_title(
        "A. From public reports to eligible appearances",
        loc="left", fontsize=12.5, fontweight="bold",
    )

    distribution = minute_distribution.set_index(["lineup_role", "event_status"])
    positions = []
    labels = []
    slot = 0.0
    for role in ("starting_lineup", "substitute_list"):
        for status, colour in (
            ("no_same_day_report", BLUE),
            ("same_day_spell_start", RED),
        ):
            if (role, status) not in distribution.index:
                continue
            _quantile_bar(axes[1], distribution.loc[(role, status)], slot, colour)
            positions.append(slot)
            labels.append(
                f"{ROLE_DISPLAY[role]}\n"
                + ("no report" if status == "no_same_day_report" else "same-day report")
            )
            slot -= 1.0
        slot -= 0.55
    axes[1].set_yticks(positions, labels, fontsize=9.4)
    axes[1].set_xlim(0, 95)
    axes[1].set_xlabel("Recorded minutes on the appearance (p10, p25, median, p75, p90)")
    axes[1].grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].set_title(
        "B. Truncation is visible among starters and absent among substitutes",
        loc="left", fontsize=12.5, fontweight="bold",
    )

    estimable = denominator_roles[denominator_roles["estimable"].astype(bool)]
    indexed = estimable.set_index(["lineup_role", "denominator"])
    rows = []
    # Every stratum the remedy is claimed over appears, including the rows with
    # no lineup status, where stratifying is impossible and the attenuation is
    # largest of all.
    for role in ("all", "starting_lineup", "substitute_list", "lineup_unavailable_or_other"):
        for denominator in ("fixed_90", "observed_minutes"):
            if (role, denominator) in indexed.index:
                rows.append((role, denominator, indexed.loc[(role, denominator)]))
    y_positions = np.arange(len(rows))[::-1].astype(float)
    tick_positions = []
    tick_labels = []
    for index, (role, denominator, row) in enumerate(rows):
        colour = MUTED if denominator == "fixed_90" else RED
        _forest_point(
            axes[2], float(row["estimate"]), float(row["ci_low"]),
            float(row["ci_high"]), float(y_positions[index]), colour,
            fill=denominator == "observed_minutes",
        )
        tick_positions.append(float(y_positions[index]))
        role_label = "All appearances" if role == "all" else ROLE_DISPLAY[role]
        # The event count belongs beside the estimate: the substitute interval
        # is wide because that stratum carries very few events, and a reader
        # cannot judge the comparison without seeing that.
        if denominator == "fixed_90":
            role_label = f"{role_label} ({int(row['n_events']):,} events)"
        tick_labels.append(f"{role_label}\n{DENOMINATOR_DISPLAY[denominator]}")
    axes[2].axvline(1.0, color=MUTED, linestyle="--", linewidth=1.1)
    axes[2].set_yticks(tick_positions, tick_labels, fontsize=9.0)
    # A log scale keeps the wide substitute intervals from compressing the
    # starter comparison, which is the convergence this panel exists to show.
    axes[2].set_xscale("log")
    ticks = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels([f"{tick:g}" for tick in ticks])
    axes[2].minorticks_off()
    axes[2].set_xlabel("Estimate per 90 prior minutes (95% CI, log scale)")
    axes[2].grid(axis="x", color=GRID, linewidth=0.8)
    axes[2].set_title(
        # "Closes" would outrun the interval: the within-starter attenuation is
        # 0.012 (0.009-0.015) and excludes zero.
        "C. The denominator gap nearly closes within starters, and nowhere else",
        loc="left", fontsize=12.5, fontweight="bold",
    )
    fig.suptitle(
        "The recorded-minute denominator: what is truncated, and what that truncation explains",
        x=0.035, ha="left", fontsize=15.5, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, output_path)


def plot_primary_and_multiverse(
    curves: pd.DataFrame,
    metric_summary: pd.DataFrame,
    output_path: Path,
    support: pd.DataFrame | None = None,
) -> None:
    """Plot one additive curve and the complete same-day exposure-metric family.

    When exposure support is supplied, a band under panel A shows the share of
    appearances in each exposure range, so readers can see which part of the
    fitted curve rests on data and which is extrapolation into a sparse tail.
    """
    _required(
        curves,
        [
            "model_id", "prior_minutes_7d", "estimate_per_1000_appearances",
            "pointwise_ci_low", "pointwise_ci_high", "simultaneous_ci_low",
            "simultaneous_ci_high",
        ],
        "additive curves",
    )
    _required(
        metric_summary,
        [
            "exposure_id", "estimate", "ci_low", "ci_high",
            "holm_p_value_63_model_family", "reject_holm_0_05",
        ],
        "exposure metric summary",
    )
    linear = curves[curves["model_id"].eq("additive_linear") & curves["prior_minutes_7d"].le(180)]
    spline = curves[curves["model_id"].eq("additive_spline") & curves["prior_minutes_7d"].le(180)]
    if linear.empty or spline.empty:
        raise ValueError("additive linear or spline curves are empty")
    forest = metric_summary[metric_summary["exposure_id"].isin(EXPOSURE_LABELS)].copy()
    if len(forest) != len(EXPOSURE_LABELS):
        raise ValueError("exposure metric forest is incomplete")
    forest = forest.set_index("exposure_id").loc[list(EXPOSURE_LABELS)].reset_index()

    fig, axes = plt.subplots(
        1, 2, figsize=(14.8, 6.6), gridspec_kw={"width_ratios": [1.12, 1]}
    )
    x = linear["prior_minutes_7d"].to_numpy(dtype=float)
    axes[0].plot(
        x, linear["estimate_per_1000_appearances"], color=BLUE, linewidth=2.6,
        label="Additive linear model",
    )
    axes[0].fill_between(
        x,
        linear["pointwise_ci_low"].to_numpy(dtype=float),
        linear["pointwise_ci_high"].to_numpy(dtype=float),
        color=BLUE, alpha=0.16, label="Linear pointwise 95% CI",
    )
    sx = spline["prior_minutes_7d"].to_numpy(dtype=float)
    axes[0].plot(
        sx, spline["estimate_per_1000_appearances"], color=RED,
        linestyle="--", linewidth=2.2, label="Spline sensitivity",
    )
    axes[0].fill_between(
        sx,
        spline["simultaneous_ci_low"].to_numpy(dtype=float),
        spline["simultaneous_ci_high"].to_numpy(dtype=float),
        color=GOLD, alpha=0.18, label="Spline simultaneous 95% band",
    )
    axes[0].set_xlim(0, 180)
    # A rare absolute risk must be read against zero, or the gradient looks
    # steeper than it is.
    axes[0].set_ylim(0, None)
    axes[0].set_xlabel("Club-match minutes in the previous 7 days")
    axes[0].set_ylabel("Same-day spell starts per 1,000 appearances")
    axes[0].grid(color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    axes[0].set_title(
        "A. One history-adjusted exposure-response curve",
        loc="left", fontsize=13, fontweight="bold",
    )
    if support is not None and not support.empty:
        _required(
            support,
            ["exposure_band", "share_of_appearances"],
            "exposure support",
        )
        spans = {
            "0": (0.0, 6.0),
            "1-45": (6.0, 45.0),
            "46-90": (45.0, 90.0),
            "91-135": (90.0, 135.0),
            "136-180": (135.0, 180.0),
        }
        rug = axes[0].inset_axes([0.0, -0.30, 1.0, 0.13])
        for _, row in support.iterrows():
            band = str(row["exposure_band"])
            if band not in spans:
                continue
            left, right = spans[band]
            share = float(row["share_of_appearances"])
            rug.barh(
                0, right - left, left=left, height=1.0,
                color=BLUE, alpha=min(0.85, 0.12 + 2.0 * share),
                edgecolor="white", linewidth=1.0,
            )
            narrow = (right - left) < 20.0
            rug.text(
                (left + right) / 2.0,
                0.85 if narrow else 0.0,
                f"{100.0 * share:.0f}%",
                ha="center", va="center", fontsize=9,
                color=INK if narrow or share <= 0.2 else "white",
            )
        rug.set_xlim(0, 180)
        rug.set_ylim(-0.6, 1.3)
        rug.set_yticks([])
        rug.set_xlabel("Share of appearances by exposure range", fontsize=9.5)
        rug.tick_params(labelsize=9)
        for spine in rug.spines.values():
            spine.set_visible(False)

    y = np.arange(len(forest))[::-1]
    for index, row in forest.iterrows():
        rejected = bool(row["reject_holm_0_05"])
        _forest_point(
            axes[1], float(row["estimate"]), float(row["ci_low"]),
            float(row["ci_high"]), float(y[index]), RED if rejected else BLUE,
            fill=rejected,
        )
    # Holm values ride on the y-axis labels; printed beside the intervals they
    # collided with the longer bars at column width.
    metric_ticks = []
    for index, row in forest.iterrows():
        q_value = float(row["holm_p_value_63_model_family"])
        formatted = f"{q_value:.3f}" if q_value < 1 else "1.000"
        metric_ticks.append(f"{EXPOSURE_LABELS[row['exposure_id']]}\nHolm p={formatted}")
    axes[1].axvline(1.0, color=MUTED, linestyle="--", linewidth=1.1)
    axes[1].set_yticks(y, metric_ticks, fontsize=9.2)
    axes[1].set_xlim(0.75, 1.85)
    axes[1].set_xlabel("Adjusted odds ratio (95% CI)")
    axes[1].grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].set_title(
        "B. Seven exposure metrics within one 63-model Holm family",
        loc="left", fontsize=13, fontweight="bold",
    )
    fig.suptitle(
        "Recent match exposure and same-day public reports",
        x=0.055, ha="left", fontsize=16, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output_path)


CONFOUNDING_LABELS = {
    "reference": "Reference (history + calendar)",
    "age_position_adjusted": "+ age and position",
    "club_season_adjusted": "+ club-season",
    "competition_adjusted": "+ competition",
    "season_adjusted": "+ season",
    "fully_adjusted": "All measured covariates",
    "premier_league_only": "Premier League matches only",
    "player_match_two_way_cluster": "Player + match clustering",
    "club_season_cluster": "Club-season clustering",
}
CLUB_CONGESTION_LABELS = {
    "plus_club_schedule": "+ club fixture schedule",
    "plus_club_schedule_and_club_season": "+ club schedule and club-season",
}
PLACEBO_LABELS = {
    "recent seven-day minutes": "Recent 7-day minutes",
    "placebo window: minutes 31-37 days earlier": "Placebo: minutes 31-37 days earlier",
    "recent window with the placebo window held constant": "Recent, placebo held constant",
    "placebo window with the recent window held constant": "Placebo, recent held constant",
}


def plot_robustness_panels(
    temporal: pd.DataFrame,
    metric_summary: pd.DataFrame,
    output_path: Path,
    confounding: pd.DataFrame,
    club_congestion: pd.DataFrame,
) -> None:
    """Plot the two sensitivity analyses that most qualify the estimate.

    Panel A asks whether measured covariates, club fixture schedule or a
    different clustering unit explain the association. Panel B asks whether it
    holds in fixed periods. The negative-control exposure moved to the
    Supplement with the outcome-behaviour results it belongs to.
    """
    _required(
        temporal,
        ["temporal_block", "estimate", "ci_low", "ci_high", "heterogeneity_p_value"],
        "temporal stability",
    )
    _required(metric_summary, ["exposure_id", "estimate", "ci_low", "ci_high"], "metric summary")
    _required(confounding, ["model_id", "estimate", "ci_low", "ci_high"], "confounding sensitivity")
    _required(
        club_congestion, ["model_id", "estimate", "ci_low", "ci_high"], "club congestion sensitivity"
    )
    overall = metric_summary[metric_summary["exposure_id"].eq("prior_minutes_7d")]
    if len(overall) != 1 or len(temporal) != 3:
        raise ValueError("temporal estimates are incomplete")

    fig, axes = plt.subplots(
        1, 2, figsize=(15.2, 7.6), gridspec_kw={"width_ratios": [1.34, 1.0]}
    )

    covariates = confounding.set_index("model_id")
    order = [item for item in CONFOUNDING_LABELS if item in covariates.index]
    club = club_congestion.set_index("model_id")
    club_order = [item for item in CLUB_CONGESTION_LABELS if item in club.index]
    labels = [CONFOUNDING_LABELS[item] for item in order] + [
        CLUB_CONGESTION_LABELS[item] for item in club_order
    ]
    estimates = list(covariates.loc[order, "estimate"]) + list(club.loc[club_order, "estimate"])
    lows = list(covariates.loc[order, "ci_low"]) + list(club.loc[club_order, "ci_low"])
    highs = list(covariates.loc[order, "ci_high"]) + list(club.loc[club_order, "ci_high"])
    yc = np.arange(len(labels))[::-1]
    for index, label in enumerate(labels):
        is_reference = index == 0
        _forest_point(
            axes[0], float(estimates[index]), float(lows[index]), float(highs[index]),
            float(yc[index]), RED if is_reference else BLUE, fill=is_reference,
        )
    axes[0].axvline(1.0, color=MUTED, linestyle="--", linewidth=1.1)
    axes[0].set_yticks(yc, labels)
    axes[0].set_xlabel("Odds ratio per 90 prior minutes (95% CI)")
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_title(
        "A. Measured covariates, club schedule and clustering\nEach covariate added singly unless stated",
        loc="left", fontsize=12.5, fontweight="bold",
    )

    forest = pd.concat(
        [
            overall.assign(display_label="All seasons")[["display_label", "estimate", "ci_low", "ci_high"]],
            temporal.rename(columns={"temporal_block": "display_label"})[
                ["display_label", "estimate", "ci_low", "ci_high"]
            ],
        ],
        ignore_index=True,
    )
    y = np.arange(len(forest))[::-1]
    for index, row in forest.iterrows():
        _forest_point(
            axes[1], float(row["estimate"]), float(row["ci_low"]),
            float(row["ci_high"]), float(y[index]), RED if index == 0 else BLUE,
        )
    heterogeneity = float(temporal["heterogeneity_p_value"].iloc[0])
    axes[1].axvline(1.0, color=MUTED, linestyle="--", linewidth=1.1)
    axes[1].set_yticks(y, forest["display_label"])
    axes[1].set_xlabel("Odds ratio per 90 prior minutes (95% CI)")
    axes[1].grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].set_title(
        f"B. Three temporal blocks (heterogeneity p={heterogeneity:.3f})",
        loc="left", fontsize=12.5, fontweight="bold",
    )

    fig.suptitle(
        "Does the association survive measured confounding and time?",
        x=0.045, ha="left", fontsize=16, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, output_path)


def plot_negative_control_exposure(placebo: pd.DataFrame, output_path: Path) -> None:
    """Plot the placebo exposure window alone and mutually adjusted.

    Minutes played five weeks earlier share a player's reporting profile but
    should contribute far less to an injury reported today. If the placebo
    window predicts as strongly as the recent one, the association reflects
    who gets reported rather than recent load.
    """
    _required(placebo, ["description", "estimate", "ci_low", "ci_high"], "placebo window analysis")
    rows = placebo[placebo["description"].isin(PLACEBO_LABELS)].copy()
    if rows.empty:
        raise ValueError("placebo window analysis has no displayable rows")
    rows["display_label"] = rows["description"].map(PLACEBO_LABELS)
    fig, axis = plt.subplots(figsize=(9.6, 5.0))
    y = np.arange(len(rows))[::-1]
    for index, row in rows.reset_index(drop=True).iterrows():
        recent = "Recent" in str(row["display_label"])
        _forest_point(
            axis, float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"]),
            float(y[index]), BLUE if recent else GOLD, fill=recent,
        )
    axis.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.1)
    axis.set_yticks(y, rows["display_label"], fontsize=9.6)
    axis.set_xlabel("Odds ratio per 90 minutes (95% CI)")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_title(
        "Negative control: a window that cannot plausibly cause today's event",
        loc="left", fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_attribution_selection_and_timing(
    audit: pd.DataFrame,
    selection: pd.DataFrame,
    multiverse: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot independent attribution, bounded selection and denominator checks."""
    _required(
        audit,
        [
            "audit_dimension", "audit_stratum", "n_sampled", "n_resolved", "n_confirmed",
            "confirmed_proportion", "ci_low", "ci_high",
        ],
        "outcome audit",
    )
    _required(
        selection,
        ["model_id", "estimate", "ci_low", "ci_high", "n_selected_appearances"],
        "selection estimates",
    )
    _required(
        multiverse,
        [
            "exposure_id", "event_col", "denominator", "estimate", "ci_low",
            "ci_high", "reject_holm_0_05",
        ],
        "exposure multiverse",
    )
    audited = audit[
        audit["audit_dimension"].eq("date_attribution")
        & audit["audit_stratum"].isin(AUDIT_LABELS)
    ].copy()
    if len(audited) != len(AUDIT_LABELS):
        raise ValueError("outcome audit is incomplete")
    audited = audited.set_index("audit_stratum").loc[list(AUDIT_LABELS)].reset_index()
    selected = selection[selection["model_id"].isin(("unweighted", "inverse_selection_weighted"))]
    if len(selected) != 2:
        raise ValueError("selection estimates are incomplete")
    sensitivity = multiverse[multiverse["exposure_id"].eq("prior_minutes_7d")].copy()
    if len(sensitivity) != 9:
        raise ValueError("seven-day timing/denominator family is incomplete")

    fig, axes = plt.subplots(1, 3, figsize=(19.6, 7.6), gridspec_kw={"width_ratios": [1.05, 0.9, 1.25]})
    y = np.arange(len(audited))[::-1]
    # Counts belong in the axis labels rather than floating inside the panel,
    # where they collided with the intervals at print size.
    audit_ticks = []
    for index, row in audited.iterrows():
        _forest_point(
            axes[0], float(row["confirmed_proportion"]), float(row["ci_low"]),
            float(row["ci_high"]), float(y[index]), TEAL,
        )
        unresolved = int(row["n_sampled"]) - int(row["n_resolved"])
        audit_ticks.append(
            f"{AUDIT_LABELS[row['audit_stratum']]}\n"
            f"{int(row['n_confirmed'])}/{int(row['n_resolved'])} resolved, {unresolved} unresolved"
        )
    axes[0].set_yticks(y, audit_ticks, fontsize=9.0)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Exact-match attribution confirmed (Wilson 95% CI)")
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_title("A. Independent source audit", loc="left", fontsize=12.5, fontweight="bold")

    selected = selected.set_index("model_id").loc[["unweighted", "inverse_selection_weighted"]].reset_index()
    y2 = np.arange(2)[::-1]
    for index, row in selected.iterrows():
        _forest_point(
            axes[1], float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"]),
            float(y2[index]), RED if row["model_id"] == "inverse_selection_weighted" else BLUE,
        )
    axes[1].axvline(1.0, color=MUTED, linestyle="--", linewidth=1.1)
    axes[1].set_yticks(y2, ["Unweighted", "Inverse-selection weighted"])
    axes[1].set_xlabel("Odds ratio per 90 prior minutes (95% CI)")
    axes[1].grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].set_title("B. Bounded appearance weighting", loc="left", fontsize=12.5, fontweight="bold")

    sensitivity["event_order"] = sensitivity["event_col"].map(
        {name: index for index, name in enumerate(EVENT_LABELS)}
    )
    sensitivity["denominator_order"] = sensitivity["denominator"].map(
        {name: index for index, name in enumerate(DENOMINATOR_LABELS)}
    )
    sensitivity = sensitivity.sort_values(["event_order", "denominator_order"]).reset_index(drop=True)
    labels = [
        f"{EVENT_LABELS[row.event_col]} | {DENOMINATOR_LABELS[row.denominator]}"
        for row in sensitivity.itertuples()
    ]
    y3 = np.arange(len(sensitivity))[::-1]
    for index, row in sensitivity.iterrows():
        rejected = bool(row["reject_holm_0_05"])
        _forest_point(
            axes[2], float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"]),
            float(y3[index]), RED if rejected else BLUE, fill=rejected,
        )
    axes[2].axvline(1.0, color=MUTED, linestyle="--", linewidth=1.1)
    axes[2].set_yticks(y3, labels)
    axes[2].set_xlabel("Ratio per 90 prior minutes (95% CI)")
    axes[2].grid(axis="x", color=GRID, linewidth=0.8)
    axes[2].set_title(
        "C. Outcome timing and denominator\nFilled points survive the 63-model Holm family",
        loc="left", fontsize=12.5, fontweight="bold",
    )
    fig.suptitle(
        "What the public-data association survives - and what remains uncertain",
        x=0.045, ha="left", fontsize=16, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, output_path)


def plot_denominator_gradient(
    gradients: pd.DataFrame,
    output_path: Path,
    threshold: float = 0.05,
) -> None:
    """Plot the denominator gradient in every league, pooled and within starters.

    The panel exists to answer one question a reader has about a single-league
    finding: does this happen anywhere else. Both estimates carry intervals, so
    the separation between them is visible rather than asserted, and the
    negligible-gradient threshold is drawn so the decision rule can be read off
    the figure.
    """
    _required(
        gradients,
        [
            "league", "gamma_pooled", "gamma_pooled_ci_low", "gamma_pooled_ci_high",
            "gamma_within_starters", "gamma_within_starters_ci_low",
            "gamma_within_starters_ci_high", "iqr_recorded_minutes",
            "iqr_starter_minutes",
        ],
        "denominator gradient",
    )
    frame = gradients.iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.62 * len(frame) + 3.0),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    positions = np.arange(len(frame), dtype=float)
    for index, row in frame.iterrows():
        _forest_point(
            axes[0], float(row["gamma_pooled"]), float(row["gamma_pooled_ci_low"]),
            float(row["gamma_pooled_ci_high"]), positions[index] + 0.17, RED,
        )
        _forest_point(
            axes[0], float(row["gamma_within_starters"]),
            float(row["gamma_within_starters_ci_low"]),
            float(row["gamma_within_starters_ci_high"]), positions[index] - 0.17,
            BLUE, fill=False,
        )
    axes[0].axvline(threshold, color=MUTED, linestyle="--", linewidth=1.1)
    axes[0].annotate(
        f"negligible gradient ({threshold:g})",
        xy=(threshold, -0.72), xytext=(threshold + 0.04, -0.72),
        color=MUTED, fontsize=8.5, va="center",
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        annotation_clip=False,
    )
    axes[0].set_yticks(positions, list(frame["league"]), fontsize=9.5)
    axes[0].set_ylim(-1.1, len(frame) - 0.4)
    axes[0].set_xlim(-0.03, max(0.72, float(frame["gamma_pooled_ci_high"].max()) + 0.06))
    axes[0].set_xlabel("Denominator gradient $\\gamma$ (95% CI)")
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color=RED, linestyle="none",
                   label="All appearances"),
            Line2D([0], [0], marker="o", color=BLUE, linestyle="none",
                   markerfacecolor="white", label="Starters only"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2,
        frameon=False, fontsize=9.5,
    )
    axes[0].set_title(
        "A. Playing time tracks the exposure everywhere,\n    and stops doing so within starters",
        loc="left", fontsize=11.5, fontweight="bold",
    )

    height = 0.34
    axes[1].barh(positions + 0.17, frame["iqr_recorded_minutes"], height=height,
                 color=RED, alpha=0.85)
    axes[1].barh(positions - 0.17, frame["iqr_starter_minutes"], height=height,
                 color=BLUE, alpha=0.85)
    for index, row in frame.iterrows():
        axes[1].text(float(row["iqr_recorded_minutes"]) + 1.0, positions[index] + 0.17,
                     f"{float(row['iqr_recorded_minutes']):.0f}", va="center",
                     fontsize=8.5, color=INK)
        axes[1].text(float(row["iqr_starter_minutes"]) + 1.0, positions[index] - 0.17,
                     f"{float(row['iqr_starter_minutes']):.0f}", va="center",
                     fontsize=8.5, color=INK)
    axes[1].set_yticks(positions, [""] * len(frame))
    axes[1].set_ylim(-1.1, len(frame) - 0.4)
    axes[1].set_xlim(0, float(frame["iqr_recorded_minutes"].max()) * 1.16)
    axes[1].set_xlabel("Interquartile range of recorded minutes")
    axes[1].grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].set_title(
        "B. Because starters have almost no\n    minutes to vary over",
        loc="left", fontsize=11.5, fontweight="bold",
    )

    fig.suptitle(
        "The denominator gradient across eight European domestic leagues",
        x=0.035, ha="left", fontsize=15.0, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    _save(fig, output_path)


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    results = root / "data" / "processed" / "results"
    figures = root / "manuscript" / "figures"
    plot_cohort_and_denominator(
        pd.read_csv(results / "jsams_cohort_flow.csv"),
        pd.read_csv(results / "jsams_revised_recorded_minute_distribution.csv"),
        pd.read_csv(results / "jsams_revised_denominator_by_lineup_role.csv"),
        figures / "J1_jsams_cohort_measurement.png",
    )
    gradients = pd.read_csv(
        results / "jsams_revised_denominator_gradient_by_league.csv"
    )
    plot_denominator_gradient(
        gradients,
        figures / "J2_jsams_denominator_gradient.png",
    )
    figure_manifest(gradients).to_csv(
        results / "jsams_revised_figure_manifest.csv", index=False
    )
    plot_primary_and_multiverse(
        pd.read_csv(results / "jsams_revised_additive_curves.csv"),
        pd.read_csv(results / "jsams_revised_exposure_metric_summary.csv"),
        figures / "J6_jsams_primary_robustness.png",
        pd.read_csv(results / "jsams_revised_exposure_support.csv"),
    )
    plot_robustness_panels(
        pd.read_csv(results / "jsams_revised_temporal_stability.csv"),
        pd.read_csv(results / "jsams_revised_exposure_metric_summary.csv"),
        figures / "J3_jsams_within_player_lineup_coverage.png",
        pd.read_csv(results / "jsams_revised_confounding_sensitivity.csv"),
        pd.read_csv(results / "jsams_revised_club_congestion_sensitivity.csv"),
    )
    plot_negative_control_exposure(
        pd.read_csv(results / "jsams_revised_placebo_window_analysis.csv"),
        figures / "J5_jsams_negative_control_exposure.png",
    )
    plot_attribution_selection_and_timing(
        pd.read_csv(results / "jsams_revised_outcome_audit_summary.csv"),
        pd.read_csv(results / "jsams_revised_appearance_selection_estimates.csv"),
        pd.read_csv(results / "jsams_revised_exposure_multiverse.csv"),
        figures / "J4_jsams_context_support.png",
    )
    print(f"Wrote six JSAMS revision figures to {figures}")


if __name__ == "__main__":  # pragma: no cover
    main()
