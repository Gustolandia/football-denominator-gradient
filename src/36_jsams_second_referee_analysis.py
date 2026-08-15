#!/usr/bin/env python
"""Run the second JSAMS referee's primary-estimand and selection checks.

This stage follows :mod:`34_jsams_referee_analysis` and changes the reporting
hierarchy without altering the frozen source data.  The reference estimand is
the odds of a same-day public spell-start report per recorded appearance.  Its
reference exposure is prior club-match minutes in the previous seven calendar
days, entered linearly per 90 minutes.  Prior report history is a continuous
additive covariate; effect modification remains a secondary sensitivity.

The stage also fits every combination of seven exposure summaries, three
outcome-timing definitions, and three denominator choices.  All 63 focal
contrasts share one Holm family because these analyses were selected after the
data had been inspected.  Additional outputs disclose temporal stability,
simultaneous spline bands, the population retained by conditional models, and
a bounded inverse-selection-weight sensitivity based on conservatively
reconstructed and overlap-resolved EPL membership intervals. A deterministic
source-audit queue is exposure-blinded, and manual decisions must pass queue,
source-independence, and immutable-field gates before they are summarised.

A later revision added six disclosures, all of which qualify the reference
estimate rather than support it. The multiverse is summarised as a distribution
of estimates instead of a rejection count, because a Holm family cannot control
the choices that defined the family. Pairwise correlations show that the
cumulative windows overlap by construction, so surviving windows are correlated
sensitivity analyses around one reference window rather than replications. A
measured-confounding table refits the reference model with age, position,
club-season, competition and season, and under player-match and club-season
clustering. The outcome audit reports the all-sampled estimand and
partial-identification bounds alongside the resolved-only proportion. The
selection stage carries explicit no-leakage gates and an included-versus-
excluded population comparison. An absolute-risk table gives the standardised
probability difference with its target population and the observed support
behind the curve.

Run after scripts 27 and 34::

    python src/36_jsams_second_referee_analysis.py

Outputs are written to ``data/processed/results`` with the prefix
``jsams_revised_``.  None of these analyses is labelled prespecified or causal.
"""

from __future__ import annotations

import importlib.util
import sys
import unicodedata
from math import erfc
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrix
from scipy.special import expit
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_identity import (  # noqa: E402
    PLAYER_KEY,
    RECORD_KEY,
    SEASON,
    SOURCE_FOUND,
    audited_surnames,
    deidentify_audit_frame,
    load_identity_map,
)


PLAYER_ID_COL = "tm_player_id"
DATE_COL = "date"
MINUTES_COL = "all_minutes_played"
HISTORY_COL = "prior_injuries_per_10000min"
HISTORY_MODEL_COL = "history_log_iqr"
SAME_DAY_COL = "injury_event_matchproxy_same_day"
LAG1_COL = "injury_event_matchproxy_lag1"
COMBINED_COL = "injury_event_matchproxy"
CALENDAR_TERMS = (
    "week_phase_sin",
    "week_phase_cos",
    "halfweek_phase_sin",
    "halfweek_phase_cos",
)
WINDOWS = (3, 5, 7, 10, 14)
PRIMARY_WINDOW = 7
REFERENCE_ANCHOR_MINUTES = 180.0
PREDICTION_GRID = tuple(float(value) for value in range(0, 181, 5))
SIMULTANEOUS_DRAWS = 10_000
CONDITIONAL_BOOTSTRAP_DRAWS = 5_000
RANDOM_SEED = 20260807
COMPLETE_SELECTION_END = pd.Timestamp("2024-06-30")
RECOVERY_REFERENCE = "6-7 days"
RECOVERY_CONGESTED = "0-3 days"
EXPOSURE_BANDS = ("0", "1-45", "46-90", "91-135", "136-180", ">180")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    """Require an explicit schema before constructing an analysis."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def load_source_module(filename: str, module_name: str):  # pragma: no cover
    """Load one numerically named pipeline module."""
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    path = src_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import pipeline script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normal_p_value(estimate: float, standard_error: float) -> float:
    """Return a two-sided normal p-value."""
    if not np.isfinite(estimate) or not np.isfinite(standard_error):
        return np.nan
    if standard_error <= 0.0:
        return 1.0 if np.isclose(estimate, 0.0) else 0.0
    return float(erfc(abs(estimate / standard_error) / np.sqrt(2.0)))


def holm_adjust(values: pd.Series) -> pd.Series:
    """Return Holm-adjusted p-values while retaining missing rows."""
    numeric = pd.to_numeric(values, errors="coerce")
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    finite = numeric.dropna().sort_values()
    running = 0.0
    count = len(finite)
    for rank, (index, value) in enumerate(finite.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted.loc[index] = running
    return adjusted


def add_prior_window_metrics(
    panel: pd.DataFrame,
    windows: Sequence[int] = WINDOWS,
) -> pd.DataFrame:
    """Add prior-only rolling minute and appearance-count summaries.

    The current appearance is excluded.  A ``w``-day window includes prior
    appearances on dates from ``date - w days`` through ``date - 1 day``.
    Player-date duplicates are rejected because their ordering is ambiguous.
    """
    _require_columns(panel, [PLAYER_ID_COL, DATE_COL, MINUTES_COL], "window panel")
    if not windows or any(int(window) <= 0 for window in windows):
        raise ValueError("windows must contain positive integers")
    out = panel.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out[MINUTES_COL] = pd.to_numeric(out[MINUTES_COL], errors="coerce").fillna(0.0)
    if out[[PLAYER_ID_COL, DATE_COL]].isna().any(axis=None):
        raise ValueError("player and date must be complete for rolling windows")
    if out.duplicated([PLAYER_ID_COL, DATE_COL]).any():
        raise ValueError("rolling windows require unique player-date rows")

    ordered = out.sort_values([PLAYER_ID_COL, DATE_COL]).copy()
    minute_arrays = {int(window): np.zeros(len(ordered)) for window in windows}
    count_arrays = {int(window): np.zeros(len(ordered), dtype=int) for window in windows}
    for positions in ordered.groupby(PLAYER_ID_COL, sort=False).indices.values():
        locations = np.asarray(positions, dtype=int)
        dates = ordered.iloc[locations][DATE_COL].to_numpy(dtype="datetime64[ns]")
        minutes = ordered.iloc[locations][MINUTES_COL].to_numpy(dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(minutes)))
        current = np.arange(len(locations), dtype=int)
        for window in windows:
            key = int(window)
            lower_dates = dates - np.timedelta64(key, "D")
            left = np.searchsorted(dates, lower_dates, side="left")
            minute_arrays[key][locations] = cumulative[current] - cumulative[left]
            count_arrays[key][locations] = current - left
    for window in windows:
        key = int(window)
        ordered[f"prior_minutes_{key}d"] = minute_arrays[key]
        ordered[f"prior_matches_{key}d"] = count_arrays[key]
    return ordered.sort_index()


def validate_reference_window(frame: pd.DataFrame) -> pd.DataFrame:
    """Verify that reconstructed seven-day minutes equal the legacy field."""
    _require_columns(
        frame,
        ["prior_minutes_7d", "all_minutes_last_7d"],
        "window validation frame",
    )
    difference = (
        pd.to_numeric(frame["prior_minutes_7d"], errors="coerce")
        - pd.to_numeric(frame["all_minutes_last_7d"], errors="coerce")
    )
    complete = difference.dropna()
    return pd.DataFrame(
        [
            {
                "comparison": "reconstructed_prior_minutes_7d_minus_legacy",
                "n_rows_compared": int(len(complete)),
                "n_exact_matches": int(np.isclose(complete, 0.0).sum()),
                "maximum_absolute_difference_minutes": float(complete.abs().max()),
                "parity_passes": bool(np.isclose(complete, 0.0).all()),
            }
        ]
    )


def exposure_specs() -> tuple[dict[str, Any], ...]:
    """Return the seven focal recent-exposure summaries."""
    minute_specs = tuple(
        {
            "exposure_id": f"prior_minutes_{window}d",
            "column": f"prior_minutes_{window}d",
            "scale": 90.0,
            "effect_label": f"per 90 prior minutes in {window} days",
            "kind": "linear",
        }
        for window in WINDOWS
    )
    return minute_specs + (
        {
            "exposure_id": "prior_matches_7d",
            "column": "prior_matches_7d",
            "scale": 1.0,
            "effect_label": "per additional prior match in 7 days",
            "kind": "linear",
        },
        {
            "exposure_id": "recovery_interval",
            "column": "recovery_interval_bin",
            "scale": 1.0,
            "effect_label": "0-3 versus 6-7 recovery days",
            "kind": "recovery",
        },
    )


def _model_formula(event_col: str, specification: Mapping[str, Any]) -> str:
    """Build an additive exposure formula with continuous prior history."""
    calendar = " + ".join(CALENDAR_TERMS)
    if specification["kind"] == "recovery":
        exposure = (
            "C(recovery_interval_bin, "
            f"Treatment(reference='{RECOVERY_REFERENCE}'))"
        )
    else:
        exposure = f"I({specification['column']} / {float(specification['scale'])})"
    return f"{event_col} ~ {exposure} + {HISTORY_MODEL_COL} + {calendar}"


def _focal_term(result: Any, specification: Mapping[str, Any]) -> str:
    """Locate the coefficient defining one exposure contrast."""
    names = list(result.params.index)
    if specification["kind"] == "recovery":
        matches = [
            name
            for name in names
            if "recovery_interval_bin" in name and f"[T.{RECOVERY_CONGESTED}]" in name
        ]
    else:
        matches = [name for name in names if specification["column"] in name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one focal term for {specification['exposure_id']}; found {matches}"
        )
    return matches[0]


def fit_exposure_model(
    frame: pd.DataFrame,
    event_col: str,
    denominator: str,
    specification: Mapping[str, Any],
) -> tuple[Any, pd.DataFrame, str]:
    """Fit one additive model and return its complete analysis frame."""
    required = [
        PLAYER_ID_COL,
        event_col,
        HISTORY_MODEL_COL,
        specification["column"],
        MINUTES_COL,
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "exposure model frame")
    work = frame.dropna(subset=required).copy()
    work = work[pd.to_numeric(work[MINUTES_COL], errors="coerce").gt(0.0)].copy()
    if work.empty or int(work[event_col].sum()) <= 0:
        raise ValueError(f"No estimable events for {event_col}")
    formula = _model_formula(event_col, specification)
    kwargs: dict[str, Any] = {}
    if denominator == "per_appearance":
        family = sm.families.Binomial()
    elif denominator == "observed_minutes":
        family = sm.families.Poisson()
        kwargs["offset"] = np.log(
            pd.to_numeric(work[MINUTES_COL], errors="coerce").clip(lower=1.0)
        )
    elif denominator == "fixed_90":
        family = sm.families.Poisson()
        kwargs["offset"] = pd.Series(np.log(90.0), index=work.index)
    else:
        raise ValueError(f"Unknown denominator: {denominator}")
    result = smf.glm(formula, data=work, family=family, **kwargs).fit(
        cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]}
    )
    return result, work, _focal_term(result, specification)


def exposure_model_row(
    frame: pd.DataFrame,
    event_col: str,
    denominator: str,
    specification: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Return one focal estimate with player-clustered uncertainty."""
    result, work, term = fit_exposure_model(
        frame, event_col, denominator, specification
    )
    estimate = float(result.params[term])
    standard_error = float(result.bse[term])
    critical = NormalDist().inv_cdf(0.975)
    effect_measure = "odds_ratio" if denominator == "per_appearance" else "incidence_rate_ratio"
    row = {
        "exposure_id": specification["exposure_id"],
        "effect_label": specification["effect_label"],
        "event_col": event_col,
        "denominator": denominator,
        "effect_measure": effect_measure,
        "estimate": float(np.exp(estimate)),
        "ci_low": float(np.exp(estimate - critical * standard_error)),
        "ci_high": float(np.exp(estimate + critical * standard_error)),
        "log_estimate": estimate,
        "standard_error": standard_error,
        "p_value": _normal_p_value(estimate, standard_error),
        "n_rows": int(len(work)),
        "n_players": int(work[PLAYER_ID_COL].nunique()),
        "n_events": int(work[event_col].sum()),
        "formula": result.model.formula,
        "cluster": "player",
        "analysis_timing": "post-data reviewer-requested multiverse",
    }
    return row, result


def exposure_multiverse(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit the complete 7 x 3 x 3 exposure sensitivity family."""
    rows = []
    for specification in exposure_specs():
        for event_col in (SAME_DAY_COL, LAG1_COL, COMBINED_COL):
            for denominator in ("per_appearance", "observed_minutes", "fixed_90"):
                row, _ = exposure_model_row(
                    frame, event_col, denominator, specification
                )
                rows.append(row)
    out = pd.DataFrame(rows)
    out["holm_p_value_63_model_family"] = holm_adjust(out["p_value"])
    out["reject_holm_0_05"] = out["holm_p_value_63_model_family"].lt(0.05)
    out["family_size"] = int(len(out))
    return out


def exposure_multiverse_summary(multiverse: pd.DataFrame) -> pd.DataFrame:
    """Describe the distribution of all 63 estimates, not only the rejections.

    A Holm family controls the tests it contains; it cannot control the choices
    that defined the family. Reporting where the estimates sit is therefore a
    more honest summary of a post-data multiverse than a rejection count, so
    this table gives the median, quartiles and range within each stratum.
    """
    _require_columns(
        multiverse,
        [
            "estimate",
            "denominator",
            "event_col",
            "holm_p_value_63_model_family",
            "reject_holm_0_05",
        ],
        "multiverse summary frame",
    )
    strata: list[tuple[str, str, pd.DataFrame]] = [
        ("all_63_models", "every exposure, timing and denominator combination", multiverse)
    ]
    for denominator, group in multiverse.groupby("denominator", sort=True):
        strata.append((f"denominator_{denominator}", f"denominator held at {denominator}", group))
    for event_col, group in multiverse.groupby("event_col", sort=True):
        strata.append((f"outcome_{event_col}", f"outcome timing held at {event_col}", group))
    rows = []
    for stratum_id, description, group in strata:
        estimates = pd.to_numeric(group["estimate"], errors="coerce").dropna()
        rows.append(
            {
                "stratum_id": stratum_id,
                "stratum_description": description,
                "n_models": int(len(group)),
                "n_holm_rejections": int(group["reject_holm_0_05"].sum()),
                "estimate_min": float(estimates.min()),
                "estimate_q1": float(estimates.quantile(0.25)),
                "estimate_median": float(estimates.median()),
                "estimate_q3": float(estimates.quantile(0.75)),
                "estimate_max": float(estimates.max()),
                "share_above_one": float(estimates.gt(1.0).mean()),
                "interpretation": (
                    "post-data multiverse distribution; adjusted p-values control the "
                    "63 stated tests, not the choices that defined the family"
                ),
            }
        )
    return pd.DataFrame(rows)


def exposure_metric_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise correlations among the seven recent-exposure summaries.

    Cumulative windows share matches by construction, so surviving windows are
    correlated sensitivity analyses around one reference window rather than
    independent replications.
    """
    specifications = exposure_specs()
    columns: dict[str, pd.Series] = {}
    for specification in specifications:
        column = specification["column"]
        if specification["kind"] == "recovery":
            _require_columns(frame, [column], "correlation frame")
            values = (
                frame[column].astype(str).eq(RECOVERY_CONGESTED).astype(float)
                .where(frame[column].astype(str).isin({RECOVERY_CONGESTED, RECOVERY_REFERENCE}))
            )
        else:
            _require_columns(frame, [column], "correlation frame")
            values = pd.to_numeric(frame[column], errors="coerce")
        columns[specification["exposure_id"]] = values
    rows = []
    identifiers = list(columns)
    reference_id = f"prior_minutes_{PRIMARY_WINDOW}d"
    for first_index, first in enumerate(identifiers):
        for second in identifiers[first_index + 1 :]:
            paired = pd.DataFrame(
                {"first": columns[first], "second": columns[second]}
            ).dropna()
            if len(paired) < 2 or paired["first"].nunique() < 2 or paired["second"].nunique() < 2:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = float(paired["first"].corr(paired["second"]))
                spearman = float(paired["first"].corr(paired["second"], method="spearman"))
            rows.append(
                {
                    "exposure_a": first,
                    "exposure_b": second,
                    "n_rows_compared": int(len(paired)),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "involves_reference_window": bool(
                        reference_id in {first, second}
                    ),
                    # The manuscript quotes one range for "the cumulative
                    # windows". Flagging the pairs that are both cumulative
                    # windows fixes which correlations that range is over, so
                    # it cannot silently become the reference-window subset.
                    "both_cumulative_windows": bool(
                        first.startswith("prior_minutes_")
                        and second.startswith("prior_minutes_")
                    ),
                    "reference_window": reference_id,
                    "interpretation": (
                        "cumulative windows overlap by construction; high correlation "
                        "means correlated sensitivity analyses, not independent replication"
                    ),
                }
            )
    return pd.DataFrame(rows)


NEGATIVE_CONTROL_COL = "same_day_illness_report"
SPECIFIC_TEXT_COL = "same_day_specific_description"
ILLNESS_TYPES = ("illness/other medical",)
UNSPECIFIC_TYPES = ("unknown", "other/unspecified")


def add_negative_control_outcomes(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    classify_injury_type: Any,
) -> pd.DataFrame:
    """Attach illness and description-specificity same-day indicators.

    Recent match minutes cannot plausibly cause an illness. If exposure
    predicts illness spell starts as strongly as it predicts muscle/tendon
    spell starts, the headline association is more likely to reflect how
    absences get reported than what happened to the tissue. The
    description-specificity flag supports the matching question of whether
    reporting detail itself varies with exposure.
    """
    _require_columns(panel, [PLAYER_ID_COL, DATE_COL, SAME_DAY_COL], "negative-control panel")
    _require_columns(
        episodes, [PLAYER_ID_COL, "start_date", "injury_desc"], "negative-control episodes"
    )
    out = panel.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    rows = episodes[[PLAYER_ID_COL, "start_date", "injury_desc"]].copy()
    rows["start_date"] = pd.to_datetime(rows["start_date"], errors="coerce")
    rows["injury_desc"] = rows["injury_desc"].fillna("").astype(str).str.strip()
    rows["episode_public_type"] = rows["injury_desc"].map(classify_injury_type)
    grouped = (
        rows.dropna(subset=[PLAYER_ID_COL, "start_date"])
        .groupby([PLAYER_ID_COL, "start_date"], as_index=False)
        .agg(
            same_day_any_illness=(
                "episode_public_type",
                lambda values: bool(pd.Series(values).isin(ILLNESS_TYPES).any()),
            ),
            same_day_any_specific=(
                "episode_public_type",
                # The parentheses matter: negating after .any() would apply
                # bitwise NOT to a Python bool and always evaluate truthy.
                lambda values: bool((~pd.Series(values).isin(UNSPECIFIC_TYPES)).any()),
            ),
        )
        .rename(columns={"start_date": DATE_COL})
    )
    out = out.merge(grouped, on=[PLAYER_ID_COL, DATE_COL], how="left", validate="many_to_one")
    same_day = pd.to_numeric(out[SAME_DAY_COL], errors="coerce").fillna(0).eq(1)
    out[NEGATIVE_CONTROL_COL] = (same_day & out["same_day_any_illness"].eq(True)).astype(int)
    out[SPECIFIC_TEXT_COL] = (same_day & out["same_day_any_specific"].eq(True)).astype(int)
    return out


PLACEBO_WINDOWS = (30, 37)
PLACEBO_COL = "prior_minutes_placebo_31_37d"


def add_placebo_exposure_window(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a distant exposure window that cannot plausibly cause today's event.

    Minutes played five weeks ago share a player's reporting propensity, club
    profile and durability, but no fatigue mechanism connects them to an injury
    today. If the distant window predicts a spell start as strongly as the
    recent one, the association reflects who gets reported rather than recent
    load. This is the negative-control exposure the illness outcome could not
    provide, because same-day illness reports are almost non-existent.
    """
    windowed = add_prior_window_metrics(frame, windows=PLACEBO_WINDOWS)
    out = windowed.copy()
    out[PLACEBO_COL] = (
        pd.to_numeric(out["prior_minutes_37d"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["prior_minutes_30d"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    return out


def placebo_window_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Contrast the recent exposure window with a distant placebo window."""
    _require_columns(
        frame,
        [SAME_DAY_COL, PLAYER_ID_COL, "prior_minutes_7d", PLACEBO_COL, HISTORY_MODEL_COL, *CALENDAR_TERMS],
        "placebo window frame",
    )
    work = frame.dropna(
        subset=[SAME_DAY_COL, "prior_minutes_7d", PLACEBO_COL, HISTORY_MODEL_COL, *CALENDAR_TERMS]
    ).copy()
    calendar = " + ".join(CALENDAR_TERMS)
    critical = NormalDist().inv_cdf(0.975)
    rows = []
    specifications = (
        ("recent_7d_alone", "I(prior_minutes_7d / 90.0)", "prior_minutes_7d", "recent seven-day minutes"),
        (
            "placebo_31_37d_alone",
            f"I({PLACEBO_COL} / 90.0)",
            PLACEBO_COL,
            "placebo window: minutes 31-37 days earlier",
        ),
        (
            "both_windows",
            f"I(prior_minutes_7d / 90.0) + I({PLACEBO_COL} / 90.0)",
            "prior_minutes_7d",
            "recent window with the placebo window held constant",
        ),
        (
            "both_windows",
            f"I(prior_minutes_7d / 90.0) + I({PLACEBO_COL} / 90.0)",
            PLACEBO_COL,
            "placebo window with the recent window held constant",
        ),
    )
    for model_id, exposure, focal, description in specifications:
        formula = f"{SAME_DAY_COL} ~ {exposure} + {HISTORY_MODEL_COL} + {calendar}"
        result = smf.glm(formula, data=work, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]}
        )
        term = next(name for name in result.params.index if focal in name)
        estimate = float(result.params[term])
        standard_error = float(result.bse[term])
        rows.append(
            {
                "model_id": model_id,
                "focal_window": focal,
                "description": description,
                "estimate": float(np.exp(estimate)),
                "ci_low": float(np.exp(estimate - critical * standard_error)),
                "ci_high": float(np.exp(estimate + critical * standard_error)),
                "p_value": _normal_p_value(estimate, standard_error),
                "n_rows": int(len(work)),
                "n_events": int(work[SAME_DAY_COL].sum()),
            }
        )
    out = pd.DataFrame(rows)
    # A mutually adjusted placebo is only interpretable beside the collinearity
    # between the two windows: both are driven by whether this man is a regular
    # starter, so a high correlation would mean the placebo term is attenuated
    # by collinearity rather than by an absent causal path.
    recent = pd.to_numeric(work["prior_minutes_7d"], errors="coerce")
    distant = pd.to_numeric(work[PLACEBO_COL], errors="coerce")
    out["pearson_r_recent_vs_placebo"] = float(recent.corr(distant))
    out["spearman_r_recent_vs_placebo"] = float(recent.corr(distant, method="spearman"))
    out["interpretation"] = (
        "a placebo association as large as the recent-window association would "
        "indicate player-level reporting propensity rather than recent load; "
        "the reported correlation shows how much collinearity the mutual "
        "adjustment had to separate"
    )
    return out


def placebo_denominator_replication(frame: pd.DataFrame) -> pd.DataFrame:
    """Refit the placebo window under all three denominators.

    The identity says a per-minute coefficient is the per-appearance one less
    the gradient. If that is a property of the arithmetic rather than of the
    seven-day exposure, then an exposure chosen for having no plausible causal
    path should show the same structure: its own gradient, its own attenuation,
    and the same ratio between what the identity predicts and what is observed.
    A second exposure cannot prove the mechanism, but it can refute the worry
    that the ratio is a coincidence of one specification.
    """
    required = [
        SAME_DAY_COL, MINUTES_COL, PLAYER_ID_COL, PLACEBO_COL,
        HISTORY_MODEL_COL, *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "placebo denominator frame")
    work = frame.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    if work.empty or int(work[SAME_DAY_COL].sum()) <= 0:
        raise ValueError("no estimable rows for the placebo denominator replication")

    calendar = " + ".join(CALENDAR_TERMS)
    critical = NormalDist().inv_cdf(0.975)
    minutes = work[MINUTES_COL].clip(lower=1.0)
    formula = (
        f"{SAME_DAY_COL} ~ I({PLACEBO_COL} / 90.0) + {HISTORY_MODEL_COL} + {calendar}"
    )

    fitted: dict[str, float] = {}
    rows = []
    for denominator, family, offset in (
        ("per_appearance", sm.families.Binomial(), None),
        ("fixed_90", sm.families.Poisson(), pd.Series(np.log(90.0), index=work.index)),
        ("observed_minutes", sm.families.Poisson(), np.log(minutes)),
    ):
        kwargs = {} if offset is None else {"offset": offset}
        result = smf.glm(formula, data=work, family=family, **kwargs).fit(
            cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]}
        )
        term = next(name for name in result.params.index if PLACEBO_COL in name)
        estimate = float(result.params[term])
        standard_error = float(result.bse[term])
        fitted[denominator] = estimate
        rows.append(
            {
                "quantity": denominator,
                "value": float(np.exp(estimate)),
                "ci_low": float(np.exp(estimate - critical * standard_error)),
                "ci_high": float(np.exp(estimate + critical * standard_error)),
                "note": f"placebo-window estimate under the {denominator} denominator",
            }
        )

    work["exposure_per_90"] = pd.to_numeric(work[PLACEBO_COL], errors="coerce") / 90.0
    work["log_recorded_minutes"] = np.log(minutes)
    gamma_fit = smf.ols(
        f"log_recorded_minutes ~ exposure_per_90 + {HISTORY_MODEL_COL} + {calendar}",
        data=work,
    ).fit(cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]})
    gamma = float(gamma_fit.params["exposure_per_90"])
    gamma_error = float(gamma_fit.bse["exposure_per_90"])
    attenuation = fitted["fixed_90"] - fitted["observed_minutes"]

    rows.extend(
        [
            {
                "quantity": "gamma_placebo",
                "value": gamma,
                "ci_low": gamma - critical * gamma_error,
                "ci_high": gamma + critical * gamma_error,
                "note": "gradient of log recorded minutes on the placebo window",
            },
            {
                "quantity": "observed_log_attenuation",
                "value": attenuation,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "note": "fixed-90 minus recorded-minute coefficient for the placebo window",
            },
            {
                "quantity": "gamma_over_observed_attenuation",
                "value": gamma / attenuation if attenuation else np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "note": (
                    "how far the first-order identity sits from the attenuation "
                    "actually observed, for an exposure with no plausible causal path"
                ),
            },
        ]
    )
    out = pd.DataFrame(rows)
    out["n_rows"] = int(len(work))
    out["n_events"] = int(work[SAME_DAY_COL].sum())
    out["interpretation"] = (
        "the placebo window carries its own gradient and its own attenuation, "
        "and the identity over-predicts it by about the same factor as for the "
        "reference exposure; the arithmetic is a property of the denominator, "
        "not of the exposure chosen"
    )
    return out


def negative_control_outcome_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the exposure association across outcomes of differing plausibility.

    Muscle/tendon reports are the most plausibly exposure-related outcome and
    illness reports the least. A similar association across both points to
    reporting behaviour rather than tissue injury; a much weaker illness
    association is the reassuring result.
    """
    specification = next(
        item for item in exposure_specs() if item["exposure_id"] == "prior_minutes_7d"
    )
    outcomes = (
        (SAME_DAY_COL, "any same-day spell start", "plausibly exposure-related"),
        (
            "same_day_muscle_tendon_report",
            "same-day muscle/tendon spell start",
            "most plausibly exposure-related",
        ),
        (
            "same_day_reported_absence_ge28d",
            "same-day spell start with at least 28 reported days",
            "least dependent on discretionary reporting",
        ),
        (
            NEGATIVE_CONTROL_COL,
            "same-day illness spell start",
            "negative control: recent minutes cannot plausibly cause illness",
        ),
    )
    rows = []
    for event_col, label, role in outcomes:
        if event_col not in frame.columns:
            raise KeyError(f"negative-control frame missing {event_col}")
        events = int(pd.to_numeric(frame[event_col], errors="coerce").fillna(0).sum())
        if events < 10:
            rows.append(
                {
                    "event_col": event_col,
                    "outcome_label": label,
                    "interpretive_role": role,
                    "n_events": events,
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "estimable": False,
                    "note": "fewer than ten events; not estimable",
                }
            )
            continue
        row, _ = exposure_model_row(frame, event_col, "per_appearance", specification)
        rows.append(
            {
                "event_col": event_col,
                "outcome_label": label,
                "interpretive_role": role,
                "n_events": int(row["n_events"]),
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "p_value": row["p_value"],
                "estimable": True,
                "note": "odds ratio per 90 previous-seven-day minutes",
            }
        )
    out = pd.DataFrame(rows)
    out["interpretation"] = (
        "a negative-control association as large as the muscle/tendon association "
        "would indicate exposure-dependent reporting rather than tissue injury"
    )
    return out


def ascertainment_by_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    """Test whether reporting detail varies with recent exposure.

    Among appearances that produced a spell start, this asks whether the
    probability that the report carries a specific description rises or falls
    with recent minutes. Exposure-dependent reporting detail would mean the
    outcome is measured differently across the exposure range.
    """
    _require_columns(
        frame,
        [SAME_DAY_COL, SPECIFIC_TEXT_COL, "prior_minutes_7d", PLAYER_ID_COL],
        "ascertainment frame",
    )
    events = frame[pd.to_numeric(frame[SAME_DAY_COL], errors="coerce").fillna(0).eq(1)].copy()
    if events.empty:
        raise ValueError("no same-day spell starts available for the ascertainment test")
    specific = pd.to_numeric(events[SPECIFIC_TEXT_COL], errors="coerce").fillna(0)
    critical = NormalDist().inv_cdf(0.975)
    if specific.nunique() < 2:
        estimate, standard_error = np.nan, np.nan
        p_value = np.nan
    else:
        result = smf.glm(
            f"{SPECIFIC_TEXT_COL} ~ I(prior_minutes_7d / 90.0)",
            data=events,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": events[PLAYER_ID_COL]})
        term = next(name for name in result.params.index if "prior_minutes_7d" in name)
        estimate = float(result.params[term])
        standard_error = float(result.bse[term])
        p_value = _normal_p_value(estimate, standard_error)
    bands = pd.cut(
        pd.to_numeric(events["prior_minutes_7d"], errors="coerce"),
        bins=[-0.001, 0.001, 45.0, 90.0, 135.0, 180.0, float("inf")],
        labels=list(EXPOSURE_BANDS),
        ordered=True,
    )
    by_band = (
        events.assign(exposure_band=bands)
        .groupby("exposure_band", observed=False)
        .agg(
            n_spell_starts=(SAME_DAY_COL, "size"),
            n_specific_description=(SPECIFIC_TEXT_COL, "sum"),
        )
        .reset_index()
    )
    by_band["share_specific"] = np.where(
        by_band["n_spell_starts"].gt(0),
        by_band["n_specific_description"] / by_band["n_spell_starts"],
        np.nan,
    )
    by_band["odds_ratio_per_90_minutes"] = float(np.exp(estimate)) if np.isfinite(estimate) else np.nan
    by_band["ci_low"] = (
        float(np.exp(estimate - critical * standard_error)) if np.isfinite(estimate) else np.nan
    )
    by_band["ci_high"] = (
        float(np.exp(estimate + critical * standard_error)) if np.isfinite(estimate) else np.nan
    )
    by_band["p_value"] = p_value
    by_band["interpretation"] = (
        "an odds ratio away from one would mean reporting detail changes with "
        "exposure, so the outcome is not measured identically across the range"
    )
    return by_band


CONFOUNDING_SPECIFICATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("reference", "", "player", "reference model: prior history and calendar phase only"),
    ("age_position_adjusted", " + age_years + C(position_group)", "player", "adds age and position group"),
    ("club_season_adjusted", " + C(club_season)", "player", "adds club-season fixed effects"),
    ("competition_adjusted", " + C(competition_context)", "player", "adds current-match competition"),
    ("season_adjusted", " + C(season)", "player", "adds season fixed effects"),
    (
        "fully_adjusted",
        " + age_years + C(position_group) + C(club_season) + C(competition_context)",
        "player",
        "adds age, position, club-season and competition together",
    ),
    ("premier_league_only", "", "player", "restricts to Premier League current matches"),
    ("player_match_two_way_cluster", "", "player_match", "player and match two-way clustered uncertainty"),
    ("club_season_cluster", "", "club_season", "club-season clustered uncertainty"),
)


def _two_way_clustered_result(frame: pd.DataFrame, formula: str, second_cluster: str):
    """Fit the additive reference logit with two-way clustered covariance."""
    if frame[PLAYER_ID_COL].nunique() < 2 or frame[second_cluster].nunique() < 2:
        raise ValueError("Two-way clustering requires at least two clusters per dimension")
    result = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit()
    first_codes = pd.factorize(frame[PLAYER_ID_COL], sort=True)[0]
    second_codes = pd.factorize(frame[second_cluster], sort=True)[0]
    covariance, _, _ = cov_cluster_2groups(result, first_codes, second_codes)
    result.cov_params_default = covariance
    result._cache.clear()
    return result


def confounding_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Refit the reference seven-day model under measured-confounder variants.

    The reference model adjusts only prior report history and calendar phase.
    Age, position, club-season, competition and season could plausibly affect
    both recent exposure and whether an absence is publicly reported, and
    observations share matches as well as players. This table puts those
    sensitivities beside the principal estimate so the qualification is visible
    rather than buried.
    """
    required = [
        SAME_DAY_COL,
        PLAYER_ID_COL,
        "prior_minutes_7d",
        HISTORY_MODEL_COL,
        "age_years",
        "position_group",
        "club_season",
        "competition_context",
        "season",
        "match_cluster_id",
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "confounding sensitivity frame")
    base = frame.dropna(subset=required).copy()
    base = base[pd.to_numeric(base[MINUTES_COL], errors="coerce").gt(0.0)].copy()
    calendar = " + ".join(CALENDAR_TERMS)
    critical = NormalDist().inv_cdf(0.975)
    rows = []
    for model_id, controls, covariance, description in CONFOUNDING_SPECIFICATIONS:
        subset = (
            base[base["competition_context"].astype(str).eq("Premier League")].copy()
            if model_id == "premier_league_only"
            else base
        )
        if subset.empty or int(subset[SAME_DAY_COL].sum()) <= 0:
            raise ValueError(f"No estimable rows for {model_id}")
        formula = (
            f"{SAME_DAY_COL} ~ I(prior_minutes_7d / 90.0) + {HISTORY_MODEL_COL} "
            f"+ {calendar}{controls}"
        )
        if covariance == "player_match":
            result = _two_way_clustered_result(subset, formula, "match_cluster_id")
        elif covariance == "club_season":
            result = smf.glm(
                formula, data=subset, family=sm.families.Binomial()
            ).fit(cov_type="cluster", cov_kwds={"groups": subset["club_season"]})
        else:
            result = smf.glm(
                formula, data=subset, family=sm.families.Binomial()
            ).fit(cov_type="cluster", cov_kwds={"groups": subset[PLAYER_ID_COL]})
        term = next(
            name for name in result.params.index if "prior_minutes_7d" in name
        )
        estimate = float(result.params[term])
        standard_error = float(np.sqrt(float(np.asarray(result.cov_params())[
            list(result.params.index).index(term),
            list(result.params.index).index(term),
        ])))
        rows.append(
            {
                "model_id": model_id,
                "description": description,
                "effect_label": "per 90 prior minutes in 7 days",
                "effect_measure": "odds_ratio",
                "estimate": float(np.exp(estimate)),
                "ci_low": float(np.exp(estimate - critical * standard_error)),
                "ci_high": float(np.exp(estimate + critical * standard_error)),
                "log_estimate": estimate,
                "standard_error": standard_error,
                "p_value": _normal_p_value(estimate, standard_error),
                "covariance": covariance,
                "controls": controls.strip(" +"),
                "n_rows": int(len(subset)),
                "n_players": int(subset[PLAYER_ID_COL].nunique()),
                "n_events": int(subset[SAME_DAY_COL].sum()),
                "analysis_timing": "post-data measured-confounding sensitivity",
                "interpretation": (
                    "measured-covariate and clustering sensitivity for the reference "
                    "estimate; unmeasured health and selection remain uncontrolled"
                ),
            }
        )
    return pd.DataFrame(rows)


def add_club_fixture_congestion(frame: pd.DataFrame) -> pd.DataFrame:
    """Add club-level fixture density beside the player's own recent minutes.

    A player's recent minutes are largely set by his club's schedule, so an
    individual-minutes association could be a club-schedule association in
    disguise. These two columns let the reference model separate them.
    """
    _require_columns(frame, [PLAYER_ID_COL, DATE_COL, "player_club_id"], "club congestion frame")
    out = frame.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    fixtures = (
        out[["player_club_id", DATE_COL]]
        .dropna()
        .drop_duplicates()
        .sort_values(["player_club_id", DATE_COL])
    )
    gaps = []
    counts = []
    for club, group in fixtures.groupby("player_club_id", sort=False):
        dates = group[DATE_COL].to_numpy(dtype="datetime64[ns]")
        previous = np.concatenate(([np.datetime64("NaT")], dates[:-1]))
        gap = (dates - previous) / np.timedelta64(1, "D")
        lower = dates - np.timedelta64(7, "D")
        left = np.searchsorted(dates, lower, side="left")
        prior_count = np.arange(len(dates)) - left
        gaps.append(pd.DataFrame({"player_club_id": club, DATE_COL: dates, "club_days_since_last_fixture": gap}))
        counts.append(pd.DataFrame({"player_club_id": club, DATE_COL: dates, "club_fixtures_last_7d": prior_count}))
    if not gaps:
        raise ValueError("no club fixtures available for congestion features")
    calendar = pd.concat(gaps, ignore_index=True).merge(
        pd.concat(counts, ignore_index=True), on=["player_club_id", DATE_COL], how="inner"
    )
    out = out.merge(calendar, on=["player_club_id", DATE_COL], how="left")
    out["club_days_since_last_fixture"] = (
        pd.to_numeric(out["club_days_since_last_fixture"], errors="coerce")
        .fillna(30.0)
        .clip(upper=30.0)
    )
    out["club_fixtures_last_7d"] = pd.to_numeric(
        out["club_fixtures_last_7d"], errors="coerce"
    ).fillna(0.0)
    return out


def club_congestion_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate the player's own minutes from his club's fixture schedule."""
    required = [
        SAME_DAY_COL,
        PLAYER_ID_COL,
        "prior_minutes_7d",
        HISTORY_MODEL_COL,
        "club_days_since_last_fixture",
        "club_fixtures_last_7d",
        "club_season",
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "club congestion sensitivity frame")
    work = frame.dropna(subset=required).copy()
    calendar = " + ".join(CALENDAR_TERMS)
    critical = NormalDist().inv_cdf(0.975)
    specifications = (
        ("reference", "", "player minutes only"),
        (
            "plus_club_schedule",
            " + club_days_since_last_fixture + club_fixtures_last_7d",
            "adds club fixture gap and club fixture count",
        ),
        (
            "plus_club_schedule_and_club_season",
            " + club_days_since_last_fixture + club_fixtures_last_7d + C(club_season)",
            "adds club schedule and club-season fixed effects",
        ),
    )
    rows = []
    for model_id, controls, description in specifications:
        formula = (
            f"{SAME_DAY_COL} ~ I(prior_minutes_7d / 90.0) + {HISTORY_MODEL_COL} "
            f"+ {calendar}{controls}"
        )
        result = smf.glm(formula, data=work, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]}
        )
        term = next(name for name in result.params.index if "prior_minutes_7d" in name)
        estimate = float(result.params[term])
        standard_error = float(result.bse[term])
        rows.append(
            {
                "model_id": model_id,
                "description": description,
                "estimate": float(np.exp(estimate)),
                "ci_low": float(np.exp(estimate - critical * standard_error)),
                "ci_high": float(np.exp(estimate + critical * standard_error)),
                "p_value": _normal_p_value(estimate, standard_error),
                "n_rows": int(len(work)),
                "n_events": int(work[SAME_DAY_COL].sum()),
                "interpretation": (
                    "if the player-minutes estimate survives club schedule and "
                    "club-season adjustment, it is not purely a team fixture effect"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_in_threshold_sensitivity(
    frame: pd.DataFrame,
    thresholds: Sequence[float] = (450.0, 900.0, 1800.0),
) -> pd.DataFrame:
    """Refit the reference model across alternative established-player run-ins."""
    _require_columns(
        frame, [SAME_DAY_COL, PLAYER_ID_COL, "prior_minutes_played"], "run-in frame"
    )
    specification = next(
        item for item in exposure_specs() if item["exposure_id"] == "prior_minutes_7d"
    )
    rows = []
    for threshold in thresholds:
        subset = frame[
            pd.to_numeric(frame["prior_minutes_played"], errors="coerce").ge(float(threshold))
        ].copy()
        if subset.empty or int(subset[SAME_DAY_COL].sum()) <= 0:
            raise ValueError(f"no estimable rows at a {threshold}-minute run-in")
        row, _ = exposure_model_row(subset, SAME_DAY_COL, "per_appearance", specification)
        rows.append(
            {
                "run_in_minutes": float(threshold),
                "run_in_description": f"at least {int(threshold)} earlier club-match minutes",
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "p_value": row["p_value"],
                "n_rows": row["n_rows"],
                "n_players": row["n_players"],
                "n_events": row["n_events"],
                "is_reference": bool(np.isclose(threshold, 900.0)),
                "interpretation": (
                    "the 900-minute reference threshold is a choice; this shows "
                    "whether the association depends on it"
                ),
            }
        )
    return pd.DataFrame(rows)


def _standardization_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Compress observed calendar phases into weighted covariate patterns."""
    reference = (
        frame.groupby(list(CALENDAR_TERMS), dropna=False, observed=False)
        .size()
        .rename("standardization_weight")
        .reset_index()
    )
    if reference.empty:
        raise ValueError("standardization reference is empty")
    return reference


def _marginal_components(
    result: Any,
    reference: pd.DataFrame,
    burden: float,
) -> tuple[float, np.ndarray]:
    """Return calendar-standardised probability and its gradient."""
    template = reference.copy()
    weights = template.pop("standardization_weight").to_numpy(dtype=float)
    weights = weights / weights.sum()
    template["prior_minutes_7d"] = float(burden)
    template[HISTORY_MODEL_COL] = 0.0
    design = np.asarray(
        build_design_matrices([result.model.data.design_info], template)[0],
        dtype=float,
    )
    probability = expit(design @ np.asarray(result.params, dtype=float))
    gradient = (weights * probability * (1.0 - probability)) @ design
    return float(weights @ probability), np.asarray(gradient, dtype=float)


def _simultaneous_critical_value(
    gradients: np.ndarray,
    covariance: np.ndarray,
    draws: int = SIMULTANEOUS_DRAWS,
    seed: int = RANDOM_SEED,
) -> float:
    """Simulate a 95% maximum-t critical value over a prediction grid."""
    if draws < 100:
        raise ValueError("simultaneous bands require at least 100 draws")
    symmetric = (covariance + covariance.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    root = vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None)))
    standard_errors = np.sqrt(
        np.clip(np.einsum("ij,jk,ik->i", gradients, symmetric, gradients), 0.0, None)
    )
    active = standard_errors > 0.0
    if not active.any():
        return NormalDist().inv_cdf(0.975)
    rng = np.random.default_rng(seed)
    coefficient_draws = rng.standard_normal((draws, symmetric.shape[0])) @ root.T
    deviations = coefficient_draws @ gradients[active].T
    maxima = np.max(np.abs(deviations / standard_errors[active]), axis=1)
    return float(np.quantile(maxima, 0.95))


def additive_curve_analysis(
    frame: pd.DataFrame,
    spline_expression: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate the additive linear curve and spline shape sensitivity."""
    specification = next(
        item for item in exposure_specs() if item["exposure_id"] == "prior_minutes_7d"
    )
    linear, work, _ = fit_exposure_model(
        frame, SAME_DAY_COL, "per_appearance", specification
    )
    spline_formula = (
        f"{SAME_DAY_COL} ~ {spline_expression} + {HISTORY_MODEL_COL} + "
        + " + ".join(CALENDAR_TERMS)
    )
    spline = smf.glm(
        spline_formula, data=work, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]})
    reference = _standardization_reference(work)
    critical = NormalDist().inv_cdf(0.975)
    rows = []
    spline_components = []
    for model_id, model in (("additive_linear", linear), ("additive_spline", spline)):
        components = []
        for burden in PREDICTION_GRID:
            estimate, gradient = _marginal_components(model, reference, burden)
            components.append((burden, estimate, gradient))
        covariance = np.asarray(model.cov_params(), dtype=float)
        gradients = np.vstack([item[2] for item in components])
        simultaneous = (
            _simultaneous_critical_value(gradients, covariance)
            if model_id == "additive_spline"
            else np.nan
        )
        for burden, estimate, gradient in components:
            standard_error = float(
                np.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
            )
            rows.append(
                {
                    "model_id": model_id,
                    "prior_minutes_7d": burden,
                    "estimate_per_1000_appearances": 1000.0 * estimate,
                    "pointwise_ci_low": 1000.0 * max(0.0, estimate - critical * standard_error),
                    "pointwise_ci_high": 1000.0 * min(1.0, estimate + critical * standard_error),
                    "simultaneous_ci_low": (
                        1000.0 * max(0.0, estimate - simultaneous * standard_error)
                        if np.isfinite(simultaneous)
                        else np.nan
                    ),
                    "simultaneous_ci_high": (
                        1000.0 * min(1.0, estimate + simultaneous * standard_error)
                        if np.isfinite(simultaneous)
                        else np.nan
                    ),
                    "simultaneous_critical_value": simultaneous,
                    "history_value": "median continuous prior history",
                    "standardization": "observed calendar phase distribution",
                }
            )
        if model_id == "additive_spline":
            spline_components = components
    names = list(spline.params.index)
    spline_terms = [name for name in names if "prior_minutes_7d" in name]
    restriction = np.zeros((len(spline_terms), len(names)))
    for row_index, term in enumerate(spline_terms):
        restriction[row_index, names.index(term)] = 1.0
    tested = spline.wald_test(restriction, scalar=True)
    tests = pd.DataFrame(
        [
            {
                "test_id": "additive_spline_global_exposure",
                "test_statistic": float(np.asarray(tested.statistic).squeeze()),
                "df": int(len(spline_terms)),
                "p_value": float(np.asarray(tested.pvalue).squeeze()),
                "band_interpretation": (
                    "pointwise intervals describe one burden at a time; simultaneous "
                    "intervals cover the displayed spline grid as one family"
                ),
                "n_grid_points": int(len(spline_components)),
            }
        ]
    )
    return pd.DataFrame(rows), tests


def absolute_risk_contrast(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the absolute 0-to-180-minute probability difference and its support.

    A ratio alone does not say how much absolute risk changes, and a fitted
    curve says nothing about where the data actually are. The first table gives
    the standardised probabilities at both anchors and their difference with a
    delta-method interval; the second gives appearances and events by exposure
    band so readers can see which part of the curve is supported.
    """
    specification = next(
        item for item in exposure_specs() if item["exposure_id"] == "prior_minutes_7d"
    )
    result, work, _ = fit_exposure_model(
        frame, SAME_DAY_COL, "per_appearance", specification
    )
    reference = _standardization_reference(work)
    covariance = np.asarray(result.cov_params(), dtype=float)
    critical = NormalDist().inv_cdf(0.975)
    # 90 minutes is reported alongside 180 because three-quarters of the
    # cohort sits at or below it; the 0-to-180 contrast reaches into a region
    # holding roughly a tenth of the data.
    SUPPORTED_ANCHOR_MINUTES = 90.0
    anchors = {}
    for burden in (0.0, SUPPORTED_ANCHOR_MINUTES, REFERENCE_ANCHOR_MINUTES):
        estimate, gradient = _marginal_components(result, reference, burden)
        standard_error = float(np.sqrt(max(float(gradient @ covariance @ gradient), 0.0)))
        anchors[burden] = (estimate, gradient, standard_error)
    difference_gradient = anchors[REFERENCE_ANCHOR_MINUTES][1] - anchors[0.0][1]
    difference = anchors[REFERENCE_ANCHOR_MINUTES][0] - anchors[0.0][0]
    difference_se = float(
        np.sqrt(max(float(difference_gradient @ covariance @ difference_gradient), 0.0))
    )
    supported_gradient = anchors[SUPPORTED_ANCHOR_MINUTES][1] - anchors[0.0][1]
    supported_difference = anchors[SUPPORTED_ANCHOR_MINUTES][0] - anchors[0.0][0]
    supported_se = float(
        np.sqrt(max(float(supported_gradient @ covariance @ supported_gradient), 0.0))
    )
    target = (
        "appearances by established players, averaged over the observed calendar-phase "
        "distribution with continuous prior history held at its sample median "
        "(history_log_iqr = 0)"
    )
    rows = [
        {
            "quantity": "standardised_probability_at_0_minutes",
            "estimate_per_1000_appearances": 1000.0 * anchors[0.0][0],
            "ci_low_per_1000_appearances": 1000.0 * max(
                0.0, anchors[0.0][0] - critical * anchors[0.0][2]
            ),
            "ci_high_per_1000_appearances": 1000.0 * min(
                1.0, anchors[0.0][0] + critical * anchors[0.0][2]
            ),
        },
        {
            "quantity": "standardised_probability_at_180_minutes",
            "estimate_per_1000_appearances": 1000.0 * anchors[REFERENCE_ANCHOR_MINUTES][0],
            "ci_low_per_1000_appearances": 1000.0 * max(
                0.0,
                anchors[REFERENCE_ANCHOR_MINUTES][0]
                - critical * anchors[REFERENCE_ANCHOR_MINUTES][2],
            ),
            "ci_high_per_1000_appearances": 1000.0 * min(
                1.0,
                anchors[REFERENCE_ANCHOR_MINUTES][0]
                + critical * anchors[REFERENCE_ANCHOR_MINUTES][2],
            ),
        },
        {
            "quantity": "standardised_probability_at_90_minutes",
            "estimate_per_1000_appearances": 1000.0 * anchors[SUPPORTED_ANCHOR_MINUTES][0],
            "ci_low_per_1000_appearances": 1000.0 * max(
                0.0,
                anchors[SUPPORTED_ANCHOR_MINUTES][0]
                - critical * anchors[SUPPORTED_ANCHOR_MINUTES][2],
            ),
            "ci_high_per_1000_appearances": 1000.0 * min(
                1.0,
                anchors[SUPPORTED_ANCHOR_MINUTES][0]
                + critical * anchors[SUPPORTED_ANCHOR_MINUTES][2],
            ),
        },
        {
            "quantity": "absolute_difference_90_minus_0",
            "estimate_per_1000_appearances": 1000.0 * supported_difference,
            "ci_low_per_1000_appearances": 1000.0 * (supported_difference - critical * supported_se),
            "ci_high_per_1000_appearances": 1000.0 * (supported_difference + critical * supported_se),
        },
        {
            "quantity": "absolute_difference_180_minus_0",
            "estimate_per_1000_appearances": 1000.0 * difference,
            "ci_low_per_1000_appearances": 1000.0 * (difference - critical * difference_se),
            "ci_high_per_1000_appearances": 1000.0 * (difference + critical * difference_se),
        },
    ]
    contrast = pd.DataFrame(rows)
    contrast["standard_error_per_1000_appearances"] = [
        1000.0 * anchors[0.0][2],
        1000.0 * anchors[REFERENCE_ANCHOR_MINUTES][2],
        1000.0 * anchors[SUPPORTED_ANCHOR_MINUTES][2],
        1000.0 * supported_se,
        1000.0 * difference_se,
    ]
    contrast["target_population"] = target
    contrast["interval_method"] = "delta method on the standardised probability scale"
    contrast["n_rows"] = int(len(work))
    contrast["n_players"] = int(work[PLAYER_ID_COL].nunique())
    contrast["n_events"] = int(work[SAME_DAY_COL].sum())

    burden = pd.to_numeric(work["prior_minutes_7d"], errors="coerce")
    bands = pd.cut(
        burden,
        bins=[-0.001, 0.001, 45.0, 90.0, 135.0, 180.0, float(burden.max()) + 1.0],
        labels=list(EXPOSURE_BANDS),
        ordered=True,
    )
    support = (
        work.assign(exposure_band=bands)
        .groupby("exposure_band", observed=False)
        .agg(
            n_appearances=(SAME_DAY_COL, "size"),
            n_same_day_events=(SAME_DAY_COL, "sum"),
            n_players=(PLAYER_ID_COL, "nunique"),
        )
        .reset_index()
    )
    support["share_of_appearances"] = support["n_appearances"] / float(len(work))
    support["events_per_1000_appearances"] = np.where(
        support["n_appearances"].gt(0),
        1000.0 * support["n_same_day_events"] / support["n_appearances"],
        np.nan,
    )
    support["interpretation"] = (
        "observed support behind the fitted curve; sparse upper bands mean the "
        "fitted values there rest on few appearances"
    )
    return contrast, support


def denominator_contrast_metadata(
    same_day_summary: pd.DataFrame,
    lineup_summary: pd.DataFrame,
    lineup_completeness: pd.DataFrame,
) -> pd.DataFrame:
    """Record how the recorded-minute contrasts were estimated.

    The contrast is descriptive: same-day rows are compared with other
    appearances. Stating the estimator, the clustering unit, which seasons
    carry complete lineup data and how the lineup mix was standardised keeps
    the comparison readable as an association rather than an event-time
    measurement.
    """
    _require_columns(
        same_day_summary,
        [
            "event_minus_non_event_minutes",
            "bootstrap_unit",
            "bootstrap_replicates",
            "interval_method",
        ],
        "same-day minute summary",
    )
    _require_columns(
        lineup_summary, ["comparison", "event_minus_non_event_minutes"], "lineup summary"
    )
    _require_columns(
        lineup_completeness,
        ["dimension", "level", "lineup_known_percent"],
        "lineup completeness",
    )
    seasons = lineup_completeness[
        lineup_completeness["dimension"].eq("season")
        & pd.to_numeric(lineup_completeness["lineup_known_percent"], errors="coerce").ge(99.9)
    ]["level"].astype(str).tolist()
    headline = same_day_summary.iloc[0]
    standardized = lineup_summary[lineup_summary["comparison"].eq("lineup_standardized")]
    rows = [
        {
            "attribute": "estimator",
            "value": "difference in mean recorded minutes between same-day-report and other appearances",
        },
        {"attribute": "uncertainty_unit", "value": str(headline["bootstrap_unit"])},
        {"attribute": "interval_method", "value": str(headline["interval_method"])},
        {
            "attribute": "bootstrap_replicates",
            "value": str(int(headline["bootstrap_replicates"])),
        },
        {
            "attribute": "complete_lineup_seasons",
            "value": ", ".join(seasons) if seasons else "none",
        },
        {
            "attribute": "n_complete_lineup_seasons",
            "value": str(len(seasons)),
        },
        {
            "attribute": "lineup_standardisation",
            "value": (
                "starter and substitute differences reweighted to the pooled recorded "
                "lineup mix across complete-lineup seasons"
            ),
        },
        {
            "attribute": "headline_difference_minutes",
            "value": f"{float(headline['event_minus_non_event_minutes']):.2f}",
        },
        {
            "attribute": "standardised_difference_minutes",
            "value": (
                f"{float(standardized['event_minus_non_event_minutes'].iloc[0]):.2f}"
                if not standardized.empty
                else "unavailable"
            ),
        },
        {
            "attribute": "causal_reading",
            "value": (
                "not licensed: the data show that same-day reports were associated with "
                "shorter recorded appearances, not that the reported event ended them"
            ),
        },
    ]
    return pd.DataFrame(rows)


LINEUP_ROLE_COL = "lineup_role_model"
STARTER_ROLE = "starting_lineup"
SUBSTITUTE_ROLE = "substitute_list"
UNKNOWN_ROLE = "lineup_unavailable_or_other"


def recorded_minute_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe recorded appearance length by event status and lineup role.

    A difference in means can be produced by a small shift in every row or by
    a subset leaving the field early. Only the second is truncation. Quantiles
    separate the two, so this table is what licenses the word truncation
    anywhere in the paper.
    """
    _require_columns(frame, [SAME_DAY_COL, MINUTES_COL, LINEUP_ROLE_COL], "minute distribution frame")
    work = frame.copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)]
    rows = []
    for role in (STARTER_ROLE, SUBSTITUTE_ROLE, UNKNOWN_ROLE, "all"):
        subset = work if role == "all" else work[work[LINEUP_ROLE_COL].astype(str).eq(role)]
        for label, is_event in (("same_day_spell_start", 1), ("no_same_day_report", 0)):
            values = subset.loc[
                pd.to_numeric(subset[SAME_DAY_COL], errors="coerce").fillna(0).eq(is_event),
                MINUTES_COL,
            ].dropna()
            if values.empty:
                continue
            quantiles = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            rows.append(
                {
                    "lineup_role": role,
                    "event_status": label,
                    "n_appearances": int(len(values)),
                    "mean_minutes": float(values.mean()),
                    "p10_minutes": float(quantiles.loc[0.10]),
                    "p25_minutes": float(quantiles.loc[0.25]),
                    "median_minutes": float(quantiles.loc[0.50]),
                    "p75_minutes": float(quantiles.loc[0.75]),
                    "p90_minutes": float(quantiles.loc[0.90]),
                }
            )
    out = pd.DataFrame(rows)
    out["interpretation"] = (
        "truncation should appear as a shifted distribution among starters and "
        "as no shift among substitutes, whose recorded minutes are set by when "
        "they came on rather than when they left"
    )
    return out


def lineup_composition_by_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    """Show how squad role, and therefore appearance length, tracks exposure.

    This is the link the denominator argument depends on. If recent exposure
    predicts starting, then it predicts recorded minutes, and a per-minute
    model divides by a quantity that is itself downstream of the exposure.
    """
    _require_columns(
        frame,
        [LINEUP_ROLE_COL, MINUTES_COL, "prior_minutes_7d", SAME_DAY_COL],
        "lineup composition frame",
    )
    work = frame.copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)]
    bands = pd.cut(
        pd.to_numeric(work["prior_minutes_7d"], errors="coerce"),
        bins=[-0.001, 0.001, 45.0, 90.0, 135.0, 180.0, float("inf")],
        labels=list(EXPOSURE_BANDS),
        ordered=True,
    )
    out = (
        work.assign(
            exposure_band=bands,
            is_starter=work[LINEUP_ROLE_COL].astype(str).eq(STARTER_ROLE),
        )
        .groupby("exposure_band", observed=False)
        .agg(
            n_appearances=("is_starter", "size"),
            share_starting_lineup=("is_starter", "mean"),
            mean_recorded_minutes=(MINUTES_COL, "mean"),
            median_recorded_minutes=(MINUTES_COL, "median"),
        )
        .reset_index()
    )
    # Neither series rises monotonically, and saying that it does would be a
    # claim the table refutes. The reversals are measured here so the caption
    # and the gate read the same fact: what the exposure predicts is an average
    # gradient, not a step-by-step increase.
    for column, prefix in (
        ("share_starting_lineup", "share_starting"),
        ("mean_recorded_minutes", "mean_minutes"),
    ):
        series = pd.to_numeric(out[column], errors="coerce")
        steps = series.diff().dropna()
        out[f"{prefix}_is_monotonic"] = bool(steps.ge(0.0).all())
        out[f"{prefix}_n_reversals"] = int(steps.lt(0.0).sum())
        out[f"{prefix}_largest_reversal"] = float(-steps.min()) if len(steps) else np.nan
    out["interpretation"] = (
        "the share of starts and mean recorded minutes are higher at high "
        "exposure than at low, so recorded minutes are a function of the "
        "exposure under study; neither series is monotonic across bands, "
        "however, so the relationship is an average gradient rather than a "
        "step-by-step rise, and gamma summarises it as a linear slope"
    )
    return out


def denominator_by_lineup_role(frame: pd.DataFrame) -> pd.DataFrame:
    """Refit every denominator within lineup role.

    Holding squad role fixed removes most of the variation in appearance
    length. If the per-minute denominator matters because minutes track
    exposure through role, the three denominators should converge inside a
    role and diverge only when roles are pooled.
    """
    _require_columns(frame, [LINEUP_ROLE_COL, SAME_DAY_COL], "denominator role frame")
    specification = next(
        item for item in exposure_specs() if item["exposure_id"] == "prior_minutes_7d"
    )
    rows = []
    for role in ("all", STARTER_ROLE, SUBSTITUTE_ROLE, UNKNOWN_ROLE):
        subset = (
            frame if role == "all" else frame[frame[LINEUP_ROLE_COL].astype(str).eq(role)]
        )
        events = int(pd.to_numeric(subset[SAME_DAY_COL], errors="coerce").fillna(0).sum())
        if events < 10:
            rows.append(
                {
                    "lineup_role": role,
                    "denominator": "per_appearance",
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "log_estimate": np.nan,
                    "n_rows": int(len(subset)),
                    "n_players": int(subset[PLAYER_ID_COL].nunique()),
                    "n_events": events,
                    "estimable": False,
                }
            )
            continue
        for denominator in ("per_appearance", "observed_minutes", "fixed_90"):
            row, _ = exposure_model_row(subset, SAME_DAY_COL, denominator, specification)
            rows.append(
                {
                    "lineup_role": role,
                    "denominator": denominator,
                    "estimate": row["estimate"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "log_estimate": row["log_estimate"],
                    "n_rows": row["n_rows"],
                    # All inference is player-clustered, so the effective sample
                    # size behind a stratum is its player count, not its rows.
                    "n_players": row["n_players"],
                    "n_events": row["n_events"],
                    "estimable": True,
                }
            )
    out = pd.DataFrame(rows)
    # The attenuation is the within-role gap between the fixed-90 and
    # recorded-minute Poisson models: same family, same link, only the offset
    # differs, so nothing else can explain a difference between them.
    # Every estimable role above receives all three denominators, so both
    # offsets are present whenever the group is estimable at all.
    gaps = []
    for role, group in out[out["estimable"]].groupby("lineup_role", sort=False):
        indexed = group.set_index("denominator")
        base = float(indexed.loc["fixed_90", "log_estimate"])
        attenuation = base - float(indexed.loc["observed_minutes", "log_estimate"])
        gaps.append(
            {
                "lineup_role": role,
                "log_attenuation_fixed90_minus_recorded": attenuation,
                # Expressed against the estimate it acts on, so a stratum with a
                # smaller base association cannot masquerade as a smaller
                # denominator effect.
                "relative_attenuation": attenuation / base if base else np.nan,
            }
        )
    out = out.merge(pd.DataFrame(gaps), on="lineup_role", how="left")
    # An offset can only attenuate a coefficient to the extent that it varies.
    # Reporting the spread of the log offset inside each role shows why the
    # remedy works among starters and not elsewhere: the 90-minute ceiling
    # leaves starters almost no minute variation for the offset to act on.
    spread = []
    for role in ("all", STARTER_ROLE, SUBSTITUTE_ROLE, UNKNOWN_ROLE):
        subset = (
            frame if role == "all" else frame[frame[LINEUP_ROLE_COL].astype(str).eq(role)]
        )
        minutes = pd.to_numeric(subset[MINUTES_COL], errors="coerce")
        minutes = minutes[minutes.gt(0.0)]
        spread.append(
            {
                "lineup_role": role,
                "sd_log_recorded_minutes": (
                    float(np.log(minutes).std(ddof=1)) if len(minutes) > 1 else np.nan
                ),
                "iqr_recorded_minutes": (
                    float(minutes.quantile(0.75) - minutes.quantile(0.25))
                    if len(minutes)
                    else np.nan
                ),
            }
        )
    out = out.merge(pd.DataFrame(spread), on="lineup_role", how="left")
    out["interpretation"] = (
        "a large pooled attenuation beside a near-zero within-starter "
        "attenuation identifies squad-role composition, not event truncation, "
        "as the reason per-minute and per-appearance answers differ; squad "
        "role is a coarse remedy rather than a general one, because the "
        "attenuation persists within substitutes and within rows of unknown "
        "lineup status, tracking the spread of the log offset in each stratum"
    )
    return out


def squad_role_association_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Ask whether squad role also drives the numerator, not just the offset.

    The denominator analysis shows recorded minutes track exposure through
    squad role. The obvious next question is whether the per-appearance
    association does too, since a player with high recent minutes is by
    construction more likely to be an established starter. This refits the
    reference per-appearance model unadjusted, adjusted for lineup role, and
    within each role, so a reader can see how much of the association is
    carried by role composition.
    """
    _require_columns(frame, [LINEUP_ROLE_COL, SAME_DAY_COL], "role association frame")
    specification = next(
        item for item in exposure_specs() if item["exposure_id"] == "prior_minutes_7d"
    )
    rows: list[dict[str, Any]] = []

    unadjusted, _ = exposure_model_row(
        frame, SAME_DAY_COL, "per_appearance", specification
    )
    rows.append({**unadjusted, "analysis": "pooled, unadjusted for squad role"})

    # Adjusting is the like-for-like comparison: same rows as the unadjusted
    # fit, one extra categorical term, so any movement is role composition.
    work = frame.copy()
    work["lineup_role_term"] = work[LINEUP_ROLE_COL].astype(str)
    adjusted_result, adjusted_work, term = fit_exposure_model(
        work, SAME_DAY_COL, "per_appearance", specification
    )
    formula = f"{adjusted_result.model.formula} + C(lineup_role_term)"
    adjusted = smf.glm(
        formula, data=adjusted_work, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": adjusted_work[PLAYER_ID_COL]})
    estimate = float(adjusted.params[term])
    standard_error = float(adjusted.bse[term])
    critical = NormalDist().inv_cdf(0.975)
    rows.append(
        {
            **unadjusted,
            "analysis": "pooled, adjusted for squad role",
            "estimate": float(np.exp(estimate)),
            "ci_low": float(np.exp(estimate - critical * standard_error)),
            "ci_high": float(np.exp(estimate + critical * standard_error)),
            "log_estimate": estimate,
            "standard_error": standard_error,
            "p_value": _normal_p_value(estimate, standard_error),
            "formula": formula,
        }
    )

    for role in (STARTER_ROLE, SUBSTITUTE_ROLE, UNKNOWN_ROLE):
        subset = frame[frame[LINEUP_ROLE_COL].astype(str).eq(role)]
        events = int(pd.to_numeric(subset[SAME_DAY_COL], errors="coerce").fillna(0).sum())
        if events < 10:
            rows.append(
                {
                    **{key: np.nan for key in unadjusted},
                    "analysis": f"within {role}",
                    "n_rows": int(len(subset)),
                    "n_events": events,
                }
            )
            continue
        row, _ = exposure_model_row(
            subset, SAME_DAY_COL, "per_appearance", specification
        )
        rows.append({**row, "analysis": f"within {role}"})

    out = pd.DataFrame(rows)
    out["interpretation"] = (
        "the per-appearance association is reported unadjusted, adjusted for "
        "squad role and within each role, because the same composition that "
        "contaminates the minute denominator could also generate the "
        "association itself; a materially weaker within-starter estimate is "
        "reported rather than set aside"
    )
    return out


def role_adjusted_denominator_refit(frame: pd.DataFrame) -> pd.DataFrame:
    """Test whether adjusting for squad role repairs the minute denominator.

    Stratifying to starters removes the attenuation because it restricts to
    rows whose offset barely varies. Adjusting is a different operation: a
    categorical role term enters the linear predictor, while the offset keeps
    its full within-role variation. This refits fixed-90 against recorded
    minutes on identical rows with a role term in both, so the remaining
    attenuation says whether a covariate can stand in for stratification.
    """
    required = [SAME_DAY_COL, MINUTES_COL, PLAYER_ID_COL, LINEUP_ROLE_COL]
    _require_columns(frame, required, "role adjusted refit frame")
    specification = next(
        item for item in exposure_specs() if item["exposure_id"] == "prior_minutes_7d"
    )
    rows = []
    for adjusted in (False, True):
        fitted: dict[str, float] = {}
        for denominator in ("fixed_90", "observed_minutes"):
            result, work, term = fit_exposure_model(
                frame, SAME_DAY_COL, denominator, specification
            )
            if adjusted:
                work = work.assign(
                    lineup_role_term=work[LINEUP_ROLE_COL].astype(str)
                )
                offset = (
                    np.log(
                        pd.to_numeric(work[MINUTES_COL], errors="coerce").clip(lower=1.0)
                    )
                    if denominator == "observed_minutes"
                    else pd.Series(np.log(90.0), index=work.index)
                )
                result = smf.glm(
                    f"{result.model.formula} + C(lineup_role_term)",
                    data=work,
                    family=sm.families.Poisson(),
                    offset=offset,
                ).fit(cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]})
            estimate = float(result.params[term])
            standard_error = float(result.bse[term])
            critical = NormalDist().inv_cdf(0.975)
            fitted[denominator] = estimate
            rows.append(
                {
                    "model": "adjusted for squad role" if adjusted else "unadjusted",
                    "denominator": denominator,
                    "estimate": float(np.exp(estimate)),
                    "ci_low": float(np.exp(estimate - critical * standard_error)),
                    "ci_high": float(np.exp(estimate + critical * standard_error)),
                    "log_estimate": estimate,
                    "n_rows": int(len(work)),
                    "n_events": int(work[SAME_DAY_COL].sum()),
                    "log_attenuation_fixed90_minus_recorded": np.nan,
                }
            )
        gap = fitted["fixed_90"] - fitted["observed_minutes"]
        for row in rows[-2:]:
            row["log_attenuation_fixed90_minus_recorded"] = gap
    out = pd.DataFrame(rows)
    out["interpretation"] = (
        "a categorical role term enters the linear predictor while the offset "
        "keeps its within-role variation, so adjusting cannot do what "
        "restricting to starters does; the attenuation that survives "
        "adjustment is the evidence for that"
    )
    return out


def run_in_exclusion_comparison(
    unrestricted: pd.DataFrame,
    threshold: float = 900.0,
) -> pd.DataFrame:
    """Describe the players the run-in removes against those it keeps.

    The established-player run-in is an eligibility choice that drops a fifth
    of the source players, so the paper should say who they were rather than
    assert it.
    """
    required = [PLAYER_ID_COL, "prior_minutes_played", "age_years", SAME_DAY_COL]
    _require_columns(unrestricted, required, "run-in exclusion frame")
    work = unrestricted.copy()
    work["prior_minutes_played"] = pd.to_numeric(
        work["prior_minutes_played"], errors="coerce"
    )
    retained_players = set(
        work.loc[work["prior_minutes_played"].ge(float(threshold)), PLAYER_ID_COL]
    )
    work["population"] = np.where(
        work[PLAYER_ID_COL].isin(retained_players),
        "retained at the run-in",
        "excluded by the run-in",
    )
    rows = []
    for population, group in work.groupby("population", sort=True):
        by_player = group.groupby(PLAYER_ID_COL)
        rows.append(
            {
                "population": population,
                "n_players": int(group[PLAYER_ID_COL].nunique()),
                "n_appearances": int(len(group)),
                "median_age_years": float(
                    pd.to_numeric(group["age_years"], errors="coerce").median()
                ),
                "median_appearances_per_player": float(by_player.size().median()),
                "median_prior_minutes_played": float(
                    by_player["prior_minutes_played"].max().median()
                ),
                "n_same_day_events": int(
                    pd.to_numeric(group[SAME_DAY_COL], errors="coerce").fillna(0).sum()
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["run_in_minutes"] = float(threshold)
    out["interpretation"] = (
        "the run-in restricts the cohort to established players; this records "
        "who it removes so the restriction can be judged rather than trusted"
    )
    return out


def exposure_window_gradient(summary: pd.DataFrame) -> pd.DataFrame:
    """Separate the effect-size and precision gradients across windows.

    Adjusted p values fall monotonically as the cumulative window lengthens,
    which invites the reading that longer windows carry a stronger signal. The
    point estimates do the opposite. Reporting both columns beside each other
    prevents that misreading.
    """
    _require_columns(
        summary,
        ["exposure_id", "estimate", "standard_error", "holm_p_value_63_model_family"],
        "window gradient summary",
    )
    windows = ("prior_minutes_3d", "prior_minutes_5d", "prior_minutes_7d",
               "prior_minutes_10d", "prior_minutes_14d")
    subset = summary[summary["exposure_id"].isin(windows)].copy()
    subset["window_days"] = (
        subset["exposure_id"].str.extract(r"(\d+)d$")[0].astype(int)
    )
    subset = subset.sort_values("window_days").reset_index(drop=True)
    out = subset[
        [
            "exposure_id",
            "window_days",
            "estimate",
            "ci_low",
            "ci_high",
            "standard_error",
            "p_value",
            "holm_p_value_63_model_family",
            "n_events",
        ]
    ].copy()
    estimates = out["estimate"].to_numpy(dtype=float)
    errors = out["standard_error"].to_numpy(dtype=float)
    out["interpretation"] = (
        "the adjusted p value falls monotonically with window length while the "
        f"odds ratio does not (peak {estimates.max():.2f} at "
        f"{int(out.loc[int(np.argmax(estimates)), 'window_days'])} days, "
        f"{estimates[-1]:.2f} at {int(out['window_days'].iloc[-1])} days) and "
        f"the standard error falls from {errors[0]:.3f} to {errors[-1]:.3f}; "
        "the gradient across windows is precision, not effect size"
    )
    return out


IMPUTATION_SCHEMES: tuple[str, ...] = (
    "role_mean",
    "role_median",
    "role_p75",
    "role_exposure_band_mean",
    "role_season_mean",
)


def _imputation_keys(work: pd.DataFrame, scheme: str) -> pd.Series:
    """Return the grouping key an imputation scheme matches event rows on."""
    roles = work[LINEUP_ROLE_COL].astype(str)
    if scheme in ("role_mean", "role_median", "role_p75"):
        return roles
    if scheme == "role_exposure_band_mean":
        bands = pd.cut(
            pd.to_numeric(work["prior_minutes_7d"], errors="coerce"),
            bins=[-0.001, 0.001, 45.0, 90.0, 135.0, 180.0, float("inf")],
            labels=list(EXPOSURE_BANDS),
            ordered=True,
        ).astype(str)
        return roles.str.cat(bands, sep="|")
    if scheme == "role_season_mean":
        return roles.str.cat(work["season_start"].astype(str), sep="|")
    raise ValueError(f"Unknown imputation scheme: {scheme}")


def _untruncated_minutes(work: pd.DataFrame, scheme: str = "role_mean") -> pd.Series:
    """Return recorded minutes with outcome truncation removed.

    Each event row is given a summary of the recorded minutes of non-event
    appearances that match it on the scheme's key, which strips the truncation
    and leaves every other property of the denominator intact. The scheme is a
    modelling choice, so every quantity derived from it is reported across the
    alternatives rather than at one setting.
    """
    minutes = pd.to_numeric(work[MINUTES_COL], errors="coerce").clip(lower=1.0)
    events = pd.to_numeric(work[SAME_DAY_COL], errors="coerce").fillna(0).eq(1)
    keys = _imputation_keys(work, scheme)
    reference = work.loc[~events].groupby(keys[~events].values)[MINUTES_COL]
    if scheme == "role_median":
        summary = reference.median()
    elif scheme == "role_p75":
        summary = reference.quantile(0.75)
    else:
        summary = reference.mean()
    out = minutes.copy()
    # A key with no non-event rows falls back to the overall non-event mean
    # rather than leaving the denominator missing.
    out.loc[events] = (
        keys[events].map(summary).astype(float).fillna(float(minutes[~events].mean()))
    )
    return out.clip(lower=1.0)


def direct_truncation_refit(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure the truncation contribution without any approximation.

    The first-order identity predicts the attenuation from a minute
    denominator, but it is an expansion and can misstate the magnitude. This
    refits the same Poisson model twice, changing only the offset: once on
    recorded minutes and once on minutes with truncation removed. The gap
    between those two coefficients is exactly what outcome truncation costs,
    with no expansion anywhere.
    """
    required = [
        SAME_DAY_COL,
        MINUTES_COL,
        PLAYER_ID_COL,
        LINEUP_ROLE_COL,
        "prior_minutes_7d",
        HISTORY_MODEL_COL,
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "direct truncation refit frame")
    work = frame.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    if work.empty or int(work[SAME_DAY_COL].sum()) <= 0:
        raise ValueError("no estimable rows for the direct truncation refit")

    calendar = " + ".join(CALENDAR_TERMS)
    formula = (
        f"{SAME_DAY_COL} ~ I(prior_minutes_7d / 90.0) + {HISTORY_MODEL_COL} + {calendar}"
    )
    critical = NormalDist().inv_cdf(0.975)
    offsets = {
        "recorded_minutes": np.log(
            pd.to_numeric(work[MINUTES_COL], errors="coerce").clip(lower=1.0)
        ),
        "untruncated_minutes": np.log(_untruncated_minutes(work)),
        "fixed_90": pd.Series(np.log(90.0), index=work.index),
    }
    rows = []
    coefficients = {}
    for offset_id, offset in offsets.items():
        result = smf.glm(
            formula, data=work, family=sm.families.Poisson(), offset=offset
        ).fit(cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]})
        term = next(name for name in result.params.index if "prior_minutes_7d" in name)
        estimate = float(result.params[term])
        standard_error = float(result.bse[term])
        coefficients[offset_id] = estimate
        rows.append(
            {
                "offset": offset_id,
                "estimate": float(np.exp(estimate)),
                "ci_low": float(np.exp(estimate - critical * standard_error)),
                "ci_high": float(np.exp(estimate + critical * standard_error)),
                "log_estimate": estimate,
                "n_rows": int(len(work)),
                "n_events": int(work[SAME_DAY_COL].sum()),
            }
        )
    out = pd.DataFrame(rows)
    # Everything except truncation is common to these two fits, so their gap
    # is the truncation contribution itself rather than a prediction of it.
    out["log_gap_untruncated_minus_recorded"] = (
        coefficients["untruncated_minutes"] - coefficients["recorded_minutes"]
    )
    out["log_attenuation_fixed90_minus_recorded"] = (
        coefficients["fixed_90"] - coefficients["recorded_minutes"]
    )
    total = coefficients["fixed_90"] - coefficients["recorded_minutes"]
    out["truncation_share_of_attenuation"] = (
        (coefficients["untruncated_minutes"] - coefficients["recorded_minutes"]) / total
        if total
        else np.nan
    )
    out["interpretation"] = (
        "changing only the offset isolates the denominator; the recorded "
        "against untruncated comparison measures outcome truncation directly, "
        "without the first-order expansion the gamma decomposition relies on"
    )
    return out


def case_restricted_exposure_bias(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantify where outcome truncation does bite: analyses restricted to cases.

    Truncation moves a whole-cohort rate by almost nothing because events are
    rare. Any quantity computed over event appearances alone has no such
    dilution, so this reports the same distortion on that scale.
    """
    _require_columns(
        frame, [SAME_DAY_COL, MINUTES_COL, LINEUP_ROLE_COL], "case-restricted frame"
    )
    work = frame.copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    events = pd.to_numeric(work[SAME_DAY_COL], errors="coerce").fillna(0).eq(1)
    if not events.any():
        raise ValueError("no same-day spell starts available")
    untruncated = _untruncated_minutes(work)
    observed_mean = float(work.loc[events, MINUTES_COL].mean())
    untruncated_mean = float(untruncated[events].mean())
    cohort_observed = float(work[MINUTES_COL].sum())
    cohort_untruncated = float(untruncated.sum())
    rows = [
        {
            "quantity": "mean_recorded_minutes_on_event_appearances",
            "observed": observed_mean,
            "truncation_removed": untruncated_mean,
            "ratio_observed_to_untruncated": observed_mean / untruncated_mean,
        },
        {
            "quantity": "total_minutes_whole_cohort",
            "observed": cohort_observed,
            "truncation_removed": cohort_untruncated,
            "ratio_observed_to_untruncated": cohort_observed / cohort_untruncated,
        },
    ]
    out = pd.DataFrame(rows)
    out["percent_understated"] = 100.0 * (1.0 - out["ratio_observed_to_untruncated"])
    out["n_event_appearances"] = int(events.sum())
    out["n_appearances"] = int(len(work))
    out["interpretation"] = (
        "truncation is diluted to nothing in a whole-cohort denominator but "
        "not in any quantity restricted to the appearances that carry an event"
    )
    return out


def truncation_imputation_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute every truncation quantity under alternative imputations.

    The counterfactual minutes are a modelling choice, and the case-restricted
    distortion is a mean against an imputed mean, so that choice is the answer
    rather than a detail of it. This repeats the case-restricted comparison,
    the cohort comparison and the direct offset refit under each scheme so the
    reader sees how much of each number is data and how much is the choice.
    """
    required = [
        SAME_DAY_COL,
        MINUTES_COL,
        PLAYER_ID_COL,
        LINEUP_ROLE_COL,
        "prior_minutes_7d",
        "season_start",
        HISTORY_MODEL_COL,
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "imputation sensitivity frame")
    work = frame.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    events = pd.to_numeric(work[SAME_DAY_COL], errors="coerce").fillna(0).eq(1)
    if not events.any():
        raise ValueError("no same-day spell starts available")

    calendar = " + ".join(CALENDAR_TERMS)
    formula = (
        f"{SAME_DAY_COL} ~ I(prior_minutes_7d / 90.0) + {HISTORY_MODEL_COL} + {calendar}"
    )
    minutes = work[MINUTES_COL].clip(lower=1.0)
    recorded_offset = np.log(minutes)

    def _exposure_coefficient(offset: pd.Series) -> float:
        result = smf.glm(
            formula, data=work, family=sm.families.Poisson(), offset=offset
        ).fit()
        term = next(name for name in result.params.index if "prior_minutes_7d" in name)
        return float(result.params[term])

    baseline = _exposure_coefficient(recorded_offset)
    fixed_90 = _exposure_coefficient(pd.Series(np.log(90.0), index=work.index))
    total_attenuation = fixed_90 - baseline

    rows = []
    for scheme in IMPUTATION_SCHEMES:
        untruncated = _untruncated_minutes(work, scheme)
        case_observed = float(minutes[events].mean())
        case_untruncated = float(untruncated[events].mean())
        cohort_observed = float(minutes.sum())
        cohort_untruncated = float(untruncated.sum())
        gap = _exposure_coefficient(np.log(untruncated)) - baseline
        rows.append(
            {
                "imputation_scheme": scheme,
                "mean_minutes_event_observed": case_observed,
                "mean_minutes_event_untruncated": case_untruncated,
                "case_restricted_percent_understated": 100.0
                * (1.0 - case_observed / case_untruncated),
                "cohort_percent_understated": 100.0
                * (1.0 - cohort_observed / cohort_untruncated),
                "log_gap_untruncated_minus_recorded": gap,
                "truncation_share_of_attenuation": (
                    gap / total_attenuation if total_attenuation else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["n_event_appearances"] = int(events.sum())
    out["interpretation"] = (
        "the case-restricted distortion is the quantity most exposed to the "
        "imputation, so it is reported as a range across schemes rather than "
        "at one setting; the attribution is stable because every scheme leaves "
        "truncation explaining a negligible share of the attenuation"
    )
    return out


def attenuation_bootstrap(
    frame: pd.DataFrame,
    replicates: int = 1000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Percentile intervals for the denominator attenuation, pooled and by role.

    The attenuation is the difference between two coefficients fitted to the
    same rows, and the mechanism argument compares that difference across
    strata. Neither quantity carries uncertainty from a single fit, so we
    resample players with replacement, refit both offsets in every stratum and
    take percentile intervals, including on the pooled-to-starter ratio that
    the composition claim rests on.
    """
    required = [
        SAME_DAY_COL,
        MINUTES_COL,
        PLAYER_ID_COL,
        LINEUP_ROLE_COL,
        "prior_minutes_7d",
        HISTORY_MODEL_COL,
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "attenuation bootstrap frame")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    work = frame.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    if work.empty or int(work[SAME_DAY_COL].sum()) <= 0:
        raise ValueError("no estimable rows for the attenuation bootstrap")

    calendar = " + ".join(CALENDAR_TERMS)
    formula = (
        f"{SAME_DAY_COL} ~ I(prior_minutes_7d / 90.0) + {HISTORY_MODEL_COL} + {calendar}"
    )
    # Every stratum the remedy is claimed over needs an interval, including the
    # rows where lineup status is missing and the remedy therefore cannot be
    # applied at all.
    strata = ("all", STARTER_ROLE, SUBSTITUTE_ROLE, UNKNOWN_ROLE)

    def _attenuations(sample: pd.DataFrame) -> dict[str, float]:
        """Return absolute and relative attenuation in each stratum."""
        roles = sample[LINEUP_ROLE_COL].astype(str)
        values: dict[str, float] = {}

        # The attribution to truncation is a difference between two fits on
        # identical rows, changing only whether the offset carries truncated or
        # untruncated minutes. It is the quantity the whole paper turns on, so
        # it is resampled alongside the attenuations rather than reported bare.
        recorded_minutes = sample[MINUTES_COL].clip(lower=1.0)
        try:
            truncated_fit = smf.glm(
                formula, data=sample, family=sm.families.Poisson(),
                offset=np.log(recorded_minutes),
            ).fit()
            untruncated_fit = smf.glm(
                formula, data=sample, family=sm.families.Poisson(),
                offset=np.log(_untruncated_minutes(sample)),
            ).fit()
            focal = next(
                n for n in truncated_fit.params.index if "prior_minutes_7d" in n
            )
            values["truncation_attribution_absolute"] = float(
                untruncated_fit.params[focal] - truncated_fit.params[focal]
            )
        except Exception:  # pragma: no cover - degenerate resample
            values["truncation_attribution_absolute"] = np.nan

        # Adjusting for squad role is offered as an alternative to stratifying
        # by it, so the attenuation it leaves is argued from and needs the same
        # interval as the stratified ones. Both fits carry the role term, so
        # only the offset differs between them.
        try:
            adjusted_frame = sample.assign(lineup_role_term=roles)
            adjusted_formula = f"{formula} + C(lineup_role_term)"
            adjusted_recorded = smf.glm(
                adjusted_formula, data=adjusted_frame, family=sm.families.Poisson(),
                offset=np.log(recorded_minutes),
            ).fit()
            adjusted_fixed = smf.glm(
                adjusted_formula, data=adjusted_frame, family=sm.families.Poisson(),
                offset=pd.Series(np.log(90.0), index=adjusted_frame.index),
            ).fit()
            focal = next(
                n for n in adjusted_recorded.params.index if "prior_minutes_7d" in n
            )
            values["role_adjusted_absolute"] = float(
                adjusted_fixed.params[focal] - adjusted_recorded.params[focal]
            )
        except Exception:  # pragma: no cover - degenerate resample
            values["role_adjusted_absolute"] = np.nan

        for stratum in strata:
            subset = sample if stratum == "all" else sample[roles.eq(stratum)]
            if int(subset[SAME_DAY_COL].sum()) < 10:
                values[stratum] = np.nan
                values[f"{stratum}_relative"] = np.nan
                continue
            minutes = subset[MINUTES_COL].clip(lower=1.0)
            try:
                recorded = smf.glm(
                    formula, data=subset, family=sm.families.Poisson(),
                    offset=np.log(minutes),
                ).fit()
                fixed = smf.glm(
                    formula, data=subset, family=sm.families.Poisson(),
                    offset=pd.Series(np.log(90.0), index=subset.index),
                ).fit()
            except Exception:  # pragma: no cover - degenerate resample
                values[stratum] = np.nan
                values[f"{stratum}_relative"] = np.nan
                continue
            term = next(n for n in recorded.params.index if "prior_minutes_7d" in n)
            base = float(fixed.params[term])
            gap = base - float(recorded.params[term])
            values[stratum] = gap
            values[f"{stratum}_relative"] = gap / base if base else np.nan
        return values

    point = _attenuations(work)
    players = work[PLAYER_ID_COL].to_numpy()
    unique_players = np.unique(players)
    index_by_player = {p: np.flatnonzero(players == p) for p in unique_players}
    generator = np.random.default_rng(seed)
    draws: list[dict[str, float]] = []
    for _ in range(replicates):
        chosen = generator.choice(unique_players, size=len(unique_players), replace=True)
        positions = np.concatenate([index_by_player[p] for p in chosen])
        draws.append(_attenuations(work.iloc[positions]))

    replicate_frame = pd.DataFrame(draws)
    # The composition claim is that pooling inflates the denominator effect.
    # The difference of absolute attenuations is the stable way to test that;
    # the ratio of relative attenuations divides by a within-starter base
    # estimate that is itself near the null, so it is retained only to show
    # how unstable it is.
    replicate_frame["pooled_minus_starter_absolute"] = (
        replicate_frame["all"] - replicate_frame[STARTER_ROLE]
    )
    replicate_frame["pooled_over_starter_relative"] = (
        replicate_frame["all_relative"] / replicate_frame[f"{STARTER_ROLE}_relative"]
    )
    point["pooled_minus_starter_absolute"] = point["all"] - point[STARTER_ROLE]
    # A NaN is truthy, so the guard has to test for a usable number rather than
    # for presence.
    starter_relative = point.get(f"{STARTER_ROLE}_relative", np.nan)
    point["pooled_over_starter_relative"] = (
        point["all_relative"] / starter_relative
        if np.isfinite(starter_relative) and starter_relative != 0.0
        else np.nan
    )

    labels = {
        "all": "pooled absolute attenuation",
        f"{STARTER_ROLE}": "absolute attenuation within starters",
        f"{SUBSTITUTE_ROLE}": "absolute attenuation within substitutes",
        f"{UNKNOWN_ROLE}": "absolute attenuation where lineup status is unknown",
        "pooled_minus_starter_absolute": "pooled minus within-starter absolute attenuation",
        "all_relative": "pooled attenuation as a share of the fixed-90 estimate",
        f"{STARTER_ROLE}_relative": "relative attenuation within starters",
        f"{SUBSTITUTE_ROLE}_relative": "relative attenuation within substitutes",
        f"{UNKNOWN_ROLE}_relative": "relative attenuation where lineup status is unknown",
        "pooled_over_starter_relative": "ratio of pooled to within-starter relative attenuation",
        "truncation_attribution_absolute": (
            "coefficient gap between recorded and untruncated offsets, the part "
            "of the attenuation outcome truncation can explain"
        ),
        "role_adjusted_absolute": (
            "attenuation surviving when squad role is adjusted for rather than "
            "stratified by, on all appearances"
        ),
    }
    rows = []
    for quantity, label in labels.items():
        values = pd.to_numeric(replicate_frame[quantity], errors="coerce").dropna()
        rows.append(
            {
                "quantity": quantity,
                "description": label,
                "estimate": float(point.get(quantity, np.nan)),
                "ci_low": float(values.quantile(0.025)) if len(values) else np.nan,
                "ci_high": float(values.quantile(0.975)) if len(values) else np.nan,
                "n_replicates_estimable": int(len(values)),
            }
        )
    out = pd.DataFrame(rows)
    out["n_replicates_requested"] = int(replicates)
    out["resampling_unit"] = "player"
    out["interpretation"] = (
        "percentile intervals from resampling players; the difference of "
        "absolute attenuations is the stable test of the composition claim, "
        "while the ratio of relative attenuations is reported to show that it "
        "is not, because its denominator is a within-starter estimate that "
        "sits near the null"
    )
    return out


def denominator_attenuation_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    """Attribute the per-minute attenuation to its two possible causes.

    For a Poisson model, ``log E[y] = log(m) + a + b_off x`` and
    ``log E[y] = a' + b_app x`` imply ``b_off = b_app - gamma`` to first order,
    where gamma is the slope of ``log(m)`` on the exposure. So the attenuation
    caused by a minute denominator is predicted by gamma alone.

    Gamma has two parts. One is outcome truncation: event appearances are
    recorded short. The other is ordinary exposure-dependence: players with
    more recent minutes start more often and therefore play longer. Replacing
    event-row minutes with the non-event mean for the same lineup role removes
    the first and leaves the second, so the difference between the two gammas
    is the share of the attenuation that truncation can actually explain.

    The identity is an expansion, so this also records how far it sits from
    the attenuation actually observed. ``direct_truncation_refit`` measures the
    same quantity without any expansion.
    """
    required = [
        SAME_DAY_COL,
        MINUTES_COL,
        PLAYER_ID_COL,
        LINEUP_ROLE_COL,
        "prior_minutes_7d",
        HISTORY_MODEL_COL,
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "attenuation decomposition frame")
    work = frame.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    if work.empty or int(work[SAME_DAY_COL].sum()) <= 0:
        raise ValueError("no estimable rows for the attenuation decomposition")
    work["exposure_per_90"] = pd.to_numeric(work["prior_minutes_7d"], errors="coerce") / 90.0
    minutes = work[MINUTES_COL].clip(lower=1.0)
    work["log_recorded_minutes"] = np.log(minutes)

    roles = work[LINEUP_ROLE_COL].astype(str)
    events = pd.to_numeric(work[SAME_DAY_COL], errors="coerce").fillna(0).eq(1)
    untruncated = _untruncated_minutes(work)
    work["log_untruncated_minutes"] = np.log(untruncated)

    calendar = " + ".join(CALENDAR_TERMS)
    covariates = f"exposure_per_90 + {HISTORY_MODEL_COL} + {calendar}"
    critical = NormalDist().inv_cdf(0.975)

    def _gamma(frame_in: pd.DataFrame, response: str) -> tuple[float, float, float]:
        """Fit one gamma with player-clustered errors and return it with bounds.

        Every gamma in this table goes through here, pooled and within-stratum
        alike, so the four values a reader compares rest on the same variance
        estimator. Clustering matters because a player contributes many
        appearances, and the pooled and stratified fits would otherwise sit on
        different footings.
        """
        fit = smf.ols(f"{response} ~ {covariates}", data=frame_in).fit(
            cov_type="cluster", cov_kwds={"groups": frame_in[PLAYER_ID_COL]}
        )
        estimate = float(fit.params["exposure_per_90"])
        error = float(fit.bse["exposure_per_90"])
        return estimate, estimate - critical * error, estimate + critical * error

    gamma, gamma_low, gamma_high = _gamma(work, "log_recorded_minutes")
    gamma_untruncated, gamma_cf_low, gamma_cf_high = _gamma(
        work, "log_untruncated_minutes"
    )

    # Gamma is a property of the denominator alone: it needs rows, not events.
    # That makes it the right quantity for the composition argument in strata
    # where the outcome is too sparse to estimate an association, and it is
    # reported for every stratum the remedy is claimed over.
    within_role: dict[str, tuple[float, float, float]] = {}
    for role in (STARTER_ROLE, SUBSTITUTE_ROLE, UNKNOWN_ROLE):
        subset = work[roles.eq(role)]
        if len(subset) < 50:
            within_role[role] = (np.nan, np.nan, np.nan)
            continue
        within_role[role] = _gamma(subset, "log_recorded_minutes")

    specification = next(
        item for item in exposure_specs() if item["exposure_id"] == "prior_minutes_7d"
    )
    fitted = {}
    for denominator in ("per_appearance", "observed_minutes", "fixed_90"):
        row, _ = exposure_model_row(frame, SAME_DAY_COL, denominator, specification)
        fitted[denominator] = row
    observed_attenuation = float(
        fitted["fixed_90"]["log_estimate"] - fitted["observed_minutes"]["log_estimate"]
    )

    minutes_lost = float((untruncated[events] - minutes[events]).sum())
    total_minutes = float(minutes.sum())

    rows = [
        {
            "quantity": "gamma_log_minutes_on_exposure",
            "value": gamma,
            "ci_low": gamma_low,
            "ci_high": gamma_high,
            "note": (
                "slope of log recorded minutes on 90 previous-seven-day minutes; "
                "the first-order predicted attenuation from a minute denominator"
            ),
        },
        {
            "quantity": "gamma_with_truncation_removed",
            "value": gamma_untruncated,
            "ci_low": gamma_cf_low,
            "ci_high": gamma_cf_high,
            "note": "same slope after giving event rows the non-event mean for their lineup role",
        },
        {
            "quantity": "gamma_attributable_to_outcome_truncation",
            "value": gamma - gamma_untruncated,
            "note": "the part of the predicted attenuation that event truncation can explain",
        },
        {
            "quantity": "truncation_share_of_gamma",
            "value": (gamma - gamma_untruncated) / gamma if gamma else np.nan,
            "note": "proportion of the minute-denominator attenuation caused by outcome truncation",
        },
        {
            "quantity": "gamma_within_starting_lineup",
            "value": within_role[STARTER_ROLE][0],
            "ci_low": within_role[STARTER_ROLE][1],
            "ci_high": within_role[STARTER_ROLE][2],
            "note": "exposure-dependence of appearance length among starters only",
        },
        {
            "quantity": "gamma_within_substitute_list",
            "value": within_role[SUBSTITUTE_ROLE][0],
            "ci_low": within_role[SUBSTITUTE_ROLE][1],
            "ci_high": within_role[SUBSTITUTE_ROLE][2],
            "note": "exposure-dependence of appearance length among substitutes only",
        },
        {
            "quantity": "gamma_within_lineup_unavailable_or_other",
            "value": within_role[UNKNOWN_ROLE][0],
            "ci_low": within_role[UNKNOWN_ROLE][1],
            "ci_high": within_role[UNKNOWN_ROLE][2],
            "note": (
                "exposure-dependence of appearance length where lineup status is "
                "unknown, the stratum in which stratifying by role is impossible"
            ),
        },
        {
            "quantity": "observed_log_attenuation_fixed90_minus_recorded",
            "value": observed_attenuation,
            "note": "measured within-Poisson attenuation; only the offset differs between these models",
        },
        {
            "quantity": "gamma_over_observed_attenuation",
            "value": gamma / observed_attenuation if observed_attenuation else np.nan,
            "note": (
                "the first-order identity is an expansion: this records how far "
                "its prediction sits from the attenuation actually observed"
            ),
        },
        {
            "quantity": "relative_attenuation_pooled",
            "value": (
                observed_attenuation / float(fitted["fixed_90"]["log_estimate"])
                if fitted["fixed_90"]["log_estimate"]
                else np.nan
            ),
            "note": (
                "attenuation as a share of the fixed-90 estimate it acts on, so "
                "strata with different base estimates can be compared"
            ),
        },
        {
            "quantity": "minutes_lost_to_truncation",
            "value": minutes_lost,
            "note": "total recorded minutes missing from event appearances",
        },
        {
            "quantity": "truncation_share_of_all_recorded_minutes",
            "value": minutes_lost / total_minutes if total_minutes else np.nan,
            "note": "why rare-event truncation cannot move a denominator built from every appearance",
        },
        {
            "quantity": "rate_inflation_factor_from_truncation",
            "value": (total_minutes + minutes_lost) / total_minutes if total_minutes else np.nan,
            "note": "multiplicative inflation of any per-minute rate caused by truncation alone",
        },
    ]
    out = pd.DataFrame(rows)
    # Only the gammas carry bounds. The remaining quantities are ratios and
    # totals derived from them, so a blank interval here means "not an
    # estimated slope", not "interval unavailable".
    out = out.reindex(columns=["quantity", "value", "ci_low", "ci_high", "note"])
    out["interpretation"] = (
        "outcome truncation and exposure-dependent playing time are separate "
        "denominator defects; this table reports which of them explains the "
        "attenuation of a per-minute exposure association; every gamma is "
        "fitted with player-clustered errors so the pooled and within-stratum "
        "values rest on the same variance estimator"
    )
    return out


def lineup_coverage_denominator_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the minute difference inside and outside the complete-lineup era.

    The starter/substitute split rests on the seasons carrying lineup data. If
    the overall event-minus-non-event difference is similar where lineup status
    is unknown, that split is not an artefact of which seasons were covered.
    """
    _require_columns(
        frame,
        [SAME_DAY_COL, MINUTES_COL, LINEUP_ROLE_COL, "season_start", PLAYER_ID_COL],
        "lineup coverage frame",
    )
    work = frame.copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    known = work[LINEUP_ROLE_COL].astype(str).ne(UNKNOWN_ROLE)
    coverage = work.assign(known=known).groupby("season_start")["known"].mean()
    complete_seasons = sorted(coverage[coverage.ge(0.999)].index.tolist())
    in_era = work["season_start"].isin(complete_seasons)

    rows = []
    for label, subset in (
        ("complete_lineup_seasons", work[in_era]),
        ("remaining_seasons", work[~in_era]),
        ("lineup_status_known", work[known]),
        ("lineup_status_unknown", work[~known]),
    ):
        events = pd.to_numeric(subset[SAME_DAY_COL], errors="coerce").fillna(0).eq(1)
        if int(events.sum()) == 0 or int((~events).sum()) == 0:
            continue
        event_minutes = subset.loc[events, MINUTES_COL]
        other_minutes = subset.loc[~events, MINUTES_COL]
        rows.append(
            {
                "stratum": label,
                "n_appearances": int(len(subset)),
                "n_same_day_events": int(events.sum()),
                "mean_minutes_event": float(event_minutes.mean()),
                "mean_minutes_non_event": float(other_minutes.mean()),
                "event_minus_non_event_minutes": float(
                    event_minutes.mean() - other_minutes.mean()
                ),
                "median_minutes_event": float(event_minutes.median()),
                "median_minutes_non_event": float(other_minutes.median()),
            }
        )
    out = pd.DataFrame(rows)
    out["complete_lineup_seasons"] = ", ".join(str(int(s)) for s in complete_seasons)
    out["interpretation"] = (
        "similar differences inside and outside the complete-lineup era mean "
        "the starter and substitute split is not an artefact of lineup coverage"
    )
    return out


def event_clustering_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report how the same-day events are distributed across players.

    Player-clustered intervals are only as informative as the clustering they
    describe, so readers need to know how many players carried an event and
    whether a minority carried many.
    """
    _require_columns(frame, [PLAYER_ID_COL, SAME_DAY_COL], "event clustering frame")
    work = frame.copy()
    work[SAME_DAY_COL] = pd.to_numeric(work[SAME_DAY_COL], errors="coerce").fillna(0)
    per_player = work.groupby(PLAYER_ID_COL)[SAME_DAY_COL].sum()
    with_event = per_player[per_player.gt(0)]
    total_events = float(per_player.sum())
    top_decile = int(max(1, round(0.1 * len(with_event))))
    share_top_decile = (
        float(with_event.sort_values(ascending=False).head(top_decile).sum() / total_events)
        if total_events
        else np.nan
    )
    return pd.DataFrame(
        [
            {
                "n_players": int(work[PLAYER_ID_COL].nunique()),
                "n_players_with_event": int(len(with_event)),
                "share_of_players_with_event": float(
                    len(with_event) / work[PLAYER_ID_COL].nunique()
                ),
                "n_events": int(total_events),
                "median_events_per_affected_player": float(with_event.median()),
                "q1_events_per_affected_player": float(with_event.quantile(0.25)),
                "q3_events_per_affected_player": float(with_event.quantile(0.75)),
                "max_events_per_player": int(with_event.max()) if len(with_event) else 0,
                "share_of_events_in_top_decile_of_players": share_top_decile,
                "interpretation": (
                    "player-clustered intervals assume events repeat within "
                    "players; these counts show how concentrated that repetition is"
                ),
            }
        ]
    )


def model_field_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    """Count the eligible rows each reference-model field would remove.

    The cohort table reports what was analysed; this reports what analysis
    itself discards, so missingness is visible rather than implied by a
    shrinking denominator between tables.
    """
    fields = [
        SAME_DAY_COL,
        MINUTES_COL,
        HISTORY_MODEL_COL,
        "prior_minutes_7d",
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, fields, "model completeness frame")
    total = int(len(frame))
    rows = []
    for field in fields:
        missing = int(frame[field].isna().sum())
        rows.append(
            {
                "model_field": field,
                "n_eligible_rows": total,
                "n_missing": missing,
                "percent_missing": 100.0 * missing / total if total else np.nan,
            }
        )
    complete = int(frame[fields].notna().all(axis=1).sum())
    positive_minutes = int(
        (
            frame[fields].notna().all(axis=1)
            & pd.to_numeric(frame[MINUTES_COL], errors="coerce").gt(0.0)
        ).sum()
    )
    rows.append(
        {
            "model_field": "all_fields_complete_and_positive_minutes",
            "n_eligible_rows": total,
            "n_missing": total - positive_minutes,
            "percent_missing": 100.0 * (total - positive_minutes) / total if total else np.nan,
        }
    )
    out = pd.DataFrame(rows)
    out["n_analysed"] = complete
    out["interpretation"] = (
        "rows entering the reference model after listwise deletion; a zero "
        "here means the eligibility rules already guaranteed completeness"
    )
    return out


def episode_type_composition(
    episodes: pd.DataFrame,
    classify_injury_type: Any,
) -> pd.DataFrame:
    """Describe what the public source records across every reconciled episode.

    The prespecified illness negative control failed for want of events. That
    scarcity is itself a measurement result: it says what kind of record this
    source keeps, so it is tabulated rather than reported only as a missing
    analysis.
    """
    _require_columns(episodes, ["injury_desc"], "episode composition frame")
    descriptions = episodes["injury_desc"].fillna("").astype(str).str.strip()
    types = descriptions.map(classify_injury_type)
    counts = types.value_counts(dropna=False).rename_axis("episode_public_type").reset_index(
        name="n_episodes"
    )
    counts["share_of_episodes"] = counts["n_episodes"] / float(len(episodes))
    counts["is_illness"] = counts["episode_public_type"].isin(ILLNESS_TYPES)
    counts["interpretation"] = (
        "a source holding thousands of absence episodes but almost no illness "
        "records is an availability and narrative record, not a health record"
    )
    return counts.sort_values("n_episodes", ascending=False).reset_index(drop=True)


def temporal_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Refit the additive reference model in three fixed season blocks."""
    _require_columns(frame, ["season_start"], "temporal frame")
    out = frame.copy()
    out["temporal_block"] = np.select(
        [out["season_start"].le(2019), out["season_start"].between(2020, 2021)],
        ["2017-18 to 2019-20", "2020-21 to 2021-22"],
        default="2022-23 to 2024-25",
    )
    specification = next(
        item for item in exposure_specs() if item["exposure_id"] == "prior_minutes_7d"
    )
    rows = []
    for block in (
        "2017-18 to 2019-20",
        "2020-21 to 2021-22",
        "2022-23 to 2024-25",
    ):
        row, _ = exposure_model_row(
            out[out["temporal_block"].eq(block)],
            SAME_DAY_COL,
            "per_appearance",
            specification,
        )
        row["temporal_block"] = block
        rows.append(row)
    formula = (
        f"{SAME_DAY_COL} ~ I(prior_minutes_7d / 90.0) * C(temporal_block) + "
        f"{HISTORY_MODEL_COL} + " + " + ".join(CALENDAR_TERMS)
    )
    result = smf.glm(formula, data=out, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": out[PLAYER_ID_COL]}
    )
    names = list(result.params.index)
    interaction_terms = [
        name
        for name in names
        if "prior_minutes_7d" in name and ":C(temporal_block)" in name
    ]
    restriction = np.zeros((len(interaction_terms), len(names)))
    for row_index, term in enumerate(interaction_terms):
        restriction[row_index, names.index(term)] = 1.0
    tested = result.wald_test(restriction, scalar=True)
    block_p_values = holm_adjust(pd.Series([row["p_value"] for row in rows]))
    for block_index, row in enumerate(rows):
        row["holm_p_value_3_block_family"] = float(block_p_values.iloc[block_index])
        row["reject_holm_3_block_0_05"] = bool(
            block_p_values.iloc[block_index] < 0.05
        )
        row["heterogeneity_test_statistic"] = float(
            np.asarray(tested.statistic).squeeze()
        )
        row["heterogeneity_df"] = int(len(interaction_terms))
        row["heterogeneity_p_value"] = float(np.asarray(tested.pvalue).squeeze())
        row["analysis_timing"] = "post-data temporal stability check"
    return pd.DataFrame(rows)


def _exposure_band(values: pd.Series) -> pd.Categorical:
    """Return fixed support bands for conditional-model disclosure."""
    numeric = pd.to_numeric(values, errors="coerce")
    labels = np.select(
        [
            numeric.eq(0.0),
            numeric.between(0.0, 45.0, inclusive="right"),
            numeric.between(45.0, 90.0, inclusive="right"),
            numeric.between(90.0, 135.0, inclusive="right"),
            numeric.between(135.0, 180.0, inclusive="right"),
            numeric.gt(180.0),
        ],
        EXPOSURE_BANDS,
        default=None,
    )
    return pd.Categorical(labels, categories=EXPOSURE_BANDS, ordered=True)


def _conditional_cluster_covariance(
    result: Any,
    work: pd.DataFrame,
    stratum_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return player-cluster sandwich covariance and player score sums."""
    strata = list(pd.unique(result.model.groups))
    score_by_stratum = {
        stratum: np.asarray(result.model.score_grp(index, result.params), dtype=float)
        for index, stratum in enumerate(strata)
    }
    mapping = work[[stratum_col, PLAYER_ID_COL]].drop_duplicates()
    if mapping[stratum_col].duplicated().any():
        raise ValueError("each conditional stratum must map to one player")
    mapping = mapping.set_index(stratum_col)[PLAYER_ID_COL]
    player_scores: dict[Any, np.ndarray] = {}
    for stratum, score in score_by_stratum.items():
        player = mapping.loc[stratum]
        player_scores[player] = player_scores.get(player, np.zeros_like(score)) + score
    scores = np.vstack(list(player_scores.values()))
    bread = np.linalg.pinv(-np.asarray(result.model.hessian(result.params), dtype=float))
    covariance = bread @ (scores.T @ scores) @ bread.T
    if len(scores) > 1:
        covariance *= len(scores) / (len(scores) - 1.0)
    return covariance, scores


def conditional_model_analysis(
    frame: pd.DataFrame,
    bootstrap_draws: int = CONDITIONAL_BOOTSTRAP_DRAWS,
    seed: int = RANDOM_SEED + 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit additive conditional models and disclose their selected population."""
    if bootstrap_draws < 100:
        raise ValueError("conditional multiplier bootstrap requires at least 100 draws")
    required = [
        PLAYER_ID_COL,
        "season_start",
        SAME_DAY_COL,
        "prior_minutes_7d",
        HISTORY_MODEL_COL,
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "conditional frame")
    estimates = []
    population_rows = []
    support_rows = []
    rng = np.random.default_rng(seed)
    for definition in ("player", "player-season"):
        work = frame.copy()
        stratum_col = "conditional_stratum"
        work[stratum_col] = work[PLAYER_ID_COL].astype(str)
        if definition == "player-season":
            work[stratum_col] += "_" + work["season_start"].astype(str)
        counts = work.groupby(stratum_col, observed=False)[SAME_DAY_COL].agg(["sum", "count"])
        discordant = counts[counts["sum"].gt(0) & counts["sum"].lt(counts["count"])].index
        included = work[work[stratum_col].isin(discordant)].copy()
        excluded = work[~work[stratum_col].isin(discordant)].copy()
        design = dmatrix(
            "0 + I(prior_minutes_7d / 90.0) + history_log_iqr + "
            + " + ".join(CALENDAR_TERMS),
            included,
            return_type="dataframe",
        )
        result = ConditionalLogit(
            included[SAME_DAY_COL].astype(int),
            design,
            groups=included[stratum_col],
        ).fit(disp=False, maxiter=250)
        term = next(name for name in result.params.index if "prior_minutes_7d" in name)
        term_index = list(result.params.index).index(term)
        covariance, scores = _conditional_cluster_covariance(
            result, included, stratum_col
        )
        bread = np.linalg.pinv(-np.asarray(result.model.hessian(result.params), dtype=float))
        signs = rng.choice((-1.0, 1.0), size=(bootstrap_draws, len(scores)))
        perturbations = (signs @ scores) @ bread.T
        multiplier_log = 2.0 * (
            float(result.params[term]) + perturbations[:, term_index]
        )
        log_estimate = 2.0 * float(result.params[term])
        standard_error = 2.0 * float(np.sqrt(max(covariance[term_index, term_index], 0.0)))
        low, high = np.quantile(multiplier_log, [0.025, 0.975])
        estimates.append(
            {
                "stratum_definition": definition,
                "contrast": "180 versus 0 prior minutes in 7 days",
                "effect_measure": "conditional_odds_ratio",
                "estimate": float(np.exp(log_estimate)),
                "player_cluster_ci_low": float(
                    np.exp(log_estimate - NormalDist().inv_cdf(0.975) * standard_error)
                ),
                "player_cluster_ci_high": float(
                    np.exp(log_estimate + NormalDist().inv_cdf(0.975) * standard_error)
                ),
                "multiplier_bootstrap_ci_low": float(np.exp(low)),
                "multiplier_bootstrap_ci_high": float(np.exp(high)),
                "player_cluster_p_value": _normal_p_value(log_estimate, standard_error),
                "bootstrap_draws": bootstrap_draws,
                "n_rows": int(len(included)),
                "n_players": int(included[PLAYER_ID_COL].nunique()),
                "n_discordant_strata": int(len(discordant)),
                "n_events": int(included[SAME_DAY_COL].sum()),
                "population_interpretation": (
                    "players or player-seasons with both event and non-event appearances; "
                    "not the full cohort"
                ),
            }
        )
        for status, subset in (("discordant_included", included), ("concordant_excluded", excluded)):
            population_rows.append(
                {
                    "stratum_definition": definition,
                    "population": status,
                    "n_rows": int(len(subset)),
                    "n_players": int(subset[PLAYER_ID_COL].nunique()),
                    "n_events": int(subset[SAME_DAY_COL].sum()),
                    "mean_prior_minutes_7d": float(subset["prior_minutes_7d"].mean()),
                    "median_prior_minutes_7d": float(subset["prior_minutes_7d"].median()),
                    "mean_history_per_10000_prior_minutes": float(subset[HISTORY_COL].mean()),
                }
            )
        included["exposure_band"] = _exposure_band(included["prior_minutes_7d"])
        grouped = included.groupby("exposure_band", observed=False)
        for band, subset in grouped:
            support_rows.append(
                {
                    "stratum_definition": definition,
                    "exposure_band": str(band),
                    "n_rows": int(len(subset)),
                    "n_players": int(subset[PLAYER_ID_COL].nunique()),
                    "n_events": int(subset[SAME_DAY_COL].sum()),
                }
            )
    estimate_frame = pd.DataFrame(estimates)
    estimate_frame["holm_p_value_2_model_family"] = holm_adjust(
        estimate_frame["player_cluster_p_value"]
    )
    estimate_frame["reject_holm_2_model_0_05"] = estimate_frame[
        "holm_p_value_2_model_family"
    ].lt(0.05)
    return estimate_frame, pd.DataFrame(population_rows), pd.DataFrame(support_rows)


def _weighted_smd(
    values: pd.Series,
    selected: pd.Series,
    weights: pd.Series,
) -> float:
    """Return an absolute weighted standardised mean difference."""
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    group = selected.astype(bool)
    means = []
    variances = []
    for mask in (group, ~group):
        group_weights = pd.to_numeric(weights[mask], errors="coerce").fillna(0.0)
        if group_weights.sum() <= 0.0:
            return np.nan
        observed = numeric[mask]
        mean = float(np.average(observed, weights=group_weights))
        means.append(mean)
        variances.append(float(np.average((observed - mean) ** 2, weights=group_weights)))
    denominator = np.sqrt(max((variances[0] + variances[1]) / 2.0, 0.0))
    return abs(means[0] - means[1]) / denominator if denominator > 0.0 else 0.0


def appearance_selection_sensitivity(
    risk_set: pd.DataFrame,
    daily_panel: pd.DataFrame,
    appearance_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Weight observed EPL appearances within conservative membership intervals.

    This is a restricted measured-selection sensitivity, not a reconstruction of
    registered squads or medical clearance, and not an adjustment for selection
    into appearance. A player is considered at opportunity only between the
    first and last observed club appearance, extended by recorded transfers.

    Returns the estimates, the diagnostics (including explicit no-leakage gates
    showing that every propensity covariate is fixed before the fixture), and a
    comparison of the appearances and players the bounded set includes against
    those it excludes.
    """
    daily_required = [
        PLAYER_ID_COL,
        DATE_COL,
        "prior_minutes_played",
        HISTORY_COL,
        "days_since_last_match",
        *CALENDAR_TERMS,
    ]
    _require_columns(
        risk_set,
        [
            PLAYER_ID_COL,
            DATE_COL,
            "played_any_minutes",
            "plausibly_available",
            "all_minutes_last_7d",
            "season",
            "player_club_id",
            "membership_evidence",
        ],
        "selection risk set",
    )
    _require_columns(daily_panel, daily_required, "selection daily panel")
    _require_columns(
        appearance_frame,
        [PLAYER_ID_COL, DATE_COL, SAME_DAY_COL],
        "selection appearance frame",
    )
    risk = risk_set.copy()
    risk[DATE_COL] = pd.to_datetime(risk[DATE_COL], errors="coerce")
    daily = daily_panel[daily_required].copy()
    daily[DATE_COL] = pd.to_datetime(daily[DATE_COL], errors="coerce")
    risk = risk.merge(daily, on=[PLAYER_ID_COL, DATE_COL], how="left")
    cohort_ids = set(appearance_frame[PLAYER_ID_COL].astype(int))
    risk = risk[
        risk[PLAYER_ID_COL].isin(cohort_ids)
        & risk["plausibly_available"].fillna(False).astype(bool)
        & risk[DATE_COL].between(pd.Timestamp("2017-07-01"), COMPLETE_SELECTION_END)
        & pd.to_numeric(risk["prior_minutes_played"], errors="coerce").ge(900.0)
    ].copy()
    risk["burden90"] = pd.to_numeric(risk["all_minutes_last_7d"], errors="coerce").fillna(0.0) / 90.0
    risk["history_log"] = np.log1p(
        pd.to_numeric(risk[HISTORY_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    risk["prior_minutes_log"] = np.log1p(
        pd.to_numeric(risk["prior_minutes_played"], errors="coerce").fillna(0.0)
    )
    risk["days_since"] = pd.to_numeric(
        risk["days_since_last_match"], errors="coerce"
    ).fillna(30.0).clip(upper=60.0)
    propensity_formula = (
        "played_any_minutes ~ burden90 + I(burden90 ** 2) + history_log + "
        "prior_minutes_log + days_since + I(days_since ** 2) + "
        "C(season) + C(player_club_id)"
    )
    propensity = smf.glm(
        propensity_formula, data=risk, family=sm.families.Binomial()
    ).fit()
    risk["selection_probability"] = propensity.predict(risk).clip(0.01, 0.99)
    selected_share = float(risk["played_any_minutes"].mean())
    risk["selection_weight"] = np.where(
        risk["played_any_minutes"].eq(1),
        selected_share / risk["selection_probability"],
        (1.0 - selected_share) / (1.0 - risk["selection_probability"]),
    )
    selected = risk[risk["played_any_minutes"].eq(1)].copy()
    outcomes = appearance_frame[[PLAYER_ID_COL, DATE_COL, SAME_DAY_COL]].copy()
    outcomes[DATE_COL] = pd.to_datetime(outcomes[DATE_COL], errors="coerce")
    outcomes = outcomes.groupby([PLAYER_ID_COL, DATE_COL], as_index=False)[SAME_DAY_COL].max()
    selected = selected.merge(outcomes, on=[PLAYER_ID_COL, DATE_COL], how="inner")
    outcome_formula = (
        f"{SAME_DAY_COL} ~ burden90 + history_log + " + " + ".join(CALENDAR_TERMS)
    )
    unweighted = smf.glm(
        outcome_formula, data=selected, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": selected[PLAYER_ID_COL]})
    weighted = sm.GEE.from_formula(
        outcome_formula,
        groups=PLAYER_ID_COL,
        data=selected,
        family=sm.families.Binomial(),
        weights=selected["selection_weight"],
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    rows = []
    for model_id, result in (("unweighted", unweighted), ("inverse_selection_weighted", weighted)):
        estimate = float(result.params["burden90"])
        standard_error = float(result.bse["burden90"])
        critical = NormalDist().inv_cdf(0.975)
        rows.append(
            {
                "model_id": model_id,
                "effect": "per 90 prior club-match minutes in 7 days",
                "estimate": float(np.exp(estimate)),
                "ci_low": float(np.exp(estimate - critical * standard_error)),
                "ci_high": float(np.exp(estimate + critical * standard_error)),
                "p_value": _normal_p_value(estimate, standard_error),
                "n_selected_appearances": int(len(selected)),
                "n_players": int(selected[PLAYER_ID_COL].nunique()),
                "n_same_day_reports": int(selected[SAME_DAY_COL].sum()),
                "estimand_limit": (
                    "EPL opportunities inside observed-appearance spans extended by "
                    "recorded transfers; registered roster and medical clearance unknown"
                ),
            }
        )
    diagnostics = pd.DataFrame(
        [
            {
                "metric": "opportunities",
                "value": float(len(risk)),
                "passes_gate": True,
            },
            {
                "metric": "selected_share",
                "value": selected_share,
                "passes_gate": True,
            },
            {
                "metric": "overlap_probability_0_05_to_0_95",
                "value": float(risk["selection_probability"].between(0.05, 0.95).mean()),
                "passes_gate": bool(risk["selection_probability"].between(0.05, 0.95).mean() >= 0.95),
            },
            {
                "metric": "selected_weight_p99",
                "value": float(selected["selection_weight"].quantile(0.99)),
                "passes_gate": bool(selected["selection_weight"].quantile(0.99) <= 10.0),
            },
            *[
                {
                    "metric": f"weighted_smd_{column}",
                    "value": _weighted_smd(
                        risk[column], risk["played_any_minutes"], risk["selection_weight"]
                    ),
                    "passes_gate": bool(
                        _weighted_smd(
                            risk[column], risk["played_any_minutes"], risk["selection_weight"]
                        )
                        <= 0.1
                    ),
                }
                for column in ("burden90", "history_log", "prior_minutes_log", "days_since")
            ],
        ]
    )
    diagnostics["all_numeric_gates_pass"] = bool(diagnostics["passes_gate"].all())
    diagnostics["membership_evidence"] = ";".join(
        sorted(risk["membership_evidence"].dropna().astype(str).unique())
    )
    propensity_covariates = (
        "burden90",
        "history_log",
        "prior_minutes_log",
        "days_since",
        "season",
        "player_club_id",
    )
    outcome_columns = (SAME_DAY_COL, LAG1_COL, COMBINED_COL)
    leaked = [column for column in outcome_columns if column in propensity_formula]
    # The opportunity set is built from Premier League fixtures, so the leakage
    # question must be asked inside that scope: given a Premier League
    # appearance, does carrying a same-day report change whether the row is
    # retained? Comparing against all competitions would instead measure the
    # deliberate competition restriction.
    cohort = appearance_frame.copy()
    cohort[DATE_COL] = pd.to_datetime(cohort[DATE_COL], errors="coerce")
    in_window = cohort[
        cohort[DATE_COL].between(pd.Timestamp("2017-07-01"), COMPLETE_SELECTION_END)
    ].copy()
    opportunity_keys = set(
        map(tuple, risk[[PLAYER_ID_COL, DATE_COL]].to_numpy().tolist())
    )
    in_window["in_opportunity_set"] = [
        tuple(key) in opportunity_keys
        for key in in_window[[PLAYER_ID_COL, DATE_COL]].to_numpy().tolist()
    ]
    if "competition_context" in in_window:
        scoped = in_window[
            in_window["competition_context"].astype(str).eq("Premier League")
        ]
        out_of_scope = int((~in_window["in_opportunity_set"]).sum())
    else:
        scoped = in_window
        out_of_scope = 0
    scoped_events = scoped[
        pd.to_numeric(scoped[SAME_DAY_COL], errors="coerce").fillna(0).gt(0)
    ]
    retained_events = (
        float(scoped_events["in_opportunity_set"].mean()) if len(scoped_events) else np.nan
    )
    retained_rows = float(scoped["in_opportunity_set"].mean()) if len(scoped) else np.nan
    leakage = pd.DataFrame(
        [
            {
                "metric": "outcome_columns_in_propensity_model",
                "value": float(len(leaked)),
                "passes_gate": not leaked,
            },
            {
                "metric": "premier_league_appearances_retained_in_opportunity_set",
                "value": retained_rows,
                "passes_gate": bool(not np.isfinite(retained_rows) or retained_rows >= 0.99),
            },
            {
                "metric": "premier_league_same_day_event_rows_retained",
                "value": retained_events,
                "passes_gate": bool(
                    not np.isfinite(retained_events) or retained_events >= 0.99
                ),
            },
            {
                "metric": "non_premier_league_appearances_out_of_scope",
                "value": float(out_of_scope),
                "passes_gate": True,
            },
        ]
    )
    leakage["all_numeric_gates_pass"] = bool(leakage["passes_gate"].all())
    leakage["membership_evidence"] = diagnostics["membership_evidence"].iloc[0]
    leakage["interpretation"] = (
        "availability and propensity covariates ("
        + ", ".join(propensity_covariates)
        + ") are fixed before the fixture; within the Premier League fixtures the "
        "opportunity set covers, a same-day report never removes its own appearance, "
        "so the weights carry no same-day outcome information. Appearances in other "
        "competitions fall outside the reconstructed EPL opportunity set by design, "
        "which is a scope restriction rather than outcome-dependent selection"
    )
    diagnostics = pd.concat([diagnostics, leakage], ignore_index=True)
    diagnostics["all_numeric_gates_pass"] = bool(diagnostics["passes_gate"].all())

    analysed = set(map(tuple, selected[[PLAYER_ID_COL, DATE_COL]].to_numpy().tolist()))
    cohort = appearance_frame.copy()
    cohort[DATE_COL] = pd.to_datetime(cohort[DATE_COL], errors="coerce")
    cohort["in_bounded_selection_set"] = [
        tuple(key) in analysed
        for key in cohort[[PLAYER_ID_COL, DATE_COL]].to_numpy().tolist()
    ]
    population_rows = []
    for included, group in cohort.groupby("in_bounded_selection_set", sort=True):
        burden = pd.to_numeric(group["prior_minutes_7d"], errors="coerce")
        population_rows.append(
            {
                "population": "included in bounded selection set"
                if included
                else "excluded from bounded selection set",
                "n_appearances": int(len(group)),
                "n_players": int(group[PLAYER_ID_COL].nunique()),
                "n_same_day_events": int(group[SAME_DAY_COL].sum()),
                "same_day_events_per_1000_appearances": (
                    1000.0 * float(group[SAME_DAY_COL].mean()) if len(group) else np.nan
                ),
                "median_prior_minutes_7d": float(burden.median()),
                "median_history_rate_per_10000min": float(
                    pd.to_numeric(group[HISTORY_COL], errors="coerce").median()
                ),
                "median_age_years": (
                    float(pd.to_numeric(group["age_years"], errors="coerce").median())
                    if "age_years" in group
                    else np.nan
                ),
                "premier_league_share": (
                    float(
                        group["competition_context"].astype(str).eq("Premier League").mean()
                    )
                    if "competition_context" in group
                    else np.nan
                ),
                "exclusion_reasons": (
                    "inside the bounded window and opportunity set"
                    if included
                    else "non-Premier-League competition, after 30 June 2024, outside a "
                    "membership interval, or marked unavailable by public episode data"
                ),
            }
        )
    population = pd.DataFrame(population_rows)
    diagnostics["interpretation"] = (
        "restricted measured-selection sensitivity within a bounded opportunity set; "
        "not an adjustment for selection into appearance and not a complete roster risk set"
    )
    return pd.DataFrame(rows), diagnostics, population


def build_non_event_audit_queue(
    frame: pd.DataFrame,
    sample_size: int = 30,
) -> pd.DataFrame:
    """Select appearances with no reported spell start, for false-negative review.

    Sampling only reported positives can estimate agreement among those
    positives but never the events the public record missed. This queue draws
    a deterministic, exposure-blinded sample of appearances carrying no
    same-day report so a reviewer can search independent sources for an injury
    the record failed to capture. Adjudication is manual: the queue ships with
    empty verdicts and the summary refuses to score it until they are filled.
    """
    required = [
        PLAYER_ID_COL,
        DATE_COL,
        SAME_DAY_COL,
        LAG1_COL,
        "player_name",
        "club_name",
        MINUTES_COL,
    ]
    _require_columns(frame, required, "non-event audit frame")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    quiet = frame[
        frame[SAME_DAY_COL].eq(0)
        & frame[LAG1_COL].eq(0)
        & pd.to_numeric(frame[MINUTES_COL], errors="coerce").gt(0.0)
    ].copy()
    if quiet.empty:
        raise ValueError("no non-event appearances available to audit")
    quiet["audit_id"] = (
        quiet[PLAYER_ID_COL].astype(int).astype(str)
        + "_"
        + pd.to_datetime(quiet[DATE_COL]).dt.strftime("%Y%m%d")
    )
    quiet["sampling_key"] = pd.util.hash_pandas_object(
        quiet[["audit_id"]], index=False
    ).astype("uint64")
    queue = quiet.sort_values("sampling_key").head(sample_size).copy()
    queue["audit_stratum"] = "no_reported_spell_start"
    queue["independent_source_url"] = ""
    queue["independent_source_type"] = ""
    queue["missed_event_verdict"] = "pending"
    queue["review_note"] = "pending independent public-source review"
    return queue[
        [
            "audit_id",
            PLAYER_ID_COL,
            "player_name",
            "club_name",
            DATE_COL,
            "audit_stratum",
            "independent_source_url",
            "independent_source_type",
            "missed_event_verdict",
            "review_note",
        ]
    ].reset_index(drop=True)


def non_event_absence_screen(
    queue: pd.DataFrame,
    appearances: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Ask whether each queued appearance was followed by an absence at all.

    A match injury the public record missed should leave a trace in the
    player's own schedule: fixtures their club played that they did not. The
    days to a player's next appearance is not that trace, because most long
    gaps here are international windows or the summer break, when nobody
    played. Counting the club's fixtures inside the gap separates the two.

    This assigns no verdicts and cannot: a player can be injured and miss
    nothing. What it does is bound the search. Where the club played no
    fixture before the player returned, there was no absence for the record to
    have missed, and the only missed events still possible are those that cost
    no playing time at all --- which by this study's outcome definition would
    not generate a spell start.
    """
    _require_columns(queue, [PLAYER_ID_COL, DATE_COL, "audit_id"], "absence screen queue")
    _require_columns(
        appearances, ["player_id", "player_club_id", "date"], "absence screen appearances"
    )
    _require_columns(
        fixtures, ["date", "home_club_id", "away_club_id"], "absence screen fixtures"
    )
    played = appearances.copy()
    played["date"] = pd.to_datetime(played["date"], errors="coerce")
    games = fixtures.copy()
    games["date"] = pd.to_datetime(games["date"], errors="coerce")

    rows = []
    for _, record in queue.iterrows():
        player = record[PLAYER_ID_COL]
        day = pd.to_datetime(record[DATE_COL])
        mine = played[played["player_id"].eq(player)].sort_values("date")
        on_day = mine[mine["date"].eq(day)]
        if on_day.empty:
            rows.append(
                {
                    "audit_id": record["audit_id"],
                    "days_to_next_appearance": np.nan,
                    "club_fixtures_missed": np.nan,
                    "club_changed": False,
                    "screen": "appearance not in the appearance snapshot",
                }
            )
            continue
        club = on_day["player_club_id"].iloc[0]
        later = mine[mine["date"].gt(day)]
        if later.empty:
            rows.append(
                {
                    "audit_id": record["audit_id"],
                    "days_to_next_appearance": np.nan,
                    "club_fixtures_missed": np.nan,
                    "club_changed": False,
                    "screen": "no subsequent appearance in the snapshot",
                }
            )
            continue
        next_row = later.iloc[0]
        gap_days = int((next_row["date"] - day).days)
        # A player who leaves the club cannot play its fixtures, so a club
        # change explains the gap without any injury.
        changed = bool(next_row["player_club_id"] != club)
        window = games[games["date"].gt(day) & games["date"].lt(next_row["date"])]
        missed = int(
            len(window[window["home_club_id"].eq(club) | window["away_club_id"].eq(club)])
        )
        if missed == 0:
            screen = "club played no fixture before the player returned"
        elif changed:
            screen = "gap spans a club change, so missed fixtures are not absences"
        else:
            screen = "player missed club fixtures; search required"
        rows.append(
            {
                "audit_id": record["audit_id"],
                "days_to_next_appearance": gap_days,
                "club_fixtures_missed": missed,
                "club_changed": changed,
                "screen": screen,
            }
        )
    # Carry only what identifies the record. The verdict lives in the reviewed
    # audit table; copying the queue's placeholder here would put two deposited
    # tables in disagreement over the same appearance.
    keys = [
        column
        for column in (PLAYER_ID_COL, DATE_COL, "audit_id", "player_name", "club_name")
        if column in queue.columns
    ]
    out = queue[keys].merge(pd.DataFrame(rows), on="audit_id", how="left")
    out["interpretation"] = (
        "the screen orders the search and assigns no verdicts; where the club "
        "played no fixture before the player returned there was no absence for "
        "the public record to have missed, which bounds what kind of missed "
        "event remains possible without establishing that none occurred"
    )
    return out


def summarize_non_event_audit(reviewed: pd.DataFrame) -> pd.DataFrame:
    """Summarise the false-negative review, or record that it is outstanding."""
    _require_columns(reviewed, ["missed_event_verdict"], "non-event audit review")
    verdicts = reviewed["missed_event_verdict"].astype(str)
    sampled = int(len(reviewed))
    pending = int(verdicts.eq("pending").sum())
    unresolved = int(verdicts.eq("unresolved").sum())
    resolved = int(verdicts.isin(("missed_event", "no_missed_event")).sum())
    missed = int(verdicts.eq("missed_event").sum())
    if resolved:
        low, high = proportion_confint(missed, resolved, method="wilson")
        proportion = missed / resolved
    else:
        low, high, proportion = np.nan, np.nan, np.nan
    # A review in which every record came back unresolved is not a completed
    # audit. Calling it complete would let a reader read the zero missed events
    # as a finding rather than as the absence of one.
    if pending:
        status = "outstanding"
    elif resolved < sampled:
        status = "reviewed, not resolved"
    else:
        status = "complete"
    return pd.DataFrame(
        [
            {
                "audit_dimension": "missed_event_detection",
                "n_sampled": sampled,
                "n_pending": pending,
                "n_unresolved": unresolved,
                "n_resolved": resolved,
                "n_missed_events": missed,
                "missed_event_proportion": proportion,
                "ci_low": float(low) if np.isfinite(low) else np.nan,
                "ci_high": float(high) if np.isfinite(high) else np.nan,
                "status": status,
                "interpretation": (
                    "an estimate of events the public record missed; a record "
                    "returned as unresolved contributes to neither numerator nor "
                    "denominator, so where none resolve the missed-event "
                    "proportion is undefined rather than zero and the outcome's "
                    "sensitivity remains unestimated"
                ),
            }
        ]
    )


def cohen_kappa(first: pd.Series, second: pd.Series) -> dict[str, float]:
    """Return Cohen's kappa for two assessors over the same audited records."""
    paired = pd.DataFrame({"first": first.astype(str), "second": second.astype(str)}).dropna()
    total = int(len(paired))
    if total == 0:
        return {"n_compared": 0, "observed_agreement": np.nan, "expected_agreement": np.nan, "kappa": np.nan}
    observed = float(paired["first"].eq(paired["second"]).mean())
    categories = sorted(set(paired["first"]) | set(paired["second"]))
    expected = float(
        sum(
            (paired["first"].eq(category).mean()) * (paired["second"].eq(category).mean())
            for category in categories
        )
    )
    kappa = np.nan if np.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)
    return {
        "n_compared": total,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": float(kappa) if np.isfinite(kappa) else np.nan,
    }


def second_assessor_agreement(reviewed: pd.DataFrame) -> pd.DataFrame:
    """Report inter-rater agreement when a second assessor's column exists."""
    column = "date_attribution_verdict_second_assessor"
    if column not in reviewed.columns:
        return pd.DataFrame(
            [
                {
                    "dimension": "date_attribution",
                    "n_compared": 0,
                    "observed_agreement": np.nan,
                    "expected_agreement": np.nan,
                    "kappa": np.nan,
                    "status": "outstanding",
                    "interpretation": (
                        "no second-assessor column present; inter-rater agreement "
                        "is unestimated and the audit remains single-assessor"
                    ),
                }
            ]
        )
    statistics = cohen_kappa(reviewed["date_attribution_verdict"], reviewed[column])
    return pd.DataFrame(
        [
            {
                "dimension": "date_attribution",
                **statistics,
                "status": "complete",
                "interpretation": "Cohen's kappa between the two independent assessors",
            }
        ]
    )


def build_outcome_audit_queue(
    frame: pd.DataFrame,
    per_stratum: int = 10,
) -> pd.DataFrame:
    """Select a deterministic, exposure-blinded same-day source-audit queue."""
    required = [
        PLAYER_ID_COL,
        DATE_COL,
        SAME_DAY_COL,
        "player_name",
        "club_name",
        MINUTES_COL,
        "matchproxy_injury_desc",
        "same_day_reported_absence_ge28d",
        "same_day_muscle_tendon_report",
    ]
    _require_columns(frame, required, "outcome audit frame")
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    events = frame[
        frame[SAME_DAY_COL].eq(1)
        & pd.to_numeric(frame[MINUTES_COL], errors="coerce").gt(0.0)
    ].copy()
    events["audit_stratum"] = np.select(
        [
            events["same_day_reported_absence_ge28d"].eq(1),
            events["same_day_muscle_tendon_report"].eq(1),
        ],
        ["reported_absence_ge28d", "muscle_tendon_nonsevere"],
        default="other_nonsevere",
    )
    events["audit_id"] = (
        events[PLAYER_ID_COL].astype(int).astype(str)
        + "_"
        + pd.to_datetime(events[DATE_COL]).dt.strftime("%Y%m%d")
    )
    events["sampling_key"] = pd.util.hash_pandas_object(
        events[["audit_id", "audit_stratum"]], index=False
    ).astype("uint64")
    queue = (
        events.sort_values(["audit_stratum", "sampling_key"])
        .groupby("audit_stratum", observed=False, group_keys=False)
        .head(per_stratum)
        .copy()
    )
    queue["independent_source_url"] = ""
    queue["independent_source_type"] = ""
    queue["date_attribution_verdict"] = "pending"
    queue["description_consistency_verdict"] = "pending"
    queue["review_note"] = "pending independent public-source review"
    return queue[
        [
            "audit_id",
            PLAYER_ID_COL,
            "player_name",
            "club_name",
            DATE_COL,
            "matchproxy_injury_desc",
            "audit_stratum",
            "independent_source_url",
            "independent_source_type",
            "date_attribution_verdict",
            "description_consistency_verdict",
            "review_note",
        ]
    ].reset_index(drop=True)


def summarize_outcome_audit(reviewed: pd.DataFrame) -> pd.DataFrame:
    """Summarise independent date-attribution and description verdicts.

    Two estimands are reported because they answer different questions. The
    resolved-only proportion conditions on having found independent evidence
    and therefore describes records the audit could adjudicate. The
    all-sampled proportion uses every drawn record and is the conservative
    reading. Because unresolved records could go either way, the table also
    carries partial-identification bounds: the lower bound counts every
    unresolved record as not confirmed and the upper bound counts them all as
    confirmed. Neither estimand measures missed events, sensitivity or
    specificity, because only reported positives were sampled.
    """
    _require_columns(
        reviewed,
        [
            "audit_stratum",
            "date_attribution_verdict",
            "description_consistency_verdict",
        ],
        "reviewed outcome audit",
    )
    rows = []
    dimensions = (
        ("date_attribution", "date_attribution_verdict"),
        ("description_consistency", "description_consistency_verdict"),
    )
    groups = list(reviewed.groupby("audit_stratum", observed=False))
    groups.append(("all_strata", reviewed))
    for dimension, verdict_column in dimensions:
        for stratum, subset in groups:
            resolved = subset[
                subset[verdict_column].isin(("confirmed", "not_confirmed"))
            ]
            confirmed = int(resolved[verdict_column].eq("confirmed").sum())
            total = int(len(resolved))
            if total:
                low, high = proportion_confint(confirmed, total, method="wilson")
                proportion = confirmed / total
            else:
                low, high, proportion = np.nan, np.nan, np.nan
            sampled = int(len(subset))
            unresolved = sampled - total
            if sampled:
                sampled_low, sampled_high = proportion_confint(
                    confirmed, sampled, method="wilson"
                )
                sampled_proportion = confirmed / sampled
                bound_low = confirmed / sampled
                bound_high = (confirmed + unresolved) / sampled
            else:
                sampled_low, sampled_high = np.nan, np.nan
                sampled_proportion, bound_low, bound_high = np.nan, np.nan, np.nan
            rows.append(
                {
                    "audit_dimension": dimension,
                    "audit_stratum": str(stratum),
                    "n_sampled": sampled,
                    "n_resolved": total,
                    "n_unresolved": unresolved,
                    "n_confirmed": confirmed,
                    "confirmed_proportion": proportion,
                    "ci_low": float(low) if np.isfinite(low) else np.nan,
                    "ci_high": float(high) if np.isfinite(high) else np.nan,
                    "confirmed_proportion_all_sampled": sampled_proportion,
                    "all_sampled_ci_low": float(sampled_low) if np.isfinite(sampled_low) else np.nan,
                    "all_sampled_ci_high": float(sampled_high) if np.isfinite(sampled_high) else np.nan,
                    "partial_identification_low": bound_low,
                    "partial_identification_high": bound_high,
                    "status": "complete" if total == sampled else "incomplete",
                    "estimand_note": (
                        "confirmed_proportion conditions on resolved records; "
                        "confirmed_proportion_all_sampled uses every sampled record; "
                        "bounds assume unresolved records are all not confirmed or all confirmed"
                    ),
                }
            )
    return pd.DataFrame(rows)


def validate_outcome_audit(
    queue: pd.DataFrame,
    reviewed: pd.DataFrame,
    require_completed_review: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that manual decisions belong to the blinded audit queue.

    Both sides arrive de-identified, so the fields held immutable are the
    surrogates rather than the names they replaced. That is not a weakening:
    the record key is drawn once per audited appearance, so it pins a verdict
    to its row exactly as the provider identifier used to, and it pins nothing
    to a person.
    """
    immutable = [
        RECORD_KEY,
        PLAYER_KEY,
        SEASON,
        "matchproxy_injury_desc",
        "audit_stratum",
    ]
    decision_columns = [
        SOURCE_FOUND,
        "independent_source_type",
        "date_attribution_verdict",
        "description_consistency_verdict",
        "review_note",
    ]
    _require_columns(queue, immutable, "outcome audit queue")
    _require_columns(reviewed, [*immutable, *decision_columns], "reviewed outcome audit")
    if queue[RECORD_KEY].duplicated().any() or reviewed[RECORD_KEY].duplicated().any():
        raise ValueError("outcome audit record keys must be unique")
    if set(queue[RECORD_KEY]) != set(reviewed[RECORD_KEY]):
        raise ValueError("reviewed outcome audit IDs do not match the blinded queue")

    forbidden = [
        column
        for column in reviewed.columns
        if any(token in column.lower() for token in ("prior_minutes", "burden", "history_log"))
    ]
    if forbidden:
        raise ValueError(f"reviewed outcome audit contains exposure fields: {forbidden}")

    expected = queue[immutable].copy()
    observed = reviewed[immutable].copy()
    for frame in (expected, observed):
        for column in immutable:
            frame[column] = frame[column].fillna("").astype(str).map(
                lambda value: unicodedata.normalize("NFKD", value)
                .encode("ascii", "ignore")
                .decode("ascii")
            )
    expected = expected.sort_values(RECORD_KEY).reset_index(drop=True)
    observed = observed.sort_values(RECORD_KEY).reset_index(drop=True)
    if not expected.equals(observed):
        raise ValueError("reviewed outcome audit changed immutable queue fields")

    allowed = {"confirmed", "not_confirmed", "unresolved"}
    verdict_columns = [
        "date_attribution_verdict",
        "description_consistency_verdict",
    ]
    for column in verdict_columns:
        verdicts = set(reviewed[column].fillna("").astype(str).str.strip())
        permitted = allowed | {"pending"}
        if not verdicts.issubset(permitted):
            raise ValueError(f"reviewed outcome audit has invalid {column} values")
    if require_completed_review and reviewed[verdict_columns].eq("pending").any(axis=None):
        raise ValueError("reviewed outcome audit still contains pending decisions")

    resolved = reviewed[verdict_columns].isin(("confirmed", "not_confirmed")).any(axis=1)
    # The source URL is withheld from the deposited record because most of these
    # slugs carry the player's surname and one carries a graded diagnosis. What
    # the verdict rests on -- that a qualifying source was found, and what kind
    # it was -- survives as a flag and a type. The URLs themselves are checked
    # against the independence rule in src/38_deidentify_audit_evidence.py,
    # where they still exist.
    found = reviewed[SOURCE_FOUND].fillna(False).astype(str).str.lower().isin(
        ("true", "1", "yes")
    )
    if (resolved & ~found).any():
        raise ValueError("resolved outcome audit rows require an independent source")
    source_types = reviewed["independent_source_type"].fillna("").astype(str)
    if source_types.str.contains("transfermarkt", case=False, regex=False).any():
        raise ValueError("outcome audit sources must be independent of Transfermarkt")
    notes = reviewed["review_note"].fillna("").astype(str).str.strip()
    if require_completed_review and (notes.eq("") | notes.str.startswith("pending")).any():
        raise ValueError("completed outcome audit rows require review notes")

    aligned = reviewed.set_index(RECORD_KEY).loc[queue[RECORD_KEY]].reset_index()
    validation = pd.DataFrame(
        [
            {"gate": "queue_ids_exact", "value": int(len(aligned)), "passes_gate": True},
            {
                "gate": "immutable_fields_exact",
                "value": int(len(immutable)),
                "passes_gate": True,
            },
            {
                "gate": "exposure_fields_absent",
                "value": int(len(forbidden)),
                "passes_gate": True,
            },
            {
                "gate": "resolved_sources_complete",
                "value": int(resolved.sum()),
                "passes_gate": True,
            },
            {
                "gate": "transfermarkt_sources_used",
                "value": int(source_types.str.contains("transfermarkt", case=False, regex=False).sum()),
                "passes_gate": True,
            },
            {
                "gate": "completed_review_required",
                "value": int(require_completed_review),
                "passes_gate": True,
            },
        ]
    )
    return aligned, validation


def build_revised_hypothesis_register(
    legacy: pd.DataFrame,
    multiverse: pd.DataFrame,
    temporal: pd.DataFrame,
    conditional: pd.DataFrame,
) -> pd.DataFrame:
    """Append every reviewer-requested inferential result to the legacy register."""
    _require_columns(legacy, ["hypothesis_id"], "legacy hypothesis register")
    _require_columns(
        multiverse,
        [
            "exposure_id", "effect_label", "event_col", "denominator",
            "effect_measure", "estimate", "ci_low", "ci_high", "p_value",
            "n_rows", "n_players", "n_events", "formula", "analysis_timing",
            "holm_p_value_63_model_family", "reject_holm_0_05",
        ],
        "revised exposure multiverse",
    )
    _require_columns(
        temporal,
        [
            "temporal_block", "estimate", "ci_low", "ci_high", "p_value",
            "holm_p_value_3_block_family", "reject_holm_3_block_0_05",
            "heterogeneity_test_statistic", "heterogeneity_df",
            "heterogeneity_p_value", "n_rows", "n_players", "n_events",
        ],
        "revised temporal register",
    )
    _require_columns(
        conditional,
        [
            "stratum_definition", "estimate", "player_cluster_ci_low",
            "player_cluster_ci_high", "player_cluster_p_value",
            "holm_p_value_2_model_family", "reject_holm_2_model_0_05",
            "n_rows", "n_players", "n_events", "n_discordant_strata",
        ],
        "revised conditional register",
    )
    columns = list(legacy.columns)
    rows: list[dict[str, Any]] = []

    def blank_row() -> dict[str, Any]:
        return {column: np.nan for column in columns}

    for item in multiverse.itertuples(index=False):
        row = blank_row()
        row.update(
            {
                "model_id": f"revised_{item.exposure_id}_{item.event_col}_{item.denominator}",
                "contrast_id": item.effect_label,
                "description": (
                    f"{item.effect_label}; {item.event_col}; {item.denominator}"
                ),
                "effect_measure": item.effect_measure,
                "p_value": item.p_value,
                "event_col": item.event_col,
                "denominator": item.denominator,
                "exposure_spec": item.exposure_id,
                "n_match_rows": item.n_rows,
                "n_players": item.n_players,
                "n_events": item.n_events,
                "family": "revised_63_model_exposure_multiverse",
                "analysis_role": "reviewer_requested_multiverse",
                "test_domain": "recent_exposure_association",
                "source_file": "jsams_revised_exposure_multiverse.csv",
                "model": item.formula,
                "model_family": (
                    "binomial_logit"
                    if item.denominator == "per_appearance"
                    else "poisson_log"
                ),
                "estimator": "player_clustered_glm",
                "group_col": PLAYER_ID_COL,
                "estimate": item.estimate,
                "ci_low": item.ci_low,
                "ci_high": item.ci_high,
                "family_size": 63,
                "analysis_timing": item.analysis_timing,
                "confirmatory_status": "exploratory",
                "estimable": True,
                "dated_prospective_analysis_plan_available": False,
                "p_holm_within_family_recomputed": item.holm_p_value_63_model_family,
                "p_adjusted_reported": item.holm_p_value_63_model_family,
                "reject_adjusted_0_05": item.reject_holm_0_05,
            }
        )
        rows.append(row)

    for item in temporal.itertuples(index=False):
        row = blank_row()
        row.update(
            {
                "model_id": "revised_temporal_prior_minutes_7d",
                "contrast_id": item.temporal_block,
                "description": f"seven-day linear exposure association in {item.temporal_block}",
                "effect_measure": "odds_ratio",
                "p_value": item.p_value,
                "event_col": SAME_DAY_COL,
                "denominator": "per_appearance",
                "exposure_spec": "prior_minutes_7d",
                "n_match_rows": item.n_rows,
                "n_players": item.n_players,
                "n_events": item.n_events,
                "family": "revised_3_block_temporal_stability",
                "analysis_role": "post_data_temporal_replication",
                "test_domain": "recent_exposure_association",
                "source_file": "jsams_revised_temporal_stability.csv",
                "estimate": item.estimate,
                "ci_low": item.ci_low,
                "ci_high": item.ci_high,
                "family_size": 3,
                "analysis_timing": "post-data temporal stability check",
                "confirmatory_status": "exploratory",
                "estimable": True,
                "dated_prospective_analysis_plan_available": False,
                "p_holm_within_family_recomputed": item.holm_p_value_3_block_family,
                "p_adjusted_reported": item.holm_p_value_3_block_family,
                "reject_adjusted_0_05": item.reject_holm_3_block_0_05,
            }
        )
        rows.append(row)

    heterogeneity = temporal.iloc[0]
    row = blank_row()
    row.update(
        {
            "model_id": "revised_temporal_heterogeneity",
            "contrast_id": "global_exposure_by_temporal_block_interaction",
            "description": "global heterogeneity of the seven-day linear association across three blocks",
            "effect_measure": "chi_square",
            "test_statistic": heterogeneity["heterogeneity_test_statistic"],
            "df": heterogeneity["heterogeneity_df"],
            "p_value": heterogeneity["heterogeneity_p_value"],
            "event_col": SAME_DAY_COL,
            "denominator": "per_appearance",
            "exposure_spec": "prior_minutes_7d_by_temporal_block",
            "family": "revised_temporal_heterogeneity",
            "analysis_role": "post_data_temporal_replication",
            "test_domain": "temporal_heterogeneity",
            "source_file": "jsams_revised_temporal_stability.csv",
            "family_size": 1,
            "analysis_timing": "post-data temporal stability check",
            "confirmatory_status": "exploratory",
            "estimable": True,
            "dated_prospective_analysis_plan_available": False,
            "p_holm_within_family_recomputed": heterogeneity["heterogeneity_p_value"],
            "p_adjusted_reported": heterogeneity["heterogeneity_p_value"],
            "reject_adjusted_0_05": bool(heterogeneity["heterogeneity_p_value"] < 0.05),
        }
    )
    rows.append(row)

    for item in conditional.itertuples(index=False):
        row = blank_row()
        row.update(
            {
                "model_id": f"revised_conditional_{item.stratum_definition}",
                "contrast_id": "180_vs_0_prior_minutes_7d",
                "description": f"conditional seven-day linear contrast within {item.stratum_definition}",
                "effect_measure": "conditional_odds_ratio",
                "p_value": item.player_cluster_p_value,
                "event_col": SAME_DAY_COL,
                "denominator": "per_appearance",
                "exposure_spec": "prior_minutes_7d_linear",
                "n_match_rows": item.n_rows,
                "n_players": item.n_players,
                "n_events": item.n_events,
                "family": "revised_2_model_conditional_sensitivity",
                "analysis_role": "selected_population_sensitivity",
                "test_domain": "within_stratum_recent_exposure_association",
                "source_file": "jsams_revised_conditional_estimates.csv",
                "estimate": item.estimate,
                "ci_low": item.player_cluster_ci_low,
                "ci_high": item.player_cluster_ci_high,
                "family_size": 2,
                "stratum_definition": item.stratum_definition,
                "analysis_timing": "post-data conditional sensitivity",
                "confirmatory_status": "exploratory",
                "n_discordant_strata": item.n_discordant_strata,
                "estimable": True,
                "dated_prospective_analysis_plan_available": False,
                "p_holm_within_family_recomputed": item.holm_p_value_2_model_family,
                "p_adjusted_reported": item.holm_p_value_2_model_family,
                "reject_adjusted_0_05": item.reject_holm_2_model_0_05,
            }
        )
        rows.append(row)

    start = len(legacy) + 1
    for offset, row in enumerate(rows):
        row["hypothesis_id"] = f"H{start + offset:04d}"
    return pd.concat([legacy, pd.DataFrame(rows, columns=columns)], ignore_index=True)


def revised_claim_hierarchy(
    multiverse: pd.DataFrame,
    temporal: pd.DataFrame,
    selection: pd.DataFrame,
    audit: pd.DataFrame,
    multiverse_summary: pd.DataFrame | None = None,
    correlations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the controlling tier registry for the revised manuscript.

    Two independent axes govern visibility. ``tier`` drives ``abstract_visible``
    and is enforced by ``visibility_rule_passes``: tiers 1-3 must appear in the
    abstract and tiers 4-5 must not. ``main_display_recommended`` is a separate
    editorial axis recording whether the claim earns a main table or figure, so
    a Tier 5 robustness check may sit in a main figure while staying out of the
    abstract. Both axes are asserted against ``manuscript.tex`` in
    ``tests/test_manuscript_tier_visibility.py``.
    """
    focal = multiverse[
        multiverse["event_col"].eq(SAME_DAY_COL)
        & multiverse["denominator"].eq("per_appearance")
    ]
    minute_rows = focal[focal["exposure_id"].str.startswith("prior_minutes_")]
    significant = int(minute_rows["reject_holm_0_05"].sum())
    total_metrics = int(len(focal))
    primary = focal[focal["exposure_id"].eq("prior_minutes_7d")].iloc[0]
    weighted = selection[selection["model_id"].eq("inverse_selection_weighted")].iloc[0]
    audit_overall = audit[
        audit["audit_dimension"].eq("date_attribution")
        & audit["audit_stratum"].eq("all_strata")
    ].iloc[0]
    resolved = int(audit_overall["n_resolved"])
    confirmed = int(audit_overall["n_confirmed"])
    sampled = int(audit_overall["n_sampled"])
    heterogeneity = float(temporal["heterogeneity_p_value"].iloc[0])
    if multiverse_summary is not None and not multiverse_summary.empty:
        overall = multiverse_summary[
            multiverse_summary["stratum_id"].eq("all_63_models")
        ].iloc[0]
        distribution = (
            f" Across all {int(overall['n_models'])} specifications the median odds "
            f"ratio was {float(overall['estimate_median']):.2f} "
            f"(interquartile range {float(overall['estimate_q1']):.2f}-"
            f"{float(overall['estimate_q3']):.2f}, range "
            f"{float(overall['estimate_min']):.2f}-{float(overall['estimate_max']):.2f})."
        )
    else:
        distribution = ""
    if correlations is not None and not correlations.empty:
        window_pairs = correlations[
            correlations["exposure_a"].str.startswith("prior_minutes_")
            & correlations["exposure_b"].str.startswith("prior_minutes_")
        ]
        correlation_note = (
            " Cumulative windows correlate up to r="
            f"{float(window_pairs['pearson_r'].max()):.2f}, so surviving windows are "
            "correlated sensitivity analyses around one reference window."
        )
    else:
        correlation_note = ""
    rows = [
        {
            "claim_id": "cumulative_recent_exposure_same_day_association",
            "tier": 2,
            "claim_role": "primary_original_observational_association",
            "tier_justification": (
                "Tier 2: one exploratory cumulative-exposure association that was "
                "relatively stable across a declared multiverse and fixed temporal "
                "refits. Every analysis was chosen after the data were inspected, so "
                "the Holm family controls the 63 stated tests but not the choices that "
                "defined them. It is therefore not Tier 1."
            ),
            "abstract_visible": True,
            "main_display_recommended": True,
            "evidence": (
                f"Reference seven-day same-day per-appearance OR {primary['estimate']:.3f} "
                f"({primary['ci_low']:.3f}-{primary['ci_high']:.3f}) per 90 minutes."
                f"{distribution}"
                f" {significant}/{total_metrics} same-day per-appearance metrics fell below "
                f"the 63-model Holm threshold.{correlation_note}"
                f" Temporal heterogeneity p={heterogeneity:.3g}; restricted selection-weighted "
                f"OR {weighted['estimate']:.3f} "
                f"({weighted['ci_low']:.3f}-{weighted['ci_high']:.3f})."
            ),
            "required_caveat": (
                "One exploratory association, not four replications: the surviving windows "
                "share matches by construction. It concerns same-day public spell-start "
                "reports among recorded appearances and does not identify clinical injury "
                "incidence or a causal workload effect."
            ),
        },
        {
            "claim_id": "independent_same_day_outcome_audit",
            "tier": 2,
            "claim_role": "original_outcome_attribution_audit",
            "tier_justification": (
                "Tier 2: an exposure-blinded independent-source audit directly tests "
                "date attribution, but its 30-record sample and single reviewer prevent "
                "clinical validation or Tier 1 status."
            ),
            "abstract_visible": True,
            "main_display_recommended": True,
            "evidence": (
                f"Independent sources confirmed exact-match attribution for "
                f"{confirmed}/{resolved} resolved records "
                f"({100.0 * confirmed / resolved:.1f}%) and {confirmed}/{sampled} "
                f"of all sampled records ({100.0 * confirmed / sampled:.1f}%); "
                f"{sampled - resolved}/{sampled} remained unresolved, bounding the "
                f"sampled proportion between {100.0 * confirmed / sampled:.1f}% and "
                f"{100.0 * (confirmed + sampled - resolved) / sampled:.1f}%."
            ),
            "required_caveat": (
                "One assessor, reported positives only. The audit checks public-source "
                "attribution, not medical diagnosis, mechanism, missed events, "
                "sensitivity, specificity or inter-rater agreement."
            ),
        },
        {
            "claim_id": "reported_event_duration_linkage",
            "tier": 2,
            "claim_role": "original_measurement_result",
            "tier_justification": (
                "Tier 2: the measured link between event-row status, recorded duration, and "
                "denominator-dependent estimates is original but cannot recover exact event time."
            ),
            "abstract_visible": True,
            "main_display_recommended": True,
            "evidence": "Inherited from jsams_claim_hierarchy.csv after the validated script-34 bootstrap.",
            "required_caveat": "Per-appearance and minute-denominator models answer different questions.",
        },
        {
            "claim_id": "denominator_gradient_measured_and_replicated",
            "tier": 2,
            "claim_role": "primary_original_measurement",
            "tier_justification": (
                "Not Tier 5: it is a positive measurement, not a null. Not Tier 4: no "
                "prior study measures this quantity, so it replicates nobody. Not Tier 3: "
                "it neither contradicts published work nor reports a surprising null. "
                "The Tier 1 case is that the gradient is central, is named and measured "
                "here for the first time, and holds without exception across eight "
                "leagues, which reaches past one specification. It is declined because "
                "the existence of the bias was already established analytically by "
                "Shrier et al., so what is original is its size, its boundary and its "
                "detectability rather than its existence, which is what Tier 2 describes "
                "as partly anticipated. Ambiguity resolves downward by rule."
            ),
            "abstract_visible": True,
            "main_display_recommended": True,
            "evidence": (
                "The gradient is estimated from appearances, dates and player "
                "identifiers with no injury data, pooled and within starters, in eight "
                "European domestic leagues; see the deposited league table, decision "
                "rule and summary."
            ),
            "required_caveat": (
                "The magnitude is local to each panel and must be measured locally; "
                "only the mechanism generalises. The cross-league fits use a different "
                "population and covariate set from the reference cohort, so the two "
                "values for the same league differ by construction."
            ),
        },
        {
            "claim_id": "denominator_gradient_decision_rule",
            "tier": 2,
            "claim_role": "original_pre_analysis_check",
            "tier_justification": (
                "Not Tier 4 or 5: a decision rule read off interval bounds is neither a "
                "replication nor a null. Not Tier 3: it contradicts nothing. Not Tier 1: "
                "it operationalises the gradient rather than establishing a further "
                "finding, and its threshold is a reporting convention rather than a "
                "tested quantity, which bounds how much weight it can carry."
            ),
            "abstract_visible": True,
            "main_display_recommended": True,
            "evidence": (
                "The rule reads a pooled lower bound and a within-starter upper bound "
                "against a negligible threshold and returns the same verdict in every "
                "league measured."
            ),
            "required_caveat": (
                "The threshold is a reporting convention, not a test. The gradient and "
                "its interval should be reported whatever the verdict."
            ),
        },
        {
            "claim_id": "truncation_explains_none_of_the_attenuation",
            "tier": 3,
            "claim_role": "surprising_null_against_the_expected_mechanism",
            "tier_justification": (
                "Tier 3: the intuitive defect was expected to explain the attenuation and "
                "two estimators, one making no approximation, agree that it explains none "
                "of it, with a resampled interval spanning zero. It is recorded as a "
                "surprising null rather than as a Tier 2 original finding because its "
                "content is the absence of an effect, and the rule assigns the lower tier "
                "where the category is arguable."
            ),
            "abstract_visible": True,
            "main_display_recommended": True,
            "evidence": (
                "Swapping recorded for untruncated minutes in the offset moves the "
                "coefficient by a fraction of a percent of the attenuation, with a "
                "player-resampled interval that includes zero."
            ),
            "required_caveat": (
                "Descriptive with an inferred mechanism: the minute an event occurred is "
                "never observed. Truncation is negligible for cohort rates and large for "
                "any quantity conditioned on cases."
            ),
        },
        {
            "claim_id": "public_report_quality_restrictions",
            "tier": 5,
            "claim_role": "secondary_null_outcome_quality_check",
            "tier_justification": (
                "Tier 5: no severe-report or muscle/tendon test survived the stated "
                "quality family, so the restrictions add no positive evidence."
            ),
            "abstract_visible": False,
            "main_display_recommended": False,
            "evidence": "Inherited from the symmetric same-day quality-restriction analysis.",
            "required_caveat": (
                "Absence of an adjusted association neither validates nor refutes a "
                "public-report exception; labels are not clinical adjudication."
            ),
        },
        {
            "claim_id": "continuous_history_effect_modification_null",
            "tier": 5,
            "claim_role": "secondary_null",
            "tier_justification": "Tier 5: an unsupported interaction is neither an original positive result nor a surprising contradiction.",
            "abstract_visible": False,
            "main_display_recommended": False,
            "evidence": "Continuous history remains an additive adjustment in the revised primary model.",
            "required_caveat": "No detected interaction does not prove homogeneity.",
        },
        {
            "claim_id": "appearance_selection_weighting",
            "tier": 5,
            "claim_role": "restricted_measured_selection_sensitivity",
            "tier_justification": (
                "Tier 5: the weighting check supports robustness but cannot reconstruct "
                "registered rosters or medical clearance. It is kept out of the abstract "
                "but shown as a supporting main-figure panel, because a robustness check "
                "on the primary estimand is most useful beside that estimand."
            ),
            "abstract_visible": False,
            "main_display_recommended": True,
            "evidence": (
                f"Bounded selection-weighted OR {weighted['estimate']:.3f} "
                f"({weighted['ci_low']:.3f}-{weighted['ci_high']:.3f})."
            ),
            "required_caveat": (
                "A restricted measured-selection sensitivity, not an adjustment for "
                "selection into appearance. Opportunity membership is inferred from "
                "Premier League fixtures inside observed-appearance spans extended by "
                "transfers, so registered rosters, symptoms and medical clearance are unknown."
            ),
        },
    ]
    out = pd.DataFrame(rows)
    out["visibility_rule_passes"] = np.where(
        out["tier"].isin((1, 2, 3)),
        out["abstract_visible"],
        ~out["abstract_visible"],
    )
    return out


def read_inputs(root: Path):  # pragma: no cover
    """Read source and audit tables required by this revision stage."""
    processed = root / "data" / "processed"
    panel = pd.read_csv(
        processed / "player_match_panel_all_comp.csv",
        parse_dates=[DATE_COL],
        low_memory=False,
    )
    injuries = pd.read_csv(processed / "tm_injuries_clean.csv", low_memory=False)
    episodes = pd.read_csv(processed / "tm_injury_episodes.csv", low_memory=False)
    lineups_path = root / "external_data" / "transfermarkt" / "game_lineups.csv"
    lineups = pd.read_csv(
        lineups_path, usecols=["date", "player_id", "type"], low_memory=False
    )
    risk_set = pd.read_csv(
        processed / "public_data_v4" / "selection_risk_set.csv", low_memory=False
    )
    daily_columns = [
        PLAYER_ID_COL,
        DATE_COL,
        "prior_minutes_played",
        HISTORY_COL,
        "days_since_last_match",
        *CALENDAR_TERMS,
    ]
    daily = pd.read_csv(
        processed / "player_day_panel_all_comp.csv",
        usecols=daily_columns,
        low_memory=False,
    )
    return panel, injuries, episodes, lineups, risk_set, daily


def write_outputs(outputs: Mapping[str, pd.DataFrame], results_dir: Path) -> None:  # pragma: no cover
    """Write every second-referee output with a stable prefix."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for stem, frame in outputs.items():
        frame.to_csv(results_dir / f"jsams_revised_{stem}.csv", index=False)


def main() -> None:  # pragma: no cover
    """Run the complete second-referee analysis layer."""
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "data" / "processed" / "results"
    primary = load_source_module(
        "18_match_proxy_poisson_splines_perminute.py", "jsams_second_primary"
    )
    previous = load_source_module(
        "34_jsams_referee_analysis.py", "jsams_second_previous"
    )
    extension = load_source_module(
        "33_matchproxy_current_data_extensions.py", "jsams_second_extension"
    )
    panel, injuries, episodes, lineups, risk_set, daily = read_inputs(root)
    panel = previous.add_same_day_quality_outcomes(
        panel, episodes, primary.classify_public_injury_type
    )
    panel = add_negative_control_outcomes(
        panel, episodes, primary.classify_public_injury_type
    )
    panel = add_prior_window_metrics(panel)
    frame, _ = previous.prepare_jsams_frame(
        primary,
        panel,
        injuries,
        lineups,
        root / "external_data" / "transfermarkt",
    )

    print("1. Validating rolling windows and fitting the 63-model family ...")
    window_validation = validate_reference_window(frame)
    if not bool(window_validation["parity_passes"].iloc[0]):
        raise RuntimeError("Reconstructed seven-day minutes do not match the legacy field")
    multiverse = exposure_multiverse(frame)
    multiverse_summary = exposure_multiverse_summary(multiverse)
    metric_correlations = exposure_metric_correlations(frame)

    print("2. Estimating additive curves, simultaneous bands and temporal stability ...")
    spline = previous.spline_expression(float(frame["prior_minutes_7d"].max())).replace(
        previous.BURDEN_COL, "prior_minutes_7d"
    )
    curves, curve_tests = additive_curve_analysis(frame, spline)
    absolute_contrast, exposure_support = absolute_risk_contrast(frame)
    temporal = temporal_stability(frame)

    print("3. Fitting measured-confounding, conditional and selection sensitivities ...")
    context_frame, _ = extension.attach_current_match_metadata(
        primary, frame, root / "external_data" / "transfermarkt"
    )
    confounding = confounding_sensitivity(context_frame)
    negative_control = negative_control_outcome_analysis(frame)
    placebo_frame = add_placebo_exposure_window(frame)
    placebo = placebo_window_analysis(placebo_frame)
    placebo_denominators = placebo_denominator_replication(placebo_frame)
    ascertainment = ascertainment_by_exposure(frame)
    club_congestion = club_congestion_sensitivity(
        add_club_fixture_congestion(context_frame)
    )
    # The reference frame already drops rows below 900 prior minutes, so a
    # lower threshold can only be tested on an unrestricted rebuild.
    unrestricted, _ = previous.prepare_jsams_frame(
        primary,
        panel,
        injuries,
        lineups,
        root / "external_data" / "transfermarkt",
        minimum_prior_minutes=0.0,
    )
    run_in = run_in_threshold_sensitivity(unrestricted)
    run_in_exclusions = run_in_exclusion_comparison(unrestricted)

    print("3b. Decomposing the minute-denominator attenuation ...")
    minute_distribution = recorded_minute_distribution(frame)
    lineup_composition = lineup_composition_by_exposure(frame)
    denominator_roles = denominator_by_lineup_role(frame)
    role_association = squad_role_association_sensitivity(frame)
    role_adjusted_refit = role_adjusted_denominator_refit(frame)
    attenuation = denominator_attenuation_decomposition(frame)
    truncation_refit = direct_truncation_refit(frame)
    case_restricted = case_restricted_exposure_bias(frame)
    imputation_sensitivity = truncation_imputation_sensitivity(frame)
    print("    bootstrapping attenuation intervals over players ...")
    attenuation_intervals = attenuation_bootstrap(frame)
    coverage_stability = lineup_coverage_denominator_stability(frame)
    clustering = event_clustering_summary(frame)
    completeness = model_field_completeness(frame)
    episode_types = episode_type_composition(episodes, primary.classify_public_injury_type)

    conditional, conditional_population, conditional_support = conditional_model_analysis(frame)
    selection, selection_diagnostics, selection_population = appearance_selection_sensitivity(
        risk_set, daily, context_frame
    )

    print("4. Building the independent source-audit queue and tier registry ...")
    players = pd.read_csv(
        root / "external_data" / "transfermarkt" / "players.csv",
        usecols=["player_id", "name"],
    ).rename(columns={"player_id": PLAYER_ID_COL, "name": "player_name"})
    clubs = pd.read_csv(
        root / "external_data" / "transfermarkt" / "clubs.csv",
        usecols=["club_id", "name"],
    ).rename(columns={"club_id": "player_club_id", "name": "club_name"})
    player_lookup = players.assign(
        _lookup_id=pd.to_numeric(players[PLAYER_ID_COL], errors="coerce")
    ).dropna(subset=["_lookup_id"]).set_index("_lookup_id")["player_name"]
    club_lookup = clubs.assign(
        _lookup_id=pd.to_numeric(clubs["player_club_id"], errors="coerce")
    ).dropna(subset=["_lookup_id"]).set_index("_lookup_id")["club_name"]
    frame["player_name"] = pd.to_numeric(
        frame[PLAYER_ID_COL], errors="coerce"
    ).map(player_lookup)
    frame["club_name"] = pd.to_numeric(
        frame["player_club_id"], errors="coerce"
    ).map(club_lookup)
    # The queues are built, joined and screened on the provider's real
    # identifiers, because the sample is drawn by hashing them and the absence
    # screen joins on them. Identity is removed on the way to disk, after every
    # value that depends on it has been computed.
    identity_map = load_identity_map(
        root / "data" / "private" / "audit_identity_map.csv"
    )
    identified_originals = [
        pd.read_csv(root / "data" / "private" / name, dtype=str)
        for name in (
            "independent_same_day_event_audit.csv",
            "independent_non_event_audit.csv",
        )
    ]
    surnames = audited_surnames(identified_originals)

    queue = deidentify_audit_frame(
        build_outcome_audit_queue(frame), identity_map, surnames
    )
    review_path = root / "data" / "manual" / "independent_same_day_event_audit.csv"
    review_exists = review_path.exists()
    reviewed = pd.read_csv(review_path) if review_exists else queue
    reviewed, audit_validation = validate_outcome_audit(
        queue,
        reviewed,
        require_completed_review=review_exists,
    )
    audit_summary = summarize_outcome_audit(reviewed)
    assessor_agreement = second_assessor_agreement(reviewed)
    # Built identified, because the absence screen below joins these rows to the
    # appearance snapshot on the provider's player identifier.
    non_event_queue_identified = build_non_event_audit_queue(frame)
    non_event_queue = deidentify_audit_frame(
        non_event_queue_identified, identity_map, surnames
    )
    non_event_path = root / "data" / "manual" / "independent_non_event_audit.csv"
    non_event_reviewed = (
        pd.read_csv(non_event_path) if non_event_path.exists() else non_event_queue
    )
    non_event_summary = summarize_non_event_audit(non_event_reviewed)
    snapshot = (
        root / "data" / "raw" / "public_data_v4" / "transfermarkt_datasets_20260804"
    )
    non_event_screen = deidentify_audit_frame(
        non_event_absence_screen(
            non_event_queue_identified,
            pd.read_csv(
                snapshot / "appearances.csv.gz",
                usecols=["player_id", "player_club_id", "date"],
                low_memory=False,
            ),
            pd.read_csv(
                snapshot / "games.csv.gz",
                usecols=["date", "home_club_id", "away_club_id"],
            ),
        ),
        identity_map,
        surnames,
    )
    history_rates = pd.to_numeric(frame[HISTORY_COL], errors="coerce").dropna()
    history_reference = pd.DataFrame(
        [
            {
                "quantity": "median_prior_history_rate_per_10000_earlier_minutes",
                "value": float(history_rates.median()),
                "q1": float(history_rates.quantile(0.25)),
                "q3": float(history_rates.quantile(0.75)),
                "interpretation": (
                    "the value at which continuous prior history is held when "
                    "standardised probabilities are reported"
                ),
            }
        ]
    )
    hierarchy = revised_claim_hierarchy(
        multiverse,
        temporal,
        selection,
        audit_summary,
        multiverse_summary,
        metric_correlations,
    )
    denominator_metadata = denominator_contrast_metadata(
        pd.read_csv(results_dir / "jsams_same_day_minute_bootstrap_summary.csv"),
        pd.read_csv(results_dir / "jsams_lineup_minute_bootstrap_summary.csv"),
        pd.read_csv(results_dir / "jsams_lineup_completeness.csv"),
    )
    legacy_register = pd.read_csv(results_dir / "jsams_hypothesis_register.csv")
    hypothesis_register = build_revised_hypothesis_register(
        legacy_register, multiverse, temporal, conditional
    )
    exposure_summary = multiverse[
        multiverse["event_col"].eq(SAME_DAY_COL)
        & multiverse["denominator"].eq("per_appearance")
    ].copy()
    window_gradient = exposure_window_gradient(exposure_summary)
    outputs = {
        "window_validation": window_validation,
        "exposure_multiverse": multiverse,
        "exposure_multiverse_summary": multiverse_summary,
        "exposure_metric_correlations": metric_correlations,
        "exposure_metric_summary": exposure_summary,
        "additive_curves": curves,
        "additive_curve_tests": curve_tests,
        "absolute_risk_contrast": absolute_contrast,
        "exposure_support": exposure_support,
        "recorded_minute_distribution": minute_distribution,
        "lineup_composition_by_exposure": lineup_composition,
        "denominator_by_lineup_role": denominator_roles,
        "squad_role_association_sensitivity": role_association,
        "role_adjusted_denominator_refit": role_adjusted_refit,
        "run_in_exclusion_comparison": run_in_exclusions,
        "exposure_window_gradient": window_gradient,
        "denominator_attenuation_decomposition": attenuation,
        "direct_truncation_refit": truncation_refit,
        "case_restricted_exposure_bias": case_restricted,
        "truncation_imputation_sensitivity": imputation_sensitivity,
        "attenuation_bootstrap": attenuation_intervals,
        "lineup_coverage_denominator_stability": coverage_stability,
        "event_clustering_summary": clustering,
        "model_field_completeness": completeness,
        "episode_type_composition": episode_types,
        "confounding_sensitivity": confounding,
        "negative_control_outcomes": negative_control,
        "placebo_window_analysis": placebo,
        "placebo_denominator_replication": placebo_denominators,
        "ascertainment_by_exposure": ascertainment,
        "club_congestion_sensitivity": club_congestion,
        "run_in_threshold_sensitivity": run_in,
        "history_reference_value": history_reference,
        "non_event_audit_queue": non_event_queue,
        "non_event_audit_summary": non_event_summary,
        "non_event_absence_screen": non_event_screen,
        "second_assessor_agreement": assessor_agreement,
        "temporal_stability": temporal,
        "conditional_estimates": conditional,
        "conditional_population": conditional_population,
        "conditional_support": conditional_support,
        "appearance_selection_estimates": selection,
        "appearance_selection_diagnostics": selection_diagnostics,
        "appearance_selection_population": selection_population,
        "denominator_contrast_metadata": denominator_metadata,
        "outcome_audit_queue": queue,
        "outcome_audit_validation": audit_validation,
        "outcome_audit_summary": audit_summary,
        "claim_hierarchy": hierarchy,
        "hypothesis_register": hypothesis_register,
    }
    write_outputs(outputs, results_dir)
    print(f"Wrote {len(outputs)} revised JSAMS tables to {results_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
