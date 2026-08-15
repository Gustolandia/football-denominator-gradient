#!/usr/bin/env python
"""
18_match_proxy_poisson_splines_perminute.py

Main inferential model for match-associated injury-proxy incidence.

The primary model keeps the corrected same-day + lag-1 match-proxy outcome and
the dynamic prior-history fragility labels. Referee-facing robustness outputs
are generated in the same run:

- pointwise confidence intervals for the primary spline prediction curve;
- selected prediction confidence intervals at 0, 90, and 180 prior minutes,
  with local support counts around each selected burden;
- higher/intermediate prior-history rate-ratio confidence intervals via the
  delta method;
- denominator sensitivity models that avoid relying only on same-day realised
  minutes as the event offset;
- clean-comparator, international-break, and COVID-disruption sensitivity
  models;
- outcome sensitivity models for same-day only, lag-1 only, non-ambiguous
  injury descriptions only, reported absences of at least 28 days, and
  muscle/tendon public descriptions;
- label sensitivity models using simpler prior-injury-history definitions;
- a type-discordant check that defines prior history from specifically
  classified musculoskeletal non-muscle public injury descriptions and predicts
  the muscle/tendon outcome; and
- selection-control sensitivity models with age, position, club-season fixed
  effects, and continuous prior-history controls.
- spline-shape sensitivity refits using alternative B-spline and natural-cubic
  specifications;
- recovery-interval denominator/link checks on the same model scales used for
  the main match-minute contrast;
- player-correlated GEE and within-player switcher fixed-effect recurrent-event
  sensitivities;
- reporting-process severity audits by reported absence-duration category;
- temporal-stability models across football-season blocks, using the same
  carried-forward day-level prior-history labels rather than recalibrating
  prior-history strata inside each period.
- formal model-contrast tests: higher/intermediate history contrasts at 0 and
  180 prior minutes, a global spline-by-history Wald test, within-stratum
  0-to-180 and 90-to-180-minute incidence-rate ratios, and the ratio of the two
  history-stratum changes for each window;
- symmetric type-frequency/recency diagnostics, including correlations among
  rows with a prior same-type report and variance-inflation factors from the
  exact fitted-model design matrix; and
- a complete specification table with Holm and Benjamini-Hochberg adjustments
  across the predefined model specifications, preventing selective p-value
  reporting.

Outputs under data/processed/results:
- poisson_spline_params_matchproxy.csv
- poisson_spline_predictions_matchproxy.csv
- poisson_spline_selected_predictions_matchproxy.csv
- poisson_spline_selected_support_matchproxy.csv
- poisson_spline_selected_ratios_matchproxy.csv
- matchproxy_same_day_denominator_audit.csv
- matchproxy_spline_curve_shape_summary.csv
- matchproxy_spline_shape_sensitivity.csv
- matchproxy_spline_shape_contrast_sensitivity.csv
- matchproxy_spline_anchor_range_summary.csv
- matchproxy_selection_band_audit.csv
- matchproxy_selection_band_joint_proxy_audit.csv
- matchproxy_reporting_process_severity_audit.csv
- matchproxy_observed_event_support_summary.csv
- matchproxy_crude_daily_history_publication.csv
- matchproxy_proxy_classification_publication.csv
- matchproxy_proxy_event_type_summary.csv
- matchproxy_recovery_interval_rates.csv
- matchproxy_recovery_interval_display.csv
- matchproxy_recovery_interval_trend_tests.csv
- matchproxy_recovery_interval_rates_reported_absence_ge28d.csv
- matchproxy_recovery_interval_trend_tests_reported_absence_ge28d.csv
- matchproxy_recovery_interval_rates_muscle_tendon_only.csv
- matchproxy_recovery_interval_trend_tests_muscle_tendon_only.csv
- matchproxy_recovery_interval_model_summary.csv
- matchproxy_recovery_interval_publication_summary.csv
- matchproxy_recurrent_event_decomposition.csv
- matchproxy_between_within_publication_summary.csv
- matchproxy_sensitivity_summary.csv
- matchproxy_outcome_history_cross_summary.csv
- matchproxy_type_discordant_history_summary.csv
- matchproxy_negative_control_magnitude_comparison.csv
- matchproxy_negative_control_anchor_selection_audit.csv
- matchproxy_negative_control_direct_comparison.csv
- matchproxy_negative_control_mutually_exclusive_type_frequency.csv
- matchproxy_negative_control_mutually_exclusive_type_binary.csv
- matchproxy_negative_control_type_frequency_distribution.csv
- matchproxy_negative_control_type_frequency_linearity_check.csv
- matchproxy_negative_control_type_frequency_linearity_formal_test.csv
- matchproxy_type_history_recency_attenuation.csv
- matchproxy_type_history_multiplicity_family.csv
- matchproxy_nominal_exposure_response_signals.csv
- matchproxy_negative_control_recent_return_excluded_model_summary.csv
- matchproxy_negative_control_recent_return_exclusion.csv
- manuscript_numeric_reconciliation.csv
- matchproxy_denominator_sensitivity_summary.csv
- matchproxy_effect_modification_tests.csv
- matchproxy_formal_model_contrast_tests.csv
- matchproxy_multiplicity_family_summary.csv
- matchproxy_effect_modification_multiplicity_family_summary.csv
- matchproxy_publication_referee_audit.csv
- matchproxy_denominator_effect_modification_tests.csv
- matchproxy_temporal_stability_summary.csv
- matchproxy_temporal_stability_predictions.csv

Run:
    python src/18_match_proxy_poisson_splines_perminute.py
"""

from math import erfc, exp, log, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices, bs, cr  # noqa: F401  (bs/cr used in formulas)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pipeline_io import (
    LABELS_45,
    add_45min_load_bins,
    add_publication_history_labels,
    merge_day_fragility,
    restrict_to_available_risk_set,
    restrict_to_fragility_risk_set,
)
from v4_statistics import percent_with_interval


PLAYER_ID_COL = "tm_player_id"
MATCH_MINUTES_COL = "all_minutes_played"
SPLINE_DF = 4
INCLUDE_TOUGH = False
SELECTED_BURDENS = [0.0, 90.0, 180.0]
DIAGNOSTIC_SUPPORT_BURDENS = [0.0, 90.0, 180.0, 220.0]
PRIMARY_GRID_MAX = 180.0
SUPPORT_WINDOW_HALF_WIDTH = 15.0
CONTRAST_WINDOWS = ((0.0, 180.0), (90.0, 180.0))
BETWEEN_HISTORY_CONTRAST_BURDENS = (0.0, 180.0)
MODEL_GROUPS = ("regular", "fragile")
PUBLICATION_GROUP_COLUMN = "publication_history_stratum"
PUBLICATION_HISTORY_LABELS = {
    "tough": "lower prior-injury-history",
    "regular": "intermediate prior-injury-history",
    "fragile": "higher prior-injury-history",
}
MULTIPLICITY_FAMILY_ORDER = [
    "primary",
    "outcome definitions",
    "prior-history definitions",
    "calendar and comparator restrictions",
    "covariate and recurrent-event checks",
]
MULTIPLICITY_ROLE_FAMILIES = {
    "primary": "primary",
    "outcome_sensitivity": "outcome definitions",
    "outcome_history_cross_sensitivity": "outcome definitions",
    "history_definition_sensitivity": "prior-history definitions",
    "calendar_sensitivity": "calendar and comparator restrictions",
    "comparator_sensitivity": "calendar and comparator restrictions",
    "selection_control_sensitivity": "covariate and recurrent-event checks",
    "overadjustment_sensitivity": "covariate and recurrent-event checks",
    "recurrent_event_sensitivity": "covariate and recurrent-event checks",
}

PRIMARY_EVENT_COL = "injury_event_matchproxy"
PRIMARY_GROUP_COL = "fragility_group"
NON_MUSCLE_HISTORY_GROUP_COL = "fragility_non_muscle_frequency_only"
NON_MUSCLE_HISTORY_THRESHOLD_COL = "q3_non_muscle_freq"
OUT_OF_TIME_GROUP_COL = "fragility_out_of_time_2017_2019_to_2020_2024"
MUSCLE_TENDON_HISTORY_COUNT_COL = "prior_muscle_tendon_n_spells"
MUSCLE_TENDON_HISTORY_RATE_COL = "prior_muscle_tendon_injuries_per_10000min"
MUSCLE_TENDON_HISTORY_HIGH_COL = "prior_muscle_tendon_high_frequency"
MUSCLE_TENDON_DAYS_SINCE_LAST_REPORT_COL = "days_since_last_prior_muscle_tendon_report"
MUSCLE_TENDON_HAS_PRIOR_REPORT_COL = "has_prior_muscle_tendon_report"
MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL = (
    "log1p_days_since_last_prior_muscle_tendon_report"
)
JOINT_BONE_HISTORY_COUNT_COL = "prior_joint_bone_n_spells"
JOINT_BONE_HISTORY_RATE_COL = "prior_joint_bone_injuries_per_10000min"
JOINT_BONE_HISTORY_HIGH_COL = "prior_joint_bone_high_frequency"
JOINT_BONE_DAYS_SINCE_LAST_REPORT_COL = "days_since_last_prior_joint_bone_report"
JOINT_BONE_HAS_PRIOR_REPORT_COL = "has_prior_joint_bone_report"
JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL = (
    "log1p_days_since_last_prior_joint_bone_report"
)
MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS = (
    MUSCLE_TENDON_HISTORY_RATE_COL,
    JOINT_BONE_HISTORY_RATE_COL,
)
OUT_OF_TIME_DERIVATION_PERIOD = {
    "period": "2017-2019",
    "season_start_min": 2017,
    "season_start_max": 2019,
}
OUT_OF_TIME_TEST_PERIOD = {
    "period": "2020-2024",
    "season_start_min": 2020,
    "season_start_max": 2024,
}

SENSITIVITY_EVENT_COLS = {
    "same_day_plus_lag1": "injury_event_matchproxy",
    "same_day_only": "injury_event_matchproxy_same_day",
    "lag1_only": "injury_event_matchproxy_lag1",
    "specific_description_only": "injury_event_matchproxy_specific",
    "reported_absence_ge28d": "injury_event_matchproxy_ge28d",
    "muscle_tendon_only": "injury_event_matchproxy_muscle_tendon",
}
OUTCOME_HISTORY_CROSS_SPECS = (
    {
        "model": "reported_absence_ge28d_frequency_only_history",
        "event_col": "injury_event_matchproxy_ge28d",
        "group_col": "fragility_frequency_only",
    },
    {
        "model": "muscle_tendon_only_frequency_only_history",
        "event_col": "injury_event_matchproxy_muscle_tendon",
        "group_col": "fragility_frequency_only",
    },
    {
        "model": "muscle_tendon_only_non_muscle_frequency_history",
        "event_col": "injury_event_matchproxy_muscle_tendon",
        "group_col": NON_MUSCLE_HISTORY_GROUP_COL,
    },
)

LINEUP_START_TYPES = {"starting_lineup"}
LINEUP_SUBSTITUTE_TYPES = {"substitutes"}

MATCHPROXY_DURATION_BUCKETS = [
    "<1 week",
    "1 week to 2 months",
    "2 months to 1 year",
    ">1 year",
    "unknown",
]
SEVERE_REPORTED_ABSENCE_DAYS = 28.0
MUSCLE_TENDON_PATTERN = (
    r"hamstring|muscle|calf|thigh|adductor|groin|quad|quadriceps|achilles|tendon"
)
KNOWN_NON_MUSCLE_HISTORY_TYPES = (
    "joint/ligament",
    "bone/fracture",
)
SPLINE_SHAPE_SENSITIVITY_SPECS = [
    {
        "specification": "bs_df3_quantile_knots",
        "spline_basis": "bs",
        "spline_df": 3,
        "spline_knots": None,
    },
    {
        "specification": "bs_df4_quantile_knots_primary",
        "spline_basis": "bs",
        "spline_df": 4,
        "spline_knots": None,
    },
    {
        "specification": "bs_df5_quantile_knots",
        "spline_basis": "bs",
        "spline_df": 5,
        "spline_knots": None,
    },
    {
        "specification": "cr_df3_quantile_knots",
        "spline_basis": "cr",
        "spline_df": 3,
        "spline_knots": None,
    },
    {
        "specification": "cr_df4_quantile_knots",
        "spline_basis": "cr",
        "spline_df": 4,
        "spline_knots": None,
    },
    {
        "specification": "cr_df5_quantile_knots",
        "spline_basis": "cr",
        "spline_df": 5,
        "spline_knots": None,
    },
    {
        "specification": "bs_fixed_knots_45_90_135",
        "spline_basis": "bs",
        "spline_df": None,
        "spline_knots": (45.0, 90.0, 135.0),
    },
    {
        "specification": "cr_fixed_knots_45_90_135",
        "spline_basis": "cr",
        "spline_df": None,
        "spline_knots": (45.0, 90.0, 135.0),
    },
]
RECOVERY_MODEL_SPECS = [
    ("observed_minutes_poisson", "poisson", "observed_minutes"),
    ("fixed_90_poisson", "poisson", "fixed_90"),
    ("per_match_logit", "binomial_logit", "per_match"),
    ("per_match_cloglog", "binomial_cloglog", "per_match"),
]

PRIOR_HISTORY_CONTROL_COLS = [
    "log_prior_minutes_played",
    "prior_n_spells",
    "log_prior_injuries_per_10000min",
    "log_prior_max_spell_duration_days",
]

TEMPORAL_PERIODS = [
    {
        "period": "2017-2019",
        "season_start_min": 2017,
        "season_start_max": 2019,
    },
    {
        "period": "2020-2021",
        "season_start_min": 2020,
        "season_start_max": 2021,
    },
    {
        "period": "2022-2024",
        "season_start_min": 2022,
        "season_start_max": 2024,
    },
]

INTERNATIONAL_BREAK_WINDOWS = [
    ("2017-08-28", "2017-09-06"),
    ("2017-10-02", "2017-10-10"),
    ("2017-11-06", "2017-11-14"),
    ("2018-03-19", "2018-03-27"),
    ("2018-09-03", "2018-09-11"),
    ("2018-10-08", "2018-10-16"),
    ("2018-11-12", "2018-11-20"),
    ("2019-03-18", "2019-03-26"),
    ("2019-09-02", "2019-09-10"),
    ("2019-10-07", "2019-10-15"),
    ("2019-11-11", "2019-11-19"),
    ("2020-09-01", "2020-09-09"),
    ("2020-10-05", "2020-10-13"),
    ("2020-11-09", "2020-11-18"),
    ("2021-03-22", "2021-03-31"),
    ("2021-08-30", "2021-09-08"),
    ("2021-10-04", "2021-10-12"),
    ("2021-11-08", "2021-11-16"),
    ("2022-03-21", "2022-03-29"),
    ("2022-09-19", "2022-09-27"),
    ("2023-03-20", "2023-03-28"),
    ("2023-09-04", "2023-09-12"),
    ("2023-10-09", "2023-10-17"),
    ("2023-11-13", "2023-11-21"),
    ("2024-03-18", "2024-03-26"),
    ("2024-09-02", "2024-09-10"),
    ("2024-10-07", "2024-10-15"),
    ("2024-11-11", "2024-11-19"),
    ("2025-03-17", "2025-03-25"),
]

COVID_DISRUPTION_WINDOWS = [
    ("2020-03-13", "2020-09-11"),
    ("2020-09-12", "2021-05-23"),
]
RECOVERY_INTERVAL_ORDER = [
    "0-3 days",
    "4-5 days",
    "6-7 days",
    "8-14 days",
    ">14 days",
    "no prior match",
]
RECOVERY_TREND_ORDER = [
    "0-3 days",
    "4-5 days",
    "6-7 days",
    "8-14 days",
    ">14 days",
]
SELECTION_AUDIT_BANDS = [
    ("0 min", 0.0, 0.0),
    ("15-45 min peak band", 15.0, 45.0),
    ("90-95 min trough band", 90.0, 95.0),
    ("180 min band", 165.0, 195.0),
]


def season_from_dates(dates: pd.Series) -> pd.Series:
    """Infer football season start year from calendar dates."""
    dt = pd.to_datetime(dates, errors="coerce")
    return np.where(dt.dt.month >= 7, dt.dt.year, dt.dt.year - 1)


def estimate_history_thresholds(day_history: pd.DataFrame) -> Dict[str, float]:
    """Estimate canonical Q1/Q3 thresholds from latest eligible player rows."""
    latest = (
        day_history.sort_values([PLAYER_ID_COL, "date"])
        .groupby(PLAYER_ID_COL, as_index=False)
        .tail(1)
    )
    eligible = latest[latest["prior_minutes_played"] >= 900.0]
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


def assign_history_labels_from_thresholds(
    day_history: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.Series:
    """Assign the canonical prior-history labels using supplied thresholds."""
    adequate = day_history["prior_minutes_played"] >= 900.0
    lower_history = (
        adequate
        & (day_history["prior_n_spells"] <= 1)
        & (day_history["prior_injuries_per_10000min"] <= thresholds["q1_freq"])
        & (day_history["prior_max_spell_duration_days"] <= thresholds["q1_sev"])
    )
    higher_history = (
        adequate
        & (day_history["prior_n_spells"] >= 2)
        & (
            (day_history["prior_injuries_per_10000min"] >= thresholds["q3_freq"])
            | (day_history["prior_max_spell_duration_days"] >= thresholds["q3_sev"])
        )
    )

    labels = pd.Series("low_exposure", index=day_history.index, dtype="object")
    labels.loc[adequate] = "regular"
    labels.loc[lower_history] = "tough"
    labels.loc[higher_history] = "fragile"
    return labels


def add_out_of_time_fragility_label(
    panel: pd.DataFrame,
    derivation_period: Dict[str, object] = OUT_OF_TIME_DERIVATION_PERIOD,
    test_period: Dict[str, object] = OUT_OF_TIME_TEST_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derive prior-history thresholds in early seasons and apply them later.

    This sensitivity check keeps row-level inputs prior-only but avoids deriving
    the threshold cut points from the same later seasons where the model is
    evaluated.
    """
    out = panel.copy()
    out["season_start"] = pd.Series(season_from_dates(out["date"]), index=out.index)
    derivation_mask = out["season_start"].between(
        int(derivation_period["season_start_min"]),
        int(derivation_period["season_start_max"]),
        inclusive="both",
    )
    test_mask = out["season_start"].between(
        int(test_period["season_start_min"]),
        int(test_period["season_start_max"]),
        inclusive="both",
    )
    thresholds = estimate_history_thresholds(out.loc[derivation_mask].copy())
    out[OUT_OF_TIME_GROUP_COL] = "outside_test_window"
    out.loc[test_mask, OUT_OF_TIME_GROUP_COL] = assign_history_labels_from_thresholds(
        out.loc[test_mask].copy(),
        thresholds,
    )

    test_counts = out.loc[test_mask, OUT_OF_TIME_GROUP_COL].value_counts().to_dict()
    audit = {
        "derivation_period": str(derivation_period["period"]),
        "test_period": str(test_period["period"]),
        "threshold_source": "latest eligible player rows in derivation period",
        "test_labels_carried_forward": False,
        "q1_freq": thresholds["q1_freq"],
        "q3_freq": thresholds["q3_freq"],
        "q1_sev": thresholds["q1_sev"],
        "q3_sev": thresholds["q3_sev"],
        "derivation_rows": int(derivation_mask.sum()),
        "derivation_players": int(out.loc[derivation_mask, PLAYER_ID_COL].nunique()),
        "test_rows": int(test_mask.sum()),
        "test_players": int(out.loc[test_mask, PLAYER_ID_COL].nunique()),
        "test_low_exposure_rows": int(test_counts.get("low_exposure", 0)),
        "test_lower_history_rows": int(test_counts.get("tough", 0)),
        "test_intermediate_history_rows": int(test_counts.get("regular", 0)),
        "test_higher_history_rows": int(test_counts.get("fragile", 0)),
    }
    return out, pd.DataFrame([audit])


def add_alternative_fragility_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add simpler prior-history labels for sensitivity analyses.

    Each sensitivity label uses the same two group names as the primary model:
    ``regular`` for the lower-risk comparator and ``fragile`` for the higher
    prior-injury-history stratum. Low-exposure rows remain labelled
    ``low_exposure`` and are filtered out before model fitting.
    """
    out = panel.copy()
    adequate = out["prior_minutes_played"] >= 900.0

    out["fragility_count_only"] = "low_exposure"
    out.loc[adequate & (out["prior_n_spells"] < 2), "fragility_count_only"] = "regular"
    out.loc[adequate & (out["prior_n_spells"] >= 2), "fragility_count_only"] = "fragile"

    out["fragility_frequency_only"] = "low_exposure"
    out.loc[
        adequate & (out["prior_injuries_per_10000min"] < out["q3_freq"]),
        "fragility_frequency_only",
    ] = "regular"
    out.loc[
        adequate & (out["prior_injuries_per_10000min"] >= out["q3_freq"]),
        "fragility_frequency_only",
    ] = "fragile"

    out["fragility_severity_only"] = "low_exposure"
    out.loc[
        adequate & (out["prior_max_spell_duration_days"] < out["q3_sev"]),
        "fragility_severity_only",
    ] = "regular"
    out.loc[
        adequate & (out["prior_max_spell_duration_days"] >= out["q3_sev"]),
        "fragility_severity_only",
    ] = "fragile"

    out["fragility_prespecified_abs"] = "low_exposure"
    prespecified_fragile = (
        (out["prior_n_spells"] >= 2)
        | (out["prior_max_spell_duration_days"] >= 28)
    )
    out.loc[adequate & ~prespecified_fragile, "fragility_prespecified_abs"] = "regular"
    out.loc[adequate & prespecified_fragile, "fragility_prespecified_abs"] = "fragile"

    return out


def non_muscle_injury_day_counts(
    injuries: pd.DataFrame,
    panel_min_date: pd.Timestamp,
    panel_max_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return daily counts of classified musculoskeletal non-muscle injuries."""
    required = {PLAYER_ID_COL, "injury_spell_id", "start_date"}
    missing = required - set(injuries.columns)
    if missing:
        raise KeyError(f"injuries missing required columns: {sorted(missing)}")

    inj = injuries.copy()
    inj[PLAYER_ID_COL] = pd.to_numeric(inj[PLAYER_ID_COL], errors="coerce")
    inj["start_date"] = pd.to_datetime(inj["start_date"], errors="coerce")
    if "injury_desc" not in inj.columns:
        inj["injury_desc"] = ""
    inj["public_injury_type"] = inj["injury_desc"].apply(classify_public_injury_type)
    inj = inj[
        inj[PLAYER_ID_COL].notna()
        & inj["start_date"].notna()
        & inj["start_date"].between(panel_min_date, panel_max_date, inclusive="both")
        & inj["public_injury_type"].isin(KNOWN_NON_MUSCLE_HISTORY_TYPES)
    ].copy()
    if inj.empty:
        return pd.DataFrame(
            columns=[PLAYER_ID_COL, "date", "n_non_muscle_spells_today"]
        )
    inj[PLAYER_ID_COL] = inj[PLAYER_ID_COL].astype(int)
    inj["injury_spell_id"] = pd.to_numeric(inj["injury_spell_id"], errors="coerce")
    return (
        inj.dropna(subset=["injury_spell_id"])
        .groupby([PLAYER_ID_COL, "start_date"], as_index=False)
        .agg(n_non_muscle_spells_today=("injury_spell_id", "nunique"))
        .rename(columns={"start_date": "date"})
    )


def classified_injury_type_day_counts(
    injuries: pd.DataFrame,
    panel_min_date: pd.Timestamp,
    panel_max_date: pd.Timestamp,
    injury_types: Sequence[str],
    output_col: str,
) -> pd.DataFrame:
    """Return daily spell counts for one or more public injury-type classes."""
    required = {PLAYER_ID_COL, "injury_spell_id", "start_date"}
    missing = required - set(injuries.columns)
    if missing:
        raise KeyError(f"injuries missing required columns: {sorted(missing)}")
    if not injury_types:
        raise ValueError("injury_types must contain at least one class")

    inj = injuries.copy()
    inj[PLAYER_ID_COL] = pd.to_numeric(inj[PLAYER_ID_COL], errors="coerce")
    inj["start_date"] = pd.to_datetime(inj["start_date"], errors="coerce")
    if "injury_desc" not in inj.columns:
        inj["injury_desc"] = ""
    inj["public_injury_type"] = inj["injury_desc"].apply(classify_public_injury_type)
    inj = inj[
        inj[PLAYER_ID_COL].notna()
        & inj["start_date"].notna()
        & inj["start_date"].between(panel_min_date, panel_max_date, inclusive="both")
        & inj["public_injury_type"].isin(set(injury_types))
    ].copy()
    if inj.empty:
        return pd.DataFrame(columns=[PLAYER_ID_COL, "date", output_col])
    inj[PLAYER_ID_COL] = inj[PLAYER_ID_COL].astype(int)
    inj["injury_spell_id"] = pd.to_numeric(inj["injury_spell_id"], errors="coerce")
    return (
        inj.dropna(subset=["injury_spell_id"])
        .groupby([PLAYER_ID_COL, "start_date"], as_index=False)
        .agg(**{output_col: ("injury_spell_id", "nunique")})
        .rename(columns={"start_date": "date"})
    )


def prior_injury_type_start_dates(
    injuries: pd.DataFrame,
    injury_types: Sequence[str],
    output_col: str,
) -> pd.DataFrame:
    """Return prior injury-report start dates for selected public injury classes."""
    required = {PLAYER_ID_COL, "start_date"}
    missing = required - set(injuries.columns)
    if missing:
        raise KeyError(f"injuries missing required columns: {sorted(missing)}")
    if not injury_types:
        raise ValueError("injury_types must contain at least one class")

    inj = injuries.copy()
    inj[PLAYER_ID_COL] = pd.to_numeric(inj[PLAYER_ID_COL], errors="coerce")
    inj["start_date"] = pd.to_datetime(inj["start_date"], errors="coerce")
    if "injury_desc" not in inj.columns:
        inj["injury_desc"] = ""
    inj["public_injury_type"] = inj["injury_desc"].apply(classify_public_injury_type)
    inj = inj[
        inj[PLAYER_ID_COL].notna()
        & inj["start_date"].notna()
        & inj["public_injury_type"].isin(set(injury_types))
    ].copy()
    if inj.empty:
        return pd.DataFrame(columns=[PLAYER_ID_COL, output_col])
    inj[PLAYER_ID_COL] = inj[PLAYER_ID_COL].astype(int)
    return (
        inj[[PLAYER_ID_COL, "start_date"]]
        .drop_duplicates()
        .sort_values([PLAYER_ID_COL, "start_date"])
        .rename(columns={"start_date": output_col})
        .reset_index(drop=True)
    )


def add_prior_injury_type_recency(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    injury_types: Sequence[str],
    days_col: str,
    has_col: str,
    log_col: str,
    start_col: str,
) -> pd.DataFrame:
    """Add row-level days since the last strictly prior report for selected types."""
    required = {PLAYER_ID_COL, "date"}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[PLAYER_ID_COL] = pd.to_numeric(out[PLAYER_ID_COL], errors="coerce")
    out[days_col] = np.nan
    out[has_col] = 0
    out[log_col] = 0.0

    base = (
        out[[PLAYER_ID_COL, "date"]]
        .dropna(subset=[PLAYER_ID_COL, "date"])
        .copy()
    )
    if base.empty:
        return out
    base[PLAYER_ID_COL] = base[PLAYER_ID_COL].astype(int)
    base["_row_index"] = base.index

    starts = prior_injury_type_start_dates(
        injuries,
        injury_types,
        start_col,
    )
    if starts.empty:
        return out

    pieces: List[pd.DataFrame] = []
    for player_id, player_rows in base.groupby(PLAYER_ID_COL, sort=False):
        player_starts = starts[starts[PLAYER_ID_COL].eq(player_id)]
        if player_starts.empty:
            continue
        matched = pd.merge_asof(
            player_rows.sort_values("date"),
            player_starts.sort_values(start_col),
            left_on="date",
            right_on=start_col,
            direction="backward",
            allow_exact_matches=False,
        )
        pieces.append(matched)
    if not pieces:
        return out

    matched = pd.concat(pieces, ignore_index=True)
    days_since = (
        matched["date"] - pd.to_datetime(matched[start_col], errors="coerce")
    ).dt.days
    valid = days_since.notna()
    if not valid.any():
        return out
    row_indices = matched.loc[valid, "_row_index"].astype(int)
    days = days_since.loc[valid].astype(float).clip(lower=0.0)
    out.loc[row_indices, days_col] = days.to_numpy()
    out.loc[row_indices, has_col] = 1
    out.loc[row_indices, log_col] = np.log1p(days.to_numpy())
    return out


def add_prior_muscle_tendon_recency(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """Add row-level days since the last strictly prior muscle/tendon report."""
    return add_prior_injury_type_recency(
        panel,
        injuries,
        ("muscle/tendon",),
        MUSCLE_TENDON_DAYS_SINCE_LAST_REPORT_COL,
        MUSCLE_TENDON_HAS_PRIOR_REPORT_COL,
        MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL,
        "last_prior_muscle_tendon_report_start_date",
    )


def add_prior_joint_bone_recency(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """Add row-level days since the last strictly prior joint/ligament or bone report."""
    return add_prior_injury_type_recency(
        panel,
        injuries,
        KNOWN_NON_MUSCLE_HISTORY_TYPES,
        JOINT_BONE_DAYS_SINCE_LAST_REPORT_COL,
        JOINT_BONE_HAS_PRIOR_REPORT_COL,
        JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL,
        "last_prior_joint_bone_report_start_date",
    )


def add_symmetric_type_recency(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """Add matched recency controls for both injury-type history variables."""
    return add_prior_joint_bone_recency(
        add_prior_muscle_tendon_recency(panel, injuries),
        injuries,
    )


def symmetric_type_recency_terms() -> str:
    """Return the matched type-specific recency terms used in type-history models."""
    return (
        f"+ {MUSCLE_TENDON_HAS_PRIOR_REPORT_COL} "
        f"+ {MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL} "
        f"+ {JOINT_BONE_HAS_PRIOR_REPORT_COL} "
        f"+ {JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL}"
    )


def symmetric_type_recency_summary(frame: pd.DataFrame) -> Dict[str, float]:
    """Summarise the missing-indicator recency coding for both history types."""
    muscle_days_since = pd.to_numeric(
        frame[MUSCLE_TENDON_DAYS_SINCE_LAST_REPORT_COL],
        errors="coerce",
    )
    joint_days_since = pd.to_numeric(
        frame[JOINT_BONE_DAYS_SINCE_LAST_REPORT_COL],
        errors="coerce",
    )
    return {
        "rows_with_prior_muscle_tendon_report": int(
            frame[MUSCLE_TENDON_HAS_PRIOR_REPORT_COL].sum()
        ),
        "rows_without_prior_muscle_tendon_report": int(
            frame[MUSCLE_TENDON_HAS_PRIOR_REPORT_COL].eq(0).sum()
        ),
        "median_days_since_last_prior_muscle_tendon_report": (
            float(muscle_days_since.dropna().median())
            if muscle_days_since.notna().any()
            else np.nan
        ),
        "rows_with_prior_joint_bone_report": int(
            frame[JOINT_BONE_HAS_PRIOR_REPORT_COL].sum()
        ),
        "rows_without_prior_joint_bone_report": int(
            frame[JOINT_BONE_HAS_PRIOR_REPORT_COL].eq(0).sum()
        ),
        "median_days_since_last_prior_joint_bone_report": (
            float(joint_days_since.dropna().median())
            if joint_days_since.notna().any()
            else np.nan
        ),
    }


def type_frequency_recency_collinearity_summary(frame: pd.DataFrame) -> Dict[str, float]:
    """Summarise frequency-recency correlations for type-history sensitivity models."""
    specs = (
        (
            "muscle_tendon",
            MUSCLE_TENDON_HISTORY_RATE_COL,
            MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL,
            MUSCLE_TENDON_HAS_PRIOR_REPORT_COL,
        ),
        (
            "joint_bone",
            JOINT_BONE_HISTORY_RATE_COL,
            JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL,
            JOINT_BONE_HAS_PRIOR_REPORT_COL,
        ),
    )
    out: Dict[str, float] = {}
    for prefix, frequency_col, log_days_col, has_col in specs:
        frequency = pd.to_numeric(frame[frequency_col], errors="coerce")
        log_days = pd.to_numeric(frame[log_days_col], errors="coerce")
        has_prior = frame[has_col].eq(1)
        out[f"{prefix}_frequency_log_recency_corr_all_rows"] = float(
            frequency.corr(log_days)
        )
        out[f"{prefix}_frequency_log_recency_corr_prior_rows"] = float(
            frequency[has_prior].corr(log_days[has_prior])
        )
    return out


def fitted_term_variance_inflation_factor(result: object, term: str) -> float:
    """Return a term's VIF from the exact fitted-model design matrix."""
    names = list(result.model.exog_names)
    if term not in names:
        raise KeyError(f"model does not contain term: {term}")
    term_index = names.index(term)
    return float(variance_inflation_factor(result.model.exog, term_index))


def add_mutually_exclusive_type_frequency_history(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add prior-only muscle/tendon and joint/bone injury-frequency histories.

    These two history variables are mutually exclusive by public injury-type
    class. They allow a same-row model to estimate prior muscle/tendon history
    over and above prior joint/ligament or bone/fracture history.
    """
    required = {PLAYER_ID_COL, "date", "prior_minutes_played"}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[PLAYER_ID_COL] = pd.to_numeric(out[PLAYER_ID_COL], errors="coerce")
    base = (
        out[[PLAYER_ID_COL, "date", "prior_minutes_played"]]
        .dropna(subset=[PLAYER_ID_COL, "date"])
        .drop_duplicates([PLAYER_ID_COL, "date"])
        .sort_values([PLAYER_ID_COL, "date"])
        .copy()
    )
    history_cols = [
        MUSCLE_TENDON_HISTORY_COUNT_COL,
        MUSCLE_TENDON_HISTORY_RATE_COL,
        JOINT_BONE_HISTORY_COUNT_COL,
        JOINT_BONE_HISTORY_RATE_COL,
    ]
    if base.empty:
        for col in history_cols:
            out[col] = 0.0
        return out

    hist = base.copy()
    type_specs = [
        (
            ("muscle/tendon",),
            "n_muscle_tendon_spells_today",
            MUSCLE_TENDON_HISTORY_COUNT_COL,
            MUSCLE_TENDON_HISTORY_RATE_COL,
        ),
        (
            KNOWN_NON_MUSCLE_HISTORY_TYPES,
            "n_joint_bone_spells_today",
            JOINT_BONE_HISTORY_COUNT_COL,
            JOINT_BONE_HISTORY_RATE_COL,
        ),
    ]
    for injury_types, today_col, prior_count_col, prior_rate_col in type_specs:
        counts = classified_injury_type_day_counts(
            injuries,
            hist["date"].min(),
            hist["date"].max(),
            injury_types,
            today_col,
        )
        hist = hist.merge(counts, on=[PLAYER_ID_COL, "date"], how="left")
        hist[today_col] = hist[today_col].fillna(0.0)
        grouped = hist.groupby(PLAYER_ID_COL, group_keys=False)
        hist[prior_count_col] = grouped[today_col].transform(
            lambda s: s.cumsum().shift(1)
        ).fillna(0.0)
        prior_minutes = hist["prior_minutes_played"].astype(float).clip(lower=0.0)
        hist[prior_rate_col] = np.where(
            prior_minutes > 0.0,
            hist[prior_count_col] / prior_minutes * 10000.0,
            0.0,
        )

    keep = [PLAYER_ID_COL, "date", *history_cols]
    return out.merge(hist[keep], on=[PLAYER_ID_COL, "date"], how="left")


def add_non_muscle_frequency_history_label(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    frequency_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Add a type-discordant prior-history label based on non-muscle injuries.

    The count uses joint/ligament and bone/fracture public descriptions only.
    Illness, concussion/head, unknown, and other/unspecified descriptions are
    excluded so the history label does not silently mix non-injury or vague
    tissue categories into the muscle/tendon outcome check. Like the main
    prior-history table, each row only sees injuries that started before that
    row's date.
    """
    required = {PLAYER_ID_COL, "date", "prior_minutes_played"}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[PLAYER_ID_COL] = pd.to_numeric(out[PLAYER_ID_COL], errors="coerce")
    base = (
        out[[PLAYER_ID_COL, "date", "prior_minutes_played"]]
        .dropna(subset=[PLAYER_ID_COL, "date"])
        .drop_duplicates([PLAYER_ID_COL, "date"])
        .sort_values([PLAYER_ID_COL, "date"])
        .copy()
    )
    if base.empty:
        out["prior_non_muscle_n_spells"] = 0.0
        out["prior_non_muscle_injuries_per_10000min"] = 0.0
        out[NON_MUSCLE_HISTORY_THRESHOLD_COL] = 0.0
        out[NON_MUSCLE_HISTORY_GROUP_COL] = "low_exposure"
        return out

    counts = non_muscle_injury_day_counts(
        injuries,
        base["date"].min(),
        base["date"].max(),
    )
    hist = base.merge(counts, on=[PLAYER_ID_COL, "date"], how="left")
    hist["n_non_muscle_spells_today"] = hist[
        "n_non_muscle_spells_today"
    ].fillna(0.0)
    grouped = hist.groupby(PLAYER_ID_COL, group_keys=False)
    hist["prior_non_muscle_n_spells"] = grouped[
        "n_non_muscle_spells_today"
    ].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    prior_minutes = hist["prior_minutes_played"].astype(float).clip(lower=0.0)
    hist["prior_non_muscle_injuries_per_10000min"] = np.where(
        prior_minutes > 0.0,
        hist["prior_non_muscle_n_spells"] / prior_minutes * 10000.0,
        0.0,
    )

    if frequency_threshold is None:
        latest = hist.groupby(PLAYER_ID_COL, as_index=False).tail(1)
        eligible = latest[latest["prior_minutes_played"].astype(float) >= 900.0]
        frequency_threshold = (
            float(eligible["prior_non_muscle_injuries_per_10000min"].quantile(0.75))
            if not eligible.empty
            else 0.0
        )

    hist[NON_MUSCLE_HISTORY_THRESHOLD_COL] = float(frequency_threshold)
    adequate = hist["prior_minutes_played"].astype(float) >= 900.0
    high_frequency = (
        (hist["prior_non_muscle_n_spells"].astype(float) > 0.0)
        & (
            hist["prior_non_muscle_injuries_per_10000min"].astype(float)
            >= float(frequency_threshold)
        )
    )
    hist[NON_MUSCLE_HISTORY_GROUP_COL] = "low_exposure"
    hist.loc[adequate & ~high_frequency, NON_MUSCLE_HISTORY_GROUP_COL] = "regular"
    hist.loc[adequate & high_frequency, NON_MUSCLE_HISTORY_GROUP_COL] = "fragile"

    keep = [
        PLAYER_ID_COL,
        "date",
        "prior_non_muscle_n_spells",
        "prior_non_muscle_injuries_per_10000min",
        NON_MUSCLE_HISTORY_THRESHOLD_COL,
        NON_MUSCLE_HISTORY_GROUP_COL,
    ]
    return out.merge(hist[keep], on=[PLAYER_ID_COL, "date"], how="left")


def epl_club_seasons(games: pd.DataFrame) -> pd.DataFrame:
    """Return season/club pairs for clubs appearing in GB1 in each season."""
    epl = games[games["competition_id"] == "GB1"]
    home = epl[["season", "home_club_id"]].rename(columns={"home_club_id": "player_club_id"})
    away = epl[["season", "away_club_id"]].rename(columns={"away_club_id": "player_club_id"})
    return (
        pd.concat([home, away], ignore_index=True)
        .dropna()
        .drop_duplicates()
        .assign(
            season=lambda d: d["season"].astype(int),
            player_club_id=lambda d: d["player_club_id"].astype(int),
        )
    )


def add_player_and_club_metadata(match_panel: pd.DataFrame, tm_dir: Path) -> pd.DataFrame:
    """Merge age, position, and dominant all-competition match-day club-season."""
    out = match_panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    players = pd.read_csv(
        tm_dir / "players.csv",
        usecols=["player_id", "date_of_birth", "position"],
        low_memory=False,
    ).rename(columns={"player_id": PLAYER_ID_COL})
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    out = out.merge(players, on=PLAYER_ID_COL, how="left")
    out["age_years"] = ((out["date"] - out["date_of_birth"]).dt.days / 365.25).fillna(
        out["date"].dt.year.sub(1990).clip(lower=16, upper=45)
    )
    out["position_group"] = out["position"].fillna("Unknown").astype(str)

    keys = out[[PLAYER_ID_COL, "date"]].drop_duplicates()
    games = pd.read_csv(
        tm_dir / "games.csv",
        usecols=["game_id", "competition_id", "season", "home_club_id", "away_club_id"],
        low_memory=False,
    )
    club_seasons = epl_club_seasons(games)
    apps = pd.read_csv(
        tm_dir / "appearances.csv",
        usecols=["game_id", "player_id", "player_club_id", "date", "minutes_played"],
        low_memory=False,
    ).rename(columns={"player_id": PLAYER_ID_COL})
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps = apps.merge(games[["game_id", "season"]], on="game_id", how="left")
    apps = apps.merge(club_seasons, on=["season", "player_club_id"], how="inner")
    apps = apps.merge(keys, on=[PLAYER_ID_COL, "date"], how="inner")
    apps = apps.sort_values([PLAYER_ID_COL, "date", "minutes_played"], ascending=[True, True, False])
    club_day = apps.groupby([PLAYER_ID_COL, "date"], as_index=False).first()[
        [PLAYER_ID_COL, "date", "season", "player_club_id"]
    ]

    out = out.merge(club_day, on=[PLAYER_ID_COL, "date"], how="left")
    out["season"] = out["season"].fillna(pd.Series(season_from_dates(out["date"]), index=out.index))
    out["season"] = out["season"].fillna(-1).astype(int).astype(str)
    out["player_club_id"] = out["player_club_id"].fillna(-1).astype(int).astype(str)
    out["club_season"] = out["season"] + "_" + out["player_club_id"]
    return out


def prepare_model_frame(
    panel: pd.DataFrame,
    event_col: str,
    group_col: str,
    include_tough: bool = INCLUDE_TOUGH,
) -> pd.DataFrame:
    """Return match rows ready for an intermediate/higher-history model."""
    missing = [c for c in [event_col, group_col, MATCH_MINUTES_COL] if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing required columns {missing}")

    match_panel = panel[panel[MATCH_MINUTES_COL] > 0].copy()
    match_panel["model_group"] = match_panel[group_col].astype(str)
    allowed = ["regular", "fragile"] + (["tough"] if include_tough else [])
    match_panel = match_panel[match_panel["model_group"].isin(allowed)].copy()
    match_panel[event_col] = match_panel[event_col].fillna(0).astype(int)
    match_panel["log_minutes_played"] = np.log(
        match_panel[MATCH_MINUTES_COL].astype(float).clip(lower=1.0)
    )
    for col in [
        "all_minutes_last_7d",
        "excess_minutes_last7d",
        "any_extra_time_last7d",
        "week_phase_sin",
        "week_phase_cos",
        "halfweek_phase_sin",
        "halfweek_phase_cos",
        "prior7_overlaps_international_break",
        "covid_disrupted_date",
    ]:
        match_panel[col] = match_panel[col].fillna(0.0) if col in match_panel else 0.0
    if "days_since_last_match" in match_panel:
        match_panel["days_since_last_match"] = pd.to_numeric(
            match_panel["days_since_last_match"], errors="coerce"
        )
    if (
        "days_since_last_match" in match_panel
        or "recovery_interval_bin" in match_panel
        or "recovery_interval_refined" in match_panel
    ):
        match_panel = add_refined_recovery_interval(match_panel)
    match_panel = add_clean_comparator_flag(match_panel)
    return match_panel


def add_prior_history_control_columns(match_panel: pd.DataFrame) -> pd.DataFrame:
    """Add stable continuous prior-history controls for sensitivity models."""
    out = match_panel.copy()
    numeric_defaults = {
        "prior_minutes_played": 0.0,
        "prior_n_spells": 0.0,
        "prior_injuries_per_10000min": 0.0,
        "prior_max_spell_duration_days": 0.0,
    }
    for col, default in numeric_defaults.items():
        values = out[col] if col in out else pd.Series(default, index=out.index)
        out[col] = pd.to_numeric(values, errors="coerce").fillna(default)
        out[col] = out[col].clip(lower=0.0)

    out["log_prior_minutes_played"] = np.log1p(out["prior_minutes_played"])
    out["log_prior_injuries_per_10000min"] = np.log1p(
        out["prior_injuries_per_10000min"]
    )
    out["log_prior_max_spell_duration_days"] = np.log1p(
        out["prior_max_spell_duration_days"]
    )
    return out


def add_temporal_period_labels(
    panel: pd.DataFrame,
    periods: Optional[List[Dict[str, object]]] = None,
) -> pd.DataFrame:
    """
    Label rows by football-season period without recalibrating fragility.

    The existing day-level prior-history labels are carried forward into each
    period. This preserves the intended behaviour that a player can enter a
    later period already labelled fragile, regular, or tough based on earlier
    observed history.
    """
    out = panel.copy()
    period_defs = periods if periods is not None else TEMPORAL_PERIODS
    if "date" not in out:
        raise KeyError("panel must contain date")

    out["season_start"] = pd.Series(season_from_dates(out["date"]), index=out.index)
    out["temporal_period"] = pd.NA
    for period in period_defs:
        label = str(period["period"])
        season_min = int(period["season_start_min"])
        season_max = int(period["season_start_max"])
        in_period = out["season_start"].between(season_min, season_max, inclusive="both")
        out.loc[in_period, "temporal_period"] = label
    return out


def parse_duration_days(value) -> float:
    """Extract numeric reported absence days from a Transfermarkt duration field."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    payload = value
    if isinstance(value, str):
        if not value.strip():
            return np.nan
        try:
            import ast

            payload = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return np.nan
    if not isinstance(payload, dict):
        return np.nan
    try:
        return float(payload.get("days"))
    except (TypeError, ValueError):
        return np.nan


def matchproxy_duration_bucket(days: float) -> str:
    """Map reported absence duration to broad severity-proxy buckets."""
    if pd.isna(days):
        return "unknown"
    if days < 7.0:
        return "<1 week"
    if days <= 60.0:
        return "1 week to 2 months"
    if days <= 365.0:
        return "2 months to 1 year"
    return ">1 year"


def classify_public_injury_type(description: object) -> str:
    """Classify public injury text into broad mechanism-adjacent categories."""
    desc = "" if pd.isna(description) else str(description).strip().lower()
    if not desc:
        return "unknown"
    if pd.Series([desc]).str.contains(MUSCLE_TENDON_PATTERN, regex=True).iloc[0]:
        return "muscle/tendon"
    if pd.Series([desc]).str.contains(
        r"acl|cruciate|ligament|meniscus|knee|ankle|sprain|shoulder",
        regex=True,
    ).iloc[0]:
        return "joint/ligament"
    if pd.Series([desc]).str.contains(r"fracture|broken|metatarsal|toe|foot|bone", regex=True).iloc[0]:
        return "bone/fracture"
    if pd.Series([desc]).str.contains(r"concussion|head", regex=True).iloc[0]:
        return "head/concussion"
    if pd.Series([desc]).str.contains(r"illness|virus|covid|infection|flu|sick", regex=True).iloc[0]:
        return "illness/other medical"
    if "unknown" in desc:
        return "unknown"
    return "other/unspecified"


def split_spell_ids(value) -> list[str]:
    """Split raw numeric or canonical episode identifiers."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    ids: list[str] = []
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


def injury_spell_metadata_lookup(injuries: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-spell lookup for duration and description metadata."""
    identifier = (
        "injury_episode_id"
        if "injury_episode_id" in injuries.columns
        else "injury_spell_id"
    )
    required = {identifier, "start_date"}
    missing = required - set(injuries.columns)
    if missing:
        raise KeyError(f"injuries missing required columns: {sorted(missing)}")
    out = injuries.copy()
    out["injury_spell_id"] = out[identifier].astype(str)
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    if "end_date" in out.columns:
        out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    else:
        out["end_date"] = pd.NaT
    if identifier == "injury_episode_id" and "duration_days" in out:
        out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce")
    else:
        details = out.get("durationDetails", pd.Series(np.nan, index=out.index))
        out["duration_days"] = details.apply(parse_duration_days)
        fallback = (out["end_date"] - out["start_date"]).dt.days.clip(lower=0)
        out["duration_days"] = out["duration_days"].fillna(fallback)
    if "injury_desc" not in out.columns:
        out["injury_desc"] = ""
    out["duration_bucket"] = out["duration_days"].apply(matchproxy_duration_bucket)
    out["public_injury_type"] = out["injury_desc"].apply(classify_public_injury_type)
    keep = [
        "injury_spell_id",
        "duration_days",
        "duration_bucket",
        "injury_desc",
        "public_injury_type",
    ]
    return out[keep].drop_duplicates("injury_spell_id")


def summarize_matchproxy_spell_ids(spell_ids, lookup: pd.DataFrame) -> Dict[str, object]:
    """Summarise one or more back-attributed injury spell ids for a match row."""
    ids = split_spell_ids(spell_ids)
    if not ids:
        return {
            "matchproxy_duration_days": np.nan,
            "matchproxy_duration_bucket": "unknown",
            "matchproxy_public_injury_type": "unknown",
            "matchproxy_lookup_desc": "",
        }
    rows = lookup[lookup["injury_spell_id"].isin(ids)]
    if rows.empty:
        return {
            "matchproxy_duration_days": np.nan,
            "matchproxy_duration_bucket": "unknown",
            "matchproxy_public_injury_type": "unknown",
            "matchproxy_lookup_desc": "",
        }
    duration = rows["duration_days"].dropna()
    duration_days = float(duration.max()) if not duration.empty else np.nan
    type_counts = rows["public_injury_type"].value_counts()
    injury_type = str(type_counts.index[0]) if not type_counts.empty else "unknown"
    return {
        "matchproxy_duration_days": duration_days,
        "matchproxy_duration_bucket": matchproxy_duration_bucket(duration_days),
        "matchproxy_public_injury_type": injury_type,
        "matchproxy_lookup_desc": "; ".join(rows["injury_desc"].fillna("").astype(str).unique()),
    }


def add_matchproxy_outcome_subsets(panel: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Add severity- and muscle/tendon-restricted match-proxy outcomes.

    Same-day rows use the current row's injury spell id. Lag-1 rows use the next
    player-day's injury spell id, mirroring the match-proxy back-attribution.
    """
    required = {
        PLAYER_ID_COL,
        "date",
        "injury_spell_id",
        PRIMARY_EVENT_COL,
        "injury_event_matchproxy_same_day",
        "injury_event_matchproxy_lag1",
    }
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")

    lookup = injury_spell_metadata_lookup(injuries)

    def per_player(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()
        next_spell = group["injury_spell_id"].shift(-1)
        group["matchproxy_spell_id"] = np.where(
            group["injury_event_matchproxy_same_day"].fillna(0).astype(int).eq(1),
            group["injury_spell_id"].fillna(""),
            np.where(
                group["injury_event_matchproxy_lag1"].fillna(0).astype(int).eq(1),
                next_spell.fillna(""),
                "",
            ),
        )
        return group

    out = pd.concat(
        [per_player(group) for _, group in panel.groupby(PLAYER_ID_COL, sort=False)],
        ignore_index=True,
    )
    meta = out["matchproxy_spell_id"].apply(
        lambda value: summarize_matchproxy_spell_ids(value, lookup)
    )
    meta_df = pd.DataFrame(list(meta))
    out = pd.concat([out.reset_index(drop=True), meta_df], axis=1)
    event = out[PRIMARY_EVENT_COL].fillna(0).astype(int).eq(1)
    fallback_type = out.get("matchproxy_injury_desc", pd.Series("", index=out.index)).apply(
        classify_public_injury_type
    )
    out["matchproxy_public_injury_type"] = np.where(
        out["matchproxy_public_injury_type"].eq("unknown"),
        fallback_type,
        out["matchproxy_public_injury_type"],
    )
    out["injury_event_matchproxy_ge28d"] = (
        event & (out["matchproxy_duration_days"].astype(float) >= SEVERE_REPORTED_ABSENCE_DAYS)
    ).astype(int)
    out["injury_event_matchproxy_muscle_tendon"] = (
        event & out["matchproxy_public_injury_type"].eq("muscle/tendon")
    ).astype(int)
    return out


def add_calendar_sensitivity_flags(
    panel: pd.DataFrame,
    international_windows: Sequence[tuple[str, str]] = INTERNATIONAL_BREAK_WINDOWS,
    covid_windows: Sequence[tuple[str, str]] = COVID_DISRUPTION_WINDOWS,
) -> pd.DataFrame:
    """Add prior-only calendar flags used for sensitivity restrictions."""
    if "date" not in panel:
        raise KeyError("panel must contain date")

    out = panel.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    prior_start = dates - pd.to_timedelta(7, unit="D")
    prior_end = dates - pd.to_timedelta(1, unit="D")

    intl_overlap = pd.Series(False, index=out.index)
    for start, end in international_windows:
        window_start = pd.Timestamp(start)
        window_end = pd.Timestamp(end)
        intl_overlap = intl_overlap | (
            prior_start.le(window_end) & prior_end.ge(window_start)
        )

    covid_date = pd.Series(False, index=out.index)
    for start, end in covid_windows:
        covid_date = covid_date | dates.between(
            pd.Timestamp(start),
            pd.Timestamp(end),
            inclusive="both",
        )

    out["prior7_overlaps_international_break"] = intl_overlap.fillna(False).astype(int)
    out["covid_disrupted_date"] = covid_date.fillna(False).astype(int)
    return out


def add_clean_comparator_flag(match_panel: pd.DataFrame) -> pd.DataFrame:
    """Flag zero-burden rows that follow no or very distant observed prior match."""
    out = match_panel.copy()
    if "zero_burden_long_rest" not in out:
        if "days_since_last_match" in out:
            days = pd.to_numeric(out["days_since_last_match"], errors="coerce")
        else:
            days = pd.Series(np.nan, index=out.index)
        out["zero_burden_long_rest"] = (
            (out["all_minutes_last_7d"].astype(float) <= 0.0)
            & (days.isna() | (days > 14.0))
        ).astype(int)
    else:
        out["zero_burden_long_rest"] = out["zero_burden_long_rest"].fillna(0).astype(int)
    out["clean_zero_or_positive_burden"] = (
        (out["all_minutes_last_7d"].astype(float) > 0.0)
        | (out["zero_burden_long_rest"] == 0)
    ).astype(int)
    return out


def delta_ratio_interval(
    params: np.ndarray,
    covariance: np.ndarray,
    numerator_design: np.ndarray,
    denominator_design: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Delta-method confidence interval for exp((x_num - x_den) beta)."""
    diff = np.asarray(numerator_design, dtype=float) - np.asarray(denominator_design, dtype=float)
    log_ratio = float(diff @ params)
    variance = float(diff @ covariance @ diff.T)
    se = float(np.sqrt(max(variance, 0.0)))
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    if se > 0.0:
        z_statistic = log_ratio / se
    elif np.isclose(log_ratio, 0.0):
        z_statistic = 0.0
    else:
        z_statistic = float(np.copysign(np.inf, log_ratio))
    with np.errstate(over="ignore"):
        rate_ratio = float(np.exp(log_ratio))
        ci_low = float(np.exp(log_ratio - z_critical * se))
        ci_high = float(np.exp(log_ratio + z_critical * se))
    return {
        "rate_ratio": rate_ratio,
        "rr_ci_low": ci_low,
        "rr_ci_high": ci_high,
        "log_rate_ratio": log_ratio,
        "log_rate_ratio_se": se,
        "z_statistic": float(z_statistic),
        "p_value": float(erfc(abs(z_statistic) / sqrt(2.0))),
    }


def count_rate_intervals(events: float, minutes: float) -> Dict[str, float]:
    """Return approximate Poisson count-rate intervals for match-minute rates."""
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
        high_10000 = -log(0.025) / float(minutes) * 10000.0
    else:
        log_margin = 1.96 / sqrt(float(events))
        low_10000 = rate_10000 * exp(-log_margin)
        high_10000 = rate_10000 * exp(log_margin)
    return {
        "events_per_10000_min": rate_10000,
        "events_per_1000_match_hours": rate_10000 * 6.0,
        "events_per_10000_min_ci_low": low_10000,
        "events_per_10000_min_ci_high": high_10000,
        "events_per_1000_match_hours_ci_low": low_10000 * 6.0,
        "events_per_1000_match_hours_ci_high": high_10000 * 6.0,
    }


def count_rate_ratio_interval(
    events_a: float,
    minutes_a: float,
    events_b: float,
    minutes_b: float,
) -> Dict[str, float]:
    """Approximate Poisson rate-ratio interval for two aggregate cells."""
    if minutes_a <= 0 or minutes_b <= 0 or events_a <= 0 or events_b <= 0:
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "fit_status": "not_estimable",
        }
    log_ratio = log((float(events_a) / float(minutes_a)) / (float(events_b) / float(minutes_b)))
    se = sqrt(1.0 / float(events_a) + 1.0 / float(events_b))
    z = log_ratio / se if se > 0 else 0.0
    return {
        "estimate": exp(log_ratio),
        "ci_low": exp(log_ratio - 1.96 * se),
        "ci_high": exp(log_ratio + 1.96 * se),
        "p_value": float(erfc(abs(z) / sqrt(2.0))),
        "fit_status": "ok",
    }


def refined_recovery_interval_from_days(days_since_last_match: object) -> str:
    """Split long recovery and no-prior-match states for recovery analyses."""
    if pd.isna(days_since_last_match):
        return "no prior match"
    days = float(days_since_last_match)
    if days <= 3.0:
        return "0-3 days"
    if days <= 5.0:
        return "4-5 days"
    if days <= 7.0:
        return "6-7 days"
    if days <= 14.0:
        return "8-14 days"
    return ">14 days"


def add_refined_recovery_interval(match_panel: pd.DataFrame) -> pd.DataFrame:
    """Add recovery bins that do not merge long gaps with no observed prior match."""
    out = match_panel.copy()
    if "days_since_last_match" in out.columns:
        out["recovery_interval_refined"] = out["days_since_last_match"].apply(
            refined_recovery_interval_from_days
        )
    elif "recovery_interval_bin" in out.columns:
        out["recovery_interval_refined"] = out["recovery_interval_bin"].astype(str).replace(
            {">14 days/no prior match": ">14 days"}
        )
    elif "recovery_interval_refined" in out.columns:
        out["recovery_interval_refined"] = out["recovery_interval_refined"].astype(str)
    else:
        raise KeyError("match_panel missing days_since_last_match or recovery_interval_bin")
    return out


def spline_curve_shape_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise full fitted-curve minima, maxima, and selected anchors."""
    required = {"fragility_group", "all_minutes_last_7d", "pred_events_per_10000_min"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing required columns: {sorted(missing)}")

    rows: List[Dict[str, object]] = []
    for group in MODEL_GROUPS:
        group_frame = predictions[predictions["fragility_group"] == group].copy()
        if group_frame.empty:
            continue
        group_frame["events_per_1000_match_hours"] = (
            group_frame["pred_events_per_10000_min"].astype(float) * 6.0
        )
        min_row = group_frame.loc[group_frame["events_per_1000_match_hours"].idxmin()]
        max_row = group_frame.loc[group_frame["events_per_1000_match_hours"].idxmax()]
        row: Dict[str, object] = {
            "history_stratum": group,
            "min_minutes_last_7d": float(min_row["all_minutes_last_7d"]),
            "min_events_per_1000_match_hours": float(min_row["events_per_1000_match_hours"]),
            "max_minutes_last_7d": float(max_row["all_minutes_last_7d"]),
            "max_events_per_1000_match_hours": float(max_row["events_per_1000_match_hours"]),
            "max_vs_min_ratio": (
                float(max_row["events_per_1000_match_hours"])
                / float(min_row["events_per_1000_match_hours"])
                if float(min_row["events_per_1000_match_hours"]) > 0
                else np.nan
            ),
        }
        for burden in SELECTED_BURDENS:
            nearest_idx = (
                group_frame["all_minutes_last_7d"].astype(float).sub(float(burden)).abs().idxmin()
            )
            nearest = group_frame.loc[nearest_idx]
            row[f"anchor_{int(burden)}_minutes_last_7d"] = float(
                nearest["all_minutes_last_7d"]
            )
            row[f"anchor_{int(burden)}_events_per_1000_match_hours"] = float(
                nearest["events_per_1000_match_hours"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_prior_spell_end_dates(injuries: pd.DataFrame) -> pd.DataFrame:
    """Prepare prior injury end dates for recent-return audits."""
    required = {PLAYER_ID_COL, "start_date"}
    missing = required - set(injuries.columns)
    if missing:
        raise KeyError(f"injuries missing required columns: {sorted(missing)}")
    out = injuries.copy()
    out[PLAYER_ID_COL] = pd.to_numeric(out[PLAYER_ID_COL], errors="coerce")
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    if "end_date" in out.columns:
        out["prior_injury_end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    else:
        out["prior_injury_end_date"] = pd.NaT
    out["prior_injury_end_date"] = out["prior_injury_end_date"].fillna(out["start_date"])
    out = out.dropna(subset=[PLAYER_ID_COL, "prior_injury_end_date"])
    out[PLAYER_ID_COL] = out[PLAYER_ID_COL].astype(int)
    return out[[PLAYER_ID_COL, "prior_injury_end_date"]].sort_values(
        [PLAYER_ID_COL, "prior_injury_end_date"]
    )


def add_recent_prior_injury_return_flags(
    match_panel: pd.DataFrame,
    injuries: pd.DataFrame,
    window_days: int = 14,
) -> pd.DataFrame:
    """Flag rows occurring soon after a recorded prior injury spell ended."""
    missing = {PLAYER_ID_COL, "date"} - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")
    spells = prepare_prior_spell_end_dates(injuries)
    rows = []
    for player_id, player_matches in match_panel.groupby(PLAYER_ID_COL, sort=False):
        left = player_matches.sort_values("date").copy()
        left["date"] = pd.to_datetime(left["date"], errors="coerce")
        left = left.dropna(subset=["date"])
        right = spells[spells[PLAYER_ID_COL] == player_id]
        if right.empty:
            left["days_since_prior_injury_end"] = np.nan
            left["returned_from_recorded_injury_within_14d"] = False
            rows.append(left)
            continue
        matched = pd.merge_asof(
            left,
            right,
            left_on="date",
            right_on="prior_injury_end_date",
            direction="backward",
            allow_exact_matches=True,
            suffixes=("", "_injury"),
        )
        matched = matched.drop(columns=[f"{PLAYER_ID_COL}_injury"], errors="ignore")
        matched["days_since_prior_injury_end"] = (
            matched["date"] - matched["prior_injury_end_date"]
        ).dt.days
        matched["returned_from_recorded_injury_within_14d"] = matched[
            "days_since_prior_injury_end"
        ].between(0, int(window_days), inclusive="both")
        rows.append(matched)
    if not rows:
        out = match_panel.copy()
        out["days_since_prior_injury_end"] = np.nan
        out["returned_from_recorded_injury_within_14d"] = False
        return out
    return pd.concat(rows, ignore_index=True)


def lineup_start_status_table(lineups: pd.DataFrame) -> pd.DataFrame:
    """Return one player-date row with starting/substitute status from lineups."""
    required = {"date", "player_id", "type"}
    missing = required - set(lineups.columns)
    if missing:
        raise KeyError(f"lineups missing required columns: {sorted(missing)}")

    out = lineups[list(required)].copy()
    out = out.rename(columns={"player_id": PLAYER_ID_COL})
    out[PLAYER_ID_COL] = pd.to_numeric(out[PLAYER_ID_COL], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["lineup_type"] = out["type"].fillna("").astype(str)
    out = out.dropna(subset=[PLAYER_ID_COL, "date"]).copy()
    out[PLAYER_ID_COL] = out[PLAYER_ID_COL].astype(int)
    out["is_starting_lineup"] = out["lineup_type"].isin(LINEUP_START_TYPES).astype(int)
    out["is_substitute_list"] = out["lineup_type"].isin(LINEUP_SUBSTITUTE_TYPES).astype(int)
    grouped = (
        out.groupby([PLAYER_ID_COL, "date"], as_index=False)
        .agg(
            lineup_rows=("lineup_type", "size"),
            is_starting_lineup=("is_starting_lineup", "max"),
            is_substitute_list=("is_substitute_list", "max"),
        )
    )
    grouped["lineup_role"] = np.select(
        [
            grouped["is_starting_lineup"].eq(1) & grouped["is_substitute_list"].eq(1),
            grouped["is_starting_lineup"].eq(1),
            grouped["is_substitute_list"].eq(1),
        ],
        ["both_start_and_substitute_same_day", "starting_lineup", "substitute_list"],
        default="listed_other",
    )
    return grouped


def add_lineup_start_status(
    match_panel: pd.DataFrame,
    lineups: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Attach Transfermarkt starting-lineup/substitute-list status when available."""
    required = {PLAYER_ID_COL, "date"}
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    out = match_panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if lineups is None or lineups.empty:
        out["lineup_rows"] = 0
        out["is_starting_lineup"] = np.nan
        out["is_substitute_list"] = np.nan
        out["lineup_role"] = "lineup_unavailable"
        return out

    status = lineup_start_status_table(lineups)
    out = out.merge(status, on=[PLAYER_ID_COL, "date"], how="left")
    out["lineup_rows"] = out["lineup_rows"].fillna(0).astype(int)
    for col in ["is_starting_lineup", "is_substitute_list"]:
        out[col] = out[col].where(out["lineup_rows"].gt(0), np.nan)
    out["lineup_role"] = out["lineup_role"].fillna("not_listed_or_missing_lineup")
    return out


def selection_band_audit(
    match_panel: pd.DataFrame,
    injuries: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
    lineups: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Summarise observed support and selection proxies in key fitted-curve bands."""
    required = {
        "model_group",
        "all_minutes_last_7d",
        MATCH_MINUTES_COL,
        PLAYER_ID_COL,
        "date",
        event_col,
    }
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    frame = add_recent_prior_injury_return_flags(match_panel, injuries)
    frame = add_lineup_start_status(frame, lineups)

    def percent_fields(
        name: str,
        selected: pd.Series,
        eligible: pd.Series,
    ) -> Dict[str, object]:
        """Return counts and a Wilson interval for one selection percentage."""
        eligible = eligible.fillna(False).astype(bool)
        selected = selected.fillna(False).astype(bool) & eligible
        numerator = int(selected.sum())
        denominator = int(eligible.sum())
        estimate, low_ci, high_ci = percent_with_interval(numerator, denominator)
        return {
            f"{name}_rows": numerator,
            f"{name}_denominator_rows": denominator,
            f"pct_{name}": estimate,
            f"pct_{name}_ci_low": low_ci,
            f"pct_{name}_ci_high": high_ci,
            f"pct_{name}_interval_method": "wilson_95",
        }

    rows: List[Dict[str, object]] = []
    for history in list(MODEL_GROUPS) + ["all_modelled"]:
        history_frame = frame if history == "all_modelled" else frame[frame["model_group"] == history]
        for label, low, high in SELECTION_AUDIT_BANDS:
            burden = history_frame["all_minutes_last_7d"].astype(float)
            if low == high:
                subset = history_frame[burden.eq(low)]
            else:
                subset = history_frame[burden.between(low, high, inclusive="both")]
            minutes = subset[MATCH_MINUTES_COL].astype(float)
            events = int(subset[event_col].sum()) if len(subset) else 0
            rates = count_rate_intervals(events, float(minutes.sum()) if len(subset) else 0.0)
            short_lt45 = minutes < 45.0
            recent_return = (
                subset["returned_from_recorded_injury_within_14d"].astype(bool)
                if len(subset)
                else pd.Series(dtype=bool)
            )
            starting = pd.to_numeric(
                subset.get("is_starting_lineup", pd.Series(np.nan, index=subset.index)),
                errors="coerce",
            )
            substitute = pd.to_numeric(
                subset.get("is_substitute_list", pd.Series(np.nan, index=subset.index)),
                errors="coerce",
            )
            all_rows = pd.Series(True, index=subset.index, dtype=bool)
            lineup_rows = subset["lineup_rows"].gt(0)
            rows.append(
                {
                    "history_stratum": history,
                    "band": label,
                    "band_low_minutes": low,
                    "band_high_minutes": high,
                    "match_rows": int(len(subset)),
                    "players": int(subset[PLAYER_ID_COL].nunique()) if len(subset) else 0,
                    "events": events,
                    "match_minutes": float(minutes.sum()) if len(subset) else 0.0,
                    "mean_current_match_minutes": float(minutes.mean()) if len(subset) else np.nan,
                    "median_current_match_minutes": float(minutes.median()) if len(subset) else np.nan,
                    **percent_fields("current_appearance_lt45", short_lt45, all_rows),
                    **percent_fields("current_appearance_lt60", minutes < 60.0, all_rows),
                    **percent_fields(
                        "returned_from_recorded_injury_within_14d",
                        recent_return,
                        all_rows,
                    ),
                    **percent_fields(
                        "short_lt45_and_recent_return",
                        short_lt45 & recent_return,
                        all_rows,
                    ),
                    **percent_fields(
                        "short_lt45_without_recent_return",
                        short_lt45 & ~recent_return,
                        all_rows,
                    ),
                    **percent_fields(
                        "recent_return_without_short_lt45",
                        ~short_lt45 & recent_return,
                        all_rows,
                    ),
                    **percent_fields("with_lineup_record", lineup_rows, all_rows),
                    **percent_fields(
                        "starting_lineup",
                        starting.eq(1),
                        starting.notna(),
                    ),
                    **percent_fields(
                        "substitute_list",
                        substitute.eq(1),
                        substitute.notna(),
                    ),
                    **rates,
                }
            )
    return pd.DataFrame(rows)


def selection_band_joint_proxy_audit(
    match_panel: pd.DataFrame,
    injuries: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
    lineups: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Cross-tabulate short-appearance and recent-return selection proxies."""
    required = {
        "model_group",
        "all_minutes_last_7d",
        MATCH_MINUTES_COL,
        PLAYER_ID_COL,
        "date",
        event_col,
    }
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    frame = add_recent_prior_injury_return_flags(match_panel, injuries)
    frame = add_lineup_start_status(frame, lineups)
    frame["short_current_appearance_lt45"] = (
        frame[MATCH_MINUTES_COL].astype(float) < 45.0
    )
    frame["recent_recorded_return_14d"] = frame[
        "returned_from_recorded_injury_within_14d"
    ].astype(bool)
    rows: List[Dict[str, object]] = []
    for history in list(MODEL_GROUPS) + ["all_modelled"]:
        history_frame = frame if history == "all_modelled" else frame[frame["model_group"] == history]
        for label, low, high in SELECTION_AUDIT_BANDS:
            burden = history_frame["all_minutes_last_7d"].astype(float)
            if low == high:
                band_frame = history_frame[burden.eq(low)]
            else:
                band_frame = history_frame[burden.between(low, high, inclusive="both")]
            band_n = int(len(band_frame))
            for short in [False, True]:
                for recent in [False, True]:
                    subset = band_frame[
                        band_frame["short_current_appearance_lt45"].eq(short)
                        & band_frame["recent_recorded_return_14d"].eq(recent)
                    ]
                    rows.append(
                        {
                            "history_stratum": history,
                            "band": label,
                            "short_current_appearance_lt45": bool(short),
                            "recent_recorded_return_14d": bool(recent),
                            "lineup_roles": ";".join(
                                sorted(subset["lineup_role"].astype(str).unique())
                            )
                            if len(subset)
                            else "",
                            "match_rows": int(len(subset)),
                            "pct_of_band_rows": (
                                float(len(subset) / band_n * 100.0) if band_n else np.nan
                            ),
                            "players": int(subset[PLAYER_ID_COL].nunique()) if len(subset) else 0,
                            "events": int(subset[event_col].sum()) if len(subset) else 0,
                            "match_minutes": float(
                                subset[MATCH_MINUTES_COL].astype(float).sum()
                            )
                            if len(subset)
                            else 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def observed_event_support_summary(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:
    """Report the observed exposure support among rows with proxy events."""
    required = {"model_group", "all_minutes_last_7d", MATCH_MINUTES_COL, event_col}
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    frame = match_panel[match_panel["model_group"].isin(MODEL_GROUPS)].copy()
    rows: List[Dict[str, object]] = []
    for history in list(MODEL_GROUPS) + ["all_modelled"]:
        history_frame = frame if history == "all_modelled" else frame[frame["model_group"] == history]
        event_rows = history_frame[history_frame[event_col].astype(int) > 0].copy()
        rows.append(
            {
                "history_stratum": history,
                "match_rows": int(len(history_frame)),
                "events": int(event_rows[event_col].sum()) if len(event_rows) else 0,
                "max_prior7_minutes_with_event": float(event_rows["all_minutes_last_7d"].max())
                if len(event_rows)
                else np.nan,
                "p95_prior7_minutes_among_events": float(event_rows["all_minutes_last_7d"].quantile(0.95))
                if len(event_rows)
                else np.nan,
                "max_current_match_minutes_with_event": float(event_rows[MATCH_MINUTES_COL].max())
                if len(event_rows)
                else np.nan,
                "events_gt150_prior7_minutes": int(
                    event_rows["all_minutes_last_7d"].astype(float).gt(150.0).sum()
                )
                if len(event_rows)
                else 0,
                "match_rows_gt150_prior7_minutes": int(
                    history_frame["all_minutes_last_7d"].astype(float).gt(150.0).sum()
                ),
                "match_minutes_gt150_prior7_minutes": float(
                    history_frame.loc[
                        history_frame["all_minutes_last_7d"].astype(float).gt(150.0),
                        MATCH_MINUTES_COL,
                    ]
                    .astype(float)
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def reporting_process_severity_audit(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
    group_col: str = PRIMARY_GROUP_COL,
) -> pd.DataFrame:
    """
    Compare public-report duration strata using a common match-minute denominator.

    This audit checks whether the higher-history contrast is restricted to
    short, easily reported absences or also appears in longer reported absences.
    It is still a public-data audit, not clinical time-loss surveillance.
    """
    required = {
        group_col,
        MATCH_MINUTES_COL,
        event_col,
        "matchproxy_duration_bucket",
        "matchproxy_duration_days",
    }
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    frame = match_panel[match_panel[group_col].isin(["tough", "regular", "fragile"])].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["reporting_history_group"] = np.where(
        frame[group_col].astype(str).eq("fragile"),
        "higher_history",
        "lower_intermediate_history",
    )
    frame[MATCH_MINUTES_COL] = pd.to_numeric(
        frame[MATCH_MINUTES_COL], errors="coerce"
    ).fillna(0.0)
    frame[event_col] = frame[event_col].fillna(0).astype(int)

    denominators = (
        frame.groupby("reporting_history_group", dropna=False)[MATCH_MINUTES_COL]
        .sum()
        .to_dict()
    )
    row_counts = frame.groupby("reporting_history_group", dropna=False).size().to_dict()
    player_counts = (
        frame.groupby("reporting_history_group", dropna=False)[PLAYER_ID_COL]
        .nunique()
        .to_dict()
        if PLAYER_ID_COL in frame.columns
        else {}
    )

    comparisons = [
        (
            duration_bucket,
            frame[event_col].eq(1)
            & frame["matchproxy_duration_bucket"].astype(str).eq(duration_bucket),
        )
        for duration_bucket in MATCHPROXY_DURATION_BUCKETS
    ]
    comparisons.append(
        (
            "reported absence >=28 days",
            frame[event_col].eq(1)
            & pd.to_numeric(frame["matchproxy_duration_days"], errors="coerce").ge(
                SEVERE_REPORTED_ABSENCE_DAYS
            ),
        )
    )

    rows: List[Dict[str, object]] = []
    for comparison, event_mask in comparisons:
        higher_events = int(
            (
                event_mask
                & frame["reporting_history_group"].eq("higher_history")
            ).sum()
        )
        lower_events = int(
            (
                event_mask
                & frame["reporting_history_group"].eq("lower_intermediate_history")
            ).sum()
        )
        higher_minutes = float(denominators.get("higher_history", 0.0))
        lower_minutes = float(denominators.get("lower_intermediate_history", 0.0))
        higher_rates = count_rate_intervals(higher_events, higher_minutes)
        lower_rates = count_rate_intervals(lower_events, lower_minutes)
        ratio = count_rate_ratio_interval(
            higher_events,
            higher_minutes,
            lower_events,
            lower_minutes,
        )
        rows.append(
            {
                "duration_or_severity_proxy": comparison,
                "higher_match_rows": int(row_counts.get("higher_history", 0)),
                "higher_players": int(player_counts.get("higher_history", 0)),
                "higher_events": higher_events,
                "higher_match_minutes": higher_minutes,
                "higher_events_per_1000_match_hours": higher_rates[
                    "events_per_1000_match_hours"
                ],
                "higher_events_per_1000_match_hours_ci_low": higher_rates[
                    "events_per_1000_match_hours_ci_low"
                ],
                "higher_events_per_1000_match_hours_ci_high": higher_rates[
                    "events_per_1000_match_hours_ci_high"
                ],
                "lower_intermediate_match_rows": int(
                    row_counts.get("lower_intermediate_history", 0)
                ),
                "lower_intermediate_players": int(
                    player_counts.get("lower_intermediate_history", 0)
                ),
                "lower_intermediate_events": lower_events,
                "lower_intermediate_match_minutes": lower_minutes,
                "lower_intermediate_events_per_1000_match_hours": lower_rates[
                    "events_per_1000_match_hours"
                ],
                "lower_intermediate_events_per_1000_match_hours_ci_low": lower_rates[
                    "events_per_1000_match_hours_ci_low"
                ],
                "lower_intermediate_events_per_1000_match_hours_ci_high": lower_rates[
                    "events_per_1000_match_hours_ci_high"
                ],
                "higher_vs_lower_intermediate_rate_ratio": ratio["estimate"],
                "higher_vs_lower_intermediate_rate_ratio_ci_low": ratio["ci_low"],
                "higher_vs_lower_intermediate_rate_ratio_ci_high": ratio["ci_high"],
                "higher_vs_lower_intermediate_rate_ratio_p": ratio["p_value"],
                "fit_status": ratio["fit_status"],
            }
        )
    return pd.DataFrame(rows)


def global_spline_interaction_test(res) -> Dict[str, float]:
    """Joint Wald test of every spline-by-history interaction coefficient."""
    names = [str(name) for name in res.params.index]
    interaction_indices = [
        idx
        for idx, name in enumerate(names)
        if name.startswith(("bs(", "cr(")) and ":model_group" in name
    ]
    if not interaction_indices:
        raise ValueError("Model has no spline-by-history interaction terms")

    restriction = np.zeros((len(interaction_indices), len(names)), dtype=float)
    for row, column in enumerate(interaction_indices):
        restriction[row, column] = 1.0
    test = res.wald_test(restriction, scalar=True)
    return {
        "test_statistic": float(np.asarray(test.statistic).squeeze()),
        "df": float(len(interaction_indices)),
        "p_value": float(np.asarray(test.pvalue).squeeze()),
    }


def effect_modification_rows(
    res,
    extra_covars: Optional[Dict[str, object]] = None,
    contrast_windows: Iterable[tuple[float, float]] = CONTRAST_WINDOWS,
    effect_measure: str = "incidence_rate_ratio",
) -> pd.DataFrame:
    """Return prespecified history contrasts and effect-modification tests."""
    design_info = res.model.data.design_info
    params = np.asarray(res.params)
    covariance = np.asarray(res.cov_params())
    windows = [(float(start), float(end)) for start, end in contrast_windows]
    burdens = sorted(
        {burden for window in windows for burden in window}
        | {float(burden) for burden in BETWEEN_HISTORY_CONTRAST_BURDENS}
    )
    designs: Dict[tuple[str, float], np.ndarray] = {}
    for group in MODEL_GROUPS:
        for burden in burdens:
            template = prediction_template([burden], group, extra_covars)
            designs[(group, burden)] = np.asarray(
                build_design_matrices([design_info], template)[0]
            )[0]

    interaction = global_spline_interaction_test(res)
    rows: List[Dict[str, object]] = [
        {
            "contrast_id": "global_spline_by_history_interaction",
            "history_stratum": "joint",
            "burden_from": np.nan,
            "burden_to": np.nan,
            "effect_measure": "chi_square",
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "log_estimate": np.nan,
            "standard_error": np.nan,
            **interaction,
        }
    ]

    for burden in BETWEEN_HISTORY_CONTRAST_BURDENS:
        interval = delta_ratio_interval(
            params,
            covariance,
            designs[("fragile", float(burden))],
            designs[("regular", float(burden))],
        )
        rows.append(
            {
                "contrast_id": f"higher_vs_intermediate_at_{float(burden):g}",
                "history_stratum": "higher_vs_intermediate",
                "burden_from": np.nan,
                "burden_to": float(burden),
                "effect_measure": effect_measure,
                "estimate": interval["rate_ratio"],
                "ci_low": interval["rr_ci_low"],
                "ci_high": interval["rr_ci_high"],
                "log_estimate": interval["log_rate_ratio"],
                "standard_error": interval["log_rate_ratio_se"],
                "test_statistic": interval["z_statistic"],
                "df": 1.0,
                "p_value": interval["p_value"],
            }
        )

    for burden_from, burden_to in windows:
        suffix = f"{burden_to:g}_vs_{burden_from:g}"
        contrasts = [
            (
                f"intermediate_history_{suffix}",
                "intermediate_history",
                designs[("regular", burden_to)],
                designs[("regular", burden_from)],
            ),
            (
                f"higher_history_{suffix}",
                "higher_history",
                designs[("fragile", burden_to)],
                designs[("fragile", burden_from)],
            ),
            (
                f"ratio_of_{suffix}_changes",
                "higher_vs_intermediate",
                designs[("fragile", burden_to)] - designs[("fragile", burden_from)],
                designs[("regular", burden_to)] - designs[("regular", burden_from)],
            ),
        ]
        for contrast_id, stratum, numerator, denominator in contrasts:
            interval = delta_ratio_interval(params, covariance, numerator, denominator)
            rows.append(
                {
                    "contrast_id": contrast_id,
                    "history_stratum": stratum,
                    "burden_from": burden_from,
                    "burden_to": burden_to,
                    "effect_measure": effect_measure,
                    "estimate": interval["rate_ratio"],
                    "ci_low": interval["rr_ci_low"],
                    "ci_high": interval["rr_ci_high"],
                    "log_estimate": interval["log_rate_ratio"],
                    "standard_error": interval["log_rate_ratio_se"],
                    "test_statistic": interval["z_statistic"],
                    "df": 1.0,
                    "p_value": interval["p_value"],
                }
            )
    return pd.DataFrame(rows)


def label_effect_modification_rows(
    effects: pd.DataFrame,
    label: str,
    event_col: str,
    group_col: str,
    controls_label: str,
    match_panel: pd.DataFrame,
    model_family: str,
    analysis_role: str,
    estimator: str = "clustered_glm",
) -> pd.DataFrame:
    """Attach specification metadata to a formal-test result table."""
    out = effects.copy()
    metadata = {
        "model": label,
        "analysis_role": analysis_role,
        "model_family": model_family,
        "estimator": estimator,
        "event_col": event_col,
        "group_col": group_col,
        "controls": controls_label,
        "n_match_rows": int(len(match_panel)),
        "n_players": int(match_panel[PLAYER_ID_COL].nunique()),
        "n_events": int(match_panel[event_col].sum()),
    }
    for column, value in reversed(list(metadata.items())):
        out.insert(0, column, value)
    return out


def add_specification_multiplicity_adjustments(effects: pd.DataFrame) -> pd.DataFrame:
    """Adjust each formal contrast across the complete specification family."""
    out = effects.copy()
    out["p_holm_across_specifications"] = np.nan
    out["p_bh_across_specifications"] = np.nan
    valid = out["p_value"].notna()
    for _, indices in out.loc[valid].groupby("contrast_id", sort=False).groups.items():
        p_values = out.loc[indices, "p_value"].astype(float).to_numpy()
        out.loc[indices, "p_holm_across_specifications"] = multipletests(
            p_values, method="holm"
        )[1]
        out.loc[indices, "p_bh_across_specifications"] = multipletests(
            p_values, method="fdr_bh"
        )[1]
    out["reject_holm_0_05"] = out["p_holm_across_specifications"] < 0.05
    out["reject_bh_0_05"] = out["p_bh_across_specifications"] < 0.05
    return out


def prediction_template(
    burdens: Iterable[float],
    group: str,
    extra_covars: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Build a neutral-covariate prediction template for one group."""
    burdens = list(burdens)
    tmp = pd.DataFrame(
        {
            "all_minutes_last_7d": burdens,
            "model_group": group,
            "excess_minutes_last7d": 0.0,
            "any_extra_time_last7d": 0,
            "week_phase_sin": 0.0,
            "week_phase_cos": 0.0,
            "halfweek_phase_sin": 0.0,
            "halfweek_phase_cos": 0.0,
            "age_years": 28.0,
            "position_group": "Attack",
            "club_season": "2019_281",
            "log_prior_minutes_played": np.log1p(900.0),
            "prior_n_spells": 1.0,
            "log_prior_injuries_per_10000min": 0.0,
            "log_prior_max_spell_duration_days": 0.0,
        }
    )
    if extra_covars:
        for key, value in extra_covars.items():
            tmp[key] = value
    tmp["log_minutes_played"] = np.log(90.0)
    return tmp


def prediction_offset(tmp: pd.DataFrame, denominator: str) -> Optional[pd.Series]:
    """Return a prediction offset for the requested denominator convention."""
    if denominator == "observed_minutes":
        return tmp["log_minutes_played"]
    if denominator == "fixed_90":
        return pd.Series(np.log(90.0), index=tmp.index)
    if denominator == "per_match":
        return None
    raise ValueError(f"Unknown denominator mode: {denominator}")


def add_prediction_intervals(
    res,
    tmp: pd.DataFrame,
    denominator: str = "observed_minutes",
) -> pd.DataFrame:  # pragma: no cover
    """Attach GLM mean prediction intervals as per-match/per-minute rates."""
    offset = prediction_offset(tmp, denominator)
    if offset is None:
        pred = res.get_prediction(tmp)
    else:
        pred = res.get_prediction(tmp, offset=offset)
    frame = pred.summary_frame(alpha=0.05)
    out = tmp.copy()
    out["pred_events_per_match"] = frame["mean"].to_numpy()
    out["pred_events_per_match_ci_low"] = frame["mean_ci_lower"].to_numpy()
    out["pred_events_per_match_ci_high"] = frame["mean_ci_upper"].to_numpy()
    for suffix, source in [
        ("", "pred_events_per_match"),
        ("_ci_low", "pred_events_per_match_ci_low"),
        ("_ci_high", "pred_events_per_match_ci_high"),
    ]:
        out[f"pred_events_per_minute{suffix}"] = out[source] / 90.0
        out[f"pred_events_per_10000_min{suffix}"] = out[f"pred_events_per_minute{suffix}"] * 10000.0
    out = out.rename(columns={"model_group": "fragility_group"})
    return out


def selected_prediction_rows(preds: pd.DataFrame, burdens: Iterable[float]) -> pd.DataFrame:
    """Return selected burden rows from a prediction grid."""
    burdens = [float(x) for x in burdens]
    return preds[preds["all_minutes_last_7d"].astype(float).isin(burdens)].copy()


def selected_support_rows(
    match_panel: pd.DataFrame,
    event_col: str,
    burdens: Iterable[float],
    window_half_width: float = SUPPORT_WINDOW_HALF_WIDTH,
) -> pd.DataFrame:
    """Summarise observed local support around selected burden values."""
    rows: List[Dict[str, float]] = []
    for group in ["regular", "fragile"]:
        group_frame = match_panel[match_panel["model_group"] == group]
        for burden in [float(x) for x in burdens]:
            lower = max(0.0, burden - window_half_width)
            upper = burden + window_half_width
            in_window = group_frame[
                group_frame["all_minutes_last_7d"].astype(float).between(
                    lower, upper, inclusive="both"
                )
            ]
            minutes = float(in_window[MATCH_MINUTES_COL].sum())
            events = int(in_window[event_col].sum())
            rows.append(
                {
                    "fragility_group": group,
                    "all_minutes_last_7d": burden,
                    "support_window_low": lower,
                    "support_window_high": upper,
                    "support_match_rows": int(len(in_window)),
                    "support_events": events,
                    "support_match_minutes": minutes,
                    "support_events_per_10000_min": (
                        events / minutes * 10000.0 if minutes > 0.0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def spline_basis_expression(
    burden_max: float,
    spline_df: Optional[int] = SPLINE_DF,
    spline_basis: str = "bs",
    spline_knots: Optional[Sequence[float]] = None,
) -> str:
    """Return a patsy spline term for burden-shape sensitivity checks."""
    burden_min = 0.0
    basis = str(spline_basis)
    if basis not in {"bs", "cr"}:
        raise ValueError(f"Unknown spline basis: {spline_basis}")
    if spline_knots is not None:
        knots = tuple(float(knot) for knot in spline_knots)
        return (
            f"{basis}(all_minutes_last_7d, knots={knots}, "
            f"lower_bound={burden_min}, upper_bound={float(burden_max)})"
        )
    if spline_df is None:
        raise ValueError("spline_df is required when spline_knots is not supplied")
    return (
        f"{basis}(all_minutes_last_7d, df={int(spline_df)}, "
        f"lower_bound={burden_min}, upper_bound={float(burden_max)})"
    )


def ratio_rows(
    res,
    selected: pd.DataFrame,
    extra_covars: Optional[Dict[str, object]] = None,
    effect_measure: str = "incidence_rate_ratio",
) -> pd.DataFrame:  # pragma: no cover
    """Compute higher/intermediate contrast rows for selected burdens."""
    design_info = res.model.data.design_info
    params = np.asarray(res.params)
    covariance = np.asarray(res.cov_params())
    rows: List[Dict[str, float]] = []
    for burden in sorted(selected["all_minutes_last_7d"].astype(float).unique()):
        fragile = prediction_template([burden], "fragile", extra_covars)
        regular = prediction_template([burden], "regular", extra_covars)
        x_fragile = np.asarray(build_design_matrices([design_info], fragile)[0])[0]
        x_regular = np.asarray(build_design_matrices([design_info], regular)[0])[0]
        interval = delta_ratio_interval(params, covariance, x_fragile, x_regular)
        interval["all_minutes_last_7d"] = float(burden)
        interval["effect_measure"] = effect_measure
        rows.append(interval)
    return pd.DataFrame(rows)


def spline_formula(
    event_col: str,
    burden_max: float,
    controls: str = "",
    include_exposure_derived_terms: bool = False,
    spline_df: Optional[int] = SPLINE_DF,
    spline_basis: str = "bs",
    spline_knots: Optional[Sequence[float]] = None,
) -> str:
    """Build the spline-by-history model formula for one specification."""
    spline_term = spline_basis_expression(
        burden_max,
        spline_df=spline_df,
        spline_basis=spline_basis,
        spline_knots=spline_knots,
    )
    formula = (
        f"{event_col} ~ "
        f"{spline_term} * model_group "
        "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
        f"{controls}"
    )
    if include_exposure_derived_terms:
        formula += " + excess_minutes_last7d + any_extra_time_last7d"
    return formula


def model_family_object(model_family: str):
    """Return a statsmodels GLM family object for a named sensitivity model."""
    if model_family == "poisson":
        return sm.families.Poisson()
    if model_family == "binomial_logit":
        return sm.families.Binomial()
    if model_family == "binomial_cloglog":
        return sm.families.Binomial(link=sm.families.links.CLogLog())
    raise ValueError(f"Unknown model family: {model_family}")


def effect_measure_for_family(model_family: str) -> str:
    """Name the exponentiated contrast scale for a fitted family/link."""
    if model_family == "poisson":
        return "incidence_rate_ratio"
    if model_family == "binomial_logit":
        return "odds_ratio"
    if model_family == "binomial_cloglog":
        return "complementary_loglog_ratio"
    raise ValueError(f"Unknown model family: {model_family}")


def fit_model(
    match_panel: pd.DataFrame,
    event_col: str,
    controls: str = "",
    model_family: str = "poisson",
    denominator: str = "observed_minutes",
    include_exposure_derived_terms: bool = False,
    spline_df: Optional[int] = SPLINE_DF,
    spline_basis: str = "bs",
    spline_knots: Optional[Sequence[float]] = None,
):  # pragma: no cover
    """Fit the clustered spline GLM used by primary and sensitivity runs."""
    burden_max = float(match_panel["all_minutes_last_7d"].max())
    formula = spline_formula(
        event_col,
        burden_max,
        controls=controls,
        include_exposure_derived_terms=include_exposure_derived_terms,
        spline_df=spline_df,
        spline_basis=spline_basis,
        spline_knots=spline_knots,
    )
    offset = None
    if denominator == "observed_minutes":
        offset = match_panel["log_minutes_played"]
    elif denominator == "fixed_90":
        offset = pd.Series(np.log(90.0), index=match_panel.index)
    elif denominator != "per_match":
        raise ValueError(f"Unknown denominator mode: {denominator}")

    kwargs = {}
    if offset is not None:
        kwargs["offset"] = offset
    model = smf.glm(
        formula=formula,
        data=match_panel,
        family=model_family_object(model_family),
        **kwargs,
    )
    return model.fit(cov_type="cluster", cov_kwds={"groups": match_panel[PLAYER_ID_COL]})


def run_prediction_bundle(
    match_panel: pd.DataFrame,
    event_col: str,
    controls: str = "",
    extra_covars: Optional[Dict[str, object]] = None,
    model_family: str = "poisson",
    denominator: str = "observed_minutes",
    include_exposure_derived_terms: bool = False,
    grid_max: float = PRIMARY_GRID_MAX,
    spline_df: Optional[int] = SPLINE_DF,
    spline_basis: str = "bs",
    spline_knots: Optional[Sequence[float]] = None,
):  # pragma: no cover
    """Fit a model and return predictions, selected rows, ratios, and metadata."""
    if match_panel[event_col].sum() <= 0:
        raise ValueError(f"No events available for {event_col}")
    if set(match_panel["model_group"].unique()) != set(MODEL_GROUPS):
        raise ValueError("Model requires exactly regular and fragile rows")

    res = fit_model(
        match_panel,
        event_col,
        controls=controls,
        model_family=model_family,
        denominator=denominator,
        include_exposure_derived_terms=include_exposure_derived_terms,
        spline_df=spline_df,
        spline_basis=spline_basis,
        spline_knots=spline_knots,
    )
    effect_measure = effect_measure_for_family(model_family)
    burden_max = float(match_panel["all_minutes_last_7d"].max())
    upper_grid = float(np.floor(min(burden_max, grid_max) / 5.0) * 5.0)
    grid = np.arange(0.0, upper_grid + 0.0001, 5.0)
    pred_frames = [
        add_prediction_intervals(
            res,
            prediction_template(grid, group, extra_covars),
            denominator=denominator,
        )
        for group in MODEL_GROUPS
    ]
    preds = pd.concat(pred_frames, ignore_index=True)
    selected_burdens = [b for b in SELECTED_BURDENS if b <= upper_grid]
    selected = selected_prediction_rows(preds, selected_burdens)
    support = selected_support_rows(match_panel, event_col, selected_burdens)
    selected = selected.merge(
        support,
        on=["fragility_group", "all_minutes_last_7d"],
        how="left",
    )
    ratios = ratio_rows(res, selected, extra_covars, effect_measure=effect_measure)
    pearson_chi2 = np.sum(res.resid_pearson ** 2)
    dispersion = pearson_chi2 / res.df_resid
    return {
        "result": res,
        "predictions": preds,
        "selected": selected,
        "support": support,
        "ratios": ratios,
        "effect_modification": effect_modification_rows(
            res,
            extra_covars,
            effect_measure=effect_measure,
        ),
        "dispersion": float(dispersion),
        "model_family": model_family,
        "denominator": denominator,
        "include_exposure_derived_terms": bool(include_exposure_derived_terms),
        "spline_df": spline_df,
        "spline_basis": spline_basis,
        "spline_knots": tuple(spline_knots) if spline_knots is not None else None,
        "estimator": "clustered_glm",
    }


def add_manual_poisson_prediction_intervals(
    res,
    tmp: pd.DataFrame,
    denominator: str = "observed_minutes",
) -> pd.DataFrame:  # pragma: no cover
    """Attach delta-method Poisson prediction intervals for GEE-style results."""
    design_info = res.model.data.design_info
    design = np.asarray(build_design_matrices([design_info], tmp)[0])
    params = np.asarray(res.params)
    covariance = np.asarray(res.cov_params())
    offset = prediction_offset(tmp, denominator)
    offset_values = 0.0 if offset is None else offset.astype(float).to_numpy()
    linear = design @ params + offset_values
    variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    se = np.sqrt(np.maximum(variance, 0.0))
    out = tmp.copy()
    for suffix, values in [
        ("", linear),
        ("_ci_low", linear - 1.96 * se),
        ("_ci_high", linear + 1.96 * se),
    ]:
        pred_per_match = np.exp(values)
        out[f"pred_events_per_match{suffix}"] = pred_per_match
        out[f"pred_events_per_minute{suffix}"] = pred_per_match / 90.0
        out[f"pred_events_per_10000_min{suffix}"] = (
            out[f"pred_events_per_minute{suffix}"] * 10000.0
        )
    out = out.rename(columns={"model_group": "fragility_group"})
    return out


def run_gee_prediction_bundle(
    match_panel: pd.DataFrame,
    event_col: str,
    controls: str = "",
    extra_covars: Optional[Dict[str, object]] = None,
    denominator: str = "observed_minutes",
    grid_max: float = PRIMARY_GRID_MAX,
) -> Dict[str, object]:  # pragma: no cover
    """Fit a player-clustered GEE Poisson sensitivity for recurrent events."""
    if match_panel[event_col].sum() <= 0:
        raise ValueError(f"No events available for {event_col}")
    if set(match_panel["model_group"].unique()) != set(MODEL_GROUPS):
        raise ValueError("Model requires exactly regular and fragile rows")
    if denominator != "observed_minutes":
        raise ValueError("GEE sensitivity currently supports observed_minutes only")

    burden_max = float(match_panel["all_minutes_last_7d"].max())
    formula = spline_formula(event_col, burden_max, controls=controls)
    res = smf.gee(
        formula=formula,
        groups=match_panel[PLAYER_ID_COL],
        data=match_panel,
        family=sm.families.Poisson(),
        offset=match_panel["log_minutes_played"],
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    upper_grid = float(np.floor(min(burden_max, grid_max) / 5.0) * 5.0)
    grid = np.arange(0.0, upper_grid + 0.0001, 5.0)
    pred_frames = [
        add_manual_poisson_prediction_intervals(
            res,
            prediction_template(grid, group, extra_covars),
            denominator=denominator,
        )
        for group in MODEL_GROUPS
    ]
    preds = pd.concat(pred_frames, ignore_index=True)
    selected_burdens = [b for b in SELECTED_BURDENS if b <= upper_grid]
    selected = selected_prediction_rows(preds, selected_burdens)
    support = selected_support_rows(match_panel, event_col, selected_burdens)
    selected = selected.merge(
        support,
        on=["fragility_group", "all_minutes_last_7d"],
        how="left",
    )
    ratios = ratio_rows(res, selected, extra_covars)
    pearson_chi2 = np.sum(np.asarray(res.resid_pearson) ** 2)
    dispersion = pearson_chi2 / res.df_resid if res.df_resid else np.nan
    return {
        "result": res,
        "predictions": preds,
        "selected": selected,
        "support": support,
        "ratios": ratios,
        "effect_modification": effect_modification_rows(res, extra_covars),
        "dispersion": float(dispersion),
        "model_family": "poisson",
        "denominator": denominator,
        "include_exposure_derived_terms": False,
        "spline_df": SPLINE_DF,
        "spline_basis": "bs",
        "spline_knots": None,
        "estimator": "gee_exchangeable_player",
    }


def spline_shape_sensitivity_table(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:  # pragma: no cover
    """Refit the primary curve under predefined spline-basis and knot choices."""
    rows: List[Dict[str, object]] = []
    for spec in SPLINE_SHAPE_SENSITIVITY_SPECS:
        row_common = {
            "specification": spec["specification"],
            "spline_basis": spec["spline_basis"],
            "spline_df": spec["spline_df"],
            "spline_knots": (
                ";".join(str(x) for x in spec["spline_knots"])
                if spec["spline_knots"] is not None
                else ""
            ),
            "event_col": event_col,
            "n_match_rows": int(len(match_panel)),
            "n_players": int(match_panel[PLAYER_ID_COL].nunique()),
            "n_events": int(match_panel[event_col].sum()),
        }
        try:
            bundle = run_prediction_bundle(
                match_panel,
                event_col,
                spline_df=spec["spline_df"],
                spline_basis=str(spec["spline_basis"]),
                spline_knots=spec["spline_knots"],
            )
            shape = spline_curve_shape_summary(bundle["predictions"])
            for _, shape_row in shape.iterrows():
                row = {**row_common, **shape_row.to_dict()}
                row["max_in_15_45_min_band"] = bool(
                    15.0 <= float(row["max_minutes_last_7d"]) <= 45.0
                )
                row["anchor_180_minus_anchor_0_per_1000_match_hours"] = (
                    float(row.get("anchor_180_events_per_1000_match_hours", np.nan))
                    - float(row.get("anchor_0_events_per_1000_match_hours", np.nan))
                )
                row["fit_status"] = "ok"
                row["fit_error"] = ""
                rows.append(row)
        except Exception as exc:
            for group in MODEL_GROUPS:
                rows.append(
                    {
                        **row_common,
                        "history_stratum": group,
                        "fit_status": "failed",
                        "fit_error": repr(exc),
                    }
                )
    return pd.DataFrame(rows)


def spline_shape_contrast_sensitivity_table(shape_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise 0- and 180-minute higher/intermediate contrasts across spline shapes."""
    required = {
        "specification",
        "history_stratum",
        "fit_status",
        "anchor_0_events_per_1000_match_hours",
        "anchor_180_events_per_1000_match_hours",
    }
    missing = required - set(shape_rows.columns)
    if missing:
        raise KeyError(f"shape_rows missing required columns: {sorted(missing)}")

    rows: List[Dict[str, object]] = []
    ok = shape_rows[shape_rows["fit_status"].astype(str).eq("ok")].copy()
    for specification, spec_rows in ok.groupby("specification", sort=False):
        by_group = spec_rows.set_index("history_stratum")
        if not set(MODEL_GROUPS).issubset(set(by_group.index)):
            rows.append(
                {
                    "specification": specification,
                    "fit_status": "not_estimable",
                    "fit_error": "missing one or both modelled history strata",
                }
            )
            continue
        regular = by_group.loc["regular"]
        fragile = by_group.loc["fragile"]
        for burden in [0, 180]:
            reg_rate = float(regular[f"anchor_{burden}_events_per_1000_match_hours"])
            frag_rate = float(fragile[f"anchor_{burden}_events_per_1000_match_hours"])
            rows.append(
                {
                    "specification": specification,
                    "burden_minutes": float(burden),
                    "intermediate_events_per_1000_match_hours": reg_rate,
                    "higher_events_per_1000_match_hours": frag_rate,
                    "higher_vs_intermediate_ratio": (
                        frag_rate / reg_rate if reg_rate > 0 else np.nan
                    ),
                    "fit_status": "ok",
                    "fit_error": "",
                }
            )
    return pd.DataFrame(rows)


def spline_anchor_range_summary(contrast_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise whether the history contrast grows from 0 to 180 minutes."""
    required = {
        "specification",
        "burden_minutes",
        "higher_vs_intermediate_ratio",
        "fit_status",
    }
    missing = required - set(contrast_rows.columns)
    if missing:
        raise KeyError(f"contrast_rows missing required columns: {sorted(missing)}")

    complete = contrast_rows[contrast_rows["fit_status"].astype(str).eq("ok")].copy()
    complete["burden_minutes"] = pd.to_numeric(
        complete["burden_minutes"], errors="coerce"
    )
    complete = complete[complete["burden_minutes"].isin([0.0, 180.0])]
    pivot = complete.pivot_table(
        index="specification",
        columns="burden_minutes",
        values="higher_vs_intermediate_ratio",
        aggfunc="first",
    ).reindex(columns=[0.0, 180.0]).dropna(subset=[0.0, 180.0])
    if pivot.empty:
        return pd.DataFrame(
            [
                {
                    "fit_status": "not_estimable",
                    "n_complete_specifications": 0,
                }
            ]
        )

    zero_ge_180 = pivot[0.0].ge(pivot[180.0])
    return pd.DataFrame(
        [
            {
                "fit_status": "ok",
                "n_complete_specifications": int(len(pivot)),
                "rr_0_min": float(pivot[0.0].min()),
                "rr_0_max": float(pivot[0.0].max()),
                "rr_180_min": float(pivot[180.0].min()),
                "rr_180_max": float(pivot[180.0].max()),
                "n_specifications_rr_0_ge_rr_180": int(zero_ge_180.sum()),
                "all_specifications_rr_0_ge_rr_180": bool(zero_ge_180.all()),
            }
        ]
    )


def player_fixed_effect_frame(
    match_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, object]:  # pragma: no cover
    """Return rows from players observed in both modelled history strata."""
    switch_counts = match_panel.groupby(PLAYER_ID_COL)["model_group"].nunique()
    switcher_ids = switch_counts[switch_counts >= 2].index
    frame = match_panel[match_panel[PLAYER_ID_COL].isin(switcher_ids)].copy()
    if frame.empty or frame["model_group"].nunique() < 2:
        raise ValueError("No within-player history-stratum switchers available")
    row_counts = frame[PLAYER_ID_COL].value_counts()
    reference_player = row_counts.index[0]
    try:
        reference_player = int(reference_player)
    except (TypeError, ValueError):
        reference_player = str(reference_player)
    return frame, reference_player


def summary_row(
    label: str,
    event_col: str,
    group_col: str,
    controls_label: str,
    match_panel: pd.DataFrame,
    bundle: Dict[str, object],
) -> Dict[str, object]:  # pragma: no cover
    """Condense a fitted bundle into a one-row sensitivity summary."""
    selected = bundle["selected"]
    ratios = bundle["ratios"]
    row: Dict[str, object] = {
        "model": label,
        "event_col": event_col,
        "group_col": group_col,
        "controls": controls_label,
        "model_family": str(bundle.get("model_family", "poisson")),
        "denominator": str(bundle.get("denominator", "observed_minutes")),
        "estimator": str(bundle.get("estimator", "clustered_glm")),
        "spline_basis": str(bundle.get("spline_basis", "bs")),
        "spline_df": bundle.get("spline_df", SPLINE_DF),
        "spline_knots": (
            ";".join(str(x) for x in bundle["spline_knots"])
            if bundle.get("spline_knots") is not None
            else ""
        ),
        "fit_status": str(bundle.get("fit_status", "ok")),
        "includes_exposure_derived_terms": bool(
            bundle.get("include_exposure_derived_terms", False)
        ),
        "n_match_rows": int(len(match_panel)),
        "n_players": int(match_panel[PLAYER_ID_COL].nunique()),
        "n_events": int(match_panel[event_col].sum()),
        "dispersion": bundle["dispersion"],
    }
    for group in MODEL_GROUPS:
        for burden in SELECTED_BURDENS:
            sub = selected[
                (selected["fragility_group"] == group)
                & (selected["all_minutes_last_7d"].astype(float) == burden)
            ]
            if not sub.empty:
                prefix = f"{group}_{int(burden)}"
                row[f"{prefix}_rate"] = float(sub["pred_events_per_10000_min"].iloc[0])
                row[f"{prefix}_ci_low"] = float(sub["pred_events_per_10000_min_ci_low"].iloc[0])
                row[f"{prefix}_ci_high"] = float(sub["pred_events_per_10000_min_ci_high"].iloc[0])
                if "support_events" in sub:
                    row[f"{prefix}_support_events"] = int(sub["support_events"].iloc[0])
                    row[f"{prefix}_support_rows"] = int(sub["support_match_rows"].iloc[0])
    for burden in SELECTED_BURDENS:
        sub = ratios[ratios["all_minutes_last_7d"].astype(float) == burden]
        if not sub.empty:
            prefix = f"rr_{int(burden)}"
            row[f"{prefix}"] = float(sub["rate_ratio"].iloc[0])
            row[f"{prefix}_ci_low"] = float(sub["rr_ci_low"].iloc[0])
            row[f"{prefix}_ci_high"] = float(sub["rr_ci_high"].iloc[0])

    effects = bundle["effect_modification"].set_index("contrast_id")
    interaction = effects.loc["global_spline_by_history_interaction"]
    row["global_interaction_chi2"] = float(interaction["test_statistic"])
    row["global_interaction_df"] = int(interaction["df"])
    row["global_interaction_p"] = float(interaction["p_value"])
    for burden_from, burden_to in CONTRAST_WINDOWS:
        suffix = f"{burden_to:g}_vs_{burden_from:g}"
        for contrast_id, prefix in [
            (f"intermediate_history_{suffix}", f"intermediate_{suffix}"),
            (f"higher_history_{suffix}", f"higher_{suffix}"),
            (f"ratio_of_{suffix}_changes", f"difference_in_{suffix}_change"),
        ]:
            effect = effects.loc[contrast_id]
            row[f"{prefix}_irr"] = float(effect["estimate"])
            row[f"{prefix}_ci_low"] = float(effect["ci_low"])
            row[f"{prefix}_ci_high"] = float(effect["ci_high"])
            row[f"{prefix}_p"] = float(effect["p_value"])
    return row


def publication_contrast_summary(
    sensitivity_summary: pd.DataFrame,
    denominator_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return compact 0/90/180 higher/intermediate contrasts for reporting."""
    frames = []
    if sensitivity_summary is not None and not sensitivity_summary.empty:
        sens = sensitivity_summary.copy()
        sens["analysis_source"] = "main_and_sensitivity"
        frames.append(sens)
    if denominator_summary is not None and not denominator_summary.empty:
        denom = denominator_summary.copy()
        denom["analysis_source"] = "denominator_and_link"
        frames.append(denom)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["effect_measure"] = out["model_family"].astype(str).map(effect_measure_for_family)
    keep_cols = [
        "analysis_source",
        "model",
        "model_family",
        "denominator",
        "estimator",
        "effect_measure",
        "n_match_rows",
        "n_players",
        "n_events",
        "dispersion",
        "rr_0",
        "rr_0_ci_low",
        "rr_0_ci_high",
        "rr_90",
        "rr_90_ci_low",
        "rr_90_ci_high",
        "rr_180",
        "rr_180_ci_low",
        "rr_180_ci_high",
        "global_interaction_p",
        "difference_in_180_vs_90_change_irr",
        "difference_in_180_vs_90_change_ci_low",
        "difference_in_180_vs_90_change_ci_high",
        "difference_in_180_vs_90_change_p",
    ]
    return out[[col for col in keep_cols if col in out.columns]].copy()


def frequency_only_publication_column(column: str) -> str:
    """Rename internal model-group prefixes for a frequency-only comparison."""
    for prefix in ("regular_", "intermediate_"):
        if column.startswith(prefix):
            return f"lower_frequency_{column[len(prefix):]}"
    for prefix in ("fragile_", "higher_"):
        if column.startswith(prefix):
            return f"higher_frequency_{column[len(prefix):]}"
    return column


def crude_daily_history_publication_table(
    panel: pd.DataFrame,
    event_col: str = "injury_event",
    group_col: str = PRIMARY_GROUP_COL,
) -> pd.DataFrame:
    """Return assumption-free daily incidence by exposure bin and history stratum."""
    required = {event_col, group_col, "all_minutes_last_7d"}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")

    frame = panel[panel[group_col].isin(PUBLICATION_HISTORY_LABELS)].copy()
    frame = frame.dropna(subset=["all_minutes_last_7d"])
    frame = add_45min_load_bins(frame)
    grouped = (
        frame.groupby([group_col, "all_minutes7d_bin"], observed=False)
        .agg(
            n_player_days=(event_col, "size"),
            n_reported_starts=(event_col, "sum"),
        )
        .reset_index()
    )
    grouped["daily_incidence"] = np.where(
        grouped["n_player_days"].gt(0),
        grouped["n_reported_starts"] / grouped["n_player_days"],
        np.nan,
    )
    grouped["daily_incidence_percent"] = grouped["daily_incidence"] * 100.0
    grouped["history_stratum"] = grouped[group_col].map(PUBLICATION_HISTORY_LABELS)
    grouped["exposure_bin_order"] = grouped["all_minutes7d_bin"].astype(str).map(
        {label: idx for idx, label in enumerate(LABELS_45)}
    )
    return grouped[
        [
            "history_stratum",
            "all_minutes7d_bin",
            "exposure_bin_order",
            "n_player_days",
            "n_reported_starts",
            "daily_incidence",
            "daily_incidence_percent",
        ]
    ].sort_values(["exposure_bin_order", "history_stratum"], ignore_index=True)


def proxy_classification_publication_table(
    panel: pd.DataFrame,
    daily_event_col: str = "injury_event",
    proxy_event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:
    """Audit how daily reported starts become the match-associated proxy."""
    required = {daily_event_col, proxy_event_col, "injury_context"}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")

    daily_starts = panel[pd.to_numeric(panel[daily_event_col], errors="coerce").eq(1)]
    context_counts = daily_starts["injury_context"].astype(str).value_counts()
    same_day = int(context_counts.get("match_same_day", 0))
    lag1 = int(context_counts.get("match_lag1_recorded_next_day", 0))
    training_other = int(context_counts.get("training_or_other", 0))
    total = int(len(daily_starts))
    candidates = same_day + lag1
    assigned = int(pd.to_numeric(panel[proxy_event_col], errors="coerce").fillna(0).sum())
    unassigned = candidates - assigned
    if unassigned < 0:
        raise ValueError("Assigned match-proxy events exceed same-day and lag-1 candidates")

    metrics = [
        ("all_reported_daily_starts", total),
        ("same_day_candidates", same_day),
        ("lag1_candidates", lag1),
        ("training_or_other", training_other),
        ("same_day_plus_lag1_candidates", candidates),
        ("assigned_matchproxy_events", assigned),
        ("unassigned_same_day_or_lag1", unassigned),
    ]
    rows = []
    for metric, count in metrics:
        rows.append(
            {
                "metric": metric,
                "n_events": count,
                "percent_of_reported_daily_starts": (
                    float(count) / total * 100.0 if total > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def proxy_event_type_summary(
    panel: pd.DataFrame,
    proxy_event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:
    """Summarise public-text injury-type categories among proxy events."""
    required = {proxy_event_col, "matchproxy_public_injury_type"}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")

    frame = panel[
        pd.to_numeric(panel[proxy_event_col], errors="coerce").fillna(0).astype(int).eq(1)
    ].copy()
    total = int(len(frame))
    if "matchproxy_lookup_desc" in frame:
        desc = frame["matchproxy_lookup_desc"].fillna("").astype(str)
    else:
        desc = pd.Series("", index=frame.index)
    covid_like = desc.str.contains(r"covid|corona", case=False, regex=True)
    if "date" in frame:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        covid_season = dates.between(
            pd.Timestamp("2020-07-01"),
            pd.Timestamp("2021-06-30"),
            inclusive="both",
        )
    else:
        covid_season = pd.Series(False, index=frame.index)

    rows: List[Dict[str, object]] = [
        {
            "metric": "all_proxy_events",
            "public_injury_type": "all",
            "n_proxy_events": total,
            "percent_of_proxy_events": 100.0 if total > 0 else np.nan,
            "n_covid_like_descriptions": int(covid_like.sum()),
            "n_covid_like_2020_2021_season": int((covid_like & covid_season).sum()),
        }
    ]
    if total == 0:
        return pd.DataFrame(rows)

    type_counts = frame["matchproxy_public_injury_type"].astype(str).value_counts()
    for injury_type, count in type_counts.items():
        mask = frame["matchproxy_public_injury_type"].astype(str).eq(injury_type)
        rows.append(
            {
                "metric": "public_injury_type",
                "public_injury_type": injury_type,
                "n_proxy_events": int(count),
                "percent_of_proxy_events": float(count) / total * 100.0,
                "n_covid_like_descriptions": int((mask & covid_like).sum()),
                "n_covid_like_2020_2021_season": int(
                    (mask & covid_like & covid_season).sum()
                ),
            }
        )
    illness_mask = frame["matchproxy_public_injury_type"].astype(str).eq(
        "illness/other medical"
    )
    rows.append(
        {
            "metric": "illness_or_other_medical_proxy_events",
            "public_injury_type": "illness/other medical",
            "n_proxy_events": int(illness_mask.sum()),
            "percent_of_proxy_events": (
                float(illness_mask.sum()) / total * 100.0 if total > 0 else np.nan
            ),
            "n_covid_like_descriptions": int((illness_mask & covid_like).sum()),
            "n_covid_like_2020_2021_season": int(
                (illness_mask & covid_like & covid_season).sum()
            ),
        }
    )
    unknown_or_unspecified = frame["matchproxy_public_injury_type"].astype(str).isin(
        ["unknown", "other/unspecified"]
    )
    rows.append(
        {
            "metric": "unknown_or_unspecified_proxy_events",
            "public_injury_type": "unknown or other/unspecified",
            "n_proxy_events": int(unknown_or_unspecified.sum()),
            "percent_of_proxy_events": (
                float(unknown_or_unspecified.sum()) / total * 100.0
                if total > 0
                else np.nan
            ),
            "n_covid_like_descriptions": int(
                (unknown_or_unspecified & covid_like).sum()
            ),
            "n_covid_like_2020_2021_season": int(
                (unknown_or_unspecified & covid_like & covid_season).sum()
            ),
        }
    )
    return pd.DataFrame(rows)


def add_p_value_adjustments(
    frame: pd.DataFrame,
    p_col: str = "p_value",
    methods: Sequence[str] = ("holm", "fdr_bh"),
) -> pd.DataFrame:
    """Add multiplicity-adjusted p-values for rows with finite p-values."""
    out = frame.copy()
    if p_col not in out:
        return out
    p_values = pd.to_numeric(out[p_col], errors="coerce")
    valid = p_values.notna() & np.isfinite(p_values)
    for method in methods:
        out[f"{p_col}_{method}"] = np.nan
        if valid.any():
            out.loc[valid, f"{p_col}_{method}"] = multipletests(
                p_values.loc[valid].astype(float),
                method=method,
            )[1]
    return out


def multiplicity_family_summary(effects: pd.DataFrame) -> pd.DataFrame:
    """Summarise all formal tests by pre-specified analysis family."""
    required = {
        "analysis_role",
        "p_value",
        "p_holm_across_specifications",
        "p_bh_across_specifications",
        "reject_holm_0_05",
        "reject_bh_0_05",
    }
    missing = required - set(effects.columns)
    if missing:
        raise KeyError(f"effects missing required columns: {sorted(missing)}")

    frame = effects.copy()
    frame["publication_test_family"] = frame["analysis_role"].map(
        MULTIPLICITY_ROLE_FAMILIES
    )
    frame = frame[frame["publication_test_family"].notna()].copy()
    for column in [
        "p_value",
        "p_holm_across_specifications",
        "p_bh_across_specifications",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["p_value"].notna()]
    if frame.empty:
        return pd.DataFrame()

    if "contrast_id" in frame:
        frame["is_history_level_contrast"] = frame["contrast_id"].astype(str).str.startswith(
            "higher_vs_intermediate_at_"
        )
    else:
        frame["is_history_level_contrast"] = False

    rows = []
    for family in MULTIPLICITY_FAMILY_ORDER:
        subset = frame[frame["publication_test_family"].eq(family)]
        if subset.empty:
            continue
        history_level = subset["is_history_level_contrast"].astype(bool)
        exposure_response = ~history_level
        rows.append(
            {
                "test_family": family,
                "n_tests": int(len(subset)),
                "n_history_level_tests": int(history_level.sum()),
                "n_exposure_response_tests": int(exposure_response.sum()),
                "minimum_raw_p": float(subset["p_value"].min()),
                "median_raw_p": float(subset["p_value"].median()),
                "minimum_holm_p": float(
                    subset["p_holm_across_specifications"].min()
                ),
                "median_holm_p": float(
                    subset["p_holm_across_specifications"].median()
                ),
                "minimum_bh_p": float(subset["p_bh_across_specifications"].min()),
                "median_bh_p": float(subset["p_bh_across_specifications"].median()),
                "holm_rejections_0_05": int(
                    subset["reject_holm_0_05"].astype(bool).sum()
                ),
                "history_level_holm_rejections_0_05": int(
                    subset.loc[history_level, "reject_holm_0_05"].astype(bool).sum()
                ),
                "exposure_response_holm_rejections_0_05": int(
                    subset.loc[exposure_response, "reject_holm_0_05"]
                    .astype(bool)
                    .sum()
                ),
                "bh_rejections_0_05": int(
                    subset["reject_bh_0_05"].astype(bool).sum()
                ),
                "history_level_bh_rejections_0_05": int(
                    subset.loc[history_level, "reject_bh_0_05"].astype(bool).sum()
                ),
                "exposure_response_bh_rejections_0_05": int(
                    subset.loc[exposure_response, "reject_bh_0_05"]
                    .astype(bool)
                    .sum()
                ),
                "adjustment_scope": "within contrast across specifications",
            }
        )
    return pd.DataFrame(rows)


def nominal_exposure_response_signal_summary(effects: pd.DataFrame) -> pd.DataFrame:
    """Return all unadjusted exposure-response signals below p=0.05."""
    required = {
        "model",
        "analysis_role",
        "estimator",
        "event_col",
        "group_col",
        "contrast_id",
        "history_stratum",
        "burden_from",
        "burden_to",
        "effect_measure",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "p_holm_across_specifications",
        "p_bh_across_specifications",
        "reject_holm_0_05",
        "reject_bh_0_05",
    }
    missing = required - set(effects.columns)
    if missing:
        raise KeyError(f"effects missing required columns: {sorted(missing)}")

    frame = effects.copy()
    frame["p_value"] = pd.to_numeric(frame["p_value"], errors="coerce")
    history_level = frame["contrast_id"].astype(str).str.startswith(
        "higher_vs_intermediate_at_"
    )
    nominal = frame.loc[
        ~history_level & frame["p_value"].notna() & frame["p_value"].lt(0.05)
    ].copy()
    ordered_columns = [
        "nominal_signal_rank",
        "model",
        "analysis_role",
        "estimator",
        "event_col",
        "group_col",
        "contrast_id",
        "history_stratum",
        "burden_from",
        "burden_to",
        "effect_measure",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "p_holm_across_specifications",
        "p_bh_across_specifications",
        "reject_holm_0_05",
        "reject_bh_0_05",
        "n_match_rows",
        "n_players",
        "n_events",
        "interpretation_note",
    ]
    if nominal.empty:
        return pd.DataFrame(columns=ordered_columns)
    nominal = nominal.sort_values(["p_value", "model", "contrast_id"]).reset_index(
        drop=True
    )
    nominal.insert(0, "nominal_signal_rank", np.arange(1, len(nominal) + 1))
    nominal["interpretation_note"] = (
        "unadjusted p<0.05 exposure-response or interaction signal; interpret "
        "through the Holm and Benjamini-Hochberg adjusted columns"
    )
    return nominal[[col for col in ordered_columns if col in nominal.columns]]


def add_cross_summary_multiplicity_columns(
    cross_summary: pd.DataFrame,
    effects: pd.DataFrame,
) -> pd.DataFrame:
    """Attach adjusted 180-minute history-contrast p-values to cross summaries."""
    if cross_summary.empty:
        return cross_summary.copy()
    required = {
        "model",
        "contrast_id",
        "p_value",
        "p_holm_across_specifications",
        "p_bh_across_specifications",
        "reject_holm_0_05",
        "reject_bh_0_05",
    }
    missing = required - set(effects.columns)
    if missing:
        raise KeyError(f"effects missing required columns: {sorted(missing)}")

    pvals = effects[
        effects["contrast_id"].astype(str).eq("higher_vs_intermediate_at_180")
    ][
        [
            "model",
            "p_value",
            "p_holm_across_specifications",
            "p_bh_across_specifications",
            "reject_holm_0_05",
            "reject_bh_0_05",
        ]
    ].rename(
        columns={
            "p_value": "rr_180_p_value",
            "p_holm_across_specifications": "rr_180_p_holm_across_specifications",
            "p_bh_across_specifications": "rr_180_p_bh_across_specifications",
            "reject_holm_0_05": "rr_180_reject_holm_0_05",
            "reject_bh_0_05": "rr_180_reject_bh_0_05",
        }
    )
    return cross_summary.merge(pvals, on="model", how="left")


def publication_referee_audit_table(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    effects: pd.DataFrame,
    cross_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return compact referee-facing quantities that support manuscript claims.

    The table intentionally duplicates only small derived quantities that a
    reader may look for: keyword classifier coverage, strict type-discordant
    history support, local anchor support for headline muscle/tendon contrasts,
    adjusted p-values for key 180-minute contrasts, and the nominal
    strict-specification interaction driver.
    """
    rows: List[Dict[str, object]] = []

    lookup = injury_spell_metadata_lookup(injuries)
    total_spells = int(len(lookup))
    muscle_spells = int(lookup["public_injury_type"].eq("muscle/tendon").sum())
    rows.append(
        {
            "metric": "muscle_tendon_keyword_spells",
            "n": muscle_spells,
            "denominator_n": total_spells,
            "percent": float(muscle_spells) / total_spells * 100.0
            if total_spells > 0
            else np.nan,
            "note": "cleaned injury spells with public text matching the muscle/tendon keyword classifier",
        }
    )

    strict_frame = prepare_model_frame(
        panel,
        SENSITIVITY_EVENT_COLS["muscle_tendon_only"],
        NON_MUSCLE_HISTORY_GROUP_COL,
    )
    nonzero_strict = strict_frame["prior_non_muscle_n_spells"].astype(float) > 0.0
    high_strict = strict_frame[NON_MUSCLE_HISTORY_GROUP_COL].astype(str).eq("fragile")
    threshold = pd.to_numeric(
        strict_frame[NON_MUSCLE_HISTORY_THRESHOLD_COL],
        errors="coerce",
    ).dropna()
    rows.extend(
        [
            {
                "metric": "strict_type_discordant_model_rows",
                "n": int(len(strict_frame)),
                "denominator_n": int(len(strict_frame)),
                "percent": 100.0,
                "n_events": int(
                    strict_frame[SENSITIVITY_EVENT_COLS["muscle_tendon_only"]].sum()
                ),
                "n_players": int(strict_frame[PLAYER_ID_COL].nunique()),
                "note": "analytic match rows for the strict type-discordant muscle/tendon model",
            },
            {
                "metric": "strict_prior_non_muscle_nonzero_rows",
                "n": int(nonzero_strict.sum()),
                "denominator_n": int(len(strict_frame)),
                "percent": float(nonzero_strict.mean() * 100.0)
                if len(strict_frame) > 0
                else np.nan,
                "n_players": int(
                    strict_frame.loc[nonzero_strict, PLAYER_ID_COL].nunique()
                ),
                "note": "rows carrying at least one prior joint/ligament or bone/fracture public report",
            },
            {
                "metric": "strict_high_non_muscle_history_rows",
                "n": int(high_strict.sum()),
                "denominator_n": int(len(strict_frame)),
                "percent": float(high_strict.mean() * 100.0)
                if len(strict_frame) > 0
                else np.nan,
                "n_players": int(strict_frame.loc[high_strict, PLAYER_ID_COL].nunique()),
                "threshold_per_10000_prior_minutes": float(threshold.iloc[0])
                if not threshold.empty
                else np.nan,
                "note": "rows at or above the strict musculoskeletal non-muscle third-quartile frequency threshold",
            },
        ]
    )

    key_contrasts = [
        (
            "reported_absence_ge28d_frequency_only_history",
            "higher_vs_intermediate_at_180",
            "ge28d_frequency_only_rr_180",
        ),
        (
            "muscle_tendon_only_frequency_only_history",
            "higher_vs_intermediate_at_180",
            "muscle_tendon_frequency_only_rr_180",
        ),
        (
            "muscle_tendon_only_non_muscle_frequency_history",
            "higher_vs_intermediate_at_180",
            "strict_type_discordant_rr_180",
        ),
        (
            "primary_same_day_plus_lag1",
            "higher_vs_intermediate_at_180",
            "primary_rr_180",
        ),
        (
            "muscle_tendon_only_non_muscle_frequency_history",
            "global_spline_by_history_interaction",
            "strict_type_discordant_global_interaction",
        ),
        (
            "muscle_tendon_only_non_muscle_frequency_history",
            "intermediate_history_180_vs_90",
            "strict_low_non_muscle_history_180_vs_90",
        ),
        (
            "muscle_tendon_only_non_muscle_frequency_history",
            "higher_history_180_vs_90",
            "strict_high_non_muscle_history_180_vs_90",
        ),
        (
            "muscle_tendon_only_non_muscle_frequency_history",
            "ratio_of_180_vs_90_changes",
            "strict_change_ratio_180_vs_90",
        ),
    ]
    for model, contrast_id, metric in key_contrasts:
        subset = effects[
            effects["model"].astype(str).eq(model)
            & effects["contrast_id"].astype(str).eq(contrast_id)
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "metric": metric,
                "model": model,
                "contrast_id": contrast_id,
                "estimate": row.get("estimate", np.nan),
                "ci_low": row.get("ci_low", np.nan),
                "ci_high": row.get("ci_high", np.nan),
                "p_value": row.get("p_value", np.nan),
                "p_holm_across_specifications": row.get(
                    "p_holm_across_specifications", np.nan
                ),
                "p_bh_across_specifications": row.get(
                    "p_bh_across_specifications", np.nan
                ),
                "reject_holm_0_05": row.get("reject_holm_0_05", np.nan),
                "reject_bh_0_05": row.get("reject_bh_0_05", np.nan),
                "n_rows": row.get("n_match_rows", np.nan),
                "n_events": row.get("n_events", np.nan),
            }
        )

    if not cross_summary.empty:
        local_support_models = [
            (
                "muscle_tendon_only_frequency_only_history",
                "muscle_tendon_all_type_frequency_history",
                "all public injury descriptions",
            ),
            (
                "muscle_tendon_only_non_muscle_frequency_history",
                "muscle_tendon_strict_type_discordant_history",
                "joint/ligament or bone/fracture prior reports",
            ),
        ]
        for model, metric_prefix, history_signal in local_support_models:
            support_subset = cross_summary[cross_summary["model"].astype(str).eq(model)]
            if support_subset.empty:
                continue
            support_row = support_subset.iloc[0]
            for stratum, stratum_label in [
                ("lower_frequency", "lower"),
                ("higher_frequency", "higher"),
            ]:
                for anchor in [0, 180]:
                    event_col = f"{stratum}_{anchor}_support_events"
                    row_col = f"{stratum}_{anchor}_support_rows"
                    rows.append(
                        {
                            "metric": (
                                f"{metric_prefix}_{stratum_label}_{anchor}_minute_local_support"
                            ),
                            "model": model,
                            "local_anchor_minutes": anchor,
                            "history_stratum": stratum_label,
                            "history_signal": history_signal,
                            "support_events": support_row.get(event_col, np.nan),
                            "support_rows": support_row.get(row_col, np.nan),
                            "note": "events and rows within plus/minus 15 minutes of the selected anchor",
                        }
                    )

        strict_cross = cross_summary[
            cross_summary["model"].astype(str).eq(
                "muscle_tendon_only_non_muscle_frequency_history"
            )
        ]
        if not strict_cross.empty:
            row = strict_cross.iloc[0]
            rows.append(
                {
                    "metric": "strict_type_discordant_cross_summary",
                    "model": row.get("model", ""),
                    "estimate": row.get("rr_180", np.nan),
                    "ci_low": row.get("rr_180_ci_low", np.nan),
                    "ci_high": row.get("rr_180_ci_high", np.nan),
                    "n_rows": row.get("n_match_rows", np.nan),
                    "n_events": row.get("n_events", np.nan),
                    "threshold_per_10000_prior_minutes": row.get(
                        "frequency_threshold_per_10000_prior_minutes",
                        np.nan,
                    ),
                    "note": "same strict type-discordant contrast as reported in the cross-summary output",
                }
            )

    return pd.DataFrame(rows)


def negative_control_magnitude_comparison(cross_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compare the all-type history signal with the type-discordant control.

    Negative-control exposure interpretation is comparative. This table keeps
    the two estimates side by side without dividing one ratio measure by the
    other, because that quotient is not a calibrated share of bias.
    """
    columns = [
        "anchor_minutes",
        "all_type_history_rr",
        "all_type_history_ci_low",
        "all_type_history_ci_high",
        "negative_control_rr",
        "negative_control_ci_low",
        "negative_control_ci_high",
        "all_type_higher_support_events",
        "negative_control_higher_support_events",
        "all_type_lower_support_events",
        "negative_control_lower_support_events",
        "all_type_higher_support_rows",
        "negative_control_higher_support_rows",
        "all_type_lower_support_rows",
        "negative_control_lower_support_rows",
        "all_type_rows_before_recent_return_exclusion",
        "all_type_events_before_recent_return_exclusion",
        "all_type_excluded_recent_return_rows",
        "all_type_excluded_recent_return_events",
        "negative_control_rows_before_recent_return_exclusion",
        "negative_control_events_before_recent_return_exclusion",
        "negative_control_excluded_recent_return_rows",
        "negative_control_excluded_recent_return_events",
        "interpretation_note",
    ]
    if cross_summary.empty:
        return pd.DataFrame(columns=columns)

    required = {"model"}
    for anchor in [0, 180]:
        required.update(
            {
                f"rr_{anchor}",
                f"rr_{anchor}_ci_low",
                f"rr_{anchor}_ci_high",
                f"higher_frequency_{anchor}_support_events",
                f"lower_frequency_{anchor}_support_events",
                f"higher_frequency_{anchor}_support_rows",
                f"lower_frequency_{anchor}_support_rows",
            }
        )
    missing = required - set(cross_summary.columns)
    if missing:
        raise KeyError(
            f"cross_summary missing required columns: {sorted(missing)}"
        )

    all_type = cross_summary[
        cross_summary["model"].astype(str).eq(
            "muscle_tendon_only_frequency_only_history"
        )
    ]
    negative_control = cross_summary[
        cross_summary["model"].astype(str).eq(
            "muscle_tendon_only_non_muscle_frequency_history"
        )
    ]
    if all_type.empty or negative_control.empty:
        return pd.DataFrame(columns=columns)

    all_row = all_type.iloc[0]
    control_row = negative_control.iloc[0]
    rows: List[Dict[str, object]] = []
    for anchor in [0, 180]:
        all_rr = float(all_row[f"rr_{anchor}"])
        control_rr = float(control_row[f"rr_{anchor}"])
        note = (
            "well-supported baseline anchor for the negative-control magnitude comparison"
            if anchor == 0
            else "exposure-anchored comparison with sparse higher-history support"
        )
        rows.append(
            {
                "anchor_minutes": anchor,
                "all_type_history_rr": all_rr,
                "all_type_history_ci_low": float(all_row[f"rr_{anchor}_ci_low"]),
                "all_type_history_ci_high": float(all_row[f"rr_{anchor}_ci_high"]),
                "negative_control_rr": control_rr,
                "negative_control_ci_low": float(
                    control_row[f"rr_{anchor}_ci_low"]
                ),
                "negative_control_ci_high": float(
                    control_row[f"rr_{anchor}_ci_high"]
                ),
                "all_type_higher_support_events": int(
                    all_row[f"higher_frequency_{anchor}_support_events"]
                ),
                "negative_control_higher_support_events": int(
                    control_row[f"higher_frequency_{anchor}_support_events"]
                ),
                "all_type_lower_support_events": int(
                    all_row[f"lower_frequency_{anchor}_support_events"]
                ),
                "negative_control_lower_support_events": int(
                    control_row[f"lower_frequency_{anchor}_support_events"]
                ),
                "all_type_higher_support_rows": int(
                    all_row[f"higher_frequency_{anchor}_support_rows"]
                ),
                "negative_control_higher_support_rows": int(
                    control_row[f"higher_frequency_{anchor}_support_rows"]
                ),
                "all_type_lower_support_rows": int(
                    all_row[f"lower_frequency_{anchor}_support_rows"]
                ),
                "negative_control_lower_support_rows": int(
                    control_row[f"lower_frequency_{anchor}_support_rows"]
                ),
                "all_type_rows_before_recent_return_exclusion": all_row.get(
                    "n_match_rows_before_restriction",
                    np.nan,
                ),
                "all_type_events_before_recent_return_exclusion": all_row.get(
                    "n_events_before_restriction",
                    np.nan,
                ),
                "all_type_excluded_recent_return_rows": all_row.get(
                    "excluded_recent_return_rows",
                    np.nan,
                ),
                "all_type_excluded_recent_return_events": all_row.get(
                    "excluded_recent_return_events",
                    np.nan,
                ),
                "negative_control_rows_before_recent_return_exclusion": control_row.get(
                    "n_match_rows_before_restriction",
                    np.nan,
                ),
                "negative_control_events_before_recent_return_exclusion": control_row.get(
                    "n_events_before_restriction",
                    np.nan,
                ),
                "negative_control_excluded_recent_return_rows": control_row.get(
                    "excluded_recent_return_rows",
                    np.nan,
                ),
                "negative_control_excluded_recent_return_events": control_row.get(
                    "excluded_recent_return_events",
                    np.nan,
                ),
                "interpretation_note": note,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def negative_control_anchor_selection_audit(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    anchor_minutes: float = 0.0,
) -> pd.DataFrame:
    """Describe recent-return and short-appearance composition at one anchor."""
    specs = [
        (
            "all_type_frequency_history",
            "fragility_frequency_only",
        ),
        (
            "type_discordant_frequency_history",
            NON_MUSCLE_HISTORY_GROUP_COL,
        ),
    ]
    rows: List[Dict[str, object]] = []
    for comparison, group_col in specs:
        frame = prepare_model_frame(
            panel,
            SENSITIVITY_EVENT_COLS["muscle_tendon_only"],
            group_col,
        )
        frame = add_recent_prior_injury_return_flags(frame, injuries)
        frame = frame[
            frame["all_minutes_last_7d"].astype(float).eq(float(anchor_minutes))
        ].copy()
        for group, publication_group in [
            ("regular", "lower frequency"),
            ("fragile", "higher frequency"),
        ]:
            subset = frame[frame["model_group"].eq(group)]
            recent_return = subset[
                "returned_from_recorded_injury_within_14d"
            ].astype(bool)
            short_appearance = subset[MATCH_MINUTES_COL].astype(float).lt(45.0)
            rows.append(
                {
                    "comparison": comparison,
                    "anchor_minutes": float(anchor_minutes),
                    "history_stratum": publication_group,
                    "match_rows": int(len(subset)),
                    "events": int(
                        subset[SENSITIVITY_EVENT_COLS["muscle_tendon_only"]].sum()
                    ),
                    "recent_return_rows": int(recent_return.sum()),
                    "recent_return_percent": float(recent_return.mean() * 100.0),
                    "short_appearance_rows": int(short_appearance.sum()),
                    "short_appearance_percent": float(
                        short_appearance.mean() * 100.0
                    ),
                    "recent_return_and_short_rows": int(
                        (recent_return & short_appearance).sum()
                    ),
                    "recent_return_and_short_percent": float(
                        (recent_return & short_appearance).mean() * 100.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def negative_control_joint_label_frame(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    exclude_recent_returns: bool,
) -> pd.DataFrame:
    """Return the shared row set carrying both negative-control history labels."""
    event_col = SENSITIVITY_EVENT_COLS["muscle_tendon_only"]
    frame = prepare_model_frame(panel, event_col, "fragility_frequency_only")
    frame = frame[frame[NON_MUSCLE_HISTORY_GROUP_COL].astype(str).isin(MODEL_GROUPS)].copy()
    frame["all_type_high_history"] = frame["model_group"].eq("fragile").astype(int)
    frame["type_discordant_high_history"] = (
        frame[NON_MUSCLE_HISTORY_GROUP_COL].astype(str).eq("fragile").astype(int)
    )
    if exclude_recent_returns:
        flagged = add_recent_prior_injury_return_flags(frame, injuries)
        recent_return = flagged["returned_from_recorded_injury_within_14d"].astype(bool)
        frame = flagged.loc[~recent_return].copy()
    return frame


def mutually_exclusive_type_frequency_frame(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    exclude_recent_returns: bool,
) -> pd.DataFrame:
    """Return the shared muscle/tendon-outcome row set with exclusive histories."""
    event_col = SENSITIVITY_EVENT_COLS["muscle_tendon_only"]
    missing_history = [
        col
        for col in MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS
        if col not in panel.columns
    ]
    enriched = (
        add_mutually_exclusive_type_frequency_history(panel, injuries)
        if missing_history
        else panel.copy()
    )
    frame = prepare_model_frame(enriched, event_col, "fragility_frequency_only")
    for col in MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    if exclude_recent_returns:
        flagged = add_recent_prior_injury_return_flags(frame, injuries)
        recent_return = flagged["returned_from_recorded_injury_within_14d"].astype(bool)
        frame = flagged.loc[~recent_return].copy()
    return frame


def latest_eligible_player_history_snapshot(
    panel: pd.DataFrame,
    history_cols: Sequence[str],
    minimum_prior_minutes: float = 900.0,
) -> pd.DataFrame:
    """Return each eligible player's latest prior-only history row."""
    required = {PLAYER_ID_COL, "date", "prior_minutes_played", *history_cols}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")
    out = panel.copy()
    out[PLAYER_ID_COL] = pd.to_numeric(out[PLAYER_ID_COL], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["prior_minutes_played"] = pd.to_numeric(
        out["prior_minutes_played"],
        errors="coerce",
    ).fillna(0.0)
    for col in history_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    eligible = out[
        out[PLAYER_ID_COL].notna()
        & out["date"].notna()
        & out["prior_minutes_played"].ge(float(minimum_prior_minutes))
    ].copy()
    if eligible.empty:
        return eligible[[PLAYER_ID_COL, "date", "prior_minutes_played", *history_cols]]
    return (
        eligible.sort_values([PLAYER_ID_COL, "date"])
        .groupby(PLAYER_ID_COL, as_index=False, group_keys=False)
        .tail(1)
        [[PLAYER_ID_COL, "date", "prior_minutes_played", *history_cols]]
    )


def distribution_statistics(values: pd.Series) -> Dict[str, float]:
    """Return compact distribution statistics for a numeric history variable."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {
            "n": 0,
            "nonzero_n": 0,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "mean": np.nan,
            "maximum": np.nan,
            "skewness": np.nan,
        }
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    return {
        "n": int(len(numeric)),
        "nonzero_n": int(numeric.gt(0.0).sum()),
        "median": float(numeric.median()),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "mean": float(numeric.mean()),
        "maximum": float(numeric.max()),
        "skewness": float(numeric.skew()) if len(numeric) >= 3 else np.nan,
    }


def q3_group_mean_statistics(values: pd.Series, threshold: float) -> Dict[str, float]:
    """Summarise observed means below and at-or-above a Q3 threshold."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    threshold = float(threshold)
    if numeric.empty or not np.isfinite(threshold):
        return {
            "below_q3_rows": 0,
            "above_q3_rows": 0,
            "below_q3_mean": np.nan,
            "above_q3_mean": np.nan,
            "above_minus_below_q3_mean_gap": np.nan,
        }
    above = numeric.ge(threshold)
    below_values = numeric.loc[~above]
    above_values = numeric.loc[above]
    below_mean = float(below_values.mean()) if not below_values.empty else np.nan
    above_mean = float(above_values.mean()) if not above_values.empty else np.nan
    if np.isfinite(below_mean) and np.isfinite(above_mean):
        mean_gap = above_mean - below_mean
    else:
        mean_gap = np.nan
    return {
        "below_q3_rows": int(len(below_values)),
        "above_q3_rows": int(len(above_values)),
        "below_q3_mean": below_mean,
        "above_q3_mean": above_mean,
        "above_minus_below_q3_mean_gap": float(mean_gap),
    }


def scaled_log_rate_ratio_interval(
    log_rate_ratio: float,
    log_rate_ratio_se: float,
    scale: float,
) -> Dict[str, float]:
    """Scale a per-unit log incidence-rate ratio to a realistic increment."""
    if not np.isfinite(float(scale)):
        return {
            "irr": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
    log_scaled = float(log_rate_ratio) * float(scale)
    se_scaled = abs(float(scale)) * float(log_rate_ratio_se)
    if se_scaled > 0.0:
        z = log_scaled / se_scaled
    elif np.isclose(log_scaled, 0.0):
        z = 0.0
    else:
        z = float(np.copysign(np.inf, log_scaled))
    return {
        "irr": float(np.exp(log_scaled)),
        "ci_low": float(np.exp(log_scaled - 1.96 * se_scaled)),
        "ci_high": float(np.exp(log_scaled + 1.96 * se_scaled)),
        "p_value": float(erfc(abs(z) / sqrt(2.0))),
    }


def scaled_log_rate_ratio_difference_interval(
    log_rate_ratio_a: float,
    log_rate_ratio_a_se: float,
    scale_a: float,
    log_rate_ratio_b: float,
    log_rate_ratio_b_se: float,
    scale_b: float,
    log_rate_ratio_difference_se: float,
) -> Dict[str, float]:
    """Delta-method interval for two same-model slopes on different scales."""
    inputs = [
        log_rate_ratio_a,
        log_rate_ratio_a_se,
        scale_a,
        log_rate_ratio_b,
        log_rate_ratio_b_se,
        scale_b,
        log_rate_ratio_difference_se,
    ]
    if any(not np.isfinite(float(value)) for value in inputs):
        return {
            "log_rate_ratio": np.nan,
            "log_rate_ratio_se": np.nan,
            "irr": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
    var_a = float(log_rate_ratio_a_se) ** 2
    var_b = float(log_rate_ratio_b_se) ** 2
    var_diff = float(log_rate_ratio_difference_se) ** 2
    covariance_ab = (var_a + var_b - var_diff) / 2.0
    log_scaled_difference = (
        float(log_rate_ratio_a) * float(scale_a)
        - float(log_rate_ratio_b) * float(scale_b)
    )
    variance = (
        float(scale_a) ** 2 * var_a
        + float(scale_b) ** 2 * var_b
        - 2.0 * float(scale_a) * float(scale_b) * covariance_ab
    )
    se = float(np.sqrt(max(variance, 0.0)))
    if se > 0.0:
        z = log_scaled_difference / se
    elif np.isclose(log_scaled_difference, 0.0):
        z = 0.0
    else:
        z = float(np.copysign(np.inf, log_scaled_difference))
    return {
        "log_rate_ratio": float(log_scaled_difference),
        "log_rate_ratio_se": se,
        "irr": float(np.exp(log_scaled_difference)),
        "ci_low": float(np.exp(log_scaled_difference - 1.96 * se)),
        "ci_high": float(np.exp(log_scaled_difference + 1.96 * se)),
        "p_value": float(erfc(abs(z) / sqrt(2.0))),
    }


def log_standard_error_from_ratio_ci(
    ci_low: float,
    ci_high: float,
    alpha: float = 0.05,
) -> float:
    """Recover an approximate log-scale standard error from a ratio CI."""
    z_critical = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
    return float((log(float(ci_high)) - log(float(ci_low))) / (2.0 * z_critical))


def observed_to_predicted_ratio_interval(
    observed_rr: float,
    observed_ci_low: float,
    observed_ci_high: float,
    predicted_log_rr: float,
    predicted_log_rr_se: float,
) -> Dict[str, float]:
    """Compare an observed binary IRR with the continuous-slope prediction."""
    observed_log_rr = log(float(observed_rr))
    observed_log_rr_se = log_standard_error_from_ratio_ci(
        observed_ci_low,
        observed_ci_high,
    )
    log_ratio = observed_log_rr - float(predicted_log_rr)
    se = sqrt(observed_log_rr_se**2 + float(predicted_log_rr_se) ** 2)
    z = log_ratio / se
    return {
        "log_ratio": float(log_ratio),
        "log_ratio_se": float(se),
        "ratio": float(exp(log_ratio)),
        "ci_low": float(exp(log_ratio - 1.96 * se)),
        "ci_high": float(exp(log_ratio + 1.96 * se)),
        "p_value": float(erfc(abs(z) / sqrt(2.0))),
    }


def negative_control_type_frequency_linearity_check(
    frequency_results: pd.DataFrame,
    binary_results: pd.DataFrame,
    distribution_context: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether binary high-history IRRs match linear continuous slopes."""
    variable_specs = [
        {
            "history_variable": "muscle_tendon",
            "history_label": "prior muscle/tendon reports",
            "log_col": "muscle_tendon_history_log_rr_per_10000min",
            "se_col": "muscle_tendon_history_log_rr_se",
            "observed_rr_col": "muscle_tendon_high_history_rr",
            "observed_ci_low_col": "muscle_tendon_high_history_ci_low",
            "observed_ci_high_col": "muscle_tendon_high_history_ci_high",
        },
        {
            "history_variable": "joint_ligament_or_bone_fracture",
            "history_label": "prior joint/ligament or bone/fracture reports",
            "log_col": "joint_bone_history_log_rr_per_10000min",
            "se_col": "joint_bone_history_log_rr_se",
            "observed_rr_col": "joint_bone_high_history_rr",
            "observed_ci_low_col": "joint_bone_high_history_ci_low",
            "observed_ci_high_col": "joint_bone_high_history_ci_high",
        },
    ]
    scopes = ["analytic_match_rows", "anchor_0_match_rows"]
    rows: List[Dict[str, object]] = []
    for restriction in frequency_results["restriction"].drop_duplicates():
        frequency_row = frequency_results[
            frequency_results["restriction"].astype(str).eq(str(restriction))
        ].iloc[0]
        binary_row = binary_results[
            binary_results["restriction"].astype(str).eq(str(restriction))
        ].iloc[0]
        for scope in scopes:
            variable_rows: Dict[str, Dict[str, float]] = {}
            for spec in variable_specs:
                context_row = distribution_context[
                    distribution_context["restriction"].astype(str).eq(str(restriction))
                    & distribution_context["distribution_scope"].astype(str).eq(scope)
                    & distribution_context["history_variable"]
                    .astype(str)
                    .eq(spec["history_variable"])
                ].iloc[0]
                gap = float(
                    context_row["above_minus_below_q3_mean_gap_per_10000min"]
                )
                continuous_log_rr = float(frequency_row[spec["log_col"]])
                continuous_log_rr_se = float(frequency_row[spec["se_col"]])
                predicted_log_rr = continuous_log_rr * gap
                predicted_log_rr_se = abs(gap) * continuous_log_rr_se
                predicted_interval = scaled_log_rate_ratio_interval(
                    continuous_log_rr,
                    continuous_log_rr_se,
                    gap,
                )
                observed_interval = observed_to_predicted_ratio_interval(
                    float(binary_row[spec["observed_rr_col"]]),
                    float(binary_row[spec["observed_ci_low_col"]]),
                    float(binary_row[spec["observed_ci_high_col"]]),
                    predicted_log_rr,
                    predicted_log_rr_se,
                )
                variable_rows[spec["history_variable"]] = {
                    "gap": gap,
                    "predicted_log_rr": predicted_log_rr,
                    "predicted_log_rr_se": predicted_log_rr_se,
                }
                rows.append(
                    {
                        "restriction": restriction,
                        "distribution_scope": scope,
                        "comparison": "single_history_variable",
                        "history_variable": spec["history_variable"],
                        "history_label": spec["history_label"],
                        "observed_mean_gap_per_10000min": gap,
                        "muscle_tendon_mean_gap_per_10000min": np.nan,
                        "joint_bone_mean_gap_per_10000min": np.nan,
                        "continuous_log_rr_per_10000min": continuous_log_rr,
                        "continuous_log_rr_se": continuous_log_rr_se,
                        "predicted_binary_irr_from_continuous_slope": predicted_interval[
                            "irr"
                        ],
                        "predicted_binary_ci_low": predicted_interval["ci_low"],
                        "predicted_binary_ci_high": predicted_interval["ci_high"],
                        "observed_binary_irr": float(
                            binary_row[spec["observed_rr_col"]]
                        ),
                        "observed_binary_ci_low": float(
                            binary_row[spec["observed_ci_low_col"]]
                        ),
                        "observed_binary_ci_high": float(
                            binary_row[spec["observed_ci_high_col"]]
                        ),
                        "observed_divided_by_predicted_ratio": observed_interval[
                            "ratio"
                        ],
                        "observed_divided_by_predicted_ci_low": observed_interval[
                            "ci_low"
                        ],
                        "observed_divided_by_predicted_ci_high": observed_interval[
                            "ci_high"
                        ],
                        "observed_divided_by_predicted_p": observed_interval[
                            "p_value"
                        ],
                        "observed_minus_predicted_percent": (
                            observed_interval["ratio"] - 1.0
                        )
                        * 100.0,
                        "ci_method": (
                            "delta method on the log scale; binary and continuous "
                            "model estimates treated as independent"
                        ),
                        "interpretation_note": (
                            "values above one mean the binary high-history estimate "
                            "is larger than the constant per-report continuous slope "
                            "predicts across the observed high-versus-lower mean gap"
                        ),
                    }
                )
            predicted_direct = scaled_log_rate_ratio_difference_interval(
                float(frequency_row["muscle_tendon_history_log_rr_per_10000min"]),
                float(frequency_row["muscle_tendon_history_log_rr_se"]),
                variable_rows["muscle_tendon"]["gap"],
                float(frequency_row["joint_bone_history_log_rr_per_10000min"]),
                float(frequency_row["joint_bone_history_log_rr_se"]),
                variable_rows["joint_ligament_or_bone_fracture"]["gap"],
                float(frequency_row["direct_ratio_log_rr_se"]),
            )
            direct_observed_interval = observed_to_predicted_ratio_interval(
                float(binary_row["direct_ratio_muscle_over_joint_bone"]),
                float(binary_row["direct_ratio_ci_low"]),
                float(binary_row["direct_ratio_ci_high"]),
                float(predicted_direct["log_rate_ratio"]),
                float(predicted_direct["log_rate_ratio_se"]),
            )
            rows.append(
                {
                    "restriction": restriction,
                    "distribution_scope": scope,
                    "comparison": "direct_muscle_tendon_over_joint_bone",
                    "history_variable": (
                        "direct_muscle_tendon_over_joint_ligament_or_bone_fracture"
                    ),
                    "history_label": (
                        "muscle/tendon history signal relative to joint/ligament "
                        "or bone/fracture history signal"
                    ),
                    "observed_mean_gap_per_10000min": np.nan,
                    "muscle_tendon_mean_gap_per_10000min": variable_rows[
                        "muscle_tendon"
                    ]["gap"],
                    "joint_bone_mean_gap_per_10000min": variable_rows[
                        "joint_ligament_or_bone_fracture"
                    ]["gap"],
                    "continuous_log_rr_per_10000min": np.nan,
                    "continuous_log_rr_se": np.nan,
                    "predicted_binary_irr_from_continuous_slope": predicted_direct[
                        "irr"
                    ],
                    "predicted_binary_ci_low": predicted_direct["ci_low"],
                    "predicted_binary_ci_high": predicted_direct["ci_high"],
                    "observed_binary_irr": float(
                        binary_row["direct_ratio_muscle_over_joint_bone"]
                    ),
                    "observed_binary_ci_low": float(binary_row["direct_ratio_ci_low"]),
                    "observed_binary_ci_high": float(
                        binary_row["direct_ratio_ci_high"]
                    ),
                    "observed_divided_by_predicted_ratio": direct_observed_interval[
                        "ratio"
                    ],
                    "observed_divided_by_predicted_ci_low": direct_observed_interval[
                        "ci_low"
                    ],
                    "observed_divided_by_predicted_ci_high": direct_observed_interval[
                        "ci_high"
                    ],
                    "observed_divided_by_predicted_p": direct_observed_interval[
                        "p_value"
                    ],
                    "observed_minus_predicted_percent": (
                        direct_observed_interval["ratio"] - 1.0
                    )
                    * 100.0,
                    "ci_method": (
                        "delta method on the log scale; binary and continuous "
                        "model estimates treated as independent"
                    ),
                    "interpretation_note": (
                        "compares the observed binary muscle-over-joint/bone ratio "
                        "with the ratio predicted by applying the two continuous "
                        "slopes across their own observed mean gaps"
                    ),
                }
            )
    return pd.DataFrame(rows)


def mutually_exclusive_type_history_thresholds(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
) -> Dict[str, float]:
    """Derive Q3 binary thresholds from latest eligible player snapshots."""
    enriched = (
        add_mutually_exclusive_type_frequency_history(panel, injuries)
        if any(col not in panel.columns for col in MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS)
        else panel.copy()
    )
    snapshots = latest_eligible_player_history_snapshot(
        enriched,
        MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS,
    )
    if snapshots.empty:
        return {
            "muscle_tendon_q3": 0.0,
            "joint_bone_q3": 0.0,
            "snapshot_players": 0,
        }
    return {
        "muscle_tendon_q3": float(
            snapshots[MUSCLE_TENDON_HISTORY_RATE_COL].quantile(0.75)
        ),
        "joint_bone_q3": float(
            snapshots[JOINT_BONE_HISTORY_RATE_COL].quantile(0.75)
        ),
        "snapshot_players": int(snapshots[PLAYER_ID_COL].nunique()),
    }


def add_mutually_exclusive_type_binary_labels(
    frame: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    """Add binary above/below-Q3 labels for exclusive injury-type histories."""
    required = set(MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS)
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"frame missing required columns: {sorted(missing)}")
    out = frame.copy()
    muscle_threshold = float(thresholds.get("muscle_tendon_q3", 0.0))
    joint_threshold = float(thresholds.get("joint_bone_q3", 0.0))
    out[MUSCLE_TENDON_HISTORY_HIGH_COL] = (
        pd.to_numeric(out[MUSCLE_TENDON_HISTORY_RATE_COL], errors="coerce")
        .fillna(0.0)
        .ge(muscle_threshold)
        .astype(int)
    )
    out[JOINT_BONE_HISTORY_HIGH_COL] = (
        pd.to_numeric(out[JOINT_BONE_HISTORY_RATE_COL], errors="coerce")
        .fillna(0.0)
        .ge(joint_threshold)
        .astype(int)
    )
    return out


def mutually_exclusive_type_binary_frame(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    exclude_recent_returns: bool,
) -> pd.DataFrame:
    """Return the shared muscle/tendon-outcome row set with exclusive Q3 labels."""
    thresholds = mutually_exclusive_type_history_thresholds(panel, injuries)
    frame = mutually_exclusive_type_frequency_frame(
        panel,
        injuries,
        exclude_recent_returns=exclude_recent_returns,
    )
    out = add_mutually_exclusive_type_binary_labels(frame, thresholds)
    out["muscle_tendon_history_threshold_per_10000min"] = thresholds[
        "muscle_tendon_q3"
    ]
    out["joint_bone_history_threshold_per_10000min"] = thresholds["joint_bone_q3"]
    out["threshold_snapshot_players"] = thresholds["snapshot_players"]
    return out


def _joint_label_prediction_template(
    anchor_minutes: float,
    all_type_high_history: int,
    type_discordant_high_history: int,
) -> pd.DataFrame:
    """Build a neutral prediction row for the direct negative-control comparison."""
    return pd.DataFrame(
        {
            "all_minutes_last_7d": [float(anchor_minutes)],
            "all_type_high_history": [int(all_type_high_history)],
            "type_discordant_high_history": [int(type_discordant_high_history)],
            "week_phase_sin": [0.0],
            "week_phase_cos": [0.0],
            "halfweek_phase_sin": [0.0],
            "halfweek_phase_cos": [0.0],
            "log_minutes_played": [np.log(90.0)],
        }
    )


def _exclusive_type_frequency_prediction_template(
    anchor_minutes: float,
    muscle_tendon_frequency: float,
    joint_bone_frequency: float,
    muscle_tendon_has_prior_report: int = 0,
    muscle_tendon_log_days_since_last_report: float = 0.0,
    joint_bone_has_prior_report: int = 0,
    joint_bone_log_days_since_last_report: float = 0.0,
) -> pd.DataFrame:
    """Build a neutral prediction row for exclusive type-frequency contrasts."""
    return pd.DataFrame(
        {
            "all_minutes_last_7d": [float(anchor_minutes)],
            MUSCLE_TENDON_HISTORY_RATE_COL: [float(muscle_tendon_frequency)],
            JOINT_BONE_HISTORY_RATE_COL: [float(joint_bone_frequency)],
            MUSCLE_TENDON_HAS_PRIOR_REPORT_COL: [int(muscle_tendon_has_prior_report)],
            MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL: [
                float(muscle_tendon_log_days_since_last_report)
            ],
            JOINT_BONE_HAS_PRIOR_REPORT_COL: [int(joint_bone_has_prior_report)],
            JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL: [
                float(joint_bone_log_days_since_last_report)
            ],
            "week_phase_sin": [0.0],
            "week_phase_cos": [0.0],
            "halfweek_phase_sin": [0.0],
            "halfweek_phase_cos": [0.0],
            "log_minutes_played": [np.log(90.0)],
        }
    )


def _exclusive_type_binary_prediction_template(
    anchor_minutes: float,
    muscle_tendon_high_history: int,
    joint_bone_high_history: int,
    muscle_tendon_has_prior_report: int = 0,
    muscle_tendon_log_days_since_last_report: float = 0.0,
    joint_bone_has_prior_report: int = 0,
    joint_bone_log_days_since_last_report: float = 0.0,
) -> pd.DataFrame:
    """Build a neutral prediction row for exclusive binary type-history contrasts."""
    return pd.DataFrame(
        {
            "all_minutes_last_7d": [float(anchor_minutes)],
            MUSCLE_TENDON_HISTORY_HIGH_COL: [int(muscle_tendon_high_history)],
            JOINT_BONE_HISTORY_HIGH_COL: [int(joint_bone_high_history)],
            MUSCLE_TENDON_HAS_PRIOR_REPORT_COL: [int(muscle_tendon_has_prior_report)],
            MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL: [
                float(muscle_tendon_log_days_since_last_report)
            ],
            JOINT_BONE_HAS_PRIOR_REPORT_COL: [int(joint_bone_has_prior_report)],
            JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL: [
                float(joint_bone_log_days_since_last_report)
            ],
            "week_phase_sin": [0.0],
            "week_phase_cos": [0.0],
            "halfweek_phase_sin": [0.0],
            "halfweek_phase_cos": [0.0],
            "log_minutes_played": [np.log(90.0)],
        }
    )


def negative_control_direct_comparison(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    anchor_minutes: float = 0.0,
) -> pd.DataFrame:  # pragma: no cover
    """
    Directly compare all-type and type-discordant history in one model.

    This is the single-model paired comparison suggested by the referee. It uses
    identical rows and events while carrying both labels, then tests whether the
    all-type label has the larger anchor association.
    """
    specs = [
        ("all eligible rows", False),
        ("exclude rows within 14 days of recorded return", True),
    ]
    event_col = SENSITIVITY_EVENT_COLS["muscle_tendon_only"]
    rows: List[Dict[str, object]] = []
    full_frame = negative_control_joint_label_frame(
        panel,
        injuries,
        exclude_recent_returns=False,
    )
    for restriction, exclude_recent_returns in specs:
        frame = (
            negative_control_joint_label_frame(
                panel,
                injuries,
                exclude_recent_returns=True,
            )
            if exclude_recent_returns
            else full_frame.copy()
        )
        burden_max = max(float(frame["all_minutes_last_7d"].max()), float(anchor_minutes))
        spline_term = spline_basis_expression(burden_max)
        formula = (
            f"{event_col} ~ {spline_term} "
            "+ all_type_high_history + type_discordant_high_history "
            f"+ {spline_term}:all_type_high_history "
            f"+ {spline_term}:type_discordant_high_history "
            "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
        )
        res = smf.glm(
            formula=formula,
            data=frame,
            family=sm.families.Poisson(),
            offset=frame["log_minutes_played"],
        ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
        design_info = res.model.data.design_info
        baseline = np.asarray(
            build_design_matrices(
                [design_info],
                _joint_label_prediction_template(anchor_minutes, 0, 0),
            )[0]
        )[0]
        all_type_high = np.asarray(
            build_design_matrices(
                [design_info],
                _joint_label_prediction_template(anchor_minutes, 1, 0),
            )[0]
        )[0]
        type_discordant_high = np.asarray(
            build_design_matrices(
                [design_info],
                _joint_label_prediction_template(anchor_minutes, 0, 1),
            )[0]
        )[0]
        params = np.asarray(res.params)
        covariance = np.asarray(res.cov_params())
        all_type_interval = delta_ratio_interval(
            params,
            covariance,
            all_type_high,
            baseline,
        )
        control_interval = delta_ratio_interval(
            params,
            covariance,
            type_discordant_high,
            baseline,
        )
        paired_interval = delta_ratio_interval(
            params,
            covariance,
            all_type_high,
            type_discordant_high,
        )
        anchor_frame = frame[
            frame["all_minutes_last_7d"].astype(float).eq(float(anchor_minutes))
        ]

        rows.append(
            {
                "restriction": restriction,
                "anchor_minutes": float(anchor_minutes),
                "all_type_conditional_rr": all_type_interval["rate_ratio"],
                "all_type_conditional_ci_low": all_type_interval["rr_ci_low"],
                "all_type_conditional_ci_high": all_type_interval["rr_ci_high"],
                "negative_control_conditional_rr": control_interval["rate_ratio"],
                "negative_control_conditional_ci_low": control_interval["rr_ci_low"],
                "negative_control_conditional_ci_high": control_interval["rr_ci_high"],
                "direct_ratio_all_type_over_negative_control": paired_interval[
                    "rate_ratio"
                ],
                "direct_ratio_ci_low": paired_interval["rr_ci_low"],
                "direct_ratio_ci_high": paired_interval["rr_ci_high"],
                "direct_ratio_p": paired_interval["p_value"],
                "shared_rows": int(len(frame)),
                "shared_events": int(frame[event_col].sum()),
                "shared_player_count": int(frame[PLAYER_ID_COL].nunique()),
                "anchor_rows": int(len(anchor_frame)),
                "anchor_events": int(anchor_frame[event_col].sum()),
                "rows_before_recent_return_exclusion": int(len(full_frame)),
                "events_before_recent_return_exclusion": int(
                    full_frame[event_col].sum()
                ),
                "excluded_recent_return_rows": int(len(full_frame) - len(frame)),
                "excluded_recent_return_events": int(
                    full_frame[event_col].sum() - frame[event_col].sum()
                ),
                "interpretation_note": (
                    "single Poisson spline model carrying both labels; values above "
                    "1 mean the all-type history association is larger than the "
                    "type-discordant comparison on the same rows and events"
                ),
            }
        )
    return pd.DataFrame(rows)


def negative_control_mutually_exclusive_type_frequency_comparison(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    anchor_minutes: float = 0.0,
) -> pd.DataFrame:  # pragma: no cover
    """Compare mutually exclusive prior muscle/tendon and non-muscle histories."""
    specs = [
        ("all eligible rows", False),
        ("exclude rows within 14 days of recorded return", True),
    ]
    event_col = SENSITIVITY_EVENT_COLS["muscle_tendon_only"]
    full_frame = mutually_exclusive_type_frequency_frame(
        panel,
        injuries,
        exclude_recent_returns=False,
    )
    rows: List[Dict[str, object]] = []
    for restriction, exclude_recent_returns in specs:
        frame = (
            mutually_exclusive_type_frequency_frame(
                panel,
                injuries,
                exclude_recent_returns=True,
            )
            if exclude_recent_returns
            else full_frame.copy()
        )
        frame = add_symmetric_type_recency(frame, injuries)
        burden_max = max(float(frame["all_minutes_last_7d"].max()), float(anchor_minutes))
        spline_term = spline_basis_expression(burden_max)
        formula = (
            f"{event_col} ~ {spline_term} "
            f"+ {MUSCLE_TENDON_HISTORY_RATE_COL} + {JOINT_BONE_HISTORY_RATE_COL} "
            f"+ {spline_term}:{MUSCLE_TENDON_HISTORY_RATE_COL} "
            f"+ {spline_term}:{JOINT_BONE_HISTORY_RATE_COL} "
            "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
        )
        res = smf.glm(
            formula=formula,
            data=frame,
            family=sm.families.Poisson(),
            offset=frame["log_minutes_played"],
        ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
        recency_res = smf.glm(
            formula=f"{formula} {symmetric_type_recency_terms()}",
            data=frame,
            family=sm.families.Poisson(),
            offset=frame["log_minutes_played"],
        ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
        intervals = exclusive_type_frequency_anchor_intervals(res, anchor_minutes)
        recency_intervals = exclusive_type_frequency_anchor_intervals(
            recency_res,
            anchor_minutes,
        )
        muscle_interval = intervals["muscle_tendon"]
        joint_bone_interval = intervals["joint_bone"]
        paired_interval = intervals["direct"]
        recency_muscle_interval = recency_intervals["muscle_tendon"]
        recency_joint_bone_interval = recency_intervals["joint_bone"]
        recency_paired_interval = recency_intervals["direct"]
        anchor_frame = frame[
            frame["all_minutes_last_7d"].astype(float).eq(float(anchor_minutes))
        ]
        recency_summary = symmetric_type_recency_summary(frame)

        rows.append(
            {
                "restriction": restriction,
                "anchor_minutes": float(anchor_minutes),
                "muscle_tendon_history_log_rr_per_10000min": muscle_interval[
                    "log_rate_ratio"
                ],
                "muscle_tendon_history_log_rr_se": muscle_interval[
                    "log_rate_ratio_se"
                ],
                "muscle_tendon_history_irr_per_10000min": muscle_interval[
                    "rate_ratio"
                ],
                "muscle_tendon_history_ci_low": muscle_interval["rr_ci_low"],
                "muscle_tendon_history_ci_high": muscle_interval["rr_ci_high"],
                "muscle_tendon_history_p": muscle_interval["p_value"],
                "joint_bone_history_log_rr_per_10000min": joint_bone_interval[
                    "log_rate_ratio"
                ],
                "joint_bone_history_log_rr_se": joint_bone_interval[
                    "log_rate_ratio_se"
                ],
                "joint_bone_history_irr_per_10000min": joint_bone_interval[
                    "rate_ratio"
                ],
                "joint_bone_history_ci_low": joint_bone_interval["rr_ci_low"],
                "joint_bone_history_ci_high": joint_bone_interval["rr_ci_high"],
                "joint_bone_history_p": joint_bone_interval["p_value"],
                "direct_ratio_muscle_over_joint_bone": paired_interval[
                    "rate_ratio"
                ],
                "direct_ratio_ci_low": paired_interval["rr_ci_low"],
                "direct_ratio_ci_high": paired_interval["rr_ci_high"],
                "direct_ratio_log_rr": paired_interval["log_rate_ratio"],
                "direct_ratio_log_rr_se": paired_interval["log_rate_ratio_se"],
                "direct_ratio_p": paired_interval["p_value"],
                "recency_adjusted_muscle_tendon_history_log_rr_per_10000min": (
                    recency_muscle_interval["log_rate_ratio"]
                ),
                "recency_adjusted_muscle_tendon_history_log_rr_se": (
                    recency_muscle_interval["log_rate_ratio_se"]
                ),
                "recency_adjusted_muscle_tendon_history_irr_per_10000min": (
                    recency_muscle_interval["rate_ratio"]
                ),
                "recency_adjusted_muscle_tendon_history_ci_low": (
                    recency_muscle_interval["rr_ci_low"]
                ),
                "recency_adjusted_muscle_tendon_history_ci_high": (
                    recency_muscle_interval["rr_ci_high"]
                ),
                "recency_adjusted_muscle_tendon_history_p": (
                    recency_muscle_interval["p_value"]
                ),
                "recency_adjusted_joint_bone_history_log_rr_per_10000min": (
                    recency_joint_bone_interval["log_rate_ratio"]
                ),
                "recency_adjusted_joint_bone_history_log_rr_se": (
                    recency_joint_bone_interval["log_rate_ratio_se"]
                ),
                "recency_adjusted_joint_bone_history_irr_per_10000min": (
                    recency_joint_bone_interval["rate_ratio"]
                ),
                "recency_adjusted_joint_bone_history_ci_low": (
                    recency_joint_bone_interval["rr_ci_low"]
                ),
                "recency_adjusted_joint_bone_history_ci_high": (
                    recency_joint_bone_interval["rr_ci_high"]
                ),
                "recency_adjusted_joint_bone_history_p": (
                    recency_joint_bone_interval["p_value"]
                ),
                "recency_adjusted_direct_ratio_muscle_over_joint_bone": (
                    recency_paired_interval["rate_ratio"]
                ),
                "recency_adjusted_direct_ratio_ci_low": recency_paired_interval[
                    "rr_ci_low"
                ],
                "recency_adjusted_direct_ratio_ci_high": recency_paired_interval[
                    "rr_ci_high"
                ],
                "recency_adjusted_direct_ratio_log_rr": recency_paired_interval[
                    "log_rate_ratio"
                ],
                "recency_adjusted_direct_ratio_log_rr_se": recency_paired_interval[
                    "log_rate_ratio_se"
                ],
                "recency_adjusted_direct_ratio_p": recency_paired_interval["p_value"],
                **recency_summary,
                "shared_rows": int(len(frame)),
                "shared_events": int(frame[event_col].sum()),
                "shared_player_count": int(frame[PLAYER_ID_COL].nunique()),
                "anchor_rows": int(len(anchor_frame)),
                "anchor_events": int(anchor_frame[event_col].sum()),
                "rows_before_recent_return_exclusion": int(len(full_frame)),
                "events_before_recent_return_exclusion": int(
                    full_frame[event_col].sum()
                ),
                "excluded_recent_return_rows": int(len(full_frame) - len(frame)),
                "excluded_recent_return_events": int(
                    full_frame[event_col].sum() - frame[event_col].sum()
                ),
                "interpretation_note": (
                    "mutually exclusive continuous history frequencies entered "
                    "simultaneously; IRRs are per one additional prior reported "
                    "injury per 10,000 previous club minutes at the anchor; "
                    "recency-adjusted columns hold symmetric type-specific "
                    "prior-report recency terms fixed"
                ),
            }
        )
    return pd.DataFrame(rows)


def negative_control_mutually_exclusive_type_binary_comparison(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    anchor_minutes: float = 0.0,
) -> pd.DataFrame:  # pragma: no cover
    """Compare mutually exclusive binary Q3 injury-type histories like for like."""
    specs = [
        ("all eligible rows", False),
        ("exclude rows within 14 days of recorded return", True),
    ]
    event_col = SENSITIVITY_EVENT_COLS["muscle_tendon_only"]
    full_frame = mutually_exclusive_type_binary_frame(
        panel,
        injuries,
        exclude_recent_returns=False,
    )
    rows: List[Dict[str, object]] = []
    for restriction, exclude_recent_returns in specs:
        frame = (
            mutually_exclusive_type_binary_frame(
                panel,
                injuries,
                exclude_recent_returns=True,
            )
            if exclude_recent_returns
            else full_frame.copy()
        )
        frame = add_symmetric_type_recency(frame, injuries)
        burden_max = max(float(frame["all_minutes_last_7d"].max()), float(anchor_minutes))
        spline_term = spline_basis_expression(burden_max)
        formula = (
            f"{event_col} ~ {spline_term} "
            f"+ {MUSCLE_TENDON_HISTORY_HIGH_COL} + {JOINT_BONE_HISTORY_HIGH_COL} "
            f"+ {spline_term}:{MUSCLE_TENDON_HISTORY_HIGH_COL} "
            f"+ {spline_term}:{JOINT_BONE_HISTORY_HIGH_COL} "
            "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
        )
        res = smf.glm(
            formula=formula,
            data=frame,
            family=sm.families.Poisson(),
            offset=frame["log_minutes_played"],
        ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
        recency_res = smf.glm(
            formula=f"{formula} {symmetric_type_recency_terms()}",
            data=frame,
            family=sm.families.Poisson(),
            offset=frame["log_minutes_played"],
        ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
        intervals = exclusive_type_binary_anchor_intervals(res, anchor_minutes)
        recency_intervals = exclusive_type_binary_anchor_intervals(
            recency_res,
            anchor_minutes,
        )
        muscle_interval = intervals["muscle_tendon"]
        joint_bone_interval = intervals["joint_bone"]
        paired_interval = intervals["direct"]
        recency_muscle_interval = recency_intervals["muscle_tendon"]
        recency_joint_bone_interval = recency_intervals["joint_bone"]
        recency_paired_interval = recency_intervals["direct"]
        anchor_frame = frame[
            frame["all_minutes_last_7d"].astype(float).eq(float(anchor_minutes))
        ]
        muscle_high = frame[MUSCLE_TENDON_HISTORY_HIGH_COL].eq(1)
        joint_high = frame[JOINT_BONE_HISTORY_HIGH_COL].eq(1)
        recency_summary = symmetric_type_recency_summary(frame)

        rows.append(
            {
                "restriction": restriction,
                "anchor_minutes": float(anchor_minutes),
                "muscle_tendon_threshold_per_10000min": float(
                    frame["muscle_tendon_history_threshold_per_10000min"].iloc[0]
                ),
                "joint_bone_threshold_per_10000min": float(
                    frame["joint_bone_history_threshold_per_10000min"].iloc[0]
                ),
                "threshold_snapshot_players": int(
                    frame["threshold_snapshot_players"].iloc[0]
                ),
                "muscle_tendon_high_history_rr": muscle_interval["rate_ratio"],
                "muscle_tendon_high_history_ci_low": muscle_interval["rr_ci_low"],
                "muscle_tendon_high_history_ci_high": muscle_interval["rr_ci_high"],
                "muscle_tendon_high_history_p": muscle_interval["p_value"],
                "joint_bone_high_history_rr": joint_bone_interval["rate_ratio"],
                "joint_bone_high_history_ci_low": joint_bone_interval["rr_ci_low"],
                "joint_bone_high_history_ci_high": joint_bone_interval["rr_ci_high"],
                "joint_bone_high_history_p": joint_bone_interval["p_value"],
                "direct_ratio_muscle_over_joint_bone": paired_interval[
                    "rate_ratio"
                ],
                "direct_ratio_ci_low": paired_interval["rr_ci_low"],
                "direct_ratio_ci_high": paired_interval["rr_ci_high"],
                "direct_ratio_p": paired_interval["p_value"],
                "recency_adjusted_muscle_tendon_high_history_rr": (
                    recency_muscle_interval["rate_ratio"]
                ),
                "recency_adjusted_muscle_tendon_high_history_ci_low": (
                    recency_muscle_interval["rr_ci_low"]
                ),
                "recency_adjusted_muscle_tendon_high_history_ci_high": (
                    recency_muscle_interval["rr_ci_high"]
                ),
                "recency_adjusted_muscle_tendon_high_history_p": (
                    recency_muscle_interval["p_value"]
                ),
                "recency_adjusted_joint_bone_high_history_rr": (
                    recency_joint_bone_interval["rate_ratio"]
                ),
                "recency_adjusted_joint_bone_high_history_ci_low": (
                    recency_joint_bone_interval["rr_ci_low"]
                ),
                "recency_adjusted_joint_bone_high_history_ci_high": (
                    recency_joint_bone_interval["rr_ci_high"]
                ),
                "recency_adjusted_joint_bone_high_history_p": (
                    recency_joint_bone_interval["p_value"]
                ),
                "recency_adjusted_direct_ratio_muscle_over_joint_bone": (
                    recency_paired_interval["rate_ratio"]
                ),
                "recency_adjusted_direct_ratio_ci_low": recency_paired_interval[
                    "rr_ci_low"
                ],
                "recency_adjusted_direct_ratio_ci_high": recency_paired_interval[
                    "rr_ci_high"
                ],
                "recency_adjusted_direct_ratio_p": recency_paired_interval["p_value"],
                **recency_summary,
                "shared_rows": int(len(frame)),
                "shared_events": int(frame[event_col].sum()),
                "shared_player_count": int(frame[PLAYER_ID_COL].nunique()),
                "anchor_rows": int(len(anchor_frame)),
                "anchor_events": int(anchor_frame[event_col].sum()),
                "muscle_tendon_high_rows": int(muscle_high.sum()),
                "muscle_tendon_high_events": int(frame.loc[muscle_high, event_col].sum()),
                "joint_bone_high_rows": int(joint_high.sum()),
                "joint_bone_high_events": int(frame.loc[joint_high, event_col].sum()),
                "both_high_rows": int((muscle_high & joint_high).sum()),
                "both_high_events": int(frame.loc[muscle_high & joint_high, event_col].sum()),
                "rows_before_recent_return_exclusion": int(len(full_frame)),
                "events_before_recent_return_exclusion": int(
                    full_frame[event_col].sum()
                ),
                "excluded_recent_return_rows": int(len(full_frame) - len(frame)),
                "excluded_recent_return_events": int(
                    full_frame[event_col].sum() - frame[event_col].sum()
                ),
                "interpretation_note": (
                    "mutually exclusive above/below-Q3 type-history labels entered "
                    "simultaneously; this is the binary-threshold counterpart to the "
                    "overlapping same-row negative-control comparison; recency-adjusted "
                    "columns hold symmetric type-specific prior-report recency terms fixed"
                ),
            }
        )
    return pd.DataFrame(rows)


def named_term_contrast_interval(
    res,
    term_weights: Dict[str, float],
) -> Dict[str, float]:  # pragma: no cover
    """Return a rate-ratio interval for a named coefficient contrast."""
    terms = list(res.params.index)
    design = np.zeros(len(terms), dtype=float)
    for term, weight in term_weights.items():
        design[terms.index(term)] = float(weight)
    return delta_ratio_interval(
        np.asarray(res.params),
        np.asarray(res.cov_params()),
        design,
        np.zeros(len(terms), dtype=float),
    )


def exclusive_type_frequency_anchor_intervals(
    res,
    anchor_minutes: float,
) -> Dict[str, Dict[str, float]]:  # pragma: no cover
    """Return matched anchor contrasts for mutually exclusive continuous histories."""
    design_info = res.model.data.design_info
    baseline = np.asarray(
        build_design_matrices(
            [design_info],
            _exclusive_type_frequency_prediction_template(anchor_minutes, 0.0, 0.0),
        )[0]
    )[0]
    muscle_history = np.asarray(
        build_design_matrices(
            [design_info],
            _exclusive_type_frequency_prediction_template(anchor_minutes, 1.0, 0.0),
        )[0]
    )[0]
    joint_bone_history = np.asarray(
        build_design_matrices(
            [design_info],
            _exclusive_type_frequency_prediction_template(anchor_minutes, 0.0, 1.0),
        )[0]
    )[0]
    params = np.asarray(res.params)
    covariance = np.asarray(res.cov_params())
    return {
        "muscle_tendon": delta_ratio_interval(
            params,
            covariance,
            muscle_history,
            baseline,
        ),
        "joint_bone": delta_ratio_interval(
            params,
            covariance,
            joint_bone_history,
            baseline,
        ),
        "direct": delta_ratio_interval(
            params,
            covariance,
            muscle_history,
            joint_bone_history,
        ),
    }


def exclusive_type_binary_anchor_intervals(
    res,
    anchor_minutes: float,
) -> Dict[str, Dict[str, float]]:  # pragma: no cover
    """Return matched anchor contrasts for mutually exclusive binary histories."""
    design_info = res.model.data.design_info
    baseline = np.asarray(
        build_design_matrices(
            [design_info],
            _exclusive_type_binary_prediction_template(anchor_minutes, 0, 0),
        )[0]
    )[0]
    muscle_history = np.asarray(
        build_design_matrices(
            [design_info],
            _exclusive_type_binary_prediction_template(anchor_minutes, 1, 0),
        )[0]
    )[0]
    joint_bone_history = np.asarray(
        build_design_matrices(
            [design_info],
            _exclusive_type_binary_prediction_template(anchor_minutes, 0, 1),
        )[0]
    )[0]
    params = np.asarray(res.params)
    covariance = np.asarray(res.cov_params())
    return {
        "muscle_tendon": delta_ratio_interval(
            params,
            covariance,
            muscle_history,
            baseline,
        ),
        "joint_bone": delta_ratio_interval(
            params,
            covariance,
            joint_bone_history,
            baseline,
        ),
        "direct": delta_ratio_interval(
            params,
            covariance,
            muscle_history,
            joint_bone_history,
        ),
    }


def negative_control_type_frequency_linearity_formal_test(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:  # pragma: no cover
    """Test high-frequency indicators after continuous type-history frequency."""
    specs = [
        ("all eligible rows", False),
        ("exclude rows within 14 days of recorded return", True),
    ]
    event_col = SENSITIVITY_EVENT_COLS["muscle_tendon_only"]
    full_frame = mutually_exclusive_type_binary_frame(
        panel,
        injuries,
        exclude_recent_returns=False,
    )
    variable_specs = [
        {
            "history_variable": "muscle_tendon",
            "history_label": "prior muscle/tendon reports",
            "term": MUSCLE_TENDON_HISTORY_HIGH_COL,
            "continuous_term": MUSCLE_TENDON_HISTORY_RATE_COL,
            "threshold_col": "muscle_tendon_history_threshold_per_10000min",
        },
        {
            "history_variable": "joint_ligament_or_bone_fracture",
            "history_label": "prior joint/ligament or bone/fracture reports",
            "term": JOINT_BONE_HISTORY_HIGH_COL,
            "continuous_term": JOINT_BONE_HISTORY_RATE_COL,
            "threshold_col": "joint_bone_history_threshold_per_10000min",
        },
    ]
    recency_model_specs = [
        {
            "recency_adjustment": "none",
            "extra_terms": "",
            "model_note": (
                "single Poisson model with burden spline, both continuous "
                "prior-frequency terms, both above-Q3 indicators, calendar terms, "
                "observed-minute offset, and player-clustered standard errors; "
                "the indicator tests excess high-frequency signal beyond a "
                "constant per-report log-linear term"
            ),
        },
        {
            "recency_adjustment": "symmetric_type_specific",
            "extra_terms": f" {symmetric_type_recency_terms()}",
            "model_note": (
                "same formal linearity model with matched muscle/tendon and "
                "joint/ligament-or-bone prior-report recency terms added; these "
                "terms define a controlled direct effect because recency is derived "
                "from each type-specific prior-report process"
            ),
        },
    ]
    rows: List[Dict[str, object]] = []
    for restriction, exclude_recent_returns in specs:
        frame = (
            mutually_exclusive_type_binary_frame(
                panel,
                injuries,
                exclude_recent_returns=True,
            )
            if exclude_recent_returns
            else full_frame.copy()
        )
        frame = add_symmetric_type_recency(frame, injuries)
        recency_summary = symmetric_type_recency_summary(frame)
        collinearity_summary = type_frequency_recency_collinearity_summary(frame)
        burden_max = float(frame["all_minutes_last_7d"].max())
        spline_term = spline_basis_expression(burden_max)
        base_formula = (
            f"{event_col} ~ {spline_term} "
            f"+ {MUSCLE_TENDON_HISTORY_RATE_COL} + {JOINT_BONE_HISTORY_RATE_COL} "
            f"+ {MUSCLE_TENDON_HISTORY_HIGH_COL} + {JOINT_BONE_HISTORY_HIGH_COL} "
            "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
        )
        for model_spec in recency_model_specs:
            res = smf.glm(
                formula=f"{base_formula}{model_spec['extra_terms']}",
                data=frame,
                family=sm.families.Poisson(),
                offset=frame["log_minutes_played"],
            ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
            frequency_vifs = {
                "muscle_tendon_frequency_vif": fitted_term_variance_inflation_factor(
                    res,
                    MUSCLE_TENDON_HISTORY_RATE_COL,
                ),
                "joint_bone_frequency_vif": fitted_term_variance_inflation_factor(
                    res,
                    JOINT_BONE_HISTORY_RATE_COL,
                ),
            }
            continuous_intervals = {
                spec["history_variable"]: named_term_contrast_interval(
                    res,
                    {spec["continuous_term"]: 1.0},
                )
                for spec in variable_specs
            }
            direct_continuous = named_term_contrast_interval(
                res,
                {
                    MUSCLE_TENDON_HISTORY_RATE_COL: 1.0,
                    JOINT_BONE_HISTORY_RATE_COL: -1.0,
                },
            )
            for spec in variable_specs:
                interval = named_term_contrast_interval(res, {spec["term"]: 1.0})
                continuous = continuous_intervals[spec["history_variable"]]
                rows.append(
                    {
                        "restriction": restriction,
                        "recency_adjustment": model_spec["recency_adjustment"],
                        "comparison": "single_history_variable",
                        "history_variable": spec["history_variable"],
                        "history_label": spec["history_label"],
                        "threshold_per_10000min": float(
                            frame[spec["threshold_col"]].iloc[0]
                        ),
                        "binary_step_rr_above_linear_frequency": interval["rate_ratio"],
                        "binary_step_ci_low": interval["rr_ci_low"],
                        "binary_step_ci_high": interval["rr_ci_high"],
                        "binary_step_log_rr": interval["log_rate_ratio"],
                        "binary_step_log_rr_se": interval["log_rate_ratio_se"],
                        "binary_step_p": interval["p_value"],
                        "continuous_rr_adjusted_per_10000min": continuous["rate_ratio"],
                        "continuous_ci_low": continuous["rr_ci_low"],
                        "continuous_ci_high": continuous["rr_ci_high"],
                        "continuous_log_rr": continuous["log_rate_ratio"],
                        "continuous_log_rr_se": continuous["log_rate_ratio_se"],
                        "continuous_p": continuous["p_value"],
                        "direct_continuous_ratio_muscle_over_joint_bone": (
                            direct_continuous["rate_ratio"]
                        ),
                        "direct_continuous_ratio_ci_low": direct_continuous["rr_ci_low"],
                        "direct_continuous_ratio_ci_high": direct_continuous["rr_ci_high"],
                        "direct_continuous_ratio_p": direct_continuous["p_value"],
                        **recency_summary,
                        **collinearity_summary,
                        **frequency_vifs,
                        "shared_rows": int(len(frame)),
                        "shared_events": int(frame[event_col].sum()),
                        "shared_player_count": int(frame[PLAYER_ID_COL].nunique()),
                        "rows_before_recent_return_exclusion": int(len(full_frame)),
                        "events_before_recent_return_exclusion": int(
                            full_frame[event_col].sum()
                        ),
                        "excluded_recent_return_rows": int(len(full_frame) - len(frame)),
                        "excluded_recent_return_events": int(
                            full_frame[event_col].sum() - frame[event_col].sum()
                        ),
                        "model_note": model_spec["model_note"],
                    }
                )
            direct = named_term_contrast_interval(
                res,
                {
                    MUSCLE_TENDON_HISTORY_HIGH_COL: 1.0,
                    JOINT_BONE_HISTORY_HIGH_COL: -1.0,
                },
            )
            rows.append(
                {
                    "restriction": restriction,
                    "recency_adjustment": model_spec["recency_adjustment"],
                    "comparison": "direct_muscle_tendon_over_joint_bone",
                    "history_variable": (
                        "direct_muscle_tendon_over_joint_ligament_or_bone_fracture"
                    ),
                    "history_label": (
                        "muscle/tendon high-frequency step relative to the joint/"
                        "ligament or bone/fracture high-frequency step"
                    ),
                    "threshold_per_10000min": np.nan,
                    "binary_step_rr_above_linear_frequency": direct["rate_ratio"],
                    "binary_step_ci_low": direct["rr_ci_low"],
                    "binary_step_ci_high": direct["rr_ci_high"],
                    "binary_step_log_rr": direct["log_rate_ratio"],
                    "binary_step_log_rr_se": direct["log_rate_ratio_se"],
                    "binary_step_p": direct["p_value"],
                    "continuous_rr_adjusted_per_10000min": np.nan,
                    "continuous_ci_low": np.nan,
                    "continuous_ci_high": np.nan,
                    "continuous_log_rr": np.nan,
                    "continuous_log_rr_se": np.nan,
                    "continuous_p": np.nan,
                    "direct_continuous_ratio_muscle_over_joint_bone": direct_continuous[
                        "rate_ratio"
                    ],
                    "direct_continuous_ratio_ci_low": direct_continuous["rr_ci_low"],
                    "direct_continuous_ratio_ci_high": direct_continuous["rr_ci_high"],
                    "direct_continuous_ratio_p": direct_continuous["p_value"],
                    **recency_summary,
                    **collinearity_summary,
                    **frequency_vifs,
                    "shared_rows": int(len(frame)),
                    "shared_events": int(frame[event_col].sum()),
                    "shared_player_count": int(frame[PLAYER_ID_COL].nunique()),
                    "rows_before_recent_return_exclusion": int(len(full_frame)),
                    "events_before_recent_return_exclusion": int(
                        full_frame[event_col].sum()
                    ),
                    "excluded_recent_return_rows": int(len(full_frame) - len(frame)),
                    "excluded_recent_return_events": int(
                        full_frame[event_col].sum() - frame[event_col].sum()
                    ),
                    "model_note": model_spec["model_note"],
                }
            )
    return pd.DataFrame(rows)


def recency_attenuation_stacked_formula(event_col: str, spline_term: str) -> str:
    """Return the two-specification stacked formula used to test coefficient change."""
    shared_terms = [
        spline_term,
        MUSCLE_TENDON_HISTORY_RATE_COL,
        JOINT_BONE_HISTORY_RATE_COL,
        MUSCLE_TENDON_HISTORY_HIGH_COL,
        JOINT_BONE_HISTORY_HIGH_COL,
        "week_phase_sin",
        "week_phase_cos",
        "halfweek_phase_sin",
        "halfweek_phase_cos",
    ]
    terms = [
        term
        for shared_term in shared_terms
        for term in (
            f"spec_unadjusted:{shared_term}",
            f"spec_adjusted:{shared_term}",
        )
    ]
    terms.extend(
        f"spec_adjusted:{term}"
        for term in (
            MUSCLE_TENDON_HAS_PRIOR_REPORT_COL,
            MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL,
            JOINT_BONE_HAS_PRIOR_REPORT_COL,
            JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL,
        )
    )
    return (
        f"{event_col} ~ 0 + spec_unadjusted + spec_adjusted + "
        + " + ".join(terms)
    )


def recency_attenuation_contrast_rows(
    result,
    restriction: str,
    frame: pd.DataFrame,
    event_col: str,
) -> List[Dict[str, object]]:
    """Extract paired adjusted-versus-unadjusted type-history contrasts."""
    contrast_specs = [
        (
            "muscle_tendon_high_step",
            "muscle/tendon high-frequency step",
            {MUSCLE_TENDON_HISTORY_HIGH_COL: 1.0},
        ),
        (
            "joint_bone_high_step",
            "joint/ligament or bone/fracture high-frequency step",
            {JOINT_BONE_HISTORY_HIGH_COL: 1.0},
        ),
        (
            "direct_high_step",
            "muscle/tendon versus joint/ligament or bone/fracture high-frequency step",
            {
                MUSCLE_TENDON_HISTORY_HIGH_COL: 1.0,
                JOINT_BONE_HISTORY_HIGH_COL: -1.0,
            },
        ),
        (
            "muscle_tendon_continuous",
            "muscle/tendon continuous frequency term",
            {MUSCLE_TENDON_HISTORY_RATE_COL: 1.0},
        ),
        (
            "joint_bone_continuous",
            "joint/ligament or bone/fracture continuous frequency term",
            {JOINT_BONE_HISTORY_RATE_COL: 1.0},
        ),
        (
            "direct_continuous",
            "muscle/tendon versus joint/ligament or bone/fracture continuous term",
            {
                MUSCLE_TENDON_HISTORY_RATE_COL: 1.0,
                JOINT_BONE_HISTORY_RATE_COL: -1.0,
            },
        ),
    ]
    rows: List[Dict[str, object]] = []
    for contrast_id, label, base_weights in contrast_specs:
        unadjusted_weights = {
            f"spec_unadjusted:{term}": weight
            for term, weight in base_weights.items()
        }
        adjusted_weights = {
            f"spec_adjusted:{term}": weight
            for term, weight in base_weights.items()
        }
        change_weights = {
            **{term: -weight for term, weight in unadjusted_weights.items()},
            **adjusted_weights,
        }
        unadjusted = named_term_contrast_interval(result, unadjusted_weights)
        adjusted = named_term_contrast_interval(result, adjusted_weights)
        change = named_term_contrast_interval(result, change_weights)
        rows.append(
            {
                "restriction": restriction,
                "contrast_id": contrast_id,
                "contrast_label": label,
                "unadjusted_irr": unadjusted["rate_ratio"],
                "unadjusted_ci_low": unadjusted["rr_ci_low"],
                "unadjusted_ci_high": unadjusted["rr_ci_high"],
                "unadjusted_p": unadjusted["p_value"],
                "recency_adjusted_irr": adjusted["rate_ratio"],
                "recency_adjusted_ci_low": adjusted["rr_ci_low"],
                "recency_adjusted_ci_high": adjusted["rr_ci_high"],
                "recency_adjusted_p": adjusted["p_value"],
                "adjusted_over_unadjusted_ratio": change["rate_ratio"],
                "adjusted_over_unadjusted_ci_low": change["rr_ci_low"],
                "adjusted_over_unadjusted_ci_high": change["rr_ci_high"],
                "adjusted_over_unadjusted_p": change["p_value"],
                "log_coefficient_change": change["log_rate_ratio"],
                "log_coefficient_change_se": change["log_rate_ratio_se"],
                "attenuation_percent": (1.0 - change["rate_ratio"]) * 100.0,
                "n_match_rows": int(len(frame)),
                "n_events": int(frame[event_col].sum()),
                "n_players": int(frame[PLAYER_ID_COL].nunique()),
                "covariance_method": "single stacked Poisson model clustered by player",
                "interpretation_note": (
                    "adjusted_over_unadjusted_ratio directly tests coefficient change "
                    "between otherwise matched specifications; it is a specification "
                    "contrast, not a causal mediation effect"
                ),
            }
        )
    return rows


def type_history_recency_attenuation_test(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:  # pragma: no cover - full-data clustered model
    """Directly test type-history coefficient change after matched recency control."""
    specs = [
        ("all eligible rows", False),
        ("exclude rows within 14 days of recorded return", True),
    ]
    event_col = SENSITIVITY_EVENT_COLS["muscle_tendon_only"]
    rows: List[Dict[str, object]] = []
    for restriction, exclude_recent_returns in specs:
        frame = mutually_exclusive_type_binary_frame(
            panel,
            injuries,
            exclude_recent_returns=exclude_recent_returns,
        )
        frame = add_symmetric_type_recency(frame, injuries)
        unadjusted = frame.copy()
        unadjusted["spec_unadjusted"] = 1.0
        unadjusted["spec_adjusted"] = 0.0
        adjusted = frame.copy()
        adjusted["spec_unadjusted"] = 0.0
        adjusted["spec_adjusted"] = 1.0
        stacked = pd.concat([unadjusted, adjusted], ignore_index=True)
        spline_term = spline_basis_expression(
            float(frame["all_minutes_last_7d"].max())
        )
        result = smf.glm(
            formula=recency_attenuation_stacked_formula(event_col, spline_term),
            data=stacked,
            family=sm.families.Poisson(),
            offset=stacked["log_minutes_played"],
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": stacked[PLAYER_ID_COL]},
        )
        rows.extend(
            recency_attenuation_contrast_rows(
                result,
                restriction,
                frame,
                event_col,
            )
        )
    return pd.DataFrame(rows)


def type_history_multiplicity_family(
    direct: pd.DataFrame,
    binary: pd.DataFrame,
    frequency: pd.DataFrame,
    linearity: pd.DataFrame,
    formal_linearity: pd.DataFrame,
    attenuation: pd.DataFrame,
) -> pd.DataFrame:
    """Collect every formal type-history p-value into one declared family."""
    rows: List[Dict[str, object]] = []

    def add(
        source_file: str,
        source_row: str,
        test_id: str,
        test_component: str,
        raw_p: object,
    ) -> None:
        value = pd.to_numeric(pd.Series([raw_p]), errors="coerce").iloc[0]
        if pd.isna(value) or not np.isfinite(value):
            return
        rows.append(
            {
                "test_id": test_id,
                "test_component": test_component,
                "source_file": source_file,
                "source_row": source_row,
                "p_value": float(value),
            }
        )

    for _, row in direct.iterrows():
        restriction = str(row["restriction"])
        add(
            "matchproxy_negative_control_direct_comparison.csv",
            f"restriction={restriction}",
            f"overlapping_direct__{restriction}",
            "overlapping_label_direct_ratio",
            row["direct_ratio_p"],
        )

    binary_tests = [
        ("muscle_high", "muscle_tendon_high_history_p"),
        ("joint_bone_high", "joint_bone_high_history_p"),
        ("direct_high", "direct_ratio_p"),
        ("recency_muscle_high", "recency_adjusted_muscle_tendon_high_history_p"),
        ("recency_joint_bone_high", "recency_adjusted_joint_bone_high_history_p"),
        ("recency_direct_high", "recency_adjusted_direct_ratio_p"),
    ]
    for _, row in binary.iterrows():
        restriction = str(row["restriction"])
        for component, column in binary_tests:
            add(
                "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
                f"restriction={restriction}",
                f"binary__{component}__{restriction}",
                component,
                row[column],
            )

    frequency_tests = [
        ("muscle_continuous", "muscle_tendon_history_p"),
        ("joint_bone_continuous", "joint_bone_history_p"),
        ("direct_continuous", "direct_ratio_p"),
        ("recency_muscle_continuous", "recency_adjusted_muscle_tendon_history_p"),
        ("recency_joint_bone_continuous", "recency_adjusted_joint_bone_history_p"),
        ("recency_direct_continuous", "recency_adjusted_direct_ratio_p"),
    ]
    for _, row in frequency.iterrows():
        restriction = str(row["restriction"])
        for component, column in frequency_tests:
            add(
                "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
                f"restriction={restriction}",
                f"frequency__{component}__{restriction}",
                component,
                row[column],
            )

    for _, row in linearity.iterrows():
        restriction = str(row["restriction"])
        scope = str(row["distribution_scope"])
        comparison = str(row["comparison"])
        history = str(row["history_variable"])
        add(
            "matchproxy_negative_control_type_frequency_linearity_check.csv",
            (
                f"restriction={restriction}; distribution_scope={scope}; "
                f"comparison={comparison}; history_variable={history}"
            ),
            f"linearity__{restriction}__{scope}__{comparison}__{history}",
            "binary_vs_log_linear_shape",
            row["observed_divided_by_predicted_p"],
        )

    for _, row in formal_linearity.iterrows():
        restriction = str(row["restriction"])
        recency = str(row["recency_adjustment"])
        comparison = str(row["comparison"])
        history = str(row["history_variable"])
        base_id = f"formal__{restriction}__{recency}__{comparison}__{history}"
        add(
            "matchproxy_negative_control_type_frequency_linearity_formal_test.csv",
            (
                f"restriction={restriction}; recency_adjustment={recency}; "
                f"comparison={comparison}; history_variable={history}"
            ),
            f"{base_id}__binary_step",
            "binary_step_beyond_continuous_frequency",
            row["binary_step_p"],
        )
        if comparison == "direct_muscle_tendon_over_joint_bone":
            add(
                "matchproxy_negative_control_type_frequency_linearity_formal_test.csv",
                f"restriction={restriction}; recency_adjustment={recency}; direct continuous",
                f"{base_id}__direct_continuous",
                "direct_continuous_frequency_ratio",
                row["direct_continuous_ratio_p"],
            )
        else:
            add(
                "matchproxy_negative_control_type_frequency_linearity_formal_test.csv",
                (
                    f"restriction={restriction}; recency_adjustment={recency}; "
                    f"history_variable={history}"
                ),
                f"{base_id}__continuous",
                "continuous_frequency_term",
                row["continuous_p"],
            )

    for _, row in attenuation.iterrows():
        restriction = str(row["restriction"])
        contrast = str(row["contrast_id"])
        add(
            "matchproxy_type_history_recency_attenuation.csv",
            f"restriction={restriction}; contrast_id={contrast}",
            f"attenuation__{restriction}__{contrast}",
            "recency_adjusted_over_unadjusted_coefficient",
            row["adjusted_over_unadjusted_p"],
        )

    family = pd.DataFrame(rows)
    if family.empty:
        return family
    if family["test_id"].duplicated().any():
        duplicates = family.loc[family["test_id"].duplicated(), "test_id"].tolist()
        raise ValueError(f"Duplicate type-history test IDs: {duplicates}")
    family["p_holm_type_history_family"] = multipletests(
        family["p_value"], method="holm"
    )[1]
    family["p_bh_type_history_family"] = multipletests(
        family["p_value"], method="fdr_bh"
    )[1]
    family["reject_holm_type_history_0_05"] = family[
        "p_holm_type_history_family"
    ].lt(0.05)
    family["reject_bh_type_history_0_05"] = family[
        "p_bh_type_history_family"
    ].lt(0.05)
    family["family_size"] = int(len(family))
    family["family_definition"] = (
        "all finite formal p-values from overlapping-label, mutually exclusive "
        "binary, mutually exclusive continuous, binary-versus-log-linear, formal "
        "threshold, and stacked recency-attenuation type-history specifications"
    )
    return family.sort_values(["p_value", "test_id"]).reset_index(drop=True)


def mutually_exclusive_type_frequency_distribution_context(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    exclusive_results: pd.DataFrame,
) -> pd.DataFrame:  # pragma: no cover
    """Translate continuous exclusive type-history slopes onto Q3 and IQR scales."""
    enriched = (
        add_mutually_exclusive_type_frequency_history(panel, injuries)
        if any(col not in panel.columns for col in MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS)
        else panel.copy()
    )
    latest = latest_eligible_player_history_snapshot(
        enriched,
        MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS,
    )
    specs = [
        ("all eligible rows", False),
        ("exclude rows within 14 days of recorded return", True),
    ]
    variable_specs = [
        (
            "muscle_tendon",
            "prior muscle/tendon reports",
            MUSCLE_TENDON_HISTORY_RATE_COL,
            "muscle_tendon_history_log_rr_per_10000min",
            "muscle_tendon_history_log_rr_se",
            "muscle_tendon_history_irr_per_10000min",
        ),
        (
            "joint_ligament_or_bone_fracture",
            "prior joint/ligament or bone/fracture reports",
            JOINT_BONE_HISTORY_RATE_COL,
            "joint_bone_history_log_rr_per_10000min",
            "joint_bone_history_log_rr_se",
            "joint_bone_history_irr_per_10000min",
        ),
    ]
    thresholds = mutually_exclusive_type_history_thresholds(enriched, injuries)
    q3_threshold_by_variable = {
        "muscle_tendon": thresholds["muscle_tendon_q3"],
        "joint_ligament_or_bone_fracture": thresholds["joint_bone_q3"],
    }
    rows: List[Dict[str, object]] = []
    for restriction, exclude_recent_returns in specs:
        model_row = exclusive_results[
            exclusive_results["restriction"].astype(str).eq(restriction)
        ].iloc[0]
        analytic = mutually_exclusive_type_frequency_frame(
            panel,
            injuries,
            exclude_recent_returns=exclude_recent_returns,
        )
        anchor = analytic[
            pd.to_numeric(analytic["all_minutes_last_7d"], errors="coerce").eq(0.0)
        ]
        for scope, data in [
            ("latest_eligible_player_snapshot", latest),
            ("analytic_match_rows", analytic),
            ("anchor_0_match_rows", anchor),
        ]:
            scope_rows: List[Dict[str, object]] = []
            scope_stats: Dict[str, Dict[str, float]] = {}
            for variable_id, label, col, log_col, se_col, irr_col in variable_specs:
                stats = distribution_statistics(data[col])
                binary_threshold = q3_threshold_by_variable[variable_id]
                group_stats = q3_group_mean_statistics(data[col], binary_threshold)
                q3_interval = scaled_log_rate_ratio_interval(
                    float(model_row[log_col]),
                    float(model_row[se_col]),
                    stats["q3"],
                )
                iqr_interval = scaled_log_rate_ratio_interval(
                    float(model_row[log_col]),
                    float(model_row[se_col]),
                    stats["iqr"],
                )
                scope_stats[variable_id] = {
                    "q3": stats["q3"],
                    "per_q3_irr": q3_interval["irr"],
                    "per_q3_ci_low": q3_interval["ci_low"],
                    "per_q3_ci_high": q3_interval["ci_high"],
                }
                scope_rows.append(
                    {
                        "restriction": restriction,
                        "distribution_scope": scope,
                        "history_variable": variable_id,
                        "history_label": label,
                        "n": stats["n"],
                        "nonzero_n": stats["nonzero_n"],
                        "median_per_10000min": stats["median"],
                        "q1_per_10000min": stats["q1"],
                        "q3_per_10000min": stats["q3"],
                        "iqr_per_10000min": stats["iqr"],
                        "mean_per_10000min": stats["mean"],
                        "maximum_per_10000min": stats["maximum"],
                        "skewness": stats["skewness"],
                        "binary_q3_threshold_per_10000min": binary_threshold,
                        "below_q3_rows": group_stats["below_q3_rows"],
                        "above_q3_rows": group_stats["above_q3_rows"],
                        "below_q3_mean_per_10000min": group_stats["below_q3_mean"],
                        "above_q3_mean_per_10000min": group_stats["above_q3_mean"],
                        "above_minus_below_q3_mean_gap_per_10000min": group_stats[
                            "above_minus_below_q3_mean_gap"
                        ],
                        "unit_irr": float(model_row[irr_col]),
                        "per_q3_irr": q3_interval["irr"],
                        "per_q3_ci_low": q3_interval["ci_low"],
                        "per_q3_ci_high": q3_interval["ci_high"],
                        "per_iqr_irr": iqr_interval["irr"],
                        "per_iqr_ci_low": iqr_interval["ci_low"],
                        "per_iqr_ci_high": iqr_interval["ci_high"],
                        "interpretation_note": (
                            "continuous-model slope translated from one prior report "
                            "per 10,000 previous club minutes to the observed Q3 and "
                            "IQR scale for that history variable"
                        ),
                    }
                )
            q3_ratio_interval = scaled_log_rate_ratio_difference_interval(
                float(model_row["muscle_tendon_history_log_rr_per_10000min"]),
                float(model_row["muscle_tendon_history_log_rr_se"]),
                scope_stats["muscle_tendon"]["q3"],
                float(model_row["joint_bone_history_log_rr_per_10000min"]),
                float(model_row["joint_bone_history_log_rr_se"]),
                scope_stats["joint_ligament_or_bone_fracture"]["q3"],
                float(model_row["direct_ratio_log_rr_se"]),
            )
            for row in scope_rows:
                row["per_q3_direct_ratio_muscle_over_joint_bone"] = q3_ratio_interval[
                    "irr"
                ]
                row["per_q3_direct_ratio_ci_low"] = q3_ratio_interval["ci_low"]
                row["per_q3_direct_ratio_ci_high"] = q3_ratio_interval["ci_high"]
                row["per_q3_direct_ratio_p"] = q3_ratio_interval["p_value"]
                rows.append(row)
    return pd.DataFrame(rows)


def manuscript_numeric_reconciliation(
    results_dir: Path,
) -> pd.DataFrame:  # pragma: no cover
    """Return a traceable ledger for quantitative manuscript claims."""
    results_dir = Path(results_dir)
    clinical = pd.read_csv(results_dir / "clinical_match_hour_rates.csv")
    proxy = pd.read_csv(results_dir / "matchproxy_proxy_classification_publication.csv")
    direct = pd.read_csv(results_dir / "matchproxy_negative_control_direct_comparison.csv")
    exclusive = pd.read_csv(
        results_dir / "matchproxy_negative_control_mutually_exclusive_type_frequency.csv"
    )
    exclusive_binary = pd.read_csv(
        results_dir / "matchproxy_negative_control_mutually_exclusive_type_binary.csv"
    )
    frequency_context = pd.read_csv(
        results_dir / "matchproxy_negative_control_type_frequency_distribution.csv"
    )
    linearity = pd.read_csv(
        results_dir / "matchproxy_negative_control_type_frequency_linearity_check.csv"
    )
    formal_linearity = pd.read_csv(
        results_dir
        / "matchproxy_negative_control_type_frequency_linearity_formal_test.csv"
    )
    recency_attenuation = pd.read_csv(
        results_dir / "matchproxy_type_history_recency_attenuation.csv"
    )
    type_history_family = pd.read_csv(
        results_dir / "matchproxy_type_history_multiplicity_family.csv"
    )
    denominator = pd.read_csv(results_dir / "matchproxy_same_day_denominator_audit.csv")
    denominator_models = pd.read_csv(
        results_dir / "matchproxy_denominator_sensitivity_summary.csv"
    )
    multiplicity = pd.read_csv(
        results_dir / "matchproxy_effect_modification_multiplicity_family_summary.csv"
    )
    nominal_path = results_dir / "matchproxy_nominal_exposure_response_signals.csv"
    nominal_signals = (
        pd.read_csv(nominal_path)
        if nominal_path.exists()
        else pd.DataFrame()
    )
    shape = pd.read_csv(results_dir / "matchproxy_spline_shape_sensitivity.csv")

    rows: List[Dict[str, object]] = []

    def add(
        claim_id: str,
        value: object,
        source_file: str,
        source_filter: str,
        source_column: str,
        practical_meaning: str,
    ) -> None:
        rows.append(
            {
                "claim_id": claim_id,
                "value": value,
                "source_file": source_file,
                "source_filter": source_filter,
                "source_column": source_column,
                "practical_meaning": practical_meaning,
            }
        )

    overall = clinical[
        clinical["rate_scope"].eq("same_day_plus_lag1")
        & clinical["group_kind"].eq("overall")
        & clinical["group"].eq("overall")
    ].iloc[0]
    add(
        "overall_match_rows",
        int(overall["match_rows"]),
        "clinical_match_hour_rates.csv",
        "rate_scope=same_day_plus_lag1; group_kind=overall; group=overall",
        "match_rows",
        "size of the match-row incidence denominator",
    )
    add(
        "overall_proxy_events",
        int(overall["events"]),
        "clinical_match_hour_rates.csv",
        "rate_scope=same_day_plus_lag1; group_kind=overall; group=overall",
        "events",
        "number of reported match-associated proxy events",
    )
    add(
        "overall_events_per_1000_match_hours",
        float(overall["events_per_1000_match_hours"]),
        "clinical_match_hour_rates.csv",
        "rate_scope=same_day_plus_lag1; group_kind=overall; group=overall",
        "events_per_1000_match_hours",
        "public proxy incidence per conventional match-exposure unit",
    )

    assigned = proxy[proxy["metric"].eq("assigned_matchproxy_events")].iloc[0]
    add(
        "assigned_matchproxy_events",
        int(assigned["n_events"]),
        "matchproxy_proxy_classification_publication.csv",
        "metric=assigned_matchproxy_events",
        "n_events",
        "proxy events successfully assigned to a match row",
    )

    for restriction in direct["restriction"]:
        row = direct[direct["restriction"].eq(restriction)].iloc[0]
        suffix = "all_rows" if restriction == "all eligible rows" else "no_recent_returns"
        add(
            f"direct_all_type_conditional_rr_{suffix}",
            float(row["all_type_conditional_rr"]),
            "matchproxy_negative_control_direct_comparison.csv",
            f"restriction={restriction}",
            "all_type_conditional_rr",
            "all-type history association conditional on the type-discordant label",
        )
        add(
            f"direct_type_discordant_conditional_rr_{suffix}",
            float(row["negative_control_conditional_rr"]),
            "matchproxy_negative_control_direct_comparison.csv",
            f"restriction={restriction}",
            "negative_control_conditional_rr",
            "type-discordant association conditional on the all-type label",
        )
        add(
            f"direct_ratio_{suffix}",
            float(row["direct_ratio_all_type_over_negative_control"]),
            "matchproxy_negative_control_direct_comparison.csv",
            f"restriction={restriction}",
            "direct_ratio_all_type_over_negative_control",
            "direct same-row comparison of the two overlapping labels",
        )
        add(
            f"direct_ratio_ci_low_{suffix}",
            float(row["direct_ratio_ci_low"]),
            "matchproxy_negative_control_direct_comparison.csv",
            f"restriction={restriction}",
            "direct_ratio_ci_low",
            "lower confidence limit for the direct same-row comparison",
        )
        add(
            f"direct_ratio_ci_high_{suffix}",
            float(row["direct_ratio_ci_high"]),
            "matchproxy_negative_control_direct_comparison.csv",
            f"restriction={restriction}",
            "direct_ratio_ci_high",
            "upper confidence limit for the direct same-row comparison",
        )
        add(
            f"direct_ratio_p_{suffix}",
            float(row["direct_ratio_p"]),
            "matchproxy_negative_control_direct_comparison.csv",
            f"restriction={restriction}",
            "direct_ratio_p",
            "p value for the direct same-row comparison",
        )
        add(
            f"direct_excluded_rows_{suffix}",
            int(row["excluded_recent_return_rows"]),
            "matchproxy_negative_control_direct_comparison.csv",
            f"restriction={restriction}",
            "excluded_recent_return_rows",
            "rows removed by the recorded recent-return exclusion",
        )
        add(
            f"direct_excluded_events_{suffix}",
            int(row["excluded_recent_return_events"]),
            "matchproxy_negative_control_direct_comparison.csv",
            f"restriction={restriction}",
            "excluded_recent_return_events",
            "events removed by the recorded recent-return exclusion",
        )

    for restriction in exclusive["restriction"]:
        row = exclusive[exclusive["restriction"].eq(restriction)].iloc[0]
        suffix = "all_rows" if restriction == "all eligible rows" else "no_recent_returns"
        add(
            f"exclusive_muscle_history_irr_{suffix}",
            float(row["muscle_tendon_history_irr_per_10000min"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "muscle_tendon_history_irr_per_10000min",
            "muscle/tendon prior-history association over and above non-muscle history",
        )
        add(
            f"exclusive_joint_bone_history_irr_{suffix}",
            float(row["joint_bone_history_irr_per_10000min"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "joint_bone_history_irr_per_10000min",
            "non-muscle prior-history association over and above muscle/tendon history",
        )
        add(
            f"exclusive_ratio_{suffix}",
            float(row["direct_ratio_muscle_over_joint_bone"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "direct_ratio_muscle_over_joint_bone",
            "direct comparison of mutually exclusive prior injury-type frequencies",
        )
        add(
            f"exclusive_ratio_ci_low_{suffix}",
            float(row["direct_ratio_ci_low"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "direct_ratio_ci_low",
            "lower confidence limit for the direct frequency comparison",
        )
        add(
            f"exclusive_ratio_ci_high_{suffix}",
            float(row["direct_ratio_ci_high"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "direct_ratio_ci_high",
            "upper confidence limit for the direct frequency comparison",
        )
        add(
            f"exclusive_ratio_p_{suffix}",
            float(row["direct_ratio_p"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "direct_ratio_p",
            "p value for the direct frequency comparison",
        )
        add(
            f"exclusive_recency_adjusted_muscle_history_irr_{suffix}",
            float(row["recency_adjusted_muscle_tendon_history_irr_per_10000min"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "recency_adjusted_muscle_tendon_history_irr_per_10000min",
            "symmetric-recency controlled muscle/tendon prior-history association per prior report per 10,000 previous club minutes",
        )
        add(
            f"exclusive_recency_adjusted_joint_bone_history_irr_{suffix}",
            float(row["recency_adjusted_joint_bone_history_irr_per_10000min"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "recency_adjusted_joint_bone_history_irr_per_10000min",
            "symmetric-recency controlled joint/ligament or bone/fracture prior-history association per prior report per 10,000 previous club minutes",
        )
        add(
            f"exclusive_recency_adjusted_ratio_{suffix}",
            float(row["recency_adjusted_direct_ratio_muscle_over_joint_bone"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "recency_adjusted_direct_ratio_muscle_over_joint_bone",
            "symmetric-recency controlled direct comparison of mutually exclusive prior injury-type frequencies",
        )
        add(
            f"exclusive_recency_adjusted_ratio_ci_low_{suffix}",
            float(row["recency_adjusted_direct_ratio_ci_low"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "recency_adjusted_direct_ratio_ci_low",
            "lower confidence limit for the recency-controlled direct frequency comparison",
        )
        add(
            f"exclusive_recency_adjusted_ratio_ci_high_{suffix}",
            float(row["recency_adjusted_direct_ratio_ci_high"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "recency_adjusted_direct_ratio_ci_high",
            "upper confidence limit for the recency-controlled direct frequency comparison",
        )
        add(
            f"exclusive_recency_adjusted_ratio_p_{suffix}",
            float(row["recency_adjusted_direct_ratio_p"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "recency_adjusted_direct_ratio_p",
            "p value for the recency-controlled direct frequency comparison",
        )
        add(
            f"exclusive_excluded_rows_{suffix}",
            int(row["excluded_recent_return_rows"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "excluded_recent_return_rows",
            "rows removed by the recorded recent-return exclusion in the mutually exclusive model",
        )
        add(
            f"exclusive_excluded_events_{suffix}",
            int(row["excluded_recent_return_events"]),
            "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            f"restriction={restriction}",
            "excluded_recent_return_events",
            "events removed by the recorded recent-return exclusion in the mutually exclusive model",
        )

    for restriction in exclusive_binary["restriction"]:
        row = exclusive_binary[exclusive_binary["restriction"].eq(restriction)].iloc[0]
        suffix = "all_rows" if restriction == "all eligible rows" else "no_recent_returns"
        add(
            f"exclusive_binary_muscle_threshold_{suffix}",
            float(row["muscle_tendon_threshold_per_10000min"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "muscle_tendon_threshold_per_10000min",
            "Q3 threshold used for the binary prior muscle/tendon history label",
        )
        add(
            f"exclusive_binary_joint_bone_threshold_{suffix}",
            float(row["joint_bone_threshold_per_10000min"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "joint_bone_threshold_per_10000min",
            "Q3 threshold used for the binary prior joint/ligament or bone/fracture history label",
        )
        add(
            f"exclusive_binary_muscle_rr_{suffix}",
            float(row["muscle_tendon_high_history_rr"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "muscle_tendon_high_history_rr",
            "binary high muscle/tendon history association conditional on high non-muscle history",
        )
        add(
            f"exclusive_binary_muscle_rr_ci_low_{suffix}",
            float(row["muscle_tendon_high_history_ci_low"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "muscle_tendon_high_history_ci_low",
            "lower confidence limit for the binary high muscle/tendon history association",
        )
        add(
            f"exclusive_binary_muscle_rr_ci_high_{suffix}",
            float(row["muscle_tendon_high_history_ci_high"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "muscle_tendon_high_history_ci_high",
            "upper confidence limit for the binary high muscle/tendon history association",
        )
        add(
            f"exclusive_binary_muscle_rr_p_{suffix}",
            float(row["muscle_tendon_high_history_p"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "muscle_tendon_high_history_p",
            "p value for the binary high muscle/tendon history association",
        )
        add(
            f"exclusive_binary_joint_bone_rr_{suffix}",
            float(row["joint_bone_high_history_rr"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "joint_bone_high_history_rr",
            "binary high non-muscle history association conditional on high muscle/tendon history",
        )
        add(
            f"exclusive_binary_joint_bone_rr_ci_low_{suffix}",
            float(row["joint_bone_high_history_ci_low"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "joint_bone_high_history_ci_low",
            "lower confidence limit for the binary high joint/ligament or bone/fracture history association",
        )
        add(
            f"exclusive_binary_joint_bone_rr_ci_high_{suffix}",
            float(row["joint_bone_high_history_ci_high"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "joint_bone_high_history_ci_high",
            "upper confidence limit for the binary high joint/ligament or bone/fracture history association",
        )
        add(
            f"exclusive_binary_joint_bone_rr_p_{suffix}",
            float(row["joint_bone_high_history_p"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "joint_bone_high_history_p",
            "p value for the binary high joint/ligament or bone/fracture history association",
        )
        add(
            f"exclusive_binary_ratio_{suffix}",
            float(row["direct_ratio_muscle_over_joint_bone"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "direct_ratio_muscle_over_joint_bone",
            "like-for-like binary comparison of mutually exclusive injury-type history labels",
        )
        add(
            f"exclusive_binary_ratio_ci_low_{suffix}",
            float(row["direct_ratio_ci_low"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "direct_ratio_ci_low",
            "lower confidence limit for the binary injury-type history ratio",
        )
        add(
            f"exclusive_binary_ratio_ci_high_{suffix}",
            float(row["direct_ratio_ci_high"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "direct_ratio_ci_high",
            "upper confidence limit for the binary injury-type history ratio",
        )
        add(
            f"exclusive_binary_ratio_p_{suffix}",
            float(row["direct_ratio_p"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "direct_ratio_p",
            "p value for the binary injury-type history ratio",
        )
        add(
            f"exclusive_binary_recency_adjusted_muscle_rr_{suffix}",
            float(row["recency_adjusted_muscle_tendon_high_history_rr"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_muscle_tendon_high_history_rr",
            "symmetric-recency controlled binary high muscle/tendon history association",
        )
        add(
            f"exclusive_binary_recency_adjusted_muscle_rr_ci_low_{suffix}",
            float(row["recency_adjusted_muscle_tendon_high_history_ci_low"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_muscle_tendon_high_history_ci_low",
            "lower confidence limit for the recency-controlled binary high muscle/tendon history association",
        )
        add(
            f"exclusive_binary_recency_adjusted_muscle_rr_ci_high_{suffix}",
            float(row["recency_adjusted_muscle_tendon_high_history_ci_high"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_muscle_tendon_high_history_ci_high",
            "upper confidence limit for the recency-controlled binary high muscle/tendon history association",
        )
        add(
            f"exclusive_binary_recency_adjusted_muscle_rr_p_{suffix}",
            float(row["recency_adjusted_muscle_tendon_high_history_p"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_muscle_tendon_high_history_p",
            "p value for the recency-controlled binary high muscle/tendon history association",
        )
        add(
            f"exclusive_binary_recency_adjusted_joint_bone_rr_{suffix}",
            float(row["recency_adjusted_joint_bone_high_history_rr"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_joint_bone_high_history_rr",
            "symmetric-recency controlled binary high joint/ligament or bone/fracture history association",
        )
        add(
            f"exclusive_binary_recency_adjusted_joint_bone_rr_ci_low_{suffix}",
            float(row["recency_adjusted_joint_bone_high_history_ci_low"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_joint_bone_high_history_ci_low",
            "lower confidence limit for the recency-controlled binary high joint/ligament or bone/fracture history association",
        )
        add(
            f"exclusive_binary_recency_adjusted_joint_bone_rr_ci_high_{suffix}",
            float(row["recency_adjusted_joint_bone_high_history_ci_high"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_joint_bone_high_history_ci_high",
            "upper confidence limit for the recency-controlled binary high joint/ligament or bone/fracture history association",
        )
        add(
            f"exclusive_binary_recency_adjusted_joint_bone_rr_p_{suffix}",
            float(row["recency_adjusted_joint_bone_high_history_p"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_joint_bone_high_history_p",
            "p value for the recency-controlled binary high joint/ligament or bone/fracture history association",
        )
        add(
            f"exclusive_binary_recency_adjusted_ratio_{suffix}",
            float(row["recency_adjusted_direct_ratio_muscle_over_joint_bone"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_direct_ratio_muscle_over_joint_bone",
            "symmetric-recency controlled like-for-like binary injury-type history ratio",
        )
        add(
            f"exclusive_binary_recency_adjusted_ratio_ci_low_{suffix}",
            float(row["recency_adjusted_direct_ratio_ci_low"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_direct_ratio_ci_low",
            "lower confidence limit for the recency-controlled binary injury-type history ratio",
        )
        add(
            f"exclusive_binary_recency_adjusted_ratio_ci_high_{suffix}",
            float(row["recency_adjusted_direct_ratio_ci_high"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_direct_ratio_ci_high",
            "upper confidence limit for the recency-controlled binary injury-type history ratio",
        )
        add(
            f"exclusive_binary_recency_adjusted_ratio_p_{suffix}",
            float(row["recency_adjusted_direct_ratio_p"]),
            "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
            f"restriction={restriction}",
            "recency_adjusted_direct_ratio_p",
            "p value for the recency-controlled binary injury-type history ratio",
        )

    latest_context = frequency_context[
        frequency_context["restriction"].eq("all eligible rows")
        & frequency_context["distribution_scope"].eq("latest_eligible_player_snapshot")
    ]
    for _, row in latest_context.iterrows():
        variable = str(row["history_variable"])
        add(
            f"{variable}_history_latest_q3",
            float(row["q3_per_10000min"]),
            "matchproxy_negative_control_type_frequency_distribution.csv",
            (
                "restriction=all eligible rows; "
                "distribution_scope=latest_eligible_player_snapshot; "
                f"history_variable={variable}"
            ),
            "q3_per_10000min",
            "latest-player Q3 used to translate the continuous per-unit slope",
        )
        add(
            f"{variable}_history_per_q3_irr",
            float(row["per_q3_irr"]),
            "matchproxy_negative_control_type_frequency_distribution.csv",
            (
                "restriction=all eligible rows; "
                "distribution_scope=latest_eligible_player_snapshot; "
                f"history_variable={variable}"
            ),
            "per_q3_irr",
            "continuous-model association translated onto the latest-player Q3 scale",
        )
        add(
            f"{variable}_history_iqr",
            float(row["iqr_per_10000min"]),
            "matchproxy_negative_control_type_frequency_distribution.csv",
            (
                "restriction=all eligible rows; "
                "distribution_scope=latest_eligible_player_snapshot; "
                f"history_variable={variable}"
            ),
            "iqr_per_10000min",
            "latest-player interquartile range used for practical interpretation",
        )
        add(
            f"{variable}_history_mean",
            float(row["mean_per_10000min"]),
            "matchproxy_negative_control_type_frequency_distribution.csv",
            (
                "restriction=all eligible rows; "
                "distribution_scope=latest_eligible_player_snapshot; "
                f"history_variable={variable}"
            ),
            "mean_per_10000min",
            "latest-player mean frequency for the continuous prior injury-type variable",
        )
        add(
            f"{variable}_history_skewness",
            float(row["skewness"]),
            "matchproxy_negative_control_type_frequency_distribution.csv",
            (
                "restriction=all eligible rows; "
                "distribution_scope=latest_eligible_player_snapshot; "
                f"history_variable={variable}"
            ),
            "skewness",
            "right-skew diagnostic for the continuous prior injury-type variable",
        )
        add(
            f"{variable}_history_above_q3_mean",
            float(row["above_q3_mean_per_10000min"]),
            "matchproxy_negative_control_type_frequency_distribution.csv",
            (
                "restriction=all eligible rows; "
                "distribution_scope=latest_eligible_player_snapshot; "
                f"history_variable={variable}"
            ),
            "above_q3_mean_per_10000min",
            "observed mean prior injury-type frequency among above-threshold players",
        )
        add(
            f"{variable}_history_below_q3_mean",
            float(row["below_q3_mean_per_10000min"]),
            "matchproxy_negative_control_type_frequency_distribution.csv",
            (
                "restriction=all eligible rows; "
                "distribution_scope=latest_eligible_player_snapshot; "
                f"history_variable={variable}"
            ),
            "below_q3_mean_per_10000min",
            "observed mean prior injury-type frequency among below-threshold players",
        )
        add(
            f"{variable}_history_above_minus_below_q3_mean_gap",
            float(row["above_minus_below_q3_mean_gap_per_10000min"]),
            "matchproxy_negative_control_type_frequency_distribution.csv",
            (
                "restriction=all eligible rows; "
                "distribution_scope=latest_eligible_player_snapshot; "
                f"history_variable={variable}"
            ),
            "above_minus_below_q3_mean_gap_per_10000min",
            "observed high-versus-low mean gap that explains the binary/continuous scale difference",
        )

    anchor_context = frequency_context[
        frequency_context["restriction"].eq("all eligible rows")
        & frequency_context["distribution_scope"].eq("anchor_0_match_rows")
    ]
    for _, row in anchor_context.iterrows():
        variable = str(row["history_variable"])
        for column, meaning in [
            ("n", "0-minute anchor rows used for the distribution check"),
            ("median_per_10000min", "0-minute anchor median prior injury-type frequency"),
            ("q1_per_10000min", "0-minute anchor first quartile prior injury-type frequency"),
            ("q3_per_10000min", "0-minute anchor third quartile prior injury-type frequency"),
            ("iqr_per_10000min", "0-minute anchor interquartile range for prior injury-type frequency"),
            ("mean_per_10000min", "0-minute anchor mean prior injury-type frequency"),
            ("maximum_per_10000min", "0-minute anchor maximum prior injury-type frequency"),
            ("skewness", "0-minute anchor right-skew diagnostic"),
            ("below_q3_mean_per_10000min", "0-minute anchor below-threshold mean using the binary Q3 threshold"),
            ("above_q3_mean_per_10000min", "0-minute anchor above-threshold mean using the binary Q3 threshold"),
            (
                "above_minus_below_q3_mean_gap_per_10000min",
                "0-minute anchor high-versus-low mean gap using the binary Q3 threshold",
            ),
        ]:
            add(
                f"{variable}_history_anchor0_{column}",
                float(row[column]) if column != "n" else int(row[column]),
                "matchproxy_negative_control_type_frequency_distribution.csv",
                (
                    "restriction=all eligible rows; "
                    "distribution_scope=anchor_0_match_rows; "
                    f"history_variable={variable}"
                ),
                column,
                meaning,
            )

    for restriction in frequency_context["restriction"].drop_duplicates():
        suffix = "all_rows" if restriction == "all eligible rows" else "no_recent_returns"
        row = frequency_context[
            frequency_context["restriction"].eq(restriction)
            & frequency_context["distribution_scope"].eq(
                "latest_eligible_player_snapshot"
            )
            & frequency_context["history_variable"].eq("muscle_tendon")
        ].iloc[0]
        for column, meaning in [
            (
                "per_q3_direct_ratio_muscle_over_joint_bone",
                "direct continuous-model muscle/tendon versus joint/bone ratio after scaling each slope to its latest-player Q3",
            ),
            (
                "per_q3_direct_ratio_ci_low",
                "lower confidence limit for the Q3-scaled direct continuous-model ratio",
            ),
            (
                "per_q3_direct_ratio_ci_high",
                "upper confidence limit for the Q3-scaled direct continuous-model ratio",
            ),
            (
                "per_q3_direct_ratio_p",
                "p value for the Q3-scaled direct continuous-model ratio",
            ),
        ]:
            add(
                f"exclusive_per_q3_direct_ratio_{suffix}_{column}",
                float(row[column]),
                "matchproxy_negative_control_type_frequency_distribution.csv",
                (
                    f"restriction={restriction}; "
                    "distribution_scope=latest_eligible_player_snapshot; "
                    "history_variable=muscle_tendon"
                ),
                column,
                meaning,
            )

    linearity_columns = [
        (
            "observed_mean_gap_per_10000min",
            "observed high-versus-lower mean gap for the binary history comparison",
        ),
        (
            "muscle_tendon_mean_gap_per_10000min",
            "muscle/tendon mean gap used in the direct linearity comparison",
        ),
        (
            "joint_bone_mean_gap_per_10000min",
            "joint/ligament or bone/fracture mean gap used in the direct linearity comparison",
        ),
        (
            "predicted_binary_irr_from_continuous_slope",
            "binary high-history IRR predicted by applying the continuous slope across the observed mean gap",
        ),
        (
            "predicted_binary_ci_low",
            "lower confidence limit for the continuous-slope prediction across the observed mean gap",
        ),
        (
            "predicted_binary_ci_high",
            "upper confidence limit for the continuous-slope prediction across the observed mean gap",
        ),
        (
            "observed_binary_irr",
            "observed binary high-history IRR from the mutually exclusive binary model",
        ),
        (
            "observed_binary_ci_low",
            "lower confidence limit for the observed binary high-history IRR",
        ),
        (
            "observed_binary_ci_high",
            "upper confidence limit for the observed binary high-history IRR",
        ),
        (
            "observed_divided_by_predicted_ratio",
            "observed binary IRR divided by the continuous-slope prediction",
        ),
        (
            "observed_divided_by_predicted_ci_low",
            "lower confidence limit for observed divided by predicted",
        ),
        (
            "observed_divided_by_predicted_ci_high",
            "upper confidence limit for observed divided by predicted",
        ),
        (
            "observed_divided_by_predicted_p",
            "p value for observed divided by predicted",
        ),
        (
            "observed_minus_predicted_percent",
            "percentage by which the observed binary estimate differs from the continuous-slope prediction",
        ),
    ]
    for _, row in linearity.iterrows():
        restriction_suffix = (
            "all_rows"
            if row["restriction"] == "all eligible rows"
            else "no_recent_returns"
        )
        base = (
            f"restriction={row['restriction']}; "
            f"distribution_scope={row['distribution_scope']}; "
            f"history_variable={row['history_variable']}"
        )
        claim_prefix = (
            f"linearity_{restriction_suffix}_{row['distribution_scope']}_"
            f"{row['history_variable']}"
        )
        for column, meaning in linearity_columns:
            add(
                f"{claim_prefix}_{column}",
                float(row[column]),
                "matchproxy_negative_control_type_frequency_linearity_check.csv",
                base,
                column,
                meaning,
            )

    formal_linearity_columns = [
        (
            "continuous_rr_adjusted_per_10000min",
            "same-model continuous prior injury-type frequency IRR per prior report per 10,000 previous club minutes",
        ),
        (
            "continuous_ci_low",
            "lower confidence limit for the same-model continuous prior injury-type frequency IRR",
        ),
        (
            "continuous_ci_high",
            "upper confidence limit for the same-model continuous prior injury-type frequency IRR",
        ),
        (
            "continuous_p",
            "p value for the same-model continuous prior injury-type frequency term",
        ),
        (
            "continuous_log_rr_se",
            "cluster-robust log standard error for the same-model continuous prior injury-type frequency term",
        ),
        (
            "binary_step_rr_above_linear_frequency",
            "above-Q3 indicator IRR after the same model already contains continuous prior injury-type frequency",
        ),
        (
            "binary_step_ci_low",
            "lower confidence limit for the formal above-linear-frequency indicator test",
        ),
        (
            "binary_step_ci_high",
            "upper confidence limit for the formal above-linear-frequency indicator test",
        ),
        (
            "binary_step_p",
            "p value for the formal above-linear-frequency indicator test",
        ),
        (
            "binary_step_log_rr_se",
            "cluster-robust log standard error for the formal above-linear-frequency indicator test",
        ),
        (
            "direct_continuous_ratio_muscle_over_joint_bone",
            "same-model continuous muscle/tendon frequency slope divided by the joint/bone slope",
        ),
        (
            "direct_continuous_ratio_ci_low",
            "lower confidence limit for the direct continuous slope ratio",
        ),
        (
            "direct_continuous_ratio_ci_high",
            "upper confidence limit for the direct continuous slope ratio",
        ),
        (
            "direct_continuous_ratio_p",
            "p value for the direct continuous slope ratio",
        ),
        (
            "rows_with_prior_muscle_tendon_report",
            "rows with a strictly prior muscle/tendon report available for recency adjustment",
        ),
        (
            "rows_without_prior_muscle_tendon_report",
            "rows coded with no strictly prior muscle/tendon report in the missing-indicator recency construction",
        ),
        (
            "median_days_since_last_prior_muscle_tendon_report",
            "median days since the most recent strictly prior muscle/tendon report among rows with one",
        ),
        (
            "rows_with_prior_joint_bone_report",
            "rows with a strictly prior joint/ligament or bone/fracture report available for recency adjustment",
        ),
        (
            "rows_without_prior_joint_bone_report",
            "rows coded with no strictly prior joint/ligament or bone/fracture report in the missing-indicator recency construction",
        ),
        (
            "median_days_since_last_prior_joint_bone_report",
            "median days since the most recent strictly prior joint/ligament or bone/fracture report among rows with one",
        ),
        (
            "muscle_tendon_frequency_log_recency_corr_all_rows",
            "Pearson correlation between prior muscle/tendon frequency and log recency under the missing-indicator coding",
        ),
        (
            "muscle_tendon_frequency_log_recency_corr_prior_rows",
            "Pearson correlation between prior muscle/tendon frequency and log recency among rows with a prior muscle/tendon report",
        ),
        (
            "joint_bone_frequency_log_recency_corr_all_rows",
            "Pearson correlation between prior joint/ligament or bone/fracture frequency and log recency under the missing-indicator coding",
        ),
        (
            "joint_bone_frequency_log_recency_corr_prior_rows",
            "Pearson correlation between prior joint/ligament or bone/fracture frequency and log recency among rows with a prior joint/ligament or bone/fracture report",
        ),
        (
            "muscle_tendon_frequency_vif",
            "variance-inflation factor for continuous muscle/tendon frequency in the exact fitted-model design matrix",
        ),
        (
            "joint_bone_frequency_vif",
            "variance-inflation factor for continuous joint/ligament or bone/fracture frequency in the exact fitted-model design matrix",
        ),
    ]
    for _, row in formal_linearity.iterrows():
        restriction_suffix = (
            "all_rows"
            if row["restriction"] == "all eligible rows"
            else "no_recent_returns"
        )
        recency_suffix = str(row.get("recency_adjustment", "none"))
        base = (
            f"restriction={row['restriction']}; "
            f"recency_adjustment={recency_suffix}; "
            f"comparison={row['comparison']}; "
            f"history_variable={row['history_variable']}"
        )
        claim_prefix = (
            f"formal_linearity_{restriction_suffix}_{recency_suffix}_"
            f"{row['history_variable']}"
        )
        for column, meaning in formal_linearity_columns:
            add(
                f"{claim_prefix}_{column}",
                float(row[column]),
                "matchproxy_negative_control_type_frequency_linearity_formal_test.csv",
                base,
                column,
                meaning,
            )

    attenuation_columns = [
        ("unadjusted_irr", "type-history IRR before matched recency control"),
        ("unadjusted_ci_low", "lower confidence limit before recency control"),
        ("unadjusted_ci_high", "upper confidence limit before recency control"),
        ("recency_adjusted_irr", "type-history IRR after matched recency control"),
        ("recency_adjusted_ci_low", "lower confidence limit after recency control"),
        ("recency_adjusted_ci_high", "upper confidence limit after recency control"),
        (
            "adjusted_over_unadjusted_ratio",
            "direct ratio of the adjusted coefficient to the unadjusted coefficient",
        ),
        (
            "adjusted_over_unadjusted_ci_low",
            "lower confidence limit for direct coefficient change",
        ),
        (
            "adjusted_over_unadjusted_ci_high",
            "upper confidence limit for direct coefficient change",
        ),
        ("adjusted_over_unadjusted_p", "raw p value for direct coefficient change"),
        ("attenuation_percent", "percentage attenuation on the IRR-ratio scale"),
    ]
    for _, row in recency_attenuation.iterrows():
        restriction_suffix = (
            "all_rows"
            if row["restriction"] == "all eligible rows"
            else "no_recent_returns"
        )
        source_filter = (
            f"restriction={row['restriction']}; contrast_id={row['contrast_id']}"
        )
        for column, meaning in attenuation_columns:
            add(
                f"recency_attenuation_{restriction_suffix}_{row['contrast_id']}_{column}",
                float(row[column]),
                "matchproxy_type_history_recency_attenuation.csv",
                source_filter,
                column,
                meaning,
            )

    add(
        "type_history_multiplicity_family_size",
        int(len(type_history_family)),
        "matchproxy_type_history_multiplicity_family.csv",
        "all rows",
        "family_size",
        "number of formal type-history tests adjusted together",
    )
    for contrast_id in ("muscle_tendon_high_step", "direct_high_step"):
        test_id = f"attenuation__all eligible rows__{contrast_id}"
        row = type_history_family[type_history_family["test_id"].eq(test_id)].iloc[0]
        for column, meaning in (
            ("p_holm_type_history_family", "Holm-adjusted p value for coefficient change"),
            ("p_bh_type_history_family", "Benjamini-Hochberg-adjusted p value for coefficient change"),
        ):
            add(
                f"recency_attenuation_all_rows_{contrast_id}_{column}",
                float(row[column]),
                "matchproxy_type_history_multiplicity_family.csv",
                f"test_id={test_id}",
                column,
                meaning,
            )

    add(
        "nominal_exposure_response_signal_count",
        int(len(nominal_signals)),
        "matchproxy_nominal_exposure_response_signals.csv",
        "all unadjusted exposure-response p<0.05 rows",
        "nominal_signal_rank",
        "number of unadjusted exposure-response or interaction signals below 0.05",
    )
    if not nominal_signals.empty:
        nominal_columns = [
            ("estimate", "nominal unadjusted exposure-response estimate"),
            ("ci_low", "lower confidence limit for the nominal estimate"),
            ("ci_high", "upper confidence limit for the nominal estimate"),
            ("p_value", "unadjusted p value for the nominal estimate"),
            (
                "p_holm_across_specifications",
                "Holm-adjusted p value for the nominal estimate",
            ),
            (
                "p_bh_across_specifications",
                "Benjamini-Hochberg-adjusted p value for the nominal estimate",
            ),
        ]
        for _, row in nominal_signals.iterrows():
            rank = int(row["nominal_signal_rank"])
            base = (
                f"rank={rank}; model={row['model']}; "
                f"contrast_id={row['contrast_id']}"
            )
            for column, meaning in nominal_columns:
                add(
                    f"nominal_exposure_response_signal_{rank}_{column}",
                    float(row[column]),
                    "matchproxy_nominal_exposure_response_signals.csv",
                    base,
                    column,
                    meaning,
                )

    same_day = denominator[
        denominator["history_stratum"].eq("all_modelled")
        & denominator["prior_load_band"].eq("all_bands")
        & denominator["row_type"].eq("same_day_proxy_event")
    ].iloc[0]
    no_event = denominator[
        denominator["history_stratum"].eq("all_modelled")
        & denominator["prior_load_band"].eq("all_bands")
        & denominator["row_type"].eq("no_proxy_event")
    ].iloc[0]
    add(
        "same_day_event_mean_minutes",
        float(same_day["mean_minutes"]),
        "matchproxy_same_day_denominator_audit.csv",
        "history_stratum=all_modelled; prior_load_band=all_bands; row_type=same_day_proxy_event",
        "mean_minutes",
        "realised minutes on same-day event rows",
    )
    add(
        "non_event_mean_minutes",
        float(no_event["mean_minutes"]),
        "matchproxy_same_day_denominator_audit.csv",
        "history_stratum=all_modelled; prior_load_band=all_bands; row_type=no_proxy_event",
        "mean_minutes",
        "realised minutes on non-event rows",
    )

    for model in ["denominator_observed_minutes_poisson", "denominator_fixed_90_poisson"]:
        row = denominator_models[denominator_models["model"].eq(model)].iloc[0]
        add(
            f"{model}_dispersion",
            float(row["dispersion"]),
            "matchproxy_denominator_sensitivity_summary.csv",
            f"model={model}",
            "dispersion",
            "Pearson dispersion under the specified denominator",
        )

    add(
        "effect_response_tests",
        int(multiplicity["n_exposure_response_tests"].sum()),
        "matchproxy_effect_modification_multiplicity_family_summary.csv",
        "all rows",
        "n_exposure_response_tests",
        "number of exposure-response and effect-modification tests adjusted",
    )
    add(
        "effect_response_holm_rejections",
        int(multiplicity["exposure_response_holm_rejections_0_05"].sum()),
        "matchproxy_effect_modification_multiplicity_family_summary.csv",
        "all rows",
        "exposure_response_holm_rejections_0_05",
        "adjusted exposure-response tests rejected at 0.05",
    )

    peak_counts = shape.groupby("publication_history_stratum")[
        "max_in_15_45_min_band"
    ].sum()
    for stratum, count in peak_counts.items():
        add(
            f"spline_peak_specifications_{stratum}",
            int(count),
            "matchproxy_spline_shape_sensitivity.csv",
            f"publication_history_stratum={stratum}",
            "max_in_15_45_min_band",
            "number of spline specifications where the fitted maximum sat in 15--45 minutes",
        )

    return pd.DataFrame(rows)


def recent_return_excluded_negative_control_cross_summary(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """Refit the two muscle/tendon history models after excluding recent returns."""
    specs = [
        (
            "muscle_tendon_only_frequency_only_history",
            "fragility_frequency_only",
        ),
        (
            "muscle_tendon_only_non_muscle_frequency_history",
            NON_MUSCLE_HISTORY_GROUP_COL,
        ),
    ]
    rows: List[Dict[str, object]] = []
    event_col = SENSITIVITY_EVENT_COLS["muscle_tendon_only"]
    for model, group_col in specs:
        full_frame = prepare_model_frame(panel, event_col, group_col)
        flagged = add_recent_prior_injury_return_flags(full_frame, injuries)
        recent_return = flagged[
            "returned_from_recorded_injury_within_14d"
        ].astype(bool)
        frame = flagged.loc[~recent_return].copy()
        bundle = run_prediction_bundle(frame, event_col)
        row = summary_row(
            model,
            event_col,
            group_col,
            "calendar_timing; excludes recorded return within 14 days",
            frame,
            bundle,
        )
        row = {
            frequency_only_publication_column(column): value
            for column, value in row.items()
        }
        row["n_match_rows_before_restriction"] = int(len(full_frame))
        row["n_events_before_restriction"] = int(full_frame[event_col].sum())
        row["excluded_recent_return_rows"] = int(recent_return.sum())
        row["excluded_recent_return_events"] = int(
            flagged.loc[recent_return, event_col].sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def recurrent_anchor_comparison_table(sensitivity_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare primary, GEE, and switcher higher/intermediate anchors."""
    required = {"model", "rr_0", "rr_180"}
    missing = required - set(sensitivity_summary.columns)
    if missing:
        raise KeyError(f"sensitivity_summary missing required columns: {sorted(missing)}")
    model_names = [
        "primary_same_day_plus_lag1",
        "recurrent_gee_exchangeable_player",
        "player_fixed_effect_within_switchers",
    ]
    rows = []
    indexed = sensitivity_summary.set_index("model", drop=False)
    primary = indexed.loc["primary_same_day_plus_lag1"] if "primary_same_day_plus_lag1" in indexed.index else None
    for model in model_names:
        if model not in indexed.index:
            continue
        row = indexed.loc[model]
        for burden in [0, 180]:
            estimate = float(row[f"rr_{burden}"])
            primary_estimate = (
                float(primary[f"rr_{burden}"]) if primary is not None else np.nan
            )
            rows.append(
                {
                    "component": "anchor_comparison",
                    "model": model,
                    "burden_minutes": float(burden),
                    "estimate": estimate,
                    "ci_low": float(row.get(f"rr_{burden}_ci_low", np.nan)),
                    "ci_high": float(row.get(f"rr_{burden}_ci_high", np.nan)),
                    "primary_glm_estimate_at_same_anchor": primary_estimate,
                    "estimate_divided_by_primary_glm": (
                        estimate / primary_estimate if primary_estimate > 0 else np.nan
                    ),
                    "interpretation": (
                        "population clustered GLM"
                        if model == "primary_same_day_plus_lag1"
                        else "GEE reweights player clusters"
                        if model == "recurrent_gee_exchangeable_player"
                        else "within-player switcher fixed-effect estimate"
                    ),
                }
            )
    return pd.DataFrame(rows)


def switcher_transition_audit(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:
    """Audit event enrichment before players first enter the higher-history state."""
    required = {PLAYER_ID_COL, "date", "model_group", MATCH_MINUTES_COL, event_col}
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    frame = match_panel[match_panel["model_group"].isin(MODEL_GROUPS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    switch_counts = frame.groupby(PLAYER_ID_COL)["model_group"].nunique()
    switcher_ids = set(switch_counts[switch_counts >= 2].index)
    first_higher = (
        frame[frame["model_group"].eq("fragile")]
        .groupby(PLAYER_ID_COL)["date"]
        .min()
        .rename("first_higher_history_date")
    )
    frame = frame.merge(first_higher, on=PLAYER_ID_COL, how="left")
    frame["is_switcher"] = frame[PLAYER_ID_COL].isin(switcher_ids)
    frame["transition_state"] = np.select(
        [
            frame["is_switcher"] & frame["date"].lt(frame["first_higher_history_date"]),
            frame["is_switcher"] & frame["date"].ge(frame["first_higher_history_date"]),
            ~frame["is_switcher"] & frame["model_group"].eq("regular"),
            ~frame["is_switcher"] & frame["model_group"].eq("fragile"),
        ],
        [
            "switcher_pre_higher_history",
            "switcher_post_higher_history",
            "non_switcher_intermediate_history",
            "non_switcher_higher_history",
        ],
        default="unclassified",
    )

    rows: List[Dict[str, object]] = []
    for state, subset in frame.groupby("transition_state", sort=False):
        minutes = pd.to_numeric(subset[MATCH_MINUTES_COL], errors="coerce").fillna(0.0)
        events = int(subset[event_col].sum())
        rates = count_rate_intervals(events, float(minutes.sum()))
        rows.append(
            {
                "component": "switcher_transition_state",
                "transition_state": state,
                "n_match_rows": int(len(subset)),
                "n_players": int(subset[PLAYER_ID_COL].nunique()),
                "n_events": events,
                "match_minutes": float(minutes.sum()),
                **rates,
            }
        )

    switcher_summary = frame[frame["is_switcher"]].groupby(PLAYER_ID_COL).agg(
        pre_higher_events=(
            event_col,
            lambda values: int(values[frame.loc[values.index, "transition_state"].eq("switcher_pre_higher_history")].sum()),
        ),
        post_higher_events=(
            event_col,
            lambda values: int(values[frame.loc[values.index, "transition_state"].eq("switcher_post_higher_history")].sum()),
        ),
    )
    if not switcher_summary.empty:
        rows.append(
            {
                "component": "switcher_player_summary",
                "transition_state": "switcher_players",
                "n_players": int(len(switcher_summary)),
                "n_players_with_pre_higher_event": int(
                    switcher_summary["pre_higher_events"].gt(0).sum()
                ),
                "n_players_with_post_higher_event": int(
                    switcher_summary["post_higher_events"].gt(0).sum()
                ),
                "pre_higher_events": int(switcher_summary["pre_higher_events"].sum()),
                "post_higher_events": int(switcher_summary["post_higher_events"].sum()),
            }
        )
    return pd.DataFrame(rows)


def within_between_history_decomposition(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:  # pragma: no cover
    """Fit offset Poisson GLMs separating between-player and within-player history."""
    required = {
        PLAYER_ID_COL,
        "model_group",
        "all_minutes_last_7d",
        "log_minutes_played",
        event_col,
    }
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")
    frame = match_panel[match_panel["model_group"].isin(MODEL_GROUPS)].copy()
    frame["higher_history_state"] = frame["model_group"].eq("fragile").astype(float)
    frame["player_higher_history_match_share"] = frame.groupby(PLAYER_ID_COL)[
        "higher_history_state"
    ].transform("mean")
    frame["within_player_higher_history_deviation"] = (
        frame["higher_history_state"] - frame["player_higher_history_match_share"]
    )
    burden_max = float(frame["all_minutes_last_7d"].max())
    spline = spline_basis_expression(burden_max)
    timing = "week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
    formulas = [
        (
            "pooled_row_history_state",
            f"{event_col} ~ higher_history_state + {spline} + {timing}",
            ["higher_history_state"],
        ),
        (
            "within_between_history_state",
            (
                f"{event_col} ~ player_higher_history_match_share "
                f"+ within_player_higher_history_deviation + {spline} + {timing}"
            ),
            [
                "player_higher_history_match_share",
                "within_player_higher_history_deviation",
            ],
        ),
    ]
    rows: List[Dict[str, object]] = []
    for model_label, formula, terms in formulas:
        res = smf.glm(
            formula=formula,
            data=frame,
            family=sm.families.Poisson(),
            offset=frame["log_minutes_played"],
        ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
        for term in terms:
            coef = float(res.params[term])
            se = float(res.bse[term])
            z = coef / se if se > 0 else 0.0
            rows.append(
                {
                    "component": "within_between_poisson",
                    "model": model_label,
                    "term": term,
                    "estimate": exp(coef),
                    "ci_low": exp(coef - 1.96 * se),
                    "ci_high": exp(coef + 1.96 * se),
                    "p_value": float(erfc(abs(z) / sqrt(2.0))),
                    "n_match_rows": int(len(frame)),
                    "n_players": int(frame[PLAYER_ID_COL].nunique()),
                    "n_events": int(frame[event_col].sum()),
                    "interpretation": (
                        "row-level higher-history association adjusted for burden spline"
                        if term == "higher_history_state"
                        else "between-player contrast by share of higher-history match rows"
                        if term == "player_higher_history_match_share"
                        else "within-player change after centering each player"
                    ),
                }
            )
    return add_p_value_adjustments(pd.DataFrame(rows))


def recurrent_event_decomposition_table(
    match_panel: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:  # pragma: no cover
    """Combine recurrent-event anchor, switcher, and within/between diagnostics."""
    frames = [
        recurrent_anchor_comparison_table(sensitivity_summary),
        switcher_transition_audit(match_panel, event_col),
        within_between_history_decomposition(match_panel, event_col),
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def between_within_publication_summary(decomposition: pd.DataFrame) -> pd.DataFrame:
    """Return the primary between-player, within-player, and non-switcher results."""
    required = {
        "component",
        "model",
        "term",
        "transition_state",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "n_events",
        "match_minutes",
        "events_per_1000_match_hours",
        "events_per_1000_match_hours_ci_low",
        "events_per_1000_match_hours_ci_high",
    }
    missing = required - set(decomposition.columns)
    if missing:
        raise KeyError(f"decomposition missing required columns: {sorted(missing)}")

    model_rows = decomposition[
        decomposition["component"].eq("within_between_poisson")
    ].copy()
    transition_rows = decomposition[
        decomposition["component"].eq("switcher_transition_state")
    ].copy()
    term_labels = [
        (
            "higher_history_row_level",
            "higher_history_state",
            "incidence rate ratio",
        ),
        (
            "between_player_higher_history_share",
            "player_higher_history_match_share",
            "incidence rate ratio",
        ),
        (
            "within_player_higher_history_deviation",
            "within_player_higher_history_deviation",
            "incidence rate ratio",
        ),
    ]
    rows: List[Dict[str, object]] = []
    for finding, term, measure in term_labels:
        subset = model_rows[model_rows["term"].eq(term)]
        if subset.empty:
            raise ValueError(f"Missing decomposition term: {term}")
        row = subset.iloc[0]
        rows.append(
            {
                "finding": finding,
                "effect_measure": measure,
                "estimate": float(row["estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "p_value": float(row["p_value"]),
                "n_events": int(row["n_events"]),
                "match_minutes": np.nan,
            }
        )

    state_labels = [
        (
            "non_switcher_intermediate_history_rate",
            "non_switcher_intermediate_history",
        ),
        (
            "non_switcher_higher_history_rate",
            "non_switcher_higher_history",
        ),
    ]
    states: Dict[str, pd.Series] = {}
    for finding, state in state_labels:
        subset = transition_rows[transition_rows["transition_state"].eq(state)]
        if subset.empty:
            raise ValueError(f"Missing transition state: {state}")
        row = subset.iloc[0]
        states[state] = row
        rows.append(
            {
                "finding": finding,
                "effect_measure": "events per 1,000 match hours",
                "estimate": float(row["events_per_1000_match_hours"]),
                "ci_low": float(row["events_per_1000_match_hours_ci_low"]),
                "ci_high": float(row["events_per_1000_match_hours_ci_high"]),
                "p_value": np.nan,
                "n_events": int(row["n_events"]),
                "match_minutes": float(row["match_minutes"]),
            }
        )

    higher = states["non_switcher_higher_history"]
    intermediate = states["non_switcher_intermediate_history"]
    contrast = count_rate_ratio_interval(
        float(higher["n_events"]),
        float(higher["match_minutes"]),
        float(intermediate["n_events"]),
        float(intermediate["match_minutes"]),
    )
    rows.append(
        {
            "finding": "non_switcher_higher_vs_intermediate",
            "effect_measure": "crude incidence rate ratio",
            "estimate": contrast["estimate"],
            "ci_low": contrast["ci_low"],
            "ci_high": contrast["ci_high"],
            "p_value": contrast["p_value"],
            "n_events": int(higher["n_events"] + intermediate["n_events"]),
            "match_minutes": float(
                higher["match_minutes"] + intermediate["match_minutes"]
            ),
        }
    )
    return pd.DataFrame(rows)


def diagnostic_support_table(
    match_panel: pd.DataFrame,
    event_col: str,
    burdens: Iterable[float] = DIAGNOSTIC_SUPPORT_BURDENS,
) -> pd.DataFrame:
    """Return local support counts for interpretive and tail burden points."""
    support = selected_support_rows(match_panel, event_col, burdens)
    support["support_role"] = np.where(
        support["all_minutes_last_7d"].astype(float).isin(SELECTED_BURDENS),
        "selected_prediction_point",
        "tail_diagnostic_only",
    )
    return support


def run_named_spline_specification(
    panel: pd.DataFrame,
    label: str,
    event_col: str,
    group_col: str = PRIMARY_GROUP_COL,
    controls: str = "",
    controls_label: str = "calendar_timing",
    model_family: str = "poisson",
    denominator: str = "observed_minutes",
    include_exposure_derived_terms: bool = False,
    analysis_role: str = "sensitivity",
    extra_covars: Optional[Dict[str, object]] = None,
    spline_df: Optional[int] = SPLINE_DF,
    spline_basis: str = "bs",
    spline_knots: Optional[Sequence[float]] = None,
) -> tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Prepare, fit, summarise, and label one spline specification."""
    frame = prepare_model_frame(panel, event_col, group_col)
    bundle = run_prediction_bundle(
        frame,
        event_col,
        controls=controls,
        extra_covars=extra_covars,
        model_family=model_family,
        denominator=denominator,
        include_exposure_derived_terms=include_exposure_derived_terms,
        spline_df=spline_df,
        spline_basis=spline_basis,
        spline_knots=spline_knots,
    )
    summary = summary_row(label, event_col, group_col, controls_label, frame, bundle)
    effects = label_effect_modification_rows(
        bundle["effect_modification"],
        label,
        event_col,
        group_col,
        controls_label,
        frame,
        model_family,
        analysis_role,
    )
    return bundle, pd.DataFrame([summary]), effects


def recovery_interval_rate_table(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:
    """Summarise match-proxy incidence by prior-match recovery interval."""
    required = {"model_group", MATCH_MINUTES_COL, event_col}
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")
    if not {
        "recovery_interval_refined",
        "days_since_last_match",
        "recovery_interval_bin",
    }.intersection(match_panel.columns):
        raise KeyError(
            "match_panel missing recovery_interval_refined, days_since_last_match, "
            "or recovery_interval_bin"
        )

    frame = add_refined_recovery_interval(match_panel)
    rows: List[Dict[str, object]] = []
    for group in MODEL_GROUPS:
        group_frame = frame[frame["model_group"] == group]
        for interval in RECOVERY_INTERVAL_ORDER:
            subset = group_frame[group_frame["recovery_interval_refined"].astype(str) == interval]
            minutes = float(subset[MATCH_MINUTES_COL].sum())
            events = int(subset[event_col].sum())
            rates = count_rate_intervals(events, minutes)
            rows.append(
                {
                    "history_stratum": group,
                    "recovery_interval_bin": interval,
                    "match_rows": int(len(subset)),
                    "match_minutes": minutes,
                    "match_hours": minutes / 60.0 if minutes > 0 else np.nan,
                    "events": events,
                    **rates,
                }
            )
    return pd.DataFrame(rows)


def recovery_interval_display_table(rate_table: pd.DataFrame) -> pd.DataFrame:
    """Return compact recovery-category rows for a journal display."""
    required = {
        "history_stratum",
        "recovery_interval_bin",
        "match_rows",
        "match_hours",
        "events",
        "events_per_1000_match_hours",
        "events_per_1000_match_hours_ci_low",
        "events_per_1000_match_hours_ci_high",
    }
    missing = required - set(rate_table.columns)
    if missing:
        raise KeyError(f"rate_table missing required columns: {sorted(missing)}")

    display = rate_table[
        rate_table["history_stratum"].isin(MODEL_GROUPS)
        & rate_table["recovery_interval_bin"].isin(RECOVERY_TREND_ORDER)
    ].copy()
    display["history_stratum"] = display["history_stratum"].map(
        {
            "regular": "intermediate prior-injury-history",
            "fragile": "higher prior-injury-history",
        }
    )
    display["recovery_interval_order"] = display["recovery_interval_bin"].map(
        {interval: idx for idx, interval in enumerate(RECOVERY_TREND_ORDER)}
    )
    display["history_stratum_order"] = display["history_stratum"].map(
        {
            "intermediate prior-injury-history": 0,
            "higher prior-injury-history": 1,
        }
    )
    return display[
        [
            "recovery_interval_bin",
            "recovery_interval_order",
            "history_stratum",
            "history_stratum_order",
            "match_rows",
            "match_hours",
            "events",
            "events_per_1000_match_hours",
            "events_per_1000_match_hours_ci_low",
            "events_per_1000_match_hours_ci_high",
        ]
    ].sort_values(
        ["recovery_interval_order", "history_stratum_order"],
        ignore_index=True,
    )


def recovery_shortness_scores(interval_values: pd.Series) -> pd.Series:
    """Map longer-to-shorter recovery bins to an ordinal trend score."""
    mapping = {
        interval: float(len(RECOVERY_TREND_ORDER) - 1 - idx)
        for idx, interval in enumerate(RECOVERY_TREND_ORDER)
    }
    return interval_values.astype(str).map(mapping)


def recovery_direct_contrast_rows(
    frame: pd.DataFrame,
    event_col: str,
    numerator_interval: str = "0-3 days",
    denominator_interval: str = "6-7 days",
) -> List[Dict[str, object]]:
    """Return aggregate rate-ratio rows for clinically familiar recovery contrasts."""
    rows: List[Dict[str, object]] = []
    for group in MODEL_GROUPS:
        group_frame = frame[frame["model_group"] == group]
        numerator = group_frame[
            group_frame["recovery_interval_refined"].astype(str) == numerator_interval
        ]
        denominator = group_frame[
            group_frame["recovery_interval_refined"].astype(str) == denominator_interval
        ]
        numerator_events = int(numerator[event_col].sum()) if len(numerator) else 0
        denominator_events = int(denominator[event_col].sum()) if len(denominator) else 0
        numerator_minutes = float(numerator[MATCH_MINUTES_COL].sum()) if len(numerator) else 0.0
        denominator_minutes = (
            float(denominator[MATCH_MINUTES_COL].sum()) if len(denominator) else 0.0
        )
        interval = count_rate_ratio_interval(
            numerator_events,
            numerator_minutes,
            denominator_events,
            denominator_minutes,
        )
        rows.append(
            {
                "model": f"{group}_direct_{numerator_interval}_vs_{denominator_interval}",
                "contrast": f"{numerator_interval}_vs_{denominator_interval}",
                "history_stratum": group,
                "n_match_rows": int(len(numerator) + len(denominator)),
                "n_players": int(
                    pd.concat([numerator, denominator], ignore_index=True)[PLAYER_ID_COL].nunique()
                )
                if len(numerator) + len(denominator)
                else 0,
                "n_events": int(numerator_events + denominator_events),
                "effect_measure": "incidence_rate_ratio_direct_recovery_interval",
                "estimate": interval["estimate"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "p_value": interval["p_value"],
                "fit_status": interval["fit_status"],
            }
        )
    return rows


def recovery_interaction_test_rows(frame: pd.DataFrame, event_col: str) -> List[Dict[str, object]]:
    """Fit recovery-interval-by-history interaction tests."""
    rows: List[Dict[str, object]] = []
    common = {
        "history_stratum": "joint",
        "n_match_rows": int(len(frame)),
        "n_players": int(frame[PLAYER_ID_COL].nunique()),
        "n_events": int(frame[event_col].sum()),
    }
    if frame.empty or frame[event_col].sum() <= 0 or frame["model_group"].nunique() < 2:
        for model, contrast, effect_measure in [
            (
                "recovery_shortness_by_history_interaction",
                "shorter_recovery_trend_higher_vs_intermediate",
                "ratio_of_irrs_per_step_shorter_recovery",
            ),
            (
                "recovery_interval_categorical_by_history_interaction",
                "global_recovery_interval_by_history_interaction",
                "chi_square",
            ),
        ]:
            rows.append(
                {
                    "model": model,
                    "contrast": contrast,
                    **common,
                    "effect_measure": effect_measure,
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "fit_status": "not_estimable",
                }
            )
        return rows

    frame = frame.copy()
    frame["model_group"] = pd.Categorical(frame["model_group"], categories=list(MODEL_GROUPS))
    frame["recovery_interval_refined"] = pd.Categorical(
        frame["recovery_interval_refined"],
        categories=RECOVERY_TREND_ORDER,
    )

    linear_res = smf.glm(
        formula=(
            f"{event_col} ~ recovery_shortness_score * "
            "C(model_group, Treatment(reference='regular'))"
        ),
        data=frame,
        family=sm.families.Poisson(),
        offset=frame["log_minutes"],
    ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
    interaction_names = [
        name
        for name in linear_res.params.index
        if "recovery_shortness_score:" in str(name)
        or ":recovery_shortness_score" in str(name)
    ]
    if interaction_names:
        name = interaction_names[0]
        coef = float(linear_res.params[name])
        se = float(linear_res.bse[name])
        z = coef / se if se > 0 else 0.0
        rows.append(
            {
                "model": "recovery_shortness_by_history_interaction",
                "contrast": "shorter_recovery_trend_higher_vs_intermediate",
                **common,
                "effect_measure": "ratio_of_irrs_per_step_shorter_recovery",
                "estimate": exp(coef),
                "ci_low": exp(coef - 1.96 * se),
                "ci_high": exp(coef + 1.96 * se),
                "p_value": float(erfc(abs(z) / sqrt(2.0))),
                "fit_status": "ok",
            }
        )
    else:
        rows.append(
            {
                "model": "recovery_shortness_by_history_interaction",
                "contrast": "shorter_recovery_trend_higher_vs_intermediate",
                **common,
                "effect_measure": "ratio_of_irrs_per_step_shorter_recovery",
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "fit_status": "not_estimable",
            }
        )

    categorical_res = smf.glm(
        formula=(
            f"{event_col} ~ C(recovery_interval_refined, Treatment(reference='6-7 days')) * "
            "C(model_group, Treatment(reference='regular'))"
        ),
        data=frame,
        family=sm.families.Poisson(),
        offset=frame["log_minutes"],
    ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
    names = [str(name) for name in categorical_res.params.index]
    interaction_indices = [
        idx
        for idx, name in enumerate(names)
        if ":" in name
        and "C(recovery_interval_refined" in name
        and "C(model_group" in name
    ]
    if interaction_indices:
        restriction = np.zeros((len(interaction_indices), len(names)), dtype=float)
        for row_idx, column_idx in enumerate(interaction_indices):
            restriction[row_idx, column_idx] = 1.0
        test = categorical_res.wald_test(restriction, scalar=True)
        rows.append(
            {
                "model": "recovery_interval_categorical_by_history_interaction",
                "contrast": "global_recovery_interval_by_history_interaction",
                **common,
                "effect_measure": "chi_square",
                "estimate": float(np.asarray(test.statistic).squeeze()),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": float(np.asarray(test.pvalue).squeeze()),
                "fit_status": "ok",
            }
        )
    else:
        rows.append(
            {
                "model": "recovery_interval_categorical_by_history_interaction",
                "contrast": "global_recovery_interval_by_history_interaction",
                **common,
                "effect_measure": "chi_square",
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "fit_status": "not_estimable",
            }
        )
    return rows


def recovery_interval_trend_tests(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:
    """Fit Poisson trend tests for shorter recovery, with and without history strata."""
    required = {
        "model_group",
        MATCH_MINUTES_COL,
        PLAYER_ID_COL,
        event_col,
    }
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")
    if not {
        "recovery_interval_refined",
        "days_since_last_match",
        "recovery_interval_bin",
    }.intersection(match_panel.columns):
        raise KeyError(
            "match_panel missing recovery_interval_refined, days_since_last_match, "
            "or recovery_interval_bin"
        )

    frame = add_refined_recovery_interval(match_panel)
    frame = frame[frame["model_group"].isin(MODEL_GROUPS)].copy()
    frame["recovery_shortness_score"] = recovery_shortness_scores(
        frame["recovery_interval_refined"]
    )
    frame = frame.dropna(
        subset=["recovery_shortness_score", MATCH_MINUTES_COL, event_col]
    ).copy()
    frame = frame[frame[MATCH_MINUTES_COL].astype(float) > 0].copy()
    if frame.empty:
        raise ValueError("No rows available for recovery-interval trend tests")
    frame["log_minutes"] = np.log(frame[MATCH_MINUTES_COL].astype(float).clip(lower=1.0))

    specs = [
        (
            "overall_adjusted_for_history",
            frame,
            f"{event_col} ~ recovery_shortness_score + C(model_group)",
            "shorter_recovery_trend_adjusted_for_history",
        )
    ]
    for group in MODEL_GROUPS:
        specs.append(
            (
                f"{group}_within_history",
                frame[frame["model_group"] == group].copy(),
                f"{event_col} ~ recovery_shortness_score",
                "shorter_recovery_trend_within_history",
            )
        )

    rows: List[Dict[str, object]] = []
    for label, spec_frame, formula, contrast_label in specs:
        if spec_frame.empty or spec_frame[event_col].sum() <= 0:
            rows.append(
                {
                    "model": label,
                    "contrast": contrast_label,
                    "history_stratum": (
                        "joint" if label == "overall_adjusted_for_history" else label.split("_")[0]
                    ),
                    "n_match_rows": int(len(spec_frame)),
                    "n_players": int(spec_frame[PLAYER_ID_COL].nunique()) if not spec_frame.empty else 0,
                    "n_events": int(spec_frame[event_col].sum()) if not spec_frame.empty else 0,
                    "effect_measure": "incidence_rate_ratio_per_step_shorter_recovery",
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "fit_status": "not_estimable",
                }
            )
            continue

        model = smf.glm(
            formula=formula,
            data=spec_frame,
            family=sm.families.Poisson(),
            offset=spec_frame["log_minutes"],
        )
        res = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": spec_frame[PLAYER_ID_COL]},
        )
        coef = float(res.params["recovery_shortness_score"])
        se = float(res.bse["recovery_shortness_score"])
        z = coef / se if se > 0 else 0.0
        rows.append(
            {
                "model": label,
                "contrast": contrast_label,
                "history_stratum": (
                    "joint" if label == "overall_adjusted_for_history" else label.split("_")[0]
                ),
                "n_match_rows": int(len(spec_frame)),
                "n_players": int(spec_frame[PLAYER_ID_COL].nunique()),
                "n_events": int(spec_frame[event_col].sum()),
                "effect_measure": "incidence_rate_ratio_per_step_shorter_recovery",
                "estimate": exp(coef),
                "ci_low": exp(coef - 1.96 * se),
                "ci_high": exp(coef + 1.96 * se),
                "p_value": float(erfc(abs(z) / sqrt(2.0))),
                "fit_status": "ok",
            }
        )
    rows.extend(recovery_interaction_test_rows(frame, event_col))
    rows.extend(recovery_direct_contrast_rows(frame, event_col))
    return pd.DataFrame(rows)


def recovery_interval_publication_summary(
    trend_tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine all-cause, muscle/tendon, and severe recovery tests for reporting."""
    rows: List[pd.DataFrame] = []
    for outcome_label, table in trend_tables.items():
        if table is None or table.empty:
            continue
        out = table.copy()
        out.insert(0, "outcome_label", outcome_label)
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = add_p_value_adjustments(combined)
    keep = [
        "outcome_label",
        "model",
        "contrast",
        "history_stratum",
        "n_match_rows",
        "n_players",
        "n_events",
        "effect_measure",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "p_value_holm",
        "p_value_fdr_bh",
        "fit_status",
    ]
    return combined[[col for col in keep if col in combined.columns]].copy()


def recovery_model_formula(event_col: str, controls: str = "") -> str:
    """Build the categorical recovery-interval-by-history formula."""
    return (
        f"{event_col} ~ "
        "C(recovery_interval_refined, Treatment(reference='6-7 days')) * "
        "C(model_group, Treatment(reference='regular'))"
        f"{controls}"
    )


def prepare_recovery_model_frame(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
) -> pd.DataFrame:
    """Prepare recovery rows for denominator/link sensitivity models."""
    required = {"model_group", MATCH_MINUTES_COL, PLAYER_ID_COL, event_col}
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")
    frame = add_refined_recovery_interval(match_panel)
    frame = frame[frame["model_group"].isin(MODEL_GROUPS)].copy()
    frame = frame[
        frame["recovery_interval_refined"].astype(str).isin(RECOVERY_TREND_ORDER)
    ].copy()
    frame[MATCH_MINUTES_COL] = pd.to_numeric(
        frame[MATCH_MINUTES_COL], errors="coerce"
    )
    frame = frame.dropna(subset=[MATCH_MINUTES_COL, event_col]).copy()
    frame = frame[frame[MATCH_MINUTES_COL].astype(float) > 0].copy()
    if frame.empty:
        raise ValueError("No rows available for recovery-interval models")
    frame["log_minutes_played"] = np.log(
        frame[MATCH_MINUTES_COL].astype(float).clip(lower=1.0)
    )
    frame["model_group"] = pd.Categorical(
        frame["model_group"],
        categories=list(MODEL_GROUPS),
    )
    frame["recovery_interval_refined"] = pd.Categorical(
        frame["recovery_interval_refined"],
        categories=RECOVERY_TREND_ORDER,
    )
    return frame


def fit_recovery_model(
    frame: pd.DataFrame,
    event_col: str,
    model_family: str,
    denominator: str,
    controls: str = "",
):  # pragma: no cover
    """Fit a recovery-interval model with the same denominator/link options."""
    offset = None
    if denominator == "observed_minutes":
        offset = frame["log_minutes_played"]
    elif denominator == "fixed_90":
        offset = pd.Series(np.log(90.0), index=frame.index)
    elif denominator != "per_match":
        raise ValueError(f"Unknown denominator mode: {denominator}")
    kwargs = {}
    if offset is not None:
        kwargs["offset"] = offset
    model = smf.glm(
        formula=recovery_model_formula(event_col, controls=controls),
        data=frame,
        family=model_family_object(model_family),
        **kwargs,
    )
    return model.fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})


def recovery_prediction_template(
    intervals: Iterable[str],
    group: str,
    extra_covars: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Build neutral prediction rows for recovery-interval contrasts."""
    intervals = [str(interval) for interval in intervals]
    tmp = pd.DataFrame(
        {
            "recovery_interval_refined": pd.Categorical(
                intervals,
                categories=RECOVERY_TREND_ORDER,
            ),
            "model_group": pd.Categorical(
                [group] * len(intervals),
                categories=list(MODEL_GROUPS),
            ),
            "week_phase_sin": 0.0,
            "week_phase_cos": 0.0,
            "halfweek_phase_sin": 0.0,
            "halfweek_phase_cos": 0.0,
        }
    )
    if extra_covars:
        for key, value in extra_covars.items():
            tmp[key] = value
    tmp["log_minutes_played"] = np.log(90.0)
    return tmp


def recovery_global_interaction_test(res) -> Dict[str, float]:
    """Joint Wald test for categorical recovery-interval-by-history terms."""
    names = [str(name) for name in res.params.index]
    interaction_indices = [
        idx
        for idx, name in enumerate(names)
        if ":" in name
        and "C(recovery_interval_refined" in name
        and "C(model_group" in name
    ]
    if not interaction_indices:
        raise ValueError("Model has no recovery-by-history interaction terms")
    restriction = np.zeros((len(interaction_indices), len(names)), dtype=float)
    for row, column in enumerate(interaction_indices):
        restriction[row, column] = 1.0
    test = res.wald_test(restriction, scalar=True)
    return {
        "test_statistic": float(np.asarray(test.statistic).squeeze()),
        "df": float(len(interaction_indices)),
        "p_value": float(np.asarray(test.pvalue).squeeze()),
    }


def recovery_model_contrast_rows(
    res,
    frame: pd.DataFrame,
    event_col: str,
    outcome_label: str,
    model_label: str,
    model_family: str,
    denominator: str,
    controls_label: str,
    extra_covars: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:  # pragma: no cover
    """Return modeled recovery contrasts on the denominator/link scale."""
    design_info = res.model.data.design_info
    params = np.asarray(res.params)
    covariance = np.asarray(res.cov_params())
    common = {
        "outcome_label": outcome_label,
        "event_col": event_col,
        "model": model_label,
        "model_family": model_family,
        "denominator": denominator,
        "controls": controls_label,
        "n_match_rows": int(len(frame)),
        "n_players": int(frame[PLAYER_ID_COL].nunique()),
        "n_events": int(frame[event_col].sum()),
        "effect_measure": effect_measure_for_family(model_family),
        "fit_status": "ok",
    }
    rows: List[Dict[str, object]] = []
    try:
        interaction = recovery_global_interaction_test(res)
        rows.append(
            {
                **common,
                "contrast": "global_recovery_interval_by_history_interaction",
                "history_stratum": "joint",
                "reference": "",
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "test_statistic": interaction["test_statistic"],
                "df": interaction["df"],
                "p_value": interaction["p_value"],
            }
        )
    except ValueError:
        rows.append(
            {
                **common,
                "contrast": "global_recovery_interval_by_history_interaction",
                "history_stratum": "joint",
                "reference": "",
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "test_statistic": np.nan,
                "df": np.nan,
                "p_value": np.nan,
                "fit_status": "not_estimable",
            }
        )

    for interval in RECOVERY_TREND_ORDER:
        fragile = recovery_prediction_template([interval], "fragile", extra_covars)
        regular = recovery_prediction_template([interval], "regular", extra_covars)
        x_fragile = np.asarray(build_design_matrices([design_info], fragile)[0])[0]
        x_regular = np.asarray(build_design_matrices([design_info], regular)[0])[0]
        contrast = delta_ratio_interval(params, covariance, x_fragile, x_regular)
        rows.append(
            {
                **common,
                "contrast": f"higher_vs_intermediate_at_{interval}",
                "history_stratum": "higher_vs_intermediate",
                "reference": interval,
                "estimate": contrast["rate_ratio"],
                "ci_low": contrast["rr_ci_low"],
                "ci_high": contrast["rr_ci_high"],
                "test_statistic": contrast["z_statistic"],
                "df": 1.0,
                "p_value": contrast["p_value"],
            }
        )

    for group, label in [
        ("regular", "intermediate_history"),
        ("fragile", "higher_history"),
    ]:
        numerator = recovery_prediction_template(["0-3 days"], group, extra_covars)
        denominator_row = recovery_prediction_template(["6-7 days"], group, extra_covars)
        x_num = np.asarray(build_design_matrices([design_info], numerator)[0])[0]
        x_den = np.asarray(build_design_matrices([design_info], denominator_row)[0])[0]
        contrast = delta_ratio_interval(params, covariance, x_num, x_den)
        rows.append(
            {
                **common,
                "contrast": "0-3_days_vs_6-7_days",
                "history_stratum": label,
                "reference": "6-7 days",
                "estimate": contrast["rate_ratio"],
                "ci_low": contrast["rr_ci_low"],
                "ci_high": contrast["rr_ci_high"],
                "test_statistic": contrast["z_statistic"],
                "df": 1.0,
                "p_value": contrast["p_value"],
            }
        )
    return rows


def recovery_interval_model_summary(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
    outcome_label: str = "same_day_plus_lag1",
    controls: str = (
        " + week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
    ),
    controls_label: str = "calendar_timing",
) -> pd.DataFrame:  # pragma: no cover
    """Run denominator/link checks for recovery-interval contrasts."""
    rows: List[Dict[str, object]] = []
    try:
        frame = prepare_recovery_model_frame(match_panel, event_col)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "outcome_label": outcome_label,
                    "event_col": event_col,
                    "model": "all_recovery_models",
                    "fit_status": "failed",
                    "fit_error": repr(exc),
                }
            ]
        )
    for model_label, model_family, denominator in RECOVERY_MODEL_SPECS:
        common_error = {
            "outcome_label": outcome_label,
            "event_col": event_col,
            "model": model_label,
            "model_family": model_family,
            "denominator": denominator,
            "controls": controls_label,
            "n_match_rows": int(len(frame)),
            "n_players": int(frame[PLAYER_ID_COL].nunique()),
            "n_events": int(frame[event_col].sum()),
            "effect_measure": effect_measure_for_family(model_family),
        }
        if frame[event_col].sum() <= 0:
            rows.append(
                {
                    **common_error,
                    "contrast": "all_recovery_contrasts",
                    "history_stratum": "joint",
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "fit_status": "not_estimable",
                }
            )
            continue
        try:
            res = fit_recovery_model(
                frame,
                event_col,
                model_family,
                denominator,
                controls=controls,
            )
            rows.extend(
                recovery_model_contrast_rows(
                    res,
                    frame,
                    event_col,
                    outcome_label,
                    model_label,
                    model_family,
                    denominator,
                    controls_label,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    **common_error,
                    "contrast": "all_recovery_contrasts",
                    "history_stratum": "joint",
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "fit_status": "failed",
                    "fit_error": repr(exc),
                }
            )
    return pd.DataFrame(rows)


def same_day_denominator_audit(
    match_panel: pd.DataFrame,
    event_col: str = PRIMARY_EVENT_COL,
    same_day_col: str = "injury_event_matchproxy_same_day",
    lag1_col: str = "injury_event_matchproxy_lag1",
) -> pd.DataFrame:
    """
    Summarise realised match minutes by proxy-event type.

    Same-day injuries can shorten observed minutes and therefore shrink their
    own Poisson offset. This audit makes that structural issue visible by
    comparing same-day proxy rows with lag-1 proxy rows and non-event rows,
    within modelled history strata and broad prior-load bands.
    """
    required = {
        "model_group",
        "all_minutes_last_7d",
        MATCH_MINUTES_COL,
        event_col,
        same_day_col,
        lag1_col,
    }
    missing = required - set(match_panel.columns)
    if missing:
        raise KeyError(f"match_panel missing required columns: {sorted(missing)}")

    out = match_panel.copy()
    out["prior_load_band"] = pd.cut(
        out["all_minutes_last_7d"].astype(float),
        bins=[-0.1, 0.0, 90.0, 180.0, np.inf],
        labels=["0 min", "1-90 min", "91-180 min", ">180 min"],
        include_lowest=True,
        right=True,
    )
    masks = {
        "same_day_proxy_event": out[same_day_col].fillna(0).astype(int).eq(1),
        "lag1_proxy_event": out[lag1_col].fillna(0).astype(int).eq(1),
        "any_proxy_event": out[event_col].fillna(0).astype(int).eq(1),
        "no_proxy_event": out[event_col].fillna(0).astype(int).eq(0),
        "all_match_rows": pd.Series(True, index=out.index),
    }
    group_values = list(MODEL_GROUPS) + ["all_modelled"]
    band_values = ["0 min", "1-90 min", "91-180 min", ">180 min", "all_bands"]

    rows: List[Dict[str, object]] = []
    for group in group_values:
        if group == "all_modelled":
            group_frame = out[out["model_group"].isin(MODEL_GROUPS)].copy()
        else:
            group_frame = out[out["model_group"] == group].copy()
        for band in band_values:
            if band == "all_bands":
                band_frame = group_frame
            else:
                band_frame = group_frame[group_frame["prior_load_band"].astype(str) == band]
            for row_type, mask in masks.items():
                subset = band_frame[mask.reindex(band_frame.index, fill_value=False)]
                minutes = subset[MATCH_MINUTES_COL].astype(float)
                rows.append(
                    {
                        "history_stratum": group,
                        "prior_load_band": band,
                        "row_type": row_type,
                        "match_rows": int(len(subset)),
                        "proxy_events": int(subset[event_col].sum()) if len(subset) else 0,
                        "same_day_events": int(subset[same_day_col].sum()) if len(subset) else 0,
                        "lag1_events": int(subset[lag1_col].sum()) if len(subset) else 0,
                        "mean_minutes": float(minutes.mean()) if len(subset) else np.nan,
                        "median_minutes": float(minutes.median()) if len(subset) else np.nan,
                        "p25_minutes": float(minutes.quantile(0.25)) if len(subset) else np.nan,
                        "p75_minutes": float(minutes.quantile(0.75)) if len(subset) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def temporal_stability_outputs(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # pragma: no cover
    """
    Fit primary per-minute models inside season blocks.

    This is a temporal stability check, not a predictive validation exercise.
    Fragility labels are the canonical day-level labels already present on each
    row, so a player can enter a later block with prior-history status acquired
    in an earlier block.
    """
    labelled = add_temporal_period_labels(panel)
    rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []
    for period in TEMPORAL_PERIODS:
        label = str(period["period"])
        period_panel = labelled[labelled["temporal_period"] == label].copy()
        row_base: Dict[str, object] = {
            "period": label,
            "season_start_min": int(period["season_start_min"]),
            "season_start_max": int(period["season_start_max"]),
            "labels_carried_forward": True,
            "fit_status": "ok",
        }
        try:
            frame = prepare_model_frame(period_panel, PRIMARY_EVENT_COL, PRIMARY_GROUP_COL)
            bundle = run_prediction_bundle(frame, PRIMARY_EVENT_COL)
            period_preds = bundle["predictions"].copy()
            period_preds.insert(0, "period", label)
            period_preds.insert(1, "season_start_min", int(period["season_start_min"]))
            period_preds.insert(2, "season_start_max", int(period["season_start_max"]))
            prediction_frames.append(period_preds)
            row = summary_row(
                f"temporal_{label}",
                PRIMARY_EVENT_COL,
                PRIMARY_GROUP_COL,
                "calendar_timing",
                frame,
                bundle,
            )
            row.update(row_base)
            row["n_players"] = int(frame[PLAYER_ID_COL].nunique())
            for group in MODEL_GROUPS:
                group_frame = frame[frame["model_group"] == group]
                row[f"{group}_match_rows"] = int(len(group_frame))
                row[f"{group}_events"] = int(group_frame[PRIMARY_EVENT_COL].sum())
            rows.append(row)
        except Exception as exc:
            row_base.update(
                {
                    "model": f"temporal_{label}",
                    "event_col": PRIMARY_EVENT_COL,
                    "group_col": PRIMARY_GROUP_COL,
                    "controls": "calendar_timing",
                    "fit_status": "failed",
                    "fit_error": repr(exc),
                    "n_match_rows": int(len(period_panel)),
                    "n_events": int(period_panel.get(PRIMARY_EVENT_COL, pd.Series(dtype=int)).sum()),
                }
            )
            rows.append(row_base)

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(rows), predictions


def temporal_stability_rows(panel: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
    """Return the period-level temporal-stability summary table."""
    summary, _ = temporal_stability_outputs(panel)
    return summary


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    proc_dir = root / "data" / "processed"
    results_dir = proc_dir / "results"
    tm_dir = root / "external_data" / "transfermarkt"
    results_dir.mkdir(exist_ok=True)

    panel_path = proc_dir / "player_day_panel_all_comp.csv"
    print(f"Repo root: {root}")
    print(f"Loading enriched panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    injuries = pd.read_csv(proc_dir / "tm_injuries_clean.csv", low_memory=False)
    injury_episodes = pd.read_csv(
        proc_dir / "tm_injury_episodes.csv",
        low_memory=False,
    )
    lineups_path = tm_dir / "game_lineups.csv"
    if lineups_path.exists():
        lineups = pd.read_csv(
            lineups_path,
            usecols=["date", "player_id", "type"],
            low_memory=False,
        )
    else:
        lineups = pd.DataFrame()

    panel = merge_day_fragility(panel, proc_dir)
    panel = add_matchproxy_outcome_subsets(panel, injury_episodes)
    panel = add_non_muscle_frequency_history_label(panel, injuries)
    panel = add_mutually_exclusive_type_frequency_history(panel, injuries)
    panel = restrict_to_fragility_risk_set(panel)
    panel = restrict_to_available_risk_set(panel)
    panel = add_calendar_sensitivity_flags(panel)
    panel = add_alternative_fragility_labels(panel)
    panel, out_of_time_audit = add_out_of_time_fragility_label(panel)

    required_events = set(SENSITIVITY_EVENT_COLS.values())
    missing = sorted(required_events - set(panel.columns))
    if missing:
        raise RuntimeError(
            f"Panel missing match-proxy sensitivity columns {missing}. "
            "Run 16_build_match_proxy_events.py first."
        )

    add_publication_history_labels(
        crude_daily_history_publication_table(panel),
        "history_stratum",
    ).to_csv(
        results_dir / "matchproxy_crude_daily_history_publication.csv",
        index=False,
    )
    proxy_classification_publication_table(panel).to_csv(
        results_dir / "matchproxy_proxy_classification_publication.csv",
        index=False,
    )
    proxy_event_type_summary(panel).to_csv(
        results_dir / "matchproxy_proxy_event_type_summary.csv",
        index=False,
    )

    print("\nFitting primary match-proxy Poisson spline model ...")
    primary_frame = prepare_model_frame(panel, PRIMARY_EVENT_COL, PRIMARY_GROUP_COL)
    all_history_match_frame = panel[
        panel[PRIMARY_GROUP_COL].isin(["tough", "regular", "fragile"])
        & (pd.to_numeric(panel[MATCH_MINUTES_COL], errors="coerce") > 0)
    ].copy()
    primary_bundle = run_prediction_bundle(primary_frame, PRIMARY_EVENT_COL)
    res = primary_bundle["result"]
    print(res.summary().tables[1])
    print(f"\n[Poisson] Pearson dispersion statistic: {primary_bundle['dispersion']:.3f}")

    res.params.to_csv(results_dir / "poisson_spline_params_matchproxy.csv")

    preds = primary_bundle["predictions"].copy()

    preds = add_publication_history_labels(preds, "fragility_group")
    selected_primary = add_publication_history_labels(
        primary_bundle["selected"], "fragility_group"
    )
    support_primary = add_publication_history_labels(
        primary_bundle["support"], "fragility_group"
    )

    preds.to_csv(results_dir / "poisson_spline_predictions_matchproxy.csv", index=False)
    selected_primary.to_csv(
        results_dir / "poisson_spline_selected_predictions_matchproxy.csv",
        index=False,
    )
    support_primary.to_csv(
        results_dir / "poisson_spline_selected_support_matchproxy.csv",
        index=False,
    )
    add_publication_history_labels(
        diagnostic_support_table(primary_frame, PRIMARY_EVENT_COL),
        "fragility_group",
    ).to_csv(
        results_dir / "poisson_spline_diagnostic_support_matchproxy.csv",
        index=False,
    )
    add_publication_history_labels(
        spline_curve_shape_summary(preds),
        "history_stratum",
    ).to_csv(
        results_dir / "matchproxy_spline_curve_shape_summary.csv",
        index=False,
    )
    shape_sensitivity = spline_shape_sensitivity_table(primary_frame, PRIMARY_EVENT_COL)
    add_publication_history_labels(shape_sensitivity, "history_stratum").to_csv(
        results_dir / "matchproxy_spline_shape_sensitivity.csv",
        index=False,
    )
    shape_contrasts = spline_shape_contrast_sensitivity_table(shape_sensitivity)
    shape_contrasts.to_csv(
        results_dir / "matchproxy_spline_shape_contrast_sensitivity.csv",
        index=False,
    )
    spline_anchor_range_summary(shape_contrasts).to_csv(
        results_dir / "matchproxy_spline_anchor_range_summary.csv",
        index=False,
    )
    add_publication_history_labels(
        selection_band_audit(primary_frame, injuries, PRIMARY_EVENT_COL, lineups),
        "history_stratum",
    ).to_csv(
        results_dir / "matchproxy_selection_band_audit.csv",
        index=False,
    )
    add_publication_history_labels(
        selection_band_joint_proxy_audit(primary_frame, injuries, PRIMARY_EVENT_COL, lineups),
        "history_stratum",
    ).to_csv(
        results_dir / "matchproxy_selection_band_joint_proxy_audit.csv",
        index=False,
    )
    add_publication_history_labels(
        observed_event_support_summary(primary_frame, PRIMARY_EVENT_COL),
        "history_stratum",
    ).to_csv(
        results_dir / "matchproxy_observed_event_support_summary.csv",
        index=False,
    )
    reporting_process_severity_audit(
        all_history_match_frame,
        PRIMARY_EVENT_COL,
        PRIMARY_GROUP_COL,
    ).to_csv(
        results_dir / "matchproxy_reporting_process_severity_audit.csv",
        index=False,
    )
    primary_recovery_rates = recovery_interval_rate_table(
        primary_frame,
        PRIMARY_EVENT_COL,
    )
    add_publication_history_labels(
        primary_recovery_rates,
        "history_stratum",
    ).to_csv(
        results_dir / "matchproxy_recovery_interval_rates.csv",
        index=False,
    )
    recovery_interval_display_table(primary_recovery_rates).to_csv(
        results_dir / "matchproxy_recovery_interval_display.csv",
        index=False,
    )
    recovery_trend_tables: Dict[str, pd.DataFrame] = {
        "same_day_plus_lag1": recovery_interval_trend_tests(
            primary_frame,
            PRIMARY_EVENT_COL,
        )
    }
    add_publication_history_labels(
        recovery_trend_tables["same_day_plus_lag1"],
        "history_stratum",
    ).to_csv(
        results_dir / "matchproxy_recovery_interval_trend_tests.csv",
        index=False,
    )
    recovery_model_tables: List[pd.DataFrame] = []
    for outcome_label, event_col in [
        ("same_day_plus_lag1", PRIMARY_EVENT_COL),
        ("reported_absence_ge28d", SENSITIVITY_EVENT_COLS["reported_absence_ge28d"]),
        ("muscle_tendon_only", SENSITIVITY_EVENT_COLS["muscle_tendon_only"]),
    ]:
        recovery_model_tables.append(
            recovery_interval_model_summary(
                primary_frame,
                event_col,
                outcome_label=outcome_label,
            )
        )
        if outcome_label != "same_day_plus_lag1":
            recovery_trend_tables[outcome_label] = recovery_interval_trend_tests(
                primary_frame,
                event_col,
            )
            add_publication_history_labels(
                recovery_interval_rate_table(primary_frame, event_col),
                "history_stratum",
            ).to_csv(
                results_dir / f"matchproxy_recovery_interval_rates_{outcome_label}.csv",
                index=False,
            )
            add_publication_history_labels(
                recovery_trend_tables[outcome_label],
                "history_stratum",
            ).to_csv(
                results_dir / f"matchproxy_recovery_interval_trend_tests_{outcome_label}.csv",
                index=False,
            )
    add_publication_history_labels(
        recovery_interval_publication_summary(recovery_trend_tables),
        "history_stratum",
    ).to_csv(
        results_dir / "matchproxy_recovery_interval_publication_summary.csv",
        index=False,
    )
    pd.concat(recovery_model_tables, ignore_index=True).to_csv(
        results_dir / "matchproxy_recovery_interval_model_summary.csv",
        index=False,
    )
    add_publication_history_labels(
        same_day_denominator_audit(primary_frame, PRIMARY_EVENT_COL),
        "history_stratum",
    ).to_csv(
        results_dir / "matchproxy_same_day_denominator_audit.csv",
        index=False,
    )
    primary_bundle["ratios"].to_csv(
        results_dir / "poisson_spline_selected_ratios_matchproxy.csv",
        index=False,
    )

    sensitivity_rows = [
        summary_row(
            "primary_same_day_plus_lag1",
            PRIMARY_EVENT_COL,
            PRIMARY_GROUP_COL,
            "calendar_timing",
            primary_frame,
            primary_bundle,
        )
    ]
    effect_tables = [
        label_effect_modification_rows(
            primary_bundle["effect_modification"],
            "primary_same_day_plus_lag1",
            PRIMARY_EVENT_COL,
            PRIMARY_GROUP_COL,
            "calendar_timing",
            primary_frame,
            "poisson",
            "primary",
        )
    ]

    print("\nFitting denominator and link sensitivity models ...")
    denominator_summary_tables: List[pd.DataFrame] = []
    denominator_effect_tables: List[pd.DataFrame] = []
    denominator_specs = [
        (
            "denominator_observed_minutes_poisson",
            "poisson",
            "observed_minutes",
            "primary denominator: observed current-match minutes",
        ),
        (
            "denominator_fixed_90_poisson",
            "poisson",
            "fixed_90",
            "fixed 90-minute match denominator",
        ),
        (
            "denominator_per_match_logit",
            "binomial_logit",
            "per_match",
            "binary match-row logit",
        ),
        (
            "denominator_per_match_cloglog",
            "binomial_cloglog",
            "per_match",
            "binary match-row complementary-log-log",
        ),
    ]
    for label, model_family, denominator, description in denominator_specs:
        try:
            bundle, summary, effects_table = run_named_spline_specification(
                panel,
                label,
                PRIMARY_EVENT_COL,
                model_family=model_family,
                denominator=denominator,
                controls_label="calendar_timing",
                analysis_role="denominator_sensitivity",
            )
            summary["denominator_description"] = description
            denominator_summary_tables.append(summary)
            denominator_effect_tables.append(effects_table)
            print(
                f"  [OK] {label}: rows={int(summary['n_match_rows'].iloc[0])}, "
                f"events={int(summary['n_events'].iloc[0])}"
            )
        except Exception as exc:
            print(f"  [FAILED] {label}: {exc!r}")

    print("\nFitting outcome sensitivity models ...")
    for label, event_col in SENSITIVITY_EVENT_COLS.items():
        if label == "same_day_plus_lag1":
            continue
        try:
            frame = prepare_model_frame(panel, event_col, PRIMARY_GROUP_COL)
            bundle = run_prediction_bundle(frame, event_col)
            sensitivity_rows.append(
                summary_row(label, event_col, PRIMARY_GROUP_COL, "calendar_timing", frame, bundle)
            )
            effect_tables.append(
                label_effect_modification_rows(
                    bundle["effect_modification"],
                    label,
                    event_col,
                    PRIMARY_GROUP_COL,
                    "calendar_timing",
                    frame,
                    "poisson",
                    "outcome_sensitivity",
                )
            )
            print(f"  [OK] {label}: rows={len(frame)}, events={int(frame[event_col].sum())}")
        except Exception as exc:
            print(f"  [FAILED] {label}: {exc!r}")

    print("\nCrossing better-captured outcomes with duration-independent history ...")
    for spec in OUTCOME_HISTORY_CROSS_SPECS:
        label = str(spec["model"])
        event_col = str(spec["event_col"])
        group_col = str(spec["group_col"])
        try:
            frame = prepare_model_frame(panel, event_col, group_col)
            bundle = run_prediction_bundle(frame, event_col)
            sensitivity_rows.append(
                summary_row(
                    label,
                    event_col,
                    group_col,
                    "calendar_timing",
                    frame,
                    bundle,
                )
            )
            effect_tables.append(
                label_effect_modification_rows(
                    bundle["effect_modification"],
                    label,
                    event_col,
                    group_col,
                    "calendar_timing",
                    frame,
                    "poisson",
                    "outcome_history_cross_sensitivity",
                )
            )
            print(
                f"  [OK] {label}: rows={len(frame)}, "
                f"events={int(frame[event_col].sum())}"
            )
        except Exception as exc:
            print(f"  [FAILED] {label}: {exc!r}")

    print("\nFitting calendar and comparator sensitivity models ...")
    restriction_specs = [
        (
            "exclude_prior7_international_break",
            panel[panel["prior7_overlaps_international_break"].astype(int) == 0].copy(),
            "calendar_timing",
            False,
            "calendar_sensitivity",
        ),
        (
            "clean_zero_comparator",
            primary_frame[primary_frame["clean_zero_or_positive_burden"].astype(int) == 1].copy(),
            "calendar_timing",
            False,
            "comparator_sensitivity",
        ),
        (
            "exclude_covid_disrupted_dates",
            panel[panel["covid_disrupted_date"].astype(int) == 0].copy(),
            "calendar_timing",
            False,
            "calendar_sensitivity",
        ),
        (
            "exposure_derived_terms_overadjustment",
            panel,
            "calendar_timing_plus_exposure_derived_terms",
            True,
            "overadjustment_sensitivity",
        ),
    ]
    for label, spec_panel, controls_label, include_exposure_derived, analysis_role in restriction_specs:
        try:
            if "model_group" in spec_panel.columns:
                frame = spec_panel
                bundle = run_prediction_bundle(
                    frame,
                    PRIMARY_EVENT_COL,
                    include_exposure_derived_terms=include_exposure_derived,
                )
            else:
                frame = prepare_model_frame(spec_panel, PRIMARY_EVENT_COL, PRIMARY_GROUP_COL)
                bundle = run_prediction_bundle(
                    frame,
                    PRIMARY_EVENT_COL,
                    include_exposure_derived_terms=include_exposure_derived,
                )
            sensitivity_rows.append(
                summary_row(label, PRIMARY_EVENT_COL, PRIMARY_GROUP_COL, controls_label, frame, bundle)
            )
            effect_tables.append(
                label_effect_modification_rows(
                    bundle["effect_modification"],
                    label,
                    PRIMARY_EVENT_COL,
                    PRIMARY_GROUP_COL,
                    controls_label,
                    frame,
                    "poisson",
                    analysis_role,
                )
            )
            print(f"  [OK] {label}: rows={len(frame)}, events={int(frame[PRIMARY_EVENT_COL].sum())}")
        except Exception as exc:
            print(f"  [FAILED] {label}: {exc!r}")

    print("\nFitting alternative fragility-label sensitivity models ...")
    for group_col in [
        "fragility_count_only",
        "fragility_frequency_only",
        "fragility_severity_only",
        "fragility_prespecified_abs",
        OUT_OF_TIME_GROUP_COL,
    ]:
        try:
            frame = prepare_model_frame(panel, PRIMARY_EVENT_COL, group_col)
            bundle = run_prediction_bundle(frame, PRIMARY_EVENT_COL)
            if group_col == "fragility_prespecified_abs":
                prespecified_labelled_effects = label_effect_modification_rows(
                    bundle["effect_modification"],
                    "label_fragility_prespecified_abs",
                    PRIMARY_EVENT_COL,
                    group_col,
                    "calendar_timing",
                    frame,
                    "poisson",
                    "history_definition_sensitivity",
                )
                add_publication_history_labels(
                    bundle["predictions"], "fragility_group"
                ).to_csv(
                    results_dir / "matchproxy_prespecified_absolute_predictions.csv",
                    index=False,
                )
                add_publication_history_labels(
                    bundle["selected"], "fragility_group"
                ).to_csv(
                    results_dir / "matchproxy_prespecified_absolute_selected_predictions.csv",
                    index=False,
                )
                add_publication_history_labels(
                    bundle["support"], "fragility_group"
                ).to_csv(
                    results_dir / "matchproxy_prespecified_absolute_support.csv",
                    index=False,
                )
                bundle["ratios"].to_csv(
                    results_dir / "matchproxy_prespecified_absolute_selected_ratios.csv",
                    index=False,
                )
                add_publication_history_labels(
                    prespecified_labelled_effects, "history_stratum"
                ).to_csv(
                    results_dir / "matchproxy_prespecified_absolute_effect_modification.csv",
                    index=False,
                )
            sensitivity_rows.append(
                summary_row(
                    f"label_{group_col}",
                    PRIMARY_EVENT_COL,
                    group_col,
                    "calendar_timing",
                    frame,
                    bundle,
                )
            )
            labelled_effects = label_effect_modification_rows(
                bundle["effect_modification"],
                f"label_{group_col}",
                PRIMARY_EVENT_COL,
                group_col,
                "calendar_timing",
                frame,
                "poisson",
                "history_definition_sensitivity",
            )
            effect_tables.append(labelled_effects)
            print(f"  [OK] {group_col}: rows={len(frame)}, events={int(frame[PRIMARY_EVENT_COL].sum())}")
        except Exception as exc:
            print(f"  [FAILED] {group_col}: {exc!r}")

    print("\nFitting selection-control sensitivity model ...")
    try:
        control_frame = add_player_and_club_metadata(primary_frame, tm_dir)
        common_club_season = control_frame["club_season"].mode().iloc[0]
        common_position = control_frame["position_group"].mode().iloc[0]
        median_age = float(control_frame["age_years"].median())
        controls = "+ age_years + C(position_group) + C(club_season)"
        extra_covars = {
            "age_years": median_age,
            "position_group": common_position,
            "club_season": common_club_season,
        }
        bundle = run_prediction_bundle(control_frame, PRIMARY_EVENT_COL, controls, extra_covars)
        sensitivity_rows.append(
            summary_row(
                "selection_controls_age_position_clubseason",
                PRIMARY_EVENT_COL,
                PRIMARY_GROUP_COL,
                "age_position_clubseason",
                control_frame,
                bundle,
            )
        )
        effect_tables.append(
            label_effect_modification_rows(
                bundle["effect_modification"],
                "selection_controls_age_position_clubseason",
                PRIMARY_EVENT_COL,
                PRIMARY_GROUP_COL,
                "age_position_clubseason",
                control_frame,
                "poisson",
                "selection_control_sensitivity",
            )
        )
        print(
            "  [OK] selection controls: "
            f"rows={len(control_frame)}, events={int(control_frame[PRIMARY_EVENT_COL].sum())}"
        )
    except Exception as exc:
        print(f"  [FAILED] selection controls: {exc!r}")

    print("\nFitting selection-control plus prior-history sensitivity model ...")
    try:
        prior_control_frame = add_prior_history_control_columns(control_frame)
        controls = (
            "+ age_years + C(position_group) + C(club_season) "
            "+ log_prior_minutes_played + prior_n_spells "
            "+ log_prior_injuries_per_10000min + log_prior_max_spell_duration_days"
        )
        extra_covars = {
            "age_years": median_age,
            "position_group": common_position,
            "club_season": common_club_season,
        }
        for col in PRIOR_HISTORY_CONTROL_COLS:
            extra_covars[col] = float(prior_control_frame[col].median())
        bundle = run_prediction_bundle(
            prior_control_frame,
            PRIMARY_EVENT_COL,
            controls,
            extra_covars,
        )
        sensitivity_rows.append(
            summary_row(
                "selection_controls_plus_priorhistory",
                PRIMARY_EVENT_COL,
                PRIMARY_GROUP_COL,
                "age_position_clubseason_priorhistory",
                prior_control_frame,
                bundle,
            )
        )
        effect_tables.append(
            label_effect_modification_rows(
                bundle["effect_modification"],
                "selection_controls_plus_priorhistory",
                PRIMARY_EVENT_COL,
                PRIMARY_GROUP_COL,
                "age_position_clubseason_priorhistory",
                prior_control_frame,
                "poisson",
                "selection_control_sensitivity",
            )
        )
        print(
            "  [OK] selection + prior-history controls: "
            f"rows={len(prior_control_frame)}, "
            f"events={int(prior_control_frame[PRIMARY_EVENT_COL].sum())}"
        )
    except Exception as exc:
        print(f"  [FAILED] selection + prior-history controls: {exc!r}")

    print("\nFitting recurrent-event sensitivity models ...")
    try:
        gee_bundle = run_gee_prediction_bundle(primary_frame, PRIMARY_EVENT_COL)
        sensitivity_rows.append(
            summary_row(
                "recurrent_gee_exchangeable_player",
                PRIMARY_EVENT_COL,
                PRIMARY_GROUP_COL,
                "calendar_timing",
                primary_frame,
                gee_bundle,
            )
        )
        effect_tables.append(
            label_effect_modification_rows(
                gee_bundle["effect_modification"],
                "recurrent_gee_exchangeable_player",
                PRIMARY_EVENT_COL,
                PRIMARY_GROUP_COL,
                "calendar_timing",
                primary_frame,
                "poisson",
                "recurrent_event_sensitivity",
                estimator="gee_exchangeable_player",
            )
        )
        print(
            "  [OK] recurrent GEE: "
            f"rows={len(primary_frame)}, events={int(primary_frame[PRIMARY_EVENT_COL].sum())}"
        )
    except Exception as exc:
        print(f"  [FAILED] recurrent GEE: {exc!r}")

    try:
        fixed_frame, reference_player = player_fixed_effect_frame(primary_frame)
        fixed_controls = (
            f"+ C({PLAYER_ID_COL}, Treatment(reference={repr(reference_player)}))"
        )
        fixed_bundle = run_prediction_bundle(
            fixed_frame,
            PRIMARY_EVENT_COL,
            controls=fixed_controls,
            extra_covars={PLAYER_ID_COL: reference_player},
        )
        fixed_bundle["estimator"] = "player_fixed_effect_switchers"
        sensitivity_rows.append(
            summary_row(
                "player_fixed_effect_within_switchers",
                PRIMARY_EVENT_COL,
                PRIMARY_GROUP_COL,
                "player_fixed_effect_switchers",
                fixed_frame,
                fixed_bundle,
            )
        )
        effect_tables.append(
            label_effect_modification_rows(
                fixed_bundle["effect_modification"],
                "player_fixed_effect_within_switchers",
                PRIMARY_EVENT_COL,
                PRIMARY_GROUP_COL,
                "player_fixed_effect_switchers",
                fixed_frame,
                "poisson",
                "recurrent_event_sensitivity",
                estimator="player_fixed_effect_switchers",
            )
        )
        print(
            "  [OK] player fixed effect switchers: "
            f"rows={len(fixed_frame)}, events={int(fixed_frame[PRIMARY_EVENT_COL].sum())}, "
            f"reference_player={reference_player}"
        )
    except Exception as exc:
        print(f"  [FAILED] player fixed effect switchers: {exc!r}")

    sens = pd.DataFrame(sensitivity_rows)
    sens.to_csv(results_dir / "matchproxy_sensitivity_summary.csv", index=False)
    print(f"\nSaved sensitivity summary -> {results_dir / 'matchproxy_sensitivity_summary.csv'}")
    effects = add_specification_multiplicity_adjustments(
        pd.concat(effect_tables, ignore_index=True)
    )
    effects = add_publication_history_labels(effects, "history_stratum")
    cross_models = {str(spec["model"]) for spec in OUTCOME_HISTORY_CROSS_SPECS}
    cross_summary = sens[sens["model"].isin(cross_models)].copy()
    cross_summary = cross_summary.rename(
        columns={
            column: frequency_only_publication_column(column)
            for column in cross_summary.columns
        }
    )
    non_muscle_model = "muscle_tendon_only_non_muscle_frequency_history"
    cross_summary["group_col"] = np.where(
        cross_summary["model"].astype(str).eq(non_muscle_model),
        "prior_non_muscle_injury_frequency_only",
        "prior_injury_frequency_only",
    )
    cross_summary["history_comparison"] = np.where(
        cross_summary["model"].astype(str).eq(non_muscle_model),
        "at_or_above_non_muscle_q3_vs_below_non_muscle_q3",
        "at_or_above_q3_vs_below_q3",
    )
    q3_freq = float(pd.to_numeric(panel["q3_freq"], errors="coerce").dropna().iloc[0])
    q3_non_muscle = float(
        pd.to_numeric(panel[NON_MUSCLE_HISTORY_THRESHOLD_COL], errors="coerce")
        .dropna()
        .iloc[0]
    )
    cross_summary["frequency_threshold_per_10000_prior_minutes"] = np.where(
        cross_summary["model"].astype(str).eq(non_muscle_model),
        q3_non_muscle,
        q3_freq,
    )
    cross_summary["history_signal"] = np.where(
        cross_summary["model"].astype(str).eq(non_muscle_model),
        "specifically classified musculoskeletal non-muscle public injury descriptions",
        "all public injury descriptions",
    )
    cross_summary["minimum_prior_minutes"] = 900.0
    cross_summary = add_cross_summary_multiplicity_columns(cross_summary, effects)
    cross_summary.to_csv(
        results_dir / "matchproxy_outcome_history_cross_summary.csv",
        index=False,
    )
    unrestricted_negative_control = negative_control_magnitude_comparison(
        cross_summary
    )
    unrestricted_negative_control.to_csv(
        results_dir / "matchproxy_negative_control_magnitude_comparison.csv",
        index=False,
    )
    negative_control_anchor_selection_audit(panel, injuries).to_csv(
        results_dir / "matchproxy_negative_control_anchor_selection_audit.csv",
        index=False,
    )
    direct_type_comparison = negative_control_direct_comparison(panel, injuries)
    direct_type_comparison.to_csv(
        results_dir / "matchproxy_negative_control_direct_comparison.csv",
        index=False,
    )
    exclusive_type_frequency = negative_control_mutually_exclusive_type_frequency_comparison(
        panel,
        injuries,
    )
    exclusive_type_frequency.to_csv(
        results_dir
        / "matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
        index=False,
    )
    exclusive_type_binary = negative_control_mutually_exclusive_type_binary_comparison(
        panel,
        injuries,
    )
    exclusive_type_binary.to_csv(
        results_dir / "matchproxy_negative_control_mutually_exclusive_type_binary.csv",
        index=False,
    )
    exclusive_type_distribution = mutually_exclusive_type_frequency_distribution_context(
        panel,
        injuries,
        exclusive_type_frequency,
    )
    exclusive_type_distribution.to_csv(
        results_dir / "matchproxy_negative_control_type_frequency_distribution.csv",
        index=False,
    )
    type_frequency_linearity = negative_control_type_frequency_linearity_check(
        exclusive_type_frequency,
        exclusive_type_binary,
        exclusive_type_distribution,
    )
    type_frequency_linearity.to_csv(
        results_dir / "matchproxy_negative_control_type_frequency_linearity_check.csv",
        index=False,
    )
    formal_type_linearity = negative_control_type_frequency_linearity_formal_test(
        panel,
        injuries,
    )
    formal_type_linearity.to_csv(
        results_dir
        / "matchproxy_negative_control_type_frequency_linearity_formal_test.csv",
        index=False,
    )
    recency_attenuation = type_history_recency_attenuation_test(panel, injuries)
    recency_attenuation.to_csv(
        results_dir / "matchproxy_type_history_recency_attenuation.csv",
        index=False,
    )
    type_history_multiplicity_family(
        direct_type_comparison,
        exclusive_type_binary,
        exclusive_type_frequency,
        type_frequency_linearity,
        formal_type_linearity,
        recency_attenuation,
    ).to_csv(
        results_dir / "matchproxy_type_history_multiplicity_family.csv",
        index=False,
    )
    recent_return_excluded_cross = (
        recent_return_excluded_negative_control_cross_summary(panel, injuries)
    )
    recent_return_excluded_cross.to_csv(
        results_dir
        / "matchproxy_negative_control_recent_return_excluded_model_summary.csv",
        index=False,
    )
    recent_return_excluded_comparison = negative_control_magnitude_comparison(
        recent_return_excluded_cross
    )
    unrestricted_negative_control.insert(0, "restriction", "all eligible rows")
    recent_return_excluded_comparison.insert(
        0,
        "restriction",
        "exclude rows within 14 days of recorded return",
    )
    pd.concat(
        [
            unrestricted_negative_control,
            recent_return_excluded_comparison,
        ],
        ignore_index=True,
    ).to_csv(
        results_dir / "matchproxy_negative_control_recent_return_exclusion.csv",
        index=False,
    )
    cross_summary[cross_summary["model"].astype(str).eq(non_muscle_model)].to_csv(
        results_dir / "matchproxy_type_discordant_history_summary.csv",
        index=False,
    )
    recurrent_decomposition = recurrent_event_decomposition_table(
        primary_frame,
        sens,
        PRIMARY_EVENT_COL,
    )
    recurrent_decomposition.to_csv(
        results_dir / "matchproxy_recurrent_event_decomposition.csv",
        index=False,
    )
    between_within_publication_summary(recurrent_decomposition).to_csv(
        results_dir / "matchproxy_between_within_publication_summary.csv",
        index=False,
    )
    print(
        "Saved recurrent-event decomposition -> "
        f"{results_dir / 'matchproxy_recurrent_event_decomposition.csv'}"
    )

    denominator_summary = pd.DataFrame()
    if denominator_summary_tables:
        denominator_summary = pd.concat(denominator_summary_tables, ignore_index=True)
        denominator_summary.to_csv(
            results_dir / "matchproxy_denominator_sensitivity_summary.csv",
            index=False,
        )
        print(
            "Saved denominator sensitivity summary -> "
            f"{results_dir / 'matchproxy_denominator_sensitivity_summary.csv'}"
        )
    publication_summary = publication_contrast_summary(sens, denominator_summary)
    publication_summary.to_csv(
        results_dir / "matchproxy_publication_contrast_summary.csv",
        index=False,
    )
    print(
        "Saved publication contrast summary -> "
        f"{results_dir / 'matchproxy_publication_contrast_summary.csv'}"
    )
    if denominator_effect_tables:
        denominator_effects = add_specification_multiplicity_adjustments(
            pd.concat(denominator_effect_tables, ignore_index=True)
        )
        denominator_effects = add_publication_history_labels(
            denominator_effects,
            "history_stratum",
        )
        denominator_effects.to_csv(
            results_dir / "matchproxy_denominator_effect_modification_tests.csv",
            index=False,
        )
        print(
            "Saved denominator formal tests -> "
            f"{results_dir / 'matchproxy_denominator_effect_modification_tests.csv'}"
        )

    effects_path = results_dir / "matchproxy_effect_modification_tests.csv"
    effects.to_csv(effects_path, index=False)
    print(f"Saved formal effect-modification tests -> {effects_path}")
    formal_path = results_dir / "matchproxy_formal_model_contrast_tests.csv"
    effects.to_csv(formal_path, index=False)
    print(f"Saved formal model-contrast tests -> {formal_path}")
    nominal_path = results_dir / "matchproxy_nominal_exposure_response_signals.csv"
    nominal_exposure_response_signal_summary(effects).to_csv(
        nominal_path,
        index=False,
    )
    print(f"Saved nominal exposure-response signal audit -> {nominal_path}")
    multiplicity_family_summary(effects).to_csv(
        results_dir / "matchproxy_multiplicity_family_summary.csv",
        index=False,
    )
    effect_mod_only = effects[
        ~effects["contrast_id"].astype(str).str.startswith("higher_vs_intermediate_at_")
    ].copy()
    multiplicity_family_summary(effect_mod_only).to_csv(
        results_dir / "matchproxy_effect_modification_multiplicity_family_summary.csv",
        index=False,
    )
    publication_referee_audit_table(panel, injuries, effects, cross_summary).to_csv(
        results_dir / "matchproxy_publication_referee_audit.csv",
        index=False,
    )
    print(
        "Saved publication referee audit -> "
        f"{results_dir / 'matchproxy_publication_referee_audit.csv'}"
    )
    out_of_time_audit.to_csv(
        results_dir / "matchproxy_out_of_time_threshold_audit.csv",
        index=False,
    )
    print(
        "Saved out-of-time threshold audit -> "
        f"{results_dir / 'matchproxy_out_of_time_threshold_audit.csv'}"
    )

    print("\nFitting temporal stability models ...")
    temporal, temporal_preds = temporal_stability_outputs(panel)
    if not temporal_preds.empty:
        temporal_preds = add_publication_history_labels(temporal_preds, "fragility_group")
    temporal.to_csv(results_dir / "matchproxy_temporal_stability_summary.csv", index=False)
    temporal_preds.to_csv(
        results_dir / "matchproxy_temporal_stability_predictions.csv",
        index=False,
    )
    print(
        "Saved temporal stability summary -> "
        f"{results_dir / 'matchproxy_temporal_stability_summary.csv'}"
    )
    print(
        "Saved temporal stability predictions -> "
        f"{results_dir / 'matchproxy_temporal_stability_predictions.csv'}"
    )
    manuscript_numeric_reconciliation(results_dir).to_csv(
        results_dir / "manuscript_numeric_reconciliation.csv",
        index=False,
    )
    print(
        "Saved manuscript numeric reconciliation -> "
        f"{results_dir / 'manuscript_numeric_reconciliation.csv'}"
    )
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
