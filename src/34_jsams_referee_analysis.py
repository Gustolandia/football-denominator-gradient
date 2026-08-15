#!/usr/bin/env python
"""Build the JSAMS reviewer-requested estimand and sensitivity analyses.

The reference estimand is the probability of a same-day reported event per
recorded appearance. Minute-denominator models are deliberately labelled as
different estimands rather than alternative estimates of the same quantity.
The module also standardises absolute predictions over observed calendar
phases, separates lineup roles, directly compares selection-adjusted effects,
and applies the same model to outcome-quality, functional-form, eligibility,
season, and national-exposure sensitivities. Prior report history remains
continuous; historical categories are descriptive support bands only.

Run after scripts 18 and 33:

    python src/34_jsams_referee_analysis.py

Outputs are written to data/processed/results with the prefix jsams_.
These are reviewer-requested revisions, not prospectively prespecified tests.
"""

from __future__ import annotations

import importlib.util
import sys
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
from scipy.stats import chi2
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


PLAYER_ID_COL = "tm_player_id"
BURDEN_COL = "all_minutes_last_7d"
MINUTES_COL = "all_minutes_played"
HISTORY_COL = "prior_injuries_per_10000min"
HISTORY_MODEL_COL = "history_log_iqr"
SAME_DAY_COL = "injury_event_matchproxy_same_day"
LAG1_COL = "injury_event_matchproxy_lag1"
COMBINED_COL = "injury_event_matchproxy"
SAME_DAY_SEVERE_COL = "same_day_reported_absence_ge28d"
SAME_DAY_MUSCLE_COL = "same_day_muscle_tendon_report"
SPLINE_KNOTS = (45.0, 90.0, 135.0)
DISPLAY_BURDENS = (0.0, 90.0, 180.0, 220.0)
PREDICTION_GRID = tuple(float(value) for value in range(0, 221, 5))
CALENDAR_TERMS = (
    "week_phase_sin",
    "week_phase_cos",
    "halfweek_phase_sin",
    "halfweek_phase_cos",
)
LINEUP_ROLES = ("starting_lineup", "substitute_list")
HISTORY_PUBLICATION_LABELS = {
    "tough": "lower prior-injury-history",
    "regular": "intermediate prior-injury-history",
    "fragile": "higher prior-injury-history",
}
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260806
REFERENCE_MINIMUM_PRIOR_MINUTES = 900.0
ELIGIBILITY_THRESHOLDS = (450.0, 900.0, 1800.0)
REFERENCE_ANCHOR_MINUTES = 180.0
COMPLETE_SEASON_END = pd.Timestamp("2024-06-30")
PANDEMIC_SEASON_STARTS = (2019, 2020)
EXPOSURE_BAND_ORDER = (
    "0",
    "1-45",
    "46-90",
    "91-135",
    "136-180",
    ">180",
)
FUNCTIONAL_FORM_SPECS = (
    "reference_bspline",
    "linear_per_90",
    "fewer_df_bspline",
    "restricted_cubic_df4",
    "fixed_match_bands",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    """Require an explicit schema before fitting or summarising a table."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def load_source_module(filename: str, module_name: str):  # pragma: no cover
    """Load a numerically named pipeline script without changing its filename."""
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    path = src_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing pipeline script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import pipeline script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history_scale(frame: pd.DataFrame) -> dict[str, float]:
    """Return the row-weighted log-history median, IQR and display anchors."""
    _require_columns(frame, [HISTORY_COL], "history frame")
    values = pd.to_numeric(frame[HISTORY_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    logged = np.log1p(values.to_numpy(dtype=float))
    q25, median, q75 = np.quantile(logged, [0.25, 0.50, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(logged))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return {
        "center_log": float(median),
        "scale_log_iqr": scale,
        "low_log": float(q25),
        "median_log": float(median),
        "high_log": float(q75),
        "low_rate": float(np.expm1(q25)),
        "median_rate": float(np.expm1(median)),
        "high_rate": float(np.expm1(q75)),
    }


def apply_history_scale(frame: pd.DataFrame, scaling: Mapping[str, float]) -> pd.DataFrame:
    """Add the continuous prior-history term measured per log-history IQR."""
    _require_columns(frame, [HISTORY_COL], "history frame")
    scale = float(scaling["scale_log_iqr"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("History scale must be positive and finite")
    out = frame.copy()
    values = pd.to_numeric(out[HISTORY_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    out[HISTORY_MODEL_COL] = (
        np.log1p(values) - float(scaling["center_log"])
    ) / scale
    return out


def season_start_year(dates: pd.Series) -> pd.Series:
    """Return the starting year of each July-to-June football season."""
    parsed = pd.to_datetime(dates, errors="coerce")
    return pd.Series(
        np.where(parsed.dt.month.ge(7), parsed.dt.year, parsed.dt.year - 1),
        index=dates.index,
        dtype="Int64",
    )


def add_same_day_quality_outcomes(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    classify_injury_type: Any,
) -> pd.DataFrame:
    """Attach severe-absence and muscle/tendon same-day outcome indicators.

    Quality restrictions use reconciled episode starts on the same player-date.
    Reported absence is not treated as clinically verified time loss, and the
    text classifier is not treated as a medical diagnosis.
    """
    _require_columns(
        panel,
        [PLAYER_ID_COL, "date", SAME_DAY_COL],
        "same-day quality panel",
    )
    _require_columns(
        episodes,
        [PLAYER_ID_COL, "start_date", "duration_days", "injury_desc"],
        "same-day quality episodes",
    )
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    episode_rows = episodes[
        [PLAYER_ID_COL, "start_date", "duration_days", "injury_desc"]
    ].copy()
    episode_rows["start_date"] = pd.to_datetime(
        episode_rows["start_date"], errors="coerce"
    )
    episode_rows["duration_days"] = pd.to_numeric(
        episode_rows["duration_days"], errors="coerce"
    )
    episode_rows["injury_desc"] = (
        episode_rows["injury_desc"].fillna("").astype(str).str.strip()
    )
    episode_rows["episode_public_type"] = episode_rows["injury_desc"].map(
        classify_injury_type
    )
    grouped = (
        episode_rows.dropna(subset=[PLAYER_ID_COL, "start_date"])
        .groupby([PLAYER_ID_COL, "start_date"], as_index=False)
        .agg(
            same_day_reported_absence_days=("duration_days", "max"),
            same_day_episode_count=("duration_days", "size"),
            same_day_any_muscle_tendon=(
                "episode_public_type",
                lambda values: bool(pd.Series(values).eq("muscle/tendon").any()),
            ),
        )
        .rename(columns={"start_date": "date"})
    )
    out = out.merge(
        grouped,
        on=[PLAYER_ID_COL, "date"],
        how="left",
        validate="many_to_one",
    )
    same_day = pd.to_numeric(out[SAME_DAY_COL], errors="coerce").fillna(0).eq(1)
    out[SAME_DAY_SEVERE_COL] = (
        same_day
        & pd.to_numeric(
            out["same_day_reported_absence_days"], errors="coerce"
        ).ge(28.0)
    ).astype(int)
    out[SAME_DAY_MUSCLE_COL] = (
        same_day & out["same_day_any_muscle_tendon"].eq(True)
    ).astype(int)
    out["same_day_quality_metadata_matched"] = (
        same_day & out["same_day_episode_count"].notna()
    ).astype(int)
    return out


def prepare_jsams_frame(
    primary_module: Any,
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    lineups: pd.DataFrame | None,
    transfermarkt_dir: Path | None = None,
    minimum_prior_minutes: float = REFERENCE_MINIMUM_PRIOR_MINUTES,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Prepare one continuous-history appearance cohort at a stated threshold."""
    if not np.isfinite(minimum_prior_minutes) or minimum_prior_minutes < 0.0:
        raise ValueError("minimum_prior_minutes must be finite and non-negative")
    required = [
        PLAYER_ID_COL,
        "date",
        MINUTES_COL,
        "available_for_injury_risk",
        "prior_minutes_played",
        HISTORY_COL,
        BURDEN_COL,
        SAME_DAY_COL,
        LAG1_COL,
        COMBINED_COL,
        *CALENDAR_TERMS,
    ]
    _require_columns(panel, required, "JSAMS source panel")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[MINUTES_COL] = pd.to_numeric(frame[MINUTES_COL], errors="coerce")
    frame["prior_minutes_played"] = pd.to_numeric(
        frame["prior_minutes_played"], errors="coerce"
    )
    for column in (BURDEN_COL, HISTORY_COL, *CALENDAR_TERMS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    complete = frame[["date", BURDEN_COL, HISTORY_COL, *CALENDAR_TERMS]].notna().all(axis=1)
    frame = frame[
        frame[MINUTES_COL].gt(0.0)
        & frame["available_for_injury_risk"].fillna(False).astype(bool)
        & frame["prior_minutes_played"].ge(float(minimum_prior_minutes))
        & frame[BURDEN_COL].ge(0.0)
        & frame[HISTORY_COL].ge(0.0)
        & complete
    ].copy()
    frame = primary_module.add_prior_history_control_columns(frame)
    frame = primary_module.add_recent_prior_injury_return_flags(frame, injuries)
    frame = primary_module.add_lineup_start_status(frame, lineups)
    frame["lineup_role_model"] = np.where(
        frame["lineup_role"].isin(LINEUP_ROLES),
        frame["lineup_role"],
        "lineup_unavailable_or_other",
    )
    frame["returned_from_recorded_injury_within_14d"] = frame[
        "returned_from_recorded_injury_within_14d"
    ].fillna(False).astype(int)
    for event_col in (SAME_DAY_COL, LAG1_COL, COMBINED_COL):
        frame[event_col] = pd.to_numeric(frame[event_col], errors="coerce").fillna(0).astype(int)
    for event_col in (SAME_DAY_SEVERE_COL, SAME_DAY_MUSCLE_COL):
        if event_col in frame:
            frame[event_col] = (
                pd.to_numeric(frame[event_col], errors="coerce").fillna(0).astype(int)
            )
    frame["analysis_minimum_prior_minutes"] = float(minimum_prior_minutes)
    frame["season_start"] = season_start_year(frame["date"])
    if transfermarkt_dir is not None:
        frame = primary_module.add_player_and_club_metadata(frame, transfermarkt_dir)
    scaling = history_scale(frame)
    return apply_history_scale(frame, scaling), scaling


def spline_expression(burden_max: float) -> str:
    """Return the fixed-knot cubic B-spline used in every JSAMS model."""
    if not np.isfinite(burden_max) or burden_max <= max(SPLINE_KNOTS):
        raise ValueError("Burden support must extend beyond the fixed spline knots")
    return (
        f"bs({BURDEN_COL}, knots={SPLINE_KNOTS}, degree=3, "
        f"include_intercept=False, lower_bound=0.0, upper_bound={float(burden_max)})"
    )


def exposure_band(values: pd.Series) -> pd.Categorical:
    """Group recent minutes into fixed, clinically readable match-load bands."""
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
        EXPOSURE_BAND_ORDER,
        default=None,
    )
    return pd.Categorical(labels, categories=EXPOSURE_BAND_ORDER, ordered=True)


def exposure_expression(exposure_spec: str, burden_max: float) -> str:
    """Return one named exposure functional form for symmetric sensitivities."""
    if exposure_spec == "reference_bspline":
        return spline_expression(burden_max)
    if exposure_spec == "linear_per_90":
        return f"I({BURDEN_COL} / 90.0)"
    if exposure_spec == "fewer_df_bspline":
        return (
            f"bs({BURDEN_COL}, df=4, degree=3, include_intercept=False, "
            f"lower_bound=0.0, upper_bound={float(burden_max)})"
        )
    if exposure_spec == "restricted_cubic_df4":
        return f"cr({BURDEN_COL}, df=4, constraints='center')"
    if exposure_spec == "fixed_match_bands":
        return "C(exposure_band_model, Treatment(reference='0'))"
    raise ValueError(f"Unknown exposure specification: {exposure_spec}")


def add_exposure_spec_columns(
    frame: pd.DataFrame, exposure_spec: str
) -> pd.DataFrame:
    """Add derived columns required by a named exposure specification."""
    out = frame.copy()
    if exposure_spec == "fixed_match_bands":
        out["exposure_band_model"] = exposure_band(out[BURDEN_COL])
    return out


def continuous_formula(
    event_col: str,
    burden_max: float,
    extra_controls: str = "",
    exposure_spec: str = "reference_bspline",
) -> str:
    """Build one symmetric exposure-by-continuous-history formula."""
    calendar = " + ".join(CALENDAR_TERMS)
    return (
        f"{event_col} ~ {exposure_expression(exposure_spec, burden_max)} "
        f"* {HISTORY_MODEL_COL} "
        f"+ {calendar}{extra_controls}"
    )


def fit_continuous_model(
    frame: pd.DataFrame,
    event_col: str,
    denominator: str = "per_appearance",
    extra_controls: str = "",
    exposure_spec: str = "reference_bspline",
):
    """Fit a player-clustered GLM for one outcome/denominator definition."""
    required = [PLAYER_ID_COL, BURDEN_COL, HISTORY_MODEL_COL, event_col, *CALENDAR_TERMS]
    if denominator in {"observed_minutes", "fixed_90"}:
        required.append(MINUTES_COL)
    _require_columns(frame, required, "continuous model frame")
    if int(frame[event_col].sum()) <= 0:
        raise ValueError(f"No events available for {event_col}")
    model_frame = add_exposure_spec_columns(frame, exposure_spec)
    formula = continuous_formula(
        event_col,
        float(model_frame[BURDEN_COL].max()),
        extra_controls,
        exposure_spec,
    )
    kwargs: dict[str, Any] = {}
    if denominator == "per_appearance":
        family = sm.families.Binomial()
    elif denominator == "observed_minutes":
        family = sm.families.Poisson()
        kwargs["offset"] = np.log(pd.to_numeric(frame[MINUTES_COL]).clip(lower=1.0))
    elif denominator == "fixed_90":
        family = sm.families.Poisson()
        kwargs["offset"] = pd.Series(np.log(90.0), index=frame.index)
    else:
        raise ValueError(f"Unknown denominator: {denominator}")
    model = smf.glm(formula=formula, data=model_frame, family=family, **kwargs)
    return model.fit(
        cov_type="cluster", cov_kwds={"groups": model_frame[PLAYER_ID_COL]}
    )


def _normal_p_value(estimate: float, standard_error: float) -> float:
    """Return a two-sided normal p-value."""
    if not np.isfinite(estimate) or not np.isfinite(standard_error):
        return np.nan
    if standard_error <= 0.0:
        return 1.0 if np.isclose(estimate, 0.0) else 0.0
    return float(erfc(abs(estimate / standard_error) / np.sqrt(2.0)))


def _joint_wald(result: Any, terms: Sequence[str]) -> dict[str, float]:
    """Run a joint Wald test for named coefficient terms."""
    names = list(result.params.index)
    selected = [term for term in terms if term in names]
    if not selected:
        return {"test_statistic": np.nan, "df": 0.0, "p_value": np.nan}
    restriction = np.zeros((len(selected), len(names)))
    for row, term in enumerate(selected):
        restriction[row, names.index(term)] = 1.0
    try:
        tested = result.wald_test(restriction, scalar=True)
    except (ValueError, np.linalg.LinAlgError):
        return {
            "test_statistic": np.nan,
            "df": float(len(selected)),
            "p_value": np.nan,
        }
    return {
        "test_statistic": float(np.asarray(tested.statistic).squeeze()),
        "df": float(len(selected)),
        "p_value": float(np.asarray(tested.pvalue).squeeze()),
    }


def formal_model_tests(result: Any, model_id: str) -> pd.DataFrame:
    """Return global recent-exposure and exposure-by-history tests."""
    names = list(result.params.index)
    exposure_names = [
        name
        for name in names
        if BURDEN_COL in name or "exposure_band_model" in name
    ]
    interaction_terms = [
        name for name in exposure_names if f":{HISTORY_MODEL_COL}" in name
    ]
    exposure_terms = [
        name
        for name in exposure_names
        if name not in interaction_terms and ":" not in name
    ]
    rows = []
    for contrast_id, terms, description in (
        (
            "global_recent_exposure_association_at_median_history",
            exposure_terms,
            "joint recent-exposure terms at the median prior-history value",
        ),
        (
            "global_recent_exposure_by_continuous_history_interaction",
            interaction_terms,
            "joint recent-exposure-by-continuous-history interaction",
        ),
        (
            "global_any_recent_exposure_term",
            [*exposure_terms, *interaction_terms],
            "all recent-exposure and exposure-by-history terms",
        ),
    ):
        rows.append(
            {
                "model_id": model_id,
                "contrast_id": contrast_id,
                "description": description,
                "effect_measure": "chi_square",
                **_joint_wald(result, terms),
            }
        )
    return pd.DataFrame(rows)


def prediction_template(
    burdens: Sequence[float],
    history_log: float,
    scaling: Mapping[str, float],
    extra_covariates: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Build a prediction frame at one observed prior-history anchor."""
    history_scaled = (
        float(history_log) - float(scaling["center_log"])
    ) / float(scaling["scale_log_iqr"])
    out = pd.DataFrame(
        {
            BURDEN_COL: [float(value) for value in burdens],
            HISTORY_MODEL_COL: history_scaled,
            **{term: 0.0 for term in CALENDAR_TERMS},
        }
    )
    for column, value in (extra_covariates or {}).items():
        out[column] = value
    return out


def prediction_intervals(
    result: Any,
    template: pd.DataFrame,
    denominator: str,
) -> pd.DataFrame:
    """Attach model-scale 95% intervals and reader-facing prediction units."""
    design = np.asarray(
        build_design_matrices([result.model.data.design_info], template)[0],
        dtype=float,
    )
    params = np.asarray(result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    linear = design @ params
    variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    critical = NormalDist().inv_cdf(0.975)
    out = template.copy()
    if denominator == "per_appearance":
        transform = lambda value: 1000.0 * expit(value)
        unit = "reported_events_per_1000_appearances"
    else:
        transform = lambda value: np.exp(np.clip(value, -690.0, 690.0)) * 60000.0
        unit = "reported_events_per_1000_match_hours"
    out["estimate"] = transform(linear)
    out["ci_low"] = transform(linear - critical * standard_error)
    out["ci_high"] = transform(linear + critical * standard_error)
    out["prediction_unit"] = unit
    return out


def standardization_reference(
    frame: pd.DataFrame,
    extra_covariates: Sequence[str] = (),
) -> pd.DataFrame:
    """Compress the observed covariate distribution into weighted patterns."""
    columns = [*CALENDAR_TERMS, *extra_covariates]
    _require_columns(frame, columns, "standardization frame")
    reference = (
        frame.groupby(columns, dropna=False, observed=False)
        .size()
        .rename("standardization_weight")
        .reset_index()
    )
    if reference.empty or reference["standardization_weight"].sum() <= 0:
        raise ValueError("Standardization reference must contain positive weight")
    return reference


def _marginal_prediction_components(
    result: Any,
    reference: pd.DataFrame,
    burden: float,
    history_scaled: float,
    denominator: str = "per_appearance",
    exposure_spec: str = "reference_bspline",
) -> tuple[float, np.ndarray, str, float]:
    """Return a marginal estimate, delta-method gradient, unit, and multiplier."""
    _require_columns(reference, ["standardization_weight"], "prediction reference")
    template = reference.copy()
    weights = pd.to_numeric(
        template.pop("standardization_weight"), errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Standardization weights must be finite and positive")
    weights = weights / weights.sum()
    template[BURDEN_COL] = float(burden)
    template[HISTORY_MODEL_COL] = float(history_scaled)
    template = add_exposure_spec_columns(template, exposure_spec)
    design = np.asarray(
        build_design_matrices([result.model.data.design_info], template)[0],
        dtype=float,
    )
    linear = design @ np.asarray(result.params, dtype=float)
    if denominator == "per_appearance":
        response = expit(linear)
        derivative = response * (1.0 - response)
        unit = "reported_events_per_1000_appearances"
        multiplier = 1000.0
    elif denominator in {"observed_minutes", "fixed_90"}:
        response = np.exp(np.clip(linear, -690.0, 690.0))
        derivative = response
        unit = "reported_events_per_1000_match_hours"
        multiplier = 60000.0
    else:
        raise ValueError(f"Unknown denominator: {denominator}")
    estimate = float(weights @ response)
    gradient = (weights * derivative) @ design
    return estimate, np.asarray(gradient, dtype=float), unit, multiplier


def marginal_prediction_interval(
    result: Any,
    reference: pd.DataFrame,
    burden: float,
    history_scaled: float,
    denominator: str = "per_appearance",
    exposure_spec: str = "reference_bspline",
) -> dict[str, float | str]:
    """Estimate an absolute risk standardized to observed calendar phases."""
    estimate, gradient, unit, multiplier = _marginal_prediction_components(
        result,
        reference,
        burden,
        history_scaled,
        denominator,
        exposure_spec,
    )
    covariance = np.asarray(result.cov_params(), dtype=float)
    standard_error = float(
        np.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
    )
    critical = NormalDist().inv_cdf(0.975)
    upper_bound = 1.0 if denominator == "per_appearance" else np.inf
    return {
        "estimate": multiplier * estimate,
        "ci_low": multiplier * max(0.0, estimate - critical * standard_error),
        "ci_high": multiplier
        * min(upper_bound, estimate + critical * standard_error),
        "prediction_unit": unit,
        "standardization": "observed_calendar_distribution",
        "standard_error_response_scale": multiplier * standard_error,
    }


def design_contrast(
    result: Any,
    first: pd.DataFrame,
    second: pd.DataFrame,
    effect_measure: str,
) -> dict[str, float | str]:
    """Return an exponentiated one-row design contrast with a 95% interval."""
    design_info = result.model.data.design_info
    first_design = np.asarray(build_design_matrices([design_info], first)[0], dtype=float)[0]
    second_design = np.asarray(build_design_matrices([design_info], second)[0], dtype=float)[0]
    weights = second_design - first_design
    params = np.asarray(result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    log_estimate = float(weights @ params)
    standard_error = float(np.sqrt(max(float(weights @ covariance @ weights), 0.0)))
    critical = NormalDist().inv_cdf(0.975)
    return {
        "effect_measure": effect_measure,
        "estimate": float(np.exp(np.clip(log_estimate, -700.0, 700.0))),
        "ci_low": float(
            np.exp(np.clip(log_estimate - critical * standard_error, -700.0, 700.0))
        ),
        "ci_high": float(
            np.exp(np.clip(log_estimate + critical * standard_error, -700.0, 700.0))
        ),
        "log_estimate": log_estimate,
        "standard_error": standard_error,
        "p_value": _normal_p_value(log_estimate, standard_error),
    }


def model_outputs(
    frame: pd.DataFrame,
    scaling: Mapping[str, float],
    event_col: str,
    denominator: str,
    model_id: str,
    extra_controls: str = "",
    exposure_spec: str = "reference_bspline",
) -> dict[str, pd.DataFrame]:
    """Fit one model and return predictions, tests, contrasts and coefficients."""
    result = fit_continuous_model(
        frame,
        event_col,
        denominator,
        extra_controls,
        exposure_spec,
    )
    effect_measure = "odds_ratio" if denominator == "per_appearance" else "incidence_rate_ratio"
    reference = standardization_reference(frame)
    predictions = []
    contrasts = []
    for anchor, key in (
        ("low", "low_log"),
        ("median", "median_log"),
        ("high", "high_log"),
    ):
        history_scaled = (
            float(scaling[key]) - float(scaling["center_log"])
        ) / float(scaling["scale_log_iqr"])
        for burden in PREDICTION_GRID:
            predictions.append(
                {
                    BURDEN_COL: burden,
                    "history_anchor": anchor,
                    "history_rate_per_10000_prior_minutes": float(
                        scaling[f"{anchor}_rate"]
                    ),
                    **marginal_prediction_interval(
                        result,
                        reference,
                        burden,
                        history_scaled,
                        denominator,
                        exposure_spec,
                    ),
                }
            )
        for burden_from, burden_to in ((0.0, 90.0), (0.0, 180.0), (90.0, 180.0)):
            first = add_exposure_spec_columns(
                prediction_template([burden_from], scaling[key], scaling),
                exposure_spec,
            )
            second = add_exposure_spec_columns(
                prediction_template([burden_to], scaling[key], scaling),
                exposure_spec,
            )
            contrasts.append(
                {
                    "model_id": model_id,
                    "event_col": event_col,
                    "denominator": denominator,
                    "history_anchor": anchor,
                    "history_rate_per_10000_prior_minutes": float(scaling[f"{anchor}_rate"]),
                    "burden_from": burden_from,
                    "burden_to": burden_to,
                    "contrast_id": f"{anchor}_history_{int(burden_to)}_vs_{int(burden_from)}",
                    "contrast_standardization": (
                        "conditional contrast; calendar terms held equal and cancel"
                    ),
                    "exposure_spec": exposure_spec,
                    **design_contrast(result, first, second, effect_measure),
                }
            )
    prediction_table = pd.DataFrame(predictions)
    prediction_table.insert(0, "model_id", model_id)
    prediction_table.insert(1, "event_col", event_col)
    prediction_table.insert(2, "denominator", denominator)
    prediction_table.insert(3, "exposure_spec", exposure_spec)
    tests = formal_model_tests(result, model_id)
    tests["event_col"] = event_col
    tests["denominator"] = denominator
    tests["exposure_spec"] = exposure_spec
    tests["n_match_rows"] = int(len(frame))
    tests["n_players"] = int(frame[PLAYER_ID_COL].nunique())
    tests["n_events"] = int(frame[event_col].sum())
    params = pd.Series(result.params)
    standard_errors = pd.Series(result.bse, index=params.index)
    coefficients = pd.DataFrame(
        {
            "model_id": model_id,
            "exposure_spec": exposure_spec,
            "term": params.index,
            "estimate": params.to_numpy(dtype=float),
            "standard_error": standard_errors.to_numpy(dtype=float),
        }
    )
    coefficients["ci_low"] = coefficients["estimate"] - 1.96 * coefficients["standard_error"]
    coefficients["ci_high"] = coefficients["estimate"] + 1.96 * coefficients["standard_error"]
    coefficients["p_value"] = [
        _normal_p_value(value, se)
        for value, se in zip(coefficients["estimate"], coefficients["standard_error"])
    ]
    coefficients["coefficient_scale"] = (
        "log_odds" if denominator == "per_appearance" else "log_rate"
    )
    return {
        "predictions": prediction_table,
        "tests": tests,
        "contrasts": pd.DataFrame(contrasts),
        "coefficients": coefficients,
    }


def run_symmetric_model_suite(
    frame: pd.DataFrame,
    scaling: Mapping[str, float],
) -> dict[str, pd.DataFrame]:
    """Apply identical models to each timing and denominator definition."""
    specifications = (
        ("primary_same_day_per_appearance", SAME_DAY_COL, "per_appearance"),
        ("lag1_per_appearance", LAG1_COL, "per_appearance"),
        ("combined_per_appearance", COMBINED_COL, "per_appearance"),
        ("same_day_observed_minutes", SAME_DAY_COL, "observed_minutes"),
        ("same_day_fixed_90", SAME_DAY_COL, "fixed_90"),
        ("lag1_observed_minutes", LAG1_COL, "observed_minutes"),
        ("lag1_fixed_90", LAG1_COL, "fixed_90"),
        ("combined_observed_minutes", COMBINED_COL, "observed_minutes"),
        ("combined_fixed_90", COMBINED_COL, "fixed_90"),
    )
    outputs: dict[str, list[pd.DataFrame]] = {
        "predictions": [],
        "tests": [],
        "contrasts": [],
        "coefficients": [],
    }
    for model_id, event_col, denominator in specifications:
        fitted = model_outputs(frame, scaling, event_col, denominator, model_id)
        for key in outputs:
            outputs[key].append(fitted[key])
    return {key: pd.concat(tables, ignore_index=True) for key, tables in outputs.items()}


def exposure_support_table(frame: pd.DataFrame, window: float = 15.0) -> pd.DataFrame:
    """Report row, player and event support around each burden by stratum."""
    _require_columns(
        frame,
        ["fragility_group", BURDEN_COL, PLAYER_ID_COL, SAME_DAY_COL, LAG1_COL, COMBINED_COL],
        "support frame",
    )
    rows = []
    for group, group_frame in frame.groupby("fragility_group", observed=False):
        for burden in DISPLAY_BURDENS:
            selected = group_frame[
                pd.to_numeric(group_frame[BURDEN_COL]).between(
                    max(0.0, burden - window), burden + window, inclusive="both"
                )
            ]
            rows.append(
                {
                    "history_stratum": HISTORY_PUBLICATION_LABELS.get(str(group), str(group)),
                    "display_burden_minutes": burden,
                    "support_window_low": max(0.0, burden - window),
                    "support_window_high": burden + window,
                    "match_rows": int(len(selected)),
                    "players": int(selected[PLAYER_ID_COL].nunique()),
                    "same_day_events": int(selected[SAME_DAY_COL].sum()),
                    "lag1_events": int(selected[LAG1_COL].sum()),
                    "combined_events": int(selected[COMBINED_COL].sum()),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_minute_difference(
    frame: pd.DataFrame,
    event_col: str = SAME_DAY_COL,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap players for the event-minus-non-event mean-minute difference."""
    _require_columns(frame, [PLAYER_ID_COL, MINUTES_COL, event_col], "minute bootstrap frame")
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    work = frame[[PLAYER_ID_COL, MINUTES_COL, event_col]].copy()
    work[event_col] = pd.to_numeric(work[event_col], errors="coerce").fillna(0).astype(int)
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    grouped = work.groupby(PLAYER_ID_COL, sort=True)
    player_stats = grouped.apply(
        lambda part: pd.Series(
            {
                "event_sum": float(part.loc[part[event_col].eq(1), MINUTES_COL].sum()),
                "event_n": int(part[event_col].eq(1).sum()),
                "nonevent_sum": float(part.loc[part[event_col].eq(0), MINUTES_COL].sum()),
                "nonevent_n": int(part[event_col].eq(0).sum()),
            }
        ),
        include_groups=False,
    )
    if player_stats["event_n"].sum() <= 0 or player_stats["nonevent_n"].sum() <= 0:
        raise ValueError("Both event and non-event rows are required")
    arrays = player_stats[
        ["event_sum", "event_n", "nonevent_sum", "nonevent_n"]
    ].to_numpy()
    rng = np.random.default_rng(seed)
    samples = []
    n_players = len(player_stats)
    for replicate in range(1, replicates + 1):
        counts = rng.multinomial(n_players, np.full(n_players, 1.0 / n_players))
        totals = counts @ arrays
        event_mean = totals[0] / totals[1]
        nonevent_mean = totals[2] / totals[3]
        samples.append(
            {
                "replicate": replicate,
                "event_mean_minutes": event_mean,
                "non_event_mean_minutes": nonevent_mean,
                "event_minus_non_event_minutes": event_mean - nonevent_mean,
            }
        )
    sample_frame = pd.DataFrame(samples)
    event_mean = float(work.loc[work[event_col].eq(1), MINUTES_COL].mean())
    nonevent_mean = float(work.loc[work[event_col].eq(0), MINUTES_COL].mean())
    difference = event_mean - nonevent_mean
    summary = pd.DataFrame(
        [
            {
                "event_col": event_col,
                "bootstrap_unit": "player",
                "bootstrap_replicates": int(replicates),
                "event_rows": int(work[event_col].sum()),
                "non_event_rows": int(work[event_col].eq(0).sum()),
                "event_mean_minutes": event_mean,
                "non_event_mean_minutes": nonevent_mean,
                "event_minus_non_event_minutes": difference,
                "difference_ci_low": float(
                    sample_frame["event_minus_non_event_minutes"].quantile(0.025)
                ),
                "difference_ci_high": float(
                    sample_frame["event_minus_non_event_minutes"].quantile(0.975)
                ),
                "interval_method": "player_cluster_percentile_bootstrap_95",
            }
        ]
    )
    return sample_frame, summary


def _weighted_standardized_prediction(
    result: Any,
    reference: pd.DataFrame,
    burden: float,
    history_scaled: float,
) -> dict[str, float]:
    """Average fitted probabilities over a weighted covariate distribution."""
    template = reference.copy()
    template[BURDEN_COL] = float(burden)
    template[HISTORY_MODEL_COL] = float(history_scaled)
    weights = template.pop("standardization_weight").to_numpy(dtype=float)
    weights = weights / weights.sum()
    design = np.asarray(
        build_design_matrices([result.model.data.design_info], template)[0],
        dtype=float,
    )
    params = np.asarray(result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    probability = expit(design @ params)
    estimate = float(weights @ probability)
    gradient = (weights * probability * (1.0 - probability)) @ design
    standard_error = float(
        np.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
    )
    critical = NormalDist().inv_cdf(0.975)
    return {
        "estimate_per_1000_appearances": 1000.0 * estimate,
        "ci_low": 1000.0 * max(0.0, estimate - critical * standard_error),
        "ci_high": 1000.0 * min(1.0, estimate + critical * standard_error),
        "standard_error_probability": standard_error,
    }


def factorized_standardization_reference(
    calendar_frame: pd.DataFrame,
    composition_frame: pd.DataFrame,
    composition_columns: Sequence[str],
) -> pd.DataFrame:
    """Cross a common calendar distribution with one observed composition."""
    calendar = standardization_reference(calendar_frame)
    _require_columns(
        composition_frame, composition_columns, "composition frame"
    )
    composition = (
        composition_frame.groupby(
            list(composition_columns), dropna=False, observed=False
        )
        .size()
        .rename("standardization_weight")
        .reset_index()
    )
    calendar = calendar.rename(
        columns={"standardization_weight": "calendar_weight"}
    )
    composition = composition.rename(
        columns={"standardization_weight": "composition_weight"}
    )
    reference = calendar.merge(composition, how="cross")
    reference["standardization_weight"] = (
        pd.to_numeric(reference.pop("calendar_weight"), errors="coerce")
        * pd.to_numeric(reference.pop("composition_weight"), errors="coerce")
    )
    return reference


def _response_difference(
    result: Any,
    first: tuple[float, np.ndarray, str, float],
    second: tuple[float, np.ndarray, str, float],
) -> dict[str, float | str]:
    """Return a delta-method difference between two marginal predictions."""
    first_estimate, first_gradient, first_unit, first_multiplier = first
    second_estimate, second_gradient, second_unit, second_multiplier = second
    if first_unit != second_unit or first_multiplier != second_multiplier:
        raise ValueError("Marginal predictions must use the same unit")
    gradient = second_gradient - first_gradient
    covariance = np.asarray(result.cov_params(), dtype=float)
    standard_error = float(
        np.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
    )
    estimate = float(second_estimate - first_estimate)
    critical = NormalDist().inv_cdf(0.975)
    return {
        "effect_measure": f"risk_difference_{first_unit}",
        "estimate": first_multiplier * estimate,
        "ci_low": first_multiplier * (estimate - critical * standard_error),
        "ci_high": first_multiplier * (estimate + critical * standard_error),
        "standard_error": first_multiplier * standard_error,
        "p_value": _normal_p_value(estimate, standard_error),
    }


def _weighted_pattern_reference(
    frame: pd.DataFrame,
    columns: Sequence[str],
    row_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Aggregate an observed covariate distribution with optional row weights."""
    _require_columns(frame, columns, "weighted reference frame")
    work = frame[list(columns)].copy()
    if row_weights is None:
        work["standardization_weight"] = 1.0
    else:
        aligned = pd.to_numeric(row_weights.reindex(frame.index), errors="coerce")
        if aligned.isna().any() or aligned.lt(0.0).any():
            raise ValueError("Bootstrap row weights must be finite and non-negative")
        work["standardization_weight"] = aligned.to_numpy(dtype=float)
    reference = (
        work.groupby(list(columns), dropna=False, observed=False)[
            "standardization_weight"
        ]
        .sum()
        .reset_index()
    )
    reference = reference[reference["standardization_weight"].gt(0.0)].copy()
    if reference.empty:
        raise ValueError("Weighted reference must contain positive weight")
    return reference


def _weighted_factorized_reference(
    calendar_frame: pd.DataFrame,
    composition_frame: pd.DataFrame,
    composition_columns: Sequence[str],
    row_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Cross weighted calendar and lineup/return distributions."""
    calendar = _weighted_pattern_reference(
        calendar_frame,
        CALENDAR_TERMS,
        row_weights,
    ).rename(columns={"standardization_weight": "calendar_weight"})
    composition = _weighted_pattern_reference(
        composition_frame,
        composition_columns,
        row_weights,
    ).rename(columns={"standardization_weight": "composition_weight"})
    reference = calendar.merge(composition, how="cross")
    reference["standardization_weight"] = (
        reference.pop("calendar_weight") * reference.pop("composition_weight")
    )
    return reference


def _selection_effect_point_estimates(
    result: Any,
    selected: pd.DataFrame,
    scaling: Mapping[str, float],
    row_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute paired changing- and fixed-composition risk differences."""
    composition_columns = [
        "lineup_role_model",
        "returned_from_recorded_injury_within_14d",
    ]
    local_frames = {
        0.0: selected[selected[BURDEN_COL].eq(0.0)],
        45.0: selected[selected[BURDEN_COL].between(30.0, 60.0)],
        REFERENCE_ANCHOR_MINUTES: selected[
            selected[BURDEN_COL].between(165.0, 195.0)
        ],
    }
    if any(local.empty for local in local_frames.values()):
        raise ValueError("Selection contrasts require support at 0, 45 and 180 minutes")
    pooled = _weighted_factorized_reference(
        selected,
        selected,
        composition_columns,
        row_weights,
    )
    local_references = {
        burden: _weighted_factorized_reference(
            selected,
            local,
            composition_columns,
            row_weights,
        )
        for burden, local in local_frames.items()
    }
    rows = []
    for anchor, key in (
        ("low", "low_log"),
        ("median", "median_log"),
        ("high", "high_log"),
    ):
        history_scaled = (
            float(scaling[key]) - float(scaling["center_log"])
        ) / float(scaling["scale_log_iqr"])
        for burden_to in (45.0, REFERENCE_ANCHOR_MINUTES):
            natural_first = _marginal_prediction_components(
                result, local_references[0.0], 0.0, history_scaled
            )[0]
            natural_second = _marginal_prediction_components(
                result,
                local_references[burden_to],
                burden_to,
                history_scaled,
            )[0]
            fixed_first = _marginal_prediction_components(
                result, pooled, 0.0, history_scaled
            )[0]
            fixed_second = _marginal_prediction_components(
                result, pooled, burden_to, history_scaled
            )[0]
            changing = 1000.0 * (natural_second - natural_first)
            fixed = 1000.0 * (fixed_second - fixed_first)
            for comparison_type, estimate in (
                ("changing_observed_lineup_return_composition", changing),
                ("fixed_pooled_lineup_return_composition", fixed),
                ("difference_between_changes", changing - fixed),
            ):
                rows.append(
                    {
                        "contrast_id": (
                            f"{anchor}_{int(burden_to)}_vs_0_{comparison_type}"
                        ),
                        "history_anchor": anchor,
                        "history_rate_per_10000_prior_minutes": float(
                            scaling[f"{anchor}_rate"]
                        ),
                        "burden_from": 0.0,
                        "burden_to": burden_to,
                        "comparison_type": comparison_type,
                        "composition_support_rows_from": int(
                            len(local_frames[0.0])
                        ),
                        "composition_support_rows_to": int(
                            len(local_frames[burden_to])
                        ),
                        "estimate": float(estimate),
                    }
                )
    return pd.DataFrame(rows)


def _fit_selection_bootstrap_model(
    selected: pd.DataFrame,
    row_weights: np.ndarray,
    start_params: pd.Series,
) -> Any:
    """Refit the lineup/return model for one player-cluster bootstrap draw."""
    weights = np.asarray(row_weights, dtype=float)
    if len(weights) != len(selected) or not np.isfinite(weights).all():
        raise ValueError("Bootstrap row weights must align with selection rows")
    formula = continuous_formula(
        SAME_DAY_COL,
        float(selected[BURDEN_COL].max()),
        " + C(lineup_role_model) + returned_from_recorded_injury_within_14d",
    )
    model = smf.glm(
        formula=formula,
        data=selected,
        family=sm.families.Binomial(),
        freq_weights=weights,
    )
    result = model.fit(start_params=start_params, maxiter=60, disp=0)
    if not bool(result.converged):
        raise ValueError("Bootstrap selection model did not converge")
    return result


def selection_effect_contrasts(
    frame: pd.DataFrame,
    scaling: Mapping[str, float],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED + 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fully refit paired player-cluster selection contrasts and intervals."""
    composition_columns = [
        "lineup_role_model",
        "returned_from_recorded_injury_within_14d",
    ]
    _require_columns(
        frame,
        [
            PLAYER_ID_COL,
            SAME_DAY_COL,
            BURDEN_COL,
            HISTORY_MODEL_COL,
            *CALENDAR_TERMS,
            *composition_columns,
        ],
        "selection-effect frame",
    )
    selected = frame[frame["lineup_role_model"].isin(LINEUP_ROLES)].copy()
    if selected.empty or selected["lineup_role_model"].nunique() < 2:
        raise ValueError("Both recorded lineup roles are required")
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    result = fit_continuous_model(
        selected,
        SAME_DAY_COL,
        "per_appearance",
        " + C(lineup_role_model) + returned_from_recorded_injury_within_14d",
    )
    point = _selection_effect_point_estimates(result, selected, scaling)
    players = np.asarray(sorted(selected[PLAYER_ID_COL].unique()), dtype=object)
    player_lookup = {player: index for index, player in enumerate(players)}
    player_codes = selected[PLAYER_ID_COL].map(player_lookup).to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    samples = []
    for replicate in range(replicates):
        sampled = rng.multinomial(
            len(players), np.full(len(players), 1.0 / len(players))
        )
        row_weight_array = sampled[player_codes].astype(float)
        try:
            fitted = _fit_selection_bootstrap_model(
                selected,
                row_weight_array,
                result.params,
            )
            row_weights = pd.Series(row_weight_array, index=selected.index)
            replicate_rows = _selection_effect_point_estimates(
                fitted,
                selected,
                scaling,
                row_weights,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        replicate_rows.insert(0, "bootstrap_replicate", replicate + 1)
        samples.append(replicate_rows)
    if len(samples) < max(2, int(0.9 * replicates)):
        raise ValueError("Too few estimable full selection bootstrap replicates")
    sample_table = pd.concat(samples, ignore_index=True)
    rows = []
    for point_row in point.to_dict("records"):
        contrast_samples = pd.to_numeric(
            sample_table.loc[
                sample_table["contrast_id"].eq(point_row["contrast_id"]),
                "estimate",
            ],
            errors="coerce",
        ).dropna()
        negative = int(contrast_samples.le(0.0).sum())
        positive = int(contrast_samples.ge(0.0).sum())
        bootstrap_p = min(
            1.0,
            2.0 * (min(negative, positive) + 1.0) / (len(contrast_samples) + 1.0),
        )
        rows.append(
            {
                **point_row,
                "calendar_standardization": "common observed calendar distribution",
                "effect_measure": (
                    "risk_difference_reported_events_per_1000_appearances"
                ),
                "ci_low": float(contrast_samples.quantile(0.025)),
                "ci_high": float(contrast_samples.quantile(0.975)),
                "standard_error": float(contrast_samples.std(ddof=1)),
                "p_value": float(bootstrap_p),
                "bootstrap_replicates_requested": int(replicates),
                "bootstrap_replicates_estimable": int(len(contrast_samples)),
                "interval_method": (
                    "full paired player-cluster percentile bootstrap; model "
                    "refitted and both standardizations recomputed in every draw"
                ),
            }
        )
    return sample_table, pd.DataFrame(rows)


def lineup_completeness_audit(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe lineup ascertainment and identify a complete-coverage era."""
    required = [
        "season_start",
        "competition_context",
        "lineup_role_model",
        BURDEN_COL,
        HISTORY_MODEL_COL,
        SAME_DAY_COL,
    ]
    _require_columns(frame, required, "lineup completeness frame")
    work = frame.copy()
    work["lineup_known"] = work["lineup_role_model"].isin(LINEUP_ROLES)
    work["exposure_band"] = exposure_band(work[BURDEN_COL]).astype(str)
    ranked_history = pd.to_numeric(
        work[HISTORY_MODEL_COL], errors="coerce"
    ).rank(method="first")
    work["history_quartile"] = pd.qcut(
        ranked_history,
        4,
        labels=("Q1 lower", "Q2", "Q3", "Q4 higher"),
    ).astype(str)
    work["same_day_status"] = np.where(
        pd.to_numeric(work[SAME_DAY_COL], errors="coerce").fillna(0).eq(1),
        "same-day reported event",
        "no same-day reported event",
    )
    rows = []
    dimensions = (
        ("overall", pd.Series("all appearances", index=work.index)),
        ("season", work["season_start"].astype(str)),
        ("recent exposure", work["exposure_band"]),
        ("continuous-history quartile", work["history_quartile"]),
        ("same-day outcome", work["same_day_status"]),
        ("competition", work["competition_context"].astype(str)),
    )
    for dimension, labels in dimensions:
        grouped = work.assign(_audit_level=labels).groupby(
            "_audit_level", observed=False, dropna=False
        )
        for level, subset in grouped:
            total = int(len(subset))
            known = int(subset["lineup_known"].sum())
            low, high = proportion_confint(
                known,
                total,
                alpha=0.05,
                method="wilson",
            )
            rows.append(
                {
                    "dimension": dimension,
                    "level": str(level),
                    "appearances": total,
                    "lineup_known_appearances": known,
                    "lineup_known_percent": 100.0 * known / total,
                    "lineup_known_percent_ci_low": 100.0 * float(low),
                    "lineup_known_percent_ci_high": 100.0 * float(high),
                    "same_day_reports": int(subset[SAME_DAY_COL].sum()),
                }
            )
    audit = pd.DataFrame(rows)
    season_rows = audit[audit["dimension"].eq("season")].copy()
    complete_seasons = sorted(
        int(value)
        for value in season_rows.loc[
            season_rows["lineup_known_appearances"].eq(
                season_rows["appearances"]
            ),
            "level",
        ]
    )
    if not complete_seasons:
        raise ValueError("No season has complete lineup-role ascertainment")
    complete = work[work["season_start"].isin(complete_seasons)].copy()
    zero_coverage_seasons = sorted(
        int(value)
        for value in season_rows.loc[
            season_rows["lineup_known_appearances"].eq(0), "level"
        ]
    )
    assessment = pd.DataFrame(
        [
            {
                "selection_analysis_seasons": ";".join(
                    str(value) for value in complete_seasons
                ),
                "selection_analysis_first_season": min(complete_seasons),
                "selection_analysis_last_season": max(complete_seasons),
                "selection_analysis_appearances": int(len(complete)),
                "selection_analysis_players": int(
                    complete[PLAYER_ID_COL].nunique()
                ),
                "selection_analysis_same_day_reports": int(
                    complete[SAME_DAY_COL].sum()
                ),
                "all_years_lineup_known_percent": 100.0
                * float(work["lineup_known"].mean()),
                "zero_coverage_seasons": ";".join(
                    str(value) for value in zero_coverage_seasons
                ),
                "inverse_probability_reweighting_performed": False,
                "reweighting_assessment": (
                    "not identifiable for the full cohort because at least one "
                    "season has zero lineup-role coverage"
                    if zero_coverage_seasons
                    else "not required for the complete-coverage season analysis"
                ),
            }
        ]
    )
    return audit, assessment, complete


def within_player_same_day_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate same-day exposure contrasts after conditioning on player."""
    required = [
        PLAYER_ID_COL,
        "season_start",
        SAME_DAY_COL,
        BURDEN_COL,
        HISTORY_MODEL_COL,
        "date",
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "within-player frame")
    rows = []
    burden_max = float(frame[BURDEN_COL].max())
    for stratum_definition in ("player", "player-season"):
        work = frame.copy()
        if stratum_definition == "player":
            work["conditional_stratum"] = work[PLAYER_ID_COL].astype(str)
        else:
            work["conditional_stratum"] = (
                work[PLAYER_ID_COL].astype(str)
                + "_"
                + work["season_start"].astype(str)
            )
        counts = work.groupby("conditional_stratum", observed=False)[
            SAME_DAY_COL
        ].agg(["sum", "count"])
        discordant = counts[
            counts["sum"].gt(0) & counts["sum"].lt(counts["count"])
        ].index
        work = work[work["conditional_stratum"].isin(discordant)].copy()
        for exposure_spec in ("reference_bspline", "linear_per_90"):
            model_id = (
                f"within_{stratum_definition.replace('-', '_')}_"
                f"{exposure_spec}_same_day"
            )
            if work.empty:
                rows.append(
                    {
                        "model_id": model_id,
                        "stratum_definition": stratum_definition,
                        "exposure_spec": exposure_spec,
                        "fit_status": "not_estimable_no_discordant_strata",
                        "p_value": np.nan,
                    }
                )
                continue
            expression = exposure_expression(exposure_spec, burden_max)
            design_formula = (
                f"0 + {expression} + {HISTORY_MODEL_COL} + "
                + " + ".join(CALENDAR_TERMS)
            )
            design = dmatrix(design_formula, work, return_type="dataframe")
            try:
                result = ConditionalLogit(
                    work[SAME_DAY_COL].astype(int),
                    design,
                    groups=work["conditional_stratum"],
                ).fit(disp=False, maxiter=200)
                template = pd.DataFrame(
                    {
                        BURDEN_COL: [0.0, REFERENCE_ANCHOR_MINUTES],
                        HISTORY_MODEL_COL: [0.0, 0.0],
                        **{term: [0.0, 0.0] for term in CALENDAR_TERMS},
                    }
                )
                template = add_exposure_spec_columns(template, exposure_spec)
                contrast_design = np.asarray(
                    build_design_matrices([design.design_info], template)[0],
                    dtype=float,
                )
                weights = contrast_design[1] - contrast_design[0]
                log_estimate = float(weights @ np.asarray(result.params))
                covariance = np.asarray(result.cov_params())
                standard_error = float(
                    np.sqrt(max(float(weights @ covariance @ weights), 0.0))
                )
                critical = NormalDist().inv_cdf(0.975)
                row = {
                    "model_id": model_id,
                    "stratum_definition": stratum_definition,
                    "exposure_spec": exposure_spec,
                    "contrast_id": "same_day_180_vs_0",
                    "effect_measure": "conditional_odds_ratio",
                    "estimate": float(np.exp(np.clip(log_estimate, -700.0, 700.0))),
                    "ci_low": float(
                        np.exp(
                            np.clip(
                                log_estimate - critical * standard_error,
                                -700.0,
                                700.0,
                            )
                        )
                    ),
                    "ci_high": float(
                        np.exp(
                            np.clip(
                                log_estimate + critical * standard_error,
                                -700.0,
                                700.0,
                            )
                        )
                    ),
                    "p_value": _normal_p_value(log_estimate, standard_error),
                    "fit_status": "ok",
                }
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                row = {
                    "model_id": model_id,
                    "stratum_definition": stratum_definition,
                    "exposure_spec": exposure_spec,
                    "contrast_id": "same_day_180_vs_0",
                    "effect_measure": "conditional_odds_ratio",
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "fit_status": "failed_convergence",
                }
            row.update(
                {
                    "analysis_timing": "post-data reviewer-requested sensitivity",
                    "confirmatory_status": "exploratory",
                    "burden_from": 0.0,
                    "burden_to": REFERENCE_ANCHOR_MINUTES,
                    "n_match_rows": int(len(work)),
                    "n_discordant_strata": int(len(discordant)),
                    "n_players": int(work[PLAYER_ID_COL].nunique()),
                    "n_events": int(work[SAME_DAY_COL].sum()),
                    "controls": (
                        "continuous prior history and calendar phase; stratum "
                        "intercept conditioned out"
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def daily_report_timing_enrichment(
    day_panel: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED + 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare report-start frequency on match, next, and other risk-set days."""
    required = [
        PLAYER_ID_COL,
        "injury_event",
        "available_for_injury_risk",
        "prior_minutes_played",
        MINUTES_COL,
        "minutes_yesterday",
        BURDEN_COL,
        HISTORY_COL,
    ]
    _require_columns(day_panel, required, "daily timing frame")
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    work = day_panel.copy()
    for column in (
        "injury_event",
        "prior_minutes_played",
        MINUTES_COL,
        "minutes_yesterday",
        BURDEN_COL,
        HISTORY_COL,
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work[
        work["available_for_injury_risk"].fillna(False).astype(bool)
        & work["prior_minutes_played"].ge(REFERENCE_MINIMUM_PRIOR_MINUTES)
        & work[BURDEN_COL].notna()
        & work[HISTORY_COL].notna()
    ].copy()
    work["timing_class"] = np.select(
        [
            work[MINUTES_COL].gt(0.0),
            work[MINUTES_COL].le(0.0) & work["minutes_yesterday"].gt(0.0),
        ],
        ["appearance day", "day after appearance"],
        default="other eligible day",
    )
    timing_order = (
        "appearance day",
        "day after appearance",
        "other eligible day",
    )
    summary_rows = []
    for timing_class in timing_order:
        subset = work[work["timing_class"].eq(timing_class)]
        total = int(len(subset))
        events = int(subset["injury_event"].sum())
        if total <= 0 or events <= 0:
            raise ValueError("Each timing class requires rows and report starts")
        low, high = proportion_confint(events, total, alpha=0.05, method="wilson")
        summary_rows.append(
            {
                "timing_class": timing_class,
                "eligible_player_days": total,
                "reported_episode_starts": events,
                "reports_per_1000_player_days": 1000.0 * events / total,
                "ci_low": 1000.0 * float(low),
                "ci_high": 1000.0 * float(high),
                "date_semantics": (
                    "Transfermarkt start/startDate field: recorded spell-start "
                    "date, not a publication timestamp or confirmed injury time"
                ),
            }
        )
    player_counts = (
        work.groupby([PLAYER_ID_COL, "timing_class"], observed=False)[
            "injury_event"
        ]
        .agg(player_days="size", report_starts="sum")
        .reset_index()
    )
    players = np.asarray(sorted(work[PLAYER_ID_COL].unique()), dtype=object)
    index = pd.MultiIndex.from_product(
        [players, timing_order], names=[PLAYER_ID_COL, "timing_class"]
    )
    player_counts = (
        player_counts.set_index([PLAYER_ID_COL, "timing_class"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    day_matrix = player_counts["player_days"].to_numpy(dtype=float).reshape(
        len(players), len(timing_order)
    )
    event_matrix = player_counts["report_starts"].to_numpy(dtype=float).reshape(
        len(players), len(timing_order)
    )
    rng = np.random.default_rng(seed)
    sample_rows = []
    for replicate in range(replicates):
        sampled = rng.multinomial(
            len(players), np.full(len(players), 1.0 / len(players))
        )
        days = sampled @ day_matrix
        events = sampled @ event_matrix
        risks = events / days
        for comparison, target_index in (
            ("appearance_day_vs_other", 0),
            ("day_after_appearance_vs_other", 1),
        ):
            sample_rows.append(
                {
                    "bootstrap_replicate": replicate + 1,
                    "comparison": comparison,
                    "risk_ratio": float(risks[target_index] / risks[2]),
                    "risk_difference_per_1000_player_days": float(
                        1000.0 * (risks[target_index] - risks[2])
                    ),
                }
            )
    samples = pd.DataFrame(sample_rows)
    point = pd.DataFrame(summary_rows).set_index("timing_class")
    contrast_rows = []
    for comparison, target in (
        ("appearance_day_vs_other", "appearance day"),
        ("day_after_appearance_vs_other", "day after appearance"),
    ):
        target_risk = point.loc[target, "reports_per_1000_player_days"] / 1000.0
        other_risk = (
            point.loc["other eligible day", "reports_per_1000_player_days"]
            / 1000.0
        )
        selected_samples = samples[samples["comparison"].eq(comparison)]
        contrast_rows.append(
            {
                "comparison": comparison,
                "risk_ratio": float(target_risk / other_risk),
                "risk_ratio_ci_low": float(
                    selected_samples["risk_ratio"].quantile(0.025)
                ),
                "risk_ratio_ci_high": float(
                    selected_samples["risk_ratio"].quantile(0.975)
                ),
                "risk_difference_per_1000_player_days": float(
                    1000.0 * (target_risk - other_risk)
                ),
                "risk_difference_ci_low": float(
                    selected_samples[
                        "risk_difference_per_1000_player_days"
                    ].quantile(0.025)
                ),
                "risk_difference_ci_high": float(
                    selected_samples[
                        "risk_difference_per_1000_player_days"
                    ].quantile(0.975)
                ),
                "bootstrap_replicates": int(replicates),
                "interval_method": "player-cluster percentile bootstrap",
                "interpretation": (
                    "descriptive timing enrichment; not validation of match causation"
                ),
            }
        )
    return pd.DataFrame(summary_rows), samples, pd.DataFrame(contrast_rows)


def eligibility_player_comparison(
    base_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compare established included players with lower-exposure exclusions."""
    required = [
        PLAYER_ID_COL,
        "date",
        MINUTES_COL,
        SAME_DAY_COL,
        "age_years",
        "position_group",
    ]
    _require_columns(base_frame, required, "eligibility base frame")
    _require_columns(reference_frame, [PLAYER_ID_COL], "eligibility reference frame")
    work = base_frame.sort_values([PLAYER_ID_COL, "date"]).copy()
    included_players = set(reference_frame[PLAYER_ID_COL].unique())
    player = (
        work.groupby(PLAYER_ID_COL, observed=False)
        .agg(
            first_age_years=("age_years", "first"),
            position=("position_group", "first"),
            recorded_appearances=(PLAYER_ID_COL, "size"),
            recorded_minutes=(MINUTES_COL, "sum"),
            same_day_reports=(SAME_DAY_COL, "sum"),
        )
        .reset_index()
    )
    player["eligibility_group"] = np.where(
        player[PLAYER_ID_COL].isin(included_players),
        "included established players",
        "excluded before reaching 900 prior minutes",
    )
    rows = []
    for group, subset in player.groupby("eligibility_group", observed=False):
        appearances = int(subset["recorded_appearances"].sum())
        events = int(subset["same_day_reports"].sum())
        low, high = proportion_confint(events, appearances, alpha=0.05, method="wilson")
        for metric, values in (
            ("age at first observed appearance, years", subset["first_age_years"]),
            ("recorded appearances per player", subset["recorded_appearances"]),
            ("recorded minutes per player", subset["recorded_minutes"]),
        ):
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            rows.append(
                {
                    "eligibility_group": group,
                    "metric": metric,
                    "value": float(numeric.median()),
                    "ci_or_iqr_low": float(numeric.quantile(0.25)),
                    "ci_or_iqr_high": float(numeric.quantile(0.75)),
                    "interval_type": "interquartile range",
                }
            )
        rows.extend(
            [
                {
                    "eligibility_group": group,
                    "metric": "players",
                    "value": int(len(subset)),
                    "ci_or_iqr_low": np.nan,
                    "ci_or_iqr_high": np.nan,
                    "interval_type": "exact count",
                },
                {
                    "eligibility_group": group,
                    "metric": "same-day reports per 1,000 appearances",
                    "value": 1000.0 * events / appearances,
                    "ci_or_iqr_low": 1000.0 * float(low),
                    "ci_or_iqr_high": 1000.0 * float(high),
                    "interval_type": "Wilson 95% confidence interval",
                },
            ]
        )
        for position, count in subset["position"].fillna("Unknown").value_counts().items():
            pos_low, pos_high = proportion_confint(
                int(count), len(subset), alpha=0.05, method="wilson"
            )
            rows.append(
                {
                    "eligibility_group": group,
                    "metric": f"position: {position}, percent",
                    "value": 100.0 * int(count) / len(subset),
                    "ci_or_iqr_low": 100.0 * float(pos_low),
                    "ci_or_iqr_high": 100.0 * float(pos_high),
                    "interval_type": "Wilson 95% confidence interval",
                }
            )
    return pd.DataFrame(rows)


def crude_outcome_summary(
    frame: pd.DataFrame,
    event_col: str,
    outcome_label: str,
) -> pd.DataFrame:
    """Report exact count-rate and Wilson appearance-risk intervals."""
    _require_columns(frame, [event_col, MINUTES_COL], "outcome summary frame")
    events = int(pd.to_numeric(frame[event_col], errors="coerce").fillna(0).sum())
    appearances = int(len(frame))
    hours = float(
        pd.to_numeric(frame[MINUTES_COL], errors="coerce").fillna(0).sum()
        / 60.0
    )
    if appearances <= 0 or hours <= 0:
        raise ValueError("Outcome summaries require appearances and positive hours")
    risk_low, risk_high = proportion_confint(
        events, appearances, alpha=0.05, method="wilson"
    )
    count_low = 0.0 if events == 0 else 0.5 * chi2.ppf(0.025, 2 * events)
    count_high = 0.5 * chi2.ppf(0.975, 2 * (events + 1))
    return pd.DataFrame(
        [
            {
                "event_col": event_col,
                "outcome_label": outcome_label,
                "match_rows": appearances,
                "events": events,
                "observed_match_hours": hours,
                "events_per_1000_appearances": 1000.0 * events / appearances,
                "appearance_risk_ci_low": 1000.0 * float(risk_low),
                "appearance_risk_ci_high": 1000.0 * float(risk_high),
                "events_per_1000_observed_match_hours": 1000.0 * events / hours,
                "hour_rate_ci_low": 1000.0 * count_low / hours,
                "hour_rate_ci_high": 1000.0 * count_high / hours,
                "appearance_interval_method": "Wilson 95%",
                "hour_rate_interval_method": "exact Poisson 95%",
            }
        ]
    )


def _model_test_and_contrast_rows(
    frame: pd.DataFrame,
    scaling: Mapping[str, float],
    event_col: str,
    model_id: str,
    exposure_spec: str = "reference_bspline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one same-day appearance model and return its formal tests and contrast."""
    result = fit_continuous_model(
        frame,
        event_col,
        "per_appearance",
        exposure_spec=exposure_spec,
    )
    tests = formal_model_tests(result, model_id)
    tests["event_col"] = event_col
    tests["denominator"] = "per_appearance"
    tests["exposure_spec"] = exposure_spec
    tests["n_match_rows"] = int(len(frame))
    tests["n_players"] = int(frame[PLAYER_ID_COL].nunique())
    tests["n_events"] = int(frame[event_col].sum())
    first = add_exposure_spec_columns(
        prediction_template([0.0], scaling["median_log"], scaling),
        exposure_spec,
    )
    second = add_exposure_spec_columns(
        prediction_template([180.0], scaling["median_log"], scaling),
        exposure_spec,
    )
    contrast = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "contrast_id": "median_history_180_vs_0",
                "event_col": event_col,
                "denominator": "per_appearance",
                "exposure_spec": exposure_spec,
                "history_anchor": "median",
                "history_rate_per_10000_prior_minutes": float(
                    scaling["median_rate"]
                ),
                "burden_from": 0.0,
                "burden_to": 180.0,
                "n_match_rows": int(len(frame)),
                "n_players": int(frame[PLAYER_ID_COL].nunique()),
                "n_events": int(frame[event_col].sum()),
                **design_contrast(result, first, second, "odds_ratio"),
            }
        ]
    )
    return tests, contrast


def functional_form_sensitivity(
    frame: pd.DataFrame,
    scaling: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the same estimand under five fixed exposure representations."""
    tests = []
    contrasts = []
    for exposure_spec in FUNCTIONAL_FORM_SPECS:
        model_id = f"same_day_per_appearance_{exposure_spec}"
        fitted_tests, fitted_contrast = _model_test_and_contrast_rows(
            frame,
            scaling,
            SAME_DAY_COL,
            model_id,
            exposure_spec,
        )
        tests.append(fitted_tests)
        contrasts.append(fitted_contrast)
    return pd.concat(tests, ignore_index=True), pd.concat(contrasts, ignore_index=True)


def outcome_quality_sensitivity(
    frame: pd.DataFrame,
    scaling: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeat the primary model for severe and muscle/tendon same-day reports."""
    specifications = (
        (SAME_DAY_COL, "all same-day reports"),
        (SAME_DAY_SEVERE_COL, "same-day reports with at least 28 reported days"),
        (SAME_DAY_MUSCLE_COL, "same-day muscle/tendon reports"),
    )
    tests = []
    contrasts = []
    summaries = []
    for event_col, label in specifications:
        _require_columns(frame, [event_col], "outcome-quality frame")
        summaries.append(crude_outcome_summary(frame, event_col, label))
        if int(frame[event_col].sum()) <= 0:
            continue
        model_id = f"quality_{event_col}_per_appearance"
        fitted_tests, fitted_contrast = _model_test_and_contrast_rows(
            frame, scaling, event_col, model_id
        )
        tests.append(fitted_tests)
        contrasts.append(fitted_contrast)
    return (
        pd.concat(tests, ignore_index=True) if tests else pd.DataFrame(),
        pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame(),
        pd.concat(summaries, ignore_index=True),
    )


def lineup_role_model_sensitivity(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the same continuous-history model within starters and substitutes."""
    _require_columns(frame, ["lineup_role_model"], "lineup-role frame")
    tests = []
    contrasts = []
    for role in LINEUP_ROLES:
        subset = frame[frame["lineup_role_model"].eq(role)].copy()
        if subset.empty or int(subset[SAME_DAY_COL].sum()) <= 0:
            continue
        role_scaling = history_scale(subset)
        subset = apply_history_scale(subset, role_scaling)
        model_id = f"same_day_per_appearance_{role}"
        fitted_tests, fitted_contrast = _model_test_and_contrast_rows(
            subset, role_scaling, SAME_DAY_COL, model_id
        )
        fitted_tests["lineup_role"] = role
        fitted_contrast["lineup_role"] = role
        tests.append(fitted_tests)
        contrasts.append(fitted_contrast)
    return (
        pd.concat(tests, ignore_index=True) if tests else pd.DataFrame(),
        pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame(),
    )


def lineup_standardized_minute_difference(
    frame: pd.DataFrame,
    event_col: str = SAME_DAY_COL,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED + 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify the minute gap by lineup role and standardize over pooled roles."""
    _require_columns(
        frame,
        [PLAYER_ID_COL, MINUTES_COL, event_col, "lineup_role_model"],
        "lineup minute frame",
    )
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    known = frame[frame["lineup_role_model"].isin(LINEUP_ROLES)].copy()
    if known["lineup_role_model"].nunique() < 2:
        raise ValueError("Both recorded lineup roles are required")

    known[event_col] = (
        pd.to_numeric(known[event_col], errors="coerce").fillna(0).astype(int)
    )
    known[MINUTES_COL] = pd.to_numeric(known[MINUTES_COL], errors="coerce")
    known["_event_minutes"] = known[MINUTES_COL] * known[event_col]
    known["_nonevent_n"] = 1 - known[event_col]
    known["_nonevent_minutes"] = known[MINUTES_COL] * known["_nonevent_n"]
    grouped = (
        known.groupby([PLAYER_ID_COL, "lineup_role_model"], observed=False)
        .agg(
            row_n=(event_col, "size"),
            event_sum=("_event_minutes", "sum"),
            event_n=(event_col, "sum"),
            nonevent_sum=("_nonevent_minutes", "sum"),
            nonevent_n=("_nonevent_n", "sum"),
        )
        .unstack("lineup_role_model", fill_value=0)
    )
    ordered_columns = [
        (metric, role)
        for role in LINEUP_ROLES
        for metric in (
            "row_n",
            "event_sum",
            "event_n",
            "nonevent_sum",
            "nonevent_n",
        )
    ]
    grouped = grouped.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns), fill_value=0)
    arrays = grouped.to_numpy(dtype=float)

    def summarize_totals(
        totals: np.ndarray,
    ) -> tuple[dict[str, dict[str, float]], float]:
        total_rows = float(sum(totals[index * 5] for index in range(len(LINEUP_ROLES))))
        role_stats: dict[str, dict[str, float]] = {}
        event_standardized = 0.0
        nonevent_standardized = 0.0
        for index, role in enumerate(LINEUP_ROLES):
            row_n, event_sum, event_n, nonevent_sum, nonevent_n = totals[
                index * 5 : index * 5 + 5
            ]
            if event_n <= 0 or nonevent_n <= 0 or total_rows <= 0:
                raise ValueError("Each lineup role requires event and non-event rows")
            weight = float(row_n / total_rows)
            event_mean = float(event_sum / event_n)
            nonevent_mean = float(nonevent_sum / nonevent_n)
            role_stats[role] = {
                "weight": weight,
                "event_rows": float(event_n),
                "non_event_rows": float(nonevent_n),
                "event_mean": event_mean,
                "non_event_mean": nonevent_mean,
                "difference": event_mean - nonevent_mean,
            }
            event_standardized += weight * event_mean
            nonevent_standardized += weight * nonevent_mean
        return role_stats, event_standardized - nonevent_standardized

    observed_roles, observed_standardized = summarize_totals(arrays.sum(axis=0))
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    n_players = len(grouped)
    for replicate in range(1, replicates + 1):
        counts = rng.multinomial(n_players, np.full(n_players, 1.0 / n_players))
        try:
            role_stats, standardized = summarize_totals(counts @ arrays)
        except ValueError:
            continue
        row = {
            "replicate": replicate,
            "lineup_standardized_difference": standardized,
        }
        for role in LINEUP_ROLES:
            row[f"{role}_difference"] = role_stats[role]["difference"]
        bootstrap_rows.append(row)
    samples = pd.DataFrame(bootstrap_rows)
    if len(samples) < max(2, int(0.9 * replicates)):
        raise ValueError("Too few estimable lineup bootstrap replicates")
    summary_rows = []
    for role in LINEUP_ROLES:
        stats = observed_roles[role]
        values = samples[f"{role}_difference"]
        summary_rows.append(
            {
                "comparison": role,
                "lineup_role": role,
                "event_rows": int(stats["event_rows"]),
                "non_event_rows": int(stats["non_event_rows"]),
                "event_mean_minutes": stats["event_mean"],
                "non_event_mean_minutes": stats["non_event_mean"],
                "event_minus_non_event_minutes": stats["difference"],
                "difference_ci_low": float(values.quantile(0.025)),
                "difference_ci_high": float(values.quantile(0.975)),
                "bootstrap_replicates_requested": int(replicates),
                "bootstrap_replicates_estimable": int(len(samples)),
            }
        )
    standardized_values = samples["lineup_standardized_difference"]
    summary_rows.append(
        {
            "comparison": "lineup_standardized",
            "lineup_role": "pooled recorded lineup mix",
            "event_rows": int(known[event_col].sum()),
            "non_event_rows": int(known[event_col].eq(0).sum()),
            "event_mean_minutes": np.nan,
            "non_event_mean_minutes": np.nan,
            "event_minus_non_event_minutes": observed_standardized,
            "difference_ci_low": float(standardized_values.quantile(0.025)),
            "difference_ci_high": float(standardized_values.quantile(0.975)),
            "bootstrap_replicates_requested": int(replicates),
            "bootstrap_replicates_estimable": int(len(samples)),
        }
    )
    return samples, pd.DataFrame(summary_rows)


def add_senior_national_exposure(
    frame: pd.DataFrame,
    exposure_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add frozen senior competitive national minutes to club-only burden."""
    national_col = "senior_competitive_national_only_minutes_last_7d"
    _require_columns(frame, [PLAYER_ID_COL, "date", BURDEN_COL], "club exposure frame")
    _require_columns(
        exposure_features,
        [PLAYER_ID_COL, "date", national_col],
        "national exposure features",
    )
    features = exposure_features[[PLAYER_ID_COL, "date", national_col]].copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    if features.duplicated([PLAYER_ID_COL, "date"]).any():
        raise ValueError("National exposure features must be unique by player-date")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.merge(features, on=[PLAYER_ID_COL, "date"], how="left", validate="many_to_one")
    national_minutes = pd.to_numeric(out[national_col], errors="coerce").fillna(0.0)
    club_minutes = pd.to_numeric(out[BURDEN_COL], errors="coerce")
    out["club_only_minutes_last_7d"] = club_minutes
    out["senior_national_minutes_last_7d"] = national_minutes
    out[BURDEN_COL] = club_minutes + national_minutes
    changed = national_minutes.gt(0.0)
    audit = pd.DataFrame(
        [
            {
                "scope": "club_plus_senior_competitive_national",
                "match_rows": int(len(out)),
                "rows_with_added_national_minutes": int(changed.sum()),
                "percent_rows_with_added_national_minutes": 100.0 * float(changed.mean()),
                "players_with_added_national_minutes": int(
                    out.loc[changed, PLAYER_ID_COL].nunique()
                ),
                "total_added_national_minutes": float(national_minutes.sum()),
                "maximum_added_national_minutes_last_7d": float(national_minutes.max()),
                "exposure_definition": (
                    "recorded club minutes plus senior competitive national minutes"
                ),
            }
        ]
    )
    return out, audit


def cohort_robustness_suite(
    cohorts: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply one model and contrast to eligibility, season, and scope cohorts."""
    if not cohorts:
        raise ValueError("At least one robustness cohort is required")
    tests = []
    contrasts = []
    audits = []
    for cohort_id, unscaled in cohorts.items():
        _require_columns(
            unscaled,
            [PLAYER_ID_COL, "date", SAME_DAY_COL, HISTORY_COL, BURDEN_COL],
            f"{cohort_id} cohort",
        )
        if unscaled.empty or int(unscaled[SAME_DAY_COL].sum()) <= 0:
            raise ValueError(f"No estimable rows for robustness cohort {cohort_id}")
        scaling = history_scale(unscaled)
        frame = apply_history_scale(unscaled, scaling)
        model_id = f"cohort_{cohort_id}"
        fitted_tests, fitted_contrast = _model_test_and_contrast_rows(
            frame, scaling, SAME_DAY_COL, model_id
        )
        fitted_tests["cohort_id"] = cohort_id
        fitted_contrast["cohort_id"] = cohort_id
        tests.append(fitted_tests)
        contrasts.append(fitted_contrast)
        dates = pd.to_datetime(frame["date"], errors="coerce")
        audits.append(
            {
                "cohort_id": cohort_id,
                "match_rows": int(len(frame)),
                "players": int(frame[PLAYER_ID_COL].nunique()),
                "same_day_events": int(frame[SAME_DAY_COL].sum()),
                "date_min": dates.min(),
                "date_max": dates.max(),
                "minimum_prior_minutes_observed": float(
                    pd.to_numeric(frame["prior_minutes_played"], errors="coerce").min()
                ),
                "median_history_rate_per_10000_prior_minutes": float(
                    scaling["median_rate"]
                ),
            }
        )
    return (
        pd.concat(tests, ignore_index=True),
        pd.concat(contrasts, ignore_index=True),
        pd.DataFrame(audits),
    )


def selection_standardization(
    frame: pd.DataFrame,
    scaling: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare unadjusted and composition-standardised same-day curves."""
    required = [
        PLAYER_ID_COL,
        SAME_DAY_COL,
        BURDEN_COL,
        HISTORY_MODEL_COL,
        "date",
        "lineup_role_model",
        "returned_from_recorded_injury_within_14d",
        *CALENDAR_TERMS,
    ]
    _require_columns(frame, required, "selection-standardization frame")
    selected = frame[frame["lineup_role_model"].isin(LINEUP_ROLES)].copy()
    if selected.empty or selected["lineup_role_model"].nunique() < 2:
        raise ValueError("Both recorded lineup roles are required")
    unadjusted = fit_continuous_model(selected, SAME_DAY_COL, "per_appearance")
    adjusted = fit_continuous_model(
        selected,
        SAME_DAY_COL,
        "per_appearance",
        " + C(lineup_role_model) + returned_from_recorded_injury_within_14d",
    )
    covariates = [
        *CALENDAR_TERMS,
        "lineup_role_model",
        "returned_from_recorded_injury_within_14d",
    ]
    reference = (
        selected.groupby(covariates, dropna=False, observed=False)
        .size()
        .rename("standardization_weight")
        .reset_index()
    )
    rows = []
    for anchor, key in (
        ("low", "low_log"),
        ("median", "median_log"),
        ("high", "high_log"),
    ):
        history_scaled = (
            float(scaling[key]) - float(scaling["center_log"])
        ) / float(scaling["scale_log_iqr"])
        for burden in PREDICTION_GRID:
            for model_name, result in (
                ("unadjusted_lineup_known", unadjusted),
                ("standardized_lineup_and_return", adjusted),
            ):
                rows.append(
                    {
                        "model": model_name,
                        "history_anchor": anchor,
                        "history_rate_per_10000_prior_minutes": float(
                            scaling[f"{anchor}_rate"]
                        ),
                        BURDEN_COL: burden,
                        **_weighted_standardized_prediction(
                            result, reference, burden, history_scaled
                        ),
                    }
                )
    curves = pd.DataFrame(rows)
    comparisons = []
    for anchor in ("low", "median", "high"):
        for burden in (0.0, 45.0, 90.0, 180.0):
            subset = curves[
                curves["history_anchor"].eq(anchor)
                & curves[BURDEN_COL].eq(burden)
            ].set_index("model")
            unadjusted_value = float(
                subset.loc[
                    "unadjusted_lineup_known", "estimate_per_1000_appearances"
                ]
            )
            adjusted_value = float(
                subset.loc[
                    "standardized_lineup_and_return",
                    "estimate_per_1000_appearances",
                ]
            )
            comparisons.append(
                {
                    "history_anchor": anchor,
                    BURDEN_COL: burden,
                    "unadjusted_per_1000_appearances": unadjusted_value,
                    "standardized_per_1000_appearances": adjusted_value,
                    "absolute_change_per_1000_appearances": (
                        adjusted_value - unadjusted_value
                    ),
                    "relative_change_percent": (
                        100.0 * (adjusted_value / unadjusted_value - 1.0)
                        if unadjusted_value > 0.0
                        else np.nan
                    ),
                }
            )
    test_frames = []
    for model_name, result in (
        ("unadjusted_lineup_known", unadjusted),
        ("adjusted_lineup_and_return", adjusted),
    ):
        tests = formal_model_tests(result, model_name)
        tests["n_match_rows"] = int(len(selected))
        tests["n_players"] = int(selected[PLAYER_ID_COL].nunique())
        tests["n_events"] = int(selected[SAME_DAY_COL].sum())
        test_frames.append(tests)
    selection_seasons = season_start_year(selected["date"])
    first_season = int(selection_seasons.min())
    last_season = int(selection_seasons.max())
    scope = (
        f"complete lineup ascertainment seasons {first_season}-{last_season}"
    )
    comparison_table = pd.DataFrame(comparisons)
    test_table = pd.concat(test_frames, ignore_index=True)
    for table in (curves, comparison_table, test_table):
        table["selection_scope"] = scope
        table["selection_first_season"] = first_season
        table["selection_last_season"] = last_season
    return curves, comparison_table, test_table


def fit_two_way_continuous_model(
    frame: pd.DataFrame,
    event_col: str,
    first_cluster: str,
    second_cluster: str,
):
    """Fit the primary logit and replace covariance with two-way clustering."""
    _require_columns(
        frame,
        [
            first_cluster,
            second_cluster,
            BURDEN_COL,
            HISTORY_MODEL_COL,
            event_col,
            *CALENDAR_TERMS,
        ],
        "two-way cluster frame",
    )
    if frame[first_cluster].nunique() < 2 or frame[second_cluster].nunique() < 2:
        raise ValueError("Two-way clustering requires at least two clusters per dimension")
    formula = continuous_formula(event_col, float(frame[BURDEN_COL].max()))
    result = smf.glm(
        formula=formula,
        data=frame,
        family=sm.families.Binomial(),
    ).fit()
    first_codes = pd.factorize(frame[first_cluster], sort=True)[0]
    second_codes = pd.factorize(frame[second_cluster], sort=True)[0]
    covariance, _, _ = cov_cluster_2groups(
        result, first_codes, second_codes
    )
    result.cov_params_default = covariance
    result._cache.clear()
    return result


def _context_model_contrast(result: Any, frame: pd.DataFrame) -> dict[str, float]:
    """Estimate the median-history 0-to-180 contrast for a context model."""
    first = frame.iloc[[0]].copy()
    first[BURDEN_COL] = 0.0
    first[HISTORY_MODEL_COL] = 0.0
    for term in CALENDAR_TERMS:
        first[term] = 0.0
    second = first.copy()
    second[BURDEN_COL] = 180.0
    return design_contrast(result, first, second, "odds_ratio")


def context_sensitivity_analysis(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Show covariate, competition, restriction, and cluster sensitivities."""
    _require_columns(
        frame,
        [
            "age_years",
            "position_group",
            "club_season",
            "competition_context",
            "match_cluster_id",
            SAME_DAY_COL,
            PLAYER_ID_COL,
        ],
        "context sensitivity frame",
    )
    specifications = [
        (
            "age_position_clubseason_adjusted",
            frame,
            " + age_years + C(position_group) + C(club_season)",
            "player_cluster",
        ),
        (
            "competition_context_adjusted",
            frame,
            " + C(competition_context)",
            "player_cluster",
        ),
        (
            "premier_league_current_match_only",
            frame[frame["competition_context"].eq("Premier League")].copy(),
            "",
            "player_cluster",
        ),
        (
            "two_way_player_match_cluster",
            frame,
            "",
            "two_way_cluster",
        ),
    ]
    outputs = []
    contrasts = []
    for model_id, subset, controls, covariance in specifications:
        if subset.empty or int(subset[SAME_DAY_COL].sum()) <= 0:
            raise ValueError(f"No estimable rows for {model_id}")
        if covariance == "two_way_cluster":
            result = fit_two_way_continuous_model(
                subset, SAME_DAY_COL, PLAYER_ID_COL, "match_cluster_id"
            )
        else:
            result = fit_continuous_model(
                subset, SAME_DAY_COL, "per_appearance", controls
            )
        tests = formal_model_tests(result, model_id)
        tests["covariance"] = covariance
        tests["controls"] = controls.strip(" +")
        tests["n_match_rows"] = int(len(subset))
        tests["n_players"] = int(subset[PLAYER_ID_COL].nunique())
        tests["n_events"] = int(subset[SAME_DAY_COL].sum())
        outputs.append(tests)
        contrasts.append(
            {
                "model_id": model_id,
                "history_anchor": "median",
                "burden_from": 0.0,
                "burden_to": 180.0,
                "contrast_id": "median_history_180_vs_0",
                "effect_measure": "odds_ratio",
                **_context_model_contrast(result, subset),
                "covariance": covariance,
                "controls": controls.strip(" +"),
                "n_match_rows": int(len(subset)),
                "n_players": int(subset[PLAYER_ID_COL].nunique()),
                "n_events": int(subset[SAME_DAY_COL].sum()),
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(contrasts)


def context_sensitivity_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the formal tests from :func:`context_sensitivity_analysis`."""
    tests, _ = context_sensitivity_analysis(frame)
    return tests


def cohort_flow_table(
    source_reports: pd.DataFrame,
    episodes: pd.DataFrame,
    raw_match_panel: pd.DataFrame,
    frame: pd.DataFrame,
    minimum_prior_minutes: float = REFERENCE_MINIMUM_PRIOR_MINUTES,
) -> pd.DataFrame:
    """Describe derivation and mutually exclusive appearance exclusions."""
    _require_columns(source_reports, [PLAYER_ID_COL], "source reports")
    _require_columns(episodes, [PLAYER_ID_COL], "injury episodes")
    _require_columns(
        raw_match_panel,
        [
            PLAYER_ID_COL,
            "date",
            "available_for_injury_risk",
            MINUTES_COL,
            "prior_minutes_played",
            BURDEN_COL,
            HISTORY_COL,
            SAME_DAY_COL,
            LAG1_COL,
            COMBINED_COL,
            *CALENDAR_TERMS,
        ],
        "raw match panel",
    )
    _require_columns(
        frame,
        [PLAYER_ID_COL, SAME_DAY_COL, LAG1_COL, COMBINED_COL],
        "match frame",
    )
    raw = raw_match_panel.copy()
    rows: list[dict[str, object]] = []

    def append_row(
        stage: str,
        subset: pd.DataFrame,
        row_type: str,
        mutually_exclusive: bool,
        remaining: int | float,
    ) -> None:
        rows.append(
            {
                "stage_order": len(rows) + 1,
                "row_type": row_type,
                "stage": stage,
                "records": int(len(subset)),
                "players": int(subset[PLAYER_ID_COL].nunique()),
                "same_day_events": (
                    int(pd.to_numeric(subset[SAME_DAY_COL], errors="coerce").fillna(0).sum())
                    if SAME_DAY_COL in subset
                    else np.nan
                ),
                "lag1_events": (
                    int(pd.to_numeric(subset[LAG1_COL], errors="coerce").fillna(0).sum())
                    if LAG1_COL in subset
                    else np.nan
                ),
                "combined_proxy_events": (
                    int(pd.to_numeric(subset[COMBINED_COL], errors="coerce").fillna(0).sum())
                    if COMBINED_COL in subset
                    else np.nan
                ),
                "mutually_exclusive_exclusion": mutually_exclusive,
                "appearance_rows_remaining": remaining,
            }
        )

    append_row(
        "cleaned public injury reports",
        source_reports,
        "outcome derivation",
        False,
        np.nan,
    )
    append_row(
        "reconciled non-overlapping reported episodes",
        episodes,
        "outcome derivation",
        False,
        np.nan,
    )
    remaining = raw.copy()
    append_row(
        "all reconstructed player-appearance rows",
        remaining,
        "appearance risk set",
        False,
        len(remaining),
    )
    minutes = pd.to_numeric(remaining[MINUTES_COL], errors="coerce")
    excluded = remaining[minutes.isna() | minutes.le(0.0)]
    remaining = remaining.drop(index=excluded.index)
    append_row(
        "excluded: no positive recorded appearance minutes",
        excluded,
        "exclusion",
        True,
        len(remaining),
    )
    available = remaining["available_for_injury_risk"].fillna(False).astype(bool)
    excluded = remaining[~available]
    remaining = remaining.drop(index=excluded.index)
    append_row(
        "excluded: not in the reconstructed injury-risk set",
        excluded,
        "exclusion",
        True,
        len(remaining),
    )
    prior_minutes = pd.to_numeric(remaining["prior_minutes_played"], errors="coerce")
    excluded = remaining[
        prior_minutes.isna() | prior_minutes.lt(float(minimum_prior_minutes))
    ]
    remaining = remaining.drop(index=excluded.index)
    append_row(
        f"excluded: fewer than {int(minimum_prior_minutes)} prior recorded match minutes",
        excluded,
        "exclusion",
        True,
        len(remaining),
    )
    complete_columns = ["date", BURDEN_COL, HISTORY_COL, *CALENDAR_TERMS]
    complete = remaining[complete_columns].copy()
    complete["date"] = pd.to_datetime(complete["date"], errors="coerce")
    for column in complete_columns[1:]:
        complete[column] = pd.to_numeric(complete[column], errors="coerce")
    invalid_numeric = (
        pd.to_numeric(complete[BURDEN_COL], errors="coerce").lt(0.0)
        | pd.to_numeric(complete[HISTORY_COL], errors="coerce").lt(0.0)
    )
    excluded = remaining[complete.isna().any(axis=1) | invalid_numeric]
    remaining = remaining.drop(index=excluded.index)
    append_row(
        "excluded: incomplete or invalid date, exposure, history, or calendar covariate",
        excluded,
        "exclusion",
        True,
        len(remaining),
    )
    if len(remaining) != len(frame):
        raise ValueError(
            "Flow exclusions do not reproduce the prepared primary analysis rows"
        )
    append_row(
        "included: primary same-day per-appearance analysis",
        frame,
        "analysis cohort",
        False,
        len(frame),
    )
    return pd.DataFrame(rows)


def cohort_descriptive_table(
    frame: pd.DataFrame,
    national_appearances: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return conventional player, exposure, event and missingness descriptors."""
    required = [
        PLAYER_ID_COL,
        "date",
        "fragility_group",
        MINUTES_COL,
        BURDEN_COL,
        SAME_DAY_COL,
        LAG1_COL,
        COMBINED_COL,
        "matchproxy_injury_desc",
    ]
    _require_columns(frame, required, "cohort descriptive frame")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    player = work.sort_values("date").groupby(PLAYER_ID_COL, as_index=False).first()
    rows: list[dict[str, object]] = []

    def add_continuous(section: str, metric: str, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        rows.append(
            {
                "section": section,
                "metric": metric,
                "n": int(len(numeric)),
                "value": float(numeric.median()) if len(numeric) else np.nan,
                "lower": (
                    float(numeric.quantile(0.25)) if len(numeric) else np.nan
                ),
                "upper": (
                    float(numeric.quantile(0.75)) if len(numeric) else np.nan
                ),
                "display": (
                    f"{numeric.median():.1f} "
                    f"({numeric.quantile(0.25):.1f}--{numeric.quantile(0.75):.1f})"
                    if len(numeric)
                    else "not available"
                ),
                "summary_type": "median_iqr",
            }
        )

    if "age_years" in player:
        add_continuous(
            "players", "age at first eligible appearance, years", player["age_years"]
        )
    appearances = work.groupby(PLAYER_ID_COL).size()
    minutes = work.groupby(PLAYER_ID_COL)[MINUTES_COL].sum()
    work["season_start"] = np.where(
        work["date"].dt.month.ge(7),
        work["date"].dt.year,
        work["date"].dt.year - 1,
    )
    seasons = work.groupby(PLAYER_ID_COL)["season_start"].nunique()
    add_continuous("players", "recorded club appearances per player", appearances)
    add_continuous("players", "recorded club match minutes per player", minutes)
    add_continuous("players", "seasons represented per player", seasons)
    if national_appearances is not None:
        _require_columns(
            national_appearances,
            [PLAYER_ID_COL, "is_senior_competitive"],
            "national appearances",
        )
        cohort_ids = set(work[PLAYER_ID_COL].unique())
        national = national_appearances[
            national_appearances[PLAYER_ID_COL].isin(cohort_ids)
            & national_appearances["is_senior_competitive"].fillna(False).astype(bool)
        ]
        national_counts = national.groupby(PLAYER_ID_COL).size().reindex(
            sorted(cohort_ids), fill_value=0
        )
        add_continuous(
            "players",
            "recorded senior competitive national appearances per player",
            national_counts,
        )
    add_continuous(
        "appearances", "previous-7-day match minutes", work[BURDEN_COL]
    )
    add_continuous(
        "appearances", "current recorded match minutes", work[MINUTES_COL]
    )

    for group, subset in work.groupby("fragility_group", observed=False):
        rows.append(
            {
                "section": "history strata (descriptive only)",
                "metric": HISTORY_PUBLICATION_LABELS.get(str(group), str(group)),
                "n": int(len(subset)),
                "value": int(subset[PLAYER_ID_COL].nunique()),
                "lower": np.nan,
                "upper": np.nan,
                "display": (
                    f"{len(subset):,} appearances; "
                    f"{subset[PLAYER_ID_COL].nunique():,} players ever represented"
                ),
                "summary_type": "rows_and_players",
            }
        )
    if "position_group" in player:
        position_counts = player["position_group"].fillna("Unknown").value_counts()
        for position, count in position_counts.items():
            rows.append(
                {
                    "section": "playing position",
                    "metric": str(position),
                    "n": int(count),
                    "value": 100.0 * float(count) / len(player),
                    "lower": np.nan,
                    "upper": np.nan,
                    "display": f"{int(count):,} ({100.0 * count / len(player):.1f}%)",
                    "summary_type": "count_percent_players",
                }
            )
    for event_col, label in (
        (SAME_DAY_COL, "same-day reported events"),
        (LAG1_COL, "lag-1 reported events"),
        (COMBINED_COL, "combined match-associated proxy events"),
        (
            SAME_DAY_SEVERE_COL,
            "same-day reports with at least 28 reported days",
        ),
        (SAME_DAY_MUSCLE_COL, "same-day muscle/tendon reports"),
    ):
        if event_col not in work:
            continue
        count = int(work[event_col].sum())
        rows.append(
            {
                "section": "outcomes",
                "metric": label,
                "n": int(len(work)),
                "value": count,
                "lower": np.nan,
                "upper": np.nan,
                "display": (
                    f"{count:,} / {len(work):,} appearances "
                    f"({100.0 * count / len(work):.2f}%)"
                ),
                "summary_type": "count_percent_appearances",
            }
        )
    if "lineup_role_model" in work:
        for role, count in work["lineup_role_model"].fillna("missing").value_counts().items():
            role_rows = work[work["lineup_role_model"].fillna("missing").eq(role)]
            rows.append(
                {
                    "section": "lineup ascertainment",
                    "metric": str(role),
                    "n": int(len(work)),
                    "value": int(count),
                    "lower": np.nan,
                    "upper": np.nan,
                    "display": (
                        f"{int(count):,} / {len(work):,} appearances "
                        f"({100.0 * count / len(work):.1f}%); "
                        f"{int(role_rows[SAME_DAY_COL].sum()):,} same-day reports"
                    ),
                    "summary_type": "count_percent_appearances_and_events",
                }
            )
    event_rows = work[work[COMBINED_COL].eq(1)]
    descriptions = (
        event_rows["matchproxy_injury_desc"].fillna("").astype(str).str.strip()
    )
    missing = int(descriptions.eq("").sum())
    rows.append(
        {
            "section": "outcome description",
            "metric": "missing or blank injury description",
            "n": int(len(event_rows)),
            "value": missing,
            "lower": np.nan,
            "upper": np.nan,
            "display": (
                f"{missing:,} / {len(event_rows):,} events "
                f"({100.0 * missing / max(len(event_rows), 1):.1f}%)"
            ),
            "summary_type": "count_percent_events",
        }
    )
    if "matchproxy_public_injury_type" in event_rows:
        for injury_type, count in (
            event_rows["matchproxy_public_injury_type"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .items()
        ):
            rows.append(
                {
                    "section": "reported event type",
                    "metric": str(injury_type),
                    "n": int(len(event_rows)),
                    "value": int(count),
                    "lower": np.nan,
                    "upper": np.nan,
                    "display": (
                        f"{int(count):,} / {len(event_rows):,} events "
                        f"({100.0 * count / max(len(event_rows), 1):.1f}%)"
                    ),
                    "summary_type": "count_percent_events",
                }
            )
    return pd.DataFrame(rows)


def _holm_adjust(values: pd.Series) -> pd.Series:
    """Return Holm-adjusted p-values while preserving missing rows."""
    numeric = pd.to_numeric(values, errors="coerce")
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    finite = numeric.dropna().sort_values()
    if finite.empty:
        return adjusted
    running = 0.0
    count = len(finite)
    for rank, (index, value) in enumerate(finite.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted.loc[index] = running
    return adjusted


def hypothesis_register(
    primary_tests: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one row for every formal test, including non-estimable tests."""
    _require_columns(
        primary_tests, ["model_id", "contrast_id", "p_value"], "primary tests"
    )
    frames = []
    primary = primary_tests.copy()
    is_primary_model = primary["model_id"].eq(
        "primary_same_day_per_appearance"
    )
    is_reference_family_test = primary["contrast_id"].isin(
        [
            "global_recent_exposure_association_at_median_history",
            "global_recent_exposure_by_continuous_history_interaction",
        ]
    )
    is_reference_association = primary["contrast_id"].eq(
        "global_recent_exposure_association_at_median_history"
    )
    is_history_interaction = primary["contrast_id"].eq(
        "global_recent_exposure_by_continuous_history_interaction"
    )
    primary["family"] = np.select(
        [
            is_primary_model & is_reference_family_test,
            is_primary_model,
        ],
        [
            "reference_model_two_test_family",
            "primary_model_component_family",
        ],
        default="timing_and_denominator_sensitivity_family",
    )
    primary["analysis_role"] = np.select(
        [
            is_primary_model & is_reference_association,
            is_primary_model & is_history_interaction,
            is_primary_model,
        ],
        [
            "primary_reference_association",
            "secondary_effect_modification",
            "primary_model_component",
        ],
        default="timing_or_denominator_sensitivity",
    )
    primary["test_domain"] = np.where(
        primary["contrast_id"].str.contains("interaction", na=False),
        "effect_modification",
        "recent_exposure_association",
    )
    primary["source_file"] = "jsams_primary_model_tests.csv"
    primary["source_row"] = np.arange(1, len(primary) + 1)
    frames.append(primary)

    source_specs = {
        "exposure": (
            "historical_exposure_response_family",
            "secondary_or_exploratory",
            "contrast_id",
            "p_value",
        ),
        "type_history": (
            "type_history_family",
            "secondary_or_exploratory",
            "test_id",
            "p_value",
        ),
        "national_status": (
            "national_status_family",
            "post_hoc_sensitivity",
            "contrast_id",
            "p_value",
        ),
        "recovery": (
            "recovery_interval_family",
            "secondary_or_exploratory",
            "contrast",
            "p_value",
        ),
        "lineup": (
            "lineup_interaction_family",
            "exploratory_selection_audit",
            "model",
            "p_value",
        ),
        "two_way_cluster": (
            "two_way_cluster_uncertainty_family",
            "uncertainty_sensitivity",
            "contrast",
            "p_value",
        ),
        "primary_contrasts": (
            "reviewer_requested_continuous_model_contrasts",
            "post_hoc_reviewer_requested_display_contrasts",
            "contrast_id",
            "p_value",
        ),
        "selection_standardized": (
            "lineup_return_standardization_family",
            "exploratory_selection_standardization",
            "contrast_id",
            "p_value",
        ),
        "context_sensitivity": (
            "primary_context_sensitivity_family",
            "covariate_context_cluster_sensitivity",
            "contrast_id",
            "p_value",
        ),
        "functional_form": (
            "reviewer_requested_functional_form_family",
            "post_hoc_functional_form_sensitivity",
            "contrast_id",
            "p_value",
        ),
        "outcome_quality": (
            "reviewer_requested_outcome_quality_family",
            "post_hoc_outcome_quality_sensitivity",
            "contrast_id",
            "p_value",
        ),
        "lineup_role": (
            "reviewer_requested_lineup_role_family",
            "post_hoc_lineup_role_sensitivity",
            "contrast_id",
            "p_value",
        ),
        "selection_effect": (
            "reviewer_requested_selection_effect_family",
            "post_hoc_selection_standardization_contrast",
            "contrast_id",
            "p_value",
        ),
        "cohort_robustness": (
            "reviewer_requested_cohort_robustness_family",
            "post_hoc_eligibility_season_scope_sensitivity",
            "contrast_id",
            "p_value",
        ),
        "within_player": (
            "reviewer_requested_within_player_family",
            "post_hoc_within_player_sensitivity",
            "model_id",
            "p_value",
        ),
    }
    for key, frame in sources.items():
        if key not in source_specs:
            raise KeyError(f"Unknown hypothesis-register source: {key}")
        family, role, id_col, p_col = source_specs[key]
        _require_columns(frame, [id_col, p_col], f"{key} hypothesis source")
        registered = frame.copy()
        registered["family"] = family
        registered["analysis_role"] = role
        registered["contrast_id"] = registered[id_col].astype(str)
        registered["p_value"] = pd.to_numeric(
            registered[p_col], errors="coerce"
        )
        registered["source_file"] = key
        registered["source_row"] = np.arange(1, len(registered) + 1)
        if key == "exposure":
            registered["test_domain"] = np.where(
                registered["contrast_id"].isin(
                    [
                        "higher_vs_intermediate_at_0",
                        "higher_vs_intermediate_at_180",
                    ]
                ),
                "between_history_level_contrast",
                "exposure_response_or_effect_modification",
            )
        elif key == "primary_contrasts":
            registered["test_domain"] = np.where(
                registered["model_id"].eq(
                    "primary_same_day_per_appearance"
                ),
                "primary_model_post_hoc_contrast",
                "timing_or_denominator_post_hoc_contrast",
            )
        elif key in {
            "functional_form",
            "outcome_quality",
            "lineup_role",
            "selection_effect",
            "cohort_robustness",
            "within_player",
        }:
            registered["test_domain"] = key
        else:
            registered["test_domain"] = key
        frames.append(registered)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.insert(
        0,
        "hypothesis_id",
        [f"H{value:04d}" for value in range(1, len(combined) + 1)],
    )
    combined["estimable"] = combined["p_value"].notna()
    combined["analysis_timing"] = "after outcome and model results were available"
    combined["confirmatory_status"] = "exploratory"
    combined["dated_prospective_analysis_plan_available"] = False
    combined["p_holm_within_family_recomputed"] = combined.groupby(
        "family", group_keys=False
    )["p_value"].apply(_holm_adjust)
    existing_adjustments = (
        "p_holm_across_specifications",
        "p_holm_type_history_family",
        "p_holm_status_family",
        "p_holm_lineup_family",
        "p_holm_extension_family",
    )
    combined["p_adjusted_reported"] = np.nan
    for column in existing_adjustments:
        if column in combined:
            combined["p_adjusted_reported"] = combined[
                "p_adjusted_reported"
            ].fillna(pd.to_numeric(combined[column], errors="coerce"))
    combined["p_adjusted_reported"] = combined[
        "p_adjusted_reported"
    ].fillna(combined["p_holm_within_family_recomputed"])
    combined["reject_adjusted_0_05"] = (
        combined["p_adjusted_reported"].lt(0.05).fillna(False)
    )
    summary = (
        combined.groupby(
            ["family", "analysis_role", "test_domain"], dropna=False
        )
        .agg(
            registered_tests=("hypothesis_id", "size"),
            estimable_tests=("estimable", "sum"),
            adjusted_rejections=("reject_adjusted_0_05", "sum"),
            minimum_unadjusted_p=("p_value", "min"),
            minimum_adjusted_p=("p_adjusted_reported", "min"),
        )
        .reset_index()
    )
    return combined, summary


def headline_inference_audit(register: pd.DataFrame) -> pd.DataFrame:
    """Separate the global spline test from the post hoc anchor contrast."""
    required = [
        "model_id",
        "contrast_id",
        "family",
        "p_value",
        "p_adjusted_reported",
        "estimate",
        "ci_low",
        "ci_high",
    ]
    _require_columns(register, required, "headline inference register")

    def one(contrast_id: str) -> pd.Series:
        selected = register[
            register["model_id"].eq("primary_same_day_per_appearance")
            & register["contrast_id"].eq(contrast_id)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one headline inference row for {contrast_id}"
            )
        return selected.iloc[0]

    global_test = one("global_recent_exposure_association_at_median_history")
    anchor = one("median_history_180_vs_0")
    family_sizes = register.groupby("family", observed=False).size()
    return pd.DataFrame(
        [
            {
                "inference_target": "global six-term exposure spline at median history",
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value_raw": float(global_test["p_value"]),
                "p_value_holm_within_family": float(
                    global_test["p_adjusted_reported"]
                ),
                "multiplicity_family": str(global_test["family"]),
                "family_size": int(family_sizes.loc[global_test["family"]]),
                "reporting_rule": (
                    "This p value tests the whole spline and must not be attached "
                    "to the 0-to-180 odds ratio."
                ),
            },
            {
                "inference_target": "post hoc median-history 0-to-180-minute anchor contrast",
                "estimate": float(anchor["estimate"]),
                "ci_low": float(anchor["ci_low"]),
                "ci_high": float(anchor["ci_high"]),
                "p_value_raw": float(anchor["p_value"]),
                "p_value_holm_within_family": float(
                    anchor["p_adjusted_reported"]
                ),
                "multiplicity_family": str(anchor["family"]),
                "family_size": int(family_sizes.loc[anchor["family"]]),
                "reporting_rule": (
                    "Report the exploratory odds ratio and 95% interval without "
                    "borrowing the global spline p value."
                ),
            },
        ]
    )


def model_specification_table(
    frame: pd.DataFrame,
    scaling: Mapping[str, float],
) -> pd.DataFrame:
    """Record each estimand, formula, support, standardization, and clustering."""
    burden_max = float(frame[BURDEN_COL].max())
    specifications = (
        ("primary_same_day_per_appearance", SAME_DAY_COL, "per_appearance", "primary"),
        ("lag1_per_appearance", LAG1_COL, "per_appearance", "timing sensitivity"),
        ("combined_per_appearance", COMBINED_COL, "per_appearance", "timing sensitivity"),
        ("same_day_observed_minutes", SAME_DAY_COL, "observed_minutes", "denominator bound"),
        ("same_day_fixed_90", SAME_DAY_COL, "fixed_90", "denominator bound"),
        ("lag1_observed_minutes", LAG1_COL, "observed_minutes", "timing and denominator sensitivity"),
        ("lag1_fixed_90", LAG1_COL, "fixed_90", "timing and denominator sensitivity"),
        ("combined_observed_minutes", COMBINED_COL, "observed_minutes", "timing and denominator sensitivity"),
        ("combined_fixed_90", COMBINED_COL, "fixed_90", "timing and denominator sensitivity"),
    )
    rows = []
    for model_id, event_col, denominator, role in specifications:
        if denominator == "per_appearance":
            estimand = "conditional reported-event probability per recorded appearance"
            family = "binomial logit"
            denominator_label = "recorded appearance"
        elif denominator == "observed_minutes":
            estimand = "conditional reported-event count per observed match minute"
            family = "Poisson log with log observed-minute offset"
            denominator_label = "observed recorded match minutes"
        else:
            estimand = (
                "conditional reported-event count per appearance expressed over "
                "a constant assigned 90-minute offset"
            )
            family = "Poisson log with fixed log(90) offset"
            denominator_label = "90 assigned minutes per appearance"
        rows.append(
            {
                "model_id": model_id,
                "analysis_role": role,
                "analysis_timing": "reviewer-requested post hoc estimand audit",
                "primary_question": (
                    "How stable is the association between previous-7-day club "
                    "match exposure and a same-day public report per appearance "
                    "under outcome, denominator, and selection checks?"
                ),
                "estimand": estimand,
                "outcome": event_col,
                "denominator": denominator_label,
                "link_and_family": family,
                "formula": continuous_formula(event_col, burden_max),
                "exposure_definition": (
                    "club competition minutes in the previous 7 days; current "
                    "appearance excluded"
                ),
                "absolute_prediction_standardization": (
                    "marginalized over the observed calendar-phase distribution"
                ),
                "spline_basis": "cubic B-spline",
                "interior_knots_minutes": ";".join(
                    str(int(value)) for value in SPLINE_KNOTS
                ),
                "lower_boundary_minutes": 0.0,
                "upper_boundary_minutes": burden_max,
                "display_range_minutes": "0--180",
                "anchor_choice": (
                    "0 and 180 minutes were chosen after data review as an "
                    "interpretable two-full-match contrast inside supported data"
                ),
                "anchor_confirmatory_status": "post hoc exploratory contrast",
                "history_measure": (
                    "log1p prior reports per 10,000 previous match minutes, "
                    "per row-weighted IQR"
                ),
                "history_observation_start": "2017-07-01 reconstructed cohort start",
                "history_transfer_tracking": (
                    "person-level Transfermarkt identifier; history is not reset at club transfer"
                ),
                "left_truncation": (
                    "events and minutes before 2017-07-01 are not observed"
                ),
                "minimum_prior_recorded_minutes": REFERENCE_MINIMUM_PRIOR_MINUTES,
                "history_effect_modification_role": "secondary",
                "history_center_log": float(scaling["center_log"]),
                "history_scale_log_iqr": float(scaling["scale_log_iqr"]),
                "cluster": "player",
                "fixed_90_algebra": (
                    "because every offset equals log(90), exposure coefficients "
                    "match a per-appearance Poisson model up to the intercept; "
                    "this is not independent time-at-risk evidence"
                    if denominator == "fixed_90"
                    else "not applicable"
                ),
                "calendar_controls": ";".join(CALENDAR_TERMS),
                "n_match_rows": int(len(frame)),
                "n_players": int(frame[PLAYER_ID_COL].nunique()),
                "n_events": int(frame[event_col].sum()),
            }
        )
    return pd.DataFrame(rows)


def enforce_claim_visibility(hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Apply the tier visibility rules and reject unjustified claims."""
    _require_columns(
        hierarchy,
        [
            "tier",
            "tier_justification",
            "abstract_recommended",
            "abstract_required_for_transparency",
            "main_display_recommended",
        ],
        "publication claim hierarchy",
    )
    out = hierarchy.copy()
    justification = out["tier_justification"].fillna("").astype(str).str.strip()
    if justification.eq("").any():
        raise ValueError("Every publication claim requires a tier justification")
    out["abstract_visible"] = (
        out["abstract_recommended"]
        | out["abstract_required_for_transparency"]
    )
    out["main_results_sentence_limit"] = np.where(
        out["tier"].ge(4), 1, np.nan
    )
    out["visibility_rule_passes"] = (
        (out["tier"].le(3) | ~out["abstract_visible"])
        & (out["tier"].le(3) | ~out["main_display_recommended"])
    )
    if not out["visibility_rule_passes"].all():
        raise ValueError(
            "A Tier 4 or Tier 5 claim was promoted to the abstract or a main display"
        )
    return out


def publication_claim_hierarchy(
    model_tests: pd.DataFrame,
    model_contrasts: pd.DataFrame,
    minute_summary: pd.DataFrame,
    lineup_minute_summary: pd.DataFrame,
    selection_effects: pd.DataFrame,
    functional_contrasts: pd.DataFrame,
    quality_contrasts: pd.DataFrame,
    cohort_contrasts: pd.DataFrame,
    hypothesis_summary: pd.DataFrame,
    headline_audit: pd.DataFrame,
    within_player: pd.DataFrame,
) -> pd.DataFrame:
    """Rank the reviewer-facing claims and enforce transparent visibility.

    The project-wide novelty tiers remain conservative: Tier 2 is an original
    but measurement-sensitive result, while Tier 5 is a null or otherwise
    uninformative result. The paper is explicitly framed as a public-data
    measurement study, so effect modification is reported in the main model
    table without being promoted to the abstract.
    """
    _require_columns(
        model_tests,
        ["model_id", "contrast_id", "p_value"],
        "claim-hierarchy model tests",
    )
    _require_columns(
        model_contrasts,
        [
            "model_id",
            "history_anchor",
            "burden_from",
            "burden_to",
            "estimate",
            "ci_low",
            "ci_high",
        ],
        "claim-hierarchy model contrasts",
    )
    _require_columns(
        minute_summary,
        [
            "event_minus_non_event_minutes",
            "difference_ci_low",
            "difference_ci_high",
            "bootstrap_replicates",
        ],
        "claim-hierarchy minute summary",
    )
    _require_columns(
        lineup_minute_summary,
        ["comparison", "event_minus_non_event_minutes", "difference_ci_low", "difference_ci_high"],
        "claim-hierarchy lineup minute summary",
    )
    _require_columns(
        selection_effects,
        ["history_anchor", "burden_to", "comparison_type", "estimate", "ci_low", "ci_high"],
        "claim-hierarchy selection effects",
    )
    for frame, label, identifier in (
        (functional_contrasts, "functional contrasts", "model_id"),
        (quality_contrasts, "quality contrasts", "model_id"),
        (cohort_contrasts, "cohort contrasts", "cohort_id"),
    ):
        _require_columns(
            frame,
            [identifier, "estimate", "ci_low", "ci_high"],
            f"claim-hierarchy {label}",
        )
    _require_columns(
        hypothesis_summary,
        ["family", "test_domain", "registered_tests", "adjusted_rejections"],
        "claim-hierarchy hypothesis summary",
    )
    _require_columns(
        headline_audit,
        [
            "inference_target",
            "p_value_raw",
            "p_value_holm_within_family",
        ],
        "claim-hierarchy headline audit",
    )
    _require_columns(
        within_player,
        [
            "model_id",
            "estimate",
            "ci_low",
            "ci_high",
            "n_discordant_strata",
        ],
        "claim-hierarchy within-player analysis",
    )

    def one(frame: pd.DataFrame, label: str, **filters: object) -> pd.Series:
        selected = frame
        for column, value in filters.items():
            selected = selected.loc[selected[column].eq(value)]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one {label} row for {filters}, found {len(selected)}"
            )
        return selected.iloc[0]

    any_exposure = one(
        model_tests,
        "primary global exposure",
        model_id="primary_same_day_per_appearance",
        contrast_id="global_recent_exposure_association_at_median_history",
    )
    interaction = one(
        model_tests,
        "primary interaction",
        model_id="primary_same_day_per_appearance",
        contrast_id="global_recent_exposure_by_continuous_history_interaction",
    )
    primary_adjusted = _holm_adjust(
        pd.Series([float(any_exposure["p_value"]), float(interaction["p_value"])])
    )
    exposure_contrast = one(
        model_contrasts,
        "primary 0-to-180 contrast",
        model_id="primary_same_day_per_appearance",
        history_anchor="median",
        burden_from=0.0,
        burden_to=180.0,
    )
    anchor_inference = one(
        headline_audit,
        "headline anchor inference",
        inference_target=(
            "post hoc median-history 0-to-180-minute anchor contrast"
        ),
    )
    within_player_spline = one(
        within_player,
        "within-player spline contrast",
        model_id="within_player_reference_bspline_same_day",
    )
    observed_contrast = one(
        model_contrasts,
        "observed-minute 0-to-180 contrast",
        model_id="same_day_observed_minutes",
        history_anchor="median",
        burden_from=0.0,
        burden_to=180.0,
    )
    minute = minute_summary.iloc[0]
    lineup_standardized = one(
        lineup_minute_summary,
        "lineup-standardized minute difference",
        comparison="lineup_standardized",
    )
    starter_minutes = one(
        lineup_minute_summary,
        "starter minute difference",
        comparison="starting_lineup",
    )
    substitute_minutes = one(
        lineup_minute_summary,
        "substitute minute difference",
        comparison="substitute_list",
    )
    selection_contribution = one(
        selection_effects,
        "selection composition contribution",
        history_anchor="median",
        burden_to=180.0,
        comparison_type="difference_between_changes",
    )
    selection_family_adjusted = _holm_adjust(selection_effects["p_value"])
    selection_adjusted_p = float(
        selection_family_adjusted.loc[selection_contribution.name]
    )
    severe_contrast = one(
        quality_contrasts,
        "severe-report contrast",
        model_id=f"quality_{SAME_DAY_SEVERE_COL}_per_appearance",
    )
    muscle_contrast = one(
        quality_contrasts,
        "muscle-report contrast",
        model_id=f"quality_{SAME_DAY_MUSCLE_COL}_per_appearance",
    )
    national_contrast = one(
        cohort_contrasts,
        "club-plus-national contrast",
        cohort_id="club_plus_senior_national_exposure",
    )
    functional_min = float(functional_contrasts["estimate"].min())
    functional_max = float(functional_contrasts["estimate"].max())
    categorical = one(
        hypothesis_summary,
        "historical categorical family",
        family="historical_exposure_response_family",
        test_domain="exposure_response_or_effect_modification",
    )

    rows = [
        {
            "claim_id": "same_day_per_appearance_recent_exposure",
            "tier": 2,
            "claim_role": "primary_original_association",
            "tier_justification": (
                "Tier 2: the same-day per-appearance public-report estimand and "
                "its explicit measurement audit are original in this setting, "
                "but the association is observational and sensitivity-bound. "
                "It is not Tier 1, a surprising Tier 3 contradiction, a direct "
                "Tier 4 replication, or an uninformative Tier 5 result."
            ),
            "abstract_recommended": True,
            "abstract_required_for_transparency": True,
            "main_display_recommended": True,
            "evidence": (
                f"The exploratory global spline test had raw "
                f"p={float(any_exposure['p_value']):.6g} and revision-family Holm "
                f"p={float(primary_adjusted.iloc[0]):.6g}; this is distinct from "
                f"the anchor contrast. At median prior "
                f"history, 0 to 180 minutes OR={float(exposure_contrast['estimate']):.3f} "
                f"({float(exposure_contrast['ci_low']):.3f}-"
                f"{float(exposure_contrast['ci_high']):.3f}); its wider post hoc "
                f"contrast-family Holm p was "
                f"{float(anchor_inference['p_value_holm_within_family']):.3f}. "
                f"The player-conditioned spline OR was "
                f"{float(within_player_spline['estimate']):.3f} "
                f"({float(within_player_spline['ci_low']):.3f}-"
                f"{float(within_player_spline['ci_high']):.3f}) across "
                f"{int(within_player_spline['n_discordant_strata'])} discordant "
                f"players; functional-form OR range={functional_min:.3f}-"
                f"{functional_max:.3f}."
            ),
            "required_caveat": (
                "This is a conditional association with a same-day public report "
                "per recorded appearance, selected after outcome inspection and "
                "not a causal clinical injury effect."
            ),
        },
        {
            "claim_id": "reported_event_duration_linkage",
            "tier": 2,
            "claim_role": "original_measurement_result",
            "tier_justification": (
                "Tier 2: the empirical size of the association between same-day "
                "report status and recorded duration, together with the resulting "
                "denominator-dependent estimate, is original. The time-at-risk "
                "problem is anticipated and exact event time remains unknown, so "
                "the result is below Tier 1 and does not meet Tier 3, 4, or 5."
            ),
            "abstract_recommended": True,
            "abstract_required_for_transparency": False,
            "main_display_recommended": True,
            "evidence": (
                f"Same-day rows contained {abs(float(minute['event_minus_non_event_minutes'])):.1f} "
                f"fewer recorded minutes ({abs(float(minute['difference_ci_high'])):.1f}-"
                f"{abs(float(minute['difference_ci_low'])):.1f} fewer; "
                f"{int(minute['bootstrap_replicates'])} player-cluster replicates). "
                f"The lineup-standardized gap was "
                f"{float(lineup_standardized['event_minus_non_event_minutes']):.1f} "
                f"minutes ({float(lineup_standardized['difference_ci_low']):.1f} to "
                f"{float(lineup_standardized['difference_ci_high']):.1f}); starter and "
                f"substitute gaps were {float(starter_minutes['event_minus_non_event_minutes']):.1f} "
                f"and {float(substitute_minutes['event_minus_non_event_minutes']):.1f}. "
                f"The observed-minute 0-to-180 IRR was "
                f"{float(observed_contrast['estimate']):.3f} "
                f"({float(observed_contrast['ci_low']):.3f}-"
                f"{float(observed_contrast['ci_high']):.3f})."
            ),
            "required_caveat": (
                "Per-appearance, observed-minute and fixed-90 estimands answer "
                "different questions; none recovers exact event time."
            ),
        },
        {
            "claim_id": "lineup_return_composition_sensitivity",
            "tier": 5,
            "claim_role": "exploratory_selection_audit",
            "tier_justification": (
                "Tier 5: the direct pointwise bootstrap interval excluded zero, "
                "but the post hoc 18-contrast selection family did not survive "
                "Holm correction. Under the lower-tier rule this remains an "
                "exploratory sensitivity, not a Tier 2 discovery."
            ),
            "abstract_recommended": False,
            "abstract_required_for_transparency": False,
            "main_display_recommended": False,
            "evidence": (
                f"At median history, changing rather than fixing the recorded "
                f"lineup/return composition added "
                f"{float(selection_contribution['estimate']):.3f} reports per "
                f"1,000 appearances to the 0-to-180 change "
                f"({float(selection_contribution['ci_low']):.3f}-"
                f"{float(selection_contribution['ci_high']):.3f}); Holm p="
                f"{selection_adjusted_p:.3f} across 18 contrasts."
            ),
            "required_caveat": (
                "The change demonstrates sensitivity to observed risk-set "
                "composition during complete-lineup seasons, not that selection "
                "created the entire curve."
            ),
        },
        {
            "claim_id": "public_report_quality_restrictions",
            "tier": 4,
            "claim_role": "corroborative_outcome_quality_checks",
            "tier_justification": (
                "Tier 4: persistence in longer-absence reports corroborates a "
                "previously proposed exception for severe public reports, while "
                "the muscle/tendon restriction is a correlated coding check. "
                "Neither is a new independent discovery or a Tier 3 contradiction."
            ),
            "abstract_recommended": False,
            "abstract_required_for_transparency": False,
            "main_display_recommended": False,
            "evidence": (
                f"Reported-absence >=28-day OR={float(severe_contrast['estimate']):.3f} "
                f"({float(severe_contrast['ci_low']):.3f}-"
                f"{float(severe_contrast['ci_high']):.3f}); muscle/tendon-report "
                f"OR={float(muscle_contrast['estimate']):.3f} "
                f"({float(muscle_contrast['ci_low']):.3f}-"
                f"{float(muscle_contrast['ci_high']):.3f})."
            ),
            "required_caveat": (
                "These post hoc correlated restrictions do not validate report "
                "attribution or clinical diagnosis."
            ),
        },
        {
            "claim_id": "national_exposure_scope_sensitivity",
            "tier": 5,
            "claim_role": "secondary_scope_sensitivity",
            "tier_justification": (
                "Tier 5: adding the limited available national exposure did not "
                "change interpretation and is neither novel nor contradictory."
            ),
            "abstract_recommended": False,
            "abstract_required_for_transparency": False,
            "main_display_recommended": False,
            "evidence": (
                f"Club-plus-national OR={float(national_contrast['estimate']):.3f} "
                f"({float(national_contrast['ci_low']):.3f}-"
                f"{float(national_contrast['ci_high']):.3f})."
            ),
            "required_caveat": (
                "Recorded national minutes changed few rows and omitted training."
            ),
        },
        {
            "claim_id": "continuous_history_effect_modification_null",
            "tier": 5,
            "claim_role": "secondary_effect_modification_null",
            "tier_justification": (
                "Tier 5: this is an ordinary unsupported interaction, not a "
                "highly surprising null or a direct contradiction of comparable "
                "published evidence, so it cannot enter Tier 3."
            ),
            "abstract_recommended": False,
            "abstract_required_for_transparency": False,
            "main_display_recommended": False,
            "evidence": (
                f"Continuous-history interaction p={float(interaction['p_value']):.3f}; "
                f"Holm p={float(primary_adjusted.iloc[1]):.3f}."
            ),
            "required_caveat": (
                "No detected interaction is not proof that clinically important "
                "effect modification is absent."
            ),
        },
        {
            "claim_id": "historical_categorical_exposure_family_null",
            "tier": 5,
            "claim_role": "secondary_null",
            "tier_justification": (
                "Tier 5: zero adjusted rejections across internally calibrated "
                "secondary categories is informative for transparency but is "
                "neither a surprising Tier 3 null nor a positive original result."
            ),
            "abstract_recommended": False,
            "abstract_required_for_transparency": False,
            "main_display_recommended": False,
            "evidence": (
                f"{int(categorical['adjusted_rejections'])}/"
                f"{int(categorical['registered_tests'])} categorical "
                "exposure-response or effect-modification tests rejected after "
                "multiplicity adjustment."
            ),
            "required_caveat": (
                "These internally calibrated categories are descriptive and "
                "secondary to continuous prior history."
            ),
        },
    ]
    hierarchy = enforce_claim_visibility(pd.DataFrame(rows))
    return hierarchy.sort_values(["tier", "claim_id"]).reset_index(drop=True)


def read_inputs(
    root: Path,
):  # pragma: no cover
    """Load appearance rows, outcomes, lineups, and exposure extensions once."""
    processed = root / "data" / "processed"
    v4_dir = processed / "public_data_v4"
    transfermarkt = root / "external_data" / "transfermarkt"
    panel = pd.read_csv(
        processed / "player_match_panel_all_comp.csv",
        parse_dates=["date"],
        low_memory=False,
    )
    injuries = pd.read_csv(
        processed / "tm_injuries_clean.csv", low_memory=False
    )
    episodes = pd.read_csv(
        processed / "tm_injury_episodes.csv",
        low_memory=False,
    )
    lineups_path = transfermarkt / "game_lineups.csv"
    lineups = (
        pd.read_csv(
            lineups_path,
            usecols=["date", "player_id", "type"],
            low_memory=False,
        )
        if lineups_path.exists()
        else None
    )
    national_path = (
        v4_dir / "international_appearances.csv"
    )
    national = (
        pd.read_csv(
            national_path,
            usecols=[PLAYER_ID_COL, "is_senior_competitive"],
            low_memory=False,
        )
        if national_path.exists()
        else None
    )
    feature_path = v4_dir / "match_exposure_scope_features.csv"
    exposure_features = pd.read_csv(
        feature_path,
        usecols=[
            PLAYER_ID_COL,
            "date",
            "senior_competitive_national_only_minutes_last_7d",
        ],
        low_memory=False,
    )
    return panel, injuries, episodes, lineups, national, exposure_features


def read_daily_timing_panel(root: Path) -> pd.DataFrame:  # pragma: no cover
    """Read only columns required for the report-onset timing audit."""
    columns = [
        PLAYER_ID_COL,
        "injury_event",
        "available_for_injury_risk",
        "prior_minutes_played",
        MINUTES_COL,
        "minutes_yesterday",
        BURDEN_COL,
        HISTORY_COL,
    ]
    return pd.read_csv(
        root / "data" / "processed" / "player_day_panel_all_comp.csv",
        usecols=columns,
        low_memory=False,
    )


def read_hypothesis_sources(
    results_dir: Path, v4_dir: Path
) -> dict[str, pd.DataFrame]:  # pragma: no cover
    """Read every formal-test family used in the disclosure register."""
    return {
        "exposure": pd.read_csv(
            results_dir / "matchproxy_effect_modification_tests.csv"
        ),
        "type_history": pd.read_csv(
            results_dir / "matchproxy_type_history_multiplicity_family.csv"
        ),
        "national_status": pd.read_csv(
            v4_dir / "v4_national_status_models.csv"
        ),
        "recovery": pd.read_csv(
            results_dir / "matchproxy_recovery_interval_model_summary.csv"
        ),
        "lineup": pd.read_csv(
            results_dir / "matchproxy_extension_lineup_spline_interaction.csv"
        ),
        "two_way_cluster": pd.read_csv(
            results_dir / "matchproxy_extension_two_way_cluster_sensitivity.csv"
        ),
    }


def write_outputs(
    outputs: Mapping[str, pd.DataFrame], results_dir: Path
) -> None:  # pragma: no cover
    """Write all non-empty JSAMS audit artifacts."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for stem, frame in outputs.items():
        frame.to_csv(results_dir / f"jsams_{stem}.csv", index=False)


def main() -> None:  # pragma: no cover
    """Run the full reviewer-requested analysis in dependency order."""
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "data" / "processed" / "results"
    v4_dir = root / "data" / "processed" / "public_data_v4"
    transfermarkt_dir = root / "external_data" / "transfermarkt"
    primary = load_source_module(
        "18_match_proxy_poisson_splines_perminute.py", "jsams_primary_source"
    )
    extension = load_source_module(
        "33_matchproxy_current_data_extensions.py", "jsams_extension_source"
    )
    panel, injuries, episodes, lineups, national, exposure_features = read_inputs(root)
    panel = add_same_day_quality_outcomes(
        panel,
        episodes,
        primary.classify_public_injury_type,
    )
    frame, scaling = prepare_jsams_frame(
        primary, panel, injuries, lineups, transfermarkt_dir
    )
    base_frame, _ = prepare_jsams_frame(
        primary,
        panel,
        injuries,
        lineups,
        transfermarkt_dir,
        minimum_prior_minutes=0.0,
    )
    metadata_frame, _ = extension.attach_current_match_metadata(
        primary, frame, transfermarkt_dir
    )
    lineup_completeness, lineup_reweighting, selection_frame = (
        lineup_completeness_audit(metadata_frame)
    )

    print("1. Fitting symmetric timing and denominator models ...")
    model_tables = run_symmetric_model_suite(frame, scaling)
    support = exposure_support_table(frame)
    specification = model_specification_table(frame, scaling)

    print("2. Running functional-form and outcome-quality checks ...")
    functional_tests, functional_contrasts = functional_form_sensitivity(
        frame, scaling
    )
    quality_tests, quality_contrasts, quality_summary = outcome_quality_sensitivity(
        frame, scaling
    )

    print("3. Standardising over lineup and recent-return composition ...")
    (
        standardized_curves,
        standardized_comparisons,
        standardized_tests,
    ) = selection_standardization(selection_frame, scaling)
    selection_bootstrap_samples, selection_effects = selection_effect_contrasts(
        selection_frame, scaling
    )
    lineup_tests, lineup_contrasts = lineup_role_model_sensitivity(
        selection_frame
    )
    context_tests, context_contrasts = context_sensitivity_analysis(metadata_frame)

    print("4. Fitting within-player and report-timing checks ...")
    within_player = within_player_same_day_analysis(frame)
    day_panel = read_daily_timing_panel(root)
    timing_summary, timing_samples, timing_contrasts = (
        daily_report_timing_enrichment(day_panel)
    )

    print("5. Bootstrapping overall and lineup-standardised minute differences ...")
    minute_samples, minute_summary = cluster_bootstrap_minute_difference(frame)
    lineup_minute_samples, lineup_minute_summary = lineup_standardized_minute_difference(
        selection_frame
    )

    print("6. Refitting eligibility, season, and national-exposure cohorts ...")
    threshold_frames: dict[float, pd.DataFrame] = {
        REFERENCE_MINIMUM_PRIOR_MINUTES: frame
    }
    for threshold in ELIGIBILITY_THRESHOLDS:
        if threshold == REFERENCE_MINIMUM_PRIOR_MINUTES:
            continue
        threshold_frame, _ = prepare_jsams_frame(
            primary,
            panel,
            injuries,
            lineups,
            minimum_prior_minutes=threshold,
        )
        threshold_frames[threshold] = threshold_frame
    national_frame, national_audit = add_senior_national_exposure(
        frame, exposure_features
    )
    cohorts = {
        "eligibility_450_prior_minutes": threshold_frames[450.0],
        "reference_900_prior_minutes": threshold_frames[900.0],
        "eligibility_1800_prior_minutes": threshold_frames[1800.0],
        "complete_seasons_through_2023_24": frame[
            pd.to_datetime(frame["date"], errors="coerce").le(COMPLETE_SEASON_END)
        ].copy(),
        "excluding_2019_20_and_2020_21": frame[
            ~frame["season_start"].isin(PANDEMIC_SEASON_STARTS)
        ].copy(),
        "club_plus_senior_national_exposure": national_frame,
    }
    cohort_tests, cohort_contrasts, cohort_audit = cohort_robustness_suite(cohorts)

    print("7. Building cohort flow, descriptors and hypothesis register ...")
    flow = cohort_flow_table(injuries, episodes, panel, frame)
    descriptive = cohort_descriptive_table(frame, national)
    eligibility_comparison = eligibility_player_comparison(base_frame, frame)
    hypothesis_sources = read_hypothesis_sources(results_dir, v4_dir)
    hypothesis_sources.update(
        {
            "primary_contrasts": model_tables["contrasts"],
            "selection_standardized": standardized_tests,
            "context_sensitivity": context_tests,
            "functional_form": functional_tests,
            "outcome_quality": quality_tests,
            "lineup_role": lineup_tests,
            "selection_effect": selection_effects,
            "cohort_robustness": cohort_tests,
            "within_player": within_player,
        }
    )
    register, register_summary = hypothesis_register(
        model_tables["tests"], hypothesis_sources
    )
    headline_audit = headline_inference_audit(register)
    claim_hierarchy = publication_claim_hierarchy(
        model_tables["tests"],
        model_tables["contrasts"],
        minute_summary,
        lineup_minute_summary,
        selection_effects,
        functional_contrasts,
        quality_contrasts,
        cohort_contrasts,
        register_summary,
        headline_audit,
        within_player,
    )

    outputs = {
        "primary_model_specification": specification,
        "primary_model_predictions": model_tables["predictions"],
        "primary_model_tests": model_tables["tests"],
        "primary_model_contrasts": model_tables["contrasts"],
        "primary_model_coefficients": model_tables["coefficients"],
        "exposure_support": support,
        "selection_standardized_curves": standardized_curves,
        "selection_standardized_comparisons": standardized_comparisons,
        "selection_standardized_tests": standardized_tests,
        "selection_effect_bootstrap_samples": selection_bootstrap_samples,
        "selection_effect_contrasts": selection_effects,
        "lineup_completeness": lineup_completeness,
        "lineup_reweighting_assessment": lineup_reweighting,
        "within_player_same_day": within_player,
        "daily_report_timing_summary": timing_summary,
        "daily_report_timing_bootstrap_samples": timing_samples,
        "daily_report_timing_contrasts": timing_contrasts,
        "functional_form_tests": functional_tests,
        "functional_form_contrasts": functional_contrasts,
        "outcome_quality_tests": quality_tests,
        "outcome_quality_contrasts": quality_contrasts,
        "outcome_quality_summary": quality_summary,
        "lineup_role_tests": lineup_tests,
        "lineup_role_contrasts": lineup_contrasts,
        "context_sensitivity_tests": context_tests,
        "context_sensitivity_contrasts": context_contrasts,
        "same_day_minute_bootstrap_samples": minute_samples,
        "same_day_minute_bootstrap_summary": minute_summary,
        "lineup_minute_bootstrap_samples": lineup_minute_samples,
        "lineup_minute_bootstrap_summary": lineup_minute_summary,
        "cohort_robustness_tests": cohort_tests,
        "cohort_robustness_contrasts": cohort_contrasts,
        "cohort_robustness_audit": cohort_audit,
        "national_exposure_audit": national_audit,
        "cohort_flow": flow,
        "cohort_descriptives": descriptive,
        "eligibility_player_comparison": eligibility_comparison,
        "hypothesis_register": register,
        "hypothesis_family_summary": register_summary,
        "headline_inference_audit": headline_audit,
        "claim_hierarchy": claim_hierarchy,
    }
    write_outputs(outputs, results_dir)
    print(f"Wrote {len(outputs)} JSAMS revision tables to {results_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
