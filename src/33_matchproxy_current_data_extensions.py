#!/usr/bin/env python
"""Run additional selection, reporting, and context audits on the match proxy.

This script deliberately follows, rather than replaces, the frozen primary
match-proxy model in ``18_match_proxy_poisson_splines_perminute.py``.  Its
analyses answer questions that the primary data can support without presenting
them as causal workload effects:

* refit the primary spline after separating recorded starters, substitute
  appearances, and recent returns;
* measure whether public injury-type classification is associated with observed
  match and reporting context, then run an inverse-probability-weighted
  muscle/tendon reporting sensitivity;
* translate public absence dates into a reported absence-day burden proxy,
  explicitly distinct from clinically confirmed injury burden;
* map the joint support for recent minutes, match count, and recovery before
  estimating a limited, supported one-versus-two-match comparison;
* fit player-season conditional-logistic (case-crossover) associations for
  recent minutes and recovery; and
* quantify curve-feature stability with a player-cluster bootstrap and
  uncertainty with two-way player/match clustering.

The script also records current-match competition context.  It uses only public
club-match records already held by the repository and is therefore a context
sensitivity, not a substitute for training, travel, or medical data.

All p-values from this script belong to a distinct exploratory extension family.
They do not alter the frozen primary multiplicity family or select manuscript
claims.

Outputs are written to ``data/processed/results`` with the prefix
``matchproxy_extension_``.

Run after script 18:

    python src/33_matchproxy_current_data_extensions.py
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from math import erfc, exp
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.stats import chi2
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from pipeline_io import merge_day_fragility, restrict_to_available_risk_set, restrict_to_fragility_risk_set
from v4_statistics import percent_with_interval


PLAYER_ID_COL = "tm_player_id"
EVENT_COL = "injury_event_matchproxy"
MATCH_MINUTES_COL = "all_minutes_played"
MODEL_GROUPS = ("regular", "fragile")
LINEUP_ROLES = ("starting_lineup", "substitute_list")
TYPE_CLASSIFIABLE = {
    "muscle/tendon",
    "joint/ligament",
    "bone/fracture",
    "head/concussion",
    "illness/other medical",
}
BURDEN_BINS = (0.0, 45.0, 90.0, 135.0, 180.0, 220.0, np.inf)
BURDEN_LABELS = ("0-45", "46-90", "91-135", "136-180", "181-220", ">220")
SUPPORTED_RECOVERY_LEVELS = ("0-3 days", "4-5 days")
MIN_JOINT_CELL_ROWS = 100
MIN_JOINT_CELL_EVENTS = 5
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260804
IPW_BOOTSTRAP_REPLICATES = 200
IPW_BOOTSTRAP_SEED = 20260805
MIN_TYPE_REPORTING_PROBABILITY = 0.10
MAX_TYPE_REPORTING_WEIGHT = 10.0
BOOTSTRAP_GRID = np.arange(0.0, 180.0 + 0.0001, 5.0)
def load_primary_module():
    """Load the numeric primary-model script as an importable module."""
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    module_path = src_dir / "18_match_proxy_poisson_splines_perminute.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing primary match-proxy script: {module_path}")
    spec = importlib.util.spec_from_file_location("matchproxy_primary_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import primary match-proxy script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    """Raise a clear error when an analysis frame lacks a required column."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def _z_interval(estimate: float, standard_error: float) -> tuple[float, float, float]:
    """Return exponentiated estimate, 95% interval, and two-sided normal p-value."""
    if not np.isfinite(estimate) or not np.isfinite(standard_error):
        return np.nan, np.nan, np.nan
    z_value = estimate / standard_error if standard_error > 0 else 0.0
    critical = NormalDist().inv_cdf(0.975)
    return (
        float(exp(estimate)),
        float(exp(estimate - critical * standard_error)),
        float(exp(estimate + critical * standard_error)),
    )


def _normal_p_value(estimate: float, standard_error: float) -> float:
    """Return a two-sided normal p-value without a SciPy model-result wrapper."""
    if not np.isfinite(estimate) or not np.isfinite(standard_error):
        return np.nan
    if standard_error <= 0:
        return 1.0 if np.isclose(estimate, 0.0) else 0.0
    return float(erfc(abs(estimate / standard_error) / np.sqrt(2.0)))


def prepare_extension_frame(
    primary_module: Any,
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    lineups: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach return and lineup markers to the frozen primary match frame."""
    frame = primary_module.prepare_model_frame(panel, EVENT_COL, "fragility_group")
    frame = primary_module.add_recent_prior_injury_return_flags(frame, injuries)
    frame = primary_module.add_lineup_start_status(frame, lineups)
    frame["lineup_role_model"] = np.where(
        frame["lineup_role"].isin(LINEUP_ROLES),
        frame["lineup_role"],
        "lineup_unavailable_or_other",
    )
    frame["returned_from_recorded_injury_within_14d"] = frame[
        "returned_from_recorded_injury_within_14d"
    ].fillna(False).astype(bool)
    return frame


def _bundle_summary(
    primary_module: Any,
    label: str,
    frame: pd.DataFrame,
    bundle: Mapping[str, Any],
    restriction: str,
) -> dict[str, Any]:
    """Summarise a primary-compatible refit with its data restriction."""
    row = primary_module.summary_row(
        label,
        EVENT_COL,
        "fragility_group",
        "calendar_timing",
        frame,
        bundle,
    )
    row["restriction"] = restriction
    row["analysis_family"] = "selection_refit"
    return row


def lineup_refit_outputs(
    primary_module: Any,
    frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Refit the primary spline in lineup and recent-return risk sets.

    Starting status is a pre-match lineup marker. Substitute-list status is an
    observed selection marker among appearances, so neither restriction is a
    causal adjustment. They test whether a curve feature persists in materially
    different observed risk-set compositions.
    """
    _require_columns(
        frame,
        ["lineup_role_model", "returned_from_recorded_injury_within_14d", EVENT_COL, "model_group"],
        "lineup extension frame",
    )
    specs = (
        (
            "lineup_known",
            "recorded starter or substitute-list appearance",
            frame[frame["lineup_role_model"].isin(LINEUP_ROLES)].copy(),
        ),
        (
            "starters_only",
            "recorded starting-lineup appearance",
            frame[frame["lineup_role_model"].eq("starting_lineup")].copy(),
        ),
        (
            "substitutes_only",
            "recorded substitute-list appearance",
            frame[frame["lineup_role_model"].eq("substitute_list")].copy(),
        ),
        (
            "exclude_recent_return",
            "exclude appearance within 14 days of recorded return",
            frame[~frame["returned_from_recorded_injury_within_14d"]].copy(),
        ),
        (
            "starters_exclude_recent_return",
            "recorded starter and not within 14 days of recorded return",
            frame[
                frame["lineup_role_model"].eq("starting_lineup")
                & ~frame["returned_from_recorded_injury_within_14d"]
            ].copy(),
        ),
    )
    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    shape_frames: list[pd.DataFrame] = []
    effect_frames: list[pd.DataFrame] = []
    for label, restriction, subset in specs:
        group_counts = set(subset["model_group"].dropna().astype(str).unique())
        if subset.empty or int(subset[EVENT_COL].sum()) == 0 or group_counts != set(MODEL_GROUPS):
            summary_rows.append(
                {
                    "model": label,
                    "restriction": restriction,
                    "analysis_family": "selection_refit",
                    "fit_status": "not_estimable",
                    "n_match_rows": int(len(subset)),
                    "n_players": int(subset[PLAYER_ID_COL].nunique()),
                    "n_events": int(subset[EVENT_COL].sum()) if EVENT_COL in subset else 0,
                }
            )
            continue
        bundle = primary_module.run_prediction_bundle(subset, EVENT_COL)
        summary_rows.append(_bundle_summary(primary_module, label, subset, bundle, restriction))
        predictions = bundle["predictions"].copy()
        predictions.insert(0, "model", label)
        predictions.insert(1, "restriction", restriction)
        prediction_frames.append(predictions)
        shapes = primary_module.spline_curve_shape_summary(predictions)
        shapes.insert(0, "model", label)
        shapes.insert(1, "restriction", restriction)
        shape_frames.append(shapes)
        effects = primary_module.label_effect_modification_rows(
            bundle["effect_modification"],
            label,
            EVENT_COL,
            "fragility_group",
            "calendar_timing",
            subset,
            "poisson",
            "selection_refit_extension",
        )
        effects.insert(0, "restriction", restriction)
        effect_frames.append(effects)
    return {
        "summary": pd.DataFrame(summary_rows),
        "predictions": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        "shape": pd.concat(shape_frames, ignore_index=True) if shape_frames else pd.DataFrame(),
        "effects": pd.concat(effect_frames, ignore_index=True) if effect_frames else pd.DataFrame(),
    }


def lineup_spline_interaction_tests(
    primary_module: Any,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether fitted burden shapes differ by recorded lineup role.

    This direct test is distinct from role-restricted refits. It is exploratory
    because recorded lineup coverage is incomplete and substitute-list status is
    part of the observed selection process.
    """
    known = frame[frame["lineup_role_model"].isin(LINEUP_ROLES)].copy()
    _require_columns(known, [EVENT_COL, "all_minutes_last_7d", "log_minutes_played"], "lineup-known frame")
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, pd.DataFrame, str]] = [
        ("pooled_history_adjusted", known, "+ C(model_group)"),
    ]
    for group in MODEL_GROUPS:
        specs.append((f"{group}_history_only", known[known["model_group"].eq(group)].copy(), ""))
    for label, subset, controls in specs:
        if subset.empty or subset[EVENT_COL].sum() <= 0 or subset["lineup_role_model"].nunique() < 2:
            rows.append(
                {
                    "model": label,
                    "effect_measure": "chi_square",
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    # `summary_row` stores this field as an integer; zero
                    # denotes that no valid joint degrees of freedom remain.
                    "df": 0,
                    "p_value": np.nan,
                    "fit_status": "not_estimable",
                    "n_match_rows": int(len(subset)),
                    "n_players": int(subset[PLAYER_ID_COL].nunique()),
                    "n_events": int(subset[EVENT_COL].sum()) if EVENT_COL in subset else 0,
                }
            )
            continue
        burden_max = float(subset["all_minutes_last_7d"].max())
        spline = primary_module.spline_basis_expression(burden_max)
        formula = (
            f"{EVENT_COL} ~ {spline} * C(lineup_role_model, "
            "Treatment(reference='starting_lineup')) "
            "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
            f"{controls}"
        )
        result = smf.glm(
            formula=formula,
            data=subset,
            family=sm.families.Poisson(),
            offset=subset["log_minutes_played"],
        ).fit(cov_type="cluster", cov_kwds={"groups": subset[PLAYER_ID_COL]})
        names = [str(name) for name in result.params.index]
        interaction_indices = [
            index
            for index, name in enumerate(names)
            if ":C(lineup_role_model" in name or "C(lineup_role_model" in name and ":" in name
        ]
        if not interaction_indices:
            fit_status = "not_estimable"
            statistic = np.nan
            p_value = np.nan
        else:
            restriction = np.zeros((len(interaction_indices), len(names)), dtype=float)
            for row_index, column_index in enumerate(interaction_indices):
                restriction[row_index, column_index] = 1.0
            test = result.wald_test(restriction, scalar=True)
            fit_status = "ok"
            statistic = float(np.asarray(test.statistic).squeeze())
            p_value = float(np.asarray(test.pvalue).squeeze())
        rows.append(
            {
                "model": label,
                "effect_measure": "chi_square",
                "estimate": statistic,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "df": float(len(interaction_indices)),
                "p_value": p_value,
                "fit_status": fit_status,
                "n_match_rows": int(len(subset)),
                "n_players": int(subset[PLAYER_ID_COL].nunique()),
                "n_events": int(subset[EVENT_COL].sum()),
            }
        )
    return primary_module.add_p_value_adjustments(pd.DataFrame(rows)).rename(
        columns={
            "p_value_holm": "p_holm_lineup_family",
            "p_value_fdr_bh": "p_bh_lineup_family",
        }
    )


def classify_reporting_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach event-level public-report classification and timing indicators."""
    _require_columns(
        frame,
        [EVENT_COL, "matchproxy_public_injury_type", "matchproxy_duration_days"],
        "reporting frame",
    )
    out = frame[frame[EVENT_COL].fillna(0).astype(int).eq(1)].copy()
    out["report_type_classifiable"] = out["matchproxy_public_injury_type"].isin(TYPE_CLASSIFIABLE)
    out["report_duration_available"] = pd.to_numeric(
        out["matchproxy_duration_days"], errors="coerce"
    ).notna()
    same_day = out.get("injury_event_matchproxy_same_day", pd.Series(0, index=out.index))
    lag1 = out.get("injury_event_matchproxy_lag1", pd.Series(0, index=out.index))
    out["proxy_timing"] = np.select(
        [same_day.astype(int).eq(1) & lag1.astype(int).eq(1), same_day.astype(int).eq(1), lag1.astype(int).eq(1)],
        ["same_day_and_lag1", "same_day", "lag1"],
        default="unclassified_proxy_timing",
    )
    return out


def reporting_completeness_by_context(events: pd.DataFrame) -> pd.DataFrame:
    """Return type and duration completeness with Wilson intervals by context."""
    _require_columns(
        events,
        ["model_group", "lineup_role_model", "proxy_timing", "report_type_classifiable", "report_duration_available"],
        "reporting events",
    )
    rows: list[dict[str, Any]] = []
    groupings = {
        "overall": [],
        "history": ["model_group"],
        "lineup": ["lineup_role_model"],
        "timing": ["proxy_timing"],
        "history_by_timing": ["model_group", "proxy_timing"],
    }
    for context, columns in groupings.items():
        groupby_arg: str | list[str] = columns[0] if len(columns) == 1 else columns
        groups = [((), events)] if not columns else events.groupby(groupby_arg, dropna=False, observed=False)
        for keys, subset in groups:
            if not isinstance(keys, tuple):
                keys = (keys,)
            n_events = int(len(subset))
            type_known = int(subset["report_type_classifiable"].sum())
            duration_known = int(subset["report_duration_available"].sum())
            type_pct, type_low, type_high = percent_with_interval(type_known, n_events)
            duration_pct, duration_low, duration_high = percent_with_interval(duration_known, n_events)
            row: dict[str, Any] = {
                "context": context,
                "events": n_events,
                "type_classifiable_events": type_known,
                "type_classifiable_percent": type_pct,
                "type_classifiable_ci_low": type_low,
                "type_classifiable_ci_high": type_high,
                "duration_available_events": duration_known,
                "duration_available_percent": duration_pct,
                "duration_available_ci_low": duration_low,
                "duration_available_ci_high": duration_high,
            }
            row.update(dict(zip(columns, keys)))
            rows.append(row)
    return pd.DataFrame(rows)


def _reporting_model_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Create finite covariates for the reporting-completeness logistic model."""
    out = events.copy()
    out["burden_per_90"] = pd.to_numeric(out["all_minutes_last_7d"], errors="coerce").fillna(0.0) / 90.0
    out["season_start"] = np.where(
        pd.to_datetime(out["date"], errors="coerce").dt.month.ge(7),
        pd.to_datetime(out["date"], errors="coerce").dt.year,
        pd.to_datetime(out["date"], errors="coerce").dt.year - 1,
    ).astype(int)
    out["lineup_role_model"] = out["lineup_role_model"].astype(str)
    out["proxy_timing"] = out["proxy_timing"].astype(str)
    out["model_group"] = out["model_group"].astype(str)
    out["recent_return_indicator"] = out[
        "returned_from_recorded_injury_within_14d"
    ].astype(int)
    return out


def reporting_type_model(events: pd.DataFrame) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Fit a clustered type-classifiability model and return predicted weights.

    The model describes observed reporting completeness, not the probability of
    a clinical diagnosis.  It deliberately excludes club fixed effects because
    sparse club-seasons can create deterministic cells; club-season variation is
    reported descriptively instead.
    """
    frame = _reporting_model_frame(events)
    if frame.empty or frame["report_type_classifiable"].nunique() < 2:
        raise ValueError("Type-classifiability model requires both observed outcomes")
    terms = ["burden_per_90"]
    for column in ["model_group", "proxy_timing", "lineup_role_model", "season_start"]:
        if frame[column].nunique(dropna=False) > 1:
            terms.append(f"C({column})")
    if frame["recent_return_indicator"].nunique(dropna=False) > 1:
        terms.append("recent_return_indicator")
    formula = "report_type_classifiable ~ " + " + ".join(terms)
    result = smf.glm(
        formula=formula,
        data=frame,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID_COL]})
    terms: list[dict[str, Any]] = []
    for term in result.params.index:
        estimate = float(result.params[term])
        standard_error = float(result.bse[term])
        odds_ratio, ci_low, ci_high = _z_interval(estimate, standard_error)
        terms.append(
            {
                "term": str(term),
                "effect_measure": "odds_ratio_type_classifiable",
                "estimate": odds_ratio,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": _normal_p_value(estimate, standard_error),
                "n_events": int(len(frame)),
                "n_players": int(frame[PLAYER_ID_COL].nunique()),
                "fit_status": "ok",
            }
        )
    terms_df = pd.DataFrame(terms)
    terms_df = terms_df.assign(
        p_holm_reporting_family=np.nan,
        p_bh_reporting_family=np.nan,
    )
    non_intercept = terms_df[~terms_df["term"].eq("Intercept")].copy()
    adjusted = primary_p_adjust(non_intercept["p_value"])
    terms_df.loc[non_intercept.index, "p_holm_reporting_family"] = adjusted["holm"]
    terms_df.loc[non_intercept.index, "p_bh_reporting_family"] = adjusted["bh"]
    # Preserve the model's probability scale for the positivity diagnostic.
    # A minimal numerical bound only prevents division by exact zero below.
    frame["predicted_type_classifiable_probability"] = result.predict(frame).clip(1e-6, 1.0)
    frame["type_reporting_ipw"] = np.where(
        frame["report_type_classifiable"],
        1.0 / frame["predicted_type_classifiable_probability"],
        0.0,
    )
    return result, terms_df, frame


def reporting_type_ipw_diagnostics(weighted_events: pd.DataFrame) -> pd.DataFrame:
    """Audit whether inverse-probability type weights have usable overlap.

    The flag is deliberately conservative: a modelled classification probability
    below 0.10 or a resulting inverse-probability weight above 10 means the
    public records do not provide adequate overlap for a stable correction.
    The reweighted curves remain an archive-only stress test in that setting.
    """
    _require_columns(
        weighted_events,
        ["report_type_classifiable", "predicted_type_classifiable_probability", "type_reporting_ipw"],
        "IPW diagnostic frame",
    )
    observed = weighted_events[weighted_events["report_type_classifiable"]].copy()
    probabilities = observed["predicted_type_classifiable_probability"].to_numpy(dtype=float)
    weights = observed["type_reporting_ipw"].to_numpy(dtype=float)
    finite_probabilities = probabilities[np.isfinite(probabilities)]
    finite_weights = weights[np.isfinite(weights)]
    min_probability = float(np.min(finite_probabilities)) if len(finite_probabilities) else np.nan
    max_weight = float(np.max(finite_weights)) if len(finite_weights) else np.nan
    stable = bool(
        len(finite_probabilities)
        and min_probability >= MIN_TYPE_REPORTING_PROBABILITY
        and max_weight <= MAX_TYPE_REPORTING_WEIGHT
    )
    effective_events = (
        float(np.sum(finite_weights) ** 2 / np.sum(finite_weights**2))
        if len(finite_weights) and np.sum(finite_weights**2) > 0
        else np.nan
    )
    return pd.DataFrame(
        [
            {
                "n_proxy_events": int(len(weighted_events)),
                "n_type_classifiable_events": int(len(observed)),
                "type_classifiable_percent": float(100.0 * len(observed) / len(weighted_events)) if len(weighted_events) else np.nan,
                "minimum_predicted_type_classifiable_probability": min_probability,
                "p01_predicted_type_classifiable_probability": float(np.quantile(finite_probabilities, 0.01)) if len(finite_probabilities) else np.nan,
                "median_predicted_type_classifiable_probability": float(np.median(finite_probabilities)) if len(finite_probabilities) else np.nan,
                "maximum_type_reporting_ipw": max_weight,
                "p99_type_reporting_ipw": float(np.quantile(finite_weights, 0.99)) if len(finite_weights) else np.nan,
                "effective_weighted_type_classifiable_events": effective_events,
                "minimum_probability_gate": MIN_TYPE_REPORTING_PROBABILITY,
                "maximum_weight_gate": MAX_TYPE_REPORTING_WEIGHT,
                "stability_status": "stable" if stable else "unstable_positivity_or_weight_tail",
            }
        ]
    )


def primary_p_adjust(values: pd.Series) -> dict[str, np.ndarray]:
    """Return Holm and Benjamini-Hochberg adjustments for finite p-values."""
    from statsmodels.stats.multitest import multipletests

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(numeric)
    out_holm = np.full(len(numeric), np.nan)
    out_bh = np.full(len(numeric), np.nan)
    if valid.any():
        out_holm[valid] = multipletests(numeric[valid], method="holm")[1]
        out_bh[valid] = multipletests(numeric[valid], method="fdr_bh")[1]
    return {"holm": out_holm, "bh": out_bh}


def reporting_type_ipw_sensitivity(
    primary_module: Any,
    full_frame: pd.DataFrame,
    weighted_events: pd.DataFrame,
    bootstrap_replicates: int = IPW_BOOTSTRAP_REPLICATES,
    seed: int = IPW_BOOTSTRAP_SEED,
) -> dict[str, pd.DataFrame]:
    """Repeat the muscle/tendon model after weighting observed classifications.

    Untypeable event rows receive zero analysis weight because their outcome
    category is unknown. Classifiable event rows are weighted by the inverse of
    their fitted reporting probability; non-event rows retain weight one.
    This is a missing-at-random reporting sensitivity, not a correction for
    unobserved clinical diagnosis.
    """
    _require_columns(weighted_events, [PLAYER_ID_COL, "date", "type_reporting_ipw", "report_type_classifiable"], "weighted reporting events")
    diagnostics = reporting_type_ipw_diagnostics(weighted_events)
    out = full_frame.copy()
    event_keys = weighted_events[[PLAYER_ID_COL, "date", "type_reporting_ipw", "report_type_classifiable"]].copy()
    out = out.merge(event_keys, on=[PLAYER_ID_COL, "date"], how="left")
    is_event = out[EVENT_COL].fillna(0).astype(int).eq(1)
    out["type_reporting_ipw"] = np.where(
        is_event,
        out["type_reporting_ipw"].fillna(0.0),
        1.0,
    )
    out["muscle_tendon_reporting_weighted_event"] = (
        out["injury_event_matchproxy_muscle_tendon"].fillna(0).astype(int)
    )
    if out["muscle_tendon_reporting_weighted_event"].sum() <= 0:
        return {
            "summary": pd.DataFrame(),
            "selected": pd.DataFrame(),
            "ratios": pd.DataFrame(),
            "bootstrap_samples": pd.DataFrame(),
            "diagnostics": diagnostics,
        }
    bundle = run_weighted_prediction_bundle(
        primary_module,
        out,
        "muscle_tendon_reporting_weighted_event",
        "type_reporting_ipw",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    summary_row = _bundle_summary(
        primary_module,
        "muscle_tendon_inverse_probability_reporting_weighted",
        out,
        bundle,
        "type-classifiable public reports inverse-probability weighted",
    )
    summary_row.update(
        {
            "event_col": "injury_event_matchproxy_muscle_tendon",
            "n_events": int(out["muscle_tendon_reporting_weighted_event"].sum()),
            "n_matchproxy_events": int(is_event.sum()),
            "weighted_muscle_tendon_event_total": float(
                (out["muscle_tendon_reporting_weighted_event"] * out["type_reporting_ipw"]).sum()
            ),
            "weight_min": float(out["type_reporting_ipw"].min()),
            "weight_max": float(out["type_reporting_ipw"].max()),
            "weight_mean": float(out["type_reporting_ipw"].mean()),
            "unclassifiable_event_rows_excluded": int((is_event & out["type_reporting_ipw"].eq(0.0)).sum()),
            "reporting_weight_stability_status": diagnostics.loc[0, "stability_status"],
            "analysis_family": "reporting_type_ipw_sensitivity",
            "fit_status": (
                "ok"
                if diagnostics.loc[0, "stability_status"] == "stable"
                else "unstable_reporting_weight_tail"
            ),
            "dispersion": np.nan,
            "dispersion_status": "not_interpretable_under_inverse_probability_weights",
        }
    )
    summary = pd.DataFrame([summary_row])
    return {
        "summary": summary,
        "selected": bundle["selected"].copy(),
        "ratios": bundle["ratios"].copy(),
        "bootstrap_samples": bundle["bootstrap_samples"].copy(),
        "diagnostics": diagnostics,
    }


def run_weighted_prediction_bundle(
    primary_module: Any,
    frame: pd.DataFrame,
    event_col: str,
    weight_col: str,
    bootstrap_replicates: int = IPW_BOOTSTRAP_REPLICATES,
    seed: int = IPW_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fit an IPW spline and obtain player-bootstrap uncertainty.

    ``statsmodels`` does not fully support clustered sandwich covariance with
    frequency weights. The model therefore supplies point estimates only, and
    player-cluster resampling supplies all reported intervals for this
    missing-at-random reporting sensitivity.
    """
    _require_columns(frame, [event_col, weight_col, "model_group", "log_minutes_played"], "weighted spline frame")
    if frame[event_col].sum() <= 0:
        raise ValueError(f"No events available for {event_col}")
    if set(frame["model_group"].unique()) != set(MODEL_GROUPS):
        raise ValueError("Weighted model requires regular and fragile rows")
    if bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be at least 2")
    burden_max = float(frame["all_minutes_last_7d"].max())
    formula = primary_module.spline_formula(event_col, burden_max)
    result = smf.glm(
        formula=formula,
        data=frame,
        family=sm.families.Poisson(),
        offset=frame["log_minutes_played"],
        freq_weights=frame[weight_col].astype(float),
    ).fit()
    selected, ratios, bootstrap_samples = weighted_player_bootstrap_predictions(
        primary_module,
        result,
        frame,
        event_col,
        weight_col,
        bootstrap_replicates,
        seed,
    )
    support = primary_module.selected_support_rows(frame, event_col, (0.0, 90.0, 180.0))
    selected = selected.merge(support, on=["fragility_group", "all_minutes_last_7d"], how="left")
    pearson = float(np.sum(result.resid_pearson**2) / result.df_resid)
    effect_modification = weighted_not_estimable_effect_rows(primary_module)
    return {
        "result": result,
        "predictions": selected.copy(),
        "selected": selected,
        "ratios": ratios,
        "effect_modification": effect_modification,
        "bootstrap_samples": bootstrap_samples,
        "dispersion": pearson,
        "model_family": "poisson",
        "denominator": "observed_minutes",
        "estimator": "inverse_probability_weighted_glm_player_cluster_bootstrap",
    }


def weighted_player_bootstrap_predictions(
    primary_module: Any,
    result: Any,
    frame: pd.DataFrame,
    event_col: str,
    weight_col: str,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return selected IPW predictions and player-bootstrap percentile intervals."""
    burdens = (0.0, 90.0, 180.0)
    design_info = result.model.data.design_info
    templates = {
        group: np.asarray(
            build_design_matrices(
                [design_info], primary_module.prediction_template(burdens, group)
            )[0]
        )
        for group in MODEL_GROUPS
    }
    point_rates = {
        group: np.exp(templates[group] @ np.asarray(result.params, dtype=float)) * 10_000.0
        for group in MODEL_GROUPS
    }
    player_codes = pd.factorize(frame[PLAYER_ID_COL], sort=True)[0]
    n_players = int(player_codes.max() + 1)
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    for replicate in range(replicates):
        draw = rng.integers(0, n_players, size=n_players)
        draw_counts = np.bincount(draw, minlength=n_players)
        bootstrap_weights = frame[weight_col].astype(float).to_numpy() * draw_counts[player_codes]
        if not np.any(bootstrap_weights):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bootstrap_result = sm.GLM(
                    frame[event_col].astype(float).to_numpy(),
                    result.model.exog,
                    family=sm.families.Poisson(),
                    offset=frame["log_minutes_played"].astype(float).to_numpy(),
                    freq_weights=bootstrap_weights,
                ).fit()
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        parameters = np.asarray(bootstrap_result.params, dtype=float)
        if not np.isfinite(parameters).all():
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            rates = {
                group: np.exp(templates[group] @ parameters) * 10_000.0
                for group in MODEL_GROUPS
            }
        if not all(np.isfinite(values).all() for values in rates.values()):
            continue
        for index, burden in enumerate(burdens):
            for group in MODEL_GROUPS:
                samples.append(
                    {
                        "replicate": replicate,
                        "measure": "predicted_events_per_10000_match_minutes",
                        "history_stratum": group,
                        "all_minutes_last_7d": burden,
                        "estimate": float(rates[group][index]),
                    }
                )
            samples.append(
                {
                    "replicate": replicate,
                    "measure": "higher_vs_intermediate_rate_ratio",
                    "history_stratum": "higher_vs_intermediate",
                    "all_minutes_last_7d": burden,
                    "estimate": float(
                        rates["fragile"][index]
                        / max(rates["regular"][index], np.finfo(float).tiny)
                    ),
                }
            )
    samples_frame = pd.DataFrame(
        samples,
        columns=["replicate", "measure", "history_stratum", "all_minutes_last_7d", "estimate"],
    )
    selected_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    for index, burden in enumerate(burdens):
        for group in MODEL_GROUPS:
            values = samples_frame.loc[
                samples_frame["measure"].eq("predicted_events_per_10000_match_minutes")
                & samples_frame["history_stratum"].eq(group)
                & samples_frame["all_minutes_last_7d"].eq(burden),
                "estimate",
            ].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            point = float(point_rates[group][index])
            ci_low = float(np.quantile(finite, 0.025)) if len(finite) else np.nan
            ci_high = float(np.quantile(finite, 0.975)) if len(finite) else np.nan
            selected_rows.append(
                {
                    "all_minutes_last_7d": burden,
                    "fragility_group": group,
                    "pred_events_per_match": point * 90.0 / 10_000.0,
                    "pred_events_per_match_ci_low": ci_low * 90.0 / 10_000.0,
                    "pred_events_per_match_ci_high": ci_high * 90.0 / 10_000.0,
                    "pred_events_per_minute": point / 10_000.0,
                    "pred_events_per_minute_ci_low": ci_low / 10_000.0,
                    "pred_events_per_minute_ci_high": ci_high / 10_000.0,
                    "pred_events_per_10000_min": point,
                    "pred_events_per_10000_min_ci_low": ci_low,
                    "pred_events_per_10000_min_ci_high": ci_high,
                    "bootstrap_successful_replicates": int(len(finite)),
                    "interval_method": "player_cluster_bootstrap_percentile",
                }
            )
        ratio_values = samples_frame.loc[
            samples_frame["measure"].eq("higher_vs_intermediate_rate_ratio")
            & samples_frame["all_minutes_last_7d"].eq(burden),
            "estimate",
        ].to_numpy(dtype=float)
        finite_ratios = ratio_values[np.isfinite(ratio_values)]
        point_ratio = float(point_rates["fragile"][index] / point_rates["regular"][index])
        ratio_rows.append(
            {
                "rate_ratio": point_ratio,
                "rr_ci_low": float(np.quantile(finite_ratios, 0.025)) if len(finite_ratios) else np.nan,
                "rr_ci_high": float(np.quantile(finite_ratios, 0.975)) if len(finite_ratios) else np.nan,
                "log_rate_ratio": float(np.log(point_ratio)),
                "log_rate_ratio_se": np.nan,
                "z_statistic": np.nan,
                "p_value": np.nan,
                "all_minutes_last_7d": burden,
                "effect_measure": "incidence_rate_ratio",
                "bootstrap_successful_replicates": int(len(finite_ratios)),
                "interval_method": "player_cluster_bootstrap_percentile",
            }
        )
    return pd.DataFrame(selected_rows), pd.DataFrame(ratio_rows), samples_frame


def weighted_not_estimable_effect_rows(primary_module: Any) -> pd.DataFrame:
    """Create the primary-compatible rows omitted from a bootstrap-only IPW fit."""
    base = {
        "estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "log_estimate": np.nan,
        "standard_error": np.nan,
        "test_statistic": np.nan,
        "p_value": np.nan,
        "fit_status": "not_estimated_bootstrap_only",
        "reason": "IPW intervals use player bootstrap; no Wald interaction test was fitted.",
    }
    rows = [
        {
            **base,
            "contrast_id": "global_spline_by_history_interaction",
            "history_stratum": "joint",
            "burden_from": np.nan,
            "burden_to": np.nan,
            "effect_measure": "chi_square",
            "df": 0,
        }
    ]
    for burden_from, burden_to in primary_module.CONTRAST_WINDOWS:
        suffix = f"{float(burden_to):g}_vs_{float(burden_from):g}"
        for contrast_id, history_stratum in [
            (f"intermediate_history_{suffix}", "intermediate_history"),
            (f"higher_history_{suffix}", "higher_history"),
            (f"ratio_of_{suffix}_changes", "higher_vs_intermediate"),
        ]:
            rows.append(
                {
                    **base,
                    "contrast_id": contrast_id,
                    "history_stratum": history_stratum,
                    "burden_from": float(burden_from),
                    "burden_to": float(burden_to),
                    "effect_measure": "incidence_rate_ratio",
                    "df": 1,
                }
            )
    return pd.DataFrame(rows)


def reported_absence_day_burden_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach public reported absence days to each match row for a burden proxy."""
    _require_columns(frame, [EVENT_COL, "matchproxy_duration_days", MATCH_MINUTES_COL, "model_group"], "burden frame")
    out = frame.copy()
    duration = pd.to_numeric(out["matchproxy_duration_days"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["reported_absence_days_proxy"] = out[EVENT_COL].fillna(0).astype(int) * duration
    out["burden_bin"] = pd.cut(
        pd.to_numeric(out["all_minutes_last_7d"], errors="coerce"),
        bins=BURDEN_BINS,
        labels=BURDEN_LABELS,
        include_lowest=True,
        right=True,
    )
    return out


def cluster_bootstrap_rate_intervals(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    numerator_col: str,
    denominator_col: str,
    scale: float = 60_000.0,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Estimate player-bootstrap intervals for aggregate exposure-standardised rates."""
    _require_columns(frame, [PLAYER_ID_COL, numerator_col, denominator_col, *group_columns], "bootstrap rate frame")
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    if frame.empty:
        return pd.DataFrame()
    player_ids = pd.Index(frame[PLAYER_ID_COL].dropna().unique())
    if len(player_ids) < 2:
        raise ValueError("At least two players are required for player bootstrap intervals")
    groupby_arg: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    grouped = frame.groupby([PLAYER_ID_COL, *group_columns], observed=False, dropna=False).agg(
        numerator=(numerator_col, "sum"), denominator=(denominator_col, "sum")
    ).reset_index()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(player_ids), size=(replicates, len(player_ids)))
    rows: list[dict[str, Any]] = []
    for keys, subset in grouped.groupby(groupby_arg, observed=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        # Keep only numeric contributions before reindexing. Group labels can
        # be categoricals, and they are not meaningful player-level values to
        # zero-fill in a bootstrap draw.
        contribution = (
            subset[[PLAYER_ID_COL, "numerator", "denominator"]]
            .set_index(PLAYER_ID_COL)
            .reindex(player_ids)
            .fillna(0.0)
        )
        numerator = contribution["numerator"].to_numpy(dtype=float)
        denominator = contribution["denominator"].to_numpy(dtype=float)
        observed_numerator = float(numerator.sum())
        observed_denominator = float(denominator.sum())
        bootstrap_numerator = numerator[draws].sum(axis=1)
        bootstrap_denominator = denominator[draws].sum(axis=1)
        rates = np.divide(
            bootstrap_numerator * scale,
            bootstrap_denominator,
            out=np.full(replicates, np.nan),
            where=bootstrap_denominator > 0,
        )
        finite = rates[np.isfinite(rates)]
        row: dict[str, Any] = {
            **dict(zip(group_columns, keys)),
            "players": int(len(player_ids)),
            "numerator": observed_numerator,
            "denominator_minutes": observed_denominator,
            "rate_per_1000_match_hours": (
                observed_numerator * scale / observed_denominator
                if observed_denominator > 0
                else np.nan
            ),
            "bootstrap_replicates": int(replicates),
            "bootstrap_successful_replicates": int(len(finite)),
            "rate_ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
            "rate_ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def conditional_reported_duration_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Model reported absence duration conditional on a proxy event.

    This is the severity component of a public-data two-part model. The outcome
    is one plus reported absence days so that zero-day records remain defined.
    The resulting ratios do not represent medically confirmed time loss.
    """
    event_rows = frame[frame[EVENT_COL].fillna(0).astype(int).eq(1)].copy()
    if event_rows.empty:
        return pd.DataFrame()
    event_rows["burden_per_90"] = event_rows["all_minutes_last_7d"].astype(float) / 90.0
    event_rows["one_plus_reported_absence_days"] = event_rows["reported_absence_days_proxy"].astype(float) + 1.0
    formula = (
        "one_plus_reported_absence_days ~ burden_per_90 * C(model_group) "
        "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
    )
    result = smf.glm(
        formula=formula,
        data=event_rows,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit(cov_type="cluster", cov_kwds={"groups": event_rows[PLAYER_ID_COL]})
    rows: list[dict[str, Any]] = []
    for term in result.params.index:
        estimate = float(result.params[term])
        standard_error = float(result.bse[term])
        ratio, ci_low, ci_high = _z_interval(estimate, standard_error)
        rows.append(
            {
                "term": str(term),
                "effect_measure": "ratio_of_one_plus_reported_absence_days",
                "estimate": ratio,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": _normal_p_value(estimate, standard_error),
                "n_proxy_events": int(len(event_rows)),
                "n_players": int(event_rows[PLAYER_ID_COL].nunique()),
                "fit_status": "ok",
            }
        )
    return primary_adjusted_table(pd.DataFrame(rows), "reported_duration")


def primary_adjusted_table(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    """Attach an isolated exploratory multiplicity family to a result table."""
    out = frame.copy()
    out["analysis_family"] = family
    adjusted = primary_p_adjust(out["p_value"])
    out["p_holm_extension_family"] = adjusted["holm"]
    out["p_bh_extension_family"] = adjusted["bh"]
    return out


def joint_burden_recovery_support(
    primary_module: Any,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Map observed support before fitting any joint burden-recovery model."""
    _require_columns(
        frame,
        ["all_minutes_last_7d", "recovery_interval_refined", "all_games_last_7d", MATCH_MINUTES_COL, EVENT_COL, "model_group"],
        "joint support frame",
    )
    out = frame.copy()
    out["burden_bin"] = pd.cut(
        out["all_minutes_last_7d"].astype(float),
        bins=BURDEN_BINS,
        labels=BURDEN_LABELS,
        include_lowest=True,
        right=True,
    )
    grouped = out.groupby(["model_group", "burden_bin", "recovery_interval_refined"], observed=False, dropna=False)
    rows: list[dict[str, Any]] = []
    for (group, burden, recovery), subset in grouped:
        minutes = float(subset[MATCH_MINUTES_COL].sum())
        events = int(subset[EVENT_COL].sum())
        rates = primary_module.count_rate_intervals(events, minutes)
        rows.append(
            {
                "history_stratum": group,
                "burden_bin": str(burden),
                "recovery_interval": str(recovery),
                "match_rows": int(len(subset)),
                "players": int(subset[PLAYER_ID_COL].nunique()),
                "events": events,
                "match_minutes": minutes,
                "all_games_last_7d_min": float(subset["all_games_last_7d"].min()),
                "all_games_last_7d_max": float(subset["all_games_last_7d"].max()),
                "support_status": (
                    "supported_for_joint_model"
                    if len(subset) >= MIN_JOINT_CELL_ROWS and events >= MIN_JOINT_CELL_EVENTS
                    else "sparse_or_structurally_unavailable"
                ),
                **rates,
            }
        )
    return pd.DataFrame(rows)


def supported_schedule_compression_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate a limited one-versus-two-match contrast in observed support.

    The comparison is restricted before fitting to 45--180 prior minutes,
    0--5 recovery days, and one or two previous club appearances. It does not
    extrapolate to the unsupported high-burden tail or claim a full two-
    dimensional exposure-response surface.
    """
    _require_columns(
        frame,
        ["all_games_last_7d", "all_minutes_last_7d", "recovery_interval_refined", EVENT_COL, "log_minutes_played"],
        "schedule compression frame",
    )
    subset = frame[
        frame["all_minutes_last_7d"].between(45.0, 180.0, inclusive="both")
        & frame["recovery_interval_refined"].isin(SUPPORTED_RECOVERY_LEVELS)
        & frame["all_games_last_7d"].isin([1, 2])
    ].copy()
    if subset.empty or subset[EVENT_COL].sum() <= 0:
        return pd.DataFrame(
            [
                {
                    "contrast": "two_vs_one_prior_club_matches",
                    "effect_measure": "incidence_rate_ratio",
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "fit_status": "not_estimable",
                    "n_match_rows": int(len(subset)),
                    "n_players": int(subset[PLAYER_ID_COL].nunique()),
                    "n_events": int(subset[EVENT_COL].sum()) if EVENT_COL in subset else 0,
                }
            ]
        )
    subset["all_games_last_7d"] = pd.Categorical(subset["all_games_last_7d"], categories=[1, 2])
    subset["model_group"] = pd.Categorical(subset["model_group"], categories=list(MODEL_GROUPS))
    formula = (
        f"{EVENT_COL} ~ C(all_games_last_7d, Treatment(reference=1)) * "
        "C(model_group, Treatment(reference='regular')) + all_minutes_last_7d "
        "+ C(recovery_interval_refined) + week_phase_sin + week_phase_cos "
        "+ halfweek_phase_sin + halfweek_phase_cos"
    )
    result = smf.glm(
        formula=formula,
        data=subset,
        family=sm.families.Poisson(),
        offset=subset["log_minutes_played"],
    ).fit(cov_type="cluster", cov_kwds={"groups": subset[PLAYER_ID_COL]})
    terms = {str(term): index for index, term in enumerate(result.params.index)}
    two_term = next((term for term in terms if "all_games_last_7d" in term and "[T.2]" in term and ":" not in term), None)
    interaction_term = next((term for term in terms if "all_games_last_7d" in term and "[T.2]" in term and ":" in term), None)
    definitions = [
        ("intermediate_history_two_vs_one", {two_term: 1.0} if two_term else {}),
        (
            "higher_history_two_vs_one",
            {term: weight for term, weight in [(two_term, 1.0), (interaction_term, 1.0)] if term},
        ),
        ("difference_in_two_vs_one_changes", {interaction_term: 1.0} if interaction_term else {}),
    ]
    rows: list[dict[str, Any]] = []
    for contrast, weights in definitions:
        if not weights:
            interval = {"rate_ratio": np.nan, "rr_ci_low": np.nan, "rr_ci_high": np.nan, "p_value": np.nan}
            status = "not_estimable"
        else:
            interval = named_coefficient_interval(result.params, result.cov_params(), weights)
            status = "ok"
        rows.append(
            {
                "contrast": contrast,
                "effect_measure": "incidence_rate_ratio",
                "estimate": interval["rate_ratio"],
                "ci_low": interval["rr_ci_low"],
                "ci_high": interval["rr_ci_high"],
                "p_value": interval["p_value"],
                "fit_status": status,
                "n_match_rows": int(len(subset)),
                "n_players": int(subset[PLAYER_ID_COL].nunique()),
                "n_events": int(subset[EVENT_COL].sum()),
                "supported_minutes": "45-180",
                "supported_recovery": "0-5 days",
                "prior_match_count_comparison": "2 versus 1",
            }
        )
    return primary_adjusted_table(pd.DataFrame(rows), "joint_schedule_compression")


def named_coefficient_interval(
    params: pd.Series,
    covariance: pd.DataFrame | np.ndarray,
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Return an exponentiated interval for a named linear coefficient contrast."""
    names = list(params.index)
    vector = np.zeros(len(names), dtype=float)
    for term, weight in weights.items():
        if term not in names:
            raise KeyError(f"Coefficient not found: {term}")
        vector[names.index(term)] = float(weight)
    estimate = float(vector @ np.asarray(params, dtype=float))
    variance = float(vector @ np.asarray(covariance, dtype=float) @ vector.T)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    ratio, ci_low, ci_high = _z_interval(estimate, standard_error)
    return {
        "rate_ratio": ratio,
        "rr_ci_low": ci_low,
        "rr_ci_high": ci_high,
        "p_value": _normal_p_value(estimate, standard_error),
        "log_rate_ratio": estimate,
        "standard_error": standard_error,
    }


def prepare_case_crossover_frame(primary_module: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Keep player-season strata containing both an event and a control appearance."""
    _require_columns(frame, [PLAYER_ID_COL, "date", EVENT_COL, "all_minutes_last_7d", "model_group"], "case-crossover frame")
    out = frame.copy()
    out["season_start"] = primary_module.season_from_dates(out["date"]).astype(int)
    out["player_season_stratum"] = out[PLAYER_ID_COL].astype(str) + "_" + out["season_start"].astype(str)
    counts = out.groupby("player_season_stratum", observed=False)[EVENT_COL].agg(["sum", "count"])
    eligible = counts[(counts["sum"] > 0) & (counts["sum"] < counts["count"])].index
    out = out[out["player_season_stratum"].isin(eligible)].copy()
    out["burden_per_90"] = out["all_minutes_last_7d"].astype(float) / 90.0
    out["higher_history"] = out["model_group"].eq("fragile").astype(float)
    out["burden_by_higher_history"] = out["burden_per_90"] * out["higher_history"]
    return out


def _conditional_interval(
    result: Any,
    term_weights: Mapping[str, float],
) -> dict[str, float]:
    """Compute a conditional-logistic odds-ratio contrast from result covariance."""
    params = pd.Series(result.params, index=result.model.exog_names)
    covariance = result.cov_params()
    interval = named_coefficient_interval(params, covariance, term_weights)
    return {
        "estimate": interval["rate_ratio"],
        "ci_low": interval["rr_ci_low"],
        "ci_high": interval["rr_ci_high"],
        "p_value": interval["p_value"],
    }


def fit_case_crossover_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit within-player-season conditional-logistic burden and recovery models."""
    _require_columns(
        frame,
        [EVENT_COL, "player_season_stratum", "burden_per_90", "higher_history", "burden_by_higher_history"],
        "case-crossover model frame",
    )
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    model_specs: list[tuple[str, pd.DataFrame, pd.DataFrame, list[tuple[str, Mapping[str, float]]], str]] = []
    burden_x = frame[
        [
            "burden_per_90",
            "higher_history",
            "burden_by_higher_history",
            "week_phase_sin",
            "week_phase_cos",
        ]
    ].astype(float)
    model_specs.append(
        (
            "previous_7day_minutes",
            frame,
            burden_x,
            [
                ("intermediate_per_90_minutes", {"burden_per_90": 1.0}),
                (
                    "higher_per_90_minutes",
                    {"burden_per_90": 1.0, "burden_by_higher_history": 1.0},
                ),
                ("difference_in_per_90_minutes", {"burden_by_higher_history": 1.0}),
            ],
            "conditional_odds_ratio_per_90_previous_minutes",
        )
    )
    recovery = frame[frame["recovery_interval_refined"].isin(("0-3 days", "4-5 days", "6-7 days"))].copy()
    if not recovery.empty:
        score_map = {"0-3 days": 2.0, "4-5 days": 1.0, "6-7 days": 0.0}
        recovery["shorter_recovery_step"] = recovery["recovery_interval_refined"].map(score_map).astype(float)
        recovery["shorter_recovery_by_higher_history"] = recovery["shorter_recovery_step"] * recovery["higher_history"]
        recovery_x = recovery[
            [
                "shorter_recovery_step",
                "higher_history",
                "shorter_recovery_by_higher_history",
                "all_minutes_last_7d",
                "week_phase_sin",
                "week_phase_cos",
            ]
        ].astype(float)
        model_specs.append(
            (
                "recovery_shortness",
                recovery,
                recovery_x,
                [
                    ("intermediate_per_category_shorter", {"shorter_recovery_step": 1.0}),
                    (
                        "higher_per_category_shorter",
                        {"shorter_recovery_step": 1.0, "shorter_recovery_by_higher_history": 1.0},
                    ),
                    ("difference_in_shorter_recovery", {"shorter_recovery_by_higher_history": 1.0}),
                ],
                "conditional_odds_ratio_per_one_category_shorter_recovery",
            )
        )
    for model_name, subset, exog, contrasts, measure in model_specs:
        counts = subset.groupby("player_season_stratum", observed=False)[EVENT_COL].agg(["sum", "count"])
        usable = counts[(counts["sum"] > 0) & (counts["sum"] < counts["count"])].index
        subset = subset[subset["player_season_stratum"].isin(usable)].copy()
        exog = exog.loc[subset.index].copy()
        if subset.empty or not len(usable):
            for contrast, _ in contrasts:
                rows.append(
                    {
                        "model": model_name,
                        "contrast": contrast,
                        "effect_measure": measure,
                        "estimate": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": np.nan,
                        "fit_status": "not_estimable",
                        "n_match_rows": int(len(subset)),
                        "n_player_seasons": 0,
                        "n_players": int(subset[PLAYER_ID_COL].nunique()),
                        "n_events": int(subset[EVENT_COL].sum()) if EVENT_COL in subset else 0,
                    }
                )
            continue
        try:
            result = ConditionalLogit(
                subset[EVENT_COL].astype(int),
                exog,
                groups=subset["player_season_stratum"],
            ).fit(disp=False, maxiter=200)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            for contrast, _ in contrasts:
                rows.append(
                    {
                        "model": model_name,
                        "contrast": contrast,
                        "effect_measure": measure,
                        "estimate": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": np.nan,
                        "fit_status": "failed_convergence",
                        "n_match_rows": int(len(subset)),
                        "n_player_seasons": int(len(usable)),
                        "n_players": int(subset[PLAYER_ID_COL].nunique()),
                        "n_events": int(subset[EVENT_COL].sum()),
                    }
                )
            continue
        for contrast, weights in contrasts:
            interval = _conditional_interval(result, weights)
            rows.append(
                {
                    "model": model_name,
                    "contrast": contrast,
                    "effect_measure": measure,
                    **interval,
                    "fit_status": "ok",
                    "n_match_rows": int(len(subset)),
                    "n_player_seasons": int(len(usable)),
                    "n_players": int(subset[PLAYER_ID_COL].nunique()),
                    "n_events": int(subset[EVENT_COL].sum()),
                }
            )
    return primary_adjusted_table(pd.DataFrame(rows), "within_player_case_crossover")


def attach_current_match_metadata(
    primary_module: Any,
    frame: pd.DataFrame,
    transfermarkt_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach a current club-match identifier and competition context.

    A player-date with one source game can share that match identifier with
    teammates. Multiple or unmatched source games are deliberately assigned a
    unique player-date cluster, preventing an invented shared-match cluster.
    """
    _require_columns(frame, [PLAYER_ID_COL, "date", MATCH_MINUTES_COL], "match metadata frame")
    games = pd.read_csv(
        transfermarkt_dir / "games.csv",
        usecols=["game_id", "competition_id", "season", "home_club_id", "away_club_id"],
        low_memory=False,
    )
    competitions = pd.read_csv(
        transfermarkt_dir / "competitions.csv",
        usecols=["competition_id", "type"],
        low_memory=False,
    )
    appearances = pd.read_csv(
        transfermarkt_dir / "appearances.csv",
        usecols=["game_id", "player_id", "player_club_id", "date", "minutes_played"],
        low_memory=False,
    ).rename(columns={"player_id": PLAYER_ID_COL})
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances = appearances.merge(games, on="game_id", how="left")
    epl_club_seasons = primary_module.epl_club_seasons(games)
    appearances = appearances.merge(epl_club_seasons, on=["season", "player_club_id"], how="inner")
    appearances = appearances.merge(competitions, on="competition_id", how="left")
    keys = frame[[PLAYER_ID_COL, "date"]].drop_duplicates()
    appearances = appearances.merge(keys, on=[PLAYER_ID_COL, "date"], how="inner")
    appearances["competition_context"] = np.select(
        [
            appearances["competition_id"].eq("GB1"),
            appearances["type"].eq("domestic_cup"),
            appearances["type"].eq("international_cup"),
        ],
        ["Premier League", "domestic cup", "UEFA/international club"],
        default="other club competition",
    )
    rows: list[dict[str, Any]] = []
    for (player_id, date), subset in appearances.groupby([PLAYER_ID_COL, "date"], sort=False):
        games_on_date = subset["game_id"].dropna().astype(str).unique()
        if len(games_on_date) == 1:
            match_cluster_id = games_on_date[0]
            context = str(subset["competition_context"].iloc[0])
            status = "unique_source_match"
        else:
            match_cluster_id = f"multiple_{player_id}_{pd.Timestamp(date).date()}"
            context = "multiple club matches on date"
            status = "multiple_source_matches"
        rows.append(
            {
                PLAYER_ID_COL: int(player_id),
                "date": pd.Timestamp(date),
                "match_cluster_id": match_cluster_id,
                "competition_context": context,
                "match_link_status": status,
                "source_games_on_date": int(len(games_on_date)),
                "source_minutes_on_date": float(pd.to_numeric(subset["minutes_played"], errors="coerce").fillna(0.0).sum()),
            }
        )
    metadata = pd.DataFrame(rows)
    metadata_columns = [
        "match_cluster_id",
        "competition_context",
        "match_link_status",
        "source_games_on_date",
        "source_minutes_on_date",
    ]
    out = frame.drop(columns=metadata_columns, errors="ignore").merge(
        metadata,
        on=[PLAYER_ID_COL, "date"],
        how="left",
    )
    missing = out["match_cluster_id"].isna()
    out.loc[missing, "match_cluster_id"] = (
        "unmatched_" + out.loc[missing, PLAYER_ID_COL].astype(str) + "_" + out.loc[missing, "date"].astype(str)
    )
    out.loc[missing, "competition_context"] = "unmatched club match date"
    out.loc[missing, "match_link_status"] = "unmatched_source_match"
    out["source_games_on_date"] = out["source_games_on_date"].fillna(0).astype(int)
    out["source_minutes_on_date"] = out["source_minutes_on_date"].fillna(0.0)
    audit = (
        out.groupby("match_link_status", dropna=False)
        .agg(
            match_rows=(PLAYER_ID_COL, "size"),
            players=(PLAYER_ID_COL, "nunique"),
            events=(EVENT_COL, "sum"),
            observed_minutes=(MATCH_MINUTES_COL, "sum"),
            source_minutes=("source_minutes_on_date", "sum"),
        )
        .reset_index()
    )
    audit["source_minus_observed_minutes"] = audit["source_minutes"] - audit["observed_minutes"]
    return out, audit


def competition_context_outputs(
    primary_module: Any,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise current-match context and refit with context control/restriction."""
    _require_columns(frame, ["competition_context", EVENT_COL, MATCH_MINUTES_COL], "competition context frame")
    rates: list[dict[str, Any]] = []
    for context, subset in frame.groupby("competition_context", observed=False, dropna=False):
        minutes = float(subset[MATCH_MINUTES_COL].sum())
        events = int(subset[EVENT_COL].sum())
        interval = primary_module.count_rate_intervals(events, minutes)
        rates.append(
            {
                "competition_context": str(context),
                "match_rows": int(len(subset)),
                "players": int(subset[PLAYER_ID_COL].nunique()),
                "events": events,
                "match_minutes": minutes,
                **interval,
            }
        )
    rate_table = pd.DataFrame(rates)
    refits: list[dict[str, Any]] = []
    specs = [
        (
            "competition_context_adjusted",
            frame,
            "+ C(competition_context)",
            {"competition_context": "Premier League"},
            "all current competition contexts, predicted at Premier League",
        ),
        (
            "premier_league_current_match_only",
            frame[frame["competition_context"].eq("Premier League")].copy(),
            "",
            None,
            "current match recorded as Premier League",
        ),
    ]
    for label, subset, controls, covars, restriction in specs:
        if subset.empty or subset[EVENT_COL].sum() <= 0 or set(subset["model_group"].unique()) != set(MODEL_GROUPS):
            refits.append(
                {
                    "model": label,
                    "restriction": restriction,
                    "fit_status": "not_estimable",
                    "n_match_rows": int(len(subset)),
                    "n_players": int(subset[PLAYER_ID_COL].nunique()),
                    "n_events": int(subset[EVENT_COL].sum()) if EVENT_COL in subset else 0,
                }
            )
            continue
        bundle = primary_module.run_prediction_bundle(
            subset,
            EVENT_COL,
            controls=controls,
            extra_covars=covars,
        )
        row = _bundle_summary(primary_module, label, subset, bundle, restriction)
        row["analysis_family"] = "competition_context_sensitivity"
        refits.append(row)
    return rate_table, pd.DataFrame(refits)


def two_way_cluster_sensitivity(
    primary_module: Any,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Recalculate selected spline contrasts with player and match clustering."""
    _require_columns(frame, ["match_cluster_id", EVENT_COL, "log_minutes_played", "model_group"], "two-way clustering frame")
    if frame.empty or frame[EVENT_COL].sum() <= 0:
        return pd.DataFrame()
    burden_max = float(frame["all_minutes_last_7d"].max())
    formula = primary_module.spline_formula(EVENT_COL, burden_max)
    result = smf.glm(
        formula=formula,
        data=frame,
        family=sm.families.Poisson(),
        offset=frame["log_minutes_played"],
    ).fit()
    player_codes = pd.factorize(frame[PLAYER_ID_COL])[0]
    match_codes = pd.factorize(frame["match_cluster_id"])[0]
    covariance, _, _ = cov_cluster_2groups(result, player_codes, match_codes)
    design_info = result.model.data.design_info
    params = np.asarray(result.params)
    rows: list[dict[str, Any]] = []
    for burden in (0.0, 90.0, 180.0):
        fragile = np.asarray(
            build_design_matrices([design_info], primary_module.prediction_template([burden], "fragile"))[0]
        )[0]
        regular = np.asarray(
            build_design_matrices([design_info], primary_module.prediction_template([burden], "regular"))[0]
        )[0]
        interval = primary_module.delta_ratio_interval(params, covariance, fragile, regular)
        rows.append(
            {
                "contrast": f"higher_vs_intermediate_at_{int(burden)}_minutes",
                "effect_measure": "incidence_rate_ratio",
                "estimate": interval["rate_ratio"],
                "ci_low": interval["rr_ci_low"],
                "ci_high": interval["rr_ci_high"],
                "p_value": interval["p_value"],
                "df": 1.0,
                "n_match_rows": int(len(frame)),
                "n_players": int(frame[PLAYER_ID_COL].nunique()),
                "n_match_clusters": int(frame["match_cluster_id"].nunique()),
                "n_events": int(frame[EVENT_COL].sum()),
                "covariance": "two_way_player_and_current_match",
                "fit_status": "ok",
            }
        )
    names = [str(name) for name in result.params.index]
    interaction_indices = [
        index for index, name in enumerate(names) if name.startswith(("bs(", "cr(")) and ":model_group" in name
    ]
    if interaction_indices:
        beta = params[interaction_indices]
        cov_sub = covariance[np.ix_(interaction_indices, interaction_indices)]
        statistic = float(beta.T @ np.linalg.pinv(cov_sub) @ beta)
        p_value = float(chi2.sf(statistic, len(interaction_indices)))
    else:
        statistic = np.nan
        p_value = np.nan
    rows.append(
        {
            "contrast": "global_spline_by_history_interaction",
            "effect_measure": "chi_square",
            "estimate": statistic,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": p_value,
            "df": float(len(interaction_indices)),
            "n_match_rows": int(len(frame)),
            "n_players": int(frame[PLAYER_ID_COL].nunique()),
            "n_match_clusters": int(frame["match_cluster_id"].nunique()),
            "n_events": int(frame[EVENT_COL].sum()),
            "covariance": "two_way_player_and_current_match",
            "fit_status": "ok" if interaction_indices else "not_estimable",
        }
    )
    return primary_adjusted_table(pd.DataFrame(rows), "two_way_cluster_uncertainty")


def curve_feature_bootstrap(
    primary_module: Any,
    frame: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap player clusters and record the fitted curve maximum location."""
    _require_columns(frame, [PLAYER_ID_COL, EVENT_COL, "log_minutes_played", "model_group"], "curve bootstrap frame")
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    players = pd.Index(frame[PLAYER_ID_COL].dropna().unique())
    if len(players) < 2:
        raise ValueError("At least two players are required for curve bootstrap")
    burden_max = float(frame["all_minutes_last_7d"].max())
    formula = primary_module.spline_formula(EVENT_COL, burden_max)
    initial = smf.glm(
        formula=formula,
        data=frame,
        family=sm.families.Poisson(),
        offset=frame["log_minutes_played"],
    ).fit()
    design_info = initial.model.data.design_info
    templates = {
        group: np.asarray(
            build_design_matrices(
                [design_info], primary_module.prediction_template(BOOTSTRAP_GRID, group)
            )[0]
        )
        for group in MODEL_GROUPS
    }
    player_codes = pd.factorize(frame[PLAYER_ID_COL], sort=True)[0]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        draw = rng.integers(0, len(players), size=len(players))
        weights_by_player = np.bincount(draw, minlength=len(players))
        weights = weights_by_player[player_codes]
        try:
            result = sm.GLM(
                frame[EVENT_COL].astype(float).to_numpy(),
                initial.model.exog,
                family=sm.families.Poisson(),
                offset=frame["log_minutes_played"].astype(float).to_numpy(),
                freq_weights=weights,
            ).fit()
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            for group in MODEL_GROUPS:
                rows.append(
                    {
                        "replicate": replicate,
                        "history_stratum": group,
                        "fit_status": "failed",
                        "global_max_minutes": np.nan,
                        "global_max_in_15_45_band": np.nan,
                        "early_band_max_vs_90_ratio": np.nan,
                    }
                )
            continue
        for group in MODEL_GROUPS:
            rates = np.exp(templates[group] @ np.asarray(result.params) + np.log(90.0)) / 90.0 * 10_000.0
            max_index = int(np.argmax(rates))
            early_rates = rates[(BOOTSTRAP_GRID >= 15.0) & (BOOTSTRAP_GRID <= 45.0)]
            rate_at_90 = float(rates[np.where(BOOTSTRAP_GRID == 90.0)[0][0]])
            rows.append(
                {
                    "replicate": replicate,
                    "history_stratum": group,
                    "fit_status": "ok",
                    "global_max_minutes": float(BOOTSTRAP_GRID[max_index]),
                    "global_max_in_15_45_band": bool(15.0 <= BOOTSTRAP_GRID[max_index] <= 45.0),
                    "early_band_max_vs_90_ratio": float(np.max(early_rates) / rate_at_90) if rate_at_90 > 0 else np.nan,
                }
            )
    samples = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for group, subset in samples.groupby("history_stratum", observed=False):
        successful = subset[subset["fit_status"].eq("ok")].copy()
        n_success = int(len(successful))
        early_count = int(successful["global_max_in_15_45_band"].sum()) if n_success else 0
        early_pct, early_low, early_high = percent_with_interval(early_count, n_success)
        max_locations = successful["global_max_minutes"].dropna().to_numpy(dtype=float)
        ratios = successful["early_band_max_vs_90_ratio"].dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "history_stratum": group,
                "bootstrap_replicates": int(replicates),
                "successful_replicates": n_success,
                "early_band_global_max_replicates": early_count,
                "early_band_global_max_percent": early_pct,
                "early_band_global_max_ci_low": early_low,
                "early_band_global_max_ci_high": early_high,
                "global_max_minutes_median": float(np.median(max_locations)) if len(max_locations) else np.nan,
                "global_max_minutes_percentile_low": float(np.quantile(max_locations, 0.025)) if len(max_locations) else np.nan,
                "global_max_minutes_percentile_high": float(np.quantile(max_locations, 0.975)) if len(max_locations) else np.nan,
                "early_band_max_vs_90_median": float(np.median(ratios)) if len(ratios) else np.nan,
                "early_band_max_vs_90_percentile_low": float(np.quantile(ratios, 0.025)) if len(ratios) else np.nan,
                "early_band_max_vs_90_percentile_high": float(np.quantile(ratios, 0.975)) if len(ratios) else np.nan,
            }
        )
    return samples, pd.DataFrame(summary_rows)


def read_extension_inputs(root: Path, primary_module: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # pragma: no cover
    """Load and prepare the same public-data frame used by the primary model."""
    processed = root / "data" / "processed"
    transfermarkt = root / "external_data" / "transfermarkt"
    panel = pd.read_csv(processed / "player_day_panel_all_comp.csv", parse_dates=["date"], low_memory=False)
    injuries = pd.read_csv(processed / "tm_injuries_clean.csv", low_memory=False)
    injury_episodes = pd.read_csv(processed / "tm_injury_episodes.csv", low_memory=False)
    lineups_path = transfermarkt / "game_lineups.csv"
    lineups = (
        pd.read_csv(lineups_path, usecols=["date", "player_id", "type"], low_memory=False)
        if lineups_path.exists()
        else pd.DataFrame()
    )
    panel = merge_day_fragility(panel, processed)
    panel = primary_module.add_matchproxy_outcome_subsets(panel, injury_episodes)
    panel = primary_module.add_non_muscle_frequency_history_label(panel, injuries)
    panel = primary_module.add_mutually_exclusive_type_frequency_history(panel, injuries)
    panel = restrict_to_fragility_risk_set(panel)
    panel = restrict_to_available_risk_set(panel)
    panel = primary_module.add_calendar_sensitivity_flags(panel)
    panel = primary_module.add_alternative_fragility_labels(panel)
    panel, _ = primary_module.add_out_of_time_fragility_label(panel)
    return panel, injuries, lineups


def write_extension_outputs(outputs: Mapping[str, pd.DataFrame], results_dir: Path) -> None:  # pragma: no cover
    """Write each non-empty audit table using a stable extension filename."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for stem, frame in outputs.items():
        frame.to_csv(results_dir / f"matchproxy_extension_{stem}.csv", index=False)


def main() -> None:  # pragma: no cover
    """Run every additional current-data audit in its documented order."""
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "data" / "processed" / "results"
    transfermarkt = root / "external_data" / "transfermarkt"
    primary_module = load_primary_module()
    panel, injuries, lineups = read_extension_inputs(root, primary_module)
    frame = prepare_extension_frame(primary_module, panel, injuries, lineups)

    print("1. Fitting lineup and return selection refits ...")
    lineup = lineup_refit_outputs(primary_module, frame)
    lineup_interaction = lineup_spline_interaction_tests(primary_module, frame)

    print("2. Auditing reporting completeness and reported absence-day proxy ...")
    reporting_events = classify_reporting_completeness(frame)
    reporting_context = reporting_completeness_by_context(reporting_events)
    _, reporting_terms, weighted_events = reporting_type_model(reporting_events)
    reporting_ipw = reporting_type_ipw_sensitivity(primary_module, frame, weighted_events)
    burden_frame = reported_absence_day_burden_frame(frame)
    burden_by_history = cluster_bootstrap_rate_intervals(
        burden_frame,
        ["model_group"],
        "reported_absence_days_proxy",
        MATCH_MINUTES_COL,
    )
    burden_by_history_burden = cluster_bootstrap_rate_intervals(
        burden_frame,
        ["model_group", "burden_bin"],
        "reported_absence_days_proxy",
        MATCH_MINUTES_COL,
    )
    duration_model = conditional_reported_duration_model(burden_frame)

    print("3. Mapping joint recent-burden and recovery support ...")
    joint_support = joint_burden_recovery_support(primary_module, frame)
    schedule_model = supported_schedule_compression_model(frame)

    print("4. Fitting player-season conditional-logistic analyses ...")
    case_crossover_frame = prepare_case_crossover_frame(primary_module, frame)
    case_crossover = fit_case_crossover_models(case_crossover_frame)

    print("5. Reconstructing current-match context and robust uncertainty checks ...")
    metadata_frame, metadata_audit = attach_current_match_metadata(primary_module, frame, transfermarkt)
    context_rates, context_refits = competition_context_outputs(primary_module, metadata_frame)
    two_way = two_way_cluster_sensitivity(primary_module, metadata_frame)
    bootstrap_samples, bootstrap_summary = curve_feature_bootstrap(primary_module, frame)

    outputs: dict[str, pd.DataFrame] = {
        "lineup_refits_summary": lineup["summary"],
        "lineup_refits_predictions": lineup["predictions"],
        "lineup_refits_shape": lineup["shape"],
        "lineup_spline_interaction": lineup_interaction,
        "reporting_completeness_context": reporting_context,
        "reporting_type_model": reporting_terms,
        "reporting_type_ipw_summary": reporting_ipw["summary"],
        "reporting_type_ipw_selected": reporting_ipw["selected"],
        "reporting_type_ipw_ratios": reporting_ipw["ratios"],
        "reporting_type_ipw_bootstrap": reporting_ipw["bootstrap_samples"],
        "reporting_type_ipw_diagnostics": reporting_ipw["diagnostics"],
        "reported_absence_day_burden_history": burden_by_history,
        "reported_absence_day_burden_history_by_burden": burden_by_history_burden,
        "reported_duration_conditional_model": duration_model,
        "joint_burden_recovery_support": joint_support,
        "joint_schedule_compression_model": schedule_model,
        "within_player_case_crossover": case_crossover,
        "current_match_metadata_audit": metadata_audit,
        "competition_context_rates": context_rates,
        "competition_context_refits": context_refits,
        "two_way_cluster_sensitivity": two_way,
        "curve_feature_bootstrap_samples": bootstrap_samples,
        "curve_feature_bootstrap_summary": bootstrap_summary,
    }
    write_extension_outputs(outputs, results_dir)
    print(f"Wrote {len(outputs)} extension tables to {results_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
