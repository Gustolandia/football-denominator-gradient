#!/usr/bin/env python
"""Audit scientific consequences of the public-data v4 exposure extension.

The module answers two distinct questions.  First, it repeats the frozen
seven-day exposure-response analysis after adding progressively broader
country-duty scopes.  Second, it explores whether a subsequent club appearance
by an EPL-cohort player shortly after senior competitive international duty
has a different within-player match-proxy incidence rate.  The second question
was identified during the v4 audit, so every result is labelled post-hoc and
hypothesis-generating.

No model in this file changes the frozen injury outcome or prior-history
classification.  A failed official-schedule coverage gate keeps all expanded
exposure results in sensitivity status.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests

from v4_statistics import percent_with_interval


PLAYER_ID = "tm_player_id"
PRIMARY_EVENT = "injury_event_matchproxy"
PRIMARY_HISTORY = "fragility_group"
WINDOW_DAYS = (3, 5, 7, 14, 28)
NATIONAL_INCREMENT_SCOPES = {
    "senior_competitive": "senior_competitive_national_only",
    "senior_all": "senior_all_national_only",
    "broader_international": "broader_international_only",
}
PUBLIC_HISTORY_LABELS = {
    "regular": "intermediate prior-injury history",
    "fragile": "higher prior-injury history",
    "tough": "lower prior-injury history with high exposure",
    "low exposure": "lower prior exposure",
}


def _required(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def load_module(path: Path, name: str) -> ModuleType:
    """Load a numbered pipeline module without copying its implementation."""
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percent(numerator: int, denominator: int) -> float:
    """Return a percentage or missing when its source denominator is empty."""
    return float(numerator / denominator * 100.0) if denominator else np.nan


def national_record_quality_audit(
    record_audit: pd.DataFrame,
    retained_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise raw-record exclusions before any exposure filtering."""
    _required(
        record_audit,
        [
            PLAYER_ID,
            "game_id",
            "participation_state",
            "minutes_played",
            "competition_type_id",
            "competition_id",
            "kickoff_time_known",
            "retained_for_exposure",
            "exclusion_reason",
        ],
        "national record audit",
    )
    _required(
        retained_appearances,
        [PLAYER_ID, "game_id", "source", "minutes_played"],
        "retained national appearances",
    )
    records = record_audit.copy()
    played = records["participation_state"].eq("played")
    minutes_known = records["minutes_played"].notna()
    senior = pd.to_numeric(records["competition_type_id"], errors="coerce").isin(
        {11, 19}
    )
    friendly = records["competition_id"].fillna("").astype(str).str.casefold().eq("fs")
    primary = played & senior & ~friendly
    retained = records["retained_for_exposure"].fillna(False).astype(bool)
    metrics = [
        ("in_window_national_records", len(records), len(records)),
        ("played_national_records", int(played.sum()), len(records)),
        ("nonplayed_national_records", int((~played).sum()), len(records)),
        (
            "played_records_with_missing_minutes",
            int((played & ~minutes_known).sum()),
            int(played.sum()),
        ),
        (
            "played_records_with_known_minutes",
            int((played & minutes_known).sum()),
            int(played.sum()),
        ),
        (
            "primary_senior_competitive_played_records",
            int(primary.sum()),
            int(played.sum()),
        ),
        (
            "primary_senior_competitive_missing_minutes",
            int((primary & ~minutes_known).sum()),
            int(primary.sum()),
        ),
        ("records_retained_for_exposure", int(retained.sum()), len(records)),
        (
            "retained_records_with_known_kickoff_time",
            int(records.loc[retained, "kickoff_time_known"].fillna(False).astype(bool).sum()),
            int(retained.sum()),
        ),
        (
            "retained_duplicate_player_games",
            int(retained_appearances.duplicated([PLAYER_ID, "game_id"]).sum()),
            len(retained_appearances),
        ),
    ]
    rows = []
    for metric, count, denominator in metrics:
        percent, ci_low, ci_high = percent_with_interval(count, denominator)
        rows.append(
            {
                "metric": metric,
                "count": count,
                "denominator": denominator,
                "percent": percent,
                "percent_ci_low": ci_low,
                "percent_ci_high": ci_high,
                "interval_method": "wilson_95",
            }
        )
    for source, group in retained_appearances.groupby("source", dropna=False):
        percent, ci_low, ci_high = percent_with_interval(
            len(group), len(retained_appearances)
        )
        rows.append(
            {
                "metric": f"retained_source_rows:{source}",
                "count": len(group),
                "denominator": len(retained_appearances),
                "percent": percent,
                "percent_ci_low": ci_low,
                "percent_ci_high": ci_high,
                "interval_method": "wilson_95",
            }
        )
    for reason, group in records.loc[~retained].groupby("exclusion_reason", dropna=False):
        percent, ci_low, ci_high = percent_with_interval(len(group), len(records))
        rows.append(
            {
                "metric": f"excluded_records:{reason}",
                "count": len(group),
                "denominator": len(records),
                "percent": percent,
                "percent_ci_low": ci_low,
                "percent_ci_high": ci_high,
                "interval_method": "wilson_95",
            }
        )
    return pd.DataFrame(rows)


def merge_national_increment(
    match_panel: pd.DataFrame,
    exposure_features: pd.DataFrame,
    increment_scope: str,
    window_days: int,
) -> pd.DataFrame:
    """Attach prior country minutes and appearances to unchanged match rows."""
    minutes = f"{increment_scope}_minutes_last_{window_days}d"
    matches = f"{increment_scope}_matches_last_{window_days}d"
    _required(match_panel, [PLAYER_ID, "date"], "match panel")
    _required(exposure_features, [PLAYER_ID, "date", minutes, matches], "exposure features")
    panel = match_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    additions = exposure_features[[PLAYER_ID, "date", minutes, matches]].copy()
    additions["date"] = pd.to_datetime(additions["date"], errors="coerce")
    if additions.duplicated([PLAYER_ID, "date"]).any():
        raise ValueError("Exposure features must have one row per player-date")
    out = panel.merge(additions, on=[PLAYER_ID, "date"], how="left", validate="many_to_one")
    out["national_minutes_in_window"] = pd.to_numeric(out[minutes], errors="coerce").fillna(0.0)
    out["national_matches_in_window"] = pd.to_numeric(out[matches], errors="coerce").fillna(0.0)
    out["recent_national_duty"] = out["national_matches_in_window"].gt(0).astype(int)
    return out


def exposure_change_audit(
    match_panel: pd.DataFrame,
    exposure_features: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify which analysed rows and events receive country-duty exposure."""
    _required(
        match_panel,
        [PLAYER_ID, "date", PRIMARY_EVENT, PRIMARY_HISTORY, "all_minutes_last_7d"],
        "match panel",
    )
    rows: list[dict[str, object]] = []
    for scope_label, increment_scope in NATIONAL_INCREMENT_SCOPES.items():
        for window_days in WINDOW_DAYS:
            panel = merge_national_increment(
                match_panel, exposure_features, increment_scope, window_days
            )
            for raw_group, group in [("all", panel), *list(panel.groupby(PRIMARY_HISTORY))]:
                changed = group["national_minutes_in_window"].gt(0)
                positive = group.loc[changed, "national_minutes_in_window"]
                percent, ci_low, ci_high = percent_with_interval(
                    int(changed.sum()), len(group)
                )
                rows.append(
                    {
                        "country_scope": scope_label,
                        "window_days": window_days,
                        "history_stratum": PUBLIC_HISTORY_LABELS.get(raw_group, raw_group),
                        "n_match_rows": int(len(group)),
                        "n_rows_with_country_minutes": int(changed.sum()),
                        "percent_rows_with_country_minutes": percent,
                        "percent_rows_with_country_minutes_ci_low": ci_low,
                        "percent_rows_with_country_minutes_ci_high": ci_high,
                        "percent_interval_method": "wilson_95",
                        "n_proxy_events_on_changed_rows": int(
                            pd.to_numeric(group.loc[changed, PRIMARY_EVENT], errors="coerce")
                            .fillna(0)
                            .sum()
                        ),
                        "total_country_minutes": float(positive.sum()),
                        "median_country_minutes_when_positive": float(positive.median())
                        if not positive.empty
                        else np.nan,
                        "p90_country_minutes_when_positive": float(positive.quantile(0.9))
                        if not positive.empty
                        else np.nan,
                        "maximum_country_minutes": float(positive.max())
                        if not positive.empty
                        else np.nan,
                        "zero_club_burden_rows_reclassified": int(
                            (
                                changed
                                & pd.to_numeric(group["all_minutes_last_7d"], errors="coerce").eq(0)
                            ).sum()
                        )
                        if window_days == 7
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def recovery_change_audit(
    match_panel: pd.DataFrame,
    exposure_features: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify recovery intervals shortened by an observed country appearance."""
    _required(
        match_panel,
        [PLAYER_ID, "date", "days_since_last_match", PRIMARY_EVENT, PRIMARY_HISTORY],
        "match panel",
    )
    panel = match_panel[[PLAYER_ID, "date", "days_since_last_match", PRIMARY_EVENT, PRIMARY_HISTORY]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    rows: list[dict[str, object]] = []
    for scope_label, increment_scope in NATIONAL_INCREMENT_SCOPES.items():
        days_column = f"{increment_scope}_days_since_previous_appearance"
        _required(exposure_features, [PLAYER_ID, "date", days_column], "exposure features")
        additions = exposure_features[[PLAYER_ID, "date", days_column]].copy()
        additions["date"] = pd.to_datetime(additions["date"], errors="coerce")
        merged = panel.merge(additions, on=[PLAYER_ID, "date"], how="left", validate="many_to_one")
        club_days = pd.to_numeric(merged["days_since_last_match"], errors="coerce")
        country_days = pd.to_numeric(merged[days_column], errors="coerce")
        expanded_days = np.fmin(club_days, country_days)
        changed = country_days.notna() & (club_days.isna() | expanded_days.ne(club_days))
        for raw_group, group_index in [("all", merged.index), *list(merged.groupby(PRIMARY_HISTORY).groups.items())]:
            group_changed = changed.loc[group_index]
            group = merged.loc[group_index]
            percent, ci_low, ci_high = percent_with_interval(
                int(group_changed.sum()), len(group)
            )
            rows.append(
                {
                    "country_scope": scope_label,
                    "history_stratum": PUBLIC_HISTORY_LABELS.get(raw_group, raw_group),
                    "n_match_rows": len(group),
                    "n_recovery_intervals_changed": int(group_changed.sum()),
                    "percent_recovery_intervals_changed": percent,
                    "percent_recovery_intervals_changed_ci_low": ci_low,
                    "percent_recovery_intervals_changed_ci_high": ci_high,
                    "percent_interval_method": "wilson_95",
                    "n_proxy_events_on_changed_rows": int(
                        pd.to_numeric(group.loc[group_changed, PRIMARY_EVENT], errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def prepare_duty_model_frame(
    base_model: Any,
    match_panel: pd.DataFrame,
    exposure_features: pd.DataFrame,
    increment_scope: str,
    window_days: int,
    event_col: str,
    history_group_col: str = PRIMARY_HISTORY,
) -> pd.DataFrame:
    """Build a two-part within/between-player international-duty exposure."""
    merged = merge_national_increment(
        match_panel, exposure_features, increment_scope, window_days
    )
    prepared = base_model.prepare_model_frame(merged, event_col, history_group_col)
    prepared["higher_history"] = prepared["model_group"].eq("fragile").astype(int)
    prepared["national_match_equivalents"] = prepared["national_minutes_in_window"] / 90.0
    prepared["national_match_count"] = prepared["national_matches_in_window"]
    exposure_columns = {
        "duty": "recent_national_duty",
        "national_match_equivalents": "national_match_equivalents",
        "national_match_count": "national_match_count",
    }
    for prefix, column in exposure_columns.items():
        prepared[f"{prefix}_between"] = prepared.groupby(PLAYER_ID)[column].transform(
            "mean"
        )
        prepared[f"{prefix}_within"] = prepared[column] - prepared[f"{prefix}_between"]
    return prepared


def _duty_formula(
    base_model: Any,
    frame: pd.DataFrame,
    event_col: str,
    controls: str,
    exposure_prefix: str = "duty",
) -> str:
    """Return the source-specific model with the frozen club-load spline."""
    burden_max = float(frame["all_minutes_last_7d"].max())
    spline = base_model.spline_basis_expression(burden_max)
    return (
        f"{event_col} ~ {exposure_prefix}_between + {exposure_prefix}_within * higher_history "
        f"+ {spline} * higher_history "
        "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
        f"{controls}"
    )


def fit_duty_model(
    base_model: Any,
    frame: pd.DataFrame,
    event_col: str,
    denominator: str,
    controls: str = "",
    exposure_prefix: str = "duty",
):  # pragma: no cover - exercised end-to-end; numerical internals belong to statsmodels
    """Fit one clustered Poisson model for recent country-duty context."""
    if frame[event_col].sum() <= 0:
        raise ValueError(f"No events available for {event_col}")
    if frame["recent_national_duty"].nunique() < 2:
        raise ValueError("Recent national duty has no exposure variation")
    offset = None
    if denominator == "observed_minutes":
        offset = frame["log_minutes_played"]
    elif denominator == "fixed_90":
        offset = pd.Series(np.log(90.0), index=frame.index)
    elif denominator != "per_match":
        raise ValueError(f"Unknown denominator mode: {denominator}")
    kwargs: dict[str, object] = {}
    if offset is not None:
        kwargs["offset"] = offset
    model = smf.glm(
        formula=_duty_formula(
            base_model, frame, event_col, controls, exposure_prefix=exposure_prefix
        ),
        data=frame,
        family=sm.families.Poisson(),
        **kwargs,
    )
    return model.fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID]})


def ratio_interval(
    params: pd.Series,
    covariance: pd.DataFrame,
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Return an exponentiated linear contrast and normal-theory interval."""
    missing = sorted(set(weights) - set(params.index))
    if missing:
        raise KeyError(f"Model is missing contrast parameters: {missing}")
    vector = pd.Series(0.0, index=params.index)
    for parameter, weight in weights.items():
        vector.loc[parameter] = float(weight)
    log_estimate = float(vector @ params)
    variance = float(vector.to_numpy() @ covariance.to_numpy() @ vector.to_numpy())
    standard_error = float(np.sqrt(max(variance, 0.0)))
    statistic = log_estimate / standard_error if standard_error > 0 else np.nan
    p_value = float(2 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
    return {
        "estimate": float(np.exp(log_estimate)),
        "ci_low": float(np.exp(log_estimate - 1.96 * standard_error)),
        "ci_high": float(np.exp(log_estimate + 1.96 * standard_error)),
        "log_estimate": log_estimate,
        "standard_error": standard_error,
        "test_statistic": statistic,
        "df": 1.0,
        "p_value": p_value,
    }


def duty_contrasts(result: Any, exposure_prefix: str = "duty") -> pd.DataFrame:
    """Extract within-player rates and the prior-history interaction."""
    params = result.params
    covariance = result.cov_params()
    within = f"{exposure_prefix}_within"
    interaction = f"{exposure_prefix}_within:higher_history"
    between = f"{exposure_prefix}_between"
    definitions = [
        (
            "intermediate_history_recent_duty_within_player",
            "intermediate prior-injury history",
            {within: 1.0},
        ),
        (
            "higher_history_recent_duty_within_player",
            "higher prior-injury history",
            {within: 1.0, interaction: 1.0},
        ),
        (
            "higher_vs_intermediate_recent_duty_interaction",
            "higher versus intermediate prior-injury history",
            {interaction: 1.0},
        ),
        (
            "between_player_recent_duty_frequency",
            "between-player selection component",
            {between: 1.0},
        ),
    ]
    rows = []
    for contrast_id, history_stratum, weights in definitions:
        row = ratio_interval(params, covariance, weights)
        row.update(
            {
                "contrast_id": contrast_id,
                "history_stratum": history_stratum,
                "effect_measure": "incidence_rate_ratio",
            }
        )
        rows.append(row)
    within_names = [within, interaction]
    indices = [params.index.get_loc(name) for name in within_names]
    beta = params.iloc[indices].to_numpy(dtype=float)
    cov = covariance.iloc[indices, indices].to_numpy(dtype=float)
    statistic = float(beta @ np.linalg.pinv(cov) @ beta)
    rows.append(
        {
            "contrast_id": "joint_within_player_duty_terms",
            "history_stratum": "joint",
            "effect_measure": "wald_chi_square",
            "estimate": statistic,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "log_estimate": np.nan,
            "standard_error": np.nan,
            "test_statistic": statistic,
            "df": 2.0,
            "p_value": float(chi2.sf(statistic, 2)),
        }
    )
    return pd.DataFrame(rows)


def _specifications() -> list[dict[str, object]]:
    """Return the complete, finite post-hoc robustness family."""
    specs: list[dict[str, object]] = [
        {
            "specification_id": f"window_{days}d_primary_observed",
            "specification_family": "country_duty_window",
            "window_days": days,
            "event_col": PRIMARY_EVENT,
            "denominator": "observed_minutes",
            "controls": "",
            "controls_label": "club_load_spline_and_calendar",
            "exposure_prefix": "duty",
            "exposure_metric": "any_senior_competitive_country_appearance",
        }
        for days in WINDOW_DAYS
    ]
    specs.extend(
        [
            {
                "specification_id": "outcome_same_day_7d",
                "specification_family": "outcome_timing",
                "window_days": 7,
                "event_col": "injury_event_matchproxy_same_day",
                "denominator": "observed_minutes",
                "controls": "",
                "controls_label": "club_load_spline_and_calendar",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
            },
            {
                "specification_id": "outcome_lag1_7d",
                "specification_family": "outcome_timing",
                "window_days": 7,
                "event_col": "injury_event_matchproxy_lag1",
                "denominator": "observed_minutes",
                "controls": "",
                "controls_label": "club_load_spline_and_calendar",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
            },
            {
                "specification_id": "outcome_description_specific_7d",
                "specification_family": "outcome_classification",
                "window_days": 7,
                "event_col": "injury_event_matchproxy_specific",
                "denominator": "observed_minutes",
                "controls": "",
                "controls_label": "club_load_spline_and_calendar",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
            },
            {
                "specification_id": "denominator_fixed90_7d",
                "specification_family": "exposure_denominator",
                "window_days": 7,
                "event_col": PRIMARY_EVENT,
                "denominator": "fixed_90",
                "controls": "",
                "controls_label": "club_load_spline_and_calendar",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
            },
            {
                "specification_id": "denominator_per_match_7d",
                "specification_family": "exposure_denominator",
                "window_days": 7,
                "event_col": PRIMARY_EVENT,
                "denominator": "per_match",
                "controls": "",
                "controls_label": "club_load_spline_and_calendar",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
            },
            {
                "specification_id": "age_position_clubseason_7d",
                "specification_family": "measured_confounding",
                "window_days": 7,
                "event_col": PRIMARY_EVENT,
                "denominator": "observed_minutes",
                "controls": " + age_years + C(position_group) + C(club_season)",
                "controls_label": "age_position_clubseason",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
            },
            {
                "specification_id": "age_position_clubseason_fixed90_7d",
                "specification_family": "measured_confounding_and_denominator",
                "window_days": 7,
                "event_col": PRIMARY_EVENT,
                "denominator": "fixed_90",
                "controls": " + age_years + C(position_group) + C(club_season)",
                "controls_label": "age_position_clubseason",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
            },
            {
                "specification_id": "continuous_history_controls_7d",
                "specification_family": "measured_confounding",
                "window_days": 7,
                "event_col": PRIMARY_EVENT,
                "denominator": "observed_minutes",
                "controls": (
                    " + age_years + C(position_group) + C(club_season)"
                    " + log_prior_minutes_played + prior_n_spells"
                    " + log_prior_injuries_per_10000min"
                    " + log_prior_max_spell_duration_days"
                ),
                "controls_label": "age_position_clubseason_continuous_prior_history",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
            },
            {
                "specification_id": "continuous_national_minutes_7d",
                "specification_family": "country_duty_exposure_definition",
                "window_days": 7,
                "event_col": PRIMARY_EVENT,
                "denominator": "observed_minutes",
                "controls": "",
                "controls_label": "club_load_spline_and_calendar",
                "exposure_prefix": "national_match_equivalents",
                "exposure_metric": "per_90_senior_competitive_country_minutes",
            },
            {
                "specification_id": "continuous_national_match_count_7d",
                "specification_family": "country_duty_exposure_definition",
                "window_days": 7,
                "event_col": PRIMARY_EVENT,
                "denominator": "observed_minutes",
                "controls": "",
                "controls_label": "club_load_spline_and_calendar",
                "exposure_prefix": "national_match_count",
                "exposure_metric": "per_senior_competitive_country_appearance",
            },
            {
                "specification_id": "international_break_calendar_control_7d",
                "specification_family": "calendar_context",
                "window_days": 7,
                "event_col": PRIMARY_EVENT,
                "denominator": "observed_minutes",
                "controls": " + prior7_overlaps_international_break + covid_disrupted_date",
                "controls_label": "club_load_spline_calendar_international_break_covid",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
            },
        ]
    )
    for history_group_col, label in [
        ("fragility_count_only", "count_only"),
        ("fragility_frequency_only", "frequency_only"),
        ("fragility_severity_only", "severity_only"),
        ("fragility_prespecified_abs", "prespecified_absolute"),
    ]:
        specs.append(
            {
                "specification_id": f"history_{label}_7d",
                "specification_family": "prior_history_definition",
                "window_days": 7,
                "event_col": PRIMARY_EVENT,
                "denominator": "observed_minutes",
                "controls": "",
                "controls_label": "club_load_spline_and_calendar",
                "exposure_prefix": "duty",
                "exposure_metric": "any_senior_competitive_country_appearance",
                "history_group_col": history_group_col,
            }
        )
    for spec in specs:
        spec.setdefault("history_group_col", PRIMARY_HISTORY)
    return specs


def add_exploratory_multiplicity(results: pd.DataFrame) -> pd.DataFrame:
    """Adjust the complete post-hoc family globally and by contrast."""
    _required(results, ["contrast_id", "p_value"], "country-duty results")
    out = results.copy()
    p_values = pd.to_numeric(out["p_value"], errors="coerce")
    valid = p_values.notna() & np.isfinite(p_values)
    out["p_holm_exploratory_family"] = np.nan
    out["p_bh_exploratory_family"] = np.nan
    if valid.any():
        out.loc[valid, "p_holm_exploratory_family"] = multipletests(
            p_values.loc[valid], method="holm"
        )[1]
        out.loc[valid, "p_bh_exploratory_family"] = multipletests(
            p_values.loc[valid], method="fdr_bh"
        )[1]
    out["p_holm_across_specifications"] = np.nan
    out["p_bh_across_specifications"] = np.nan
    for _, indices in out.loc[valid].groupby("contrast_id", sort=False).groups.items():
        values = p_values.loc[indices].to_numpy(dtype=float)
        out.loc[indices, "p_holm_across_specifications"] = multipletests(
            values, method="holm"
        )[1]
        out.loc[indices, "p_bh_across_specifications"] = multipletests(
            values, method="fdr_bh"
        )[1]
    out["reject_holm_exploratory_0_05"] = out["p_holm_exploratory_family"].lt(0.05)
    out["reject_bh_exploratory_0_05"] = out["p_bh_exploratory_family"].lt(0.05)
    return out


def run_country_duty_family(
    base_model: Any,
    match_panel: pd.DataFrame,
    exposure_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # pragma: no cover - full data GLMs are integration work
    """Fit the declared post-hoc source-specific robustness family."""
    result_tables: list[pd.DataFrame] = []
    support_rows: list[dict[str, object]] = []
    for spec in _specifications():
        event_col = str(spec["event_col"])
        frame = prepare_duty_model_frame(
            base_model,
            match_panel,
            exposure_features,
            NATIONAL_INCREMENT_SCOPES["senior_competitive"],
            int(spec["window_days"]),
            event_col,
            history_group_col=str(spec["history_group_col"]),
        )
        if "continuous_history" in str(spec["specification_id"]):
            frame = base_model.add_prior_history_control_columns(frame)
        result = fit_duty_model(
            base_model,
            frame,
            event_col,
            str(spec["denominator"]),
            str(spec["controls"]),
            exposure_prefix=str(spec["exposure_prefix"]),
        )
        contrasts = duty_contrasts(result, exposure_prefix=str(spec["exposure_prefix"]))
        for key, value in spec.items():
            if key != "controls":
                contrasts[key] = value
        contrasts["analysis_role"] = "post_hoc_hypothesis_generating"
        contrasts["n_match_rows"] = len(frame)
        contrasts["n_players"] = frame[PLAYER_ID].nunique()
        contrasts["n_events"] = int(frame[event_col].sum())
        result_tables.append(contrasts)
        for raw_group, group in frame.groupby("model_group", sort=False):
            exposed = group["recent_national_duty"].eq(1)
            support_rows.append(
                {
                    "specification_id": spec["specification_id"],
                    "window_days": spec["window_days"],
                    "event_col": event_col,
                    "history_stratum": PUBLIC_HISTORY_LABELS.get(raw_group, raw_group),
                    "n_match_rows": len(group),
                    "n_exposed_rows": int(exposed.sum()),
                    "n_events": int(group[event_col].sum()),
                    "n_events_on_exposed_rows": int(group.loc[exposed, event_col].sum()),
                }
            )
    results = add_exploratory_multiplicity(pd.concat(result_tables, ignore_index=True))
    return results, pd.DataFrame(support_rows)


def run_total_burden_scope_family(
    comparison_module: Any,
    base_model: Any,
    match_panel: pd.DataFrame,
    exposure_features: pd.DataFrame,
    gate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # pragma: no cover - integration GLMs
    """Repeat the unchanged primary model across all v4 country scopes."""
    selected_tables: list[pd.DataFrame] = []
    test_tables: list[pd.DataFrame] = []
    audit_tables: list[pd.DataFrame] = []
    label = comparison_module.coverage_decision(gate)
    for scope in comparison_module.AUDIT_SCOPES:
        panel = comparison_module.prepare_scope_model_panel(
            match_panel, exposure_features, scope
        )
        selected, burden, recovery, audit = comparison_module.run_scope_model(
            base_model, panel, scope, label
        )
        selected_tables.append(selected)
        test_tables.extend([burden, recovery])
        audit_tables.append(audit)
    tests = comparison_module.add_comparison_multiplicity(
        pd.concat(test_tables, ignore_index=True)
    )
    return (
        pd.concat(selected_tables, ignore_index=True),
        tests,
        pd.concat(audit_tables, ignore_index=True),
    )


def conclusion_audit(
    gate: pd.DataFrame,
    total_tests: pd.DataFrame,
    selected: pd.DataFrame,
    duty_results: pd.DataFrame,
    selection_diagnostics: pd.DataFrame,
    travel_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Translate the audit tables into bounded claim decisions."""
    _required(gate, ["primary_v4_exposure_allowed"], "coverage gate")
    _required(
        total_tests,
        ["contrast_id", "exposure_scope", "p_value", "reject_holm_v4_0_05"],
        "total-burden tests",
    )
    _required(
        selected,
        [
            "exposure_scope",
            "fragility_group",
            "all_minutes_last_7d",
            "pred_events_per_10000_min",
        ],
        "selected predictions",
    )
    _required(
        duty_results,
        ["specification_id", "contrast_id", "estimate", "ci_low", "ci_high", "p_value", "p_holm_exploratory_family"],
        "country-duty results",
    )
    direct_scopes = ["frozen_club_all", "frozen_club_plus_senior_national"]
    interactions = total_tests.loc[
        total_tests["contrast_id"].eq("global_spline_by_history_interaction")
        & total_tests["exposure_scope"].isin(direct_scopes)
    ]
    p_by_scope = interactions.set_index("exposure_scope")["p_value"]
    predictions = selected.loc[selected["exposure_scope"].isin(direct_scopes)].copy()
    pivot = predictions.pivot_table(
        index=["fragility_group", "all_minutes_last_7d"],
        columns="exposure_scope",
        values="pred_events_per_10000_min",
        aggfunc="first",
    ).dropna()
    relative_shift = (
        (
            pivot["frozen_club_plus_senior_national"]
            / pivot["frozen_club_all"]
            - 1
        )
        .abs()
        .mul(100)
    )
    primary_duty = duty_results.loc[
        duty_results["specification_id"].eq("window_7d_primary_observed")
        & duty_results["contrast_id"].eq("higher_history_recent_duty_within_player")
    ].iloc[0]
    interaction_duty = duty_results.loc[
        duty_results["specification_id"].eq("window_7d_primary_observed")
        & duty_results["contrast_id"].eq("higher_vs_intermediate_recent_duty_interaction")
    ].iloc[0]
    continuous_duty = duty_results.loc[
        duty_results["specification_id"].eq("continuous_national_minutes_7d")
        & duty_results["contrast_id"].eq("higher_history_recent_duty_within_player")
    ].iloc[0]
    break_control = duty_results.loc[
        duty_results["specification_id"].eq("international_break_calendar_control_7d")
        & duty_results["contrast_id"].eq("higher_history_recent_duty_within_player")
    ].iloc[0]
    alternative_history = duty_results.loc[
        duty_results["specification_family"].eq("prior_history_definition")
        & duty_results["contrast_id"].eq("higher_history_recent_duty_within_player")
    ]
    ipw_usable = (
        selection_diagnostics["ipw_usable"].fillna(False).astype(bool).all()
        if "ipw_usable" in selection_diagnostics and not selection_diagnostics.empty
        else False
    )
    travel_rows = (
        travel_audit.loc[
            travel_audit["metric"].eq("timeline_rows_with_travel_distance"), "value"
        ]
        if {"metric", "value"}.issubset(travel_audit.columns)
        else pd.Series(dtype=float)
    )
    travel_pairs = int(float(travel_rows.iloc[0])) if not travel_rows.empty else 0
    rows = [
        {
            "audit_question": "Does adding senior competitive country minutes change the primary total-burden conclusion?",
            "decision": "no_material_change",
            "evidence": (
                f"Global interaction p={p_by_scope.get('frozen_club_all', np.nan):.3f} before and "
                f"p={p_by_scope.get('frozen_club_plus_senior_national', np.nan):.3f} after; "
                f"maximum absolute selected prediction shift={relative_shift.max():.1f}%."
            ),
            "permitted_interpretation": "Observed country minutes did not overturn the frozen seven-day exposure-response result.",
        },
        {
            "audit_question": "Can the expanded exposure replace the primary frozen exposure?",
            "decision": "sensitivity_only"
            if not gate["primary_v4_exposure_allowed"].all()
            else "primary_allowed",
            "evidence": "The prespecified official-schedule coverage gate did not pass; independent secondary-source coverage is reported separately."
            if not gate["primary_v4_exposure_allowed"].all()
            else "Every prespecified coverage condition passed.",
            "permitted_interpretation": "The extension addresses observed exposure misclassification but cannot establish complete national exposure.",
        },
        {
            "audit_question": "Does recent senior competitive country duty identify a source-specific signal?",
            "decision": "hypothesis_generating",
            "evidence": (
                f"Higher-history within-player IRR={primary_duty['estimate']:.2f} "
                f"({primary_duty['ci_low']:.2f}-{primary_duty['ci_high']:.2f}), "
                f"raw p={primary_duty['p_value']:.4f}, global Holm p={primary_duty['p_holm_exploratory_family']:.3f}; "
                f"per 90 country minutes IRR={continuous_duty['estimate']:.2f} "
                f"({continuous_duty['ci_low']:.2f}-{continuous_duty['ci_high']:.2f}); "
                f"break-adjusted IRR={break_control['estimate']:.2f} "
                f"({break_control['ci_low']:.2f}-{break_control['ci_high']:.2f}); "
                f"alternative-history IRR range={alternative_history['estimate'].min():.2f}-"
                f"{alternative_history['estimate'].max():.2f}; "
                f"history interaction IRR={interaction_duty['estimate']:.2f}, p={interaction_duty['p_value']:.3f}."
            ),
            "permitted_interpretation": "This post-hoc association warrants prospective testing; it is not a causal effect of national-team minutes.",
        },
        {
            "audit_question": "Did observed selection weighting resolve selection into appearances?",
            "decision": "not_supported" if not ipw_usable else "supported",
            "evidence": "The prespecified balance/overlap/weight gate failed."
            if not ipw_usable
            else "The prespecified balance/overlap/weight gate passed.",
            "permitted_interpretation": "Selection into minutes remains an important limitation.",
        },
        {
            "audit_question": "Can geographic travel be modelled from verified venue data?",
            "decision": "not_supported" if travel_pairs == 0 else "supported",
            "evidence": f"Verified consecutive venue pairs with distance={travel_pairs}.",
            "permitted_interpretation": "No travel effect is estimated when verified coordinates are absent.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:  # pragma: no cover
    """Run the complete v4 consequence audit and write machine-readable tables."""
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed" / "public_data_v4"
    base_model = load_module(
        root / "src" / "18_match_proxy_poisson_splines_perminute.py",
        "v4_primary_model_for_scientific_audit",
    )
    comparison = load_module(
        root / "src" / "28_public_data_v4_model_comparison.py",
        "v4_scope_comparison_for_scientific_audit",
    )
    match_panel = pd.read_csv(
        root / "data" / "processed" / "player_match_panel_all_comp.csv",
        low_memory=False,
    )
    features = pd.read_csv(
        processed / "match_exposure_scope_features.csv", low_memory=False
    )
    gate = pd.read_csv(processed / "exposure_coverage_audit.csv", low_memory=False)

    changes = exposure_change_audit(match_panel, features)
    changes.to_csv(processed / "v4_exposure_change_audit.csv", index=False)
    recovery_changes = recovery_change_audit(match_panel, features)
    recovery_changes.to_csv(processed / "v4_recovery_change_audit.csv", index=False)

    record_audit = pd.read_csv(
        processed / "international_performance_record_audit.csv", low_memory=False
    )
    retained_national = pd.read_csv(
        processed / "international_appearances.csv", low_memory=False
    )
    national_record_quality_audit(record_audit, retained_national).to_csv(
        processed / "v4_national_record_quality_audit.csv", index=False
    )

    selected, total_tests, total_audit = run_total_burden_scope_family(
        comparison, base_model, match_panel, features, gate
    )
    selected.to_csv(processed / "v4_all_scope_selected_predictions.csv", index=False)
    total_tests.to_csv(processed / "v4_all_scope_model_comparison.csv", index=False)
    total_audit.to_csv(processed / "v4_all_scope_model_input_audit.csv", index=False)

    enriched = base_model.add_player_and_club_metadata(
        match_panel, root / "external_data" / "transfermarkt"
    )
    enriched = base_model.add_calendar_sensitivity_flags(enriched)
    enriched = base_model.add_alternative_fragility_labels(enriched)
    duty_results, duty_support = run_country_duty_family(
        base_model, enriched, features
    )
    duty_results.to_csv(processed / "v4_country_duty_between_within.csv", index=False)
    duty_support.to_csv(processed / "v4_country_duty_support.csv", index=False)

    selection = pd.read_csv(
        processed / "selection_weight_diagnostics.csv", low_memory=False
    )
    travel = pd.read_csv(
        processed / "geographic_travel_coverage_audit.csv", low_memory=False
    )
    conclusions = conclusion_audit(
        gate, total_tests, selected, duty_results, selection, travel
    )
    conclusions.to_csv(processed / "v4_conclusion_audit.csv", index=False)
    print(conclusions.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
