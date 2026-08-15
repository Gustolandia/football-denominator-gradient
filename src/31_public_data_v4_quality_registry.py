#!/usr/bin/env python
"""Create machine-readable v4 quality checks and publication claim tiers.

The registry is deliberately stricter than a narrative audit. Structural
failures block the expanded-data sensitivity analysis, the prespecified
official-schedule gate controls whether it may replace the frozen exposure,
and sparse model cells must remain explicit non-estimates. Proportions use
Wilson 95% confidence intervals; incidence and model intervals are audited in
their source tables rather than recomputed here. Every substantive result
reported in the manuscript is assigned its lowest defensible tier so that
abstract and main-display prominence can be checked independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from v4_statistics import percent_with_interval


EXPECTED_STATUS_CATEGORIES = {
    "no_recent_senior_record",
    "explicitly_not_in_squad",
    "recorded_unavailable",
    "squad_only",
    "played",
}


def _required(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def _proportion_row(
    domain: str,
    check_id: str,
    numerator: int,
    denominator: int,
    *,
    threshold: float | None = None,
    threshold_direction: str = "at_least",
    severity: str = "informational",
    interpretation: str,
) -> dict[str, object]:
    estimate, low, high = percent_with_interval(numerator, denominator)
    if threshold is None:
        passes: object = pd.NA
    elif threshold_direction == "at_least":
        passes = bool(estimate >= threshold)
    elif threshold_direction == "at_most":
        passes = bool(estimate <= threshold)
    else:
        raise ValueError(f"Unknown threshold direction: {threshold_direction}")
    return {
        "domain": domain,
        "check_id": check_id,
        "numerator": numerator,
        "denominator": denominator,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "interval_method": "wilson_95",
        "threshold": threshold,
        "threshold_direction": threshold_direction if threshold is not None else "not_applicable",
        "passes": passes,
        "severity": severity,
        "interpretation": interpretation,
    }


def _count_row(
    domain: str,
    check_id: str,
    value: int | float,
    *,
    threshold: float,
    threshold_direction: str,
    severity: str,
    interpretation: str,
) -> dict[str, object]:
    if threshold_direction == "at_least":
        passes = bool(value >= threshold)
    elif threshold_direction == "at_most":
        passes = bool(value <= threshold)
    elif threshold_direction == "equals":
        passes = bool(value == threshold)
    else:
        raise ValueError(f"Unknown threshold direction: {threshold_direction}")
    return {
        "domain": domain,
        "check_id": check_id,
        "numerator": pd.NA,
        "denominator": pd.NA,
        "estimate": value,
        "ci_low": pd.NA,
        "ci_high": pd.NA,
        "interval_method": "not_applicable_exact_count",
        "threshold": threshold,
        "threshold_direction": threshold_direction,
        "passes": passes,
        "severity": severity,
        "interpretation": interpretation,
    }


def independent_schedule_summary(validation: pd.DataFrame) -> pd.DataFrame:
    """Summarise unique-match and score agreement with the CC0 schedule."""
    _required(
        validation,
        [
            "game_id",
            "verified",
            "score_values_available",
            "score_agreement",
            "independent_shootout_match",
            "score_internal_consistency",
        ],
        "independent schedule validation",
    )
    rows = []
    for _game_id, group in validation.groupby("game_id", dropna=False):
        verified = _bool_series(group["verified"]).any()
        comparable = _bool_series(group["score_values_available"])
        agreements = _bool_series(group.loc[comparable, "score_agreement"])
        shootout = _bool_series(group["independent_shootout_match"]).any()
        rows.append(
            {
                "verified": bool(verified),
                "score_comparable": bool(comparable.any()),
                "score_agreement": bool(agreements.all()) if comparable.any() else pd.NA,
                "shootout_match": bool(shootout),
            }
        )
    games = pd.DataFrame(rows)
    comparable = _bool_series(games["score_comparable"])
    non_shootout_comparable = comparable & ~_bool_series(games["shootout_match"])
    internal_conflicts = (
        validation["score_internal_consistency"].eq(False).fillna(False)
    )
    accepted_conflicts = internal_conflicts & validation[
        "score_values_available"
    ].astype("boolean").fillna(False)
    return pd.DataFrame(
        [
            _count_row(
                "cross_source_schedule",
                "accepted_internally_conflicting_score_games",
                int(validation.loc[accepted_conflicts, "game_id"].nunique()),
                threshold=0,
                threshold_direction="equals",
                severity="critical",
                interpretation=f"No player-level score pair is accepted when opposite team orientations conflict; {int(validation.loc[internal_conflicts, 'game_id'].nunique())} detected conflicts remain listed as non-comparable.",
            ),
            _proportion_row(
                "cross_source_schedule",
                "independent_unique_match_coverage",
                int(_bool_series(games["verified"]).sum()),
                len(games),
                interpretation="Unique senior matches confirmed by exact date and unordered team pair.",
            ),
            _proportion_row(
                "cross_source_schedule",
                "independent_score_agreement",
                int(_bool_series(games.loc[comparable, "score_agreement"]).sum()),
                int(comparable.sum()),
                interpretation="Exact home-away score agreement where both sources recorded scores and orientation.",
            ),
            _proportion_row(
                "cross_source_schedule",
                "independent_non_shootout_score_agreement",
                int(
                    _bool_series(
                        games.loc[non_shootout_comparable, "score_agreement"]
                    ).sum()
                ),
                int(non_shootout_comparable.sum()),
                interpretation="Exact home-away score agreement after excluding independently identified penalty-shootout fixtures, whose score conventions differ across sources.",
            ),
        ]
    )


def worldcup_player_validation_summary(validation: pd.DataFrame) -> pd.DataFrame:
    """Summarise the independent CC0 World Cup player-level validation sample."""
    _required(
        validation,
        [
            "source_match_found",
            "source_player_found",
            "starter_agreement",
            "minutes_within_5",
        ],
        "World Cup player validation",
    )
    match_found = _bool_series(validation["source_match_found"])
    player_found = _bool_series(validation["source_player_found"])
    starter_comparable = validation["starter_agreement"].notna() & player_found
    minute_comparable = validation["minutes_within_5"].notna() & player_found
    return pd.DataFrame(
        [
            _proportion_row(
                "cross_source_player",
                "worldcup_match_coverage",
                int(match_found.sum()),
                len(validation),
                interpretation="Cohort World Cup played records whose date and national-team pair occur in the independent lineup source.",
            ),
            _proportion_row(
                "cross_source_player",
                "worldcup_played_player_agreement",
                int(player_found.sum()),
                int(match_found.sum()),
                interpretation="Transfermarkt played records found under the same normalized player name in an independently compiled lineup.",
            ),
            _proportion_row(
                "cross_source_player",
                "worldcup_starter_agreement",
                int(_bool_series(validation.loc[starter_comparable, "starter_agreement"]).sum()),
                int(starter_comparable.sum()),
                interpretation="Starter/substitute classification agreement among exact player matches.",
            ),
            _proportion_row(
                "cross_source_player",
                "worldcup_minutes_within_5",
                int(_bool_series(validation.loc[minute_comparable, "minutes_within_5"]).sum()),
                int(minute_comparable.sum()),
                interpretation="Observed minutes within five minutes of the lineup/substitution-clock approximation.",
            ),
        ]
    )


def _metric_value(frame: pd.DataFrame, metric: str) -> float:
    _required(frame, ["metric", "value"], "metric table")
    values = pd.to_numeric(frame.loc[frame["metric"].eq(metric), "value"], errors="coerce")
    if len(values) != 1 or pd.isna(values.iloc[0]):
        raise ValueError(f"Expected one finite value for metric {metric!r}")
    return float(values.iloc[0])


def build_quality_registry(
    appearances: pd.DataFrame,
    acquisition_log: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    gate: pd.DataFrame,
    baseline_parity: pd.DataFrame,
    independent_validation: pd.DataFrame,
    worldcup_validation: pd.DataFrame,
    status_rates: pd.DataFrame,
    status_models: pd.DataFrame,
    status_support: pd.DataFrame,
) -> pd.DataFrame:
    """Build one harmonised registry of structural and statistical checks."""
    _required(
        appearances,
        ["tm_player_id", "game_id", "participation_state", "minutes_played", "team_name", "opponent_team_name"],
        "international appearances",
    )
    _required(acquisition_log, ["status"], "acquisition log")
    _required(duplicate_audit, ["resolution"], "duplicate audit")
    _required(
        gate,
        [
            "metric",
            "value",
            "numerator",
            "denominator",
            "ci_low",
            "ci_high",
            "primary_v4_exposure_allowed",
        ],
        "coverage gate",
    )
    _required(status_rates, ["national_status", "events_per_1000_match_hours", "ci_low", "ci_high"], "status rates")
    _required(status_models, ["fit_status", "estimate", "ci_low", "ci_high", "p_value", "p_holm_status_family", "support_adequate"], "status models")
    _required(status_support, ["support_adequate", "national_status", "specification_id"], "status support")

    rows: list[dict[str, object]] = []
    errors = int(acquisition_log["status"].eq("error").sum())
    rows.append(
        _count_row(
            "acquisition",
            "player_endpoint_acquisition_errors",
            errors,
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="Failed player histories are never interpreted as zero exposure.",
        )
    )
    invalid_retained = ~appearances["participation_state"].eq("played") | pd.to_numeric(
        appearances["minutes_played"], errors="coerce"
    ).isna()
    rows.append(
        _count_row(
            "harmonization",
            "retained_rows_violating_played_known_minutes_rule",
            int(invalid_retained.sum()),
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="Every exposure row must represent observed play with known minutes.",
        )
    )
    rows.append(
        _count_row(
            "harmonization",
            "duplicate_retained_player_game_rows",
            int(appearances.duplicated(["tm_player_id", "game_id"]).sum()),
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="The retained exposure chronology has one row per player-game.",
        )
    )
    unresolved = duplicate_audit["resolution"].astype(str).str.contains(
        "unresolved", case=False, na=False
    )
    rows.append(
        _count_row(
            "harmonization",
            "unresolved_source_collisions",
            int(unresolved.sum()),
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="No conflicting duplicate contributes exposure.",
        )
    )
    complete_names = appearances["team_name"].notna() & appearances[
        "opponent_team_name"
    ].notna()
    rows.append(
        _proportion_row(
            "harmonization",
            "retained_team_pair_name_completeness",
            int(complete_names.sum()),
            len(appearances),
            severity="informational",
            interpretation="Both national-team names are available for descriptive reconciliation; IDs, dates, and minutes remain the exposure keys.",
        )
    )

    for metric, check_id, threshold, severity in (
        ("verified_match_coverage_percent", "official_unique_match_coverage", 95.0, "scope_gate"),
        ("independent_match_coverage_percent", "independent_senior_competitive_match_coverage", None, "informational"),
        ("primary_senior_competitive_nonmissing_minutes_percent", "senior_competitive_minutes_completeness", 95.0, "critical"),
        ("all_played_national_records_nonmissing_minutes_percent", "all_played_minutes_completeness", None, "informational"),
    ):
        selected = gate.loc[gate["metric"].eq(metric)]
        if len(selected) != 1:
            raise ValueError(f"Expected one coverage-gate row for {metric}")
        row = selected.iloc[0]
        rows.append(
            {
                "domain": "coverage",
                "check_id": check_id,
                "numerator": row["numerator"],
                "denominator": row["denominator"],
                "estimate": float(row["value"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "interval_method": row.get("interval_method", "wilson_95"),
                "threshold": threshold,
                "threshold_direction": "at_least" if threshold is not None else "not_applicable",
                "passes": bool(float(row["value"]) >= threshold) if threshold is not None else pd.NA,
                "severity": severity,
                "interpretation": "Official verification controls primary replacement." if metric.startswith("verified") else "Coverage or completeness of observed national exposure.",
            }
        )
    rows.extend(independent_schedule_summary(independent_validation).to_dict("records"))
    rows.extend(worldcup_player_validation_summary(worldcup_validation).to_dict("records"))

    rows.append(
        _count_row(
            "baseline_parity",
            "frozen_burden_mismatch_rows",
            int(_metric_value(baseline_parity, "frozen_comparator_burden_mismatch_rows")),
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="Adding public sources does not mutate the frozen club comparator.",
        )
    )
    statuses_valid = set(status_rates["national_status"]).issubset(
        EXPECTED_STATUS_CATEGORIES
    )
    rows.append(
        _count_row(
            "status_definition",
            "unexpected_status_categories",
            0 if statuses_valid else len(set(status_rates["national_status"]) - EXPECTED_STATUS_CATEGORIES),
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="Status labels distinguish played, squad-only matchday records, unavailability, explicit non-selection, and no record; none is called training.",
        )
    )
    rate = pd.to_numeric(status_rates["events_per_1000_match_hours"], errors="coerce")
    rate_low = pd.to_numeric(status_rates["ci_low"], errors="coerce")
    rate_high = pd.to_numeric(status_rates["ci_high"], errors="coerce")
    invalid_rates = rate.isna() | rate_low.isna() | rate_high.isna() | (rate_low > rate) | (rate > rate_high)
    rows.append(
        _count_row(
            "uncertainty",
            "status_rate_rows_with_invalid_or_missing_ci",
            int(invalid_rates.sum()),
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="Every displayed status-specific incidence rate has an ordered exact Poisson 95% interval.",
        )
    )
    fitted = status_models["fit_status"].eq("fitted")
    fitted_numeric = status_models.loc[fitted, ["estimate", "ci_low", "ci_high", "p_value", "p_holm_status_family"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_fitted = (
        fitted_numeric.isna().any(axis=1)
        | (fitted_numeric["ci_low"] > fitted_numeric["estimate"])
        | (fitted_numeric["estimate"] > fitted_numeric["ci_high"])
        | ~fitted_numeric["p_value"].between(0, 1)
        | ~fitted_numeric["p_holm_status_family"].between(0, 1)
    )
    rows.append(
        _count_row(
            "uncertainty",
            "fitted_status_contrasts_with_invalid_inference",
            int(invalid_fitted.sum()),
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="Every fitted contrast has an estimate, ordered 95% interval, raw p-value, and family-adjusted p-value.",
        )
    )
    sparse = status_models["fit_status"].eq("not_fitted_sparse_support")
    sparse_numeric = status_models.loc[sparse, ["estimate", "ci_low", "ci_high", "p_value"]].apply(
        pd.to_numeric, errors="coerce"
    )
    rows.append(
        _count_row(
            "sparse_support",
            "sparse_status_rows_containing_pseudo_estimates",
            int(sparse_numeric.notna().any(axis=1).sum()),
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="Sparse status cells remain explicit non-estimates rather than separation-driven IRRs.",
        )
    )
    support_keys = status_support.groupby(["national_status", "specification_id"])[
        "support_adequate"
    ].apply(lambda values: _bool_series(values).all())
    model_keys = status_models.groupby(["national_status", "specification_id"])[
        "support_adequate"
    ].apply(lambda values: _bool_series(values).all())
    inconsistent_support = int((support_keys.sort_index() != model_keys.sort_index()).sum())
    rows.append(
        _count_row(
            "sparse_support",
            "model_support_flag_mismatches",
            inconsistent_support,
            threshold=0,
            threshold_direction="equals",
            severity="critical",
            interpretation="Pre-fit support decisions agree with every model output row.",
        )
    )
    registry = pd.DataFrame(rows)
    critical = registry["severity"].eq("critical")
    critical_passes = _bool_series(registry.loc[critical, "passes"]).all()
    registry = pd.concat(
        [
            registry,
            pd.DataFrame(
                [
                    _count_row(
                        "readiness",
                        "v4_sensitivity_analysis_ready",
                        int(critical_passes),
                        threshold=1,
                        threshold_direction="equals",
                        severity="critical",
                        interpretation="All structural and inference checks required for sensitivity use pass; the separate official gate still controls primary replacement.",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    return registry


def _one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def _validate_result_tier_registry(registry: pd.DataFrame) -> None:
    """Reject duplicate claims or tier/visibility combinations that over-promote results."""
    if registry["claim_id"].duplicated().any():
        raise ValueError("Result-tier registry contains duplicate claim IDs")
    if not registry["abstract_rule_passes"].all():
        raise ValueError("Result-tier registry violates an abstract visibility rule")
    if not registry["main_display_rule_passes"].all():
        raise ValueError("Result-tier registry violates a main-display visibility rule")


def build_result_tier_registry(
    clinical_rates: pd.DataFrame,
    outcome_sensitivity: pd.DataFrame,
    binary_type: pd.DataFrame,
    frequency_type: pd.DataFrame,
    recency_attenuation: pd.DataFrame,
    type_history_family: pd.DataFrame,
    denominator: pd.DataFrame,
    same_day_denominator: pd.DataFrame,
    effect_family: pd.DataFrame,
    shape_sensitivity: pd.DataFrame,
    selection_bands: pd.DataFrame,
    status_models: pd.DataFrame,
    status_rates: pd.DataFrame,
    conclusion_audit: pd.DataFrame,
    quality_registry: pd.DataFrame,
    recurrent_decomposition: pd.DataFrame,
    duration_risk: pd.DataFrame,
    case_crossover: pd.DataFrame,
    temporal_stability: pd.DataFrame,
    reporting_completeness: pd.DataFrame,
    reporting_ipw: pd.DataFrame,
    selection_bootstrap: pd.DataFrame,
    lineup_interaction: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each claim its lowest defensible novelty and visibility tier.

    Tier 1 contains central, strongly original findings with implications beyond
    one narrow specification. Tier 2 contains original but narrower, partly
    anticipated, or sensitivity-bound findings. Tier 3 contains only surprising
    nulls or direct contradictions, Tier 4 replications, and Tier 5 other null
    or uninformative results. Ambiguous claims receive the lower tier.
    """
    overall = _one(
        clinical_rates,
        rate_scope="same_day_plus_lag1",
        group_kind="overall",
        group="overall",
    )
    severe_outcome = _one(outcome_sensitivity, model="reported_absence_ge28d")
    muscle_outcome = _one(outcome_sensitivity, model="muscle_tendon_only")
    binary = _one(binary_type, restriction="all eligible rows")
    frequency = _one(frequency_type, restriction="all eligible rows")
    attenuation = _one(
        recency_attenuation,
        restriction="all eligible rows",
        contrast_id="muscle_tendon_high_step",
    )
    attenuation_family = _one(
        type_history_family,
        test_id="attenuation__all eligible rows__muscle_tendon_high_step",
    )
    observed = _one(denominator, model="denominator_observed_minutes_poisson")
    fixed = _one(denominator, model="denominator_fixed_90_poisson")
    event_minutes = _one(
        same_day_denominator,
        history_stratum="all_modelled",
        prior_load_band="all_bands",
        row_type="same_day_proxy_event",
    )
    nonevent_minutes = _one(
        same_day_denominator,
        history_stratum="all_modelled",
        prior_load_band="all_bands",
        row_type="no_proxy_event",
    )
    n_tests = int(pd.to_numeric(effect_family["n_tests"], errors="raise").sum())
    n_holm = int(
        (
            pd.to_numeric(effect_family["minimum_holm_p"], errors="coerce") < 0.05
        ).sum()
    )
    regular_shape = shape_sensitivity.loc[shape_sensitivity["history_stratum"].eq("regular")]
    fragile_shape = shape_sensitivity.loc[shape_sensitivity["history_stratum"].eq("fragile")]
    regular_peaks = int(_bool_series(regular_shape["max_in_15_45_min_band"]).sum())
    fragile_peaks = int(_bool_series(fragile_shape["max_in_15_45_min_band"]).sum())
    regular_band = _one(selection_bands, history_stratum="regular", band="15-45 min peak band")
    fragile_band = _one(selection_bands, history_stratum="fragile", band="15-45 min peak band")
    regular_trough = _one(selection_bands, history_stratum="regular", band="90-95 min trough band")
    fragile_trough = _one(selection_bands, history_stratum="fragile", band="90-95 min trough band")
    primary_played = _one(
        status_models,
        specification_id="window_7d_observed",
        national_status="played",
        contrast_id="played_higher_vs_no_involvement",
    )
    fixed_played = _one(
        status_models,
        specification_id="fixed90_7d",
        national_status="played",
        contrast_id="played_higher_vs_no_involvement",
    )
    played_interaction = _one(
        status_models,
        specification_id="window_7d_observed",
        national_status="played",
        contrast_id="played_history_interaction",
    )
    squad_rate = _one(
        status_rates,
        history_stratum="fragile",
        national_status="squad_only",
    )
    schedule_quality = _one(
        quality_registry,
        check_id="independent_senior_competitive_match_coverage",
    )
    player_quality = _one(
        quality_registry, check_id="worldcup_played_player_agreement"
    )
    between_player = _one(
        recurrent_decomposition,
        component="within_between_poisson",
        model="within_between_history_state",
        term="player_higher_history_match_share",
    )
    within_player = _one(
        recurrent_decomposition,
        component="within_between_poisson",
        model="within_between_history_state",
        term="within_player_higher_history_deviation",
    )
    pre_transition = _one(
        recurrent_decomposition,
        component="switcher_transition_state",
        transition_state="switcher_pre_higher_history",
    )
    post_transition = _one(
        recurrent_decomposition,
        component="switcher_transition_state",
        transition_state="switcher_post_higher_history",
    )
    duration_regular_short = _one(
        duration_risk,
        group="regular",
        prior_injury_duration_bucket="<1 week",
    )
    duration_regular_long = _one(
        duration_risk,
        group="regular",
        prior_injury_duration_bucket="2 months to 1 year",
    )
    duration_fragile_short = _one(
        duration_risk,
        group="fragile",
        prior_injury_duration_bucket="<1 week",
    )
    duration_fragile_long = _one(
        duration_risk,
        group="fragile",
        prior_injury_duration_bucket="2 months to 1 year",
    )
    case_crossover_difference = _one(
        case_crossover,
        model="previous_7day_minutes",
        contrast="difference_in_per_90_minutes",
    )
    temporal_rows = temporal_stability.sort_values("season_start_min")
    reporting_overall = _one(reporting_completeness, context="overall")
    reporting_same_day = _one(
        reporting_completeness, context="timing", proxy_timing="same_day"
    )
    reporting_lag1 = _one(
        reporting_completeness, context="timing", proxy_timing="lag1"
    )
    if len(reporting_ipw) != 1:
        raise ValueError(
            f"Expected one reporting-IPW diagnostic row, found {len(reporting_ipw)}"
        )
    reporting_ipw_row = reporting_ipw.iloc[0]
    bootstrap_regular = _one(selection_bootstrap, history_stratum="regular")
    bootstrap_fragile = _one(selection_bootstrap, history_stratum="fragile")
    lineup_pooled = _one(lineup_interaction, model="pooled_history_adjusted")
    total_burden = conclusion_audit.loc[
        conclusion_audit["audit_question"].str.startswith(
            "Does adding senior competitive"
        )
    ].iloc[0]
    regular_rate = _one(
        clinical_rates,
        rate_scope="same_day_plus_lag1",
        group_kind="fragility_group",
        group="regular",
    )
    fragile_rate = _one(
        clinical_rates,
        rate_scope="same_day_plus_lag1",
        group_kind="fragility_group",
        group="fragile",
    )

    rows = [
        {
            "claim_id": "public_proxy_retains_bounded_relative_information",
            "tier": 2,
            "tier_label": "medium_new_result",
            "claim": "Public reports undercount absolute match injury incidence but retain an auditable prior-history ordering in better-captured outcome subsets.",
            "evidence": f"Overall proxy incidence {float(overall['events_per_1000_match_hours']):.2f} ({float(overall['events_per_1000_match_hours_ci_low']):.2f}-{float(overall['events_per_1000_match_hours_ci_high']):.2f}) per 1,000 match-hours, or {float(overall['events_per_1000_match_hours']) / 36 * 100:.1f}% and {float(overall['events_per_1000_match_hours']) / 23.8 * 100:.1f}% of the two clinical benchmarks. At the zero-recent-minute anchor, the established higher/intermediate-history ordering remained {float(severe_outcome['rr_0']):.2f} ({float(severe_outcome['rr_0_ci_low']):.2f}-{float(severe_outcome['rr_0_ci_high']):.2f}) for reported absences of at least 28 days and {float(muscle_outcome['rr_0']):.2f} ({float(muscle_outcome['rr_0_ci_low']):.2f}-{float(muscle_outcome['rr_0_ci_high']):.2f}) for muscle/tendon descriptions.",
            "evidence_file": "clinical_match_hour_rates.csv; matchproxy_sensitivity_summary.csv",
            "abstract_eligible": True,
            "abstract_recommended": True,
            "main_display_recommended": True,
            "maximum_visibility": "abstract_results_discussion_conclusion",
            "required_caveat": "Proxy incidence is not clinical incidence and benchmark differences are not a formal validation study.",
        },
        {
            "claim_id": "matched_recency_attenuates_apparent_type_threshold",
            "tier": 2,
            "tier_label": "medium_new_result",
            "claim": "Matched type-specific recency attenuates the apparent muscle/tendon high-frequency step; continuous muscle/tendon and type-discordant comparator slopes are similar after the same adjustment.",
            "evidence": f"The formal muscle/tendon high-frequency step changed from {float(attenuation['unadjusted_irr']):.2f} ({float(attenuation['unadjusted_ci_low']):.2f}-{float(attenuation['unadjusted_ci_high']):.2f}) to {float(attenuation['recency_adjusted_irr']):.2f} ({float(attenuation['recency_adjusted_ci_low']):.2f}-{float(attenuation['recency_adjusted_ci_high']):.2f}); the direct adjusted/unadjusted ratio was {float(attenuation['adjusted_over_unadjusted_ratio']):.2f} ({float(attenuation['adjusted_over_unadjusted_ci_low']):.2f}-{float(attenuation['adjusted_over_unadjusted_ci_high']):.2f}; Holm p={float(attenuation_family['p_holm_type_history_family']):.3g}). The recency-adjusted continuous type ratio was {float(frequency['recency_adjusted_direct_ratio_muscle_over_joint_bone']):.2f} ({float(frequency['recency_adjusted_direct_ratio_ci_low']):.2f}-{float(frequency['recency_adjusted_direct_ratio_ci_high']):.2f}).",
            "evidence_file": "matchproxy_type_history_recency_attenuation.csv; matchproxy_type_history_multiplicity_family.csv; matchproxy_negative_control_mutually_exclusive_type_frequency.csv",
            "abstract_eligible": True,
            "abstract_recommended": True,
            "main_display_recommended": True,
            "maximum_visibility": "abstract_main_results_discussion_conclusion",
            "required_caveat": "Binary and continuous specifications disagree. The binary result may reflect a tail-concentrated association or a dichotomisation artefact; the similar continuous comparator slope means shared reporting or player-profile bias remains plausible.",
        },
        {
            "claim_id": "reported_event_duration_linkage",
            "tier": 2,
            "tier_label": "medium_new_result",
            "claim": "Same-day-report rows contain less observed playing time, making per-minute estimates and model dispersion sensitive to the exposure denominator.",
            "evidence": f"Same-day event rows averaged {float(event_minutes['mean_minutes']):.1f} minutes versus {float(nonevent_minutes['mean_minutes']):.1f} without an event; Poisson dispersion changed from {float(observed['dispersion']):.2f} with observed minutes to {float(fixed['dispersion']):.2f} with fixed 90-minute exposure.",
            "evidence_file": "matchproxy_same_day_denominator_audit.csv; matchproxy_denominator_sensitivity_summary.csv",
            "abstract_eligible": True,
            "abstract_recommended": True,
            "main_display_recommended": True,
            "maximum_visibility": "abstract_main_results_discussion_conclusion",
            "required_caveat": "The duration difference does not show why an appearance was shorter. Observed minutes may stop at an event, while fixed 90 minutes adds unplayed or post-event time; the comparison brackets denominator sensitivity rather than identifying the true event time.",
        },
        {
            "claim_id": "selection_shapes_early_spline_peak",
            "tier": 2,
            "tier_label": "medium_new_result",
            "claim": "The fitted early exposure-response peak is specification-sensitive and concentrated in substitute and recent-return appearances.",
            "evidence": f"An early peak survived {regular_peaks}/{len(regular_shape)} intermediate-history and {fragile_peaks}/{len(fragile_shape)} higher-history spline specifications. Substitute-list records were {float(regular_band['pct_substitute_list']):.1f}% versus {float(regular_trough['pct_substitute_list']):.1f}% in intermediate-history rows and {float(fragile_band['pct_substitute_list']):.1f}% versus {float(fragile_trough['pct_substitute_list']):.1f}% in higher-history rows; recent-return shares were {float(regular_band['pct_returned_from_recorded_injury_within_14d']):.1f}% versus {float(regular_trough['pct_returned_from_recorded_injury_within_14d']):.1f}% and {float(fragile_band['pct_returned_from_recorded_injury_within_14d']):.1f}% versus {float(fragile_trough['pct_returned_from_recorded_injury_within_14d']):.1f}%. In player bootstraps, the early band was the global maximum in {float(bootstrap_regular['early_band_global_max_percent']):.1f}% and {float(bootstrap_fragile['early_band_global_max_percent']):.1f}%; the pooled lineup-role interaction p was {float(lineup_pooled['p_value']):.3f}.",
            "evidence_file": "matchproxy_spline_shape_sensitivity.csv; matchproxy_selection_band_audit.csv",
            "abstract_eligible": True,
            "abstract_recommended": True,
            "main_display_recommended": True,
            "maximum_visibility": "abstract_main_results_discussion_conclusion",
            "required_caveat": "This demonstrates sensitivity to risk-set composition, not the exact magnitude of selection bias.",
        },
        {
            "claim_id": "expanded_source_validation",
            "tier": 2,
            "tier_label": "medium_new_result",
            "claim": "Independent CC0 sources verify a substantial part of the national-match schedule and a tournament subset of played player records.",
            "evidence": f"Unique-match coverage was {float(schedule_quality['estimate']):.1f}% ({float(schedule_quality['ci_low']):.1f}-{float(schedule_quality['ci_high']):.1f}); exact normalized-player agreement in matched World Cup records was {float(player_quality['estimate']):.1f}% ({float(player_quality['ci_low']):.1f}-{float(player_quality['ci_high']):.1f}).",
            "evidence_file": "v4_data_quality_registry.csv",
            "abstract_eligible": True,
            "abstract_recommended": False,
            "main_display_recommended": True,
            "maximum_visibility": "main_figure_methods_results_discussion",
            "required_caveat": "The player-level validation sample covers World Cups 2018 and 2022 only; official full-calendar coverage remains below the primary-use gate.",
        },
        {
            "claim_id": "recent_senior_national_match_context",
            "tier": 5,
            "tier_label": "null_or_uninformative_result",
            "claim": "Recent senior national-match participation did not provide a multiplicity-adjusted source-specific signal or a between-history-stratum interaction.",
            "evidence": f"Seven-day observed-minute IRR {float(primary_played['estimate']):.2f} ({float(primary_played['ci_low']):.2f}-{float(primary_played['ci_high']):.2f}; Holm p={float(primary_played['p_holm_status_family']):.3f}); fixed-90 IRR {float(fixed_played['estimate']):.2f} ({float(fixed_played['ci_low']):.2f}-{float(fixed_played['ci_high']):.2f}; Holm p={float(fixed_played['p_holm_status_family']):.3f}); history interaction {float(played_interaction['estimate']):.2f} ({float(played_interaction['ci_low']):.2f}-{float(played_interaction['ci_high']):.2f}).",
            "evidence_file": "v4_national_status_models.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results_and_one_sentence_discussion_or_conclusion",
            "required_caveat": "Post-hoc status sensitivity; match selection, travel, reporting, and return-to-club context remain inseparable.",
        },
        {
            "claim_id": "no_history_effect_modification",
            "tier": 5,
            "tier_label": "null_or_uninformative_result",
            "claim": "No multiplicity-adjusted exposure-response contrast supports effect modification by prior injury history.",
            "evidence": f"Zero Holm rejections across {n_tests} tests; family minima produced {n_holm} family-level minima below 0.05.",
            "evidence_file": "matchproxy_effect_modification_multiplicity_family_summary.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results_and_one_sentence_discussion_or_conclusion",
            "required_caveat": "Absence of rejection is not proof that all clinically relevant heterogeneity is zero.",
        },
        {
            "claim_id": "national_minutes_do_not_change_total_burden_conclusion",
            "tier": 5,
            "tier_label": "null_or_uninformative_result",
            "claim": "Adding observed senior competitive national-team minutes does not change the total-burden conclusion.",
            "evidence": str(total_burden["evidence"]),
            "evidence_file": "v4_conclusion_audit.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results_and_one_sentence_discussion_or_conclusion",
            "required_caveat": "The expanded chronology remains incomplete and sensitivity-only.",
        },
        {
            "claim_id": "prior_history_incidence_gradient",
            "tier": 4,
            "tier_label": "reproduced_result",
            "claim": "Higher prior injury history identifies higher subsequent proxy incidence, consistent with established prior-injury susceptibility.",
            "evidence": f"Intermediate-history rate {float(regular_rate['events_per_1000_match_hours']):.1f} ({float(regular_rate['events_per_1000_match_hours_ci_low']):.1f}-{float(regular_rate['events_per_1000_match_hours_ci_high']):.1f}) and higher-history rate {float(fragile_rate['events_per_1000_match_hours']):.1f} ({float(fragile_rate['events_per_1000_match_hours_ci_low']):.1f}-{float(fragile_rate['events_per_1000_match_hours_ci_high']):.1f}) per 1,000 match-hours.",
            "evidence_file": "clinical_match_hour_rates.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results_and_one_sentence_discussion_or_conclusion",
            "required_caveat": "Corroborative, not novel.",
        },
        {
            "claim_id": "squad_only_status_inference",
            "tier": 5,
            "tier_label": "null_or_uninformative_result",
            "claim": "Seven-day squad-only national matchday involvement is too sparse for adjusted inference.",
            "evidence": f"Higher-history descriptive rate {float(squad_rate['events_per_1000_match_hours']):.1f} ({float(squad_rate['ci_low']):.1f}-{float(squad_rate['ci_high']):.1f}) per 1,000 match-hours from {int(squad_rate['n_events'])} events; the prespecified model support rule failed.",
            "evidence_file": "v4_national_status_rates.csv; v4_national_status_model_support.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results_and_one_sentence_discussion_or_conclusion",
            "required_caveat": "Squad-only matchday records do not measure national-team training.",
        },
        {
            "claim_id": "between_player_susceptibility_corroboration",
            "tier": 4,
            "tier_label": "reproduced_result",
            "claim": "The between-player history component is consistent with established player-level susceptibility; the within-player reversal is not interpretable as an effect.",
            "evidence": f"Between-player IRR {float(between_player['estimate']):.2f} ({float(between_player['ci_low']):.2f}-{float(between_player['ci_high']):.2f}); within-player IRR {float(within_player['estimate']):.2f} ({float(within_player['ci_low']):.2f}-{float(within_player['ci_high']):.2f}). Switcher incidence was {float(pre_transition['events_per_1000_match_hours']):.1f} before versus {float(post_transition['events_per_1000_match_hours']):.1f} after transition.",
            "evidence_file": "matchproxy_recurrent_event_decomposition.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results_and_one_sentence_discussion_or_conclusion",
            "required_caveat": "The injury that changes history status contributes to the pre-transition period, creating index-event bias.",
        },
        {
            "claim_id": "prior_absence_duration_descriptive_pattern",
            "tier": 5,
            "tier_label": "descriptive_uninterpretable_result",
            "claim": "Longer prior reported absences preceded lower crude proxy incidence in both history strata, but the contrast cannot identify a duration or rehabilitation effect.",
            "evidence": f"Two-month-to-one-year versus under-one-week rates were {float(duration_regular_long['events_per_1000_match_hours']):.1f} ({float(duration_regular_long['events_per_1000_match_hours_ci_low']):.1f}-{float(duration_regular_long['events_per_1000_match_hours_ci_high']):.1f}) versus {float(duration_regular_short['events_per_1000_match_hours']):.1f} ({float(duration_regular_short['events_per_1000_match_hours_ci_low']):.1f}-{float(duration_regular_short['events_per_1000_match_hours_ci_high']):.1f}) in intermediate-history rows and {float(duration_fragile_long['events_per_1000_match_hours']):.1f} ({float(duration_fragile_long['events_per_1000_match_hours_ci_low']):.1f}-{float(duration_fragile_long['events_per_1000_match_hours_ci_high']):.1f}) versus {float(duration_fragile_short['events_per_1000_match_hours']):.1f} ({float(duration_fragile_short['events_per_1000_match_hours_ci_low']):.1f}-{float(duration_fragile_short['events_per_1000_match_hours_ci_high']):.1f}) in higher-history rows.",
            "evidence_file": "prior_injury_duration_next_risk_canonical.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results",
            "required_caveat": "Reported absence duration mixes injury type, severity, rehabilitation, return-to-sport selection, and reporting practice.",
        },
        {
            "claim_id": "within_player_case_crossover_no_history_difference",
            "tier": 5,
            "tier_label": "null_or_uninformative_result",
            "claim": "Within-player case-crossover slopes did not differ detectably between history strata.",
            "evidence": f"The higher/intermediate per-90-minute slope ratio was {float(case_crossover_difference['estimate']):.2f} ({float(case_crossover_difference['ci_low']):.2f}-{float(case_crossover_difference['ci_high']):.2f}; Holm p={float(case_crossover_difference['p_holm_extension_family']):.3f}).",
            "evidence_file": "matchproxy_extension_within_player_case_crossover.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results",
            "required_caveat": "The two positive stratum-specific slopes do not establish a difference between strata.",
        },
        {
            "claim_id": "temporal_block_instability",
            "tier": 5,
            "tier_label": "null_or_uninformative_result",
            "claim": "Temporal-block estimates were imprecise and did not provide prospective validation of history-specific exposure response.",
            "evidence": "The 180-minute higher/intermediate ratios were " + ", ".join(
                f"{float(row.rr_180):.2f} ({float(row.rr_180_ci_low):.2f}-{float(row.rr_180_ci_high):.2f})"
                for row in temporal_rows.itertuples(index=False)
            ) + ".",
            "evidence_file": "matchproxy_temporal_stability_summary.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results",
            "required_caveat": "These are internal time blocks with carried-forward labels, not an independent prospective validation cohort.",
        },
        {
            "claim_id": "type_reporting_completeness_gate",
            "tier": 5,
            "tier_label": "measurement_limitation",
            "claim": "Public report text was incompletely type-classifiable, and inverse-probability correction failed prespecified stability gates.",
            "evidence": f"Type was classifiable for {float(reporting_overall['type_classifiable_percent']):.1f}% ({float(reporting_overall['type_classifiable_ci_low']):.1f}-{float(reporting_overall['type_classifiable_ci_high']):.1f}) of modelled events, including {float(reporting_same_day['type_classifiable_percent']):.1f}% of same-day and {float(reporting_lag1['type_classifiable_percent']):.1f}% of lag-1 events. Minimum fitted probability was {float(reporting_ipw_row['minimum_predicted_type_classifiable_probability']):.3f}; maximum weight was {float(reporting_ipw_row['maximum_type_reporting_ipw']):.1f}.",
            "evidence_file": "matchproxy_extension_reporting_completeness_context.csv; matchproxy_extension_reporting_type_ipw_diagnostics.csv",
            "abstract_eligible": False,
            "abstract_recommended": False,
            "main_display_recommended": False,
            "maximum_visibility": "one_sentence_results_and_limitations",
            "required_caveat": "The failed positivity and weight gates prohibit presenting the weighted model as correction.",
        },
    ]
    registry = pd.DataFrame(rows)
    registry["main_display_eligible"] = registry["tier"].le(3)
    registry["abstract_rule_passes"] = (
        registry["abstract_eligible"].eq(registry["tier"].le(3))
        & (~registry["abstract_recommended"] | registry["abstract_eligible"])
    )
    registry["main_display_rule_passes"] = (
        (~registry["main_display_recommended"] | registry["main_display_eligible"])
        & ~(
            registry["tier"].ge(4)
            & registry["main_display_recommended"]
        )
    )
    _validate_result_tier_registry(registry)
    return registry.sort_values(["tier", "claim_id"]).reset_index(drop=True)


def main() -> None:  # pragma: no cover - file orchestration
    """Write v4 quality, source-agreement, and result-tier registries."""
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed" / "public_data_v4"
    results = root / "data" / "processed" / "results"
    quality = build_quality_registry(
        pd.read_csv(processed / "international_appearances.csv", low_memory=False),
        pd.read_csv(processed / "national_acquisition_log.csv", low_memory=False),
        pd.read_csv(processed / "national_duplicate_audit.csv", low_memory=False),
        pd.read_csv(processed / "exposure_coverage_audit.csv", low_memory=False),
        pd.read_csv(processed / "baseline_parity_report.csv", low_memory=False),
        pd.read_csv(processed / "independent_schedule_validation.csv", low_memory=False),
        pd.read_csv(processed / "openfootball_worldcup_player_validation.csv", low_memory=False),
        pd.read_csv(processed / "v4_national_status_rates.csv", low_memory=False),
        pd.read_csv(processed / "v4_national_status_models.csv", low_memory=False),
        pd.read_csv(processed / "v4_national_status_model_support.csv", low_memory=False),
    )
    quality.to_csv(processed / "v4_data_quality_registry.csv", index=False)
    tiers = build_result_tier_registry(
        pd.read_csv(results / "clinical_match_hour_rates.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_sensitivity_summary.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_negative_control_mutually_exclusive_type_binary.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_negative_control_mutually_exclusive_type_frequency.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_type_history_recency_attenuation.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_type_history_multiplicity_family.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_denominator_sensitivity_summary.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_same_day_denominator_audit.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_effect_modification_multiplicity_family_summary.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_spline_shape_sensitivity.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_selection_band_audit.csv", low_memory=False),
        pd.read_csv(processed / "v4_national_status_models.csv", low_memory=False),
        pd.read_csv(processed / "v4_national_status_rates.csv", low_memory=False),
        pd.read_csv(processed / "v4_conclusion_audit.csv", low_memory=False),
        quality,
        pd.read_csv(results / "matchproxy_recurrent_event_decomposition.csv", low_memory=False),
        pd.read_csv(results / "prior_injury_duration_next_risk_canonical.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_extension_within_player_case_crossover.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_temporal_stability_summary.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_extension_reporting_completeness_context.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_extension_reporting_type_ipw_diagnostics.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_extension_curve_feature_bootstrap_summary.csv", low_memory=False),
        pd.read_csv(results / "matchproxy_extension_lineup_spline_interaction.csv", low_memory=False),
    )
    tiers.to_csv(processed / "v4_result_tier_registry.csv", index=False)
    print(quality.to_string(index=False))
    print(tiers[["tier", "claim_id", "abstract_recommended"]].to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
