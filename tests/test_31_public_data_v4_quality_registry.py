"""Tests for the harmonised v4 quality and result-tier registries."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _module():
    path = Path(__file__).parents[1] / "src" / "31_public_data_v4_quality_registry.py"
    spec = importlib.util.spec_from_file_location("v4_quality_registry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _quality_inputs():
    appearances = pd.DataFrame(
        [
            {"tm_player_id": 1, "game_id": "g1", "participation_state": "played", "minutes_played": 90, "team_name": "A", "opponent_team_name": "B"},
            {"tm_player_id": 2, "game_id": "g1", "participation_state": "played", "minutes_played": 45, "team_name": "B", "opponent_team_name": "A"},
        ]
    )
    acquisition = pd.DataFrame({"status": ["cached", "downloaded"]})
    duplicates = pd.DataFrame({"resolution": ["single_source"]})
    gate = pd.DataFrame(
        [
            {"metric": "verified_match_coverage_percent", "value": 0.0, "numerator": 0, "denominator": 2, "ci_low": 0.0, "ci_high": 10.0, "interval_method": "wilson_95", "primary_v4_exposure_allowed": False},
            {"metric": "independent_match_coverage_percent", "value": 75.0, "numerator": 3, "denominator": 4, "ci_low": 60.0, "ci_high": 85.0, "interval_method": "wilson_95", "primary_v4_exposure_allowed": False},
            {"metric": "primary_senior_competitive_nonmissing_minutes_percent", "value": 100.0, "numerator": 2, "denominator": 2, "ci_low": 90.0, "ci_high": 100.0, "interval_method": "wilson_95", "primary_v4_exposure_allowed": False},
            {"metric": "all_played_national_records_nonmissing_minutes_percent", "value": 100.0, "numerator": 2, "denominator": 2, "ci_low": 90.0, "ci_high": 100.0, "interval_method": "wilson_95", "primary_v4_exposure_allowed": False},
        ]
    )
    baseline = pd.DataFrame(
        {"metric": ["frozen_comparator_burden_mismatch_rows"], "value": [0]}
    )
    independent = pd.DataFrame(
        [
            {"game_id": "g1", "verified": True, "score_values_available": True, "score_agreement": True, "independent_shootout_match": False, "score_internal_consistency": True},
            {"game_id": "g1", "verified": True, "score_values_available": True, "score_agreement": True, "independent_shootout_match": False, "score_internal_consistency": True},
            {"game_id": "g2", "verified": False, "score_values_available": False, "score_agreement": pd.NA, "independent_shootout_match": False, "score_internal_consistency": pd.NA},
        ]
    )
    worldcup = pd.DataFrame(
        [
            {"source_match_found": True, "source_player_found": True, "starter_agreement": True, "minutes_within_5": True},
            {"source_match_found": True, "source_player_found": False, "starter_agreement": pd.NA, "minutes_within_5": pd.NA},
        ]
    )
    rates = pd.DataFrame(
        [
            {"history_stratum": "fragile", "national_status": status, "n_events": 1, "events_per_1000_match_hours": 10.0, "ci_low": 1.0, "ci_high": 20.0}
            for status in sorted(_module().EXPECTED_STATUS_CATEGORIES)
        ]
    )
    models = pd.DataFrame(
        [
            {"national_status": "played", "specification_id": "fit", "fit_status": "fitted", "estimate": 1.2, "ci_low": 0.8, "ci_high": 1.8, "p_value": 0.4, "p_holm_status_family": 0.8, "support_adequate": True},
            {"national_status": "squad_only", "specification_id": "sparse", "fit_status": "not_fitted_sparse_support", "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "p_holm_status_family": np.nan, "support_adequate": False},
        ]
    )
    support = pd.DataFrame(
        [
            {"national_status": "played", "specification_id": "fit", "support_adequate": True},
            {"national_status": "squad_only", "specification_id": "sparse", "support_adequate": False},
        ]
    )
    return appearances, acquisition, duplicates, gate, baseline, independent, worldcup, rates, models, support


def test_source_summaries_and_quality_registry():
    module = _module()
    inputs = _quality_inputs()
    schedule = module.independent_schedule_summary(inputs[5])
    assert schedule.loc[schedule.check_id.eq("independent_unique_match_coverage"), "estimate"].iloc[0] == 50.0
    assert schedule.loc[schedule.check_id.eq("independent_score_agreement"), "estimate"].iloc[0] == 100.0
    assert schedule.loc[schedule.check_id.eq("independent_non_shootout_score_agreement"), "estimate"].iloc[0] == 100.0
    assert schedule.loc[schedule.check_id.eq("accepted_internally_conflicting_score_games"), "estimate"].iloc[0] == 0
    worldcup = module.worldcup_player_validation_summary(inputs[6])
    assert worldcup.loc[worldcup.check_id.eq("worldcup_played_player_agreement"), "estimate"].iloc[0] == 50.0
    registry = module.build_quality_registry(*inputs)
    assert registry.loc[registry.check_id.eq("v4_sensitivity_analysis_ready"), "passes"].iloc[0]
    assert not registry.loc[registry.check_id.eq("official_unique_match_coverage"), "passes"].iloc[0]
    coverage = registry.loc[
        registry.check_id.eq("independent_senior_competitive_match_coverage")
    ].iloc[0]
    assert (coverage["numerator"], coverage["denominator"]) == (3, 4)
    assert registry.loc[registry.check_id.eq("fitted_status_contrasts_with_invalid_inference"), "estimate"].iloc[0] == 0


def test_quality_helpers_reject_bad_inputs():
    module = _module()
    assert module._proportion_row(
        "x",
        "minimum",
        2,
        2,
        threshold=90,
        threshold_direction="at_least",
        interpretation="x",
    )["passes"]
    assert module._proportion_row(
        "x",
        "maximum",
        0,
        2,
        threshold=10,
        threshold_direction="at_most",
        interpretation="x",
    )["passes"]
    assert module._count_row(
        "x",
        "minimum_count",
        2,
        threshold=1,
        threshold_direction="at_least",
        severity="critical",
        interpretation="x",
    )["passes"]
    assert module._count_row(
        "x",
        "maximum_count",
        0,
        threshold=1,
        threshold_direction="at_most",
        severity="critical",
        interpretation="x",
    )["passes"]
    with pytest.raises(ValueError, match="Unknown threshold direction"):
        module._proportion_row("x", "x", 1, 2, threshold=1, threshold_direction="sideways", interpretation="x")
    with pytest.raises(ValueError, match="Unknown threshold direction"):
        module._count_row("x", "x", 1, threshold=1, threshold_direction="sideways", severity="critical", interpretation="x")
    with pytest.raises(KeyError, match="independent schedule validation"):
        module.independent_schedule_summary(pd.DataFrame())
    with pytest.raises(KeyError, match="World Cup player validation"):
        module.worldcup_player_validation_summary(pd.DataFrame())
    with pytest.raises(ValueError, match="Expected one finite value"):
        module._metric_value(pd.DataFrame({"metric": ["a"], "value": [1]}), "missing")
    with pytest.raises(KeyError, match="international appearances"):
        module.build_quality_registry(pd.DataFrame(), *(_quality_inputs()[1:]))
    with pytest.raises(ValueError, match="Expected one coverage-gate row"):
        bad = list(_quality_inputs())
        bad[3] = bad[3].iloc[:-1]
        module.build_quality_registry(*bad)
    with pytest.raises(ValueError, match="Expected one row"):
        module._one(pd.DataFrame({"a": [1, 1]}), a=1)


def test_result_tiers_enforce_abstract_visibility():
    module = _module()
    clinical = pd.DataFrame(
        [
            {"rate_scope": "same_day_plus_lag1", "group_kind": "overall", "group": "overall", "events_per_1000_match_hours": 17.5, "events_per_1000_match_hours_ci_low": 16.7, "events_per_1000_match_hours_ci_high": 18.4},
            {"rate_scope": "same_day_plus_lag1", "group_kind": "fragility_group", "group": "regular", "events_per_1000_match_hours": 17.2, "events_per_1000_match_hours_ci_low": 16.0, "events_per_1000_match_hours_ci_high": 18.5},
            {"rate_scope": "same_day_plus_lag1", "group_kind": "fragility_group", "group": "fragile", "events_per_1000_match_hours": 24.6, "events_per_1000_match_hours_ci_low": 22.5, "events_per_1000_match_hours_ci_high": 26.8},
        ]
    )
    outcome_sensitivity = pd.DataFrame(
        [
            {"model": "reported_absence_ge28d", "rr_0": 1.91, "rr_0_ci_low": 1.40, "rr_0_ci_high": 2.60},
            {"model": "muscle_tendon_only", "rr_0": 1.61, "rr_0_ci_low": 1.20, "rr_0_ci_high": 2.20},
        ]
    )
    binary = pd.DataFrame(
        [{"restriction": "all eligible rows", "direct_ratio_muscle_over_joint_bone": 2.35, "direct_ratio_ci_low": 1.37, "direct_ratio_ci_high": 4.01, "recency_adjusted_direct_ratio_muscle_over_joint_bone": 1.35, "recency_adjusted_direct_ratio_ci_low": 0.79, "recency_adjusted_direct_ratio_ci_high": 2.31}]
    )
    frequency = pd.DataFrame(
        [{"restriction": "all eligible rows", "recency_adjusted_direct_ratio_muscle_over_joint_bone": 0.98, "recency_adjusted_direct_ratio_ci_low": 0.94, "recency_adjusted_direct_ratio_ci_high": 1.03}]
    )
    attenuation = pd.DataFrame(
        [
            {
                "restriction": "all eligible rows",
                "contrast_id": "muscle_tendon_high_step",
                "unadjusted_irr": 1.59,
                "unadjusted_ci_low": 1.23,
                "unadjusted_ci_high": 2.06,
                "recency_adjusted_irr": 1.15,
                "recency_adjusted_ci_low": 0.91,
                "recency_adjusted_ci_high": 1.46,
                "adjusted_over_unadjusted_ratio": 0.72,
                "adjusted_over_unadjusted_ci_low": 0.66,
                "adjusted_over_unadjusted_ci_high": 0.80,
            }
        ]
    )
    type_history_family = pd.DataFrame(
        [
            {
                "test_id": "attenuation__all eligible rows__muscle_tendon_high_step",
                "p_holm_type_history_family": 1e-8,
            }
        ]
    )
    denominator = pd.DataFrame(
        [
            {"model": "denominator_observed_minutes_poisson", "dispersion": 2.16},
            {"model": "denominator_fixed_90_poisson", "dispersion": 0.98},
        ]
    )
    minutes = pd.DataFrame(
        [
            {"history_stratum": "all_modelled", "prior_load_band": "all_bands", "row_type": "same_day_proxy_event", "mean_minutes": 51.6},
            {"history_stratum": "all_modelled", "prior_load_band": "all_bands", "row_type": "no_proxy_event", "mean_minutes": 71.3},
        ]
    )
    effect = pd.DataFrame(
        [{"n_tests": 100, "minimum_holm_p": 0.2}, {"n_tests": 54, "minimum_holm_p": 0.3}]
    )
    shape = pd.DataFrame(
        [
            {"history_stratum": "regular", "max_in_15_45_min_band": True},
            {"history_stratum": "regular", "max_in_15_45_min_band": False},
            {"history_stratum": "fragile", "max_in_15_45_min_band": True},
            {"history_stratum": "fragile", "max_in_15_45_min_band": True},
        ]
    )
    bands = pd.DataFrame(
        [
            {"history_stratum": "regular", "band": "15-45 min peak band", "pct_substitute_list": 32.7, "pct_returned_from_recorded_injury_within_14d": 11.9},
            {"history_stratum": "regular", "band": "90-95 min trough band", "pct_substitute_list": 7.0, "pct_returned_from_recorded_injury_within_14d": 5.0},
            {"history_stratum": "fragile", "band": "15-45 min peak band", "pct_substitute_list": 40.4, "pct_returned_from_recorded_injury_within_14d": 17.9},
            {"history_stratum": "fragile", "band": "90-95 min trough band", "pct_substitute_list": 9.9, "pct_returned_from_recorded_injury_within_14d": 8.2},
        ]
    )
    status_models = pd.DataFrame(
        [
            {"specification_id": "window_7d_observed", "national_status": "played", "contrast_id": "played_higher_vs_no_involvement", "estimate": 1.7, "ci_low": 1.1, "ci_high": 2.6, "p_holm_status_family": 0.37},
            {"specification_id": "fixed90_7d", "national_status": "played", "contrast_id": "played_higher_vs_no_involvement", "estimate": 1.98, "ci_low": 1.31, "ci_high": 3.0, "p_holm_status_family": 0.04},
            {"specification_id": "window_7d_observed", "national_status": "played", "contrast_id": "played_history_interaction", "estimate": 1.74, "ci_low": 0.97, "ci_high": 3.13, "p_holm_status_family": 1.0},
        ]
    )
    status_rates = pd.DataFrame(
        [{"history_stratum": "fragile", "national_status": "squad_only", "n_events": 3, "events_per_1000_match_hours": 32.9, "ci_low": 6.8, "ci_high": 96.1}]
    )
    conclusion = pd.DataFrame(
        [{"audit_question": "Does adding senior competitive country minutes change the primary total-burden conclusion?", "evidence": "p=0.527 before and p=0.632 after."}]
    )
    quality = pd.DataFrame(
        [
            {"check_id": "independent_senior_competitive_match_coverage", "estimate": 67.2, "ci_low": 65.5, "ci_high": 68.8},
            {"check_id": "worldcup_played_player_agreement", "estimate": 95.0, "ci_low": 93.0, "ci_high": 97.0},
        ]
    )
    recurrent = pd.DataFrame(
        [
            {"component": "within_between_poisson", "model": "within_between_history_state", "term": "player_higher_history_match_share", "transition_state": "", "estimate": 2.20, "ci_low": 1.80, "ci_high": 2.60, "events_per_1000_match_hours": np.nan},
            {"component": "within_between_poisson", "model": "within_between_history_state", "term": "within_player_higher_history_deviation", "transition_state": "", "estimate": 0.55, "ci_low": 0.44, "ci_high": 0.68, "events_per_1000_match_hours": np.nan},
            {"component": "switcher_transition_state", "model": "", "term": "", "transition_state": "switcher_pre_higher_history", "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "events_per_1000_match_hours": 33.3},
            {"component": "switcher_transition_state", "model": "", "term": "", "transition_state": "switcher_post_higher_history", "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "events_per_1000_match_hours": 21.7},
        ]
    )
    duration = pd.DataFrame(
        [
            {"group": group, "prior_injury_duration_bucket": bucket, "events_per_1000_match_hours": rate, "events_per_1000_match_hours_ci_low": rate - 2, "events_per_1000_match_hours_ci_high": rate + 2}
            for group, bucket, rate in (
                ("regular", "<1 week", 17.0),
                ("regular", "2 months to 1 year", 14.1),
                ("fragile", "<1 week", 27.4),
                ("fragile", "2 months to 1 year", 20.0),
            )
        ]
    )
    case_crossover = pd.DataFrame(
        [{"model": "previous_7day_minutes", "contrast": "difference_in_per_90_minutes", "estimate": 1.06, "ci_low": 0.88, "ci_high": 1.27, "p_holm_extension_family": 1.0}]
    )
    temporal = pd.DataFrame(
        [
            {"season_start_min": start, "rr_180": estimate, "rr_180_ci_low": low, "rr_180_ci_high": high}
            for start, estimate, low, high in (
                (2017, 1.97, 1.22, 3.17),
                (2020, 1.28, 0.60, 2.73),
                (2022, 1.12, 0.60, 2.11),
            )
        ]
    )
    reporting = pd.DataFrame(
        [
            {"context": "overall", "proxy_timing": "", "type_classifiable_percent": 76.0, "type_classifiable_ci_low": 73.8, "type_classifiable_ci_high": 78.0},
            {"context": "timing", "proxy_timing": "same_day", "type_classifiable_percent": 85.6, "type_classifiable_ci_low": 82.3, "type_classifiable_ci_high": 88.4},
            {"context": "timing", "proxy_timing": "lag1", "type_classifiable_percent": 71.6, "type_classifiable_ci_low": 68.9, "type_classifiable_ci_high": 74.2},
        ]
    )
    reporting_ipw = pd.DataFrame(
        [{"minimum_predicted_type_classifiable_probability": 0.074, "maximum_type_reporting_ipw": 13.5}]
    )
    bootstrap = pd.DataFrame(
        [
            {"history_stratum": "regular", "early_band_global_max_percent": 50.5},
            {"history_stratum": "fragile", "early_band_global_max_percent": 25.0},
        ]
    )
    lineup = pd.DataFrame(
        [{"model": "pooled_history_adjusted", "p_value": 0.894}]
    )
    tier_args = [
        clinical,
        outcome_sensitivity,
        binary,
        frequency,
        attenuation,
        type_history_family,
        denominator,
        minutes,
        effect,
        shape,
        bands,
        status_models,
        status_rates,
        conclusion,
        quality,
        recurrent,
        duration,
        case_crossover,
        temporal,
        reporting,
        reporting_ipw,
        bootstrap,
        lineup,
    ]
    tiers = module.build_result_tier_registry(*tier_args)
    assert tiers.abstract_rule_passes.all()
    assert tiers.main_display_rule_passes.all()
    assert not tiers.loc[tiers.tier.ge(4), "abstract_eligible"].any()
    expected_tiers = {
        "reported_event_duration_linkage": 2,
        "public_proxy_retains_bounded_relative_information": 2,
        "selection_shapes_early_spline_peak": 2,
        "expanded_source_validation": 2,
        "matched_recency_attenuates_apparent_type_threshold": 2,
        "recent_senior_national_match_context": 5,
        "prior_history_incidence_gradient": 4,
        "national_minutes_do_not_change_total_burden_conclusion": 5,
        "no_history_effect_modification": 5,
        "squad_only_status_inference": 5,
        "between_player_susceptibility_corroboration": 4,
        "prior_absence_duration_descriptive_pattern": 5,
        "within_player_case_crossover_no_history_difference": 5,
        "temporal_block_instability": 5,
        "type_reporting_completeness_gate": 5,
    }
    assert dict(zip(tiers["claim_id"], tiers["tier"], strict=True)) == expected_tiers
    assert not tiers["tier"].eq(3).any()
    assert not tiers["tier"].eq(1).any()
    recommended = tiers.loc[tiers["abstract_recommended"], "claim_id"].tolist()
    assert recommended == [
        "matched_recency_attenuates_apparent_type_threshold",
        "public_proxy_retains_bounded_relative_information",
        "reported_event_duration_linkage",
        "selection_shapes_early_spline_peak",
    ]
    assert not tiers.loc[tiers.tier.ge(4), "main_display_recommended"].any()
    assert tiers.loc[tiers.tier.eq(2), "main_display_recommended"].all()
    assert not tiers.loc[
        tiers["claim_id"].eq("recent_senior_national_match_context"),
        "abstract_eligible",
    ].iloc[0]
    proxy_evidence = tiers.loc[
        tiers["claim_id"].eq("public_proxy_retains_bounded_relative_information"),
        "evidence",
    ].iloc[0]
    assert "zero-recent-minute" in proxy_evidence
    assert "1.91" in proxy_evidence and "1.61" in proxy_evidence
    bad_args = tier_args.copy()
    bad_args[20] = reporting_ipw.iloc[0:0]
    with pytest.raises(ValueError, match="reporting-IPW"):
        module.build_result_tier_registry(*bad_args)


def test_result_tier_registry_validator_rejects_bad_visibility_and_duplicates():
    module = _module()
    base = pd.DataFrame(
        {
            "claim_id": ["a"],
            "abstract_rule_passes": [True],
            "main_display_rule_passes": [True],
        }
    )
    duplicate = pd.concat([base, base], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        module._validate_result_tier_registry(duplicate)
    bad_abstract = base.assign(abstract_rule_passes=False)
    with pytest.raises(ValueError, match="abstract visibility"):
        module._validate_result_tier_registry(bad_abstract)
    bad_display = base.assign(main_display_rule_passes=False)
    with pytest.raises(ValueError, match="main-display"):
        module._validate_result_tier_registry(bad_display)
