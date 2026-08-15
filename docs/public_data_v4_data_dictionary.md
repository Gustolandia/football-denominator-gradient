# Public-data v4 data dictionary

This dictionary covers the public-data v4 extension specified in
[`public_data_v4_protocol.md`](public_data_v4_protocol.md). It is separate
from the immutable raw acquisition and shares the corrected canonical risk set
with the current club-only analysis. All v4 files are
local, generated artifacts under `data/raw/public_data_v4/` or
`data/processed/public_data_v4/`, both ignored by Git because they contain
large snapshots, cached public responses, or reproducible derived data.

## Raw snapshot and cache

`transfermarkt_datasets_YYYYMMDD.zip` is the downloaded archive.  Its
extracted sibling directory contains compressed source CSVs and
`snapshot_manifest.json`.  The manifest records source URLs, retrieval UTC,
upstream commit, source terms URL, file bytes, SHA-256 digests, and source CSV
headers.  A date directory is never overwritten.

`public_data_source_catalog.csv` records each additional public source, its
pinned URL, accepted or rejected role, and known limit. The matching immutable
independent senior-results and World Cup lineup files live under dated raw v4
directories with their own `source_manifest.json` files and retained licences.

`national_performance_cache/<tm_player_id>.json` is an unmodified response
from the publicly served Transfermarkt player-performance page endpoint.
These are source caches, not an official API.  `national_acquisition_log.csv`
records one attempt summary per stable cohort player:

| Column | Meaning |
| --- | --- |
| `tm_player_id` | Stable Transfermarkt player identifier from the frozen EPL cohort. |
| `source_url` | Exact public endpoint URL requested. |
| `cache_file` | Local raw JSON cache path. |
| `status` | `cached`, `downloaded`, or `error`; errors are never treated as zero national exposure. |
| `attempts` | HTTP attempts used for a non-cached record. |
| `n_normalised_rows` | Observed in-window national appearances retained after source filtering. |
| `n_national_records` | All in-window national records before participation/minute filtering. |
| `n_played_records` | Records whose source participation state is `played`. |
| `n_played_missing_minutes` | Played records excluded because source minutes are missing. |
| `n_nonplayed_records` | Bench, squad, injured, unavailable, or other non-played records retained only in the audit ledger. |
| `error` | Stored error text when acquisition failed. |
| `retrieved_at_utc` | Request/cache audit time. |

## Cohort and appearances

`epl_cohort_manifest.csv` contains every stable identifier used in the current
EPL panel.  `observed_club_seasons`, `observed_first_epl_club_appearance`, and
`observed_last_epl_club_appearance` come from actual club appearance records.
`country_of_citizenship` and `current_national_team_id` are contextual profile
fields only.  They never infer a national appearance.  `recorded_transfer_dates`
is a semicolon-separated audit value, and `unresolved_cohort_id` flags any
cohort ID lacking observed EPL-club membership.

`international_performance_record_audit.csv` is the pre-filter ledger. It
contains every in-window national record, including non-played participation
states and played records with missing minutes. `retained_for_exposure` and
`exclusion_reason` make the exposure decision explicit. This file, rather
than the already-filtered appearance table, supplies the minute-completeness
denominator.

`international_appearances.csv` contains one retained observed player-game
record after source reconciliation. Essential fields are:

| Column | Meaning |
| --- | --- |
| `tm_player_id`, `game_id`, `match_key` | Player and game identifiers. `match_key` is source-independent (`national:<game_id>`) after reconciliation. |
| `date`, `kickoff_utc`, `kickoff_time_known` | Match calendar date; true UTC kickoff only where the public source supplied an explicit time. |
| `team_id`, `team_name`, `opponent_team_id`, `opponent_team_name`, `team_venue` | National teams taken from the actual performance record. A missing name is not imputed from citizenship. |
| `competition_id`, `competition_name`, `competition_type` | Source competition information. |
| `team_level` | `senior`, `youth_or_olympic`, or `unknown`, derived from source competition type and, only as a fallback, source team naming. |
| `competition_status` | `competitive`, `friendly`, or `unknown`; missing source classification remains unknown. |
| `minutes_played`, `is_starter`, `participation_state` | Observed public performance fields. Missing minutes are not changed to zero. |
| `stadium_id`, `stadium_name`, `venue_key` | Public venue references used only for optional geographic proxies. |
| `source`, `source_url`, `cache_file`, `retrieved_at_utc` | Provenance for published-snapshot and endpoint sources. |
| `duplicate_resolution` | `single_source`, `published_preferred_consistent`, or `endpoint_duplicate_consistent`. Conflicting minutes are excluded and recorded in the duplicate audit. |

`national_duplicate_audit.csv` contains every player-game source collision.  A
row with `resolution = unresolved_minutes_conflict_excluded` prevents primary
use of expanded exposure until resolved.

`snapshot_national_appearance_audit.csv` explains why the bulk Transfermarkt
snapshot alone cannot supply the study-window national player exposure: it
counts its national matches and appearances by date range and records the zero
in-window cohort contribution. `international_appearances_raw.csv` is the
normalized endpoint-derived appearance table before cross-source duplicate
resolution.

`national_team_id_crosswalk.csv` maps source team IDs to normalized and display
names with a recorded mapping source. `independent_schedule_validation.csv`
contains the date/team/tournament/score reconciliation against the pinned
independent senior-results source. It is a secondary chronology validation,
not the binding official-source gate. `openfootball_worldcup_player_validation.csv`
matches the 2018 and 2022 World Cup subset on date, teams, normalized player
name, starter status, and approximate minutes. It validates a narrow public
subset and cannot establish complete national-player coverage.

## Unified timeline and exposure features

`public_match_timeline.csv` combines observed competitive EPL-club and
reconciled national appearances.  The five indicator columns are source-row
memberships for the preregistered scopes:

- `is_club_competitive`
- `is_senior_competitive`
- `is_senior_friendly`
- `is_youth_international`
- `is_club_friendly`

The corresponding refreshed-timeline scope names are `club_competitive`,
`club_plus_senior_national`, `club_plus_senior_all`,
`club_plus_broader_international`, and `club_plus_all_public`.  The internal
`senior_competitive_national_only` feature supplies the observed national-only
increment for the frozen comparator. `senior_all_national_only` adds senior
friendlies, and `broader_international_only` also adds youth/Olympic matches.
These are exposure scopes, not clinical subgroups. The
refreshed club scopes are reconstruction sensitivities because the frozen
all-competition variable includes recorded club friendlies.

`match_exposure_scope_features.csv` has one row for every eligible source
club-match row. For each scope and `N` in 3, 5, 7, 14, and 28 it
contains:

| Field pattern | Meaning |
| --- | --- |
| `<scope>_minutes_last_<N>d` | Sum of source minutes on dates in `[index date - N, index date)`; the index match is excluded. |
| `<scope>_matches_last_<N>d` | Number of distinct observed source games in the same shifted window. |
| `<scope>_national_minutes_last_<N>d` | National contribution within that scope and window. |
| `<scope>_days_since_previous_appearance` | Whole calendar days to the prior observed source appearance. |
| `<scope>_consecutive_match_sequence` | Count of prior observed matches in the contiguous sequence, reset after a gap greater than 14 calendar days. |

`recovery_measure` is always `calendar_days` in this v4 build because club
source files do not provide sufficiently complete UTC kickoffs.  It must not be
renamed as a 72-hour recovery variable.

`exposure_scope_comparison.csv` is a compact refreshed-timeline reconciliation
table.  `frozen_baseline_national_scope_comparison.csv` is the direct
comparison table: it shows how often the frozen prior-seven-day club burden
changes when observed senior competitive national minutes are added, how often
zero burden becomes positive burden, and total added minutes.
`baseline_parity_report.csv` records both the zero-row frozen-comparator
parity check and the refreshed strict-club reconstruction differences.  The
latter must be read before interpreting refreshed-scope estimates.

`international_status_ledger.csv` retains played, squad-only, recorded-
unavailable, and not-in-squad senior/youth records separately, together with
the independent schedule fields available for each row. `national_status_features.csv`
then attaches shifted 3-, 5-, 7-, 14-, and 28-day status indicators and match-
day counts to every eligible club appearance. A non-played status never adds
minutes and is not converted into a played appearance.

## Coverage, travel, outcome, and selection audits

`international_exposure_coverage_audit.csv` provides observed appearance and
minute completeness by competition, season, national team and player.
`official_schedule_validation.csv` is a manual source-validation queue.  Its
`verified` value is false until a dated official FIFA, UEFA, confederation, or
equivalent match source has been recorded in `official_source_url`.

`exposure_coverage_audit.csv` distinguishes binding gate rows from secondary
chronology checks. `binding_for_primary_use=True` applies to the protocol-recorded official-
schedule coverage, primary senior-competitive minute completeness, failed
cohort-ID count, and unexplained duplicate count. The independently
reconstructed schedule coverage and all-played-record minute completeness are
informational rows with no registered threshold. `gate_role` records each
row's function. `primary_v4_exposure_allowed` is true only when all four
binding rules pass; otherwise `decision` is `sensitivity_only`.

`venue_geocodes.csv` is a coordinate validation template.  Coordinates require
`source_url` and `match_confidence`; `evidence_status` is `verified` only when
both latitude and longitude are present.  `geographic_travel_km` and
`geographic_timezone_change_hours` in the unified timeline are distances and
offset changes between consecutive observed venues.  They are geographic travel
proxies, not measurements of player travel.  Unknown venues stay missing.
`geographic_travel_coverage_audit.csv` reports verified venue-coordinate
coverage and the number of timeline rows with an estimable distance or time-
zone-change proxy.  A zero usable-row count rules out a travel analysis rather
than implying zero travel.

`injury_source_validation.csv` is a deterministic public-source review queue
stratified by reported absence of at least 28 days, muscle/tendon description,
ambiguous description, and other unmatched description.  `official_evidence_grade`
remains `unreviewed` until a reviewer adds an official source URL and evidence
assessment.  This file never changes the primary Transfermarkt injury outcome
automatically.

`selection_risk_set.csv` contains conservative EPL player-fixture opportunities
defined by observed club appearance spans, recorded transfers, scheduled EPL
fixtures, and the existing public injury availability flag. Transfer dates use
the latest arrival on or before the observed stint and the earliest departure
on or after it; career-wide earliest/latest dates are not propagated into every
season. Appearance status is matched on player, fixture, and club. Overlapping
player-date memberships retain the observed club when one appearance identifies
it and otherwise are excluded. `played_any_minutes` is observed playing status.
`plausibly_available` is not a medical clearance variable.
`selection_membership_resolution_audit.csv` reports input rows, exact
duplicates, overlapping player-dates, observed-club resolutions, ambiguous
dates excluded, final rows, and the unique-player-date gate.
`selection_probability` and `stabilized_ipw` are populated only for an
estimable prior-information logistic selection model.
`selection_weight_diagnostics.csv` records overlap, 99th-percentile weights,
weighted standardised mean differences, and `ipw_usable`.  A false gate means
no weighted outcome model should be reported.

## Model outputs

`v4_scope_selected_predictions.csv` contains the existing primary spline
prediction grid for `frozen_club_all` and
`frozen_club_plus_senior_national`. `v4_model_comparison.csv` combines the
original formal previous-seven-day spline contrasts and recovery-interval
trend tests across those identical-row comparators, with Holm and
Benjamini-Hochberg adjustment across the v4 comparison family.
`v4_model_input_audit.csv` records the number of model rows, match-proxy events,
dispersion and estimator for each fit.  These outputs are sensitivity outputs
when the coverage gate fails; their numerical existence does not change the
gate decision.

`v4_national_record_quality_audit.csv` summarises the pre-filter source
denominators, exclusions, kickoff completeness, duplicate count, and retained
source distribution. `v4_exposure_change_audit.csv` reports changed rows,
events, minutes and zero-burden reclassification for all three country scopes
and five windows. `v4_recovery_change_audit.csv` reports recovery intervals
shortened by an observed country appearance.

`v4_all_scope_selected_predictions.csv`,
`v4_all_scope_model_comparison.csv`, and
`v4_all_scope_model_input_audit.csv` extend the identical primary model to
senior-all and broader-international increments while preserving the frozen
club comparator.

`v4_country_duty_between_within.csv` contains the post-hoc source-specific
country-duty analysis. No current contrast survives multiplicity adjustment.
`duty_between` is each player's mean probability of a
recent country appearance; `duty_within` is the row-level deviation from that
mean. Continuous alternatives use country match-equivalents (`minutes/90`) or
appearance counts. Every row records its window, outcome, denominator,
controls, history definition, raw p-value, Holm/BH values across the complete
exploratory family, and Holm/BH values for the same contrast across
specifications. `analysis_role` is always
`post_hoc_hypothesis_generating`.

`v4_country_duty_support.csv` records rows, exposed rows, events, and exposed-
row events by specification and history stratum. `v4_conclusion_audit.csv`
contains the machine-readable bounded decision for each potential v4 claim.

`v4_national_status_rates.csv` reports exact Poisson rate intervals by recent
national-match status. `v4_national_status_models.csv` reports fitted
participation contrasts, history interactions, raw p-values, and Holm/BH
values across the complete status family. Rows without adequate support remain
explicit non-estimates. `v4_national_status_model_support.csv` records the
pre-fit support decision. Outcome identifiers use `description_specific_7d`
rather than `mechanism_specific_7d`, because public text cannot establish an
injury mechanism.

`data/processed/results/risk_set_history_reconciliation.csv` records the core
source-to-risk-set integrity checks used before v4 fitting: source appearance
days, represented match rows, rows added beyond the old span, observed
appearances still marked unavailable, cleaned reports, canonical episodes,
absorbed reports, and return-truncated episodes.

`v4_data_quality_registry.csv` harmonises structural, coverage, uncertainty,
and support checks. `v4_result_tier_registry.csv` preserves the earlier v4
extension audit but no longer controls manuscript visibility. The controlling
publication registry is `jsams_revised_claim_hierarchy.csv`, generated after
all v4 and current-data checks by script 36. `jsams_claim_hierarchy.csv` remains
the predecessor script-34 registry.

Both registries use the lowest defensible tier: Tier 1 is a central, strongly
original result with implications beyond one narrow specification; Tier 2 is
original but narrower, partly anticipated, or sensitivity-bound; Tier 3 is
reserved for a highly surprising null or a direct contradiction of comparable
published evidence; Tier 4 corroborates an established result; and Tier 5
contains other null or uninformative results. Ambiguous claims receive the
lower tier, and a tier is never raised to populate the abstract.

The current publication registry assigns Tier 2 to the multiverse-supported
same-day per-appearance recent-exposure association, the independent-source
outcome-attribution audit, and the link between a same-day record and the
recorded-minute denominator. It assigns Tier 5 to bounded selection weighting,
the continuous
history-interaction null, historical categorical null, national chronology
sensitivity, and the outcome-quality family with no adjusted rejection. The
prior-history gradient is a Tier 4 corroboration. This hierarchy is why
selection, national chronology, and type-recency results are not main displays.

Two primary-pipeline files support the type-history recency audit.
`data/processed/results/matchproxy_type_history_recency_attenuation.csv` uses a
stacked, player-clustered Poisson fit to estimate the adjusted/unadjusted
coefficient ratio and its 95% confidence interval. The component coefficients
are reproduced within the same fit. `matchproxy_type_history_multiplicity_family.csv`
combines all finite p-values from the overlapping-label, mutually exclusive
binary, continuous, shape, threshold, and attenuation specifications and adds
Holm and Benjamini--Hochberg values across the 74-test family.

`matchproxy_negative_control_type_frequency_linearity_formal_test.csv` also
contains two distinct collinearity diagnostics. The
`*_frequency_log_recency_corr_prior_rows` fields describe the substantive
frequency--recency relation only where a prior same-type report exists. The
`*_frequency_vif` fields are variance-inflation factors calculated from the
exact fitted-model design matrix and therefore include the spline, calendar,
binary-threshold, both frequency, and matched-recency terms in that
specification. The `*_corr_all_rows` fields are retained for auditability but
must not be read as model collinearity: assigning zero frequency and zero
recency to no-prior rows can reverse the correlation sign, while the fitted
has-prior indicator absorbs that presence/absence split.

## Post-primary current-data extension outputs

`src/33_matchproxy_current_data_extensions.py` writes the following files to
`data/processed/results/`. These output names use the internal `regular` and
`fragile` model labels; publication text must translate them to intermediate
and higher prior-injury-history strata. All extension analyses use the current
canonical match-proxy rows from the primary spline model and are observational
diagnostics, not replacements for the primary estimand.

| Output | Contents and permitted interpretation |
| --- | --- |
| `matchproxy_extension_lineup_refits_summary.csv` | Model audit, selected predictions, support and interaction fields for recorded-lineup, starters-only, substitutes-only, return-excluded, and starter-plus-return-excluded refits. They diagnose observed risk-set composition; absent lineups and latent fitness remain unobserved. |
| `matchproxy_extension_lineup_refits_predictions.csv` | Full burden prediction grid for each lineup/return restriction, with player-clustered confidence intervals. |
| `matchproxy_extension_lineup_refits_shape.csv` | Minimum/maximum predicted burden locations and rates by restriction and history stratum. A maximum is a fitted feature, not a workload threshold. |
| `matchproxy_extension_lineup_spline_interaction.csv` | Global recorded-lineup-role-by-spline Wald tests in pooled and stratum-specific models, including Holm/BH values across the three tests. |
| `matchproxy_extension_reporting_completeness_context.csv` | Wilson intervals for the proportion of proxy-event descriptions classifiable into the public type dictionary, overall and by timing, history, lineup and joint contexts. This is text completeness, not diagnostic accuracy. |
| `matchproxy_extension_reporting_type_model.csv` | Player-clustered logistic associations with type-classifiable text. The comparison identifies conditions related to public text detail, not injury mechanism. |
| `matchproxy_extension_reporting_type_ipw_diagnostics.csv` | Predicted classification probabilities, weight distribution, effective event count, predeclared gate values and `stability_status`. A non-`stable` result prohibits using the weighted model as a correction. |
| `matchproxy_extension_reporting_type_ipw_summary.csv` | Weighted muscle/tendon spline point predictions plus 200-player-bootstrap intervals. `fit_status=unstable_reporting_weight_tail` means archive-only stress test; its blank interaction fields are intentionally non-estimates. |
| `matchproxy_extension_reporting_type_ipw_selected.csv` | Selected weighted prediction anchors and player-bootstrap percentile intervals. Interpret only if the diagnostics gate is stable. |
| `matchproxy_extension_reporting_type_ipw_ratios.csv` | Weighted history-rate ratios and player-bootstrap percentile intervals. Interpret only if the diagnostics gate is stable. |
| `matchproxy_extension_reporting_type_ipw_bootstrap.csv` | Individual player-bootstrap weighted predictions and success status; retained for reproducibility of percentile intervals. |
| `matchproxy_extension_reported_absence_day_burden_history.csv` | Player-bootstrap intervals for reported dated-episode days per 1,000 recorded match hours by history stratum. This is a public absence-date proxy, not clinical injury burden or confirmed time loss. |
| `matchproxy_extension_reported_absence_day_burden_history_by_burden.csv` | The same public absence-date proxy cross-tabulated by prior-burden band. Sparse bins are descriptive. |
| `matchproxy_extension_reported_duration_conditional_model.csv` | Gamma-model ratios for `1 + reported absence days` among proxy events. These conditional reporting-date associations do not establish severity, rehabilitation effect, or causal workload effect. |
| `matchproxy_extension_joint_burden_recovery_support.csv` | Row, player, event and minute support by joint prior-minute/recovery cell. It is the support audit that determines which joint comparisons can be fit. |
| `matchproxy_extension_joint_schedule_compression_model.csv` | Support-limited one-versus-two-prior-club-match rate ratios at 45--180 prior minutes and 0--5 recovery days. The difference contrast is the history-specific comparison; neither rate ratio is causal. |
| `matchproxy_extension_within_player_case_crossover.csv` | Conditional-logistic within-player-season previous-minute and recovery associations, with raw/Holm/BH p-values. The risk set contains observed appearances in event-containing player-seasons and does not solve time-varying selection. |
| `matchproxy_extension_current_match_metadata_audit.csv` | Exact reconciliation of model rows, players, events and source versus observed current-match minutes after matching each row to a unique source match. |
| `matchproxy_extension_competition_context_rates.csv` | Crude current-competition-specific proxy rates and intervals. Context labels describe the current observed match, not all recent burden. |
| `matchproxy_extension_competition_context_refits.csv` | Primary-spline-format refits with current competition adjustment and current-Premier-League-only restriction. These are context sensitivity models. |
| `matchproxy_extension_two_way_cluster_sensitivity.csv` | Primary model contrasts with sandwich covariance clustered jointly by player and current match. It quantifies dependence sensitivity, not causal confounding. |
| `matchproxy_extension_curve_feature_bootstrap_samples.csv` | One row per player-bootstrap replicate and history stratum, recording fit success, global maximum location and early-band maximum ratio. |
| `matchproxy_extension_curve_feature_bootstrap_summary.csv` | Bootstrap success counts, the proportion of replicates whose global maximum lies in 15--45 minutes, percentile interval, maximum-location percentiles and early-band/90-minute ratio percentiles. It assesses fitted-feature stability, not a clinical threshold. |

The extension has two hard interpretation gates. First, an inverse-probability
reporting result is usable only when `stability_status` is `stable`; failed
positivity/weight diagnostics leave it archive-only. Second, a joint
burden/recovery result must be restricted to cells with recorded support in the
support table; sparse high-burden cells cannot support a two-dimensional
exposure-response conclusion.

## Predecessor JSAMS measurement-analysis outputs

`src/34_jsams_referee_analysis.py` writes 41 CSV files under
`data/processed/results/`. They remain the source for cohort, denominator,
quality, historical, and contextual checks, but script 36 now controls the
reference model and publication hierarchy.

| Output | Contents and permitted interpretation |
| --- | --- |
| `jsams_primary_model_specification.csv` | One row per symmetric model defining the outcome, denominator, formula, fixed knots, continuous-history scale, observed-calendar standardisation, cluster and analysis counts. |
| `jsams_primary_model_coefficients.csv` | All coefficients and clustered uncertainty for the nine same-day/lag-1/combined by appearance/observed-minute/fixed-90 models. Coefficients are model parameters, not causal effects. |
| `jsams_primary_model_predictions.csv` | Observed-calendar-standardised prediction grids from 0 to 220 minutes at continuous low, median and high history anchors. Publication figures stop at 180 because 220 lacks local events. |
| `jsams_primary_model_contrasts.csv` | Direct 0-to-90, 0-to-180 and 90-to-180 ORs or IRRs, each with 95% confidence intervals and its estimand label. |
| `jsams_primary_model_tests.csv` | Global exposure, exposure-by-continuous-history and any-exposure Wald tests for all nine symmetric models. |
| `jsams_functional_form_tests.csv` | Three formal tests for each of the reference B-spline, linear-per-90, four-df B-spline, four-df restricted cubic spline and fixed-band exposure forms. |
| `jsams_functional_form_contrasts.csv` | Median-history 0-to-180 same-day per-appearance ORs for the five exposure forms. These assess form sensitivity, not five discoveries. |
| `jsams_outcome_quality_summary.csv` | Crude per-appearance and per-observed-hour rates with Wilson and exact-Poisson 95% intervals for all same-day, at-least-28-reported-day and muscle/tendon-text outcomes. |
| `jsams_outcome_quality_tests.csv` | Reference-form global tests under the three same-day outcome definitions. Restrictions remain public-report proxies. |
| `jsams_outcome_quality_contrasts.csv` | Median-history 0-to-180 per-appearance ORs for the three same-day outcome definitions. The severe anchor must be read with its unsupported global spline test. |
| `jsams_cohort_robustness_audit.csv` | Row, player, event and exposure-support counts for every eligibility, season and exposure-scope cohort. |
| `jsams_cohort_robustness_tests.csv` | Reference-form global tests for the 450-, 900- and 1,800-minute, complete-season, pandemic-excluded and club-plus-national cohorts. |
| `jsams_cohort_robustness_contrasts.csv` | Median-history 0-to-180 per-appearance ORs and 95% intervals for those six cohorts. |
| `jsams_cohort_flow.csv` | Mutually exclusive flow from 12,440 public reports through 11,993 episodes, 101,042 source appearances and the 88,573-appearance reference risk set, including lineup ascertainment. |
| `jsams_cohort_descriptives.csv` | Player age, position, seasons, club and national appearances, exposure, history-row membership, outcome timing, report type and missingness. |
| `jsams_exposure_support.csv` | Appearance, player and event counts in local windows around 0, 90, 180 and 220 minutes by descriptive history stratum. |
| `jsams_same_day_minute_bootstrap_samples.csv` | The 1,000 player-resampling replicates for the overall mean recorded-minute difference between same-day-report and non-event appearances. |
| `jsams_same_day_minute_bootstrap_summary.csv` | Overall point means, difference and 95% player-bootstrap percentile interval. It does not estimate event time or minutes lost. |
| `jsams_lineup_minute_bootstrap_samples.csv` | The 1,000 player-resampling replicates for starter, substitute and pooled-lineup-standardised minute differences. |
| `jsams_lineup_minute_bootstrap_summary.csv` | Role-specific and lineup-standardised minute differences, support and 95% percentile intervals. |
| `jsams_lineup_role_tests.csv` | Global same-day per-appearance tests from separate starter and substitute model refits. These do not compare roles directly. |
| `jsams_lineup_role_contrasts.csv` | Role-specific median-history 0-to-180 ORs. Sparse substitute events limit interpretation. |
| `jsams_selection_standardized_curves.csv` | Lineup-known curves before composition adjustment and after standardisation for recorded role and recent return, each over a common observed calendar distribution. |
| `jsams_selection_standardized_comparisons.csv` | Legacy pointwise comparison of the two fitted curves; use the direct effect file for interval-supported change contrasts. |
| `jsams_selection_standardized_tests.csv` | Global exposure and interaction tests before and after lineup/return adjustment. A p-value threshold change is not itself evidence that the estimates differ. |
| `jsams_selection_effect_bootstrap_samples.csv` | Paired estimates from 1,000 player-cluster bootstrap samples of the complete-lineup-season frame. Every estimable replicate refits the selection model and recomputes changing-composition, fixed-composition and difference estimates. |
| `jsams_selection_effect_contrasts.csv` | Direct 0-to-45 and 0-to-180 risk changes under changing versus fixed composition, and their paired difference, at three continuous-history anchors. Intervals are percentiles of the full refit bootstrap; Holm values cover all 18 contrasts. |
| `jsams_lineup_completeness.csv` | Recorded lineup-role coverage by season, exposure band, history quartile, same-day outcome and competition, with binomial intervals. |
| `jsams_lineup_reweighting_assessment.csv` | Positivity assessment for full-cohort lineup reweighting and the complete-season restriction used by the selection analysis. |
| `jsams_within_player_same_day.csv` | Conditional-logistic 0-to-180 spline and linear contrasts within player and within player-season, including discordant strata, rows, players and event counts. |
| `jsams_daily_report_timing_summary.csv` | Recorded spell-start frequencies on appearance days, next days and other eligible player-days. |
| `jsams_daily_report_timing_bootstrap_samples.csv` | Player-cluster bootstrap frequencies used for timing-enrichment intervals. |
| `jsams_daily_report_timing_contrasts.csv` | Appearance-day and next-day risk ratios and risk differences versus other eligible player-days. These are chronology checks, not event-onset validation. |
| `jsams_eligibility_player_comparison.csv` | Age, appearance, minute and crude same-day-frequency comparison for players included and excluded by the 900-prior-minute rule. |
| `jsams_context_sensitivity_tests.csv` | Global tests for age/position/club-season, competition, EPL-only and player/match-cluster checks. |
| `jsams_context_sensitivity_contrasts.csv` | Median-history 0-to-180 ORs and 95% confidence intervals for each measured-context check. |
| `jsams_national_exposure_audit.csv` | Reference-row count, player count, total minutes and maximum seven-day increment when senior competitive national minutes are added. |
| `jsams_hypothesis_register.csv` | One row per formal test with source, role, family, estimability, raw and adjusted p-values. This is Supplementary Data S1. |
| `jsams_hypothesis_family_summary.csv` | Registered, estimable and adjusted-rejection counts by family and test domain, including all reviewer-requested post hoc families. |
| `jsams_claim_hierarchy.csv` | Publication tier, explicit tier justification, role, evidence, caveat, abstract/main-display visibility rule, and one-sentence main-Results limit for Tier 4--5 claims. |

## Current additive and reviewer-requested outputs

`src/36_jsams_second_referee_analysis.py` writes the following 16 CSV files.
All use the prefix `jsams_revised_` under `data/processed/results/`.

| Output | Contents and permitted interpretation |
| --- | --- |
| `jsams_revised_window_validation.csv` | Exact row-level parity between independently reconstructed and legacy previous-seven-day minutes. A failed gate stops fitting. |
| `jsams_revised_exposure_multiverse.csv` | All 63 exposure-summary by outcome-timing by denominator estimates, player-clustered 95% intervals, raw values, and one common Holm adjustment. |
| `jsams_revised_exposure_multiverse_summary.csv` | Distribution of the 63 estimates overall and within each denominator and outcome timing: count, Holm rejections, minimum, quartiles, median, maximum, and share above one. Report this rather than a rejection count, because a Holm family cannot control the choices that defined the family. |
| `jsams_revised_exposure_metric_correlations.csv` | Pearson and Spearman correlations for all 21 exposure-metric pairs, flagging pairs involving the seven-day reference window. High values mean the cumulative windows are correlated sensitivity analyses, not independent replications. |
| `jsams_revised_exposure_metric_summary.csv` | The seven same-day per-appearance rows extracted from the complete family for the main forest plot. |
| `jsams_revised_additive_curves.csv` | Calendar-standardised additive linear predictions and flexible spline predictions on 0--180 minutes. Linear intervals are pointwise; spline intervals include a 10,000-draw simultaneous band. |
| `jsams_revised_additive_curve_tests.csv` | Global spline departure-from-flat test and an explicit warning that it does not test a monotonic increase. |
| `jsams_revised_absolute_risk_contrast.csv` | Standardised probabilities per 1,000 appearances at zero and 180 previous-seven-day minutes and their difference, with delta-method intervals. `target_population` names the estimand: observed calendar-phase distribution with continuous prior history fixed at its sample median. |
| `jsams_revised_exposure_support.csv` | Appearances, same-day events, players, share and event rate in each fixed exposure band. Shows which part of the fitted curve rests on data; the band above 180 minutes carries 181 appearances and no events. |
| `jsams_revised_confounding_sensitivity.csv` | The reference seven-day estimate refitted with age, position, club-season, competition and season added singly and together, restricted to Premier League matches, and under player-match two-way and club-season clustering. Bounds measured confounding only; unmeasured health and selection remain. |
| `jsams_revised_temporal_stability.csv` | Three locked additive refits, Holm adjustment across their slopes, and one global block-heterogeneity test. Players may recur across blocks. |
| `jsams_revised_conditional_estimates.csv` | Player and player-season conditional 180-versus-zero estimates with player-cluster sandwich intervals, 5,000-draw multiplier intervals, and two-test Holm values. |
| `jsams_revised_conditional_population.csv` | Included discordant and excluded concordant rows, players, events, exposure, and history summaries for each conditional target population. |
| `jsams_revised_conditional_support.csv` | Rows, players, and events in fixed recent-minute bands for each conditional model. |
| `jsams_revised_appearance_selection_estimates.csv` | Unweighted and stabilised inverse-selection-weighted seven-day estimates in the bounded opportunity set. A restricted measured-selection sensitivity, not an adjustment for selection into appearance, and not a complete registered roster. |
| `jsams_revised_appearance_selection_diagnostics.csv` | Opportunity count, selected share, propensity overlap, weight tail, weighted standardised mean differences, and no-leakage gates. The gates confirm that no outcome column enters the propensity model and that every Premier League appearance and same-day spell start is retained; appearances in other competitions fall outside the reconstructed EPL fixture set by design, which restricts scope rather than conditioning on the outcome. |
| `jsams_revised_appearance_selection_population.csv` | Appearances, players, same-day events and event rate, median recent minutes, history and age, and Premier League share, for the included and excluded populations. |
| `jsams_revised_denominator_contrast_metadata.csv` | Estimator, uncertainty unit, interval method, bootstrap replicates, the complete-lineup seasons, the lineup standardisation rule, and an explicit statement that the recorded-minute contrast licenses no causal reading. |
| `jsams_revised_outcome_audit_queue.csv` | Deterministic exposure-blinded 30-record queue, as issued to the searcher. Columns: `record_key`, `player_key`, `season`, `matchproxy_injury_desc`, `audit_stratum`, and the pending verdict and note columns. It omits recent exposure, history, and fitted-model values, and carries no player name, provider identifier or match date. |
| `jsams_revised_non_event_audit_queue.csv` | The same queue construction applied to 30 appearances carrying no same-day report, for false-negative review. Same surrogate keying; verdicts ship `pending`, and the adjudications live in `data/manual/`. |
| `jsams_revised_non_event_absence_screen.csv` | For each queued report-free appearance, the fixtures the player's own club played between that appearance and the player's next one. Columns: `record_key`, `player_key`, `days_to_next_appearance`, `club_fixtures_missed`, `club_changed`, `screen`, `interpretation`. It assigns no verdict; it establishes whether an absence existed for the record to have missed. |
| `jsams_revised_outcome_audit_validation.csv` | Gates for exact queue identity, immutable fields, absence of exposure fields, source completeness, and independence from Transfermarkt. |
| `jsams_revised_outcome_audit_summary.csv` | Date-attribution and description-consistency proportions by audit stratum and overall, with Wilson 95% intervals and unresolved counts. This is not a clinical validation sample. |
| `jsams_revised_claim_hierarchy.csv` | Controlling claim tier, justification, evidence, caveat, and abstract/display visibility gates. |
| `jsams_revised_hypothesis_register.csv` | Complete 706-row register: 637 predecessor rows, 63 multiverse tests, three temporal slopes, one heterogeneity test, and two conditional estimates. |

The hand-reviewed decisions are versioned separately in
`data/manual/independent_same_day_event_audit.csv` and
`data/manual/independent_non_event_audit.csv`, keyed by the same surrogates as
the queues above. One author performed the audit. “Independent” refers to
source independence from Transfermarkt, not an independent assessor or clinical
adjudication.

The source URL behind each verdict is not deposited: most of these slugs
contain the player's surname and one contains a graded diagnosis, so the URL
identifies as surely as the name column did. What the verdict rests on is kept
--- `independent_source_found` records whether a qualifying source was located
and `independent_source_type` records what kind it was --- and the URLs
themselves are held with the reviewer's identified copy, which is deposited
nowhere. `src/38_deidentify_audit_evidence.py` performs the transformation and
checks the withheld URLs against the Transfermarkt-independence rule while they
still exist.

`src/35_plot_jsams_revision.py` consumes script-34 and script-36 outputs. It writes
`J1_jsams_cohort_measurement.png`, `J2_jsams_primary_robustness.png`,
`J3_jsams_within_player_lineup_coverage.png`, and
`J4_jsams_context_support.png` to `manuscript/figures/`.
