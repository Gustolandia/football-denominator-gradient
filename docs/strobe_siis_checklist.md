# STROBE-SIIS-oriented submission checklist

This working checklist maps the current public-data manuscript to reporting
items likely to matter under STROBE-SIIS and sports-injury-surveillance review.
It is not a completed journal form. Page and line references must be generated
after the target journal template is fixed.

## Design, setting, and ethics

- The title, abstract, and Methods identify a retrospective observational
  public-data cohort.
- The study covers 1 July 2017 through 7 April 2025; the final season is
  right-censored.
- Player identity uses stable Transfermarkt IDs internally. They are replaced
  by surrogates at the boundary where anything is deposited, so no published
  file carries them.
- The analysis uses public, pre-existing records with no recruitment, player
  contact, intervention, or private medical records.
- The manuscript gives the reason formal approval was not sought and states
  that no institutional determination or reference number was obtained. A
  later institutional determination, if required by the journal, is external
  to this pipeline and will not be invented retrospectively.
- The manuscript states the data-protection basis. The records concern
  identifiable professional athletes and the absence labels are health-related,
  so they are special-category personal data notwithstanding their public
  availability; public availability is not treated as exemption. Processing is
  for scientific research in the public interest under the GDPR research
  provisions, restricted to data already published by the source, with no
  attempt to obtain, infer or link clinical information. No individual is
  identified in any reported result or deposited output: no deposited file
  carries a player name, the provider's player identifier, or a match date.
  Deposited records are aggregated or keyed by a surrogate drawn at random
  once, including the hand-built audit files; the map that reverses the
  surrogates and the reviewer's identified copy of the audit are retained by
  the authors and deposited nowhere. Dates are coarsened to seasons and
  row-level model diagnostics are not deposited. This is de-identification of
  the archive, not anonymisation of the underlying people: the sampling rule is
  published, so a reader holding the provider's snapshot could redraw the same
  appearances.

## Participants and risk set

- Figure 1 shows the flow from 12,440 public reports and the 1,558-player base
  cohort to 88,573 eligible appearances by 1,208 players. Table 1 describes
  player age, position, seasons, club and national appearances, exposure,
  time-varying history membership, outcome timing, report type, and missingness.
- The all-competition grid covers all 101,042 valid EPL-club-season appearance
  days, including 898 match rows omitted by the old first/last-EPL span.
- Canonical episodes collapse touching or overlapping reports and end before
  an observed return. Zero observed appearance rows remain unavailable.
- History is strictly prior to each row and includes 2,266 pre-entry episodes
  plus 2,875,586 pre-entry club minutes.
- Continuous prior report history is primary. Lower, intermediate, and higher
  prior-history strata are descriptive, internally calibrated risk strata,
  not diagnoses or stable phenotypes.

## Exposure definitions

- Previous 3-, 5-, 7-, 10-, and 14-day club minutes are shifted and exclude the
  index match. The independently rebuilt seven-day field equals the legacy
  field on all 88,573 eligible rows.
- All-competition club exposure means recorded appearances for clubs that
  competed in the EPL in that season.
- Recovery interval is reported in calendar-day categories and is not called
  a less-than-72-hour measure.
- The v4 sensitivity adds identified senior competitive, senior friendly, and
  youth/Olympic national appearances. It does not include national training.
- Expanded national exposure remains sensitivity-only because the official
  full-calendar source gate was unexecuted: its ledger had no retained source
  URLs or identifiers.

## Outcome definitions

- The daily outcome is a canonical public-report episode start.
- The reference outcome is a same-day report per recorded appearance. A lag-1
  next-day report and the combined same-day plus lag-1 proxy are separate
  timing sensitivities.
- Other sensitivities include description-specific,
  reported absence of at least 28 days, and muscle/tendon text.
- Public text cannot verify exact onset, mechanism, contact status, diagnosis,
  recurrence, clinical availability, or confirmed time loss.
- Reporting follows STROBE (von Elm et al. 2007) and its sport injury and
  illness surveillance extension, STROBE-SIIS, which is published as part
  of the 2020 IOC consensus statement. An earlier revision cited the
  METHODS MATTER statement for the extension, which was wrong.
- The manuscript uses the 2020 IOC consensus and its 2023 football-specific
  extension to identify which exposure, health-problem, onset, and return-to-
  football fields public reports cannot supply.
- A deterministic exposure-blinded sample checks 30 same-day positives against
  club, league, or contemporaneous news sources independent of Transfermarkt.
  One author performed the audit. Date attribution and description consistency
  are separate outcomes with Wilson intervals and unresolved counts.

## Incidence, severity, and reporting

- The reference scale is reported events per 1,000 appearances with 95%
  confidence intervals. Per-observed-minute and per-fixed-90-minute rates are
  opposing denominator sensitivities.
- Clinical benchmarks are used for scale, not as a formal validation sample;
  their cohorts and definitions differ.
- Reported absence duration is a public-date proxy. It is not called clinical
  severity, confirmed time loss, or injury burden.
- Type text was classifiable for 1,210/1,592 primary proxy events (76.0%, 95%
  CI 73.8--78.0), with 85.6% completeness for same-day and 71.6% for lag-1
  events.
- The reporting inverse-probability sensitivity failed its overlap/weight
  gate (minimum fitted probability 0.074; maximum weight 13.5) and is retained
  only as an archive stress test.

## Statistical methods

- Daily logistic analyses use the full eligible panel rather than sampled
  controls, so no intercept recalibration is needed.
- The reference analysis is a player-clustered binomial-logit model for a
  same-day report per appearance. Previous-seven-day minutes enter linearly per
  90 minutes. Continuous prior history and calendar phase are additive; the
  unsupported history interaction is secondary.
- The complete post-data family crosses seven exposure summaries, same-day,
  lag-1 and combined outcomes, and per-appearance, observed-minute and fixed-90
  denominators. All 63 focal values share one Holm correction.
- A cubic B-spline with fixed 45-, 90-, and 135-minute knots is a non-monotonic
  shape sensitivity. Pointwise intervals and a 10,000-draw simultaneous band
  are distinguished. Its global test detects any shape, not a rising gradient.
- Absolute risks are marginalised over the observed calendar-phase
  distribution. Contrasts compare common calendar distributions rather than
  predictions with periodic terms set to zero.
- Earlier functional-form and cohort checks remain secondary. The current
  exposure audit adds five cumulative windows, appearance count, recovery
  interval, and three fixed temporal blocks using one locked additive model.
- The recorded-minute audit is reported overall, within starters and
  substitutes, and after standardising over the pooled lineup mix, each with a
  1,000-player-bootstrap 95% interval.
- The bounded selection-into-appearance analysis uses unique EPL player-date
  opportunities inferred from observed stints and nearest relevant transfer
  bounds. Overlaps are resolved only by an observed club; unresolved
  non-appearance dates are excluded. Stabilised inverse-selection weights are
  used only after overlap, weight-tail, and balance gates. The result cannot
  adjust for complete roster membership, symptoms, clearance, or tactical need.
- Conditional logistic models compare exposure within player and within
  player-season, using only discordant strata. Player-cluster sandwich intervals
  and 5,000-draw multiplier intervals allow one player to contribute several
  seasons. Included/excluded populations and exposure support are disclosed.
- The historical categorical family contains 154 secondary/exploratory tests
  with Holm and Benjamini-Hochberg adjustment; neither method rejects an
  exposure-response or effect-modification contrast. It is not called
  prospectively prespecified.
- Recovery models use the same observed-minute, fixed-90, logit, and
  complementary-log-log denominator/link checks.
- Recurrent-event checks include player-correlated GEE, switcher fixed effects,
  and within/between decomposition. They do not replace a shared-frailty or
  PAMM recurrent-event model.
- Spline-shape checks cover eight B-spline/natural-cubic specifications. A
  1,000-player bootstrap measures the location and stability of the early
  fitted feature.
- The same-day minute difference uses a separate 1,000-player percentile
  bootstrap and is reported with a 95% confidence interval.
- Current-data extensions include lineup/return restrictions, conditional
  player-season logistic models, competition-context refits, and two-way
  player/current-match clustered uncertainty.
- The v4 national model reports exposure-change support, total-burden refits,
  national-status models, and one multiplicity family. Unsupported squad-only
  models remain explicit non-estimates.

## Bias and limitations

- Selection into appearances is central: players reaching high burden are
  selected on observed and unobserved health, medical, tactical, and squad
  factors.
- The bounded opportunity source started with 198,543 rows. Eleven of 34
  overlapping player-dates were resolved by an observed club and 23 unresolved
  non-appearance dates were excluded; the retained table has one row per
  player-date.
- Same-day event rows contain fewer recorded minutes than non-event rows, so
  observed-minute and fixed-90 denominators represent different timing errors;
  neither is assumed to bound the true event-time denominator.
- In the six complete-lineup seasons, the minute gap is concentrated among
  recorded starters: -31.2 minutes (95% CI -33.9 to -28.6) versus 2.2 (-3.0
  to 7.5) among substitutes. The lineup-standardised gap is -24.7 (-27.1 to
  -22.2).
- Role-specific exposure refits are disclosed with their support: 364 starter
  events and 26 substitute events. The sparse substitute estimate is not
  interpreted as protection or as a direct role contrast.
- Public ascertainment can differ by player profile, outcome timing, injury
  type, and severity.
- Type-specific recency attenuation is a paired specification contrast, not
  causal mediation or a clinical recurrence threshold.
- The official-source national chronology audit is incomplete; unobserved
  national records, national training, and travel remain possible exposure
  error.
- Venue coordinates, medical clearance, symptoms, training load, diagnoses,
  and confirmed time loss are unavailable.
- Null interactions do not prove the absence of clinically meaningful
  heterogeneity.

## Reproducibility

- The attenuation comparison carries a player-resampled percentile interval
  (`jsams_revised_attenuation_bootstrap.csv`, 1,000 replicates): the pooled
  minus within-starter difference is 0.139 (0.128 to 0.150). The ratio of
  relative attenuations is reported alongside it and its interval spans one,
  so the comparison is made on the absolute scale and the Supplement says why.
- The counterfactual untruncated-minutes quantity is imputation-dependent, so
  `jsams_revised_truncation_imputation_sensitivity.csv` refits it under five
  schemes. The reported scheme is the most conservative of the five and the
  attribution to truncation is unchanged across all of them.
- The denominator remedy is reported over every stratum it is claimed for.
  Stratifying by squad role removes the attenuation among starters (0.012)
  and not among substitutes (0.117) or where lineup status is missing
  (0.181, larger than the pooled 0.151). Main-text Table 2 carries all four
  strata with row counts, event counts, gamma and the attenuation interval,
  so no reader needs the Supplement to see where the remedy fails.
- The prespecified illness negative-control outcome yielded one same-day
  event in 88,573 appearances and is reported as not estimable rather than
  failed, since an unestimable control and a large one have opposite
  implications.
- Whether squad role also generates the association is reported rather than
  set aside: adjusting moves the estimate from 1.27 to 1.19 (1.04-1.36), and
  within starters it is 1.181 (0.998-1.398), printed to three decimals so a
  null-spanning interval is not rounded into one that merely touches.
- Every gamma is fitted with player-clustered standard errors, pooled and
  within stratum alike, and reported with a 95% interval: 0.303
  (0.283-0.323) pooled, 0.011 (0.007-0.014) among starters, 0.214
  (0.182-0.247) among substitutes and 0.351 (0.317-0.386) where lineup
  status is unknown. The within-starter interval is disjoint from all
  three others, so the composition claim rests on separated intervals
  rather than on separated point estimates.
- No estimate the paper argues from is printed without uncertainty. The
  within-starter attenuation excludes zero, so the paper says the
  contamination is nearly removed rather than removed, and the recorded-
  minute estimate stays above one, so the artefact is described as
  nullifying a conclusion rather than reversing it.
- Stratifying and adjusting are reported as the different operations they
  are. Adjusting for squad role leaves an attenuation of 0.071
  (0.062-0.080) against 0.151 (0.140-0.163) pooled and 0.012 (0.009-0.015)
  within starters, because a categorical term enters the linear predictor
  while the offset keeps its within-role variation. The three intervals do
  not overlap, so the ordering is not read off point estimates. The
  recommendation names restriction to starters, not stratification by role.
- The attribution of the attenuation to outcome truncation carries a
  player-resampled interval: -0.095% of the attenuation, -0.41% to 0.20%,
  which includes zero. No quantity the paper argues from is now reported
  without uncertainty.
- The 350 players removed by the 900-minute run-in are described rather
  than characterised in passing: median age 22.6 against 26.7 years, 5
  appearances against 56.5, and 180 earlier club minutes against 6,598.
- Directional claims about the role split are gated against the signs in
  the generated table. Starters lose 31.19 minutes on event appearances
  and substitutes gain 2.15 with an interval spanning zero, so the text
  says the pooled difference is driven by starters and that substitutes
  show no shift, and a test refuses any claim that the effect is uniform
  across roles.
- The paper's contribution is a measurement, and every quantity it argues
  from carries an interval: the denominator gradient by stratum and by
  league, all four stratum attenuations, the role-adjusted attenuation and
  the attribution to outcome truncation.
- The cross-league gradient is fitted from appearance records alone --- player
  identifier, date and recorded minutes --- with no injury data, in 628,487
  appearances across eight domestic leagues. The pooled gradient exceeds the
  0.05 reporting threshold in all eight and the within-starter gradient falls
  below it in all eight.
- Dropping the calendar terms, which is what a two-line implementation gives,
  moves no league's gradient by more than 0.056 and changes no verdict.
- Every retained Python source file has a matching test file.
- The final full-suite result was 289 tests passed with 100.00% statement and
  branch coverage across 8,258 statements and 2,080 branches.
- TeXcount counts 3,189 main-text words and 252 abstract words, against
  JSAMS limits of 3,500 and 250. **Both are within limit.**
- The paper is committed fully to the denominator claim. Reaching the limit
  came from compressing explanation rather than deleting findings: the
  rationale for each analysis moved to the Supplement and the main text
  states results. Every number a referee required in the main text is still
  there, including the attenuation decomposition, the direct truncation
  refit, the identity over-prediction, the player-resampled interval on the
  attenuation difference, the case-restricted bias result, the within-role
  remedy and both audit estimands. The exposure-robustness figure, the
  outcome-behaviour section and the outcome-audit figure moved to the
  Supplement.
- `risk_set_history_reconciliation.csv` and
  `history_chronology_reconciliation.csv` audit the two corrected pipeline
  defects.
- `v4_data_quality_registry.csv` records v4 gates.
  `jsams_revised_hypothesis_register.csv` and
  `jsams_revised_claim_hierarchy.csv` control the current analysis families and
  publication visibility. The register contains 706 unique rows. Script 34
  generates 41 predecessor audit tables, script 36 generates 49 current
  tables and script 37 generates the three denominator-gradient tables.
- `selection_membership_resolution_audit.csv`,
  `jsams_revised_window_validation.csv`, and
  `jsams_revised_outcome_audit_validation.csv` must pass before the associated
  results are reported.
- Code and sanitized derived-output releases are documented in the repository;
  generated local source caches remain excluded from Git.
