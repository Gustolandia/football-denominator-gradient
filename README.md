[![Software DOI](https://zenodo.org/badge/1110783957.svg)](https://doi.org/10.5281/zenodo.17835593)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17835137.svg)](https://doi.org/10.5281/zenodo.17835137)

# EPL Congestion And Injury Risk Pipeline

This repository builds the data, models, diagnostics, tests, and figures for an
English Premier League fixture-congestion injury-risk analysis covering matches
from the 2017-18 season through `2025-04-07`.

Window note: raw local Transfermarkt-derived files may contain earlier seasons,
but generated processed panels, cleaned injury starts, results, figures, and
the manuscript now use only the 2017-18 onward analysis window. This is encoded
as `ANALYSIS_START_DATE = "2017-07-01"` in `config.py`.

The current manuscript is explicitly a public-data measurement study. Its
primary question is: how stable is the association between previous-7-day
match exposure and a same-day public report per recorded appearance when the
outcome clock, exposure denominator, and observed match selection are changed?
The reference estimand is the conditional probability of a same-day reported
event per appearance among players who reached the match risk set. It is not a
clinical injury-incidence or causal workload estimand.

`src/36_jsams_second_referee_analysis.py` generates the controlling publication
hierarchy in `data/processed/results/jsams_revised_claim_hierarchy.csv`. No
current result meets the strict Tier 1 or Tier 3 definitions. Three original,
sensitivity-bounded Tier 2 results lead the manuscript and abstract:

1. Each additional 90 club minutes in the preceding seven days is associated
   with 1.27 times the odds of a same-day public spell start per appearance
   (95% CI 1.11--1.44), an absolute rise from 5.6 to 8.9 per 1,000 appearances
   (difference 3.3, 1.3--5.4). Across all 63 exposure, outcome-timing and
   denominator combinations the median odds ratio is 1.16 (interquartile range
   1.06--1.23). Seven days is the single reference window: the cumulative
   windows overlap by construction (r up to 0.85), so the 5-, 10- and 14-day
   results are correlated sensitivity analyses, not replications. The estimate
   moves little under measured-covariate adjustment or player-match clustering
   (1.27--1.32). The result is post-data, conditional on appearing, and not a
   causal workload threshold.
2. An exposure-blinded 30-record audit against public sources independent of
   Transfermarkt supports exact-match attribution for 24 of 27 resolved records
   (88.9%, 95% CI 71.9--96.1) and 24 of 30 sampled records (80.0%, 62.7--90.5),
   bounding the sampled proportion between 80% and 90%. One author performed
   the audit and only reported positives were sampled, so this is
   source-independent scrutiny of date attribution rather than independent
   clinical adjudication or case ascertainment.
3. Same-day-report appearances contain 21.0 fewer recorded minutes than other
   appearances (95% CI 19.0--23.1 fewer). None of seven observed-minute
   associations survives the 63-model correction. This demonstrates
   denominator sensitivity; it does not reveal exact event time.

The continuous history-interaction result (`p = 0.153` after Holm correction),
the 0/154 adjusted categorical exposure-response/effect-modification family,
national-exposure robustness, and restricted measured-selection weighting are
Tier 5. Severe-report restrictions are also Tier 5 because no test survived
their quality family; the prior-history incidence gradient is a Tier 4
corroboration. They remain visible in Results or the Supplement but
not in the abstract. All analyses were selected after outcomes and models were
available; none is confirmatory. `v4_result_tier_registry.csv` and
`jsams_claim_hierarchy.csv` preserve earlier audit stages;
`jsams_revised_claim_hierarchy.csv` controls the current paper.

The corrected pipeline is Transfermarkt-centered. It reads local
Transfermarkt dataset dumps for fixtures, appearances, clubs, and minutes;
  fetches or reuses Transfermarkt injury-report data; reconciles reports into
  non-overlapping episodes; builds a stable player-day
panel; computes all-competition rolling load; assigns day-level prior-injury-
history labels; then runs daily, match-minute, stratified, spline, FFT,
plotting, diagnostic, and clinician-facing bridge analyses.

Sports-medicine terminology in the manuscript follows football and IOC
surveillance conventions where the public data allow it. Match-minute rates are
shown in the familiar unit of events per 1,000 match hours. Reported absence
duration is a severity proxy, not confirmed time loss; the project does not
claim clinical injury burden because that requires confirmed days lost per
exposure hours.

The terms are used as follows:

- **Match exposure** is observed all-competition club playing time.
- **Incidence rate** is the number of proxy events divided by match exposure.
- **Reported absence duration** is a public-data proxy for time-loss severity.
- **Injury burden** is not estimated because confirmed time-loss days and total
  football exposure are unavailable.
- **Same-day reported event** is the reference outcome: Transfermarkt's
  `start`/`startDate` spell-start field recorded on an appearance date. It is
  not a publication timestamp, verified onset, or proof that the event occurred
  during that match.
- **Lag-1 reported event** is a next-day public episode start attributed to the
  preceding appearance as a timing sensitivity. It may be a delayed report,
  training event, or unrelated event.
- **Combined match-associated proxy** means same-day plus lag-1 reports. It is
  a sensitivity outcome, not the reference outcome or a clinically confirmed
  match injury.
- **Per-appearance probability** is the reference denominator because exact
  event time is unavailable. Observed-minute and fixed-90-minute models are
  opposing sensitivity bounds, not corrections.
- **Exposure-standardised onset rate** is used only for descriptive plots that
  divide all reported starts by observed match hours. It is not called match
  injury incidence because the numerator includes non-match-associated starts.
- **Prior-injury-history stratum** is the publication-facing description of
  the internal data labels. The manuscript uses lower, intermediate, and
  higher prior-injury-history throughout to avoid stigmatizing or clinically
  misleading terminology.

## Current Status

The current codebase has been corrected for the main reasoning issues found in
the previous implementation:

- Stable player identity is now `tm_player_id`, one row per Transfermarkt
  player. The legacy column `fbref_player_id` is retained only as a backwards
  compatibility alias and equals `tm_player_id`.
- Prior-injury-history status is dynamic and prior-only at the player-day level. A player can
  move from `low_exposure` into `tough`, `regular`, or `fragile` only after
  enough prior observed exposure has accumulated. These internal labels mean
  lower, intermediate, and higher prior-injury-history strata; they are not
  diagnoses or stable player phenotypes.
- All-competition rolling load uses player IDs rather than name joins, is
  restricted to appearances for EPL club-seasons, and uses shifted rolling
  windows so today's minutes are not included in prior 7-day burden.
- Daily and per-minute analyses use an availability-adjusted risk set. Shared
  canonical episodes collapse overlapping/touching reports and end on the day
  before an observed return, so an observed appearance is never unavailable.
- The all-competition panel spans every valid EPL-club-season appearance rather
  than stopping at each player's first and last Premier League appearance.
- Prior history includes strictly earlier episodes and club exposure before a
  player's first eligible risk-set day; 2,266 pre-entry episodes and 2,875,586
  pre-entry club minutes are now represented.
- Sparse 5-minute dummy-bin GLMs have been removed. Five-minute burden tables
  are descriptive only because extreme bins can be separated or too sparse for
  reliable inference.
- Forty-five-minute GLMs keep every bin in crude summaries but exclude sparse
  or separated bins from model fitting. Estimability tables are written beside
  the model outputs.
- Daily logistic models now fit the full eligible panel rather than a sampled
  case-control subset. Model-audit CSVs report full-risk rows, fitted rows,
  events, and prediction rows, so no intercept recalibration is needed.
- The match-proxy Poisson spline model uses a 4 df spline. The previous 5 df
  interacted spline was too flexible for the rare event counts and produced
  singular or undefined clustered standard errors. The current run also writes
  local support counts around selected prediction points, denominator/link
  checks, clean-comparator and calendar restrictions, and selection-control
  sensitivities with and without continuous prior-history controls.
- Formal effect-modification output now separates three questions: the global
  spline-by-history interaction, the literal 0-to-180-minute congestion
  comparison, and the incremental 90-to-180-minute comparison. Every planned
  outcome, history-definition, calendar restriction, comparator restriction,
  denominator/link, distribution, and control specification is
  retained in one long table with Holm and Benjamini-Hochberg adjustments.
  This table is an anti-p-hunting audit, not a mechanism for selecting whichever
  specification gives the smallest p-value.
- The primary spline model no longer adjusts for exposure-derived surplus
  terms such as excess minutes beyond `90 x match-days`; those are retained as
  an overadjustment sensitivity. This avoids controlling away part of the
  short-term exposure pattern being studied.
- The primary selected prediction grid is now 0, 90, and 180 previous-7-day
  minutes. The 220-minute tail is generated only in
  `poisson_spline_diagnostic_support_matchproxy.csv`; in the current data there
  are no local proxy events within +/- 15 minutes of 220 in either modelled
  stratum.
- A threshold-transport sensitivity derives the canonical prior-history
  thresholds from the 2017--2019 seasons only and applies those fixed cut points
  to 2020--2024 rows. This checks whether the broad match-minute result depends
  on calibrating thresholds with the full cohort.
- The pre-specified absolute prior-history rule now has dedicated prediction,
  support, ratio, and formal-test CSVs as well as its row in the full
  sensitivity table.
- A compact publication contrast table now combines the main sensitivity rows
  and denominator/link rows, reporting the higher-to-intermediate contrast at
  0, 90, and 180 previous-7-day minutes for every retained specification.
- The match-proxy spline pipeline now also writes reported-absence `>=28 days`
  and muscle/tendon-restricted outcome sensitivities, frequency-only and
  stricter type-discordant musculoskeletal non-muscle history crosses,
  a qualitative negative-control magnitude comparison at the 0- and
  180-minute anchors, a direct same-row negative-control comparison, clean
  mutually exclusive binary and continuous type-history checks, type-frequency
  distribution translations, an exact 0-minute selection audit, and a refit
  excluding appearances within 14 days of a recorded return,
  B-spline/natural-cubic spline-shape sensitivities, player-correlated GEE,
  within-player switcher fixed-effect, and within/between recurrent-event
  sensitivity summaries. It also writes selection-band audits with
  short-appearance, recent-return, and available lineup starter/substitute
  proxies.
- The formal type-history file now reports model-relevant variance-inflation
  factors for both continuous frequency terms. In the recency-adjusted model,
  VIFs are 2.76 for muscle/tendon frequency and 2.63 for the type-discordant
  comparator. Prior-report-only frequency--log-recency correlations are
  similarly moderate at -0.361 and -0.356. The muscle/tendon high-step log
  standard error changes from 0.1113 to 0.1110, so the fall from 1.84 to 1.31
  reflects movement of the estimate rather than reduced precision.
- After matched recency, mutually exclusive continuous muscle/tendon and
  comparator IRRs are 1.041 and 1.040 per additional prior report per 10,000
  previous club minutes; their direct ratio is 1.00 (95% CI 0.95--1.06).
  The binary tail contrast and continuous slope therefore disagree. The
  pipeline records both because the pattern may reflect either concentration
  in the high-frequency tail or an artefact of data-derived categorisation;
  the similar comparator slope leaves shared reporting or player-profile bias
  plausible.
- A proxy-event type audit now reports public-text category shares among the
  1,845 same-day/lag-1 proxy events. Unknown or other/unspecified descriptions
  account for 453 events (24.6%), defining the limit of tissue-type checks.
- Recovery-interval outputs now include approximate Poisson count-rate
  intervals plus formal shorter-recovery trend tests adjusted for
  prior-injury-history stratum, with matched observed-minute, fixed-90-minute,
  per-match logit, and per-match complementary-log-log recovery models. A
  publication summary combines the all-cause, muscle/tendon, and reported
  absence `>=28 days` recovery tests with multiplicity-adjusted p-values.
- Publication-facing result tables add neutral lower/intermediate/higher
  prior-injury-history labels beside the internal `tough`, `regular`, and
  `fragile` labels.
- `src/24_prepare_public_deposit.py` creates a sanitized `public_deposit/`
  export where legacy internal terms are replaced in CSV columns, string
  values, and filenames before external archiving.
- Temporal-stability models refit the primary per-minute spline inside
  football-season blocks (`2017-2019`, `2020-2021`, `2022-2024`). These blocks
  keep the canonical day-level prior-history labels, so a player can enter a
  later block already labelled `regular`, `tough`, or `fragile`; labels are not
  reset or recalibrated inside the block.
- Clinical-surveillance bridge outputs translate the public-data
  match-associated proxy signal into events per 1,000 match hours, events per
  1,000 appearances, and reported absence-duration strata. These are
  descriptive translations for clinical readability, not standardised club
  medical surveillance rates.
- Prior-injury-duration outputs attach the most recent completed prior injury
  episode to each match row and estimate subsequent match-associated proxy
  incidence by prior-duration category. A frequency-only grouping is included
  so duration is not built into the group definition being tested.
- `src/33_matchproxy_current_data_extensions.py` adds seven named, post-primary
  robustness and measurement audits to the frozen intermediate/higher-history
  match-proxy model: observed lineup/return selection refits; report-type
  completeness and gated inverse-probability weighting; public absence-date
  summaries; joint burden/recovery support; within-player-season conditional
  logistic models; player-bootstrap curve stability and two-way clustering;
  and current-competition sensitivities. These checks do not change the
  primary estimand. An unstable reporting-weight gate prevents its weighted
  result from being used as a correction.
- `src/34_jsams_referee_analysis.py` builds the predecessor JSAMS audit layer and
  fits all nine same-day/lag-1/combined by per-appearance/observed-minute/
  fixed-90 models symmetrically. It also standardises absolute predictions over
  observed calendar phases; compares five exposure forms; restricts same-day
  outcomes to reported absence of at least 28 days and muscle/tendon text;
  creates conventional cohort descriptors; estimates overall, role-specific,
  and lineup-standardised minute gaps with 1,000-player bootstraps; audits
  lineup coverage and restricts selection analyses to complete-coverage
  seasons; refits each changing-versus-fixed composition contrast inside a
  paired 1,000-player bootstrap; fits conditional logistic models within player
  and player-season; checks spell-start timing against other eligible days;
  compares included and excluded players; refits 450-, 900-, and 1,800-minute
  eligibility cohorts, complete seasons, pandemic-excluded seasons, and
  club-plus-senior-national exposure; applies measured-context and
  two-way-cluster checks; and writes the complete test and tier registries.
- `src/36_jsams_second_referee_analysis.py` supersedes the script-34 spline as
  the current reference model. It fits an additive continuous-history model,
  all 63 exposure/timing/denominator combinations, a simultaneous spline band,
  three fixed temporal blocks, conditional target-population checks, restricted
  measured-selection weighting, and the independent-source outcome audit.
  It writes a 706-row hypothesis register and the controlling tier registry.
  It also writes seven tables that qualify the reference estimate rather than
  support it: the multiverse distribution, the exposure-metric correlations,
  the measured-confounding and clustering refits, the absolute-risk contrast
  and its exposure support, the included-versus-excluded selection population,
  and the denominator-contrast metadata.
- `src/35_plot_jsams_revision.py` generates the four main manuscript figures
  from the script-34 and script-36 CSVs. It does not refit models.
- Obsolete Python scripts and stale result CSVs have been removed.
- Every retained Python file has a corresponding test file. The final suite
  passes 289 tests with 100.00% statement and branch coverage.

Secondary effect-modification interpretation: the earlier categorical model
uses 72,445 intermediate/higher-history match rows and 1,592 combined proxy
events. Its global spline-by-history test is unsupported (Wald chi-square(4) =
2.39, `p = 0.664`). From 90 to 180 previous-7-day minutes, the
intermediate-history IRR is 1.06 (95% CI 0.87--1.29), the higher-history IRR is
1.06 (0.77--1.45), and the ratio of changes is 1.00 (0.70--1.42; `p = 0.992`).
The current all-strata, same-day, per-appearance reference model uses one
linear seven-day exposure term and additive continuous history. The unsupported
exposure-by-history interaction (`p = 0.153` after Holm correction) is
secondary; its absence does not prove equivalence.

Across 154 registered post-data exposure-response tests, no Holm- or
Benjamini-Hochberg-adjusted contrast is significant. The family contains every
planned outcome, history, denominator, calendar, comparator, and recurrent-
event specification. This is a Tier 5 null and remains a secondary result.

Denominator checks are now central measurement results. In the 88,573-row
reference risk set, same-day-report appearances average 50.1 observed minutes
versus 71.1 in other appearances; the player-bootstrap difference is -21.0
minutes (95% CI -23.1 to -19.0; 1,000 replicates). For the seven-day linear
model, the per-appearance OR is 1.27 (1.11--1.44), the observed-minute IRR is
1.09 (0.95--1.25), and the fixed-90 IRR is 1.27. Across all timings and metrics,
0/21 observed-minute models survive correction versus 12/21 per-appearance
and 12/21 fixed-90 models. Fixed 90 is a constant-denominator count model, not
an independent reconstruction of time at risk.

The minute gap is not uniform by recorded lineup role. Among lineup-known rows,
in the six complete-lineup seasons, same-day reports occur on starter
appearances that are 31.2 minutes shorter (95% CI 28.6--33.9 shorter) than
non-event starter appearances; the substitute difference is 2.2 minutes
(-3.0 to 7.5). Standardising over the pooled lineup mix gives 24.7 fewer
minutes (22.2--27.1 fewer). These are distributional checks,
not estimates of event time or minutes lost.

Role-specific same-day per-appearance refits are also disclosed. The starter
0-to-180 OR is 1.48 (95% CI 0.99--2.22; 364 events; global `p = 0.078`); the
substitute OR is 0.018 (0.0003--0.981; 26 events; global `p = 0.125`). The
substitute fit is sparse and unstable, so it is not interpreted as protection
or as a direct role difference.

The recorded-selection audit first measures lineup completeness. Role coverage
is 100% from 2017-18 through 2022-23, 22.5% in 2023-24, and 0% in 2024-25.
Because the final season has no positive chance of observed role, weighting the
lineup-complete rows back to the full cohort is not identifiable. The selection
analysis is therefore restricted to the six complete seasons: 64,387
appearances, 987 players, and 360 same-day records.

At median prior history, the modelled 0-to-180 increase is 2.27 records per
1,000 appearances (95% CI -0.50--4.88) when recorded lineup/return composition
changes with exposure and 1.51 (-1.07--3.82) when that composition is fixed.
Every one of 1,000 player-cluster draws refits the model and recomputes both
standardisations; 999 draws are estimable. Their paired difference is 0.75
(0.36--1.08), with raw `p = 0.046` and Holm-adjusted `p = 0.828` across 18
selection contrasts. Measured selection is therefore a Tier 5 sensitivity, not
a headline result. Unrecorded health, medical, training, and tactical selection
remain uncontrolled.

Reviewer-requested conditional logistic models remove stable player or
player-season intercepts. The 0-to-180 spline OR is 1.64 (1.17--2.30) within
362 discordant players and 2.09 (1.45--2.99) within 505 discordant
player-seasons. These models do not control time-varying symptoms, fitness,
training, or medical decisions.

The same-day per-appearance 0-to-180 OR is 1.48--1.61 across five exposure
forms. Post hoc quality restrictions give OR 1.63 (1.07--2.47) for 337 reports
with at least 28 reported absence days and 2.13 (1.33--3.41) for 266
muscle/tendon reports. The severe-report global spline test is `p = 0.139`, so
its positive anchor contrast is not treated as evidence for a complete severe-
event curve. Eligibility, complete-season, pandemic-exclusion, measured-context,
and club-plus-senior-national checks retain the same positive anchor direction;
they are internal sensitivities, not independent discoveries or external
validation.

Outcome-subset checks address public-report ascertainment. At zero recent
minutes, higher versus intermediate history gives IRRs of 1.81 (1.37--2.40)
for reports lasting at least 28 days and 1.77 (1.31--2.38) for muscle/tendon
reports. Frequency-only history gives 1.94 (1.45--2.61) and 2.12 (1.57--2.88).
These are established prior-history level differences, not exposure effects.

The mutually exclusive binary type audit uses third-quartile thresholds of
2.81 muscle/tendon and 2.18 joint/ligament or bone/fracture reports per 10,000
previous club minutes. High muscle/tendon history gives an IRR of 2.55
(1.90--3.42), the control gives 0.91 (0.64--1.30), and their direct ratio is
2.80 (1.64--4.76; Holm p = 0.009). With matched recency, that ratio is 1.56
(0.90--2.70; Holm p = 1.000). Continuous frequency gives a direct per-unit
ratio of 1.05 (1.00--1.11; raw p = 0.059) and 1.00 (0.95--1.06) after recency.

The formal stacked model quantifies attenuation between matched specifications.
The muscle/tendon high-frequency step falls from 1.84 (1.48--2.29) to 1.31
(1.05--1.63); adjusted/unadjusted ratio 0.71 (0.65--0.78; Holm p < 0.001).
The direct muscle/tendon-versus-control step falls from 1.84 (1.25--2.70) to
1.27 (0.88--1.85), ratio 0.69 (0.60--0.80; Holm p < 0.001). Excluding 6,740
rows and 103 events within 14 days of a recorded return leaves the main
attenuation at 0.74 (0.68--0.81; Holm p < 0.001), while the binary direct ratio
no longer survives Holm correction. These are specification contrasts, not
causal mediation, recurrence, or clinical thresholds.
Recurrent-event checks remain cautionary. The GEE 180-minute history contrast is
1.20 (0.80--1.81), while the switcher fixed-effect estimate is 0.78
(0.51--1.18). The between-player IRR is 2.16 (1.85--2.52), but the within-player
deviation is 0.52 (0.42--0.63). Switcher incidence is already 35.5 per 1,000
hours before versus 20.4 after transition, so index-event timing prevents a
causal interpretation of the within-player reversal.

Spline-shape sensitivity argues against an early workload threshold. The
15--45-minute peak appears in 3/8 intermediate-history and 0/8 higher-history
specifications. Substitute shares are 32.8% versus 7.2% in the early and
90-minute bands for intermediate history, and 41.6% versus 9.6% for higher
history. Recent-return shares are 10.9% versus 4.8% and 15.9% versus 7.1%.
Player bootstrap refits put the early band at the global maximum in 51.4%
(48.3--54.5) and 29.6% (26.9--32.5) of samples. The curve therefore reflects
risk-set composition as well as recent burden.

Recovery-interval, out-of-time threshold, case-crossover, joint burden/recovery,
competition-context, and two-way-cluster checks are retained as secondary
audits. None establishes history-specific effect modification after family
correction. Temporal 180-minute history ratios are 1.97 (1.22--3.17), 1.28
(0.60--2.73), and 1.12 (0.60--2.11) for 2017--2019, 2020--2021, and 2022--2024;
the wide, overlapping intervals make these internal stability checks rather
than prospective validation.

Clinical bridge interpretation: the proxy rate is 17.6 (16.8--18.4) events per
1,000 all-competition match hours. Rates are 12.4, 16.6, and 23.6 in lower-,
intermediate-, and higher-history rows. This reproduces known prior-injury
susceptibility; it does not supply training exposure, exact mechanisms,
diagnoses, recurrence, confirmed time loss, or clinical injury burden.

Prior-duration interpretation remains descriptive. After a two-month-to-one-
year versus under-one-week reported absence, rates are 14.1 (11.3--17.5) versus
17.0 (14.7--19.6) in intermediate-history rows and 20.0 (16.7--23.9) versus
27.4 (23.0--32.7) in higher-history rows. Injury type, rehabilitation, and
return-to-sport selection may explain the pattern; longer injury is not treated
as protective.

## Releases And Citation

The software and derived dataset have separate Zenodo records:

- Software concept DOI: `10.5281/zenodo.17835593`
- Software v1.0.0 DOI: `10.5281/zenodo.17835594`
- Software v2.0.0 DOI: `10.5281/zenodo.21498460`
- Software v3.0.0 DOI: `10.5281/zenodo.21673177`
- Dataset concept DOI: `10.5281/zenodo.17835137`
- Dataset v1.0.0 DOI: `10.5281/zenodo.17835138`
- Dataset v2.0.0 DOI: `10.5281/zenodo.21498770`
- Dataset v3.0.0 DOI: `10.5281/zenodo.21673210`

Version 3.0.0 was published on 29 July 2026. It preserves the v1 and v2 records
and archives the pipeline, manuscript, tests, figures, and sanitized derived
outputs as they stood at that release. Compared with v2.0.0, it adds formal
denominator and link checks, severe-reported-absence and muscle/tendon outcome restrictions,
mutually adjusted and mutually exclusive injury-type checks, 154-test
effect-modification and 198-test formal contrast audits, selection and
recent-return diagnostics, recurrent-event decomposition, manuscript numeric
reconciliation, and regenerated figures. The public dataset archive contains
the 184 sanitized derived-output and figure payloads, its sanitization
manifest, release metadata, and a complete SHA-256 payload manifest.

Use the concept DOIs in badges or links that should always resolve to the latest
published version. The current checkout contains later v4 acquisition, scripts
34--36, revised figures, and a newer manuscript; the v3.0.0 version-specific
DOIs therefore identify a historical release and do not exactly reproduce this
checkout. A future public release must create matching new software and
derived-output versions before their exact DOIs are cited for the current
paper. The v2.0.0 publication remains documented in
[`docs/zenodo_v2_release.md`](docs/zenodo_v2_release.md); the v3.0.0 archives,
checksums, and publication checks are documented in
[`docs/zenodo_v3_release.md`](docs/zenodo_v3_release.md). The arXiv source-package
contents and submission checks are documented in
[`docs/arxiv_preprint_release.md`](docs/arxiv_preprint_release.md). The
canonical current/historical/local-only map is
[`docs/repository_inventory.md`](docs/repository_inventory.md).

## Repository Layout

```text
epl_congestion/
|- README.md
|- CITATION.cff
|- .zenodo.json
|- config.py
|- requirements.txt
|- pytest.ini
|- .coveragerc
|- docs/
|  |- zenodo_dataset_v2_metadata.json
|  |- zenodo_dataset_v2_manifest.txt
|  |- zenodo_v2_release.md
|  |- zenodo_dataset_v3_metadata.json
|  |- zenodo_dataset_v3_manifest.txt
|  |- zenodo_v3_release.md
|  |- arxiv_preprint_release.md
|  |- public_data_v4_protocol.md
|  |- public_data_v4_data_dictionary.md
|  |- public_data_v4_scientific_audit.md
|  |- repository_inventory.md
|  |- strobe_siis_checklist.md
|  `- versioned Zenodo v2/v3 metadata, manifests, and release notes
|- src/
|  |- 00_list_result_columns.py
|  |- 01_fetch_fbref.py
|  |- 02_build_player_index.py
|  |- 03a_build_player_mapping_tm.py
|  |- 04_fetch_injuries_transfermarkt.py
|  |- 05_clean_injuries.py
|  |- 06_build_player_day_panel.py
|  |- 07_build_all_comp_minutes.py
|  |- 08_baseline_hazard_all_comp.py
|  |- 09_build_fragility_groups.py
|  |- 10_hazard_by_fragility_45min.py
|  |- 11_hazard_5min_all_comp_and_fragility.py
|  |- 12_fft_injury_rates_fragility.py
|  |- 13_max_daily_load_features.py
|  |- 14_hazard_by_fragility_45min_weekly.py
|  |- 16_build_match_proxy_events.py
|  |- 17_match_proxy_perminute_descriptives_5min.py
|  |- 18_match_proxy_poisson_splines_perminute.py
|  |- 19_plot_everything_for_paper.py
|  |- 20_model_diagnostics.py
|  |- 21_panel_restriction_counts.py
|  |- 22_clinical_surveillance_bridge.py
|  |- 23_prior_injury_duration_next_risk.py
|  |- 24_prepare_public_deposit.py
|  |- 25_public_data_v4.py
|  |- 25b_acquire_public_sources.py
|  |- 26_build_public_match_timeline.py
|  |- 27_public_data_v4_audits.py
|  |- 28_public_data_v4_model_comparison.py
|  |- 29_public_data_v4_scientific_audit.py
|  |- 30_national_status_analysis.py
|  |- 31_public_data_v4_quality_registry.py
|  |- 32_plot_v4_paper_figures.py
|  |- 33_matchproxy_current_data_extensions.py
|  |- 34_jsams_referee_analysis.py
|  |- 35_plot_jsams_revision.py
|  |- 36_jsams_second_referee_analysis.py
|  |- pipeline_io.py
|  |- public_data_sources.py
|  `- v4_statistics.py
|- tests/
|  `- test files matching each retained Python module
|- data/
|  |- raw/
|  |- manual/
|  `- processed/
|     |- public_data_v4/
|     `- results/
|- external_data/
|  `- transfermarkt/
`- manuscript/
   |- manuscript.tex
   |- manuscript.pdf
   |- supplement.tex
   |- title_page.tex
   `- figures/
```

`data/`, `external_data/`, `output/`, `public_deposit/`, local dependency
folders, test artifacts, and auxiliary manuscript builds are ignored by Git
because they are large or generated. The clean `manuscript/manuscript.pdf` and
the vetted manuscript figures are versioned so the paper remains inspectable
without rerunning the local data pipeline.

## Environment

Create a Python environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Git Bash or WSL:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Current dependencies:

```text
pandas
python-dateutil
tqdm
soccerdata
transfermarkt-wrapper
requests
python-dotenv
statsmodels
scipy
patsy
numpy
matplotlib
pillow
pytest
pytest-cov
```

`soccerdata` remains in `requirements.txt` for historical compatibility, but
the active production path reads local Transfermarkt CSVs and does not call
FBref through `soccerdata`.

If you are using the Codex-local dependency folder that was created during this
cleanup, run tests with:

```powershell
$env:PYTHONPATH = (Resolve-Path '.codex_deps').Path
python -m pytest --cov --cov-branch
```

In a normal virtual environment, `PYTHONPATH` is not needed:

```powershell
.\.venv312\Scripts\python.exe -m pytest --cov=. --cov-branch
```

Expected coverage gate:

```text
256 passed
100% statement coverage
100% branch coverage
```

## Required Inputs

The pipeline expects local Transfermarkt dataset dumps at:

```text
external_data/transfermarkt/appearances.csv
external_data/transfermarkt/club_games.csv
external_data/transfermarkt/clubs.csv
external_data/transfermarkt/competitions.csv
external_data/transfermarkt/game_events.csv
external_data/transfermarkt/game_lineups.csv
external_data/transfermarkt/games.csv
external_data/transfermarkt/player_valuations.csv
external_data/transfermarkt/players.csv
external_data/transfermarkt/transfers.csv
```

The active scripts use:

- `games.csv` for Premier League schedules, dates, clubs, scores, and seasons.
- `appearances.csv` for EPL and all-competition player minutes.
  All-competition minutes are restricted to clubs that were EPL clubs in the
  corresponding season.
- `clubs.csv` and `players.csv` for stable identifier and display-name context.
- `competitions.csv` for current-match competition grouping.
- `game_lineups.csv` for observed starter/substitute coverage and role-specific
  sensitivity analyses.
- `transfers.csv` for the bounded player-club opportunity reconstruction used
  by the v4 selection audit.

`club_games.csv`, `game_events.csv`, and `player_valuations.csv` are retained as
part of the dated provider snapshot but are not read by the current production
path. The README lists them so a local snapshot can be checked as a complete
provider bundle rather than silently mixing retrieval dates.

Injury reports are stored separately:

```text
data/raw/tm_injuries_raw.csv
data/processed/tm_injuries_clean.csv
```

`src/04_fetch_injuries_transfermarkt.py` can refetch injuries from
Transfermarkt through `transfermarkt-wrapper` / `tmkt`. Because that is a live
network operation, downstream reruns can instead reuse an existing
`data/raw/tm_injuries_raw.csv` and start at `src/05_clean_injuries.py`.

Three inputs are not derived from the provider snapshot and cannot be rebuilt:

```text
data/manual/independent_same_day_event_audit.csv
data/manual/independent_non_event_audit.csv
data/manual/per_minute_denominator_scoping.csv
```

The first records one author's manual adjudication of 30 exposure-blinded
sampled reports, with the independent source URL behind each verdict, and is
the only evidence for the outcome-attribution result. The second applies the
same protocol to the 30 report-free appearances and returns `unresolved` on
every one; it is the record that the search was run and settled nothing, which
is why the paper reports outcome sensitivity as unknown. The third is the
scoping search for published studies using minute denominators on public data,
reported as a floor rather than a systematic count.

These are the only files under `data/` kept in version control. `.gitignore`
negates the whole `data/manual/` directory rather than listing them by name, so
the next hand-built file is versioned by default; an allow-list previously
dropped two of these three silently. Do not delete or regenerate them;
`src/36_jsams_second_referee_analysis.py` validates the audits against their
blinded queues before summarising them.

## Data Model

### Stable Player Identity

`src/02_build_player_index.py` writes:

```text
data/processed/player_index_fbref.csv
```

Despite the filename and legacy column name, this file is now a stable
Transfermarkt player index:

```text
tm_player_id      stable Transfermarkt player ID
fbref_player_id   compatibility alias equal to tm_player_id
player            display name from appearances
teams             observed club names
first_season      first EPL season in local data
last_season       last EPL season in local data
```

Current regenerated count:

```text
stable players: 1,558
```

### Base EPL Player-Day Panel

`src/06_build_player_day_panel.py` writes:

```text
data/processed/player_day_panel.csv
```

Each row is one stable player on one calendar day between that player's first
and last observed EPL appearance. Important columns include:

```text
fbref_player_id
tm_player_id
date
minutes_played
games_played
games_last_7d
minutes_last_7d
injury_spell_id        legacy column name containing the canonical episode ID
injury_desc
n_injury_spells        legacy column name containing the episode count
injury_event
```

`injury_event` is `1` when a canonical injury episode starts on that
player-day. The shared episode builder collapses duplicate, touching, and
overlapping reports; an observed return truncates the preceding episode; and a
new report on an observed appearance day starts a new episode. This preserves
one player-day row while preventing overlapping reports from being counted as
independent starts.

Current regenerated count:

```text
rows:        1,330,896
players:    1,558
match-days: 84,191
events:     4,541
date span:  2017-08-11 to 2025-04-07
```

### All-Competition Load Panel

`src/07_build_all_comp_minutes.py` computes prior all-competition load from the
local Transfermarkt `appearances.csv`. In this project, all-competition means
recorded club appearances for clubs that were EPL clubs in the corresponding
season. This keeps EPL, domestic cup, League Cup, Community Shield, European,
and other senior club appearances by EPL clubs while excluding non-EPL-club
career appearances and national-team matches. Internally it creates a complete
player-date grid before rolling. The final panel extends the base EPL-date
panel whenever an eligible player has a valid appearance for an EPL
club-season outside that narrower span:

```text
data/processed/player_day_panel_all_comp.csv
rows:                         1,398,297
players:                      1,558
rows added beyond EPL span:      67,401
date span: 2017-07-27 to 2025-04-07
```

Current all-competition match rows inside the EPL player-day grid:

```text
all-competition player-match days: 101,042
all-competition minutes:            7,022,189
non-GB1 domestic league rows:      0
```

The reconciliation audit confirms that all 101,042 source EPL-club-season
appearance days are represented. It adds 898 match rows beyond the old
first/last-EPL-appearance span, leaves zero observed appearances marked
unavailable, reconciles 12,440 cleaned reports into 11,993 episodes, absorbs
395 reports into continuous episodes, and truncates 704 episodes at an
observed return.

Important added columns include:

```text
all_minutes_last_7d
non_epl_minutes_last_7d
minutes_yesterday
minutes_last_match
max_daily_minutes_last_7d
any_day_last7_over_90
any_day_last7_full_match
excess_minutes_last7d
any_extra_time_last7d
day_index
week_phase_sin
week_phase_cos
halfweek_phase_sin
halfweek_phase_cos
```

The rolling windows are shifted by one day, so `all_minutes_last_7d` means the
previous seven calendar days and excludes the current row's minutes.

### Fragility Labels

`src/09_build_fragility_groups.py` writes two files:

```text
data/processed/player_day_fragility.csv
data/processed/player_fragility_groups.csv
```

`player_day_fragility.csv` is canonical for analysis. It contains day-level
prior-history labels and is merged by `tm_player_id` plus `date`.

`player_fragility_groups.csv` is a latest-snapshot summary for reporting and
sanity checks. It should not be used to assign historical day-level labels.

Day-level fragility inputs are strictly prior to the current date:

```text
prior_minutes_played
prior_n_spells                 legacy column name for prior episodes
prior_total_days_injured
prior_max_spell_duration_days  legacy column name for maximum episode duration
prior_years_at_risk
prior_injuries_per_year
prior_injuries_per_10000min
```

The label rule is:

- `low_exposure`: `prior_minutes_played < 900`.
- `tough`: `prior_minutes_played >= 900`, `prior_n_spells <= 1`,
  prior injury frequency <= Q1 threshold, and prior maximum episode duration <=
  Q1 threshold.
- `fragile`: `prior_minutes_played >= 900`, `prior_n_spells >= 2`, and prior
  injury frequency >= Q3 threshold or prior maximum episode duration >= Q3
  threshold.
- `regular`: eligible prior-history rows that are neither `tough` nor
  `fragile`.

The Q1/Q3 thresholds are estimated once from each eligible player's latest
prior-history snapshot, then applied to every player-day using only that day's
prior information. Current frequency thresholds are 2.92 and 11.01 prior
episodes per 10,000 club minutes; current maximum-duration thresholds are 24
and 103 reported days. Strictly earlier records are retained even when they
precede a player's first eligible risk-set day: the current build includes
2,266 pre-entry episodes for 641 players and 2,875,586 pre-entry club minutes.

Current regenerated day-level label counts:

```text
regular       562,330
fragile       404,873
low_exposure 247,116
tough         183,978
```

Current latest-snapshot player counts:

```text
regular      570
fragile      477
low_exposure 350
tough        161
```

Most stratified analyses use the dynamic risk set:

```text
fragility_group in {"tough", "regular", "fragile"}
```

That excludes low-exposure days, not necessarily entire players forever.

### Match-Proxy Outcomes

`src/16_build_match_proxy_events.py` adds per-minute match-proxy outcomes to
`player_day_panel_all_comp.csv`.

The raw injury date is a Transfermarkt report start date, not an observed injury
minute. The proxy logic is:

- If an injury starts on an all-competition match day with
  `all_minutes_played > 0`, classify it as `match_same_day`.
- If an injury starts on a non-match day and the player played the previous
  day, classify the onset day as `match_lag1_recorded_next_day` and
  back-attribute the match-proxy event to the previous match row.
- Otherwise classify the onset as `training_or_other`.

Added columns:

```text
match_injury_same_day
match_injury_lag1_recorded_next_day
injury_event_matchproxy
matchproxy_source
injury_context
injury_minute_proxy
injury_event_trainingproxy
```

Current onset-day context counts within the dynamic fragility risk set:

```text
training_or_other                 2,268
match_lag1_recorded_next_day      1,278
match_same_day                      576
```

Current onset-day match-context counts by fragility group:

```text
fragile: same_day 219, lag1 428
regular: same_day 281, lag1 669
tough:   same_day 76,  lag1 181
```

The per-minute model counts events on match rows after lag-1 back-attribution,
so it is not identical to the onset-day context table:

```text
dynamic risk-set players:       1,208
dynamic risk-set player-days:   1,018,202
dynamic risk-set match-days:    88,573
match-proxy match-row events:   1,845
```

For the main Poisson spline model, the lower prior-injury-history stratum is
excluded by default because its match-proxy event count is sparse. The
intermediate plus higher match-row subset is:

```text
players:    1,057
match-days: 72,445
events:     1,592
```

## Pipeline Order

Run commands from the repository root.

### Core Data Build

```powershell
python src/01_fetch_fbref.py
python src/02_build_player_index.py
python src/03a_build_player_mapping_tm.py
python src/04_fetch_injuries_transfermarkt.py
python src/05_clean_injuries.py
python src/06_build_player_day_panel.py
python src/07_build_all_comp_minutes.py
python src/09_build_fragility_groups.py
python src/13_max_daily_load_features.py
python src/16_build_match_proxy_events.py
```

If you do not want to refetch live Transfermarkt injury pages, skip
`src/04_fetch_injuries_transfermarkt.py` only when
`data/raw/tm_injuries_raw.csv` already exists, then continue with
`src/05_clean_injuries.py`.

### Core And Historical Analyses

```powershell
python src/08_baseline_hazard_all_comp.py
python src/10_hazard_by_fragility_45min.py
python src/11_hazard_5min_all_comp_and_fragility.py
python src/12_fft_injury_rates_fragility.py
python src/14_hazard_by_fragility_45min_weekly.py
python src/17_match_proxy_perminute_descriptives_5min.py
python src/18_match_proxy_poisson_splines_perminute.py
python src/20_model_diagnostics.py
python src/21_panel_restriction_counts.py
python python src/22_clinical_surveillance_bridge.py
python src/23_prior_injury_duration_next_risk.py
```

## Public-data v4 Extension

The frozen analysis measures recorded appearances for clubs that competed in
the EPL in the relevant season.  It does not measure senior national-team,
youth, Olympic, or geographic travel exposure.  The v4
extension is a separate, hash-recorded public-data build designed to test
whether that missing match exposure changes the club-only conclusions.  Its
protocol is fixed in [`docs/public_data_v4_protocol.md`](docs/public_data_v4_protocol.md)
with a documented post-audit comparator amendment. The output fields and audit meanings are
defined in [`docs/public_data_v4_data_dictionary.md`](docs/public_data_v4_data_dictionary.md).

The direct v4 comparison retains the frozen all-club exposure and adds observed
competitive senior-national minutes.  This is necessary because the frozen
all-competition variable includes recorded club friendlies whereas the
refreshed strict competitive-club reconstruction excludes them.  The match-
proxy outcome and prior-injury-history variables are unchanged, so this direct
comparison changes national exposure measurement only.  Refreshed club-only,
senior-friendly, U21/U23/Olympic, and club-friendly scopes remain separate
reconstruction or secondary sensitivity scopes.

Raw v4 data are intentionally stored in a new ignored directory:

```text
data/raw/public_data_v4/
|- transfermarkt_datasets_YYYYMMDD.zip
|- transfermarkt_datasets_YYYYMMDD/
|  `- snapshot_manifest.json
`- national_performance_cache/
```

The manifest records the retrieval time, upstream commit, source URLs, file
sizes, SHA-256 hashes and CSV schemas.  The original
`external_data/transfermarkt/` input directory is never overwritten.  Cached
national histories use a publicly served but undocumented Transfermarkt player
performance endpoint; they are not an official API and may change.  Each cache
record and normalised appearance retains its retrieval URL and cache path.

Run v4 only after the frozen core pipeline has produced
`player_day_panel.csv` and `player_match_panel_all_comp.csv`:

```powershell
python src/25_public_data_v4.py
python src/25b_acquire_public_sources.py
python src/26_build_public_match_timeline.py
python src/27_public_data_v4_audits.py
python src/28_public_data_v4_model_comparison.py
python src/29_public_data_v4_scientific_audit.py
python src/30_national_status_analysis.py
```

### Current Manuscript Layer

Run this layer only after the core script-18 outputs and v4 scripts 25--30
exist. Script 31 incorporates script-33 extension outputs, script 34 consumes
script-33 and v4 context tables, and script 36 consumes scripts 27 and 34.
Generate schemas and the public deposit last.

Both plotting scripts are required for a complete submission: script 35 writes
six figures --- the two the main paper displays (`J1`, `J2`) and four the
Supplement displays (`J3`--`J6`) --- and script 32 writes two more Supplement
figures (`I1`, `I2`). Skipping script 32 leaves `supplement.tex` without two of
its displays.

```powershell
python src/33_matchproxy_current_data_extensions.py
python src/31_public_data_v4_quality_registry.py
python src/34_jsams_referee_analysis.py
python src/36_jsams_second_referee_analysis.py
python src/37_denominator_gradient.py
python src/35_plot_jsams_revision.py
python src/00_list_result_columns.py
python src/24_prepare_public_deposit.py
```

The v4 scripts write the following audit artifacts under
`data/processed/public_data_v4/`:

```text
public_data_source_catalog.csv
epl_cohort_manifest.csv
national_acquisition_log.csv
international_performance_record_audit.csv
international_appearances_raw.csv
snapshot_national_appearance_audit.csv
international_appearances.csv
national_duplicate_audit.csv
national_team_id_crosswalk.csv
independent_schedule_validation.csv
openfootball_worldcup_player_validation.csv
public_match_timeline.csv
venue_geocodes.csv
geographic_travel_coverage_audit.csv
match_exposure_scope_features.csv
international_exposure_coverage_audit.csv
exposure_coverage_audit.csv
official_schedule_validation.csv
selection_risk_set.csv
selection_membership_resolution_audit.csv
selection_weight_diagnostics.csv
injury_source_validation.csv
baseline_parity_report.csv
exposure_scope_comparison.csv
frozen_baseline_national_scope_comparison.csv
v4_model_comparison.csv
v4_model_input_audit.csv
v4_scope_selected_predictions.csv
v4_national_record_quality_audit.csv
v4_exposure_change_audit.csv
v4_recovery_change_audit.csv
v4_all_scope_model_comparison.csv
v4_all_scope_model_input_audit.csv
v4_all_scope_selected_predictions.csv
v4_country_duty_between_within.csv
v4_country_duty_support.csv
v4_conclusion_audit.csv
international_status_ledger.csv
national_status_features.csv
v4_national_status_rates.csv
v4_national_status_models.csv
v4_national_status_model_support.csv
v4_data_quality_registry.csv
v4_result_tier_registry.csv
```

`exposure_coverage_audit.csv` is a gate, not a descriptive nicety. The
club-plus-senior-national scope may be interpreted as a primary v4 exposure
only when the protocol-recorded official-schedule check verifies at least 95% of
identified matches, at least 95% of identified senior-competitive appearances
have minutes, no cohort player request fails, and no unexplained player-game
duplicate remains. The file marks these rows with
`binding_for_primary_use=True`. Independent schedule reconstruction is a
secondary chronology row with no registered threshold; it cannot replace the
official gate. A blank official-schedule template is not evidence of coverage,
and unidentified minutes remain missing rather than being converted to zero.

The completed 3 August 2026 v4 acquisition found `33,083` played national
records, of which `33,081` had known minutes and were retained. The two
missing-minute records were senior friendlies; primary senior-competitive
minute completeness was `17,491/17,491` (`100%`). Across all played records,
minute completeness was `99.994%`. The build had zero acquisition errors and
zero unresolved player-game duplicates. It added senior-national minutes to
`2,379/101,042` eligible match rows (`2.35%`) and reclassified `2,171` rows
from zero to positive prior-seven-day burden. The binding official-schedule
audit has not been completed (`0/3,077` verified); the 95% gate was unexecuted,
not failed by disagreement. The independent
secondary reconstruction covered `2,785/3,077` matches (`90.5%`, 95% CI
`89.4--91.5`) but was informational. Every v4 exposure output is therefore
correctly marked `sensitivity_only`. The total-burden
conclusion was unchanged: the global spline-by-history interaction was
`p=0.664` before and `p=0.664` after senior competitive national minutes were
added; senior-all and broader-international scopes gave `p=0.562` and
`p=0.541`. The maximum absolute change among selected predictions was `4.3%`.

The post-hoc national-status models did not yield a multiplicity-adjusted
signal. In the primary higher-history stratum, recent senior competitive
national participation had an observed-minute IRR of `1.29` (95% CI
`0.83--2.03`; Holm `p=1.000`) and a fixed-90 IRR of `1.50` (`0.96--2.35`;
Holm `p=1.000`). Squad-only status was too sparse for adjusted inference. These
are Tier 5 sensitivities rather than positive findings. The full decision audit is
[`docs/public_data_v4_scientific_audit.md`](docs/public_data_v4_scientific_audit.md).

The observed-selection IPW diagnostic also failed
its prespecified post-weight balance criterion for prior-seven-day minutes
(`standardised mean difference 0.117 > 0.10`), so no weighted outcome model is
fitted or interpreted.  Travel is likewise unavailable in this build: none of
the `1,907` observed venues yet has a retained verified coordinate source, so
the geographic proxy remains missing rather than being treated as zero travel.

The timeline supplies shifted prior 3-, 5-, 7-, 14- and 28-day minutes and
appearance counts for every scope, plus prior national contribution, calendar
days since the previous observed appearance, and consecutive-match sequences.
Club fixture source data provide dates but not sufficiently complete kickoff
times, so v4 recovery is explicitly a calendar-day measure; it is not called a
less-than-72-hour measure.  Travel fields are geographic venue-to-venue
proxies only.  Coordinates without a verified retained source remain missing,
not zero-distance travel.

The selection-risk set uses observed EPL-club appearance spans and recorded
transfers to create plausible player-fixture opportunities.  It cannot reveal
squad selection, symptoms, medical clearance or complete roster membership.
Stabilised inverse-probability weights are produced only when overlap,
post-weight balance and weight stability pass their predeclared checks.
`injury_source_validation.csv` is a stratified review queue for official club
or competition sources; unreviewed Transfermarkt reports remain public-source
records and are never silently upgraded to clinical evidence.

Recommended order note:

- Run `src/25b_acquire_public_sources.py` after script 25 and before script 26;
  script 26 consumes its pinned independent schedule and World Cup lineup
  snapshots when building validation tables.
- Run `src/31_public_data_v4_quality_registry.py` after script 33 because its
  combined quality registry reads current-data extension diagnostics.
- Run `src/13_max_daily_load_features.py` before `src/14`, `src/18`, or `src/20`.
- Run `src/16_build_match_proxy_events.py` before `src/17`, `src/18`, `src/20`,
  `src/22`, or `src/23`.
- Run `src/19_plot_everything_for_paper.py` after analysis CSVs exist.
- Run `src/22_clinical_surveillance_bridge.py` after the match-proxy columns
  exist; it writes its own clinical bridge figure and mirrors it into
  `manuscript/figures/`.
- Run `src/23_prior_injury_duration_next_risk.py` after the match-proxy columns
  exist and after cleaned injuries exist; it writes the prior-duration next-risk
  tables and mirrors its H2 figure into `manuscript/figures/`.
- Run `src/00_list_result_columns.py` last when you want an inventory of
  generated result schemas.
- Run `src/24_prepare_public_deposit.py` after the result and figure outputs
  exist when preparing a neutral public archive export.

## Script Reference

`src/00_list_result_columns.py`

Builds `data/processed/results/columns_inventory.csv`, a schema inventory for
all result CSVs.

`src/01_fetch_fbref.py`

Historical filename. Reads local Transfermarkt data and writes:

```text
data/raw/epl_matches.csv
data/raw/epl_player_appearances.csv
```

Current regenerated counts:

```text
matches:      4,869
appearances: 135,950
```

`src/02_build_player_index.py`

Builds the stable player index. `fbref_player_id` is a compatibility alias for
`tm_player_id`.

`src/03a_build_player_mapping_tm.py`

Builds `data/processed/player_mapping_tm.csv` from local appearances and the
stable player index. This keeps downstream scripts using a mapping file without
returning to name-based matching.

`src/04_fetch_injuries_transfermarkt.py`

Fetches Transfermarkt injury histories and writes
`data/raw/tm_injuries_raw.csv`. This is the only step in the normal data build
that depends on live network access.

`src/05_clean_injuries.py`

Normalizes raw injury records, parses dates, removes unusable rows, estimates
reported durations where possible, and writes:

```text
data/processed/tm_injuries_clean.csv
```

Current regenerated count:

```text
clean injury reports: 12,440
```

`src/06_build_player_day_panel.py`

Builds the base EPL player-day panel, joins reports by stable player ID,
constructs canonical non-overlapping episodes with the shared builder, expands
episode days into unavailable days, and computes EPL-only rolling match counts
and minutes. Observed EPL appearances truncate a preceding episode.

`src/07_build_all_comp_minutes.py`

Computes shifted all-competition rolling burden from Transfermarkt appearances
for EPL club-seasons. It expands the risk grid to every valid club-season
appearance, reapplies canonical episode availability with observed-return
truncation, and writes `risk_set_history_reconciliation.csv`.

`src/08_baseline_hazard_all_comp.py`

Runs the overall daily discrete-time hazard model. This script intentionally
uses an analysis-window total all-competition minutes restriction (`>= 900`) as
a broad baseline, not the dynamic fragility risk set. It writes crude 45-minute rates,
estimable bin diagnostics, odds ratios, full-panel predictions, and a model
audit table.

`src/09_build_fragility_groups.py`

Builds prior-history day-level fragility labels and the latest-snapshot
summary. Strictly prior episodes and club exposure before the first eligible
risk-set day are included. This is the canonical source for `tough`, `regular`,
`fragile`, and `low_exposure`.

`src/10_hazard_by_fragility_45min.py`

Runs dynamic-fragility-stratified 45-minute daily hazard analyses. Crude tables
keep all bins; GLMs use the estimable-bin rule.

`src/11_hazard_5min_all_comp_and_fragility.py`

Writes high-resolution 5-minute crude daily and per-minute tables overall and
by dynamic fragility group. It deletes obsolete 5-minute GLM result CSVs from
older runs because those models are not considered valid.

`src/12_fft_injury_rates_fragility.py`

Builds daily injury-rate time series by dynamic fragility group and exports FFT
power spectra for exploratory cyclicity checks.

`src/13_max_daily_load_features.py`

Adds match-load covariates used by the richer weekly/load-adjusted models:
recent maximum daily load, last-match minutes, extra-time indicators, and
weekly/half-weekly harmonic terms.

`src/14_hazard_by_fragility_45min_weekly.py`

Fits dynamic-fragility-stratified daily hazard models with 45-minute burden
bins plus richer recent-load and weekly timing features. It also writes one
model-audit row per fitted history stratum.

`src/16_build_match_proxy_events.py`

Creates match-proxy outcomes and context tables, mutating
`player_day_panel_all_comp.csv` in place while preserving all player-days.

`src/17_match_proxy_perminute_descriptives_5min.py`

Writes descriptive 5-minute match-proxy per-minute crude rates overall, by
fragility, and by today's minutes bin.

`src/18_match_proxy_poisson_splines_perminute.py`

Fits the main per-minute match-proxy Poisson spline model with
`log(all_minutes_played)` as the exposure offset. The default model compares
the intermediate and higher prior-injury-history strata and excludes the lower
stratum for sparse-event stability. It writes pointwise confidence intervals,
selected 0/90/180-minute prediction confidence intervals, local support counts
around selected prediction points, higher-to-intermediate model-contrast
confidence intervals, denominator/link checks, match-proxy outcome sensitivity
models, calendar and clean-comparator restrictions, alternative
prior-injury-history label sensitivity models, age/position/club-season control
sensitivities with and without continuous prior-history controls,
severe-reported-absence and muscle/tendon outcome restrictions, clean
negative-control comparisons, spline-shape sensitivity refits, recurrent event
GEE, within-player switcher fixed-effect, and within/between decomposition
outputs, an out-of-time threshold sensitivity, and temporal-stability refits
for the three football-season blocks. It also writes
the period-specific prediction grids used by the
temporal-stability curve figure. Formal outputs include a joint Wald test of all
spline-by-history terms, within-stratum IRRs for 0 versus 180 and 90 versus 180
prior minutes, and ratios comparing those changes across strata.
`matchproxy_effect_modification_tests.csv` and
`matchproxy_denominator_effect_modification_tests.csv` report unadjusted,
Holm-adjusted, and Benjamini-Hochberg-adjusted p-values within each contrast
family. `matchproxy_nominal_exposure_response_signals.csv` lists every
unadjusted exposure-response or interaction signal below p = 0.05 so nominal
findings are reported symmetrically. `matchproxy_publication_contrast_summary.csv` gives a compact
publication-facing view of the 0-, 90-, and 180-minute contrasts for every
main, sensitivity, recurrent-event, and denominator/link specification.
`matchproxy_type_history_recency_attenuation.csv` fits the no-recency and
matched-recency specifications in one stacked, player-clustered Poisson model
and tests their coefficient change with the fitted covariance.
`matchproxy_type_history_multiplicity_family.csv` combines those p-values with
the overlapping-label, mutually exclusive binary, continuous, shape, and
threshold tests and reports Holm and Benjamini--Hochberg values across the
complete 74-test type-history family.
Negative-control outputs include
`matchproxy_negative_control_magnitude_comparison.csv`,
`matchproxy_negative_control_direct_comparison.csv`,
`matchproxy_negative_control_mutually_exclusive_type_binary.csv`,
`matchproxy_negative_control_mutually_exclusive_type_frequency.csv`, and
`matchproxy_negative_control_type_frequency_distribution.csv`, plus
`matchproxy_negative_control_type_frequency_linearity_check.csv` and
`matchproxy_negative_control_type_frequency_linearity_formal_test.csv`. The
distribution file carries median, quartiles, IQR, mean, maximum, skewness,
binary-threshold high/low means, and the delta-method Q3-scaled direct ratio for
the continuous mutually exclusive model. The linearity file compares observed
binary high-history IRRs with the IRRs predicted by applying continuous slopes
across the same observed high-versus-lower mean gaps. The formal linearity file
fits matched no-recency and symmetric-recency models containing both continuous
type-frequency terms and both high-frequency indicators. The symmetric-recency
model adds any-prior-report status and log days since the last prior report for
both muscle/tendon and joint/ligament or bone/fracture histories, so the
indicator tests excess high-frequency signal beyond a constant per-report slope
as a controlled direct effect. It also reports prior-report-only frequency--
log-recency correlations and term-specific variance-inflation factors computed
from each exact fitted design matrix. The all-row correlations remain in the
archive only to expose the sign change created when no-prior rows share zero
frequency and zero coded recency; they are not used as fitted-model
collinearity diagnostics. Curve-shape and
support outputs are written to
`matchproxy_spline_curve_shape_summary.csv`,
`matchproxy_spline_shape_sensitivity.csv`,
`matchproxy_spline_shape_contrast_sensitivity.csv`,
`matchproxy_selection_band_audit.csv`,
`matchproxy_selection_band_joint_proxy_audit.csv`,
`matchproxy_recurrent_event_decomposition.csv`,
`matchproxy_reporting_process_severity_audit.csv`, and
`matchproxy_observed_event_support_summary.csv`. Recovery-interval outputs split
`>14 days` from `no prior match` in `matchproxy_recovery_interval_rates.csv`
and report pooled trends, within-stratum trends, recovery-by-history
interaction tests, and direct 0-3 versus 6-7 day contrasts in
`matchproxy_recovery_interval_trend_tests.csv`. The matched recovery denominator
and link checks are written to `matchproxy_recovery_interval_model_summary.csv`.
`matchproxy_recovery_interval_publication_summary.csv` combines all-cause,
muscle/tendon, and reported-absence `>=28 days` recovery tests for manuscript
reporting.

`src/33_matchproxy_current_data_extensions.py`

Runs seven post-primary audits on the frozen intermediate/higher-history
match-proxy analysis frame after `src/18_match_proxy_poisson_splines_perminute.py`:

1. Refit the spline among rows with recorded lineups, starters only,
   substitutes only, rows outside a 14-day recorded return window, and starters
   outside that window. A recorded lineup-role-by-spline test asks whether the
   fitted curve differs by observed role; it does not measure clinical
   clearance or latent fitness.
2. Measure report-text type completeness by proxy timing, history stratum and
   recorded role. A type-report inverse-probability sensitivity is retained
   only if minimum fitted classification probability is at least `0.10` and
   maximum weight is at most `10`; otherwise it is labelled archive-only.
   Its confidence intervals come from 200 player-bootstrap resamples because
   the installed `statsmodels` covariance implementation does not provide a
   supported clustered sandwich estimate with frequency weights.
3. Summarise dated public-episode absence days per recorded match hour and fit a
   conditional duration model. These are reporting/date proxies, not medical
   time loss, severity, or clinical injury burden.
4. Tabulate joint previous-seven-day-minute and recovery support before the
   restricted one-versus-two-prior-club-match model. Sparse joint cells are
   displayed rather than extrapolated.
5. Fit conditional logistic models within event-containing player-seasons. The
   estimates compare observed rows within the same player-season and remain
   conditional on selection into appearances; they are not causal workload
   effects.
6. Resample players 1,000 times to measure how often the 15--45-minute band is a
   fitted global maximum, and refit the primary curve with player and current
   match as two clustering dimensions.
7. Refit after current-match competition adjustment and in the current
   Premier-League-only subset. Current-match metadata are reconciled exactly
   with the observed match-minute total before these models are fit.

All extension artifacts start with `matchproxy_extension_`; their fields,
restrictions, and permitted interpretation are documented in
[`docs/public_data_v4_data_dictionary.md`](docs/public_data_v4_data_dictionary.md).

`src/34_jsams_referee_analysis.py`

Builds the predecessor JSAMS audit layer after scripts 18 and 33. Its spline,
quality, cohort, denominator, bootstrap, and historical outputs remain inputs
to the current paper, but its spline is no longer the reference exposure model.
Script 36 supplies the current additive estimand, complete 63-model family, and
controlling tier registry.

The script writes 41 `jsams_*.csv` files under
`data/processed/results/`, including:

- the exact model specification, coefficients, standardised predictions, direct contrasts,
  and global tests;
- all nine symmetric outcome/denominator combinations;
- conventional player, exposure, event-type, and cohort-flow descriptors;
- local support at 0, 90, 180, and 220 minutes;
- five exposure-form refits and two same-day outcome-quality restrictions;
- 450-, 900-, and 1,800-minute, complete-season, pandemic-excluded, and
  club-plus-senior-national cohort refits;
- 1,000-replicate player-cluster bootstraps for the overall, starter,
  substitute, and lineup-standardised same-day minute differences;
- lineup completeness by season, exposure, history, outcome, and competition;
- full paired player-cluster selection bootstraps that refit the model and
  recompute both standardisations in every draw;
- within-player and within-player-season conditional logistic models;
- appearance-day, next-day, and other-day timing-enrichment estimates;
- included-versus-excluded player comparisons for the 900-minute threshold;
- age/position/club-season, competition, EPL-only, and two-way-cluster checks;
- a headline-inference audit that separates the global spline test from the
  post hoc anchor contrast;
- a complete test-level hypothesis register and family summary; and
- `jsams_claim_hierarchy.csv`, the predecessor tier registry retained for
  provenance; script 36's `jsams_revised_claim_hierarchy.csv` controls the
  current manuscript.

Every claim-registry row explains why its tier is not lower. The code rejects
blank justifications and any Tier 4--5 promotion to the abstract or a main
display; those tiers have a one-sentence main-Results limit.

The hypothesis register preserves non-estimable tests and labels every row as
reference, secondary, exploratory, or post hoc. Because no dated prospective
analysis plan exists, publication text must not call this family
"prespecified."

`src/36_jsams_second_referee_analysis.py`

Runs after scripts 27 and 34. It independently reconstructs 3-, 5-, 7-, 10-,
and 14-day prior-minute windows and requires exact parity for the legacy
seven-day field. The reference model is a player-clustered binomial-logit model
for a same-day public report per recorded appearance, with one linear exposure
term per 90 previous-seven-day minutes, additive continuous prior history, and
calendar phase. A cubic spline with 45-, 90-, and 135-minute interior knots is
a non-monotonic shape sensitivity with a 10,000-draw simultaneous band.

The complete multiverse crosses seven exposure summaries, three outcome
timings, and three denominators. All 63 focal tests share one Holm correction.
The stage also writes three fixed temporal refits, player and player-season
conditional models with player-cluster uncertainty and 5,000 multiplier draws,
a bounded selection-into-appearance sensitivity, and an exposure-blinded
30-record independent-source audit. The audit validator requires exact queue
IDs and immutable fields, no exposure columns, independent URLs for resolved
records, and no Transfermarkt audit source. The revised hypothesis register has
706 unique rows: 637 legacy rows plus 63 multiverse tests, three temporal tests,
one temporal-heterogeneity test, and two conditional estimates.

`src/35_plot_jsams_revision.py`

Reads generated script-34 and script-36 tables and produces the four main
manuscript figures:

```text
J1_jsams_cohort_measurement.png
J2_jsams_primary_robustness.png
J3_jsams_within_player_lineup_coverage.png
J4_jsams_context_support.png
```

The plotter shows 95% confidence intervals where an effect estimate is drawn,
uses readable text at journal-page scale, and never refits a model. The
national chronology and type-recency figures are supplementary.

`src/19_plot_everything_for_paper.py`

Generates the current figure set under:

```text
data/processed/results/figures/
```

It removes known stale figure names from older separated 5-minute GLM runs,
plots only from current result CSVs, and mirrors the generated PNG set into:

```text
manuscript/figures/
```

`src/20_model_diagnostics.py`

Refits the corrected daily logistic and per-minute Poisson models for
residual, Q-Q, and fitted-value diagnostics. The daily diagnostic now mirrors
the baseline analysis-window cohort rather than the narrower dynamic
prior-history risk set; the Poisson diagnostic mirrors the intermediate/higher
match-proxy model. It writes CSV diagnostics under
`data/processed/results/diagnostics/` and PNG diagnostics in the same directory.

`src/21_panel_restriction_counts.py`

Prints full-panel, dynamic-risk-set, and match-proxy per-minute denominators.
Current output:

```text
Full EPL panel      : players = 1558, player-days = 1330896, match-days = 84191
All-comp risk panel : players = 1558, player-days = 1398297, match-days = 101042
Dynamic risk set    : players = 1208, player-days = 1018202, match-days = 88573
Primary model set   : players = 1057, match-days = 72445, match-proxy events = 1592
```

`src/22_clinical_surveillance_bridge.py`

Builds clinician-facing descriptive summaries from the corrected public-data
panel. This script does not create true club medical surveillance data. It
translates the match-proxy outcome into units sports medicine readers expect
and makes the missing clinical fields explicit.

It writes:

```text
data/processed/results/clinical_match_hour_rates.csv
data/processed/results/clinical_duration_context_summary.csv
data/processed/results/clinical_matchproxy_duration_rates_by_group.csv
data/processed/results/clinical_matchproxy_duration_rates_by_burden.csv
data/processed/results/figures/H1_clinical_bridge_rates.png
manuscript/figures/H1_clinical_bridge_rates.png
```

Current regenerated clinical bridge headline:

```text
same-day + lag-1 match-proxy rate: 17.6 (95% CI 16.8-18.4) per 1,000 match hours
lower-history rate:                12.4 (11.0-14.0) per 1,000 match hours
intermediate-history rate:         16.6 (15.5-17.7) per 1,000 match hours
higher-history rate:               23.6 (21.9-25.5) per 1,000 match hours
lower/intermediate appearance rate: 19.0 (17.9-20.1) per 1,000 appearances
higher-history appearance rate:     25.4 (23.6-27.5) per 1,000 appearances
```

The duration-specific rate table uses the reported Transfermarkt absence
duration attached to the same-day injury episode or the next-day lag-1 episode.
Duration buckets are `<1 week`, `1 week to 2 months`, `2 months to 1 year`,
`>1 year`, and `unknown`.

`src/23_prior_injury_duration_next_risk.py`

Attaches the most recent completed prior injury episode to each all-competition
match row and asks whether that previous episode's reported duration is associated
with the next match-proxy injury rate per minute. The as-of join is strictly
prior-only: an episode must have ended before the match date to be eligible.

It writes:

```text
data/processed/results/prior_injury_duration_next_risk_canonical.csv
data/processed/results/prior_injury_duration_next_risk_frequency_only.csv
data/processed/results/prior_injury_duration_type_mix.csv
data/processed/results/figures/H2_prior_injury_duration_next_risk.png
manuscript/figures/H2_prior_injury_duration_next_risk.png
```

The canonical table uses the manuscript's intermediate and higher
prior-injury-history groups. The frequency-only table repeats the analysis with
a duration-independent high-prior-frequency label so that prior injury duration
is not part of the group definition. The rate tables include descriptive rates
and approximate 95% Poisson count-rate intervals. Current regenerated canonical
rates per 1,000 match hours are:

```text
intermediate, <1 week prior injury:       17.0 (95% CI 14.7-19.6)
intermediate, 1 week to 2 months:         16.8 (15.6-18.1)
intermediate, 2 months to 1 year:         14.1 (11.3-17.5)
higher, <1 week prior injury:             27.4 (23.0-32.7)
higher, 1 week to 2 months:               24.0 (21.8-26.4)
higher, 2 months to 1 year:               20.0 (16.7-23.9)
```

The type-mix table is a descriptive check because duration can also proxy for
injury type and return-to-play pathway. In the current run, match rows following
1-week-to-2-month prior injuries are most often preceded by muscle/tendon
descriptions (37.3%), while rows following 2-month-to-1-year prior injuries are
more often preceded by joint/ligament descriptions (47.2%). This is why the
paper treats the downward duration pattern as compatible with management and
selection, not as evidence that longer injuries are inherently safer.

The `>1 year` bucket is retained in the CSVs but has very little match-hour
support in both groups, so it is not foregrounded in the manuscript figure.

`src/24_prepare_public_deposit.py`

Creates a neutral public-deposit export under:

```text
public_deposit/
```

The source pipeline keeps legacy internal labels for backwards compatibility,
but this export rewrites public CSV columns, string values, and filenames to use
lower/intermediate/higher prior-injury-history terminology. It also writes:

```text
public_deposit/sanitization_manifest.csv
```

`src/25_public_data_v4.py`

Creates the immutable v4 acquisition layer. It snapshots the pinned public
Transfermarkt dataset, builds the stable EPL cohort manifest, reuses or fetches
one cached national-performance response per cohort player, and writes the raw
national appearance and acquisition-audit tables. It never overwrites
`external_data/transfermarkt/` and never treats a failed request or missing
minutes as zero exposure.

`src/25b_acquire_public_sources.py`

Caches the pinned independent senior-international results and World Cup
lineup sources used by script 26, including their licences and SHA-256 source
manifests. It also writes `public_data_source_catalog.csv`, which records every
candidate source, its accepted or rejected role, and its known limitation.

`src/26_build_public_match_timeline.py`

Reconciles endpoint and bulk-snapshot national appearances, records every
duplicate decision, crosswalks national-team IDs, validates dates/scores and a
World Cup player subset against independent sources, and builds the unified
club-plus-country timeline. It then attaches shifted 3-, 5-, 7-, 14-, and
28-day exposure features to all `101,042` eligible club appearance rows.

`src/27_public_data_v4_audits.py`

Applies the binding official-schedule/minute/identity/duplicate gates, writes
the independent chronology result separately, creates the injury-source review
queue, and rebuilds the bounded player-fixture opportunity set. Its
`selection_membership_resolution_audit.csv` must pass the one-player-date gate
before script 36 may use `selection_risk_set.csv`.

`src/28_public_data_v4_model_comparison.py`

Compares the frozen all-club exposure with the club-plus-senior-competitive
scope on the same match rows, writes model-input parity, selected predictions,
and formal comparison tables, and obeys the coverage gate's
`sensitivity_only` decision.

`src/29_public_data_v4_scientific_audit.py`

Audits exposure and recovery changes across every v4 scope, national-record
quality, all-scope models, country-duty between/within estimates, support, and
claim eligibility. `v4_conclusion_audit.csv` is the machine-readable decision
table for what the national extension can and cannot support.

`src/30_national_status_analysis.py`

Classifies played, squad-only, recorded-unavailable, and not-in-squad national
records without converting one state into another. It creates shifted status
features, descriptive rates with common interval methods, model support gates,
and the multiplicity-adjusted national-status model family.

`src/31_public_data_v4_quality_registry.py`

Collects structural, coverage, support, uncertainty, and interpretation gates
from the v4 and predecessor result files. It writes
`v4_data_quality_registry.csv` and the historical
`v4_result_tier_registry.csv`; neither supersedes script 36's current
publication registry.

`src/32_plot_v4_paper_figures.py`

Generates `I1_type_history_recency_audit.png` and
`I2_v4_national_validation.png` from existing tables without refitting models.
These are retained supplementary/historical figures, not current main-paper
display items.

`src/pipeline_io.py`

Shared helper module for day-level fragility merges, dynamic risk-set
restriction, canonical 45-minute bins, representative bin values, and
estimable-bin checks.

`src/public_data_sources.py`

Pins, downloads, hashes, parses, and reconciles the reusable independent
international-results and World Cup lineup sources. Source commits, URLs,
licences, aliases, and normalization decisions live here rather than in the
analysis scripts.

`src/v4_statistics.py`

Provides the shared Wilson proportion interval, exact central Poisson rate
interval, and percentage conversion used by v4 quality and status outputs.
This prevents separate scripts from silently applying different uncertainty
conventions.

## Result Files

Result CSVs are written under:

```text
data/processed/results/
```

Main result families:

```text
baseline_injury_rate_by_all_minutes7d_bin_45min.csv
baseline_injury_rate_per_minute_by_all_minutes7d_bin_45min.csv
glm_estimable_bins_all_minutes7d_45min.csv
glm_or_all_minutes7d_bins_45min.csv
glm_predicted_probs_all_minutes7d_bins_45min.csv
glm_model_audit_all_minutes7d_45min.csv

hazard_by_fragility_<group>_daily_45min.csv
hazard_by_fragility_<group>_per_minute_45min.csv
hazard_by_fragility_<group>_glm_estimable_bins_45min.csv
hazard_by_fragility_<group>_glm_or_45min.csv
hazard_by_fragility_<group>_glm_predicted_45min.csv

hazard_5min_overall_perday.csv
hazard_5min_overall_perminute.csv
hazard_5min_<group>_perday.csv
hazard_5min_<group>_perminute.csv

hazard_45min_<group>_perday_weekly.csv
hazard_45min_<group>_perminute_weekly.csv
glm_estimable_bins_45min_<group>_weekly.csv
glm_or_45min_<group>_weekly_bins.csv
glm_or_45min_<group>_weekly_loadvars.csv
glm_predicted_probs_45min_<group>_weekly.csv
glm_model_audit_45min_weekly.csv

fft_spectrum_<group>.csv

match_proxy_counts_overall.csv
match_proxy_counts_by_fragility.csv
matchproxy_hazard_5min_overall_perminute.csv
matchproxy_hazard_5min_<group>_perminute.csv
matchproxy_hazard_5min_<group>_perminute_by_todaybin.csv

poisson_spline_params_matchproxy.csv
poisson_spline_predictions_matchproxy.csv
poisson_spline_selected_predictions_matchproxy.csv
poisson_spline_selected_support_matchproxy.csv
poisson_spline_selected_ratios_matchproxy.csv
poisson_spline_diagnostic_support_matchproxy.csv
matchproxy_sensitivity_summary.csv
matchproxy_outcome_history_cross_summary.csv
matchproxy_type_discordant_history_summary.csv
matchproxy_negative_control_magnitude_comparison.csv
matchproxy_negative_control_anchor_selection_audit.csv
matchproxy_negative_control_direct_comparison.csv
matchproxy_negative_control_mutually_exclusive_type_frequency.csv
matchproxy_negative_control_mutually_exclusive_type_binary.csv
matchproxy_negative_control_type_frequency_distribution.csv
matchproxy_negative_control_recent_return_excluded_model_summary.csv
matchproxy_negative_control_recent_return_exclusion.csv
manuscript_numeric_reconciliation.csv
matchproxy_proxy_classification_publication.csv
matchproxy_proxy_event_type_summary.csv
matchproxy_effect_modification_tests.csv
matchproxy_formal_model_contrast_tests.csv
matchproxy_multiplicity_family_summary.csv
matchproxy_effect_modification_multiplicity_family_summary.csv
matchproxy_publication_referee_audit.csv
matchproxy_denominator_sensitivity_summary.csv
matchproxy_denominator_effect_modification_tests.csv
matchproxy_publication_contrast_summary.csv
matchproxy_same_day_denominator_audit.csv
matchproxy_spline_curve_shape_summary.csv
matchproxy_spline_shape_sensitivity.csv
matchproxy_spline_shape_contrast_sensitivity.csv
matchproxy_selection_band_audit.csv
matchproxy_selection_band_joint_proxy_audit.csv
matchproxy_recurrent_event_decomposition.csv
matchproxy_reporting_process_severity_audit.csv
matchproxy_observed_event_support_summary.csv
match_proxy_backattribution_reconciliation.csv
matchproxy_out_of_time_threshold_audit.csv
matchproxy_prespecified_absolute_effect_modification.csv
matchproxy_prespecified_absolute_predictions.csv
matchproxy_prespecified_absolute_selected_predictions.csv
matchproxy_prespecified_absolute_selected_ratios.csv
matchproxy_prespecified_absolute_support.csv
matchproxy_temporal_stability_summary.csv
matchproxy_temporal_stability_predictions.csv
matchproxy_recovery_interval_rates.csv
matchproxy_recovery_interval_trend_tests.csv
matchproxy_recovery_interval_rates_reported_absence_ge28d.csv
matchproxy_recovery_interval_trend_tests_reported_absence_ge28d.csv
matchproxy_recovery_interval_rates_muscle_tendon_only.csv
matchproxy_recovery_interval_trend_tests_muscle_tendon_only.csv
matchproxy_recovery_interval_model_summary.csv
matchproxy_recovery_interval_publication_summary.csv

matchproxy_extension_lineup_refits_summary.csv
matchproxy_extension_lineup_refits_predictions.csv
matchproxy_extension_lineup_refits_shape.csv
matchproxy_extension_lineup_spline_interaction.csv
matchproxy_extension_reporting_completeness_context.csv
matchproxy_extension_reporting_type_model.csv
matchproxy_extension_reporting_type_ipw_diagnostics.csv
matchproxy_extension_reporting_type_ipw_summary.csv
matchproxy_extension_reporting_type_ipw_selected.csv
matchproxy_extension_reporting_type_ipw_ratios.csv
matchproxy_extension_reporting_type_ipw_bootstrap.csv
matchproxy_extension_reported_absence_day_burden_history.csv
matchproxy_extension_reported_absence_day_burden_history_by_burden.csv
matchproxy_extension_reported_duration_conditional_model.csv
matchproxy_extension_joint_burden_recovery_support.csv
matchproxy_extension_joint_schedule_compression_model.csv
matchproxy_extension_within_player_case_crossover.csv
matchproxy_extension_current_match_metadata_audit.csv
matchproxy_extension_competition_context_rates.csv
matchproxy_extension_competition_context_refits.csv
matchproxy_extension_two_way_cluster_sensitivity.csv
matchproxy_extension_curve_feature_bootstrap_samples.csv
matchproxy_extension_curve_feature_bootstrap_summary.csv

jsams_revised_absolute_risk_contrast.csv
jsams_revised_additive_curve_tests.csv
jsams_revised_additive_curves.csv
jsams_revised_appearance_selection_diagnostics.csv
jsams_revised_appearance_selection_estimates.csv
jsams_revised_appearance_selection_population.csv
jsams_revised_ascertainment_by_exposure.csv
jsams_revised_attenuation_bootstrap.csv
jsams_revised_case_restricted_exposure_bias.csv
jsams_revised_claim_hierarchy.csv
jsams_revised_club_congestion_sensitivity.csv
jsams_revised_conditional_estimates.csv
jsams_revised_conditional_population.csv
jsams_revised_conditional_support.csv
jsams_revised_confounding_sensitivity.csv
jsams_revised_denominator_attenuation_decomposition.csv
jsams_revised_denominator_by_lineup_role.csv
jsams_revised_denominator_contrast_metadata.csv
jsams_revised_denominator_gradient_by_league.csv
jsams_revised_denominator_gradient_clip_sensitivity.csv
jsams_revised_denominator_gradient_decision_rule.csv
jsams_revised_denominator_gradient_estimator_sensitivity.csv
jsams_revised_denominator_gradient_specifications.csv
jsams_revised_denominator_gradient_summary.csv
jsams_revised_direct_truncation_refit.csv
jsams_revised_episode_type_composition.csv
jsams_revised_event_clustering_summary.csv
jsams_revised_exposure_metric_correlations.csv
jsams_revised_exposure_metric_summary.csv
jsams_revised_exposure_multiverse.csv
jsams_revised_exposure_multiverse_summary.csv
jsams_revised_exposure_support.csv
jsams_revised_exposure_window_gradient.csv
jsams_revised_history_reference_value.csv
jsams_revised_hypothesis_register.csv
jsams_revised_lineup_composition_by_exposure.csv
jsams_revised_lineup_coverage_denominator_stability.csv
jsams_revised_model_field_completeness.csv
jsams_revised_negative_control_outcomes.csv
jsams_revised_non_event_absence_screen.csv
jsams_revised_non_event_audit_queue.csv
jsams_revised_non_event_audit_summary.csv
jsams_revised_outcome_audit_queue.csv
jsams_revised_outcome_audit_summary.csv
jsams_revised_outcome_audit_validation.csv
jsams_revised_placebo_denominator_replication.csv
jsams_revised_placebo_window_analysis.csv
jsams_revised_recorded_minute_distribution.csv
jsams_revised_role_adjusted_denominator_refit.csv
jsams_revised_run_in_exclusion_comparison.csv
jsams_revised_run_in_threshold_sensitivity.csv
jsams_revised_second_assessor_agreement.csv
jsams_revised_squad_role_association_sensitivity.csv
jsams_revised_temporal_stability.csv
jsams_revised_truncation_imputation_sensitivity.csv
jsams_revised_window_validation.csv

clinical_match_hour_rates.csv
clinical_duration_context_summary.csv
clinical_matchproxy_duration_rates_by_group.csv
clinical_matchproxy_duration_rates_by_burden.csv

prior_injury_duration_next_risk_canonical.csv
prior_injury_duration_next_risk_frequency_only.csv
prior_injury_duration_type_mix.csv

columns_inventory.csv
```

Current local result inventory after the 6 August 2026 review-round rerun:

```text
249 top-level result CSVs under data/processed/results/
252 result CSVs including the diagnostics subdirectory
54 figures under data/processed/results/figures/
303 public-deposit payload artifacts plus sanitization_manifest.csv (304 files total)
```

`data/processed/results/columns_inventory.csv` indexes the schemas of all 276
processed CSVs, including 44 v4 tables. The exact 41 script-34 and 49 script-36
file definitions are in
[`docs/public_data_v4_data_dictionary.md`](docs/public_data_v4_data_dictionary.md),
while [`docs/repository_inventory.md`](docs/repository_inventory.md) records
ownership, current-versus-historical status, and local/versioned policy.

## Figures And Manuscript Assets

The active plotter writes to:

```text
data/processed/results/figures/
```

The four main submitted figures are generated directly into
`manuscript/figures/` by `src/35_plot_jsams_revision.py`. They show cohort
construction and event-linked recorded time; the additive reference curve,
simultaneous spline sensitivity and seven exposure metrics; fixed temporal and
conditional estimates; and the independent-source, bounded-selection, outcome-
timing and denominator checks. The main paper does not use the older FFT,
daily, categorical spline, type-recency, or national chronology figures.

Legacy and supplementary figures are generated by
`src/19_plot_everything_for_paper.py` and the specialised scripts below.
The clinical bridge figure `H1_clinical_bridge_rates.png` is generated by
`src/22_clinical_surveillance_bridge.py` because it depends on the duration
join and match-hour summaries built in that script.
The prior-injury-duration figure `H2_prior_injury_duration_next_risk.png` is
generated by `src/23_prior_injury_duration_next_risk.py` because it depends on
a prior-only most-recent-completed-injury as-of join. It is retained as a
repository-only diagnostic rather than a submitted display item.

`G1_public_data_measurement_audit.png` is retained as a historical diagnostic.
It has been replaced in the main paper by `J1` and `J2`, which use the
same-day per-appearance reference estimand and 1,000-replicate uncertainty.

The LaTeX manuscript references files under:

```text
manuscript/figures/
```

Do not manually copy the `J1`--`J4` figures: script 35 writes their publication
copies. After any data rerun, rerun scripts 27, 34, 36 and 35, compile both
`manuscript.tex` and `supplement.tex`, and reconcile every manuscript number
against the generated CSVs. The prose remains a reviewed paper-writing layer,
not an automatic source of truth.

Build the clean manuscript, blinded review copy, Supplement and separate title
page from `manuscript/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex

# manuscript_blind.tex is the versioned wrapper carrying the same macro
# definitions, so the anonymized review copy is a committed artifact rather
# than a command-line incantation. This is the file to upload for review.
pdflatex -interaction=nonstopmode -halt-on-error manuscript_blind.tex
bibtex manuscript_blind
pdflatex -interaction=nonstopmode -halt-on-error manuscript_blind.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript_blind.tex

pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
bibtex supplement
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex

# The supplement travels to referees, so it has an anonymized build too.
pdflatex -interaction=nonstopmode -halt-on-error supplement_blind.tex
bibtex supplement_blind
pdflatex -interaction=nonstopmode -halt-on-error supplement_blind.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement_blind.tex

pdflatex -interaction=nonstopmode -halt-on-error title_page.tex
pdflatex -interaction=nonstopmode -halt-on-error title_page.tex
```

The default manuscript contains authors and no margin line numbers. The review
copy is double spaced, line numbered and anonymous. The generated review,
Supplement and title-page PDFs are intentionally ignored by Git; their TeX
sources are versioned, while the current clean `manuscript.pdf` is versioned.

## Tests And Coverage

There is one test file for every retained Python file, including
`config.py` and `src/pipeline_io.py`.

Run the full suite:

```powershell
python -m pytest --cov --cov-branch
```

Coverage is configured in `.coveragerc`:

```text
branch = True
source = .
fail_under = 100
```

Current verified result: 289 tests passed; total statement and branch coverage
was 100.00% across 7,559 statements and 1,958 branches.

Final manuscript counts were measured with TeXcount 3.1.1 on generated slices
of `manuscript/manuscript.tex`: `manuscript/tmp/wordcount_main.tex` contains
4,543 main-text words and `manuscript/tmp/wordcount_abstract.tex` contains 249
abstract words, against JSAMS limits of 6,000 and under 250. The main-text
slice includes the Practical implications and excludes the abstract, displays,
captions, references and declarations.

Heavy CLI entrypoints and plotting side-effect wrappers are marked
`# pragma: no cover`; deterministic helpers, branch logic, estimability rules,
fragility assignment, and IO failure paths are tested directly.

## Validation Commands

Useful smoke checks after rerunning the pipeline:

```powershell
.\.venv312\Scripts\python.exe src/21_panel_restriction_counts.py
.\.venv312\Scripts\python.exe src/22_clinical_surveillance_bridge.py
.\.venv312\Scripts\python.exe src/23_prior_injury_duration_next_risk.py
.\.venv312\Scripts\python.exe src/33_matchproxy_current_data_extensions.py
.\.venv312\Scripts\python.exe src/27_public_data_v4_audits.py
.\.venv312\Scripts\python.exe src/34_jsams_referee_analysis.py
.\.venv312\Scripts\python.exe src/36_jsams_second_referee_analysis.py
.\.venv312\Scripts\python.exe src/35_plot_jsams_revision.py
.\.venv312\Scripts\python.exe src/00_list_result_columns.py
.\.venv312\Scripts\python.exe -m pytest --cov=. --cov-branch
```

Useful stale-output checks:

```powershell
Get-ChildItem data\processed\results -Filter "*5min*glm*"
rg -n "glm_or_all_minutes7d_bins_5min|fragileonly" src tests
```

The first command should find no 5-minute GLM result CSVs. The second command
should only find deliberate stale-cleanup references in
`src/11_hazard_5min_all_comp_and_fragility.py` or
`src/19_plot_everything_for_paper.py`.

## Known Limits

- Transfermarkt injury data are public reporting data, not club medical
  surveillance records.
- Injury dates are Transfermarkt recorded spell-start dates, not webpage
  publication timestamps, medically verified onset or observed injury time.
- The clinical bridge tables report public-data translations. They cannot
  provide true training injury incidence per 1,000 training hours because no
  training exposure hours are observed. They also cannot classify injury
  mechanism, contact status, clinical diagnosis, recurrence, or club-confirmed
  time-loss definitions.
- Reported absence-duration buckets use Transfermarkt `durationDetails` and
  `missedGamesCount`. These fields are useful proxies for severity/time loss,
  but they are not equivalent to standardised medical time-loss surveillance.
- The prior-injury-duration next-risk analysis uses the most recent completed
  prior injury episode before each match row. It is prior-only, but it is still
  descriptive: prior injury duration is correlated with injury type, player
  selection, rehabilitation, squad role, return-to-play screening, and medical
  management that are not observed in the public data. Lower next per-minute
  rates after longer completed prior injuries should therefore be read as a
  management/selection-compatible pattern, not as evidence that longer injuries
  are protective.
- Match-proxy per-minute outcomes are heuristic. Same-day and lag-1 rules are
  now refit as separate sensitivity outcomes, and ambiguous injury descriptions
  can be excluded, but these checks do not prove the exact injury minute or
  mechanism.
- In the post-primary report-type audit, 1,210/1,592 (76.0%) modelled proxy events had
  type-classifiable public text. The inverse-probability sensitivity fails its
  recorded overlap gate (minimum fitted probability 0.074; maximum weight
  13.5), so it is an archive stress test rather than a correction for
  differential reporting. Public episode dates can describe reported absence
  days but cannot establish clinical time loss or injury burden.
- Lineup, recent-return, conditional player-season, and competition-context
  checks reduce only selected observed sources of composition and dependence.
  They cannot measure symptoms, medical clearance, latent fitness, or the
  counterfactual effect of assigning minutes. Joint burden/recovery cells are
  sparse above 180 prior minutes, so the associated one-versus-two-match model
  is restricted to observed support rather than extrapolated.
- The baseline script `src/08_baseline_hazard_all_comp.py` uses an
  analysis-window `>= 900` total all-competition minutes restriction. Fragility-stratified scripts use
  dynamic prior-history labels and exclude only low-exposure player-days.
- Frozen all-competition exposure is limited to club appearances for EPL
  club-seasons. The v4 sensitivity extension adds identified national-team
  appearances, but the post-data official-source gate was not executed because
  its ledger lacked official URLs and identifiers. Independent schedule
  reconstruction matched 90.5% and is a secondary chronology audit, not
  outcome validation; unobserved national or club records may remain.
- Five-minute burden outputs are descriptive only. The current publication
  inference comes from the script-34 same-day per-appearance model and its
  symmetric denominator, functional-form, context and cohort checks.
- The match-proxy Poisson spline excludes the lower prior-injury-history
  stratum by default because there are too few match-proxy events for stable
  interacted spline inference in that group.
- Selection into congested match appearances remains a major observational
  limitation. Sensitivity models adjust for age, position, club-season, and in
  one deliberately conservative check continuous prior-history controls, but
  they still cannot observe tactical role, travel, training load, medical
  management, or full player-level selection mechanisms.
- Temporal-stability refits are internal period checks, not predictive
  validation. They show whether the central 180-minute contrast points in a
  similar direction across calendar blocks, but each block has much wider
  uncertainty because the split reduces event counts.
- Generated data are ignored by git and must be regenerated or restored before
  rerunning the numerical pipeline. Vetted manuscript figures are versioned,
  while the compiled PDF is rebuilt locally from `manuscript/manuscript.tex`.
