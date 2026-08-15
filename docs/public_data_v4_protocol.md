# Public-data v4 protocol: club and international match exposure

> **Publication note.** This protocol governs the historical v4
> national-exposure extension only. It does not define the current manuscript's
> reference model or publication hierarchy. The current same-day,
> per-appearance measurement analysis is revision-defined in
> `src/36_jsams_second_referee_analysis.py`; it is not described as prospectively
> prespecified.

## Status and purpose

This protocol was written before acquiring or analysing the new international
appearance data.  It defines a public-data extension to the frozen club-only
analysis.  The purpose is to test whether the existing findings are sensitive
to incomplete measurement of match exposure outside a player's English Premier
League (EPL) club schedule.  It is not a search for a favourable association or
a substitute for clinical injury surveillance, training-load data, or medical
clearance records.

The acquisition baseline is the repository state at commit
`61648fd5f1aad64a52c0aac0f42f4fa2e8f31fbd`.  Its local input inventory and
test result are recorded in `data/raw/public_data_v4/baseline_manifest.json`.
The raw v4 acquisition and `external_data/transfermarkt/` must never be
overwritten by v4 analysis code. After acquisition, the club comparator and
every expanded scope are rebuilt on the same canonical risk set. This permits
pipeline-integrity corrections to be applied symmetrically rather than
preserving a known defect in the comparator.

## Study population, outcome, and prior injury history

The analysis period remains 1 July 2017 through 7 April 2025.  The cohort is
the existing EPL player-day cohort, identified by stable `tm_player_id` values.
National-appearance collection starts on 3 June 2017, 28 calendar days before
the outcome period, so a 28-day rolling window is complete on the first study
day.

The match-proxy injury outcome, canonical episode construction, availability
rules, and row-level prior-injury-history variables are identical across all
v4 scopes. Canonical episodes collapse touching or overlapping reports and end
before an observed return. The risk grid spans every valid EPL-club-season
appearance, and history includes strictly prior pre-entry episodes and club
minutes. Therefore each within-run scope comparison changes *exposure
measurement only*. No report is re-dated, reclassified, or removed because of
a national appearance record. Public reports remain a reproducible proxy
outcome, not clinical surveillance.

## Prespecified exposure scopes

All scopes use recorded player appearances and exclude the index match from
prior-window variables.

| Scope | Included exposure | Role |
| --- | --- | --- |
| `frozen_club_all` | Stored club all-competition minutes and recovery values from the frozen analysis | Direct comparator |
| `frozen_club_plus_senior_national` | `frozen_club_all` plus recorded senior national-team appearances in competitive matches | Direct v4 exposure, conditional on the coverage gate |
| `frozen_club_plus_senior_all` | Frozen club burden plus senior competitive and friendly national appearances | Secondary scope audit |
| `frozen_club_plus_broader_international` | Frozen club burden plus senior, youth and Olympic national appearances | Secondary scope audit |
| `club_competitive` | Refreshed recorded competitive club appearances for a club that was an EPL club in that season, including league, domestic-cup, UEFA and FIFA club competition matches | Reconstruction sensitivity check |
| `club_plus_senior_national` | `club_competitive` plus recorded senior national-team appearances in competitive matches | Reconstruction sensitivity check |
| `club_plus_senior_all` | Primary scope plus senior national-team friendlies | Secondary sensitivity analysis |
| `club_plus_broader_international` | Senior-all scope plus U21, U23 and Olympic appearances | Secondary sensitivity analysis |
| `club_plus_all_public` | Broader-international scope plus recorded club friendlies, where a source labels them reliably | Exploratory sensitivity analysis |

Senior competitive national matches include World Cup, continental
championships, continental and World Cup qualifying, UEFA/CONCACAF Nations
League, Gold Cup, AFCON, Asian Cup, Copa America, and other senior matches
whose source competition classification is not `friendly`.  A missing
classification remains missing: it is not assumed to be competitive or to have
zero minutes.

`club_competitive` is intentionally defined to reproduce the existing
all-competition club exposure definition as closely as the dated upstream
snapshot permits.  The baseline-parity report is a mandatory gate before any
club-plus-country estimate is interpreted.

### Comparator amendment after the parity audit

The frozen `all_minutes_last_7d` variable includes recorded club friendlies,
whereas the refreshed `club_competitive` scope excludes them.  The first v4
parity audit therefore found 610 of 101,042 match rows with a non-zero
difference (maximum 90 minutes).  This is a source-definition difference, not
a reason to rewrite the frozen analysis.  To keep the direct v4 comparison to
an exposure-only change, the model comparator is now `frozen_club_all` (the
stored frozen club burden and recovery interval) versus
`frozen_club_plus_senior_national` (that same frozen burden plus observed
senior-competitive-national minutes, and the earlier of the frozen club or
observed national prior appearance).  Refreshed club scopes remain available
for reconstruction sensitivity checks, but they are not substituted for the
frozen comparator.

## Acquisition and provenance

The dated upstream snapshot is stored under
`data/raw/public_data_v4/transfermarkt_datasets_YYYYMMDD/`; that directory is
ignored by Git and is immutable after its manifest has been written.  The
manifest records retrieval time in UTC, source URLs, upstream Git commit,
licence/redistribution notice, byte sizes, SHA-256 hashes and CSV column
schemas.  The v4 scripts must not use a latest URL without recording the
retrieved file hash and timestamp.

Published `transfermarkt-datasets` tables are the first source for national
tournament appearances.  They currently cover selected senior finals but not
the full qualifier, Nations League, Gold Cup, friendly, youth, or Olympic
calendar.  For the remaining cohort-specific histories, the acquisition script
uses Transfermarkt's publicly served player performance record keyed by
`tm_player_id`.  It caches each raw response, resumes failed IDs, rate-limits
requests, retries transient errors and records the source URL and retrieval
time.  This endpoint is an undocumented public web endpoint, not an official
Transfermarkt API; its availability and fields must be audited on every run.

National-team identity is assigned only from an actual national appearance
record.  Citizenship or a current national-team field is used only as cohort
metadata and never to impute an appearance or national load.

## Required fields and data-quality rules

Each normalised national appearance must retain the player, game, team and
opponent identifiers; local and UTC kick-off data where supplied; competition
identifier and source type; senior/youth level; competitive/friendly status;
minutes; starter/substitute status; team venue; stadium identifier; source URL;
and raw-response cache key.  Player-game duplicates are not silently summed.
They must be explained, resolved using the raw source, or cause the coverage
gate to fail.

The acquisition audit covers World Cup, continental championships, qualifying,
UEFA and CONCACAF Nations League, Gold Cup, AFCON, Asian Cup, Copa America,
senior friendlies, Olympics and U21/U23 matches.  It tabulates coverage by
competition, season, national team and EPL player.  The audit compares source
match totals with an official FIFA, UEFA or confederation schedule where an
official schedule is available, and includes a stratified manual validation
sample against official match reports.  Identified appearances with missing
minutes remain missing and are excluded from the relevant complete-case
calculation; they are never converted to zero.

The `club_plus_senior_national` scope may be used as the primary v4 exposure
only if all of these conditions hold:

1. at least 95% verified match coverage in the official-schedule audit;
2. at least 95% non-missing minutes among identified appearances;
3. zero unresolved cohort player IDs; and
4. zero unexplained duplicate player-game records.

If any condition fails, club-plus-country exposure is reported only as a
sensitivity analysis.  The coverage output records both the gate inputs and
the decision; no analyst may override it in the model script.

## Generic exposure and recovery engine

One engine creates the same variables for every exposure scope: prior 3-, 5-,
7-, 14- and 28-day minutes and appearance counts; days since previous observed
appearance; current-match minutes; consecutive-match sequence measures; and
national-team minutes within each window.  Source rows are a unified player
match chronology, not separately coded club and international rolling loops.

When both UTC kick-offs are present, recovery is measured as elapsed UTC hours.
Otherwise it is measured in whole calendar days and is labelled as such.  The
existing display categories are retained: 0--3, 4--5, 6--7, 8--14 and more than
14 days.  Calendar-day intervals are never described as a `less than 72 hours`
recovery measure.

## Geographic travel proxies

The v4 pipeline stores stadium coordinates only when they are matched to a
verified Wikidata item or another retained public source URL.  It records match
confidence and source provenance, then calculates great-circle distance and
time-zone change between consecutive *observed match venues*.  These variables
are geographic travel proxies, not observed player travel.  An unknown stadium
or coordinate remains missing, not zero kilometres or zero time-zone change.
Travel analyses are secondary regardless of coverage.

## Selection-risk-set analysis

Scheduled EPL fixtures, observed club-season membership, transfer dates and
canonical injury episodes define player-fixture opportunities plausibly
available for selection. Transfer bounds are tied to each observed stint:
latest arrival on or before its first appearance and earliest departure on or
after its last appearance. Appearance status is matched by player, fixture and
club. When public intervals name multiple clubs on one date, the observed club
resolves an appearance date; unresolved non-appearance dates are excluded.
The resolution audit must show one row per player-date before modelling. A
selection model predicts whether a player appears for any minutes using prior
information only. Stabilised inverse-probability weights are fitted only if the
audit shows adequate covariate overlap, acceptable post-weight balance and
stable weights. These models address measured selection inside an inferred
membership window; they cannot reconstruct a registered roster or remove
selection on symptoms, medical clearance or tactical decisions.

## Outcome-confidence audit

Transfermarkt remains the primary reproducible injury source. The v4 queue
stratifies severe, muscle/tendon, ambiguous and unmatched reports for official
source review. The current publication adds a separate deterministic sample of
30 same-day positives across three strata. Its queue hides exposure and fitted
results. One author checks club, league or contemporaneous news sources that
are independent of Transfermarkt. Code gates require exact queue membership,
unchanged descriptive fields, no exposure columns, an independent URL for
every resolved row, and explicit unresolved status otherwise. This audit does
not silently alter the outcome, estimate missed cases, provide clinical
diagnosis or provide inter-rater agreement.

## Prespecified analysis and interpretation

The direct v4 comparison uses previous-seven-calendar-day minutes and the
existing recovery-interval analysis.  It compares `frozen_club_all` and
`frozen_club_plus_senior_national` using the same outcome rows, prior-history
variables, spline basis, covariates and multiplicity family.  It reports the
number and percentage of EPL match rows whose seven-day burden changes, the
number of zero-burden rows reclassified, and whether model estimates and
conclusions change.  The name `primary v4 exposure` remains conditional on the
coverage gate; a failed gate labels every v4 comparison `sensitivity_only`.

Travel, 14-/28-day windows, youth appearances, friendlies, and selection
weighting are secondary or sensitivity analyses.  No new classification,
subgroup or alternative window becomes a primary result after examining the
results.  The mandatory first deliverables are the acquisition manifest,
coverage audit, unified timeline, baseline-parity report and model-comparison
report. The manuscript is not changed until those deliverables are complete
and their gate decisions are documented.

### Post-audit scientific extension

After the prespecified comparison showed no material total-burden change,
`src/29_public_data_v4_scientific_audit.py` evaluated recent senior competitive
country duty as a post-hoc, hypothesis-generating exposure. It
uses a between/within-player decomposition, controls the frozen club-load
spline and calendar terms, and tests 3-, 5-, 7-, 14- and 28-day windows,
outcome timing, description-specific reports, observed/fixed/per-match
denominators, measured covariates, continuous country minutes and match counts,
international-break context, and alternative prior-history definitions.

These analyses do not amend the prespecified primary exposure. They carry a
single explicit exploratory multiplicity family and cannot be promoted to a
causal conclusion after inspection. In the corrected build, no national-status
contrast survived adjustment; the result is Tier 5. The audit decision and
numerical results are recorded in
[`public_data_v4_scientific_audit.md`](public_data_v4_scientific_audit.md).

### Post-primary current-data robustness extension

`src/33_matchproxy_current_data_extensions.py` is a separate extension of the
frozen intermediate/higher-history match-proxy model. It runs after
`src/18_match_proxy_poisson_splines_perminute.py`; it does not acquire new
injury or exposure data and does not amend the primary previous-seven-day
club-minute estimand. Its purpose is to test whether the main measurement and
selection interpretation changes after directly observable checks.

The analyses are executed in this fixed order:

1. **Observed selection proxies.** Refit among lineup-known rows, starters,
   substitutes, rows outside 14 days of a recorded return, and starters outside
   that window. Test the recorded lineup-role-by-spline interaction. These are
   risk-set composition checks, not measures of medical clearance or fitness.
2. **Differential report detail.** Estimate type-classifiable public text by
   proxy timing, history stratum and lineup role. The inverse-probability
   reporting sensitivity may be interpreted only when fitted classification
   probabilities are at least 0.10 and maximum weights are at most 10. If either
   gate fails, its weighted predictions are archived as a stress test and do
   not correct the primary model. Player-bootstrap percentile intervals replace
   unsupported clustered weighted-GLM covariance.
3. **Reported absence-day proxy.** Summarise public dated-episode days per
   recorded match hour and model `1 + reported absence days` only as a reporting
   proxy. Do not call this clinical time loss, injury severity, or injury burden.
4. **Joint support.** Display previous-seven-day-minute by recovery cells before
   fitting the support-limited one-versus-two-prior-club-match comparison. Empty
   and sparse cells remain non-estimates; no two-dimensional response surface
   is inferred.
5. **Within-player-season comparison.** Use conditional logistic models in
   event-containing player-seasons. This controls stable player-season factors
   only within the observed match risk set; it does not solve time-varying
   health, medical, or tactical selection.
6. **Curve and uncertainty stability.** Use 1,000 player-resampling replicates to
   measure the probability and location of an early fitted global maximum, and
   report an additional player/current-match two-way cluster covariance model.
7. **Current-match context.** Reconcile all model rows to a unique source match
   and refit with current competition adjustment and among current
   Premier-League matches only.

The script writes 23 `matchproxy_extension_*.csv` files to
`data/processed/results/`. Every output contains its restriction, model family,
row/event support, or explicit non-estimability status. The data dictionary
lists the individual files. The manuscript may use these checks only to qualify
the pre-existing observational interpretation; it must not treat them as a
causal adjustment set.

### Reviewer-requested publication analysis

`src/34_jsams_referee_analysis.py` follows scripts 18 and 33 and preserves the
predecessor spline, cohort, denominator, outcome-quality, historical and
contextual checks. `src/36_jsams_second_referee_analysis.py` follows scripts 27
and 34 and defines the current manuscript estimand: the probability of a
same-day public report per appearance among established players who reached
that appearance. The exposure is previous-seven-day club minutes entered as an
additive linear term per 90 minutes. Continuous prior report history and
calendar phase are additive controls. No dated prospective plan exists; every
test is post-data and exploratory.

The current stage applies one symmetric treatment:

1. independently rebuild 3-, 5-, 7-, 10-, and 14-day prior-minute windows and
   stop if the seven-day reconstruction differs from the legacy field;
2. cross five minute windows, seven-day appearance count, and recovery interval
   with same-day, lag-1, and combined outcomes and with per-appearance,
   observed-minute, and fixed-90 denominators;
3. adjust all 63 focal tests as one Holm family;
4. standardise additive probabilities over observed calendar phase and use a
   10,000-draw simultaneous band for the non-monotonic spline sensitivity;
5. lock the additive model before three fixed temporal refits;
6. disclose player and player-season conditional target populations, exposure
   support, player-cluster intervals and 5,000 multiplier draws;
7. fit bounded inverse-selection weighting only after unique-opportunity,
   overlap, weight-tail and balance gates; and
8. validate the independent-source audit and append every test to one 706-row
   register before tier visibility rules are applied.

Continuous prior report history is primary; internally calibrated categories
and their unsupported interaction are secondary. The display range stops at
180 recent minutes. The flexible spline is a shape sensitivity, not evidence
of a rising threshold. None of these analyses turns the public outcome into
clinical surveillance or removes unmeasured medical and tactical selection.
