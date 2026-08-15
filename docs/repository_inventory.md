# Repository inventory and source-of-truth map

This file is the canonical index for the repository as a working research
project. It separates current analysis artifacts from historical releases and
from large local files that Git intentionally excludes. The detailed operator
guide remains [`../README.md`](../README.md), and field-level result definitions
remain in
[`public_data_v4_data_dictionary.md`](public_data_v4_data_dictionary.md).

## Authority order

When two artifacts appear to disagree, use this order:

1. The current Python source and its tests define computation.
2. Generated CSV registries and audit gates define the current numerical state.
3. `manuscript/manuscript.tex` defines the submitted scientific narrative.
4. The README and current audit documents explain operation and interpretation.
5. Zenodo v2/v3 files describe immutable historical releases only.

The current publication hierarchy is
`data/processed/results/jsams_revised_claim_hierarchy.csv`. The current formal
test index is `jsams_revised_hypothesis_register.csv`. Earlier
`v4_result_tier_registry.csv`, `jsams_claim_hierarchy.csv`, and
`jsams_hypothesis_register.csv` are retained as provenance for predecessor
analysis stages; they do not override the revised files.

## Verified checkout snapshot

The local checkout was audited on 7 August 2026 after rerunning the synchronized
current-paper chain: scripts 27, 18, 33, 31, 32, 34, 36, 37, 35, 00, and 24 with
`.venv312`. Counts below describe that verified local snapshot and will change
when raw caches or generated analyses are rebuilt.

| Area | Files | Bytes | Git policy |
| --- | ---: | ---: | --- |
| `src/` | 39 Python modules | 1,397,340 | Versioned |
| `tests/` | 41 `test_*.py` files plus `conftest.py` | 521,207 | Versioned |
| `docs/` including this index | 9 files | 177,146 | Versioned |
| `data/raw/` | 1,601 | 3,524,344,717 | Local/ignored; manifests retain provenance |
| `data/processed/` | 371 | 1,627,710,055 | Local/ignored; reproducible derived data |
| `external_data/` | 10 | 555,824,911 | Local/ignored provider snapshot |
| `output/` | 374 | 1,080,478,520 | Local/ignored release and build staging |
| `public_deposit/` | 307 | 805,376,842 | Local/ignored sanitized export |
| `manuscript/figures/` | 6 PNG files | 2,194,805 | Versioned paper assets |

The figure count fell from 58 to 6 when the reframe purged every figure the
current paper does not display. Anything cited by an earlier draft lives in the
Zenodo releases, not here.

## Code index

Every module under `src/` has a matching test module under `tests/`. The two
additional test files cover `config.py` and manuscript tier visibility. The
README's **Script Reference** describes every executable module individually.

| Code group | Role | Current status |
| --- | --- | --- |
| `src/01`--`src/07`, `src/09`, `src/13`, `src/16` | Core Transfermarkt ingestion, episode reconciliation, panel construction, rolling exposure, prior-only history, and match-proxy outcomes | Required build path |
| `src/08`, `src/10`--`src/12`, `src/14`, `src/17`--`src/23` | Historical models, descriptive checks, FFT, diagnostics, clinical bridge, and prior-duration analyses | Retained secondary or diagnostic path |
| `src/24_prepare_public_deposit.py` | Neutral-label export of generated result CSVs and manuscript figures | Current release utility |
| `src/25_public_data_v4.py`, `src/25b_acquire_public_sources.py`, `src/26`--`src/30` | v4 acquisition, independent public-source caches, national timeline, quality gates, and sensitivity models | Required v4 build path |
| `src/31_public_data_v4_quality_registry.py` | Combined v4 and current-data quality/tier registry | Run after script 33 |
| `src/32_plot_v4_paper_figures.py` | The two Supplement figures `I1` and `I2` | Required for a complete submission; produces no main-paper figure |
| `src/33_matchproxy_current_data_extensions.py` | Current-data robustness and measurement audits used by script 34 | Required predecessor input |
| `src/34_jsams_referee_analysis.py` | Forty-one cohort, denominator, quality, context, bootstrap, and predecessor registry tables | Required predecessor layer |
| `src/36_jsams_second_referee_analysis.py` | Current estimand, 63-model family and its distribution, exposure-metric correlations, measured-confounding and clustering refits, absolute-risk contrast and support, temporal and conditional analyses, restricted measured-selection check with no-leakage gates, independent-source audit, player-clustered gamma intervals and player-resampled attenuation intervals in every lineup stratum, truncation-imputation sensitivity, squad-role-adjusted association and denominator refit, exposure-window gradient, run-in exclusion comparison, and controlling registries | Current analysis authority |
| `src/37_denominator_gradient.py` | Denominator gradient, decision rule, cross-league summary, floor and estimator sensitivities, the specification register, and the scoping and adjudication protocols, fitted from appearance records alone | Current diagnostic authority |
| `src/35_plot_jsams_revision.py` | Six figures: two main-paper (`J1`, `J2`) and four Supplement (`J3`--`J6`), from scripts 34, 36 and 37; each written as PNG for the sources and vector PDF for the journal's accepted formats, with a deposited manifest of drawn labels that the gates compare against the current tables | Current figure authority |
| `src/00_list_result_columns.py` | Schema inventory for every processed CSV | Run after all analyses |
| `src/pipeline_io.py`, `src/public_data_sources.py`, `src/v4_statistics.py` | Shared IO, acquisition, and statistical helpers | Imported support modules |

The minimal dependency chain for the current paper is:

```text
core build -> script 18 -> scripts 25, 25b, 26--30 -> script 33
           -> script 31 -> script 32 -> script 34 -> script 36 -> script 37
           -> script 35 -> script 00 -> script 24
```

Script 32 is part of this chain, not an optional extra: it writes the two
figures `manuscript/supplement.tex` includes (`I1`, `I2`). Script 35 writes six
figures: the two the main paper displays (`J1`, `J2`) and four the Supplement
displays (`J3`--`J6`). The main paper kept only the two that carry the
denominator claim when it committed fully to that claim; the rest moved to the
Supplement rather than being dropped. Both scripts are needed for a complete
submission. Script 27 is part of the 26--30 block; scripts 27 and 34 must both
be current before script 36. Script 37 must be current before script 35, which
reads its cross-league gradient table to draw figure `J2`. Script 37 reads no
other script's output --- only the appearance snapshot --- so it can be rerun
on its own, and it is placed after 36 in the chain only because 35 follows
both. The numbered commands in the README give the full
executable order, including historical analyses that are not required to
regenerate the eight submitted figures.

Source numbering runs `00`--`37` with **no `src/15`**. The gap is intentional:
`src/15_plot_hazard_and_fft_for_paper.py` was removed in commit `3fc1374`
(11 June 2026) when its hazard and FFT plotting was consolidated into
`src/19_plot_everything_for_paper.py`. The number was not reused.

## Data index

| Location | Contents | Canonical provenance/index |
| --- | --- | --- |
| `external_data/transfermarkt/` | Ten dated Transfermarkt dataset CSVs used by the core build and context/lineup checks | `data/raw/public_data_v4/baseline_manifest.json` records hashes for the core files used to freeze v4 |
| `data/raw/` | Injury-page cache, core raw tables, immutable v4 snapshots, 1,208 player-performance caches, and independent-source snapshots | Snapshot-specific JSON manifests under `data/raw/public_data_v4/` |
| `data/manual/` | The deposited form of the hand-built evidence: the exposure-blinded same-day outcome audit, the report-free queue review, and the denominator scoping search. The two audit files are keyed by surrogate and carry no name, provider identifier, match date or source URL; `src/38_deidentify_audit_evidence.py` writes them from `data/private/` | `independent_same_day_event_audit.csv`, `independent_non_event_audit.csv`, `per_minute_denominator_scoping.csv`; `.gitignore` negates the whole directory rather than listing them, so the next hand-built file is versioned by default |
| `data/private/` | The reviewer's own identified copy of both audits and the surrogate map that reverses them. Tracked here so the evidence survives a lost machine; carried by no deposit builder, both of which read `data/manual` and `data/processed` only | `independent_*_audit.csv`, `audit_identity_map.csv`, `deposit_player_map.csv` |
| `data/processed/diagnostics_private/` | Row-level residual frames, one row per player-day with a provider identifier beside an injury description. Nothing reads them back and they verify no reported number; they sit outside every exported subtree so a diagnostic cannot publish special-category data as a side effect | `logit_residuals.csv`, `poisson_matchproxy_residuals.csv` |
| `data/processed/` | Core player-day/player-match panels, cleaned reports, canonical episodes, prior-history tables, v4 tables, and analysis results | `data/processed/results/columns_inventory.csv` and `.txt` index all 276 processed CSV schemas |
| `data/processed/public_data_v4/` | Forty-four acquisition, reconciliation, status, exposure, model, quality, and tier tables | The v4 data dictionary defines every table and gate |
| `data/processed/results/` | Current and historical model outputs, diagnostics, and plot inputs | Prefix ownership below; schema inventory is regenerated last |

Raw provider data and large generated panels are not committed. Missing raw
files are not represented as zeros, and ignored data are not silently replaced
by the published sanitized deposit. A clean clone therefore needs the documented
provider inputs or a compatible archived dataset before the full pipeline can
run.

One deliberate exception exists. Everything under `data/` is reproducible by
rerunning the pipeline **except** `data/manual/`, which holds the three files a
human made and no script can rebuild:

| File | What it records |
| --- | --- |
| `independent_same_day_event_audit.csv` | One author's exposure-blinded adjudication of 30 sampled reports. The only evidence for a Tier 2, abstract-visible claim. The deposited copy records whether an independent source was found and of what kind; the URL itself is withheld, because most of these slugs carry the player's surname and one carries a graded diagnosis. |
| `independent_non_event_audit.csv` | The same protocol applied to the 30 report-free appearances, and de-identified the same way. Every verdict is `unresolved`: the searches that could be run failed on identification, so the file records that the search happened and settled nothing. |
| `per_minute_denominator_scoping.csv` | The scoping search for published studies using minute denominators on public data, with the verbatim denominator sentence and source URL per record. Reported as a floor, not a systematic count. |

`.gitignore` negates the directory as a whole rather than naming the files, and
`src/24_prepare_public_deposit.py` exports it. Enumerating them was a real
defect: two of the three were silently ignored until the negation was widened.

### Reading a result CSV: check its cohort first

`data/processed/results/` holds current and predecessor outputs side by side,
and some answer the same question on different cohorts. Prefix ownership above
gives the authority; cohort size is the quickest discriminator.

| Cohort | Rows | Written by | Example |
| --- | ---: | --- | --- |
| Current per-appearance cohort | 88,573 appearances, 1,208 players | Scripts 33, 34, 36 | `matchproxy_type_history_recency_attenuation.csv` (muscle/tendon step 1.84 to 1.31) |
| Predecessor per-minute cohort | 80,598 match rows, 1,063 players | Script 18 | `matchproxy_negative_control_mutually_exclusive_type_binary.csv` (2.55 to 1.53) |

Both are correct for their own cohort, and the manuscript and Supplement quote
the current one. Confirm `n_match_rows`/`n_players` before reusing any value.

## Result index

The verified result directory contains 249 top-level CSVs and three diagnostic
CSVs, for 252 result CSVs in total. It also contains 54 generated result
figures, six diagnostic PNGs, and `columns_inventory.txt` (305 files total).

| Prefix/location | Owner | Count or role |
| --- | --- | --- |
| `jsams_revised_*` | Scripts 36 and 37 | 52 current tables; controlling analysis layer |
| `jsams_*` excluding `jsams_revised_*` | Script 34 | 41 predecessor/current-support tables |
| `matchproxy_extension_*` | Script 33 | 23 robustness and measurement tables |
| Other `matchproxy_*`, `poisson_*`, and historical prefixes | Scripts 16--23 | Auditable predecessor, sensitivity, and diagnostic outputs |
| `data/processed/public_data_v4/*.csv` | Scripts 25--31 | 44 v4 tables |
| `columns_inventory.csv` and `.txt` | Script 00 | File-level schema index for all processed CSVs |

The forty-two controlling script-36 outputs are:

```text
jsams_revised_recorded_minute_distribution.csv
jsams_revised_lineup_composition_by_exposure.csv
jsams_revised_denominator_by_lineup_role.csv
jsams_revised_denominator_attenuation_decomposition.csv
jsams_revised_direct_truncation_refit.csv
jsams_revised_case_restricted_exposure_bias.csv
jsams_revised_lineup_coverage_denominator_stability.csv
jsams_revised_event_clustering_summary.csv
jsams_revised_model_field_completeness.csv
jsams_revised_episode_type_composition.csv
jsams_revised_window_validation.csv
jsams_revised_exposure_multiverse.csv
jsams_revised_exposure_multiverse_summary.csv
jsams_revised_exposure_metric_correlations.csv
jsams_revised_exposure_metric_summary.csv
jsams_revised_additive_curves.csv
jsams_revised_additive_curve_tests.csv
jsams_revised_absolute_risk_contrast.csv
jsams_revised_exposure_support.csv
jsams_revised_history_reference_value.csv
jsams_revised_confounding_sensitivity.csv
jsams_revised_club_congestion_sensitivity.csv
jsams_revised_run_in_threshold_sensitivity.csv
jsams_revised_placebo_window_analysis.csv
jsams_revised_negative_control_outcomes.csv
jsams_revised_ascertainment_by_exposure.csv
jsams_revised_temporal_stability.csv
jsams_revised_conditional_estimates.csv
jsams_revised_conditional_population.csv
jsams_revised_conditional_support.csv
jsams_revised_appearance_selection_estimates.csv
jsams_revised_appearance_selection_diagnostics.csv
jsams_revised_appearance_selection_population.csv
jsams_revised_denominator_contrast_metadata.csv
jsams_revised_outcome_audit_queue.csv
jsams_revised_outcome_audit_validation.csv
jsams_revised_outcome_audit_summary.csv
jsams_revised_non_event_absence_screen.csv
jsams_revised_non_event_audit_queue.csv
jsams_revised_non_event_audit_summary.csv
jsams_revised_second_assessor_agreement.csv
jsams_revised_claim_hierarchy.csv
jsams_revised_hypothesis_register.csv
```

The nine tables added in an earlier revision test whether the outcome
behaves like an injury or like a report. `placebo_window_analysis` fits a
negative-control exposure (minutes 31--37 days earlier) alone and mutually
adjusted with the recent window; `negative_control_outcomes` contrasts
outcome definitions and records that the prespecified illness control had one
event and is not estimable; `ascertainment_by_exposure` tests whether
reporting detail varies with exposure. `club_congestion_sensitivity` and
`run_in_threshold_sensitivity` separate the player's own minutes from his
club's fixture calendar and from the 900-minute run-in choice.
`non_event_audit_queue`, `non_event_absence_screen`, `non_event_audit_summary`
and `second_assessor_agreement` carry the two quantities the audit cannot
estimate --- missed events and inter-rater kappa --- and report them as
unestimated rather than substituting a weaker proxy. The queue has now been
searched under the deposited protocol and returned `unresolved` on all 30
records, so `non_event_audit_summary` reads `reviewed, not resolved` with an
undefined missed-event proportion; zero confirmed missed events is the absence
of an estimate, not an estimate of zero. `non_event_absence_screen` explains
why so little was searchable: it counts the fixtures each player's own club
played between the queued appearance and the player's next one, and in 28 of 30
that count is zero, so no absence existed for a report to have missed. The
screen assigns no verdicts and carries no verdict column, because a player can
be injured and miss nothing.
`history_reference_value` records the median prior-history rate at which
standardised probabilities are held.

The two tables added in the sixth round exist because the fifth round's
attribution leaned on a first-order identity that turns out to
over-predict. `direct_truncation_refit` fits one Poisson model three
times, changing nothing but the offset, so the recorded-minute against
untruncated-minute gap measures outcome truncation without any expansion:
-0.095% of the attenuation, agreeing with the identity's -0.13%.
`case_restricted_exposure_bias` then answers where truncation does bite,
since a whole-cohort denominator dilutes it to 0.24% while a
case-restricted quantity carries the full 34.6%.

The eight tables added in the fifth round exist to test the paper's own
mechanism, and they refuted it. `recorded_minute_distribution` shows the
truncation as a distribution rather than a mean, which is what licenses the
word truncation at all. `denominator_attenuation_decomposition` then applies
the first-order identity `b_off = b_app - gamma` and finds that outcome
truncation explains -0.13% of the attenuation, because the minutes it removes
are 0.243% of the total. `lineup_composition_by_exposure` and
`denominator_by_lineup_role` identify the real cause: recorded minutes rise
with the exposure through squad role, and the pooled attenuation of 0.151 on
the log scale falls to 0.012 within starters.
`lineup_coverage_denominator_stability` shows the minute difference is the
same inside and outside the complete-lineup era, so the starter/substitute
split is not a coverage artefact. `event_clustering_summary`,
`model_field_completeness` and `episode_type_composition` describe how events
cluster within players, that no eligible row is lost to listwise deletion, and
what the source records across all 11,993 episodes.

The seven tables added in an earlier revision all qualify the reference
estimate rather than support it: `exposure_multiverse_summary` reports the
distribution of the 63 estimates instead of a rejection count,
`exposure_metric_correlations` shows that the cumulative windows overlap,
`confounding_sensitivity` refits the reference model with measured covariates
and alternative clustering, `absolute_risk_contrast` and `exposure_support`
give the absolute contrast and the data behind the curve,
`appearance_selection_population` compares included with excluded appearances,
and `denominator_contrast_metadata` records how the recorded-minute comparison
was estimated and why it carries no causal reading.

The local sanitized export contains 303 payload artifacts: 297 neutralised CSVs
and 6 figures, plus `sanitization_manifest.csv` (304 files total). The CSVs are
248 from `results/`, 44 from `public_data_v4/`, and the one `data/manual/`
adjudication file, so every artifact the manuscript or Supplement cites is
present. It is a working export, not a published v4 Zenodo version.

## Manuscript and figure index

`manuscript/manuscript.tex` is the clean, author-visible main source;
`manuscript/manuscript.pdf` is its versioned compiled counterpart.
`supplement.tex` and `title_page.tex` are versioned sources, while their local
PDFs and the anonymous line-numbered review PDF are ignored build outputs.
`manuscript_blind.tex` and `supplement_blind.tex` are the versioned wrappers
that produce the anonymized review copies the double-anonymized process
requires --- both documents travel to referees, so both blind --- and
`cover_letter.md` carries the journal-fit argument and the ethics and
data-protection position to the editor; a gate asserts all three exist and
say what they must. The institution's name appears in the manuscript body
only through the `\ethicsinstitution` macro, whose blind branch redacts it
and drops the institutional-guidance citation with it, so the anonymized
copies carry neither author names nor the institution. References are set in
Vancouver format (`vancouver.bst`, citation-order numbering); journal-name
abbreviation is left to the publisher's production stage rather than done by
hand, which would risk exactly the class of reference error round fourteen
existed to remove.

Word, abstract, display-item and reference counts are no longer recorded here
or anywhere else by hand: `tests/test_manuscript_tier_visibility.py` computes
them from `manuscript.tex` on every run, asserts them against the journal's
limits (5,000 main-text words, 250 abstract words, three to five Practical
Implications bullets, six combined tables and figures, forty references, at
most three references per citation point), and asserts that the counts the
title page states equal the computed ones. A hand-recorded count survives
edits that a computed one does not, which is how this paragraph once carried
a word count from two papers ago. These journal-style counts use
TeXcount's text and heading fields rather than adding inline mathematics as
extra words; the generated slices exclude material according to the documented
journal-counting rules.

The main manuscript includes only these four figures, all generated by script
35:

```text
J1_jsams_cohort_measurement.png
J2_jsams_primary_robustness.png
J3_jsams_within_player_lineup_coverage.png
J4_jsams_context_support.png
```

The remaining 54 versioned PNGs are supplementary, diagnostic, or historical
assets. They are retained for traceability but are not evidence that the main
paper uses those analyses. LaTeX include checks confirmed that all four current
figure paths and `references.bib` resolve.

## Documentation index

| File | Scope | Status |
| --- | --- | --- |
| `README.md` | Installation, inputs, pipeline order, script reference, outputs, figures, tests, and limitations | Current operator guide |
| `repository_inventory.md` | Authority, file ownership, local/versioned policy, and current counts | Current canonical index |
| `public_data_v4_protocol.md` | Frozen protocol for the national-exposure extension | Historical protocol; not the current model plan |
| `public_data_v4_data_dictionary.md` | Field and artifact definitions across v4 and current outputs | Current reference |
| `public_data_v4_scientific_audit.md` | Scientific interpretation of v4 and later checks | Current audit |
| `strobe_siis_checklist.md` | Reporting checklist and unresolved external submission actions | Current working checklist |
| `arxiv_preprint_release.md` | Source-package and preprint replacement procedure | Current release guide |
| `zenodo_v2_release.md` plus v2 metadata/manifest | Immutable v2 publication record | Historical |
| `zenodo_v3_release.md`, v3 metadata/manifest, and `zenodo_referee_revision_metadata.json` | Immutable v3 publication record | Historical |

## Release-state distinction

Zenodo v3.0.0 is the latest published software/dataset pair. `.zenodo.json` and
`CITATION.cff` intentionally identify that published release. The present
checkout contains later v4 acquisition, revised analyses, and a newer
manuscript, so the v3 version-specific DOIs do **not** exactly reproduce the
current paper. Before releasing this checkout, create new software and derived-
output versions under the existing concept DOIs, update their exact metadata,
and then update any preprint or journal data-availability statement.

## Audit gates

The following checks were completed before this index was written:

- 40 source modules have matching tests; there are no unmatched source files.
- Every directly imported third-party package is declared; `scipy` and `patsy`
  are explicit rather than accidental `statsmodels` transitive dependencies.
- All local links in extant versioned Markdown files resolve.
- All six LaTeX figure includes resolve: `J1`--`J4` in `manuscript.tex` from
  script 35, and `I1`--`I2` in `supplement.tex` from script 32. Both
  bibliographies resolve.
- All 41 script-34 and all 23 script-36 expected CSVs exist.
- All 44 v4 CSVs exist; seven that were missing from the old README list are
  now documented.
- Script 27 rebuilt a unique 198,486-row bounded opportunity table from
  198,543 source rows; script 36 then regenerated all current outputs.
- Script 35 regenerated the four main figures without changing their tracked
  bytes, showing that the final source reproduces the committed figures.
- Script 00 regenerated the 276-file processed-CSV schema inventory.
- Script 24 regenerated the 326-artifact sanitized payload, which now includes
  the `public_data_v4` tables and the `data/manual` adjudication file.
- The clean, anonymous-review, supplement, and title-page TeX targets compiled;
  all pages were rendered and visually checked for clipping and broken glyphs.
- Pytest collected 229 tests and reached 100% statement and branch coverage
  across 7,559 statements and 1,958 branches.
- Rerunning script 36 changed one of its sixteen outputs
  (`jsams_revised_claim_hierarchy.csv`, the intended `main_display_recommended`
  correction) and left the other fifteen byte-identical, confirming that the
  stage is deterministic under `RANDOM_SEED`.

Run the full test and manuscript-build commands in the README after any source,
data, result, or manuscript change. A successful test run and clean LaTeX build
complete the audit; documentation counts alone do not establish scientific
correctness.

## Revision, 6 August 2026

A reviewer returned "major revision necessary" on the 21-page draft, accepting
the study as coherent and transparent but objecting that the paper gave the
cumulative-minute association more inferential prominence than a post-data
design supports. All eight major and seven moderate points were addressed in
the pipeline first, then in the manuscript.

| Reviewer point | Pipeline response | Manuscript response |
| --- | --- | --- |
| 1. Inference remains post-data | `exposure_multiverse_summary` reports the median, quartiles and range of the 63 estimates | Abstract, Results and Discussion lead with the distribution (median 1.16, IQR 1.06--1.23); "survived correction" removed |
| 2. Four windows are dependent evidence | `exposure_metric_correlations` gives Pearson and Spearman for all 21 pairs | Seven days named the single reference window; others described as correlated sensitivities (r up to 0.85); "four of seven" removed from the abstract |
| 3. Measured confounding not visible | `confounding_sensitivity` refits the reference model with age, position, club-season, competition and season, plus player-match and club-season clustering | New Figure J3 panel A in the main paper, with a Supplement table |
| 4. Audit conditions on verification | `summarize_outcome_audit` adds the all-sampled estimand and partial-identification bounds | Both estimands reported: 24/27 resolved (88.9%) and 24/30 sampled (80.0%, 62.7--90.5), bounds 80--90% |
| 5. Weighting is not squad-level selection | Leakage gates prove every covariate is pre-fixture and that all 401 Premier League same-day starts were retained; new `appearance_selection_population` table | Relabelled a restricted measured-selection sensitivity throughout, with the included/excluded comparison reported |
| 6. Absolute predictions underspecified | `absolute_risk_contrast` and `exposure_support` give the target population, the 0-to-180 difference and the support behind the curve | Absolute rise 5.6 to 8.9 per 1,000 (difference 3.3, 1.3--5.4); exposure-support band added beneath Figure J2 panel A |
| 7. Denominator described causally | `denominator_contrast_metadata` records estimator, clustering, complete-lineup seasons and standardisation, and states that no causal reading is licensed | "shortened recorded exposure" replaced with association wording in text, figure title and caption; Methods now state the estimator |
| 8. Conditional models called "support" | None required | Section and figure retitled sensitivity; non-collapsibility stated explicitly |

Moderate points: the title and outcome wording now say "injury/absence spell
starts"; Table 1 carries a denominator column; the Introduction distinguishes
this audit from Krutsch's severe-injury protocol and Szymski's prospective
multi-source Bundesliga study; Figures J3 and J4 were enlarged; the practitioner
message leads the Practical implications; the abstract is 249 words; and the
ethics paragraph states plainly that no determination was obtained and that
retrospective approval is not available.

**External action still outstanding.** An institutional determination that no
review is required would reduce editorial risk, and only the authors can obtain
it. The manuscript does not imply that one exists.

## Contributor audit, 6 August 2026

A full-contributor audit reviewed structure, pipeline, data, tests, docs and
release state, and reconciled roughly fifty manuscript and Supplement values
against generated CSVs. No scientific defect was found: every headline number
reconciled, the seven-day exposure parity gate passed on all 88,573 rows, and
the suite passed at 100% statement and branch coverage. Six repository defects
were found and fixed.

| # | Defect | Fix |
| --- | --- | --- |
| 1 | `data/manual/independent_same_day_event_audit.csv` was untracked, though it is the only non-regenerable input and the sole evidence for a Tier 2 abstract claim. This index previously described it as versioned, which was false. | `.gitignore` now negates the file; this index states the true policy. |
| 2 | The deposit exported only `data/processed/results`, so two artifacts the Supplement cites by name were absent. | `src/24_prepare_public_deposit.py` also exports `data/processed/public_data_v4` and `data/manual`; a regression test covers both. |
| 3 | Script 32 was documented as optional and historical, but writes the two figures `supplement.tex` includes. Its own docstring called them main-paper figures. | Script 32 is now in the documented chain in this index and the README, and its docstring describes the Supplement correctly. |
| 4 | Current and predecessor cohorts sat side by side in `results/` with no quick way to tell them apart. | The data index now carries a cohort-discriminator table with worked examples. |
| 5 | `jsams_revised_claim_hierarchy.csv` marked `appearance_selection_weighting` as not main-display-recommended while it appears as Figure J4 Panel B. | The flag is corrected to `True` with a stated justification, the two visibility axes are documented in `revised_claim_hierarchy`, and `tests/test_manuscript_tier_visibility.py` now pins the main-display axis to the manuscript. |
| 6 | `src/15` was missing with no explanation; stray `.codex_jsams34_*.log` files sat in the repository root; `referee_report.md` was tracked but deleted in the working tree. | The gap is explained above; the logs are removed; the stale report, which reviewed the superseded manuscript, is deleted in-tree and remains recoverable at `git show 29f1b0f:referee_report.md`. |
