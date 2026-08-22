"""Gates tying the manuscript's claims to the tables that generate them.

The paper's contribution is a measurement, so every number it argues from has
to come from a deposited table and has to arrive with its uncertainty. These
tests read the tables, not the prose, and fail when the two drift apart. They
also refuse the specific overclaims that successive drafts have introduced:
saying a contamination is removed when its interval excludes zero, saying an
artefact reverses a conclusion when it only nullifies one, and saying an
effect is uniform across strata whose signs disagree.

The second group enforces the target journal's own rules, which are recorded
in ``docs/smf_author_guide.md``. Where a test and that file disagree the file
wins and the test is wrong, because the file is a transcript of what the
journal requires and the test is only our reading of it. The journal
desk-rejects a manuscript that claims to follow a reporting guideline without
following it, so the checklist is gated rather than trusted.

Interval formatting: prose writes ``(0.283 to 0.323)`` because a bound can be
negative and an en-dash between two negative numbers is unreadable; tables
write ``(0.283--0.323)`` because a column has no room. Gates accept either,
via ``_interval``, so the choice stays an editorial one.
"""

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
MANUSCRIPT = ROOT / "manuscript" / "manuscript.tex"
SUPPLEMENT = ROOT / "manuscript" / "supplement.tex"
TITLE_PAGE = ROOT / "manuscript" / "title_page.tex"
COVER_LETTER = ROOT / "manuscript" / "cover_letter.md"
APPENDIX = ROOT / "manuscript" / "appendix_statistical_output.txt"
GUIDE = ROOT / "docs" / "smf_author_guide.md"
RESULTS = ROOT / "data" / "processed" / "results"

# Science and Medicine in Football, Original Investigation.
WORD_LIMIT = 4500
ABSTRACT_LIMIT = 250
KEYWORD_RANGE = (1, 6)

#: The journal states no numeric limit on displays, so this is a house
#: convention recorded in docs/smf_author_guide.md rather than a journal rule.
#: Six is the working ceiling; the manuscript must use at least half of it,
#: because this paper's arguments are visual ones and hiding a gradient, a
#: calibration or a decision rule in prose to look austere serves nobody.
FIGURE_CEILING = 6
FIGURE_FLOOR = FIGURE_CEILING // 2
TABLE_COUNT = 3

#: Figure files, in the order the manuscript first cites them. The extension is
#: deliberate: the manuscript embeds the vector PDFs, because the journal asks
#: for 1200 dpi line art and a vector drawing satisfies that at any size.
MAIN_FIGURES = (
    "J1_jsams_cohort_measurement.pdf",
    "J8_jsams_identity_calibration.pdf",
    "J5_jsams_negative_control_exposure.pdf",
    "J7_jsams_gradient_by_population.pdf",
    "J9_jsams_decision_rule.pdf",
)
SUPPLEMENT_FIGURES = (
    "J2_jsams_denominator_gradient.png",
    "J3_jsams_within_player_lineup_coverage.png",
    "J4_jsams_context_support.png",
    "J6_jsams_primary_robustness.png",
)


def _between(text: str, start: str, end: str) -> str:
    """Return manuscript text between two unique markers."""
    return text.split(start, 1)[1].split(end, 1)[0]


def _interval(value: float, low: float, high: float, places: int = 3) -> tuple[str, str]:
    """The two accepted renderings of one estimate and its interval.

    The closing bracket is deliberately absent: some parentheses carry a p-value
    after the interval, and requiring the bracket would reject a correctly
    reported estimate for being followed by more of itself.
    """
    body = f"{value:.{places}f}"
    return (
        f"{body} ({low:.{places}f} to {high:.{places}f}",
        f"{body} ({low:.{places}f}--{high:.{places}f}",
    )


def _carries(text: str, value: float, low: float, high: float, places: int = 3) -> bool:
    return any(form in text for form in _interval(value, low, high, places))


def test_paper_is_positioned_against_the_work_it_extends():
    """The contribution completes a numerator paper and an analytical one, so
    both must be cited and the gap stated rather than implied."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    for key in (
        "hoenig2022citizen_transfermarkt",   # the numerator cannot be trusted
        "shrier2022_causal_workload",        # the offset is a constrained covariate
        "schisterman2009_overadjustment",    # the bias has a standard name
        "naimi2013_healthy_worker",          # and a precedent outside sport
        "stovitz2012_exposure_time",         # the denominator, measured correctly
        "wang2024_immortal_time_load",       # the same bias family, same group
        "mandorino2026_minutes_in_legs",     # the gradient, found by other means
        "krutsch2020media_validity",         # the severity skew is not new
        "sprouse2025_fa_womens_surveillance",  # why not gold-standard women's data
        "ortiz2026_acl_top_five",            # a live study the argument applies to
    ):
        assert key in text, key

    intro = _between(text, "\\section{Introduction}", "\\section{Materials and methods}")
    # The gap this paper fills has to be stated, not implied.
    assert "not how large it is" in intro
    # And it has to be stated accurately. An earlier draft claimed the
    # denominator had never been examined, which is contradicted by a paper the
    # argument leans on: Stovitz and Shrier examined whether exposure time is
    # measured correctly. The distinction between that question and this one is
    # what makes the contribution survive contact with the literature, so the
    # sweeping version is forbidden and the precise one required.
    assert "The denominator has not been examined at all" not in text
    assert "whether it is neutral" in intro
    assert "even when measured exactly" in intro
    # And the aims must be stated as aims, because the journal requires the
    # abstract's aims to be the manuscript's aims.
    assert "We aimed to" in intro


def test_abstract_carries_the_measured_claim_with_uncertainty():
    text = MANUSCRIPT.read_text(encoding="utf-8")
    abstract = _between(text, r"\begin{abstract}", r"\end{abstract}")

    decomposition = pd.read_csv(
        RESULTS / "jsams_revised_denominator_attenuation_decomposition.csv"
    ).set_index("quantity")

    boot = pd.read_csv(RESULTS / "jsams_revised_attenuation_bootstrap.csv").set_index(
        "quantity"
    )
    row = boot.loc["all"]
    assert _carries(abstract, row["estimate"], row["ci_low"], row["ci_high"])

    summary = pd.read_csv(
        RESULTS / "jsams_revised_denominator_gradient_summary.csv"
    ).set_index("quantity")["value"]
    assert f"{summary['gamma_pooled_min']:.3f} to {summary['gamma_pooled_max']:.3f}" in abstract

    # The women's range is the reason the paper can say the mechanism is not a
    # fact about men's football, so the abstract carries it too.
    contrast = pd.read_csv(
        RESULTS / "jsams_denominator_gradient_population_contrast.csv"
    ).set_index("population")
    women = contrast.loc["women"]
    assert (
        f"{women['min_gamma_pooled']:.3f} to {women['max_gamma_pooled']:.3f}" in abstract
    )
    assert int(contrast["leagues"].sum()) == 15
    assert "15 leagues" in abstract


def test_every_lineup_stratum_reaches_the_main_text():
    """A reader must not have to open the Supplement to learn that the remedy
    fails outside the starter stratum."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    roles = pd.read_csv(RESULTS / "jsams_revised_denominator_by_lineup_role.csv")
    per_role = roles.drop_duplicates("lineup_role").set_index("lineup_role")

    for role in ("all", "starting_lineup", "substitute_list", "lineup_unavailable_or_other"):
        row = per_role.loc[role]
        assert f"{int(row['n_rows']):,}" in text, role
        assert str(int(row["n_events"])) in text, role
        assert f"{float(row['log_attenuation_fixed90_minus_recorded']):.3f}" in text, role

    decomposition = pd.read_csv(
        RESULTS / "jsams_revised_denominator_attenuation_decomposition.csv"
    ).set_index("quantity")
    strata = (
        "gamma_log_minutes_on_exposure",
        "gamma_within_starting_lineup",
        "gamma_within_substitute_list",
        "gamma_within_lineup_unavailable_or_other",
    )
    for quantity in strata:
        row = decomposition.loc[quantity]
        assert _carries(text, row["value"], row["ci_low"], row["ci_high"]), quantity

    # The within-starter interval must be separated from every other, and the
    # paper must say so rather than leaving the reader to compare bounds.
    starter_high = float(decomposition.loc["gamma_within_starting_lineup", "ci_high"])
    for quantity in strata:
        if quantity != "gamma_within_starting_lineup":
            assert starter_high < float(decomposition.loc[quantity, "ci_low"]), quantity
    assert "disjoint from all three" in text
    assert "clustered on player identifier" in text


def test_mens_cross_league_gradients_match_the_generated_table():
    """The men's per-league detail moved to the Supplement when the main text
    took on two populations, so the binding follows it there. The main text
    still states the range, and the range has to be the table's."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    leagues = pd.read_csv(RESULTS / "jsams_revised_denominator_gradient_by_league.csv")
    assert len(leagues) == 8

    for _, row in leagues.iterrows():
        assert str(row["league"]) in supplement, row["league"]
        assert f"{int(row['n_appearances']):,}" in supplement
        assert _carries(
            supplement, row["gamma_pooled"], row["gamma_pooled_ci_low"],
            row["gamma_pooled_ci_high"],
        ), row["league"]
        assert _carries(
            supplement, row["gamma_within_starters"],
            row["gamma_within_starters_ci_low"], row["gamma_within_starters_ci_high"],
        ), row["league"]

    rule = pd.read_csv(RESULTS / "jsams_revised_denominator_gradient_decision_rule.csv")
    assert rule["recommendation"].eq("restrict to starters").all()
    assert int(leagues["n_appearances"].sum()) > 600_000
    assert f"{int(leagues['n_appearances'].sum()):,}" in text

    summary = pd.read_csv(
        RESULTS / "jsams_revised_denominator_gradient_summary.csv"
    ).set_index("quantity")["value"]
    assert summary["n_leagues_pooled_above_threshold"] == summary["n_leagues"]
    assert summary["n_leagues_starters_below_threshold"] == summary["n_leagues"]
    assert f"{summary['max_abs_gap_adjusted_vs_unadjusted']:.3f}" in supplement


def test_womens_results_match_the_generated_tables():
    """The women's extension is the paper's answer to a rejection, so every
    number in it is bound to the table that produced it -- including the two
    leagues where the recommended remedy does not work, which is the result a
    careless draft would round away."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    both = text + supplement

    womens = pd.read_csv(RESULTS / "jsams_womens_denominator_gradient_by_league.csv")
    assert len(womens) == 7
    for _, row in womens.iterrows():
        assert _carries(
            supplement, row["gamma_pooled"], row["gamma_pooled_ci_low"],
            row["gamma_pooled_ci_high"],
        ), row["league"]
        assert f"{int(row['n_appearances']):,}" in supplement, row["league"]

    contrast = pd.read_csv(
        RESULTS / "jsams_denominator_gradient_population_contrast.csv"
    ).set_index("population")
    women = contrast.loc["women"]
    assert f"{int(women['appearances']):,}" in text
    assert f"{women['min_gamma_pooled']:.3f}" in text
    assert f"{women['max_gamma_pooled']:.3f}" in text
    assert f"{women['median_gamma_pooled']:.3f}" in text

    # The exception must be reported, with both bounds, and must not be
    # described as though the remedy worked everywhere.
    rule = pd.read_csv(RESULTS / "jsams_womens_denominator_gradient_decision_rule.csv")
    per_appearance = rule[rule["recommendation"].eq("report per appearance")]
    assert len(per_appearance) == 2, "the exception is two leagues"
    assert int(women["leagues_negligible_within_starters"]) == 5
    assert not bool(women["pattern_holds_in_every_league"])
    assert "five of seven" in text
    for _, row in per_appearance.iterrows():
        match = womens[womens["league"].eq(row["league"])].iloc[0]
        assert f"{match['gamma_within_starters_ci_high']:.3f}" in text, row["league"]

    # And the men's population must not be described as failing anywhere.
    men = contrast.loc["men"]
    assert int(men["leagues_negligible_within_starters"]) == 8
    assert bool(men["pattern_holds_in_every_league"])
    assert "eight of eight men's leagues" in text

    # Coverage and the cross-source check live in the Supplement.
    completeness = pd.read_csv(
        RESULTS / "jsams_womens_league_season_completeness.csv"
    )
    assert bool(completeness["admitted"].all())
    assert f"{int(completeness['usable_matches'].sum())}" in both
    assert f"{int(completeness['scheduled_matches'].sum())}" in both

    cross = pd.read_csv(RESULTS / "jsams_denominator_gradient_cross_source.csv").iloc[0]
    assert bool(cross["intervals_overlap"])
    assert f"{cross['absolute_difference']:.3f}" in both


def test_injury_record_audit_justifies_the_outcome_restriction():
    """The paper withholds a women's outcome analysis. That is a scope claim,
    so it has to be argued with the audit's numbers rather than asserted, and
    the numbers have to be the deposited ones."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    contrast = pd.read_csv(
        RESULTS / "jsams_injury_record_audit_contrast.csv"
    ).set_index("population")
    severity = pd.read_csv(
        RESULTS / "jsams_injury_record_severity_mix.csv"
    ).set_index("population")

    for population in ("women", "men"):
        row = contrast.loc[population]
        assert f"{100 * float(row['share_with_history']):.1f}\\%" in text, population
        assert f"{float(row['spells_per_player']):.1f}" in text, population
        assert f"{float(row['early_recording_relative_to_reference']):.2f}" in text, population
        assert f"{100 * float(severity.loc[population, 'catastrophic_share']):.1f}\\%" in text
        assert f"{100 * float(severity.loc[population, 'unknown_share']):.1f}\\%" in text

    # The direction of every comparison must favour the men's record, or the
    # restriction the paper defends is not the one the data supports.
    assert float(contrast.loc["men", "share_with_history"]) > float(
        contrast.loc["women", "share_with_history"]
    )
    assert float(contrast.loc["men", "early_recording_relative_to_reference"]) > float(
        contrast.loc["women", "early_recording_relative_to_reference"]
    )
    assert float(severity.loc["women", "catastrophic_share"]) > float(
        severity.loc["men", "catastrophic_share"]
    )

    # The restriction is stated as a decision with a reason, not an omission.
    assert "we did not run one" in text
    # And the audit's own weakness is admitted where a reader will see it.
    assert "not a random sample of either" in text


def test_truncation_is_reported_as_the_defect_that_does_not_matter():
    """The decomposition is the paper's most original finding: the intuitive
    defect is measured and shown not to be the operative one."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    boot = pd.read_csv(RESULTS / "jsams_revised_attenuation_bootstrap.csv").set_index(
        "quantity"
    )
    row = boot.loc["truncation_attribution_absolute"]
    pooled = float(boot.loc["all", "estimate"])
    share = 100.0 * float(row["estimate"]) / pooled
    low = 100.0 * float(row["ci_low"]) / pooled
    high = 100.0 * float(row["ci_high"]) / pooled

    assert f"{share:.3f}\\%" in text
    assert f"{low:.2f}" in text and f"{high:.2f}" in text
    assert low < 0.0 < high
    assert "includes zero" in text

    case = pd.read_csv(RESULTS / "jsams_revised_case_restricted_exposure_bias.csv")
    indexed = case.set_index("quantity")["percent_understated"]
    cohort = float(indexed["total_minutes_whole_cohort"])
    restricted = float(indexed["mean_recorded_minutes_on_event_appearances"])
    assert f"{restricted:.1f}\\%" in text and f"{cohort:.2f}\\%" in text


def test_role_split_direction_matches_the_generated_signs():
    """Starters and substitutes move in opposite directions on event
    appearances, and that contrast is the point of Figure 1B."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    summary = pd.read_csv(RESULTS / "jsams_lineup_minute_bootstrap_summary.csv")
    indexed = summary.set_index("comparison")
    starter = float(indexed.loc["starting_lineup", "event_minus_non_event_minutes"])
    substitute = float(indexed.loc["substitute_list", "event_minus_non_event_minutes"])
    assert starter < 0.0 < substitute

    for uniformity in (
        "runs the same way inside each role",
        "the same way in each role",
        "in both roles",
    ):
        assert uniformity not in text, uniformity
    assert "indistinguishable" in text

    low = float(indexed.loc["substitute_list", "difference_ci_low"])
    high = float(indexed.loc["substitute_list", "difference_ci_high"])
    assert low < 0.0 < high


def test_supplement_keeps_the_controls_the_main_text_no_longer_carries():
    """Trimming to a shorter journal moved the negative controls, the placebo
    replication and the imputation range out of the main text. Content that
    moves must still be gated, or a cut becomes a quiet deletion."""
    supplement = SUPPLEMENT.read_text(encoding="utf-8")

    controls = pd.read_csv(
        RESULTS / "jsams_revised_negative_control_outcomes.csv"
    ).set_index("event_col")
    illness = controls.loc["same_day_illness_report"]
    assert not bool(illness["estimable"])
    assert int(illness["n_events"]) == 1
    assert "reported as not estimable rather than as a null result" in supplement
    assert "failed illness control" not in supplement

    table = pd.read_csv(RESULTS / "jsams_revised_placebo_window_analysis.csv")
    indexed = table.set_index(["model_id", "focal_window"])
    alone = indexed.loc[("placebo_31_37d_alone", "prior_minutes_placebo_31_37d")]
    recent = indexed.loc[("both_windows", "prior_minutes_7d")]
    placebo = indexed.loc[("both_windows", "prior_minutes_placebo_31_37d")]
    for row in (alone, recent, placebo):
        assert _carries(supplement, row["estimate"], row["ci_low"], row["ci_high"]), (
            row["description"]
        )
    assert float(recent["ci_low"]) > 1.0
    assert float(placebo["ci_low"]) < 1.0 < float(placebo["ci_high"])

    placebo_rep = pd.read_csv(
        RESULTS / "jsams_revised_placebo_denominator_replication.csv"
    ).set_index("quantity")
    reference = pd.read_csv(
        RESULTS / "jsams_revised_denominator_attenuation_decomposition.csv"
    ).set_index("quantity")["value"]
    gamma = placebo_rep.loc["gamma_placebo"]
    assert _carries(supplement, gamma["value"], gamma["ci_low"], gamma["ci_high"])
    ratio = float(placebo_rep.loc["gamma_over_observed_attenuation", "value"])
    assert abs(ratio - float(reference["gamma_over_observed_attenuation"])) < 0.10
    assert f"{ratio:.2f}" in supplement


def test_no_claim_beyond_what_the_intervals_support():
    """Three overclaims the paper's own intervals contradict."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    plotter = (ROOT / "src" / "35_plot_jsams_revision.py").read_text(encoding="utf-8")
    boot = pd.read_csv(RESULTS / "jsams_revised_attenuation_bootstrap.csv").set_index(
        "quantity"
    )
    roles = pd.read_csv(RESULTS / "jsams_revised_denominator_by_lineup_role.csv")

    # The within-starter attenuation excludes zero, so nothing is "removed".
    assert float(boot.loc["starting_lineup", "ci_low"]) > 0.0
    assert "starters removes" not in text
    for source in (text, plotter):
        assert "gap closes within starters" not in source
    assert "nearly closes within starters" in plotter

    # The recorded-minute estimate stays above one, so nothing is "reversed".
    pooled_recorded = roles[
        roles["lineup_role"].eq("all") & roles["denominator"].eq("observed_minutes")
    ]["estimate"].iloc[0]
    assert float(pooled_recorded) > 1.0
    for overclaim in ("reverse a study's conclusion", "reverse a conclusion"):
        assert overclaim not in text, overclaim

    # Adjusting is not offered as an equivalent to restricting.
    assert "linear predictor" in text or "not a substitute" in text

    # The women's replication must not be described as exceptionless, because
    # it is not; that phrase was true of the men's eight and survived a paste.
    replication = _between(text, "\\subsection{The gradient in fifteen", "\\subsection{What the outcome")
    assert "without exception" not in replication


def test_main_displays_match_the_smf_submission():
    """Two figures and three tables carry the paper; everything else is
    supplementary. The journal has no display cap, so the discipline is ours."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")

    figures = text.count(r"\begin{figure}")
    assert figures == len(MAIN_FIGURES)
    assert FIGURE_FLOOR <= figures <= FIGURE_CEILING, figures
    assert text.count(r"\begin{table}") == TABLE_COUNT

    for figure in MAIN_FIGURES:
        assert figure in text, figure
    for demoted in SUPPLEMENT_FIGURES:
        assert demoted not in text, demoted
        assert demoted in supplement, demoted
    # A figure printed in both documents is a figure the reader has to
    # reconcile, so the two sets must not intersect.
    stems = {name.rsplit(".", 1)[0] for name in MAIN_FIGURES}
    assert not stems & {name.rsplit(".", 1)[0] for name in SUPPLEMENT_FIGURES}

    # The journal wants figure captions as a list, separate from the figures.
    assert "\\section*{Figure captions}" in text
    assert text.index("\\section*{Figures}") < text.index("\\section*{Figure captions}")
    for number in ("Figure 1.", "Figure 2."):
        assert number in text, number


def test_worked_example_is_not_presented_as_a_finding():
    """The association is the thing being divided, not a result, and the paper
    has to say so where a reader would otherwise take it for one. The journal
    desk-rejects implicit causal readings, so this is doubly load-bearing."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    assert "worked example" in text
    assert "not as a finding" in text
    # And the estimator is explicitly described as arithmetic, not biology.
    assert "not a biological effect" in text


def test_tier_registry_still_governs_visibility():
    hierarchy = pd.read_csv(RESULTS / "jsams_revised_claim_hierarchy.csv")
    assert hierarchy["visibility_rule_passes"].all()
    assert hierarchy.query("tier <= 3")["abstract_visible"].all()
    assert hierarchy.query("tier > 3")["abstract_visible"].eq(False).all()

    # Anything in the abstract must also earn a main display.
    abstract_visible = hierarchy["abstract_visible"].astype(bool)
    main_display = hierarchy["main_display_recommended"].astype(bool)
    assert (~abstract_visible | main_display).all()

    # Every tier must argue itself down before it is allowed up, so a blank or
    # token justification is not a tier assignment.
    justification = hierarchy["tier_justification"].astype(str)
    assert justification.str.split().str.len().ge(12).all()
    assert justification.str.contains(r"Tier [1-5]").all()

    # The women's claims must be registered, or the manuscript is making claims
    # the registry never ranked.
    for claim in (
        "denominator_gradient_replicated_in_womens_leagues",
        "starter_restriction_is_league_specific",
        "womens_public_injury_record_quality",
    ):
        assert claim in set(hierarchy["claim_id"]), claim


ABSTRACT_MARKERS = {
    "cumulative_recent_exposure_same_day_association": ("1.27",),
    "independent_same_day_outcome_audit": ("attribution", "88.9"),
    "reported_event_duration_linkage": ("90 to 53", "truncated"),
    "denominator_gradient_measured_and_replicated": ("gradient",),
    "denominator_gradient_decision_rule": ("before", "reported"),
    "truncation_explains_none_of_the_attenuation": ("explained none",),
    "denominator_gradient_replicated_in_womens_leagues": ("women's", "0.444"),
    "starter_restriction_is_league_specific": ("five of seven", "0.058"),
}


def test_every_abstract_visible_claim_actually_reaches_the_abstract():
    """The registry decides what belongs in the abstract, so the abstract has to
    contain it. Without this the two drift silently: a claim can sit in the
    registry marked abstract-visible while the abstract never mentions it, which
    is what an earlier reframe did to the outcome-attribution audit."""
    abstract = _between(
        MANUSCRIPT.read_text(encoding="utf-8"), r"\begin{abstract}", r"\end{abstract}"
    )
    hierarchy = pd.read_csv(RESULTS / "jsams_revised_claim_hierarchy.csv")
    visible = hierarchy[hierarchy["abstract_visible"].astype(bool)]

    unmapped = set(visible["claim_id"]) - set(ABSTRACT_MARKERS)
    assert not unmapped, f"abstract-visible claims with no marker: {sorted(unmapped)}"

    for claim_id in visible["claim_id"]:
        markers = ABSTRACT_MARKERS[claim_id]
        assert any(marker in abstract for marker in markers), (
            f"{claim_id} is marked abstract-visible but the abstract carries none "
            f"of {markers}"
        )

    # And nothing kept out of the abstract may smuggle itself in.
    hidden = hierarchy[~hierarchy["abstract_visible"].astype(bool)]
    for phrase in ("inverse-selection", "Holm-adjusted interaction"):
        assert phrase not in abstract, phrase
    assert len(hidden) >= 3


def test_denominator_values_match_generated_tables():
    """Keep complete-season lineup estimates synchronised with the paper.

    The role split sits in the Supplement, so both files are searched: the
    requirement is that every generated value is reported somewhere, and that
    its direction is unambiguous from a word or a printed sign.
    """
    text = MANUSCRIPT.read_text(encoding="utf-8") + SUPPLEMENT.read_text(
        encoding="utf-8"
    )
    summary = pd.read_csv(RESULTS / "jsams_lineup_minute_bootstrap_summary.csv")
    indexed = summary.set_index("comparison")

    for comparison in ("starting_lineup", "substitute_list", "lineup_standardized"):
        row = indexed.loc[comparison]
        for value in (
            abs(float(row["event_minus_non_event_minutes"])),
            abs(float(row["difference_ci_low"])),
            abs(float(row["difference_ci_high"])),
        ):
            assert f"{value:.1f}" in text or f"{value:.2f}" in text, (comparison, value)


def test_scoping_count_is_reported_as_a_floor():
    """The count motivates the paper, so it has to match the deposited records
    and must never be presented as exhaustive."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    summary = pd.read_csv(
        RESULTS / "jsams_revised_per_minute_denominator_scoping.csv"
    ).set_index("quantity")["value"]

    assert "seven peer-reviewed studies" in text
    assert int(summary["n_records_retrieved"]) == 7
    assert int(summary["n_denominator_confirmed"]) == 5
    assert int(summary["n_denominator_unverified"]) == 2
    assert "a floor, not a ceiling" in text

    records = pd.read_csv(ROOT / "data" / "manual" / "per_minute_denominator_scoping.csv")
    assert len(records) == int(summary["n_records_retrieved"])
    assert records["source_url"].str.startswith("http").all()

    for key in SCOPING_KEYS:
        assert key in text, key

    confirmed = records[records["gradient_applies"].astype(str).eq("yes")]
    assert len(confirmed) == 5
    assert confirmed["denominator_verbatim"].str.len().gt(30).all()

    assert int(summary["n_provenance_flagged"]) == 1
    assert "provenance flag" in text

    protocol = pd.read_csv(
        RESULTS / "jsams_revised_per_minute_denominator_scoping_protocol.csv"
    ).set_index("rule_id")
    for rule_id in ("S2_sources_searched", "S3_search_dates", "S7_bound"):
        assert rule_id in protocol.index, rule_id
    assert "deposited beside the count" in text


# The seven scoping records, as citation keys. The count in the Introduction
# rests on these, so the gate walks the list rather than trusting the prose.
SCOPING_KEYS = (
    "szymski2023bundesliga_media",
    "wilke2022_media_muscle",
    "krutsch2021_bundesliga_restart",
    "palmer2023_english_case_series",
    "ortiz2026_acl_top_five",
    "dambrosi2026_rtp_acl",
    "hoenig2022citizen_transfermarkt",
)


def _main_text_word_count(tex: str) -> int:
    """Count main-text words the way the journal does: Introduction through the
    end of the Discussion, excluding floats, captions, citations and the back
    matter."""
    body = tex[tex.index("\\section{Introduction}"):tex.index("\\section*{Acknowledgements}")]
    for environment in ("table", "figure"):
        body = re.sub(
            rf"\\begin\{{{environment}\}}.*?\\end\{{{environment}\}}", " ", body, flags=re.S
        )
    body = re.sub(r"\\cite\{[^}]*\}", " ", body)
    body = re.sub(r"\\ref\{[^}]*\}", "X", body)
    body = re.sub(r"\\label\{[^}]*\}", " ", body)
    body = re.sub(r"\$[^$]*\$", "X", body)
    body = re.sub(r"\\(sub)*section\*?\{([^}]*)\}", r" \2 ", body)
    body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    body = re.sub(r"[{}\\~]", " ", body)
    return len([w for w in body.split() if re.search(r"[A-Za-z0-9]", w)])


def _abstract_word_count(tex: str) -> int:
    abstract = tex[tex.index(r"\begin{abstract}"):tex.index(r"\end{abstract}")]
    abstract = re.sub(r"\$[^$]*\$", "X", abstract)
    abstract = re.sub(r"\\[a-zA-Z]+\*?", " ", abstract)
    abstract = re.sub(r"[{}\\~]", " ", abstract)
    return len([w for w in abstract.split() if re.search(r"[A-Za-z0-9]", w)])


def test_smf_limits_and_satellite_documents_agree():
    """The journal's limits are requirements, so the suite enforces them; and
    every count a satellite document states must be the computed one. Three
    rounds running, a satellite carried stale numbers or a stale title, because
    compiled documents fail loudly and stated counts fail silently."""
    tex = MANUSCRIPT.read_text(encoding="utf-8")

    main_words = _main_text_word_count(tex)
    abstract_words = _abstract_word_count(tex)
    assert main_words <= WORD_LIMIT, main_words
    assert abstract_words <= ABSTRACT_LIMIT, abstract_words

    applications = tex[
        tex.index("\\subsection{Practical applications}"):tex.index("\\section*{Acknowledgements}")
    ]
    assert 3 <= applications.count("\\item") <= 5

    keywords = re.search(r"\\newcommand\{\\keywordlist\}\{([^}]*)\}", tex).group(1)
    count = len([k for k in keywords.split(";") if k.strip()])
    assert KEYWORD_RANGE[0] <= count <= KEYWORD_RANGE[1], count

    cited = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", tex):
        cited.update(key.strip() for key in group.split(","))
    assert len(cited) <= 40, len(cited)

    # The title page must describe this manuscript, not a previous one.
    title_page = TITLE_PAGE.read_text(encoding="utf-8")
    manuscript_title = re.search(r"\\newcommand\{\\papertitle\}\{([^}]*)\}", tex).group(1)
    title_page_title = re.search(
        r"\\newcommand\{\\papertitle\}\{([^}]*)\}", title_page
    ).group(1)
    assert title_page_title == manuscript_title

    stated_main = re.search(
        r"\\newcommand\{\\mainwordcount\}\{([^}]*)\}", title_page
    ).group(1)
    stated_abstract = re.search(
        r"\\newcommand\{\\abstractwordcount\}\{([^}]*)\}", title_page
    ).group(1)
    assert stated_main == f"{main_words:,}", (stated_main, main_words)
    assert stated_abstract == str(abstract_words), (stated_abstract, abstract_words)
    assert f"Tables: {TABLE_COUNT}. Figures: {len(MAIN_FIGURES)}." in title_page
    assert "Original Investigation" in title_page

    # The league count the title states has to be the number of leagues the
    # deposited population table actually carries.
    population = pd.read_csv(RESULTS / "jsams_denominator_gradient_by_population.csv")
    assert "fifteen" in manuscript_title.lower()
    assert len(population) == 15

    # The cover letter may not hard-code a word count; a stated number goes
    # stale with the next edit, and did.
    letter = COVER_LETTER.read_text(encoding="utf-8")
    assert not re.search(r"\b\d[\d,]* words", letter)
    assert "4,500-word" in letter


#: Files that restate the manuscript's title. Each one is a separate document
#: an editor may open, and each has at some point carried a title the paper no
#: longer had.
TITLE_BEARING = (
    Path("manuscript") / "title_page.tex",
    Path("manuscript") / "supplement.tex",
    Path("manuscript") / "cover_letter.md",
    Path("manuscript") / "cover_letter.tex",
    Path("manuscript") / "credit_author_statement.tex",
    Path("manuscript") / "declaration_of_interest.tex",
    Path("manuscript") / "acknowledgement_funding_ethics.tex",
    Path("manuscript") / "strobe_checklist.tex",
    Path("tools") / "build_public_release.py",
)


def _normalise(text: str) -> str:
    """Flatten a document so a title can be found however it was wrapped.

    Line breaks, LaTeX escapes and Python's adjacent-string-literal
    concatenation all split a title across the source without changing what a
    reader sees, so all three are removed before matching.
    """
    flat = " ".join(text.replace("\\", "").split())
    return flat.replace('" "', "")


def _orcid_check_digit(digits: str) -> str:
    """The ISO 7064 MOD 11-2 check character an ORCID iD must end in."""
    total = 0
    for digit in digits[:15]:
        total = (total + int(digit)) * 2
    remainder = (12 - total % 11) % 11
    return "X" if remainder == 10 else str(remainder)


def test_the_title_page_carries_a_valid_orcid():
    """An ORCID is sixteen digits nobody reads, so a transposition survives
    every human check and resolves to a stranger or to nothing. The identifier
    validates itself, and the gate is the only reader that will notice."""
    title_page = TITLE_PAGE.read_text(encoding="utf-8")
    found = re.findall(r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", title_page)
    assert found, "the corresponding author's ORCID is missing from the title page"

    for iD in found:
        digits = iD.replace("-", "")
        assert _orcid_check_digit(digits) == digits[15], (
            f"{iD} fails its own check digit and is not a real ORCID"
        )


def test_no_document_still_carries_a_superseded_title():
    """A title lives in ten files, and a rename reaches nine of them. The
    tenth goes to an editor saying the paper is about something else."""
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    title = _normalise(
        re.search(r"\\newcommand\{\\papertitle\}\{([^}]*)\}", tex).group(1)
    )
    superseded = "measuring the denominator gradient in eight European leagues"

    for relative in TITLE_BEARING:
        body = _normalise((ROOT / relative).read_text(encoding="utf-8"))
        assert superseded not in body, relative.as_posix()
        assert title in body, relative.as_posix()


def test_smf_element_order_is_followed():
    """The journal lists a required element order for Original Investigations.
    It desk-rejects a manuscript claiming to follow guidance it does not, so
    the order is checked rather than assumed."""
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    required = [
        r"\begin{abstract}",
        r"\textbf{Keywords:}",
        r"\section{Introduction}",
        r"\section{Materials and methods}",
        r"\section{Results}",
        r"\section{Discussion}",
        r"\subsection{Limitations}",
        r"\section*{Acknowledgements}",
        r"\section*{Author contributions}",
        r"\section*{Funding}",
        r"\section*{Disclosure of interest}",
        r"\section*{Declaration of generative AI use}",
        r"\section*{Data availability statement}",
        r"\section*{Code availability statement}",
        r"\section*{Data deposition}",
        r"\section*{Ethics approval and informed consent}",
        r"\bibliography{references}",
        r"\section*{Tables}",
        r"\section*{Figures}",
        r"\section*{Figure captions}",
    ]
    positions = []
    for element in required:
        assert element in tex, element
        positions.append(tex.index(element))
    assert positions == sorted(positions), "element order does not follow the guide"

    # The limitation section is inside the discussion, as the guide requires.
    assert tex.index(r"\section{Discussion}") < tex.index(r"\subsection{Limitations}")


def test_abstract_is_unstructured_and_carries_design_aims_conclusion():
    """The journal requires an unstructured abstract that nonetheless always
    states the design, the aims, and a conclusion based on the primary
    outcomes. Structured headings inside it are a formatting rejection."""
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    abstract = _between(tex, r"\begin{abstract}", r"\end{abstract}")

    # Unstructured: no bold run-in headings of the Objectives/Design kind.
    assert r"\textbf" not in abstract
    for heading in ("Objectives:", "Design:", "Methods:", "Results:", "Conclusions:"):
        assert heading not in abstract, heading

    tokens = pd.read_csv(ROOT / "data" / "manual" / "smf_abstract_tokens.csv")
    for _, row in tokens.iterrows():
        options = [t.strip() for t in str(row["tokens"]).split("|")]
        assert any(t in abstract for t in options), (row["component"], options)

    # The aims sentence in the abstract must be the manuscript's aims sentence.
    intro = _between(tex, "\\section{Introduction}", "\\section{Materials and methods}")
    assert "We aimed to" in abstract and "We aimed to" in intro


CAUSAL_TERMS = (
    "caused", "causal", "causes", "effect of", "drives", "leads to",
    "increases risk", "reduces risk", "significant",
)


def test_causal_language_is_absent_from_the_abstract():
    """The journal desk-rejects implicit causal interpretation, and says so in
    terms: hypothesised causal readings must be explicit in the text and must
    never frame the abstract. The gradient is arithmetic on an estimator, so
    causal verbs would be wrong as well as unwelcome."""
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    abstract = _between(tex, r"\begin{abstract}", r"\end{abstract}").lower()
    for term in CAUSAL_TERMS:
        assert term not in abstract, term


US_SPELLINGS = (
    "analyze", "analyzed", "modeled", "modeling", "labeled",
    "generalize", "generalized", "minimize", "behavior", "favor",
    "artifact", "italicized",
)


def test_uk_spelling_is_consistent():
    """The journal allows either variety but requires consistency. The text is
    UK throughout, so a US variant is a drift rather than a choice."""
    for path in (MANUSCRIPT, SUPPLEMENT, TITLE_PAGE):
        text = path.read_text(encoding="utf-8").lower()
        for variant in US_SPELLINGS:
            assert variant not in text, (path.name, variant)
        # Rates are written "per 1000 hours", without the thousands comma the
        # counts use; the supplement once mixed both forms in one section.
        assert "per 1,000" not in text, path.name


def test_the_governing_analogy_appears_exactly_twice():
    """One analogy, used twice, in the manuscript only. An explanation that
    needs a second metaphor is an explanation that has not been made, and an
    analogy repeated on every page stops being an aid and becomes a tic."""
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert manuscript.count("exposure clock") == 2, manuscript.count("exposure clock")
    # Once where the mechanism is introduced, once where it is resolved.
    first = manuscript.index("exposure clock")
    second = manuscript.index("exposure clock", first + 1)
    assert first < manuscript.index("\\section{Materials and methods}")
    assert second > manuscript.index("\\section{Discussion}")
    # And nowhere else, so the supplement does not dilute it.
    assert "exposure clock" not in SUPPLEMENT.read_text(encoding="utf-8")


def test_statistical_output_appendix_is_current():
    """The journal requires the software's original full output as an appendix.
    The appendix is generated by refitting every reported gradient and checking
    it against the deposited estimate, so what is gated here is that the file
    exists, covers every model, and was not hand-edited afterwards."""
    assert APPENDIX.exists(), "the statistical output appendix has not been generated"
    appendix = APPENDIX.read_text(encoding="utf-8")
    manifest = pd.read_csv(RESULTS / "jsams_statistical_output_manifest.csv")

    mens = pd.read_csv(RESULTS / "jsams_revised_denominator_gradient_by_league.csv")
    womens = pd.read_csv(RESULTS / "jsams_womens_denominator_gradient_by_league.csv")
    assert len(manifest) == 2 * (len(mens) + len(womens))

    for model_id in manifest["model_id"]:
        assert model_id in appendix, model_id
    assert appendix.count("OLS Regression Results") == len(manifest)

    # Every pooled fit in the appendix must be the published one.
    pooled = manifest[manifest["stratum"].eq("all appearances")].set_index("league")
    for _, row in pd.concat([mens, womens]).iterrows():
        assert abs(
            float(pooled.loc[row["league"], "gamma"]) - float(row["gamma_pooled"])
        ) < 1e-9, row["league"]

    # And the manuscript has to tell the reader the appendix exists.
    text = MANUSCRIPT.read_text(encoding="utf-8")
    assert "Appendix A" in text
    assert f"{len(manifest)} gradient models" in text or f"all {len(manifest)}" in text


def test_backmatter_statements_are_present_and_specific():
    """Six statements the journal makes mandatory. Each has been supplied as an
    empty gesture in some submission somewhere, so each is checked for content
    rather than for a heading."""
    tex = MANUSCRIPT.read_text(encoding="utf-8")

    funding = _between(tex, r"\section*{Funding}", r"\section*{Disclosure of interest}")
    assert "no specific grant" in funding

    disclosure = _between(
        tex, r"\section*{Disclosure of interest}", r"\section*{Declaration of generative AI use}"
    )
    assert "no competing interests to declare" in disclosure

    ai = _between(
        tex, r"\section*{Declaration of generative AI use}", r"\section*{Data availability statement}"
    )
    assert len(ai.split()) > 30, "the AI declaration must say what was used and how"
    assert "verified by the authors" in ai

    das = _between(
        tex, r"\section*{Data availability statement}", r"\section*{Code availability statement}"
    )
    cas = _between(
        tex, r"\section*{Code availability statement}", r"\section*{Data deposition}"
    )
    deposition = _between(
        tex, r"\section*{Data deposition}", r"\section*{Ethics approval and informed consent}"
    )
    identifiers = pd.read_csv(ROOT / "data" / "manual" / "deposit_identifiers.csv")
    concept = identifiers[identifiers["role"].eq("concept")].iloc[0]["identifier"]
    for section, label in ((das, "data availability"), (cas, "code availability"),
                           (deposition, "data deposition")):
        assert concept in section or "archivedoi" in section, label
    # The statement must say why anything withheld is withheld.
    assert "not redistributed" in das

    ethics = _between(
        tex, r"\section*{Ethics approval and informed consent}", r"\begingroup"
    )
    # The guide accepts an explanation where approval is not required, but it
    # must be an explanation rather than a silence.
    assert "no primary data were collected" in ethics
    assert "informed consent was not applicable" in ethics
    assert "none can now be issued" in ethics


def test_strobe_checklist_resolves_to_sections_that_exist():
    """The journal desk-rejects a manuscript that states it follows a reporting
    guideline without following it, so the checklist is not decorative: every
    item must name a section the manuscript actually contains."""
    checklist = pd.read_csv(ROOT / "data" / "manual" / "strobe_checklist.csv")
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    assert len(checklist) >= 22, len(checklist)

    for _, row in checklist.iterrows():
        assert str(row["manuscript_text"]).strip(), row["item"]
        for anchor in str(row["anchor"]).split("|"):
            assert anchor.strip() in tex, (row["item"], anchor)

    assert "STROBE" in tex


def test_figures_are_current_and_in_both_formats(load_src_module):
    """A figure drawn from a table that has since changed fails here rather
    than waiting for a reader: the manifest records what script 35 drew, and
    the league labels it drew must equal the league table as it stands now."""
    figures_dir = ROOT / "manuscript" / "figures"
    both = MANUSCRIPT.read_text(encoding="utf-8") + SUPPLEMENT.read_text(
        encoding="utf-8"
    )
    included = set(re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}", both))
    assert len(included) == len(MAIN_FIGURES) + len(SUPPLEMENT_FIGURES)
    for name in included:
        # Both twins must exist whichever one the document embeds: the vector
        # is what the journal prints, the raster is what a reader without a TeX
        # toolchain can open out of the deposit.
        stem = (figures_dir / name).with_suffix("")
        assert stem.with_suffix(".pdf").exists(), name
        assert stem.with_suffix(".png").exists(), name

    manifest = pd.read_csv(
        RESULTS / "jsams_revised_figure_manifest.csv"
    ).set_index("figure")
    assert manifest["formats"].eq("png+pdf").all()
    assert len(manifest) == len(MAIN_FIGURES) + len(SUPPLEMENT_FIGURES)
    drawn = str(manifest.loc["J2_jsams_denominator_gradient", "league_labels"])
    leagues = pd.read_csv(RESULTS / "jsams_revised_denominator_gradient_by_league.csv")
    assert drawn == "|".join(str(label) for label in leagues["league"])

    plotting = load_src_module("35_plot_jsams_revision.py")
    assert (
        str(manifest.loc["J2_jsams_denominator_gradient", "source_digest"])
        == plotting.gradient_source_digest(leagues)
    )
    # The men-and-women panel plots a league table too, and is gated the same
    # way. It was added after this check was written, and exempting everything
    # but J2 would have let a stale second forest plot through unnoticed.
    population_path = RESULTS / "jsams_denominator_gradient_by_population.csv"
    digested = ["J2_jsams_denominator_gradient"]
    if "J7_jsams_gradient_by_population" in manifest.index:
        assert population_path.exists(), "J7 is in the manifest but its table is not"
        population = pd.read_csv(population_path)
        digested.append("J7_jsams_gradient_by_population")
        assert (
            str(manifest.loc["J7_jsams_gradient_by_population", "source_digest"])
            == plotting.gradient_source_digest(population)
        )
        drawn_population = str(
            manifest.loc["J7_jsams_gradient_by_population", "league_labels"]
        )
        assert drawn_population == "|".join(
            f"{group}:{label}"
            for group, label in zip(population["population"], population["league"])
        )

    # The figures that plot no league table record no digest to compare.
    assert (
        manifest.drop(index=digested)["source_digest"].fillna("").eq("").all()
    )


def test_preprint_is_declared_where_the_journal_requires_it():
    """A stated negative goes stale silently, so the preprint declaration is
    gated in both places that carry it."""
    title_page = TITLE_PAGE.read_text(encoding="utf-8")
    letter = " ".join(COVER_LETTER.read_text(encoding="utf-8").split())

    for source, label in ((title_page, "title page"), (letter, "cover letter")):
        assert ARXIV_ID in source, label
        assert "arxiv.org/abs/" in source.lower(), label

    # The journal warns that a preprint defeats anonymity; the letter should
    # show we understood that rather than leaving it to be discovered.
    assert "anonymity cannot be guaranteed" in letter

    # The posted v2 reports eight leagues and claims the within-starter
    # pattern held without exception, which this manuscript contradicts. Until
    # the replacement is live, both documents must say so plainly: a referee
    # who follows the declared preprint must not be left to reconcile two
    # papers on their own.
    for source, label in ((title_page, "title page"), (letter, "cover letter")):
        assert "supersede" in source, label
        assert "without exception" in source, label

    # And the letter must not carry the two errors the manuscript has shed:
    # the headline estimate is a rate ratio, and the division cost two thirds
    # of the excess association, not a third.
    assert "odds ratio" not in letter
    assert "roughly a third" not in letter
    assert "rate ratio" in letter


#: Published Science and Medicine in Football articles the cover letter names
#: as precedent for the submission's topic and level of technicality:
#: Bache-Mathiesen 2022 (statistical methodology, 6(4):452-464), Hecksteden
#: 2026 (injury-risk data analytics), Mkumbuzi 2023 (women's surveillance,
#: 7(1):74-80), Impellizzeri 2019 (the journal's statistical recommendations,
#: 3(1):1-2).
SMF_PRECEDENTS = ("Bache-Mathiesen", "Hecksteden", "Mkumbuzi", "Impellizzeri")


def test_cover_letter_precedent_articles_survive_in_both_copies():
    """The letter grounds journal fit in the journal's own pages. It exists
    twice (Markdown record, LaTeX for the PDF), and prose that lives twice
    drifts unless both copies are held to the same names."""
    md = COVER_LETTER.read_text(encoding="utf-8")
    tex = (ROOT / "manuscript" / "cover_letter.tex").read_text(encoding="utf-8")
    for name in SMF_PRECEDENTS:
        assert name in md, ("cover_letter.md", name)
        assert name in tex, ("cover_letter.tex", name)


def test_paper_cites_the_record_it_is_archived_inside():
    """The deposit contains the manuscript, so the manuscript must cite the
    deposit: a reader holding the archive should be able to confirm that the
    text they have is the text the data supports."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    title_page = TITLE_PAGE.read_text(encoding="utf-8")
    identifiers = pd.read_csv(ROOT / "data" / "manual" / "deposit_identifiers.csv")

    concept = identifiers[identifiers["role"].eq("concept")].iloc[0]["identifier"]
    assert concept.startswith("10.5281/zenodo.")
    assert concept in text or "archivedoi" in text
    assert concept in title_page

    # A concept identifier resolves to the newest version, which is what a
    # reader wants; a version identifier pinned in prose goes stale.
    assert identifiers["role"].eq("concept").sum() == 1
    for _, row in identifiers.iterrows():
        assert str(row["identifier"]).startswith("10.5281/zenodo."), row["role"]


# The arXiv record this work is posted on. Declaring it is a journal
# requirement, and the declaration went stale the moment the preprint existed.
ARXIV_ID = "2608.11228"

#: Every string that would tell a referee who wrote this. Review is
#: single-anonymous and a preprint exists, so none of this is a secret; the
#: point is that identity lives on the title page rather than scattered through
#: a manuscript, where one forgotten instance is found only after upload.
IDENTIFYING = (
    "Gustavo",
    "Ricou",
    "Mahony",
    "James Ng",
    "Adam Field",
    "Trinity College Dublin",
    "Manchester Metropolitan",
    "github.com/Gustolandia",
    "10.5281/zenodo.",
    "pedrorig@tcd.ie",
    "njmahony@tcd.ie",
    # Our own preprint identifier, not the word: the bibliography cites other
    # people's arXiv preprints, and flagging those would be a false alarm that
    # trains the reader of this gate to ignore it.
    ARXIV_ID,
)

BLIND_BLOCK = re.compile(
    r"\\ifdefined\\journalblindcopy(?P<blind>.*?)(?:\\else(?P<open>.*?))?\\fi",
    re.S,
)


def _blinded(tex: str) -> str:
    """Resolve the source the way the anonymised build resolves it."""
    return BLIND_BLOCK.sub(lambda match: match.group("blind"), tex)


def _open_copy(tex: str) -> str:
    return BLIND_BLOCK.sub(lambda match: match.group("open") or "", tex)


def test_the_anonymised_copy_carries_no_identifying_string():
    """Blinding by hand fails on the instance nobody remembered, so the check
    resolves the conditionals the way the build does and reads the result."""
    for path in (MANUSCRIPT, SUPPLEMENT):
        blinded = _blinded(path.read_text(encoding="utf-8"))
        for token in IDENTIFYING:
            assert token not in blinded, f"{path.name} leaks {token!r} when blinded"

    # The blinded copy must say where the missing material went, rather than
    # silently dropping sections an editor expects to find.
    blinded = _blinded(MANUSCRIPT.read_text(encoding="utf-8"))
    assert "separate title-page file" in blinded
    assert "blinded for review" in blinded


def test_acknowledgements_survive_in_every_build():
    """Named people must be thanked in the copy that is published, and the
    blinding must not be achieved by deleting them. Consent was given for an
    acknowledgement, not for a silence."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    title_page = TITLE_PAGE.read_text(encoding="utf-8")
    open_copy = _open_copy(text)

    for name in ("James Ng", "Adam Field"):
        assert name in text, name
        assert name in title_page, name
        assert name in open_copy, name

    # And the identifying material must genuinely be in the open copy, or the
    # gate above would pass on a manuscript that blinds by having nothing.
    for token in ("github.com/Gustolandia", "10.5281/zenodo.", "Gustavo"):
        assert token in open_copy, token


def test_practical_applications_speak_to_a_lay_audience():
    """The journal prefers applicability to complexity, and the applications
    are the part a practitioner reads. They must be usable without the
    estimator, and must not overreach into clinical advice."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    applications = _between(
        text, r"\subsection{Practical applications}", r"\section*{Acknowledgements}"
    )
    assert "appearance records" in applications
    # The refusal the paper owes a practitioner.
    assert "clear a player" in applications
    for jargon in ("offset", "Poisson", "heteroskedastic", "estimand"):
        assert jargon not in applications, jargon


def test_references_the_count_rests_on_are_complete():
    """Every cited key must exist in the bibliography, or the compiled PDF
    carries a question mark where a citation should be."""
    text = MANUSCRIPT.read_text(encoding="utf-8") + SUPPLEMENT.read_text(
        encoding="utf-8"
    )
    bib = (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8")
    cited = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", text):
        cited.update(key.strip() for key in group.split(","))
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = sorted(cited - defined)
    assert not missing, missing


def test_every_estimate_is_labelled_by_the_model_that_produced_it():
    """A Poisson estimate is a rate ratio and a logistic one is an odds ratio.

    In this cohort the two nearly coincide, because the outcome is rare: the
    per-appearance odds ratio is 1.267 and the fixed-90 rate ratio 1.265. A
    first submission collapsed the two and called a Poisson estimate an odds
    ratio in its own abstract. The label now travels in the deposited table
    beside the estimate, and this gate is what stops the manuscript reapplying
    it from memory.
    """
    text = MANUSCRIPT.read_text(encoding="utf-8")
    roles = pd.read_csv(RESULTS / "jsams_revised_denominator_by_lineup_role.csv")
    measures = (
        roles.drop_duplicates("denominator").set_index("denominator")["effect_measure"]
    )
    assert measures["fixed_90"] == "rate_ratio"
    assert measures["observed_minutes"] == "rate_ratio"
    assert measures["per_appearance"] == "odds_ratio"

    assert "rate ratio of 1.27" in text
    assert "an odds ratio of 1.27" not in text
    # The phrase may appear once, where the logistic model is named, and the
    # naming has to be in the same sentence rather than somewhere nearby.
    # The phrase may appear only where the logistic model is named in the same
    # breath: once in Results and once in the caption of the negative-control
    # figure, whose fits are per-appearance and therefore genuinely odds ratios.
    occurrences = [m.start() for m in re.finditer("odds ratio", text)]
    assert len(occurrences) == 2, len(occurrences)
    for position in occurrences:
        assert "logistic" in text[position - 260:position], position

    # The coincidence is the robustness argument, so the paper has to make it.
    assert "carrying no exposure gradient agree" in text


def test_identity_calibration_reaches_the_main_text():
    """The identity over-predicts the attenuation, and by an amount that is
    itself a function of the gradient. Reporting a single pooled ratio put a
    factor of two into the neighbourhood of the reporting threshold, which is
    the one part of the range the decision rule uses, so the main text now
    carries the measured range rather than a constant."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    calibration = pd.read_csv(RESULTS / "jsams_identity_calibration.csv").set_index(
        "stratum"
    )
    curve = pd.read_csv(RESULTS / "jsams_calibration_curve.csv")

    pooled = calibration.loc["all"]
    assert f"{float(pooled['over_prediction_ratio']):.2f}" in text
    assert f"{float(pooled['observed_percent']):.1f}\\%" in text
    assert f"{float(pooled['gamma_implied_percent']):.1f}\\%" in text

    # The ratio is not a constant, and the paper must say so with its range.
    assert f"{float(curve['over_prediction_ratio'].min()):.2f}" in text
    assert f"{float(curve['over_prediction_ratio'].max()):.2f}" in text
    assert "function of $\\gamma$" in text

    # Within starters the identity is nearly exact, which is why the ratio
    # cannot be pooled: it is small where the gradient is small.
    starters = calibration.loc["starting_lineup"]
    assert abs(float(starters["over_prediction_ratio"]) - 1.0) < 0.15
    assert float(curve["over_prediction_ratio"].min()) < 1.0
    assert float(curve["over_prediction_ratio"].max()) > 1.9

    # And the supplement must carry the per-stratum detail the text summarises.
    for stratum in ("substitute_list", "lineup_unavailable_or_other"):
        ratio = float(calibration.loc[stratum, "over_prediction_ratio"])
        assert f"{ratio:.2f}" in supplement, stratum


def test_the_calibration_ratio_is_measured_where_the_threshold_sits():
    """The reporting threshold must fall inside the swept range, not beyond it.

    A previous version interpolated nothing and applied the pooled ratio at
    every gradient, which is an extrapolation into the only region a
    practitioner reads, and it ran in the direction that makes the threshold
    look twice as tolerant as it is.
    """
    text = MANUSCRIPT.read_text(encoding="utf-8")
    curve = pd.read_csv(RESULTS / "jsams_calibration_curve.csv")
    translation = pd.read_csv(RESULTS / "jsams_threshold_translation.csv")

    # The sweep has to bracket the threshold, or the calibration there is a guess.
    assert float(curve["gamma"].min()) < 0.05 < float(curve["gamma"].max())
    assert len(curve) >= 5
    # Raising the minute floor must lower the gradient, or the sweep is not
    # doing what it claims.
    ordered = curve.sort_values("minute_floor", ascending=False)
    assert ordered["gamma"].is_monotonic_increasing

    at_threshold = translation[translation["is_reporting_threshold"].astype(bool)]
    assert len(at_threshold) == 1
    row = at_threshold.iloc[0]
    assert bool(row["ratio_is_measured"])
    assert f"{float(row['over_prediction_ratio']):.2f}" in text
    assert f"{float(row['calibrated_percent_understatement']):.1f}\\%" in text

    # Every ratio on the sweep now carries an interval, because the sweep was
    # the one place the paper broke its own every-estimate-gets-an-interval
    # rule, and the threshold cost is reported as a range wherever it appears.
    for column in ("ratio_ci_low", "ratio_ci_high",
                   "attenuation_ci_low", "attenuation_ci_high"):
        assert column in curve.columns, column
        assert curve[column].notna().all(), column
    assert (curve["ratio_ci_low"] <= curve["over_prediction_ratio"]).all()
    assert (curve["over_prediction_ratio"] <= curve["ratio_ci_high"]).all()

    interval = pd.read_csv(RESULTS / "jsams_threshold_cost_interval.csv").iloc[0]
    assert int(interval["n_boot_valid"]) >= 950
    cost_range = (
        f"{float(interval['cost_percent_ci_low']):.1f} to "
        f"{float(interval['cost_percent_ci_high']):.1f}"
    )
    ratio_range = (
        f"{float(interval['ratio_at_threshold_ci_low']):.2f} to "
        f"{float(interval['ratio_at_threshold_ci_high']):.2f}"
    )
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    assert text.count(ratio_range) >= 2, ratio_range
    assert cost_range in text, cost_range
    assert ratio_range in supplement and (
        f"{float(interval['cost_percent_ci_low']):.1f}\\%" in supplement
    )

    # The softened claim: the small-gradient end is stated to what its own
    # interval will bear, and the widest interval is quoted rather than hidden.
    lowest = curve.sort_values("gamma").iloc[0]
    assert (
        f"{float(lowest['ratio_ci_low']):.2f} to {float(lowest['ratio_ci_high']):.2f}"
        in supplement
    )
    assert "too noisy to constrain" in supplement
    assert "accurate to within a few per cent" not in supplement

    # And the curve's scope is stated: measured in this cohort, sweep your own
    # data, and -- correcting a slip in the report that asked for this -- the
    # sweep needs an outcome, which is why the women's panels cannot supply a
    # second population.
    assert "in this cohort" in text
    assert "sweep their own data" in text
    assert "no second-population sweep is possible" in supplement
    # The pooled ratio would have given roughly half this cost; that number
    # must no longer appear anywhere as the threshold's cost.
    assert "corresponds to 2.5\\%" not in text
    assert "roughly 2.5\\% of the association" not in text


def test_the_threshold_is_translated_rather_than_declared():
    """The decision rule turns on 0.05 and the Spain bound sits exactly on it,
    so the threshold has to mean something. It is reported in the units of the
    answer, which is what a practitioner can weigh."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    translation = pd.read_csv(RESULTS / "jsams_threshold_translation.csv")

    threshold = translation[translation["is_reporting_threshold"].astype(bool)]
    assert len(threshold) == 1
    row = threshold.iloc[0]
    assert float(row["gamma"]) == 0.05
    assert f"{float(row['calibrated_percent_understatement']):.1f}\\%" in text
    assert f"{float(row['naive_percent_understatement']):.1f}\\%" in text
    # Every row of the grid records whether its ratio was measured, and the
    # supplement must mark the extrapolated ones rather than print them plain.
    assert not translation["ratio_is_measured"].all()
    assert "extrapolat" in supplement

    # The supplement carries the full grid so a reader can pick another
    # tolerance, which is the only honest defence of a convention.
    for _, entry in translation.iterrows():
        assert f"{float(entry['gamma']):.3f}" in supplement, entry["gamma"]
    assert "a different tolerance can be substituted" in supplement


def test_precision_is_reported_in_place_of_a_power_statement():
    """No target sample size was set, so STROBE item 10 is answered with the
    precision the panels actually delivered, smallest first."""
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    profile = pd.read_csv(RESULTS / "jsams_precision_profile.csv")

    women = profile[profile["population"].eq("women")]
    men = profile[profile["population"].eq("men")]
    for group in (women, men):
        low = 100.0 * float(group["relative_half_width"].min())
        high = 100.0 * float(group["relative_half_width"].max())
        assert f"{low:.1f}\\%" in supplement, low
        assert f"{high:.1f}\\%" in supplement, high

    # The least precise panel has to be named, because a reader's first
    # question is whether the smallest league is carrying a verdict.
    smallest = profile.sort_values("n_appearances").iloc[0]
    widest = profile.sort_values("relative_half_width").iloc[-1]
    assert str(widest["league"]) in supplement
    assert f"{int(widest['n_appearances']):,}" in supplement
    # Even the widest interval must clear the threshold it is read against, or
    # the verdict is an artefact of precision rather than a measurement.
    assert float(widest["ci_low"]) > 0.05
    assert int(smallest["n_appearances"]) < int(men["n_appearances"].min())


def test_ascertainment_by_role_is_reported_where_the_remedy_is_recommended():
    """The remedy is restriction to starters, and starters carry three times the
    event rate per appearance. If that gap were reporting rather than exposure,
    restriction would trade a denominator problem for a numerator one, so the
    comparison is made in both units and reported in the main text."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    rates = pd.read_csv(RESULTS / "jsams_ascertainment_by_role.csv").set_index(
        "lineup_role"
    )

    starters = rates.loc["starting_lineup"]
    substitutes = rates.loc["substitute_list"]

    for row in (starters, substitutes):
        assert f"{float(row['events_per_1000_appearances']):.1f}" in text
        assert f"{float(row['events_per_1000_minutes']):.3f}" in text

    # Per appearance the starters are higher; per minute they are lower. That
    # inversion is the whole finding, and it must not be reported one way only.
    assert float(starters["events_per_1000_appearances"]) > float(
        substitutes["events_per_1000_appearances"]
    )
    assert float(starters["events_per_1000_minutes"]) < float(
        substitutes["events_per_1000_minutes"]
    )
    assert "the ordering inverts" in text

    # The supplement carries the intervals and the third stratum.
    for bound in ("events_per_1000_appearances_ci_low", "events_per_1000_appearances_ci_high"):
        assert f"{float(starters[bound]):.1f}" in supplement, bound
    unknown = rates.loc["lineup_unavailable_or_other"]
    assert f"{float(unknown['events_per_1000_appearances']):.1f}" in supplement
    # And the honest boundary of what the comparison can detect.
    assert "acting on both alike" in text


def test_clustering_sensitivity_changes_no_estimate_and_no_verdict():
    """Clustering is a covariance choice. If the three rows disagreed on the
    point estimate the table would be reporting three models, which is what a
    first attempt at it did."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    clustering = pd.read_csv(RESULTS / "jsams_clustering_sensitivity.csv")
    decomposition = pd.read_csv(
        RESULTS / "jsams_revised_denominator_attenuation_decomposition.csv"
    ).set_index("quantity")

    published = float(decomposition.loc["gamma_log_minutes_on_exposure", "value"])
    assert clustering["gamma"].nunique() == 1
    assert abs(float(clustering["gamma"].iloc[0]) - published) < 1e-9

    assert bool(clustering["all_bounds_exceed_threshold"].all())
    ratio = float(clustering["max_width_ratio"].iloc[0])
    assert f"{ratio:.2f}" in text

    for _, row in clustering.iterrows():
        assert f"{int(row['n_groups']):,}" in supplement, row["clustering"]
    # The lower bound the main text quotes must be the true floor.
    assert float(clustering["ci_low"].min()) > 0.28
    assert "stays above 0.28" in text


def test_supplement_tables_are_typeset_in_the_order_they_are_cited():
    """A reader met Table 27 in the prose before Table 26, then found them in
    the opposite order on the page. Within the calibration section, the first
    table cited must be the first table typeset."""
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    section = supplement[
        supplement.index(r"\section{Calibrating the identity"):
        supplement.index(r"\section{Ascertainment across squad roles}")
    ]
    for label in ("tab:supp_curve", "tab:supp_calibration", "tab:supp_threshold"):
        first_reference = section.index(f"\\ref{{{label}}}")
        typeset_at = section.index(f"\\label{{{label}}}")
        assert first_reference < typeset_at, label
    # And the typeset order must follow the citation order, so the numbering
    # ascends down the page.
    references = [
        section.index(f"\\ref{{{label}}}")
        for label in ("tab:supp_curve", "tab:supp_calibration", "tab:supp_threshold")
    ]
    labels = [
        section.index(f"\\label{{{label}}}")
        for label in ("tab:supp_curve", "tab:supp_calibration", "tab:supp_threshold")
    ]
    assert references == sorted(references)
    assert labels == sorted(labels)
