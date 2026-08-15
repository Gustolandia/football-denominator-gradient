"""Gates tying the manuscript's claims to the tables that generate them.

The paper's contribution is a measurement, so every number it argues from has
to come from a deposited table and has to arrive with its uncertainty. These
tests read the tables, not the prose, and fail when the two drift apart. They
also refuse the specific overclaims that successive drafts have introduced:
saying a contamination is removed when its interval excludes zero, saying an
artefact reverses a conclusion when it only nullifies one, and saying an
effect is uniform across strata whose signs disagree.
"""

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
MANUSCRIPT = ROOT / "manuscript" / "manuscript.tex"
SUPPLEMENT = ROOT / "manuscript" / "supplement.tex"
RESULTS = ROOT / "data" / "processed" / "results"


def _between(text: str, start: str, end: str) -> str:
    """Return manuscript text between two unique markers."""
    return text.split(start, 1)[1].split(end, 1)[0]


def test_paper_is_positioned_against_the_two_papers_it_extends():
    """The contribution is the third leg of a numerator paper and an
    analytical one, so both must be cited and named as such."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    for key in (
        "hoenig2022citizen_transfermarkt",   # the numerator cannot be trusted
        "shrier2022_causal_workload",        # the offset is a constrained covariate
        "wang2024_immortal_time_load",       # the same group on the numerator's version
        "schisterman2009_overadjustment",    # the bias has a standard name
        "naimi2013_healthy_worker",          # and a precedent outside sport
        "ortiz2026_acl_top_five",            # a live study the argument applies to
    ):
        assert key in text, key

    intro = _between(text, "\\section{Introduction}", "\\section{Methods}")
    # The gap this paper fills has to be stated, not implied.
    assert "They say the bias exists" in intro
    assert "We measure it" in intro
    # The adjacent bias must be named as adjacent, not silently absorbed: one is
    # the numerator's misalignment and the other the denominator's.
    assert "immortal-time" in intro


def test_abstract_carries_the_measured_claim_with_uncertainty():
    text = MANUSCRIPT.read_text(encoding="utf-8")
    abstract = _between(text, r"\begin{abstract}", r"\end{abstract}")

    decomposition = pd.read_csv(
        RESULTS / "jsams_revised_denominator_attenuation_decomposition.csv"
    ).set_index("quantity")
    gamma = decomposition.loc["gamma_log_minutes_on_exposure"]
    assert (
        f"{gamma['value']:.3f} ({gamma['ci_low']:.3f}--{gamma['ci_high']:.3f})"
        in abstract
    )

    boot = pd.read_csv(RESULTS / "jsams_revised_attenuation_bootstrap.csv").set_index(
        "quantity"
    )
    for key in ("all", "starting_lineup", "role_adjusted_absolute"):
        row = boot.loc[key]
        assert (
            f"{row['estimate']:.3f} ({row['ci_low']:.3f}--{row['ci_high']:.3f})"
            in abstract
        ), key
    for key in ("substitute_list", "lineup_unavailable_or_other"):
        assert f"{boot.loc[key, 'estimate']:.3f}" in abstract, key

    summary = pd.read_csv(
        RESULTS / "jsams_revised_denominator_gradient_summary.csv"
    ).set_index("quantity")["value"]
    assert f"{summary['gamma_pooled_min']:.2f} to {summary['gamma_pooled_max']:.2f}" in abstract
    assert (
        f"{summary['gamma_within_starters_min']:.3f} to "
        f"{summary['gamma_within_starters_max']:.3f}" in abstract
    )
    assert "without exception" in abstract


def test_every_lineup_stratum_reaches_the_main_text():
    """A reader must not have to open the Supplement to learn that the remedy
    fails outside the starter stratum."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    roles = pd.read_csv(RESULTS / "jsams_revised_denominator_by_lineup_role.csv")
    per_role = roles.drop_duplicates("lineup_role").set_index("lineup_role")

    for role in ("all", "starting_lineup", "substitute_list", "lineup_unavailable_or_other"):
        row = per_role.loc[role]
        assert f"{int(row['n_rows']):,}" in text, role
        assert f"{int(row['n_players']):,}" in text, role
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
        assert f"{row['value']:.3f} ({row['ci_low']:.3f}--{row['ci_high']:.3f})" in text

    # The within-starter interval must be separated from every other, and the
    # paper must say so rather than leaving the reader to compare bounds.
    starter_high = float(decomposition.loc["gamma_within_starting_lineup", "ci_high"])
    for quantity in strata:
        if quantity != "gamma_within_starting_lineup":
            assert starter_high < float(decomposition.loc[quantity, "ci_low"]), quantity
    assert "disjoint from all three" in text
    assert "player-clustered errors" in text


def test_cross_league_gradients_match_the_generated_table():
    """The generalisation claim is the paper's strongest sentence, so every
    league's numbers have to come from the table that produced them."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    leagues = pd.read_csv(RESULTS / "jsams_revised_denominator_gradient_by_league.csv")
    assert len(leagues) == 8

    for _, row in leagues.iterrows():
        assert str(row["league"]) in text, row["league"]
        assert f"{int(row['n_appearances']):,}" in text
        assert (
            f"{row['gamma_pooled']:.3f} ({row['gamma_pooled_ci_low']:.3f}--"
            f"{row['gamma_pooled_ci_high']:.3f})" in text
        ), row["league"]
        assert (
            f"{row['gamma_within_starters']:.3f} "
            f"({row['gamma_within_starters_ci_low']:.3f}--"
            f"{row['gamma_within_starters_ci_high']:.3f})" in text
        ), row["league"]

    # The replication is only worth reporting because it is exceptionless.
    rule = pd.read_csv(RESULTS / "jsams_revised_denominator_gradient_decision_rule.csv")
    assert rule["recommendation"].eq("restrict to starters").all()
    assert int(leagues["n_appearances"].sum()) > 600_000
    assert f"{int(leagues['n_appearances'].sum()):,}" in text

    summary = pd.read_csv(
        RESULTS / "jsams_revised_denominator_gradient_summary.csv"
    ).set_index("quantity")["value"]
    assert summary["n_leagues_pooled_above_threshold"] == summary["n_leagues"]
    assert summary["n_leagues_starters_below_threshold"] == summary["n_leagues"]
    # The two-line shortcut is only recommendable if it lands near the adjusted fit.
    assert f"{summary['max_abs_gap_adjusted_vs_unadjusted']:.3f}" in text


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
    assert f"{low:.2f}\\%" in text and f"{high:.2f}\\%" in text
    assert low < 0.0 < high
    assert "includes zero" in text

    case = pd.read_csv(RESULTS / "jsams_revised_case_restricted_exposure_bias.csv")
    indexed = case.set_index("quantity")["percent_understated"]
    cohort = float(indexed["total_minutes_whole_cohort"])
    restricted = float(indexed["mean_recorded_minutes_on_event_appearances"])
    assert f"{restricted:.1f}\\%" in text and f"{cohort:.2f}\\%" in text
    assert f"a factor of {round(restricted / cohort)}" in text


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
    assert "driven by starters" in text
    assert "substitutes show no shift" in text

    low = float(indexed.loc["substitute_list", "difference_ci_low"])
    high = float(indexed.loc["substitute_list", "difference_ci_high"])
    assert low < 0.0 < high


def test_illness_control_is_reported_as_unestimable_not_failed():
    """An unestimable control and a large one have opposite implications."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    controls = pd.read_csv(
        RESULTS / "jsams_revised_negative_control_outcomes.csv"
    ).set_index("event_col")
    illness = controls.loc["same_day_illness_report"]
    assert not bool(illness["estimable"])
    assert int(illness["n_events"]) == 1

    assert "could not be estimated" in text
    assert "uninformative rather than failed" in text
    assert "failed illness control" not in text


def test_negative_control_values_match_the_deposited_table():
    """The mutually adjusted negative-control estimates appear in the main text
    at two decimals and in the Supplement at three. Both must be the deposited
    values rounded once from source -- not roundings of roundings, which is how
    1.4247 once became 1.43 by passing through 1.425."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    table = pd.read_csv(RESULTS / "jsams_revised_placebo_window_analysis.csv")
    indexed = table.set_index(["model_id", "focal_window"])

    alone = indexed.loc[("placebo_31_37d_alone", "prior_minutes_placebo_31_37d")]
    recent = indexed.loc[("both_windows", "prior_minutes_7d")]
    placebo = indexed.loc[("both_windows", "prior_minutes_placebo_31_37d")]

    for row in (alone, recent, placebo):
        assert (
            f"{row['estimate']:.2f} ({row['ci_low']:.2f}--{row['ci_high']:.2f})"
            in text
        ), row["description"]
        assert (
            f"{row['estimate']:.3f} ({row['ci_low']:.3f}--{row['ci_high']:.3f}"
            in supplement
        ), row["description"]

    # The contrast the sentence draws must hold in the table: the recent window
    # survives mutual adjustment and the placebo does not.
    assert float(recent["ci_low"]) > 1.0
    assert float(placebo["ci_low"]) < 1.0 < float(placebo["ci_high"])


def test_placebo_replicates_the_identity_over_prediction():
    """The identity's twofold over-prediction must be shown to be a property
    of the denominator rather than of the exposure chosen."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    both = text + SUPPLEMENT.read_text(encoding="utf-8")
    placebo = pd.read_csv(
        RESULTS / "jsams_revised_placebo_denominator_replication.csv"
    ).set_index("quantity")
    reference = pd.read_csv(
        RESULTS / "jsams_revised_denominator_attenuation_decomposition.csv"
    ).set_index("quantity")["value"]

    gamma = placebo.loc["gamma_placebo"]
    assert (
        f"{gamma['value']:.3f} ({gamma['ci_low']:.3f}--{gamma['ci_high']:.3f})" in text
    )
    ratio = float(placebo.loc["gamma_over_observed_attenuation", "value"])
    assert f"{ratio:.2f}" in text
    assert f"{float(reference['gamma_over_observed_attenuation']):.2f}" in text
    # The two ratios must actually agree, or the sentence claiming they do is
    # not supported.
    assert abs(ratio - float(reference["gamma_over_observed_attenuation"])) < 0.10
    assert "1.99" in both and "2.01" in both


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
    assert "nullify a conclusion" in text

    # Adjusting is not offered as an equivalent to restricting.
    assert "within starters}, or adjust for squad role" not in text
    assert "linear predictor" in text

    # Referring words that earlier compressions left stranded.
    for loose in ("disjoint from both", "overlaps neither.", "and adjusting no substitute"):
        assert loose not in text, loose


def test_main_displays_match_the_reframed_claim():
    """Two figures and three tables, all serving the denominator claim; the
    worked-example and audit displays belong in the Supplement."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")

    assert text.count(r"\begin{figure}") == 2
    assert text.count(r"\begin{table}") == 3
    for figure in ("J1_jsams_cohort_measurement.png", "J2_jsams_denominator_gradient.png"):
        assert figure in text, figure
    for demoted in (
        "J3_jsams_within_player_lineup_coverage.png",
        "J4_jsams_context_support.png",
        "J5_jsams_negative_control_exposure.png",
        "J6_jsams_primary_robustness.png",
    ):
        assert demoted not in text, demoted
        assert demoted in supplement, demoted

    assert "Return to Sport" not in text
    assert "Bias (Epidemiology)" in text


def test_worked_example_is_not_presented_as_a_finding():
    """The association is the thing being divided, not a result, and the paper
    has to say so where a reader would otherwise take it for one."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    assert "worked example, not a claim" in text
    assert "not as a finding" in text
    association = pd.read_csv(
        RESULTS / "jsams_revised_squad_role_association_sensitivity.csv"
    ).set_index("analysis")
    starter = association.loc["within starting_lineup"]
    # Three decimals: rounding 0.998 to 1.00 would disguise a null-spanning
    # interval as one that merely touches the null.
    assert (
        f"{starter['estimate']:.2f} ({starter['ci_low']:.3f}--{starter['ci_high']:.3f}"
        in text
    )
    assert float(starter["ci_low"]) < 1.0


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
    assert justification.str.len().gt(80).all()
    assert justification.str.contains("Tier").all()


# What each abstract-visible claim must actually say in the abstract. The
# registry alone cannot enforce this: it can mark a claim abstract-visible while
# the abstract says nothing about it, which is exactly how the reframe left the
# outcome audit stranded. The mapping ties the two together.
ABSTRACT_MARKERS = {
    "cumulative_recent_exposure_same_day_association": ("1.27",),
    "independent_same_day_outcome_audit": ("attribution", "88.9"),
    "reported_event_duration_linkage": ("90 to 53", "truncated"),
    "denominator_gradient_measured_and_replicated": ("gradient", "eight"),
    "denominator_gradient_decision_rule": ("before", "reported"),
    "truncation_explains_none_of_the_attenuation": ("explained none",),
}


def test_every_abstract_visible_claim_actually_reaches_the_abstract():
    """The registry decides what belongs in the abstract, so the abstract has to
    contain it. Without this the two drift silently: a claim can sit in the
    registry marked abstract-visible while the abstract never mentions it, which
    is what the reframe did to the outcome-attribution audit."""
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

    for comparison, words in (
        ("starting_lineup", ("lost", "fewer", "driven by starters")),
        ("substitute_list", ("gained", "more", "no shift")),
    ):
        signed = float(indexed.loc[comparison, "event_minus_non_event_minutes"])
        magnitude = abs(signed)
        lines = [
            line
            for line in text.splitlines()
            if f"{magnitude:.1f}" in line or f"{magnitude:.2f}" in line
        ]
        assert lines, comparison
        assert any(
            any(word in line for word in words)
            or f"{signed:.2f}" in line
            or f"$-{magnitude:.2f}$" in line
            for line in lines
        ), comparison


def test_scoping_count_is_reported_as_a_floor():
    """The count motivates the paper, so it has to match the deposited records
    and must never be presented as exhaustive."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    summary = pd.read_csv(
        RESULTS / "jsams_revised_per_minute_denominator_scoping.csv"
    ).set_index("quantity")["value"]

    assert f"seven peer-reviewed studies" in text
    assert int(summary["n_records_retrieved"]) == 7
    assert int(summary["n_denominator_confirmed"]) == 5
    assert int(summary["n_denominator_unverified"]) == 2
    assert "a floor, not a ceiling" in text

    # Unclassified records may not be quietly folded into the confirmed count.
    records = pd.read_csv(ROOT / "data" / "manual" / "per_minute_denominator_scoping.csv")
    assert len(records) == int(summary["n_records_retrieved"])
    assert records["source_url"].str.startswith("http").all()

    # Every study the count rests on has to be citable by the reader, not just
    # deposited: an uncited count cannot be checked from the paper alone. Seven
    # records, seven citations -- a reader who counts must find them agreeing.
    for key in SCOPING_KEYS:
        assert key in text, key

    # A confirmed record carries the sentence that confirmed it.
    confirmed = records[records["gradient_applies"].astype(str).eq("yes")]
    assert len(confirmed) == 5
    assert confirmed["denominator_verbatim"].str.len().gt(30).all()

    # One confirmed record carries a provenance flag, and the prose has to be
    # exactly as cautious as the deposited table about it.
    assert int(summary["n_provenance_flagged"]) == 1
    assert "provenance flag" in text

    # And the strategy that produced the count travels with it.
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
    """Count main-text words the way the journal does: Introduction through
    Conclusion, excluding floats, captions, citations and the back matter."""
    body = tex[tex.index("\\section{Introduction}"):tex.index("\\section*{Practical implications}")]
    for environment in ("table", "figure"):
        body = re.sub(
            rf"\\begin\{{{environment}\}}.*?\\end\{{{environment}\}}", " ", body, flags=re.S
        )
    body = re.sub(r"\\FloatBarrier", " ", body)
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


def test_journal_limits_and_satellite_documents_agree():
    """The journal's limits are requirements, so the suite enforces them; and
    every count a satellite document states must be the computed one. Three
    rounds running, a satellite carried stale numbers or a stale title --- the
    supplement's title, the cover letter's word count, then the entire title
    page --- because compiled documents fail loudly and stated counts fail
    silently. This gate makes them fail loudly."""
    tex = MANUSCRIPT.read_text(encoding="utf-8")

    main_words = _main_text_word_count(tex)
    abstract_words = _abstract_word_count(tex)
    assert main_words <= 5000, main_words
    assert abstract_words <= 250, abstract_words

    implications = tex[
        tex.index("\\section*{Practical implications}"):tex.index("\\section*{Funding}")
    ]
    assert 3 <= implications.count("\\item") <= 5

    assert tex.count(r"\begin{table}") + tex.count(r"\begin{figure}") <= 6

    cited = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", tex):
        cited.update(key.strip() for key in group.split(","))
    assert len(cited) <= 40, len(cited)

    # No citation cluster may exceed three keys: the guide allows at most
    # three references per point.
    for group in re.findall(r"\\cite\{([^}]*)\}", tex):
        assert len(group.split(",")) <= 3, group

    # The title page must describe this manuscript, not a previous one.
    title_page = (ROOT / "manuscript" / "title_page.tex").read_text(encoding="utf-8")
    manuscript_title = re.search(
        r"\\newcommand\{\\papertitle\}\{([^}]*)\}", tex
    ).group(1)
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
    assert "Tables: 3. Figures: 2." in title_page

    # The cover letter may not hard-code a word count at all; a stated number
    # goes stale with the next edit, and did.
    letter = (ROOT / "manuscript" / "cover_letter.md").read_text(encoding="utf-8")
    assert not re.search(r"\b\d[\d,]* words", letter)
    assert "5,000-word" in letter


def test_figures_are_current_and_in_both_formats(load_src_module):
    """A figure drawn from a table that has since changed fails here rather
    than waiting for a reader: the manifest records what script 35 drew, and
    the league labels it drew must equal the league table as it stands now.
    Every included figure must also exist in the journal-accepted vector
    format beside the PNG the sources include."""
    figures_dir = ROOT / "manuscript" / "figures"
    both = MANUSCRIPT.read_text(encoding="utf-8") + SUPPLEMENT.read_text(
        encoding="utf-8"
    )
    included = set(re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}", both))
    assert len(included) == 6
    for name in included:
        assert (figures_dir / name).exists(), name
        assert (figures_dir / name).with_suffix(".pdf").exists(), name

    manifest = pd.read_csv(
        RESULTS / "jsams_revised_figure_manifest.csv"
    ).set_index("figure")
    assert manifest["formats"].eq("png+pdf").all()
    drawn = str(manifest.loc["J2_jsams_denominator_gradient", "league_labels"])
    leagues = pd.read_csv(RESULTS / "jsams_revised_denominator_gradient_by_league.csv")
    assert drawn == "|".join(str(label) for label in leagues["league"])

    # And the figure was drawn from these numbers, not merely from a table with
    # the same league names. This was a comparison of file modification times
    # until the archived record was unpacked and every source file arrived with
    # one packaging timestamp, which made a current figure look stale. The
    # digest travels inside the deposited table and asks the sharper question.
    plotting = load_src_module("35_plot_jsams_revision.py")
    assert (
        str(manifest.loc["J2_jsams_denominator_gradient", "source_digest"])
        == plotting.gradient_source_digest(leagues)
    )
    # The figures that plot no league table record no digest to compare.
    assert (
        manifest.drop(index="J2_jsams_denominator_gradient")["source_digest"]
        .fillna("")
        .eq("")
        .all()
    )


# The arXiv record this work is posted on. Declaring it is a journal
# requirement, and the declaration went stale the moment the preprint existed.
ARXIV_ID = "2608.11228"


def test_preprint_is_declared_where_the_journal_requires_it():
    """The guide asks the corresponding author to state whether the paper is on
    a preprint server, with the link, in the cover letter *and* on the title
    page. Both said there was none, which stopped being true when the
    replacement was submitted. A stated negative goes stale silently, so it is
    gated in both places."""
    title_page = (ROOT / "manuscript" / "title_page.tex").read_text(encoding="utf-8")
    letter = " ".join(
        (ROOT / "manuscript" / "cover_letter.md").read_text(encoding="utf-8").split()
    )

    for source, label in ((title_page, "title page"), (letter, "cover letter")):
        assert ARXIV_ID in source, label
        assert "arxiv.org/abs/" in source.lower(), label
        # The old negative must not survive anywhere alongside the new positive.
        for stale in ("Preprint: none", "not available on any preprint"):
            assert stale not in source, (label, stale)

    # The journal also asks for the preprint to be annotated once under review;
    # the letter commits to that so the obligation is not forgotten.
    assert "under peer review" in letter


# The archived record this paper is deposited inside. The DOI is reserved
# before publication precisely so the archived PDF can cite the record it
# sits in; a paper that cites a different record is not a replication object.
DEPOSIT_DOI = "10.5281/zenodo.21940597"


def test_paper_cites_the_record_it_is_archived_inside():
    """The deposit is one object: paper, code and data under one DOI. The
    manuscript and the title page must both cite that DOI, and must not
    describe the archive as forthcoming once it exists."""
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    title_page = (ROOT / "manuscript" / "title_page.tex").read_text(encoding="utf-8")

    for source, label in ((manuscript, "manuscript"), (title_page, "title page")):
        assert DEPOSIT_DOI in source, label
        for stale in ("in preparation", "archive of this release is in"):
            assert stale not in source, (label, stale)

    # The earlier records stay cited, as earlier versions rather than as this one.
    for previous in ("10.5281/zenodo.21673177", "10.5281/zenodo.21673210"):
        assert previous in manuscript, previous
    assert "earlier versions" in manuscript


def test_acknowledgements_survive_in_every_build():
    """The guide makes an Acknowledgements section compulsory. The manuscript
    has two back-matter branches and the anonymised one must still carry the
    section while deferring its content to the title page, so a change to one
    branch cannot silently drop it from the other."""
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert manuscript.count("\\section*{Acknowledgements}") == 2

    blind = _blind_projection(manuscript)
    assert "\\section*{Acknowledgements}" in blind
    # Named individuals belong on the title page, never in the blinded copy.
    for name in ("James Ng", "Adam Field"):
        assert name not in blind, name
        assert name in manuscript, name

    title_page = (ROOT / "manuscript" / "title_page.tex").read_text(encoding="utf-8")
    assert "Acknowledgements" in title_page
    for name in ("James Ng", "Adam Field"):
        assert name in title_page, name


def test_practical_implications_speak_to_a_lay_audience():
    """The guide addresses this one section to a coach or clinician: bullet
    points a lay audience can understand, avoiding overly scientific terms and
    abbreviations, with no further-research recommendations. The technical
    formulation lives in the Discussion; the bullets may not lean on it."""
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    implications = tex[
        tex.index("\\section*{Practical implications}"):tex.index("\\section*{Funding}")
    ]
    assert 3 <= implications.count("\\item") <= 5

    # No mathematics, no citations, no statistician's vocabulary.
    for banned in (
        "$", "\\cite", "\\pval", "regress", "log scale", "clustered",
        "point estimate", "coefficient", "denominate",
    ):
        assert banned not in implications, banned
    # And no recommendations for further research, per the guide.
    for banned in ("further research", "future research", "future studies"):
        assert banned not in implications.lower(), banned

    # The bullets still have to carry the paper's practical content.
    for required in ("two-line", "starters", "per appearance"):
        assert required in implications, required


def test_imputation_claim_matches_the_deposited_ordering():
    """The truncation headline may not claim to be the least alarming scheme
    when the deposited table shows one scheme below it."""
    table = pd.read_csv(
        RESULTS / "jsams_revised_truncation_imputation_sensitivity.csv"
    )
    case_column = next(
        column for column in table.columns if "case_restricted" in column
    )
    values = pd.to_numeric(table[case_column], errors="coerce")
    reference = float(values.iloc[0])
    # The fact itself: the reference sits near, but not at, the minimum.
    assert reference > float(values.min())
    assert reference < float(values.median())

    text = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    for source in (text, supplement):
        assert "is the least alarming" not in source
        assert "least alarming value the reasonable schemes produce" not in source
    assert "least-alarming end" in text
    assert "least-alarming end" in supplement


def _blind_projection(source: str) -> str:
    """Render what the anonymized build keeps: the if-branch of every
    \\ifdefined\\journalblindcopy block, and everything outside them."""
    return re.sub(
        r"\\ifdefined\\journalblindcopy(.*?)(?:\\else(.*?))?\\fi",
        lambda match: match.group(1),
        source,
        flags=re.S,
    )


def _bib_blocks() -> dict:
    """Split references.bib into per-key entry bodies."""
    bib = (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8")
    blocks = {}
    for chunk in bib.split("@")[1:]:
        head, _, body = chunk.partition("{")
        key = body.split(",", 1)[0].strip()
        blocks[key] = body
    return blocks


def test_references_the_count_rests_on_are_complete():
    """A reference with no authors or no journal cannot be retrieved, and the
    scoping count is only checkable if every study behind it can be. Each of
    the seven must carry authors, a venue, a year and a DOI, and the entries
    must not be reconstructed from memory: the metadata was read from the
    primary source or its repository record."""
    blocks = _bib_blocks()
    # A field is present only when it appears as a field, at the start of its
    # own line: matching the bare word would find "year" inside "5-year".
    def has_field(body: str, field: str) -> bool:
        return re.search(rf"(?m)^\s*{field}\s*=", body) is not None

    for key in SCOPING_KEYS + (
        "wang2024_immortal_time_load", "bachemathiesen2024_acute_chronic",
        "kronmal1993_ratio_standard", "nevill1995_allometric",
    ):
        assert key in blocks, key
        for field in ("author", "journal", "year", "doi"):
            assert has_field(blocks[key], field), (key, field)

    # The seventh study's author list was initially truncated behind a paywall;
    # the repository record supplied it in full, and "and others" may not creep
    # back in place of the five named authors.
    assert "Sonnery-Cottet" in blocks["dambrosi2026_rtp_acl"]
    assert "and others" not in blocks["dambrosi2026_rtp_acl"]

    # The lineage claims must be in the text, not just retrievable: the paper
    # has statistical ancestors and owns them.
    text = MANUSCRIPT.read_text(encoding="utf-8")
    for key in ("kronmal1993_ratio_standard", "nevill1995_allometric"):
        assert key in text, key


def test_submission_artifacts_exist_and_are_blind():
    """JSAMS review is double-anonymized: the uploaded manuscript must carry no
    author identity, which lives in the separate title page. The blind wrapper
    has to define the macro before it inputs the manuscript, and the manuscript
    itself has to blank the author block under that macro."""
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert "\\ifdefined\\journalblindcopy" in manuscript
    assert "\\author{}" in manuscript

    wrapper = (ROOT / "manuscript" / "manuscript_blind.tex").read_text(encoding="utf-8")
    define = wrapper.index("\\journalblindcopy")
    assert define < wrapper.index("manuscript.tex")
    # Line numbers belong on the review copy, so referees can point at lines.
    assert "\\journalreviewcopy" in wrapper

    # What must be anonymous is the blind rendering, not the raw source: the
    # source legitimately carries names inside \else branches. Projecting the
    # blind branch of every \ifdefined\journalblindcopy block gives the text a
    # referee would actually receive, and that text may identify nobody.
    blind = _blind_projection(manuscript)
    for identifier in (
        "Trinity College Dublin", "Ricou", "Mahony", "njmahony", "pedrorig",
        "tcd2026_ethics_guidance",
    ):
        assert identifier not in blind, identifier
    # The redaction macro, not the literal name, is what the ethics sentence
    # uses; the name may live only in the non-blind branch.
    assert "\\ethicsinstitution" in blind
    assert "Trinity College Dublin" in manuscript

    # The supplement travels to referees too, so it needs the same treatment:
    # a blanked author line under the macro and its own wrapper.
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    assert "\\ifdefined\\journalblindcopy" in supplement
    assert "\\author{}" in supplement
    for identifier in ("Ricou", "Mahony"):
        assert identifier not in _blind_projection(supplement), identifier
    supplement_wrapper = (ROOT / "manuscript" / "supplement_blind.tex").read_text(
        encoding="utf-8"
    )
    assert supplement_wrapper.index("\\journalblindcopy") < supplement_wrapper.index(
        "supplement.tex"
    )

    # Whitespace-normalised so hard-wrapped prose cannot hide a phrase.
    letter = " ".join(
        (ROOT / "manuscript" / "cover_letter.md")
        .read_text(encoding="utf-8")
        .split()
    )
    # The fit argument names the venue's own prior work rather than implying it.
    assert "Shrier" in letter
    assert "Journal of Science and Medicine in Sport" in letter
    # The ethics position is stated to the editor, not left to be discovered.
    assert "ethic" in letter.lower()
    assert "General Data Protection Regulation" in letter
    assert "no determination was obtained" in letter
    # The guide's mandatory cover-letter contents: category, sub-discipline,
    # the funding position in the journal's own formula, and preprint status.
    assert "Original Research" in letter
    assert "Sport Injury" in letter
    assert "not-for-profit sectors" in letter
    assert "preprint" in letter.lower()

    # The supplement's review wrapper carries line numbers like the
    # manuscript's, so referees can point at supplement lines too.
    supplement_wrapper = (ROOT / "manuscript" / "supplement_blind.tex").read_text(
        encoding="utf-8"
    )
    assert "\\journalreviewcopy" in supplement_wrapper


def test_missed_event_queue_was_searched_and_resolved_nothing():
    """The queue is the paper's one acknowledged hole. It has now been searched
    under the pre-specified protocol and returned no verdict on any record, so
    the manuscript must report zero confirmed missed events as the absence of an
    estimate and never as a reassuring one."""
    text = MANUSCRIPT.read_text(encoding="utf-8")
    protocol = pd.read_csv(
        RESULTS / "jsams_revised_missed_event_adjudication_protocol.csv"
    ).set_index("rule_id")
    assert "B7_absence_of_evidence" in protocol.index

    audit = pd.read_csv(RESULTS / "jsams_revised_non_event_audit_summary.csv").iloc[0]
    assert int(audit["n_sampled"]) == 30
    assert int(audit["n_unresolved"]) == 30
    assert int(audit["n_resolved"]) == 0
    # Nothing resolved, so the proportion is undefined rather than zero, and the
    # status may not read as a finished audit.
    assert pd.isna(audit["missed_event_proportion"])
    assert str(audit["status"]) == "reviewed, not resolved"

    # The screen that ordered the search reports what it found, and the count of
    # searchable records in the manuscript must match it.
    screen = pd.read_csv(RESULTS / "jsams_revised_non_event_absence_screen.csv")
    assert len(screen) == 30
    assert "assigns no verdicts" in screen["interpretation"].iloc[0]
    assert int(screen["club_fixtures_missed"].eq(0).sum()) == 28
    assert "in 28 of 30 the player's club played no fixture" in text
    # It reports no verdict of its own, which is what keeps it a screen.
    assert "missed_event_verdict" not in screen.columns

    assert "all 30 remain unadjudicated" in text
    assert "never to no missed event" in text
    assert "lower bound on missed events and never an upper bound" in text
    assert "Zero confirmed missed events is therefore not an estimate of zero" in text
    assert "remains unestimated" in text
    for reassurance in ("no missed events were found", "sensitivity is adequate"):
        assert reassurance not in text, reassurance
