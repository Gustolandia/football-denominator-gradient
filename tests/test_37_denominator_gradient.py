"""Tests for the denominator-gradient diagnostic.

The gradient is the quantity the paper asks readers to compute before they
divide by playing time, so these tests check that it means what the paper
says it means: that it rises when playing time tracks the exposure, that it
collapses when it does not, and that the decision rule reads interval bounds
rather than point estimates.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _panel(module, players: int = 40, appearances: int = 20, coupled: bool = True,
           competition: str = "GB1", seed: int = 7) -> pd.DataFrame:
    """Build an appearance panel where minutes may or may not track exposure.

    With ``coupled`` the player alternates between long and short outings in
    runs, so recent minutes predict the current appearance length -- the real
    pattern that squad rotation produces. Without it, appearance length is
    drawn independently of history.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for player in range(players):
        start = pd.Timestamp("2018-08-01") + pd.Timedelta(days=int(rng.integers(0, 20)))
        regular = player % 2 == 0
        for index in range(appearances):
            date = start + pd.Timedelta(days=4 * index + int(rng.integers(0, 2)))
            if coupled:
                minutes = 90.0 if regular else 15.0 + rng.normal(0.0, 2.0)
            else:
                minutes = float(rng.choice([90.0, 15.0])) + rng.normal(0.0, 2.0)
            rows.append(
                {
                    "player_id": 1000 + player,
                    "date": date,
                    "minutes_played": max(1.0, minutes),
                    "competition_id": competition,
                    "lineup_role": "starting_lineup" if minutes > 60 else "substitutes",
                }
            )
    frame = pd.DataFrame(rows)
    windowed = module.add_prior_window_minutes(frame)
    return module.add_calendar_phase(windowed)


def test_calendar_phase_needs_complete_dates(load_src_module):
    module = load_src_module("37_denominator_gradient.py")
    frame = pd.DataFrame({"date": ["2019-01-05", "2019-01-12"]})
    out = module.add_calendar_phase(frame)
    for column in module.CALENDAR_TERMS:
        assert column in out.columns
    # The phase terms are bounded trigonometric functions of weekday.
    assert out[list(module.CALENDAR_TERMS)].abs().le(1.0 + 1e-9).all().all()

    with pytest.raises(ValueError, match="complete appearance dates"):
        module.add_calendar_phase(pd.DataFrame({"date": ["2019-01-05", None]}))
    with pytest.raises(KeyError, match="calendar frame missing columns"):
        module.add_calendar_phase(pd.DataFrame({"other": [1]}))


def test_prior_window_excludes_the_current_appearance(load_src_module):
    module = load_src_module("37_denominator_gradient.py")
    frame = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "date": pd.to_datetime(["2019-01-01", "2019-01-04", "2019-01-20"]),
            "minutes_played": [90.0, 60.0, 30.0],
        }
    )
    out = module.add_prior_window_minutes(frame, window=7).set_index("date")
    # First appearance has no history; the second sees only the first; the
    # third falls outside the window and sees nothing.
    assert out.loc["2019-01-01", "prior_minutes_7d"] == 0.0
    assert out.loc["2019-01-04", "prior_minutes_7d"] == 90.0
    assert out.loc["2019-01-20", "prior_minutes_7d"] == 0.0

    with pytest.raises(ValueError, match="positive number of days"):
        module.add_prior_window_minutes(frame, window=0)
    with pytest.raises(KeyError, match="window frame missing columns"):
        module.add_prior_window_minutes(pd.DataFrame({"player_id": [1]}))


def test_prior_window_drops_ambiguous_player_dates(load_src_module):
    """A player recorded twice on one date has no defined ordering, so both
    rows are dropped rather than summed into a guess."""
    module = load_src_module("37_denominator_gradient.py")
    frame = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "date": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-01-01"]),
            "minutes_played": [90.0, 45.0, 90.0],
        }
    )
    out = module.add_prior_window_minutes(frame)
    assert list(out["player_id"]) == [2]


def test_gradient_rises_when_minutes_track_exposure(load_src_module):
    """The gradient is the whole diagnostic: it must be large when playing
    time follows recent playing time and small when it does not."""
    module = load_src_module("37_denominator_gradient.py")
    coupled = module.denominator_gradient(_panel(module, coupled=True))
    independent = module.denominator_gradient(_panel(module, coupled=False, seed=11))

    assert coupled["estimable"] and independent["estimable"]
    assert coupled["gamma"] > independent["gamma"]
    assert coupled["ci_low"] <= coupled["gamma"] <= coupled["ci_high"]
    assert coupled["ci_low"] > 0.0
    assert coupled["n_players"] == 40

    # The two-line unadjusted form should reach the same conclusion.
    plain = module.denominator_gradient(_panel(module, coupled=True), adjusted=False)
    assert plain["gamma"] > 0.0

    with pytest.raises(KeyError, match="gradient frame missing columns"):
        module.denominator_gradient(pd.DataFrame({"player_id": [1]}))


def test_gradient_withholds_thin_panels(load_src_module):
    """Too few rows means no estimate rather than an unstable one."""
    module = load_src_module("37_denominator_gradient.py")
    thin = _panel(module, players=3, appearances=4)
    out = module.denominator_gradient(thin)
    assert not out["estimable"]
    assert np.isnan(out["gamma"]) and np.isnan(out["ci_low"])


def test_gradient_by_league_reports_pooled_and_within_starters(load_src_module):
    module = load_src_module("37_denominator_gradient.py")
    england = _panel(module, competition="GB1", seed=3)
    spain = _panel(module, competition="ES1", seed=5)
    out = module.gradient_by_league(pd.concat([spain, england], ignore_index=True))

    # Leagues are ordered against the reference league, not by input order.
    assert list(out["competition_id"]) == ["GB1", "ES1"]
    assert out["league"].iloc[0].startswith("England")
    for _, row in out.iterrows():
        assert row["gamma_pooled"] > row["gamma_within_starters"]
        assert row["iqr_starter_minutes"] <= row["iqr_recorded_minutes"]
        assert row["n_starter_appearances"] < row["n_appearances"]
        assert row["window_days"] == module.PRIMARY_WINDOW

    with pytest.raises(KeyError, match="league frame missing columns"):
        module.gradient_by_league(pd.DataFrame({"competition_id": ["GB1"]}))


def test_gradient_by_league_handles_a_league_without_lineup_roles(load_src_module):
    """Lineup status is the one field that is routinely missing, so a league
    without it still reports a pooled gradient and an absent within-starter
    one rather than failing."""
    module = load_src_module("37_denominator_gradient.py")
    frame = _panel(module, competition="XX1").drop(columns=["lineup_role"])
    out = module.gradient_by_league(frame)
    row = out.iloc[0]
    assert row["gamma_pooled"] > 0.0
    assert np.isnan(row["gamma_within_starters"])
    assert int(row["n_starter_appearances"]) == 0
    # An unrecognised competition falls back to its own code as a label.
    assert row["league"] == "XX1"


def test_decision_rule_reads_interval_bounds(load_src_module):
    """Three verdicts, each driven by a bound rather than a point estimate."""
    module = load_src_module("37_denominator_gradient.py")
    gradients = pd.DataFrame(
        {
            "league": ["material, starters clean", "negligible pooled", "material everywhere"],
            "gamma_pooled": [0.50, 0.02, 0.50],
            "gamma_pooled_ci_low": [0.45, 0.01, 0.45],
            "gamma_within_starters": [0.02, 0.01, 0.30],
            "gamma_within_starters_ci_high": [0.03, 0.02, 0.35],
        }
    )
    out = module.diagnostic_decision_rule(gradients).set_index("league")
    assert out.loc["material, starters clean", "recommendation"] == "restrict to starters"
    assert out.loc["negligible pooled", "recommendation"] == "per-minute defensible pooled"
    assert out.loc["material everywhere", "recommendation"] == "report per appearance"
    assert out["negligible_threshold"].eq(module.NEGLIGIBLE_GAMMA).all()

    # A league with no estimable within-starter bound cannot be told to
    # restrict to starters.
    unknown = gradients.head(1).assign(gamma_within_starters_ci_high=[np.nan])
    assert (
        module.diagnostic_decision_rule(unknown)["recommendation"].iloc[0]
        == "report per appearance"
    )

    with pytest.raises(KeyError, match="decision rule frame missing columns"):
        module.diagnostic_decision_rule(pd.DataFrame({"league": ["x"]}))


def test_gradient_summary_reduces_to_the_defensible_sentence(load_src_module):
    module = load_src_module("37_denominator_gradient.py")
    gradients = pd.DataFrame(
        {
            "gamma_pooled": [0.40, 0.60, np.nan],
            "gamma_pooled_unadjusted": [0.38, 0.57, np.nan],
            "gamma_within_starters": [0.02, 0.03, np.nan],
            "estimable": [True, True, False],
        }
    )
    out = module.gradient_summary(gradients).set_index("quantity")["value"]
    assert out["n_leagues"] == 2
    assert np.isclose(out["gamma_pooled_min"], 0.40)
    assert np.isclose(out["gamma_pooled_max"], 0.60)
    assert out["n_leagues_pooled_above_threshold"] == 2
    assert out["n_leagues_starters_below_threshold"] == 2
    # The two-line shortcut is only defensible if it lands near the adjusted
    # fit, so the largest gap across leagues is reported.
    assert np.isclose(out["max_abs_gap_adjusted_vs_unadjusted"], 0.03)

    # Without the unadjusted column or within-starter values the summary
    # still returns what it can.
    lean = gradients.drop(columns=["gamma_pooled_unadjusted"]).assign(
        gamma_within_starters=[np.nan, np.nan, np.nan]
    )
    lean_out = module.gradient_summary(lean).set_index("quantity")["value"]
    assert "max_abs_gap_adjusted_vs_unadjusted" not in lean_out.index
    assert "gamma_within_starters_min" not in lean_out.index

    # A panel that carries the unadjusted column but never fitted it reports
    # no gap rather than a gap computed from nothing.
    blank = gradients.assign(gamma_pooled_unadjusted=[np.nan, np.nan, np.nan])
    blank_out = module.gradient_summary(blank).set_index("quantity")["value"]
    assert "max_abs_gap_adjusted_vs_unadjusted" not in blank_out.index
    assert blank_out["n_leagues"] == 2

    with pytest.raises(KeyError, match="gradient summary frame missing columns"):
        module.gradient_summary(pd.DataFrame({"gamma_pooled": [0.1]}))
    with pytest.raises(ValueError, match="no estimable leagues"):
        module.gradient_summary(gradients.assign(estimable=False))


def test_scoping_summary_reports_a_floor_not_a_count(load_src_module):
    """A web-based scoping search can show that exposed studies exist and count
    the ones it found. It cannot show that no others do, so every quantity it
    produces has to be labelled as a lower bound, and records whose denominator
    was not read directly must not be folded into the confirmed count."""
    module = load_src_module("37_denominator_gradient.py")
    records = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "public_source": ["transfermarkt", "media_reports", "transfermarkt", "media_reports"],
            "gradient_applies": ["yes", "yes", "unknown", "yes"],
            "verification": [
                "rate_verbatim_confirmed",
                "rate_verbatim_confirmed_source_unverified",
                "public_source_confirmed_denominator_unverified",
                "rate_verbatim_confirmed",
            ],
        }
    )
    out = module.scoping_search_summary(records).set_index("quantity")["value"]
    assert out["n_records_retrieved"] == 4
    assert out["n_denominator_confirmed"] == 3
    assert out["n_denominator_unverified"] == 1
    assert out["n_provenance_flagged"] == 1
    # Confirmed and unverified must partition the records, so neither can
    # quietly absorb the other.
    assert out["n_denominator_confirmed"] + out["n_denominator_unverified"] == out["n_records_retrieved"]

    frame = module.scoping_search_summary(records)
    assert frame["bound"].eq("lower").all()
    assert frame["search_type"].eq("scoping").all()
    assert "never a ceiling" in frame["interpretation"].iloc[0]

    with pytest.raises(KeyError, match="scoping records missing columns"):
        module.scoping_search_summary(pd.DataFrame({"record_id": ["a"]}))


def test_deposited_scoping_records_are_internally_consistent(load_src_module):
    """The committed evidence file is the paper's claim, so it has to carry a
    resolvable source for every record and never mark a denominator confirmed
    without having read the rate."""
    module = load_src_module("37_denominator_gradient.py")
    root = Path(__file__).resolve().parents[1]
    records = pd.read_csv(root / "data" / "manual" / "per_minute_denominator_scoping.csv")

    assert records["record_id"].is_unique
    assert records["source_url"].str.startswith("http").all()
    assert records["search_query"].str.len().gt(0).all()
    assert set(records["gradient_applies"]) <= {"yes", "unknown", "no"}

    confirmed = records[records["gradient_applies"].eq("yes")]
    assert confirmed["verification"].str.startswith("rate_verbatim_confirmed").all()
    assert confirmed["denominator_verbatim"].str.len().gt(0).all()

    unverified = records[records["gradient_applies"].eq("unknown")]
    assert unverified["denominator_reported"].eq("not_established").all()

    summary = module.scoping_search_summary(records).set_index("quantity")["value"]
    assert summary["n_denominator_confirmed"] >= 3


def test_adjudication_protocol_forbids_treating_absence_as_a_negative(load_src_module):
    """The protocol's whole purpose is to stop a reviewer converting a failed
    search into evidence of no missed event, which would bias the sensitivity
    estimate in the paper's own favour."""
    module = load_src_module("37_denominator_gradient.py")
    out = module.adjudication_protocol().set_index("rule_id")

    for rule_id in (
        "B1_source_hierarchy", "B2_temporal_window", "B3_specificity",
        "B4_match_identification", "B5_verdict_set", "B6_assessors",
        "B7_absence_of_evidence",
    ):
        assert rule_id in out.index, rule_id
        assert len(str(out.loc[rule_id, "rule"])) > 40

    absence = str(out.loc["B7_absence_of_evidence", "rule"])
    assert "never to no_missed_event" in absence
    assert "unresolved" in absence

    # Aggregators must be excluded, or the audit checks the database against
    # a copy of itself.
    assert "circular" in str(out.loc["B1_source_hierarchy", "rule"])
    # Blinding and inter-rater agreement are part of the protocol, not optional.
    assert "kappa" in str(out.loc["B6_assessors", "rule"])
    assert "blinded" in str(out.loc["B6_assessors", "rule"])

    frame = module.adjudication_protocol()
    # The rules were applied, and applying them resolved nothing. Both halves
    # have to survive together, or the deposited status starts implying that a
    # completed search produced a negative finding.
    assert frame["status"].str.contains("applied").all()
    assert frame["status"].str.contains("no record resolved").all()
    assert "cannot yield an upper bound" in frame["interpretation"].iloc[0]


def test_scoping_protocol_states_its_own_limits(load_src_module):
    """The count motivates the paper, so how it was produced has to travel with
    it: which sources were searched, when, what counted as exposed, and the
    fact that a scoping search can only ever establish a floor."""
    module = load_src_module("37_denominator_gradient.py")
    out = module.scoping_search_protocol().set_index("rule_id")

    for rule_id in (
        "S1_question", "S2_sources_searched", "S3_search_dates", "S4_inclusion",
        "S5_denominator_classification", "S6_provenance_flag", "S7_bound",
    ):
        assert rule_id in out.index, rule_id
        assert len(str(out.loc[rule_id, "rule"])) > 40

    # It must not present itself as a systematic review.
    assert not module.scoping_search_protocol()["registered"].any()
    searched = str(out.loc["S2_sources_searched", "rule"])
    assert "No grey-literature database" in searched
    # A denominator may only be counted when it was read, never inferred.
    classification = str(out.loc["S5_denominator_classification", "rule"])
    assert "read directly" in classification
    assert "never counted as exposed" in classification
    assert "floor" in str(out.loc["S7_bound", "rule"])


def test_gradient_survives_the_floor_applied_before_the_log(load_src_module):
    """The reference fit logs minutes clipped at one. Short substitute
    appearances are common, so that floor is a modelling choice, and the
    contrast the paper claims must not depend on it."""
    module = load_src_module("37_denominator_gradient.py")
    panel = _panel(module)

    out = module.gradient_clip_sensitivity(panel)
    assert list(out["minute_floor"]) == [1.0, 5.0, 10.0]
    # The reference row is its own baseline.
    assert out["gamma_pooled_gap_vs_reference"].iloc[0] == pytest.approx(0.0)
    # Whatever the floor, minutes still track exposure pooled and barely move
    # within starters, which is the whole claim.
    assert (out["gamma_pooled"] > out["gamma_within_starters"]).all()
    assert "does not depend on it" in out["interpretation"].iloc[0]


def test_gradient_survives_estimators_that_do_not_assume_a_log(load_src_module):
    """Recorded minutes are heaped at a full appearance and bounded there, so
    least squares on their log is a choice. The composition result has to hold
    under estimators that make different assumptions about that shape."""
    module = load_src_module("37_denominator_gradient.py")
    panel = _panel(module)

    out = module.gradient_estimator_sensitivity(panel).set_index("estimator")
    for estimator in (
        "ols_log_minutes", "median_regression", "fractional_response",
        "excluding_ceiling",
    ):
        assert estimator in out.index, estimator

    # Wherever the contrast is interpretable, it points the same way.
    usable = out[out["contrast_interpretable"].astype(bool)]
    assert len(usable) == 3
    assert usable["pooled_exceeds_starters"].all()

    # Dropping the ceiling keeps, among starters, exactly the appearances that
    # ended early. That fit may still produce a number, and the number answers a
    # different question, so it must never be counted toward the robustness claim.
    ceiling = out.loc["excluding_ceiling"]
    assert not bool(ceiling["contrast_interpretable"])
    assert pd.isna(ceiling["pooled_exceeds_starters"])
    assert "not interpretable here" in str(ceiling["note"])
    assert "wherever the contrast is interpretable" in out["interpretation"].iloc[0]

    # The median fit is far flatter than the reference one, and the table has to
    # explain why rather than leave it looking like a contradiction.
    assert "expected rather than contradictory" in str(
        out.loc["median_regression", "note"]
    )
    assert "mean of log minutes" in out["interpretation"].iloc[0]


def test_gradient_estimators_are_reachable_individually(load_src_module):
    """Each estimator is a branch a reader might rerun, so each is fitted."""
    module = load_src_module("37_denominator_gradient.py")
    panel = _panel(module)

    median = module.denominator_gradient(panel, estimator="median")
    assert median["estimable"]
    # Quantile regression carries no clustered covariance, so it contributes a
    # point estimate and explicitly no interval.
    assert np.isnan(median["ci_low"]) and np.isnan(median["ci_high"])

    fractional = module.denominator_gradient(panel, estimator="fractional")
    assert fractional["estimable"]
    assert fractional["ci_low"] < fractional["gamma"] < fractional["ci_high"]

    below = module.denominator_gradient(panel, estimator="below_ceiling")
    # The ceiling rows are gone, so this fit sees strictly fewer appearances.
    assert below["n_rows"] < int(len(panel))


def test_gamma_pair_reports_an_absent_starter_stratum_rather_than_guessing(
    load_src_module,
):
    """A panel with no lineup field still yields a pooled gradient, and the
    within-starter one has to come back missing rather than silently pooled."""
    module = load_src_module("37_denominator_gradient.py")
    panel = _panel(module).drop(columns=["lineup_role"])

    assert module._starter_rows(panel).empty
    pooled, within = module._gamma_pair(panel, module.PRIMARY_WINDOW)
    assert np.isfinite(pooled)
    assert np.isnan(within)


def test_specification_registry_explains_why_the_two_fits_differ(load_src_module):
    """The reference cohort and the cross-league diagnostic give different
    numbers for the same league. A reader meeting both without being told will
    conclude one is wrong, so the register says which is which and why."""
    module = load_src_module("37_denominator_gradient.py")
    out = module.gradient_specification_registry().set_index("specification_id")

    for specification in (
        "reference_cohort", "cross_league_diagnostic", "two_line_shortcut",
    ):
        assert specification in out.index, specification
        assert len(str(out.loc[specification, "why_it_differs"])) > 30
        assert len(str(out.loc[specification, "used_for"])) > 20

    assert "history term" in str(out.loc["reference_cohort", "why_it_differs"])
    assert "no eligibility" in str(out.loc["cross_league_diagnostic", "model"])
    assert out["clustering"].eq("player").all()
