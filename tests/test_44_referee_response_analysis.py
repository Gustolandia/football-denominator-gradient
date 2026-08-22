"""Tests for the referee-response quantities.

Each of these functions exists because a reviewer could not check something
from what the paper reported. That makes the failure mode specific: not a crash
but a plausible number that nobody can trace. So the tests check arithmetic
against hand-computed values wherever a closed form exists, and check that the
functions refuse rather than guess when an input is not what it claims to be.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(name="referee")
def _referee(load_src_module):
    return load_src_module("44_referee_response_analysis.py")


@pytest.fixture(name="decomposition")
def _decomposition():
    return pd.DataFrame(
        {
            "quantity": [
                "gamma_log_minutes_on_exposure",
                "gamma_within_starting_lineup",
                "gamma_within_substitute_list",
                "gamma_within_lineup_unavailable_or_other",
            ],
            "value": [0.30, 0.01, 0.20, 0.35],
        }
    ).set_index("quantity")


@pytest.fixture(name="roles")
def _roles():
    return pd.DataFrame(
        {
            "lineup_role": [
                "all",
                "starting_lineup",
                "substitute_list",
                "lineup_unavailable_or_other",
            ],
            "log_attenuation_fixed90_minus_recorded": [0.15, 0.01, 0.10, 0.175],
        }
    )


def test_percent_understatement_is_the_exponential_of_the_attenuation(referee):
    # An attenuation of 0.151 multiplies the ratio by exp(-0.151) = 0.8598,
    # so 14.02% of the association has been divided away.
    assert referee.percent_understatement(0.151) == pytest.approx(14.02, abs=0.01)
    assert referee.percent_understatement(0.0) == pytest.approx(0.0)
    # Small attenuations are close to the naive percentage, which is why the
    # approximation is tempting and why it is worth not making.
    assert referee.percent_understatement(0.01) == pytest.approx(0.995, abs=0.01)


def test_calibration_pairs_every_stratum_with_its_observed_attenuation(
    referee, decomposition, roles
):
    out = referee.identity_calibration(decomposition, roles)
    assert list(out["stratum"]) == [
        "all",
        "starting_lineup",
        "substitute_list",
        "lineup_unavailable_or_other",
    ]
    # The identity predicts gamma; pooled it over-predicts twofold by
    # construction here (0.30 predicted against 0.15 observed).
    pooled = out[out["stratum"].eq("all")].iloc[0]
    assert pooled["over_prediction_ratio"] == pytest.approx(2.0)
    # Within starters the two agree, which is the pattern the paper reports.
    starters = out[out["stratum"].eq("starting_lineup")].iloc[0]
    assert starters["over_prediction_ratio"] == pytest.approx(1.0)


def test_calibration_skips_a_stratum_the_tables_do_not_carry(referee, decomposition, roles):
    out = referee.identity_calibration(
        decomposition.drop(index="gamma_within_substitute_list"),
        roles[~roles["lineup_role"].eq("lineup_unavailable_or_other")],
    )
    assert set(out["stratum"]) == {"all", "starting_lineup"}


def test_calibration_refuses_when_nothing_can_be_paired(referee, roles):
    empty = pd.DataFrame({"quantity": ["something_else"], "value": [1.0]}).set_index(
        "quantity"
    )
    with pytest.raises(ValueError, match="no stratum could be calibrated"):
        referee.identity_calibration(empty, roles)


def test_a_zero_attenuation_yields_no_ratio_rather_than_an_infinity(referee, decomposition):
    roles = pd.DataFrame(
        {
            "lineup_role": ["all"],
            "log_attenuation_fixed90_minus_recorded": [0.0],
        }
    )
    out = referee.identity_calibration(decomposition, roles)
    assert np.isnan(out["over_prediction_ratio"].iloc[0])


def test_missing_columns_are_named_not_guessed(referee, roles):
    with pytest.raises(KeyError, match="decomposition table"):
        referee.identity_calibration(pd.DataFrame({"quantity": []}).set_index("quantity"), roles)


def test_calibration_factor_reads_the_requested_stratum(referee, decomposition, roles):
    calibration = referee.identity_calibration(decomposition, roles)
    assert referee.calibration_factor(calibration) == pytest.approx(2.0)
    assert referee.calibration_factor(calibration, "starting_lineup") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no calibration row"):
        referee.calibration_factor(calibration, "no_such_stratum")


@pytest.fixture(name="curve")
def _curve():
    """A sweep whose ratio rises with the gradient, as the real one does.

    Ratio 1.0 at the small end and 2.0 at the large end, so an interpolation at
    the midpoint has an unambiguous right answer.
    """
    gamma = [0.010, 0.050, 0.100, 0.200, 0.300]
    ratio = [1.00, 1.20, 1.40, 1.70, 2.00]
    return pd.DataFrame(
        {
            "minute_floor": [80.0, 60.0, 40.0, 20.0, 0.0],
            "gamma": gamma,
            "over_prediction_ratio": ratio,
            "observed_attenuation": [g / r for g, r in zip(gamma, ratio)],
        }
    )


def test_ratio_is_read_at_the_gradient_not_pooled(referee, curve):
    """The whole point of the sweep: the ratio depends on where you stand."""
    ratio, measured = referee.ratio_at_gamma(curve, 0.05)
    assert ratio == pytest.approx(1.20)
    assert measured

    # Interpolation between nodes.
    midpoint, _ = referee.ratio_at_gamma(curve, 0.075)
    assert 1.20 < midpoint < 1.40

    # Outside the swept range the ratio is a guess and says so.
    _, below = referee.ratio_at_gamma(curve, 0.001)
    _, above = referee.ratio_at_gamma(curve, 0.90)
    assert not below and not above


def test_threshold_translation_uses_the_local_ratio(referee, curve):
    out = referee.threshold_translation(curve, grid=(0.05, 0.30, 0.90))
    threshold = out[out["gamma"].eq(0.05)].iloc[0]

    # Naive: 0.05 read straight off the identity is a 4.88% understatement.
    assert threshold["naive_percent_understatement"] == pytest.approx(4.88, abs=0.01)
    # Calibrated at the local ratio of 1.20, not the pooled 2.0: 0.05/1.2.
    assert threshold["over_prediction_ratio"] == pytest.approx(1.20)
    assert threshold["calibrated_attenuation"] == pytest.approx(0.0417, abs=0.0005)
    assert threshold["calibrated_percent_understatement"] == pytest.approx(4.08, abs=0.02)
    assert bool(threshold["is_reporting_threshold"])
    assert bool(threshold["ratio_is_measured"])

    # A row beyond the swept range is flagged rather than quietly reported.
    assert not bool(out[out["gamma"].eq(0.90)].iloc[0]["ratio_is_measured"])
    assert not bool(out[out["gamma"].eq(0.30)].iloc[0]["is_reporting_threshold"])


def test_threshold_translation_refuses_a_curve_it_cannot_read(referee, curve):
    with pytest.raises(KeyError, match="calibration curve"):
        referee.threshold_translation(curve.drop(columns=["over_prediction_ratio"]))


def test_excess_association_lost_is_not_the_same_as_the_ratio_change(referee):
    """A rate ratio of 1.27 falling to 1.09 keeps a third of its excess, so the
    division cost two thirds of it --- not the 14% by which the ratio itself
    shrank. A draft used one word for both and reported the wrong one."""
    assert referee.excess_association_lost(1.267408, 1.088407) == pytest.approx(66.9, abs=0.1)
    assert referee.excess_association_lost(2.0, 1.5) == pytest.approx(50.0)
    assert referee.excess_association_lost(1.5, 1.5) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="must exceed 1"):
        referee.excess_association_lost(1.0, 1.0)


def test_curve_baseline_must_reproduce_both_published_quantities(referee, curve):
    baseline = curve[curve["minute_floor"].eq(0.0)].iloc[0]
    gamma = float(baseline["gamma"])
    attenuation = float(baseline["observed_attenuation"])

    assert referee.verify_curve_baseline(curve, gamma, attenuation) is None

    with pytest.raises(ValueError, match="re-derived gamma"):
        referee.verify_curve_baseline(curve, gamma + 0.01, attenuation)
    with pytest.raises(ValueError, match="re-derived attenuation"):
        referee.verify_curve_baseline(curve, gamma, attenuation + 0.01)
    with pytest.raises(ValueError, match="no unrestricted row"):
        referee.verify_curve_baseline(
            curve[~curve["minute_floor"].eq(0.0)], gamma, attenuation
        )
    with pytest.raises(KeyError, match="calibration curve"):
        referee.verify_curve_baseline(curve.drop(columns=["gamma"]), gamma, attenuation)


@pytest.fixture(name="sweep_frame")
def _sweep_frame(referee):
    """A cohort where longer appearances carry less minute variation.

    Recorded minutes rise with exposure and saturate at 90, so restricting to
    a higher floor leaves less room for the offset to move and the gradient
    falls. That is the mechanism the real sweep exploits.
    """
    rng = np.random.default_rng(20260822)
    size = 1500
    exposure = rng.uniform(0.0, 270.0, size)
    minutes = np.clip(25.0 + 0.24 * exposure + rng.normal(0.0, 8.0, size), 5.0, 90.0)
    events = rng.binomial(1, 0.05, size)
    return pd.DataFrame(
        {
            referee.MINUTES_COL: minutes,
            referee.SAME_DAY_COL: events,
            referee.PLAYER_ID_COL: rng.integers(0, 120, size),
            "prior_minutes_7d": exposure,
            referee.HISTORY_MODEL_COL: rng.normal(0.0, 1.0, size),
            "phase": rng.normal(0.0, 1.0, size),
        }
    )


def test_denominator_pair_fits_both_offsets_on_identical_rows(referee, sweep_frame):
    out = referee.denominator_pair(
        sweep_frame, "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL
    )
    assert out["n_rows"] == len(sweep_frame)
    assert out["gamma"] > 0
    assert out["gamma_ci_low"] < out["gamma"] < out["gamma_ci_high"]
    assert np.isfinite(out["observed_attenuation"])
    assert out["over_prediction_ratio"] == pytest.approx(
        out["gamma"] / out["observed_attenuation"]
    )


def test_denominator_pair_names_the_column_it_lacks(referee, sweep_frame):
    with pytest.raises(KeyError, match="denominator pair frame"):
        referee.denominator_pair(
            sweep_frame.drop(columns=["phase"]), "prior_minutes_7d", ["phase"],
            referee.HISTORY_MODEL_COL,
        )


def test_the_sweep_lowers_the_gradient_as_the_floor_rises(referee, sweep_frame):
    """Raising the floor removes the short appearances, which is where the
    minute variation lives, so the gradient must fall."""
    out = referee.calibration_curve(
        sweep_frame, "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL,
        floors=(0.0, 40.0, 70.0), min_events=10,
    )
    assert len(out) >= 2
    by_floor = out.sort_values("minute_floor")
    assert by_floor["gamma"].iloc[0] > by_floor["gamma"].iloc[-1]
    # Sorted by gamma on the way out, so a reader can interpolate.
    assert out["gamma"].is_monotonic_increasing


def test_the_sweep_drops_a_stratum_whose_gradient_is_indistinguishable_from_zero(
    referee, sweep_frame
):
    """At the highest floors every appearance is a full match and the gradient
    has nothing to vary over. Its ratio is noise with a division sign, and the
    first version of this sweep reported one of minus one."""
    flat = sweep_frame.copy()
    flat[referee.MINUTES_COL] = 90.0
    with pytest.raises(ValueError, match="at least two estimable strata"):
        referee.calibration_curve(
            flat, "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL,
            floors=(0.0, 40.0), min_events=10,
        )


def test_the_sweep_skips_a_stratum_with_too_few_events(referee, sweep_frame):
    out = referee.calibration_curve(
        sweep_frame, "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL,
        floors=(0.0, 40.0, 89.5), min_events=10,
    )
    assert 89.5 not in set(out["minute_floor"])


def test_the_sweep_needs_a_frame_it_can_read(referee, sweep_frame):
    with pytest.raises(KeyError, match="calibration curve frame"):
        referee.calibration_curve(
            sweep_frame.drop(columns=[referee.MINUTES_COL]),
            "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL,
        )


@pytest.fixture(name="cohort")
def _cohort(referee):
    """A cohort where starters play 90 minutes and substitutes play 18.

    Event rates are set so that starters and substitutes carry the same rate
    per appearance, which means the per-minute rates must differ fivefold. That
    is the arithmetic the ascertainment table exists to expose.
    """
    rows = []
    for index in range(100):
        rows.append(
            {
                referee.ROLE_COL: referee.STARTER_ROLE,
                referee.SAME_DAY_COL: 1 if index < 10 else 0,
                referee.MINUTES_COL: 90.0,
            }
        )
        rows.append(
            {
                referee.ROLE_COL: referee.SUBSTITUTE_ROLE,
                referee.SAME_DAY_COL: 1 if index < 10 else 0,
                referee.MINUTES_COL: 18.0,
            }
        )
    return pd.DataFrame(rows)


def test_ascertainment_separates_per_appearance_from_per_minute(referee, cohort):
    out = referee.ascertainment_by_role(cohort).set_index("lineup_role")
    assert out.loc[referee.STARTER_ROLE, "events_per_1000_appearances"] == pytest.approx(100.0)
    assert out.loc[referee.SUBSTITUTE_ROLE, "events_per_1000_appearances"] == pytest.approx(100.0)
    # Equal per appearance, fivefold apart per minute: 90 against 18 minutes.
    assert out.loc[referee.STARTER_ROLE, "starter_over_substitute_per_appearance"] == pytest.approx(1.0)
    assert out.loc[referee.STARTER_ROLE, "starter_over_substitute_per_minute"] == pytest.approx(0.2)
    assert out.loc[referee.STARTER_ROLE, "mean_recorded_minutes"] == pytest.approx(90.0)


def test_ascertainment_intervals_bracket_the_point_estimate(referee, cohort):
    out = referee.ascertainment_by_role(cohort)
    assert (out["events_per_1000_appearances_ci_low"] < out["events_per_1000_appearances"]).all()
    assert (out["events_per_1000_appearances_ci_high"] > out["events_per_1000_appearances"]).all()


def test_ascertainment_skips_a_role_with_no_appearances(referee, cohort):
    out = referee.ascertainment_by_role(cohort)
    assert referee.UNKNOWN_ROLE not in set(out["lineup_role"])
    # And with only one role present, no cross-role ratio is claimed.
    single = cohort[cohort[referee.ROLE_COL].eq(referee.STARTER_ROLE)]
    assert "starter_over_substitute_per_minute" not in referee.ascertainment_by_role(single)


def test_ascertainment_reports_no_per_minute_rate_without_minutes(referee, cohort):
    zeroed = cohort.copy()
    zeroed.loc[zeroed[referee.ROLE_COL].eq(referee.SUBSTITUTE_ROLE), referee.MINUTES_COL] = 0.0
    out = referee.ascertainment_by_role(zeroed).set_index("lineup_role")
    assert np.isnan(out.loc[referee.SUBSTITUTE_ROLE, "events_per_1000_minutes"])


def test_ascertainment_refuses_a_frame_with_no_known_role(referee, cohort):
    stripped = cohort.copy()
    stripped[referee.ROLE_COL] = "something_else"
    with pytest.raises(ValueError, match="no squad role carried any appearance"):
        referee.ascertainment_by_role(stripped)


@pytest.fixture(name="gradient_frame")
def _gradient_frame(referee):
    rng = np.random.default_rng(20260822)
    size = 600
    exposure = rng.uniform(0.0, 270.0, size)
    minutes = np.clip(20.0 + 0.2 * exposure + rng.normal(0.0, 5.0, size), 1.0, 90.0)
    return pd.DataFrame(
        {
            referee.MINUTES_COL: minutes,
            "prior_minutes_7d": exposure,
            "phase": rng.normal(0.0, 1.0, size),
            "player": rng.integers(0, 60, size),
            "club_season": rng.integers(0, 12, size),
        }
    )


def test_clustering_changes_the_interval_and_never_the_estimate(referee, gradient_frame):
    out = referee.clustering_sensitivity(
        gradient_frame,
        {"player (published)": "player", "club-season": "club_season"},
        "prior_minutes_7d",
        ["phase"],
    )
    assert len(out) == 2
    # A covariance choice cannot move a point estimate.
    assert out["gamma"].nunique() == 1
    assert (out["ci_width"] > 0).all()
    assert out["width_ratio_to_narrowest"].min() == pytest.approx(1.0)
    assert out["n_groups"].tolist() == [
        gradient_frame["player"].nunique(),
        gradient_frame["club_season"].nunique(),
    ]


def test_clustering_records_whether_the_verdict_survives(referee, gradient_frame):
    out = referee.clustering_sensitivity(
        gradient_frame,
        {"player (published)": "player"},
        "prior_minutes_7d",
        ["phase"],
    )
    expected = bool(out["ci_low"].iloc[0] > referee.NEGLIGIBLE_GAMMA)
    assert bool(out["all_bounds_exceed_threshold"].iloc[0]) is expected


def test_clustering_requires_the_grouping_column_to_exist(referee, gradient_frame):
    with pytest.raises(KeyError, match="clustering frame"):
        referee.clustering_sensitivity(
            gradient_frame, {"ghost": "not_a_column"}, "prior_minutes_7d", ["phase"]
        )


def test_precision_profile_orders_the_smallest_panel_first(referee):
    leagues = pd.DataFrame(
        {
            "league": ["Big", "Small"],
            "n_appearances": [100000, 2500],
            "gamma_pooled": [0.50, 0.40],
            "gamma_pooled_ci_low": [0.48, 0.30],
            "gamma_pooled_ci_high": [0.52, 0.50],
        }
    )
    out = referee.precision_profile(leagues, "gamma_pooled", "men")
    assert list(out["league"]) == ["Small", "Big"]
    assert out["population"].eq("men").all()
    assert out.loc[0, "ci_half_width"] == pytest.approx(0.10)
    assert out.loc[1, "ci_half_width"] == pytest.approx(0.02)
    # Relative precision is what tells a reader whether a small panel can carry
    # a verdict: a quarter of the estimate against four percent of it.
    assert out.loc[0, "relative_half_width"] == pytest.approx(0.25)
    assert out.loc[1, "relative_half_width"] == pytest.approx(0.04)


def test_precision_profile_names_the_table_it_could_not_read(referee):
    with pytest.raises(KeyError, match="women"):
        referee.precision_profile(pd.DataFrame({"league": ["x"]}), "gamma_pooled", "women")


def test_clustering_baseline_must_reproduce_the_published_gradient(referee, gradient_frame):
    """Clustering is a covariance choice, so the point estimate cannot move.

    The first run of this analysis dropped the history term the published
    gradient conditions on and returned 0.321 where the paper reports 0.303. It
    looked like a clustering result and was a different model. The baseline row
    is now checked against the published value, and a mismatch stops the run.
    """
    clustering = referee.clustering_sensitivity(
        gradient_frame,
        {"player (published)": "player", "club-season": "club_season"},
        "prior_minutes_7d",
        ["phase"],
    )
    published = float(clustering["gamma"].iloc[0])

    assert referee.verify_published_clustering(clustering, published) is None

    with pytest.raises(ValueError, match="the specification\nchanged|specification"):
        referee.verify_published_clustering(clustering, published + 0.01)

    with pytest.raises(ValueError, match="no clustering row labelled"):
        referee.verify_published_clustering(clustering, published, label="ghost")

    with pytest.raises(KeyError, match="clustering sensitivity"):
        referee.verify_published_clustering(
            clustering.drop(columns=["gamma"]), published
        )


def _design_for(referee, frame):
    return referee._sweep_design(
        frame, "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL
    )


def test_sweep_design_precomputes_what_a_replicate_needs(referee, sweep_frame):
    design = _design_for(referee, sweep_frame)
    assert design["exog"].shape == (len(sweep_frame), 4)
    assert len(design["minutes"]) == len(sweep_frame)
    total_rows = sum(len(rows) for rows in design["rows_by_player"])
    assert total_rows == len(sweep_frame)
    with pytest.raises(KeyError, match="sweep design frame"):
        referee._sweep_design(
            sweep_frame.drop(columns=["phase"]), "prior_minutes_7d", ["phase"],
            referee.HISTORY_MODEL_COL,
        )


def test_sweep_points_returns_floor_gamma_attenuation_ratio(referee, sweep_frame):
    design = _design_for(referee, sweep_frame)
    rows = np.arange(len(sweep_frame))
    points = referee._sweep_points(design, rows, (0.0, 40.0), min_events=10)
    assert points, "the unrestricted floor must be estimable"
    for floor, gamma, attenuation, ratio in points:
        assert floor in (0.0, 40.0)
        assert gamma > 0.0
        assert attenuation > 0.0
        assert ratio == pytest.approx(gamma / attenuation)


def test_sweep_points_skips_a_floor_with_too_few_events(referee, sweep_frame):
    design = _design_for(referee, sweep_frame)
    rows = np.arange(len(sweep_frame))
    points = referee._sweep_points(design, rows, (0.0, 89.9), min_events=10)
    floors = [floor for floor, *_ in points]
    assert 89.9 not in floors


def test_sweep_points_skips_a_floor_whose_gradient_is_not_positive(referee, sweep_frame):
    """Minutes falling with exposure give a strictly negative gradient, which
    is a floor the sweep must drop rather than divide with. (Exactly constant
    minutes would sit on the boundary, where floating point decides the sign,
    so the test uses a sign that cannot be argued with.)"""
    declining = sweep_frame.copy()
    declining[referee.MINUTES_COL] = 90.0 - 0.2 * declining["prior_minutes_7d"]
    design = _design_for(referee, declining)
    points = referee._sweep_points(
        design, np.arange(len(declining)), (0.0,), min_events=10
    )
    assert points == []


def test_sweep_points_skips_a_negative_attenuation(referee):
    """A cohort whose log minutes rise with exposure while its arithmetic
    minutes fall gives a positive gradient beside a per-minute rate that grows
    faster than the per-row rate, so the attenuation is negative and the ratio
    would be meaningless. The guard drops it. The construction uses skew: at
    low exposure half the appearances are one minute and half eighty-one, so
    the log mean is low while the arithmetic mean is high."""
    rows = []
    for index in range(100):
        rows.append({"m": 1.0, "e": 0.0, "y": 1 if index < 5 else 0})
    for index in range(100):
        rows.append({"m": 81.0, "e": 0.0, "y": 1 if index < 5 else 0})
    for index in range(200):
        rows.append({"m": 15.0, "e": 270.0, "y": 1 if index < 12 else 0})
    frame = pd.DataFrame(
        {
            "all_minutes_played": [r["m"] for r in rows],
            "prior_minutes_7d": [r["e"] for r in rows],
            "injury_event_matchproxy_same_day": [r["y"] for r in rows],
            "tm_player_id": np.arange(len(rows)),
            "history_log_iqr": 0.0,
            "phase": 0.0,
        }
    )
    # The gradient is positive (long appearances sit at high exposure overall)
    # but the events sit on the short high-exposure rows, which is the
    # configuration that flips the attenuation's sign.
    design = referee._sweep_design(
        frame, "prior_minutes_7d", [], referee.HISTORY_MODEL_COL
    )
    points = referee._sweep_points(
        design, np.arange(len(frame)), (0.0,), min_events=5
    )
    assert points == []


def test_bootstrap_reports_intervals_per_floor_and_at_the_threshold(referee, sweep_frame):
    curve = referee.calibration_curve(
        sweep_frame, "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL,
        floors=(0.0, 40.0, 70.0), min_events=10,
    )
    # A threshold chosen inside the swept range, so replicates can bracket it.
    threshold = float(curve["gamma"].median())

    intervals, summary = referee.bootstrap_calibration_sweep(
        sweep_frame, "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL,
        floors=(0.0, 40.0, 70.0, 89.9), n_boot=25, seed=7, min_events=10,
        threshold=threshold,
    )
    # The impossible floor never produces a ratio, so it never reaches the table.
    assert 89.9 not in set(intervals["minute_floor"])
    assert (intervals["ratio_ci_low"] <= intervals["ratio_ci_high"]).all()
    assert (intervals["attenuation_ci_low"] <= intervals["attenuation_ci_high"]).all()
    assert (intervals["n_boot_valid"] <= 25).all()

    assert summary["threshold"] == pytest.approx(threshold)
    assert summary["cost_percent_ci_low"] <= summary["cost_percent_ci_high"]
    assert summary["ratio_at_threshold_ci_low"] <= summary["ratio_at_threshold_ci_high"]
    assert 0 < summary["n_boot_valid"] <= 25


def test_bootstrap_refuses_a_threshold_the_sweep_cannot_bracket(referee, sweep_frame):
    with pytest.raises(ValueError, match="never bracket"):
        referee.bootstrap_calibration_sweep(
            sweep_frame, "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL,
            floors=(0.0, 40.0), n_boot=5, seed=7, min_events=10, threshold=5.0,
        )


def test_bootstrap_needs_two_floors_to_interpolate(referee, sweep_frame):
    """One floor gives one point per replicate, and one point is not a curve.
    The threshold interpolation must refuse rather than extrapolate."""
    with pytest.raises(ValueError, match="never bracket"):
        referee.bootstrap_calibration_sweep(
            sweep_frame, "prior_minutes_7d", ["phase"], referee.HISTORY_MODEL_COL,
            floors=(0.0,), n_boot=5, seed=7, min_events=10, threshold=0.1,
        )
