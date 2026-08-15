"""Tests for the second JSAMS referee analysis layer."""

from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf


def _raw_panel(players: int = 30, appearances: int = 14) -> pd.DataFrame:
    """Return a deterministic player-appearance frame with estimable events."""
    rows = []
    for player in range(players):
        for index in range(appearances):
            same_day = int(index == (player % (appearances - 2)) + 1)
            lag1 = int(index == ((player + 3) % (appearances - 2)) + 1)
            date = pd.Timestamp("2020-08-01") + pd.Timedelta(
                days=2 * index + player % 3
            )
            rows.append(
                {
                    "tm_player_id": player,
                    "date": date,
                    "all_minutes_played": float((30, 70, 100)[(index + player) % 3]),
                    "all_minutes_last_7d": 0.0,
                    "prior_injuries_per_10000min": float((player % 7) / 2 + index / 30),
                    "history_log_iqr": float((player % 7) / 4 + index / 50),
                    "prior_minutes_played": 1000.0 + index * 90.0,
                    "injury_event_matchproxy_same_day": same_day,
                    "injury_event_matchproxy_lag1": lag1,
                    "injury_event_matchproxy": int(same_day or lag1),
                    "week_phase_sin": np.sin((index + player) / 3),
                    "week_phase_cos": np.cos((index + player) / 3),
                    "halfweek_phase_sin": np.sin((index + 2 * player) / 5),
                    "halfweek_phase_cos": np.cos((index + 2 * player) / 5),
                    "recovery_interval_bin": (
                        "0-3 days" if index % 3 == 0 else "6-7 days"
                    ),
                    "season_start": 2020,
                    "matchproxy_injury_desc": "hamstring" if same_day else "",
                    "player_name": f"Player {player}",
                    "club_name": f"Club {player % 4}",
                    "same_day_reported_absence_ge28d": int(same_day and player % 3 == 0),
                    "same_day_muscle_tendon_report": int(same_day and player % 3 == 1),
                }
            )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["tm_player_id", "date"]).reset_index(drop=True)
    rolling = []
    for _, group in frame.groupby("tm_player_id", sort=False):
        minutes = group["all_minutes_played"].to_numpy()
        for index in range(len(group)):
            rolling.append(float(minutes[max(0, index - 1) : index].sum()))
    frame["all_minutes_last_7d"] = rolling
    return frame


def _with_windows(module, players: int = 30, appearances: int = 14) -> pd.DataFrame:
    return module.add_prior_window_metrics(_raw_panel(players, appearances))


def test_require_normal_holm_and_windows(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    with pytest.raises(KeyError, match="missing columns"):
        module._require_columns(pd.DataFrame(), ["x"], "sample")
    assert np.isnan(module._normal_p_value(np.nan, 1.0))
    assert module._normal_p_value(0.0, 0.0) == 1.0
    assert module._normal_p_value(1.0, 0.0) == 0.0
    assert 0.0 < module._normal_p_value(1.0, 1.0) < 1.0
    assert module.holm_adjust(pd.Series([np.nan])).isna().all()
    adjusted = module.holm_adjust(pd.Series([0.04, 0.01, np.nan]))
    assert adjusted.dropna().between(0, 1).all()

    small = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1],
            "date": pd.to_datetime(["2020-01-01", "2020-01-04", "2020-01-08"]),
            "all_minutes_played": [90.0, 45.0, 30.0],
        }
    )
    rolled = module.add_prior_window_metrics(small, windows=(3, 7))
    assert rolled["prior_minutes_3d"].tolist() == [0.0, 90.0, 0.0]
    assert rolled["prior_minutes_7d"].tolist() == [0.0, 90.0, 135.0]
    assert rolled["prior_matches_7d"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="positive"):
        module.add_prior_window_metrics(small, windows=())
    with pytest.raises(ValueError, match="complete"):
        module.add_prior_window_metrics(small.assign(date=pd.NaT))
    with pytest.raises(ValueError, match="unique"):
        module.add_prior_window_metrics(pd.concat([small, small.iloc[[0]]]))


def test_window_validation_specs_formulas_and_terms(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    frame["all_minutes_last_7d"] = frame["prior_minutes_7d"]
    validation = module.validate_reference_window(frame)
    assert bool(validation.loc[0, "parity_passes"])
    specs = module.exposure_specs()
    assert len(specs) == 7
    linear = specs[2]
    recovery = specs[-1]
    assert "history_log_iqr" in module._model_formula(module.SAME_DAY_COL, linear)
    assert "Treatment" in module._model_formula(module.SAME_DAY_COL, recovery)

    result, _, term = module.fit_exposure_model(
        frame, module.SAME_DAY_COL, "per_appearance", linear
    )
    assert module._focal_term(result, linear) == term
    recovery_result, _, recovery_term = module.fit_exposure_model(
        frame, module.SAME_DAY_COL, "per_appearance", recovery
    )
    assert "0-3 days" in recovery_term
    fake = SimpleNamespace(params=pd.Series([1.0], index=["Intercept"]))
    with pytest.raises(ValueError, match="Expected one focal"):
        module._focal_term(fake, linear)


def test_exposure_models_and_multiverse(load_src_module, monkeypatch):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    specification = module.exposure_specs()[2]
    row, _ = module.exposure_model_row(
        frame, module.SAME_DAY_COL, "per_appearance", specification
    )
    assert row["effect_measure"] == "odds_ratio"
    assert row["n_rows"] == len(frame[frame[module.MINUTES_COL] > 0])
    observed, _, _ = module.fit_exposure_model(
        frame, module.SAME_DAY_COL, "observed_minutes", specification
    )
    fixed, _, _ = module.fit_exposure_model(
        frame, module.SAME_DAY_COL, "fixed_90", specification
    )
    assert len(observed.params) == len(fixed.params)
    with pytest.raises(ValueError, match="Unknown denominator"):
        module.fit_exposure_model(frame, module.SAME_DAY_COL, "bad", specification)
    with pytest.raises(ValueError, match="No estimable events"):
        module.fit_exposure_model(
            frame.assign(injury_event_matchproxy_same_day=0),
            module.SAME_DAY_COL,
            "per_appearance",
            specification,
        )

    calls = []

    def fake_row(frame, event_col, denominator, specification):
        calls.append((event_col, denominator, specification["exposure_id"]))
        return (
            {
                "event_col": event_col,
                "denominator": denominator,
                "exposure_id": specification["exposure_id"],
                "p_value": 0.01 + len(calls) / 10000,
            },
            None,
        )

    monkeypatch.setattr(module, "exposure_model_row", fake_row)
    multiverse = module.exposure_multiverse(frame)
    assert len(multiverse) == 63
    assert multiverse["family_size"].eq(63).all()


def test_standardisation_components_and_simultaneous_band(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    specification = module.exposure_specs()[2]
    result, work, _ = module.fit_exposure_model(
        frame, module.SAME_DAY_COL, "per_appearance", specification
    )
    reference = module._standardization_reference(work)
    estimate, gradient = module._marginal_components(result, reference, 90.0)
    assert 0.0 < estimate < 1.0
    assert len(gradient) == len(result.params)
    with pytest.raises(ValueError, match="empty"):
        module._standardization_reference(frame.iloc[0:0])
    with pytest.raises(ValueError, match="at least 100"):
        module._simultaneous_critical_value(
            np.ones((2, 2)), np.eye(2), draws=10
        )
    fallback = module._simultaneous_critical_value(
        np.zeros((2, 2)), np.zeros((2, 2)), draws=100
    )
    active = module._simultaneous_critical_value(
        np.eye(2), np.eye(2), draws=100, seed=1
    )
    assert fallback > 1.9
    assert active > 1.0


def test_curve_and_temporal_analysis(load_src_module, monkeypatch):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module, players=40, appearances=16)
    monkeypatch.setattr(module, "_simultaneous_critical_value", lambda *args, **kwargs: 2.5)
    spline = (
        "bs(prior_minutes_7d, knots=(45.0, 90.0, 135.0), degree=3, "
        "include_intercept=False, lower_bound=0.0, "
        f"upper_bound={float(frame['prior_minutes_7d'].max())})"
    )
    curves, tests = module.additive_curve_analysis(frame, spline)
    assert set(curves["model_id"]) == {"additive_linear", "additive_spline"}
    assert curves.query("model_id == 'additive_spline'")["simultaneous_ci_low"].notna().all()
    assert tests.loc[0, "df"] > 0

    frame.loc[frame.index[:150], "season_start"] = 2018
    frame.loc[frame.index[150:350], "season_start"] = 2021
    frame.loc[frame.index[350:], "season_start"] = 2023
    temporal = module.temporal_stability(frame)
    assert len(temporal) == 3
    assert temporal["heterogeneity_df"].eq(2).all()
    assert temporal["holm_p_value_3_block_family"].between(0, 1).all()


def test_exposure_bands_and_conditional_analysis(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    bands = module._exposure_band(pd.Series([0, 20, 70, 120, 170, 210, np.nan]))
    assert list(bands.astype(object)[:6]) == list(module.EXPOSURE_BANDS)
    frame = _with_windows(module, players=24, appearances=16)
    estimates, populations, support = module.conditional_model_analysis(
        frame, bootstrap_draws=100, seed=3
    )
    assert len(estimates) == 2
    assert estimates["multiplier_bootstrap_ci_low"].gt(0).all()
    assert estimates["holm_p_value_2_model_family"].between(0, 1).all()
    assert set(populations["population"]) == {
        "discordant_included",
        "concordant_excluded",
    }
    assert len(support) == 12
    with pytest.raises(ValueError, match="at least 100"):
        module.conditional_model_analysis(frame, bootstrap_draws=10)

    fake_model = SimpleNamespace(
        groups=pd.Series(["one"]),
        score_grp=lambda index, params: np.array([1.0]),
        hessian=lambda params: np.array([[-1.0]]),
    )
    fake_result = SimpleNamespace(
        model=fake_model,
        params=pd.Series([0.0], index=["x"]),
    )
    covariance, scores = module._conditional_cluster_covariance(
        fake_result,
        pd.DataFrame({"stratum": ["one"], "tm_player_id": [1]}),
        "stratum",
    )
    assert covariance.shape == (1, 1)
    assert scores.shape == (1, 1)
    with pytest.raises(ValueError, match="must map to one player"):
        module._conditional_cluster_covariance(
            fake_result,
            pd.DataFrame(
                {"stratum": ["one", "one"], "tm_player_id": [1, 2]}
            ),
            "stratum",
        )


def _selection_frames(module):
    rng = np.random.default_rng(22)
    risk_rows = []
    daily_rows = []
    appearance_rows = []
    start = pd.Timestamp("2020-09-01")
    for player in range(30):
        history = float(player % 5) / 2.0
        for game in range(18):
            date = start + pd.Timedelta(days=7 * game + player % 2)
            burden = float((game % 4) * 60)
            probability = 0.55 + 0.15 * (game % 3 == 0) - 0.05 * history
            played = int(rng.random() < probability)
            risk_rows.append(
                {
                    "tm_player_id": player,
                    "date": date,
                    "played_any_minutes": played,
                    "plausibly_available": True,
                    "all_minutes_last_7d": burden,
                    "season": 2020,
                    "player_club_id": player % 4,
                    "membership_evidence": "observed span",
                }
            )
            daily_rows.append(
                {
                    "tm_player_id": player,
                    "date": date,
                    "prior_minutes_played": 1000.0 + game * 90,
                    "prior_injuries_per_10000min": history,
                    "days_since_last_match": float(3 + game % 12),
                    "week_phase_sin": np.sin(game),
                    "week_phase_cos": np.cos(game),
                    "halfweek_phase_sin": np.sin(game / 2),
                    "halfweek_phase_cos": np.cos(game / 2),
                }
            )
            if played:
                appearance_rows.append(
                    {
                        "tm_player_id": player,
                        "date": date,
                        "injury_event_matchproxy_same_day": int(
                            (player + game) % 29 == 0
                        ),
                        "prior_minutes_7d": burden,
                        "prior_injuries_per_10000min": history,
                        "age_years": 24.0 + (player % 9),
                        "competition_context": (
                            "Premier League" if game % 5 else "domestic cup"
                        ),
                    }
                )
    return pd.DataFrame(risk_rows), pd.DataFrame(daily_rows), pd.DataFrame(appearance_rows)


def test_weighted_smd_and_selection_sensitivity(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    selected = pd.Series([1, 1, 0, 0])
    assert module._weighted_smd(values, selected, pd.Series([1.0] * 4)) > 0
    assert np.isnan(module._weighted_smd(values, selected, pd.Series([0.0] * 4)))
    assert module._weighted_smd(
        pd.Series([1.0] * 4), selected, pd.Series([1.0] * 4)
    ) == 0.0
    risk, daily, appearances = _selection_frames(module)
    # Appearances outside every membership interval must land in the excluded
    # population, so the comparison has both arms to describe.
    outside = appearances.head(4).copy()
    outside["date"] = pd.Timestamp("2026-01-15")
    appearances = pd.concat([appearances, outside], ignore_index=True)
    estimates, diagnostics, population = module.appearance_selection_sensitivity(
        risk, daily, appearances
    )
    assert set(estimates["model_id"]) == {
        "unweighted",
        "inverse_selection_weighted",
    }
    assert "all_numeric_gates_pass" in diagnostics

    # The leakage gates must be present and must confirm that no outcome column
    # entered the propensity model.
    gates = diagnostics.set_index("metric")["value"]
    assert gates["outcome_columns_in_propensity_model"] == 0.0
    assert "premier_league_same_day_event_rows_retained" in gates.index
    assert set(population["population"]) == {
        "included in bounded selection set",
        "excluded from bounded selection set",
    }
    assert population["premier_league_share"].notna().all()

    # Without competition metadata the scope falls back to every appearance,
    # which must still produce a complete population comparison.
    _, plain_diagnostics, plain_population = module.appearance_selection_sensitivity(
        risk, daily, appearances.drop(columns=["competition_context"])
    )
    assert (
        plain_diagnostics.set_index("metric")
        .loc["non_premier_league_appearances_out_of_scope", "value"]
        == 0.0
    )
    assert plain_population["premier_league_share"].isna().all()


def test_exposure_multiverse_summary_and_correlations(load_src_module):
    """The multiverse must be summarised as a distribution, not a rejection count."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module, players=26, appearances=16)
    multiverse = module.exposure_multiverse(frame)
    summary = module.exposure_multiverse_summary(multiverse)
    overall = summary[summary["stratum_id"].eq("all_63_models")].iloc[0]
    assert int(overall["n_models"]) == len(multiverse)
    assert overall["estimate_min"] <= overall["estimate_median"] <= overall["estimate_max"]
    assert overall["estimate_q1"] <= overall["estimate_q3"]
    assert 0.0 <= overall["share_above_one"] <= 1.0
    assert {"denominator_per_appearance", "denominator_observed_minutes"} <= set(
        summary["stratum_id"]
    )

    correlations = module.exposure_metric_correlations(frame)
    assert len(correlations) == 21
    assert correlations["involves_reference_window"].any()
    windows = correlations[
        correlations["exposure_a"].eq("prior_minutes_7d")
        & correlations["exposure_b"].eq("prior_minutes_10d")
    ].iloc[0]
    assert windows["pearson_r"] > 0.5
    assert correlations["reference_window"].eq("prior_minutes_7d").all()


def test_exposure_correlations_handle_degenerate_columns(load_src_module):
    """A constant metric cannot yield a correlation and must return missing."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module, players=24, appearances=12)
    frame["prior_minutes_3d"] = 0.0
    frame["recovery_interval_bin"] = "8-14 days"
    correlations = module.exposure_metric_correlations(frame)
    degenerate = correlations[correlations["exposure_a"].eq("prior_minutes_3d")]
    assert degenerate["pearson_r"].isna().all()
    recovery = correlations[correlations["exposure_b"].eq("recovery_interval")]
    assert recovery["n_rows_compared"].eq(0).all()


def test_confounding_sensitivity_covers_covariates_and_clustering(load_src_module):
    """Every declared confounding and clustering variant must be estimable."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module, players=40, appearances=18)
    rng = np.random.default_rng(11)
    frame["age_years"] = 22.0 + rng.integers(0, 12, len(frame))
    frame["position_group"] = np.where(frame["tm_player_id"] % 2 == 0, "Defender", "Attack")
    frame["club_season"] = "2020_" + (frame["tm_player_id"] % 4).astype(str)
    frame["competition_context"] = np.where(
        frame["tm_player_id"] % 5 == 0, "domestic cup", "Premier League"
    )
    frame["season"] = "2020"
    frame["match_cluster_id"] = frame["date"].astype(str)
    result = module.confounding_sensitivity(frame)
    assert list(result["model_id"]) == [item[0] for item in module.CONFOUNDING_SPECIFICATIONS]
    assert (result["ci_low"] <= result["estimate"]).all()
    assert (result["estimate"] <= result["ci_high"]).all()
    assert set(result["covariance"]) == {"player", "player_match", "club_season"}
    premier = result[result["model_id"].eq("premier_league_only")].iloc[0]
    assert premier["n_rows"] < result[result["model_id"].eq("reference")].iloc[0]["n_rows"]

    with pytest.raises(ValueError, match="No estimable rows"):
        empty = frame.copy()
        empty["competition_context"] = "domestic cup"
        module.confounding_sensitivity(empty)


def test_two_way_cluster_requires_two_clusters(load_src_module):
    """Two-way clustering is undefined with a single cluster on either axis."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module, players=20, appearances=10)
    frame["match_cluster_id"] = "single"
    formula = (
        "injury_event_matchproxy_same_day ~ I(prior_minutes_7d / 90.0) + history_log_iqr"
    )
    with pytest.raises(ValueError, match="at least two clusters"):
        module._two_way_clustered_result(frame, formula, "match_cluster_id")


def test_absolute_risk_contrast_and_support(load_src_module):
    """The absolute contrast must name its target and expose observed support."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module, players=34, appearances=16)
    contrast, support = module.absolute_risk_contrast(frame)
    # 90 minutes is reported beside 180 because most of the cohort sits at or
    # below it; both contrasts must be present and internally consistent.
    assert list(contrast["quantity"]) == [
        "standardised_probability_at_0_minutes",
        "standardised_probability_at_180_minutes",
        "standardised_probability_at_90_minutes",
        "absolute_difference_90_minus_0",
        "absolute_difference_180_minus_0",
    ]
    indexed = contrast.set_index("quantity")
    zero = indexed.loc["standardised_probability_at_0_minutes"]
    for anchor_name, difference_name in (
        ("standardised_probability_at_90_minutes", "absolute_difference_90_minus_0"),
        ("standardised_probability_at_180_minutes", "absolute_difference_180_minus_0"),
    ):
        anchor = indexed.loc[anchor_name]
        difference = indexed.loc[difference_name]
        assert (
            difference["ci_low_per_1000_appearances"]
            <= difference["estimate_per_1000_appearances"]
            <= difference["ci_high_per_1000_appearances"]
        )
        assert np.isclose(
            difference["estimate_per_1000_appearances"],
            anchor["estimate_per_1000_appearances"] - zero["estimate_per_1000_appearances"],
        )
    assert "median" in contrast["target_population"].iloc[0]
    assert set(support["exposure_band"]) <= set(module.EXPOSURE_BANDS)
    assert np.isclose(support["share_of_appearances"].sum(), 1.0)
    assert support["n_appearances"].sum() == int(contrast["n_rows"].iloc[0])


def test_denominator_contrast_metadata(load_src_module):
    """Denominator metadata must name the estimator and refuse a causal reading."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    same_day = pd.DataFrame(
        [
            {
                "event_minus_non_event_minutes": -21.05,
                "bootstrap_unit": "player",
                "bootstrap_replicates": 1000,
                "interval_method": "player_cluster_percentile_bootstrap_95",
            }
        ]
    )
    lineup = pd.DataFrame(
        [
            {"comparison": "starting_lineup", "event_minus_non_event_minutes": -31.19},
            {"comparison": "lineup_standardized", "event_minus_non_event_minutes": -24.66},
        ]
    )
    completeness = pd.DataFrame(
        [
            {"dimension": "overall", "level": "all", "lineup_known_percent": 76.0},
            {"dimension": "season", "level": "2017", "lineup_known_percent": 100.0},
            {"dimension": "season", "level": "2023", "lineup_known_percent": 22.4},
        ]
    )
    metadata = module.denominator_contrast_metadata(same_day, lineup, completeness)
    values = metadata.set_index("attribute")["value"]
    assert values["complete_lineup_seasons"] == "2017"
    assert values["n_complete_lineup_seasons"] == "1"
    assert values["standardised_difference_minutes"] == "-24.66"
    assert "not licensed" in values["causal_reading"]

    without_standardised = module.denominator_contrast_metadata(
        same_day, lineup[lineup["comparison"].eq("starting_lineup")], completeness
    )
    assert (
        without_standardised.set_index("attribute")
        .loc["standardised_difference_minutes", "value"]
        == "unavailable"
    )
    no_seasons = module.denominator_contrast_metadata(
        same_day, lineup, completeness[completeness["dimension"].eq("overall")]
    )
    assert no_seasons.set_index("attribute").loc["complete_lineup_seasons", "value"] == "none"


def test_outcome_audit_and_claim_hierarchy(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module, players=36, appearances=14)
    identified_queue = module.build_outcome_audit_queue(frame, per_stratum=3)
    assert len(identified_queue) == 9
    assert not any("minutes" in column for column in identified_queue.columns)
    with pytest.raises(ValueError, match="positive"):
        module.build_outcome_audit_queue(frame, per_stratum=0)

    # The queue is built on real identifiers, because the sample is drawn by
    # hashing them, and de-identified on its way to disk. Validation therefore
    # sees surrogates on both sides.
    identity = load_src_module("audit_identity.py")
    keys = iter(f"{index:012X}" for index in range(1000))
    identity_map = identity.build_identity_map(
        [identified_queue], key_factory=lambda: next(keys)
    )
    queue = identity.deidentify_audit_frame(identified_queue, identity_map)
    assert "player_name" not in queue.columns
    assert "tm_player_id" not in queue.columns

    reviewed = queue.copy()
    reviewed[identity.SOURCE_FOUND] = True
    reviewed["independent_source_type"] = "independent_news"
    reviewed["review_note"] = "independent report reviewed"
    reviewed.loc[reviewed.index[:2], "date_attribution_verdict"] = "confirmed"
    reviewed.loc[reviewed.index[2], "date_attribution_verdict"] = "not_confirmed"
    reviewed.loc[reviewed.index[3:], "date_attribution_verdict"] = "unresolved"
    reviewed["description_consistency_verdict"] = "unresolved"
    reviewed, validation = module.validate_outcome_audit(queue, reviewed)
    assert validation["passes_gate"].all()
    summary = module.summarize_outcome_audit(reviewed)
    assert set(summary["status"]) == {"complete", "incomplete"}
    overall = summary.loc[
        summary["audit_dimension"].eq("date_attribution")
        & summary["audit_stratum"].eq("all_strata")
    ].iloc[0]
    assert overall["n_sampled"] == 9
    assert 0.0 <= overall["ci_low"] <= overall["ci_high"] <= 1.0
    assert set(summary["audit_dimension"]) == {
        "date_attribution",
        "description_consistency",
    }

    rows = []
    for exposure in [item["exposure_id"] for item in module.exposure_specs()]:
        rows.append(
            {
                "event_col": module.SAME_DAY_COL,
                "denominator": "per_appearance",
                "exposure_id": exposure,
                "estimate": 1.2,
                "ci_low": 1.0,
                "ci_high": 1.4,
                "reject_holm_0_05": exposure != "recovery_interval",
            }
        )
    temporal = pd.DataFrame([{"heterogeneity_p_value": 0.8}])
    selection = pd.DataFrame(
        [
            {
                "model_id": "inverse_selection_weighted",
                "estimate": 1.2,
                "ci_low": 1.0,
                "ci_high": 1.5,
            }
        ]
    )
    hierarchy = module.revised_claim_hierarchy(
        pd.DataFrame(rows), temporal, selection, summary
    )
    assert hierarchy["visibility_rule_passes"].all()
    assert hierarchy.query("tier > 3")["abstract_visible"].eq(False).all()
    assert "independent_same_day_outcome_audit" in set(hierarchy["claim_id"])

    # Both estimands and the partial-identification bounds must reach the
    # abstract-visible audit claim.
    audit_claim = hierarchy.set_index("claim_id").loc["independent_same_day_outcome_audit"]
    assert "of all sampled records" in audit_claim["evidence"]
    assert "bounding the sampled proportion" in audit_claim["evidence"]

    # Supplying the multiverse distribution and the window correlations adds the
    # exploratory framing the primary claim needs; omitting them must not fail.
    frame = _with_windows(module, players=26, appearances=16)
    multiverse = module.exposure_multiverse(frame)
    enriched = module.revised_claim_hierarchy(
        pd.DataFrame(rows),
        temporal,
        selection,
        summary,
        module.exposure_multiverse_summary(multiverse),
        module.exposure_metric_correlations(frame),
    )
    primary = enriched.set_index("claim_id").loc[
        "cumulative_recent_exposure_same_day_association"
    ]
    assert "median odds ratio" in primary["evidence"]
    assert "correlated sensitivity analyses" in primary["evidence"]
    assert "not four replications" in primary["required_caveat"]

    bare = module.revised_claim_hierarchy(
        pd.DataFrame(rows),
        temporal,
        selection,
        summary,
        pd.DataFrame(),
        pd.DataFrame(),
    )
    bare_primary = bare.set_index("claim_id").loc[
        "cumulative_recent_exposure_same_day_association"
    ]
    assert "median odds ratio" not in bare_primary["evidence"]
    assert "correlated sensitivity analyses" not in bare_primary["evidence"]


def test_summarize_outcome_audit_handles_an_empty_stratum(load_src_module):
    """An unused stratum has no denominator, so every proportion is missing."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    reviewed = pd.DataFrame(
        [
            {
                "audit_stratum": "muscle_tendon_nonsevere",
                "date_attribution_verdict": "confirmed",
                "description_consistency_verdict": "confirmed",
            }
        ]
    )
    reviewed["audit_stratum"] = pd.Categorical(
        reviewed["audit_stratum"],
        categories=["muscle_tendon_nonsevere", "reported_absence_ge28d"],
    )
    summary = module.summarize_outcome_audit(reviewed)
    empty = summary[summary["audit_stratum"].eq("reported_absence_ge28d")]
    assert len(empty) == 2
    assert empty["n_sampled"].eq(0).all()
    assert empty["confirmed_proportion_all_sampled"].isna().all()
    assert empty["partial_identification_low"].isna().all()
    assert empty["partial_identification_high"].isna().all()


def test_outcome_audit_validation_rejects_tampering(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module, players=36, appearances=14)
    identity = load_src_module("audit_identity.py")
    identified_queue = module.build_outcome_audit_queue(frame, per_stratum=2)
    keys = iter(f"{index:012X}" for index in range(1000))
    identity_map = identity.build_identity_map(
        [identified_queue], key_factory=lambda: next(keys)
    )
    queue = identity.deidentify_audit_frame(identified_queue, identity_map)
    reviewed = queue.copy()
    reviewed["date_attribution_verdict"] = "unresolved"
    reviewed["description_consistency_verdict"] = "unresolved"
    reviewed["review_note"] = "review complete"

    pending, validation = module.validate_outcome_audit(
        queue,
        queue,
        require_completed_review=False,
    )
    assert len(pending) == len(queue)
    assert validation["passes_gate"].all()

    duplicate = pd.concat([reviewed, reviewed.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="record keys must be unique"):
        module.validate_outcome_audit(queue, duplicate)

    wrong_ids = reviewed.copy()
    wrong_ids.loc[0, identity.RECORD_KEY] = "different"
    with pytest.raises(ValueError, match="IDs do not match"):
        module.validate_outcome_audit(queue, wrong_ids)

    exposed = reviewed.assign(prior_minutes_7d=90)
    with pytest.raises(ValueError, match="contains exposure fields"):
        module.validate_outcome_audit(queue, exposed)

    changed = reviewed.copy()
    changed.loc[0, identity.SEASON] = "1999-00"
    with pytest.raises(ValueError, match="immutable queue fields"):
        module.validate_outcome_audit(queue, changed)

    invalid = reviewed.copy()
    invalid.loc[0, "date_attribution_verdict"] = "maybe"
    with pytest.raises(ValueError, match="invalid date_attribution_verdict"):
        module.validate_outcome_audit(queue, invalid)

    pending_completed = reviewed.copy()
    pending_completed.loc[0, "date_attribution_verdict"] = "pending"
    with pytest.raises(ValueError, match="still contains pending"):
        module.validate_outcome_audit(queue, pending_completed)

    resolved = reviewed.copy()
    resolved.loc[0, "date_attribution_verdict"] = "confirmed"
    with pytest.raises(ValueError, match="require an independent source"):
        module.validate_outcome_audit(queue, resolved)

    transfermarkt = reviewed.copy()
    transfermarkt["independent_source_type"] = "transfermarkt_profile"
    with pytest.raises(ValueError, match="independent of Transfermarkt"):
        module.validate_outcome_audit(queue, transfermarkt)

    no_note = reviewed.copy()
    no_note.loc[0, "review_note"] = ""
    with pytest.raises(ValueError, match="require review notes"):
        module.validate_outcome_audit(queue, no_note)

    with pytest.raises(KeyError, match="reviewed outcome audit"):
        module.validate_outcome_audit(queue, reviewed.drop(columns="review_note"))


def test_revised_hypothesis_register(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    columns = [
        "hypothesis_id", "model_id", "contrast_id", "description",
        "effect_measure", "test_statistic", "df", "p_value", "event_col",
        "denominator", "exposure_spec", "n_match_rows", "n_players",
        "n_events", "family", "analysis_role", "test_domain", "source_file",
        "model", "model_family", "estimator", "group_col", "estimate",
        "ci_low", "ci_high", "family_size", "stratum_definition",
        "analysis_timing", "confirmatory_status", "n_discordant_strata",
        "estimable", "dated_prospective_analysis_plan_available",
        "p_holm_within_family_recomputed", "p_adjusted_reported",
        "reject_adjusted_0_05",
    ]
    legacy = pd.DataFrame([["H0001", *([np.nan] * (len(columns) - 1))]], columns=columns)
    multiverse = pd.DataFrame(
        [
            {
                "exposure_id": "prior_minutes_7d",
                "effect_label": "per 90 prior minutes in 7 days",
                "event_col": module.SAME_DAY_COL,
                "denominator": denominator,
                "effect_measure": "odds_ratio" if denominator == "per_appearance" else "incidence_rate_ratio",
                "estimate": 1.2,
                "ci_low": 1.0,
                "ci_high": 1.4,
                "p_value": 0.02,
                "n_rows": 100,
                "n_players": 20,
                "n_events": 10,
                "formula": "event ~ exposure",
                "analysis_timing": "post-data reviewer-requested multiverse",
                "holm_p_value_63_model_family": 0.04,
                "reject_holm_0_05": True,
            }
            for denominator in ("per_appearance", "observed_minutes")
        ]
    )
    temporal = pd.DataFrame(
        [
            {
                "temporal_block": block,
                "estimate": 1.2,
                "ci_low": 1.0,
                "ci_high": 1.4,
                "p_value": 0.03,
                "holm_p_value_3_block_family": 0.09,
                "reject_holm_3_block_0_05": False,
                "heterogeneity_test_statistic": 0.2,
                "heterogeneity_df": 2,
                "heterogeneity_p_value": 0.9,
                "n_rows": 100,
                "n_players": 20,
                "n_events": 10,
            }
            for block in ("early", "middle", "late")
        ]
    )
    conditional = pd.DataFrame(
        [
            {
                "stratum_definition": definition,
                "estimate": 1.5,
                "player_cluster_ci_low": 1.1,
                "player_cluster_ci_high": 2.0,
                "player_cluster_p_value": 0.01,
                "holm_p_value_2_model_family": 0.02,
                "reject_holm_2_model_0_05": True,
                "n_rows": 50,
                "n_players": 10,
                "n_events": 8,
                "n_discordant_strata": 10,
            }
            for definition in ("player", "player-season")
        ]
    )
    register = module.build_revised_hypothesis_register(
        legacy, multiverse, temporal, conditional
    )
    assert len(register) == 1 + 2 + 3 + 1 + 2
    assert register["hypothesis_id"].is_unique
    assert set(register.loc[register["family_size"].eq(63), "model_family"]) == {
        "binomial_logit",
        "poisson_log",
    }


def _episodes_for(frame: pd.DataFrame) -> pd.DataFrame:
    """Return public episodes aligned to the same-day events in ``frame``.

    Every third event becomes an illness so the negative-control column is
    populated, and every fourth carries unspecified text so the
    description-specificity flag has both values.
    """
    events = frame[frame["injury_event_matchproxy_same_day"].eq(1)].reset_index(drop=True)
    descriptions = []
    for index in range(len(events)):
        if index % 3 == 0:
            descriptions.append("flu")
        elif index % 4 == 0:
            descriptions.append("unknown")
        else:
            descriptions.append("hamstring strain")
    return pd.DataFrame(
        {
            "tm_player_id": events["tm_player_id"],
            "start_date": events["date"],
            "injury_desc": descriptions,
        }
    )


def _classifier(text: str) -> str:
    lowered = str(text).lower()
    if "flu" in lowered:
        return "illness/other medical"
    if "unknown" in lowered:
        return "unknown"
    return "muscle/tendon"


def test_negative_control_columns_and_placebo_window(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    labelled = module.add_negative_control_outcomes(
        frame, _episodes_for(frame), _classifier
    )
    # Illness and specificity flags only ever fire on a same-day appearance.
    same_day = labelled["injury_event_matchproxy_same_day"].eq(1)
    assert labelled.loc[~same_day, module.NEGATIVE_CONTROL_COL].eq(0).all()
    assert labelled.loc[~same_day, module.SPECIFIC_TEXT_COL].eq(0).all()
    assert labelled[module.NEGATIVE_CONTROL_COL].sum() > 0
    # The specificity flag must separate "unknown" from real descriptions,
    # which is exactly what the operator-precedence bug would have hidden.
    assert 0 < labelled[module.SPECIFIC_TEXT_COL].sum() < same_day.sum()

    placebo = module.add_placebo_exposure_window(frame)
    assert module.PLACEBO_COL in placebo.columns
    assert (placebo[module.PLACEBO_COL] >= 0).all()


def test_placebo_window_analysis_reports_four_specifications(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    # The placebo window looks back 31-37 days, so the panel must span more
    # than five weeks or that column is identically zero and unestimable.
    frame = module.add_placebo_exposure_window(
        _with_windows(module, players=30, appearances=40)
    )
    assert frame[module.PLACEBO_COL].gt(0).any()
    out = module.placebo_window_analysis(frame)
    assert len(out) == 4
    assert list(out["model_id"]) == [
        "recent_7d_alone",
        "placebo_31_37d_alone",
        "both_windows",
        "both_windows",
    ]
    # The mutually adjusted rows must report different focal windows.
    both = out[out["model_id"].eq("both_windows")]
    assert set(both["focal_window"]) == {"prior_minutes_7d", module.PLACEBO_COL}
    assert (out["ci_low"] <= out["estimate"]).all()
    assert (out["estimate"] <= out["ci_high"]).all()
    with pytest.raises(KeyError, match="placebo window frame missing columns"):
        module.placebo_window_analysis(pd.DataFrame())


def test_negative_control_outcomes_flag_sparse_events(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    labelled = module.add_negative_control_outcomes(
        frame, _episodes_for(frame), _classifier
    )
    out = module.negative_control_outcome_analysis(labelled)
    assert len(out) == 4
    # Every row is either estimable with an interval or explicitly not.
    estimable = out[out["estimable"]]
    assert estimable["estimate"].notna().all()
    assert (out.loc[~out["estimable"], "n_events"] < 10).all()
    assert (
        out.loc[~out["estimable"], "note"]
        .eq("fewer than ten events; not estimable")
        .all()
    )

    sparse = labelled.copy()
    sparse[module.NEGATIVE_CONTROL_COL] = 0
    sparse_out = module.negative_control_outcome_analysis(sparse)
    row = sparse_out[sparse_out["event_col"].eq(module.NEGATIVE_CONTROL_COL)].iloc[0]
    assert not bool(row["estimable"])
    assert int(row["n_events"]) == 0

    with pytest.raises(KeyError, match="negative-control frame missing"):
        module.negative_control_outcome_analysis(
            labelled.drop(columns=[module.NEGATIVE_CONTROL_COL])
        )


def test_ascertainment_by_exposure_and_degenerate_paths(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    labelled = module.add_negative_control_outcomes(
        frame, _episodes_for(frame), _classifier
    )
    out = module.ascertainment_by_exposure(labelled)
    assert list(out["exposure_band"]) == list(module.EXPOSURE_BANDS)
    assert out["odds_ratio_per_90_minutes"].notna().all()
    populated = out[out["n_spell_starts"].gt(0)]
    assert populated["share_specific"].between(0.0, 1.0).all()
    assert out.loc[out["n_spell_starts"].eq(0), "share_specific"].isna().all()

    # With no variation in reporting detail there is nothing to model, so the
    # estimate is withheld rather than fabricated from a degenerate fit.
    constant = labelled.copy()
    constant[module.SPECIFIC_TEXT_COL] = 1
    degenerate = module.ascertainment_by_exposure(constant)
    assert degenerate["odds_ratio_per_90_minutes"].isna().all()
    assert degenerate["p_value"].isna().all()

    with pytest.raises(ValueError, match="no same-day spell starts"):
        module.ascertainment_by_exposure(
            labelled.assign(injury_event_matchproxy_same_day=0)
        )
    with pytest.raises(KeyError, match="ascertainment frame missing columns"):
        module.ascertainment_by_exposure(pd.DataFrame())


def test_club_congestion_features_and_sensitivity(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    frame = frame.assign(
        player_club_id=frame["tm_player_id"] % 4,
        club_season=(frame["tm_player_id"] % 4).astype(str) + "_2020",
    )
    featured = module.add_club_fixture_congestion(frame)
    assert featured["club_days_since_last_fixture"].between(0.0, 30.0).all()
    assert featured["club_fixtures_last_7d"].ge(0).all()
    # A club's first fixture has no predecessor, so the gap is capped, not null.
    assert featured["club_days_since_last_fixture"].notna().all()

    out = module.club_congestion_sensitivity(featured)
    assert list(out["model_id"]) == [
        "reference",
        "plus_club_schedule",
        "plus_club_schedule_and_club_season",
    ]
    assert (out["ci_low"] <= out["estimate"]).all()
    assert (out["estimate"] <= out["ci_high"]).all()

    with pytest.raises(ValueError, match="no club fixtures available"):
        module.add_club_fixture_congestion(
            featured.assign(player_club_id=np.nan, date=pd.NaT)
        )
    with pytest.raises(
        KeyError, match="club congestion sensitivity frame missing columns"
    ):
        module.club_congestion_sensitivity(pd.DataFrame())


def test_run_in_threshold_sensitivity_marks_the_reference(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    out = module.run_in_threshold_sensitivity(frame, thresholds=(0.0, 900.0))
    assert list(out["run_in_minutes"]) == [0.0, 900.0]
    assert list(out["is_reference"]) == [False, True]
    # A lower run-in must retain at least as many rows as a higher one.
    assert out.iloc[0]["n_rows"] >= out.iloc[1]["n_rows"]

    with pytest.raises(ValueError, match="no estimable rows at a"):
        module.run_in_threshold_sensitivity(frame, thresholds=(1e12,))
    with pytest.raises(KeyError, match="run-in frame missing columns"):
        module.run_in_threshold_sensitivity(pd.DataFrame())


def test_non_event_audit_queue_is_blinded_and_deterministic(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    queue = module.build_non_event_audit_queue(frame, sample_size=12)
    assert len(queue) == 12
    assert queue["audit_id"].is_unique
    assert queue["missed_event_verdict"].eq("pending").all()
    # No exposure or fitted value may reach the reviewer.
    assert not {"prior_minutes_7d", "all_minutes_last_7d"} & set(queue.columns)
    assert module.build_non_event_audit_queue(frame, sample_size=12).equals(queue)

    with pytest.raises(ValueError, match="sample_size must be positive"):
        module.build_non_event_audit_queue(frame, sample_size=0)
    with pytest.raises(ValueError, match="no non-event appearances"):
        module.build_non_event_audit_queue(
            frame.assign(
                injury_event_matchproxy_same_day=1, injury_event_matchproxy_lag1=0
            )
        )
    with pytest.raises(KeyError, match="non-event audit frame missing columns"):
        module.build_non_event_audit_queue(pd.DataFrame())


def test_summarize_non_event_audit_refuses_to_score_pending_rows(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    pending = pd.DataFrame({"missed_event_verdict": ["pending"] * 5})
    outstanding = module.summarize_non_event_audit(pending).iloc[0]
    assert outstanding["status"] == "outstanding"
    assert int(outstanding["n_resolved"]) == 0
    assert pd.isna(outstanding["missed_event_proportion"])

    adjudicated = pd.DataFrame(
        {
            "missed_event_verdict": [
                "missed_event",
                "no_missed_event",
                "no_missed_event",
                "no_missed_event",
            ]
        }
    )
    complete = module.summarize_non_event_audit(adjudicated).iloc[0]
    assert complete["status"] == "complete"
    assert int(complete["n_resolved"]) == 4
    assert complete["missed_event_proportion"] == pytest.approx(0.25)
    assert complete["ci_low"] < 0.25 < complete["ci_high"]

    # A review that was carried out but resolved nothing is not a completed
    # audit. Calling it complete would let a reader read zero missed events as
    # a finding rather than as the absence of one.
    reviewed = pd.DataFrame({"missed_event_verdict": ["unresolved"] * 4 + ["missed_event"]})
    partial = module.summarize_non_event_audit(reviewed).iloc[0]
    assert partial["status"] == "reviewed, not resolved"
    assert int(partial["n_unresolved"]) == 4
    assert int(partial["n_pending"]) == 0
    assert int(partial["n_resolved"]) == 1

    with pytest.raises(KeyError, match="non-event audit review missing columns"):
        module.summarize_non_event_audit(pd.DataFrame())


def test_cohen_kappa_and_second_assessor_status(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    empty = module.cohen_kappa(pd.Series(dtype=str), pd.Series(dtype=str))
    assert empty["n_compared"] == 0
    assert np.isnan(empty["kappa"])

    first = pd.Series(["confirmed", "confirmed", "not_confirmed", "unresolved"])
    second = pd.Series(["confirmed", "not_confirmed", "not_confirmed", "unresolved"])
    statistics = module.cohen_kappa(first, second)
    assert statistics["n_compared"] == 4
    assert statistics["observed_agreement"] == pytest.approx(0.75)
    assert np.isfinite(statistics["kappa"])

    # Perfect agreement on a single category leaves kappa undefined rather
    # than reporting a spurious 1.0.
    constant = pd.Series(["confirmed"] * 4)
    assert np.isnan(module.cohen_kappa(constant, constant)["kappa"])

    reviewed = pd.DataFrame({"date_attribution_verdict": list(first)})
    outstanding = module.second_assessor_agreement(reviewed).iloc[0]
    assert outstanding["status"] == "outstanding"
    assert int(outstanding["n_compared"]) == 0

    paired = reviewed.assign(date_attribution_verdict_second_assessor=list(second))
    complete = module.second_assessor_agreement(paired).iloc[0]
    assert complete["status"] == "complete"
    assert int(complete["n_compared"]) == 4


def _with_lineup(module, players: int = 30, appearances: int = 14) -> pd.DataFrame:
    """Attach squad roles so recorded minutes depend on exposure the way they
    do in the real panel: high-exposure rows start and play long, low-exposure
    rows come on and play little, and event rows among starters are truncated.
    """
    frame = _with_windows(module, players=players, appearances=appearances)
    exposure = pd.to_numeric(frame["prior_minutes_7d"], errors="coerce").fillna(0.0)
    starter = exposure.ge(exposure.median())
    frame = frame.assign(
        lineup_role_model=np.where(starter, "starting_lineup", "substitute_list")
    )
    minutes = np.where(starter, 90.0, 20.0)
    event = pd.to_numeric(
        frame["injury_event_matchproxy_same_day"], errors="coerce"
    ).fillna(0).eq(1)
    minutes = np.where(event & starter, 50.0, minutes)
    return frame.assign(all_minutes_played=minutes)


def test_recorded_minute_distribution_separates_roles(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.recorded_minute_distribution(_with_lineup(module))
    assert set(out["lineup_role"]) <= {
        "starting_lineup",
        "substitute_list",
        "lineup_unavailable_or_other",
        "all",
    }
    indexed = out.set_index(["lineup_role", "event_status"])
    # Starters carrying a report are recorded shorter; substitutes are not.
    starters_event = indexed.loc[("starting_lineup", "same_day_spell_start")]
    starters_clean = indexed.loc[("starting_lineup", "no_same_day_report")]
    assert starters_event["median_minutes"] < starters_clean["median_minutes"]
    subs_event = indexed.loc[("substitute_list", "same_day_spell_start")]
    subs_clean = indexed.loc[("substitute_list", "no_same_day_report")]
    assert subs_event["median_minutes"] == subs_clean["median_minutes"]
    assert (out["p10_minutes"] <= out["median_minutes"]).all()
    assert (out["median_minutes"] <= out["p90_minutes"]).all()

    with pytest.raises(KeyError, match="minute distribution frame missing columns"):
        module.recorded_minute_distribution(pd.DataFrame())


def test_lineup_composition_tracks_exposure(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.lineup_composition_by_exposure(_with_lineup(module))
    assert list(out["exposure_band"]) == list(module.EXPOSURE_BANDS)
    populated = out[out["n_appearances"].gt(0)]
    assert populated["share_starting_lineup"].between(0.0, 1.0).all()
    # The construction ties starting to exposure, so the share must not be flat.
    assert populated["share_starting_lineup"].nunique() > 1

    with pytest.raises(KeyError, match="lineup composition frame missing columns"):
        module.lineup_composition_by_exposure(pd.DataFrame())


def test_denominator_by_lineup_role_reports_within_role_gaps(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.denominator_by_lineup_role(_with_lineup(module))
    assert {"all", "starting_lineup", "substitute_list"} <= set(out["lineup_role"])
    estimable = out[out["estimable"].astype(bool)]
    assert set(estimable["denominator"]) == {
        "per_appearance",
        "observed_minutes",
        "fixed_90",
    }
    # Every estimable role carries the within-Poisson attenuation gap.
    assert estimable["log_attenuation_fixed90_minus_recorded"].notna().all()

    # An offset can only attenuate a coefficient to the extent that it varies,
    # so the spread of the log offset is reported beside the attenuation. In
    # this fixture starters play a constant 90 minutes unless truncated, so
    # their spread must sit below the pooled spread.
    spread = out.drop_duplicates("lineup_role").set_index("lineup_role")
    assert spread.loc["starting_lineup", "sd_log_recorded_minutes"] < spread.loc[
        "all", "sd_log_recorded_minutes"
    ]
    assert spread.loc["all", "iqr_recorded_minutes"] > 0.0

    # A role with fewer than ten events is recorded as not estimable rather
    # than fitted on a handful of rows.
    sparse = _with_lineup(module).copy()
    substitutes = sparse["lineup_role_model"].eq("substitute_list")
    sparse.loc[substitutes, "injury_event_matchproxy_same_day"] = 0
    sparse_out = module.denominator_by_lineup_role(sparse)
    row = sparse_out[sparse_out["lineup_role"].eq("substitute_list")].iloc[0]
    assert not bool(row["estimable"])
    assert pd.isna(row["estimate"])

    with pytest.raises(KeyError, match="denominator role frame missing columns"):
        module.denominator_by_lineup_role(pd.DataFrame())


def test_attenuation_decomposition_attributes_the_gap(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.denominator_attenuation_decomposition(_with_lineup(module))
    values = out.set_index("quantity")["value"]
    for quantity in (
        "gamma_log_minutes_on_exposure",
        "gamma_with_truncation_removed",
        "gamma_attributable_to_outcome_truncation",
        "truncation_share_of_gamma",
        "gamma_within_starting_lineup",
        "gamma_within_substitute_list",
        "gamma_within_lineup_unavailable_or_other",
        "observed_log_attenuation_fixed90_minus_recorded",
        "minutes_lost_to_truncation",
        "truncation_share_of_all_recorded_minutes",
        "rate_inflation_factor_from_truncation",
    ):
        assert quantity in values.index

    # Gamma carries the composition argument, so every gamma must arrive with
    # bounds that bracket it. The remaining quantities are ratios and totals
    # derived from those slopes and carry no interval of their own.
    indexed = out.set_index("quantity")
    gammas = [q for q in indexed.index if q.startswith("gamma_") and "within" in q]
    gammas += ["gamma_log_minutes_on_exposure", "gamma_with_truncation_removed"]
    estimated = 0
    for quantity in gammas:
        row = indexed.loc[quantity]
        if pd.isna(row["value"]):
            # An unestimable stratum reports a blank interval rather
            # than a fabricated one.
            assert pd.isna(row["ci_low"]) and pd.isna(row["ci_high"]), quantity
            continue
        assert row["ci_low"] <= row["value"] <= row["ci_high"], quantity
        estimated += 1
    assert estimated >= 3
    assert pd.isna(indexed.loc["truncation_share_of_gamma", "ci_low"])

    # The truncation component is the difference between the two gammas, and
    # rare events cannot move a denominator built from every appearance.
    assert np.isclose(
        values["gamma_attributable_to_outcome_truncation"],
        values["gamma_log_minutes_on_exposure"] - values["gamma_with_truncation_removed"],
    )
    assert values["minutes_lost_to_truncation"] >= 0
    assert values["truncation_share_of_all_recorded_minutes"] < 0.1
    assert values["rate_inflation_factor_from_truncation"] >= 1.0

    with pytest.raises(KeyError, match="attenuation decomposition frame missing columns"):
        module.denominator_attenuation_decomposition(pd.DataFrame())
    with pytest.raises(ValueError, match="no estimable rows"):
        module.denominator_attenuation_decomposition(
            _with_lineup(module).assign(injury_event_matchproxy_same_day=0)
        )


def test_lineup_coverage_stability_compares_eras(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_lineup(module)
    # Two seasons, one with complete lineup status and one without, so the
    # complete-lineup era and the remainder are both populated.
    half = len(frame) // 2
    frame = frame.copy()
    frame.iloc[:half, frame.columns.get_loc("season_start")] = 2020
    frame.iloc[half:, frame.columns.get_loc("season_start")] = 2021
    frame.iloc[half:, frame.columns.get_loc("lineup_role_model")] = (
        "lineup_unavailable_or_other"
    )
    out = module.lineup_coverage_denominator_stability(frame)
    assert {"complete_lineup_seasons", "remaining_seasons"} <= set(out["stratum"])
    assert {"lineup_status_known", "lineup_status_unknown"} <= set(out["stratum"])
    assert (
        out["event_minus_non_event_minutes"]
        == out["mean_minutes_event"] - out["mean_minutes_non_event"]
    ).all()

    with pytest.raises(KeyError, match="lineup coverage frame missing columns"):
        module.lineup_coverage_denominator_stability(pd.DataFrame())


def test_event_clustering_summary_describes_repetition(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.event_clustering_summary(_with_windows(module)).iloc[0]
    assert out["n_players_with_event"] <= out["n_players"]
    assert 0.0 <= out["share_of_players_with_event"] <= 1.0
    assert out["median_events_per_affected_player"] >= 1
    assert 0.0 <= out["share_of_events_in_top_decile_of_players"] <= 1.0

    # With no events at all the shares are undefined rather than invented.
    empty = module.event_clustering_summary(
        _with_windows(module).assign(injury_event_matchproxy_same_day=0)
    ).iloc[0]
    assert int(empty["n_players_with_event"]) == 0
    assert pd.isna(empty["share_of_events_in_top_decile_of_players"])

    with pytest.raises(KeyError, match="event clustering frame missing columns"):
        module.event_clustering_summary(pd.DataFrame())


def test_model_field_completeness_counts_listwise_deletion(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module)
    out = module.model_field_completeness(frame)
    assert out["n_missing"].eq(0).all()
    assert out["n_analysed"].eq(len(frame)).all()

    holed = frame.copy()
    holed.loc[holed.index[:5], module.HISTORY_MODEL_COL] = np.nan
    holed_out = module.model_field_completeness(holed)
    history = holed_out[holed_out["model_field"].eq(module.HISTORY_MODEL_COL)].iloc[0]
    assert int(history["n_missing"]) == 5
    assert history["percent_missing"] > 0

    with pytest.raises(KeyError, match="model completeness frame missing columns"):
        module.model_field_completeness(pd.DataFrame())


def test_episode_type_composition_shows_what_the_source_records(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    episodes = pd.DataFrame(
        {
            "injury_desc": ["flu"] * 3 + ["hamstring strain"] * 6 + ["unknown"],
        }
    )
    out = module.episode_type_composition(episodes, _classifier)
    assert np.isclose(out["share_of_episodes"].sum(), 1.0)
    # Sorted by frequency, and the illness flag is set on the illness row only.
    assert out["n_episodes"].is_monotonic_decreasing
    illness = out[out["is_illness"]]
    assert len(illness) == 1
    assert int(illness["n_episodes"].iloc[0]) == 3

    with pytest.raises(KeyError, match="episode composition frame missing columns"):
        module.episode_type_composition(pd.DataFrame(), _classifier)


def test_placebo_analysis_reports_window_correlation(load_src_module):
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = module.add_placebo_exposure_window(
        _with_windows(module, players=30, appearances=40)
    )
    out = module.placebo_window_analysis(frame)
    # The referee needs the collinearity the mutual adjustment had to separate.
    assert out["pearson_r_recent_vs_placebo"].between(-1.0, 1.0).all()
    assert out["spearman_r_recent_vs_placebo"].between(-1.0, 1.0).all()
    assert out["pearson_r_recent_vs_placebo"].nunique() == 1


def test_decomposition_and_coverage_handle_thin_strata(load_src_module):
    """Thin strata are skipped rather than fitted on a handful of rows."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_lineup(module)

    # Leave fewer than 50 substitute rows, so the within-role gamma for that
    # role is withheld instead of estimated from noise.
    roles = frame["lineup_role_model"].astype(str)
    substitute_index = frame.index[roles.eq("substitute_list")]
    thin = frame.copy()
    thin.loc[substitute_index[:-10], "lineup_role_model"] = "starting_lineup"
    values = module.denominator_attenuation_decomposition(thin).set_index("quantity")[
        "value"
    ]
    assert pd.isna(values["gamma_within_substitute_list"])
    assert np.isfinite(values["gamma_within_starting_lineup"])

    # A stratum containing no events contributes no row to the coverage table.
    single_era = frame.copy()
    single_era["season_start"] = 2020
    out = module.lineup_coverage_denominator_stability(single_era)
    assert "remaining_seasons" not in set(out["stratum"])
    assert "complete_lineup_seasons" in set(out["stratum"])


def test_direct_truncation_refit_isolates_the_offset(load_src_module):
    """Only the offset differs between the three fits, so their gaps are
    measurements rather than predictions."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_lineup(module)
    out = module.direct_truncation_refit(frame)
    assert list(out["offset"]) == ["recorded_minutes", "untruncated_minutes", "fixed_90"]
    assert out["n_rows"].nunique() == 1
    assert out["n_events"].nunique() == 1
    assert (out["ci_low"] <= out["estimate"]).all()
    assert (out["estimate"] <= out["ci_high"]).all()

    indexed = out.set_index("offset")["log_estimate"]
    assert np.isclose(
        out["log_gap_untruncated_minus_recorded"].iloc[0],
        indexed["untruncated_minutes"] - indexed["recorded_minutes"],
    )
    assert np.isclose(
        out["log_attenuation_fixed90_minus_recorded"].iloc[0],
        indexed["fixed_90"] - indexed["recorded_minutes"],
    )
    # Events are rare, so removing their truncation cannot move a denominator
    # built from every appearance by much.
    assert abs(float(out["truncation_share_of_attenuation"].iloc[0])) < 0.25

    with pytest.raises(KeyError, match="direct truncation refit frame missing columns"):
        module.direct_truncation_refit(pd.DataFrame())
    with pytest.raises(ValueError, match="no estimable rows"):
        module.direct_truncation_refit(
            frame.assign(injury_event_matchproxy_same_day=0)
        )


def test_case_restricted_exposure_bias_separates_the_scales(load_src_module):
    """The same truncation is diluted in a cohort denominator and undiluted in
    a case-restricted one."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.case_restricted_exposure_bias(_with_lineup(module)).set_index("quantity")
    cases = out.loc["mean_recorded_minutes_on_event_appearances"]
    cohort = out.loc["total_minutes_whole_cohort"]
    assert cases["observed"] <= cases["truncation_removed"]
    assert cohort["observed"] <= cohort["truncation_removed"]
    # Rare events cannot move the cohort total as far as they move their own.
    assert cases["percent_understated"] > cohort["percent_understated"]
    assert (out["ratio_observed_to_untruncated"] <= 1.0).all()

    with pytest.raises(KeyError, match="case-restricted frame missing columns"):
        module.case_restricted_exposure_bias(pd.DataFrame())
    with pytest.raises(ValueError, match="no same-day spell starts"):
        module.case_restricted_exposure_bias(
            _with_lineup(module).assign(injury_event_matchproxy_same_day=0)
        )


def test_untruncated_minutes_only_touches_event_rows(load_src_module):
    """The counterfactual must leave non-event appearances exactly as found."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_lineup(module)
    work = frame[pd.to_numeric(frame["all_minutes_played"], errors="coerce").gt(0)].copy()
    replaced = module._untruncated_minutes(work)
    events = work["injury_event_matchproxy_same_day"].eq(1)
    original = pd.to_numeric(work["all_minutes_played"], errors="coerce").clip(lower=1.0)
    assert replaced[~events].equals(original[~events])
    assert (replaced > 0).all()

    # A role with no non-event rows falls back to the overall non-event mean
    # rather than producing a missing denominator.
    orphan = work.copy()
    orphan.loc[orphan.index[:3], "lineup_role_model"] = "unseen_role"
    orphan.loc[orphan.index[:3], "injury_event_matchproxy_same_day"] = 1
    assert module._untruncated_minutes(orphan).notna().all()


def test_relative_attenuation_guards_against_weaker_base_estimates(load_src_module):
    """Attenuation must be reported against the estimate it acts on."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.denominator_by_lineup_role(_with_lineup(module))
    estimable = out[out["estimable"].astype(bool)]
    for role, group in estimable.groupby("lineup_role"):
        indexed = group.set_index("denominator")
        base = float(indexed.loc["fixed_90", "log_estimate"])
        expected = (base - float(indexed.loc["observed_minutes", "log_estimate"])) / base
        assert np.isclose(float(group["relative_attenuation"].iloc[0]), expected)


def test_decomposition_records_identity_overprediction(load_src_module):
    """The identity is an expansion; the table must expose how far it sits from
    the attenuation actually observed."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    values = module.denominator_attenuation_decomposition(
        _with_lineup(module)
    ).set_index("quantity")["value"]
    assert "gamma_over_observed_attenuation" in values.index
    assert "relative_attenuation_pooled" in values.index
    assert np.isclose(
        values["gamma_over_observed_attenuation"],
        values["gamma_log_minutes_on_exposure"]
        / values["observed_log_attenuation_fixed90_minus_recorded"],
    )


def test_untruncated_minutes_honours_every_imputation_scheme(load_src_module):
    """Each scheme must leave non-event rows untouched and produce a usable
    denominator for every event row."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_lineup(module)
    work = frame[pd.to_numeric(frame["all_minutes_played"], errors="coerce").gt(0)].copy()
    events = work["injury_event_matchproxy_same_day"].eq(1)
    original = pd.to_numeric(work["all_minutes_played"], errors="coerce").clip(lower=1.0)

    for scheme in module.IMPUTATION_SCHEMES:
        replaced = module._untruncated_minutes(work, scheme)
        assert replaced[~events].equals(original[~events]), scheme
        assert replaced.notna().all(), scheme
        assert (replaced > 0).all(), scheme

    with pytest.raises(ValueError, match="Unknown imputation scheme"):
        module._untruncated_minutes(work, "not_a_scheme")


def test_imputation_sensitivity_reports_a_range(load_src_module):
    """The case-restricted distortion is the quantity most exposed to the
    imputation, so every scheme must be reported."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.truncation_imputation_sensitivity(_with_lineup(module))
    assert list(out["imputation_scheme"]) == list(module.IMPUTATION_SCHEMES)

    # Rare events cannot move a cohort denominator as far as they move their
    # own, whichever scheme is used.
    assert (
        out["case_restricted_percent_understated"]
        >= out["cohort_percent_understated"]
    ).all()
    # The attribution must be stable across schemes even where the magnitude
    # is not: truncation explains a negligible share under all of them.
    assert out["truncation_share_of_attenuation"].abs().max() < 0.25

    with pytest.raises(KeyError, match="imputation sensitivity frame missing columns"):
        module.truncation_imputation_sensitivity(pd.DataFrame())
    with pytest.raises(ValueError, match="no same-day spell starts"):
        module.truncation_imputation_sensitivity(
            _with_lineup(module).assign(injury_event_matchproxy_same_day=0)
        )


def test_attenuation_bootstrap_brackets_its_point_estimates(load_src_module):
    """The mechanism argument compares two attenuations, so both need
    intervals, and so does the ratio between them."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.attenuation_bootstrap(
        _with_lineup(module, players=40, appearances=14), replicates=12
    )
    expected = {
        "all",
        "starting_lineup",
        "substitute_list",
        "lineup_unavailable_or_other",
        "pooled_minus_starter_absolute",
        "all_relative",
        "starting_lineup_relative",
        "substitute_list_relative",
        "lineup_unavailable_or_other_relative",
        "pooled_over_starter_relative",
        "truncation_attribution_absolute",
        "role_adjusted_absolute",
    }
    assert set(out["quantity"]) == expected

    # The difference of absolute attenuations is the stable statistic, so it
    # must be internally consistent with the two it is built from.
    indexed = out.set_index("quantity")["estimate"]
    assert np.isclose(
        indexed["pooled_minus_starter_absolute"],
        indexed["all"] - indexed["starting_lineup"],
    )
    assert out["resampling_unit"].eq("player").all()
    assert out["n_replicates_requested"].eq(12).all()

    estimable = out[out["n_replicates_estimable"].gt(0)]
    assert not estimable.empty
    assert (estimable["ci_low"] <= estimable["ci_high"]).all()

    with pytest.raises(ValueError, match="replicates must be positive"):
        module.attenuation_bootstrap(_with_lineup(module), replicates=0)
    with pytest.raises(KeyError, match="attenuation bootstrap frame missing columns"):
        module.attenuation_bootstrap(pd.DataFrame())
    with pytest.raises(ValueError, match="no estimable rows"):
        module.attenuation_bootstrap(
            _with_lineup(module).assign(injury_event_matchproxy_same_day=0)
        )


def test_attenuation_bootstrap_withholds_thin_strata(load_src_module):
    """A stratum with fewer than ten events is withheld rather than fitted, and
    a missing starter stratum leaves the ratio undefined rather than infinite."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_lineup(module)
    starters = frame["lineup_role_model"].astype(str).eq("starting_lineup")
    thin = frame.copy()
    thin.loc[starters, "injury_event_matchproxy_same_day"] = 0
    out = module.attenuation_bootstrap(thin, replicates=4).set_index("quantity")

    assert pd.isna(out.loc["starting_lineup", "estimate"])
    # NaN is truthy, so the ratio guard has to test for a usable number.
    assert pd.isna(out.loc["pooled_over_starter_relative", "estimate"])
    assert int(out.loc["starting_lineup", "n_replicates_estimable"]) == 0


def test_squad_role_association_reports_adjusted_and_stratified_estimates(
    load_src_module,
):
    """The composition that contaminates the denominator could also generate
    the association, so the association is reported adjusted for squad role and
    within each role rather than pooled only."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.squad_role_association_sensitivity(_with_lineup(module))
    analyses = set(out["analysis"])
    assert "pooled, unadjusted for squad role" in analyses
    assert "pooled, adjusted for squad role" in analyses
    assert "within starting_lineup" in analyses

    indexed = out.set_index("analysis")
    unadjusted = indexed.loc["pooled, unadjusted for squad role"]
    adjusted = indexed.loc["pooled, adjusted for squad role"]
    # Adjusting adds one categorical term to the same rows, so the comparison
    # is like-for-like and only the estimate may move.
    assert int(adjusted["n_rows"]) == int(unadjusted["n_rows"])
    assert int(adjusted["n_events"]) == int(unadjusted["n_events"])
    assert "lineup_role_term" in str(adjusted["formula"])
    assert adjusted["log_estimate"] != unadjusted["log_estimate"]
    assert (out["ci_low"].dropna() <= out["estimate"].dropna()).all()

    # A role carrying fewer than ten events is recorded as a row with counts
    # and no estimate, never fitted on a handful of appearances. The fixture
    # has no unknown-lineup rows, so that stratum takes this path.
    unknown = indexed.loc["within lineup_unavailable_or_other"]
    assert pd.isna(unknown["estimate"])
    assert int(unknown["n_events"]) == 0

    with pytest.raises(KeyError, match="role association frame missing columns"):
        module.squad_role_association_sensitivity(pd.DataFrame())


def test_exposure_window_gradient_separates_precision_from_effect(load_src_module):
    """Adjusted p values falling with window length invite the reading that
    longer windows carry a stronger signal; the estimates must be printed
    beside them so that reading is checkable."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    summary = pd.DataFrame(
        {
            "exposure_id": [
                "prior_minutes_3d",
                "prior_minutes_5d",
                "prior_minutes_14d",
                "prior_minutes_7d",
                "prior_minutes_10d",
                "prior_matches_7d",
            ],
            "estimate": [1.26, 1.40, 1.19, 1.27, 1.23, 1.20],
            "ci_low": [0.98, 1.15, 1.10, 1.11, 1.11, 1.05],
            "ci_high": [1.62, 1.69, 1.29, 1.44, 1.38, 1.36],
            "standard_error": [0.130, 0.098, 0.041, 0.067, 0.055, 0.067],
            "p_value": [0.076, 0.001, 0.000, 0.000, 0.000, 0.008],
            "holm_p_value_63_model_family": [1.0, 0.027, 0.001, 0.016, 0.006, 0.278],
            "n_events": [576] * 6,
        }
    )
    out = module.exposure_window_gradient(summary)

    # Only the five cumulative windows, and in ascending order regardless of
    # the order they arrive in.
    assert out["window_days"].tolist() == [3, 5, 7, 10, 14]
    assert "prior_matches_7d" not in set(out["exposure_id"])

    # The two gradients run in opposite directions, which is the point.
    assert out["holm_p_value_63_model_family"].is_monotonic_decreasing
    assert not out["estimate"].is_monotonic_increasing
    assert out["standard_error"].is_monotonic_decreasing
    note = str(out["interpretation"].iloc[0])
    assert "precision, not effect size" in note
    assert "1.40" in note and "5 days" in note

    with pytest.raises(KeyError, match="window gradient summary missing columns"):
        module.exposure_window_gradient(pd.DataFrame())


def test_within_stratum_gamma_uses_the_same_clustered_variance_as_pooled(
    load_src_module,
):
    """Gamma carries the composition argument, so the pooled and within-stratum
    values must rest on one variance estimator.

    A player contributes many appearances. If the stratified fits used ordinary
    errors while the pooled fit clustered, the table would invite a comparison
    across strata that its own intervals could not support.
    """
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_lineup(module, players=40, appearances=16)
    out = module.denominator_attenuation_decomposition(frame).set_index("quantity")

    # Rebuild the within-starter fit both ways and see which interval was used.
    required = [
        "injury_event_matchproxy_same_day",
        "all_minutes_played",
        "tm_player_id",
        "lineup_role_model",
        "prior_minutes_7d",
        "history_log_iqr",
        *module.CALENDAR_TERMS,
    ]
    work = frame.dropna(subset=required).copy()
    work["all_minutes_played"] = pd.to_numeric(
        work["all_minutes_played"], errors="coerce"
    )
    work = work[work["all_minutes_played"].gt(0.0)].copy()
    work["exposure_per_90"] = (
        pd.to_numeric(work["prior_minutes_7d"], errors="coerce") / 90.0
    )
    work["log_recorded_minutes"] = np.log(work["all_minutes_played"].clip(lower=1.0))
    starters = work[work["lineup_role_model"].astype(str).eq("starting_lineup")]

    formula = (
        "log_recorded_minutes ~ exposure_per_90 + history_log_iqr + "
        + " + ".join(module.CALENDAR_TERMS)
    )
    model = smf.ols(formula, data=starters)
    clustered = model.fit(
        cov_type="cluster", cov_kwds={"groups": starters["tm_player_id"]}
    ).bse["exposure_per_90"]
    ordinary = model.fit().bse["exposure_per_90"]

    row = out.loc["gamma_within_starting_lineup"]
    half_width = (float(row["ci_high"]) - float(row["ci_low"])) / 2.0
    critical = NormalDist().inv_cdf(0.975)
    assert np.isclose(half_width, critical * clustered, rtol=1e-6)

    # The check is only meaningful if the two estimators actually differ here.
    assert not np.isclose(clustered, ordinary, rtol=1e-3)


def test_denominator_roles_report_player_counts(load_src_module):
    """Inference is player-clustered, so the effective sample size behind a
    stratum is its player count rather than its row count."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.denominator_by_lineup_role(_with_lineup(module))
    per_role = out.drop_duplicates("lineup_role").set_index("lineup_role")
    assert (per_role["n_players"] > 0).any()
    for role in ("starting_lineup", "substitute_list"):
        row = per_role.loc[role]
        assert 0 < int(row["n_players"]) <= int(row["n_rows"])
    assert int(per_role.loc["all", "n_players"]) >= int(
        per_role.loc["starting_lineup", "n_players"]
    )


def test_correlation_table_marks_cumulative_window_pairs(load_src_module):
    """The manuscript quotes one correlation range for the cumulative windows,
    so the table has to fix which pairs that range is taken over."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.exposure_metric_correlations(_with_windows(module))
    assert "both_cumulative_windows" in out.columns

    flagged = out[out["both_cumulative_windows"].astype(bool)]
    assert not flagged.empty
    for _, row in flagged.iterrows():
        assert str(row["exposure_a"]).startswith("prior_minutes_")
        assert str(row["exposure_b"]).startswith("prior_minutes_")

    # Metrics that are not cumulative windows must be excluded, otherwise the
    # quoted range would silently widen to include the recovery contrast.
    unflagged = out[~out["both_cumulative_windows"].astype(bool)]
    assert not unflagged.empty
    for _, row in unflagged.iterrows():
        assert not (
            str(row["exposure_a"]).startswith("prior_minutes_")
            and str(row["exposure_b"]).startswith("prior_minutes_")
        )


def test_role_adjusted_refit_shows_adjustment_is_not_stratification(load_src_module):
    """A categorical role term enters the linear predictor; the offset keeps its
    within-role variation. Adjusting therefore cannot do what restricting to
    starters does, and the surviving attenuation is the evidence."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_lineup(module, players=40, appearances=16)
    out = module.role_adjusted_denominator_refit(frame)

    assert set(out["model"]) == {"unadjusted", "adjusted for squad role"}
    assert set(out["denominator"]) == {"fixed_90", "observed_minutes"}
    # Adjusting adds a term to the same rows, so the comparison is like-for-like.
    assert out["n_rows"].nunique() == 1
    assert out["n_events"].nunique() == 1
    assert out["log_attenuation_fixed90_minus_recorded"].notna().all()

    def _adjusted_gap(result: pd.DataFrame) -> float:
        per_model = result.drop_duplicates("model").set_index("model")
        return float(
            per_model.loc[
                "adjusted for squad role", "log_attenuation_fixed90_minus_recorded"
            ]
        )

    # What a role term can absorb is exactly the between-role part of the
    # offset. In this fixture minutes are constant inside each role, so the
    # term absorbs all of it and the attenuation collapses -- adjusting and
    # stratifying coincide only in that special case.
    assert abs(_adjusted_gap(out)) < 0.01

    # Give minutes a spread inside each role, as they have in the real panel,
    # and the same adjustment leaves the attenuation standing: the offset still
    # varies with exposure within the categories the term controls for.
    varied = frame.copy()
    exposure = pd.to_numeric(varied["prior_minutes_7d"], errors="coerce").fillna(0.0)
    within_role_spread = 1.0 + 0.9 * (exposure / max(exposure.max(), 1.0))
    varied["all_minutes_played"] = (
        pd.to_numeric(varied["all_minutes_played"], errors="coerce") * within_role_spread
    ).clip(lower=1.0, upper=90.0)
    varied_out = module.role_adjusted_denominator_refit(varied)
    assert _adjusted_gap(varied_out) > 10 * abs(_adjusted_gap(out))

    with pytest.raises(KeyError, match="role adjusted refit frame missing columns"):
        module.role_adjusted_denominator_refit(pd.DataFrame())


def test_run_in_exclusion_comparison_describes_who_is_dropped(load_src_module):
    """The run-in is an eligibility choice that removes a fifth of the source
    players, so the paper reports who they were instead of asserting it."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    frame = _with_windows(module, players=40, appearances=14)
    rng = np.random.default_rng(11)
    players = frame["tm_player_id"].unique()
    established = set(players[: len(players) // 2])
    frame = frame.assign(
        prior_minutes_played=[
            1500.0 if p in established else 300.0 for p in frame["tm_player_id"]
        ],
        age_years=[
            28.0 if p in established else 21.0 for p in frame["tm_player_id"]
        ],
    )
    out = module.run_in_exclusion_comparison(frame)

    assert set(out["population"]) == {"retained at the run-in", "excluded by the run-in"}
    indexed = out.set_index("population")
    assert (
        indexed.loc["excluded by the run-in", "median_age_years"]
        < indexed.loc["retained at the run-in", "median_age_years"]
    )
    assert (
        indexed.loc["excluded by the run-in", "median_prior_minutes_played"]
        < indexed.loc["retained at the run-in", "median_prior_minutes_played"]
    )
    assert int(out["n_players"].sum()) == len(players)
    assert out["run_in_minutes"].eq(900.0).all()

    with pytest.raises(KeyError, match="run-in exclusion frame missing columns"):
        module.run_in_exclusion_comparison(pd.DataFrame())


def test_bootstrap_gives_the_truncation_attribution_an_interval(load_src_module):
    """The attribution to truncation was the last quantity the paper argued
    from without uncertainty; it is resampled with the attenuations."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    out = module.attenuation_bootstrap(
        _with_lineup(module, players=40, appearances=14), replicates=12
    ).set_index("quantity")

    assert "truncation_attribution_absolute" in out.index
    row = out.loc["truncation_attribution_absolute"]
    assert int(row["n_replicates_estimable"]) > 0
    assert float(row["ci_low"]) <= float(row["ci_high"])
    assert np.isfinite(float(row["estimate"]))

    # It is measured on the same scale as the attenuation it is a share of, so
    # the two can be divided without rescaling.
    assert abs(float(row["estimate"])) < abs(float(out.loc["all", "estimate"]))

    # Adjusting for squad role is offered as an alternative to stratifying, so
    # the attenuation it leaves is argued from and carries the same interval.
    adjusted = out.loc["role_adjusted_absolute"]
    assert int(adjusted["n_replicates_estimable"]) > 0
    assert float(adjusted["ci_low"]) <= float(adjusted["estimate"]) <= float(
        adjusted["ci_high"]
    )


def test_placebo_denominator_replication_repeats_the_identity(load_src_module):
    """The identity's over-prediction should be a property of the denominator
    rather than of the seven-day exposure, so a second exposure with no
    plausible causal path must show the same structure."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    # The placebo window looks 31-37 days back, so the panel has to span well
    # beyond that or the placebo exposure is constant at zero.
    frame = module.add_placebo_exposure_window(
        _with_lineup(module, players=40, appearances=44)
    )
    assert frame["prior_minutes_placebo_31_37d"].gt(0).any()
    out = module.placebo_denominator_replication(frame).set_index("quantity")

    for denominator in ("per_appearance", "fixed_90", "observed_minutes"):
        row = out.loc[denominator]
        assert row["ci_low"] <= row["value"] <= row["ci_high"]

    # The comparison is like-for-like: one row set, one event count.
    assert out["n_rows"].nunique() == 1
    assert out["n_events"].nunique() == 1

    gamma = out.loc["gamma_placebo"]
    assert gamma["ci_low"] <= gamma["value"] <= gamma["ci_high"]

    # The attenuation is the gap between the two Poisson fits on identical
    # rows, so it must reconcile with the coefficients reported above.
    attenuation = float(out.loc["observed_log_attenuation", "value"])
    assert np.isclose(
        attenuation,
        np.log(out.loc["fixed_90", "value"]) - np.log(out.loc["observed_minutes", "value"]),
    )
    ratio = float(out.loc["gamma_over_observed_attenuation", "value"])
    assert np.isclose(ratio, float(gamma["value"]) / attenuation)

    with pytest.raises(KeyError, match="placebo denominator frame missing columns"):
        module.placebo_denominator_replication(pd.DataFrame())
    with pytest.raises(ValueError, match="no estimable rows"):
        module.placebo_denominator_replication(
            frame.assign(injury_event_matchproxy_same_day=0)
        )


def test_absence_screen_separates_calendar_gaps_from_real_absences(load_src_module):
    """A long gap before a player's next appearance is not an absence if the
    club played nothing, and is not an absence if the player left the club.
    The screen has to tell those apart, because otherwise every international
    window looks like a missed injury."""
    module = load_src_module("36_jsams_second_referee_analysis.py")
    queue = pd.DataFrame(
        {
            "audit_id": ["calendar", "real", "transfer", "absent_row"],
            "tm_player_id": [1, 2, 3, 4],
            "date": pd.to_datetime(["2020-01-01"] * 4),
        }
    )
    appearances = pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3, 3],
            "player_club_id": [10, 10, 20, 20, 30, 31],
            "date": pd.to_datetime(
                [
                    "2020-01-01", "2020-01-25",   # club idle across the gap
                    "2020-01-01", "2020-01-25",   # club played twice in the gap
                    "2020-01-01", "2020-01-25",   # player changed club
                ]
            ),
        }
    )
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-10", "2020-01-18", "2020-01-12"]),
            "home_club_id": [20, 99, 30],
            "away_club_id": [99, 20, 99],
        }
    )
    out = module.non_event_absence_screen(queue, appearances, fixtures).set_index(
        "audit_id"
    )

    assert int(out.loc["calendar", "club_fixtures_missed"]) == 0
    assert "no fixture" in out.loc["calendar", "screen"]
    assert int(out.loc["real", "club_fixtures_missed"]) == 2
    assert "search required" in out.loc["real", "screen"]
    assert bool(out.loc["transfer", "club_changed"])
    assert "club change" in out.loc["transfer", "screen"]
    # A queued appearance missing from the snapshot is reported, not guessed.
    assert pd.isna(out.loc["absent_row", "club_fixtures_missed"])
    assert "not in the appearance snapshot" in out.loc["absent_row", "screen"]

    # A player with no later appearance cannot have a gap measured.
    lone = module.non_event_absence_screen(
        queue.head(1), appearances.head(1), fixtures
    ).iloc[0]
    assert pd.isna(lone["days_to_next_appearance"])
    assert "no subsequent appearance" in lone["screen"]

    assert "assigns no verdicts" in out["interpretation"].iloc[0]
    # The screen must not restate a verdict it did not reach: the queue's
    # placeholder would otherwise contradict the reviewed audit table.
    carried = module.non_event_absence_screen(
        queue.assign(missed_event_verdict="pending"), appearances, fixtures
    )
    assert "missed_event_verdict" not in carried.columns

    with pytest.raises(KeyError, match="absence screen queue missing columns"):
        module.non_event_absence_screen(pd.DataFrame(), appearances, fixtures)
    with pytest.raises(KeyError, match="absence screen appearances missing columns"):
        module.non_event_absence_screen(queue, pd.DataFrame(), fixtures)
    with pytest.raises(KeyError, match="absence screen fixtures missing columns"):
        module.non_event_absence_screen(queue, appearances, pd.DataFrame())
