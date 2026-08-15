import numpy as np
import pandas as pd
import pytest


def test_18_imports(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    assert module.SPLINE_DF == 4
    assert module.PLAYER_ID_COL == "tm_player_id"
    assert module.MATCH_MINUTES_COL == "all_minutes_played"
    assert module.SELECTED_BURDENS == [0.0, 90.0, 180.0]
    assert module.DIAGNOSTIC_SUPPORT_BURDENS == [0.0, 90.0, 180.0, 220.0]
    assert module.PRIMARY_GRID_MAX == 180.0
    assert module.CONTRAST_WINDOWS == ((0.0, 180.0), (90.0, 180.0))
    assert module.RECOVERY_INTERVAL_ORDER[-2:] == [">14 days", "no prior match"]
    assert module.RECOVERY_TREND_ORDER[-1] == ">14 days"
    assert module.OUT_OF_TIME_GROUP_COL == "fragility_out_of_time_2017_2019_to_2020_2024"
    assert module.OUT_OF_TIME_TEST_PERIOD["period"] == "2020-2024"
    assert module.SENSITIVITY_EVENT_COLS["reported_absence_ge28d"] == "injury_event_matchproxy_ge28d"
    assert module.SENSITIVITY_EVENT_COLS["muscle_tendon_only"] == "injury_event_matchproxy_muscle_tendon"
    assert [spec["model"] for spec in module.OUTCOME_HISTORY_CROSS_SPECS] == [
        "reported_absence_ge28d_frequency_only_history",
        "muscle_tendon_only_frequency_only_history",
        "muscle_tendon_only_non_muscle_frequency_history",
    ]
    assert {
        spec["group_col"] for spec in module.OUTCOME_HISTORY_CROSS_SPECS
    } == {"fragility_frequency_only", module.NON_MUSCLE_HISTORY_GROUP_COL}
    assert module.NON_MUSCLE_HISTORY_GROUP_COL == "fragility_non_muscle_frequency_only"
    assert "joint/ligament" in module.KNOWN_NON_MUSCLE_HISTORY_TYPES
    assert "illness/other medical" not in module.KNOWN_NON_MUSCLE_HISTORY_TYPES
    assert "other/unspecified" not in module.KNOWN_NON_MUSCLE_HISTORY_TYPES
    assert module.BETWEEN_HISTORY_CONTRAST_BURDENS == (0.0, 180.0)
    assert module.SEVERE_REPORTED_ABSENCE_DAYS == 28.0
    assert module.LINEUP_START_TYPES == {"starting_lineup"}
    assert module.LINEUP_SUBSTITUTE_TYPES == {"substitutes"}
    assert module.MATCHPROXY_DURATION_BUCKETS[:3] == [
        "<1 week",
        "1 week to 2 months",
        "2 months to 1 year",
    ]
    assert module.RECOVERY_MODEL_SPECS[-1][0] == "per_match_cloglog"
    assert [p["period"] for p in module.TEMPORAL_PERIODS] == [
        "2017-2019",
        "2020-2021",
        "2022-2024",
    ]


def test_season_from_dates(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    seasons = module.season_from_dates(pd.Series(["2024-06-30", "2024-07-01"]))
    assert seasons.tolist() == [2023, 2024]


def test_out_of_time_fragility_thresholds(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    empty_thresholds = module.estimate_history_thresholds(
        pd.DataFrame(
            {
                "tm_player_id": [1],
                "date": pd.to_datetime(["2018-01-01"]),
                "prior_minutes_played": [100],
                "prior_injuries_per_10000min": [0.0],
                "prior_max_spell_duration_days": [0.0],
            }
        )
    )
    assert empty_thresholds == {"q1_freq": 0.0, "q3_freq": 0.0, "q1_sev": 0.0, "q3_sev": 0.0}

    history = pd.DataFrame(
        {
            "tm_player_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "date": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                    "2018-01-01",
                    "2018-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                ]
            ),
            "prior_minutes_played": [900, 900, 900, 900, 900, 900, 900, 100],
            "prior_n_spells": [0, 1, 2, 3, 1, 2, 3, 0],
            "prior_injuries_per_10000min": [0.0, 1.0, 8.0, 20.0, 0.0, 8.0, 25.0, 0.0],
            "prior_max_spell_duration_days": [0.0, 2.0, 20.0, 60.0, 0.0, 25.0, 80.0, 0.0],
        }
    )
    thresholds = module.estimate_history_thresholds(history.iloc[:4])
    labels = module.assign_history_labels_from_thresholds(history.iloc[4:], thresholds)
    assert labels.tolist() == ["tough", "regular", "fragile", "low_exposure"]

    labelled, audit = module.add_out_of_time_fragility_label(history)
    assert labelled[module.OUT_OF_TIME_GROUP_COL].tolist()[:4] == ["outside_test_window"] * 4
    assert labelled[module.OUT_OF_TIME_GROUP_COL].tolist()[4:] == [
        "tough",
        "regular",
        "fragile",
        "low_exposure",
    ]
    assert audit.loc[0, "derivation_period"] == "2017-2019"
    assert audit.loc[0, "test_higher_history_rows"] == 1


def test_alternative_fragility_labels(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            "prior_minutes_played": [899, 900, 900, 900],
            "prior_n_spells": [0, 1, 2, 1],
            "prior_injuries_per_10000min": [0, 1, 10, 1],
            "prior_max_spell_duration_days": [0, 5, 10, 30],
            "q3_freq": [8, 8, 8, 8],
            "q3_sev": [20, 20, 20, 20],
        }
    )
    out = module.add_alternative_fragility_labels(panel)
    assert out["fragility_count_only"].tolist() == [
        "low_exposure",
        "regular",
        "fragile",
        "regular",
    ]
    assert out["fragility_frequency_only"].tolist()[2] == "fragile"
    assert out["fragility_severity_only"].tolist()[3] == "fragile"
    assert out["fragility_prespecified_abs"].tolist()[3] == "fragile"


def test_non_muscle_frequency_history_label(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 2, 2],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "prior_minutes_played": [900, 1000, 1100, 900, 1000],
        }
    )
    injuries = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2, 2],
            "injury_spell_id": [101, 102, 201, 202],
            "start_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-01",
                "2024-01-02",
            ],
            "injury_desc": [
                "Hamstring injury",
                "Knee injury",
                "Metatarsal fracture",
                "Covid infection",
            ],
        }
    )

    counts = module.non_muscle_injury_day_counts(
        injuries,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-03"),
    )
    assert counts["n_non_muscle_spells_today"].sum() == 2
    assert counts["tm_player_id"].tolist() == [1, 2]
    empty_counts = module.non_muscle_injury_day_counts(
        injuries[["tm_player_id", "injury_spell_id", "start_date"]].iloc[[0]],
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-03"),
    )
    assert empty_counts.empty
    assert empty_counts.columns.tolist() == [
        "tm_player_id",
        "date",
        "n_non_muscle_spells_today",
    ]

    out = module.add_non_muscle_frequency_history_label(
        panel,
        injuries,
        frequency_threshold=5.0,
    )
    assert out["prior_non_muscle_n_spells"].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]
    assert out[module.NON_MUSCLE_HISTORY_GROUP_COL].tolist() == [
        "regular",
        "regular",
        "fragile",
        "regular",
        "fragile",
    ]
    assert out[module.NON_MUSCLE_HISTORY_THRESHOLD_COL].unique().tolist() == [5.0]

    estimated = module.add_non_muscle_frequency_history_label(panel, injuries)
    assert module.NON_MUSCLE_HISTORY_THRESHOLD_COL in estimated.columns
    empty = module.add_non_muscle_frequency_history_label(
        panel.iloc[0:0],
        injuries.iloc[0:0],
    )
    assert empty[module.NON_MUSCLE_HISTORY_GROUP_COL].empty
    with pytest.raises(KeyError):
        module.add_non_muscle_frequency_history_label(
            panel.drop(columns=["prior_minutes_played"]),
            injuries,
        )
    with pytest.raises(KeyError):
        module.non_muscle_injury_day_counts(
            injuries.drop(columns=["start_date"]),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-03"),
        )


def test_matchproxy_duration_type_and_outcome_subsets(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")

    assert module.parse_duration_days({"days": 12}) == 12.0
    assert module.parse_duration_days("{'days': 40}") == 40.0
    assert np.isnan(module.parse_duration_days(""))
    assert np.isnan(module.parse_duration_days("not a dict"))
    assert np.isnan(module.parse_duration_days(123))
    assert np.isnan(module.parse_duration_days({"days": "not numeric"}))
    assert module.matchproxy_duration_bucket(np.nan) == "unknown"
    assert module.matchproxy_duration_bucket(6) == "<1 week"
    assert module.matchproxy_duration_bucket(60) == "1 week to 2 months"
    assert module.matchproxy_duration_bucket(365) == "2 months to 1 year"
    assert module.matchproxy_duration_bucket(366) == ">1 year"
    assert module.classify_public_injury_type("Hamstring injury") == "muscle/tendon"
    assert module.classify_public_injury_type("ACL rupture") == "joint/ligament"
    assert module.classify_public_injury_type("Metatarsal fracture") == "bone/fracture"
    assert module.classify_public_injury_type("Head concussion") == "head/concussion"
    assert module.classify_public_injury_type("Covid infection") == "illness/other medical"
    assert module.classify_public_injury_type("") == "unknown"
    assert module.classify_public_injury_type("Unknown injury") == "unknown"
    assert module.classify_public_injury_type("Bruise") == "other/unspecified"
    assert module.split_spell_ids(None) == []
    assert module.split_spell_ids("101; 102.0; 1.5; bad;1:2024-01-01:1") == [
        "101",
        "102",
        "1:2024-01-01:1",
    ]

    injuries = pd.DataFrame(
        {
            "injury_spell_id": [101, 102, 103],
            "start_date": ["2024-01-01", "2024-01-04", "2024-02-01"],
            "end_date": ["2024-01-05", "2024-02-13", "2024-02-03"],
            "durationDetails": ["{'days': 5}", "{'days': 40}", None],
            "injury_desc": ["Hamstring injury", "Knee injury", None],
        }
    )
    lookup = module.injury_spell_metadata_lookup(injuries)
    assert lookup.loc[lookup["injury_spell_id"].eq("102"), "duration_days"].iloc[0] == 40.0
    fallback_duration = lookup.loc[lookup["injury_spell_id"].eq("103"), "duration_days"].iloc[0]
    assert fallback_duration == 2.0
    multi = module.summarize_matchproxy_spell_ids("101;102", lookup)
    assert multi["matchproxy_duration_days"] == 40.0
    assert multi["matchproxy_duration_bucket"] == "1 week to 2 months"
    assert multi["matchproxy_public_injury_type"] in {"muscle/tendon", "joint/ligament"}
    missing = module.summarize_matchproxy_spell_ids("", lookup)
    assert missing["matchproxy_duration_bucket"] == "unknown"
    missing_lookup = module.summarize_matchproxy_spell_ids("999", lookup)
    assert missing_lookup["matchproxy_duration_bucket"] == "unknown"
    minimal_lookup = module.injury_spell_metadata_lookup(
        injuries[["injury_spell_id", "start_date", "durationDetails"]]
    )
    assert minimal_lookup["injury_desc"].tolist() == ["", "", ""]
    episode_lookup = module.injury_spell_metadata_lookup(
        pd.DataFrame(
            {
                "injury_episode_id": ["1:2024-01-01:1"],
                "start_date": ["2024-01-01"],
                "end_date": ["2024-01-30"],
                "duration_days": [10],
                "injury_desc": ["calf strain"],
            }
        )
    )
    assert episode_lookup.loc[0, "duration_days"] == 10
    with pytest.raises(KeyError):
        module.injury_spell_metadata_lookup(injuries.drop(columns=["start_date"]))

    panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 2],
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "injury_spell_id": ["101", "", "102", ""],
            "injury_event_matchproxy": [1, 1, 0, 1],
            "injury_event_matchproxy_same_day": [1, 0, 0, 0],
            "injury_event_matchproxy_lag1": [0, 1, 0, 1],
            "matchproxy_injury_desc": ["", "", "", "calf strain"],
        }
    )
    out = module.add_matchproxy_outcome_subsets(panel, injuries)
    assert out["matchproxy_spell_id"].tolist() == ["101", "102", "", ""]
    assert out["injury_event_matchproxy_ge28d"].tolist() == [0, 1, 0, 0]
    assert out["injury_event_matchproxy_muscle_tendon"].tolist() == [1, 0, 0, 1]
    with pytest.raises(KeyError):
        module.add_matchproxy_outcome_subsets(panel.drop(columns=["injury_spell_id"]), injuries)


def test_epl_club_seasons(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    games = pd.DataFrame(
        {
            "competition_id": ["GB1", "ES1"],
            "season": [2024, 2024],
            "home_club_id": [1, 3],
            "away_club_id": [2, 4],
        }
    )
    out = module.epl_club_seasons(games)
    assert sorted(out["player_club_id"].tolist()) == [1, 2]


def test_prepare_model_frame_and_missing(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            "injury_event_matchproxy": [1, 0, 1],
            "fragility_group": ["regular", "fragile", "tough"],
            "all_minutes_played": [90, 45, 30],
            "all_minutes_last_7d": [0, None, 90],
            "excess_minutes_last7d": [0, None, 0],
            "any_extra_time_last7d": [0, 0, 0],
            "week_phase_sin": [0, 0, 0],
            "week_phase_cos": [0, 0, 0],
            "halfweek_phase_sin": [0, 0, 0],
            "halfweek_phase_cos": [0, 0, 0],
            "days_since_last_match": ["20", "not available", "3"],
        }
    )
    out = module.prepare_model_frame(panel, "injury_event_matchproxy", "fragility_group")
    assert out["model_group"].tolist() == ["regular", "fragile"]
    assert out["all_minutes_last_7d"].tolist() == [0.0, 0.0]
    assert np.isclose(out["log_minutes_played"].iloc[0], np.log(90))
    assert out["days_since_last_match"].isna().tolist() == [False, True]
    assert out["recovery_interval_refined"].tolist() == [">14 days", "no prior match"]
    assert out["clean_zero_or_positive_burden"].tolist() == [False, False]
    out_without_days = module.prepare_model_frame(
        panel.drop(columns=["days_since_last_match"]),
        "injury_event_matchproxy",
        "fragility_group",
    )
    assert "days_since_last_match" not in out_without_days.columns
    assert out_without_days["zero_burden_long_rest"].tolist() == [1, 1]
    with pytest.raises(KeyError):
        module.prepare_model_frame(panel.drop(columns=["all_minutes_played"]), "injury_event_matchproxy", "fragility_group")


def test_prior_history_controls_and_support_rows(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            "model_group": ["regular", "regular", "fragile", "fragile"],
            "all_minutes_last_7d": [0.0, 92.0, 178.0, 222.0],
            "all_minutes_played": [90.0, 90.0, 60.0, 30.0],
            "injury_event_matchproxy": [0, 1, 1, 0],
            "prior_minutes_played": [900, 1000, None, -5],
            "prior_n_spells": [1, 2, 3, 4],
            "prior_injuries_per_10000min": [0, 2, 3, 4],
            "prior_max_spell_duration_days": [0, 7, 28, 56],
        }
    )
    controlled = module.add_prior_history_control_columns(panel)
    assert controlled["log_prior_minutes_played"].iloc[0] == pytest.approx(np.log1p(900))
    assert controlled["log_prior_minutes_played"].iloc[2] == pytest.approx(0.0)
    assert controlled["prior_minutes_played"].iloc[3] == 0
    assert set(module.PRIOR_HISTORY_CONTROL_COLS).issubset(controlled.columns)

    support = module.selected_support_rows(
        panel,
        "injury_event_matchproxy",
        [90.0, 220.0],
        window_half_width=15.0,
    )
    reg_90 = support[
        (support["fragility_group"] == "regular")
        & (support["all_minutes_last_7d"] == 90.0)
    ].iloc[0]
    fragile_220 = support[
        (support["fragility_group"] == "fragile")
        & (support["all_minutes_last_7d"] == 220.0)
    ].iloc[0]
    assert reg_90["support_events"] == 1
    assert reg_90["support_match_rows"] == 1
    assert fragile_220["support_match_minutes"] == 30.0

    diagnostic = module.diagnostic_support_table(
        panel,
        "injury_event_matchproxy",
        [90.0, 220.0],
    )
    assert diagnostic.loc[
        diagnostic["all_minutes_last_7d"].eq(90.0),
        "support_role",
    ].iloc[0] == "selected_prediction_point"
    assert diagnostic.loc[
        diagnostic["all_minutes_last_7d"].eq(220.0),
        "support_role",
    ].iloc[0] == "tail_diagnostic_only"


def test_calendar_and_denominator_helpers(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-26", "2024-04-10", "2020-06-20"]),
        }
    )
    flags = module.add_calendar_sensitivity_flags(
        panel,
        international_windows=[("2024-03-18", "2024-03-26")],
        covid_windows=[("2020-03-13", "2020-09-11")],
    )
    assert flags["prior7_overlaps_international_break"].tolist() == [1, 0, 0]
    assert flags["covid_disrupted_date"].tolist() == [0, 0, 1]
    with pytest.raises(KeyError):
        module.add_calendar_sensitivity_flags(pd.DataFrame({"x": [1]}))

    clean = module.add_clean_comparator_flag(
        pd.DataFrame(
            {
                "all_minutes_last_7d": [0.0, 0.0, 90.0],
                "days_since_last_match": [10.0, 20.0, np.nan],
            }
        )
    )
    assert clean["zero_burden_long_rest"].tolist() == [0, 1, 0]
    assert clean["clean_zero_or_positive_burden"].tolist() == [1, 0, 1]
    no_days = module.add_clean_comparator_flag(
        pd.DataFrame({"all_minutes_last_7d": [0.0, 90.0]})
    )
    assert no_days["zero_burden_long_rest"].tolist() == [1, 0]
    assert no_days["clean_zero_or_positive_burden"].tolist() == [0, 1]
    preflagged = module.add_clean_comparator_flag(
        pd.DataFrame(
            {
                "all_minutes_last_7d": [0.0, 30.0],
                "zero_burden_long_rest": [None, 1],
            }
        )
    )
    assert preflagged["zero_burden_long_rest"].tolist() == [0, 1]
    assert preflagged["clean_zero_or_positive_burden"].tolist() == [1, 1]

    tmp = pd.DataFrame({"log_minutes_played": [np.log(45.0)]})
    assert module.prediction_offset(tmp, "observed_minutes").iloc[0] == pytest.approx(np.log(45.0))
    assert module.prediction_offset(tmp, "fixed_90").iloc[0] == pytest.approx(np.log(90.0))
    assert module.prediction_offset(tmp, "per_match") is None
    with pytest.raises(ValueError):
        module.prediction_offset(tmp, "bad")

    assert module.effect_measure_for_family("poisson") == "incidence_rate_ratio"
    assert module.effect_measure_for_family("binomial_logit") == "odds_ratio"
    assert module.effect_measure_for_family("binomial_cloglog") == "complementary_loglog_ratio"
    assert module.model_family_object("poisson").__class__.__name__ == "Poisson"
    assert module.model_family_object("binomial_logit").__class__.__name__ == "Binomial"
    assert module.model_family_object("binomial_cloglog").__class__.__name__ == "Binomial"
    with pytest.raises(ValueError):
        module.effect_measure_for_family("bad")

    assert "excess_minutes_last7d" not in module.spline_formula("event", 180.0)
    assert "excess_minutes_last7d" in module.spline_formula(
        "event",
        180.0,
        include_exposure_derived_terms=True,
    )
    assert "cr(all_minutes_last_7d" in module.spline_formula(
        "event",
        180.0,
        spline_basis="cr",
        spline_df=3,
    )
    assert "knots=(45.0, 90.0, 135.0)" in module.spline_formula(
        "event",
        180.0,
        spline_df=None,
        spline_knots=(45, 90, 135),
    )
    with pytest.raises(ValueError):
        module.spline_basis_expression(180.0, spline_basis="bad")
    with pytest.raises(ValueError):
        module.spline_basis_expression(180.0, spline_df=None)
    with pytest.raises(ValueError):
        module.model_family_object("bad")


def test_run_named_spline_specification_delegates(load_src_module, monkeypatch):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame({"sentinel": [1]})
    prepared = pd.DataFrame(
        {
            "tm_player_id": [10, 11],
            "model_group": ["regular", "fragile"],
            "injury_event_matchproxy": [1, 0],
        }
    )
    effects_in = pd.DataFrame(
        {
            "contrast_id": ["global_spline_by_history_interaction"],
            "p_value": [0.5],
        }
    )

    def fake_prepare(panel_arg, event_col, group_col):
        assert panel_arg is panel
        assert event_col == "injury_event_matchproxy"
        assert group_col == "custom_group"
        return prepared

    def fake_bundle(
        frame,
        event_col,
        controls="",
        extra_covars=None,
            model_family="poisson",
            denominator="observed_minutes",
            include_exposure_derived_terms=False,
            spline_df=4,
            spline_basis="bs",
            spline_knots=None,
        ):
            assert frame is prepared
            assert controls == " + age_years"
            assert extra_covars == {"age_years": 28.0}
            assert model_family == "binomial_logit"
            assert denominator == "per_match"
            assert include_exposure_derived_terms is True
            assert spline_df == 4
            assert spline_basis == "bs"
            assert spline_knots is None
            return {"effect_modification": effects_in}

    def fake_summary(label, event_col, group_col, controls_label, frame, bundle):
        return {
            "model": label,
            "event_col": event_col,
            "group_col": group_col,
            "controls": controls_label,
            "n_events": int(frame[event_col].sum()),
        }

    def fake_label_effects(
        effects,
        label,
        event_col,
        group_col,
        controls_label,
            frame,
            model_family,
            analysis_role,
            estimator="clustered_glm",
        ):
            assert effects is effects_in
            assert estimator == "clustered_glm"
            return effects.assign(
                model=label,
                event_col=event_col,
            group_col=group_col,
            controls=controls_label,
            model_family=model_family,
            analysis_role=analysis_role,
            n_match_rows=len(frame),
        )

    monkeypatch.setattr(module, "prepare_model_frame", fake_prepare)
    monkeypatch.setattr(module, "run_prediction_bundle", fake_bundle)
    monkeypatch.setattr(module, "summary_row", fake_summary)
    monkeypatch.setattr(module, "label_effect_modification_rows", fake_label_effects)

    bundle, summary, effects = module.run_named_spline_specification(
        panel,
        "delegated",
        "injury_event_matchproxy",
        group_col="custom_group",
        controls=" + age_years",
        controls_label="age_adjusted",
        model_family="binomial_logit",
        denominator="per_match",
        include_exposure_derived_terms=True,
        analysis_role="denominator_sensitivity",
        extra_covars={"age_years": 28.0},
    )

    assert bundle["effect_modification"] is effects_in
    assert summary.to_dict("records") == [
        {
            "model": "delegated",
            "event_col": "injury_event_matchproxy",
            "group_col": "custom_group",
            "controls": "age_adjusted",
            "n_events": 1,
        }
    ]
    assert effects.loc[0, "model_family"] == "binomial_logit"
    assert effects.loc[0, "analysis_role"] == "denominator_sensitivity"


def test_publication_contrast_summary(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    sensitivity = pd.DataFrame(
        {
            "model": ["primary"],
            "model_family": ["poisson"],
            "denominator": ["observed_minutes"],
            "n_match_rows": [10],
            "n_players": [3],
            "n_events": [2],
            "dispersion": [1.1],
            "rr_0": [1.2],
            "rr_0_ci_low": [0.9],
            "rr_0_ci_high": [1.6],
            "rr_90": [1.3],
            "rr_90_ci_low": [1.0],
            "rr_90_ci_high": [1.8],
            "rr_180": [1.4],
            "rr_180_ci_low": [1.1],
            "rr_180_ci_high": [1.9],
            "global_interaction_p": [0.5],
            "difference_in_180_vs_90_change_irr": [1.1],
            "difference_in_180_vs_90_change_ci_low": [0.8],
            "difference_in_180_vs_90_change_ci_high": [1.4],
            "difference_in_180_vs_90_change_p": [0.7],
        }
    )
    denominator = sensitivity.assign(
        model="per_match",
        model_family="binomial_cloglog",
        denominator="per_match",
    )
    out = module.publication_contrast_summary(sensitivity, denominator)
    assert out["analysis_source"].tolist() == [
        "main_and_sensitivity",
        "denominator_and_link",
    ]
    assert out["effect_measure"].tolist() == [
        "incidence_rate_ratio",
        "complementary_loglog_ratio",
    ]
    assert out["rr_180"].tolist() == [1.4, 1.4]
    assert module.publication_contrast_summary(pd.DataFrame()).empty

    adjusted = module.add_p_value_adjustments(pd.DataFrame({"p_value": [0.01, 0.20]}))
    assert adjusted["p_value_holm"].iloc[0] <= adjusted["p_value_holm"].iloc[1]
    all_missing_p = module.add_p_value_adjustments(pd.DataFrame({"p_value": [np.nan]}))
    assert np.isnan(all_missing_p["p_value_holm"].iloc[0])
    unchanged = module.add_p_value_adjustments(pd.DataFrame({"estimate": [1.0]}))
    assert unchanged.columns.tolist() == ["estimate"]

    recurrent = module.recurrent_anchor_comparison_table(
        pd.DataFrame(
            {
                "model": [
                    "primary_same_day_plus_lag1",
                    "recurrent_gee_exchangeable_player",
                    "player_fixed_effect_within_switchers",
                ],
                "rr_0": [1.5, 1.2, 0.7],
                "rr_0_ci_low": [1.0, 0.9, 0.5],
                "rr_0_ci_high": [2.0, 1.6, 0.9],
                "rr_180": [1.6, 1.3, 0.8],
                "rr_180_ci_low": [1.1, 0.8, 0.5],
                "rr_180_ci_high": [2.2, 2.0, 1.2],
            }
        )
    )
    switcher_0 = recurrent[
        recurrent["model"].eq("player_fixed_effect_within_switchers")
        & recurrent["burden_minutes"].eq(0.0)
    ].iloc[0]
    assert switcher_0["estimate_divided_by_primary_glm"] == pytest.approx(0.7 / 1.5)
    primary_only = module.recurrent_anchor_comparison_table(
        pd.DataFrame(
            {
                "model": ["primary_same_day_plus_lag1"],
                "rr_0": [1.5],
                "rr_0_ci_low": [1.0],
                "rr_0_ci_high": [2.0],
                "rr_180": [1.6],
                "rr_180_ci_low": [1.1],
                "rr_180_ci_high": [2.2],
            }
        )
    )
    assert primary_only["model"].tolist() == ["primary_same_day_plus_lag1"] * 2
    with pytest.raises(KeyError):
        module.recurrent_anchor_comparison_table(pd.DataFrame({"model": ["primary"]}))


def test_recovery_interval_rate_table(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    assert module.refined_recovery_interval_from_days(np.nan) == "no prior match"
    assert module.refined_recovery_interval_from_days(3) == "0-3 days"
    assert module.refined_recovery_interval_from_days(5) == "4-5 days"
    assert module.refined_recovery_interval_from_days(7) == "6-7 days"
    assert module.refined_recovery_interval_from_days(14) == "8-14 days"
    assert module.refined_recovery_interval_from_days(15) == ">14 days"

    refined_only = module.add_refined_recovery_interval(
        pd.DataFrame({"recovery_interval_refined": ["no prior match"]})
    )
    assert refined_only["recovery_interval_refined"].tolist() == ["no prior match"]
    with pytest.raises(KeyError):
        module.add_refined_recovery_interval(pd.DataFrame({"x": [1]}))

    match_panel = pd.DataFrame(
        {
            "model_group": ["regular", "regular", "regular", "fragile"],
            "days_since_last_match": [2.0, 4.0, np.nan, 2.0],
            "all_minutes_played": [90.0, 45.0, 30.0, 45.0],
            "injury_event_matchproxy": [1, 0, 0, 1],
        }
    )
    out = module.recovery_interval_rate_table(match_panel)
    reg_short = out[
        (out["history_stratum"] == "regular")
        & (out["recovery_interval_bin"] == "0-3 days")
    ].iloc[0]
    fragile_short = out[
        (out["history_stratum"] == "fragile")
        & (out["recovery_interval_bin"] == "0-3 days")
    ].iloc[0]
    assert reg_short["events"] == 1
    assert reg_short["events_per_1000_match_hours"] == pytest.approx(666.6666667)
    assert reg_short["events_per_1000_match_hours_ci_low"] < reg_short[
        "events_per_1000_match_hours"
    ]
    assert reg_short["events_per_1000_match_hours_ci_high"] > reg_short[
        "events_per_1000_match_hours"
    ]
    assert fragile_short["match_minutes"] == 45.0
    no_prior = out[
        (out["history_stratum"] == "regular")
        & (out["recovery_interval_bin"] == "no prior match")
    ].iloc[0]
    long_gap = out[
        (out["history_stratum"] == "regular")
        & (out["recovery_interval_bin"] == ">14 days")
    ].iloc[0]
    assert no_prior["match_minutes"] == 30.0
    assert np.isnan(long_gap["events_per_10000_min"])

    legacy = module.recovery_interval_rate_table(
        pd.DataFrame(
            {
                "model_group": ["regular"],
                "recovery_interval_bin": [">14 days/no prior match"],
                "all_minutes_played": [90.0],
                "injury_event_matchproxy": [0],
            }
        )
    )
    assert legacy.loc[
        legacy["recovery_interval_bin"].eq(">14 days"),
        "match_minutes",
    ].iloc[0] == 90.0
    with pytest.raises(KeyError):
        module.recovery_interval_rate_table(match_panel.drop(columns=["days_since_last_match"]))
    with pytest.raises(KeyError):
        module.recovery_interval_rate_table(match_panel.drop(columns=["model_group"]))


def test_curve_shape_selection_and_event_support_outputs(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")

    interval = module.count_rate_ratio_interval(2, 100.0, 1, 100.0)
    assert interval["estimate"] == pytest.approx(2.0)
    assert interval["fit_status"] == "ok"
    assert module.count_rate_ratio_interval(0, 100.0, 1, 100.0)["fit_status"] == "not_estimable"
    assert module.count_rate_ratio_interval(1, 0.0, 1, 100.0)["fit_status"] == "not_estimable"

    predictions = pd.DataFrame(
        {
            "fragility_group": ["regular", "regular", "regular", "fragile", "fragile"],
            "all_minutes_last_7d": [0.0, 30.0, 180.0, 0.0, 25.0],
            "pred_events_per_10000_min": [2.0, 3.0, 2.5, 4.0, 5.0],
        }
    )
    shape = module.spline_curve_shape_summary(predictions)
    regular = shape[shape["history_stratum"] == "regular"].iloc[0]
    fragile = shape[shape["history_stratum"] == "fragile"].iloc[0]
    assert regular["max_minutes_last_7d"] == 30.0
    assert regular["max_events_per_1000_match_hours"] == pytest.approx(18.0)
    assert fragile["anchor_180_minutes_last_7d"] == 25.0
    one_group_shape = module.spline_curve_shape_summary(
        predictions[predictions["fragility_group"] == "regular"]
    )
    assert one_group_shape["history_stratum"].tolist() == ["regular"]
    with pytest.raises(KeyError):
        module.spline_curve_shape_summary(predictions.drop(columns=["fragility_group"]))

    shape_rows = pd.DataFrame(
        {
            "specification": ["s1", "s1", "s2"],
            "history_stratum": ["regular", "fragile", "regular"],
            "fit_status": ["ok", "ok", "ok"],
            "anchor_0_events_per_1000_match_hours": [10.0, 15.0, 12.0],
            "anchor_180_events_per_1000_match_hours": [20.0, 30.0, 24.0],
        }
    )
    contrast_shape = module.spline_shape_contrast_sensitivity_table(shape_rows)
    assert contrast_shape.loc[
        contrast_shape["specification"].eq("s1")
        & contrast_shape["burden_minutes"].eq(180.0),
        "higher_vs_intermediate_ratio",
    ].iloc[0] == pytest.approx(1.5)
    assert contrast_shape.loc[
        contrast_shape["specification"].eq("s2"),
        "fit_status",
    ].iloc[0] == "not_estimable"
    with pytest.raises(KeyError):
        module.spline_shape_contrast_sensitivity_table(shape_rows.drop(columns=["fit_status"]))

    injuries = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2, None],
            "start_date": ["2024-01-01", "2024-03-01", "2024-01-05", "2024-01-01"],
            "end_date": ["2024-01-10", None, "2024-01-20", "2024-01-02"],
        }
    )
    spells = module.prepare_prior_spell_end_dates(injuries)
    assert spells["prior_injury_end_date"].max() == pd.Timestamp("2024-03-01")
    no_end = module.prepare_prior_spell_end_dates(
        injuries.drop(columns=["end_date"]).dropna(subset=["tm_player_id"])
    )
    assert no_end["prior_injury_end_date"].min() == pd.Timestamp("2024-01-01")
    with pytest.raises(KeyError):
        module.prepare_prior_spell_end_dates(injuries.drop(columns=["start_date"]))

    match_panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2, 3],
            "date": ["2024-01-18", "2024-02-15", "2024-01-25", "2024-01-05"],
            "fragility_group": ["regular", "regular", "fragile", "tough"],
            "model_group": ["regular", "regular", "fragile", "fragile"],
            "all_minutes_last_7d": [30.0, 92.0, 180.0, 0.0],
            "all_minutes_played": [30.0, 90.0, 55.0, 10.0],
            "injury_event_matchproxy": [1, 0, 1, 0],
            "matchproxy_duration_bucket": [
                "<1 week",
                "unknown",
                "2 months to 1 year",
                "unknown",
            ],
            "matchproxy_duration_days": [5.0, np.nan, 90.0, np.nan],
        }
    )
    flagged = module.add_recent_prior_injury_return_flags(match_panel, injuries)
    assert flagged.loc[flagged["tm_player_id"].eq(1), "returned_from_recorded_injury_within_14d"].iloc[0]
    assert not flagged.loc[flagged["tm_player_id"].eq(3), "returned_from_recorded_injury_within_14d"].iloc[0]
    empty_flags = module.add_recent_prior_injury_return_flags(match_panel.iloc[0:0], injuries)
    assert empty_flags["returned_from_recorded_injury_within_14d"].empty
    with pytest.raises(KeyError):
        module.add_recent_prior_injury_return_flags(match_panel.drop(columns=["date"]), injuries)

    lineups = pd.DataFrame(
        {
            "date": ["2024-01-18", "2024-02-15", "2024-01-25", "2024-01-25"],
            "player_id": [1, 1, 2, 2],
            "type": ["substitutes", "starting_lineup", "starting_lineup", "substitutes"],
        }
    )
    lineup_status = module.lineup_start_status_table(lineups)
    assert set(lineup_status["lineup_role"]) == {
        "substitute_list",
        "starting_lineup",
        "both_start_and_substitute_same_day",
    }
    with pytest.raises(KeyError):
        module.lineup_start_status_table(lineups.drop(columns=["type"]))
    merged_lineups = module.add_lineup_start_status(match_panel, lineups)
    assert merged_lineups.loc[
        merged_lineups["tm_player_id"].eq(2),
        "lineup_role",
    ].iloc[0] == "both_start_and_substitute_same_day"
    unavailable_lineups = module.add_lineup_start_status(match_panel, None)
    assert unavailable_lineups["lineup_role"].eq("lineup_unavailable").all()
    with pytest.raises(KeyError):
        module.add_lineup_start_status(match_panel.drop(columns=["date"]), lineups)

    audit = module.selection_band_audit(match_panel, injuries, lineups=lineups)
    peak = audit[
        (audit["history_stratum"] == "regular")
        & (audit["band"] == "15-45 min peak band")
    ].iloc[0]
    assert peak["events"] == 1
    assert peak["pct_current_appearance_lt45"] == 100.0
    assert peak["pct_returned_from_recorded_injury_within_14d"] == 100.0
    assert peak["pct_substitute_list"] == 100.0
    assert peak["substitute_list_rows"] == 1
    assert peak["substitute_list_denominator_rows"] == 1
    assert peak["pct_substitute_list_ci_low"] < 100.0
    assert peak["pct_substitute_list_interval_method"] == "wilson_95"
    assert peak["pct_short_lt45_and_recent_return"] == 100.0
    all_modelled = audit[
        (audit["history_stratum"] == "all_modelled")
        & (audit["band"] == "180 min band")
    ].iloc[0]
    assert all_modelled["events"] == 1
    empty_band = audit[
        (audit["history_stratum"] == "regular")
        & (audit["band"] == "180 min band")
    ].iloc[0]
    assert empty_band["current_appearance_lt45_denominator_rows"] == 0
    assert np.isnan(empty_band["pct_current_appearance_lt45"])
    with pytest.raises(KeyError):
        module.selection_band_audit(match_panel.drop(columns=["all_minutes_played"]), injuries)

    joint = module.selection_band_joint_proxy_audit(match_panel, injuries, lineups=lineups)
    joint_peak = joint[
        (joint["history_stratum"] == "regular")
        & (joint["band"] == "15-45 min peak band")
        & joint["short_current_appearance_lt45"]
        & joint["recent_recorded_return_14d"]
    ].iloc[0]
    assert joint_peak["match_rows"] == 1
    assert joint_peak["pct_of_band_rows"] == 100.0
    assert joint_peak["lineup_roles"] == "substitute_list"
    with pytest.raises(KeyError):
        module.selection_band_joint_proxy_audit(
            match_panel.drop(columns=["all_minutes_last_7d"]),
            injuries,
        )

    support = module.observed_event_support_summary(match_panel)
    assert support.loc[
        support["history_stratum"].eq("all_modelled"),
        "max_prior7_minutes_with_event",
    ].iloc[0] == 180.0
    assert support.loc[
        support["history_stratum"].eq("all_modelled"),
        "events_gt150_prior7_minutes",
    ].iloc[0] == 1
    assert support.loc[
        support["history_stratum"].eq("fragile"),
        "match_rows_gt150_prior7_minutes",
    ].iloc[0] == 1
    no_event_support = module.observed_event_support_summary(
        match_panel.assign(injury_event_matchproxy=0)
    )
    assert np.isnan(
        no_event_support.loc[
            no_event_support["history_stratum"].eq("regular"),
            "max_prior7_minutes_with_event",
        ].iloc[0]
    )
    with pytest.raises(KeyError):
        module.observed_event_support_summary(match_panel.drop(columns=["all_minutes_last_7d"]))

    severity = module.reporting_process_severity_audit(match_panel)
    severe_row = severity[
        severity["duration_or_severity_proxy"].eq("reported absence >=28 days")
    ].iloc[0]
    assert severe_row["higher_events"] == 1
    assert severe_row["lower_intermediate_events"] == 0
    assert severe_row["fit_status"] == "not_estimable"
    short_row = severity[severity["duration_or_severity_proxy"].eq("<1 week")].iloc[0]
    assert short_row["lower_intermediate_events"] == 1
    empty = module.reporting_process_severity_audit(
        match_panel.assign(fragility_group="low_exposure")
    )
    assert empty.empty
    with pytest.raises(KeyError):
        module.reporting_process_severity_audit(
            match_panel.drop(columns=["matchproxy_duration_days"])
        )


def test_recovery_interval_trend_tests(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    match_panel = pd.DataFrame(
        {
            "tm_player_id": np.repeat(np.arange(1, 9), 2),
            "model_group": ["regular", "fragile"] * 8,
            "recovery_interval_bin": [
                "0-3 days",
                "0-3 days",
                "4-5 days",
                "4-5 days",
                "6-7 days",
                "6-7 days",
                "8-14 days",
                "8-14 days",
                ">14 days/no prior match",
                ">14 days/no prior match",
                "0-3 days",
                "4-5 days",
                "6-7 days",
                "8-14 days",
                ">14 days/no prior match",
                "0-3 days",
            ],
            "all_minutes_played": [90.0] * 16,
            "injury_event_matchproxy": [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        }
    )
    refined = module.add_refined_recovery_interval(match_panel)
    scores = module.recovery_shortness_scores(refined["recovery_interval_refined"])
    assert scores.iloc[0] > scores.iloc[8]
    out = module.recovery_interval_trend_tests(match_panel)
    assert {
        "overall_adjusted_for_history",
        "regular_within_history",
        "fragile_within_history",
        "recovery_shortness_by_history_interaction",
        "recovery_interval_categorical_by_history_interaction",
        "regular_direct_0-3 days_vs_6-7 days",
        "fragile_direct_0-3 days_vs_6-7 days",
    }.issubset(set(out["model"]))
    direct = out[out["effect_measure"] == "incidence_rate_ratio_direct_recovery_interval"]
    assert len(direct) == 2
    assert direct["contrast"].eq("0-3 days_vs_6-7 days").all()
    linear_interaction = out[
        out["model"].eq("recovery_shortness_by_history_interaction")
    ].iloc[0]
    assert linear_interaction["effect_measure"] == "ratio_of_irrs_per_step_shorter_recovery"
    fitted = out[out["fit_status"] == "ok"]
    assert fitted["estimate"].gt(0.0).all()
    assert fitted["p_value"].between(0.0, 1.0).all()

    empty = match_panel.copy()
    empty["injury_event_matchproxy"] = 0
    not_estimable = module.recovery_interval_trend_tests(empty)
    assert set(not_estimable["fit_status"]) == {"not_estimable"}

    no_usable_rows = match_panel.assign(recovery_interval_bin="not a real bin")
    with pytest.raises(ValueError):
        module.recovery_interval_trend_tests(no_usable_rows)

    with pytest.raises(KeyError):
        module.recovery_interval_trend_tests(match_panel.drop(columns=["tm_player_id"]))
    with pytest.raises(KeyError):
        module.recovery_interval_trend_tests(match_panel.drop(columns=["recovery_interval_bin"]))

    summary = module.recovery_interval_publication_summary(
        {
            "same_day_plus_lag1": out,
            "empty_outcome": pd.DataFrame(),
            "muscle_tendon_only": out.assign(p_value=0.053),
            "missing_outcome": None,
            "reported_absence_ge28d": out.assign(p_value=0.72),
        }
    )
    assert set(summary["outcome_label"]) == {
        "same_day_plus_lag1",
        "muscle_tendon_only",
        "reported_absence_ge28d",
    }
    assert "p_value_holm" in summary.columns
    assert "p_value_fdr_bh" in summary.columns
    assert module.recovery_interval_publication_summary({}).empty


def test_recovery_model_preparation_and_formula_helpers(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    match_panel = pd.DataFrame(
        {
            "tm_player_id": [1, 2, 3, 4],
            "model_group": ["regular", "fragile", "regular", "fragile"],
            "recovery_interval_bin": ["0-3 days", "6-7 days", "no prior match", "8-14 days"],
            "all_minutes_played": [90.0, 45.0, 90.0, 0.0],
            "injury_event_matchproxy": [1, 0, 0, 1],
        }
    )
    formula = module.recovery_model_formula("injury_event_matchproxy", controls=" + age")
    assert "C(recovery_interval_refined" in formula
    assert "C(model_group" in formula
    assert formula.endswith(" + age")

    frame = module.prepare_recovery_model_frame(match_panel)
    assert frame["recovery_interval_refined"].tolist() == ["0-3 days", "6-7 days"]
    assert frame["model_group"].cat.categories.tolist() == list(module.MODEL_GROUPS)
    assert frame["recovery_interval_refined"].cat.categories.tolist() == module.RECOVERY_TREND_ORDER
    assert frame["log_minutes_played"].iloc[0] == pytest.approx(np.log(90.0))

    pred = module.recovery_prediction_template(
        ["0-3 days", "6-7 days"],
        "fragile",
        {"age": 31.0},
    )
    assert pred["model_group"].cat.categories.tolist() == list(module.MODEL_GROUPS)
    assert pred["age"].tolist() == [31.0, 31.0]
    assert pred["log_minutes_played"].iloc[0] == pytest.approx(np.log(90.0))
    no_extra = module.recovery_prediction_template(["0-3 days"], "regular")
    assert no_extra["model_group"].astype(str).tolist() == ["regular"]

    with pytest.raises(ValueError):
        module.prepare_recovery_model_frame(
            match_panel.assign(recovery_interval_bin="no prior match")
        )
    with pytest.raises(KeyError):
        module.prepare_recovery_model_frame(match_panel.drop(columns=["tm_player_id"]))


def test_recovery_global_interaction_test(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")

    class FakeTest:
        statistic = np.array(3.0)
        pvalue = np.array(0.22)

    class FakeResult:
        params = pd.Series(
            [0.0, 0.0],
            index=[
                "Intercept",
                "C(recovery_interval_refined, Treatment(reference='6-7 days'))"
                "[T.0-3 days]:C(model_group, Treatment(reference='regular'))[T.fragile]",
            ],
        )

        def wald_test(self, restriction, scalar=True):
            assert scalar is True
            assert restriction.shape == (1, 2)
            return FakeTest()

    out = module.recovery_global_interaction_test(FakeResult())
    assert out["test_statistic"] == 3.0
    assert out["df"] == 1.0
    assert out["p_value"] == 0.22

    class ResultWithoutInteraction:
        params = pd.Series([0.0], index=["Intercept"])

    with pytest.raises(ValueError, match="no recovery-by-history interaction"):
        module.recovery_global_interaction_test(ResultWithoutInteraction())


def test_recovery_interaction_tests_without_named_interactions(load_src_module, monkeypatch):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    frame = pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "model_group": ["regular", "fragile"],
            "recovery_interval_refined": ["0-3 days", "6-7 days"],
            "recovery_shortness_score": [4.0, 2.0],
            "all_minutes_played": [90.0, 90.0],
            "log_minutes": [np.log(90.0), np.log(90.0)],
            "injury_event_matchproxy": [1, 1],
        }
    )

    class FakeResult:
        params = pd.Series([0.0], index=["Intercept"])
        bse = pd.Series([1.0], index=["Intercept"])

    class FakeModel:
        def fit(self, *args, **kwargs):
            return FakeResult()

    monkeypatch.setattr(module.smf, "glm", lambda *args, **kwargs: FakeModel())
    rows = module.recovery_interaction_test_rows(frame, "injury_event_matchproxy")
    assert [row["model"] for row in rows] == [
        "recovery_shortness_by_history_interaction",
        "recovery_interval_categorical_by_history_interaction",
    ]
    assert {row["fit_status"] for row in rows} == {"not_estimable"}


def test_same_day_denominator_audit(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    match_panel = pd.DataFrame(
        {
            "model_group": ["regular", "regular", "fragile", "fragile"],
            "all_minutes_last_7d": [0.0, 100.0, 100.0, 220.0],
            "all_minutes_played": [90.0, 30.0, 75.0, 15.0],
            "injury_event_matchproxy": [0, 1, 1, 0],
            "injury_event_matchproxy_same_day": [0, 1, 0, 0],
            "injury_event_matchproxy_lag1": [0, 0, 1, 0],
        }
    )
    audit = module.same_day_denominator_audit(match_panel)
    same_day_regular = audit[
        (audit["history_stratum"] == "regular")
        & (audit["prior_load_band"] == "91-180 min")
        & (audit["row_type"] == "same_day_proxy_event")
    ].iloc[0]
    lag1_fragile = audit[
        (audit["history_stratum"] == "fragile")
        & (audit["prior_load_band"] == "91-180 min")
        & (audit["row_type"] == "lag1_proxy_event")
    ].iloc[0]
    all_modelled_tail = audit[
        (audit["history_stratum"] == "all_modelled")
        & (audit["prior_load_band"] == ">180 min")
        & (audit["row_type"] == "all_match_rows")
    ].iloc[0]
    empty_same_day_tail = audit[
        (audit["history_stratum"] == "regular")
        & (audit["prior_load_band"] == ">180 min")
        & (audit["row_type"] == "same_day_proxy_event")
    ].iloc[0]
    assert same_day_regular["match_rows"] == 1
    assert same_day_regular["mean_minutes"] == 30.0
    assert lag1_fragile["lag1_events"] == 1
    assert lag1_fragile["median_minutes"] == 75.0
    assert all_modelled_tail["match_rows"] == 1
    assert np.isnan(empty_same_day_tail["mean_minutes"])
    with pytest.raises(KeyError):
        module.same_day_denominator_audit(
            match_panel.drop(columns=["injury_event_matchproxy_same_day"])
        )


def test_switcher_transition_audit(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    match_panel = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 2, 2, 3],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-01-01",
                ]
            ),
            "model_group": [
                "regular",
                "regular",
                "fragile",
                "regular",
                "regular",
                "fragile",
            ],
            "all_minutes_played": [90.0, 90.0, 45.0, 90.0, 90.0, 30.0],
            "injury_event_matchproxy": [0, 1, 0, 0, 0, 1],
        }
    )
    audit = module.switcher_transition_audit(match_panel)
    states = set(audit["transition_state"].dropna())
    assert {
        "switcher_pre_higher_history",
        "switcher_post_higher_history",
        "non_switcher_intermediate_history",
        "non_switcher_higher_history",
        "switcher_players",
    }.issubset(states)
    pre = audit[audit["transition_state"].eq("switcher_pre_higher_history")].iloc[0]
    assert pre["n_events"] == 1
    player_summary = audit[audit["transition_state"].eq("switcher_players")].iloc[0]
    assert player_summary["n_players_with_pre_higher_event"] == 1

    no_switcher = module.switcher_transition_audit(
        match_panel[match_panel["tm_player_id"].isin([2, 3])]
    )
    assert "switcher_players" not in set(no_switcher["transition_state"])
    with pytest.raises(KeyError):
        module.switcher_transition_audit(match_panel.drop(columns=["date"]))


def test_temporal_period_labels_carry_existing_fragility(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            "date": ["2016-05-01", "2018-09-01", "2021-01-10", "2023-01-10"],
            "fragility_group": ["regular", "fragile", "fragile", "regular"],
        }
    )
    out = module.add_temporal_period_labels(panel)
    assert pd.isna(out["temporal_period"].iloc[0])
    assert out["temporal_period"].tolist()[1:] == [
        "2017-2019",
        "2020-2021",
        "2022-2024",
    ]
    assert out["fragility_group"].tolist() == panel["fragility_group"].tolist()

    custom = module.add_temporal_period_labels(
        panel,
        periods=[
            {
                "period": "custom_recent",
                "season_start_min": 2022,
                "season_start_max": 2024,
            }
        ],
    )
    assert pd.isna(custom["temporal_period"].iloc[0])
    assert pd.isna(custom["temporal_period"].iloc[1])
    assert pd.isna(custom["temporal_period"].iloc[2])
    assert custom["temporal_period"].iloc[3] == "custom_recent"

    with pytest.raises(KeyError):
        module.add_temporal_period_labels(pd.DataFrame({"fragility_group": ["regular"]}))


def test_delta_ratio_interval_and_selected_rows(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    interval = module.delta_ratio_interval(
        np.array([0.0, np.log(2.0)]),
        np.eye(2) * 0.01,
        np.array([1.0, 1.0]),
        np.array([1.0, 0.0]),
    )
    assert interval["rate_ratio"] == pytest.approx(2.0)
    assert interval["rr_ci_low"] < 2.0 < interval["rr_ci_high"]
    assert 0.0 < interval["p_value"] < 1.0

    zero_null = module.delta_ratio_interval(
        np.zeros(1),
        np.zeros((1, 1)),
        np.ones(1),
        np.zeros(1),
    )
    assert zero_null["z_statistic"] == 0.0
    assert zero_null["p_value"] == 1.0

    zero_nonnull = module.delta_ratio_interval(
        np.array([np.log(2.0)]),
        np.zeros((1, 1)),
        np.ones(1),
        np.zeros(1),
        alpha=0.10,
    )
    assert np.isinf(zero_nonnull["z_statistic"])
    assert zero_nonnull["p_value"] == 0.0
    assert zero_nonnull["rr_ci_low"] == pytest.approx(2.0)

    preds = pd.DataFrame({"all_minutes_last_7d": [0.0, 5.0, 90.0]})
    selected = module.selected_prediction_rows(preds, [0, 90])
    assert selected["all_minutes_last_7d"].tolist() == [0.0, 90.0]


def test_formal_effect_modification_and_multiplicity(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    rng = np.random.default_rng(20260722)
    n = 720
    burden = np.tile(np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0]), n // 6)
    group = np.where(np.arange(n) % 2 == 0, "regular", "fragile")
    probability = (
        0.080
        + 0.00008 * burden
        + 0.035 * (group == "fragile")
        + 0.020 * ((group == "fragile") & (burden >= 135.0))
    )
    frame = pd.DataFrame(
        {
            "tm_player_id": np.repeat(np.arange(n // 6), 6),
            "model_group": group,
            "all_minutes_last_7d": burden,
            "all_minutes_played": 90.0,
            "log_minutes_played": np.log(90.0),
            "injury_event_matchproxy": rng.binomial(1, probability),
            "excess_minutes_last7d": np.maximum(burden - 180.0, 0.0),
            "any_extra_time_last7d": rng.binomial(1, 0.1, n),
            "week_phase_sin": rng.normal(size=n),
            "week_phase_cos": rng.normal(size=n),
            "halfweek_phase_sin": rng.normal(size=n),
            "halfweek_phase_cos": rng.normal(size=n),
        }
    )
    result = module.fit_model(frame, "injury_event_matchproxy")
    effects = module.effect_modification_rows(result)
    assert effects["contrast_id"].tolist() == [
        "global_spline_by_history_interaction",
        "higher_vs_intermediate_at_0",
        "higher_vs_intermediate_at_180",
        "intermediate_history_180_vs_0",
        "higher_history_180_vs_0",
        "ratio_of_180_vs_0_changes",
        "intermediate_history_180_vs_90",
        "higher_history_180_vs_90",
        "ratio_of_180_vs_90_changes",
    ]
    interaction = effects.iloc[0]
    assert interaction["df"] == module.SPLINE_DF
    assert 0.0 <= interaction["p_value"] <= 1.0
    assert effects.iloc[1:]["estimate"].gt(0.0).all()
    between = effects[effects["contrast_id"].str.startswith("higher_vs_intermediate_at_")]
    assert between["burden_to"].tolist() == [0.0, 180.0]

    labelled = module.label_effect_modification_rows(
        effects,
        "test_specification",
        "injury_event_matchproxy",
        "fragility_group",
        "load_timing_only",
        frame,
        "poisson",
        "test",
    )
    assert labelled["model"].unique().tolist() == ["test_specification"]
    assert labelled["n_match_rows"].unique().tolist() == [n]
    assert labelled["n_players"].unique().tolist() == [n // 6]
    assert labelled["n_events"].unique().tolist() == [
        int(frame["injury_event_matchproxy"].sum())
    ]

    duplicate = labelled.copy()
    duplicate["model"] = "second_specification"
    duplicate.loc[duplicate.index[0], "p_value"] = np.nan
    adjusted = module.add_specification_multiplicity_adjustments(
        pd.concat([labelled, duplicate], ignore_index=True)
    )
    valid = adjusted["p_value"].notna()
    assert (
        adjusted.loc[valid, "p_holm_across_specifications"]
        >= adjusted.loc[valid, "p_value"]
    ).all()
    assert adjusted.loc[valid, "p_bh_across_specifications"].between(0.0, 1.0).all()
    assert adjusted.loc[~valid, "p_holm_across_specifications"].isna().all()
    assert not adjusted.loc[~valid, "reject_holm_0_05"].any()


def test_global_interaction_requires_interaction_terms(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")

    class ResultWithoutInteraction:
        params = pd.Series([0.0], index=["Intercept"])

    with pytest.raises(ValueError, match="no spline-by-history interaction"):
        module.global_spline_interaction_test(ResultWithoutInteraction())


def test_prediction_template(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    out = module.prediction_template([0, 90], "fragile", {"age_years": 31.0})
    assert out["model_group"].tolist() == ["fragile", "fragile"]
    assert out["age_years"].tolist() == [31.0, 31.0]
    assert "log_minutes_played" in out.columns

    default = module.prediction_template([0], "regular")
    assert default["age_years"].iloc[0] == 28.0


def test_add_player_and_club_metadata(load_src_module, tmp_path):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    tm_dir = tmp_path
    pd.DataFrame(
        {
            "player_id": [1, 2],
            "date_of_birth": ["2000-01-01", None],
            "position": ["Attack", None],
        }
    ).to_csv(tm_dir / "players.csv", index=False)
    pd.DataFrame(
        {
            "game_id": [10, 11],
            "competition_id": ["GB1", "FAC"],
            "season": [2024, 2024],
            "home_club_id": [100, 100],
            "away_club_id": [200, 300],
        }
    ).to_csv(tm_dir / "games.csv", index=False)
    pd.DataFrame(
        {
            "game_id": [11],
            "player_id": [1],
            "player_club_id": [100],
            "date": ["2024-08-01"],
            "minutes_played": [90],
        }
    ).to_csv(tm_dir / "appearances.csv", index=False)
    match_panel = pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "date": ["2024-08-01", "2024-06-01"],
        }
    )
    out = module.add_player_and_club_metadata(match_panel, tm_dir)
    assert out.loc[0, "position_group"] == "Attack"
    assert out.loc[0, "club_season"] == "2024_100"
    assert out.loc[1, "position_group"] == "Unknown"
    assert out.loc[1, "club_season"] == "2023_-1"
    assert out.loc[1, "age_years"] >= 16


def test_spline_anchor_range_summary(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    contrasts = pd.DataFrame(
        {
            "specification": ["a", "a", "b", "b", "failed"],
            "burden_minutes": [0.0, 180.0, 0.0, 180.0, 0.0],
            "higher_vs_intermediate_ratio": [1.6, 1.5, 1.7, 1.4, 9.0],
            "fit_status": ["ok", "ok", "ok", "ok", "failed"],
        }
    )
    out = module.spline_anchor_range_summary(contrasts).iloc[0]
    assert out["fit_status"] == "ok"
    assert out["n_complete_specifications"] == 2
    assert out["rr_0_min"] == 1.6
    assert out["rr_180_max"] == 1.5
    assert out["n_specifications_rr_0_ge_rr_180"] == 2
    assert bool(out["all_specifications_rr_0_ge_rr_180"])

    unsupported = module.spline_anchor_range_summary(
        contrasts[contrasts["fit_status"].eq("failed")]
    ).iloc[0]
    assert unsupported["fit_status"] == "not_estimable"
    assert unsupported["n_complete_specifications"] == 0
    with pytest.raises(KeyError):
        module.spline_anchor_range_summary(contrasts.drop(columns=["fit_status"]))


def test_publication_crude_daily_and_proxy_classification_tables(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            "fragility_group": ["tough", "regular", "fragile", "fragile"],
            "all_minutes_last_7d": [0.0, 90.0, 180.0, np.nan],
            "injury_event": [1, 0, 1, 1],
            "injury_context": [
                "match_same_day",
                "none",
                "match_lag1_recorded_next_day",
                "training_or_other",
            ],
            "injury_event_matchproxy": [1, 0, 1, 0],
        }
    )
    crude = module.crude_daily_history_publication_table(panel)
    lower_zero = crude[
        crude["history_stratum"].eq("lower prior-injury-history")
        & crude["all_minutes7d_bin"].astype(str).eq("0-45")
    ].iloc[0]
    assert lower_zero["n_player_days"] == 1
    assert lower_zero["n_reported_starts"] == 1
    assert lower_zero["daily_incidence_percent"] == 100.0
    assert crude["daily_incidence"].isna().any()

    proxy = module.proxy_classification_publication_table(panel).set_index("metric")
    assert proxy.loc["all_reported_daily_starts", "n_events"] == 3
    assert proxy.loc["same_day_plus_lag1_candidates", "n_events"] == 2
    assert proxy.loc["assigned_matchproxy_events", "n_events"] == 2
    assert proxy.loc["unassigned_same_day_or_lag1", "n_events"] == 0

    typed = panel.assign(
        matchproxy_public_injury_type=[
            "illness/other medical",
            "muscle/tendon",
            "illness/other medical",
            "joint/ligament",
        ],
        matchproxy_lookup_desc=[
            "Corona virus",
            "Hamstring injury",
            "Flu",
            "Knee injury",
        ],
        date=pd.to_datetime(["2020-12-01", "2024-01-01", "2021-02-01", "2021-01-01"]),
    )
    typed_summary = module.proxy_event_type_summary(typed).set_index("metric")
    assert typed_summary.loc["all_proxy_events", "n_proxy_events"] == 2
    assert typed_summary.loc[
        "illness_or_other_medical_proxy_events",
        "n_proxy_events",
    ] == 2
    assert typed_summary.loc[
        "illness_or_other_medical_proxy_events",
        "n_covid_like_2020_2021_season",
    ] == 1
    assert typed_summary.loc[
        "unknown_or_unspecified_proxy_events",
        "n_proxy_events",
    ] == 0
    with pytest.raises(KeyError):
        module.proxy_event_type_summary(typed.drop(columns=["matchproxy_public_injury_type"]))
    minimal_typed = pd.DataFrame(
        {
            "injury_event_matchproxy": [1, 0],
            "matchproxy_public_injury_type": ["muscle/tendon", "joint/ligament"],
        }
    )
    minimal_summary = module.proxy_event_type_summary(minimal_typed).set_index("metric")
    assert minimal_summary.loc["all_proxy_events", "n_covid_like_descriptions"] == 0
    assert minimal_summary.loc[
        "unknown_or_unspecified_proxy_events",
        "n_proxy_events",
    ] == 0
    empty_typed = minimal_typed.assign(injury_event_matchproxy=0)
    empty_summary = module.proxy_event_type_summary(empty_typed)
    assert empty_summary["n_proxy_events"].tolist() == [0]
    assert empty_summary["percent_of_proxy_events"].isna().all()

    no_starts = panel.assign(
        injury_event=0,
        injury_context="none",
        injury_event_matchproxy=0,
    )
    empty_proxy = module.proxy_classification_publication_table(no_starts)
    assert empty_proxy["percent_of_reported_daily_starts"].isna().all()

    with pytest.raises(ValueError, match="exceed"):
        module.proxy_classification_publication_table(
            no_starts.assign(injury_event_matchproxy=[1, 0, 0, 0])
        )
    with pytest.raises(KeyError):
        module.crude_daily_history_publication_table(
            panel.drop(columns=["all_minutes_last_7d"])
        )
    with pytest.raises(KeyError):
        module.proxy_classification_publication_table(
            panel.drop(columns=["injury_context"])
        )


def test_multiplicity_family_summary(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    effects = pd.DataFrame(
        {
            "analysis_role": [
                "primary",
                "primary",
                "outcome_history_cross_sensitivity",
                "unmapped",
            ],
            "contrast_id": [
                "higher_vs_intermediate_at_180",
                "global_spline_by_history_interaction",
                "ratio_of_180_vs_90_changes",
                "higher_vs_intermediate_at_180",
            ],
            "p_value": [0.01, 0.40, 0.20, 0.001],
            "p_holm_across_specifications": [0.08, 0.80, 0.40, 0.01],
            "p_bh_across_specifications": [0.06, 0.60, 0.30, 0.01],
            "reject_holm_0_05": [False, False, False, True],
            "reject_bh_0_05": [False, False, False, True],
        }
    )
    out = module.multiplicity_family_summary(effects)
    assert out["test_family"].tolist() == ["primary", "outcome definitions"]
    assert out.loc[0, "n_tests"] == 2
    assert out.loc[0, "n_history_level_tests"] == 1
    assert out.loc[0, "n_exposure_response_tests"] == 1
    assert out.loc[0, "minimum_holm_p"] == 0.08
    assert out.loc[0, "holm_rejections_0_05"] == 0
    assert out.loc[0, "history_level_holm_rejections_0_05"] == 0
    no_contrast = module.multiplicity_family_summary(effects.drop(columns=["contrast_id"]))
    assert no_contrast.loc[0, "n_history_level_tests"] == 0
    assert no_contrast.loc[0, "n_exposure_response_tests"] == 2
    assert module.multiplicity_family_summary(
        effects.assign(analysis_role="unmapped")
    ).empty
    with pytest.raises(KeyError):
        module.multiplicity_family_summary(effects.drop(columns=["p_value"]))


def test_nominal_exposure_response_signal_summary(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    common = {
        "analysis_role": ["primary"] * 4,
        "estimator": ["clustered_glm"] * 4,
        "event_col": ["injury_event_matchproxy"] * 4,
        "group_col": ["model_group"] * 4,
        "history_stratum": [
            "joint",
            "higher_history",
            "higher_vs_intermediate",
            "higher_vs_intermediate",
        ],
        "burden_from": [np.nan, 90.0, 90.0, np.nan],
        "burden_to": [np.nan, 180.0, 180.0, 180.0],
        "effect_measure": ["incidence_rate_ratio"] * 4,
        "estimate": [np.nan, 1.66, 1.19, 1.54],
        "ci_low": [np.nan, 1.13, 0.80, 1.10],
        "ci_high": [np.nan, 2.44, 1.75, 2.15],
        "p_value": [0.048, 0.010, 0.394, 0.001],
        "p_holm_across_specifications": [1.0, 0.216, 1.0, 0.01],
        "p_bh_across_specifications": [1.0, 0.150, 1.0, 0.01],
        "reject_holm_0_05": [False, False, False, True],
        "reject_bh_0_05": [False, False, False, True],
        "n_match_rows": [100] * 4,
        "n_players": [20] * 4,
        "n_events": [5] * 4,
    }
    effects = pd.DataFrame(
        {
            **common,
            "model": ["model_b", "model_a", "model_c", "model_d"],
            "contrast_id": [
                "global_spline_by_history_interaction",
                "higher_history_180_vs_90",
                "ratio_of_180_vs_90_changes",
                "higher_vs_intermediate_at_180",
            ],
        }
    )
    out = module.nominal_exposure_response_signal_summary(effects)
    assert out["contrast_id"].tolist() == [
        "higher_history_180_vs_90",
        "global_spline_by_history_interaction",
    ]
    assert out["nominal_signal_rank"].tolist() == [1, 2]
    assert out["p_holm_across_specifications"].tolist() == [0.216, 1.0]
    assert out["interpretation_note"].str.contains("unadjusted").all()

    empty = module.nominal_exposure_response_signal_summary(
        effects.assign(p_value=[0.20, 0.10, 0.394, 0.001])
    )
    assert empty.empty
    assert "nominal_signal_rank" in empty.columns
    with pytest.raises(KeyError):
        module.nominal_exposure_response_signal_summary(effects.drop(columns=["model"]))


def test_cross_summary_and_referee_publication_audit(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    cross = pd.DataFrame(
        {
            "model": [
                "muscle_tendon_only_non_muscle_frequency_history",
                "muscle_tendon_only_frequency_only_history",
            ],
            "rr_180": [0.9, 2.3],
            "rr_180_ci_low": [0.4, 1.3],
            "rr_180_ci_high": [1.6, 4.0],
            "rr_0": [0.9, 2.3],
            "rr_0_ci_low": [0.5, 1.4],
            "rr_0_ci_high": [1.5, 3.7],
            "n_match_rows": [4, 5],
            "n_events": [2, 3],
            "frequency_threshold_per_10000_prior_minutes": [3.0, 10.0],
            "lower_frequency_0_support_events": [11, 21],
            "lower_frequency_0_support_rows": [110, 210],
            "lower_frequency_180_support_events": [12, 22],
            "lower_frequency_180_support_rows": [120, 220],
            "higher_frequency_0_support_events": [13, 23],
            "higher_frequency_0_support_rows": [130, 230],
            "higher_frequency_180_support_events": [14, 24],
            "higher_frequency_180_support_rows": [140, 240],
        }
    )
    effects = pd.DataFrame(
        {
            "model": [
                "muscle_tendon_only_frequency_only_history",
                "muscle_tendon_only_non_muscle_frequency_history",
                "muscle_tendon_only_non_muscle_frequency_history",
                "muscle_tendon_only_non_muscle_frequency_history",
                "muscle_tendon_only_non_muscle_frequency_history",
                "muscle_tendon_only_non_muscle_frequency_history",
                "reported_absence_ge28d_frequency_only_history",
                "primary_same_day_plus_lag1",
            ],
            "contrast_id": [
                "higher_vs_intermediate_at_180",
                "higher_vs_intermediate_at_180",
                "global_spline_by_history_interaction",
                "intermediate_history_180_vs_90",
                "higher_history_180_vs_90",
                "ratio_of_180_vs_90_changes",
                "higher_vs_intermediate_at_180",
                "higher_vs_intermediate_at_180",
            ],
            "estimate": [2.3, 0.9, np.nan, 1.4, 0.8, 0.57, 1.8, 1.5],
            "ci_low": [1.3, 0.4, np.nan, 1.1, 0.4, 0.3, 0.9, 1.1],
            "ci_high": [4.0, 1.6, np.nan, 1.8, 1.5, 1.1, 3.6, 2.1],
            "p_value": [0.002, 0.66, 0.048, 0.013, 0.44, 0.097, 0.086, 0.018],
            "p_holm_across_specifications": [0.047, 1.0, 0.216, 0.216, 1.0, 1.0, 0.6, 0.229],
            "p_bh_across_specifications": [0.014, 0.66, 0.216, 0.216, 0.7, 0.7, 0.12, 0.05],
            "reject_holm_0_05": [True, False, False, False, False, False, False, False],
            "reject_bh_0_05": [True, False, False, False, False, False, False, False],
            "n_match_rows": [4] * 8,
            "n_events": [2] * 8,
        }
    )
    enriched = module.add_cross_summary_multiplicity_columns(cross, effects)
    assert enriched.loc[0, "rr_180_p_holm_across_specifications"] == 1.0
    assert module.add_cross_summary_multiplicity_columns(pd.DataFrame(), effects).empty
    with pytest.raises(KeyError):
        module.add_cross_summary_multiplicity_columns(cross, effects.drop(columns=["p_value"]))

    comparison = module.negative_control_magnitude_comparison(enriched)
    assert comparison["anchor_minutes"].tolist() == [0, 180]
    baseline = comparison.set_index("anchor_minutes").loc[0]
    assert baseline["all_type_history_rr"] == 2.3
    assert baseline["negative_control_rr"] == 0.9
    assert baseline["all_type_higher_support_events"] == 23
    assert baseline["negative_control_higher_support_events"] == 13
    assert "all_type_to_negative_control_ratio" not in comparison.columns
    assert "negative_control_percent_of_all_type" not in comparison.columns
    assert "well-supported baseline" in baseline["interpretation_note"]
    sparse = comparison.set_index("anchor_minutes").loc[180]
    assert "sparse higher-history support" in sparse["interpretation_note"]
    assert module.negative_control_magnitude_comparison(pd.DataFrame()).empty
    assert module.negative_control_magnitude_comparison(
        enriched[enriched["model"].eq("missing_model")]
    ).empty
    assert module.negative_control_magnitude_comparison(
        enriched[
            enriched["model"].eq("muscle_tendon_only_frequency_only_history")
        ]
    ).empty
    with pytest.raises(KeyError):
        module.negative_control_magnitude_comparison(enriched.drop(columns=["rr_0"]))

    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 1, 2, 2],
            "all_minutes_played": [90.0, 45.0, 90.0, 30.0],
            "all_minutes_last_7d": [0.0, 180.0, 90.0, 180.0],
            "injury_event_matchproxy_muscle_tendon": [0, 1, 0, 1],
            module.NON_MUSCLE_HISTORY_GROUP_COL: [
                "regular",
                "fragile",
                "regular",
                "fragile",
            ],
            "prior_non_muscle_n_spells": [0.0, 1.0, 0.0, 2.0],
            module.NON_MUSCLE_HISTORY_THRESHOLD_COL: [3.0, 3.0, 3.0, 3.0],
        }
    )
    injuries = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2, 3],
            "injury_spell_id": [10, 11, 12],
            "start_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "end_date": pd.to_datetime(["2020-01-08", "2020-01-03", "2020-02-03"]),
            "durationDetails": ["7 days", "1 days", "31 days"],
            "injury_desc": ["Hamstring injury", "ACL rupture", "Metatarsal fracture"],
        }
    )
    audit = module.publication_referee_audit_table(
        panel,
        injuries,
        effects,
        enriched,
    ).set_index("metric")
    assert audit.loc["muscle_tendon_keyword_spells", "n"] == 1
    assert audit.loc["strict_type_discordant_model_rows", "n_events"] == 2
    assert audit.loc["strict_prior_non_muscle_nonzero_rows", "n"] == 2
    assert audit.loc["strict_high_non_muscle_history_rows", "n"] == 2
    assert bool(audit.loc["muscle_tendon_frequency_only_rr_180", "reject_holm_0_05"])
    assert audit.loc["strict_low_non_muscle_history_180_vs_90", "estimate"] == 1.4
    assert (
        audit.loc[
            "muscle_tendon_all_type_frequency_history_higher_180_minute_local_support",
            "support_events",
        ]
        == 24
    )
    assert (
        audit.loc[
            "muscle_tendon_strict_type_discordant_history_lower_0_minute_local_support",
            "support_rows",
        ]
        == 110
    )
    without_cross = module.publication_referee_audit_table(
        panel,
        injuries,
        effects,
        pd.DataFrame(),
    )
    assert "strict_type_discordant_cross_summary" not in without_cross["metric"].tolist()
    partial = module.publication_referee_audit_table(
        panel,
        injuries,
        effects.iloc[[0]].copy(),
        pd.DataFrame({"model": ["not_the_strict_model"]}),
    )
    assert "strict_type_discordant_rr_180" not in partial["metric"].tolist()
    assert "strict_type_discordant_cross_summary" not in partial["metric"].tolist()


def test_negative_control_recent_return_outputs(load_src_module, monkeypatch):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2, 3, 4],
            "date": pd.to_datetime(
                ["2020-01-10", "2020-01-20", "2020-01-11", "2020-01-21"]
            ),
            module.MATCH_MINUTES_COL: [30.0, 90.0, 60.0, 90.0],
            "all_minutes_last_7d": [0.0, 0.0, 180.0, 180.0],
            "injury_event_matchproxy_muscle_tendon": [0, 1, 1, 0],
            "fragility_frequency_only": [
                "regular",
                "fragile",
                "regular",
                "fragile",
            ],
            module.NON_MUSCLE_HISTORY_GROUP_COL: [
                "regular",
                "fragile",
                "regular",
                "fragile",
            ],
        }
    )
    injuries = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2, 3, 4],
            "start_date": pd.to_datetime(
                ["2019-12-01", "2019-11-01", "2020-01-01", "2019-10-01"]
            ),
            "end_date": pd.to_datetime(
                ["2020-01-01", "2019-12-01", "2020-01-05", "2019-11-01"]
            ),
        }
    )

    audit = module.negative_control_anchor_selection_audit(
        panel,
        injuries,
    )
    assert len(audit) == 4
    all_type_lower = audit[
        audit["comparison"].eq("all_type_frequency_history")
        & audit["history_stratum"].eq("lower frequency")
    ].iloc[0]
    assert all_type_lower["recent_return_percent"] == 100.0
    assert all_type_lower["short_appearance_percent"] == 100.0
    all_type_higher = audit[
        audit["comparison"].eq("all_type_frequency_history")
        & audit["history_stratum"].eq("higher frequency")
    ].iloc[0]
    assert all_type_higher["recent_return_percent"] == 0.0
    assert all_type_higher["short_appearance_percent"] == 0.0

    def fake_bundle(frame, event_col):
        return {"frame": frame, "event_col": event_col}

    def fake_summary(label, event_col, group_col, controls, frame, bundle):
        rr = (
            2.0
            if label == "muscle_tendon_only_frequency_only_history"
            else 1.1
        )
        return {
            "model": label,
            "event_col": event_col,
            "group_col": group_col,
            "controls": controls,
            "n_match_rows": len(frame),
            "n_events": int(frame[event_col].sum()),
            "rr_0": rr,
            "rr_0_ci_low": rr - 0.2,
            "rr_0_ci_high": rr + 0.2,
            "rr_180": rr + 0.1,
            "rr_180_ci_low": rr - 0.1,
            "rr_180_ci_high": rr + 0.3,
            "regular_0_support_events": 1,
            "regular_0_support_rows": 10,
            "fragile_0_support_events": 2,
            "fragile_0_support_rows": 20,
            "regular_180_support_events": 3,
            "regular_180_support_rows": 30,
            "fragile_180_support_events": 4,
            "fragile_180_support_rows": 40,
        }

    monkeypatch.setattr(module, "run_prediction_bundle", fake_bundle)
    monkeypatch.setattr(module, "summary_row", fake_summary)
    restricted = module.recent_return_excluded_negative_control_cross_summary(
        panel,
        injuries,
    )
    assert len(restricted) == 2
    assert restricted["excluded_recent_return_rows"].tolist() == [2, 2]
    assert restricted["n_match_rows"].tolist() == [2, 2]
    comparison = module.negative_control_magnitude_comparison(restricted)
    baseline = comparison.set_index("anchor_minutes").loc[0]
    assert baseline["all_type_history_rr"] == 2.0
    assert baseline["negative_control_rr"] == 1.1


def test_negative_control_joint_label_frame_filters_recent_returns(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2],
            "date": pd.to_datetime(["2020-01-10", "2020-01-20"]),
            module.MATCH_MINUTES_COL: [30.0, 90.0],
            "all_minutes_last_7d": [0.0, 0.0],
            "injury_event_matchproxy_muscle_tendon": [0, 1],
            "fragility_frequency_only": ["regular", "fragile"],
            module.NON_MUSCLE_HISTORY_GROUP_COL: ["fragile", "regular"],
        }
    )
    injuries = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2],
            "start_date": pd.to_datetime(["2019-12-01", "2019-10-01"]),
            "end_date": pd.to_datetime(["2020-01-01", "2019-11-01"]),
        }
    )

    unrestricted = module.negative_control_joint_label_frame(
        panel,
        injuries,
        exclude_recent_returns=False,
    )
    restricted = module.negative_control_joint_label_frame(
        panel,
        injuries,
        exclude_recent_returns=True,
    )
    assert len(unrestricted) == 2
    assert unrestricted["all_type_high_history"].tolist() == [0, 1]
    assert unrestricted["type_discordant_high_history"].tolist() == [1, 0]
    assert len(restricted) == 1
    assert restricted[module.PLAYER_ID_COL].tolist() == [2]


def test_classified_injury_type_day_counts(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    injuries = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 1, 1, 2, 3],
            "injury_spell_id": [10, 11, 12, 20, 30],
            "start_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-03", "2020-01-02", None]
            ),
            "injury_desc": [
                "Hamstring injury",
                "Calf strain",
                "ACL rupture",
                "Toe fracture",
                "Muscle injury",
            ],
        }
    )
    out = module.classified_injury_type_day_counts(
        injuries,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-03"),
        ("muscle/tendon",),
        "n_muscle_today",
    )
    assert out["n_muscle_today"].sum() == 2
    assert out.loc[out[module.PLAYER_ID_COL].eq(1), "n_muscle_today"].iloc[0] == 2

    empty = module.classified_injury_type_day_counts(
        injuries.drop(columns=["injury_desc"]),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-03"),
        ("muscle/tendon",),
        "n_muscle_today",
    )
    assert empty.empty
    assert empty.columns.tolist() == [
        module.PLAYER_ID_COL,
        "date",
        "n_muscle_today",
    ]

    with pytest.raises(KeyError):
        module.classified_injury_type_day_counts(
            injuries.drop(columns=["start_date"]),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-03"),
            ("muscle/tendon",),
            "n_muscle_today",
        )
    with pytest.raises(ValueError):
        module.classified_injury_type_day_counts(
            injuries,
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-03"),
            (),
            "n_muscle_today",
        )


def test_add_mutually_exclusive_type_frequency_history(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 1, 1, 2],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-02"]
            ),
            "prior_minutes_played": [0.0, 1000.0, 2000.0, 1000.0],
        }
    )
    injuries = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 1, 2],
            "injury_spell_id": [10, 11, 20],
            "start_date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-01"]
            ),
            "injury_desc": ["Hamstring injury", "ACL rupture", "Toe fracture"],
        }
    )
    out = module.add_mutually_exclusive_type_frequency_history(panel, injuries)
    row_2 = out[out["date"].eq(pd.Timestamp("2020-01-02")) & out[module.PLAYER_ID_COL].eq(1)].iloc[0]
    row_3 = out[out["date"].eq(pd.Timestamp("2020-01-03")) & out[module.PLAYER_ID_COL].eq(1)].iloc[0]
    assert row_2[module.MUSCLE_TENDON_HISTORY_COUNT_COL] == 1
    assert row_2[module.MUSCLE_TENDON_HISTORY_RATE_COL] == pytest.approx(10.0)
    assert row_3[module.JOINT_BONE_HISTORY_COUNT_COL] == 1
    assert row_3[module.JOINT_BONE_HISTORY_RATE_COL] == pytest.approx(5.0)

    empty = module.add_mutually_exclusive_type_frequency_history(
        panel.assign(date=pd.NaT),
        injuries,
    )
    assert empty[module.MUSCLE_TENDON_HISTORY_RATE_COL].eq(0.0).all()
    with pytest.raises(KeyError):
        module.add_mutually_exclusive_type_frequency_history(
            panel.drop(columns=["prior_minutes_played"]),
            injuries,
        )


def test_prior_muscle_tendon_recency_is_prior_only(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 1, 1, 2, 3],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-03", "2020-01-10", "2020-01-05", None]
            ),
        }
    )
    injuries = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 1, 2, None],
            "start_date": pd.to_datetime(
                ["2020-01-01", "2020-01-08", "2020-01-02", "2020-01-01"]
            ),
            "injury_desc": [
                "Hamstring injury",
                "Calf muscle injury",
                "Knee fracture",
                "Hamstring injury",
            ],
        }
    )

    starts = module.prior_injury_type_start_dates(
        injuries,
        ("muscle/tendon",),
        "prior_start_date",
    )
    assert starts["prior_start_date"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-08"),
    ]

    out = module.add_prior_muscle_tendon_recency(panel, injuries)
    same_day = out[out["date"].eq(pd.Timestamp("2020-01-01"))].iloc[0]
    two_days = out[out["date"].eq(pd.Timestamp("2020-01-03"))].iloc[0]
    after_second = out[out["date"].eq(pd.Timestamp("2020-01-10"))].iloc[0]
    joint_only = out[out[module.PLAYER_ID_COL].eq(2)].iloc[0]
    assert same_day[module.MUSCLE_TENDON_HAS_PRIOR_REPORT_COL] == 0
    assert np.isnan(same_day[module.MUSCLE_TENDON_DAYS_SINCE_LAST_REPORT_COL])
    assert two_days[module.MUSCLE_TENDON_DAYS_SINCE_LAST_REPORT_COL] == pytest.approx(
        2.0
    )
    assert two_days[module.MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL] == pytest.approx(
        np.log1p(2.0)
    )
    assert after_second[module.MUSCLE_TENDON_DAYS_SINCE_LAST_REPORT_COL] == pytest.approx(
        2.0
    )
    assert joint_only[module.MUSCLE_TENDON_HAS_PRIOR_REPORT_COL] == 0
    joint_recency = module.add_prior_joint_bone_recency(panel, injuries)
    joint_only = joint_recency[joint_recency[module.PLAYER_ID_COL].eq(2)].iloc[0]
    assert joint_only[module.JOINT_BONE_HAS_PRIOR_REPORT_COL] == 1
    assert joint_only[module.JOINT_BONE_DAYS_SINCE_LAST_REPORT_COL] == pytest.approx(
        3.0
    )
    assert joint_only[module.JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL] == pytest.approx(
        np.log1p(3.0)
    )
    symmetric = module.add_symmetric_type_recency(panel, injuries)
    summary = module.symmetric_type_recency_summary(symmetric)
    assert summary["rows_with_prior_muscle_tendon_report"] == 2
    assert summary["rows_without_prior_muscle_tendon_report"] == 3
    assert summary["median_days_since_last_prior_muscle_tendon_report"] == pytest.approx(
        2.0
    )
    assert summary["rows_with_prior_joint_bone_report"] == 1
    assert summary["rows_without_prior_joint_bone_report"] == 4
    assert summary["median_days_since_last_prior_joint_bone_report"] == pytest.approx(
        3.0
    )
    diagnostic_frame = pd.DataFrame(
        {
            module.MUSCLE_TENDON_HISTORY_RATE_COL: [0.0, 1.0, 2.0, 4.0, 5.0],
            module.MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL: [
                0.0,
                1.0,
                2.0,
                1.0,
                3.0,
            ],
            module.MUSCLE_TENDON_HAS_PRIOR_REPORT_COL: [0, 1, 1, 1, 1],
            module.JOINT_BONE_HISTORY_RATE_COL: [0.0, 2.0, 4.0, 6.0, 8.0],
            module.JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL: [
                0.0,
                4.0,
                2.0,
                3.0,
                1.0,
            ],
            module.JOINT_BONE_HAS_PRIOR_REPORT_COL: [0, 1, 1, 1, 1],
        }
    )
    diagnostics = module.type_frequency_recency_collinearity_summary(diagnostic_frame)
    expected_muscle_all = pd.Series([0.0, 1.0, 2.0, 4.0, 5.0]).corr(
        pd.Series(
            diagnostic_frame[module.MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL]
        )
    )
    expected_joint_prior = pd.Series([2.0, 4.0, 6.0, 8.0]).corr(
        pd.Series([4.0, 2.0, 3.0, 1.0])
    )
    assert diagnostics["muscle_tendon_frequency_log_recency_corr_all_rows"] == (
        pytest.approx(expected_muscle_all)
    )
    assert diagnostics["joint_bone_frequency_log_recency_corr_prior_rows"] == (
        pytest.approx(expected_joint_prior)
    )
    fitted = module.smf.ols(
        "outcome ~ muscle_frequency + joint_frequency",
        data=pd.DataFrame(
            {
                "outcome": [1.0, 2.0, 1.5, 3.5, 2.8, 4.2],
                "muscle_frequency": [0.0, 1.0, 0.5, 2.0, 1.5, 3.0],
                "joint_frequency": [1.0, 0.0, 1.5, 0.5, 2.0, 1.0],
            }
        ),
    ).fit()
    assert module.fitted_term_variance_inflation_factor(
        fitted, "muscle_frequency"
    ) >= 1.0
    with pytest.raises(KeyError, match="model does not contain term"):
        module.fitted_term_variance_inflation_factor(fitted, "missing_frequency")
    terms = module.symmetric_type_recency_terms()
    assert module.MUSCLE_TENDON_LOG_DAYS_SINCE_LAST_REPORT_COL in terms
    assert module.JOINT_BONE_LOG_DAYS_SINCE_LAST_REPORT_COL in terms

    no_desc = module.prior_injury_type_start_dates(
        injuries.drop(columns=["injury_desc"]),
        ("muscle/tendon",),
        "prior_start_date",
    )
    assert no_desc.empty
    with pytest.raises(ValueError):
        module.prior_injury_type_start_dates(injuries, (), "prior_start_date")
    with pytest.raises(KeyError):
        module.prior_injury_type_start_dates(
            injuries.drop(columns=["start_date"]),
            ("muscle/tendon",),
            "prior_start_date",
        )
    with pytest.raises(KeyError):
        module.add_prior_muscle_tendon_recency(panel.drop(columns=["date"]), injuries)

    no_panel_dates = module.add_prior_muscle_tendon_recency(
        panel.assign(date=pd.NaT),
        injuries,
    )
    assert no_panel_dates[module.MUSCLE_TENDON_HAS_PRIOR_REPORT_COL].eq(0).all()
    no_muscle = module.add_prior_muscle_tendon_recency(
        panel,
        injuries.assign(injury_desc="Knee fracture"),
    )
    assert no_muscle[module.MUSCLE_TENDON_HAS_PRIOR_REPORT_COL].eq(0).all()
    other_player_only = module.add_prior_muscle_tendon_recency(
        panel.assign(**{module.PLAYER_ID_COL: 9}),
        injuries,
    )
    assert other_player_only[module.MUSCLE_TENDON_HAS_PRIOR_REPORT_COL].eq(0).all()
    before_first = module.add_prior_muscle_tendon_recency(
        pd.DataFrame(
            {
                module.PLAYER_ID_COL: [1],
                "date": pd.to_datetime(["2019-12-31"]),
            }
        ),
        injuries,
    )
    assert before_first[module.MUSCLE_TENDON_HAS_PRIOR_REPORT_COL].eq(0).all()


def test_mutually_exclusive_type_frequency_frame_filters_recent_returns(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2],
            "date": pd.to_datetime(["2020-01-10", "2020-01-20"]),
            module.MATCH_MINUTES_COL: [30.0, 90.0],
            "all_minutes_last_7d": [0.0, 90.0],
            "injury_event_matchproxy_muscle_tendon": [0, 1],
            "fragility_frequency_only": ["regular", "fragile"],
            module.MUSCLE_TENDON_HISTORY_RATE_COL: [2.0, 0.0],
            module.JOINT_BONE_HISTORY_RATE_COL: [0.0, 1.5],
        }
    )
    injuries = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2],
            "start_date": pd.to_datetime(["2019-12-01", "2019-10-01"]),
            "end_date": pd.to_datetime(["2020-01-01", "2019-11-01"]),
        }
    )
    unrestricted = module.mutually_exclusive_type_frequency_frame(
        panel,
        injuries,
        exclude_recent_returns=False,
    )
    restricted = module.mutually_exclusive_type_frequency_frame(
        panel,
        injuries,
        exclude_recent_returns=True,
    )
    assert unrestricted[module.MUSCLE_TENDON_HISTORY_RATE_COL].tolist() == [2.0, 0.0]
    assert unrestricted[module.JOINT_BONE_HISTORY_RATE_COL].tolist() == [0.0, 1.5]
    assert len(restricted) == 1
    assert restricted[module.PLAYER_ID_COL].tolist() == [2]


def test_latest_eligible_player_history_snapshot(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 1, 2, 3],
            "date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-02", None]),
            "prior_minutes_played": [1000.0, 1500.0, 500.0, 2000.0],
            module.MUSCLE_TENDON_HISTORY_RATE_COL: [1.0, 3.0, 5.0, 7.0],
            module.JOINT_BONE_HISTORY_RATE_COL: [0.0, 2.0, 4.0, 6.0],
        }
    )
    out = module.latest_eligible_player_history_snapshot(
        panel,
        module.MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS,
    )
    assert out[module.PLAYER_ID_COL].tolist() == [1]
    assert out[module.MUSCLE_TENDON_HISTORY_RATE_COL].tolist() == [3.0]

    empty = module.latest_eligible_player_history_snapshot(
        panel.assign(prior_minutes_played=10.0),
        module.MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS,
    )
    assert empty.empty
    with pytest.raises(KeyError):
        module.latest_eligible_player_history_snapshot(
            panel.drop(columns=[module.JOINT_BONE_HISTORY_RATE_COL]),
            module.MUTUALLY_EXCLUSIVE_TYPE_HISTORY_RATE_COLS,
        )


def test_distribution_statistics(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    stats = module.distribution_statistics(pd.Series([0.0, 1.0, 2.0, 4.0]))
    assert stats["n"] == 4
    assert stats["nonzero_n"] == 3
    assert stats["median"] == pytest.approx(1.5)
    assert stats["q1"] == pytest.approx(0.75)
    assert stats["q3"] == pytest.approx(2.5)
    assert stats["iqr"] == pytest.approx(1.75)
    assert stats["mean"] == pytest.approx(1.75)
    assert stats["maximum"] == pytest.approx(4.0)
    assert np.isfinite(stats["skewness"])

    empty = module.distribution_statistics(pd.Series([np.nan, "bad"]))
    assert empty["n"] == 0
    assert np.isnan(empty["q3"])
    short = module.distribution_statistics(pd.Series([1.0, 2.0]))
    assert np.isnan(short["skewness"])


def test_q3_group_mean_statistics(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    stats = module.q3_group_mean_statistics(pd.Series([0.0, 1.0, 5.0, 9.0]), 5.0)
    assert stats["below_q3_rows"] == 2
    assert stats["above_q3_rows"] == 2
    assert stats["below_q3_mean"] == pytest.approx(0.5)
    assert stats["above_q3_mean"] == pytest.approx(7.0)
    assert stats["above_minus_below_q3_mean_gap"] == pytest.approx(6.5)

    empty = module.q3_group_mean_statistics(pd.Series([np.nan, "bad"]), 1.0)
    assert empty["below_q3_rows"] == 0
    assert np.isnan(empty["above_minus_below_q3_mean_gap"])
    missing_threshold = module.q3_group_mean_statistics(pd.Series([1.0]), np.nan)
    assert np.isnan(missing_threshold["below_q3_mean"])
    one_sided = module.q3_group_mean_statistics(pd.Series([5.0, 6.0]), 1.0)
    assert one_sided["below_q3_rows"] == 0
    assert np.isnan(one_sided["above_minus_below_q3_mean_gap"])


def test_scaled_log_rate_ratio_interval(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    out = module.scaled_log_rate_ratio_interval(np.log(1.1), 0.1, 2.0)
    assert out["irr"] == pytest.approx(1.21)
    assert out["ci_low"] < out["irr"] < out["ci_high"]

    zero = module.scaled_log_rate_ratio_interval(0.0, 0.0, 3.0)
    assert zero["irr"] == pytest.approx(1.0)
    assert zero["p_value"] == pytest.approx(1.0)

    infinite_z = module.scaled_log_rate_ratio_interval(np.log(2.0), 0.0, 1.0)
    assert infinite_z["p_value"] == pytest.approx(0.0)

    missing = module.scaled_log_rate_ratio_interval(np.log(2.0), 0.1, np.nan)
    assert np.isnan(missing["irr"])
    assert np.isnan(missing["p_value"])


def test_scaled_log_rate_ratio_difference_interval(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    out = module.scaled_log_rate_ratio_difference_interval(
        np.log(1.1),
        0.10,
        2.0,
        np.log(1.05),
        0.08,
        1.0,
        0.12,
    )
    expected = (1.1**2.0) / 1.05
    assert out["log_rate_ratio"] == pytest.approx(np.log(expected))
    assert out["log_rate_ratio_se"] > 0
    assert out["irr"] == pytest.approx(expected)
    assert out["ci_low"] < out["irr"] < out["ci_high"]

    no_variance = module.scaled_log_rate_ratio_difference_interval(
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    assert no_variance["log_rate_ratio"] == pytest.approx(0.0)
    assert no_variance["log_rate_ratio_se"] == pytest.approx(0.0)
    assert no_variance["irr"] == pytest.approx(1.0)
    assert no_variance["p_value"] == pytest.approx(1.0)

    infinite_z = module.scaled_log_rate_ratio_difference_interval(
        np.log(2.0),
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    assert infinite_z["p_value"] == pytest.approx(0.0)

    missing = module.scaled_log_rate_ratio_difference_interval(
        np.log(2.0),
        0.1,
        np.nan,
        0.0,
        0.1,
        1.0,
        0.1,
    )
    assert np.isnan(missing["log_rate_ratio"])
    assert np.isnan(missing["log_rate_ratio_se"])
    assert np.isnan(missing["irr"])


def test_log_and_observed_to_predicted_ratio_intervals(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    se = module.log_standard_error_from_ratio_ci(1.0, 4.0)
    assert se == pytest.approx(np.log(4.0) / (2 * 1.959963984540054))

    out = module.observed_to_predicted_ratio_interval(
        observed_rr=2.0,
        observed_ci_low=1.0,
        observed_ci_high=4.0,
        predicted_log_rr=np.log(1.5),
        predicted_log_rr_se=0.1,
    )
    assert out["ratio"] == pytest.approx(2.0 / 1.5)
    assert out["ci_low"] < out["ratio"] < out["ci_high"]
    assert 0.0 < out["p_value"] < 1.0


def test_negative_control_type_frequency_linearity_check(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    frequency_results = pd.DataFrame(
        {
            "restriction": ["all eligible rows"],
            "muscle_tendon_history_log_rr_per_10000min": [np.log(1.10)],
            "muscle_tendon_history_log_rr_se": [0.05],
            "joint_bone_history_log_rr_per_10000min": [np.log(1.02)],
            "joint_bone_history_log_rr_se": [0.04],
            "direct_ratio_log_rr_se": [0.06],
        }
    )
    binary_results = pd.DataFrame(
        {
            "restriction": ["all eligible rows"],
            "muscle_tendon_high_history_rr": [2.0],
            "muscle_tendon_high_history_ci_low": [1.5],
            "muscle_tendon_high_history_ci_high": [2.7],
            "joint_bone_high_history_rr": [1.1],
            "joint_bone_high_history_ci_low": [0.8],
            "joint_bone_high_history_ci_high": [1.5],
            "direct_ratio_muscle_over_joint_bone": [1.8],
            "direct_ratio_ci_low": [1.1],
            "direct_ratio_ci_high": [3.0],
        }
    )
    distribution_context = pd.DataFrame(
        [
            {
                "restriction": "all eligible rows",
                "distribution_scope": scope,
                "history_variable": variable,
                "above_minus_below_q3_mean_gap_per_10000min": gap,
            }
            for scope, muscle_gap, joint_gap in [
                ("analytic_match_rows", 5.0, 3.0),
                ("anchor_0_match_rows", 6.0, 4.0),
            ]
            for variable, gap in [
                ("muscle_tendon", muscle_gap),
                ("joint_ligament_or_bone_fracture", joint_gap),
            ]
        ]
    )

    out = module.negative_control_type_frequency_linearity_check(
        frequency_results,
        binary_results,
        distribution_context,
    )
    assert len(out) == 6
    anchor_muscle = out[
        out["distribution_scope"].eq("anchor_0_match_rows")
        & out["history_variable"].eq("muscle_tendon")
    ].iloc[0]
    assert anchor_muscle["predicted_binary_irr_from_continuous_slope"] == pytest.approx(
        1.10**6.0
    )
    assert anchor_muscle["observed_divided_by_predicted_ratio"] == pytest.approx(
        2.0 / (1.10**6.0)
    )

    anchor_direct = out[
        out["distribution_scope"].eq("anchor_0_match_rows")
        & out["comparison"].eq("direct_muscle_tendon_over_joint_bone")
    ].iloc[0]
    expected_direct = (1.10**6.0) / (1.02**4.0)
    assert anchor_direct["predicted_binary_irr_from_continuous_slope"] == pytest.approx(
        expected_direct
    )
    assert anchor_direct["observed_divided_by_predicted_ratio"] == pytest.approx(
        1.8 / expected_direct
    )
    assert anchor_direct["ci_method"].startswith("delta method")


def test_mutually_exclusive_type_history_thresholds(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2, 3, 4],
            "date": pd.to_datetime(["2020-01-01"] * 4),
            "prior_minutes_played": [1000.0, 1000.0, 1000.0, 1000.0],
            module.MUSCLE_TENDON_HISTORY_RATE_COL: [0.0, 2.0, 4.0, 8.0],
            module.JOINT_BONE_HISTORY_RATE_COL: [0.0, 1.0, 3.0, 5.0],
        }
    )
    thresholds = module.mutually_exclusive_type_history_thresholds(
        panel,
        pd.DataFrame(),
    )
    assert thresholds["muscle_tendon_q3"] == pytest.approx(5.0)
    assert thresholds["joint_bone_q3"] == pytest.approx(3.5)
    assert thresholds["snapshot_players"] == 4

    empty = module.mutually_exclusive_type_history_thresholds(
        panel.assign(prior_minutes_played=1.0),
        pd.DataFrame(),
    )
    assert empty["muscle_tendon_q3"] == 0.0
    assert empty["snapshot_players"] == 0


def test_add_mutually_exclusive_type_binary_labels(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    frame = pd.DataFrame(
        {
            module.MUSCLE_TENDON_HISTORY_RATE_COL: [1.0, 5.0, np.nan],
            module.JOINT_BONE_HISTORY_RATE_COL: [4.0, 1.0, np.nan],
        }
    )
    out = module.add_mutually_exclusive_type_binary_labels(
        frame,
        {"muscle_tendon_q3": 3.0, "joint_bone_q3": 2.0},
    )
    assert out[module.MUSCLE_TENDON_HISTORY_HIGH_COL].tolist() == [0, 1, 0]
    assert out[module.JOINT_BONE_HISTORY_HIGH_COL].tolist() == [1, 0, 0]

    with pytest.raises(KeyError):
        module.add_mutually_exclusive_type_binary_labels(
            frame.drop(columns=[module.JOINT_BONE_HISTORY_RATE_COL]),
            {"muscle_tendon_q3": 3.0, "joint_bone_q3": 2.0},
        )


def test_mutually_exclusive_type_binary_frame_filters_recent_returns(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2, 3, 4],
            "date": pd.to_datetime(
                ["2020-01-10", "2020-01-20", "2020-01-21", "2020-01-22"]
            ),
            "prior_minutes_played": [1000.0, 1000.0, 1000.0, 1000.0],
            module.MATCH_MINUTES_COL: [30.0, 90.0, 90.0, 90.0],
            "all_minutes_last_7d": [0.0, 90.0, 0.0, 0.0],
            "injury_event_matchproxy_muscle_tendon": [0, 1, 0, 0],
            "fragility_frequency_only": ["regular", "fragile", "regular", "fragile"],
            module.MUSCLE_TENDON_HISTORY_RATE_COL: [1.0, 2.0, 6.0, 8.0],
            module.JOINT_BONE_HISTORY_RATE_COL: [4.0, 1.0, 3.0, 5.0],
        }
    )
    injuries = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2],
            "start_date": pd.to_datetime(["2019-12-01", "2019-10-01"]),
            "end_date": pd.to_datetime(["2020-01-01", "2019-11-01"]),
        }
    )
    unrestricted = module.mutually_exclusive_type_binary_frame(
        panel,
        injuries,
        exclude_recent_returns=False,
    )
    restricted = module.mutually_exclusive_type_binary_frame(
        panel,
        injuries,
        exclude_recent_returns=True,
    )
    assert unrestricted["muscle_tendon_history_threshold_per_10000min"].iloc[0] == pytest.approx(6.5)
    assert unrestricted["joint_bone_history_threshold_per_10000min"].iloc[0] == pytest.approx(4.25)
    assert unrestricted[module.MUSCLE_TENDON_HISTORY_HIGH_COL].tolist() == [0, 0, 0, 1]
    assert unrestricted[module.JOINT_BONE_HISTORY_HIGH_COL].tolist() == [0, 0, 0, 1]
    assert restricted[module.PLAYER_ID_COL].tolist() == [2, 3, 4]


def test_joint_label_prediction_template(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    out = module._joint_label_prediction_template(
        anchor_minutes=45.0,
        all_type_high_history=1,
        type_discordant_high_history=0,
    )
    assert out["all_minutes_last_7d"].tolist() == [45.0]
    assert out["all_type_high_history"].tolist() == [1]
    assert out["type_discordant_high_history"].tolist() == [0]
    assert out["week_phase_sin"].tolist() == [0.0]
    assert out["log_minutes_played"].iloc[0] == pytest.approx(np.log(90.0))


def test_exclusive_type_frequency_prediction_template(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    out = module._exclusive_type_frequency_prediction_template(
        anchor_minutes=45.0,
        muscle_tendon_frequency=2.5,
        joint_bone_frequency=1.5,
    )
    assert out["all_minutes_last_7d"].tolist() == [45.0]
    assert out[module.MUSCLE_TENDON_HISTORY_RATE_COL].tolist() == [2.5]
    assert out[module.JOINT_BONE_HISTORY_RATE_COL].tolist() == [1.5]
    assert out["halfweek_phase_cos"].tolist() == [0.0]
    assert out["log_minutes_played"].iloc[0] == pytest.approx(np.log(90.0))


def test_exclusive_type_binary_prediction_template(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    out = module._exclusive_type_binary_prediction_template(
        anchor_minutes=45.0,
        muscle_tendon_high_history=1,
        joint_bone_high_history=0,
    )
    assert out["all_minutes_last_7d"].tolist() == [45.0]
    assert out[module.MUSCLE_TENDON_HISTORY_HIGH_COL].tolist() == [1]
    assert out[module.JOINT_BONE_HISTORY_HIGH_COL].tolist() == [0]
    assert out["week_phase_cos"].tolist() == [0.0]
    assert out["log_minutes_played"].iloc[0] == pytest.approx(np.log(90.0))


def test_mutually_exclusive_type_frequency_distribution_context(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    panel = pd.DataFrame(
        {
            module.PLAYER_ID_COL: [1, 2, 3, 4],
            "date": pd.to_datetime(
                ["2020-01-10", "2020-01-11", "2020-01-12", "2020-01-13"]
            ),
            "prior_minutes_played": [1000.0, 1000.0, 1000.0, 1000.0],
            module.MATCH_MINUTES_COL: [90.0, 90.0, 90.0, 90.0],
            "all_minutes_last_7d": [0.0, 0.0, 90.0, 0.0],
            "fragility_frequency_only": ["regular", "regular", "fragile", "fragile"],
            "injury_event_matchproxy_muscle_tendon": [0, 1, 0, 1],
            module.MUSCLE_TENDON_HISTORY_RATE_COL: [0.0, 2.0, 6.0, 10.0],
            module.JOINT_BONE_HISTORY_RATE_COL: [0.0, 1.0, 4.0, 5.0],
        }
    )
    exclusive_results = pd.DataFrame(
        {
            "restriction": [
                "all eligible rows",
                "exclude rows within 14 days of recorded return",
            ],
            "muscle_tendon_history_log_rr_per_10000min": [np.log(1.1), np.log(1.1)],
            "muscle_tendon_history_log_rr_se": [0.10, 0.10],
            "muscle_tendon_history_irr_per_10000min": [1.1, 1.1],
            "joint_bone_history_log_rr_per_10000min": [np.log(1.05), np.log(1.05)],
            "joint_bone_history_log_rr_se": [0.08, 0.08],
            "joint_bone_history_irr_per_10000min": [1.05, 1.05],
            "direct_ratio_log_rr_se": [0.12, 0.12],
        }
    )
    injuries = pd.DataFrame(
        {
            module.PLAYER_ID_COL: pd.Series(dtype=int),
            "start_date": pd.Series(dtype="datetime64[ns]"),
            "end_date": pd.Series(dtype="datetime64[ns]"),
        }
    )
    out = module.mutually_exclusive_type_frequency_distribution_context(
        panel,
        injuries,
        exclusive_results,
    )
    assert len(out) == 12
    assert set(out["distribution_scope"]) == {
        "latest_eligible_player_snapshot",
        "analytic_match_rows",
        "anchor_0_match_rows",
    }
    latest_muscle = out[
        out["restriction"].eq("all eligible rows")
        & out["distribution_scope"].eq("latest_eligible_player_snapshot")
        & out["history_variable"].eq("muscle_tendon")
    ].iloc[0]
    assert latest_muscle["q3_per_10000min"] == pytest.approx(7.0)
    assert latest_muscle["binary_q3_threshold_per_10000min"] == pytest.approx(7.0)
    assert latest_muscle["above_q3_mean_per_10000min"] == pytest.approx(10.0)
    assert latest_muscle["per_q3_irr"] == pytest.approx(1.1**7.0)
    assert latest_muscle["per_q3_direct_ratio_muscle_over_joint_bone"] > 1.0

    anchor_muscle = out[
        out["restriction"].eq("all eligible rows")
        & out["distribution_scope"].eq("anchor_0_match_rows")
        & out["history_variable"].eq("muscle_tendon")
    ].iloc[0]
    assert anchor_muscle["n"] == 3
    assert anchor_muscle["above_q3_rows"] == 1
    assert anchor_muscle["below_q3_mean_per_10000min"] == pytest.approx(1.0)


def test_frequency_only_publication_column(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    assert module.frequency_only_publication_column("regular_180_rate") == (
        "lower_frequency_180_rate"
    )
    assert module.frequency_only_publication_column("intermediate_180_vs_90_irr") == (
        "lower_frequency_180_vs_90_irr"
    )
    assert module.frequency_only_publication_column("fragile_0_rate") == (
        "higher_frequency_0_rate"
    )
    assert module.frequency_only_publication_column("higher_180_vs_90_irr") == (
        "higher_frequency_180_vs_90_irr"
    )
    assert module.frequency_only_publication_column("rr_180") == "rr_180"


def _publication_decomposition_fixture():
    shared = {
        "component": np.nan,
        "model": np.nan,
        "term": np.nan,
        "transition_state": np.nan,
        "estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "n_events": np.nan,
        "match_minutes": np.nan,
        "events_per_1000_match_hours": np.nan,
        "events_per_1000_match_hours_ci_low": np.nan,
        "events_per_1000_match_hours_ci_high": np.nan,
    }
    rows = []
    for term, estimate in [
        ("higher_history_state", 1.4),
        ("player_higher_history_match_share", 2.5),
        ("within_player_higher_history_deviation", 0.55),
    ]:
        rows.append(
            {
                **shared,
                "component": "within_between_poisson",
                "model": "within_between",
                "term": term,
                "estimate": estimate,
                "ci_low": estimate * 0.8,
                "ci_high": estimate * 1.2,
                "p_value": 0.01,
                "n_events": 20,
            }
        )
    for state, events, minutes, rate in [
        ("non_switcher_intermediate_history", 10, 1000.0, 600.0),
        ("non_switcher_higher_history", 10, 500.0, 1200.0),
    ]:
        rows.append(
            {
                **shared,
                "component": "switcher_transition_state",
                "transition_state": state,
                "n_events": events,
                "match_minutes": minutes,
                "events_per_1000_match_hours": rate,
                "events_per_1000_match_hours_ci_low": rate * 0.8,
                "events_per_1000_match_hours_ci_high": rate * 1.2,
            }
        )
    return pd.DataFrame(rows)


def test_between_within_publication_summary(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    decomposition = _publication_decomposition_fixture()
    out = module.between_within_publication_summary(decomposition)
    assert out["finding"].tolist()[-1] == "non_switcher_higher_vs_intermediate"
    contrast = out.iloc[-1]
    assert contrast["estimate"] == pytest.approx(2.0)
    assert contrast["ci_low"] < 2.0 < contrast["ci_high"]
    assert out.loc[
        out["finding"].eq("between_player_higher_history_share"), "estimate"
    ].iloc[0] == 2.5

    with pytest.raises(KeyError):
        module.between_within_publication_summary(
            decomposition.drop(columns=["transition_state"])
        )
    with pytest.raises(ValueError, match="Missing decomposition term"):
        module.between_within_publication_summary(
            decomposition[
                ~decomposition["term"].eq("player_higher_history_match_share")
            ]
        )
    with pytest.raises(ValueError, match="Missing transition state"):
        module.between_within_publication_summary(
            decomposition[
                ~decomposition["transition_state"].eq(
                    "non_switcher_higher_history"
                )
            ]
        )


def test_recovery_interval_display_table(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    rates = pd.DataFrame(
        {
            "history_stratum": ["fragile", "regular", "regular"],
            "recovery_interval_bin": ["0-3 days", "0-3 days", "no prior match"],
            "match_rows": [2, 3, 1],
            "match_hours": [3.0, 4.0, 1.0],
            "events": [1, 1, 0],
            "events_per_1000_match_hours": [333.3, 250.0, 0.0],
            "events_per_1000_match_hours_ci_low": [50.0, 40.0, 0.0],
            "events_per_1000_match_hours_ci_high": [800.0, 700.0, 100.0],
        }
    )
    out = module.recovery_interval_display_table(rates)
    assert out["history_stratum"].tolist() == [
        "intermediate prior-injury-history",
        "higher prior-injury-history",
    ]
    assert "no prior match" not in out["recovery_interval_bin"].tolist()
    with pytest.raises(KeyError):
        module.recovery_interval_display_table(rates.drop(columns=["events"]))


def test_recency_attenuation_contrasts_and_type_history_family(load_src_module):
    module = load_src_module("18_match_proxy_poisson_splines_perminute.py")
    terms = []
    for term in (
        module.MUSCLE_TENDON_HISTORY_HIGH_COL,
        module.JOINT_BONE_HISTORY_HIGH_COL,
        module.MUSCLE_TENDON_HISTORY_RATE_COL,
        module.JOINT_BONE_HISTORY_RATE_COL,
    ):
        terms.extend([f"spec_unadjusted:{term}", f"spec_adjusted:{term}"])
    params = pd.Series(
        [0.4, 0.2, -0.1, -0.1, 0.06, 0.03, 0.03, 0.04],
        index=terms,
    )
    covariance = pd.DataFrame(
        np.eye(len(terms)) * 0.0025,
        index=terms,
        columns=terms,
    )

    class Result:
        def __init__(self):
            self.params = params

        @staticmethod
        def cov_params():
            return covariance

    frame = pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "injury_event_matchproxy_muscle_tendon": [1, 0],
        }
    )
    formula = module.recency_attenuation_stacked_formula(
        "injury_event_matchproxy_muscle_tendon",
        "all_minutes_last_7d",
    )
    assert "spec_adjusted:has_prior_muscle_tendon_report" in formula
    attenuation_rows = module.recency_attenuation_contrast_rows(
        Result(),
        "all eligible rows",
        frame,
        "injury_event_matchproxy_muscle_tendon",
    )
    attenuation = pd.DataFrame(attenuation_rows)
    assert len(attenuation) == 6
    muscle = attenuation.loc[
        attenuation["contrast_id"].eq("muscle_tendon_high_step")
    ].iloc[0]
    assert muscle["adjusted_over_unadjusted_ratio"] == pytest.approx(np.exp(-0.2))
    assert muscle["n_events"] == 1

    direct = pd.DataFrame(
        [{"restriction": "all eligible rows", "direct_ratio_p": 0.01}]
    )
    binary = pd.DataFrame(
        [
            {
                "restriction": "all eligible rows",
                "muscle_tendon_high_history_p": 0.01,
                "joint_bone_high_history_p": np.nan,
                "direct_ratio_p": 0.03,
                "recency_adjusted_muscle_tendon_high_history_p": 0.04,
                "recency_adjusted_joint_bone_high_history_p": 0.05,
                "recency_adjusted_direct_ratio_p": 0.06,
            }
        ]
    )
    frequency = pd.DataFrame(
        [
            {
                "restriction": "all eligible rows",
                "muscle_tendon_history_p": 0.01,
                "joint_bone_history_p": 0.02,
                "direct_ratio_p": 0.03,
                "recency_adjusted_muscle_tendon_history_p": 0.04,
                "recency_adjusted_joint_bone_history_p": 0.05,
                "recency_adjusted_direct_ratio_p": 0.06,
            }
        ]
    )
    linearity = pd.DataFrame(
        [
            {
                "restriction": "all eligible rows",
                "distribution_scope": "analytic_match_rows",
                "comparison": "direct_muscle_tendon_over_joint_bone",
                "history_variable": "direct",
                "observed_divided_by_predicted_p": 0.07,
            }
        ]
    )
    formal = pd.DataFrame(
        [
            {
                "restriction": "all eligible rows",
                "recency_adjustment": "none",
                "comparison": "single_history_variable",
                "history_variable": "muscle_tendon",
                "binary_step_p": 0.01,
                "continuous_p": 0.02,
                "direct_continuous_ratio_p": 0.03,
            },
            {
                "restriction": "all eligible rows",
                "recency_adjustment": "none",
                "comparison": "direct_muscle_tendon_over_joint_bone",
                "history_variable": "direct",
                "binary_step_p": 0.04,
                "continuous_p": np.nan,
                "direct_continuous_ratio_p": 0.05,
            },
        ]
    )
    family = module.type_history_multiplicity_family(
        direct,
        binary,
        frequency,
        linearity,
        formal,
        attenuation,
    )
    assert family["test_id"].is_unique
    assert family["family_size"].eq(len(family)).all()
    assert family["p_holm_type_history_family"].notna().all()
    assert module.type_history_multiplicity_family(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    ).empty
    with pytest.raises(ValueError, match="Duplicate type-history test IDs"):
        module.type_history_multiplicity_family(
            pd.concat([direct, direct], ignore_index=True),
            binary,
            frequency,
            linearity,
            formal,
            attenuation,
        )
