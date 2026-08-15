import pandas as pd
import pytest


def games_frame():
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "competition_id": ["GB1", "FIWC", "GB1"],
            "competition_type": ["domestic_league", "national_team_competition", "domestic_league"],
            "season": [2017, 2017, 2017],
            "date": ["2017-08-01", "2017-08-02", "2017-08-03"],
            "home_club_id": [10, 100, 10],
            "away_club_id": [20, 200, 30],
            "home_club_name": ["Club", "England", "Club"],
            "away_club_name": ["Away", "Spain", "Other"],
            "home_club_goals": [2, 1, 0],
            "away_club_goals": [0, 0, 1],
            "stadium": ["Club ground", "National stadium", "Club ground"],
            "url": ["club-url", "national-url", "club-url-2"],
        }
    )


def endpoint_frame():
    return pd.DataFrame(
        {
            "tm_player_id": [1, 1],
            "game_id": [2, 4],
            "date": ["2017-08-02", "2017-08-04"],
            "kickoff_utc": ["2017-08-02T18:00:00+00:00", "2017-08-04T18:00:00+00:00"],
            "kickoff_time_known": [True, True],
            "team_id": [100, 100],
            "opponent_team_id": [200, 200],
            "team_venue": ["home", "away"],
            "competition_id": ["FIWC", "FS"],
            "competition_group_id": ["A", pd.NA],
            "competition_type_id": [11, 11],
            "season": [2017, 2017],
            "stadium_id": [5, 6],
            "minutes_played": [60, 30],
            "is_starter": [True, False],
            "participation_state": ["played", "played"],
            "source_url": ["endpoint-2", "endpoint-4"],
            "cache_file": ["2.json", "4.json"],
            "retrieved_at_utc": ["now", "now"],
        }
    )


def metadata_frames():
    teams = pd.DataFrame({"national_team_id": [100, 200], "name": ["England", "Spain"]})
    competitions = pd.DataFrame(
        {
            "competition_id": ["FIWC", "FS"],
            "name": ["World Cup", "Friendlies"],
            "type": ["national_team_competition", "national_team_competition"],
        }
    )
    return teams, competitions


def test_classifiers_and_epl_seasons(load_src_module, tmp_path):
    module = load_src_module("26_build_public_match_timeline.py")
    (tmp_path / "transfermarkt_datasets_20250101").mkdir()
    (tmp_path / "transfermarkt_datasets_20260803").mkdir()
    assert module.newest_snapshot_dir(tmp_path).name.endswith("20260803")
    with pytest.raises(FileNotFoundError):
        module.newest_snapshot_dir(tmp_path / "missing")
    games = games_frame()
    assert len(module.epl_club_seasons(games)) == 3
    with pytest.raises(KeyError, match="away_club_id"):
        module.epl_club_seasons(games.drop(columns="away_club_id"))
    assert module.classify_team_level(11) == "senior"
    assert module.classify_team_level(17) == "youth_or_olympic"
    assert module.classify_team_level(None, "England U21") == "youth_or_olympic"
    assert module.classify_team_level(None, "England") == "unknown"
    assert module.classify_competition_status("FS") == "friendly"
    assert module.classify_competition_status("x", "Test friendly") == "friendly"
    assert module.classify_competition_status("EURO") == "competitive"
    assert module.classify_competition_status(None) == "unknown"
    assert module._venue_key(7, "ignored") == "tm_stadium_id:7"
    assert module._venue_key(None, " Ground ") == "tm_stadium_name:ground"
    assert module._venue_key(None, None) is pd.NA


def test_enrichment_published_and_duplicate_resolution(load_src_module):
    module = load_src_module("26_build_public_match_timeline.py")
    teams, competitions = metadata_frames()
    endpoint = module.enrich_endpoint_national_appearances(endpoint_frame(), teams, competitions)
    assert endpoint.loc[0, "team_name"] == "England"
    assert endpoint.loc[0, "is_senior_competitive"]
    assert endpoint.loc[1, "is_senior_friendly"]
    endpoint_from_games = module.enrich_endpoint_national_appearances(
        endpoint_frame(),
        pd.DataFrame({"national_team_id": [], "name": []}),
        competitions,
        games_frame(),
    )
    assert endpoint_from_games.loc[0, "team_name"] == "England"
    assert endpoint_from_games.loc[0, "opponent_team_name"] == "Spain"
    assert endpoint_from_games.loc[0, "stadium_name"] == "National stadium"
    assert endpoint_from_games.loc[0, "validation_team_goals"] == 1
    assert endpoint_from_games.loc[0, "validation_opponent_goals"] == 0
    assert endpoint_from_games.loc[0, "validation_score_source"] == "published_snapshot_game"
    full_match = endpoint_frame().iloc[[0]].copy()
    full_match["minutes_played"] = 90
    full_match["game_duration_minutes"] = 90
    full_match["team_goals_on_pitch"] = 2
    full_match["opponent_goals_on_pitch"] = 1
    full_match["is_starter"] = True
    scored = module.enrich_endpoint_national_appearances(
        full_match, teams, competitions
    )
    assert scored.loc[0, "validation_score_source"] == "full_match_on_pitch"
    full_match["is_starter"] = False
    substitute = module.enrich_endpoint_national_appearances(
        full_match, teams, competitions
    )
    assert substitute.loc[0, "validation_score_source"] == "unavailable"
    assert pd.isna(endpoint_from_games.loc[1, "team_name"])
    with pytest.raises(KeyError, match="snapshot games"):
        module.enrich_endpoint_national_appearances(
            endpoint_frame(), teams, competitions, games_frame().drop(columns="stadium")
        )
    with pytest.raises(KeyError, match="team_id"):
        module.enrich_endpoint_national_appearances(endpoint_frame().drop(columns="team_id"), teams, competitions)

    apps = pd.DataFrame(
        {
            "player_id": [1, 1],
            "game_id": [2, 1],
            "player_club_id": [100, 10],
            "competition_id": ["FIWC", "GB1"],
            "date": ["1999-01-01", "1999-01-02"],
            "minutes_played": [60, 90],
        }
    )
    published = module.published_national_appearances(apps, games_frame(), [1], competitions)
    assert len(published) == 1
    assert published.loc[0, "stadium_name"] == "National stadium"
    resolved, audit = module.resolve_national_duplicates(endpoint, published)
    assert len(resolved) == 2
    assert "published_preferred_consistent" in audit["resolution"].tolist()
    conflicting = endpoint.copy()
    conflicting.loc[0, "minutes_played"] = 61
    _, conflict_audit = module.resolve_national_duplicates(conflicting.iloc[[0]], published)
    assert conflict_audit.loc[0, "resolution"] == "unresolved_minutes_conflict_excluded"
    duplicate_endpoint = pd.concat([endpoint.iloc[[1]], endpoint.iloc[[1]]], ignore_index=True)
    duplicated, duplicated_audit = module.resolve_national_duplicates(duplicate_endpoint, published.iloc[0:0])
    assert len(duplicated) == 1
    assert duplicated_audit.loc[0, "resolution"] == "endpoint_duplicate_consistent"
    empty, empty_audit = module.resolve_national_duplicates(endpoint.iloc[0:0], published.iloc[0:0])
    assert empty.empty and empty_audit.empty
    minimal, _ = module.resolve_national_duplicates(
        pd.DataFrame({"tm_player_id": [9], "game_id": [99], "minutes_played": [1], "source": ["minimal"]}),
        pd.DataFrame(),
    )
    assert pd.isna(minimal.loc[0, "competition_name"])


def test_club_timeline_timeline_scopes_and_features(load_src_module):
    module = load_src_module("26_build_public_match_timeline.py")
    teams, competitions = metadata_frames()
    apps = pd.DataFrame(
        {
            "player_id": [1, 2],
            "game_id": [1, 3],
            "player_club_id": [10, 10],
            "minutes_played": [90, 90],
            "date": ["2017-08-01", "2017-08-03"],
            "competition_id": ["GB1", "GB1"],
        }
    )
    club = module.build_club_timeline(apps, games_frame(), [1, 2])
    endpoint = module.enrich_endpoint_national_appearances(endpoint_frame(), teams, competitions)
    timeline = module.build_public_match_timeline(club, endpoint)
    assert set(timeline["source"]) == {"club", "player_performance_endpoint"}
    for scope in module.SCOPES:
        assert len(module.scope_mask(timeline, scope)) == len(timeline)
    assert module.scope_mask(timeline, module.SENIOR_NATIONAL_ONLY_SCOPE).sum() == 1
    assert module.scope_mask(timeline, module.SENIOR_ALL_NATIONAL_ONLY_SCOPE).sum() == 2
    assert module.scope_mask(timeline, module.BROADER_NATIONAL_ONLY_SCOPE).sum() == 2
    with pytest.raises(ValueError, match="Unknown"):
        module.scope_mask(timeline, "p-hack")
    with pytest.raises(ValueError, match="duplicate"):
        module.build_public_match_timeline(club, pd.concat([endpoint, endpoint.iloc[[0]]], ignore_index=True))
    assert module.build_public_match_timeline(club.iloc[0:0], endpoint.iloc[0:0]).empty
    sparse_timeline = module.build_public_match_timeline(
        pd.DataFrame({"tm_player_id": [9], "match_key": ["x"], "date": ["2017-08-01"], "minutes_played": [1]}),
        pd.DataFrame(),
    )
    assert pd.isna(sparse_timeline.loc[0, "source"])

    targets = pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "date": ["2017-08-05", "2017-08-05"],
            "all_minutes_last_7d": [90, 90],
        }
    )
    features = module.build_scope_exposure_features(
        timeline,
        targets,
        scopes=(*module.SCOPES[:2], *module.NATIONAL_ONLY_SCOPES),
    )
    player_one = features.loc[features["tm_player_id"].eq(1)].iloc[0]
    player_two = features.loc[features["tm_player_id"].eq(2)].iloc[0]
    assert player_one["club_competitive_minutes_last_7d"] == 90
    assert player_one["club_plus_senior_national_minutes_last_7d"] == 150
    assert player_one["club_plus_senior_national_national_minutes_last_7d"] == 60
    assert player_one["club_plus_senior_national_days_since_previous_appearance"] == 3
    assert player_one["club_plus_senior_national_consecutive_match_sequence"] == 2
    assert player_two["club_plus_senior_national_minutes_last_7d"] == 90
    assert player_two["club_plus_senior_national_days_since_previous_appearance"] == 2
    assert features["recovery_measure"].eq("calendar_days").all()
    assert player_one[f"{module.SENIOR_NATIONAL_ONLY_SCOPE}_minutes_last_7d"] == 60
    assert player_one[f"{module.SENIOR_ALL_NATIONAL_ONLY_SCOPE}_minutes_last_7d"] == 90
    assert player_one[f"{module.BROADER_NATIONAL_ONLY_SCOPE}_minutes_last_7d"] == 90
    no_events = module.build_scope_exposure_features(
        timeline,
        pd.DataFrame({"tm_player_id": [3], "date": ["2017-08-05"]}),
        scopes=("club_competitive",),
    )
    assert no_events.loc[0, "club_competitive_matches_last_28d"] == 0
    assert module.build_scope_exposure_features(
        timeline,
        targets.iloc[0:0],
        scopes=("club_competitive",),
    ).empty

    comparison = module.exposure_scope_comparison(features)
    assert comparison.set_index("metric").loc["rows_changed_previous_7d_minutes", "value"] == 1
    frozen_comparison = module.frozen_baseline_scope_comparison(targets, features)
    assert frozen_comparison.set_index("metric").loc["rows_changed_previous_7d_minutes", "value"] == 1
    with pytest.raises(KeyError, match="scope features"):
        module.exposure_scope_comparison(pd.DataFrame())
    with pytest.raises(KeyError, match="target matches"):
        module.frozen_baseline_scope_comparison(targets.drop(columns="all_minutes_last_7d"), features)


def test_window_sequence_geocodes_and_travel(load_src_module):
    module = load_src_module("26_build_public_match_timeline.py")
    dates = pd.to_datetime(["2017-01-01", "2017-01-20"]).to_numpy(dtype="datetime64[ns]")
    assert module._window_values(dates, pd.Series([90, 60]).to_numpy(), pd.to_datetime(["2017-01-25"]).to_numpy(dtype="datetime64[ns]"), 7)[0] == 60
    events = pd.DataFrame({"date": pd.to_datetime(["2017-01-01", "2017-01-03", "2017-01-25"]), "n_matches": [1, 1, 1]})
    days, sequence = module._prior_match_features(events, pd.to_datetime(["2017-01-04", "2017-01-26", "2016-12-31"]).to_numpy(dtype="datetime64[ns]"))
    assert days.tolist()[:2] == [1.0, 1.0]
    assert sequence.tolist() == [2.0, 1.0, 0.0]
    no_prior_days, no_prior_sequence = module._prior_match_features(
        events,
        pd.to_datetime(["2016-12-31"]).to_numpy(dtype="datetime64[ns]"),
    )
    assert pd.isna(no_prior_days[0]) and no_prior_sequence[0] == 0

    timeline = pd.DataFrame(
        {
            "tm_player_id": [1, 1],
            "match_key": ["a", "b"],
            "date": pd.to_datetime(["2017-01-01", "2017-01-03"]),
            "venue_key": ["venue-a", "venue-b"],
            "stadium_id": [pd.NA, pd.NA],
            "stadium_name": ["A", "B"],
        }
    )
    template = module.venue_geocode_template(timeline)
    assert template["evidence_status"].eq("unresolved").all()
    verified = pd.DataFrame(
        {
            "venue_key": ["venue-a", "venue-b"],
            "latitude": [0.0, 0.0],
            "longitude": [0.0, 1.0],
            "source_url": ["wiki-a", "wiki-b"],
            "match_confidence": ["exact", "exact"],
            "timezone_offset_hours": [0, 1],
        }
    )
    template = module.venue_geocode_template(timeline, verified)
    assert template["evidence_status"].eq("verified").all()
    with pytest.raises(KeyError, match="verified geocodes"):
        module.venue_geocode_template(timeline, verified.drop(columns="latitude"))
    assert module.great_circle_km(None, 0, 0, 1) is pd.NA
    assert 110 < module.great_circle_km(0, 0, 0, 1) < 112
    travelled = module.add_geographic_travel_proxies(timeline, verified)
    assert pd.isna(travelled.loc[0, "geographic_travel_km"])
    assert travelled.loc[1, "geographic_timezone_change_hours"] == 1
    audit = module.geographic_travel_coverage_audit(travelled, template)
    assert audit.set_index("metric").loc["travel_proxy_usable", "value"]
    empty_audit = module.geographic_travel_coverage_audit(
        travelled.assign(geographic_travel_km=pd.NA, geographic_timezone_change_hours=pd.NA),
        template.assign(evidence_status="unresolved"),
    )
    assert not empty_audit.set_index("metric").loc["travel_proxy_usable", "value"]
    with pytest.raises(KeyError, match="geocodes"):
        module.add_geographic_travel_proxies(timeline, verified.drop(columns="longitude"))
    with pytest.raises(KeyError, match="timeline"):
        module.geographic_travel_coverage_audit(timeline, template)
