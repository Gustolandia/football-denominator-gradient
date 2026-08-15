import json

import numpy as np
import pandas as pd
import pytest


def national_appearances():
    return pd.DataFrame(
        {
            "tm_player_id": [1, 1],
            "game_id": [10, 11],
            "competition_id": ["EURO", "EURO"],
            "season": [2024, 2024],
            "team_id": [100, 100],
            "opponent_team_id": [200, 300],
            "minutes_played": [90, np.nan],
            "date": ["2024-06-01", "2024-06-05"],
        }
    )


def test_baseline_manifest_and_schedule_key(load_src_module, tmp_path):
    module = load_src_module("27_public_data_v4_audits.py")
    (tmp_path / "transfermarkt_datasets_20240101").mkdir()
    (tmp_path / "transfermarkt_datasets_20250101").mkdir()
    assert module.newest_snapshot_dir(tmp_path).name.endswith("20250101")
    with pytest.raises(FileNotFoundError):
        module.newest_snapshot_dir(tmp_path / "none")
    path = tmp_path / "input.txt"
    path.write_text("baseline", encoding="utf-8")
    manifest = module.baseline_input_manifest([path], "commit")
    assert manifest["baseline_commit"] == "commit"
    assert len(manifest["files"][0]["sha256"]) == 64
    with pytest.raises(FileNotFoundError):
        module.baseline_input_manifest([tmp_path / "missing"], "commit")
    key = module._schedule_match_key(national_appearances())
    assert key.iloc[0] == "100|200|EURO|2024-06-01"


def test_coverage_audit_gate_and_template(load_src_module):
    module = load_src_module("27_public_data_v4_audits.py")
    appearances = national_appearances()
    log = pd.DataFrame({"tm_player_id": [1], "status": ["downloaded"]})
    duplicates = pd.DataFrame({"resolution": ["single_source"]})
    detailed, gate = module.coverage_audit(appearances, log, duplicates)
    assert detailed.loc[0, "official_schedule_audit_available"] == False
    assert detailed.loc[0, "independent_schedule_audit_available"] == False
    assert gate.loc[0, "decision"] == "sensitivity_only"
    assert gate.loc[0, "interval_method"] == "wilson_95"
    assert gate.set_index("metric").loc[
        "verified_match_coverage_percent", "denominator"
    ] == 2
    template = module.official_schedule_template(appearances)
    assert template["verified"].eq(False).all()

    schedule = template.copy()
    schedule["verified"] = True
    schedule["official_source_url"] = "https://official"
    complete = appearances.copy()
    complete["minutes_played"] = 90
    detailed, gate = module.coverage_audit(complete, log, duplicates, schedule)
    assert detailed["percent_verified_match_coverage"].eq(100).all()
    assert gate["primary_v4_exposure_allowed"].all()
    independent = schedule.drop(columns="official_source_url").copy()
    _, independent_gate = module.coverage_audit(
        complete,
        log,
        duplicates,
        schedule,
        independent_schedule=independent,
    )
    assert independent_gate.set_index("metric").loc[
        "independent_match_coverage_percent", "value"
    ] == 100
    assert independent_gate.set_index("metric").loc[
        "independent_match_coverage_percent", "numerator"
    ] == 2
    complete["is_senior_competitive"] = [True, False]
    record_audit = pd.DataFrame(
        {
            "participation_state": ["played", "played", "played"],
            "minutes_played": [90, 45, np.nan],
            "competition_type_id": [11, 19, 11],
            "competition_id": ["EURO", "WCQ", "FS"],
        }
    )
    _, raw_gate = module.coverage_audit(
        complete, log, duplicates, schedule, record_audit=record_audit
    )
    values = raw_gate.set_index("metric")["value"]
    assert values["primary_senior_competitive_nonmissing_minutes_percent"] == 100
    assert values["all_played_national_records_nonmissing_minutes_percent"] == pytest.approx(
        200 / 3
    )
    counts = raw_gate.set_index("metric")[["numerator", "denominator"]]
    assert tuple(
        counts.loc["primary_senior_competitive_nonmissing_minutes_percent"]
    ) == (2, 2)
    assert tuple(
        counts.loc["all_played_national_records_nonmissing_minutes_percent"]
    ) == (2, 3)
    no_played = record_audit.assign(participation_state="bench")
    _, empty_raw_gate = module.coverage_audit(
        complete, log, duplicates, schedule, record_audit=no_played
    )
    assert empty_raw_gate.loc[
        empty_raw_gate["metric"].eq("primary_senior_competitive_nonmissing_minutes_percent"),
        "value",
    ].item() == 0
    with pytest.raises(KeyError, match="national record audit"):
        module.coverage_audit(
            complete,
            log,
            duplicates,
            schedule,
            record_audit=record_audit.drop(columns="competition_type_id"),
        )
    with pytest.raises(KeyError, match="independent schedule"):
        module.coverage_audit(
            complete,
            log,
            duplicates,
            schedule,
            independent_schedule=independent.drop(columns="verified"),
        )
    bad_log = pd.DataFrame({"tm_player_id": [1], "status": ["error"]})
    bad_dups = pd.DataFrame({"resolution": ["unresolved_minutes_conflict_excluded"]})
    _, bad_gate = module.coverage_audit(complete, bad_log, bad_dups, schedule)
    assert bad_gate["decision"].eq("sensitivity_only").all()
    with pytest.raises(KeyError, match="international appearances"):
        module.coverage_audit(appearances.drop(columns="team_id"), log, duplicates)


def test_duration_and_outcome_validation_queue(load_src_module):
    module = load_src_module("27_public_data_v4_audits.py")
    assert module.parse_reported_duration(None) != module.parse_reported_duration(None)
    assert module.parse_reported_duration({"days": 12}) == 12
    assert module.parse_reported_duration("{'days': 28}") == 28
    assert module.parse_reported_duration("not python") != module.parse_reported_duration("not python")
    assert module.parse_reported_duration(5) != module.parse_reported_duration(5)
    assert module.parse_reported_duration("{'other': 1}") != module.parse_reported_duration("{'other': 1}")
    injuries = pd.DataFrame(
        {
            "tm_player_id": [1, 2, 3, 4],
            "injury_spell_id": [1, 2, 3, 4],
            "start_date": ["2024-01-01"] * 4,
            "end_date": ["2024-01-02"] * 4,
            "injury_desc": ["toe injury", "hamstring injury", "unknown injury", "bruise"],
            "durationDetails": ["{'days': 28}", "{'days': 10}", "{'days': 2}", "{'days': 3}"],
        }
    )
    queue = module.outcome_validation_queue(injuries, per_stratum=1)
    assert set(queue["validation_stratum"]) == {
        "reported_absence_ge28d",
        "muscle_tendon_description",
        "ambiguous_description",
        "unmatched_other_description",
    }
    assert queue["official_evidence_grade"].eq("unreviewed").all()
    with pytest.raises(KeyError, match="cleaned injuries"):
        module.outcome_validation_queue(injuries.drop(columns="injury_desc"))


def selection_inputs():
    games = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "competition_id": ["GB1"] * 4,
            "season": [2024] * 4,
            "date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
            "home_club_id": [10, 10, 10, 10],
            "away_club_id": [20, 20, 20, 20],
        }
    )
    appearances = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "game_id": [1, 2, 4],
            "player_club_id": [10, 10, 10],
            "date": ["2024-01-01", "2024-01-08", "2024-01-22"],
            "minutes_played": [90, 45, 90],
        }
    )
    transfers = pd.DataFrame(
        {
            "player_id": [1, 1],
            "to_club_id": [10, 99],
            "from_club_id": [98, 10],
            "transfer_date": ["2023-12-20", "2024-01-23"],
        }
    )
    daily = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 1, 1],
            "date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
            "available_for_injury_risk": [True, True, True, True],
            "all_minutes_last_7d": [0, 90, 45, 0],
            "prior_n_spells": [0, 0, 1, 1],
        }
    )
    return appearances, games, transfers, daily


def test_membership_risk_set_and_weights(load_src_module):
    module = load_src_module("27_public_data_v4_audits.py")
    appearances, games, transfers, daily = selection_inputs()
    intervals = module.epl_membership_intervals(appearances, games, transfers, [1])
    assert intervals.loc[0, "membership_start"] == pd.Timestamp("2023-12-20")
    assert intervals.loc[0, "membership_end"] == pd.Timestamp("2024-01-23")
    no_transfers = module.epl_membership_intervals(appearances, games, pd.DataFrame(), [1])
    assert no_transfers["arrival_date"].isna().all()
    risk, resolution = module.build_selection_risk_set(
        appearances,
        games,
        transfers,
        [1],
        daily,
        return_resolution_audit=True,
    )
    assert risk["played_any_minutes"].sum() == 3
    assert risk["plausibly_available"].all()
    assert resolution.loc[
        resolution["metric"].eq("unique_player_date_gate"), "passes_gate"
    ].item()
    default_risk = module.build_selection_risk_set(
        appearances, games, transfers, [1], daily
    )
    pd.testing.assert_frame_equal(default_risk, risk)
    with pytest.raises(KeyError, match="player-day panel"):
        module.build_selection_risk_set(appearances, games, transfers, [1], daily.drop(columns="prior_n_spells"))

    no_fit, no_fit_diag = module.fit_selection_weights(risk.assign(played_any_minutes=1))
    assert no_fit["ipw_usable"].eq(False).all()
    assert no_fit_diag.loc[0, "value"] == "not_estimable"
    weighted, diagnostics = module.fit_selection_weights(risk)
    assert weighted["selection_probability"].notna().all()
    assert diagnostics.loc[0, "value"] == "estimable"
    assert module.sqrt_safe(-1) == 0
    assert module.sqrt_safe(4) == 2
    assert module._weighted_standardised_difference(pd.Series([1, 2]), pd.Series([True, True]), pd.Series([1, 1])) != module._weighted_standardised_difference(pd.Series([1, 2]), pd.Series([True, True]), pd.Series([1, 1]))
    assert module._weighted_standardised_difference(pd.Series([1, 2]), pd.Series([True, False]), pd.Series([0, 1])) != module._weighted_standardised_difference(pd.Series([1, 2]), pd.Series([True, False]), pd.Series([0, 1]))
    assert module._weighted_standardised_difference(pd.Series([1, 1]), pd.Series([True, False]), pd.Series([1, 1])) == 0
    with pytest.raises(KeyError, match="selection risk set"):
        module.fit_selection_weights(risk.drop(columns="season"))


def test_selection_opportunity_overlap_resolution(load_src_module):
    module = load_src_module("27_public_data_v4_audits.py")
    risk = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "player_club_id": [10, 20, 10, 20, 10, 20, 10, 10],
            "game_id": [1, 2, 3, 4, 5, 6, 7, 7],
            "played_any_minutes": [1, 0, 0, 0, 1, 1, 0, 0],
        }
    )
    resolved, audit = module.resolve_selection_opportunities(risk)
    assert list(resolved["tm_player_id"]) == [1, 4]
    assert resolved["opportunity_resolution"].tolist() == [
        "observed_club_selected_from_overlap",
        "unique_public_membership",
    ]
    values = audit.set_index("metric")["value"]
    assert values["exact_duplicate_rows_removed"] == 1
    assert values["overlaps_resolved_by_observed_club"] == 1
    assert values["unresolved_zero_appearance_dates_excluded"] == 1
    assert values["unresolved_multiple_appearance_dates_excluded"] == 1
    assert not audit.loc[
        audit["metric"].eq("unresolved_multiple_appearance_dates_excluded"),
        "passes_gate",
    ].item()

    with pytest.raises(KeyError, match="selection opportunities"):
        module.resolve_selection_opportunities(risk.drop(columns="game_id"))
