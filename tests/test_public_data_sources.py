"""Tests for public-source acquisition and match reconciliation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from public_data_sources import (
    _assign_external_names,
    _seed_team_crosswalk,
    _split_lineup_entries,
    acquire_independent_results,
    acquire_worldcup_lineups,
    download_file,
    harmonize_national_matches,
    newest_independent_snapshot,
    newest_worldcup_lineup_snapshot,
    normalize_person_name,
    normalize_team_name,
    parse_openfootball_worldcup_file,
    parse_openfootball_worldcup_snapshot,
    public_source_catalog,
    sha256_file,
    validate_worldcup_lineups,
)


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = 0

    def get(self, _url, timeout):
        assert timeout == 60
        self.calls += 1
        return _Response(next(self.payloads))


def _appearances():
    return pd.DataFrame(
        [
            {"game_id": "1", "date": "2024-01-01", "team_id": 10, "opponent_team_id": 20, "team_name": "Türkiye", "opponent_team_name": pd.NA, "team_venue": "away", "team_goals": 1, "opponent_goals": 0, "team_level": "senior", "competition_id": "FS"},
            {"game_id": "2", "date": "2024-01-03", "team_id": 20, "opponent_team_id": 30, "team_name": pd.NA, "opponent_team_name": pd.NA, "team_venue": "away", "team_level": "senior", "competition_id": "Q"},
            {"game_id": "3", "date": "2024-01-04", "team_id": 99, "opponent_team_id": 98, "team_name": pd.NA, "opponent_team_name": pd.NA, "team_venue": "unknown", "team_level": "youth_or_olympic", "competition_id": "U21"},
        ]
    )


def _results():
    return pd.DataFrame(
        [
            {"date": "2024-01-01", "home_team": "Turkey", "away_team": "France", "home_score": 1, "away_score": 0, "tournament": "Friendly", "city": "Ankara", "country": "Turkey", "neutral": False},
            {"date": "2024-01-03", "home_team": "Spain", "away_team": "France", "home_score": 0, "away_score": 0, "tournament": "Qualifier", "city": "Madrid", "country": "Spain", "neutral": False},
        ]
    )


def test_download_hash_snapshot_and_catalog(tmp_path, monkeypatch):
    session = _Session([b"abc"])
    target = tmp_path / "file.csv"
    assert download_file("https://example.test/file", target, session) == target
    assert download_file("https://example.test/file", target, session) == target
    assert session.calls == 1
    assert sha256_file(target) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(FileNotFoundError):
        newest_independent_snapshot(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        newest_worldcup_lineup_snapshot(tmp_path / "missing")
    catalog = public_source_catalog()
    assert {
        "accepted_independent_validation",
        "accepted_independent_player_validation",
        "rejected_automated_acquisition",
        "rejected_usage_constraints",
    }.issubset(set(catalog.decision))
    assert catalog.loc[
        catalog.source.eq("Transfermarkt datasets"), "reason_or_limit"
    ].str.contains("zero cohort exposure rows").all()
    assert {
        "rejected_scope_and_access_mismatch",
        "rejected_schema_not_dataset",
        "external_methodology_benchmark_only",
    }.issubset(set(catalog.decision))

    csv = b"date,home_team,away_team,tournament,city,country,neutral\n2024-01-01,A,B,Friendly,X,Y,FALSE\n"
    shootouts_csv = b"date,home_team,away_team,winner,first_shooter\n2024-01-01,A,B,A,A\n"
    license_text = b"CC0"
    payloads = {
        "results.csv": csv,
        "shootouts.csv": shootouts_csv,
        "LICENSE": license_text,
    }
    monkeypatch.setattr("public_data_sources.download_file", lambda url, destination: (destination.parent.mkdir(parents=True, exist_ok=True), destination.write_bytes(payloads[destination.name]), destination)[2])
    snapshot = acquire_independent_results(tmp_path, retrieval_date="20240102")
    assert newest_independent_snapshot(tmp_path) == snapshot
    manifest = json.loads((snapshot / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 1
    assert manifest["shootout_rows"] == 1
    assert manifest["license"] == "CC0-1.0"

    monkeypatch.setattr(
        "public_data_sources.download_file",
        lambda url, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_text("CC0" if destination.name == "LICENSE.md" else "= World Cup", encoding="utf-8"),
            destination,
        )[2],
    )
    lineup_snapshot = acquire_worldcup_lineups(tmp_path, retrieval_date="20240103")
    assert newest_worldcup_lineup_snapshot(tmp_path) == lineup_snapshot
    lineup_manifest = json.loads(
        (lineup_snapshot / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert lineup_manifest["source_commit"]
    assert set(lineup_manifest["files"]) == {
        "2018_worldcup.txt",
        "2022_worldcup.txt",
        "LICENSE.md",
    }


def test_normalization_and_harmonization_are_strict_and_auditable():
    assert normalize_team_name(pd.NA) == ""
    assert normalize_team_name("Bosnia-Herzegovina") == "bosnia and herzegovina"
    assert normalize_person_name("Joãoquim Félixson") == "joaoquim felixson"
    assert normalize_person_name(pd.NA) == ""
    shootouts = pd.DataFrame(
        [{"date": "2024-01-01", "home_team": "Turkey", "away_team": "France"}]
    )
    harmonized, validation, crosswalk = harmonize_national_matches(
        _appearances(), _results(), shootouts
    )
    first = harmonized.loc[harmonized.game_id.eq("1")].iloc[0]
    second = harmonized.loc[harmonized.game_id.eq("2")].iloc[0]
    youth = harmonized.loc[harmonized.game_id.eq("3")].iloc[0]
    assert first.opponent_team_name == "France"
    assert second.team_name == "France"
    assert second.opponent_team_name == "Spain"
    assert first.independent_schedule_verified
    assert second.external_city == "Madrid"
    assert pd.isna(youth.team_name)
    assert validation.verified.sum() == 2
    assert bool(validation.loc[validation.game_id.eq("1"), "score_agreement"].iloc[0])
    assert bool(
        validation.loc[
            validation.game_id.eq("1"), "independent_shootout_match"
        ].iloc[0]
    )
    assert validation.loc[
        validation.game_id.eq("1"), "score_orientation_method"
    ].iloc[0] == "matched_team_names"
    assert bool(
        validation.loc[
            validation.game_id.eq("1"), "score_internal_consistency"
        ].iloc[0]
    )
    assert validation.loc[
        validation.game_id.eq("1"), "score_discrepancy_reason"
    ].iloc[0] == "exact_agreement"
    assert set(crosswalk.national_team_id) == {10, 20, 30}


def test_harmonization_validates_columns_and_handles_ambiguity_and_no_metadata():
    with pytest.raises(KeyError, match="national appearances missing"):
        harmonize_national_matches(pd.DataFrame(), _results())
    with pytest.raises(KeyError, match="independent results missing"):
        harmonize_national_matches(_appearances(), pd.DataFrame())
    with pytest.raises(KeyError, match="independent shootouts missing"):
        harmonize_national_matches(_appearances(), _results(), pd.DataFrame())

    apps = _appearances().iloc[[0]].copy()
    ambiguous = pd.concat([_results().iloc[[0]], _results().iloc[[0]]], ignore_index=True)
    harmonized, validation, _ = harmonize_national_matches(apps, ambiguous)
    assert not harmonized.independent_schedule_verified.any()
    assert validation.validation_method.eq("ambiguous").all()

    unmatched = _results().assign(date="2020-01-01")
    harmonized, validation, _ = harmonize_national_matches(apps, unmatched)
    assert not harmonized.independent_schedule_verified.any()
    assert validation.validation_method.eq("unmatched").all()

    conflicting = pd.DataFrame(
        [
            {**_appearances().iloc[0].to_dict(), "opponent_team_name": "France", "validation_team_goals": 1, "validation_opponent_goals": 0, "validation_score_source": "full_match_on_pitch"},
            {**_appearances().iloc[0].to_dict(), "team_id": 20, "opponent_team_id": 10, "team_name": "France", "opponent_team_name": "Turkey", "validation_team_goals": 1, "validation_opponent_goals": 1, "validation_score_source": "full_match_on_pitch"},
        ]
    )
    _, validation, _ = harmonize_national_matches(conflicting, _results())
    assert not validation["score_values_available"].any()
    assert validation["score_internal_consistency"].eq(False).all()
    assert validation["score_discrepancy_reason"].eq(
        "conflicting_player_level_score_pairs"
    ).all()


def test_parse_and_validate_openfootball_worldcup_lineups(tmp_path):
    content = """= World Cup 2022

» Group stage
Sun Nov/20 2022 @ Stadium › City, Country
  Türkiye v France  1-0

Türkiye: Joãoquim Félixson (70' Fresh Player), Full Match
France: Other Player, Away Starter (90+2' Late Player)
"""
    first = tmp_path / "2022_worldcup.txt"
    first.write_text(content, encoding="utf-8")
    parsed = parse_openfootball_worldcup_file(first, 2022)
    assert len(parsed) == 6
    starter = parsed.loc[parsed.player_key.eq("joaoquim felixson")].iloc[0]
    substitute = parsed.loc[parsed.player_key.eq("fresh player")].iloc[0]
    late = parsed.loc[parsed.player_key.eq("late player")].iloc[0]
    assert starter.is_starter and starter.approx_minutes == 70
    assert not substitute.is_starter and substitute.approx_minutes == 20
    assert late.approx_minutes == 0

    second = tmp_path / "2018_worldcup.txt"
    second.write_text(content.replace("2022", "2018"), encoding="utf-8")
    combined = parse_openfootball_worldcup_snapshot(tmp_path)
    assert set(combined.tournament_year) == {2018, 2022}

    audit = pd.DataFrame(
        [
            {"tm_player_id": 1, "game_id": "10", "date": "2022-11-20", "team_id": 10, "opponent_team_id": 20, "competition_id": "FIWC", "participation_state": "played", "minutes_played": 70, "is_starter": True},
            {"tm_player_id": 2, "game_id": "10", "date": "2022-11-20", "team_id": 10, "opponent_team_id": 20, "competition_id": "FIWC", "participation_state": "played", "minutes_played": 10, "is_starter": False},
            {"tm_player_id": 3, "game_id": "11", "date": "2022-11-20", "team_id": 10, "opponent_team_id": 20, "competition_id": "FS", "participation_state": "played", "minutes_played": 90, "is_starter": True},
        ]
    )
    cohort = pd.DataFrame(
        {"tm_player_id": [1, 2, 3], "name": ["Joãoquim Félixson", "Missing Person", "Full Match"]}
    )
    crosswalk = pd.DataFrame(
        {"national_team_id": [10, 20], "normalized_team_name": ["turkey", "france"]}
    )
    validated = validate_worldcup_lineups(audit, cohort, crosswalk, parsed)
    assert len(validated) == 2
    assert validated.loc[validated.tm_player_id.eq(1), "source_player_found"].iloc[0]
    assert validated.loc[validated.tm_player_id.eq(1), "minutes_within_5"].iloc[0]
    assert not validated.loc[validated.tm_player_id.eq(2), "source_player_found"].iloc[0]


def test_worldcup_validation_rejects_missing_schema():
    with pytest.raises(KeyError, match="national record audit missing"):
        validate_worldcup_lineups(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def test_parser_and_crosswalk_edge_paths(tmp_path):
    assert _split_lineup_entries("") == []
    assert _split_lineup_entries("Player A (70' Player B, Player C), Player D,") == [
        "Player A (70' Player B, Player C)",
        "Player D",
    ]
    empty = tmp_path / "empty.txt"
    empty.write_text("= World Cup\nOfficials: Nobody\n", encoding="utf-8")
    assert parse_openfootball_worldcup_file(empty, 2022).empty

    multiline = tmp_path / "multiline.txt"
    multiline.write_text(
        """Sun Nov/20 2022 @ Stadium › City, Country
  Turkey v France  1-0

Officials: Referee
Turkey: Player A,
   Player B
End of match
""",
        encoding="utf-8",
    )
    parsed = parse_openfootball_worldcup_file(multiline, 2022)
    assert set(parsed["player_name"]) == {"Player A", "Player B"}

    ambiguous_names = pd.DataFrame(
        [
            {
                "team_id": 1,
                "team_name": "Alpha",
                "opponent_team_id": 2,
                "opponent_team_name": "Beta",
            },
            {
                "team_id": 1,
                "team_name": "Gamma",
                "opponent_team_id": pd.NA,
                "opponent_team_name": pd.NA,
            },
            {
                "team_id": "not-an-id",
                "team_name": "Ignored",
                "opponent_team_id": 3,
                "opponent_team_name": "",
            },
        ]
    )
    mapping, labels = _seed_team_crosswalk(ambiguous_names)
    assert 1 not in mapping
    assert mapping[2] == "beta"
    assert labels[2] == "Beta"


def test_external_name_orientation_paths():
    external = pd.Series(
        {
            "home_key": "home",
            "away_key": "away",
            "home_team": "Home",
            "away_team": "Away",
        }
    )
    assert _assign_external_names(
        pd.Series({"team_id": pd.NA, "opponent_team_id": 2}), external, {}
    ) == []
    opponent_home = _assign_external_names(
        pd.Series(
            {
                "team_id": 1,
                "opponent_team_id": 2,
                "team_name": "Unknown",
                "opponent_team_name": "Home",
            }
        ),
        external,
        {},
    )
    assert opponent_home[0][1] == "away"
    opponent_away = _assign_external_names(
        pd.Series(
            {
                "team_id": 1,
                "opponent_team_id": 2,
                "team_name": "Unknown",
                "opponent_team_name": "Away",
            }
        ),
        external,
        {},
    )
    assert opponent_away[0][1] == "home"
    assert _assign_external_names(
        pd.Series(
            {
                "team_id": 1,
                "opponent_team_id": 2,
                "team_name": "Unknown",
                "opponent_team_name": "Also Unknown",
            }
        ),
        external,
        {},
    ) == []


def test_harmonization_crosswalk_iteration_and_empty_validation(monkeypatch):
    youth_only = _appearances().loc[lambda frame: frame["team_level"].eq("youth_or_olympic")]
    harmonized, validation, _ = harmonize_national_matches(youth_only, _results())
    assert validation.empty
    assert not harmonized["independent_schedule_verified"].any()

    call_count = {"value": 0}

    def always_add(_row, _external, _mapping):
        call_count["value"] += 1
        identifier = 1000 + call_count["value"]
        return [(identifier, f"team {identifier}", f"Team {identifier}")]

    monkeypatch.setattr("public_data_sources._assign_external_names", always_add)
    harmonize_national_matches(_appearances(), _results())
    assert call_count["value"] >= 8

    monkeypatch.setattr(
        "public_data_sources._assign_external_names",
        lambda _row, _external, _mapping: [(777, "alpha", "Alpha"), (777, "beta", "Beta")],
    )
    harmonize_national_matches(_appearances(), _results())

    monkeypatch.setattr(
        "public_data_sources._assign_external_names",
        lambda _row, _external, _mapping: [(10, "france", "France")],
    )
    harmonized, _, _ = harmonize_national_matches(_appearances(), _results())
    assert harmonized.loc[harmonized["team_id"].eq(10), "team_name"].eq("Türkiye").all()
