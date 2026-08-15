import json
import gzip
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
import requests


def performance_payload(
    *,
    national=True,
    date="2018-01-02T20:00:00+00:00",
    played=True,
    minutes=90,
):
    """Return a small public-performance response fixture."""
    return {
        "data": {
            "performance": [
                {
                    "gameInformation": {
                        "isNationalGame": national,
                        "gameId": "10",
                        "competitionId": "EURO",
                        "competitionGroupId": "ECQ",
                        "competitionTypeId": 11,
                        "seasonId": 2017,
                        "stadiumId": 20,
                        "gameDuration": 90,
                        "date": {
                            "dateTimeUTC": date,
                            "dateTimeLocalized": date,
                            "isTimeDefined": True,
                        },
                    },
                    "clubsInformation": {
                        "club": {
                            "clubId": "1",
                            "venue": "home",
                            "goalsTotal": "2",
                            "opponentGoalsTotal": "1",
                        },
                        "opponent": {"clubId": "2"},
                    },
                    "statistics": {
                        "generalStatistics": {
                            "participationState": "played" if played else "bench",
                        },
                        "playingTimeStatistics": {
                            "playedMinutes": minutes,
                            "isStarting": True,
                        },
                        "goalStatistics": {
                            "teamGoalsOnThePitch": 2,
                            "opponentGoalsOnThePitch": 1,
                        },
                    },
                }
            ]
        }
    }


def test_hash_schema_records_and_json(load_src_module, tmp_path):
    module = load_src_module("25_public_data_v4.py")
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("first,second\n1,2\n", encoding="utf-8")
    text_path = tmp_path / "notes.txt"
    text_path.write_text("hello", encoding="utf-8")
    gz_path = tmp_path / "rows.csv.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write("compressed\n1\n")

    assert len(module.sha256_file(csv_path, chunk_size=1)) == 64
    assert module.csv_schema(csv_path) == ["first", "second"]
    assert module.file_record(csv_path, "https://source") ["columns"] == ["first", "second"]
    assert module.file_record(gz_path)["columns"] == ["compressed"]
    assert "columns" not in module.file_record(text_path)
    out_path = tmp_path / "deep" / "audit.json"
    module.write_json(out_path, {"b": 2, "a": 1})
    assert json.loads(out_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert module.utc_now().endswith("+00:00")


class FakeDownloadResponse:
    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def iter_content(self, chunk_size):
        del chunk_size
        return iter(self.chunks)


def test_download_extract_and_snapshot_manifest(load_src_module, tmp_path):
    module = load_src_module("25_public_data_v4.py")
    archive_path = tmp_path / "snapshot.zip"
    module.download_snapshot("https://source", archive_path, get=lambda *args, **kwargs: FakeDownloadResponse([b"x", b"", b"y"]))
    assert archive_path.read_bytes() == b"xy"
    with pytest.raises(FileExistsError):
        module.download_snapshot("https://source", archive_path, get=lambda *args, **kwargs: FakeDownloadResponse([]))

    valid_archive = tmp_path / "valid.zip"
    with ZipFile(valid_archive, "w") as archive:
        archive.writestr("data.csv", "one\n1\n")
        archive.writestr("empty_directory/", "")
        archive.writestr("nested/two.txt", "two")
    extracted = module.extract_snapshot(valid_archive, tmp_path / "extracted")
    assert {path.name for path in extracted} == {"data.csv", "two.txt"}
    with pytest.raises(FileExistsError):
        module.extract_snapshot(valid_archive, tmp_path / "extracted")
    manifest = module.snapshot_manifest(valid_archive, extracted, "now", "commit", "terms")
    assert manifest["archive"]["source_url"] == module.SNAPSHOT_URL
    assert manifest["files"][0]["path"] == "data.csv"

    unsafe_archive = tmp_path / "unsafe.zip"
    with ZipFile(unsafe_archive, "w") as archive:
        archive.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Unsafe"):
        module.extract_snapshot(unsafe_archive, tmp_path / "unsafe")


def test_epl_seasons_and_cohort_manifest(load_src_module):
    module = load_src_module("25_public_data_v4.py")
    games = pd.DataFrame(
        {
            "game_id": [1, 2],
            "competition_id": ["GB1", "ES1"],
            "season": [2017, 2017],
            "home_club_id": [10, 11],
            "away_club_id": [20, 12],
        }
    )
    assert len(module.epl_club_seasons(games)) == 2
    with pytest.raises(KeyError, match="home_club_id"):
        module.epl_club_seasons(games.drop(columns="home_club_id"))

    panel = pd.DataFrame({"tm_player_id": [1, 2]})
    appearances = pd.DataFrame(
        {
            "player_id": [1, 2],
            "game_id": [1, 2],
            "player_club_id": [10, 11],
            "date": ["2017-08-01", "2017-08-02"],
        }
    )
    players = pd.DataFrame(
        {
            "player_id": [1, 2],
            "name": ["A", "B"],
            "country_of_citizenship": ["England", "Spain"],
            "current_national_team_id": [100, 200],
        }
    )
    transfers = pd.DataFrame({"player_id": [1], "transfer_date": ["2017-07-01"]})
    manifest = module.build_cohort_manifest(panel, appearances, games, players, transfers)
    assert manifest.loc[manifest["tm_player_id"].eq(1), "observed_club_seasons"].item() == "2017_10"
    assert manifest.loc[manifest["tm_player_id"].eq(2), "unresolved_cohort_id"].item()
    assert manifest.loc[manifest["tm_player_id"].eq(1), "recorded_transfer_dates"].item().startswith("2017-07-01")

    no_transfer_date = module.build_cohort_manifest(panel, appearances, games, players[["player_id"]], pd.DataFrame())
    assert no_transfer_date["recorded_transfer_dates"].eq("").all()
    no_player_id = module.build_cohort_manifest(panel, appearances, games, pd.DataFrame(), pd.DataFrame())
    assert no_player_id["name"].isna().all()
    with pytest.raises(KeyError, match="tm_player_id"):
        module.build_cohort_manifest(pd.DataFrame(), appearances, games, players, transfers)


def test_snapshot_national_appearance_audit_is_window_and_cohort_specific(
    load_src_module,
):
    module = load_src_module("25_public_data_v4.py")
    competitions = pd.DataFrame(
        {
            "competition_id": ["FIWC", "GB1"],
            "type": ["national_team_competition", "domestic_league"],
        }
    )
    games = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "competition_id": ["FIWC", "FIWC", "GB1"],
            "date": ["2022-12-18", "2026-06-01", "2022-12-18"],
        }
    )
    appearances = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 3],
            "player_id": [10, 11, 10, 10],
            "date": ["2022-12-18", "2022-12-18", "2026-06-01", "2022-12-18"],
        }
    )
    audit = module.audit_snapshot_national_appearances(
        competitions, games, appearances, [10]
    ).set_index("metric")
    assert audit.loc[
        "snapshot_national_appearance_rows_in_acquisition_window", "value"
    ] == 2
    assert audit.loc[
        "snapshot_cohort_national_appearance_rows_in_acquisition_window", "value"
    ] == 1
    assert audit.loc["snapshot_first_national_appearance_date", "value"] == "2022-12-18"
    with pytest.raises(KeyError, match="snapshot competitions"):
        module.audit_snapshot_national_appearances(
            competitions.drop(columns="type"), games, appearances, [10]
        )
    with pytest.raises(KeyError, match="snapshot games"):
        module.audit_snapshot_national_appearances(
            competitions, games.drop(columns="date"), appearances, [10]
        )
    with pytest.raises(KeyError, match="snapshot appearances"):
        module.audit_snapshot_national_appearances(
            competitions, games, appearances.drop(columns="date"), [10]
        )


def test_snapshot_national_appearance_audit_handles_no_player_rows(load_src_module):
    module = load_src_module("25_public_data_v4.py")
    competitions = pd.DataFrame(
        {"competition_id": ["FIWC"], "type": ["national_team_competition"]}
    )
    games = pd.DataFrame(
        {"game_id": [1], "competition_id": ["FIWC"], "date": ["2022-12-18"]}
    )
    appearances = pd.DataFrame(columns=["game_id", "player_id", "date"])
    audit = module.audit_snapshot_national_appearances(
        competitions, games, appearances, [10]
    ).set_index("metric")
    assert audit.loc["snapshot_national_appearance_rows_all_dates", "value"] == 0
    assert audit.loc["snapshot_first_national_appearance_date", "value"] == ""
    assert audit.loc["snapshot_last_national_appearance_date", "value"] == ""


def test_timestamp_and_normalisation_filters_rows(load_src_module):
    module = load_src_module("25_public_data_v4.py")
    assert module._timestamp_parts(None) == (pd.NaT, pd.NA, False)
    unknown_time = module._timestamp_parts({"dateTimeLocalized": "2018-01-02", "isTimeDefined": False})
    assert unknown_time[0] == pd.Timestamp("2018-01-02")
    assert unknown_time[1] is pd.NA and not unknown_time[2]
    timezone_local = module._timestamp_parts({"dateTimeLocalized": "2018-01-02T19:00:00+01:00", "isTimeDefined": False})
    assert timezone_local[0] == pd.Timestamp("2018-01-02")

    frame = module.normalise_national_performance(
        performance_payload(), 7, "url", "cache", "now"
    )
    assert frame.loc[0, "tm_player_id"] == 7
    assert frame.loc[0, "kickoff_time_known"]
    assert frame.loc[0, "is_starter"]
    assert frame.loc[0, "minutes_played"] == 90.0
    assert frame.loc[0, "team_goals"] == 2
    assert frame.loc[0, "opponent_goals"] == 1
    assert frame.loc[0, "game_duration_minutes"] == 90
    assert frame.loc[0, "team_goals_on_pitch"] == 2
    assert frame.loc[0, "opponent_goals_on_pitch"] == 1
    audit = module.national_performance_record_audit(
        performance_payload(played=False), 7, "url", "cache", "now"
    )
    assert not audit.loc[0, "retained_for_exposure"]
    assert audit.loc[0, "exclusion_reason"] == "not_played"
    missing_minutes = module.national_performance_record_audit(
        performance_payload(minutes=None), 7, "url", "cache", "now"
    )
    assert missing_minutes.loc[0, "exclusion_reason"] == "played_missing_minutes"
    assert pd.isna(missing_minutes.loc[0, "minutes_played"])
    retained = module.national_performance_record_audit(
        performance_payload(), 7, "url", "cache", "now"
    )
    assert retained.loc[0, "retained_for_exposure"]
    assert retained.loc[0, "exclusion_reason"] == ""
    assert module.normalise_national_performance(performance_payload(national=False), 7, "url", "cache", "now").empty
    assert module.normalise_national_performance(performance_payload(date="2016-01-02T00:00:00+00:00"), 7, "url", "cache", "now").empty
    assert module.normalise_national_performance(performance_payload(played=False), 7, "url", "cache", "now").empty
    assert module.normalise_national_performance(performance_payload(minutes=None), 7, "url", "cache", "now").empty
    with pytest.raises(ValueError, match="list-valued"):
        module.normalise_national_performance({"data": {"performance": {}}}, 7, "url", "cache", "now")


class FakePerformanceResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, response_by_player):
        self.headers = {}
        self.response_by_player = response_by_player

    def get(self, url, timeout):
        del timeout
        player_id = int(url.rsplit("/", 1)[-1])
        response = self.response_by_player[player_id]
        if isinstance(response, Exception):
            raise response
        return response


def test_acquisition_uses_cache_logs_downloads_and_errors(load_src_module, tmp_path):
    module = load_src_module("25_public_data_v4.py")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "1.json").write_text(json.dumps(performance_payload()), encoding="utf-8")
    session = FakeSession(
        {
            2: FakePerformanceResponse(performance_payload()),
            3: requests.ConnectionError("offline"),
        }
    )
    sleeps = []
    appearances, request_log = module.acquire_national_appearances(
        [1, 2, 3],
        cache_dir,
        tmp_path / "request_log.csv",
        session=session,
        sleep=sleeps.append,
        request_interval_seconds=0.1,
        retries=2,
        record_audit_path=tmp_path / "record_audit.csv",
    )
    assert len(appearances) == 2
    assert request_log.set_index("tm_player_id").loc[1, "status"] == "cached"
    assert request_log.set_index("tm_player_id").loc[2, "status"] == "downloaded"
    assert request_log.set_index("tm_player_id").loc[3, "status"] == "error"
    assert request_log.set_index("tm_player_id").loc[1, "n_played_records"] == 1
    assert request_log.set_index("tm_player_id").loc[3, "n_national_records"] == 0
    assert len(pd.read_csv(tmp_path / "record_audit.csv")) == 2
    assert (cache_dir / "2.json").exists()
    assert len(sleeps) == 3

    empty, log = module.acquire_national_appearances(
        [],
        cache_dir,
        tmp_path / "empty.csv",
        session=session,
        request_interval_seconds=0,
        record_audit_path=tmp_path / "empty_record_audit.csv",
    )
    assert empty.empty and log.empty
    assert pd.read_csv(tmp_path / "empty_record_audit.csv").empty
    module.acquire_national_appearances(
        [1], cache_dir, tmp_path / "zero_interval.csv", session=session, request_interval_seconds=0
    )
    (cache_dir / "4.json").write_text(json.dumps(performance_payload(national=False)), encoding="utf-8")
    no_national_rows, _ = module.acquire_national_appearances(
        [4],
        cache_dir,
        tmp_path / "no_national_rows.csv",
        session=session,
        request_interval_seconds=0,
        record_audit_path=tmp_path / "no_national_record_audit.csv",
    )
    assert no_national_rows.empty
    assert pd.read_csv(tmp_path / "no_national_record_audit.csv").empty
