"""Tests for the boundary where audited records stop naming people."""

import pandas as pd
import pytest


def _identified() -> pd.DataFrame:
    """Two appearances by one player and one by another, as the audit sampled."""
    return pd.DataFrame(
        {
            "audit_id": ["8000001_20181212", "8000001_20190104", "8000002_20220821"],
            "tm_player_id": ["8000001", "8000001", "8000002"],
            "player_name": ["Ardal Kavanagh", "Ardal Kavanagh", "Brunel Aitchison"],
            "club_name": ["Manchester City FC", "Manchester City FC", "West Ham FC"],
            "date": ["2018-12-12", "2019-01-04", "2022-08-21"],
            "matchproxy_injury_desc": ["Muscle injury", "Calf strain", "Foot injury"],
            "independent_source_url": [
                "https://example.org/kavanagh-injury-update",
                "",
                "https://example.org/aitchison-latest",
            ],
            "independent_source_type": ["official_club", "", "independent_news"],
            "review_note": [
                "Kavanagh was withdrawn at half time; the club confirms a muscle problem.",
                "No qualifying source found for Ardal.",
                "Screened: the club played no fixture in the gap.",
            ],
        }
    )


def _map(module, frame) -> pd.DataFrame:
    keys = iter(f"{index:012X}" for index in range(100))
    return module.build_identity_map([frame], key_factory=lambda: next(keys))


def test_seasons_replace_dates_because_a_match_day_identifies(load_src_module):
    """A club plus a match day plus a body part is a singleton; a season is not."""
    module = load_src_module("audit_identity.py")
    seasons = module.season_label(
        pd.Series(["2018-12-12", "2019-01-04", "2022-07-01", "2022-06-30"])
    )
    # July starts a season, so January belongs to the year before it.
    assert list(seasons) == ["2018-19", "2018-19", "2022-23", "2021-22"]

    with pytest.raises(ValueError, match="must all parse"):
        module.season_label(pd.Series(["not a date"]))


def test_surrogates_are_drawn_once_per_record_and_once_per_player(load_src_module):
    """The record key joins a verdict to its row; the player key keeps the
    sample's clustering visible without naming who is repeated."""
    module = load_src_module("audit_identity.py")
    frame = _identified()
    identity_map = _map(module, frame)

    assert list(identity_map.columns) == list(module.IDENTITY_MAP_COLUMNS)
    assert len(identity_map) == 3
    assert identity_map["record_key"].nunique() == 3
    # Two rows are the same player, so two of the three share a player key.
    assert identity_map["player_key"].nunique() == 2
    assert identity_map["record_key"].str.startswith("A").all()
    assert identity_map["player_key"].str.startswith("P").all()

    with pytest.raises(ValueError, match="at least one audited frame"):
        module.build_identity_map([])

    conflicting = pd.DataFrame(
        {"audit_id": ["x", "x"], "tm_player_id": ["1", "2"]}
    )
    with pytest.raises(ValueError, match="cannot belong to two players"):
        module.build_identity_map([conflicting])

    with pytest.raises(KeyError, match="identity map source"):
        module.build_identity_map([pd.DataFrame({"audit_id": ["x"]})])


def test_keys_are_drawn_not_derived(load_src_module):
    """A hash of a public identifier is reversible by hashing public
    identifiers, and there are only a few thousand candidates. Two maps built
    from the same rows must therefore disagree."""
    module = load_src_module("audit_identity.py")
    frame = _identified()
    first = module.build_identity_map([frame])
    second = module.build_identity_map([frame])
    assert set(first["record_key"]) != set(second["record_key"])
    assert first["record_key"].str.len().eq(13).all()


def test_deidentified_frame_keeps_no_way_back(load_src_module):
    module = load_src_module("audit_identity.py")
    frame = _identified()
    identity_map = _map(module, frame)
    surnames = module.audited_surnames([frame])
    public = module.deidentify_audit_frame(frame, identity_map, surnames)

    for column in ("audit_id", "tm_player_id", "player_name", "club_name", "date"):
        assert column not in public.columns, column
    assert list(public.columns[:3]) == ["record_key", "player_key", "season"]
    assert list(public["season"]) == ["2018-19", "2018-19", "2022-23"]

    # The injury label survives: it is what the audit adjudicates, and attached
    # to an opaque key it describes nobody.
    assert list(public["matchproxy_injury_desc"]) == [
        "Muscle injury",
        "Calf strain",
        "Foot injury",
    ]

    # The URL is withheld because its slug carries the surname; whether a source
    # was found is what the verdict rests on, and that survives.
    assert "independent_source_url" not in public.columns
    assert list(public["independent_source_found"]) == [True, False, True]

    joined = " ".join(public["review_note"])
    for name in ("Kavanagh", "Ardal", "Aitchison", "Brunel"):
        assert name not in joined, name
    assert module.WITHHELD in joined
    # Scrubbing a name must not eat the reasoning around it.
    assert "withdrawn at half time" in joined
    assert "played no fixture in the gap" in joined


def test_deidentification_fails_rather_than_dropping_a_row(load_src_module):
    """A silently dropped row would shrink an audit whose whole value is that
    it reports what it could not resolve."""
    module = load_src_module("audit_identity.py")
    frame = _identified()
    identity_map = _map(module, frame)

    stranger = frame.assign(audit_id=["8000001_20181212", "8000001_20190104", "999_20200101"])
    with pytest.raises(KeyError, match="no surrogate for audit records"):
        module.deidentify_audit_frame(stranger, identity_map)

    with pytest.raises(KeyError, match="audit frame"):
        module.deidentify_audit_frame(pd.DataFrame({"x": [1]}), identity_map)

    with pytest.raises(KeyError, match="identity map"):
        module.deidentify_audit_frame(frame, pd.DataFrame({"audit_id": ["x"]}))


def test_frames_without_dates_urls_or_notes_still_de_identify(load_src_module):
    """The absence screen carries none of the three, and must still pass."""
    module = load_src_module("audit_identity.py")
    frame = _identified()
    identity_map = _map(module, frame)

    bare = frame[["audit_id", "tm_player_id"]].assign(club_fixtures_missed=[0, 1, 0])
    public = module.deidentify_audit_frame(bare, identity_map)
    assert list(public.columns) == ["record_key", "player_key", "club_fixtures_missed"]
    assert "season" not in public.columns
    assert "independent_source_found" not in public.columns


def test_name_scrubbing_ignores_fragments_too_short_to_identify(load_src_module):
    module = load_src_module("audit_identity.py")
    # A two-letter part would match inside ordinary words and destroy the note.
    assert module.scrub_names("Di Fixtura played on", ["Di"]) == "Di Fixtura played on"
    assert module.scrub_names("Kavanagh played on", ["Kavanagh"]).startswith(
        module.WITHHELD
    )
    # Matching is case-insensitive and whole-word only.
    assert module.scrub_names("KAVANAGH and Kavanaghson", ["Kavanagh"]) == (
        f"{module.WITHHELD} and Kavanaghson"
    )

    # "Di" is dropped as too short to match safely; the rest of the name is
    # still collected, so a two-letter particle cannot suppress a surname.
    named = pd.DataFrame({"player_name": ["Ardal Kavanagh", "Di Fixtura", None]})
    unnamed = pd.DataFrame({"other": [1]})
    parts = module.audited_surnames([named, unnamed])
    assert parts == ["Kavanagh", "Fixtura", "Ardal"]
    assert "Di" not in parts


def test_identity_map_round_trips_through_disk(load_src_module, tmp_path):
    module = load_src_module("audit_identity.py")
    frame = _identified()
    identity_map = _map(module, frame)
    path = tmp_path / "audit_identity_map.csv"
    identity_map.to_csv(path, index=False)

    loaded = module.load_identity_map(path)
    assert loaded.equals(identity_map)

    with pytest.raises(FileNotFoundError, match="identity map not found"):
        module.load_identity_map(tmp_path / "absent.csv")

    duplicated_records = pd.concat([identity_map, identity_map.iloc[[0]]])
    duplicated_records.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate audit_id"):
        module.load_identity_map(path)

    clashing = identity_map.copy()
    clashing.loc[1, "record_key"] = clashing.loc[0, "record_key"]
    clashing.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate record keys"):
        module.load_identity_map(path)

    missing_column = identity_map.drop(columns="player_key")
    missing_column.to_csv(path, index=False)
    with pytest.raises(KeyError, match="identity map"):
        module.load_identity_map(path)
