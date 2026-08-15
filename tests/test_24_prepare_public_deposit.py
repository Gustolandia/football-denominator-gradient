from pathlib import Path

import pandas as pd
import pytest


def test_sanitize_text_path_and_frame(load_src_module):
    module = load_src_module("24_prepare_public_deposit.py")
    assert module.sanitize_text("fragile_regular_tough_fragility") == (
        "higher_history_intermediate_history_lower_history_prior_history"
    )
    assert module.sanitize_text(10) == 10
    assert module.sanitize_path_name(Path("A4_fragile") / "tough.csv") == (
        Path("A4_higher_history") / "lower_history.csv"
    )

    frame = pd.DataFrame(
        {
            "fragility_group": ["fragile", "regular", "tough"],
            "count": [1, 2, 3],
        }
    )
    out = module.sanitize_frame(frame)
    assert out.columns.tolist() == ["prior_history_group", "count"]
    assert out["prior_history_group"].tolist() == [
        "higher_history",
        "intermediate_history",
        "lower_history",
    ]


def test_prepare_public_deposit_sanitizes_csvs_and_figures(load_src_module, tmp_path):
    module = load_src_module("24_prepare_public_deposit.py")
    results_dir = tmp_path / "data" / "processed" / "results"
    figures_dir = tmp_path / "manuscript" / "figures"
    results_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    stale_file = tmp_path / "public_deposit" / "stale.csv"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old")
    pd.DataFrame(
        {
            "fragility_group": ["fragile", "regular"],
            "note": ["Fragile label", "tough comparator"],
        }
    ).to_csv(results_dir / "matchproxy_fragile_summary.csv", index=False)
    (results_dir / "ignore_me.txt").write_text("not a result csv")
    (figures_dir / "A4_allinjury_crude_45min_tough.png").write_bytes(b"png")

    manifest = module.prepare_public_deposit(tmp_path)
    assert manifest["artifact_type"].tolist() == ["csv", "figure"]
    public_csv = (
        tmp_path
        / "public_deposit"
        / "data"
        / "processed"
        / "results"
        / "matchproxy_higher_history_summary.csv"
    )
    public_figure = (
        tmp_path
        / "public_deposit"
        / "manuscript"
        / "figures"
        / "A4_allinjury_crude_45min_lower_history.png"
    )
    assert public_csv.exists()
    assert public_figure.read_bytes() == b"png"
    assert not stale_file.exists()
    public_frame = pd.read_csv(public_csv)
    assert public_frame.columns.tolist() == ["prior_history_group", "note"]
    assert public_frame["prior_history_group"].tolist() == [
        "higher_history",
        "intermediate_history",
    ]
    assert public_frame["note"].tolist() == [
        "Higher history label",
        "lower_history comparator",
    ]
    manifest_path = tmp_path / "public_deposit" / "sanitization_manifest.csv"
    assert manifest_path.exists()


def test_prepare_public_deposit_exports_v4_and_manual_sources(load_src_module, tmp_path):
    """Every directory the manuscript or supplement cites must be deposited.

    ``data/manual`` is the critical case: no script regenerates it, so omitting
    it would leave the outcome audit uncheckable from the archive alone.
    """
    module = load_src_module("24_prepare_public_deposit.py")
    v4_dir = tmp_path / "data" / "processed" / "public_data_v4"
    manual_dir = tmp_path / "data" / "manual"
    v4_dir.mkdir(parents=True)
    manual_dir.mkdir(parents=True)
    pd.DataFrame({"metric": ["opportunities"], "value": [86281]}).to_csv(
        v4_dir / "selection_membership_resolution_audit.csv", index=False
    )
    pd.DataFrame({"audit_id": ["8000001_20181212"], "verdict": ["confirmed"]}).to_csv(
        manual_dir / "independent_same_day_event_audit.csv", index=False
    )

    module.prepare_public_deposit(tmp_path)

    deposit = tmp_path / "public_deposit"
    assert (
        deposit
        / "data"
        / "processed"
        / "public_data_v4"
        / "selection_membership_resolution_audit.csv"
    ).exists()
    assert (
        deposit / "data" / "manual" / "independent_same_day_event_audit.csv"
    ).exists()


def test_prepare_public_deposit_handles_missing_sources(load_src_module, tmp_path):
    module = load_src_module("24_prepare_public_deposit.py")
    manifest = module.prepare_public_deposit(tmp_path)
    assert manifest.empty
    assert (tmp_path / "public_deposit" / "sanitization_manifest.csv").exists()


def test_deposit_drops_names_and_replaces_person_identifiers(load_src_module):
    """The step named "sanitize" renamed history strata and left every
    identifier untouched, so the deposit named 56 players while the manuscript
    said no individual was identified in it."""
    module = load_src_module("24_prepare_public_deposit.py")
    frame = pd.DataFrame(
        {
            "tm_player_id": ["8000001", "8000001", "8000002", None],
            "player_name": ["Ardal Kavanagh", "Ardal Kavanagh", "X", "Y"],
            "player_club_id": [281, 281, 379, 379],
            "injury_desc": ["Muscle injury", "Calf strain", "Foot injury", None],
        }
    )
    keys = iter(f"P{index:03d}" for index in range(100))
    surrogates = {}
    public = module.deidentify_frame(frame, surrogates, key_factory=lambda: next(keys))

    assert "player_name" not in public.columns
    # One surrogate per player, not per row; a missing identifier stays missing.
    assert list(public["tm_player_id"][:3]) == ["P000", "P000", "P001"]
    assert pd.isna(public["tm_player_id"].iloc[3])
    assert surrogates == {"8000001": "P000", "8000002": "P001"}
    # The club is not a person and is left alone; so is the injury label, which
    # now describes an opaque key.
    assert list(public["player_club_id"]) == [281, 281, 379, 379]
    assert public["injury_desc"].iloc[0] == "Muscle injury"

    # A table with no person columns passes through untouched.
    plain = pd.DataFrame({"season": ["2018-19"], "n": [3]})
    assert module.deidentify_frame(plain, surrogates).equals(plain)

    # A later table reuses the surrogate already drawn, so rows join across the
    # deposit instead of the same player appearing under two keys.
    later = pd.DataFrame({"tm_player_id": ["8000001", "8000003"], "n": [1, 2]})
    joined = module.deidentify_frame(later, surrogates, key_factory=lambda: next(keys))
    assert list(joined["tm_player_id"]) == ["P000", "P002"]
    assert surrogates["8000001"] == "P000"


def test_surrogates_survive_a_rebuild(load_src_module, tmp_path):
    """A rebuild that renumbered every player would silently invalidate any
    analysis a reader had already done against the deposit."""
    module = load_src_module("24_prepare_public_deposit.py")
    path = tmp_path / "deposit_player_map.csv"

    assert module.load_player_surrogates(path) == {}

    surrogates = {"8000001": "PAAA", "8000002": "PBBB"}
    module.save_player_surrogates(path, surrogates)
    assert module.load_player_surrogates(path) == surrogates

    pd.DataFrame(
        {"source_id": ["8000001", "8000001"], "player_key": ["PAAA", "PCCC"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate source identifiers"):
        module.load_player_surrogates(path)

    pd.DataFrame({"source_id": ["8000001"]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="player map missing columns"):
        module.load_player_surrogates(path)


def test_writing_without_a_surrogate_map_still_sanitises_terminology(
    load_src_module, tmp_path
):
    """The de-identification is opt-in per call, so the terminology pass must
    keep working for callers that supply no map."""
    module = load_src_module("24_prepare_public_deposit.py")
    source = tmp_path / "in.csv"
    pd.DataFrame({"fragility_group": ["fragile"], "n": [1]}).to_csv(source, index=False)

    target = tmp_path / "out.csv"
    module.write_sanitized_csv(source, target)

    written = pd.read_csv(target)
    assert list(written.columns) == ["prior_history_group", "n"]
    assert written["prior_history_group"].iloc[0] == "higher_history"
