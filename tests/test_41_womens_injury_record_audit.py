"""Tests for the women's injury-record audit.

Two things are being protected. The first is the measurement: the audit exists
to justify a restriction in the manuscript, so its numbers have to mean what
the limitation paragraph says they mean. The second is the boundary: the rows
this module reads name identifiable athletes and give their diagnoses, and no
table it writes may carry either.
"""

import numpy as np
import pandas as pd
import pytest


MODULE = "41_womens_injury_record_audit.py"


def _rows(league, specs):
    """Build sampled rows from (player, spells, seasons, types) tuples."""
    return [
        {
            "league": league,
            "club": f"{league}-club",
            "player": player,
            "n_spells": spells,
            "seasons": seasons,
            "types": types,
        }
        for player, spells, seasons, types in specs
    ]


def test_normalize_marks_presence_and_absent_sections(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_audit_records(
        _rows("WSL", [
            ("a", 3, "22/23;23/24", "Hamstring Injury;Cruciate Ligament Rupture"),
            ("b", 0, "", ""),
            ("c", module.NO_SECTION, "", ""),
        ])
    )
    assert list(frame["has_history"]) == [True, False, False]
    assert list(frame["section_absent"]) == [False, False, True]
    # A missing section must not subtract from a spell count.
    assert list(frame["recorded_spells"]) == [3, 0, 0]


def test_normalize_rejects_malformed_input(load_src_module):
    module = load_src_module(MODULE)
    with pytest.raises(ValueError, match="at least one sampled player"):
        module.normalize_audit_records([])

    with pytest.raises(KeyError, match="injury audit input missing columns"):
        module.normalize_audit_records([{"league": "WSL"}])

    bad = _rows("WSL", [("a", "many", "", "")])
    with pytest.raises(ValueError, match="readable spell count"):
        module.normalize_audit_records(bad)


def test_coverage_by_league_counts_presence(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_audit_records(
        _rows("WSL", [("a", 2, "23/24", "x"), ("b", 0, "", ""), ("c", 2, "24/25", "y")])
        + _rows("BL1", [("d", 0, "", ""), ("e", 0, "", "")])
    )
    coverage = module.coverage_by_league(frame).set_index("league")
    assert coverage.loc["WSL", "players_sampled"] == 3
    assert coverage.loc["WSL", "players_with_history"] == 2
    assert coverage.loc["WSL", "spells_per_player"] == pytest.approx(4 / 3)
    assert coverage.loc["BL1", "share_with_history"] == 0.0
    assert coverage.loc["WSL", "clubs_sampled"] == 1

    # A frame without club provenance still reports coverage.
    clubless = frame.drop(columns=["club"])
    assert module.coverage_by_league(clubless).iloc[0]["clubs_sampled"] == 0

    with pytest.raises(KeyError, match="coverage input missing columns"):
        module.coverage_by_league(pd.DataFrame({"other": [1]}))


def test_season_profile_counts_players_per_season(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_audit_records(
        _rows("WSL", [
            ("a", 2, "23/24;24/25", "x;y"),
            ("b", 1, "24/25", "z"),
            ("c", 0, "", ""),
        ])
    )
    profile = module.season_recording_profile(frame, "women").set_index("season")
    assert profile.loc["24/25", "players_with_spell"] == 2
    assert profile.loc["23/24", "players_with_spell"] == 1
    assert profile.loc["24/25", "players_sampled"] == 3
    assert profile.loc["24/25", "per_capita"] == pytest.approx(2 / 3)

    with pytest.raises(KeyError, match="season profile input missing columns"):
        module.season_recording_profile(pd.DataFrame({"other": [1]}), "women")


def test_recording_decay_indexes_to_the_reference_season(load_src_module):
    module = load_src_module(MODULE)
    profile = pd.DataFrame(
        [
            {"population": "women", "season": "19/20", "per_capita": 0.02},
            {"population": "women", "season": "24/25", "per_capita": 0.40},
            {"population": "men", "season": "19/20", "per_capita": 0.30},
            {"population": "men", "season": "24/25", "per_capita": 0.60},
        ]
    )
    decay = module.recording_decay(profile, "24/25").set_index(["population", "season"])
    assert decay.loc[("women", "19/20"), "relative_to_reference"] == pytest.approx(0.05)
    assert decay.loc[("men", "19/20"), "relative_to_reference"] == pytest.approx(0.50)

    with pytest.raises(ValueError, match="reference season"):
        module.recording_decay(profile, "26/27")
    with pytest.raises(KeyError, match="decay input missing columns"):
        module.recording_decay(pd.DataFrame({"other": [1]}), "24/25")


def test_recording_decay_handles_an_empty_reference(load_src_module):
    module = load_src_module(MODULE)
    profile = pd.DataFrame(
        [
            {"population": "women", "season": "19/20", "per_capita": 0.02},
            {"population": "women", "season": "24/25", "per_capita": 0.0},
        ]
    )
    decay = module.recording_decay(profile, "24/25")
    assert decay["relative_to_reference"].isna().all()


def test_severity_mix_reports_the_catastrophic_share(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_audit_records(
        _rows("WSL", [
            ("a", 3, "23/24", "Cruciate Ligament Rupture;Hamstring Injury;Unknown Injury"),
            ("b", 1, "24/25", "Cruciate Ligament Rupture"),
        ])
    )
    mix = module.severity_mix(frame, "women").iloc[0]
    assert mix["recorded_type_entries"] == 4
    assert mix["catastrophic_entries"] == 2
    assert mix["catastrophic_share"] == pytest.approx(0.5)
    assert mix["unknown_entries"] == 1

    empty = module.normalize_audit_records(_rows("WSL", [("a", 0, "", "")]))
    blank = module.severity_mix(empty, "women").iloc[0]
    assert blank["recorded_type_entries"] == 0
    assert np.isnan(blank["catastrophic_share"])
    assert np.isnan(blank["unknown_share"])

    with pytest.raises(KeyError, match="severity input missing columns"):
        module.severity_mix(pd.DataFrame({"other": [1]}), "women")


def test_severity_matching_survives_a_vocabulary_difference(load_src_module):
    """The two sites name the same injury differently.

    Transfermarkt records a cruciate ligament "tear"; Soccerdonna records a
    "rupture". Matching exact labels scored the men's sample at zero
    catastrophic injuries, which reads as a spectacular finding and is in fact
    a spelling difference. Both spellings must count.
    """
    module = load_src_module(MODULE)
    rupture = module.normalize_audit_records(
        _rows("WSL", [("a", 1, "24/25", "Cruciate Ligament Rupture")])
    )
    tear = module.normalize_audit_records(
        _rows("EPL", [("b", 1, "24/25", "Cruciate ligament tear")])
    )
    injury = module.normalize_audit_records(
        _rows("EPL", [("c", 1, "24/25", "Cruciate ligament injury")])
    )
    for frame in (rupture, tear, injury):
        assert module.severity_mix(frame, "x").iloc[0]["catastrophic_entries"] == 1

    # An Achilles complaint is not season-ending unless it ruptured or tore.
    grumbling = module.normalize_audit_records(
        _rows("EPL", [("d", 1, "24/25", "Achilles tendon problems")])
    )
    assert module.severity_mix(grumbling, "x").iloc[0]["catastrophic_entries"] == 0
    ruptured = module.normalize_audit_records(
        _rows("EPL", [("e", 1, "24/25", "Achilles tendon rupture")])
    )
    assert module.severity_mix(ruptured, "x").iloc[0]["catastrophic_entries"] == 1


def test_audit_contrast_reports_the_numbers_the_paper_cites(load_src_module):
    module = load_src_module(MODULE)
    womens = module.normalize_audit_records(
        _rows("WSL", [("a", 2, "24/25", "x"), ("b", 1, "24/25", "y"), ("c", 0, "", "")])
    )
    mens = module.normalize_audit_records(
        _rows("EPL", [("d", 8, "19/20;24/25", "x"), ("e", 6, "19/20;24/25", "y")])
    )
    contrast = module.audit_contrast(womens, mens, "24/25", "19/20").set_index("population")

    assert contrast.loc["women", "players_sampled"] == 3
    assert contrast.loc["women", "share_with_history"] == pytest.approx(2 / 3)
    assert contrast.loc["men", "spells_per_player"] == pytest.approx(7.0)
    # The men's record reaches the early season as densely as the reference.
    assert contrast.loc["men", "early_recording_relative_to_reference"] == pytest.approx(1.0)
    # The women's record has nothing there at all.
    assert np.isnan(contrast.loc["women", "early_recording_relative_to_reference"])

    with pytest.raises(KeyError, match="women's audit missing columns"):
        module.audit_contrast(pd.DataFrame({"other": [1]}), mens, "24/25", "19/20")
    with pytest.raises(KeyError, match="men's audit missing columns"):
        module.audit_contrast(womens, pd.DataFrame({"other": [1]}), "24/25", "19/20")


def test_deposited_tables_may_not_name_anybody(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_audit_records(_rows("WSL", [("a", 1, "24/25", "x")]))

    # The aggregates are safe.
    module.assert_no_personal_columns(module.coverage_by_league(frame), "coverage")
    module.assert_no_personal_columns(module.severity_mix(frame, "women"), "severity")

    # The sampled rows themselves are not, and the check says which columns.
    with pytest.raises(ValueError, match="would publish personal columns"):
        module.assert_no_personal_columns(frame, "raw sample")
    with pytest.raises(ValueError, match="player"):
        module.assert_no_personal_columns(pd.DataFrame({"player": ["x"]}), "names")
