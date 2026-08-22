"""Tests for the women's appearance snapshot.

The records this module assembles are extracted by a browser, outside any
test's reach, so what matters here is that the module refuses malformed input
rather than turning it into a gradient. A half-read match report looks exactly
like a heavily rotated squad, and the whole paper is about not mistaking a
recording artefact for a rotation signal.
"""

from itertools import count

import pandas as pd
import pytest


MODULE = "39_fetch_fbref_womens_appearances.py"


def _match_records(module, match_id="m1", competition="189", season="2024-2025",
                   date="2024-09-21", starters=22, subs=4, minutes=None):
    """Build one complete, well-formed match's worth of records."""
    records = []
    per_sub = 15.0
    starter_minutes = minutes if minutes is not None else 90.0
    for index in range(starters):
        records.append([
            "ENG", competition, season, date, match_id, f"{match_id}p{index}",
            starter_minutes if index >= subs else starter_minutes - per_sub,
            module.STARTER_ROLE,
        ])
    for index in range(subs):
        records.append([
            "ENG", competition, season, date, match_id, f"{match_id}s{index}",
            per_sub, module.SUBSTITUTE_ROLE,
        ])
    return records


def test_normalize_records_types_and_orders(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_records(_match_records(module))
    assert len(frame) == 26
    assert str(frame[module.DATE_COL].dtype).startswith("datetime64")
    assert frame[module.MINUTES_COL].dtype.kind == "f"
    assert frame[module.COMPETITION_COL].dtype == object


def test_normalize_records_rejects_malformed_input(load_src_module):
    module = load_src_module(MODULE)

    with pytest.raises(ValueError, match="no appearance records"):
        module.normalize_records([])

    bad_date = _match_records(module)
    bad_date[0][3] = "not-a-date"
    with pytest.raises(ValueError, match="readable date"):
        module.normalize_records(bad_date)

    bad_minutes = _match_records(module)
    bad_minutes[0][6] = "ninety"
    with pytest.raises(ValueError, match="readable minutes"):
        module.normalize_records(bad_minutes)

    bad_comp = _match_records(module, competition="999")
    with pytest.raises(ValueError, match="unregistered competitions"):
        module.normalize_records(bad_comp)

    bad_role = _match_records(module)
    bad_role[0][7] = "water_carrier"
    with pytest.raises(ValueError, match="unrecognised lineup roles"):
        module.normalize_records(bad_role)

    duplicated = _match_records(module)
    duplicated[1][5] = duplicated[0][5]
    with pytest.raises(ValueError, match="twice in one match"):
        module.normalize_records(duplicated)


def test_match_integrity_accepts_a_complete_report(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_records(_match_records(module))
    integrity = module.match_integrity(frame)
    assert len(integrity) == 1
    row = integrity.iloc[0]
    assert row["n_starters"] == module.STARTERS_PER_MATCH
    assert bool(row["usable"])
    assert bool(row["roles_known"])
    assert row["exclusion_reason"] == ""


def test_a_missing_lineup_box_costs_roles_not_the_match(load_src_module):
    """Some leagues are covered for minutes but carry no lineup box.

    The pooled gradient needs only minutes, so discarding those matches would
    throw away a whole league to protect a fit that league was never going to
    contribute to.
    """
    module = load_src_module(MODULE)
    no_lineup = _match_records(module)
    for record in no_lineup:
        record[7] = module.SUBSTITUTE_ROLE
    integrity = module.match_integrity(module.normalize_records(no_lineup))
    row = integrity.iloc[0]
    assert bool(row["usable"])
    assert not bool(row["roles_known"])
    assert row["exclusion_reason"] == ""


def test_match_integrity_flags_each_way_a_report_fails(load_src_module):
    module = load_src_module(MODULE)

    # Both elevens read, but the minutes total is far short of a match: the
    # shape of a report whose substitutions were dropped.
    short = _match_records(module, starters=22, subs=0, minutes=60.0)
    integrity = module.match_integrity(module.normalize_records(short))
    row = integrity.iloc[0]
    assert bool(row["roles_known"])
    assert not bool(row["minutes_ok"])
    assert not bool(row["usable"])
    assert "outside match bounds" in row["exclusion_reason"]

    # A misread minutes cell that still totals plausibly is caught by bounds.
    outsized = _match_records(module, starters=22, subs=0, minutes=90.0)
    outsized[0][6] = 900.0
    outsized[1][6] = -810.0 + 90.0
    integrity = module.match_integrity(module.normalize_records(outsized))
    row = integrity.iloc[0]
    assert bool(row["roles_known"]) and bool(row["minutes_ok"])
    assert not bool(row["bounds_ok"])
    assert "plausible minutes" in row["exclusion_reason"]


def test_match_integrity_requires_its_columns(load_src_module):
    module = load_src_module(MODULE)
    with pytest.raises(KeyError, match="match integrity input missing columns"):
        module.match_integrity(pd.DataFrame({"other": [1]}))


def test_completeness_reports_missing_and_thin_seasons(load_src_module):
    module = load_src_module(MODULE)
    records = []
    for index in range(10):
        records.extend(_match_records(module, match_id=f"m{index}"))
    integrity = module.match_integrity(module.normalize_records(records))

    scheduled = {("189", "2024-2025"): 10, ("183", "2024-2025"): 12}
    completeness = module.completeness_by_league_season(integrity, scheduled)

    admitted = completeness.set_index("competition_id")
    assert bool(admitted.loc["189", "admitted"])
    assert admitted.loc["189", "match_coverage"] == 1.0
    # A league-season that produced nothing is present and excluded, not absent.
    assert not bool(admitted.loc["183", "admitted"])
    assert admitted.loc["183", "usable_matches"] == 0
    assert admitted.loc["183", "parsed_matches"] == 0


def test_completeness_falls_back_to_parsed_count_and_handles_empty(load_src_module):
    module = load_src_module(MODULE)
    records = _match_records(module)
    integrity = module.match_integrity(module.normalize_records(records))

    # No fixture list for this league-season: the parsed count stands in.
    completeness = module.completeness_by_league_season(integrity, {})
    assert completeness.iloc[0]["scheduled_matches"] == 1
    assert completeness.iloc[0]["match_coverage"] == 1.0

    # A fixture list claiming zero matches cannot divide by itself.
    zeroed = module.completeness_by_league_season(integrity, {("189", "2024-2025"): 0})
    assert zeroed.iloc[0]["match_coverage"] == 0.0
    assert not bool(zeroed.iloc[0]["admitted"])

    with pytest.raises(KeyError, match="completeness input missing columns"):
        module.completeness_by_league_season(pd.DataFrame({"other": [1]}), {})


def test_gate_keeps_only_usable_matches_in_admitted_seasons(load_src_module):
    module = load_src_module(MODULE)
    good = []
    for index in range(10):
        good.extend(_match_records(module, match_id=f"g{index}"))
    # Broken means the appearance record is wrong, not that roles are missing.
    broken = _match_records(module, match_id="bad", starters=22, subs=0, minutes=40.0)
    frame = module.normalize_records(good + broken)
    integrity = module.match_integrity(frame)
    completeness = module.completeness_by_league_season(integrity, {("189", "2024-2025"): 11})

    gated = module.apply_completeness_gate(frame, integrity, completeness)
    assert set(gated[module.MATCH_COL]) == {f"g{i}" for i in range(10)}

    with pytest.raises(KeyError, match="gate input missing columns"):
        module.apply_completeness_gate(pd.DataFrame({"other": [1]}), integrity, completeness)


def test_gate_refuses_to_return_an_empty_snapshot(load_src_module):
    module = load_src_module(MODULE)
    # Broken means the appearance record is wrong, not that roles are missing.
    broken = _match_records(module, match_id="bad", starters=22, subs=0, minutes=40.0)
    frame = module.normalize_records(broken)
    integrity = module.match_integrity(frame)
    completeness = module.completeness_by_league_season(integrity, {("189", "2024-2025"): 1})
    with pytest.raises(ValueError, match="admitted no appearances"):
        module.apply_completeness_gate(frame, integrity, completeness)


def test_roles_are_marked_unknown_where_they_could_not_be_read(load_src_module):
    """A league-season either has lineup boxes or it does not.

    Marking the role unknown rather than calling everyone a substitute matters:
    the gradient module reports an absent within-starter fit for a league with
    no starters, which is true, but would report a confident and wrong one if
    every appearance were labelled a substitute.
    """
    module = load_src_module(MODULE)
    with_roles = []
    for index in range(5):
        with_roles.extend(_match_records(module, match_id=f"r{index}"))
    without = []
    for index in range(5):
        block = _match_records(module, match_id=f"n{index}", competition="185", season="2024")
        for record in block:
            record[7] = module.SUBSTITUTE_ROLE
        without.extend(block)

    frame = module.normalize_records(with_roles + without)
    integrity = module.match_integrity(frame)
    completeness = module.completeness_by_league_season(
        integrity, {("189", "2024-2025"): 5, ("185", "2024"): 5}
    )
    by_league = completeness.set_index("competition_id")
    assert bool(by_league.loc["189", "roles_admitted"])
    assert bool(by_league.loc["185", "admitted"])
    assert not bool(by_league.loc["185", "roles_admitted"])
    assert by_league.loc["185", "role_coverage"] == 0.0

    harmonized = module.harmonize_roles(frame, integrity, completeness)
    english = harmonized[harmonized[module.COMPETITION_COL] == "189"]
    norwegian = harmonized[harmonized[module.COMPETITION_COL] == "185"]
    assert module.UNKNOWN_ROLE not in set(english[module.ROLE_COL])
    assert set(norwegian[module.ROLE_COL]) == {module.UNKNOWN_ROLE}

    with pytest.raises(KeyError, match="role harmonisation input missing columns"):
        module.harmonize_roles(pd.DataFrame({"other": [1]}), integrity, completeness)
    with pytest.raises(KeyError, match="completeness frame missing columns"):
        module.harmonize_roles(frame, integrity, pd.DataFrame({"other": [1]}))


def test_surrogates_are_drawn_not_derived(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_records(_match_records(module))

    counter = count()
    surrogates = module.build_player_surrogates(
        frame, key_factory=lambda: f"{next(counter):06d}"
    )
    assert len(surrogates) == frame[module.SOURCE_PLAYER_COL].nunique()
    assert surrogates[module.PLAYER_ID_COL].is_unique
    assert surrogates[module.PLAYER_ID_COL].str.startswith("W").all()
    # No surrogate contains the source identifier it stands for.
    for source, key in zip(surrogates[module.SOURCE_PLAYER_COL], surrogates[module.PLAYER_ID_COL]):
        assert source not in key

    default = module.build_player_surrogates(frame)
    assert default[module.PLAYER_ID_COL].is_unique

    with pytest.raises(KeyError, match="surrogate source missing columns"):
        module.build_player_surrogates(pd.DataFrame({"other": [1]}))


def test_gradient_schema_carries_no_source_identifier(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_records(_match_records(module))
    surrogates = module.build_player_surrogates(frame)
    projected = module.to_gradient_schema(frame, surrogates)

    assert module.SOURCE_PLAYER_COL not in projected.columns
    for column in (module.PLAYER_ID_COL, module.DATE_COL, module.MINUTES_COL,
                   module.COMPETITION_COL, module.ROLE_COL):
        assert column in projected.columns

    with pytest.raises(KeyError, match="gradient projection input missing columns"):
        module.to_gradient_schema(pd.DataFrame({"other": [1]}), surrogates)
    with pytest.raises(KeyError, match="surrogate map missing columns"):
        module.to_gradient_schema(frame, pd.DataFrame({"other": [1]}))


def test_gradient_schema_refuses_an_unmapped_player(load_src_module):
    module = load_src_module(MODULE)
    frame = module.normalize_records(_match_records(module))
    surrogates = module.build_player_surrogates(frame).iloc[:-1]
    with pytest.raises(ValueError, match="needs a surrogate"):
        module.to_gradient_schema(frame, surrogates)


def test_snapshot_summary_describes_what_was_deposited(load_src_module):
    module = load_src_module(MODULE)
    records = []
    for index in range(4):
        records.extend(_match_records(module, match_id=f"m{index}", date=f"2024-09-0{index+1}"))
    frame = module.normalize_records(records)
    projected = module.to_gradient_schema(frame, module.build_player_surrogates(frame))

    summary = module.snapshot_summary(projected)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["matches"] == 4
    assert row["appearances"] == len(projected)
    assert row["league"] == module.WOMENS_LEAGUES["189"]
    assert 0.0 < row["starter_share"] < 1.0
    assert row["first_date"] == "2024-09-01"

    with pytest.raises(KeyError, match="summary input missing columns"):
        module.snapshot_summary(pd.DataFrame({"other": [1]}))


def test_adding_a_league_does_not_renumber_the_players_already_deposited(load_src_module):
    """A surrogate that moves is not a pseudonym, it is a new identity.

    The women's panel gained six leagues and changed source after one league
    had already been deposited. If that redraws every key, any table published
    against the old one silently stops matching.
    """
    module = load_src_module(MODULE)
    first = module.normalize_records(_match_records(module, match_id="a"))
    original = module.build_player_surrogates(first)

    second = module.normalize_records(
        _match_records(module, match_id="b", competition="ENG1", season="2024")
    )
    combined = pd.concat([first, second], ignore_index=True)
    extended = module.extend_player_surrogates(original, combined)

    before = dict(zip(original[module.SOURCE_PLAYER_COL], original[module.PLAYER_ID_COL]))
    after = dict(zip(extended[module.SOURCE_PLAYER_COL], extended[module.PLAYER_ID_COL]))
    for source, key in before.items():
        assert after[source] == key, source
    assert len(extended) == combined[module.SOURCE_PLAYER_COL].nunique()
    assert extended[module.PLAYER_ID_COL].is_unique

    # Re-running with nothing new is a no-op rather than a redraw.
    assert module.extend_player_surrogates(extended, combined).equals(extended)

    with pytest.raises(KeyError, match="existing surrogate map missing columns"):
        module.extend_player_surrogates(pd.DataFrame({"other": [1]}), combined)
    with pytest.raises(KeyError, match="surrogate source missing columns"):
        module.extend_player_surrogates(original, pd.DataFrame({"other": [1]}))
