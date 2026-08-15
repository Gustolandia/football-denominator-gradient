"""Refuse to deposit anything that identifies a person.

The manuscript says no individual is identified in any deposited output. That
sentence was false for five files at once --- two hand-built audit inputs and
three machine-generated tables --- and nothing in the suite noticed, because
every gate checked whether numbers agreed and none checked who the rows were
about. A published DOI cannot be recalled, so this runs before one exists.

The rules are deliberately about shape rather than about the five known files.
A gate that named them would pass the moment a sixth appeared.
"""

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]

# What travels: the hand-built evidence, and the derived tables the archived
# record carries beside it. data/processed/results is absent from a released
# checkout until the pipeline runs, so it is scanned when present.
DEPOSITED_TREES = (
    ROOT / "data" / "manual",
    ROOT / "data" / "processed" / "results",
)

# Directories the public release copies wholesale. Source files ship too, so a
# name in a docstring or a test fixture is as deposited as a name in a table.
SHIPPED_SOURCE_TREES = ("src", "tests", "manuscript")
SOURCE_SUFFIXES = (".py", ".tex", ".bib", ".md", ".txt")

# A person's name, and the provider identifiers that resolve to a public
# profile page. None of these belongs in a deposited file in any form.
FORBIDDEN_COLUMNS = (
    "player_name",
    "tm_player_id",
    "player_id",
    "fbref_player_id",
    "audit_id",
)

# The old audit key was "{tm_player_id}_{yyyymmdd}", which is a direct
# identifier wearing a key's clothes. It must not survive in any cell either.
RAW_AUDIT_ID = re.compile(r"^\d{3,}_\d{8}$")

AUDIT_FILES = (
    "independent_same_day_event_audit.csv",
    "independent_non_event_audit.csv",
)


def _deposited_csvs() -> list[Path]:
    return [
        path
        for tree in DEPOSITED_TREES
        if tree.exists()
        for path in sorted(tree.rglob("*.csv"))
    ]


def _shipped_source_files() -> list[Path]:
    return [
        path
        for tree in SHIPPED_SOURCE_TREES
        for path in sorted((ROOT / tree).rglob("*"))
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and "__pycache__" not in path.parts
        # The release build excludes these, so neither does this gate.
        and "tmp" not in path.parts
    ]


def _whole_word_pattern(tokens) -> re.Pattern:
    """Match any token as a whole word.

    Word boundaries matter: a four-letter name part such as a common forename
    otherwise matches inside an ordinary English word and reports a leak in a
    column inventory.
    """
    ordered = sorted(tokens, key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(re.escape(token) for token in ordered) + r")\b",
        re.IGNORECASE,
    )


def _audited_name_tokens() -> tuple[set, set]:
    """Return (every name part, surnames only) from the reviewer's own record."""
    parts, surnames = set(), set()
    for name in AUDIT_FILES:
        frame = pd.read_csv(ROOT / "data" / "private" / name, dtype=str)
        for full_name in frame["player_name"].dropna():
            tokens = [t for t in re.split(r"[\s\-']+", full_name.strip()) if t]
            surnames.add(tokens[-1])
            parts.update(token for token in tokens if len(token) >= 4)
    return parts, surnames


def test_no_deposited_table_carries_a_person_column():
    paths = _deposited_csvs()
    assert paths, "no deposited tables were scanned"
    # data/manual ships with every release, so it must always be among them.
    assert any("manual" in str(path) for path in paths)

    offenders = []
    for path in paths:
        header = pd.read_csv(path, nrows=0)
        for column in FORBIDDEN_COLUMNS:
            if column in header.columns:
                offenders.append(f"{path.relative_to(ROOT)}: {column}")
    assert not offenders, "deposited tables carry person columns: " + "; ".join(
        offenders
    )


def test_no_deposited_cell_carries_a_raw_audit_key():
    offenders = []
    for path in _deposited_csvs():
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        for column in frame.columns:
            values = frame[column].dropna().astype(str)
            if values.map(lambda value: bool(RAW_AUDIT_ID.match(value))).any():
                offenders.append(f"{path.relative_to(ROOT)}: {column}")
    assert not offenders, "deposited cells carry raw audit keys: " + "; ".join(
        offenders
    )


def test_audit_files_are_keyed_by_surrogate_and_withhold_their_urls():
    """The verdicts still have to join to the rows they judge, and the source
    URLs cannot travel: most of these slugs carry the player's surname and one
    carries a graded diagnosis."""
    for name in AUDIT_FILES:
        path = ROOT / "data" / "manual" / name
        frame = pd.read_csv(path, dtype=str)
        assert len(frame) == 30, name
        for required in ("record_key", "player_key", "season"):
            assert required in frame.columns, f"{name}: {required}"
        assert frame["record_key"].is_unique, name
        assert "independent_source_url" not in frame.columns, name
        assert "independent_source_found" in frame.columns, name
        # A season, not a match day: a club plus a date plus a body part is a
        # singleton.
        assert frame["season"].str.fullmatch(r"\d{4}-\d{2}").all(), name


def test_no_deposited_table_names_an_audited_player():
    """Checked against the reviewer's own identified record, which stays out of
    every deposit. Skipped in a released checkout, which does not carry it."""
    if not all((ROOT / "data" / "private" / name).exists() for name in AUDIT_FILES):
        return
    parts, _ = _audited_name_tokens()
    assert parts, "the identified originals name nobody"

    pattern = _whole_word_pattern(parts)
    offenders = []
    for path in _deposited_csvs():
        match = pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
        if match:
            offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
    assert not offenders, "deposited tables name audited players: " + "; ".join(
        offenders
    )


def test_no_shipped_source_file_names_an_audited_player():
    """Source files ship, so a name in a docstring is deposited too.

    The first version of this gate read tables only, and the very commit that
    removed the names from the data put four of them back --- one in a module
    docstring, three in test fixtures, two of those inside example URLs whose
    slugs carried a surname, which is the exact pattern the fixtures existed to
    describe.

    Surnames only. Forenames collide constantly with legitimate text: eleven of
    these players share a first name with an author cited in references.bib and
    one with the statistician thanked in the acknowledgements, while a surname
    match has so far only ever been a real leak.
    """
    if not all((ROOT / "data" / "private" / name).exists() for name in AUDIT_FILES):
        return
    _, surnames = _audited_name_tokens()
    assert surnames, "the identified originals name nobody"

    pattern = _whole_word_pattern(surnames)
    offenders = []
    for path in _shipped_source_files():
        match = pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
        if match:
            offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
    assert not offenders, "shipped source files name audited players: " + "; ".join(
        offenders
    )


def test_the_identity_map_never_travels():
    """The map is the only thing that reverses the surrogates, so a deposit
    carrying it would be pseudonymised in name only."""
    for tree in DEPOSITED_TREES:
        if not tree.exists():
            continue
        for path in tree.rglob("*"):
            assert "identity_map" not in path.name, path
            assert "player_map" not in path.name, path
