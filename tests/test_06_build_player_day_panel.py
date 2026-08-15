import pandas as pd
import pytest


def test_06_imports(load_src_module):
    module = load_src_module("06_build_player_day_panel.py")
    assert callable(module.main)


def test_expand_unavailable_spell_days(load_src_module):
    module = load_src_module("06_build_player_day_panel.py")
    injuries = pd.DataFrame(
        {
            "tm_player_id": [1, 1, 2, 3],
            "start_date": ["2024-01-01", "2024-01-05", "2024-01-02", None],
            "end_date": ["2024-01-03", "2024-01-04", None, "2024-01-04"],
        }
    )
    out = module.expand_unavailable_spell_days(
        injuries,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-10"),
    )
    assert out[["tm_player_id", "date"]].to_dict("records") == [
        {"tm_player_id": 1, "date": pd.Timestamp("2024-01-02")},
        {"tm_player_id": 1, "date": pd.Timestamp("2024-01-03")},
    ]


def test_expand_unavailable_spell_days_requires_columns(load_src_module):
    module = load_src_module("06_build_player_day_panel.py")
    with pytest.raises(KeyError, match="tm_player_id"):
        module.expand_unavailable_spell_days(
            pd.DataFrame({"start_date": ["2024-01-01"]}),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-10"),
        )

    out = module.expand_unavailable_spell_days(
        pd.DataFrame({"tm_player_id": [1], "start_date": ["2024-01-01"]}),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-10"),
    )
    assert out.empty


def test_expand_unavailable_spell_days_empty_cleaned_and_no_rows(load_src_module):
    module = load_src_module("06_build_player_day_panel.py")

    cleaned_empty = module.expand_unavailable_spell_days(
        pd.DataFrame(
            {
                "tm_player_id": [1],
                "start_date": ["2024-01-01"],
                "end_date": [None],
            }
        ),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-10"),
    )
    assert cleaned_empty.empty

    no_unavailable_days = module.expand_unavailable_spell_days(
        pd.DataFrame(
            {
                "tm_player_id": [1],
                "start_date": ["2024-01-05"],
                "end_date": ["2024-01-04"],
            }
        ),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-10"),
    )
    assert no_unavailable_days.empty
