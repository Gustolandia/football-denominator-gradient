import pandas as pd
import pytest


def test_pick_col_returns_first_existing(load_src_module):
    module = load_src_module("01_fetch_fbref.py")
    df = pd.DataFrame({"b": [1], "c": [2]})
    assert module._pick_col(df, ["a", "b", "c"], "demo") == "b"


def test_pick_col_exits_with_columns(load_src_module):
    module = load_src_module("01_fetch_fbref.py")
    with pytest.raises(SystemExit) as exc:
        module._pick_col(pd.DataFrame({"x": []}), ["a"], "demo")
    assert "Available columns" in str(exc.value)
