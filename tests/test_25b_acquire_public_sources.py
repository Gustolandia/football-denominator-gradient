"""Import-level test for the public-source acquisition entry point."""

import importlib.util
from pathlib import Path


def test_public_source_entry_point_exposes_main():
    path = Path(__file__).parents[1] / "src" / "25b_acquire_public_sources.py"
    spec = importlib.util.spec_from_file_location("public_source_entry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert callable(module.main)
