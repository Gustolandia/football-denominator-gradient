import importlib.util
import sys
from pathlib import Path
from typing import Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in [ROOT, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def load_src_module():
    def _load(filename: str, module_name: Optional[str] = None):
        name = module_name or filename.replace(".py", "").replace("-", "_")
        spec = importlib.util.spec_from_file_location(name, SRC / filename)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    return _load
