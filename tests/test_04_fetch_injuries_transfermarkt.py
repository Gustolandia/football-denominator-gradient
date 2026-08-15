import asyncio
import builtins
import sys
import types

import pytest


def test_load_tmkt_client_success_and_missing(load_src_module, monkeypatch):
    module = load_src_module("04_fetch_injuries_transfermarkt.py")
    fake_client = object()
    monkeypatch.setitem(sys.modules, "tmkt", types.SimpleNamespace(TMKT=fake_client))
    assert module._load_tmkt_client() is fake_client

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tmkt":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "tmkt", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit):
        module._load_tmkt_client()


def test_fetch_injuries_for_all_handles_payload_shapes(load_src_module, monkeypatch):
    module = load_src_module("04_fetch_injuries_transfermarkt.py")

    class FakeApi:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get_player_injuries(self, pid):
            if pid == 1:
                raise RuntimeError("boom")
            payloads = {
                2: ["not", "a", "dict"],
                3: {"success": False, "message": "nope"},
                4: {"success": True, "data": []},
                5: {"success": True, "data": {"injuries": "bad"}},
                6: {"data": {"injuries": [7, {"injuryId": 99}]}},
            }
            return payloads[pid]

    monkeypatch.setattr(module, "_load_tmkt_client", lambda: FakeApi)
    rows = asyncio.run(module.fetch_injuries_for_all([1, 2, 3, 4, 5, 6]))
    assert rows == [{"injuryId": 99, "tm_player_id": 6}]
