"""
Fetch explicit injury histories from Transfermarkt for mapped players.

Requires:
  data/processed/player_mapping_tm.csv

Outputs:
  data/raw/tm_injuries_raw.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

import asyncio
from typing import Any, Dict, List

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from config import DATA_PROCESSED, DATA_RAW


def _load_tmkt_client():
    """Load the optional Transfermarkt client only for the network fetch path."""
    try:
        from tmkt import TMKT
    except ImportError as exc:
        raise SystemExit(
            "transfermarkt-wrapper is not installed. Run:\n"
            "  pip install transfermarkt-wrapper"
        ) from exc
    return TMKT


async def fetch_injuries_for_all(tm_player_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Fetch injuries for a list of Transfermarkt player IDs.

    The tmkt client returns a dict like:
      {
        "success": True,
        "message": "OK",
        "data": {
          "playerId": "...",
          "injuries": [
            {
              "playerId": "...",
              "injuryId": 66,
              "missedGamesCount": 16,
              "seasonId": 2022,
              "start": "2023-02-12",
              "end": "2023-05-10",
              "durationDetails": {...},
              "name": "Toe injury",
              "category": "Unknown",
            },
            ...
          ]
        }
      }

    We flatten this into one row per injury spell, carrying tm_player_id.
    """
    rows: List[Dict[str, Any]] = []
    TMKT = _load_tmkt_client()

    async with TMKT() as api:
        for pid in tqdm(tm_player_ids, desc="injuries"):
            try:
                resp = await api.get_player_injuries(pid)
            except Exception as exc:  # pragma: no cover - network issues, rate limits, etc.
                print(f"Warning: get_player_injuries failed for {pid}: {exc}")
                continue

            if not isinstance(resp, dict):
                print(
                    f"Warning: unexpected injuries payload for {pid}: "
                    f"{type(resp)} -> {resp!r}"
                )
                continue

            success = resp.get("success", True)
            if not success:
                print(
                    f"Warning: injuries call for {pid} not successful: "
                    f"{resp.get('message')!r}"
                )
                continue

            data = resp.get("data")
            if not isinstance(data, dict):
                print(
                    f"Warning: injuries 'data' field for {pid} is not a dict: "
                    f"{type(data)} -> {data!r}"
                )
                continue

            injuries = data.get("injuries") or []
            if not isinstance(injuries, list):
                print(
                    f"Warning: injuries list for {pid} is not a list: "
                    f"{type(injuries)} -> {injuries!r}"
                )
                continue

            for inj in injuries:
                if not isinstance(inj, dict):
                    print(
                        f"Warning: skipping non-dict injury row for {pid}: "
                        f"{type(inj)} -> {inj!r}"
                    )
                    continue

                rec = dict(inj)
                rec["tm_player_id"] = int(pid)
                rows.append(rec)

    return rows


def main() -> None:  # pragma: no cover
    mapping = pd.read_csv(DATA_PROCESSED / "player_mapping_tm.csv")

    tm_ids = (
        mapping["tm_player_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not tm_ids:
        raise SystemExit(
            "No tm_player_id values found in player_mapping_tm.csv. "
            "Have you filled it in?"
        )

    inj_rows = asyncio.run(fetch_injuries_for_all(tm_ids))

    injuries = pd.DataFrame(inj_rows)
    out_path = DATA_RAW / "tm_injuries_raw.csv"
    injuries.to_csv(out_path, index=False)
    print(f"Saved injuries: {len(injuries)} rows -> {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
