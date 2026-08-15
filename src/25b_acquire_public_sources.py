"""Cache the pinned reusable public sources used by the v4 extension."""

from pathlib import Path

from public_data_sources import (
    acquire_independent_results,
    acquire_worldcup_lineups,
    public_source_catalog,
)


def main() -> None:  # pragma: no cover - network/cache orchestration
    """Acquire immutable source files and write the source decision catalog."""
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw" / "public_data_v4"
    processed = root / "data" / "processed" / "public_data_v4"
    processed.mkdir(parents=True, exist_ok=True)
    snapshot = acquire_independent_results(raw)
    lineup_snapshot = acquire_worldcup_lineups(raw)
    public_source_catalog().to_csv(processed / "public_data_source_catalog.csv", index=False)
    print(f"Independent senior-results snapshot: {snapshot}")
    print(f"Independent World Cup lineup snapshot: {lineup_snapshot}")


if __name__ == "__main__":  # pragma: no cover
    main()
