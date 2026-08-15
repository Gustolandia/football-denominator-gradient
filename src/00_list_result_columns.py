#!/usr/bin/env python
"""
00_list_result_columns.py

Utility: scan data/processed for CSV outputs and inventory their columns.

Outputs:
- data/processed/results/columns_inventory.csv
- data/processed/results/columns_inventory.txt

Printed summary:
- number of CSVs found
- per-file column counts
- column names per file

Run from repo root:
    python src/00_list_result_columns.py
"""

from pathlib import Path
import pandas as pd


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    results_dir = processed_dir / "results"

    print(f"Repo root: {root}")
    print(f"Scanning processed dir: {processed_dir}")
    print(f"Results dir (for outputs): {results_dir}")

    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results dir: {results_dir}")

    # Include all CSVs anywhere under data/processed (including results/)
    csv_paths = sorted(processed_dir.rglob("*.csv"))
    print(f"\nFound {len(csv_paths)} CSVs.\n")

    inventory_rows = []
    txt_lines = []

    for path in csv_paths:
        rel = path.relative_to(root)

        try:
            df = pd.read_csv(path, nrows=5, low_memory=False)
        except Exception as e:
            print(f"[WARN] Could not read {rel}: {e}")
            inventory_rows.append({"file": str(rel), "n_cols": 0, "columns": ""})
            txt_lines.append(f"{rel}\n  <READ ERROR>\n")
            continue

        cols = list(df.columns)
        inventory_rows.append(
            {"file": str(rel), "n_cols": len(cols), "columns": ", ".join(cols)}
        )

        txt_lines.append(f"{rel}")
        if cols:
            txt_lines.append("  " + ", ".join(cols))
        else:
            txt_lines.append("  <NO COLUMNS?>")
        txt_lines.append("")  # blank line spacer

    inv = pd.DataFrame(inventory_rows).sort_values("file").reset_index(drop=True)

    # Print compact summary like your current output
    print(inv[["file", "n_cols"]].to_string(index=False))

    out_csv = results_dir / "columns_inventory.csv"
    out_txt = results_dir / "columns_inventory.txt"

    inv.to_csv(out_csv, index=False)
    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"\nSaved -> {out_csv}")
    print(f"Saved -> {out_txt}")


if __name__ == "__main__":  # pragma: no cover
    main()
# =====================================================================
# End of file
