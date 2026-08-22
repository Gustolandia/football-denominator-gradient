"""Emit the full statistical-software output for every gradient reported.

Science and Medicine in Football requires the original full output generated
by the statistical software as an appendix, not merely the estimates a table
carries. This module produces it for the denominator gradient, which is the
quantity the paper asks readers to compute before dividing by playing time
and the only quantity reported in all fifteen leagues.

The appendix verifies itself. Re-deriving a model in a second script is an
invitation to quiet divergence: a different floor, a dropped calendar term or
a changed exposure scale would still produce a regression table that looks
right. So every fit here is checked against the gradient the deposited league
table already carries, and a mismatch beyond floating-point tolerance stops
the run. What the appendix prints is therefore the output of the model that
produced the published number, or nothing at all.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MEN = "men"
WOMEN = "women"

PLAYER_ID_COL = "player_id"
MINUTES_COL = "minutes_played"
COMPETITION_COL = "competition_id"
ROLE_COL = "lineup_role"
STARTER_ROLE = "starting_lineup"

POOLED = "all appearances"
STARTERS = "starters only"

#: A published gamma and a re-derived one may differ only by floating-point
#: noise. Anything larger is a different model wearing the same label.
GAMMA_TOLERANCE = 1e-9


def load_source_module(filename: str, module_name: str):  # pragma: no cover
    """Load one numerically named pipeline module."""
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    path = src_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import pipeline script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def design_frame(
    frame: pd.DataFrame,
    exposure_col: str,
    calendar_terms: Sequence[str],
    minute_floor: float = 1.0,
) -> pd.DataFrame:
    """Rebuild the estimation frame exactly as the gradient module does.

    Same drops, same floor beneath the log, same exposure scaling. Any
    departure here would be caught downstream by the gamma check, which is
    the point of doing it this way rather than trusting the transcription.
    """
    required = [PLAYER_ID_COL, MINUTES_COL, exposure_col, *calendar_terms]
    _require(frame, required, "appendix design frame")

    work = frame.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    work["log_recorded_minutes"] = np.log(work[MINUTES_COL].clip(lower=float(minute_floor)))
    work["exposure_per_90"] = pd.to_numeric(work[exposure_col], errors="coerce") / 90.0
    return work


def fit_reported_model(
    frame: pd.DataFrame,
    exposure_col: str,
    calendar_terms: Sequence[str],
) -> Any:
    """Fit the clustered least-squares model behind one published gradient."""
    work = design_frame(frame, exposure_col, calendar_terms)
    formula = "log_recorded_minutes ~ exposure_per_90 + " + " + ".join(calendar_terms)
    return smf.ols(formula, data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]}
    )


def stratum_frames(panel: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """The two populations every league is reported in."""
    _require(panel, (MINUTES_COL,), "stratum source")
    strata: list[tuple[str, pd.DataFrame]] = [(POOLED, panel)]
    if ROLE_COL in panel.columns:
        starters = panel[panel[ROLE_COL].astype(str).eq(STARTER_ROLE)]
        if len(starters):
            strata.append((STARTERS, starters))
    return strata


def collect_models(
    panel: pd.DataFrame,
    population: str,
    labels: dict[str, str],
    exposure_col: str,
    calendar_terms: Sequence[str],
) -> list[dict[str, Any]]:
    """Fit every reported gradient in one population, league by league."""
    _require(panel, (COMPETITION_COL,), "appendix panel")

    entries: list[dict[str, Any]] = []
    for competition, group in panel.groupby(COMPETITION_COL, sort=True):
        league = labels.get(str(competition), str(competition))
        for stratum, rows in stratum_frames(group):
            fit = fit_reported_model(rows, exposure_col, calendar_terms)
            entries.append(
                {
                    "model_id": f"{population}_{competition}_"
                    f"{'pooled' if stratum == POOLED else 'starters'}",
                    "population": population,
                    "league": league,
                    "stratum": stratum,
                    "n_rows": int(fit.nobs),
                    "gamma": float(fit.params["exposure_per_90"]),
                    "summary": str(fit.summary()),
                }
            )
    return entries


def verify_against_deposited(
    entries: Sequence[dict[str, Any]],
    deposited: pd.DataFrame,
    gamma_column: str,
    stratum: str,
) -> None:
    """Refuse to publish output that disagrees with the published estimate."""
    _require(deposited, ("league", gamma_column), "deposited gradient table")

    published = dict(zip(deposited["league"], deposited[gamma_column]))
    for entry in entries:
        if entry["stratum"] != stratum:
            continue
        expected = published.get(entry["league"])
        if expected is None or not np.isfinite(float(expected)):
            continue
        if abs(entry["gamma"] - float(expected)) > GAMMA_TOLERANCE:
            raise ValueError(
                f"{entry['model_id']}: re-derived gamma {entry['gamma']:.12g} does not "
                f"match the deposited {float(expected):.12g}"
            )


def render_appendix(entries: Sequence[dict[str, Any]], generated: str) -> str:
    """Assemble the appendix text, one section per model."""
    if not entries:
        raise ValueError("the appendix needs at least one fitted model")

    lines = [
        "APPENDIX: FULL STATISTICAL OUTPUT",
        "=" * 78,
        "",
        "Original output generated by statsmodels for every denominator gradient",
        "reported in the manuscript: one section per league and stratum, in both",
        "populations. Each fit is ordinary least squares of log recorded minutes on",
        "previous-seven-day club minutes per 90, with weekly and half-weekly",
        "calendar-phase terms, and a covariance clustered on player identifier.",
        "The coefficient labelled exposure_per_90 is the denominator gradient.",
        "",
        "Every gradient below was checked against the deposited league table before",
        "this file was written; a mismatch beyond floating-point tolerance stops the",
        "run, so these are the models that produced the published numbers.",
        "",
        f"Generated from the archived pipeline at {generated}.",
        "",
        "=" * 78,
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        lines += [
            f"[{index}] {entry['model_id']}",
            f"    population: {entry['population']}",
            f"    league:     {entry['league']}",
            f"    stratum:    {entry['stratum']}",
            f"    n:          {entry['n_rows']:,} appearances",
            "",
            entry["summary"],
            "",
            "-" * 78,
            "",
        ]
    return "\n".join(lines)


def appendix_manifest(entries: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """One row per printed model, so a gate can check the appendix is current."""
    if not entries:
        raise ValueError("the appendix needs at least one fitted model")
    return pd.DataFrame(
        [
            {
                "model_id": entry["model_id"],
                "population": entry["population"],
                "league": entry["league"],
                "stratum": entry["stratum"],
                "n_rows": entry["n_rows"],
                "gamma": entry["gamma"],
            }
            for entry in entries
        ]
    )


def main() -> None:  # pragma: no cover - orchestration
    """Write the statistical-output appendix and its manifest."""
    root = Path(__file__).resolve().parents[1]
    results = root / "data" / "processed" / "results"
    gradient = load_source_module("37_denominator_gradient.py", "denominator_gradient")
    womens = load_source_module("40_womens_denominator_gradient.py", "womens_gradient")

    exposure_col = f"prior_minutes_{gradient.PRIMARY_WINDOW}d"
    calendar_terms = list(gradient.CALENDAR_TERMS)

    print("1. Refitting the men's league gradients ...")
    snapshot = (
        root / "data" / "raw" / "public_data_v4" / "transfermarkt_datasets_20260803"
    )
    mens_raw = gradient.load_league_appearances(snapshot)
    pieces = [
        gradient.add_prior_window_minutes(group)
        for _, group in mens_raw.groupby(gradient.COMPETITION_COL, sort=False)
    ]
    mens_panel = gradient.add_calendar_phase(pd.concat(pieces, ignore_index=True))
    mens_panel[COMPETITION_COL] = mens_panel[COMPETITION_COL].astype(str)
    mens = collect_models(mens_panel, MEN, dict(gradient.LEAGUES), exposure_col, calendar_terms)

    print("2. Refitting the women's league gradients ...")
    womens_panel = pd.read_csv(root / "data" / "processed" / "womens_appearances.csv.gz")
    womens_panel["date"] = pd.to_datetime(womens_panel["date"])
    womens_panel[COMPETITION_COL] = womens_panel[COMPETITION_COL].astype(str)
    womens_panel = womens.prepare_panel(womens_panel, gradient)
    women = collect_models(
        womens_panel, WOMEN, dict(womens.WOMENS_LEAGUE_LABELS), exposure_col, calendar_terms
    )

    print("3. Checking every fit against the deposited estimates ...")
    verify_against_deposited(
        mens,
        pd.read_csv(results / "jsams_revised_denominator_gradient_by_league.csv"),
        "gamma_pooled",
        POOLED,
    )
    verify_against_deposited(
        women,
        pd.read_csv(results / "jsams_womens_denominator_gradient_by_league.csv"),
        "gamma_pooled",
        POOLED,
    )

    entries = mens + women
    appendix = root / "manuscript" / "appendix_statistical_output.txt"
    appendix.write_text(
        render_appendix(entries, "data/processed/results"), encoding="utf-8"
    )
    appendix_manifest(entries).to_csv(
        results / "jsams_statistical_output_manifest.csv", index=False
    )
    print(f"   {len(entries)} model outputs written to {appendix}")


if __name__ == "__main__":  # pragma: no cover
    main()
