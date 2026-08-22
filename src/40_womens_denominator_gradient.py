"""Measure the denominator gradient in women's leagues and set it beside men's.

The gradient module makes no reference to the source its panel came from, so
nothing in it needs changing to run here: this module supplies the same five
columns from the women's snapshot, calls the same fits, and relabels the
competition codes. Reusing the estimator rather than reimplementing it is the
point --- a difference between the two populations has to come from the data,
not from two versions of the same regression.

Two sources for one comparison is the obvious objection, since the men's panel
comes from a Transfermarkt dump and the women's from FBref match reports. The
answer is not an argument but a measurement: ``cross_source_agreement`` fits the
same league-season on both sources and reports whether the intervals overlap.
If they do, the source is not carrying the result. If they do not, the
comparison is not reportable and the paper says so.

What the comparison is for: the manuscript claims the gradient is a property of
how squads are rotated rather than of English football or of one data
provider. Women's leagues are the sharpest available test of that claim,
because almost everything else about them differs --- squad sizes, fixture
density, season geometry, and the recording regime itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

MENS = "men"
WOMENS = "women"

COMPETITION_COL = "competition_id"
POPULATION_COL = "population"

#: Labels for the women's competitions. The gradient module falls back to the
#: bare competition code for anything outside the men's registry, so the codes
#: are turned back into league names here rather than by editing that module.
WOMENS_LEAGUE_LABELS: Mapping[str, str] = {
    # FBref competition codes
    "189": "England, FA Women's Super League",
    "183": "Germany, Frauen-Bundesliga",
    "193": "France, Première Ligue",
    "208": "Italy, Serie A Femminile",
    "195": "Netherlands, Eredivisie Vrouwen",
    "185": "Norway, Toppserien",
    "187": "Sweden, Damallsvenskan",
    "340": "Denmark, Kvindeligaen",
    # Soccerdonna competition codes
    "ENG1": "England, FA Women's Super League",
    "BL1": "Germany, Frauen-Bundesliga",
    "IT1": "Italy, Serie A Femminile",
    "ESP1": "Spain, Primera División Femenina",
    "SWE1": "Sweden, Damallsvenskan",
    "NOR1": "Norway, Toppserien",
    "SUI1": "Switzerland, Women's Super League",
}


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


def _median(values: pd.Series) -> float:
    """Median of the finite values, or NaN when there are none.

    A league whose within-starter fit was not estimable contributes no number
    here; asking numpy for the median of nothing is a warning, not an answer.
    """
    finite = values[np.isfinite(values)]
    if finite.empty:
        return float("nan")
    return float(finite.median())


def prepare_panel(appearances: pd.DataFrame, gradient_module: Any) -> pd.DataFrame:
    """Add the exposure window and calendar phase the gradient fit expects.

    The men's pipeline does this per competition so that one league's fixture
    calendar cannot leak into another's exposure window. The same applies here,
    and more sharply: the Nordic leagues play through a summer when the rest of
    Europe is idle.
    """
    _require(appearances, (COMPETITION_COL,), "women's panel")
    pieces = []
    for _, group in appearances.groupby(COMPETITION_COL, sort=False):
        pieces.append(gradient_module.add_prior_window_minutes(group))
    return gradient_module.add_calendar_phase(pd.concat(pieces, ignore_index=True))


def relabel_leagues(
    gradients: pd.DataFrame,
    labels: Mapping[str, str] = WOMENS_LEAGUE_LABELS,
) -> pd.DataFrame:
    """Replace fallback competition codes with league names."""
    _require(gradients, (COMPETITION_COL, "league"), "gradient frame")
    out = gradients.copy()
    out["league"] = [
        labels.get(str(code), str(existing))
        for code, existing in zip(out[COMPETITION_COL], out["league"])
    ]
    return out


def combine_populations(mens: pd.DataFrame, womens: pd.DataFrame) -> pd.DataFrame:
    """Stack the two league tables with the population named in a column.

    One table, not two, because the claim under test is about both together:
    if the gradient behaves the same way in leagues that share almost nothing
    but the practice of rotating a squad, the mechanism is the rotation.
    """
    for frame, label in ((mens, "men's gradients"), (womens, "women's gradients")):
        _require(frame, (COMPETITION_COL, "league", "gamma_pooled"), label)

    left = mens.copy()
    right = womens.copy()
    left[POPULATION_COL] = MENS
    right[POPULATION_COL] = WOMENS
    combined = pd.concat([left, right], ignore_index=True)
    columns = [POPULATION_COL] + [c for c in combined.columns if c != POPULATION_COL]
    return combined[columns]


def population_contrast(
    combined: pd.DataFrame,
    negligible: float,
) -> pd.DataFrame:
    """Summarise, per population, what the gradients say as a group.

    The paper's claim is not that the two populations share a number. It is
    that they share a pattern: the pooled gradient is materially above
    negligible, and restricting to starters removes it. Counting leagues where
    each holds is what makes that claim checkable rather than rhetorical.
    """
    _require(
        combined,
        (POPULATION_COL, "gamma_pooled", "gamma_pooled_ci_low",
         "gamma_within_starters", "gamma_within_starters_ci_high"),
        "combined gradients",
    )

    rows = []
    for population, group in combined.groupby(POPULATION_COL, sort=True):
        pooled = pd.to_numeric(group["gamma_pooled"], errors="coerce")
        pooled_low = pd.to_numeric(group["gamma_pooled_ci_low"], errors="coerce")
        starter = pd.to_numeric(group["gamma_within_starters"], errors="coerce")
        starter_high = pd.to_numeric(group["gamma_within_starters_ci_high"], errors="coerce")

        material = pooled_low > negligible
        collapses = starter_high.notna() & (starter_high <= negligible)
        rows.append(
            {
                POPULATION_COL: population,
                "leagues": int(len(group)),
                "appearances": int(pd.to_numeric(group["n_appearances"]).sum()),
                "median_gamma_pooled": _median(pooled),
                "min_gamma_pooled": float(pooled.min()),
                "max_gamma_pooled": float(pooled.max()),
                "leagues_material_pooled": int(material.sum()),
                "median_gamma_within_starters": _median(starter),
                "leagues_negligible_within_starters": int(collapses.sum()),
                "pattern_holds_in_every_league": bool(material.all() and collapses.all()),
            }
        )
    return pd.DataFrame(rows)


def cross_source_agreement(fits: pd.DataFrame) -> pd.DataFrame:
    """Compare gradients fitted on the same league from two different sources.

    An overlap is not proof the sources agree, but a non-overlap is proof they
    do not, and that is the finding that would stop the men-and-women
    comparison being reportable at all.
    """
    _require(
        fits,
        ("source", COMPETITION_COL, "gamma", "ci_low", "ci_high"),
        "cross-source fits",
    )
    rows = []
    for competition, group in fits.groupby(COMPETITION_COL, sort=True):
        if len(group) != 2:
            raise ValueError(
                f"cross-source check needs exactly two fits for {competition}, got {len(group)}"
            )
        first, second = group.iloc[0], group.iloc[1]
        overlap = (
            float(first["ci_low"]) <= float(second["ci_high"])
            and float(second["ci_low"]) <= float(first["ci_high"])
        )
        rows.append(
            {
                COMPETITION_COL: competition,
                "source_a": first["source"],
                "gamma_a": float(first["gamma"]),
                "source_b": second["source"],
                "gamma_b": float(second["gamma"]),
                "absolute_difference": abs(float(first["gamma"]) - float(second["gamma"])),
                "intervals_overlap": bool(overlap),
                "verdict": (
                    "sources agree within uncertainty"
                    if overlap
                    else "sources disagree; the comparison is not reportable"
                ),
            }
        )
    return pd.DataFrame(rows)


def womens_coverage_note(completeness: pd.DataFrame) -> str:
    """One sentence naming exactly which league-seasons the gradients rest on."""
    _require(completeness, ("league", "season", "admitted"), "completeness frame")
    admitted = completeness.loc[completeness["admitted"]]
    if admitted.empty:
        return "No women's league-season met the fixture-coverage threshold."
    leagues = admitted["league"].nunique()
    seasons = sorted(admitted["season"].unique())
    return (
        f"Women's gradients are fitted on {len(admitted)} league-seasons across "
        f"{leagues} leagues, seasons {seasons[0]} to {seasons[-1]}, each retained "
        "only where the fixture list and the parsed match reports agree."
    )


def main() -> None:  # pragma: no cover - orchestration
    """Fit the gradient in women's leagues and contrast it with the men's."""
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    results = processed / "results"
    results.mkdir(parents=True, exist_ok=True)

    gradient = load_source_module("37_denominator_gradient.py", "denominator_gradient")

    print("1. Reading the women's appearance snapshot ...")
    appearances = pd.read_csv(processed / "womens_appearances.csv.gz")
    appearances["date"] = pd.to_datetime(appearances["date"])
    appearances[COMPETITION_COL] = appearances[COMPETITION_COL].astype(str)
    print(f"   {len(appearances):,} appearances in {appearances[COMPETITION_COL].nunique()} leagues")

    print("2. Building exposure windows and calendar phase ...")
    panel = prepare_panel(appearances, gradient)

    print("3. Fitting the gradient in every women's league ...")
    womens = relabel_leagues(gradient.gradient_by_league(panel))
    womens_rule = gradient.diagnostic_decision_rule(womens)

    print("4. Contrasting with the men's leagues ...")
    mens = pd.read_csv(results / "jsams_revised_denominator_gradient_by_league.csv")
    mens[COMPETITION_COL] = mens[COMPETITION_COL].astype(str)
    combined = combine_populations(mens, womens)
    contrast = population_contrast(combined, gradient.NEGLIGIBLE_GAMMA)

    completeness = pd.read_csv(results / "jsams_womens_league_season_completeness.csv")
    note = womens_coverage_note(completeness)
    print(f"   {note}")

    outputs = {
        "womens_denominator_gradient_by_league": womens,
        "womens_denominator_gradient_decision_rule": womens_rule,
        "denominator_gradient_by_population": combined,
        "denominator_gradient_population_contrast": contrast,
        "womens_coverage_note": pd.DataFrame({"note": [note]}),
    }

    cross_path = results / "jsams_womens_cross_source_fits.csv"
    if cross_path.exists():
        outputs["denominator_gradient_cross_source"] = cross_source_agreement(
            pd.read_csv(cross_path)
        )

    for name, table in outputs.items():
        table.to_csv(results / f"jsams_{name}.csv", index=False)
    print(f"Wrote {len(outputs)} tables to {results}")


if __name__ == "__main__":  # pragma: no cover
    main()
