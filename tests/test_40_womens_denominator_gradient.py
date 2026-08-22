"""Tests for the women's gradient run and the men-women contrast.

The estimator itself is tested against the men's module; what is tested here
is that the women's panel reaches it unchanged, that the two populations are
compared on the same terms, and that a disagreement between data sources is
reported as disqualifying rather than smoothed over.
"""

import numpy as np
import pandas as pd
import pytest


MODULE = "40_womens_denominator_gradient.py"
GRADIENT = "37_denominator_gradient.py"


def _appearances(competition="189", players=30, appearances=16, coupled=True, seed=3):
    """An appearance panel in the five columns the gradient module reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for player in range(players):
        regular = player % 2 == 0
        start = pd.Timestamp("2024-02-01") + pd.Timedelta(days=int(rng.integers(0, 10)))
        for index in range(appearances):
            minutes = (90.0 if regular else 18.0) if coupled else float(
                rng.choice([90.0, 18.0])
            )
            minutes = max(1.0, minutes + rng.normal(0.0, 1.5))
            rows.append(
                {
                    "player_id": f"W{player:04d}",
                    "date": start + pd.Timedelta(days=4 * index),
                    "minutes_played": minutes,
                    "competition_id": competition,
                    "lineup_role": "starting_lineup" if minutes > 60 else "substitutes",
                }
            )
    return pd.DataFrame(rows)


def _gradient_table(competition, league, pooled, pooled_low, starter, starter_high, n=5000):
    return pd.DataFrame(
        [
            {
                "competition_id": competition,
                "league": league,
                "n_appearances": n,
                "gamma_pooled": pooled,
                "gamma_pooled_ci_low": pooled_low,
                "gamma_within_starters": starter,
                "gamma_within_starters_ci_high": starter_high,
            }
        ]
    )


def test_panel_reaches_the_gradient_module_unchanged(load_src_module):
    module = load_src_module(MODULE)
    gradient = load_src_module(GRADIENT)

    panel = module.prepare_panel(_appearances(), gradient)
    for column in gradient.CALENDAR_TERMS:
        assert column in panel.columns
    assert f"prior_minutes_{gradient.PRIMARY_WINDOW}d" in panel.columns

    fitted = gradient.gradient_by_league(panel)
    assert len(fitted) == 1
    assert bool(fitted.iloc[0]["estimable"])

    with pytest.raises(KeyError, match="women's panel missing columns"):
        module.prepare_panel(pd.DataFrame({"other": [1]}), gradient)


def test_panel_windows_are_computed_within_each_league(load_src_module):
    module = load_src_module(MODULE)
    gradient = load_src_module(GRADIENT)
    # One player identifier appearing in two leagues must not accumulate
    # exposure across them.
    shared = pd.concat(
        [_appearances(competition="189", players=4), _appearances(competition="185", players=4)],
        ignore_index=True,
    )
    panel = module.prepare_panel(shared, gradient)
    assert set(panel["competition_id"]) == {"189", "185"}
    assert len(panel) == len(shared)


def test_relabel_restores_league_names(load_src_module):
    module = load_src_module(MODULE)
    gradients = pd.DataFrame(
        {"competition_id": ["189", "999"], "league": ["189", "999"], "gamma_pooled": [0.3, 0.2]}
    )
    out = module.relabel_leagues(gradients)
    assert out.loc[0, "league"] == module.WOMENS_LEAGUE_LABELS["189"]
    # A code outside the registry keeps whatever label it arrived with.
    assert out.loc[1, "league"] == "999"

    with pytest.raises(KeyError, match="gradient frame missing columns"):
        module.relabel_leagues(pd.DataFrame({"other": [1]}))


def test_combine_populations_names_the_population(load_src_module):
    module = load_src_module(MODULE)
    mens = _gradient_table("GB1", "England, Premier League", 0.30, 0.28, 0.02, 0.04)
    womens = _gradient_table("189", "England, FA WSL", 0.26, 0.22, 0.01, 0.03)

    combined = module.combine_populations(mens, womens)
    assert list(combined.columns)[0] == module.POPULATION_COL
    assert set(combined[module.POPULATION_COL]) == {module.MENS, module.WOMENS}
    assert len(combined) == 2

    with pytest.raises(KeyError, match="men's gradients missing columns"):
        module.combine_populations(pd.DataFrame({"other": [1]}), womens)
    with pytest.raises(KeyError, match="women's gradients missing columns"):
        module.combine_populations(mens, pd.DataFrame({"other": [1]}))


def test_population_contrast_counts_where_the_pattern_holds(load_src_module):
    module = load_src_module(MODULE)
    mens = pd.concat(
        [
            _gradient_table("GB1", "England", 0.30, 0.28, 0.02, 0.04),
            _gradient_table("ES1", "Spain", 0.26, 0.24, 0.01, 0.03),
        ],
        ignore_index=True,
    )
    womens = pd.concat(
        [
            _gradient_table("189", "WSL", 0.24, 0.20, 0.02, 0.04),
            # A league where the gradient survives restriction to starters.
            _gradient_table("185", "Toppserien", 0.22, 0.18, 0.14, 0.19),
        ],
        ignore_index=True,
    )
    combined = module.combine_populations(mens, womens)
    contrast = module.population_contrast(combined, 0.05).set_index(module.POPULATION_COL)

    assert contrast.loc[module.MENS, "leagues"] == 2
    assert contrast.loc[module.MENS, "leagues_material_pooled"] == 2
    assert contrast.loc[module.MENS, "leagues_negligible_within_starters"] == 2
    assert bool(contrast.loc[module.MENS, "pattern_holds_in_every_league"])

    assert contrast.loc[module.WOMENS, "leagues_negligible_within_starters"] == 1
    assert not bool(contrast.loc[module.WOMENS, "pattern_holds_in_every_league"])
    assert contrast.loc[module.WOMENS, "appearances"] == 10000

    with pytest.raises(KeyError, match="combined gradients missing columns"):
        module.population_contrast(pd.DataFrame({"other": [1]}), 0.05)


def test_population_contrast_treats_an_unestimable_starter_fit_as_not_collapsing(load_src_module):
    module = load_src_module(MODULE)
    mens = _gradient_table("GB1", "England", 0.30, 0.28, 0.02, 0.04)
    womens = _gradient_table("189", "WSL", 0.24, 0.20, np.nan, np.nan)
    combined = module.combine_populations(mens, womens)
    contrast = module.population_contrast(combined, 0.05).set_index(module.POPULATION_COL)
    # An absent within-starter interval cannot be counted as evidence that
    # restricting to starters fixes anything.
    assert contrast.loc[module.WOMENS, "leagues_negligible_within_starters"] == 0
    assert not bool(contrast.loc[module.WOMENS, "pattern_holds_in_every_league"])


def test_cross_source_agreement_reads_intervals(load_src_module):
    module = load_src_module(MODULE)
    agreeing = pd.DataFrame(
        [
            {"source": "transfermarkt", "competition_id": "GB1",
             "gamma": 0.303, "ci_low": 0.29, "ci_high": 0.32},
            {"source": "fbref", "competition_id": "GB1",
             "gamma": 0.297, "ci_low": 0.28, "ci_high": 0.31},
        ]
    )
    out = module.cross_source_agreement(agreeing)
    assert bool(out.iloc[0]["intervals_overlap"])
    assert "agree" in out.iloc[0]["verdict"]
    assert out.iloc[0]["absolute_difference"] == pytest.approx(0.006)

    disagreeing = agreeing.copy()
    disagreeing.loc[1, ["gamma", "ci_low", "ci_high"]] = [0.05, 0.02, 0.08]
    out = module.cross_source_agreement(disagreeing)
    assert not bool(out.iloc[0]["intervals_overlap"])
    assert "not reportable" in out.iloc[0]["verdict"]


def test_cross_source_agreement_needs_exactly_two_fits(load_src_module):
    module = load_src_module(MODULE)
    lonely = pd.DataFrame(
        [{"source": "fbref", "competition_id": "GB1",
          "gamma": 0.3, "ci_low": 0.28, "ci_high": 0.32}]
    )
    with pytest.raises(ValueError, match="exactly two fits"):
        module.cross_source_agreement(lonely)

    with pytest.raises(KeyError, match="cross-source fits missing columns"):
        module.cross_source_agreement(pd.DataFrame({"other": [1]}))


def test_coverage_note_names_the_league_seasons(load_src_module):
    module = load_src_module(MODULE)
    completeness = pd.DataFrame(
        {
            "league": ["WSL", "WSL", "Toppserien"],
            "season": ["2023-2024", "2024-2025", "2024"],
            "admitted": [True, True, False],
        }
    )
    note = module.womens_coverage_note(completeness)
    assert "2 league-seasons" in note
    assert "1 leagues" in note

    empty = completeness.assign(admitted=False)
    assert module.womens_coverage_note(empty).startswith("No women's league-season")

    with pytest.raises(KeyError, match="completeness frame missing columns"):
        module.womens_coverage_note(pd.DataFrame({"other": [1]}))
