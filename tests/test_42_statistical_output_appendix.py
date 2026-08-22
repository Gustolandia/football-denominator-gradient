"""Tests for the statistical-output appendix.

The journal wants the software's own output, not a transcription of it, and a
second script that re-derives a model is exactly where a transcription error
hides: a different floor or a dropped calendar term still yields a regression
table that looks entirely respectable. So the behaviour under test is mostly
the refusal -- that a fit which disagrees with the deposited estimate stops
the run rather than being printed.
"""

import numpy as np
import pandas as pd
import pytest


MODULE = "42_statistical_output_appendix.py"
GRADIENT = "37_denominator_gradient.py"


def _panel(module, gradient, competition="GB1", players=30, appearances=16, seed=5):
    """An appearance panel carrying the columns the gradient fit reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for player in range(players):
        regular = player % 2 == 0
        start = pd.Timestamp("2024-02-01") + pd.Timedelta(days=int(rng.integers(0, 10)))
        for index in range(appearances):
            minutes = max(1.0, (90.0 if regular else 18.0) + rng.normal(0.0, 1.5))
            rows.append(
                {
                    module.PLAYER_ID_COL: f"P{player:04d}",
                    "date": start + pd.Timedelta(days=4 * index),
                    module.MINUTES_COL: minutes,
                    module.COMPETITION_COL: competition,
                    module.ROLE_COL: module.STARTER_ROLE if minutes > 60 else "substitutes",
                }
            )
    frame = pd.DataFrame(rows)
    return gradient.add_calendar_phase(gradient.add_prior_window_minutes(frame))


def _terms(gradient):
    return f"prior_minutes_{gradient.PRIMARY_WINDOW}d", list(gradient.CALENDAR_TERMS)


def test_the_refit_reproduces_the_published_gradient(load_src_module):
    """The whole appendix rests on this: the model printed here is the model
    that produced the number in the table."""
    module = load_src_module(MODULE)
    gradient = load_src_module(GRADIENT)
    exposure, calendar = _terms(gradient)
    panel = _panel(module, gradient)

    published = gradient.denominator_gradient(panel)
    refit = module.fit_reported_model(panel, exposure, calendar)
    assert float(refit.params["exposure_per_90"]) == pytest.approx(
        published["gamma"], abs=1e-12
    )
    assert int(refit.nobs) == published["n_rows"]


def test_design_frame_drops_what_the_estimator_drops(load_src_module):
    module = load_src_module(MODULE)
    gradient = load_src_module(GRADIENT)
    exposure, calendar = _terms(gradient)
    panel = _panel(module, gradient)

    dirty = pd.concat([panel, panel.head(3).assign(**{module.MINUTES_COL: 0.0})])
    work = module.design_frame(dirty, exposure, calendar)
    assert len(work) == len(panel)
    assert work[module.MINUTES_COL].gt(0).all()
    assert "log_recorded_minutes" in work and "exposure_per_90" in work

    with pytest.raises(KeyError, match="appendix design frame missing columns"):
        module.design_frame(pd.DataFrame({"other": [1]}), exposure, calendar)


def test_collect_models_covers_both_strata_and_labels_leagues(load_src_module):
    module = load_src_module(MODULE)
    gradient = load_src_module(GRADIENT)
    exposure, calendar = _terms(gradient)
    panel = _panel(module, gradient)

    entries = module.collect_models(
        panel, module.MEN, {"GB1": "England, Premier League"}, exposure, calendar
    )
    assert [e["stratum"] for e in entries] == [module.POOLED, module.STARTERS]
    assert all(e["league"] == "England, Premier League" for e in entries)
    assert all("OLS Regression Results" in e["summary"] for e in entries)
    assert entries[0]["model_id"].endswith("_pooled")

    with pytest.raises(KeyError, match="appendix panel missing columns"):
        module.collect_models(
            pd.DataFrame({"other": [1]}), module.MEN, {}, exposure, calendar
        )


def test_a_league_without_lineup_status_reports_only_the_pooled_fit(load_src_module):
    """Some sources publish minutes but no lineup box. That league still has a
    pooled gradient and must not acquire an invented starter one."""
    module = load_src_module(MODULE)
    gradient = load_src_module(GRADIENT)
    exposure, calendar = _terms(gradient)
    panel = _panel(module, gradient).drop(columns=[module.ROLE_COL])

    entries = module.collect_models(panel, module.WOMEN, {}, exposure, calendar)
    assert len(entries) == 1 and entries[0]["stratum"] == module.POOLED

    # A role column present but naming no starters is the same situation.
    none_started = _panel(module, gradient).assign(**{module.ROLE_COL: "substitutes"})
    assert len(module.collect_models(none_started, module.WOMEN, {}, exposure, calendar)) == 1

    with pytest.raises(KeyError, match="stratum source missing columns"):
        module.stratum_frames(pd.DataFrame({"other": [1]}))


def test_a_disagreeing_fit_stops_the_run(load_src_module):
    module = load_src_module(MODULE)
    gradient = load_src_module(GRADIENT)
    exposure, calendar = _terms(gradient)
    entries = module.collect_models(
        _panel(module, gradient), module.MEN, {"GB1": "England"}, exposure, calendar
    )

    truthful = pd.DataFrame(
        {"league": ["England"], "gamma_pooled": [entries[0]["gamma"]]}
    )
    module.verify_against_deposited(entries, truthful, "gamma_pooled", module.POOLED)

    wrong = pd.DataFrame({"league": ["England"], "gamma_pooled": [entries[0]["gamma"] + 0.01]})
    with pytest.raises(ValueError, match="does not match the deposited"):
        module.verify_against_deposited(entries, wrong, "gamma_pooled", module.POOLED)

    # A league absent from the table, or carrying no estimable gradient, is
    # skipped rather than treated as a contradiction.
    absent = pd.DataFrame({"league": ["Somewhere else"], "gamma_pooled": [0.5]})
    module.verify_against_deposited(entries, absent, "gamma_pooled", module.POOLED)
    unestimable = pd.DataFrame({"league": ["England"], "gamma_pooled": [np.nan]})
    module.verify_against_deposited(entries, unestimable, "gamma_pooled", module.POOLED)

    with pytest.raises(KeyError, match="deposited gradient table missing columns"):
        module.verify_against_deposited(entries, pd.DataFrame({"x": [1]}), "gamma_pooled", module.POOLED)


def test_appendix_prints_one_section_per_model(load_src_module):
    module = load_src_module(MODULE)
    gradient = load_src_module(GRADIENT)
    exposure, calendar = _terms(gradient)
    entries = module.collect_models(
        _panel(module, gradient), module.MEN, {"GB1": "England"}, exposure, calendar
    )

    text = module.render_appendix(entries, "data/processed/results")
    assert text.startswith("APPENDIX: FULL STATISTICAL OUTPUT")
    for entry in entries:
        assert entry["model_id"] in text
        assert entry["summary"] in text
    assert text.count("OLS Regression Results") == len(entries)
    # The reader is told what the coefficient means and that it was checked.
    assert "exposure_per_90 is the denominator gradient" in text
    assert "stops the" in text

    manifest = module.appendix_manifest(entries)
    assert list(manifest["model_id"]) == [e["model_id"] for e in entries]
    assert manifest["gamma"].notna().all()

    with pytest.raises(ValueError, match="at least one fitted model"):
        module.render_appendix([], "x")
    with pytest.raises(ValueError, match="at least one fitted model"):
        module.appendix_manifest([])
