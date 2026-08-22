"""Tests for the reviewer-aligned JSAMS figure builders."""

import numpy as np
import pandas as pd
import pytest


def _flow() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage_order": stage,
                "stage": f"stage {stage}",
                "records": 1000 - stage,
                "players": 100 - stage,
                "same_day_events": 20 + stage,
                "lag1_events": 40 + stage,
                "combined_proxy_events": 60 + stage,
            }
            for stage in range(1, 9)
        ]
    )


def _minute_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "comparison": comparison,
                "event_minus_non_event_minutes": estimate,
                "difference_ci_low": estimate - 3.0,
                "difference_ci_high": estimate + 3.0,
                "bootstrap_replicates_estimable": 1000,
            }
            for comparison, estimate in (
                ("starting_lineup", -31.0),
                ("substitute_list", 1.0),
                ("lineup_standardized", -25.0),
            )
        ]
    )


def _minute_distribution() -> pd.DataFrame:
    """Return quantiles that show truncation among starters but not substitutes."""
    rows = []
    for role, (median_no_report, median_report) in (
        ("starting_lineup", (90.0, 53.0)),
        ("substitute_list", (18.0, 18.5)),
        ("lineup_unavailable_or_other", (90.0, 45.0)),
        ("all", (90.0, 45.0)),
    ):
        for status, median in (
            ("no_same_day_report", median_no_report),
            ("same_day_spell_start", median_report),
        ):
            rows.append(
                {
                    "lineup_role": role,
                    "event_status": status,
                    "n_appearances": 500,
                    "mean_minutes": median,
                    "p10_minutes": max(1.0, median - 30.0),
                    "p25_minutes": max(1.0, median - 15.0),
                    "median_minutes": median,
                    "p75_minutes": median + 10.0,
                    "p90_minutes": median + 20.0,
                }
            )
    return pd.DataFrame(rows)


def _denominator_roles() -> pd.DataFrame:
    rows = []
    for role, gap in (
        ("all", 0.15),
        ("starting_lineup", 0.01),
        ("substitute_list", 0.12),
        ("lineup_unavailable_or_other", 0.18),
    ):
        for denominator, estimate in (
            ("per_appearance", 1.27),
            ("fixed_90", 1.26),
            ("observed_minutes", 1.26 - gap),
        ):
            rows.append(
                {
                    "lineup_role": role,
                    "denominator": denominator,
                    "estimate": estimate,
                    "ci_low": estimate - 0.15,
                    "ci_high": estimate + 0.18,
                    "log_estimate": float(np.log(estimate)),
                    "n_rows": 5000,
                    "n_events": 300,
                    "estimable": True,
                }
            )
    return pd.DataFrame(rows)


def _curves() -> pd.DataFrame:
    rows = []
    for model_id, shift in (("additive_linear", 0.0), ("additive_spline", 0.3)):
        for burden in range(0, 181, 30):
            estimate = 5.0 + shift + burden / 60.0
            rows.append(
                {
                    "model_id": model_id,
                    "prior_minutes_7d": float(burden),
                    "estimate_per_1000_appearances": estimate,
                    "pointwise_ci_low": estimate - 0.7,
                    "pointwise_ci_high": estimate + 0.7,
                    "simultaneous_ci_low": estimate - 1.2,
                    "simultaneous_ci_high": estimate + 1.2,
                }
            )
    return pd.DataFrame(rows)


def _metric_summary(module) -> pd.DataFrame:
    rows = []
    for index, exposure_id in enumerate(module.EXPOSURE_LABELS):
        estimate = 1.12 + index / 30.0
        rows.append(
            {
                "exposure_id": exposure_id,
                "estimate": estimate,
                "ci_low": estimate - 0.12,
                "ci_high": estimate + 0.12,
                "holm_p_value_63_model_family": 0.02 if index == 0 else 1.0,
                "reject_holm_0_05": index == 0,
            }
        )
    return pd.DataFrame(rows)


def _temporal() -> pd.DataFrame:
    rows = []
    for index, block in enumerate(
        ("2017-18 to 2019-20", "2020-21 to 2021-22", "2022-23 to 2024-25")
    ):
        estimate = 1.20 + index / 20.0
        rows.append(
            {
                "temporal_block": block,
                "estimate": estimate,
                "ci_low": estimate - 0.18,
                "ci_high": estimate + 0.18,
                "heterogeneity_p_value": 0.89,
            }
        )
    return pd.DataFrame(rows)


def _conditional() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stratum_definition": stratum,
                "estimate": estimate,
                "player_cluster_ci_low": estimate - 0.3,
                "player_cluster_ci_high": estimate + 0.3,
                "multiplier_bootstrap_ci_low": estimate - 0.28,
                "multiplier_bootstrap_ci_high": estimate + 0.28,
                "n_players": 300,
                "n_discordant_strata": 360 if stratum == "player" else 570,
            }
            for stratum, estimate in (("player", 1.65), ("player-season", 2.02))
        ]
    )


def _population() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stratum_definition": stratum,
                "population": population,
                "n_rows": 1000,
                "n_players": 800 if population == "concordant_excluded" else 300,
                "n_events": 2 if population == "concordant_excluded" else 570,
            }
            for stratum in ("player", "player-season")
            for population in ("discordant_included", "concordant_excluded")
        ]
    )


def _confounding(module) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "estimate": 1.27 + 0.01 * index,
                "ci_low": 1.11 + 0.01 * index,
                "ci_high": 1.44 + 0.01 * index,
            }
            for index, model_id in enumerate(module.CONFOUNDING_LABELS)
        ]
    )


def _club_congestion(module) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "estimate": 1.36 + 0.01 * index,
                "ci_low": 1.13 + 0.01 * index,
                "ci_high": 1.65 + 0.01 * index,
            }
            for index, model_id in enumerate(module.CLUB_CONGESTION_LABELS)
        ]
    )


def _placebo(module) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "description": description,
                "estimate": 1.27 - 0.04 * index,
                "ci_low": 1.11 - 0.04 * index,
                "ci_high": 1.44 - 0.04 * index,
            }
            for index, description in enumerate(module.PLACEBO_LABELS)
        ]
    )


def _support() -> pd.DataFrame:
    shares = {"0": 0.33, "1-45": 0.11, "46-90": 0.43, "91-135": 0.04, "136-180": 0.09, ">180": 0.0}
    return pd.DataFrame(
        [
            {"exposure_band": band, "share_of_appearances": share}
            for band, share in shares.items()
        ]
    )


def _audit(module) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_dimension": "date_attribution",
                "audit_stratum": audit_stratum,
                "n_sampled": 10,
                "n_resolved": 9,
                "n_confirmed": 7 + index,
                "confirmed_proportion": (7 + index) / 9,
                "ci_low": 0.45 + index / 10.0,
                "ci_high": max((7 + index) / 9, min(1.0, 0.93 + index / 40.0)),
            }
            for index, audit_stratum in enumerate(module.AUDIT_LABELS)
        ]
    )


def _selection() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "estimate": estimate,
                "ci_low": estimate - 0.15,
                "ci_high": estimate + 0.15,
                "n_selected_appearances": 1000,
            }
            for model_id, estimate in (
                ("unweighted", 1.30),
                ("inverse_selection_weighted", 1.31),
            )
        ]
    )


def _multiverse(module) -> pd.DataFrame:
    rows = []
    for event_index, event_col in enumerate(module.EVENT_LABELS):
        for denominator_index, denominator in enumerate(module.DENOMINATOR_LABELS):
            estimate = 1.05 + event_index / 10.0 + denominator_index / 20.0
            rows.append(
                {
                    "exposure_id": "prior_minutes_7d",
                    "event_col": event_col,
                    "denominator": denominator,
                    "estimate": estimate,
                    "ci_low": estimate - 0.12,
                    "ci_high": estimate + 0.12,
                    "reject_holm_0_05": denominator == "per_appearance",
                }
            )
    return pd.DataFrame(rows)


def test_plot_builders_write_images(load_src_module, tmp_path):
    module = load_src_module("35_plot_jsams_revision.py")
    outputs = [tmp_path / f"figure_{index}.png" for index in range(5)]
    module.plot_cohort_and_denominator(
        _flow(), _minute_distribution(), _denominator_roles(), outputs[0]
    )
    module.plot_primary_and_multiverse(
        _curves(), _metric_summary(module), outputs[1], _support()
    )
    module.plot_robustness_panels(
        _temporal(),
        _metric_summary(module),
        outputs[2],
        _confounding(module),
        _club_congestion(module),
    )
    module.plot_attribution_selection_and_timing(
        _audit(module), _selection(), _multiverse(module), outputs[3]
    )
    module.plot_negative_control_exposure(_placebo(module), outputs[4])
    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 1000
        # The journal's artwork instructions accept vector PDF but not PNG at
        # the revision stage, so every figure writes a vector twin.
        assert output.with_suffix(".pdf").exists()
        assert output.with_suffix(".pdf").stat().st_size > 1000


def _gradient_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "league": ["England, Premier League", "Turkey, Süper Lig"],
            "gamma_pooled": [0.303, 0.291],
            "gamma_pooled_ci_low": [0.290, 0.276],
            "gamma_pooled_ci_high": [0.316, 0.306],
            "gamma_within_starters": [0.021, 0.018],
            "gamma_within_starters_ci_low": [0.014, 0.010],
            "gamma_within_starters_ci_high": [0.028, 0.026],
        }
    )


def test_figure_manifest_records_what_was_drawn(load_src_module):
    """Figures were the one artifact class nothing gated: a drawn label could
    outlive the table it came from and only a human reading the image would
    notice. The manifest deposits the drawn labels so a test can compare them
    with the current table."""
    module = load_src_module("35_plot_jsams_revision.py")
    gradients = _gradient_table()
    manifest = module.figure_manifest(gradients).set_index("figure")

    assert len(manifest) == 8
    assert manifest["formats"].eq("png+pdf").all()
    drawn = manifest.loc["J2_jsams_denominator_gradient", "league_labels"]
    assert drawn == "England, Premier League|Turkey, Süper Lig"
    # Only the league figure carries labels; the rest record formats alone.
    assert manifest.drop(index="J2_jsams_denominator_gradient")[
        "league_labels"
    ].eq("").all()

    # The digest belongs to the one figure drawn from the league table.
    assert manifest.loc[
        "J2_jsams_denominator_gradient", "source_digest"
    ] == module.gradient_source_digest(gradients)
    assert manifest.drop(index="J2_jsams_denominator_gradient")[
        "source_digest"
    ].eq("").all()

    with pytest.raises(KeyError, match="figure manifest gradients"):
        module.figure_manifest(pd.DataFrame())


def test_gradient_digest_tracks_the_plotted_numbers(load_src_module):
    """Currency is a question about values, not about clocks.

    The check this replaces compared file modification times, which an archive
    rewrites: unpacking the deposited record gave every source file one
    timestamp and made a current figure look stale. The digest has to move when
    a plotted number moves, and hold still when nothing plotted changes.
    """
    module = load_src_module("35_plot_jsams_revision.py")
    gradients = _gradient_table()
    baseline = module.gradient_source_digest(gradients)

    assert baseline == module.gradient_source_digest(gradients.copy())

    moved = gradients.copy()
    moved.loc[0, "gamma_pooled"] = 0.304
    assert module.gradient_source_digest(moved) != baseline

    relabelled = gradients.copy()
    relabelled.loc[1, "league"] = "Türkiye, Süper Lig"
    assert module.gradient_source_digest(relabelled) != baseline

    # A column the figure does not draw must not raise a false alarm.
    unplotted = gradients.copy()
    unplotted["n_appearances"] = [88_573, 61_204]
    assert module.gradient_source_digest(unplotted) == baseline

    with pytest.raises(KeyError, match="gradient figure digest"):
        module.gradient_source_digest(pd.DataFrame({"league": ["England"]}))


def test_cohort_validates_inputs(load_src_module, tmp_path):
    module = load_src_module("35_plot_jsams_revision.py")
    valid = (_flow(), _minute_distribution(), _denominator_roles())
    for index, label in enumerate(
        ("flow", "minute distribution", "denominator by lineup role")
    ):
        frames = list(valid)
        frames[index] = pd.DataFrame()
        with pytest.raises(KeyError, match=f"{label} missing columns"):
            module.plot_cohort_and_denominator(*frames, tmp_path / f"bad_{index}.png")
    with pytest.raises(ValueError, match="missing a required display stage"):
        module.plot_cohort_and_denominator(
            _flow().query("stage_order != 6"), valid[1], valid[2], tmp_path / "d.png"
        )


def test_cohort_panel_skips_unestimable_roles(load_src_module, tmp_path):
    """A role with too few events carries no interval and must not be drawn."""
    module = load_src_module("35_plot_jsams_revision.py")
    sparse = _denominator_roles()
    sparse.loc[sparse["lineup_role"].eq("substitute_list"), "estimable"] = False
    output = tmp_path / "sparse.png"
    module.plot_cohort_and_denominator(
        _flow(), _minute_distribution(), sparse, output
    )
    assert output.exists()
    assert output.stat().st_size > 1000


def test_cohort_panel_skips_absent_distribution_rows(load_src_module, tmp_path):
    """A role/status pair with no rows is omitted rather than drawn empty."""
    module = load_src_module("35_plot_jsams_revision.py")
    partial = _minute_distribution()
    partial = partial[
        ~(
            partial["lineup_role"].eq("substitute_list")
            & partial["event_status"].eq("same_day_spell_start")
        )
    ]
    output = tmp_path / "partial.png"
    module.plot_cohort_and_denominator(
        _flow(), partial, _denominator_roles(), output
    )
    assert output.exists()
    assert output.stat().st_size > 1000


def test_primary_validates_inputs(load_src_module, tmp_path):
    module = load_src_module("35_plot_jsams_revision.py")
    with pytest.raises(KeyError, match="additive curves missing columns"):
        module.plot_primary_and_multiverse(
            pd.DataFrame(), _metric_summary(module), tmp_path / "a.png"
        )
    with pytest.raises(KeyError, match="exposure metric summary missing columns"):
        module.plot_primary_and_multiverse(_curves(), pd.DataFrame(), tmp_path / "b.png")
    with pytest.raises(ValueError, match="curves are empty"):
        module.plot_primary_and_multiverse(
            _curves().assign(model_id="other"), _metric_summary(module), tmp_path / "c.png"
        )
    with pytest.raises(ValueError, match="forest is incomplete"):
        module.plot_primary_and_multiverse(
            _curves(), _metric_summary(module).iloc[:-1], tmp_path / "d.png"
        )


def test_primary_exposure_support_is_optional(load_src_module, tmp_path):
    """Panel A must render with or without the exposure-support band."""
    module = load_src_module("35_plot_jsams_revision.py")
    without = tmp_path / "no_support.png"
    module.plot_primary_and_multiverse(_curves(), _metric_summary(module), without)
    assert without.exists()

    empty = tmp_path / "empty_support.png"
    module.plot_primary_and_multiverse(
        _curves(), _metric_summary(module), empty, _support().iloc[0:0]
    )
    assert empty.exists()

    with pytest.raises(KeyError, match="exposure support missing columns"):
        module.plot_primary_and_multiverse(
            _curves(), _metric_summary(module), tmp_path / "bad.png",
            pd.DataFrame([{"exposure_band": "0"}]),
        )


def test_robustness_panels_validate_every_input(load_src_module, tmp_path):
    """Each of the four frames behind Figure 3 is mandatory and checked."""
    module = load_src_module("35_plot_jsams_revision.py")
    frames = {
        "temporal stability": _temporal(),
        "metric summary": _metric_summary(module),
        "confounding sensitivity": _confounding(module),
        "club congestion sensitivity": _club_congestion(module),
    }
    for index, label in enumerate(frames):
        broken = dict(frames)
        broken[label] = pd.DataFrame()
        temporal, metric, confounding, club = broken.values()
        with pytest.raises(KeyError, match=f"{label} missing columns"):
            module.plot_robustness_panels(
                temporal,
                metric,
                tmp_path / f"missing_{index}.png",
                confounding,
                club,
            )

    # A dropped temporal block must fail loudly rather than plot two of three.
    with pytest.raises(ValueError, match="estimates are incomplete"):
        module.plot_robustness_panels(
            _temporal().iloc[:-1],
            frames["metric summary"],
            tmp_path / "short.png",
            frames["confounding sensitivity"],
            frames["club congestion sensitivity"],
        )


def test_robustness_panels_skip_unlabelled_rows(load_src_module, tmp_path):
    """Rows without a display label are dropped, not plotted unlabelled."""
    module = load_src_module("35_plot_jsams_revision.py")
    extra_confounding = pd.concat(
        [
            _confounding(module),
            pd.DataFrame(
                [{"model_id": "not_displayed", "estimate": 9.0, "ci_low": 8.0, "ci_high": 10.0}]
            ),
        ],
        ignore_index=True,
    )
    output = tmp_path / "filtered.png"
    module.plot_robustness_panels(
        _temporal(),
        _metric_summary(module),
        output,
        extra_confounding,
        _club_congestion(module),
    )
    assert output.exists()
    assert output.stat().st_size > 1000


def test_negative_control_figure_filters_and_validates(load_src_module, tmp_path):
    """The supplementary placebo figure drops unlabelled rows and refuses an
    input that carries none of the displayable specifications."""
    module = load_src_module("35_plot_jsams_revision.py")
    extra_placebo = pd.concat(
        [
            _placebo(module),
            pd.DataFrame(
                [{"description": "not displayed", "estimate": 9.0, "ci_low": 8.0, "ci_high": 10.0}]
            ),
        ],
        ignore_index=True,
    )
    output = tmp_path / "negative_control.png"
    module.plot_negative_control_exposure(extra_placebo, output)
    assert output.exists()
    assert output.stat().st_size > 1000

    with pytest.raises(KeyError, match="placebo window analysis missing columns"):
        module.plot_negative_control_exposure(pd.DataFrame(), tmp_path / "bad.png")
    with pytest.raises(ValueError, match="no displayable rows"):
        module.plot_negative_control_exposure(
            _placebo(module).assign(description="not displayed"), tmp_path / "empty.png"
        )


def test_attribution_validates_inputs(load_src_module, tmp_path):
    module = load_src_module("35_plot_jsams_revision.py")
    valid = (_audit(module), _selection(), _multiverse(module))
    labels = ("outcome audit", "selection estimates", "exposure multiverse")
    for index, label in enumerate(labels):
        frames = list(valid)
        frames[index] = pd.DataFrame()
        with pytest.raises(KeyError, match=f"{label} missing columns"):
            module.plot_attribution_selection_and_timing(
                *frames, tmp_path / f"missing_{index}.png"
            )
    with pytest.raises(ValueError, match="outcome audit is incomplete"):
        module.plot_attribution_selection_and_timing(
            valid[0].iloc[:-1], valid[1], valid[2], tmp_path / "audit.png"
        )
    with pytest.raises(ValueError, match="selection estimates are incomplete"):
        module.plot_attribution_selection_and_timing(
            valid[0], valid[1].iloc[:-1], valid[2], tmp_path / "selection.png"
        )
    with pytest.raises(ValueError, match="family is incomplete"):
        module.plot_attribution_selection_and_timing(
            valid[0], valid[1], valid[2].iloc[:-1], tmp_path / "family.png"
        )


def test_forest_point_can_draw_unfilled_marker(load_src_module):
    module = load_src_module("35_plot_jsams_revision.py")
    figure, axis = module.plt.subplots()
    module._forest_point(axis, 1.2, 1.0, 1.4, 0.0, module.GOLD, marker="s", fill=False)
    assert len(axis.collections) == 1
    module.plt.close(figure)


def _gradients() -> pd.DataFrame:
    rows = []
    for league, pooled, starter in (
        ("England, Premier League", 0.47, 0.020),
        ("Spain, LaLiga", 0.43, 0.027),
        ("Portugal, Liga Portugal", 0.66, 0.031),
    ):
        rows.append(
            {
                "league": league,
                "gamma_pooled": pooled,
                "gamma_pooled_ci_low": pooled - 0.02,
                "gamma_pooled_ci_high": pooled + 0.02,
                "gamma_within_starters": starter,
                "gamma_within_starters_ci_low": starter - 0.004,
                "gamma_within_starters_ci_high": starter + 0.004,
                "iqr_recorded_minutes": 45.0,
                "iqr_starter_minutes": 10.0,
            }
        )
    return pd.DataFrame(rows)


def test_denominator_gradient_figure_renders(tmp_path, load_src_module):
    """The cross-league panel is the display that answers 'does this happen
    anywhere else', so it must render every league with both intervals."""
    module = load_src_module("35_plot_jsams_revision.py")
    output = tmp_path / "gradient.png"
    module.plot_denominator_gradient(_gradients(), output)
    assert output.exists() and output.stat().st_size > 0

    with pytest.raises(KeyError, match="denominator gradient missing columns"):
        module.plot_denominator_gradient(
            _gradients().drop(columns=["gamma_within_starters"]), output
        )


def _population_gradients() -> pd.DataFrame:
    """Two men's leagues and two women's, one of them without lineup data."""
    return pd.DataFrame(
        {
            "population": ["men", "men", "women", "women"],
            "league": [
                "England, Premier League",
                "Türkiye, Süper Lig",
                "England, FA Women's Super League",
                "Norway, Toppserien",
            ],
            "n_appearances": [84317, 78550, 3924, 3937],
            "gamma_pooled": [0.303, 0.291, 0.264, 0.248],
            "gamma_pooled_ci_low": [0.290, 0.276, 0.240, 0.222],
            "gamma_pooled_ci_high": [0.316, 0.306, 0.288, 0.274],
            "gamma_within_starters": [0.021, 0.018, 0.026, float("nan")],
            "gamma_within_starters_ci_low": [0.014, 0.010, 0.011, float("nan")],
            "gamma_within_starters_ci_high": [0.028, 0.026, 0.041, float("nan")],
        }
    )


def test_population_figure_renders_both_populations(tmp_path, load_src_module):
    """The men-and-women panel is what separates 'squads are rotated' from
    'men's football is recorded this way', so it has to draw both populations
    on one axis and mark the leagues whose source has no lineup box."""
    module = load_src_module("35_plot_jsams_revision.py")
    output = tmp_path / "population.png"
    module.plot_gradient_by_population(_population_gradients(), output)
    assert output.exists() and output.stat().st_size > 0

    with pytest.raises(KeyError, match="population gradient missing columns"):
        module.plot_gradient_by_population(
            _population_gradients().drop(columns=["population"]), output
        )


def test_the_dagger_is_explained_only_when_something_carries_one(tmp_path, load_src_module):
    """A panel whose sources all published their lineups should not advertise
    a caveat it has no instance of."""
    module = load_src_module("35_plot_jsams_revision.py")
    complete = _population_gradients()
    complete.loc[3, ["gamma_within_starters", "gamma_within_starters_ci_low",
                     "gamma_within_starters_ci_high"]] = [0.031, 0.019, 0.043]
    output = tmp_path / "no_dagger.png"
    module.plot_gradient_by_population(complete, output)
    assert output.exists() and output.stat().st_size > 0


def test_manifest_gains_a_row_only_when_the_population_figure_is_drawn(load_src_module):
    """The women's extension must not silently change what the existing
    manifest claims, because the manuscript gates read it."""
    module = load_src_module("35_plot_jsams_revision.py")
    assert len(module.figure_manifest(_gradient_table())) == 8

    extended = module.figure_manifest(_gradient_table(), _population_gradients())
    assert len(extended) == 9
    row = extended.set_index("figure").loc["J7_jsams_gradient_by_population"]
    assert row["league_labels"].startswith("men:England, Premier League|")
    assert "women:Norway, Toppserien" in row["league_labels"]
    assert len(row["source_digest"]) == 64

    # The digest has to move when a plotted number moves.
    moved = _population_gradients()
    moved.loc[2, "gamma_pooled"] = 0.265
    other = module.figure_manifest(_gradient_table(), moved).set_index("figure")
    assert other.loc["J7_jsams_gradient_by_population", "source_digest"] != row["source_digest"]

    with pytest.raises(KeyError, match="population gradients missing columns"):
        module.figure_manifest(_gradient_table(), pd.DataFrame({"other": [1]}))


def test_the_exception_leagues_are_named_in_the_legend(tmp_path, load_src_module):
    """The two leagues where restriction does not remove the gradient are the
    paper's most qualified finding. A figure that draws them like the other
    thirteen makes the reader take the exception on trust, so the panel names
    them."""
    module = load_src_module("35_plot_jsams_revision.py")
    frame = _population_gradients()
    # Push one league's within-starter bound above the threshold.
    frame.loc[2, "gamma_within_starters_ci_high"] = 0.058
    output = tmp_path / "exception.png"
    module.plot_gradient_by_population(frame, output)
    assert output.exists() and output.stat().st_size > 0


def _calibration_table():
    return pd.DataFrame(
        {
            "stratum": ["all", "starting_lineup", "substitute_list"],
            "quantity": [
                "gamma_log_minutes_on_exposure",
                "gamma_within_starting_lineup",
                "gamma_within_substitute_list",
            ],
            "gamma_predicted_attenuation": [0.303, 0.011, 0.214],
            "observed_attenuation": [0.151, 0.012, 0.117],
            "over_prediction_ratio": [2.01, 0.92, 1.83],
        }
    )


def _translation_table():
    return pd.DataFrame(
        {
            "gamma": [0.01, 0.05, 0.10, 0.30, 0.50],
            "naive_percent_understatement": [1.0, 4.9, 9.5, 25.9, 39.3],
            "calibrated_attenuation": [0.005, 0.025, 0.05, 0.149, 0.249],
            "calibrated_percent_understatement": [0.5, 2.5, 4.9, 13.9, 22.0],
            "over_prediction_ratio": [0.96, 1.20, 1.42, 2.00, 2.01],
            "ratio_is_measured": [True, True, True, True, False],
        }
    )


def _calibration_curve():
    """A sweep whose ratio rises with the gradient, as the measured one does."""
    gamma = [0.001, 0.010, 0.025, 0.080, 0.303]
    ratio = [0.94, 0.96, 1.06, 1.36, 2.01]
    attenuation = [g / r for g, r in zip(gamma, ratio)]
    return pd.DataFrame(
        {
            "minute_floor": [82.0, 65.0, 40.0, 20.0, 0.0],
            "gamma": gamma,
            "gamma_ci_low": [g * 0.7 for g in gamma],
            "gamma_ci_high": [g * 1.3 for g in gamma],
            "over_prediction_ratio": ratio,
            "observed_attenuation": attenuation,
            "attenuation_ci_low": [a * 0.6 for a in attenuation],
            "attenuation_ci_high": [a * 1.4 for a in attenuation],
        }
    )


def test_calibration_figure_draws_both_panels(tmp_path, load_src_module):
    """The identity over-predicts by twofold and the paper's advice is built on
    the quantity it over-predicts, so the discrepancy gets a figure rather than
    a sentence in a supplement nobody opens."""
    module = load_src_module("35_plot_jsams_revision.py")
    output = tmp_path / "calibration.png"
    module.plot_identity_calibration(
        _calibration_table(), _translation_table(), _calibration_curve(), output
    )
    assert output.exists() and output.stat().st_size > 0
    assert output.with_suffix(".pdf").exists()


def test_calibration_figure_survives_a_grid_without_the_threshold(tmp_path, load_src_module):
    """The threshold annotation is drawn only when the translation grid
    contains the threshold, so a grid that does not carry it still renders."""
    module = load_src_module("35_plot_jsams_revision.py")
    translation = _translation_table()
    translation = translation[~translation["gamma"].eq(0.05)]
    output = tmp_path / "calibration_nothreshold.png"
    module.plot_identity_calibration(
        _calibration_table(), translation, _calibration_curve(), output
    )
    assert output.exists() and output.stat().st_size > 0


def test_calibration_figure_names_the_table_it_cannot_read(tmp_path, load_src_module):
    module = load_src_module("35_plot_jsams_revision.py")
    output = tmp_path / "bad.png"
    with pytest.raises(KeyError, match="identity calibration"):
        module.plot_identity_calibration(
            _calibration_table().drop(columns=["over_prediction_ratio"]),
            _translation_table(),
            _calibration_curve(),
            output,
        )
    with pytest.raises(KeyError, match="threshold translation"):
        module.plot_identity_calibration(
            _calibration_table(), _translation_table().drop(columns=["gamma"]),
            _calibration_curve(), output,
        )
    # The curve is what makes the calibrated line a measurement rather than an
    # extrapolation, so a figure cannot be drawn without it.
    with pytest.raises(KeyError, match="calibration curve"):
        module.plot_identity_calibration(
            _calibration_table(), _translation_table(),
            _calibration_curve().drop(columns=["over_prediction_ratio"]), output,
        )


def test_decision_rule_figure_renders_with_its_tallies(tmp_path, load_src_module):
    """The decision rule is the only thing in the paper a practitioner runs, so
    it gets drawn rather than described, with the fifteen measured leagues
    distributed across its branches."""
    module = load_src_module("35_plot_jsams_revision.py")
    output = tmp_path / "rule.png"
    module.plot_decision_rule(
        {"neutral": 0, "restrict": 13, "per_appearance": 2}, output
    )
    assert output.exists() and output.stat().st_size > 0
    assert output.with_suffix(".pdf").exists()


def test_decision_rule_figure_defaults_a_missing_tally_to_zero(tmp_path, load_src_module):
    """A branch nothing landed in is drawn as zero rather than left blank: an
    empty box reads as an oversight, and a zero reads as a result."""
    module = load_src_module("35_plot_jsams_revision.py")
    output = tmp_path / "rule_partial.png"
    module.plot_decision_rule({"restrict": 13}, output)
    assert output.exists() and output.stat().st_size > 0
