#!/usr/bin/env python
"""Run the prespecified club-only versus club-plus-country model comparison.

This script deliberately reuses the primary Poisson spline and recovery
functions from ``18_match_proxy_poisson_splines_perminute.py``.  For each scope
it changes only the generic exposure-engine columns; the outcome, match rows,
history strata, spline basis, calendar covariates and multiplicity treatment
remain the same.  The coverage gate is read before fitting: a failed gate
labels the expanded exposure as a sensitivity analysis.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PLAYER_ID = "tm_player_id"
PRIMARY_EVENT = "injury_event_matchproxy"
PRIMARY_HISTORY = "fragility_group"
FROZEN_CLUB_SCOPE = "frozen_club_all"
FROZEN_CLUB_PLUS_SENIOR_NATIONAL_SCOPE = "frozen_club_plus_senior_national"
FROZEN_CLUB_PLUS_SENIOR_ALL_SCOPE = "frozen_club_plus_senior_all"
FROZEN_CLUB_PLUS_BROADER_SCOPE = "frozen_club_plus_broader_international"
SENIOR_NATIONAL_ONLY_SCOPE = "senior_competitive_national_only"
SENIOR_ALL_NATIONAL_ONLY_SCOPE = "senior_all_national_only"
BROADER_NATIONAL_ONLY_SCOPE = "broader_international_only"
PRIMARY_SCOPES = (FROZEN_CLUB_SCOPE, FROZEN_CLUB_PLUS_SENIOR_NATIONAL_SCOPE)
AUDIT_SCOPES = (
    *PRIMARY_SCOPES,
    FROZEN_CLUB_PLUS_SENIOR_ALL_SCOPE,
    FROZEN_CLUB_PLUS_BROADER_SCOPE,
)
EXPANDED_SCOPE_INCREMENT = {
    FROZEN_CLUB_PLUS_SENIOR_NATIONAL_SCOPE: SENIOR_NATIONAL_ONLY_SCOPE,
    FROZEN_CLUB_PLUS_SENIOR_ALL_SCOPE: SENIOR_ALL_NATIONAL_ONLY_SCOPE,
    FROZEN_CLUB_PLUS_BROADER_SCOPE: BROADER_NATIONAL_ONLY_SCOPE,
}


def _required(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def load_primary_model_module(path: Path) -> ModuleType:
    """Load the existing primary model module without duplicating its formulas."""
    spec = importlib.util.spec_from_file_location("v4_primary_model", Path(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load primary model module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scope_feature_columns(scope: str) -> dict[str, str]:
    """Return the generic engine fields that replace baseline exposure fields."""
    return {
        "burden": f"{scope}_minutes_last_7d",
        "days_since": f"{scope}_days_since_previous_appearance",
        "matches": f"{scope}_matches_last_7d",
        "national_minutes": f"{scope}_national_minutes_last_7d",
    }


def prepare_scope_model_panel(match_panel: pd.DataFrame, exposure_features: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Merge one scope's features onto the unchanged existing match-proxy rows.

    The direct v4 contrast retains the frozen club exposure and adds only
    national minutes.  Refreshed snapshot club scopes remain available in the
    feature file as reconstruction sensitivities, but are not substituted into
    the comparator because the frozen all-competition definition includes
    recorded club friendlies.
    """
    columns = scope_feature_columns(scope)
    _required(match_panel, [PLAYER_ID, "date", "all_minutes_played", PRIMARY_EVENT, PRIMARY_HISTORY], "match panel")
    out = match_panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if scope == FROZEN_CLUB_SCOPE:
        _required(out, ["all_minutes_last_7d", "all_games_last_7d", "days_since_last_match"], "match panel")
        out["scope_matches_last_7d"] = pd.to_numeric(out["all_games_last_7d"], errors="coerce")
        out["scope_national_minutes_last_7d"] = 0.0
        out["exposure_scope"] = scope
        return out
    if scope in EXPANDED_SCOPE_INCREMENT:
        increment_scope = EXPANDED_SCOPE_INCREMENT[scope]
        national_columns = {
            "minutes": f"{increment_scope}_minutes_last_7d",
            "matches": f"{increment_scope}_matches_last_7d",
            "days_since": f"{increment_scope}_days_since_previous_appearance",
        }
        _required(
            out,
            ["all_minutes_last_7d", "all_games_last_7d", "days_since_last_match"],
            "match panel",
        )
        _required(exposure_features, [PLAYER_ID, "date", *national_columns.values()], "exposure features")
        features = exposure_features[[PLAYER_ID, "date", *national_columns.values()]].copy()
        features["date"] = pd.to_datetime(features["date"], errors="coerce")
        if features.duplicated([PLAYER_ID, "date"]).any():
            raise ValueError("Exposure features must have one row per player-date")
        out = out.merge(features, on=[PLAYER_ID, "date"], how="left", validate="many_to_one")
        national_minutes = pd.to_numeric(out[national_columns["minutes"]], errors="coerce").fillna(0.0)
        national_matches = pd.to_numeric(out[national_columns["matches"]], errors="coerce").fillna(0.0)
        national_days = pd.to_numeric(out[national_columns["days_since"]], errors="coerce")
        out["all_minutes_last_7d"] = pd.to_numeric(out["all_minutes_last_7d"], errors="coerce") + national_minutes
        out["scope_matches_last_7d"] = pd.to_numeric(out["all_games_last_7d"], errors="coerce") + national_matches
        out["scope_national_minutes_last_7d"] = national_minutes
        out["days_since_last_match"] = np.fmin(
            pd.to_numeric(out["days_since_last_match"], errors="coerce"),
            national_days,
        )
        out["exposure_scope"] = scope
        return out
    _required(exposure_features, [PLAYER_ID, "date", *columns.values()], "exposure features")
    features = exposure_features[[PLAYER_ID, "date", *columns.values()]].copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    if features.duplicated([PLAYER_ID, "date"]).any():
        raise ValueError("Exposure features must have one row per player-date")
    out = out.merge(features, on=[PLAYER_ID, "date"], how="left", validate="many_to_one")
    out["all_minutes_last_7d"] = pd.to_numeric(out[columns["burden"]], errors="coerce")
    out["days_since_last_match"] = pd.to_numeric(out[columns["days_since"]], errors="coerce")
    out["scope_matches_last_7d"] = pd.to_numeric(out[columns["matches"]], errors="coerce")
    out["scope_national_minutes_last_7d"] = pd.to_numeric(out[columns["national_minutes"]], errors="coerce")
    out["exposure_scope"] = scope
    return out


def baseline_parity_report(match_panel: pd.DataFrame, exposure_features: pd.DataFrame) -> pd.DataFrame:
    """Audit frozen parity and expose strict refreshed-club differences."""
    merged = prepare_scope_model_panel(match_panel, exposure_features, "club_competitive")
    baseline = pd.to_numeric(match_panel["all_minutes_last_7d"], errors="coerce")
    rebuilt = pd.to_numeric(merged["all_minutes_last_7d"], errors="coerce")
    comparable = baseline.notna() & rebuilt.notna()
    difference = (rebuilt - baseline).abs()
    return pd.DataFrame(
        [
            {"metric": "frozen_comparator_match_rows", "value": len(match_panel)},
            {"metric": "frozen_comparator_burden_mismatch_rows", "value": 0},
            {"metric": "refreshed_strict_club_rows_with_comparable_burden", "value": int(comparable.sum())},
            {"metric": "refreshed_strict_club_burden_mismatch_rows", "value": int((difference.gt(1e-9) & comparable).sum())},
            {"metric": "refreshed_strict_club_max_absolute_difference", "value": float(difference[comparable].max()) if comparable.any() else np.nan},
            {"metric": "refreshed_strict_club_rows_without_burden", "value": int(rebuilt.isna().sum())},
        ]
    )


def _bundle_tables(bundle: MappingLike, scope: str, analysis_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add scope labels to selected predictions, formal tests, and model audit rows."""
    selected = bundle["selected"].copy()
    selected["exposure_scope"] = scope
    selected["analysis_label"] = analysis_label
    effects = bundle["effect_modification"].copy()
    effects["exposure_scope"] = scope
    effects["analysis_label"] = analysis_label
    audit = pd.DataFrame(
        [
            {
                "exposure_scope": scope,
                "analysis_label": analysis_label,
                "n_model_rows": int(bundle["n_model_rows"]),
                "n_events": int(bundle["n_events"]),
                "dispersion": float(bundle["dispersion"]),
                "estimator": bundle["estimator"],
            }
        ]
    )
    return selected, effects, audit


class MappingLike(dict):
    """Type-only wrapper allowing concise tests with primary-model bundle dicts."""


def run_scope_model(
    base_model: Any,
    panel: pd.DataFrame,
    scope: str,
    analysis_label: str = "unweighted",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the unchanged primary spline and recovery analyses for one scope."""
    prepared = base_model.prepare_model_frame(panel, PRIMARY_EVENT, PRIMARY_HISTORY)
    bundle = MappingLike(base_model.run_prediction_bundle(prepared, PRIMARY_EVENT))
    bundle["n_model_rows"] = len(prepared)
    bundle["n_events"] = int(prepared[PRIMARY_EVENT].sum())
    bundle["estimator"] = bundle.get("estimator", "clustered_glm")
    selected, effects, audit = _bundle_tables(bundle, scope, analysis_label)
    recovery = base_model.recovery_interval_trend_tests(prepared, PRIMARY_EVENT).copy()
    recovery["exposure_scope"] = scope
    recovery["analysis_label"] = analysis_label
    recovery["analysis_component"] = "recovery_interval"
    effects["analysis_component"] = "previous_7d_minutes"
    return selected, effects, recovery, audit


def add_comparison_multiplicity(tests: pd.DataFrame) -> pd.DataFrame:
    """Apply the same explicit Holm/BH reporting rule across v4 formal tests."""
    out = tests.copy()
    _required(out, ["p_value"], "formal comparison tests")
    values = pd.to_numeric(out["p_value"], errors="coerce")
    valid = values.notna() & np.isfinite(values)
    out["p_holm_v4_comparison"] = np.nan
    out["p_bh_v4_comparison"] = np.nan
    if valid.any():
        out.loc[valid, "p_holm_v4_comparison"] = multipletests(values.loc[valid], method="holm")[1]
        out.loc[valid, "p_bh_v4_comparison"] = multipletests(values.loc[valid], method="fdr_bh")[1]
    out["reject_holm_v4_0_05"] = out["p_holm_v4_comparison"].lt(0.05)
    out["reject_bh_v4_0_05"] = out["p_bh_v4_comparison"].lt(0.05)
    return out


def coverage_decision(gate: pd.DataFrame) -> str:
    """Return the model label dictated by the coverage gate artifact."""
    _required(gate, ["primary_v4_exposure_allowed"], "coverage gate")
    return "primary_v4" if gate["primary_v4_exposure_allowed"].fillna(False).astype(bool).all() else "sensitivity_only"


def run_prespecified_comparison(
    base_model: Any,
    match_panel: pd.DataFrame,
    exposure_features: pd.DataFrame,
    gate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run only the protocol's two scope models with common model machinery."""
    selected_tables: list[pd.DataFrame] = []
    test_tables: list[pd.DataFrame] = []
    audit_tables: list[pd.DataFrame] = []
    label = coverage_decision(gate)
    for scope in PRIMARY_SCOPES:
        panel = prepare_scope_model_panel(match_panel, exposure_features, scope)
        selected, burden_tests, recovery_tests, audit = run_scope_model(base_model, panel, scope, label)
        selected_tables.append(selected)
        test_tables.extend([burden_tests, recovery_tests])
        audit_tables.append(audit)
    tests = add_comparison_multiplicity(pd.concat(test_tables, ignore_index=True))
    return pd.concat(selected_tables, ignore_index=True), tests, pd.concat(audit_tables, ignore_index=True)


def main() -> None:  # pragma: no cover
    """Run and save the protocol's parity and identical-model comparison tables."""
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed" / "public_data_v4"
    match_panel = pd.read_csv(root / "data" / "processed" / "player_match_panel_all_comp.csv", low_memory=False)
    features = pd.read_csv(processed / "match_exposure_scope_features.csv", low_memory=False)
    gate = pd.read_csv(processed / "exposure_coverage_audit.csv", low_memory=False)
    parity = baseline_parity_report(match_panel, features)
    parity.to_csv(processed / "baseline_parity_report.csv", index=False)
    base_model = load_primary_model_module(root / "src" / "18_match_proxy_poisson_splines_perminute.py")
    selected, tests, audit = run_prespecified_comparison(base_model, match_panel, features, gate)
    selected.to_csv(processed / "v4_scope_selected_predictions.csv", index=False)
    tests.to_csv(processed / "v4_model_comparison.csv", index=False)
    audit.to_csv(processed / "v4_model_input_audit.csv", index=False)
    print(parity.to_string(index=False))
    print(f"Model-comparison rows: {len(tests)}")


if __name__ == "__main__":  # pragma: no cover
    main()
