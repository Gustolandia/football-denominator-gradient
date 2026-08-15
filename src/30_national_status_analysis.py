"""Analyse recorded national-team matchday status without calling it training.

Transfermarkt's participation ledger distinguishes played, in-squad,
injured/absent, and not-in-squad records. The source does not observe training.
This module therefore uses ``squad_only`` as a matchday involvement proxy and
keeps every other state explicit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from v4_statistics import poisson_rate_interval


PLAYER_ID = "tm_player_id"
EVENT = "injury_event_matchproxy"
HISTORY = "fragility_group"
WINDOW_DAYS = (3, 5, 7, 14, 28)
MODEL_STATUSES = (
    "no_recent_senior_record",
    "explicitly_not_in_squad",
    "recorded_unavailable",
    "squad_only",
    "played",
)
MODEL_EXPOSURES = ("squad_only", "played")
MIN_EXPOSED_EVENTS_PER_HISTORY = 5
MIN_EXPOSED_ROWS_PER_HISTORY = 50
SENIOR_TYPE_IDS = {11, 19}
YOUTH_TYPE_IDS = {17, 20}


def _required(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def load_module(path: Path, name: str) -> ModuleType:
    """Load a numbered pipeline module without renaming the source file."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_national_record(
    participation_state: object,
    competition_type_id: object,
    competition_id: object,
) -> str:
    """Classify one public national matchday record with no training inference."""
    type_id = pd.to_numeric(competition_type_id, errors="coerce")
    if pd.isna(type_id):
        level = "unknown"
    elif int(type_id) in SENIOR_TYPE_IDS:
        level = "senior"
    elif int(type_id) in YOUTH_TYPE_IDS:
        level = "youth_or_olympic"
    else:
        level = "unknown"
    competition = "friendly" if str(competition_id).strip().casefold() == "fs" else "competitive"
    state = str(participation_state).strip().casefold()
    if state == "played":
        prefix = "played"
    elif state == "in squad":
        prefix = "squad_only"
    elif state in {"injured", "absent"}:
        prefix = "recorded_unavailable"
    elif state == "not in squad":
        prefix = "not_in_squad"
    else:
        prefix = "other"
    return f"{prefix}_{level}_{competition}"


def build_status_ledger(
    records: pd.DataFrame,
    harmonized_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """Attach record classes and independently matched game context."""
    _required(
        records,
        [
            PLAYER_ID,
            "game_id",
            "date",
            "participation_state",
            "competition_type_id",
            "competition_id",
        ],
        "national record audit",
    )
    out = records.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["game_id"] = out["game_id"].astype(str)
    out["national_record_class"] = [
        classify_national_record(state, type_id, competition)
        for state, type_id, competition in zip(
            out["participation_state"],
            out["competition_type_id"],
            out["competition_id"],
        )
    ]
    context_columns = [
        "game_id",
        "independent_schedule_verified",
        "external_tournament",
        "external_city",
        "external_country",
        "external_neutral",
    ]
    available = [column for column in context_columns if column in harmonized_appearances]
    if len(available) > 1:
        context = harmonized_appearances[available].copy()
        context["game_id"] = context["game_id"].astype(str)
        context = context.drop_duplicates("game_id")
        out = out.merge(context, on="game_id", how="left")
    if "independent_schedule_verified" not in out:
        out["independent_schedule_verified"] = False
    out["independent_schedule_verified"] = (
        out["independent_schedule_verified"].astype("boolean").fillna(False).astype(bool)
    )
    return out


def _window_count(
    event_dates: np.ndarray,
    event_indicator: np.ndarray,
    target_dates: np.ndarray,
    days: int,
) -> np.ndarray:
    right = np.searchsorted(event_dates, target_dates, side="left")
    left = np.searchsorted(
        event_dates, target_dates - np.timedelta64(days, "D"), side="left"
    )
    cumulative = np.concatenate(([0], np.cumsum(event_indicator.astype(int))))
    return cumulative[right] - cumulative[left]


def build_status_features(
    ledger: pd.DataFrame,
    targets: pd.DataFrame,
    windows: Iterable[int] = WINDOW_DAYS,
) -> pd.DataFrame:
    """Create rolling national matchday-state counts for each club-match row."""
    _required(ledger, [PLAYER_ID, "date", "national_record_class"], "status ledger")
    _required(targets, [PLAYER_ID, "date"], "target matches")
    out = targets[[PLAYER_ID, "date"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    classes = (
        "played_senior_competitive",
        "played_senior_friendly",
        "squad_only_senior_competitive",
        "squad_only_senior_friendly",
        "recorded_unavailable_senior_competitive",
        "recorded_unavailable_senior_friendly",
        "not_in_squad_senior_competitive",
        "not_in_squad_senior_friendly",
        "played_youth_or_olympic_competitive",
        "squad_only_youth_or_olympic_competitive",
    )
    window_values = tuple(int(value) for value in windows)
    for window in window_values:
        for record_class in classes:
            out[f"{record_class}_last_{window}d"] = 0
    status_source = ledger.copy()
    status_source["date"] = pd.to_datetime(status_source["date"], errors="coerce").dt.normalize()
    grouped = {identifier: group.sort_values("date") for identifier, group in status_source.groupby(PLAYER_ID)}
    for identifier, indices in out.groupby(PLAYER_ID, sort=False).groups.items():
        player_records = grouped.get(identifier)
        if player_records is None or player_records.empty:
            continue
        event_dates = player_records["date"].to_numpy(dtype="datetime64[ns]")
        target_dates = out.loc[indices, "date"].to_numpy(dtype="datetime64[ns]")
        for record_class in classes:
            indicator = player_records["national_record_class"].eq(record_class).to_numpy()
            for window in window_values:
                out.loc[indices, f"{record_class}_last_{window}d"] = _window_count(
                    event_dates, indicator, target_dates, window
                )
    for window in window_values:
        played = out[
            [
                f"played_senior_competitive_last_{window}d",
                f"played_senior_friendly_last_{window}d",
            ]
        ].sum(axis=1)
        squad_only = out[
            [
                f"squad_only_senior_competitive_last_{window}d",
                f"squad_only_senior_friendly_last_{window}d",
            ]
        ].sum(axis=1)
        unavailable = out[
            [
                f"recorded_unavailable_senior_competitive_last_{window}d",
                f"recorded_unavailable_senior_friendly_last_{window}d",
            ]
        ].sum(axis=1)
        not_selected = out[
            [
                f"not_in_squad_senior_competitive_last_{window}d",
                f"not_in_squad_senior_friendly_last_{window}d",
            ]
        ].sum(axis=1)
        out[f"senior_played_matchdays_last_{window}d"] = played
        out[f"senior_squad_only_matchdays_last_{window}d"] = squad_only
        out[f"senior_recorded_unavailable_matchdays_last_{window}d"] = unavailable
        out[f"senior_not_in_squad_matchdays_last_{window}d"] = not_selected
        out[f"national_status_last_{window}d"] = np.select(
            [played.gt(0), squad_only.gt(0), unavailable.gt(0), not_selected.gt(0)],
            ["played", "squad_only", "recorded_unavailable", "explicitly_not_in_squad"],
            default="no_recent_senior_record",
        )
    return out


def attach_status_features(panel: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Merge one-to-one status features onto the frozen club-match panel."""
    _required(panel, [PLAYER_ID, "date"], "match panel")
    _required(features, [PLAYER_ID, "date"], "status features")
    if features.duplicated([PLAYER_ID, "date"]).any():
        raise ValueError("status features contain duplicate player-date rows")
    left = panel.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right = features.copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    merged = left.merge(right, on=[PLAYER_ID, "date"], how="left", validate="one_to_one")
    status_columns = [column for column in right if column.startswith("national_status_last_")]
    for column in status_columns:
        merged[column] = merged[column].fillna("no_recent_senior_record")
    count_columns = [column for column in right if column not in {PLAYER_ID, "date", *status_columns}]
    merged[count_columns] = merged[count_columns].fillna(0)
    return merged


def status_rate_table(
    panel: pd.DataFrame,
    window_days: int = 7,
    event_col: str = EVENT,
) -> pd.DataFrame:
    """Report observed-minute proxy incidence with exact Poisson intervals."""
    status_column = f"national_status_last_{window_days}d"
    _required(panel, [PLAYER_ID, "minutes_played", event_col, status_column, HISTORY], "status panel")
    match_rows = panel.loc[pd.to_numeric(panel["minutes_played"], errors="coerce").gt(0)].copy()
    rows: list[dict[str, object]] = []
    for (history, status), group in match_rows.groupby([HISTORY, status_column], dropna=False, sort=False):
        events = int(pd.to_numeric(group[event_col], errors="coerce").fillna(0).sum())
        minutes = float(pd.to_numeric(group["minutes_played"], errors="coerce").fillna(0).sum())
        exposure_hours = minutes / 60.0
        rate = events / exposure_hours * 1_000.0 if exposure_hours > 0 else np.nan
        low, high = poisson_rate_interval(events, exposure_hours, scale=1_000.0)
        rows.append(
            {
                "window_days": window_days,
                "history_stratum": history,
                "national_status": status,
                "n_players": group[PLAYER_ID].nunique(),
                "n_match_rows": len(group),
                "n_events": events,
                "observed_match_hours": exposure_hours,
                "events_per_1000_match_hours": rate,
                "ci_low": low,
                "ci_high": high,
                "interval_method": "exact_poisson_95",
            }
        )
    return pd.DataFrame(rows)


def prepare_status_model_frame(
    base_model: Any,
    panel: pd.DataFrame,
    window_days: int,
    event_col: str,
    exposure_status: str,
) -> pd.DataFrame:
    """Prepare one supported status contrast on the unchanged primary risk set."""
    if exposure_status not in MODEL_EXPOSURES:
        raise ValueError(f"Unknown model exposure: {exposure_status}")
    frame = base_model.prepare_model_frame(panel, event_col, HISTORY)
    source = f"national_status_last_{window_days}d"
    _required(frame, [source], "prepared status panel")
    reference = frame[source].isin(
        ["no_recent_senior_record", "explicitly_not_in_squad"]
    )
    exposed = frame[source].eq(exposure_status)
    frame = frame.loc[reference | exposed].copy()
    frame["national_status"] = np.where(exposed.loc[frame.index], exposure_status, "no_observed_senior_squad_involvement")
    frame["recent_status"] = frame["national_status"].eq(exposure_status).astype(int)
    frame["higher_history"] = frame["model_group"].eq("fragile").astype(int)
    return frame


def _status_formula(base_model: Any, frame: pd.DataFrame, event_col: str, controls: str) -> str:
    burden_max = float(frame["all_minutes_last_7d"].max())
    spline = base_model.spline_basis_expression(burden_max)
    return (
        f"{event_col} ~ recent_status * higher_history + {spline} * higher_history "
        "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
        f"{controls}"
    )


def fit_status_model(
    base_model: Any,
    frame: pd.DataFrame,
    event_col: str,
    denominator: str,
    controls: str = "",
):  # pragma: no cover - statsmodels numerical integration
    """Fit one clustered Poisson model for recorded national status."""
    if frame[event_col].sum() <= 0:
        raise ValueError(f"No events available for {event_col}")
    if frame["recent_status"].nunique() < 2:
        raise ValueError("National status has no variation")
    if denominator == "observed_minutes":
        offset = frame["log_minutes_played"]
    elif denominator == "fixed_90":
        offset = pd.Series(np.log(90.0), index=frame.index)
    elif denominator == "per_match":
        offset = None
    else:
        raise ValueError(f"Unknown denominator mode: {denominator}")
    kwargs: dict[str, object] = {}
    if offset is not None:
        kwargs["offset"] = offset
    model = smf.glm(
        formula=_status_formula(base_model, frame, event_col, controls),
        data=frame,
        family=sm.families.Poisson(),
        **kwargs,
    )
    return model.fit(cov_type="cluster", cov_kwds={"groups": frame[PLAYER_ID]})


def linear_ratio(
    params: pd.Series,
    covariance: pd.DataFrame,
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Return an exponentiated linear contrast with a normal-theory CI."""
    missing = sorted(set(weights) - set(params.index))
    if missing:
        raise KeyError(f"Model is missing contrast parameters: {missing}")
    vector = pd.Series(0.0, index=params.index)
    for name, weight in weights.items():
        vector.loc[name] = float(weight)
    log_estimate = float(vector @ params)
    variance = float(vector.to_numpy() @ covariance.to_numpy() @ vector.to_numpy())
    standard_error = float(np.sqrt(max(variance, 0.0)))
    statistic = log_estimate / standard_error if standard_error > 0 else np.nan
    return {
        "estimate": float(np.exp(log_estimate)),
        "ci_low": float(np.exp(log_estimate - 1.96 * standard_error)),
        "ci_high": float(np.exp(log_estimate + 1.96 * standard_error)),
        "log_estimate": log_estimate,
        "standard_error": standard_error,
        "test_statistic": statistic,
        "df": 1.0,
        "p_value": float(2.0 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan,
    }


def status_contrasts(result: Any, exposure_status: str) -> pd.DataFrame:
    """Extract one status IRR in each history stratum and their interaction."""
    params = result.params
    covariance = result.cov_params()
    rows: list[dict[str, object]] = []
    definitions = (
        ("intermediate_vs_no_involvement", "intermediate prior-injury history", {"recent_status": 1.0}),
        ("higher_vs_no_involvement", "higher prior-injury history", {"recent_status": 1.0, "recent_status:higher_history": 1.0}),
        ("history_interaction", "higher versus intermediate history", {"recent_status:higher_history": 1.0}),
    )
    for contrast, history, weights in definitions:
        row = linear_ratio(params, covariance, weights)
        row.update(
            {
                "national_status": exposure_status,
                "contrast_id": f"{exposure_status}_{contrast}",
                "history_stratum": history,
                "effect_measure": "incidence_rate_ratio",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def status_support(frame: pd.DataFrame, event_col: str) -> pd.DataFrame:
    """Count exposed rows and events before deciding whether a model is estimable."""
    rows = []
    for group_name, group in frame.groupby("model_group", sort=False):
        exposed = group["recent_status"].eq(1)
        rows.append(
            {
                "model_group": group_name,
                "n_rows": len(group),
                "n_exposed_rows": int(exposed.sum()),
                "n_events": int(group[event_col].sum()),
                "n_exposed_events": int(group.loc[exposed, event_col].sum()),
            }
        )
    return pd.DataFrame(rows)


def support_is_adequate(support: pd.DataFrame) -> bool:
    """Require stable exposed support in both primary history strata."""
    required_groups = {"regular", "fragile"}
    if set(support["model_group"]) != required_groups:
        return False
    return bool(
        support["n_exposed_events"].ge(MIN_EXPOSED_EVENTS_PER_HISTORY).all()
        and support["n_exposed_rows"].ge(MIN_EXPOSED_ROWS_PER_HISTORY).all()
    )


def sparse_contrasts(exposure_status: str) -> pd.DataFrame:
    """Return explicit non-estimates when a status contrast is too sparse."""
    rows = []
    for contrast, history in (
        ("intermediate_vs_no_involvement", "intermediate prior-injury history"),
        ("higher_vs_no_involvement", "higher prior-injury history"),
        ("history_interaction", "higher versus intermediate history"),
    ):
        rows.append(
            {
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "log_estimate": np.nan,
                "standard_error": np.nan,
                "test_statistic": np.nan,
                "df": 1.0,
                "p_value": np.nan,
                "national_status": exposure_status,
                "contrast_id": f"{exposure_status}_{contrast}",
                "history_stratum": history,
                "effect_measure": "incidence_rate_ratio",
            }
        )
    return pd.DataFrame(rows)


def status_specifications() -> list[dict[str, object]]:
    """Return the finite status-model robustness family."""
    specifications = [
        {
            "specification_id": f"window_{window}d_observed",
            "window_days": window,
            "event_col": EVENT,
            "denominator": "observed_minutes",
            "controls": "",
            "family": "window",
        }
        for window in WINDOW_DAYS
    ]
    specifications.extend(
        [
            {"specification_id": "same_day_7d", "window_days": 7, "event_col": "injury_event_matchproxy_same_day", "denominator": "observed_minutes", "controls": "", "family": "outcome_timing"},
            {"specification_id": "lag1_7d", "window_days": 7, "event_col": "injury_event_matchproxy_lag1", "denominator": "observed_minutes", "controls": "", "family": "outcome_timing"},
            {"specification_id": "description_specific_7d", "window_days": 7, "event_col": "injury_event_matchproxy_specific", "denominator": "observed_minutes", "controls": "", "family": "outcome_classification"},
            {"specification_id": "fixed90_7d", "window_days": 7, "event_col": EVENT, "denominator": "fixed_90", "controls": "", "family": "denominator"},
            {"specification_id": "per_match_7d", "window_days": 7, "event_col": EVENT, "denominator": "per_match", "controls": "", "family": "denominator"},
            {"specification_id": "measured_controls_7d", "window_days": 7, "event_col": EVENT, "denominator": "observed_minutes", "controls": " + age_years + C(position_group) + C(club_season)", "family": "measured_confounding"},
        ]
    )
    return specifications


def add_status_multiplicity(results: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm and BH correction across the complete status family."""
    _required(results, ["p_value"], "status results")
    out = results.copy()
    p_values = pd.to_numeric(out["p_value"], errors="coerce")
    valid = p_values.notna() & np.isfinite(p_values)
    out["p_holm_status_family"] = np.nan
    out["p_bh_status_family"] = np.nan
    if valid.any():
        out.loc[valid, "p_holm_status_family"] = multipletests(
            p_values.loc[valid], method="holm"
        )[1]
        out.loc[valid, "p_bh_status_family"] = multipletests(
            p_values.loc[valid], method="fdr_bh"
        )[1]
    out["reject_holm_status_0_05"] = out["p_holm_status_family"].lt(0.05)
    out["reject_bh_status_0_05"] = out["p_bh_status_family"].lt(0.05)
    return out


def run_status_family(
    base_model: Any,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # pragma: no cover - integration GLMs
    """Fit every declared status specification and retain support counts."""
    result_tables: list[pd.DataFrame] = []
    support_tables: list[pd.DataFrame] = []
    for specification in status_specifications():
        event_col = str(specification["event_col"])
        for exposure_status in MODEL_EXPOSURES:
            frame = prepare_status_model_frame(
                base_model,
                panel,
                int(specification["window_days"]),
                event_col,
                exposure_status,
            )
            support = status_support(frame, event_col)
            adequate = support_is_adequate(support)
            if adequate:
                result = fit_status_model(
                    base_model,
                    frame,
                    event_col,
                    str(specification["denominator"]),
                    str(specification["controls"]),
                )
                contrasts = status_contrasts(result, exposure_status)
                fit_status = "fitted"
            else:
                contrasts = sparse_contrasts(exposure_status)
                fit_status = "not_fitted_sparse_support"
            for key, value in specification.items():
                if key != "controls":
                    contrasts[key] = value
            contrasts["analysis_role"] = "post_hoc_status_sensitivity"
            contrasts["fit_status"] = fit_status
            contrasts["support_adequate"] = adequate
            contrasts["n_match_rows"] = len(frame)
            contrasts["n_players"] = frame[PLAYER_ID].nunique()
            contrasts["n_events"] = int(frame[event_col].sum())
            result_tables.append(contrasts)
            support["national_status"] = exposure_status
            support["specification_id"] = specification["specification_id"]
            support["support_adequate"] = adequate
            support_tables.append(support)
    return add_status_multiplicity(pd.concat(result_tables, ignore_index=True)), pd.concat(support_tables, ignore_index=True)


def main() -> None:  # pragma: no cover
    """Build status features, fit the full robustness family, and write outputs."""
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed" / "public_data_v4"
    records = pd.read_csv(
        processed / "international_performance_record_audit.csv", low_memory=False
    )
    appearances = pd.read_csv(
        processed / "international_appearances.csv", low_memory=False
    )
    panel = pd.read_csv(
        root / "data" / "processed" / "player_match_panel_all_comp.csv",
        low_memory=False,
    )
    ledger = build_status_ledger(records, appearances)
    features = build_status_features(ledger, panel)
    enriched_panel = attach_status_features(panel, features)
    ledger.to_csv(processed / "international_status_ledger.csv", index=False)
    features.to_csv(processed / "national_status_features.csv", index=False)
    status_rate_table(enriched_panel).to_csv(
        processed / "v4_national_status_rates.csv", index=False
    )

    base_model = load_module(
        root / "src" / "18_match_proxy_poisson_splines_perminute.py",
        "match_proxy_model_for_status",
    )
    model_panel = base_model.add_player_and_club_metadata(
        enriched_panel, root / "external_data" / "transfermarkt"
    )
    results, support = run_status_family(base_model, model_panel)
    results.to_csv(processed / "v4_national_status_models.csv", index=False)
    support.to_csv(processed / "v4_national_status_model_support.csv", index=False)
    print(results.loc[results["specification_id"].eq("window_7d_observed")].to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
