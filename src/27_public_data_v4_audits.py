#!/usr/bin/env python
"""Audit v4 public-data coverage, outcome evidence, and observed selection.

The functions in this module operationalise the public-data v4 protocol.  In
particular, a missing official-schedule audit cannot pass the coverage gate,
and an unreviewed public injury report cannot be upgraded to clinical evidence.
The selection-risk set is deliberately labelled as *plausible observed
membership*: public appearances and transfer dates do not reveal selection,
medical clearance, or complete squad availability.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from v4_statistics import percent_with_interval


PLAYER_ID = "tm_player_id"
MATCH_COVERAGE_THRESHOLD = 95.0
MINUTES_COVERAGE_THRESHOLD = 95.0
VALIDATION_COLUMNS = [
    PLAYER_ID,
    "injury_spell_id",
    "start_date",
    "end_date",
    "injury_desc",
    "reported_duration_days",
    "validation_stratum",
    "transfermarkt_record_status",
    "candidate_official_url",
    "official_evidence_grade",
    "review_status",
]
SENIOR_TYPE_IDS = {11, 19}


def _required(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def newest_snapshot_dir(raw_root: Path) -> Path:
    """Return the latest dated v4 snapshot without mutating a prior snapshot."""
    candidates = sorted(
        path for path in Path(raw_root).glob("transfermarkt_datasets_*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"No immutable Transfermarkt snapshot under {raw_root}")
    return candidates[-1]


def sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    """Return a bounded-memory SHA-256 digest for a frozen local input."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_input_manifest(paths: Iterable[Path], baseline_commit: str) -> dict[str, Any]:
    """Record the exact current inputs without modifying any frozen source file."""
    records = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Frozen baseline input is missing: {path}")
        records.append({"path": str(path).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"baseline_commit": baseline_commit, "files": records}


def _schedule_match_key(frame: pd.DataFrame) -> pd.Series:
    """Create an auditable key for independently reviewed national schedules."""
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    team = pd.to_numeric(frame["team_id"], errors="coerce").fillna(-1).astype(int)
    opponent = pd.to_numeric(frame["opponent_team_id"], errors="coerce").fillna(-1).astype(int)
    first = pd.concat([team, opponent], axis=1).min(axis=1).astype(int).astype(str)
    second = pd.concat([team, opponent], axis=1).max(axis=1).astype(int).astype(str)
    competition = frame["competition_id"].fillna("").astype(str)
    return first + "|" + second + "|" + competition + "|" + dates


def coverage_audit(
    appearances: pd.DataFrame,
    acquisition_log: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    official_schedule: pd.DataFrame | None = None,
    record_audit: pd.DataFrame | None = None,
    independent_schedule: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return detailed coverage rows and a non-overridable primary-use gate.

    Minute completeness is measured before exposure filtering when the raw
    record audit is supplied.  The gate itself uses identified senior
    competitive appearances because that is the prespecified primary country
    exposure; completeness across every played national record is retained as
    a separate descriptive metric.
    """
    _required(appearances, [PLAYER_ID, "game_id", "competition_id", "season", "team_id", "minutes_played", "date", "opponent_team_id"], "international appearances")
    _required(acquisition_log, [PLAYER_ID, "status"], "acquisition log")
    _required(duplicate_audit, ["resolution"], "duplicate audit")
    source = appearances.copy()
    source["minutes_played"] = pd.to_numeric(source["minutes_played"], errors="coerce")
    source["schedule_key"] = _schedule_match_key(source)
    if official_schedule is None:
        official_schedule = pd.DataFrame(columns=["team_id", "opponent_team_id", "competition_id", "date", "verified", "official_source_url"])
    _required(official_schedule, ["team_id", "opponent_team_id", "competition_id", "date", "verified", "official_source_url"], "official schedule")
    schedule = official_schedule.copy()
    schedule["schedule_key"] = _schedule_match_key(schedule)
    verified = schedule.loc[schedule["verified"].fillna(False).astype(bool), "schedule_key"].drop_duplicates()
    source["official_schedule_verified"] = source["schedule_key"].isin(verified)
    if independent_schedule is None:
        independent_schedule = pd.DataFrame(
            columns=["team_id", "opponent_team_id", "competition_id", "date", "verified"]
        )
    _required(
        independent_schedule,
        ["team_id", "opponent_team_id", "competition_id", "date", "verified"],
        "independent schedule",
    )
    independent = independent_schedule.copy()
    independent["schedule_key"] = _schedule_match_key(independent)
    independent_verified = independent.loc[
        independent["verified"].fillna(False).astype(bool), "schedule_key"
    ].drop_duplicates()
    source["independent_schedule_verified"] = source["schedule_key"].isin(
        independent_verified
    )
    if "is_senior_competitive" in source:
        primary_source = source.loc[
            source["is_senior_competitive"].fillna(False).astype(bool)
        ].copy()
    else:
        primary_source = source
    grouped = source.groupby(["competition_id", "season", "team_id", PLAYER_ID], dropna=False, as_index=False).agg(
        identified_appearance_rows=("game_id", "size"),
        unique_identified_games=("game_id", "nunique"),
        nonmissing_minutes_rows=("minutes_played", lambda series: int(series.notna().sum())),
        official_schedule_verified_games=("official_schedule_verified", "sum"),
        independent_schedule_verified_games=("independent_schedule_verified", "sum"),
    )
    grouped["percent_minutes_known"] = grouped["nonmissing_minutes_rows"] / grouped["identified_appearance_rows"] * 100.0
    grouped["percent_verified_match_coverage"] = grouped["official_schedule_verified_games"] / grouped["unique_identified_games"] * 100.0
    grouped["percent_independent_match_coverage"] = grouped["independent_schedule_verified_games"] / grouped["unique_identified_games"] * 100.0
    minute_intervals = grouped.apply(
        lambda row: percent_with_interval(
            int(row["nonmissing_minutes_rows"]), int(row["identified_appearance_rows"])
        )[1:],
        axis=1,
        result_type="expand",
    )
    grouped[["percent_minutes_known_ci_low", "percent_minutes_known_ci_high"]] = minute_intervals
    official_intervals = grouped.apply(
        lambda row: percent_with_interval(
            int(row["official_schedule_verified_games"]), int(row["unique_identified_games"])
        )[1:],
        axis=1,
        result_type="expand",
    )
    grouped[["percent_verified_match_coverage_ci_low", "percent_verified_match_coverage_ci_high"]] = official_intervals
    independent_intervals = grouped.apply(
        lambda row: percent_with_interval(
            int(row["independent_schedule_verified_games"]), int(row["unique_identified_games"])
        )[1:],
        axis=1,
        result_type="expand",
    )
    grouped[["percent_independent_match_coverage_ci_low", "percent_independent_match_coverage_ci_high"]] = independent_intervals
    grouped["percent_interval_method"] = "wilson_95"
    grouped["official_schedule_audit_available"] = not schedule.empty
    grouped["independent_schedule_audit_available"] = not independent.empty
    minutes_coverage = float(source["minutes_played"].notna().mean() * 100.0) if len(source) else 0.0
    primary_games = primary_source.drop_duplicates("game_id")
    match_coverage = (
        float(primary_games["official_schedule_verified"].mean() * 100.0)
        if len(primary_games) and not schedule.empty
        else np.nan
    )
    independent_match_coverage = (
        float(primary_games["independent_schedule_verified"].mean() * 100.0)
        if len(primary_games) and not independent.empty
        else np.nan
    )
    overall_played_minutes_coverage = minutes_coverage
    if record_audit is not None:
        _required(
            record_audit,
            ["participation_state", "minutes_played", "competition_type_id", "competition_id"],
            "national record audit",
        )
        records = record_audit.copy()
        played = records["participation_state"].eq("played")
        senior = pd.to_numeric(records["competition_type_id"], errors="coerce").isin(
            SENIOR_TYPE_IDS
        )
        competitive = ~records["competition_id"].fillna("").astype(str).str.casefold().eq("fs")
        primary_played = played & senior & competitive
        overall_played_minutes_coverage = (
            float(records.loc[played, "minutes_played"].notna().mean() * 100.0)
            if played.any()
            else 0.0
        )
        minutes_coverage = (
            float(records.loc[primary_played, "minutes_played"].notna().mean() * 100.0)
            if primary_played.any()
            else 0.0
        )
    unresolved_ids = int(acquisition_log["status"].eq("error").sum())
    unresolved_duplicates = int(duplicate_audit["resolution"].eq("unresolved_minutes_conflict_excluded").sum())
    gate_checks = {
        "official_schedule_coverage_at_least_95": bool(not pd.isna(match_coverage) and match_coverage >= MATCH_COVERAGE_THRESHOLD),
        "identified_minutes_at_least_95": bool(minutes_coverage >= MINUTES_COVERAGE_THRESHOLD),
        "zero_unresolved_cohort_ids": unresolved_ids == 0,
        "zero_unexplained_duplicates": unresolved_duplicates == 0,
    }
    gate_rows = [
            {
                "metric": "verified_match_coverage_percent",
                "value": match_coverage,
                "threshold": MATCH_COVERAGE_THRESHOLD,
                "passes": gate_checks["official_schedule_coverage_at_least_95"],
                "binding_for_primary_use": True,
                "gate_role": "prespecified_official_schedule_gate",
            },
            {
                "metric": "independent_match_coverage_percent",
                "value": independent_match_coverage,
                "threshold": np.nan,
                "passes": pd.NA,
                "binding_for_primary_use": False,
                "gate_role": "informational_secondary_validation",
            },
            {
                "metric": "primary_senior_competitive_nonmissing_minutes_percent",
                "value": minutes_coverage,
                "threshold": MINUTES_COVERAGE_THRESHOLD,
                "passes": gate_checks["identified_minutes_at_least_95"],
                "binding_for_primary_use": True,
                "gate_role": "prespecified_minute_completeness_gate",
            },
            {
                "metric": "all_played_national_records_nonmissing_minutes_percent",
                "value": overall_played_minutes_coverage,
                "threshold": np.nan,
                "passes": pd.NA,
                "binding_for_primary_use": False,
                "gate_role": "informational_all_scope_completeness",
            },
            {
                "metric": "unresolved_cohort_player_ids",
                "value": unresolved_ids,
                "threshold": 0,
                "passes": gate_checks["zero_unresolved_cohort_ids"],
                "binding_for_primary_use": True,
                "gate_role": "prespecified_identity_gate",
            },
            {
                "metric": "unexplained_duplicate_player_games",
                "value": unresolved_duplicates,
                "threshold": 0,
                "passes": gate_checks["zero_unexplained_duplicates"],
                "binding_for_primary_use": True,
                "gate_role": "prespecified_duplicate_resolution_gate",
            },
        ]
    gate = pd.DataFrame(gate_rows)
    gate["numerator"] = pd.NA
    gate["denominator"] = pd.NA
    gate["ci_low"] = np.nan
    gate["ci_high"] = np.nan
    gate["interval_method"] = "not_applicable"
    interval_inputs = {
        "verified_match_coverage_percent": (
            int(primary_games["official_schedule_verified"].sum()),
            len(primary_games),
        ),
        "independent_match_coverage_percent": (
            int(primary_games["independent_schedule_verified"].sum()),
            len(primary_games),
        ),
    }
    if record_audit is not None:
        interval_inputs[
            "primary_senior_competitive_nonmissing_minutes_percent"
        ] = (
            int(records.loc[primary_played, "minutes_played"].notna().sum()),
            int(primary_played.sum()),
        )
        interval_inputs["all_played_national_records_nonmissing_minutes_percent"] = (
            int(records.loc[played, "minutes_played"].notna().sum()),
            int(played.sum()),
        )
    for metric, (successes, total) in interval_inputs.items():
        _, low, high = percent_with_interval(successes, total)
        selected = gate["metric"].eq(metric)
        gate.loc[
            selected,
            ["numerator", "denominator", "ci_low", "ci_high", "interval_method"],
        ] = [
            successes,
            total,
            low,
            high,
            "wilson_95",
        ]
    primary_allowed = bool(all(gate_checks.values()))
    gate["primary_v4_exposure_allowed"] = primary_allowed
    gate["decision"] = "primary_allowed" if primary_allowed else "sensitivity_only"
    return grouped, gate


def official_schedule_template(appearances: pd.DataFrame) -> pd.DataFrame:
    """Create a manual-validation queue without pretending an unknown schedule is complete."""
    _required(appearances, ["team_id", "opponent_team_id", "competition_id", "date"], "international appearances")
    template = appearances[["team_id", "opponent_team_id", "competition_id", "date"]].drop_duplicates().copy()
    template["verified"] = False
    template["official_source_url"] = ""
    template["official_match_id"] = ""
    template["review_note"] = "pending official schedule or match-report validation"
    return template.sort_values(["team_id", "date", "opponent_team_id"]).reset_index(drop=True)


def parse_reported_duration(value: object) -> float:
    """Extract a reported spell length from a Transfermarkt duration payload."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    if isinstance(value, Mapping):
        payload = value
    elif isinstance(value, str) and value.strip():
        try:
            payload = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return np.nan
    else:
        return np.nan
    try:
        return float(payload.get("days"))
    except (AttributeError, TypeError, ValueError):
        return np.nan


def outcome_validation_queue(injuries: pd.DataFrame, per_stratum: int = 12) -> pd.DataFrame:
    """Create a deterministic public-source validation queue by outcome type."""
    _required(injuries, [PLAYER_ID, "injury_spell_id", "start_date", "end_date", "injury_desc", "durationDetails"], "cleaned injuries")
    out = injuries.copy()
    out["reported_duration_days"] = out["durationDetails"].map(parse_reported_duration)
    description = out["injury_desc"].fillna("").astype(str).str.lower()
    out["validation_stratum"] = np.select(
        [
            out["reported_duration_days"].ge(28),
            description.str.contains("muscle|hamstring|tendon|achilles", regex=True),
            description.str.contains("unknown|unspecified", regex=True),
        ],
        ["reported_absence_ge28d", "muscle_tendon_description", "ambiguous_description"],
        default="unmatched_other_description",
    )
    out = out.sort_values(["validation_stratum", PLAYER_ID, "injury_spell_id"])
    sampled = out.groupby("validation_stratum", group_keys=False).head(per_stratum).copy()
    sampled["transfermarkt_record_status"] = "primary_public_record"
    sampled["candidate_official_url"] = ""
    sampled["official_evidence_grade"] = "unreviewed"
    sampled["review_status"] = "pending manual official-source check"
    return sampled[VALIDATION_COLUMNS].reset_index(drop=True)


def epl_membership_intervals(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    transfers: pd.DataFrame,
    cohort_ids: Iterable[int],
) -> pd.DataFrame:
    """Infer conservative observed EPL-club membership intervals from public records."""
    _required(appearances, ["player_id", "game_id", "player_club_id", "date"], "appearances")
    _required(games, ["game_id", "competition_id", "season", "home_club_id", "away_club_id"], "games")
    cohort = {int(value) for value in cohort_ids}
    epl_games = games.loc[games["competition_id"].eq("GB1")]
    epl_clubs = pd.concat(
        [
            epl_games[["season", "home_club_id"]].rename(columns={"home_club_id": "player_club_id"}),
            epl_games[["season", "away_club_id"]].rename(columns={"away_club_id": "player_club_id"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    app = appearances.loc[appearances["player_id"].isin(cohort)].merge(games[["game_id", "season"]], on="game_id", how="left")
    app["date"] = pd.to_datetime(app["date"], errors="coerce")
    app = app.merge(epl_clubs, on=["season", "player_club_id"], how="inner")
    intervals = app.groupby(["player_id", "player_club_id", "season"], as_index=False).agg(
        observed_first_appearance=("date", "min"),
        observed_last_appearance=("date", "max"),
    ).rename(columns={"player_id": PLAYER_ID})
    intervals["membership_start"] = intervals["observed_first_appearance"]
    intervals["membership_end"] = intervals["observed_last_appearance"]
    if {"player_id", "to_club_id", "from_club_id", "transfer_date"}.issubset(transfers.columns):
        transfer = transfers.copy()
        transfer["transfer_date"] = pd.to_datetime(transfer["transfer_date"], errors="coerce")
        intervals["_interval_id"] = np.arange(len(intervals), dtype=int)
        arrivals = transfer[["player_id", "to_club_id", "transfer_date"]].rename(
            columns={"player_id": PLAYER_ID, "to_club_id": "player_club_id"}
        )
        arrival_candidates = intervals[
            ["_interval_id", PLAYER_ID, "player_club_id", "observed_first_appearance"]
        ].merge(arrivals, on=[PLAYER_ID, "player_club_id"], how="left")
        arrival_candidates = arrival_candidates[
            arrival_candidates["transfer_date"].le(
                arrival_candidates["observed_first_appearance"]
            )
        ]
        arrival_dates = arrival_candidates.groupby("_interval_id")["transfer_date"].max()

        departures = transfer[["player_id", "from_club_id", "transfer_date"]].rename(
            columns={"player_id": PLAYER_ID, "from_club_id": "player_club_id"}
        )
        departure_candidates = intervals[
            ["_interval_id", PLAYER_ID, "player_club_id", "observed_last_appearance"]
        ].merge(departures, on=[PLAYER_ID, "player_club_id"], how="left")
        departure_candidates = departure_candidates[
            departure_candidates["transfer_date"].ge(
                departure_candidates["observed_last_appearance"]
            )
        ]
        departure_dates = departure_candidates.groupby("_interval_id")["transfer_date"].min()

        intervals["arrival_date"] = intervals["_interval_id"].map(arrival_dates)
        intervals["departure_date"] = intervals["_interval_id"].map(departure_dates)
        intervals["membership_start"] = intervals[["membership_start", "arrival_date"]].min(axis=1)
        intervals["membership_end"] = intervals[["membership_end", "departure_date"]].max(axis=1)
        intervals = intervals.drop(columns="_interval_id")
    else:
        intervals["arrival_date"] = pd.NaT
        intervals["departure_date"] = pd.NaT
    intervals["membership_evidence"] = "observed_appearance_span_with_recorded_transfer_extension"
    return intervals


def resolve_selection_opportunities(
    risk_set: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve overlapping public membership intervals conservatively.

    A player can have only one EPL club-match opportunity on a date. Exact
    duplicate opportunities are removed. If overlapping public intervals name
    more than one club and exactly one row records an appearance, that observed
    club is retained. Dates with no unique observed club are excluded rather
    than assigned to a club using future information.
    """
    key = [PLAYER_ID, "date"]
    exact_key = [PLAYER_ID, "date", "player_club_id", "game_id"]
    _required(
        risk_set,
        [*exact_key, "played_any_minutes"],
        "selection opportunities",
    )
    work = risk_set.copy()
    input_rows = len(work)
    exact_duplicates = int(work.duplicated(exact_key, keep="first").sum())
    work = work.drop_duplicates(exact_key, keep="first").copy()

    group_size = work.groupby(key, observed=False)["game_id"].transform("size")
    selected_count = work.groupby(key, observed=False)["played_any_minutes"].transform("sum")
    overlapping = group_size.gt(1)
    one_observed_club = overlapping & selected_count.eq(1)
    unresolved_none = overlapping & selected_count.eq(0)
    unresolved_multiple = overlapping & selected_count.gt(1)
    keep = ~overlapping | (one_observed_club & work["played_any_minutes"].eq(1))
    resolved = work.loc[keep].copy()
    resolved["opportunity_resolution"] = np.where(
        group_size.loc[keep].eq(1),
        "unique_public_membership",
        "observed_club_selected_from_overlap",
    )
    if resolved.duplicated(key).any():  # pragma: no cover - prevented by the groupwise keep rule
        raise RuntimeError("selection opportunity resolution left duplicate player-dates")

    audit = pd.DataFrame(
        [
            {"metric": "input_rows", "value": int(input_rows), "passes_gate": True},
            {
                "metric": "exact_duplicate_rows_removed",
                "value": exact_duplicates,
                "passes_gate": True,
            },
            {
                "metric": "overlapping_player_dates",
                "value": int(work.loc[overlapping, key].drop_duplicates().shape[0]),
                "passes_gate": True,
            },
            {
                "metric": "overlaps_resolved_by_observed_club",
                "value": int(work.loc[one_observed_club, key].drop_duplicates().shape[0]),
                "passes_gate": True,
            },
            {
                "metric": "unresolved_zero_appearance_dates_excluded",
                "value": int(work.loc[unresolved_none, key].drop_duplicates().shape[0]),
                "passes_gate": True,
            },
            {
                "metric": "unresolved_multiple_appearance_dates_excluded",
                "value": int(work.loc[unresolved_multiple, key].drop_duplicates().shape[0]),
                "passes_gate": bool(
                    work.loc[unresolved_multiple, key].drop_duplicates().empty
                ),
            },
            {
                "metric": "resolved_rows",
                "value": int(len(resolved)),
                "passes_gate": True,
            },
            {
                "metric": "unique_player_date_gate",
                "value": int(not resolved.duplicated(key).any()),
                "passes_gate": bool(not resolved.duplicated(key).any()),
            },
        ]
    )
    return resolved.sort_values([PLAYER_ID, "date", "game_id"]).reset_index(drop=True), audit


def build_selection_risk_set(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    transfers: pd.DataFrame,
    cohort_ids: Iterable[int],
    player_day: pd.DataFrame,
    return_resolution_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Create conservative player-fixture opportunities and observed playing status."""
    _required(player_day, [PLAYER_ID, "date", "available_for_injury_risk", "all_minutes_last_7d", "prior_n_spells"], "player-day panel")
    intervals = epl_membership_intervals(appearances, games, transfers, cohort_ids)
    fixtures = games.loc[games["competition_id"].eq("GB1"), ["game_id", "season", "date", "home_club_id", "away_club_id"]].copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    club_fixtures = pd.concat(
        [
            fixtures.rename(columns={"home_club_id": "player_club_id"})[["game_id", "season", "date", "player_club_id"]],
            fixtures.rename(columns={"away_club_id": "player_club_id"})[["game_id", "season", "date", "player_club_id"]],
        ],
        ignore_index=True,
    )
    risk = intervals.merge(club_fixtures, on=["season", "player_club_id"], how="inner")
    risk = risk.loc[risk["date"].between(risk["membership_start"], risk["membership_end"], inclusive="both")].copy()
    observed = appearances[
        ["player_id", "game_id", "player_club_id", "minutes_played"]
    ].copy()
    observed["minutes_played"] = pd.to_numeric(observed["minutes_played"], errors="coerce")
    observed = (
        observed.groupby(
            ["player_id", "game_id", "player_club_id"], as_index=False
        )["minutes_played"]
        .sum()
        .rename(columns={"player_id": PLAYER_ID})
    )
    risk = risk.merge(
        observed,
        on=[PLAYER_ID, "game_id", "player_club_id"],
        how="left",
    )
    risk["played_any_minutes"] = risk["minutes_played"].fillna(0).gt(0).astype(int)
    daily = player_day.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    risk = risk.merge(
        daily[[PLAYER_ID, "date", "available_for_injury_risk", "all_minutes_last_7d", "prior_n_spells"]],
        on=[PLAYER_ID, "date"],
        how="left",
    )
    risk["plausibly_available"] = risk["available_for_injury_risk"].fillna(False).astype(bool)
    resolved, resolution_audit = resolve_selection_opportunities(risk)
    if return_resolution_audit:
        return resolved, resolution_audit
    return resolved


def _weighted_standardised_difference(values: pd.Series, treated: pd.Series, weights: pd.Series) -> float:
    """Return an absolute weighted SMD or missing when a group has no support."""
    value = pd.to_numeric(values, errors="coerce")
    group = treated.astype(bool)
    if not group.any() or group.all():
        return np.nan
    means = []
    variances = []
    for mask in (group, ~group):
        weight = pd.to_numeric(weights.loc[mask], errors="coerce").fillna(0)
        observed = value.loc[mask].fillna(0)
        if weight.sum() <= 0:
            return np.nan
        mean = np.average(observed, weights=weight)
        means.append(mean)
        variances.append(np.average((observed - mean) ** 2, weights=weight))
    denominator = sqrt_safe((variances[0] + variances[1]) / 2)
    return abs(means[0] - means[1]) / denominator if denominator > 0 else 0.0


def sqrt_safe(value: float) -> float:
    """Return zero for a non-positive variance used in balance diagnostics."""
    return float(np.sqrt(value)) if value > 0 else 0.0


def fit_selection_weights(risk_set: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a prior-information selection model and gate stabilised IP weights."""
    _required(risk_set, ["played_any_minutes", "plausibly_available", "all_minutes_last_7d", "prior_n_spells", "season", "player_club_id"], "selection risk set")
    frame = risk_set.loc[risk_set["plausibly_available"].astype(bool)].copy()
    frame["all_minutes_last_7d"] = pd.to_numeric(frame["all_minutes_last_7d"], errors="coerce").fillna(0)
    frame["prior_n_spells"] = pd.to_numeric(frame["prior_n_spells"], errors="coerce").fillna(0)
    if frame.empty or frame["played_any_minutes"].nunique() < 2:
        out = risk_set.copy()
        out["selection_probability"] = np.nan
        out["stabilized_ipw"] = np.nan
        out["ipw_usable"] = False
        return out, pd.DataFrame([{"metric": "model_status", "value": "not_estimable", "passes": False}])
    model = smf.glm(
        "played_any_minutes ~ all_minutes_last_7d + prior_n_spells + C(season) + C(player_club_id)",
        data=frame,
        family=sm.families.Binomial(),
    ).fit()
    frame["selection_probability"] = model.predict(frame).clip(lower=0.001, upper=0.999)
    marginal = float(frame["played_any_minutes"].mean())
    frame["stabilized_ipw"] = np.where(
        frame["played_any_minutes"].eq(1),
        marginal / frame["selection_probability"],
        (1 - marginal) / (1 - frame["selection_probability"]),
    )
    balance_minutes = _weighted_standardised_difference(frame["all_minutes_last_7d"], frame["played_any_minutes"], frame["stabilized_ipw"])
    balance_history = _weighted_standardised_difference(frame["prior_n_spells"], frame["played_any_minutes"], frame["stabilized_ipw"])
    overlap = float(frame["selection_probability"].between(0.05, 0.95).mean())
    weight_p99 = float(frame["stabilized_ipw"].quantile(0.99))
    usable = bool(overlap >= 0.95 and weight_p99 <= 10 and max(balance_minutes, balance_history) <= 0.1)
    out = risk_set.copy().merge(frame[[PLAYER_ID, "game_id", "selection_probability", "stabilized_ipw"]], on=[PLAYER_ID, "game_id"], how="left")
    out["ipw_usable"] = usable
    diagnostics = pd.DataFrame(
        [
            {"metric": "model_status", "value": "estimable", "passes": True},
            {"metric": "overlap_share_probability_0_05_to_0_95", "value": overlap, "passes": overlap >= 0.95},
            {"metric": "weight_99th_percentile", "value": weight_p99, "passes": weight_p99 <= 10},
            {"metric": "weighted_smd_prior_7d_minutes", "value": balance_minutes, "passes": balance_minutes <= 0.1},
            {"metric": "weighted_smd_prior_injury_count", "value": balance_history, "passes": balance_history <= 0.1},
        ]
    )
    diagnostics["ipw_usable"] = usable
    return out, diagnostics


def main() -> None:  # pragma: no cover
    """Write the protocol's baseline, coverage, outcome, and selection audits."""
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed" / "public_data_v4"
    raw = root / "data" / "raw" / "public_data_v4"
    snapshot = newest_snapshot_dir(raw)
    processed.mkdir(parents=True, exist_ok=True)
    baseline_paths = [
        root / "external_data" / "transfermarkt" / name
        for name in ("appearances.csv", "games.csv", "players.csv", "transfers.csv", "competitions.csv")
    ] + [
        root / "data" / "processed" / name
        for name in ("player_day_panel.csv", "player_day_panel_all_comp.csv", "player_match_panel_all_comp.csv", "tm_injuries_clean.csv")
    ]
    manifest = baseline_input_manifest(baseline_paths, "61648fd5f1aad64a52c0aac0f42f4fa2e8f31fbd")
    (raw / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    national = pd.read_csv(processed / "international_appearances.csv", low_memory=False)
    record_audit = pd.read_csv(
        processed / "international_performance_record_audit.csv", low_memory=False
    )
    log = pd.read_csv(processed / "national_acquisition_log.csv", low_memory=False)
    duplicates = pd.read_csv(processed / "national_duplicate_audit.csv", low_memory=False)
    schedule_path = processed / "official_schedule_validation.csv"
    schedule = pd.read_csv(schedule_path, low_memory=False) if schedule_path.exists() else None
    independent_path = processed / "independent_schedule_validation.csv"
    independent = (
        pd.read_csv(independent_path, low_memory=False)
        if independent_path.exists()
        else None
    )
    detailed, gate = coverage_audit(
        national,
        log,
        duplicates,
        schedule,
        record_audit=record_audit,
        independent_schedule=independent,
    )
    detailed.to_csv(processed / "international_exposure_coverage_audit.csv", index=False)
    gate.to_csv(processed / "exposure_coverage_audit.csv", index=False)
    if schedule is None:
        official_schedule_template(national).to_csv(schedule_path, index=False)
    injuries = pd.read_csv(root / "data" / "processed" / "tm_injuries_clean.csv", low_memory=False)
    outcome_validation_queue(injuries).to_csv(processed / "injury_source_validation.csv", index=False)
    apps = pd.read_csv(snapshot / "appearances.csv.gz", low_memory=False)
    games = pd.read_csv(snapshot / "games.csv.gz", low_memory=False)
    transfers = pd.read_csv(snapshot / "transfers.csv.gz", low_memory=False)
    cohort = pd.read_csv(processed / "epl_cohort_manifest.csv", low_memory=False)
    panel = pd.read_csv(root / "data" / "processed" / "player_day_panel_all_comp.csv", low_memory=False)
    risk_set, resolution_audit = build_selection_risk_set(
        apps,
        games,
        transfers,
        cohort[PLAYER_ID],
        panel,
        return_resolution_audit=True,
    )
    weighted, diagnostics = fit_selection_weights(risk_set)
    weighted.to_csv(processed / "selection_risk_set.csv", index=False)
    resolution_audit.to_csv(
        processed / "selection_membership_resolution_audit.csv", index=False
    )
    diagnostics.to_csv(processed / "selection_weight_diagnostics.csv", index=False)
    print(gate.to_string(index=False))
    print(f"Selection-risk opportunities: {len(weighted)}")


if __name__ == "__main__":  # pragma: no cover
    main()
