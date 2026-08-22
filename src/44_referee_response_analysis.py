"""Quantities the first round of review asked for, computed rather than argued.

Five things were missing from the paper and each is a number, not a sentence,
so each is produced here instead of being written into the manuscript by hand.

1.  **Calibration.** The first-order identity predicts an attenuation equal to
    the gradient. It over-predicts by about a factor of two, and the paper's
    practical advice is built on the gradient, so a practitioner following it
    would have overstated their own bias. The ratio is measured per stratum.
2.  **Translation.** A gradient of 0.303 means nothing to a practitioner. The
    same quantity expressed as "the association is understated by 14%" means
    something immediately, and it is the same number through one exponential.
3.  **A threshold with a meaning.** The negligible threshold of 0.05 was a
    declaration. Here it is carried through the calibration so a reader can see
    what it costs in the units of the answer they care about.
4.  **Ascertainment.** Events per appearance differ threefold between starters
    and substitutes. Since the remedy this paper recommends is restriction to
    starters, the reader is owed that comparison rather than left to derive it
    from two columns of a table.
5.  **Clustering.** Appearances repeat within player, and the published
    intervals are clustered there. They also repeat within club-season and
    within fixture, and squad rotation is a club decision, so the gradient is
    refitted against each of those groupings.

Nothing here refits a published estimate under a different specification and
then reports the new one. Every function takes the deposited quantity as its
input and reports what it implies, so the manuscript keeps one set of numbers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "processed" / "results"

PLAYER_ID_COL = "tm_player_id"
DATE_COL = "date"
MINUTES_COL = "all_minutes_played"
HISTORY_MODEL_COL = "history_log_iqr"
SAME_DAY_COL = "injury_event_matchproxy_same_day"
ROLE_COL = "lineup_role_model"
CLUB_COL = "player_club_id"

STARTER_ROLE = "starting_lineup"
SUBSTITUTE_ROLE = "substitute_list"
UNKNOWN_ROLE = "lineup_unavailable_or_other"

#: The reporting threshold the decision rule reads. It is a convention, and
#: this module's job is to say what the convention costs rather than to defend
#: the number.
NEGLIGIBLE_GAMMA = 0.05

#: Gradients worth translating for a reader who wants to interpolate.
THRESHOLD_GRID = (0.01, 0.025, NEGLIGIBLE_GAMMA, 0.10, 0.20, 0.30, 0.50)

#: Recorded-minute floors used to sweep the gradient.
#:
#: The over-prediction ratio is not a constant. Measured over four squad-role
#: strata it runs from 0.89 where the gradient is 0.011 to 2.01 where it is
#: 0.303, and the reporting threshold of 0.05 sits in the gap between those two
#: observations. Applying the pooled ratio there is an extrapolation into the
#: one region that matters, and it happens to run in the unsafe direction: it
#: makes the threshold look about twice as tolerant as it is.
#:
#: Raising a floor on recorded minutes shrinks the minute variation the offset
#: has to act on, so the gradient falls continuously. That sweeps the whole
#: range with the outcome, the exposure and the specification held fixed, and
#: the only thing changing is the population's minute spread -- which is the
#: quantity the ratio is supposed to depend on.
MINUTE_FLOORS = (0.0, 20.0, 40.0, 55.0, 65.0, 75.0, 82.0, 86.0)

#: A stratum with too few events cannot support a Poisson pair, and a ratio
#: computed from an unstable attenuation is worse than no ratio.
MIN_EVENTS_FOR_CURVE = 40

#: The strata the decomposition reports, paired with the lineup role each one
#: describes. Keeping the pairing in one place is what lets the calibration
#: table be built by lookup instead of by index arithmetic.
STRATUM_ROLES = (
    ("gamma_log_minutes_on_exposure", "all"),
    ("gamma_within_starting_lineup", STARTER_ROLE),
    ("gamma_within_substitute_list", SUBSTITUTE_ROLE),
    ("gamma_within_lineup_unavailable_or_other", UNKNOWN_ROLE),
)


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


def percent_understatement(log_attenuation: float) -> float:
    """Turn an attenuation on the log scale into a percentage of the answer.

    An attenuation of ``a`` multiplies the reported ratio by ``exp(-a)``, so the
    association is understated by ``1 - exp(-a)`` of itself. For the reference
    cohort's 0.151 that is 14.0%, which is the sentence a practitioner can act
    on; 0.151 is not.
    """
    return float((1.0 - np.exp(-float(log_attenuation))) * 100.0)


def identity_calibration(
    decomposition: pd.DataFrame, roles: pd.DataFrame
) -> pd.DataFrame:
    """Compare the attenuation the identity predicts with the one observed.

    The identity is a first-order expansion that weights every appearance
    equally, while the Poisson score weights by fitted means. Where the gradient
    is large the two diverge, and the divergence is not noise: it is systematic,
    it runs one way, and it is the difference between a practitioner correctly
    and incorrectly estimating what their own denominator costs them.
    """
    _require(decomposition, ("value",), "decomposition table")
    _require(roles, ("lineup_role", "log_attenuation_fixed90_minus_recorded"), "role table")

    observed = (
        roles.drop_duplicates("lineup_role")
        .set_index("lineup_role")["log_attenuation_fixed90_minus_recorded"]
        .astype(float)
    )

    rows = []
    for quantity, role in STRATUM_ROLES:
        if quantity not in decomposition.index or role not in observed.index:
            continue
        gamma = float(decomposition.loc[quantity, "value"])
        actual = float(observed.loc[role])
        rows.append(
            {
                "stratum": role,
                "quantity": quantity,
                "gamma_predicted_attenuation": gamma,
                "observed_attenuation": actual,
                "over_prediction_ratio": gamma / actual if actual else np.nan,
                "gamma_implied_percent": percent_understatement(gamma),
                "observed_percent": percent_understatement(actual),
            }
        )
    if not rows:
        raise ValueError("no stratum could be calibrated against the deposited tables")
    return pd.DataFrame(rows)


def calibration_factor(calibration: pd.DataFrame, stratum: str = "all") -> float:
    """The pooled ratio by which the identity over-predicts."""
    _require(calibration, ("stratum", "over_prediction_ratio"), "calibration table")
    match = calibration[calibration["stratum"].eq(stratum)]
    if match.empty:
        raise ValueError(f"no calibration row for stratum {stratum!r}")
    return float(match["over_prediction_ratio"].iloc[0])


def threshold_translation(
    curve: pd.DataFrame, grid: Sequence[float] = THRESHOLD_GRID
) -> pd.DataFrame:
    """Say what each candidate gradient threshold costs in the reported answer.

    The calibrated column divides the gradient by the over-prediction ratio
    measured *at that gradient*, not by the pooled ratio. Those differ by a
    factor of two near the reporting threshold, in the direction that makes the
    threshold look more tolerant than it is, and the threshold is the one place
    on this grid a practitioner will actually read.

    Every row records whether its ratio was measured or extrapolated, so a
    reader can see where the curve stops being evidence.
    """
    rows = []
    for gamma in grid:
        ratio, measured = ratio_at_gamma(curve, gamma)
        calibrated = float(gamma) / ratio
        rows.append(
            {
                "gamma": float(gamma),
                "over_prediction_ratio": ratio,
                "ratio_is_measured": measured,
                "naive_percent_understatement": percent_understatement(gamma),
                "calibrated_attenuation": calibrated,
                "calibrated_percent_understatement": percent_understatement(calibrated),
                "is_reporting_threshold": bool(np.isclose(gamma, NEGLIGIBLE_GAMMA)),
            }
        )
    return pd.DataFrame(rows)


def denominator_pair(
    subset: pd.DataFrame,
    exposure_col: str,
    calendar_terms: Sequence[str],
    history_col: str,
    minute_floor: float = 1.0,
) -> dict[str, float]:
    """Fit the gradient and the fixed-90/recorded-minute attenuation on one set of rows.

    Both Poisson models take the same rows, the same family, the same link and
    the same right-hand side; only the offset differs, which is what makes their
    difference an attenuation rather than a comparison of two models.
    """
    required = [MINUTES_COL, exposure_col, history_col, SAME_DAY_COL, PLAYER_ID_COL,
                *calendar_terms]
    _require(subset, required, "denominator pair frame")

    work = subset.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    work["log_recorded_minutes"] = np.log(work[MINUTES_COL].clip(lower=float(minute_floor)))
    work["exposure_per_90"] = pd.to_numeric(work[exposure_col], errors="coerce") / 90.0

    calendar = " + ".join(calendar_terms)
    gradient = smf.ols(
        f"log_recorded_minutes ~ exposure_per_90 + {history_col} + {calendar}", data=work
    ).fit(cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]})
    gamma = float(gradient.params["exposure_per_90"])
    low, high = gradient.conf_int().loc["exposure_per_90"]

    formula = f"{SAME_DAY_COL} ~ exposure_per_90 + {history_col} + {calendar}"
    minutes = work[MINUTES_COL].clip(lower=float(minute_floor))
    coefficients = {}
    for label, offset in (
        ("fixed_90", pd.Series(np.log(90.0), index=work.index)),
        ("observed_minutes", np.log(minutes)),
    ):
        fit = smf.glm(
            formula, data=work, family=sm.families.Poisson(), offset=offset
        ).fit(cov_type="cluster", cov_kwds={"groups": work[PLAYER_ID_COL]})
        coefficients[label] = float(fit.params["exposure_per_90"])

    attenuation = coefficients["fixed_90"] - coefficients["observed_minutes"]
    return {
        "n_rows": int(len(work)),
        "n_players": int(work[PLAYER_ID_COL].nunique()),
        "n_events": int(work[SAME_DAY_COL].sum()),
        "mean_recorded_minutes": float(work[MINUTES_COL].mean()),
        "iqr_recorded_minutes": float(
            work[MINUTES_COL].quantile(0.75) - work[MINUTES_COL].quantile(0.25)
        ),
        "gamma": gamma,
        "gamma_ci_low": float(low),
        "gamma_ci_high": float(high),
        "observed_attenuation": attenuation,
        "over_prediction_ratio": gamma / attenuation if attenuation else np.nan,
    }


def calibration_curve(
    frame: pd.DataFrame,
    exposure_col: str,
    calendar_terms: Sequence[str],
    history_col: str,
    floors: Sequence[float] = MINUTE_FLOORS,
    min_events: int = MIN_EVENTS_FOR_CURVE,
) -> pd.DataFrame:
    """Measure the over-prediction ratio across the whole range of the gradient.

    Each row restricts the cohort to appearances at or above a recorded-minute
    floor. Raising the floor removes the short appearances, which is where most
    of the minute variation lives, so the gradient falls; nothing else about the
    model changes. The result is the ratio as a function of the gradient, over a
    range that includes the reporting threshold instead of straddling it.
    """
    _require(frame, (MINUTES_COL,), "calibration curve frame")

    rows = []
    for floor in floors:
        minutes = pd.to_numeric(frame[MINUTES_COL], errors="coerce")
        subset = frame[minutes.ge(float(floor))]
        events = int(
            pd.to_numeric(subset.get(SAME_DAY_COL), errors="coerce").fillna(0).sum()
        )
        if events < int(min_events):
            continue
        entry = denominator_pair(
            subset, exposure_col, calendar_terms, history_col
        )
        # A ratio of two quantities that are both indistinguishable from zero
        # is not a calibration, it is noise with a division sign. At the
        # highest floors every appearance is a full 90 minutes, the gradient
        # has nothing left to vary over, and its interval straddles zero.
        if (
            not np.isfinite(entry["over_prediction_ratio"])
            or not np.isfinite(entry["gamma_ci_low"])
            or entry["gamma_ci_low"] <= 0.0
        ):
            continue
        entry["minute_floor"] = float(floor)
        rows.append(entry)

    if len(rows) < 2:
        raise ValueError(
            "the calibration curve needs at least two estimable strata; "
            "lower the floors or the event minimum"
        )
    return pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)


#: Resamples for the sweep's intervals. The house convention everywhere else
#: in this pipeline is 1000 player resamples, and the sweep keeps it.
CURVE_BOOTSTRAP_DRAWS = 1000


def _sweep_design(
    frame: pd.DataFrame,
    exposure_col: str,
    calendar_terms: Sequence[str],
    history_col: str,
    minute_floor: float = 1.0,
) -> dict[str, object]:
    """Precompute the arrays one bootstrap replicate needs.

    The bootstrap refits three models per floor per replicate, and building a
    patsy design from a formula each time would spend more time parsing strings
    than fitting. Everything a fit needs is therefore assembled once as plain
    arrays, and replicates index into them.
    """
    required = [MINUTES_COL, exposure_col, history_col, SAME_DAY_COL, PLAYER_ID_COL,
                *calendar_terms]
    _require(frame, required, "sweep design frame")

    work = frame.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)]

    minutes = work[MINUTES_COL].to_numpy(dtype=float)
    exog = np.column_stack(
        [
            np.ones(len(work)),
            pd.to_numeric(work[exposure_col], errors="coerce").to_numpy(dtype=float) / 90.0,
            pd.to_numeric(work[history_col], errors="coerce").to_numpy(dtype=float),
        ]
        + [
            pd.to_numeric(work[term], errors="coerce").to_numpy(dtype=float)
            for term in calendar_terms
        ]
    )
    codes, _ = pd.factorize(work[PLAYER_ID_COL])
    order = np.argsort(codes, kind="stable")
    boundaries = np.searchsorted(codes[order], np.arange(codes.max() + 2))
    rows_by_player = [
        order[boundaries[k]:boundaries[k + 1]] for k in range(codes.max() + 1)
    ]
    return {
        "minutes": minutes,
        "log_minutes": np.log(np.clip(minutes, float(minute_floor), None)),
        "events": pd.to_numeric(work[SAME_DAY_COL], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        "exog": exog,
        "rows_by_player": rows_by_player,
    }


def _sweep_points(
    design: Mapping[str, object],
    rows: np.ndarray,
    floors: Sequence[float],
    min_events: int,
) -> list[tuple[float, float, float, float]]:
    """One replicate's sweep: (floor, gamma, attenuation, ratio) per estimable floor.

    The guards mirror the deterministic curve's. A floor with too few events
    cannot support the Poisson pair; a non-positive gradient means the floor
    has exhausted the minute variation; and a non-positive attenuation makes
    the ratio meaningless rather than merely noisy.
    """
    minutes = design["minutes"][rows]
    log_minutes = design["log_minutes"][rows]
    events = design["events"][rows]
    exog = design["exog"][rows]

    points: list[tuple[float, float, float, float]] = []
    log_90 = float(np.log(90.0))
    for floor in floors:
        mask = minutes >= float(floor)
        if float(events[mask].sum()) < float(min_events):
            continue
        x = exog[mask]
        gamma = float(np.linalg.lstsq(x, log_minutes[mask], rcond=None)[0][1])
        if gamma <= 0.0:
            continue
        coefficients = {}
        for label, offset in (
            ("fixed_90", np.full(int(mask.sum()), log_90)),
            ("observed", log_minutes[mask]),
        ):
            fit = sm.GLM(
                events[mask], x, family=sm.families.Poisson(), offset=offset
            ).fit()
            coefficients[label] = float(fit.params[1])
        attenuation = coefficients["fixed_90"] - coefficients["observed"]
        if attenuation <= 0.0:
            continue
        points.append((float(floor), gamma, attenuation, gamma / attenuation))
    return points


def bootstrap_calibration_sweep(
    frame: pd.DataFrame,
    exposure_col: str,
    calendar_terms: Sequence[str],
    history_col: str,
    floors: Sequence[float] = MINUTE_FLOORS,
    n_boot: int = CURVE_BOOTSTRAP_DRAWS,
    seed: int = 20260822,
    min_events: int = MIN_EVENTS_FOR_CURVE,
    threshold: float = NEGLIGIBLE_GAMMA,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Percentile intervals for the sweep's ratios, and for the threshold cost.

    Players are resampled with replacement, each carrying all their rows, so
    the within-player correlation the published intervals respect is respected
    here too. Within each replicate the whole sweep is recomputed and the
    ratio is interpolated at the reporting threshold on that replicate's own
    curve, which makes the threshold-cost interval a joint statement rather
    than a splice of per-floor margins.

    This exists because the paper's stated discipline is that every estimate
    arrives with an interval, and the sweep's ratios were the one place it
    broke its own rule.
    """
    design = _sweep_design(frame, exposure_col, calendar_terms, history_col)
    rows_by_player = design["rows_by_player"]
    n_players = len(rows_by_player)

    rng = np.random.default_rng(seed)
    per_floor: dict[float, dict[str, list[float]]] = {
        float(floor): {"gamma": [], "attenuation": [], "ratio": []} for floor in floors
    }
    threshold_costs: list[float] = []
    threshold_ratios: list[float] = []

    for _ in range(int(n_boot)):
        sampled = rng.integers(0, n_players, n_players)
        rows = np.concatenate([rows_by_player[player] for player in sampled])
        points = _sweep_points(design, rows, floors, min_events)

        for floor, gamma, attenuation, ratio in points:
            bucket = per_floor[floor]
            bucket["gamma"].append(gamma)
            bucket["attenuation"].append(attenuation)
            bucket["ratio"].append(ratio)

        if len(points) >= 2:
            ordered = sorted((gamma, ratio) for _, gamma, _, ratio in points)
            grid = [gamma for gamma, _ in ordered]
            ratios = [ratio for _, ratio in ordered]
            if grid[0] <= float(threshold) <= grid[-1]:
                ratio_here = float(np.interp(float(threshold), grid, ratios))
                threshold_ratios.append(ratio_here)
                threshold_costs.append(
                    percent_understatement(float(threshold) / ratio_here)
                )

    rows_out = []
    for floor in floors:
        bucket = per_floor[float(floor)]
        if not bucket["ratio"]:
            continue
        rows_out.append(
            {
                "minute_floor": float(floor),
                "n_boot_valid": len(bucket["ratio"]),
                "ratio_ci_low": float(np.percentile(bucket["ratio"], 2.5)),
                "ratio_ci_high": float(np.percentile(bucket["ratio"], 97.5)),
                "attenuation_ci_low": float(np.percentile(bucket["attenuation"], 2.5)),
                "attenuation_ci_high": float(np.percentile(bucket["attenuation"], 97.5)),
            }
        )
    if not threshold_costs:
        raise ValueError(
            "no bootstrap replicate could evaluate the threshold: the swept "
            "gradients never bracket it"
        )

    summary = {
        "threshold": float(threshold),
        "n_boot_valid": len(threshold_costs),
        "ratio_at_threshold_ci_low": float(np.percentile(threshold_ratios, 2.5)),
        "ratio_at_threshold_ci_high": float(np.percentile(threshold_ratios, 97.5)),
        "cost_percent_ci_low": float(np.percentile(threshold_costs, 2.5)),
        "cost_percent_ci_high": float(np.percentile(threshold_costs, 97.5)),
    }
    return pd.DataFrame(rows_out), summary


def ratio_at_gamma(curve: pd.DataFrame, gamma: float) -> tuple[float, bool]:
    """The over-prediction ratio at one gradient, and whether it was measured.

    Interpolation inside the measured range is honest; outside it is a guess,
    and the guess is reported as one. The previous version of this analysis
    applied the pooled ratio everywhere, which put a factor of two into the one
    place a practitioner reads -- the neighbourhood of the threshold.
    """
    _require(curve, ("gamma", "over_prediction_ratio"), "calibration curve")
    ordered = curve.sort_values("gamma")
    grid = ordered["gamma"].to_numpy(dtype=float)
    ratios = ordered["over_prediction_ratio"].to_numpy(dtype=float)
    measured = bool(grid.min() <= float(gamma) <= grid.max())
    return float(np.interp(float(gamma), grid, ratios)), measured


def excess_association_lost(base_ratio: float, attenuated_ratio: float) -> float:
    """The share of the excess association the denominator removes.

    A rate ratio of 1.27 falling to 1.09 keeps 0.09 of 0.27 of its excess over
    the null, so the division costs two thirds of it. That is a different number
    from the 14% by which the ratio itself shrinks, and a draft that used the
    word "association" for both in one sentence said the division cost a third
    when it cost two.
    """
    base, attenuated = float(base_ratio) - 1.0, float(attenuated_ratio) - 1.0
    if base <= 0.0:
        raise ValueError(f"base ratio must exceed 1, got {base_ratio}")
    return float(100.0 * (1.0 - attenuated / base))


def ascertainment_by_role(frame: pd.DataFrame) -> pd.DataFrame:
    """Events per appearance and per 1000 recorded minutes, by squad role.

    A substitute plays a fifth of a starter's minutes, so a lower event count
    per appearance is expected. Per 1000 minutes it should not be, and if it
    still is, the record is finding events in starters that it misses in
    substitutes. That matters here because the remedy this paper recommends is
    restriction to starters: if reporting is differential by role, restriction
    trades a denominator problem for a numerator one, and the reader has to be
    told so.
    """
    _require(frame, (ROLE_COL, SAME_DAY_COL, MINUTES_COL), "ascertainment frame")

    work = frame.copy()
    work[SAME_DAY_COL] = pd.to_numeric(work[SAME_DAY_COL], errors="coerce").fillna(0.0)
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")

    rows = []
    for role in (STARTER_ROLE, SUBSTITUTE_ROLE, UNKNOWN_ROLE):
        subset = work[work[ROLE_COL].astype(str).eq(role)]
        if subset.empty:
            continue
        appearances = int(len(subset))
        events = int(subset[SAME_DAY_COL].sum())
        minutes = float(subset[MINUTES_COL].fillna(0.0).sum())
        low, high = proportion_confint(events, appearances, alpha=0.05, method="wilson")
        rows.append(
            {
                "lineup_role": role,
                "appearances": appearances,
                "events": events,
                "recorded_minutes": minutes,
                "events_per_1000_appearances": 1000.0 * events / appearances,
                "events_per_1000_appearances_ci_low": 1000.0 * float(low),
                "events_per_1000_appearances_ci_high": 1000.0 * float(high),
                "events_per_1000_minutes": (
                    1000.0 * events / minutes if minutes > 0 else np.nan
                ),
                "mean_recorded_minutes": minutes / appearances,
            }
        )
    if not rows:
        raise ValueError("no squad role carried any appearance")

    out = pd.DataFrame(rows)
    indexed = out.set_index("lineup_role")
    if STARTER_ROLE in indexed.index and SUBSTITUTE_ROLE in indexed.index:
        per_appearance = (
            indexed.loc[STARTER_ROLE, "events_per_1000_appearances"]
            / indexed.loc[SUBSTITUTE_ROLE, "events_per_1000_appearances"]
        )
        per_minute = (
            indexed.loc[STARTER_ROLE, "events_per_1000_minutes"]
            / indexed.loc[SUBSTITUTE_ROLE, "events_per_1000_minutes"]
        )
        out["starter_over_substitute_per_appearance"] = float(per_appearance)
        out["starter_over_substitute_per_minute"] = float(per_minute)
    return out


def clustering_sensitivity(
    frame: pd.DataFrame,
    groupings: Mapping[str, str],
    exposure_col: str,
    calendar_terms: Sequence[str],
    minute_floor: float = 1.0,
) -> pd.DataFrame:
    """Refit the gradient under each clustering, changing nothing else.

    The point estimate cannot move: clustering is a covariance choice, not an
    estimation one. What can move is the interval, and if it moves enough to
    change a verdict read off interval bounds then the published clustering was
    doing work it should not have been.
    """
    required = [MINUTES_COL, exposure_col, *calendar_terms, *groupings.values()]
    _require(frame, required, "clustering frame")

    work = frame.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    work["log_recorded_minutes"] = np.log(work[MINUTES_COL].clip(lower=float(minute_floor)))
    work["exposure_per_90"] = pd.to_numeric(work[exposure_col], errors="coerce") / 90.0

    formula = "log_recorded_minutes ~ exposure_per_90 + " + " + ".join(calendar_terms)
    model = smf.ols(formula, data=work)

    rows = []
    for label, column in groupings.items():
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": work[column]})
        estimate = float(fit.params["exposure_per_90"])
        error = float(fit.bse["exposure_per_90"])
        low, high = fit.conf_int().loc["exposure_per_90"]
        rows.append(
            {
                "clustering": label,
                "column": column,
                "n_groups": int(work[column].nunique()),
                "gamma": estimate,
                "standard_error": error,
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_width": float(high) - float(low),
            }
        )
    out = pd.DataFrame(rows)
    narrowest = float(out["ci_width"].min())
    out["width_ratio_to_narrowest"] = out["ci_width"] / narrowest
    # The verdict the paper reads off these bounds must survive the choice of
    # clustering, or the choice was doing the work.
    out["all_bounds_exceed_threshold"] = bool(out["ci_low"].gt(NEGLIGIBLE_GAMMA).all())
    out["max_width_ratio"] = float(out["ci_width"].max() / narrowest)
    return out


#: A published gradient and a refitted one may differ only by floating-point
#: noise. Anything larger means the sensitivity check changed the model rather
#: than the covariance, which is the mistake this constant exists to catch.
GAMMA_TOLERANCE = 1e-9


def verify_published_clustering(
    clustering: pd.DataFrame, published_gamma: float, label: str = "player (published)"
) -> None:
    """Refuse a sensitivity table whose baseline is not the published fit.

    Clustering is a covariance choice, so the point estimate cannot move. If the
    row labelled as the published clustering does not reproduce the published
    gradient exactly, the specification drifted and the other rows are not
    comparable to anything.
    """
    _require(clustering, ("clustering", "gamma"), "clustering sensitivity")
    match = clustering[clustering["clustering"].eq(label)]
    if match.empty:
        raise ValueError(f"no clustering row labelled {label!r}")
    refitted = float(match["gamma"].iloc[0])
    if abs(refitted - float(published_gamma)) > GAMMA_TOLERANCE:
        raise ValueError(
            f"clustering sensitivity re-derived gamma {refitted:.12g}, which does not "
            f"match the published {float(published_gamma):.12g}; the specification "
            "changed, not just the covariance"
        )


def verify_curve_baseline(
    curve: pd.DataFrame, published_gamma: float, published_attenuation: float
) -> None:
    """Refuse a curve whose unrestricted row is not the published cohort.

    The floor-zero row applies no restriction, so it must reproduce both
    published quantities exactly. If it does not, the sweep is measuring some
    other model and every ratio on the curve is uninterpretable.
    """
    _require(curve, ("minute_floor", "gamma", "observed_attenuation"), "calibration curve")
    baseline = curve[curve["minute_floor"].eq(0.0)]
    if baseline.empty:
        raise ValueError("the calibration curve has no unrestricted row")
    row = baseline.iloc[0]
    for name, refitted, published in (
        ("gamma", float(row["gamma"]), float(published_gamma)),
        ("attenuation", float(row["observed_attenuation"]), float(published_attenuation)),
    ):
        if abs(refitted - published) > GAMMA_TOLERANCE:
            raise ValueError(
                f"calibration curve re-derived {name} {refitted:.12g}, which does not "
                f"match the published {published:.12g}"
            )


def precision_profile(leagues: pd.DataFrame, gamma_column: str, label: str) -> pd.DataFrame:
    """Report the precision actually achieved, smallest panel first.

    No target sample size was set, so the honest substitute for a power
    statement is the precision the data delivered. Ordering by panel size lets a
    reader see immediately whether the smallest league is carrying a verdict its
    interval cannot support.
    """
    low_col, high_col = f"{gamma_column}_ci_low", f"{gamma_column}_ci_high"
    _require(leagues, ("league", "n_appearances", gamma_column, low_col, high_col), label)

    out = leagues[["league", "n_appearances", gamma_column, low_col, high_col]].copy()
    out = out.rename(
        columns={gamma_column: "gamma", low_col: "ci_low", high_col: "ci_high"}
    )
    out["population"] = label
    out["ci_half_width"] = (out["ci_high"] - out["ci_low"]) / 2.0
    out["relative_half_width"] = out["ci_half_width"] / out["gamma"].abs()
    return out.sort_values("n_appearances").reset_index(drop=True)


def main() -> None:  # pragma: no cover - orchestration
    """Compute every referee-requested quantity and deposit it."""
    results = RESULTS
    results.mkdir(parents=True, exist_ok=True)

    referee = load_source_module("36_jsams_second_referee_analysis.py", "referee36")
    gradient = load_source_module("37_denominator_gradient.py", "gradient37")
    primary = load_source_module(
        "18_match_proxy_poisson_splines_perminute.py", "primary18"
    )
    previous = load_source_module("34_jsams_referee_analysis.py", "previous34")

    print("1. Calibrating the first-order identity against the observed attenuation ...")
    decomposition = pd.read_csv(
        results / "jsams_revised_denominator_attenuation_decomposition.csv"
    ).set_index("quantity")
    roles = pd.read_csv(results / "jsams_revised_denominator_by_lineup_role.csv")
    calibration = identity_calibration(decomposition, roles)
    calibration.to_csv(results / "jsams_identity_calibration.csv", index=False)
    print(f"   identity over-predicts by {calibration_factor(calibration):.2f}x pooled")

    print("2. Rebuilding the reference cohort ...")
    panel, injuries, episodes, lineups, _, _ = referee.read_inputs(ROOT)
    panel = previous.add_same_day_quality_outcomes(
        panel, episodes, primary.classify_public_injury_type
    )
    panel = referee.add_negative_control_outcomes(
        panel, episodes, primary.classify_public_injury_type
    )
    panel = referee.add_prior_window_metrics(panel)
    frame, _ = previous.prepare_jsams_frame(
        primary, panel, injuries, lineups, ROOT / "external_data" / "transfermarkt"
    )

    print("3. Sweeping the gradient to measure the ratio where the threshold sits ...")
    curve = calibration_curve(
        frame, "prior_minutes_7d", list(referee.CALENDAR_TERMS), HISTORY_MODEL_COL
    )
    verify_curve_baseline(
        curve,
        float(decomposition.loc["gamma_log_minutes_on_exposure", "value"]),
        float(
            roles.drop_duplicates("lineup_role")
            .set_index("lineup_role")
            .loc["all", "log_attenuation_fixed90_minus_recorded"]
        ),
    )
    print(
        f"   gamma swept from {curve['gamma'].min():.3f} to {curve['gamma'].max():.3f}; "
        f"ratio {curve['over_prediction_ratio'].min():.2f} to "
        f"{curve['over_prediction_ratio'].max():.2f}"
    )

    print("3b. Bootstrapping the sweep, 1000 player resamples ...")
    intervals, threshold_interval = bootstrap_calibration_sweep(
        frame, "prior_minutes_7d", list(referee.CALENDAR_TERMS), HISTORY_MODEL_COL
    )
    curve = curve.merge(intervals, on="minute_floor", how="left")
    curve.to_csv(results / "jsams_calibration_curve.csv", index=False)
    pd.DataFrame([threshold_interval]).to_csv(
        results / "jsams_threshold_cost_interval.csv", index=False
    )
    print(
        f"   threshold cost {threshold_interval['cost_percent_ci_low']:.1f}% to "
        f"{threshold_interval['cost_percent_ci_high']:.1f}% "
        f"over {threshold_interval['n_boot_valid']} replicates"
    )

    translation = threshold_translation(curve)
    translation.to_csv(results / "jsams_threshold_translation.csv", index=False)
    at_threshold = translation[translation["is_reporting_threshold"].astype(bool)].iloc[0]
    print(
        f"   threshold {at_threshold['gamma']:g} costs "
        f"{at_threshold['calibrated_percent_understatement']:.2f}% of the ratio "
        f"(ratio {at_threshold['over_prediction_ratio']:.2f}, "
        f"measured={bool(at_threshold['ratio_is_measured'])})"
    )

    print("4. Comparing ascertainment across squad roles ...")
    ascertainment = ascertainment_by_role(frame)
    ascertainment.to_csv(results / "jsams_ascertainment_by_role.csv", index=False)

    print("5. Refitting the gradient under three clusterings ...")
    work = frame.copy()
    work["club_season"] = (
        work[CLUB_COL].astype(str) + "_" + pd.to_datetime(work[DATE_COL]).dt.year.astype(str)
    )
    work["fixture"] = (
        work[CLUB_COL].astype(str) + "_" + pd.to_datetime(work[DATE_COL]).dt.date.astype(str)
    )
    clustering = clustering_sensitivity(
        work,
        {
            "player (published)": PLAYER_ID_COL,
            "club-season": "club_season",
            "fixture": "fixture",
        },
        "prior_minutes_7d",
        # The published gradient conditions on prior report history as well as
        # calendar phase. Omitting it here would refit a different model and
        # call the difference a clustering effect.
        [HISTORY_MODEL_COL, *referee.CALENDAR_TERMS],
    )
    verify_published_clustering(
        clustering, float(decomposition.loc["gamma_log_minutes_on_exposure", "value"])
    )
    clustering.to_csv(results / "jsams_clustering_sensitivity.csv", index=False)

    print("6. Recording the precision the panels actually delivered ...")
    profiles = pd.concat(
        [
            precision_profile(
                pd.read_csv(results / "jsams_revised_denominator_gradient_by_league.csv"),
                "gamma_pooled",
                "men",
            ),
            precision_profile(
                pd.read_csv(results / "jsams_womens_denominator_gradient_by_league.csv"),
                "gamma_pooled",
                "women",
            ),
        ],
        ignore_index=True,
    )
    profiles.to_csv(results / "jsams_precision_profile.csv", index=False)

    print("Wrote 6 tables to data/processed/results")


if __name__ == "__main__":  # pragma: no cover
    main()
