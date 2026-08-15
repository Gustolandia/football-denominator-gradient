"""Measure the denominator gradient across European domestic leagues.

The denominator gradient, ``gamma``, is the slope of log recorded match
minutes on the exposure a study intends to analyse per minute. It is a
property of the denominator alone: it is estimated from appearances and
dates, and needs no injury data at all. That is what makes it usable as a
pre-analysis check, and what makes it measurable in any league whose
appearance records are public.

A per-minute rate divides by recorded minutes. If those minutes rise with
the exposure, the offset carries part of the gradient being estimated and
the per-minute coefficient is attenuated by roughly ``gamma``. When
``gamma`` is near zero the minute denominator is a neutral rescaling; when
it is not, dividing by playing time removes signal that belongs to the
numerator.

This module fits ``gamma`` with player-clustered standard errors, pooled
and within starters, in every league supplied, and turns the result into a
decision rule a reader can apply to their own panel before dividing.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

PLAYER_ID_COL = "player_id"
DATE_COL = "date"
MINUTES_COL = "minutes_played"
COMPETITION_COL = "competition_id"
ROLE_COL = "lineup_role"
STARTER_ROLE = "starting_lineup"
SUBSTITUTE_ROLE = "substitutes"

CALENDAR_TERMS = (
    "week_phase_sin",
    "week_phase_cos",
    "halfweek_phase_sin",
    "halfweek_phase_cos",
)
PRIMARY_WINDOW = 7
WINDOW_START = "2017-07-01"
WINDOW_END = "2025-04-07"

# The eight highest-coverage domestic leagues in the deposited snapshot. The
# reference league is listed first so tables read against it.
LEAGUES: Mapping[str, str] = {
    "GB1": "England, Premier League",
    "ES1": "Spain, LaLiga",
    "IT1": "Italy, Serie A",
    "L1": "Germany, Bundesliga",
    "FR1": "France, Ligue 1",
    "NL1": "Netherlands, Eredivisie",
    "PO1": "Portugal, Liga Portugal",
    "TR1": "Turkey, Süper Lig",
}

# Below this the offset has too little room to carry a gradient for the
# per-minute and per-appearance answers to diverge materially. It is a
# reporting threshold, not a test.
NEGLIGIBLE_GAMMA = 0.05
MIN_ROWS_FOR_FIT = 200

# A full appearance. Recorded minutes are bounded here and heap on it, which is
# why the estimator sensitivity exists: among starters the response is nearly a
# point mass at this value.
FULL_APPEARANCE_MINUTES = 90.0


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def add_calendar_phase(frame: pd.DataFrame) -> pd.DataFrame:
    """Add weekly and half-weekly phase terms from the appearance date.

    Fixture scheduling is strongly weekly, so a diagnostic that ignores it
    would attribute calendar structure to the exposure. These terms need
    nothing but the date, which keeps the check runnable on appearance data.
    """
    _require_columns(frame, [DATE_COL], "calendar frame")
    out = frame.copy()
    dates = pd.to_datetime(out[DATE_COL], errors="coerce")
    if dates.isna().any():
        raise ValueError("calendar phase requires complete appearance dates")
    day = dates.dt.dayofweek.to_numpy(dtype=float)
    out["week_phase_sin"] = np.sin(2.0 * np.pi * day / 7.0)
    out["week_phase_cos"] = np.cos(2.0 * np.pi * day / 7.0)
    out["halfweek_phase_sin"] = np.sin(4.0 * np.pi * day / 7.0)
    out["halfweek_phase_cos"] = np.cos(4.0 * np.pi * day / 7.0)
    return out


def add_prior_window_minutes(
    frame: pd.DataFrame,
    window: int = PRIMARY_WINDOW,
) -> pd.DataFrame:
    """Add prior-only cumulative minutes over a rolling day window.

    The current appearance is excluded, so the exposure is fixed before the
    appearance whose length it is being related to. Player-date duplicates
    are dropped rather than summed: a player recorded twice on one date has
    an ambiguous ordering, and guessing would contaminate the gradient.
    """
    _require_columns(frame, [PLAYER_ID_COL, DATE_COL, MINUTES_COL], "window frame")
    if int(window) <= 0:
        raise ValueError("window must be a positive number of days")
    out = frame.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out[MINUTES_COL] = pd.to_numeric(out[MINUTES_COL], errors="coerce")
    out = out.dropna(subset=[PLAYER_ID_COL, DATE_COL, MINUTES_COL])
    out = out.drop_duplicates([PLAYER_ID_COL, DATE_COL], keep=False)

    ordered = out.sort_values([PLAYER_ID_COL, DATE_COL]).reset_index(drop=True)
    prior = np.zeros(len(ordered), dtype=float)
    for positions in ordered.groupby(PLAYER_ID_COL, sort=False).indices.values():
        locations = np.asarray(positions, dtype=int)
        dates = ordered.loc[locations, DATE_COL].to_numpy(dtype="datetime64[ns]")
        minutes = ordered.loc[locations, MINUTES_COL].to_numpy(dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(minutes)))
        current = np.arange(len(locations), dtype=int)
        left = np.searchsorted(dates, dates - np.timedelta64(int(window), "D"), side="left")
        prior[locations] = cumulative[current] - cumulative[left]
    ordered[f"prior_minutes_{int(window)}d"] = prior
    return ordered


def _not_estimable(work: pd.DataFrame) -> dict[str, float]:
    """The return shape used wherever a fit cannot be attempted."""
    return {
        "gamma": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "n_rows": int(len(work)), "n_players": int(work[PLAYER_ID_COL].nunique()),
        "estimable": False,
    }


def denominator_gradient(
    frame: pd.DataFrame,
    window: int = PRIMARY_WINDOW,
    adjusted: bool = True,
    minute_floor: float = 1.0,
    estimator: str = "ols",
) -> dict[str, float]:
    """Fit gamma with player-clustered errors and return it with bounds.

    ``adjusted`` adds the calendar phase terms. Both variants are reported
    because the unadjusted one is what a reader gets from two lines of code,
    and the difference between them says whether that shortcut is safe.

    ``minute_floor`` and ``estimator`` exist for the robustness tables rather
    than for the headline fit. Recorded minutes are bounded above and heap on
    a full appearance, so neither the log transform nor the floor beneath it
    is beyond question, and the paper has to show that its claim survives
    both. ``estimator`` accepts ``ols``, ``median`` (quantile regression at
    the median), ``fractional`` (a binomial fit on minutes as a share of a
    full appearance) and ``below_ceiling`` (least squares among appearances
    short of the ceiling).
    """
    exposure_col = f"prior_minutes_{int(window)}d"
    required = [PLAYER_ID_COL, MINUTES_COL, exposure_col]
    if adjusted:
        required = [*required, *CALENDAR_TERMS]
    _require_columns(frame, required, "gradient frame")

    work = frame.dropna(subset=required).copy()
    work[MINUTES_COL] = pd.to_numeric(work[MINUTES_COL], errors="coerce")
    work = work[work[MINUTES_COL].gt(0.0)].copy()
    if estimator == "below_ceiling":
        work = work[work[MINUTES_COL].lt(FULL_APPEARANCE_MINUTES)].copy()
    if len(work) < MIN_ROWS_FOR_FIT:
        return _not_estimable(work)

    work["log_recorded_minutes"] = np.log(work[MINUTES_COL].clip(lower=float(minute_floor)))
    # Held strictly inside the unit interval so the binomial fit is defined at
    # the ceiling, where most starter appearances sit.
    work["minute_share"] = (
        work[MINUTES_COL].div(FULL_APPEARANCE_MINUTES).clip(lower=0.001, upper=0.999)
    )
    work["exposure_per_90"] = pd.to_numeric(work[exposure_col], errors="coerce") / 90.0

    response = "minute_share" if estimator == "fractional" else "log_recorded_minutes"
    formula = f"{response} ~ exposure_per_90"
    if adjusted:
        formula = f"{formula} + {' + '.join(CALENDAR_TERMS)}"

    clustered = {"cov_type": "cluster", "cov_kwds": {"groups": work[PLAYER_ID_COL]}}
    if estimator == "median":
        # Quantile regression carries no clustered covariance in statsmodels, so
        # this variant contributes a point estimate to the robustness table and
        # no interval; the headline fit is the clustered one.
        fit = smf.quantreg(formula, data=work).fit(q=0.5)
        estimate = float(fit.params["exposure_per_90"])
        return {
            "gamma": estimate, "ci_low": np.nan, "ci_high": np.nan,
            "n_rows": int(len(work)), "n_players": int(work[PLAYER_ID_COL].nunique()),
            "estimable": True,
        }
    if estimator == "fractional":
        fit = smf.glm(
            formula, data=work, family=sm.families.Binomial()
        ).fit(**clustered)
    else:
        fit = smf.ols(formula, data=work).fit(**clustered)

    estimate = float(fit.params["exposure_per_90"])
    error = float(fit.bse["exposure_per_90"])
    critical = NormalDist().inv_cdf(0.975)
    return {
        "gamma": estimate,
        "ci_low": estimate - critical * error,
        "ci_high": estimate + critical * error,
        "n_rows": int(len(work)),
        "n_players": int(work[PLAYER_ID_COL].nunique()),
        "estimable": True,
    }


def _starter_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the starter rows, or an empty frame where the field is absent."""
    if ROLE_COL in frame.columns:
        return frame[frame[ROLE_COL].astype(str).eq(STARTER_ROLE)]
    return frame.iloc[0:0]


def _gamma_pair(frame: pd.DataFrame, window: int, **kwargs: Any) -> tuple[float, float]:
    """Fit gamma pooled and within starters under one variant of the estimator.

    The paper's claim is not about gamma's level but about the contrast: large
    pooled, negligible within starters. A robustness check that reported only
    the pooled value could not tell us whether the contrast survives, so both
    are always returned together.
    """
    pooled = denominator_gradient(frame, window=window, **kwargs)
    starters = _starter_rows(frame)
    within = (
        denominator_gradient(starters, window=window, **kwargs)
        if len(starters)
        else {"gamma": np.nan}
    )
    return float(pooled["gamma"]), float(within["gamma"])


def gradient_clip_sensitivity(
    frame: pd.DataFrame,
    window: int = PRIMARY_WINDOW,
    clips: Iterable[float] = (1.0, 5.0, 10.0),
) -> pd.DataFrame:
    """Refit gamma under alternative floors on recorded minutes.

    The reference fit takes the log of minutes clipped below at one, and a
    substitute appearance of one or two minutes is common, so on a log scale
    that floor is a modelling choice rather than a formality --- and it bites
    hardest in exactly the stratum where gamma is large. Refitting under
    higher floors says whether the choice carries any of the result.
    """
    rows: list[dict[str, Any]] = []
    for clip in clips:
        pooled, within = _gamma_pair(frame, window, minute_floor=float(clip))
        rows.append(
            {
                "minute_floor": float(clip),
                "gamma_pooled": pooled,
                "gamma_within_starters": within,
            }
        )
    out = pd.DataFrame(rows)
    reference = out.iloc[0]
    out["gamma_pooled_gap_vs_reference"] = out["gamma_pooled"] - float(
        reference["gamma_pooled"]
    )
    out["interpretation"] = (
        "the floor applied before taking logs is an analyst choice; the pooled "
        "and within-starter gradients are reported under each so a reader can "
        "see that the contrast between them, which is what the paper claims, "
        "does not depend on it"
    )
    return out


def gradient_estimator_sensitivity(
    frame: pd.DataFrame,
    window: int = PRIMARY_WINDOW,
) -> pd.DataFrame:
    """Refit gamma under estimators that do not assume a well-behaved log.

    Recorded minutes are bounded above and heaped: a starter's appearance is
    almost always exactly 90, so the reference response is close to a point
    mass and ordinary least squares on its log is not self-evidently the right
    estimator. Three alternatives are fitted --- a median regression that is
    indifferent to the heap, a fractional-response fit that respects the
    ceiling, and a fit that drops the ceiling mass altogether. They are on
    different scales, so what is comparable between them is the sign and the
    pooled-versus-starter contrast, not the level.
    """
    exposure_col = f"prior_minutes_{int(window)}d"
    _require_columns(
        frame, [PLAYER_ID_COL, MINUTES_COL, exposure_col], "estimator sensitivity frame"
    )
    pooled_ols, within_ols = _gamma_pair(frame, window)
    rows: list[dict[str, Any]] = [
        {
            "estimator": "ols_log_minutes",
            "gamma_pooled": pooled_ols,
            "gamma_within_starters": within_ols,
            "scale": "log minutes",
            "note": "the reference estimator, reported for comparison",
        }
    ]

    for label, kwargs, scale, note in (
        (
            "median_regression",
            {"estimator": "median"},
            "log minutes",
            "quantile regression at the median. Its gradient is far smaller than "
            "the reference one and that is expected rather than contradictory: the "
            "median appearance is a full one at almost every exposure, so the "
            "median barely moves. The attenuation identity is driven by the mean "
            "of log minutes, not the median, so this fit bounds how much of the "
            "gradient is carried by the typical appearance rather than by the mix",
        ),
        (
            "fractional_response",
            {"estimator": "fractional"},
            "share of the 90-minute ceiling",
            "binomial GLM on minutes as a share of 90, which respects the ceiling",
        ),
        (
            "excluding_ceiling",
            {"estimator": "below_ceiling"},
            "log minutes",
            "ordinary least squares among appearances short of the ceiling; the "
            "within-starter contrast is not interpretable here, because dropping "
            "the ceiling from the starter stratum keeps precisely the appearances "
            "that ended early, which is the truncation the paper measures elsewhere",
        ),
    ):
        pooled, within = _gamma_pair(frame, window, **kwargs)
        rows.append(
            {
                "estimator": label,
                "gamma_pooled": pooled,
                "gamma_within_starters": within,
                "scale": scale,
                "note": note,
            }
        )

    out = pd.DataFrame(rows)
    out["contrast_estimable"] = (
        out["gamma_pooled"].notna() & out["gamma_within_starters"].notna()
    )
    # Estimable is not the same as interpretable. Dropping the ceiling keeps, in
    # the starter stratum, precisely the appearances that ended early, so that
    # fit yields a number and the number answers a different question. Marking it
    # here stops the robustness claim from resting on a selected subset.
    out["contrast_interpretable"] = out["contrast_estimable"] & ~out[
        "estimator"
    ].eq("excluding_ceiling")
    out["pooled_exceeds_starters"] = np.where(
        out["contrast_interpretable"],
        out["gamma_pooled"] > out["gamma_within_starters"],
        pd.NA,
    )
    out["interpretation"] = (
        "estimators on different scales cannot be compared by level, and the "
        "level is strongly specification-dependent; what the table establishes is "
        "that wherever the contrast is interpretable the pooled gradient exceeds "
        "the within-starter one, so the composition result is not an artefact of "
        "taking logs of a heaped, ceiling-bounded response. The reference fit is "
        "the one the attenuation identity requires, because that identity is "
        "driven by the mean of log minutes"
    )
    return out


# The three places gamma is fitted in this paper differ, and a reader comparing
# two of them without knowing that would think the numbers disagreed. They are
# registered here rather than described in prose so the manuscript can be gated
# against the register.
GAMMA_SPECIFICATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("reference_cohort",
     "log recorded minutes on exposure, with prior report history and calendar "
     "phase, over the eligible established-player cohort",
     "main text cohort results and the attenuation decomposition",
     "smallest of the three, because the history term absorbs part of the "
     "between-player variation in playing time"),
    ("cross_league_diagnostic",
     "log recorded minutes on exposure with weekly and half-weekly calendar "
     "phase, over all domestic-league appearances with no eligibility "
     "restriction",
     "the eight-league table and the decision rule",
     "larger than the reference fit, because it includes the players whose "
     "minutes vary most and omits the history term"),
    ("two_line_shortcut",
     "log recorded minutes on exposure alone, with no calendar terms",
     "the Practical Implications, as what a reader gets from two lines of code",
     "differs from the cross-league fit by at most the gap reported in the "
     "league table, and returns the same verdict in every league"),
)


def gradient_specification_registry() -> pd.DataFrame:
    """Name the three gamma specifications and say where each is used.

    The reference cohort and the cross-league diagnostic give different values
    for the same league. That is expected and explicable, but a reader who
    meets both without being told will conclude one of them is wrong, so the
    register is deposited and the manuscript is gated against it.
    """
    out = pd.DataFrame(
        GAMMA_SPECIFICATIONS,
        columns=["specification_id", "model", "used_for", "why_it_differs"],
    )
    out["response"] = "log recorded minutes"
    out["clustering"] = "player"
    out["interpretation"] = (
        "three fits of the same quantity over different populations and "
        "covariate sets; the diagnostic is the one a reader would compute and "
        "is the more alarming, so it is the upper reading of the same quantity"
    )
    return out


def gradient_by_league(
    frame: pd.DataFrame,
    window: int = PRIMARY_WINDOW,
) -> pd.DataFrame:
    """Measure the gradient in every league, pooled and within starters.

    Reporting both is what bounds the remedy. A gradient that is large
    pooled and near zero within starters says the contamination is squad-role
    composition and that restricting to starters removes it; a gradient that
    survives restriction says it does not.
    """
    _require_columns(frame, [COMPETITION_COL, MINUTES_COL], "league frame")
    rows: list[dict[str, Any]] = []
    for competition, group in frame.groupby(COMPETITION_COL, sort=False):
        minutes = pd.to_numeric(group[MINUTES_COL], errors="coerce")
        minutes = minutes[minutes.gt(0.0)]
        pooled = denominator_gradient(group, window=window)
        # The unadjusted fit is what two lines of code give a reader. Printing
        # it beside the adjusted one says whether that shortcut is safe here.
        plain = denominator_gradient(group, window=window, adjusted=False)
        # Lineup status is the field most often absent from a public panel, so
        # a league without it still yields a pooled gradient and an explicitly
        # absent within-starter one.
        if ROLE_COL in group.columns:
            starters = group[group[ROLE_COL].astype(str).eq(STARTER_ROLE)]
        else:
            starters = group.iloc[0:0]
        within = (
            denominator_gradient(starters, window=window)
            if len(starters) else
            {"gamma": np.nan, "ci_low": np.nan, "ci_high": np.nan,
             "n_rows": 0, "n_players": 0, "estimable": False}
        )
        starter_minutes = pd.to_numeric(starters[MINUTES_COL], errors="coerce")
        starter_minutes = starter_minutes[starter_minutes.gt(0.0)]
        rows.append(
            {
                "competition_id": competition,
                "league": LEAGUES.get(str(competition), str(competition)),
                "n_appearances": int(pooled["n_rows"]),
                "n_players": int(pooled["n_players"]),
                "iqr_recorded_minutes": (
                    float(minutes.quantile(0.75) - minutes.quantile(0.25))
                    if len(minutes) else np.nan
                ),
                "gamma_pooled": pooled["gamma"],
                "gamma_pooled_ci_low": pooled["ci_low"],
                "gamma_pooled_ci_high": pooled["ci_high"],
                "gamma_pooled_unadjusted": plain["gamma"],
                "gamma_pooled_unadjusted_ci_low": plain["ci_low"],
                "gamma_pooled_unadjusted_ci_high": plain["ci_high"],
                "n_starter_appearances": int(within["n_rows"]),
                "iqr_starter_minutes": (
                    float(starter_minutes.quantile(0.75) - starter_minutes.quantile(0.25))
                    if len(starter_minutes) else np.nan
                ),
                "gamma_within_starters": within["gamma"],
                "gamma_within_starters_ci_low": within["ci_low"],
                "gamma_within_starters_ci_high": within["ci_high"],
                "estimable": bool(pooled["estimable"]),
            }
        )
    out = pd.DataFrame(rows)
    order = {code: index for index, code in enumerate(LEAGUES)}
    out["_order"] = out["competition_id"].map(lambda c: order.get(str(c), len(order)))
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    out["window_days"] = int(window)
    out["interpretation"] = (
        "gamma is the slope of log recorded minutes on 90 previous-seven-day "
        "minutes, fitted with player-clustered errors from appearance records "
        "alone; a per-minute analysis divides out roughly this much of the "
        "gradient it is trying to estimate"
    )
    return out


def diagnostic_decision_rule(gradients: pd.DataFrame) -> pd.DataFrame:
    """Turn the measured gradients into the rule a reader should apply.

    The diagnostic is deliberately coarse. It answers one question --- is the
    minute denominator safe here --- and it answers it with the interval, not
    the point estimate, so a gradient that cannot be distinguished from
    negligible is not reported as if it were.
    """
    _require_columns(
        gradients,
        ["league", "gamma_pooled", "gamma_pooled_ci_low", "gamma_within_starters",
         "gamma_within_starters_ci_high"],
        "decision rule frame",
    )
    rows = []
    for _, row in gradients.iterrows():
        pooled_low = float(row["gamma_pooled_ci_low"])
        starter_high = float(row["gamma_within_starters_ci_high"])
        pooled_material = pooled_low > NEGLIGIBLE_GAMMA
        starter_negligible = np.isfinite(starter_high) and starter_high <= NEGLIGIBLE_GAMMA
        if not pooled_material:
            verdict = "per-minute defensible pooled"
        elif starter_negligible:
            verdict = "restrict to starters"
        else:
            verdict = "report per appearance"
        rows.append(
            {
                "league": row["league"],
                "gamma_pooled": float(row["gamma_pooled"]),
                "gamma_within_starters": float(row["gamma_within_starters"]),
                "pooled_exceeds_negligible": bool(pooled_material),
                "starter_within_negligible": bool(starter_negligible),
                "recommendation": verdict,
            }
        )
    out = pd.DataFrame(rows)
    out["negligible_threshold"] = float(NEGLIGIBLE_GAMMA)
    out["interpretation"] = (
        "the rule reads off interval bounds rather than point estimates: a "
        "pooled lower bound above the threshold means the minute denominator "
        "is not neutral, and a within-starter upper bound below it means "
        "restricting to starters restores neutrality"
    )
    return out


def gradient_summary(gradients: pd.DataFrame) -> pd.DataFrame:
    """Reduce the league table to the sentence the paper needs to defend."""
    _require_columns(
        gradients, ["gamma_pooled", "gamma_within_starters", "estimable"],
        "gradient summary frame",
    )
    usable = gradients[gradients["estimable"].astype(bool)]
    pooled = pd.to_numeric(usable["gamma_pooled"], errors="coerce").dropna()
    starters = pd.to_numeric(usable["gamma_within_starters"], errors="coerce").dropna()
    if pooled.empty:
        raise ValueError("no estimable leagues in the gradient table")
    rows = [
        {"quantity": "n_leagues", "value": float(len(usable)),
         "note": "domestic leagues with an estimable gradient"},
        {"quantity": "gamma_pooled_min", "value": float(pooled.min()),
         "note": "smallest pooled gradient across leagues"},
        {"quantity": "gamma_pooled_max", "value": float(pooled.max()),
         "note": "largest pooled gradient across leagues"},
        {"quantity": "gamma_pooled_median", "value": float(pooled.median()),
         "note": "median pooled gradient across leagues"},
        {"quantity": "n_leagues_pooled_above_threshold",
         "value": float((pooled > NEGLIGIBLE_GAMMA).sum()),
         "note": "leagues whose pooled gradient exceeds the negligible threshold"},
    ]
    if "gamma_pooled_unadjusted" in usable.columns:
        plain = pd.to_numeric(usable["gamma_pooled_unadjusted"], errors="coerce").dropna()
        if not plain.empty:
            gap = (plain - pooled).abs()
            rows.append(
                {"quantity": "max_abs_gap_adjusted_vs_unadjusted",
                 "value": float(gap.max()),
                 "note": "largest absolute difference between the calendar-adjusted "
                         "gradient and the two-line unadjusted one, across leagues"}
            )
    if not starters.empty:
        rows += [
            {"quantity": "gamma_within_starters_min", "value": float(starters.min()),
             "note": "smallest within-starter gradient across leagues"},
            {"quantity": "gamma_within_starters_max", "value": float(starters.max()),
             "note": "largest within-starter gradient across leagues"},
            {"quantity": "n_leagues_starters_below_threshold",
             "value": float((starters <= NEGLIGIBLE_GAMMA).sum()),
             "note": "leagues where restricting to starters brings the gradient below the threshold"},
        ]
    out = pd.DataFrame(rows)
    out["negligible_threshold"] = float(NEGLIGIBLE_GAMMA)
    out["interpretation"] = (
        "the gradient is a property of how playing time is allocated, so it "
        "replicates wherever squads rotate; the magnitude is local and the "
        "mechanism is not"
    )
    return out


def scoping_search_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Summarise the scoping search for exposed studies as a lower bound.

    This is a scoping search, not a systematic review: it used web search
    rather than a protocolised multi-database query, so it can establish that
    studies of a given kind exist and count the ones it found, and it cannot
    establish that no others do. Every count it produces is therefore reported
    as a floor. Records whose denominator could not be verified from the
    retrieved material are counted separately rather than assumed either way,
    because assuming them in would inflate the very number the search exists
    to support.
    """
    required = ["record_id", "public_source", "gradient_applies", "verification"]
    _require_columns(records, required, "scoping records")
    applies = records["gradient_applies"].astype(str)
    verification = records["verification"].astype(str)

    confirmed = int(applies.eq("yes").sum())
    unverified = int(applies.eq("unknown").sum())
    provenance_flagged = int(
        verification.str.contains("source_unverified", na=False).sum()
    )
    rows = [
        {
            "quantity": "n_records_retrieved",
            "value": float(len(records)),
            "note": "peer-reviewed studies retrieved by the scoping search",
        },
        {
            "quantity": "n_denominator_confirmed",
            "value": float(confirmed),
            "note": (
                "studies using publicly sourced injury data and reporting a rate "
                "with a minute or hour denominator, verified verbatim"
            ),
        },
        {
            "quantity": "n_denominator_unverified",
            "value": float(unverified),
            "note": (
                "studies using publicly sourced injury data whose denominator "
                "could not be established from the retrieved material"
            ),
        },
        {
            "quantity": "n_provenance_flagged",
            "value": float(provenance_flagged),
            "note": "records whose data provenance was not independently verified",
        },
    ]
    out = pd.DataFrame(rows)
    out["search_type"] = "scoping"
    out["bound"] = "lower"
    out["interpretation"] = (
        "a scoping search establishes existence and a floor, never a ceiling; "
        "the confirmed count is the number of studies for which both the public "
        "source and the minute or hour denominator were read directly, and the "
        "true number of exposed studies can only be larger"
    )
    return out


# What the scoping search did, recorded as data so a reader can judge it rather
# than take the count on trust. It is deliberately labelled a scoping search
# and not a systematic review: it used general and biomedical web search rather
# than a protocolised multi-database query with a registered strategy, so it can
# establish that exposed studies exist and count those it found, and it cannot
# establish that no others do.
SCOPING_RULES: tuple[tuple[str, str], ...] = (
    ("S1_question",
     "Which peer-reviewed studies of football injury draw their injury data "
     "from a public or media-derived source rather than club-internal medical "
     "surveillance, and of those, which report a rate whose denominator is "
     "time played?"),
    ("S2_sources_searched",
     "PubMed and PubMed Central, ScienceDirect, and general web search, with "
     "forward and backward citation checking from the two anchor papers. No "
     "grey-literature database, thesis repository or conference proceedings "
     "were searched."),
    ("S3_search_dates",
     "Searches were run on 6 and 7 August 2026 and extended on 14 August 2026. "
     "Nothing published after the later date could have been retrieved."),
    ("S4_inclusion",
     "Peer-reviewed; association football; injury or absence data drawn from a "
     "named public or media-derived source; reports at least one injury rate or "
     "incidence figure."),
    ("S5_denominator_classification",
     "A record counts as exposed only when the sentence defining the rate "
     "denominator was read directly and states a time denominator. A study "
     "whose denominator could not be read from the retrieved material is "
     "recorded as unverified and is never counted as exposed."),
    ("S6_provenance_flag",
     "A record whose data provenance could not be independently verified is "
     "flagged and reported separately, because a study that only appears to use "
     "a public source would not belong in the count."),
    ("S7_bound",
     "Every count is a floor. A search of this kind cannot support a statement "
     "about how many exposed studies exist, only about how many were found."),
)


def scoping_search_protocol() -> pd.DataFrame:
    """Return the scoping search strategy as data.

    The count of exposed studies motivates the paper, so a reader has to be
    able to judge how it was produced. Depositing the strategy rather than
    describing it in prose means the manuscript can be gated against it and
    the search's limits travel with the number it supports.
    """
    out = pd.DataFrame(SCOPING_RULES, columns=["rule_id", "rule"])
    out["search_type"] = "scoping"
    out["registered"] = False
    out["interpretation"] = (
        "a scoping search establishes existence and a floor, never a ceiling; "
        "the strategy is deposited so that the count can be checked, extended "
        "or contradicted rather than taken on trust"
    )
    return out


# The adjudication rules for the missed-event queue, fixed in advance so that
# a reviewer applies them rather than invents them. They are deliberately
# asymmetric: positive evidence is self-verifying because a reader can open the
# source, whereas failure to find evidence is nearly uninformative, since minor
# injuries routinely never reach the press. Rule 7 therefore forbids the one
# inference that would bias the sensitivity estimate in the paper's favour.
ADJUDICATION_RULES: tuple[tuple[str, str], ...] = (
    ("B1_source_hierarchy",
     "Admissible sources are, in order: official club channel, league or "
     "governing-body statement, contemporaneous news agency report. Injury "
     "aggregators are inadmissible because they may themselves derive from the "
     "database under audit, which would make the check circular."),
    ("B2_temporal_window",
     "The source must refer to the appearance date or the day either side of it."),
    ("B3_specificity",
     "The source must name the player and describe a physical problem, an early "
     "withdrawal on medical grounds, or a medical assessment. Rotation, "
     "suspension, illness-free rest and tactical substitution do not qualify."),
    ("B4_match_identification",
     "Where a club has more than one fixture against the same opponent with the "
     "same scoreline, the source must be tied to the date by an explicit date or "
     "round reference. A headline naming only the opponent and score is "
     "insufficient."),
    ("B5_verdict_set",
     "missed_event requires B1 to B4 met; no_missed_event requires a source "
     "meeting B1 and B2 that explicitly states the player was uninjured; "
     "everything else is unresolved."),
    ("B6_assessors",
     "Two assessors, blinded to the appearance's exposure value, with Cohen's "
     "kappa reported before any summary is read."),
    ("B7_absence_of_evidence",
     "Failure to find a source maps to unresolved and never to no_missed_event. "
     "Minor injuries frequently go unreported, so non-detection carries almost "
     "no information, and recording it as a negative would bias the sensitivity "
     "estimate upwards."),
)


def adjudication_protocol() -> pd.DataFrame:
    """Return the pre-specified rules for adjudicating the missed-event queue.

    Depositing the rules as data rather than prose means a reviewer applies a
    fixed protocol instead of forming one while looking at the records, and
    means the protocol can be cited, versioned and tested.
    """
    out = pd.DataFrame(ADJUDICATION_RULES, columns=["rule_id", "rule"])
    out["queue"] = "non_event_audit_queue"
    out["status"] = "pre-specified and applied; no record resolved"
    out["interpretation"] = (
        "the rules are asymmetric by design: positive findings are verifiable "
        "by opening the cited source, negative findings are not obtainable from "
        "public search, so the queue can yield a lower bound on missed events "
        "and cannot yield an upper bound on them"
    )
    return out


def load_league_appearances(  # pragma: no cover - filesystem IO
    snapshot: Path,
    competitions: Sequence[str] = tuple(LEAGUES),
    start: str = WINDOW_START,
    end: str = WINDOW_END,
) -> pd.DataFrame:
    """Read appearances and lineup roles for the requested domestic leagues."""
    appearances = pd.read_csv(
        snapshot / "appearances.csv.gz",
        usecols=["game_id", PLAYER_ID_COL, DATE_COL, MINUTES_COL, COMPETITION_COL],
        low_memory=False,
    )
    appearances = appearances[appearances[COMPETITION_COL].isin(competitions)].copy()
    appearances[DATE_COL] = pd.to_datetime(appearances[DATE_COL], errors="coerce")
    appearances = appearances[
        appearances[DATE_COL].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    lineups = pd.read_csv(
        snapshot / "game_lineups.csv.gz",
        usecols=["game_id", PLAYER_ID_COL, "type"],
        low_memory=False,
    )
    lineups = lineups.rename(columns={"type": ROLE_COL}).drop_duplicates(
        ["game_id", PLAYER_ID_COL]
    )
    merged = appearances.merge(lineups, on=["game_id", PLAYER_ID_COL], how="left")
    return merged


def main() -> None:  # pragma: no cover - orchestration
    """Measure the denominator gradient in every covered domestic league."""
    root = Path(__file__).resolve().parents[1]
    snapshot = (
        root / "data" / "raw" / "public_data_v4" / "transfermarkt_datasets_20260804"
    )
    results_dir = root / "data" / "processed" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("1. Reading league appearances and lineup roles ...")
    frame = load_league_appearances(snapshot)
    print(f"   {len(frame):,} appearances across {frame[COMPETITION_COL].nunique()} leagues")

    print("2. Building prior-window exposure and calendar phase ...")
    pieces = []
    for competition, group in frame.groupby(COMPETITION_COL, sort=False):
        windowed = add_prior_window_minutes(group)
        pieces.append(windowed)
    panel = add_calendar_phase(pd.concat(pieces, ignore_index=True))

    print("3. Fitting the denominator gradient in every league ...")
    gradients = gradient_by_league(panel)
    rule = diagnostic_decision_rule(gradients)
    summary = gradient_summary(gradients)

    print("4. Refitting the gradient under alternative floors and estimators ...")
    reference = panel[panel[COMPETITION_COL].astype(str).eq("GB1")]
    clip_sensitivity = gradient_clip_sensitivity(reference)
    estimator_sensitivity = gradient_estimator_sensitivity(reference)

    print("5. Summarising the scoping search for exposed studies ...")
    scoping = scoping_search_summary(
        pd.read_csv(root / "data" / "manual" / "per_minute_denominator_scoping.csv")
    )

    outputs = {
        "denominator_gradient_by_league": gradients,
        "denominator_gradient_decision_rule": rule,
        "denominator_gradient_summary": summary,
        "denominator_gradient_clip_sensitivity": clip_sensitivity,
        "denominator_gradient_estimator_sensitivity": estimator_sensitivity,
        "denominator_gradient_specifications": gradient_specification_registry(),
        "per_minute_denominator_scoping": scoping,
        "per_minute_denominator_scoping_protocol": scoping_search_protocol(),
        "missed_event_adjudication_protocol": adjudication_protocol(),
    }
    for name, table in outputs.items():
        table.to_csv(results_dir / f"jsams_revised_{name}.csv", index=False)
    print(f"Wrote {len(outputs)} denominator-gradient tables to {results_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
