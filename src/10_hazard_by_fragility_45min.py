#!/usr/bin/env python
"""
10_hazard_by_fragility_45min.py

Estimate crude and model-based injury risk by recent 7-day load,
separately for prior-injury-history groups.

For each group:

1) Compute crude per-day injury rates by all_minutes_last_7d bins
   (45-min bins: 0–45, 46–90, …, 271–300).

2) Compute crude match-day injury rates PER MINUTE:
       events_per_10000_min = (events / total_minutes) * 10,000
   using only rows where all_minutes_played > 0.

3) For groups with enough events (>= MIN_EVENTS_FOR_GLM), fit a
   discrete-time logistic model:

       injury_event ~ C(all_minutes7d_bin, Treatment('0-45'))
                      + all_games_last_7d

   using the full eligible panel and cluster-robust SEs by stable tm_player_id.

   Predictions are daily probabilities. Model-based per-minute risk is computed
   only for display by dividing those probabilities by a typical match-day
   denominator.

Crude summaries retain every 45-minute bin. GLMs use an estimable-bin screen
that removes sparse or separated bins, so an extreme tail bin with no events
does not create a misleading near-zero/infinite odds ratio.

Inputs:
    data/processed/player_day_panel_all_comp.csv
    data/processed/player_day_fragility.csv

Outputs (per group, under data/processed/results):
    hazard_by_fragility_<group>_daily_45min.csv
    hazard_by_fragility_<group>_per_minute_45min.csv
    hazard_by_fragility_<group>_glm_estimable_bins_45min.csv
    hazard_by_fragility_<group>_glm_or_45min.csv           (if GLM fits)
    hazard_by_fragility_<group>_glm_predicted_45min.csv    (if GLM fits)

Run from repo root:

    python src/10_hazard_by_fragility_45min.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pipeline_io import (
    merge_day_fragility,
    restrict_to_available_risk_set,
    restrict_to_fragility_risk_set,
)
from pipeline_io import LABELS_45, REP_VALUES_45, add_45min_load_bins, estimable_bin_labels

MIN_EVENTS_FOR_GLM = 200  # require this many events to fit GLM reliably
CURRENT_MINUTES_COL = "all_minutes_played"
CURRENT_GAMES_COL = "all_games_played"
RECENT_GAMES_COL = "all_games_last_7d"


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    proc_dir = root / "data" / "processed"

    panel_path = proc_dir / "player_day_panel_all_comp.csv"
    out_dir = proc_dir / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"Repo root: {root}")
    print(f"Loading player-day panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    print("Panel shape:", panel.shape)

    print("Merging prior-history day-level fragility labels ...")
    panel = merge_day_fragility(panel, proc_dir)
    panel = restrict_to_fragility_risk_set(panel)
    panel = restrict_to_available_risk_set(panel)

    print("\nFragility_group value counts in panel:")
    print(panel["fragility_group"].value_counts(dropna=False))

    groups_to_analyse = ["tough", "regular", "fragile"]

    for g in groups_to_analyse:
        print(f"\n================ {g.upper()} PLAYERS ================")

        sub = panel[panel["fragility_group"] == g].copy()
        if sub.empty:
            print(f"No rows for group {g}; skipping.")
            continue

        print(f"Sub-panel shape ({g}): {sub.shape}")

        # Drop rows with missing load (should be rare: early days, etc.)
        sub = sub.dropna(subset=["all_minutes_last_7d"]).copy()

        # True per-day event rate in this group
        true_prev = sub["injury_event"].mean()
        print(f"True daily injury event rate ({g}): {true_prev:.6f}")

        # Bin the 7-day load
        sub = add_45min_load_bins(sub)

        # ------------------------------------------------------------------
        # 1) Crude per-day injury rates by 7-day load
        # ------------------------------------------------------------------
        daily_rates = (
            sub
            .groupby("all_minutes7d_bin", dropna=False, observed=False)
            .agg(
                n_days=("injury_event", "size"),
                n_events=("injury_event", "sum"),
            )
            .assign(
                injury_rate=lambda d: d["n_events"] / d["n_days"]
            )
        )

        print("\nCrude per-day injury rate by 7-day load bin (45-min bins):")
        print(daily_rates)

        # ------------------------------------------------------------------
        # 2) Crude per-minute injury rates on match days
        # ------------------------------------------------------------------
        sub_match = sub[sub[CURRENT_MINUTES_COL] > 0].copy()

        per_minute = (
            sub_match
            .groupby("all_minutes7d_bin", dropna=False, observed=False)
            .agg(
                total_events=("injury_event", "sum"),
                total_minutes=(CURRENT_MINUTES_COL, "sum"),
            )
            .assign(
                events_per_minute=lambda d: d["total_events"] / d["total_minutes"],
                events_per_10000_min=lambda d: d["events_per_minute"] * 10000,
            )
        )

        print("\nCrude match-day injury rate per minute by 7-day load bin (45-min bins):")
        print(per_minute)

        # Save crude summaries
        daily_out = out_dir / f"hazard_by_fragility_{g}_daily_45min.csv"
        permin_out = out_dir / f"hazard_by_fragility_{g}_per_minute_45min.csv"
        daily_rates.to_csv(daily_out)
        per_minute.to_csv(permin_out)

        # ------------------------------------------------------------------
        # 3) GLM: only if enough events in this group
        # ------------------------------------------------------------------
        n_events = int(sub["injury_event"].sum())
        print(f"\nTotal injury events in group {g}: {n_events}")

        if n_events < MIN_EVENTS_FOR_GLM:
            print(
                f"Too few events (< {MIN_EVENTS_FOR_GLM}) for stable GLM in group {g}; "
                "skipping model fit."
            )
            continue

        model_labels, glm_bin_counts = estimable_bin_labels(sub)
        estimability_out = out_dir / f"hazard_by_fragility_{g}_glm_estimable_bins_45min.csv"
        glm_bin_counts.to_csv(estimability_out, index=False)

        if "0-45" not in model_labels or len(model_labels) < 2:
            print(
                "Not enough estimable burden bins with reference '0-45'; "
                f"skipping GLM for group {g}. Bin table saved to {estimability_out}"
            )
            continue

        excluded_bins = glm_bin_counts[~glm_bin_counts["estimable"]]
        if not excluded_bins.empty:
            print("\nBins excluded from GLM because they are sparse or separated:")
            print(excluded_bins)

        glm_panel = sub[sub["all_minutes7d_bin"].isin(model_labels)].copy()
        glm_panel["all_minutes7d_bin"] = pd.Categorical(
            glm_panel["all_minutes7d_bin"].astype(str),
            categories=model_labels,
            ordered=True,
        )

        # Drop any rows with missing bin or covariates
        glm_data = glm_panel.dropna(
            subset=["all_minutes7d_bin", RECENT_GAMES_COL]
        ).copy()

        print(f"Full-panel GLM shape ({g}) after dropping NAs:", glm_data.shape)
        print(f"Model event rate ({g}): {glm_data['injury_event'].mean():.6f}")

        # Typical covariates for predictions
        typ_minutes_played = sub.loc[sub[CURRENT_GAMES_COL] == 1, CURRENT_MINUTES_COL].median()
        typ_games_last_7d = 1

        print(f"Typical {CURRENT_MINUTES_COL} on match days ({g}): {typ_minutes_played}")
        print(f"Using {RECENT_GAMES_COL} = {typ_games_last_7d} for predictions.")

        # Fit GLM with cluster-robust SEs – crucial: groups from glm_data
        formula = (
            "injury_event ~ C(all_minutes7d_bin, Treatment('0-45')) "
            f"+ {RECENT_GAMES_COL}"
        )

        try:
            glm_binom = smf.glm(
                formula=formula,
                data=glm_data,
                family=sm.families.Binomial()
            )

            res_glm = glm_binom.fit(
                cov_type="cluster",
                cov_kwds={"groups": glm_data["tm_player_id"]}
            )
        except Exception as e:
            print(f"GLM failed for group {g}: {e}")
            continue

        print("\nModel fit summary (coeff table):")
        print(res_glm.summary().tables[1])

        # Odds ratios for the load bins
        params = res_glm.params
        conf_int = res_glm.conf_int()

        or_table = pd.DataFrame({
            "coef": params,
            "OR": np.exp(params),
            "CI_lower": np.exp(conf_int[0]),
            "CI_upper": np.exp(conf_int[1]),
        })

        or_bins = or_table[or_table.index.str.contains("all_minutes7d_bin")]
        print(f"\nOdds ratios for 7-day load bins (45-min bins, vs 0-45) in group {g}:")
        print(or_bins)

        # Predictions by bin
        pred_df = pd.DataFrame({
            "all_minutes7d_bin": pd.Categorical(
                model_labels,
                categories=model_labels,
                ordered=True,
            ),
            "all_minutes_last_7d": [REP_VALUES_45[label] for label in model_labels],
            CURRENT_MINUTES_COL: typ_minutes_played,
            RECENT_GAMES_COL: typ_games_last_7d,
        })

        pred_df["all_minutes7d_bin"] = pd.Categorical(
            pred_df["all_minutes7d_bin"],
            categories=model_labels,
            ordered=True,
        )

        pred_prob_true = res_glm.predict(pred_df)

        pred_df["pred_prob_true"] = pred_prob_true
        pred_df["risk_per_minute"] = pred_df["pred_prob_true"] / pred_df[CURRENT_MINUTES_COL]
        pred_df["events_per_10000_min"] = pred_df["risk_per_minute"] * 10000

        print(
            f"\nPredicted daily and per-minute injury risk by 7-day load bin "
            f"({g} players, 45-min bins):"
        )
        print(pred_df[[
            "all_minutes7d_bin",
            "all_minutes_last_7d",
            "pred_prob_true",
            "events_per_10000_min",
        ]])

        # Save GLM outputs
        or_out = out_dir / f"hazard_by_fragility_{g}_glm_or_45min.csv"
        pred_out = out_dir / f"hazard_by_fragility_{g}_glm_predicted_45min.csv"
        or_bins.to_csv(or_out)
        pred_df.to_csv(pred_out, index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
