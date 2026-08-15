#!/usr/bin/env python
"""
08_baseline_hazard_all_comp.py

Baseline discrete-time hazard model using all-competition club load, with
a minimum-exposure restriction on players:

- We EXCLUDE players with very low total all-competition minutes, because their
  injury rates are unstable and they add lots of "days at risk" with
  essentially no exposure.

- Outcome: injury_event (1 if new injury spell starts on that player-day).
- Main exposure: all_minutes_last_7d (total minutes in ALL competitions
  in the previous 7 days, excluding today), binned.
- Controls: all_games_last_7d. Current-day minutes are intentionally not used
  as a covariate because same-day injury starts can shorten realised minutes.

Steps:
  1. Load data/processed/player_day_panel_all_comp.csv
  2. Compute total all-competition minutes per tm_player_id and RESTRICT to players
     with total_minutes_played >= MIN_MINUTES_FOR_ANALYSIS (default 900).
  3. Compute crude injury rates by all_minutes_last_7d bins (45-min bins)
     both per player-day and per minute on match days.
  4. Screen 45-minute bins before GLM fitting so sparse/separated tail bins
     remain in crude tables but do not create unstable model coefficients.
  5. Fit a full-panel logistic regression with cluster-robust SEs by
     tm_player_id.
  6. Print odds ratios for the bins and predicted daily injury probabilities
     by load bin, plus model-based per-minute rates for a typical match-day
     denominator.

Run from repo root:

    python src/08_baseline_hazard_all_comp.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pipeline_io import LABELS_45, REP_VALUES_45, add_45min_load_bins, estimable_bin_labels
from pipeline_io import restrict_to_available_risk_set


CURRENT_MINUTES_COL = "all_minutes_played"
CURRENT_GAMES_COL = "all_games_played"
RECENT_GAMES_COL = "all_games_last_7d"

# Minimum total all-competition minutes for a player to be included in the analysis
# (10 full matches = 900 minutes; adjust if desired)
MIN_MINUTES_FOR_ANALYSIS = 900.0


def main() -> None:  # pragma: no cover
    # Paths
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"
    panel_path = data_dir / "player_day_panel_all_comp.csv"

    print(f"Repo root: {root}")
    print(f"Loading enriched player-day panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)

    print("Panel shape (all players):", panel.shape)
    total_events_before = int((panel["injury_event"] == 1).sum())
    panel = restrict_to_available_risk_set(panel)
    print("Panel shape after availability restriction:", panel.shape)

    # -------------------------------------------------------------------------
    # 0. Restrict to players with sufficient exposure (total minutes)
    # -------------------------------------------------------------------------
    player_minutes = (
        panel
        .groupby("tm_player_id")[CURRENT_MINUTES_COL]
        .sum()
        .rename("total_minutes_played")
    )

    n_players_total = player_minutes.shape[0]
    print(f"\nTotal tm_player_id in panel: {n_players_total}")

    keep_players = player_minutes[player_minutes >= MIN_MINUTES_FOR_ANALYSIS].index
    n_players_keep = keep_players.shape[0]
    print(f"Players with total_minutes_played >= {MIN_MINUTES_FOR_ANALYSIS}: {n_players_keep}")

    # Filter panel to those players
    panel = panel[panel["tm_player_id"].isin(keep_players)].copy()
    print("Panel shape after exposure restriction:", panel.shape)

    # Check how many injury events we keep
    total_events_after = (panel["injury_event"] == 1).sum()
    print(f"Total injury events before restriction: {total_events_before}")
    print(f"Total injury events after  restriction: {total_events_after}")

    # Drop any rows with missing all_minutes_last_7d (should be none after our build)
    panel = panel.dropna(subset=["all_minutes_last_7d"]).copy()

    # Basic sanity: injury_event value counts and true prevalence
    print("\nInjury_event value counts (restricted cohort):")
    print(panel["injury_event"].value_counts(dropna=False))

    true_prev = panel["injury_event"].mean()
    print(f"True daily injury event rate (restricted): {true_prev:.6f} (events per player-day)")

    # -------------------------------------------------------------------------
    # 1. Crude injury rates by ALL-competitions minutes_last_7d bins (45-min bins)
    # -------------------------------------------------------------------------
    panel = add_45min_load_bins(panel)

    print("\nBin counts for all_minutes_last_7d (45-min bins, restricted cohort):")
    print(panel["all_minutes7d_bin"].value_counts(dropna=False))

    injury_by_bin = (
        panel
        .groupby("all_minutes7d_bin", dropna=False, observed=False)
        .agg(
            n_days=("injury_event", "size"),
            n_events=("injury_event", "sum"),
        )
        .assign(
            injury_rate=lambda d: d["n_events"] / d["n_days"]
        )
    )

    print("\nCrude injury event rate by ALL-minutes_last_7d bins (45-min bins):")
    print(injury_by_bin)

    # --- Crude per-minute injury rates on match days -------------------------
    panel_match = panel[panel[CURRENT_MINUTES_COL] > 0].copy()

    per_minute = (
        panel_match
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

    print("\nCrude match-day injury rate per minute by 7-day load bin (restricted cohort):")
    print(per_minute)

    # -------------------------------------------------------------------------
    # 2. Fit full-panel logistic GLM with cluster-robust SEs
    # -------------------------------------------------------------------------
    # Typical covariate values for prediction later
    typ_minutes_played = panel.loc[panel[CURRENT_GAMES_COL] == 1, CURRENT_MINUTES_COL].median()
    typ_games_last_7d = 1  # common value

    print(f"\nTypical {CURRENT_MINUTES_COL} on match days (for predictions): {typ_minutes_played}")
    print(f"Using {RECENT_GAMES_COL} = {typ_games_last_7d} for predictions.\n")

    model_labels, glm_bin_counts = estimable_bin_labels(panel)
    if "0-45" not in model_labels:
        raise RuntimeError("Reference bin '0-45' is not estimable; cannot fit GLM.")

    excluded_bins = glm_bin_counts[~glm_bin_counts["estimable"]]
    if not excluded_bins.empty:
        print("\nBins excluded from GLM because they are sparse or separated:")
        print(excluded_bins)

    glm_panel = panel[panel["all_minutes7d_bin"].isin(model_labels)].copy()
    glm_panel["all_minutes7d_bin"] = pd.Categorical(
        glm_panel["all_minutes7d_bin"].astype(str),
        categories=model_labels,
        ordered=True,
    )
    print("Full-panel GLM shape after estimable-bin restriction:", glm_panel.shape)

    formula = (
        "injury_event ~ C(all_minutes7d_bin, Treatment('0-45')) "
        f"+ {RECENT_GAMES_COL}"
    )

    print("Fitting logistic GLM (Binomial) on full risk panel...")
    glm_binom = smf.glm(
        formula=formula,
        data=glm_panel,
        family=sm.families.Binomial()
    )

    res_glm = glm_binom.fit(
        cov_type="cluster",
        cov_kwds={"groups": glm_panel["tm_player_id"]}
    )

    print("\nModel fit summary (truncated):")
    print(res_glm.summary().tables[1])

    # -------------------------------------------------------------------------
    # 4. Odds ratios for load bins
    # -------------------------------------------------------------------------
    params = res_glm.params
    conf_int = res_glm.conf_int()

    or_table = pd.DataFrame({
        "coef": params,
        "OR": np.exp(params),
        "CI_lower": np.exp(conf_int[0]),
        "CI_upper": np.exp(conf_int[1]),
    })

    or_bins = or_table[or_table.index.str.contains("all_minutes7d_bin")]
    print("\nOdds ratios for ALL-minutes_last_7d bins (45-min bins, vs 0-45 minutes):")
    print(or_bins)

    # -------------------------------------------------------------------------
    # 4. Predicted daily injury probabilities by load bin
    #    + model-based per-minute rates using a typical match-day denominator
    # -------------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "all_minutes7d_bin": pd.Categorical(
            model_labels,
            categories=model_labels,
            ordered=True
        ),
        "all_minutes_last_7d": [REP_VALUES_45[label] for label in model_labels],
        CURRENT_MINUTES_COL: typ_minutes_played,
        RECENT_GAMES_COL: typ_games_last_7d,
    })

    pred_df["all_minutes7d_bin"] = pd.Categorical(
        pred_df["all_minutes7d_bin"],
        categories=model_labels,
        ordered=True
    )

    pred_prob_true = res_glm.predict(pred_df)

    pred_df["pred_prob_true"] = pred_prob_true

    pred_df["risk_per_minute"] = pred_df["pred_prob_true"] / pred_df[CURRENT_MINUTES_COL]
    pred_df["events_per_10000_min"] = pred_df["risk_per_minute"] * 10000
    model_audit = pd.DataFrame(
        [
            {
                "model": "overall_daily_logistic_45min",
                "history_stratum": "all_prior_history",
                "n_full_risk_rows": int(len(panel)),
                "n_model_rows": int(len(glm_panel)),
                "n_full_events": int(panel["injury_event"].sum()),
                "n_model_events": int(glm_panel["injury_event"].sum()),
                "non_event_sample_fraction": 1.0,
                "true_event_rate": float(true_prev),
                "model_event_rate": float(glm_panel["injury_event"].mean()),
                "n_prediction_rows": int(len(pred_df)),
                "mean_pred_prob_true": float(np.mean(pred_prob_true)),
            }
        ]
    )

    print("\nPredicted daily injury probabilities and per-minute rates "
          "by ALL-minutes_last_7d bin (45-min bins, restricted cohort)")
    print("(full-panel predictions; events_per_10000_min is model-based per-minute "
          "rate on a typical match-day denominator):")
    print(pred_df[["all_minutes7d_bin", "all_minutes_last_7d",
                   "pred_prob_true", "events_per_10000_min"]])

    # Save outputs
    out_dir = data_dir / "results"
    out_dir.mkdir(exist_ok=True)

    injury_by_bin.to_csv(out_dir / "baseline_injury_rate_by_all_minutes7d_bin_45min.csv")
    per_minute.to_csv(out_dir / "baseline_injury_rate_per_minute_by_all_minutes7d_bin_45min.csv")
    or_bins.to_csv(out_dir / "glm_or_all_minutes7d_bins_45min.csv")
    pred_df.to_csv(out_dir / "glm_predicted_probs_all_minutes7d_bins_45min.csv", index=False)
    glm_bin_counts.to_csv(out_dir / "glm_estimable_bins_all_minutes7d_45min.csv", index=False)
    model_audit.to_csv(out_dir / "glm_model_audit_all_minutes7d_45min.csv", index=False)

    print(f"\nSaved:")
    print(f"  - crude per-day rates  -> {out_dir / 'baseline_injury_rate_by_all_minutes7d_bin_45min.csv'}")
    print(f"  - crude per-minute     -> {out_dir / 'baseline_injury_rate_per_minute_by_all_minutes7d_bin_45min.csv'}")
    print(f"  - ORs                  -> {out_dir / 'glm_or_all_minutes7d_bins_45min.csv'}")
    print(f"  - model preds (per day & per minute) -> "
          f"{out_dir / 'glm_predicted_probs_all_minutes7d_bins_45min.csv'}")
    print(f"  - GLM bin estimability -> {out_dir / 'glm_estimable_bins_all_minutes7d_45min.csv'}")
    print(f"  - model audit          -> {out_dir / 'glm_model_audit_all_minutes7d_45min.csv'}")


if __name__ == "__main__":  # pragma: no cover
    main()
