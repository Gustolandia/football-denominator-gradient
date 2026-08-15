#!/usr/bin/env python
"""
14_hazard_by_fragility_45min_weekly.py

Prior-history-stratified daily hazard models with short-term load controls.

This script uses day-level prior-history fragility labels from
player_day_fragility.csv, keeps the dynamic risk set
{tough, regular, fragile}, and excludes low_exposure days. Crude summaries keep
all 45-minute burden bins, while GLMs are fit only on bins with enough days and
at least one event to avoid separation-driven odds ratios.

Model features:
- full-panel logistic GLMs within each prior-history stratum
- 45-minute bins of all_minutes_last_7d
- recent-load and calendar-timing covariates from src/13_max_daily_load_features.py
- clustered SEs by stable tm_player_id
- per-minute display rates computed as pred_prob_true / typical match-day minutes * 10,000

Inputs:
    data/processed/player_day_panel_all_comp.csv
    data/processed/player_day_fragility.csv

Outputs (per group, under data/processed/results/):
    hazard_45min_{group}_perday_weekly.csv
    hazard_45min_{group}_perminute_weekly.csv
    glm_or_45min_{group}_weekly_bins.csv
    glm_or_45min_{group}_weekly_loadvars.csv
    glm_predicted_probs_45min_{group}_weekly.csv

Run from repo root:

    python src/14_hazard_by_fragility_45min_weekly.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pipeline_io import (
    LABELS_45,
    REP_VALUES_45,
    add_45min_load_bins,
    estimable_bin_labels,
    merge_day_fragility,
    restrict_to_available_risk_set,
    restrict_to_fragility_risk_set,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_EVENTS_FOR_GLM = 200
CURRENT_MINUTES_COL = "all_minutes_played"
CURRENT_GAMES_COL = "all_games_played"
RECENT_GAMES_COL = "all_games_last_7d"

# Columns that MUST exist in the panel (from script 13)
REQUIRED_LOAD_COLS = [
    CURRENT_MINUTES_COL,
    CURRENT_GAMES_COL,
    RECENT_GAMES_COL,
    "minutes_yesterday",
    "minutes_last_match",
    "max_daily_minutes_last_7d",
    "any_day_last7_over_90",
    "any_day_last7_full_match",
    "week_phase_sin",
    "week_phase_cos",
    "halfweek_phase_sin",
    "halfweek_phase_cos",
]


def main() -> None:  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    proc_dir = root / "data" / "processed"
    results_dir = proc_dir / "results"
    results_dir.mkdir(exist_ok=True)

    panel_path = proc_dir / "player_day_panel_all_comp.csv"

    print(f"Repo root: {root}")
    print(f"Loading player-day panel from {panel_path} ...")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    print("Panel shape (all players):", panel.shape)

    # Check that load features from script 13 exist
    missing = [c for c in REQUIRED_LOAD_COLS if c not in panel.columns]
    if missing:
        raise RuntimeError(
            f"Panel is missing required load / weekly columns {missing}. "
            "Run 13_max_daily_load_features.py first."
        )

    print("Merging prior-history day-level fragility labels ...")
    panel = merge_day_fragility(panel, proc_dir)
    panel = restrict_to_fragility_risk_set(panel)
    panel = restrict_to_available_risk_set(panel)

    print("\nFragility_group value counts in restricted panel:")
    print(panel["fragility_group"].value_counts())

    # 45-min bins for ALL competitions all_minutes_last_7d
    panel = add_45min_load_bins(panel)

    groups = ["tough", "regular", "fragile"]

    model_audit_rows = []

    for group in groups:
        print("\n" + "=" * 16 + f" {group.upper()} PLAYERS " + "=" * 16)
        sub = panel[panel["fragility_group"] == group].copy()
        print(f"Sub-panel shape ({group}): {sub.shape}")

        true_prev = sub["injury_event"].mean()
        print(
            f"True daily injury event rate ({group}): "
            f"{true_prev:.6f} (events per player-day)"
        )

        # ------------------------------------------------------------------
        # Crude per-day rates by 45-min bins
        # ------------------------------------------------------------------
        injury_by_bin = (
            sub.groupby("all_minutes7d_bin", dropna=False, observed=False)
            .agg(
                n_days=("injury_event", "size"),
                n_events=("injury_event", "sum"),
            )
            .assign(injury_rate=lambda d: d["n_events"] / d["n_days"])
            .reset_index()
        )

        print("Crude per-day injury rate by 7-day load bin (45-min bins):")
        print(injury_by_bin)

        # Match-day per-minute rates
        sub_match = sub[sub[CURRENT_MINUTES_COL] > 0].copy()
        per_minute = (
            sub_match.groupby("all_minutes7d_bin", dropna=False, observed=False)
            .agg(
                total_events=("injury_event", "sum"),
                total_minutes=(CURRENT_MINUTES_COL, "sum"),
            )
            .assign(
                events_per_minute=lambda d: d["total_events"] / d["total_minutes"],
                events_per_10000_min=lambda d: d["events_per_minute"] * 10000,
            )
            .reset_index()
        )

        print(
            "Crude match-day per-minute injury rate by 7-day load bin (45-min bins):"
        )
        print(per_minute)

        # Save crude tables
        injury_by_bin.to_csv(
            results_dir / f"hazard_45min_{group}_perday_weekly.csv", index=False
        )
        per_minute.to_csv(
            results_dir / f"hazard_45min_{group}_perminute_weekly.csv",
            index=False,
        )

        # ------------------------------------------------------------------
        # GLM: full-panel model with weekly terms & load features
        # ------------------------------------------------------------------
        total_events = int(sub["injury_event"].sum())
        print(f"Total injury events in group {group}: {total_events}")
        if total_events < MIN_EVENTS_FOR_GLM:
            print(
                f"Too few events (< {MIN_EVENTS_FOR_GLM}) for stable GLM in "
                f"group {group}; skipping model fit."
            )
            continue

        model_labels, glm_bin_counts = estimable_bin_labels(sub)
        estimability_out = results_dir / f"glm_estimable_bins_45min_{group}_weekly.csv"
        glm_bin_counts.to_csv(estimability_out, index=False)
        if "0-45" not in model_labels or len(model_labels) < 2:
            print(
                "Not enough estimable burden bins with reference '0-45'; "
                f"skipping GLM for group {group}. Bin table saved to {estimability_out}"
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

        # Fill any residual NaNs in the new features with zeros
        for col in REQUIRED_LOAD_COLS:
            glm_panel[col] = glm_panel[col].fillna(0.0)

        print(
            f"Full-panel GLM shape ({group}) after estimable-bin restriction:"
            f" {glm_panel.shape}"
        )
        print(
            f"Model event rate ({group}): {glm_panel['injury_event'].mean():.6f}"
        )

        # Typical covariates for prediction
        typ_minutes_played = sub.loc[sub[CURRENT_GAMES_COL] == 1, CURRENT_MINUTES_COL].median()
        typ_games_last_7d = 1
        typ_minutes_last_match = sub.loc[
            sub["minutes_last_match"] > 0.0, "minutes_last_match"
        ].median()
        if pd.isna(typ_minutes_last_match):
            typ_minutes_last_match = typ_minutes_played

        print(
            f"Typical {CURRENT_MINUTES_COL} on match days ({group}): {typ_minutes_played}"
        )
        print(f"Typical minutes_last_match ({group}): {typ_minutes_last_match}")
        print(f"Using {RECENT_GAMES_COL} = {typ_games_last_7d} for predictions.")
        print("For predictions we set:")
        print("  minutes_yesterday = 0 (no match yesterday)")
        print("  max_daily_minutes_last_7d = 90")
        print("  any_day_last7_over_90 = 1  (represents full-match exposure >=85 mins)")
        print("  weekly sin/cos terms = 0 (average over the cycle)\n")

        # ------------------------------------------------------------------
        # Build formula dynamically, dropping constant columns and
        # removing exact-alias columns.
        # ------------------------------------------------------------------
        bin_term = "C(all_minutes7d_bin, Treatment('0-45'))"

        # IMPORTANT: do NOT include both full-match flags in model
        # (they are identical aliases). Also do not include current-day minutes
        # or algebraic extra-time transforms of all_minutes_last_7d.
        candidate_continuous = [
            RECENT_GAMES_COL,
            "minutes_yesterday",
            "minutes_last_match",
            "max_daily_minutes_last_7d",
            "any_day_last7_over_90",
            # "any_day_last7_full_match",  # alias → excluded
            "week_phase_sin",
            "week_phase_cos",
            "halfweek_phase_sin",
            "halfweek_phase_cos",
        ]

        varying_terms = [
            c
            for c in candidate_continuous
            if glm_panel[c].nunique(dropna=False) > 1
        ]

        if len(varying_terms) < len(candidate_continuous):
            dropped = sorted(set(candidate_continuous) - set(varying_terms))
            print(
                f"Columns constant within group {group}; "
                f"excluding from GLM: {dropped}"
            )

        formula = "injury_event ~ " + bin_term
        if varying_terms:
            formula += " + " + " + ".join(varying_terms)

        print("Fitting logistic GLM (Binomial) for this group...")

        # 1) Initial fit just to discover which rows are actually used
        glm_binom = smf.glm(
            formula=formula,
            data=glm_panel,
            family=sm.families.Binomial(),
        )
        base_res = glm_binom.fit()

        # Rows used by the formula (after dropping NAs internally)
        used_index = base_res.model.data.row_labels

        # Align cluster groups with those rows only
        groups_used = glm_panel.loc[used_index, "tm_player_id"]

        # 2) Refit on the reduced dataset with cluster-robust SEs
        glm_used = glm_panel.loc[used_index].copy()

        glm_binom_cluster = smf.glm(
            formula=formula,
            data=glm_used,
            family=sm.families.Binomial(),
        )

        res_glm = glm_binom_cluster.fit(
            cov_type="cluster",
            cov_kwds={"groups": groups_used},
        )

        print("\nModel fit summary (coeff table):")
        print(res_glm.summary().tables[1])

        # ------------------------------------------------------------------
        # Odds ratios: bins + key load variables
        # ------------------------------------------------------------------
        params = res_glm.params
        conf_int = res_glm.conf_int()

        or_table = pd.DataFrame(
            {
                "coef": params,
                "OR": np.exp(params),
                "CI_lower": np.exp(conf_int[0]),
                "CI_upper": np.exp(conf_int[1]),
            }
        )

        or_bins = or_table[or_table.index.str.contains("all_minutes7d_bin")].copy()
        print(
            "\nOdds ratios for 7-day load bins "
            f"(45-min bins, vs 0-45) in group {group}:"
        )
        print(or_bins)

        loadvar_names = [
            "minutes_yesterday",
            "minutes_last_match",
            "max_daily_minutes_last_7d",
            "any_day_last7_over_90",
            RECENT_GAMES_COL,
        ]
        present_loadvars = [name for name in loadvar_names if name in or_table.index]
        or_loadvars = or_table.loc[present_loadvars].copy()

        print("\nOdds ratios for key load covariates:")
        print(or_loadvars)

        or_bins.to_csv(results_dir / f"glm_or_45min_{group}_weekly_bins.csv")
        or_loadvars.to_csv(results_dir / f"glm_or_45min_{group}_weekly_loadvars.csv")

        # ------------------------------------------------------------------
        # Predictions by 45-min bin
        # ------------------------------------------------------------------
        pred_df = pd.DataFrame(
            {
                "all_minutes7d_bin": pd.Categorical(
                    model_labels, categories=model_labels, ordered=True
                ),
                "all_minutes_last_7d": [REP_VALUES_45[label] for label in model_labels],
                CURRENT_MINUTES_COL: typ_minutes_played,
                RECENT_GAMES_COL: typ_games_last_7d,
                "minutes_yesterday": 0.0,
                "minutes_last_match": typ_minutes_last_match,
                "max_daily_minutes_last_7d": 90.0,
                "any_day_last7_over_90": 1,
                "week_phase_sin": 0.0,
                "week_phase_cos": 0.0,
                "halfweek_phase_sin": 0.0,
                "halfweek_phase_cos": 0.0,
            }
        )

        pred_prob_true = res_glm.predict(pred_df)

        pred_df["pred_prob_true"] = pred_prob_true

        # Per-minute risk on a typical match day
        pred_df["risk_per_minute"] = pred_df["pred_prob_true"] / pred_df[CURRENT_MINUTES_COL]
        pred_df["events_per_10000_min"] = pred_df["risk_per_minute"] * 10000.0

        print(
            f"\nPredicted daily and per-minute injury risk by 7-day load bin "
            f"({group} players, 45-min bins):"
        )
        print(
            pred_df[
                [
                    "all_minutes7d_bin",
                    "all_minutes_last_7d",
                    "pred_prob_true",
                    "events_per_10000_min",
                ]
            ]
        )

        pred_df.to_csv(
            results_dir / f"glm_predicted_probs_45min_{group}_weekly.csv",
            index=False,
        )
        model_audit_rows.append(
            {
                "model": f"{group}_daily_logistic_45min_weekly",
                "history_stratum": group,
                "n_full_risk_rows": int(len(sub)),
                "n_model_rows": int(len(glm_used)),
                "n_full_events": int(sub["injury_event"].sum()),
                "n_model_events": int(glm_used["injury_event"].sum()),
                "non_event_sample_fraction": 1.0,
                "true_event_rate": float(true_prev),
                "model_event_rate": float(glm_used["injury_event"].mean()),
                "n_prediction_rows": int(len(pred_df)),
                "mean_pred_prob_true": float(np.mean(pred_prob_true)),
            }
        )

    if model_audit_rows:
        model_audit = pd.DataFrame(model_audit_rows)
        model_audit.to_csv(
            results_dir / "glm_model_audit_45min_weekly.csv",
            index=False,
        )
        print(
            "\nSaved model audit -> "
            f"{results_dir / 'glm_model_audit_45min_weekly.csv'}"
        )
    print("\nAll groups processed.")


if __name__ == "__main__":  # pragma: no cover
    main()
