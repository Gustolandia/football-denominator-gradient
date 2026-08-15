#!/usr/bin/env python
"""
20_model_diagnostics.py

Residual, Q-Q, and influence diagnostics for:

(1) Daily logistic hazard model on the baseline analysis-window cohort
(2) Match-proxy Poisson model with a 4 df burden spline and observed-minute
    offset

This script DOES NOT modify any processed files. It only reads the
final panel, refits the key models, computes residuals, and exports
diagnostic figures and CSVs.

The logistic diagnostic uses the full eligible panel and the same estimable-bin
rule as the corrected 45-minute GLMs. The Poisson diagnostic mirrors
src/18_match_proxy_poisson_splines_perminute.py by excluding tough players
unless that model is intentionally changed.

Run:
    python src/20_model_diagnostics.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pipeline_io import (
    add_45min_load_bins,
    estimable_bin_labels,
    merge_day_fragility,
    restrict_to_available_risk_set,
    restrict_to_fragility_risk_set,
)

# =====================================================================
# Configuration
# =====================================================================

PLAYER_ID = "tm_player_id"
CURRENT_MINUTES_COL = "all_minutes_played"
RECENT_GAMES_COL = "all_games_last_7d"
MIN_MINUTES_FOR_ANALYSIS = 900.0

SPLINE_DF = 4

# =====================================================================
# Utility functions
# =====================================================================

def qq_plot(residuals, title, outpath):  # pragma: no cover
    """Deviance or Pearson residual Q–Q plot."""
    fig = plt.figure(figsize=(6, 6))
    sm.qqplot(residuals, line="45", ax=plt.gca())
    plt.title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Q-Q plot -> {outpath}")

def residual_vs_fitted_plot(residuals, fitted, title, outpath):  # pragma: no cover
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(fitted, residuals, s=4, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residual")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved residual vs fitted -> {outpath}")

def histogram(residuals, title, outpath):  # pragma: no cover
    fig = plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=60, alpha=0.75)
    plt.title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved residual histogram -> {outpath}")


def diagnostic_panel(residuals, fitted, title, outpath):  # pragma: no cover
    """Three-panel diagnostic figure for manuscript use."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    sm.qqplot(residuals, line="45", ax=axes[0])
    axes[0].set_title("Q-Q")
    axes[1].scatter(fitted, residuals, s=4, alpha=0.3)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.0)
    axes[1].set_title("Residual vs fitted")
    axes[1].set_xlabel("Fitted")
    axes[1].set_ylabel("Deviance residual")
    axes[2].hist(residuals, bins=60, alpha=0.75)
    axes[2].set_title("Residual histogram")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved diagnostic panel -> {outpath}")


def restrict_to_daily_analysis_window(
    panel: pd.DataFrame,
    min_minutes: float = MIN_MINUTES_FOR_ANALYSIS,
) -> pd.DataFrame:
    """Mirror the baseline daily model's availability and total-minute screen."""
    required = {PLAYER_ID, CURRENT_MINUTES_COL}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing required columns: {sorted(missing)}")

    available = restrict_to_available_risk_set(panel)
    player_minutes = available.groupby(PLAYER_ID)[CURRENT_MINUTES_COL].sum()
    keep_players = player_minutes[player_minutes >= min_minutes].index
    return available[available[PLAYER_ID].isin(keep_players)].copy()


# =====================================================================
# Main
# =====================================================================

def main():  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"
    results = proc / "results"
    diag_dir = results / "diagnostics"
    fig_dir = results / "figures"
    # Row-level residual frames are one row per player-day and carry the
    # provider's player identifier beside an injury description. They verify no
    # reported number -- nothing in this repository reads them back -- but both
    # deposit builders export everything under data/processed/results, so
    # writing them there published special-category data about identifiable
    # people as a side effect of a diagnostic. They are written outside every
    # exported subtree instead, and stay available to the authors.
    private_diag_dir = proc / "diagnostics_private"
    diag_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    private_diag_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------------------
    # Load panel
    # -----------------------------------------------------------------
    panel_path = proc / "player_day_panel_all_comp.csv"
    print(f"Loading: {panel_path}")
    panel = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)

    panel = merge_day_fragility(panel, proc)
    daily_panel = restrict_to_daily_analysis_window(panel)
    poisson_panel = restrict_to_available_risk_set(
        restrict_to_fragility_risk_set(panel)
    )

    # =================================================================
    # (1) DAILY LOGISTIC HAZARD DIAGNOSTICS
    # =================================================================
    print("Refitting daily logistic hazard model for diagnostics...")
    daily_panel = add_45min_load_bins(daily_panel)
    model_labels, glm_bin_counts = estimable_bin_labels(daily_panel)
    if "0-45" not in model_labels:
        raise RuntimeError("Reference bin '0-45' is not estimable; cannot fit diagnostics.")
    glm_bin_counts.to_csv(diag_dir / "logit_estimable_bins.csv", index=False)

    logit_frame = daily_panel[daily_panel["all_minutes7d_bin"].isin(model_labels)].copy()
    logit_frame["all_minutes7d_bin"] = pd.Categorical(
        logit_frame["all_minutes7d_bin"].astype(str),
        categories=model_labels,
        ordered=True,
    )

    print(f"  Full-panel shape after estimable-bin restriction: {logit_frame.shape}")
    print(f"  Event rate in model frame: {logit_frame['injury_event'].mean():.5f}")

    logit_formula = (
        "injury_event ~ C(all_minutes7d_bin, Treatment('0-45')) "
        f"+ {RECENT_GAMES_COL}"
    )

    logit_model = smf.glm(
        formula=logit_formula,
        data=logit_frame,
        family=sm.families.Binomial(),
    )
    logit_res = logit_model.fit()

    print("\n[Logit] Key coefficients (first 10):")
    print(logit_res.params.head(10))


    # Fitted values and residuals
    logit_frame["fitted"] = logit_res.predict(logit_frame)
    logit_frame["pearson"] = (logit_frame["injury_event"] - logit_frame["fitted"]) / np.sqrt(
        logit_frame["fitted"] * (1 - logit_frame["fitted"])
    )
    logit_frame["deviance"] = logit_res.resid_deviance

    # Save residuals to CSV
    logit_frame.to_csv(private_diag_dir / "logit_residuals.csv", index=False)
    print(f"[OK] Saved logit residuals -> {private_diag_dir / 'logit_residuals.csv'}")

    # --- Simple influence / outlier scan for logit model ---
    max_idx = logit_frame["deviance"].abs().idxmax()
    max_row = logit_frame.loc[
        max_idx,
        [PLAYER_ID, "date", "all_minutes_last_7d", CURRENT_MINUTES_COL,
         "injury_event", "fitted", "deviance"],
    ]

    print("\n[Logit] Largest absolute deviance residual:")
    print(max_row)

    print(f"\n[Logit] Max |deviance|: {logit_frame['deviance'].abs().max():.3f}")
    print("[Logit] Deviance residual quantiles:")
    print(logit_frame["deviance"].quantile([0.5, 0.9, 0.99, 0.999]))


    # Plots
    qq_plot(
        logit_frame["deviance"],
        "Daily Logit – Deviance Q–Q",
        diag_dir / "logit_deviance_qq.png",
    )
    residual_vs_fitted_plot(
        logit_frame["deviance"],
        logit_frame["fitted"],
        "Daily Logit – Deviance vs Fitted",
        diag_dir / "logit_deviance_vs_fitted.png",
    )
    histogram(
        logit_frame["deviance"],
        "Daily Logit – Deviance Residuals",
        diag_dir / "logit_deviance_hist.png",
    )
    diagnostic_panel(
        logit_frame["deviance"],
        logit_frame["fitted"],
        "Daily logit diagnostics",
        fig_dir / "diag_logit.png",
    )


    # =================================================================
    # (2) MATCH-PROXY POISSON PER-MINUTE DIAGNOSTICS
    # =================================================================
    print("Refitting Poisson per-minute model for diagnostics...")

    mp = poisson_panel[poisson_panel[CURRENT_MINUTES_COL] > 0].copy()
    mp = mp[mp["fragility_group"].isin(["regular", "fragile"])]

    # Clean NA covars used by the corrected primary spline model.
    covars = [
        "all_minutes_last_7d",
        "week_phase_sin",
        "week_phase_cos",
        "halfweek_phase_sin",
        "halfweek_phase_cos",
    ]
    for c in covars:
        mp[c] = mp[c].fillna(0.0)

    # Offset
    mp["log_minutes"] = np.log(mp[CURRENT_MINUTES_COL].clip(lower=1.0))

    burden_min = 0.0
    burden_max = float(mp["all_minutes_last_7d"].max())

    formula_pois = (
        f"injury_event_matchproxy ~ "
        f"bs(all_minutes_last_7d, df={SPLINE_DF}, lower_bound={burden_min}, upper_bound={burden_max})"
        " * fragility_group "
        "+ week_phase_sin + week_phase_cos + halfweek_phase_sin + halfweek_phase_cos"
    )

    print("Refitting Poisson per-minute model for diagnostics...")
    print(f"  Match-panel shape: {mp.shape}")
    print(f"  Match-proxy event rate: {mp['injury_event_matchproxy'].mean():.6f}")

    pois_model = smf.glm(
        formula=formula_pois,
        data=mp,
        family=sm.families.Poisson(),
        offset=mp["log_minutes"],
    )

    pois_res = pois_model.fit(
        cov_type="cluster",
        cov_kwds={"groups": mp[PLAYER_ID]},
    )

    print("\n[Poisson] Key coefficients (first 10):")
    print(pois_res.params.head(10))

    pearson_chi2 = np.sum(pois_res.resid_pearson ** 2)
    dispersion = pearson_chi2 / pois_res.df_resid
    print(f"[Poisson] Pearson dispersion (from diagnostics run): {dispersion:.3f}")


    mp["fitted"] = pois_res.mu
    mp["pearson"] = (mp["injury_event_matchproxy"] - mp["fitted"]) / np.sqrt(mp["fitted"])
    mp["deviance"] = pois_res.resid_deviance


    mp.to_csv(private_diag_dir / "poisson_matchproxy_residuals.csv", index=False)

    # --- Simple influence / outlier scan for Poisson model ---
    max_idx_p = mp["deviance"].abs().idxmax()
    max_row_p = mp.loc[max_idx_p, [PLAYER_ID, "date", "all_minutes_last_7d",
                                   CURRENT_MINUTES_COL, "injury_event_matchproxy",
                                   "fitted", "deviance"]]

    print("\n[Poisson] Largest absolute deviance residual:")
    print(max_row_p)

    print(f"\n[Poisson] Max |deviance|: {mp['deviance'].abs().max():.3f}")
    print("[Poisson] Deviance residual quantiles:")
    print(mp['deviance'].quantile([0.5, 0.9, 0.99, 0.999]))


    # Plots
    qq_plot(mp["deviance"], "Match-Proxy Poisson – Deviance Q–Q", diag_dir / "poisson_deviance_qq.png")
    residual_vs_fitted_plot(mp["deviance"], mp["fitted"], "Match-Proxy Poisson – Deviance vs Fitted", diag_dir / "poisson_deviance_vs_fitted.png")
    histogram(mp["deviance"], "Match-Proxy Poisson – Deviance Residuals", diag_dir / "poisson_deviance_hist.png")
    diagnostic_panel(
        mp["deviance"],
        mp["fitted"],
        "Match-proxy Poisson diagnostics",
        fig_dir / "diag_poisson.png",
    )

    print("\nDiagnostics complete.")
    print(f"Saved diagnostics to: {diag_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
