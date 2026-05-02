#!/usr/bin/env python3
"""Per-bucket ML stock selection: train on all tickers <=2025, predict Q1 2026 (or latest quarter with data)."""

import argparse
import os
import sqlite3
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Allow running as standalone script: python3 src/strategies/ml_bucket_selection.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sector -> Bucket mapping (synced with group_selection_by_gics.py v1.2.2)
# ---------------------------------------------------------------------------
SECTOR_TO_BUCKET = {
    "information technology": "growth_tech",
    "technology": "growth_tech",
    "communication services": "growth_tech",
    "consumer discretionary": "cyclical",
    "consumer cyclical": "cyclical",
    "financials": "cyclical",
    "financial services": "cyclical",
    "industrials": "cyclical",
    "energy": "real_assets",
    "materials": "real_assets",
    "basic materials": "real_assets",
    "real estate": "real_assets",
    "health care": "defensive",
    "healthcare": "defensive",
    "consumer staples": "defensive",
    "consumer defensive": "defensive",
    "utilities": "defensive",
}

FEATURE_COLS = [
    # Valuation (5)
    "pe", "ps", "pb", "peg", "ev_multiple",
    # Profitability (4)
    "EPS", "roe", "gross_margin", "operating_margin",
    # Cash Flow (5)
    "fcf_per_share", "cash_per_share", "capex_per_share", "fcf_to_ocf", "ocf_ratio",
    # Leverage (3)
    "debt_ratio", "debt_to_equity", "debt_to_mktcap",
    # Liquidity (1)
    "cur_ratio",
    # Efficiency (3)
    "acc_rec_turnover", "asset_turnover", "payables_turnover",
    # Coverage (2)
    "interest_coverage", "debt_service_coverage",
    # Dividend (1)
    "dividend_yield",
    # Solvency (1)
    "solvency_ratio",
    # Per-Share (1)
    "BPS",
]

# Momentum features computed from sequential data (not stored in DB)
MOMENTUM_COLS = [
    # Price momentum (from trade_price)
    "ret_1q", "ret_4q", "ret_accel",
    # Fundamental momentum (QoQ changes)
    "eps_chg", "roe_chg", "gm_chg", "om_chg",
]

# datadate → tradedate mapping (first day of next-next month)
DATADATE_TO_TRADEDATE_MAP = {
    "03-31": ("06-01", 0),   # Mar 31 → Jun 1 (same year)
    "06-30": ("09-01", 0),   # Jun 30 → Sep 1 (same year)
    "09-30": ("12-01", 0),   # Sep 30 → Dec 1 (same year)
    "12-31": ("03-01", 1),   # Dec 31 → Mar 1 (next year)
}


def datadate_to_tradedate(datadate_str):
    """Map datadate to tradedate: 03-31→06-01, 06-30→09-01, 09-30→12-01, 12-31→03-01(+1yr)."""
    if not isinstance(datadate_str, str) or len(datadate_str) < 10:
        return None
    mm_dd = datadate_str[5:]  # e.g. "03-31"
    try:
        year = int(datadate_str[:4])
    except ValueError:
        return None
    mapping = DATADATE_TO_TRADEDATE_MAP.get(mm_dd)
    if mapping is None:
        return None
    target_mmdd, year_add = mapping
    return f"{year + year_add}-{target_mmdd}"


def build_models():
    models = {
        "RF": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "XGB": None,
        "LGBM": None,
        "HistGBM": HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),
    }
    try:
        from xgboost import XGBRegressor
        models["XGB"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    except ImportError:
        del models["XGB"]

    try:
        from lightgbm import LGBMRegressor
        models["LGBM"] = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    except ImportError:
        del models["LGBM"]

    return models


def run_bucket(bucket, bdf, feature_cols, val_cutoff="2025-12-31", val_quarters=3):
    """Train models for one bucket, return (predictions_df, model_results_list)."""

    # Validation: last N quarters up to val_cutoff (inclusive)
    all_dates = sorted(bdf[bdf["y_return"].notna()]["datadate"].unique())
    val_end_idx = None
    for i, d in enumerate(all_dates):
        if str(d) <= val_cutoff:
            val_end_idx = i
    if val_end_idx is not None:
        val_start_idx = max(0, val_end_idx - val_quarters + 1)
        val_dates = set(all_dates[val_start_idx : val_end_idx + 1])
    else:
        val_dates = set()

    train_b = bdf[(~bdf["datadate"].isin(val_dates)) & (bdf["datadate"] <= val_cutoff) & (bdf["y_return"].notna())]
    val_b = bdf[(bdf["datadate"].isin(val_dates)) & (bdf["y_return"].notna())]
    # Infer on quarters after val_cutoff
    infer_dates = sorted(bdf[bdf["datadate"] > val_cutoff]["datadate"].unique())
    if infer_dates:
        infer_b = bdf[bdf["datadate"].isin(infer_dates)]
    else:
        infer_b = pd.DataFrame()

    print(f"\n{'=' * 60}")
    print(f"  Bucket: {bucket.upper()}")
    val_date_range = f"{sorted(val_dates)[0]} ~ {sorted(val_dates)[-1]}" if val_dates else "none"
    print(f"  Train: {len(train_b)} | Val: {len(val_b)} ({len(val_dates)}Q: {val_date_range}) | Infer: {len(infer_b)}")
    if len(infer_b) > 0:
        print(f"  Infer dates: {infer_dates} ({len(infer_dates)}Q)")
    print(f"{'=' * 60}")

    if len(train_b) < 20 or len(infer_b) == 0:
        print("  SKIP: insufficient data")
        return pd.DataFrame(), [], []

    X_train, y_train = train_b[feature_cols].values, train_b["y_return"].values
    X_val, y_val = val_b[feature_cols].values, val_b["y_return"].values
    X_infer = infer_b[feature_cols].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(val_b) > 0 else None
    X_infer_s = scaler.transform(X_infer)

    models = build_models()
    fitted = {}
    model_results = []
    best_name, best_mse, best_model = None, float("inf"), None

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        fitted[name] = model
        if X_val_s is not None and len(X_val_s) > 0:
            mse = mean_squared_error(y_val, model.predict(X_val_s))
        else:
            mse = float("inf")
        model_results.append({
            "bucket": bucket, "model": name, "val_mse": round(mse, 6),
            "train_size": len(train_b), "val_size": len(val_b), "infer_size": len(infer_b),
        })
        print(f"  {name:12s}: MSE = {mse:.6f}")
        if mse < best_mse:
            best_name, best_mse, best_model = name, mse, model

    # Stacking top 3
    if X_val_s is not None and len(X_val_s) > 0:
        sorted_m = sorted(
            [(n, mean_squared_error(y_val, fitted[n].predict(X_val_s))) for n in fitted],
            key=lambda x: x[1],
        )
    else:
        sorted_m = [(n, 0) for n in fitted]
    top3 = [n for n, _ in sorted_m[:3]]
    stacking = StackingRegressor(
        estimators=[(n, fitted[n]) for n in top3],
        final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1,
    )
    stacking.fit(X_train_s, y_train)
    fitted["Stacking"] = stacking
    if X_val_s is not None and len(X_val_s) > 0:
        stack_mse = mean_squared_error(y_val, stacking.predict(X_val_s))
    else:
        stack_mse = float("inf")
    model_results.append({
        "bucket": bucket, "model": "Stacking", "val_mse": round(stack_mse, 6),
        "train_size": len(train_b), "val_size": len(val_b), "infer_size": len(infer_b),
    })
    print(f"  {'Stacking':12s}: MSE = {stack_mse:.6f}  (base: {top3})")

    if stack_mse < best_mse:
        best_name, best_mse, best_model = "Stacking", stack_mse, stacking

    print(f"  >> Best: {best_name} (MSE={best_mse:.6f})")

    # Retrain all models on train + val before inference
    full_train = pd.concat([train_b, val_b], ignore_index=True)
    X_full, y_full = full_train[feature_cols].values, full_train["y_return"].values
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_full)
    X_infer_s = scaler_full.transform(X_infer)
    print(f"  Retrained on train+val: {len(full_train)} samples")

    for name, model in fitted.items():
        if name == "Stacking":
            continue  # rebuild stacking below
        model.fit(X_full_s, y_full)
    # Rebuild stacking with retrained base models
    stacking = StackingRegressor(
        estimators=[(n, fitted[n]) for n in top3],
        final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1,
    )
    stacking.fit(X_full_s, y_full)
    fitted["Stacking"] = stacking
    if best_name == "Stacking":
        best_model = stacking

    # Predict
    infer_b = infer_b.copy()
    infer_b["tradedate"] = infer_b["datadate"].apply(datadate_to_tradedate)
    infer_b["predicted_return"] = best_model.predict(X_infer_s)
    infer_b["best_model"] = best_name
    for n, m in fitted.items():
        infer_b[f"pred_{n}"] = m.predict(X_infer_s)

    # Inverse-MSE weighted ensemble (weights from val MSE, predictions from retrained models)
    mse_map = {r["model"]: r["val_mse"] for r in model_results}
    pred_model_cols = [c for c in infer_b.columns if c.startswith("pred_") and c != "pred_ensemble_avg"]
    weights = {}
    for col in pred_model_cols:
        name = col.replace("pred_", "")
        mse = mse_map.get(name, None)
        weights[col] = (1.0 / mse) if mse and mse > 0 else 0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    infer_b["pred_ensemble_avg"] = sum(infer_b[col] * w for col, w in weights.items())

    infer_b = infer_b.sort_values(["datadate", "predicted_return"], ascending=[True, False])

    # Print ranking per quarter
    for idate in infer_dates:
        qdf = infer_b[infer_b["datadate"] == idate].sort_values("predicted_return", ascending=False)
        actual_col = "y_return" if "y_return" in qdf.columns and qdf["y_return"].notna().any() else None
        print(f"\n  Ranking ({idate}, {len(qdf)} stocks):")
        for i, (_, r) in enumerate(qdf.head(10).iterrows()):
            marker = " ***" if i < 3 else ""
            actual = f"  actual={r['y_return'] * 100:+.1f}%" if actual_col and pd.notna(r.get("y_return")) else ""
            print(f"    {i + 1:2d}. {r['tic']:6s}  pred={r['predicted_return'] * 100:+6.1f}%{actual}{marker}")

    # Feature importance (collect from all models that expose it)
    importance_records = []
    for name, model in fitted.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            for rank_idx, (feat, val) in enumerate(imp.items(), 1):
                importance_records.append({
                    "bucket": bucket, "model": name,
                    "is_best": name == best_name,
                    "feature": feat, "importance": round(val, 6),
                    "rank": rank_idx,
                })
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            imp = pd.Series(coefs, index=feature_cols).sort_values(ascending=False)
            total = imp.sum()
            for rank_idx, (feat, val) in enumerate(imp.items(), 1):
                importance_records.append({
                    "bucket": bucket, "model": name,
                    "is_best": name == best_name,
                    "feature": feat, "importance": round(val / total if total > 0 else 0, 6),
                    "rank": rank_idx,
                })

    # For Stacking: use top base estimator's importance as proxy
    if best_name == "Stacking" and hasattr(fitted["Stacking"], "estimators_"):
        for est in fitted["Stacking"].estimators_:
            est_name = type(est).__name__
            if hasattr(est, "feature_importances_"):
                imp = pd.Series(est.feature_importances_, index=feature_cols).sort_values(ascending=False)
                for rank_idx, (feat, val) in enumerate(imp.items(), 1):
                    importance_records.append({
                        "bucket": bucket, "model": "Stacking",
                        "is_best": True,
                        "feature": feat, "importance": round(val, 6),
                        "rank": rank_idx,
                    })
                break  # use first tree-based estimator

    # Print top 5 for best model
    best_imp = [r for r in importance_records if r["model"] == best_name]
    if best_imp:
        best_imp_sorted = sorted(best_imp, key=lambda x: x["importance"], reverse=True)
        print(f"\n  Top 5 Features ({best_name}):")
        for r in best_imp_sorted[:5]:
            print(f"    {r['feature']:20s} {r['importance']:.3f}")

    return infer_b, model_results, importance_records


def run_dual_ensemble(df, bucket_preds, bucket_importances, val_cutoff, val_quarters, alpha=0.5):
    """Dual model ensemble: unified global model + bucket-split models combined via rank percentiles.

    Architecture:
        Stage 1 — Unified model trains on ALL stocks with sector dummies as features.
                  Captures cross-sector patterns with larger sample size.
        Stage 2 — Bucket-split models (already run) capture within-sector nuances.
        Stage 3 — Rank-based combination:
                  dual_score = α × unified_rank_pctile + (1-α) × bucket_rank_pctile
                  Uses rank percentiles (0-1) to normalize across different model scales.
    """
    print(f"\n{'='*70}")
    print(f"  DUAL ENSEMBLE — Stage 1: Unified Model (all stocks)")
    print(f"{'='*70}")

    all_sector_cols = sorted(c for c in df.columns if c.startswith("sector_") and df[c].sum() > 0)
    unified_features = FEATURE_COLS + MOMENTUM_COLS + all_sector_cols
    print(f"  Features: {len(FEATURE_COLS)} fund + {len(MOMENTUM_COLS)} mom + {len(all_sector_cols)} sector = {len(unified_features)}")

    unified_preds, unified_results, unified_importances = run_bucket(
        "unified_all", df, unified_features, val_cutoff=val_cutoff, val_quarters=val_quarters,
    )

    if len(unified_preds) == 0:
        print("  No unified predictions — skipping ensemble")
        return None, [], []

    # ---- Merge unified + bucket predictions ----
    u_slim = unified_preds[["tic", "datadate", "predicted_return"]].rename(
        columns={"predicted_return": "unified_pred"})

    dual = bucket_preds.copy()
    dual = dual.rename(columns={"predicted_return": "bucket_pred"})
    dual = dual.merge(u_slim, on=["tic", "datadate"], how="left")

    # Within-bucket rank percentiles (0→worst, 1→best)
    dual["bucket_pctile"] = dual.groupby(["bucket", "datadate"])["bucket_pred"].rank(pct=True)
    dual["unified_pctile"] = dual.groupby(["bucket", "datadate"])["unified_pred"].rank(pct=True)

    # Ensemble scores for multiple α values
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    for a in alphas:
        tag = f"a{int(a*100):02d}"
        dual[f"score_{tag}"] = a * dual["unified_pctile"] + (1 - a) * dual["bucket_pctile"]
        dual[f"rank_{tag}"] = dual.groupby(["bucket", "datadate"])[f"score_{tag}"].rank(ascending=False).astype(int)

    # Primary ensemble
    ptag = f"a{int(alpha*100):02d}"
    dual["dual_score"] = dual[f"score_{ptag}"]
    dual["rank_dual"] = dual[f"rank_{ptag}"]

    has_actual = "y_return" in dual.columns and dual["y_return"].notna().any()

    # ---- Stage 3: Comparison Report ----
    print(f"\n{'='*70}")
    print(f"  STAGE 3: Per-Bucket Top 5 Comparison")
    print(f"  dual_score = α×unified_rank%%ile + (1-α)×bucket_rank%%ile")
    print(f"{'='*70}")

    buckets = ["growth_tech", "cyclical", "real_assets", "defensive"]
    methods = [
        ("Bucket-Split(α=0)", "rank_a00"),
        ("Dual α=0.3",        "rank_a30"),
        ("Dual α=0.5",        "rank_a50"),
        ("Dual α=0.7",        "rank_a70"),
        ("Unified(α=1)",      "rank_a100"),
    ]

    for idate in sorted(dual["datadate"].unique()):
        for bucket in buckets:
            bq = dual[(dual["bucket"] == bucket) & (dual["datadate"] == idate)]
            if len(bq) == 0:
                continue

            print(f"\n  [{bucket.upper()}] ({idate}, {len(bq)} stocks)")
            for method_name, rank_col in methods:
                top5 = bq.nsmallest(5, rank_col)
                parts = []
                for _, r in top5.iterrows():
                    if has_actual and pd.notna(r.get("y_return")):
                        parts.append(f"{r['tic']}{r['y_return']*100:+.0f}%")
                    else:
                        parts.append(r["tic"])
                avg = top5["y_return"].mean() * 100 if has_actual and top5["y_return"].notna().any() else float("nan")
                avg_s = f"avg={avg:+.1f}%" if not np.isnan(avg) else ""
                print(f"    {method_name:<22} {', '.join(parts):<50} {avg_s}")

    # ---- Alpha Sensitivity ----
    if has_actual:
        print(f"\n{'='*70}")
        print(f"  ALPHA SENSITIVITY (per-bucket Top 5 avg actual return)")
        print(f"{'='*70}")

        # SPY proxy: equal-weight avg of all inference stocks
        spy_proxy = dual["y_return"].dropna().mean() * 100

        print(f"  {'α':<8} {'Label':<18} {'Top5 Avg':>10} {'SPY≈':>8} {'Alpha':>8}")
        print(f"  {'-'*54}")
        best_a, best_ret = None, -999
        for a in alphas:
            tag = f"a{int(a*100):02d}"
            avgs = []
            for idate in dual["datadate"].unique():
                for bucket in buckets:
                    bq = dual[(dual["bucket"] == bucket) & (dual["datadate"] == idate)]
                    if len(bq) == 0:
                        continue
                    top5 = bq.nsmallest(5, f"rank_{tag}")
                    v = top5["y_return"].mean()
                    if pd.notna(v):
                        avgs.append(v)
            if avgs:
                overall = np.mean(avgs) * 100
                alpha_pp = overall - spy_proxy
                label = {0.0: "Pure Bucket", 1.0: "Pure Unified"}.get(a, f"Blend {int(a*100)}/{int((1-a)*100)}")
                marker = ""
                if overall > best_ret:
                    best_a, best_ret = a, overall
                    marker = " *"
                print(f"  {a:<8.1f} {label:<18} {overall:>+9.1f}% {spy_proxy:>+7.1f}% {alpha_pp:>+7.1f}pp{marker}")
        if best_a is not None:
            print(f"\n  * Best α={best_a} on this period (in-sample, not predictive)")

    # ---- Feature Importance Comparison ----
    print(f"\n{'='*70}")
    print(f"  FEATURE IMPORTANCE: Unified vs Bucket Models")
    print(f"{'='*70}")

    u_best = sorted([r for r in unified_importances if r.get("is_best")],
                     key=lambda x: x["importance"], reverse=True)
    if u_best:
        print(f"\n  {'Unified Top 10':<30} {'Importance':>10}")
        print(f"  {'-'*42}")
        for r in u_best[:10]:
            print(f"  {r['feature']:<30} {r['importance']:.4f}")

    for bucket in buckets:
        b_best = sorted([r for r in bucket_importances if r["bucket"] == bucket and r.get("is_best")],
                         key=lambda x: x["importance"], reverse=True)
        if b_best:
            print(f"\n  {bucket} Top 5:")
            for r in b_best[:5]:
                print(f"    {r['feature']:<28} {r['importance']:.4f}")

    return dual, unified_results, unified_importances


def main():
    parser = argparse.ArgumentParser(description="Per-bucket ML stock selection")
    parser.add_argument("--db", default=os.path.join(project_root, "data", "finrl_trading.db"))
    parser.add_argument("--universe", default=None,
                        help="Filter to a stock universe: sp500, nasdaq100, or path to CSV with 'tickers' column")
    parser.add_argument("--val-cutoff", default="2025-12-31", help="Validation end date (last val quarter)")
    parser.add_argument("--val-quarters", type=int, default=3, help="Number of validation quarters (default: 3)")
    parser.add_argument("--output-dir", default=os.path.join(project_root, "data"))
    parser.add_argument("--latest-snapshot", action="store_true",
                        help="Inference using latest available filing per ticker (fill missing with previous quarter)")
    parser.add_argument("--mixed-vintage", action="store_true",
                        help="Inference: use latest available datadate per current universe members (mix Q4/Q1)")
    parser.add_argument("--ref-date", default=None,
                        help="Reference date for actual return calculation in latest-snapshot mode (default: last quarter end)")
    parser.add_argument("--end-date", default=None,
                        help="End date for return calculation (default: today)")
    parser.add_argument("--infer-date", default=None,
                        help="Only infer on this specific datadate (e.g. 2025-12-31). Default: all quarters after val-cutoff")
    parser.add_argument("--dual-ensemble", action="store_true",
                        help="Run dual model ensemble: unified + bucket-split combined via rank percentiles")
    parser.add_argument("--ensemble-alpha", type=float, default=0.5,
                        help="Ensemble weight: α×unified + (1-α)×bucket (default: 0.5)")
    parser.add_argument("--mixed-alpha", action="store_true",
                        help="Per-bucket optimal α: growth_tech=0(bucket), cyclical=0.7(dual), "
                             "real_assets=1.0(unified), defensive=1.0(unified)")
    parser.add_argument("--unified-only", action="store_true",
                        help="Skip per-bucket split; train single unified model and rank all stocks together")
    args = parser.parse_args()

    # Derive universe name for file paths and print statements
    universe_name = (args.universe or "sp500").lower()

    # Load data
    conn = sqlite3.connect(args.db)
    _feat_sql = ", ".join(FEATURE_COLS)
    df = pd.read_sql(
        f"""SELECT ticker as tic, datadate, gsector, adj_close_q, trade_price,
           filing_date, accepted_date,
           {_feat_sql}, y_return
           FROM fundamental_data ORDER BY ticker, datadate""",
        conn,
    )
    conn.close()

    # Filter to universe if specified
    if args.universe:
        if args.universe.lower() == "nasdaq100":
            import sys as _sys; _sys.path.insert(0, os.path.join(project_root, "src"))
            from data.data_fetcher import fetch_nasdaq100_tickers
            univ = fetch_nasdaq100_tickers()
            univ_tickers = set(univ["tickers"].tolist())
        elif args.universe.lower() == "sp500":
            from data.data_fetcher import fetch_sp500_tickers
            univ = fetch_sp500_tickers()
            univ_tickers = set(univ["tickers"].tolist())
        elif os.path.exists(args.universe):
            univ_tickers = set(pd.read_csv(args.universe)["tickers"].tolist())
        else:
            raise ValueError(f"Unknown universe: {args.universe}")
        before = len(df)
        df = df[df["tic"].isin(univ_tickers)].copy()
        print(f"Universe filter ({args.universe}): {before} -> {len(df)} records ({df['tic'].nunique()} tickers)")

    print(f"Loaded {len(df)} records, {df['tic'].nunique()} tickers")
    print(f"Date range: {df['datadate'].min()} ~ {df['datadate'].max()}")
    print(f"Val cutoff: {args.val_cutoff}")

    # Load historical membership (used for point-in-time universe filtering)
    hist_csv = os.path.join(project_root, "data", f"{universe_name}_historical_constituents.csv")
    _hist_df = None
    if os.path.exists(hist_csv):
        _hist_df = pd.read_csv(hist_csv)
        _hist_df['date'] = pd.to_datetime(_hist_df['date'])
    else:
        print(f"WARNING: {hist_csv} not found — point-in-time filtering will be skipped for {universe_name.upper()}")

    def get_universe_at(quarter_str):
        """Return set of universe tickers at a given quarter date."""
        if _hist_df is None:
            return set()
        q_dt = pd.to_datetime(quarter_str)
        valid = _hist_df[_hist_df['date'] <= q_dt]
        if not valid.empty:
            return set(t.strip() for t in valid.iloc[-1]['tickers'].split(','))
        return set()

    # Momentum features: price momentum (from trade_price) + fundamental QoQ changes
    df = df.sort_values(["tic", "datadate"]).copy()
    df["adj_close_q"] = pd.to_numeric(df["adj_close_q"], errors="coerce")
    df["trade_price"] = pd.to_numeric(df["trade_price"], errors="coerce")

    # Fill NULL trade_price (future tradedate) with today's price for momentum calc
    import yfinance as yf
    from datetime import date, timedelta
    today = date.today()

    null_tp_mask = df["trade_price"].isna()
    if null_tp_mask.any():
        null_tickers = df.loc[null_tp_mask, "tic"].unique().tolist()
        print(f"Filling {null_tp_mask.sum()} NULL trade_price ({len(null_tickers)} tickers) with today's price ...")
        yf_tickers = [t.replace(".", "-") for t in null_tickers]
        try:
            px = yf.download(yf_tickers, start=(today - timedelta(days=5)).isoformat(),
                             end=(today + timedelta(days=1)).isoformat(),
                             auto_adjust=True, progress=False)
            if isinstance(px.columns, pd.MultiIndex):
                close = px["Close"]
            else:
                close = px[["Close"]]
            filled = 0
            for tic in null_tickers:
                yf_t = tic.replace(".", "-")
                if yf_t in close.columns:
                    s = close[yf_t].dropna()
                    if len(s) > 0:
                        df.loc[(df["tic"] == tic) & null_tp_mask, "trade_price"] = float(s.iloc[-1])
                        filled += 1
            print(f"  Filled {filled}/{len(null_tickers)} tickers with today's price")
        except Exception as e:
            print(f"  WARNING: yfinance download failed: {e}")

    # Mixed-vintage: override ALL latest inference rows' trade_price with today's price
    # so Q4 reporters (real 3/1 price) and Q1 reporters (already filled) are aligned —
    # everyone's ret_1q reflects "momentum from prev tradedate to TODAY"
    if args.mixed_vintage:
        infer_rows = df[df["datadate"] > args.val_cutoff].copy()
        # Find the latest inference row per ticker (the one mixed-vintage will use)
        latest_infer = infer_rows.sort_values(["tic", "datadate"]).drop_duplicates(subset="tic", keep="last")
        mv_tickers = latest_infer["tic"].unique().tolist()
        # Only download for tickers whose trade_price is NOT already today's price
        stale_tics = latest_infer.loc[latest_infer["trade_price"].notna(), "tic"].unique().tolist()
        if stale_tics:
            print(f"Mixed-vintage: aligning {len(stale_tics)} tickers' trade_price to today's price ...")
            yf_mv = [t.replace(".", "-") for t in stale_tics]
            try:
                px_mv = yf.download(yf_mv, start=(today - timedelta(days=5)).isoformat(),
                                    end=(today + timedelta(days=1)).isoformat(),
                                    auto_adjust=True, progress=False)
                if isinstance(px_mv.columns, pd.MultiIndex):
                    close_mv = px_mv["Close"]
                else:
                    close_mv = px_mv[["Close"]]
                aligned = 0
                for tic in stale_tics:
                    yf_t = tic.replace(".", "-")
                    if yf_t in close_mv.columns:
                        s = close_mv[yf_t].dropna()
                        if len(s) > 0:
                            # Find this ticker's latest inference row index in df
                            tic_infer = df[(df["tic"] == tic) & (df["datadate"] > args.val_cutoff)]
                            last_idx = tic_infer.sort_values("datadate").index[-1]
                            df.loc[last_idx, "trade_price"] = float(s.iloc[-1])
                            aligned += 1
                print(f"  Aligned {aligned}/{len(stale_tics)} tickers")
            except Exception as e:
                print(f"  WARNING: yfinance download failed: {e}")

    df["ret_1q"] = df.groupby("tic")["trade_price"].pct_change(1)
    df["ret_4q"] = df.groupby("tic")["trade_price"].pct_change(4)
    # ret_accel computed AFTER winsorize to maintain algebraic consistency
    for src, dst in [("EPS", "eps_chg"), ("roe", "roe_chg"),
                     ("gross_margin", "gm_chg"), ("operating_margin", "om_chg")]:
        df[src] = pd.to_numeric(df[src], errors="coerce")
        df[dst] = df.groupby("tic")[src].diff()

    # Prep features (ret_accel excluded here, computed after winsorize)
    pre_accel_feats = [c for c in FEATURE_COLS + MOMENTUM_COLS if c != "ret_accel"]
    for c in pre_accel_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    global_medians = df[pre_accel_feats].median()
    df[pre_accel_feats] = df[pre_accel_feats].fillna(global_medians).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Winsorize: clip at 1st/99th percentile to reduce outlier impact
    for c in pre_accel_feats:
        p01, p99 = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(lower=p01, upper=p99)

    # Compute ret_accel from winsorized ret_1q/ret_4q, then winsorize it too
    df["ret_accel"] = df["ret_1q"] - df["ret_4q"] / 4
    df["ret_accel"] = df["ret_accel"].fillna(0)
    p01, p99 = df["ret_accel"].quantile(0.01), df["ret_accel"].quantile(0.99)
    df["ret_accel"] = df["ret_accel"].clip(lower=p01, upper=p99)
    print(f"Added {len(MOMENTUM_COLS)} momentum features: {MOMENTUM_COLS}")

    # Assign buckets
    df["bucket"] = df["gsector"].str.lower().map(SECTOR_TO_BUCKET)
    unmapped = df[df["bucket"].isna()]["gsector"].unique()
    if len(unmapped) > 0:
        print(f"WARNING: unmapped sectors: {unmapped}")
    df = df[df["bucket"].notna()].copy()

    # Latest-snapshot mode: for inference, use the most recent filing per ticker
    # instead of only future quarter dates. This gives ~500 stocks to rank.
    if args.latest_snapshot:
        if universe_name == "nasdaq100":
            from data.data_fetcher import fetch_nasdaq100_tickers
            univ_snap = fetch_nasdaq100_tickers()
        else:
            from data.data_fetcher import fetch_sp500_tickers
            univ_snap = fetch_sp500_tickers()
        univ_tickers = set(univ_snap["tickers"].tolist()) if univ_snap is not None else set()
        print(f"\nLatest-snapshot mode: {len(univ_tickers)} current {universe_name.upper()} tickers")

        # For each universe ticker, take the latest available record (up to ref_date if set)
        import yfinance as yf
        from datetime import date
        ref_date = pd.Timestamp(args.ref_date) if args.ref_date else pd.Timestamp("2026-03-31")
        end_date = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp(date.today())

        train_part = df[df["datadate"] <= args.val_cutoff].copy()
        latest_rows = []
        for tic in sorted(univ_tickers):
            tic_df = df[(df["tic"] == tic) & (df["datadate"] <= ref_date.strftime("%Y-%m-%d"))]
            if len(tic_df) == 0:
                continue
            latest = tic_df.sort_values("datadate").iloc[-1].copy()
            latest["datadate"] = "latest"  # synthetic date for inference
            latest_rows.append(latest)

        if latest_rows:
            snapshot_df = pd.DataFrame(latest_rows)
            # Download actual returns: price on ref_date -> end_date via FMP
            import requests
            from data.data_fetcher import FMPFetcher
            fmp = FMPFetcher()
            all_tickers_list = list(snapshot_df["tic"].unique())
            dl_start = (ref_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            dl_end = (end_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            print(f"  Ref: {ref_date.strftime('%Y-%m-%d')} -> End: {end_date.strftime('%Y-%m-%d')}")
            print(f"  Downloading prices via FMP for {len(all_tickers_list)} tickers ...")

            # Use FMP historical-price-eod API per ticker
            all_close = {}  # tic -> pd.Series(date->close)
            for i, tic in enumerate(all_tickers_list):
                try:
                    url = f"{fmp.base_url}/historical-price-eod/full?symbol={tic}&from={dl_start}&to={dl_end}&apikey={fmp.api_key}"
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    rows = data.get("historical", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    if rows:
                        s = pd.Series(
                            {pd.Timestamp(r["date"]): r["adjClose"] if "adjClose" in r else r["close"] for r in rows}
                        ).sort_index()
                        if len(s) > 0:
                            all_close[tic] = s
                except Exception:
                    pass
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(all_tickers_list)} done ...")

            print(f"  Got prices for {len(all_close)} / {len(all_tickers_list)} tickers")

            # Get price at ref_date and end_date for each ticker
            # Use first trading day ON or AFTER the date (consistent with DB tradedate logic)
            y_returns = {}
            for tic in snapshot_df["tic"].unique():
                if tic not in all_close:
                    continue
                series = all_close[tic]
                # Buy: first trading day >= ref_date
                mask_ref = series.index >= ref_date
                if not mask_ref.any():
                    continue
                p_ref = float(series[mask_ref].iloc[0])
                # Sell: first trading day >= end_date; if future, use latest available
                mask_end = series.index >= end_date
                if mask_end.any():
                    p_end = float(series[mask_end].iloc[0])
                else:
                    p_end = float(series.iloc[-1])  # end_date in future, use latest
                if p_ref > 0:
                    y_returns[tic] = np.log(p_end / p_ref)

            snapshot_df["y_return"] = snapshot_df["tic"].map(y_returns)
            filled = snapshot_df["y_return"].notna().sum()
            print(f"  Snapshot: {len(snapshot_df)} tickers, y_return filled: {filled}")

            # Replace inference data: keep train_part + snapshot
            df = pd.concat([train_part, snapshot_df], ignore_index=True)

    # Mixed-vintage mode: for inference, keep only the latest record per ticker
    # among current universe members. Early reporters get Q1 2026 data, rest keep Q4 2025.
    if args.mixed_vintage:
        mv_hist_csv = os.path.join(project_root, "data", f"{universe_name}_historical_constituents.csv")
        if os.path.exists(mv_hist_csv):
            _hist_mv = pd.read_csv(mv_hist_csv)
            _hist_mv['date'] = pd.to_datetime(_hist_mv['date'])
            latest_members = set(
                t.strip() for t in _hist_mv.loc[_hist_mv['date'] == _hist_mv['date'].max(), 'tickers'].iloc[0].split(',')
            )
        else:
            # No historical CSV — fall back to current members from API
            print(f"  No historical CSV for {universe_name.upper()}, fetching current members from API...")
            if universe_name == "nasdaq100":
                from data.data_fetcher import fetch_nasdaq100_tickers
                _mv_snap = fetch_nasdaq100_tickers()
            else:
                from data.data_fetcher import fetch_sp500_tickers
                _mv_snap = fetch_sp500_tickers()
            latest_members = set(_mv_snap["tickers"].tolist()) if _mv_snap is not None else set()

        train_val_part = df[df["datadate"] <= args.val_cutoff].copy()
        infer_part = df[(df["datadate"] > args.val_cutoff) & (df["tic"].isin(latest_members))].copy()

        # Keep only latest datadate per ticker
        infer_part = infer_part.sort_values(["tic", "datadate"]).drop_duplicates(subset="tic", keep="last")

        # Tag vintage for output
        infer_part["data_vintage"] = infer_part["datadate"].apply(
            lambda d: "Q1_2026" if d >= "2026-03-01" else "Q4_2025"
        )

        n_q1 = (infer_part["data_vintage"] == "Q1_2026").sum()
        n_q4 = (infer_part["data_vintage"] == "Q4_2025").sum()
        print(f"\nMixed-vintage mode: {len(infer_part)} current {universe_name.upper()} tickers for inference")
        print(f"  Q1 2026 (early reporters): {n_q1}")
        print(f"  Q4 2025 (not yet reported): {n_q4}")

        # Set all inference records to a common synthetic datadate so they rank together
        infer_part["original_datadate"] = infer_part["datadate"]
        infer_part["datadate"] = "mixed"

        df = pd.concat([train_val_part, infer_part], ignore_index=True)

    # --infer-date: keep only the specified inference quarter (drop other future quarters)
    if args.infer_date and not args.mixed_vintage:
        before = len(df)
        df = df[(df["datadate"] <= args.val_cutoff) | (df["datadate"] == args.infer_date)].copy()
        print(f"Infer-date filter: keep only {args.infer_date} for inference ({before} -> {len(df)} records)")

    # ---------------------------------------------------------------
    # Point-in-time universe membership filter (Principles A, B, C)
    #   A: Universe is point-in-time at each tradedate
    #   B: Features already computed on full data (above)
    #   C: Same rule for train + val + inference
    # For each (ticker, datadate), keep only if the ticker was in
    # the universe at the corresponding tradedate.
    # ---------------------------------------------------------------
    if _hist_df is not None:
        synthetic_dates = {"latest", "mixed"}
        real_mask = ~df["datadate"].isin(synthetic_dates)

        # Map each real datadate to its tradedate
        df["_tradedate_pit"] = None
        df.loc[real_mask, "_tradedate_pit"] = df.loc[real_mask, "datadate"].apply(datadate_to_tradedate)

        # Build keep mask
        pit_keep = ~real_mask  # always keep synthetic rows (latest-snapshot / mixed-vintage)
        unique_tradedates = sorted(df.loc[real_mask, "_tradedate_pit"].dropna().unique())

        print(f"\nPoint-in-time {universe_name.upper()} filter ({len(unique_tradedates)} tradedates):")

        for td in unique_tradedates:
            universe_members = get_universe_at(td)
            td_mask = df["_tradedate_pit"] == td
            pit_keep = pit_keep | (td_mask & df["tic"].isin(universe_members))

        # Drop rows with unmappable datadates (non-standard quarter ends)
        unmappable = real_mask & df["_tradedate_pit"].isna()
        if unmappable.any():
            print(f"  Dropped {unmappable.sum()} rows with non-standard datadates")

        before_pit = len(df)
        df = df[pit_keep].copy()
        train_count = (df["datadate"] <= args.val_cutoff).sum() if not df.empty else 0
        infer_count = len(df) - train_count
        print(f"  {before_pit} -> {len(df)} records ({df['tic'].nunique()} tickers)")
        print(f"  Train+Val: {train_count} | Inference: {infer_count}")

        # Show last few tradedates for verification
        for td in unique_tradedates[-4:]:
            universe_members = get_universe_at(td)
            td_rows = (df["_tradedate_pit"] == td).sum()
            print(f"  tradedate {td}: {td_rows} stocks ({universe_name.upper()}={len(universe_members)})")

        df = df.drop(columns=["_tradedate_pit"], errors="ignore")
    else:
        print(f"\nSkipping point-in-time filter (no historical constituents for {universe_name.upper()})")

    # Sub-sector indicator features: one-hot encode gsector so models can
    # distinguish sectors within each bucket (e.g. Energy vs Real Estate).
    sector_dummies = pd.get_dummies(df["gsector"], prefix="sector")
    df = pd.concat([df, sector_dummies], axis=1)

    # ---- Unified-only mode: single model, rank all stocks together ----
    if args.unified_only:
        all_sector_cols = sorted(c for c in df.columns if c.startswith("sector_") and df[c].sum() > 0)
        unified_features = FEATURE_COLS + MOMENTUM_COLS + all_sector_cols
        print(f"\n  [Unified-only mode]: {len(FEATURE_COLS)} fundamental + {len(MOMENTUM_COLS)} momentum + {len(all_sector_cols)} sector = {len(unified_features)} features")

        pred_all, all_model_results, all_importances = run_bucket(
            "unified_all", df, unified_features,
            val_cutoff=args.val_cutoff, val_quarters=args.val_quarters,
        )

        if len(pred_all) == 0:
            print("\nNo predictions generated.")
            return

        # Overall ranking (no bucket split)
        pred_all["rank_best"] = pred_all.groupby("datadate")["predicted_return"].rank(ascending=False).astype(int)
        pred_all["rank_ensemble"] = pred_all.groupby("datadate")["pred_ensemble_avg"].rank(ascending=False).astype(int)

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        prefix = f"{args.universe}_" if args.universe else "sp500_"
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        real_dates = pred_all["datadate"][~pred_all["datadate"].isin({"latest", "mixed"})]
        if "tradedate" in pred_all.columns and pred_all["tradedate"].notna().any():
            trade_tag = pred_all["tradedate"].max().replace("-", "")
        elif not real_dates.empty:
            trade_tag = datadate_to_tradedate(real_dates.max()).replace("-", "")
        else:
            trade_tag = pd.Timestamp.now().strftime("%Y%m%d")

        pred_path = os.path.join(args.output_dir, f"{prefix}ml_unified_predictions_{trade_tag}_{timestamp}.csv")
        pred_all.to_csv(pred_path, index=False)
        print(f"\nSaved: {pred_path} ({len(pred_all)} stocks)")

        # Excel dashboard
        excel_path = os.path.join(args.output_dir, f"{prefix}ml_dashboard_{trade_tag}_{timestamp}.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pred_all.sort_values("rank_best").to_excel(writer, sheet_name="Rankings", index=False)
            pd.DataFrame(all_model_results).to_excel(writer, sheet_name="Models", index=False)
            if all_importances:
                pd.DataFrame(all_importances).to_excel(writer, sheet_name="Features", index=False)
        print(f"Saved: {excel_path}")
        print(f"Dashboard: python3 src/tools/dashboard.py {excel_path}")

        # Summary: top 10 overall
        print(f"\n{'='*60}")
        print(f"  UNIFIED MODEL: Top 10 Overall Picks")
        print(f"{'='*60}")
        has_actual = "y_return" in pred_all.columns and pred_all["y_return"].notna().any()
        for idate in sorted(pred_all["datadate"].unique()):
            qdf = pred_all[pred_all["datadate"] == idate].sort_values("rank_best")
            print(f"\n  --- {idate} ({len(qdf)} stocks) ---")
            for i, (_, r) in enumerate(qdf.head(10).iterrows(), 1):
                actual = f"  actual={r['y_return']*100:+.1f}%" if has_actual and pd.notna(r.get("y_return")) else ""
                print(f"    {i:>2}. {r['tic']:<8} pred={r['predicted_return']*100:+.1f}%{actual}")
            if has_actual and qdf.head(5)["y_return"].notna().any():
                avg5 = qdf.head(5)["y_return"].mean() * 100
                avg10 = qdf.head(10)["y_return"].mean() * 100
                print(f"\n    Top 5 avg: {avg5:+.1f}% | Top 10 avg: {avg10:+.1f}%")

        return

    # Run per bucket
    all_preds = []
    all_model_results = []
    all_importances = []

    for bucket in ["growth_tech", "cyclical", "real_assets", "defensive"]:
        bdf = df[df["bucket"] == bucket].copy()
        # Keep only sector dummy columns that have variance within this bucket
        sector_cols = [c for c in bdf.columns if c.startswith("sector_") and bdf[c].sum() > 0]
        mom_cols = MOMENTUM_COLS
        bucket_features = FEATURE_COLS + mom_cols + sector_cols
        print(f"\n  [Features for {bucket}]: {len(FEATURE_COLS)} fundamental + {len(mom_cols)} momentum + {len(sector_cols)} sector = {len(bucket_features)}")
        preds, results, importances = run_bucket(bucket, bdf, bucket_features, val_cutoff=args.val_cutoff, val_quarters=args.val_quarters)
        if len(preds) > 0:
            all_preds.append(preds)
        all_model_results.extend(results)
        all_importances.extend(importances)

    if not all_preds:
        print("\nNo predictions generated.")
        return

    pred_all = pd.concat(all_preds, ignore_index=True)

    # Per-bucket-per-quarter ranking
    pred_all["rank_best"] = pred_all.groupby(["bucket", "datadate"])["predicted_return"].rank(ascending=False).astype(int)
    pred_all["rank_ensemble"] = pred_all.groupby(["bucket", "datadate"])["pred_ensemble_avg"].rank(ascending=False).astype(int)

    # ---- Actual return calculation for inference rows (ref-date → end-date) ----
    if args.ref_date and not args.latest_snapshot:
        from datetime import date
        ref_dt = pd.Timestamp(args.ref_date)
        end_dt = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp(date.today())
        # Identify inference rows (datadate > val_cutoff or synthetic like "mixed")
        infer_mask = pred_all["datadate"].apply(
            lambda d: d in ("mixed", "latest") or (isinstance(d, str) and d > args.val_cutoff)
        )
        infer_tickers = pred_all.loc[infer_mask, "tic"].unique().tolist()
        if infer_tickers:
            print(f"\n  Calculating actual returns: {ref_dt.date()} → {end_dt.date()} for {len(infer_tickers)} tickers ...")
            import yfinance as yf
            yf_tickers = [t.replace(".", "-") for t in infer_tickers] + ["SPY"]
            dl_start = (ref_dt - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            dl_end = (end_dt + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            try:
                px = yf.download(yf_tickers, start=dl_start, end=dl_end, auto_adjust=True, progress=False)
                close = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px[["Close"]]
                # Buy: first trading day >= ref_date
                buy_dates = close.loc[ref_dt.strftime("%Y-%m-%d"):].index
                sell_dates = close.loc[:end_dt.strftime("%Y-%m-%d")].index
                if len(buy_dates) > 0 and len(sell_dates) > 0:
                    buy_d = buy_dates[0]
                    sell_d = sell_dates[-1]
                    y_rets = {}
                    for tic in infer_tickers:
                        yf_t = tic.replace(".", "-")
                        if yf_t in close.columns:
                            try:
                                p_buy = float(close.loc[buy_d, yf_t])
                                p_sell = float(close.loc[sell_d, yf_t])
                                if p_buy > 0:
                                    y_rets[tic] = (p_sell / p_buy) - 1
                            except Exception:
                                pass
                    pred_all.loc[infer_mask, "y_return"] = pred_all.loc[infer_mask, "tic"].map(y_rets)
                    filled = pred_all.loc[infer_mask, "y_return"].notna().sum()
                    spy_ret = ""
                    if "SPY" in close.columns:
                        try:
                            spy_ret = f" (SPY: {(float(close.loc[sell_d, 'SPY']) / float(close.loc[buy_d, 'SPY']) - 1) * 100:+.1f}%)"
                        except Exception:
                            pass
                    print(f"  Actual returns filled: {filled}/{len(infer_tickers)}{spy_ret}")
            except Exception as e:
                print(f"  WARNING: yfinance download failed: {e}")

    # Save — prefix filenames with universe name + tradedate tag + timestamp
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{args.universe}_" if args.universe else "sp500_"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    # Derive trade period tag from the latest tradedate in predictions
    if "tradedate" in pred_all.columns and pred_all["tradedate"].notna().any():
        trade_tag = pred_all["tradedate"].max().replace("-", "")
    else:
        # Fallback: derive from latest real datadate (skip synthetic like "mixed"/"latest")
        real_dates = pred_all["datadate"][~pred_all["datadate"].isin({"latest", "mixed"})]
        if not real_dates.empty:
            trade_tag = datadate_to_tradedate(real_dates.max()).replace("-", "")
        else:
            trade_tag = pd.Timestamp.now().strftime("%Y%m%d")

    pred_path = os.path.join(args.output_dir, f"{prefix}ml_bucket_predictions_{trade_tag}_{timestamp}.csv")
    pred_all.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path} ({len(pred_all)} stocks)")

    model_path = os.path.join(args.output_dir, f"{prefix}ml_bucket_model_results_{trade_tag}_{timestamp}.csv")
    pd.DataFrame(all_model_results).to_csv(model_path, index=False)

    if all_importances:
        imp_df = pd.DataFrame(all_importances)
        imp_df = imp_df.sort_values(["bucket", "model", "rank"])
        imp_path = os.path.join(args.output_dir, f"{prefix}ml_feature_importance_{trade_tag}_{timestamp}.csv")
        imp_df.to_csv(imp_path, index=False)
        print(f"Saved: {imp_path} ({len(imp_df)} rows)")
    print(f"Saved: {model_path} ({len(all_model_results)} rows)")

    # Excel dashboard
    excel_path = os.path.join(args.output_dir, f"{prefix}ml_dashboard_{trade_tag}_{timestamp}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pred_all.sort_values("rank_best").to_excel(writer, sheet_name="Rankings", index=False)
        pd.DataFrame(all_model_results).to_excel(writer, sheet_name="Models", index=False)
        if all_importances:
            pd.DataFrame(all_importances).to_excel(writer, sheet_name="Features", index=False)
    print(f"Saved: {excel_path}")
    print(f"Dashboard: python3 src/tools/dashboard.py {excel_path}")

    # Summary per quarter
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: Top picks per bucket per quarter")
    print(f"{'=' * 60}")
    for idate in sorted(pred_all["datadate"].unique()):
        print(f"\n  --- {idate} ---")
        for bucket in ["growth_tech", "cyclical", "real_assets", "defensive"]:
            bp = pred_all[(pred_all["bucket"] == bucket) & (pred_all["datadate"] == idate)].sort_values("rank_best").head(3)
            if len(bp) == 0:
                print(f"  {bucket:15s}: no data")
                continue
            best_m = bp.iloc[0]["best_model"]
            picks = ", ".join(
                f"{r['tic']}({r['predicted_return'] * 100:+.1f}%)" for _, r in bp.iterrows()
            )
            print(f"  {bucket:15s} [{best_m}]: {picks}")

    # ---- Dual Model Ensemble ----
    if args.dual_ensemble:
        dual_result, dual_model_results, dual_importances = run_dual_ensemble(
            df, pred_all, all_importances,
            args.val_cutoff, args.val_quarters, args.ensemble_alpha,
        )
        if dual_result is not None:
            dual_path = os.path.join(args.output_dir, f"{prefix}ml_dual_ensemble_{trade_tag}_{timestamp}.csv")
            dual_result.to_csv(dual_path, index=False)
            print(f"\nSaved dual ensemble: {dual_path} ({len(dual_result)} stocks)")

    # ---- Mixed Alpha: per-bucket optimal strategy ----
    if args.mixed_alpha:
        # Optimal α per bucket (derived from 5-period backtest)
        BUCKET_ALPHA = {
            "growth_tech": 0.0,   # Pure bucket-split — tech has unique features
            "cyclical":    0.7,   # Heavy unified — diverse sectors need cross-sector signal
            "real_assets": 1.0,   # Pure unified — small sample, bucket model overfits
            "defensive":   1.0,   # Pure unified — fundamentals-driven, best Sharpe
        }

        print(f"\n{'='*70}")
        print(f"  MIXED-ALPHA STRATEGY")
        print(f"  Per-bucket optimal α from 5-period backtest:")
        for b, a in BUCKET_ALPHA.items():
            label = {0.0: "Pure Bucket-Split", 1.0: "Pure Unified"}.get(a, f"Dual {int(a*100)}/{int((1-a)*100)}")
            print(f"    {b:<15} α={a:.1f}  ({label})")
        print(f"{'='*70}")

        # Need unified model for buckets with α > 0
        needs_unified = any(a > 0 for a in BUCKET_ALPHA.values())
        unified_preds_df = pd.DataFrame()
        unified_imps = []

        if needs_unified:
            all_sector_cols = sorted(c for c in df.columns if c.startswith("sector_") and df[c].sum() > 0)
            unified_features = FEATURE_COLS + MOMENTUM_COLS + all_sector_cols
            print(f"\n  Training unified model ({len(unified_features)} features) ...")

            unified_preds_df, _, unified_imps = run_bucket(
                "unified_all", df, unified_features,
                val_cutoff=args.val_cutoff, val_quarters=args.val_quarters,
            )

        if len(unified_preds_df) == 0 and needs_unified:
            print("  WARNING: Unified model produced no predictions, falling back to bucket-split")
            BUCKET_ALPHA = {b: 0.0 for b in BUCKET_ALPHA}

        # Build final predictions by applying per-bucket α
        u_slim = pd.DataFrame()
        if len(unified_preds_df) > 0:
            u_slim = unified_preds_df[["tic", "datadate", "predicted_return"]].rename(
                columns={"predicted_return": "unified_pred"})

        mixed = pred_all.copy()
        mixed = mixed.rename(columns={"predicted_return": "bucket_pred"})
        if len(u_slim) > 0:
            mixed = mixed.merge(u_slim, on=["tic", "datadate"], how="left")
        else:
            mixed["unified_pred"] = mixed["bucket_pred"]

        # Compute within-bucket rank percentiles
        mixed["bucket_pctile"] = mixed.groupby(["bucket", "datadate"])["bucket_pred"].rank(pct=True)
        mixed["unified_pctile"] = mixed.groupby(["bucket", "datadate"])["unified_pred"].rank(pct=True)

        # Apply per-bucket α
        mixed["mixed_score"] = 0.0
        for bucket, alpha in BUCKET_ALPHA.items():
            mask = mixed["bucket"] == bucket
            mixed.loc[mask, "mixed_score"] = (
                alpha * mixed.loc[mask, "unified_pctile"] +
                (1 - alpha) * mixed.loc[mask, "bucket_pctile"]
            )
            mixed.loc[mask, "applied_alpha"] = alpha

        mixed["rank_mixed"] = mixed.groupby(["bucket", "datadate"])["mixed_score"].rank(ascending=False).astype(int)

        # Also compute uniform α=0.5 for comparison
        mixed["score_uniform"] = 0.5 * mixed["unified_pctile"] + 0.5 * mixed["bucket_pctile"]
        mixed["rank_uniform"] = mixed.groupby(["bucket", "datadate"])["score_uniform"].rank(ascending=False).astype(int)

        has_actual = "y_return" in mixed.columns and mixed["y_return"].notna().any()
        buckets = ["growth_tech", "cyclical", "real_assets", "defensive"]

        # ---- Results ----
        print(f"\n{'='*70}")
        print(f"  MIXED-ALPHA RESULTS: Top 5 per Bucket")
        print(f"{'='*70}")

        for idate in sorted(mixed["datadate"].unique()):
            for bucket in buckets:
                bq = mixed[(mixed["bucket"] == bucket) & (mixed["datadate"] == idate)]
                if len(bq) == 0:
                    continue
                a = BUCKET_ALPHA[bucket]
                label = {0.0: "Bucket", 1.0: "Unified"}.get(a, f"Dual α={a}")

                print(f"\n  [{bucket.upper()}] α={a} ({label}), {len(bq)} stocks")

                for method_name, rank_col in [("Bucket-Split", "rank_best"), ("Mixed-Alpha", "rank_mixed")]:
                    if rank_col not in bq.columns:
                        continue
                    top5 = bq.nsmallest(5, rank_col)
                    parts = []
                    for _, r in top5.iterrows():
                        if has_actual and pd.notna(r.get("y_return")):
                            parts.append(f"{r['tic']}{r['y_return']*100:+.0f}%")
                        else:
                            parts.append(r["tic"])
                    avg = top5["y_return"].mean() * 100 if has_actual and top5["y_return"].notna().any() else float("nan")
                    avg_s = f"avg={avg:+.1f}%" if not np.isnan(avg) else ""
                    print(f"    {method_name:<16} {', '.join(parts):<55} {avg_s}")

        # ---- Overall comparison ----
        if has_actual:
            spy_proxy = mixed["y_return"].dropna().mean() * 100

            print(f"\n{'='*70}")
            print(f"  COMPARISON: Bucket-Split vs Mixed-Alpha vs Uniform α=0.5")
            print(f"{'='*70}")
            print(f"\n  {'Bucket':<15} {'α':<6} {'Bucket-Split':>14} {'Mixed-Alpha':>14} {'Uniform 0.5':>14}")
            print(f"  {'-'*65}")

            totals = {"bs": [], "mixed": [], "uniform": []}
            for bucket in buckets:
                a = BUCKET_ALPHA[bucket]
                for idate in mixed["datadate"].unique():
                    bq = mixed[(mixed["bucket"] == bucket) & (mixed["datadate"] == idate)]
                    if len(bq) == 0:
                        continue
                    bs5 = bq.nsmallest(5, "rank_best")["y_return"].mean() * 100 if "rank_best" in bq.columns else float("nan")
                    mx5 = bq.nsmallest(5, "rank_mixed")["y_return"].mean() * 100
                    uf5 = bq.nsmallest(5, "rank_uniform")["y_return"].mean() * 100
                    totals["bs"].append(bs5)
                    totals["mixed"].append(mx5)
                    totals["uniform"].append(uf5)
                    print(f"  {bucket:<15} {a:<6.1f} {bs5:>+13.1f}% {mx5:>+13.1f}% {uf5:>+13.1f}%")

            bs_avg = np.nanmean(totals["bs"])
            mx_avg = np.nanmean(totals["mixed"])
            uf_avg = np.nanmean(totals["uniform"])
            print(f"  {'-'*65}")
            print(f"  {'OVERALL':<15} {'':6} {bs_avg:>+13.1f}% {mx_avg:>+13.1f}% {uf_avg:>+13.1f}%")
            print(f"  {'SPY≈':<15} {'':6} {spy_proxy:>+13.1f}% {spy_proxy:>+13.1f}% {spy_proxy:>+13.1f}%")
            print(f"  {'ALPHA':<15} {'':6} {bs_avg-spy_proxy:>+12.1f}pp {mx_avg-spy_proxy:>+12.1f}pp {uf_avg-spy_proxy:>+12.1f}pp")

        # ---- Save ----
        mixed_path = os.path.join(args.output_dir, f"{prefix}ml_mixed_alpha_{trade_tag}_{timestamp}.csv")
        mixed.to_csv(mixed_path, index=False)
        print(f"\nSaved: {mixed_path} ({len(mixed)} stocks)")


if __name__ == "__main__":
    main()
