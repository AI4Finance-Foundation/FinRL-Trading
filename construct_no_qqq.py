#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
construct_no_qqq.py
Global-QQQ-removed variant of construct.py

- Removes QQQ from all computations and outputs (KPI summary, single-series charts, comparison charts).
- IR benchmark is fixed to SPX (no CLI choice).
- Everything else remains: SPX baseline + MeanVar/MinVar/Equal/DRL strategies.

Defaults preserved per your setup:
  --freq M
  --rf 0.02
  --mar 0.0
  --start 2015-01-01 --end 2025-06-30
  --out test_back
  --equity-file ./data_processor/sp500_tickers_daily_price_20250712.csv
  --weights-mean ./output/mean_weighted.xlsx
  --weights-min  ./output/minimum_weighted.xlsx
  --weights-equal ./output/equally_weighted.xlsx
  --weights-drl ./output/drl_weight.csv
"""

import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ----------------------- User-editable constant -----------------------
TRANSACTION_COST = 0.001  # 0.1% per turnover on rebalance dates
# ---------------------------------------------------------------------

_PERIODS = {"D": 252, "W": 52, "M": 12, "Q": 4, "Y": 1}

def setup_logging(verbosity:int=1):
    level = logging.INFO if verbosity<=1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------------------- I/O Helpers ----------------------------
def load_index_series(path: str, date_col: str="date", price_col: str="close") -> pd.Series:
    df = pd.read_csv(path)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"Index file {path} must contain columns [{date_col}, {price_col}]")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    s = df[price_col].astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s

def load_equity_prices(path: str, gvkey_col="gvkey", date_col="datadate",
                       prccd_col="prccd", ajexdi_col="ajexdi") -> pd.DataFrame:
    """
    Returns a wide DataFrame indexed by date with columns as gvkey,
    containing adjusted close 'adj_close_q = prccd / ajexdi'.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    for c in [gvkey_col, date_col, prccd_col, ajexdi_col]:
        if c not in df.columns:
            raise ValueError(f"Equity file {path} must contain column '{c}'")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, gvkey_col])
    df["adj_close_q"] = df[prccd_col].astype(float) / df[ajexdi_col].astype(float)
    pivot = df.pivot_table(index=date_col, columns=gvkey_col, values="adj_close_q", aggfunc="last")
    pivot = pivot.sort_index()
    pivot = pivot.replace([np.inf, -np.inf], np.nan)
    return pivot

def load_weights(path: str, gvkey_col="gvkey", date_col="trade_date") -> pd.DataFrame:
    """
    Robust loader: accepts either 'weights' or 'weight' column name; renames to 'weights'.
    Disallows negative weights. Normalizes per-trade_date to sum==1 with a warning if needed.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # detect weight column name
    weight_col = None
    for cand in ["weights", "weight", "Weight", "WEIGHT"]:
        if cand in df.columns:
            weight_col = cand
            break
    if weight_col is None:
        raise ValueError(f"Weight file {path} must contain a 'weights' (or 'weight') column")

    for c in [gvkey_col, date_col, weight_col]:
        if c not in df.columns:
            raise ValueError(f"Weight file {path} must contain columns '{gvkey_col}', '{date_col}', '{weight_col}'")
    df[date_col] = pd.to_datetime(df[date_col])
    df["weights"] = df[weight_col].astype(float)

    # constraints
    if (df["weights"] < 0).any():
        bad = df[df["weights"] < 0].head()
        raise ValueError(f"Negative weights encountered (short-selling not allowed). Sample:\n{bad}")

    # sum-to-one per trade_date (normalize if not)
    def _normalize(g):
        s = g["weights"].sum()
        if not np.isclose(s, 1.0, atol=1e-6):
            logging.warning(f"[normalize weights] {os.path.basename(path)} {g.name.date()} sum={s:.6f} → normalized to 1.0")
            g["weights"] = g["weights"] / s if s != 0 else 0.0
        return g
    df = df.groupby(date_col, group_keys=False).apply(_normalize)
    return df[[gvkey_col, date_col, "weights"]]

def align_calendar_forward_fill(objs: List[pd.DataFrame] | List[pd.Series]) -> List[pd.DataFrame] | List[pd.Series]:
    """Align all to the union of dates and forward-fill."""
    idx = None
    for o in objs:
        oi = o.index if isinstance(o, (pd.DataFrame, pd.Series)) else None
        idx = oi if idx is None else idx.union(oi)
    idx = idx.sort_values()
    out = []
    for o in objs:
        o2 = o.reindex(idx).sort_index()
        o2 = o2.ffill()
        out.append(o2)
    return out

# -------------------------- Weight Alignment -------------------------
def unify_quarterly_weights(weights_map: Dict[str, pd.DataFrame], anchor_key: str,
                            gvkey_col="gvkey", date_col="trade_date", weight_col="weights") -> Dict[str, pd.DataFrame]:
    """
    Align all strategy weight tables to the anchor strategy's trade_date set.
    After alignment, weights per date are renormalized to sum=1.
    """
    if anchor_key not in weights_map:
        raise ValueError(f"Anchor key '{anchor_key}' not in weights_map keys: {list(weights_map.keys())}")
    anchor_dates = sorted(weights_map[anchor_key][date_col].dropna().unique())

    out = {}
    for k, wdf in weights_map.items():
        w = wdf[wdf[date_col].isin(anchor_dates)].copy()
        if w.empty:
            logging.warning(f"[unify] weights for '{k}' empty after aligning to anchor dates")
        def _renorm(g):
            s = g[weight_col].sum()
            if not np.isclose(s, 1.0, atol=1e-6):
                logging.warning(f"[unify normalize] strategy={k} date={g.name.date()} sum={s:.6f} → normalized to 1.0")
                g[weight_col] = g[weight_col] / s if s != 0 else 0.0
            return g
        w = w.groupby(date_col, group_keys=False).apply(_renorm)
        out[k] = w
    return out

# ------------------------ Portfolio Construction ---------------------
def build_portfolio_daily_returns(price_wide: pd.DataFrame,
                                  weights_df: pd.DataFrame,
                                  gvkey_col="gvkey", date_col="trade_date",
                                  weight_col="weights",
                                  cost: float = TRANSACTION_COST) -> pd.Series:
    """
    Build daily returns of a rebalanced long-only portfolio using adjusted prices.
    """
    asset_ret = price_wide.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tdates = sorted(weights_df[date_col].unique())
    if len(tdates) == 0:
        raise ValueError("No trade_date in weights_df")
    port_ret = pd.Series(index=asset_ret.index, dtype=float)

    prev_w = None
    for i, t in enumerate(tdates):
        w_slice = weights_df[weights_df[date_col] == t][[gvkey_col, weight_col]].set_index(gvkey_col)[weight_col]
        w = w_slice.reindex(asset_ret.columns).fillna(0.0).astype(float)

        start_idx = asset_ret.index.get_indexer([t], method='nearest')[0]
        start_date = asset_ret.index[start_idx]
        end_date = tdates[i+1] if i < len(tdates)-1 else asset_ret.index[-1]
        mask = (asset_ret.index > start_date) & (asset_ret.index <= end_date)
        sub_ret = asset_ret.loc[mask]

        turnover = w.abs().sum() if prev_w is None else (w - prev_w).abs().sum()
        day_cost = cost * turnover

        first = True
        for dt, row in sub_ret.iterrows():
            r = float(np.nansum(w.values * row.values))
            if first:
                r -= day_cost
                first = False
            port_ret.loc[dt] = r

        prev_w = w

    return port_ret.dropna()

# ----------------------------- Frequency -----------------------------
def resample_returns(ret_series: pd.Series, freq: str = "M", how: str = "compound") -> pd.Series:
    if freq not in _PERIODS:
        raise ValueError("freq must be in {D,W,M,Q,Y}")
    rs_freq = {"M":"M", "Q":"Q", "Y":"Y", "W":"W", "D":"D"}[freq]
    if how == "compound":
        return (1.0 + ret_series).resample(rs_freq).prod().sub(1.0)
    elif how == "sum":
        return ret_series.resample(rs_freq).sum()
    else:
        raise ValueError("how must be 'compound' or 'sum'")

def _per_period_rate(annual_rate: float, freq: str) -> float:
    n = _PERIODS[freq]
    return (1.0 + annual_rate) ** (1.0 / n) - 1.0

# ------------------------------ Metrics ------------------------------
def compute_drawdown(equity: pd.Series) -> Tuple[pd.Series, int]:
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1.0
    dur = (dd < 0).astype(int)
    run = dur.groupby((dur != dur.shift()).cumsum()).cumsum()
    dd_duration = int(run.max()) if not run.empty else 0
    return dd, dd_duration

def compute_metrics(ret: pd.Series,
                    freq: str = "M",
                    rf_annual: float = 0.02,
                    benchmark: Optional[pd.Series] = None,
                    mar_annual: float = 0.0,
                    var_levels=(0.95, 0.99),
                    label: str = "Strategy") -> dict:
    assert freq in _PERIODS, "freq must be one of D/W/M/Q/Y"
    n = _PERIODS[freq]
    ret = ret.dropna()
    if ret.empty:
        raise ValueError(f"Empty return series for {label}")

    mu = ret.mean()
    sd = ret.std(ddof=0)
    mar_p = _per_period_rate(mar_annual, freq)

    ann_ret = mu * n
    ann_vol = sd * np.sqrt(n)

    equity = (1.0 + ret).cumprod()
    cum_ret = equity.iloc[-1] - 1.0

    downside = np.clip(ret - mar_p, a_min=None, a_max=0.0)
    downside_sd = downside.std(ddof=0)
    ann_downside = downside_sd * np.sqrt(n)

    sharpe = np.nan if ann_vol == 0 else (ann_ret - rf_annual) / ann_vol
    sortino = np.nan if ann_downside == 0 else (ann_ret - mar_annual) / ann_downside

    dd, dd_dur = compute_drawdown(equity)
    max_dd = dd.min() if not dd.empty else 0.0
    calmar = np.nan if max_dd == 0 else ann_ret / abs(max_dd)

    ir = np.nan
    if benchmark is not None:
        diff = (ret - benchmark.reindex_like(ret)).dropna()
        te = diff.std(ddof=0) * np.sqrt(n)
        ann_alpha = diff.mean() * n
        ir = np.nan if te == 0 else ann_alpha / te

    losses = -ret
    var_cvar = {}
    for q in var_levels:
        var_q = losses.quantile(q)
        cvar_q = losses[losses >= var_q].mean()
        var_cvar[f"VaR_{int(q*100)}"] = float(var_q)
        var_cvar[f"CVaR_{int(q*100)}"] = float(cvar_q)

    return {
        "label": label, "freq": freq, "n_periods": n,
        "cum_return": float(cum_ret),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_dd),
        "max_dd_duration": int(dd_dur),
        "ir": float(ir),
        **var_cvar,
        "_equity": equity,
        "_dd": dd,
        "_ret": ret,
    }

# ------------------------------ Plotting -----------------------------
def save_equity_and_dd_plots(label: str, eq: pd.Series, dd: pd.Series, out_dir: str, suffix: str=""):
    fn_suffix = f"_{suffix}" if suffix else ""
    # Equity
    plt.figure()
    eq.plot(title=f"{label} – Equity Curve{(' ('+suffix+')') if suffix else ''}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{label.lower()}_equity{fn_suffix}.png"), dpi=150)
    plt.close()

    # Drawdown
    plt.figure()
    dd.plot(title=f"{label} – Drawdown{(' ('+suffix+')') if suffix else ''}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{label.lower()}_drawdown{fn_suffix}.png"), dpi=150)
    plt.close()

def save_timeseries_csv(label: str, eq: pd.Series, dd: pd.Series, out_dir: str, suffix: str=""):
    fn_suffix = f"_{suffix}" if suffix else ""
    eq.to_frame("equity").to_csv(os.path.join(out_dir, f"{label.lower()}_equity{fn_suffix}.csv"))
    dd.to_frame("drawdown").to_csv(os.path.join(out_dir, f"{label.lower()}_drawdown{fn_suffix}.csv"))

# ----------------------- Rolling Comparison (No QQQ) -----------------
_ROLL_WINDOWS = {"M": 12, "Q": 8, "Y": 3}

def _plot_multi(df: pd.DataFrame, title: str, out_path: str):
    plt.figure()
    for col in df.columns:
        df[col].plot(label=col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def generate_comparison_charts_no_qqq(daily_returns_map: Dict[str, pd.Series], out_dir: str,
                                      rf_annual: float, mar_annual: float):
    """
    Build rolling comparison charts for KPIs across SPX, MeanVar, MinVar, Equal, DRL.
    Frequencies: M/Q/Y; Windows: 12M / 8Q / 3Y.
    """
    labels = ["SPX","MeanVar","MinVar","Equal","DRL"]
    for freq, win in _ROLL_WINDOWS.items():
        n = _PERIODS[freq]
        # Resample
        ret_map = {k: resample_returns(daily_returns_map[k], freq=freq) for k in labels}

        # Equity comparison
        eq_df = pd.DataFrame({k: (1.0 + ret_map[k]).cumprod() for k in labels})
        _plot_multi(eq_df, f"Equity Curve Comparison ({freq})", os.path.join(out_dir, f"comparison_equity_{freq}.png"))
        eq_df.to_csv(os.path.join(out_dir, f"comparison_equity_{freq}.csv"))

        # Rolling metrics
        idx = ret_map["SPX"].index
        ann_ret_df = pd.DataFrame(index=idx)
        ann_vol_df = pd.DataFrame(index=idx)
        sharpe_df  = pd.DataFrame(index=idx)
        sortino_df = pd.DataFrame(index=idx)
        calmar_df  = pd.DataFrame(index=idx)
        maxdd_df   = pd.DataFrame(index=idx)
        var95_df   = pd.DataFrame(index=idx)
        var99_df   = pd.DataFrame(index=idx)
        ir_spx_df  = pd.DataFrame(index=idx)

        for k in labels:
            r = ret_map[k].dropna()
            if r.empty: 
                continue
            roll = r.rolling(win, min_periods=win)
            ann_ret_df[k] = roll.apply(lambda x: np.mean(x)*n, raw=True)
            ann_vol_df[k] = roll.apply(lambda x: np.std(x, ddof=0)*np.sqrt(n), raw=True)
            # Sharpe
            sharpe_df[k]  = roll.apply(lambda x: ((np.mean(x)*n - rf_annual) / (np.std(x, ddof=0)*np.sqrt(n) if np.std(x, ddof=0)>0 else np.nan)), raw=True)
            # Sortino
            mar_p = _per_period_rate(mar_annual, freq)
            def _sort(x):
                downside = np.minimum(x - mar_p, 0.0)
                sd = np.std(downside, ddof=0)
                return ((np.mean(x)*n - mar_annual) / (sd*np.sqrt(n))) if sd>0 else np.nan
            sortino_df[k] = roll.apply(_sort, raw=True)
            # Calmar
            def _calmar(x):
                eq = np.cumprod(1.0 + x)
                peak = np.maximum.accumulate(eq)
                dd = eq/peak - 1.0
                mdd = dd.min() if dd.size else 0.0
                return (np.mean(x)*n)/abs(mdd) if mdd!=0 else np.nan
            calmar_df[k] = roll.apply(_calmar, raw=True)
            # Max drawdown
            def _maxdd(x):
                eq = np.cumprod(1.0 + x)
                peak = np.maximum.accumulate(eq)
                dd = eq/peak - 1.0
                return dd.min() if dd.size else 0.0
            maxdd_df[k] = roll.apply(_maxdd, raw=True)
            # VaR
            var95_df[k] = roll.apply(lambda x: np.quantile(-x, 0.95), raw=True)
            var99_df[k] = roll.apply(lambda x: np.quantile(-x, 0.99), raw=True)
            # IR vs SPX
            if k != "SPX":
                diff_spx = (r - ret_map["SPX"].reindex_like(r)).dropna()
                roll_d = diff_spx.rolling(win, min_periods=win)
                ir_spx_df[k] = roll_d.apply(lambda x: (np.mean(x)*n) / (np.std(x, ddof=0)*np.sqrt(n) if np.std(x, ddof=0)>0 else np.nan), raw=True)

        # Export
        metrics_to_export = [
            ("ann_return", ann_ret_df),
            ("ann_vol", ann_vol_df),
            ("sharpe", sharpe_df),
            ("sortino", sortino_df),
            ("calmar", calmar_df),
            ("max_drawdown", maxdd_df),
            ("VaR_95", var95_df),
            ("VaR_99", var99_df),
            ("IR_vs_SPX", ir_spx_df),
        ]
        for metric_name, df in metrics_to_export:
            df = df[[c for c in labels if c in df.columns]]
            csv_path = os.path.join(out_dir, f"comparison_{metric_name}_{freq}.csv")
            png_path = os.path.join(out_dir, f"comparison_{metric_name}_{freq}.png")
            df.to_csv(csv_path)
            _plot_multi(df, f"{metric_name} (Rolling {win} {freq})", png_path)

# ------------------------------- Main --------------------------------
def main():
    parser = argparse.ArgumentParser(description="Backtest & KPI tool (QQQ globally removed)")
    parser.add_argument("--freq", choices=list(_PERIODS.keys()), default="M", help="frequency for KPI annualization")
    parser.add_argument("--rf", type=float, default=0.02, help="annual risk-free rate")
    parser.add_argument("--mar", type=float, default=0.0, help="annual minimum acceptable return (Sortino MAR)")
    parser.add_argument("--cost", type=float, default=None, help="(ignored) static TRANSACTION_COST in code")
    parser.add_argument("--start", type=str, default="2015-01-01", help="start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-06-30", help="end date YYYY-MM-DD")
    parser.add_argument("--out", type=str, default="test_back", help="output directory")
    parser.add_argument("--verbosity", type=int, default=1, help="0=warning,1=info,2=debug")

    # input files (configurable)
    parser.add_argument("--index-spx", type=str, default="SPX.csv")
    parser.add_argument("--equity-file", type=str, default="./data_processor/sp500_tickers_daily_price_20250712.csv")
    parser.add_argument("--weights-mean", type=str, default="./output/mean_weighted.xlsx")
    parser.add_argument("--weights-min", type=str, default="./output/minimum_weighted.xlsx")
    parser.add_argument("--weights-equal", type=str, default="./output/equally_weighted.xlsx")
    parser.add_argument("--weights-drl", type=str, default="./output/drl_weight.csv")
    parser.add_argument("--anchor", type=str, default="MeanVar", choices=["MeanVar","MinVar","Equal","DRL"],
                        help="anchor strategy for aligning quarterly trade dates")

    args = parser.parse_args()
    setup_logging(args.verbosity)

    if args.cost is not None:
        logging.warning("`--cost` is ignored. Using static TRANSACTION_COST=%.6f in source.", TRANSACTION_COST)

    os.makedirs(args.out, exist_ok=True)

    # ------------------- Load prices -------------------
    spx = load_index_series(args.index_spx, date_col="date", price_col="close")
    eq_prices = load_equity_prices(args.equity_file)  # wide df

    # align calendar
    spx, eq_prices = align_calendar_forward_fill([spx, eq_prices])

    # date range clip
    if args.start:
        start = pd.to_datetime(args.start)
        spx = spx.loc[spx.index >= start]
        eq_prices = eq_prices.loc[eq_prices.index >= start]
    if args.end:
        end = pd.to_datetime(args.end)
        spx = spx.loc[spx.index <= end]
        eq_prices = eq_prices.loc[eq_prices.index <= end]

    # ------------------- Load weights ------------------
    w_mean  = load_weights(args.weights_mean)
    w_min   = load_weights(args.weights_min)
    w_equal = load_weights(args.weights_equal)
    w_drl   = load_weights(args.weights_drl)

    weights_map = {"MeanVar": w_mean, "MinVar": w_min, "Equal": w_equal, "DRL": w_drl}
    weights_map = unify_quarterly_weights(weights_map, anchor_key=args.anchor)

    # ------------------- Build portfolios --------------
    spx_ret_daily = spx.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean_ret_daily  = build_portfolio_daily_returns(eq_prices, weights_map["MeanVar"])
    min_ret_daily   = build_portfolio_daily_returns(eq_prices, weights_map["MinVar"])
    equal_ret_daily = build_portfolio_daily_returns(eq_prices, weights_map["Equal"])
    drl_ret_daily   = build_portfolio_daily_returns(eq_prices, weights_map["DRL"])

    # ------------------- Resample to KPI freq ----------
    freq = args.freq
    returns_dict = {
        "SPX": resample_returns(spx_ret_daily, freq=freq),
        "MeanVar": resample_returns(mean_ret_daily, freq=freq),
        "MinVar": resample_returns(min_ret_daily, freq=freq),
        "Equal": resample_returns(equal_ret_daily, freq=freq),
        "DRL": resample_returns(drl_ret_daily, freq=freq),
    }

    # Benchmark for IR (fixed to SPX)
    benchmark_series = returns_dict["SPX"]

    # ------------------- Compute KPIs ------------------
    summary_rows = []
    metrics_map = {}
    for label in ["MeanVar","MinVar","Equal","SPX","DRL"]:
        series = returns_dict[label]
        bmk = None if label == "SPX" else benchmark_series
        m = compute_metrics(series, freq=freq, rf_annual=args.rf, mar_annual=args.mar, benchmark=bmk, label=label)
        metrics_map[label] = m
        keep = {k: m[k] for k in [
            "label","freq","cum_return","ann_return","ann_vol","sharpe","sortino","calmar",
            "max_drawdown","max_dd_duration","ir","VaR_95","CVaR_95","VaR_99","CVaR_99"
        ]}
        summary_rows.append(keep)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.out, "risk_metrics_summary.csv"), index=False)

    # ------------------- Save KPI-frequency charts -----
    for label, m in metrics_map.items():
        eq = m["_equity"]
        dd = m["_dd"]
        save_timeseries_csv(label, eq, dd, args.out, suffix=freq)
        save_equity_and_dd_plots(label, eq, dd, args.out, suffix=freq)

    # ------------------- Also export M/Q/Y chart sets --
    chart_freqs = ["M","Q","Y"]
    for chart_f in chart_freqs:
        for label, daily_series in {
            "SPX": spx_ret_daily,
            "MeanVar": mean_ret_daily, "MinVar": min_ret_daily,
            "Equal": equal_ret_daily, "DRL": drl_ret_daily,
        }.items():
            rs = resample_returns(daily_series, freq=chart_f)
            eq = (1.0 + rs).cumprod()
            dd, _ = compute_drawdown(eq)
            save_timeseries_csv(label, eq, dd, args.out, suffix=chart_f)
            save_equity_and_dd_plots(label, eq, dd, args.out, suffix=chart_f)

    # ------------------- Comparison charts (NO QQQ) ----
    daily_map = {
        "SPX": spx_ret_daily,
        "MeanVar": mean_ret_daily, "MinVar": min_ret_daily,
        "Equal": equal_ret_daily, "DRL": drl_ret_daily,
    }
    generate_comparison_charts_no_qqq(daily_map, args.out, rf_annual=args.rf, mar_annual=args.mar)

    logging.info("Done. Summary written to %s", os.path.join(args.out, "risk_metrics_summary.csv"))

if __name__ == "__main__":
    main()
