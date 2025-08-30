"""
Step2_preprocess_fundmental_data

CLI/script version of the original notebook `Step2_preprocess_fundmental_data.ipynb`.

Behavior parity with the notebook:
- Load fundamentals CSV and daily price CSV
- Align quarterly report dates to standardized trade dates per quarter
- Compute adjusted quarterly close price (adj_close_q)
- Map `tic` to `gvkey` using daily price data; filter unmatched tickers
- Compute next-quarter log return (y_return) per `gvkey`
- Engineer financial ratios; build final dataset; clean NaN/inf; format dates
- Save final CSV and per-sector Excel outputs

No hard-coded paths: use CLI arguments for all inputs/outputs.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def _tqdm(iterable=None, total=None, desc=None):  # type: ignore
        return iterable if iterable is not None else range(total or 0)


def handle_nan(df: pd.DataFrame, features_column_financial: List[str]) -> pd.DataFrame:
    """Replicate notebook's NaN/inf handling.

    - Drop rows where adj_close_q == 0
    - Coerce selected columns to numeric
    - Replace non-finite values with NaN, then drop rows with NaN
    - Drop any feature columns that still contain non-finite values
    - Reset index
    """
    df = df.drop(list(df[df.adj_close_q == 0].index)).reset_index(drop=True)
    df["y_return"] = pd.to_numeric(df["y_return"], errors="coerce")
    for col in features_column_financial:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["y_return"].replace([np.nan, np.inf, -np.inf], np.nan, inplace=True)
    df[features_column_financial].replace([np.nan, np.inf, -np.inf], np.nan, inplace=True)

    dropped_col: List[str] = []
    for col in list(features_column_financial):
        # If any remaining non-finite, drop this feature column
        if np.any(~np.isfinite(df[col])):
            df.drop(columns=[col], axis=1, inplace=True)
            dropped_col.append(col)
    df.dropna(axis=0, inplace=True)
    df = df.reset_index(drop=True)
    print("dropped_col: ", dropped_col)
    return df


def _require_columns(df: pd.DataFrame, required: List[str], source_name: str) -> None:
    """Validate that a DataFrame contains required columns; raise with clear message if not.

    Parameters
    - df: DataFrame to validate
    - required: list of required column names
    - source_name: human-readable label for the DataFrame (e.g., 'fundamentals CSV')
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        available_preview = ", ".join(list(df.columns)[:30])
        raise ValueError(
            f"Missing required columns in {source_name}: {missing}.\n"
            f"Available columns sample: [{available_preview}]"
        )


def _align_to_quarter_trade_dates(datadate_values: List[int]) -> List[pd.Timestamp]:
    """Align raw datadate integers to quarter trade dates as in the notebook.

    The notebook uses integer operations with thresholds (301, 601, 901, 1201)
    to map into 03-01, 06-01, 09-01, 12-01 of the appropriate year.
    """
    times: List[int] = [int(x) for x in datadate_values]
    for i in range(len(times)):
        quarter = times[i] - int(times[i] / 10000) * 10000
        if 1201 < quarter:
            times[i] = int(times[i] / 10000 + 1) * 10000 + 301
        if quarter <= 301:
            times[i] = int(times[i] / 10000) * 10000 + 301
        if 301 < quarter <= 601:
            times[i] = int(times[i] / 10000) * 10000 + 601
        if 601 < quarter <= 901:
            times[i] = int(times[i] / 10000) * 10000 + 901
        if 901 < quarter <= 1201:
            times[i] = int(times[i] / 10000) * 10000 + 1201
    # Convert to pandas Timestamp with explicit format
    return list(pd.to_datetime(times, format="%Y%m%d"))


def _map_tic_to_gvkey(df_daily_price: pd.DataFrame) -> Dict[str, str]:
    """Create a mapping from ticker (tic) to gvkey using daily price data."""
    tic_to_gvkey: Dict[str, str] = {}
    for tic, df_group in df_daily_price.groupby("tic"):
        tic_to_gvkey[tic] = df_group.gvkey.iloc[0]
    return tic_to_gvkey


def _compute_next_quarter_log_return(fund_df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Compute next-quarter log return per gvkey group using adj_close_q."""
    grouped = list(fund_df.groupby("gvkey"))
    if len(grouped) == 0:
        raise ValueError(
            "No gvkey groups found after mapping. Ensure tickers in fundamentals match those in daily price CSV."
        )
    processed_groups: List[pd.DataFrame] = []
    iterator = grouped
    if show_progress:
        iterator = _tqdm(grouped, total=len(grouped), desc="Compute next-q returns")
    for gvkey, df_group in iterator:
        df_local = df_group.copy().reset_index(drop=True)
        df_local = df_local.sort_values("date")
        df_local["y_return"] = np.log(df_local["adj_close_q"].shift(-1) / df_local["adj_close_q"])
        processed_groups.append(df_local)
    return pd.concat(processed_groups, axis=0, ignore_index=True)


def run(
    fundamentals_csv: str,
    daily_price_csv: str,
    output_csv: str,
    sectors_output_dir: str,
    raw_output_csv: str | None = None,
    show_progress: bool = True,
) -> None:
    # Load data
    fund_df = pd.read_csv(
        fundamentals_csv,
        low_memory=False,
        dtype={"tic": str},
    )
    df_daily_price = pd.read_csv(
        daily_price_csv,
        low_memory=False,
        dtype={"tic": str, "gvkey": str},
    )

    # Normalize identifiers and dtypes early to avoid mixed-types warnings
    if "tic" in df_daily_price.columns:
        df_daily_price["tic"] = df_daily_price["tic"].astype(str).str.upper().str.strip()
    if "gvkey" in df_daily_price.columns:
        # Remove accidental trailing decimal artifacts like "12345.0"
        df_daily_price["gvkey"] = (
            df_daily_price["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        )
    if "tic" in fund_df.columns:
        fund_df["tic"] = fund_df["tic"].astype(str).str.upper().str.strip()

    # 1.1 Use Trade date instead of quarterly report date
    _require_columns(fund_df, ["datadate", "tic"], source_name="fundamentals CSV")
    # Robustly coerce datadate to numeric YYYYMMDD before alignment
    fund_df["datadate"] = pd.to_numeric(fund_df["datadate"], errors="coerce").astype("Int64")
    fund_df = fund_df[fund_df["datadate"].notna()].copy()
    aligned_trade_dates = _align_to_quarter_trade_dates(list(fund_df["datadate"].astype(int)))
    fund_df["tradedate"] = aligned_trade_dates

    # Match fundamentals to daily prices, preferring gvkey when available
    _require_columns(df_daily_price, ["tic", "gvkey"], source_name="daily price CSV")

    matched_via: str = ""
    if "gvkey" in fund_df.columns and fund_df["gvkey"].notna().any():
        # Normalize fundamentals gvkey as string without trailing .0, to match daily
        fund_df["gvkey"] = (
            fund_df["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        )
        # Filter by gvkey intersection
        daily_gvkeys = set(pd.Series(df_daily_price["gvkey"], dtype=str).dropna().unique())
        fund_df = fund_df[fund_df["gvkey"].isin(daily_gvkeys)].copy()

        # Ensure we have a reasonable ticker column; backfill tic from daily if missing or blank
        if "tic" not in fund_df.columns:
            fund_df["tic"] = np.nan
        gvkey_to_tic = (
            df_daily_price.dropna(subset=["gvkey", "tic"])  # safety
            .drop_duplicates(["gvkey"])  # first occurrence
            .set_index("gvkey")["tic"].to_dict()
        )
        fund_df["tic"] = fund_df.apply(
            lambda r: r["tic"] if isinstance(r["tic"], str) and r["tic"].strip() != "" else gvkey_to_tic.get(str(r["gvkey"]), np.nan),
            axis=1,
        )
        matched_via = "gvkey"
    else:
        # Fallback: map by ticker to gvkey from daily
        tic_to_gvkey = _map_tic_to_gvkey(df_daily_price)
        fund_df = fund_df[np.isin(fund_df.tic, list(tic_to_gvkey.keys()))].copy()
        fund_df["gvkey"] = [tic_to_gvkey[x] for x in fund_df["tic"]]
        matched_via = "tic"

    # Validate we still have data after matching
    if fund_df.empty:
        if matched_via == "gvkey":
            fundament_keys = set(pd.Series(fund_df.get("gvkey", pd.Series(dtype=str))).dropna().unique())
            price_keys = set(pd.Series(df_daily_price.get("gvkey", pd.Series(dtype=str))).dropna().unique())
            overlap = fundament_keys.intersection(price_keys)
            raise ValueError(
                "No rows remain after matching by gvkey. "
                f"Fundamentals gvkeys: {len(fundament_keys)}, Daily price gvkeys: {len(price_keys)}, Overlap: {len(overlap)}.\n"
                "Ensure both files use consistent gvkey identifiers."
            )
        else:
            fundament_tics = set(pd.Series(fund_df.get("tic", pd.Series(dtype=str))).dropna().unique())
            price_tics = set(pd.Series(df_daily_price.get("tic", pd.Series(dtype=str))).dropna().unique())
            overlap = fundament_tics.intersection(price_tics)
            raise ValueError(
                "No rows remain after matching tickers to gvkey. "
                f"Fundamentals tickers: {len(fundament_tics)}, Daily price tickers: {len(price_tics)}, Overlap: {len(overlap)}.\n"
                "Check that both files use the same ticker symbols."
            )

    # Date preparation and deduplication
    fund_df["date"] = pd.to_datetime(fund_df["tradedate"], errors="coerce")
    fund_df.drop_duplicates(["date", "gvkey"], keep="last", inplace=True)

    # Ensure numeric dtypes for all columns used in arithmetic below
    numeric_cols = [
        "prccq",
        "adjex",
        "epspxq",
        "revtq",
        "cshoq",
        "atq",
        "ltq",
        "teqq",
        "epspiy",
        "ceqq",
        "dvpspq",
        "actq",
        "lctq",
        "cheq",
        "rectq",
        "cogsq",
        "invtq",
        "apq",
        "dlttq",
        "dlcq",
        "niq",
        "oiadpq",
        "gsector",
    ]
    for col in numeric_cols:
        if col in fund_df.columns:
            fund_df[col] = pd.to_numeric(fund_df[col], errors="coerce")

    # 1.2 Adjusted close price (after numeric coercion), guard against div-by-zero
    if "prccq" in fund_df.columns and "adjex" in fund_df.columns:
        fund_df["adj_close_q"] = fund_df["prccq"] / fund_df["adjex"].replace({0: np.nan})
    else:
        _require_columns(fund_df, ["prccq", "adjex"], source_name="fundamentals CSV for adj_close_q")

    # Next quarter's return
    fund_df = _compute_next_quarter_log_return(fund_df, show_progress=show_progress)

    # 1.3 Calculate Financial Ratios (valuation)
    fund_df["pe"] = fund_df.prccq / fund_df.epspxq
    fund_df["ps"] = fund_df.prccq / (fund_df.revtq / fund_df.cshoq)
    fund_df["pb"] = fund_df.prccq / ((fund_df.atq - fund_df.ltq) / fund_df.cshoq)

    # Select core fields (keep exact order to match notebook indexing assumptions)
    items = [
        "date",
        "gvkey",
        "tic",
        "gsector",
        "oiadpq",
        "revtq",
        "niq",
        "atq",
        "teqq",
        "epspiy",
        "ceqq",
        "cshoq",
        "dvpspq",
        "actq",
        "lctq",
        "cheq",
        "rectq",
        "cogsq",
        "invtq",
        "apq",
        "dlttq",
        "dlcq",
        "ltq",
        "pe",
        "ps",
        "pb",
        "adj_close_q",
        "y_return",
    ]
    # Ensure all required fields exist before selection
    _require_columns(fund_df, items, source_name="fundamentals (post-processing)")
    fund_data = fund_df[items].copy()

    # Rename for readability
    fund_data = fund_data.rename(
        columns={
            "oiadpq": "op_inc_q",
            "revtq": "rev_q",
            "niq": "net_inc_q",
            "atq": "tot_assets",
            "teqq": "sh_equity",
            "epspiy": "eps_incl_ex",
            "ceqq": "com_eq",
            "cshoq": "sh_outstanding",
            "dvpspq": "div_per_sh",
            "actq": "cur_assets",
            "lctq": "cur_liabilities",
            "cheq": "cash_eq",
            "rectq": "receivables",
            "cogsq": "cogs_q",
            "invtq": "inventories",
            "apq": "payables",
            "dlttq": "long_debt",
            "dlcq": "short_debt",
            "ltq": "tot_liabilities",
        }
    )

    # Build Series for columns used downstream
    date = fund_data["date"].to_frame("date").reset_index(drop=True)
    tic = fund_data["tic"].to_frame("tic").reset_index(drop=True)
    gvkey = fund_data["gvkey"].to_frame("gvkey").reset_index(drop=True)
    adj_close_q = fund_data["adj_close_q"].to_frame("adj_close_q").reset_index(drop=True)
    y_return = fund_data["y_return"].to_frame("y_return").reset_index(drop=True)
    gsector = fund_data["gsector"].to_frame("gsector").reset_index(drop=True)
    pe = fund_data["pe"].to_frame("pe").reset_index(drop=True)
    ps = fund_data["ps"].to_frame("ps").reset_index(drop=True)
    pb = fund_data["pb"].to_frame("pb").reset_index(drop=True)

    # Profitability ratios (3-quarter rolling sums if same gvkey)
    def _rolling_ratio(numerator_col: str, denominator_col: str, name: str) -> pd.DataFrame:
        series = pd.Series(np.empty(fund_data.shape[0], dtype=object), name=name)
        iterator = range(0, fund_data.shape[0])
        if show_progress:
            iterator = _tqdm(iterator, total=fund_data.shape[0], desc=f"{name}")
        for i in iterator:
            if i - 3 < 0:
                series[i] = np.nan
            elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:  # column index 1 is 'gvkey'
                series.iloc[i] = np.nan
            else:
                series.iloc[i] = (
                    np.sum(fund_data[numerator_col].iloc[i - 3 : i])
                    / np.sum(fund_data[denominator_col].iloc[i - 3 : i])
                )
        return pd.Series(series).to_frame().reset_index(drop=True)

    OPM = _rolling_ratio("op_inc_q", "rev_q", "OPM")
    NPM = _rolling_ratio("net_inc_q", "rev_q", "NPM")

    # ROA, ROE use current quarter denominators (not rolling)
    def _ratio_to_current(numerator_col: str, denom_col: str, name: str) -> pd.DataFrame:
        series = pd.Series(np.empty(fund_data.shape[0], dtype=object), name=name)
        iterator = range(0, fund_data.shape[0])
        if show_progress:
            iterator = _tqdm(iterator, total=fund_data.shape[0], desc=f"{name}")
        for i in iterator:
            if i - 3 < 0:
                series[i] = np.nan
            elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
                series.iloc[i] = np.nan
            else:
                series.iloc[i] = (
                    np.sum(fund_data[numerator_col].iloc[i - 3 : i]) / fund_data[denom_col].iloc[i]
                )
        return pd.Series(series).to_frame().reset_index(drop=True)

    ROA = _ratio_to_current("net_inc_q", "tot_assets", "ROA")
    ROE = _ratio_to_current("net_inc_q", "sh_equity", "ROE")

    # Per-share and liquidity ratios
    EPS = fund_data["eps_incl_ex"].to_frame("EPS").reset_index(drop=True)
    BPS = (fund_data["com_eq"] / fund_data["sh_outstanding"]).to_frame("BPS").reset_index(drop=True)
    DPS = fund_data["div_per_sh"].to_frame("DPS").reset_index(drop=True)

    cur_ratio = (fund_data["cur_assets"] / fund_data["cur_liabilities"]).to_frame("cur_ratio").reset_index(drop=True)
    quick_ratio = (
        (fund_data["cash_eq"] + fund_data["receivables"]) / fund_data["cur_liabilities"]
    ).to_frame("quick_ratio").reset_index(drop=True)
    cash_ratio = (fund_data["cash_eq"] / fund_data["cur_liabilities"]).to_frame("cash_ratio").reset_index(drop=True)

    # Efficiency ratios
    def _rolling_to_current(numerator_col: str, denom_col: str, name: str) -> pd.DataFrame:
        series = pd.Series(np.empty(fund_data.shape[0], dtype=object), name=name)
        iterator = range(0, fund_data.shape[0])
        if show_progress:
            iterator = _tqdm(iterator, total=fund_data.shape[0], desc=f"{name}")
        for i in iterator:
            if i - 3 < 0:
                series[i] = np.nan
            elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
                series.iloc[i] = np.nan
            else:
                series.iloc[i] = np.sum(fund_data[numerator_col].iloc[i - 3 : i]) / fund_data[denom_col].iloc[i]
        return pd.Series(series).to_frame().reset_index(drop=True)

    inv_turnover = _rolling_to_current("cogs_q", "inventories", "inv_turnover")
    acc_rec_turnover = _rolling_to_current("rev_q", "receivables", "acc_rec_turnover")
    acc_pay_turnover = _rolling_to_current("cogs_q", "payables", "acc_pay_turnover")

    # Leverage ratios
    debt_ratio = (fund_data["tot_liabilities"] / fund_data["tot_assets"]).to_frame("debt_ratio").reset_index(drop=True)
    debt_to_equity = (fund_data["tot_liabilities"] / fund_data["sh_equity"]).to_frame("debt_to_equity").reset_index(drop=True)

    # Merge all ratios
    ratios = pd.concat(
        [
            date,
            gvkey,
            tic,
            gsector,
            adj_close_q,
            y_return,
            OPM,
            NPM,
            ROA,
            ROE,
            EPS,
            BPS,
            DPS,
            cur_ratio,
            quick_ratio,
            cash_ratio,
            inv_turnover,
            acc_rec_turnover,
            acc_pay_turnover,
            debt_ratio,
            debt_to_equity,
            pe,
            ps,
            pb,
        ],
        axis=1,
    ).reset_index(drop=True)

    # Replace NAs infinite values with zero (raw stage)
    final_ratios = ratios.copy()
    final_ratios = final_ratios.fillna(0)
    final_ratios = final_ratios.replace(np.inf, 0)

    # Optionally save raw
    if raw_output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(raw_output_csv)), exist_ok=True)
        final_ratios.to_csv(raw_output_csv, index=False)

    # Feature selection for cleaning
    features_column_financial = [
        "OPM",
        "NPM",
        "ROA",
        "ROE",
        "EPS",
        "BPS",
        "DPS",
        "cur_ratio",
        "quick_ratio",
        "cash_ratio",
        "inv_turnover",
        "acc_rec_turnover",
        "acc_pay_turnover",
        "debt_ratio",
        "debt_to_equity",
        "pe",
        "ps",
        "pb",
    ]

    # Clean and possibly drop unstable columns
    final_ratios = handle_nan(final_ratios, features_column_financial)

    # Format date
    final_ratios["date"] = pd.to_datetime(final_ratios["date"]).dt.strftime("%Y-%m-%d")

    # Write final CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    final_ratios.to_csv(output_csv, index=False)

    # Write per-sector Excel files
    os.makedirs(sectors_output_dir, exist_ok=True)
    sec_groups = list(final_ratios.groupby("gsector"))
    iterator = sec_groups
    if show_progress:
        iterator = _tqdm(sec_groups, total=len(sec_groups), desc="Write sector Excel")
    for sec, df_sec in iterator:
        sec_label = "NA" if pd.isna(sec) else str(sec)
        # sanitize filename-friendly label
        sec_label = (
            sec_label.replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
            .replace("*", "x")
            .replace("?", "")
            .replace("\"", "")
            .replace("<", "(")
            .replace(">", ")")
            .replace("|", "-")
        )
        out_path = os.path.join(sectors_output_dir, f"sector{sec_label}.xlsx")
        df_sec.to_excel(out_path, index=False)

    print(
        f"Wrote final ratios CSV to: {output_csv} and sector files to: {sectors_output_dir}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess fundamentals and price data to produce engineered ratios "
            "and next-quarter returns; outputs a final CSV and per-sector Excel files."
        )
    )
    parser.add_argument(
        "--fundamentals-csv",
        required=True,
        help="Path to fundamentals CSV (e.g., sp500_fundamental_199601_202210.csv)",
    )
    parser.add_argument(
        "--daily-price-csv",
        required=True,
        help="Path to daily price CSV (must include columns 'tic' and 'gvkey')",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to write final ratios CSV (cleaned).",
    )
    parser.add_argument(
        "--sectors-output-dir",
        required=True,
        help="Directory to write per-sector Excel files (sector{gsector}.xlsx).",
    )
    parser.add_argument(
        "--raw-output-csv",
        required=False,
        help="Optional path to also write the raw (pre-cleaning) ratios CSV.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars during processing.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run(
        fundamentals_csv=args.fundamentals_csv,
        daily_price_csv=args.daily_price_csv,
        output_csv=args.output_csv,
        sectors_output_dir=args.sectors_output_dir,
        raw_output_csv=args.raw_output_csv,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()


