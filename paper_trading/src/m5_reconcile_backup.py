#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M5 — Reconciliation
双重检查：
1. 单次执行对账：planned_qty vs actual_qty（检查 M4 执行情况）
2. 最终持仓对账：target_qty vs final_actual_qty（检查最终结果）
   包含权重百分比比较
"""
import argparse
import sys
from pathlib import Path
import datetime as dt
import pandas as pd
import yaml

BUY = {"BUY", "BOT", "BUY_TO_OPEN", "BUY_TO_COVER"}
SELL = {"SELL", "SLD", "SELL_TO_CLOSE", "SELL_SHORT"}

def _read_csv_safe(p: Path) -> pd.DataFrame:
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def _sign(side: str) -> int:
    s = str(side).upper()
    if s in BUY: return +1
    if s in SELL: return -1
    return 0

def load_config(cfg_path: Path) -> dict:
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def filter_execution_log_by_latest_run(exec_log_df: pd.DataFrame, run_date: str = None) -> pd.DataFrame:
    """
    过滤 execution_log.csv，只保留最新一次运行的记录
    """
    if exec_log_df.empty:
        return exec_log_df
    
    # 从 client_order_id 中提取时间戳
    def extract_timestamp_from_client_id(client_id):
        if pd.isna(client_id) or not client_id:
            return None
        parts = str(client_id).split('_')
        if len(parts) >= 2:
            return parts[1]  # 返回 YYYYMMDD 部分
        return None
    
    # 添加时间戳列
    exec_log_df['extracted_timestamp'] = exec_log_df['client_order_id'].apply(extract_timestamp_from_client_id)
    
    # 找到执行日志中的最新时间戳
    valid_timestamps = [ts for ts in exec_log_df['extracted_timestamp'].dropna().unique() if ts]
    if not valid_timestamps:
        print("[M5] Warning: No valid timestamps found in execution log")
        return pd.DataFrame()
    
    latest_timestamp = max(valid_timestamps)
    print(f"[M5] Found latest timestamp in execution log: {latest_timestamp}")
    
    # 如果指定了 run_date，检查是否匹配
    if run_date:
        target_timestamp = run_date.replace('-', '')
        if target_timestamp != latest_timestamp:
            print(f"[M5] Warning: Run date {target_timestamp} doesn't match execution log date {latest_timestamp}")
            print(f"[M5] Using execution log date: {latest_timestamp}")
    
    # 过滤出最新一次运行的记录
    filtered_df = exec_log_df[exec_log_df['extracted_timestamp'] == latest_timestamp].copy()
    print(f"[M5] Filtering execution log for date: {latest_timestamp}, found {len(filtered_df)} records")
    
    # 删除临时列
    filtered_df = filtered_df.drop('extracted_timestamp', axis=1)
    
    return filtered_df

def reconcile_single_execution(planned_csv: Path, exec_log_csv: Path, run_date: str = None) -> pd.DataFrame:
    """
    检查 1：单次执行对账
    比较：planned_qty vs actual_qty（单次运行）
    """
    planned = _read_csv_safe(planned_csv)
    actual = _read_csv_safe(exec_log_csv)

    if planned.empty:
        planned = pd.DataFrame(columns=["symbol","side","qty"])
    if actual.empty:
        actual = pd.DataFrame(columns=["symbol","side","qty","status"])

    # 过滤出指定运行日期的记录
    if not actual.empty:
        actual = filter_execution_log_by_latest_run(actual, run_date)
        print(f"[M5] Single execution check: {len(actual)} records")

    # Normalize columns
    for df in (planned, actual):
        if "symbol" not in df.columns and "Symbol" in df.columns: df.rename(columns={"Symbol":"symbol"}, inplace=True)
        if "side" not in df.columns and "Side" in df.columns: df.rename(columns={"Side":"side"}, inplace=True)
        if "qty" not in df.columns and "quantity" in df.columns: df.rename(columns={"quantity":"qty"}, inplace=True)

    # Planned signed qty
    planned["signed_qty"] = planned.apply(lambda r: float(r.get("qty", 0)) * _sign(r.get("side","")), axis=1)
    p = planned.groupby("symbol", as_index=False)["signed_qty"].sum().rename(columns={"signed_qty":"planned_qty"})

    # Actual signed qty
    actual["signed_qty"] = actual.apply(lambda r: float(r.get("qty", 0)) * _sign(r.get("side","")), axis=1)
    a = actual.groupby("symbol", as_index=False)["signed_qty"].sum().rename(columns={"signed_qty":"actual_qty"})

    recon = p.merge(a, on="symbol", how="outer").fillna(0.0)
    recon["delta_qty"] = recon["planned_qty"] - recon["actual_qty"]
    recon["status"] = recon["delta_qty"].apply(lambda x: "OK" if abs(float(x)) < 1e-6 else "MISMATCH")
    
    return recon.sort_values("symbol")

def reconcile_final_positions(plan_csv: Path, bod_csv: Path, exec_log_csv: Path, prices_csv: Path = None) -> pd.DataFrame:
    """
    检查 2：最终持仓对账
    比较：target_qty vs final_actual_qty（最终结果）
    包含权重百分比比较
    """
    # 1. 读取目标持仓（从 rebalance_plan.csv）
    plan_df = _read_csv_safe(plan_csv)
    target_positions = {}
    target_weights = {}
    if not plan_df.empty:
        for _, row in plan_df.iterrows():
            symbol = row['symbol']
            target_qty = float(row['target_qty'])  # 目标持仓数量
            target_weight = float(row.get('target_weight', 0))  # 目标权重
            target_positions[symbol] = target_qty
            target_weights[symbol] = target_weight
    
    # 2. 读取期初持仓
    bod_df = _read_csv_safe(bod_csv)
    bod_positions = {}
    if not bod_df.empty:
        for _, row in bod_df.iterrows():
            symbol = row['symbol']
            qty = float(row['qty'])
            bod_positions[symbol] = qty
    
    # 3. 计算执行变化（从 execution_log.csv）
    exec_df = _read_csv_safe(exec_log_csv)
    execution_changes = {}
    if not exec_df.empty:
        for _, row in exec_df.iterrows():
            symbol = row['symbol']
            side = row['side']
            qty = float(row['qty'])
            change = qty if side == 'BUY' else -qty
            execution_changes[symbol] = execution_changes.get(symbol, 0) + change
    
    # 4. 读取价格信息
    prices = {}
    if prices_csv and prices_csv.exists():
        prices_df = _read_csv_safe(prices_csv)
        if not prices_df.empty:
            for _, row in prices_df.iterrows():
                symbol = row['symbol']
                price = float(row['price'])
                prices[symbol] = price
    
    # 5. 计算最终实际持仓
    final_actual_positions = {}
    all_symbols = set(target_positions.keys()) | set(bod_positions.keys()) | set(execution_changes.keys())
    
    for symbol in all_symbols:
        bod_qty = bod_positions.get(symbol, 0)
        exec_change = execution_changes.get(symbol, 0)
        final_qty = bod_qty + exec_change
        final_actual_positions[symbol] = final_qty
    
    # 6. 计算权重百分比
    # 计算总市值
    total_market_value = 0
    for symbol in all_symbols:
        qty = final_actual_positions.get(symbol, 0)
        price = prices.get(symbol, 0)
        market_value = qty * price
        total_market_value += market_value
    
    print(f"[M5] Total market value: ${total_market_value:,.2f}")
    
    # 计算实际权重
    actual_weights = {}
    for symbol in all_symbols:
        qty = final_actual_positions.get(symbol, 0)
        price = prices.get(symbol, 0)
        market_value = qty * price
        if total_market_value > 0:
            actual_weight = (market_value / total_market_value) * 100  # 转换为百分比
        else:
            actual_weight = 0
        actual_weights[symbol] = actual_weight
    
    # 验证权重总和
    total_actual_weight = sum(actual_weights.values())
    print(f"[M5] Total actual weight: {total_actual_weight:.2f}%")
    
    # 如果权重总和不为100%，进行归一化
    if abs(total_actual_weight - 100.0) > 0.01 and total_actual_weight > 0:
        print(f"[M5] Warning: Total weight is {total_actual_weight:.2f}%, normalizing to 100%")
        for symbol in actual_weights:
            actual_weights[symbol] = (actual_weights[symbol] / total_actual_weight) * 100
    
    # 7. 对账比较
    recon_rows = []
    for symbol in all_symbols:
        target_qty = target_positions.get(symbol, 0)
        actual_qty = final_actual_positions.get(symbol, 0)
        delta_qty = target_qty - actual_qty
        
        target_weight = target_weights.get(symbol, 0)
        actual_weight = actual_weights.get(symbol, 0)
        delta_weight = target_weight - actual_weight
        
        price = prices.get(symbol, 0)
        target_market_value = target_qty * price
        actual_market_value = actual_qty * price
        
        recon_rows.append({
            'symbol': symbol,
            'target_qty': target_qty,
            'actual_qty': actual_qty,
            'delta_qty': delta_qty,
            'price': price,
            'target_market_value': target_market_value,
            'actual_market_value': actual_market_value,
            'target_weight_pct': target_weight,
            'actual_weight_pct': actual_weight,
            'delta_weight_pct': delta_weight,
            'status': 'OK' if abs(delta_qty) < 1e-6 else 'MISMATCH'
        })
    
    recon_df = pd.DataFrame(recon_rows).sort_values("symbol")
    
    # 8. 添加汇总信息
    summary = {
        'total_symbols': len(recon_df),
        'total_target_market_value': recon_df['target_market_value'].sum(),
        'total_actual_market_value': recon_df['actual_market_value'].sum(),
        'total_target_weight_pct': recon_df['target_weight_pct'].sum(),
        'total_actual_weight_pct': recon_df['actual_weight_pct'].sum(),
        'mismatch_count': int((recon_df['status'] == 'MISMATCH').sum()),
        'ok_count': int((recon_df['status'] == 'OK').sum())
    }
    
    return recon_df, summary

def build_eod_positions(final_recon: pd.DataFrame) -> pd.DataFrame:
    """
    构建 EOD 持仓文件
    使用最终持仓对账的结果
    """
    # 从最终对账结果中提取实际持仓
    eod_positions = []
    for _, row in final_recon.iterrows():
        symbol = row['symbol']
        actual_qty = row['actual_qty']
        if abs(actual_qty) > 1e-6:  # 只保留有持仓的股票
            eod_positions.append({
                'symbol': symbol,
                'qty': actual_qty
            })
    
    return pd.DataFrame(eod_positions).sort_values("symbol")

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_summary(summary: dict, path: Path):
    """保存汇总信息到文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Final Position Reconciliation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Symbols: {summary['total_symbols']}\n")
        f.write(f"Total Target Market Value: ${summary['total_target_market_value']:,.2f}\n")
        f.write(f"Total Actual Market Value: ${summary['total_actual_market_value']:,.2f}\n")
        f.write(f"Total Target Weight: {summary['total_target_weight_pct']:.2f}%\n")
        f.write(f"Total Actual Weight: {summary['total_actual_weight_pct']:.2f}%\n")
        f.write(f"OK Count: {summary['ok_count']}\n")
        f.write(f"Mismatch Count: {summary['mismatch_count']}\n")
        f.write(f"Success Rate: {(summary['ok_count']/summary['total_symbols']*100):.1f}%\n")

def main():
    ap = argparse.ArgumentParser(description="M5 — Reconciliation (Dual Check with Weight Analysis)")
    ap.add_argument("--config", default="./config/config.yaml")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--time", default="0900", help="HHMM format (default: 0900)")
    ap.add_argument("--source", choices=["alpaca","logs"], default="logs")
    ap.add_argument("--planned", default=None, help="Override planned_orders.csv path")
    ap.add_argument("--exec-log", dest="exec_log", default=None, help="Override execution_log.csv path")
    ap.add_argument("--rebalance-plan", dest="rebalance_plan", default=None, help="Optional price source for notionals")
    ap.add_argument("--positions-in", dest="positions_in", default="./data/outputs/positions_bod.csv")
    ap.add_argument("--prices", default=None, help="Price file for weight calculation")
    ap.add_argument("--out-dir", default="./data/outputs")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_date = (args.date or dt.datetime.now().strftime("%Y-%m-%d"))
    run_time = args.time
    yyyymmdd = run_date.replace("-","")
    timestamp = f"{yyyymmdd}_{run_time}"

    # Resolve inputs
    planned_path = Path(args.planned or cfg.get("execution",{}).get("out_orders","./data/outputs/planned_orders.csv"))
    exec_log_path = Path(args.exec_log or "./data/outputs/execution_log.csv")
    plan_csv = Path(args.rebalance_plan or cfg.get("execution",{}).get("out_plan","./data/outputs/rebalance_plan.csv"))
    bod_path = Path(args.positions_in) if args.positions_in else None
    prices_path = Path(args.prices or "./data/outputs/real_prices.csv")

    out_dir = Path(args.out_dir)
    single_recon_out = out_dir / f"recon_single_{timestamp}.csv"
    final_recon_out = out_dir / f"recon_final_{timestamp}.csv"
    summary_out = out_dir / f"recon_summary_{timestamp}.txt"
    eod_out = out_dir / f"positions_eod_{timestamp}.csv"

    if args.source == "alpaca":
        print("[M5] alpaca source not implemented in test environment; fallback to logs")
        args.source = "logs"

    # 检查 1：单次执行对账
    print("[M5] === Check 1: Single Execution Reconciliation ===")
    single_recon = reconcile_single_execution(planned_path, exec_log_path, run_date)
    save_csv(single_recon, single_recon_out)
    
    single_mism = int((single_recon["status"]=="MISMATCH").sum())
    single_total = int(len(single_recon))
    print(f"[M5] Single execution: {single_total} symbols, {single_mism} mismatches")

    # 检查 2：最终持仓对账
    print("[M5] === Check 2: Final Position Reconciliation ===")
    final_recon, summary = reconcile_final_positions(plan_csv, bod_path, exec_log_path, prices_path)
    save_csv(final_recon, final_recon_out)
    save_summary(summary, summary_out)
    
    final_mism = int((final_recon["status"]=="MISMATCH").sum())
    final_total = int(len(final_recon))
    print(f"[M5] Final positions: {final_total} symbols, {final_mism} mismatches")
    print(f"[M5] Total market value: ${summary['total_actual_market_value']:,.2f}")
    print(f"[M5] Total weight: {summary['total_actual_weight_pct']:.2f}%")
    print(f"[M5] Success rate: {(summary['ok_count']/summary['total_symbols']*100):.1f}%")

    # 构建 EOD 持仓
    eod = build_eod_positions(final_recon)
    save_csv(eod, eod_out)

    # Summary
    print(f"[M5] date={run_date} summary:")
    print(f"[M5] - Single execution: {single_recon_out}")
    print(f"[M5] - Final positions: {final_recon_out}")
    print(f"[M5] - Summary report: {summary_out}")
    print(f"[M5] - EOD positions: {eod_out}")
    
    if args.debug:
        print("=== Single Execution Sample ===")
        print(single_recon.head(10).to_string(index=False))
        print("=== Final Positions Sample ===")
        print(final_recon.head(10).to_string(index=False))
        print("=== Summary ===")
        print(f"Total Market Value: ${summary['total_actual_market_value']:,.2f}")
        print(f"Total Weight: {summary['total_actual_weight_pct']:.2f}%")
        print("=== EOD Positions Sample ===")
        print(eod.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
