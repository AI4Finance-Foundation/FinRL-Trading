#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys, importlib.util, yaml
from collections import Counter
import pandas as pd
from datetime import datetime, timedelta
import glob

# load engine dynamically
spec = importlib.util.spec_from_file_location("execution_engine", str(Path(__file__).parent / "execution_engine.py"))
ee = importlib.util.module_from_spec(spec)
sys.modules["execution_engine"] = ee
spec.loader.exec_module(ee)

# load alp_execution for price fetching
spec = importlib.util.spec_from_file_location("alp_execution", str(Path(__file__).parent / "alp_execution.py"))
alp = importlib.util.module_from_spec(spec)
sys.modules["alp_execution"] = alp
spec.loader.exec_module(alp)

def _fmt(p: Path):
    try:
        return str(p.resolve())
    except Exception:
        return str(p)

def find_latest_positions_file(outputs_dir: Path, current_timestamp: str = None) -> Path:
    """
    find the latest positions file by timestamp
    """
    if current_timestamp is None:
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # parse current timestamp
    try:
        current_dt = datetime.strptime(current_timestamp, "%Y%m%d_%H%M")
    except ValueError:
        try:
            current_dt = datetime.strptime(current_timestamp, "%Y%m%d")
        except ValueError:
            print(f"Warning: Invalid timestamp format: {current_timestamp}, using current time")
            current_dt = datetime.now()
    
    # find all positions_eod_*.csv files
    position_files = list(outputs_dir.glob("positions_eod_*.csv"))
    
    if not position_files:
        default_file = Path("./data/outputs/sample_positions.csv")
        print(f"No position files found, using default: {default_file}")
        return default_file
    
    # parse the timestamp in the file name and find the closest one
    closest_file = None
    min_time_diff = float('inf')
    
    for file_path in position_files:
        filename = file_path.stem
        if filename.startswith("positions_eod_"):
            timestamp_str = filename[14:]
            
            try:
                if "_" in timestamp_str:
                    file_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
                else:
                    file_dt = datetime.strptime(timestamp_str, "%Y%m%d")
                
                time_diff = abs((current_dt - file_dt).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_file = file_path
                    
            except ValueError as e:
                print(f"Warning: Could not parse timestamp from {filename}: {e}")
                continue
    
    if closest_file:
        print(f"Using closest position file: {closest_file} (time diff: {min_time_diff/60:.1f} minutes)")
        return closest_file
    else:
        default_file = Path("./data/outputs/sample_positions.csv")
        print(f"No valid position files found, using default: {default_file}")
        return default_file

def get_alpaca_account_info(conf: dict) -> tuple:
    """
    from Alpaca API get account information
    
    Returns:
        tuple: (equity, positions_df)
    """
    try:
        # Create Alpaca client
        alp_conf = conf.get("alpaca", {})
        key_id = alp_conf.get("api_key")
        secret = alp_conf.get("api_secret")
        trading_base = alp_conf.get("trading_endpoint", "").rstrip("/")
        
        if not (key_id and secret and trading_base):
            raise ValueError("Alpaca credentials missing")
        
        # Create Alpaca client
        alp_client = alp.AlpacaClient(trading_base, trading_base, key_id, secret)
        
        # Get account information
        account = alp_client.get_account()
        if not account:
            raise ValueError("Failed to get account information")
        
        equity = float(account.get("equity", 0))
        print(f"[M2] Alpaca account equity: ${equity:,.2f}")
        
        # Get current positions
        positions = []
        try:
            alpaca_positions = alp_client.get_positions()
            if alpaca_positions:
                for pos in alpaca_positions:
                    symbol = pos.get('symbol', '')
                    qty = float(pos.get('qty', 0))
                    market_value = float(pos.get('market_value', 0))
                    if abs(qty) > 1e-6:  # Only keep stocks with positions
                        positions.append({
                            'symbol': symbol,
                            'qty': qty,
                            'market_value': market_value
                        })
                print(f"[M2] Fetched {len(positions)} positions from Alpaca")
            else:
                print("[M2] No positions found in Alpaca account")
        except Exception as e:
            print(f"[M2] Warning: Failed to fetch positions: {e}")
        
        positions_df = pd.DataFrame(positions, columns=["symbol", "qty", "market_value"])
        
        return equity, positions_df
        
    except Exception as e:
        print(f"[M2] Error getting Alpaca account info: {e}")
        return None, None

def update_config_equity(conf: dict, new_equity: float, config_path: str):
    """
    Update equity value in config file
    """
    try:
        # Update configuration in memory
        conf["scheduler"]["m2"]["equity"] = new_equity
        
        # Write back to config file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(conf, f, default_flow_style=False, allow_unicode=True)
        
        print(f"[M2] Updated config equity to: ${new_equity:,.2f}")
        
    except Exception as e:
        print(f"[M2] Warning: Failed to update config file: {e}")

def main():
    ap = argparse.ArgumentParser(description="Execution Engine (M2) - Generate planned orders from targets")
    
    # Core parameters (required)
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--targets", required=True, help="Consolidated targets (csv/parquet) from M1")
    ap.add_argument("--equity", required=True, type=float, help="Total account equity in USD")
    ap.add_argument("--execution-mode", choices=["dryrun", "online"], required=True, 
                    help="Execution mode: dryrun (use config equity) or online (fetch from Alpaca)")
    
    # Optional parameters (minimal)
    ap.add_argument("--debug", action="store_true", help="Print diagnostics")
    ap.add_argument("--timestamp", help="Current timestamp (YYYYMMDD_HHMM) for finding latest positions")
    
    args = ap.parse_args()

    # Load config
    conf = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mapping_file = conf["mapping"]["file"]
    id_col = conf["mapping"]["id_column"]
    sym_col = conf["mapping"]["symbol_column"]

    # Get output paths from config file
    exec_conf = conf.get("execution", {})
    out_orders = exec_conf.get("out_orders", "./data/outputs/planned_orders.csv")
    out_plan = exec_conf.get("out_plan", "./data/outputs/rebalance_plan.csv")
    
    # Get execution parameters from config file
    tp_pct = float(exec_conf.get("tp_pct", 0.05))
    sl_pct = float(exec_conf.get("sl_pct", 0.03))

    # Read M2 configuration
    scheduler_conf = conf.get("scheduler", {})
    m2_conf = scheduler_conf.get("m2", {})
    
    print(f"[M2] Execution mode: {args.execution_mode}")
    
    # Process according to execution mode
    if args.execution_mode == "online":
        # Online mode: Fetch account information from Alpaca
        print("[M2] Online mode: Fetching account information from Alpaca...")
        alpaca_equity, alpaca_positions = get_alpaca_account_info(conf)
        
        if alpaca_equity is not None:
            # Update equity value
            args.equity = alpaca_equity
            
            # Update config file
            if m2_conf.get("online", {}).get("auto_update_equity", True):
                update_config_equity(conf, alpaca_equity, args.config)
            
            # If there are positions, save to file
            if not alpaca_positions.empty and m2_conf.get("online", {}).get("auto_update_positions", True):
                positions_save_path = m2_conf.get("online", {}).get("save_current_positions", "./data/outputs/current_positions.csv")
                alpaca_positions.to_csv(positions_save_path, index=False)
                print(f"[M2] Saved current positions to: {positions_save_path}")
        
        # Online mode must fetch real prices
        fetch_real_prices = True
        price_source = m2_conf.get("online", {}).get("price_source", "snapshot")
        save_prices_path = m2_conf.get("online", {}).get("save_prices", "./data/outputs/real_prices.csv")
        debug_mode = args.debug or m2_conf.get("online", {}).get("debug", False)
        
    else:
        # DryRun mode: Use config price fetch settings
        fetch_real_prices = m2_conf.get("dryrun", {}).get("fetch_real_prices", False)
        price_source = m2_conf.get("dryrun", {}).get("price_source", "latest")
        save_prices_path = m2_conf.get("dryrun", {}).get("save_prices", "./data/outputs/real_prices.csv")
        debug_mode = args.debug or m2_conf.get("dryrun", {}).get("debug", False)
    
    if debug_mode:
        print(f"[DEBUG] Execution mode: {args.execution_mode}")
        print(f"[DEBUG] Fetch real prices: {fetch_real_prices}")
        print(f"[DEBUG] Price source: {price_source}")
        print(f"[DEBUG] Equity: ${args.equity:,.2f}")
        print(f"[DEBUG] Timestamp: {args.timestamp}")

    # Input file paths (according to SPEC file structure)
    targets_path = Path(args.targets)
    mapping_path = Path(mapping_file)
    prices_path = Path("./data/outputs/sample_prices.csv")  # Default price file
    
    # Positions file path
    outputs_dir = Path("./data/outputs")
    
    # Online mode priority use current_positions.csv
    if args.execution_mode == "online":
        current_positions_path = Path("./data/outputs/current_positions.csv")
        if current_positions_path.exists():
            positions_path = current_positions_path
            print(f"[M2] Using current positions from Alpaca: {positions_path}")
        else:
            positions_path = find_latest_positions_file(outputs_dir, args.timestamp)
            print(f"[M2] Current positions not found, using latest: {positions_path}")
    else:
        positions_path = find_latest_positions_file(outputs_dir, args.timestamp)

    # Load data
    targets = ee.load_targets(targets_path)
    mapping = ee.load_mapping(mapping_path, id_col, sym_col)
    targets2 = ee.map_targets_to_symbols(targets, mapping)

    # Price fetch logic
    if fetch_real_prices:
        print("Fetching real prices from Alpaca API...")
        mapping_df = pd.read_csv(mapping_path)
        
        real_prices = alp.fetch_prices_for_targets(
            cfg=conf,
            targets_df=targets,
            mapping_df=mapping_df,
            price_source=price_source,
            debug=debug_mode
        )
        
        if real_prices:
            prices = real_prices
            if save_prices_path:
                alp.save_prices_to_csv(real_prices, save_prices_path)
        else:
            print("Warning: Failed to fetch real prices, falling back to sample prices")
            prices = ee.load_prices_from_csv(prices_path)
    else:
        prices = ee.load_prices_from_csv(prices_path)
    
    positions = ee.load_positions_from_csv(positions_path)
    
    # Debug diagnostics BEFORE planning
    if debug_mode:
        print("=== DEBUG: Inputs ===")
        print(f"config:       {_fmt(Path(args.config))}")
        print(f"targets:      {_fmt(targets_path)}  rows={len(targets)}  uniq_assets={targets['asset_id'].nunique() if not targets.empty else 0}")
        print(f"mapping:      {_fmt(mapping_path)}  rows={len(mapping)}")
        print(f"prices:       {len(prices)} symbols loaded")
        print(f"positions:    {_fmt(positions_path)}  rows={len(positions)}")
        if not targets.empty:
            wsum = float(pd.to_numeric(targets['target_weight'], errors='coerce').fillna(0).sum())
            print(f"target_weight sum = {wsum:.6f}")
        missing_after_map = max(0, len(targets) - len(targets2))
        print(f"mapped targets rows = {len(targets2)}  dropped (missing symbol) = {missing_after_map}")
        if not targets2.empty:
            syms = set(targets2['symbol'].astype(str).tolist())
            missing_prices = [s for s in syms if s not in prices]
            print(f"symbols without price = {len(missing_prices)} (e.g., {missing_prices[:5]})")

    # Generate plan
    orders, plan = ee.plan_orders(
        targets_df=targets2,
        prices=prices,
        positions=positions,
        equity=args.equity,
        tp_pct=tp_pct,
        sl_pct=sl_pct
    )

    # Debug diagnostics AFTER planning
    if debug_mode:
        from collections import Counter
        print("=== DEBUG: Plan Summary ===")
        acts = Counter(plan['action']) if not plan.empty else Counter()
        print(f"actions: {dict(acts)}")
        if not plan.empty:
            print("sample plan rows:")
            print(plan.head(10).to_string(index=False))

    # Fractional shares processing: generate actual executable orders
    risk_conf = conf.get("risk", {})
    fractional_conf = risk_conf.get("fractional_shares", {})
    
    min_order_notional = float(fractional_conf.get("min_order_notional", 2.0))
    quantity_precision = int(fractional_conf.get("quantity_precision", 2))
    max_total_weight = float(fractional_conf.get("max_total_weight", 1.0))
    
    print(f"[M2] Applying fractional share rules:")
    print(f"[M2]   - Min order notional: ${min_order_notional}")
    print(f"[M2]   - Quantity precision: {quantity_precision} decimal places")
    print(f"[M2]   - Max total weight: {max_total_weight*100:.1f}%")
    
    # Generate actual orders
    real_orders, real_plan = ee.generate_real_orders(
        orders=orders,
        plan_df=plan,
        min_order_notional=min_order_notional,
        quantity_precision=quantity_precision,
        max_total_weight=max_total_weight
    )
    
    # Save original file (for debugging)
    ee.save_orders_csv(orders, Path(out_orders))
    ee.save_plan_csv(plan, Path(out_plan))
    
    # Save actual execution file
    real_orders_path = out_orders.replace(".csv", "_real.csv")
    real_plan_path = out_plan.replace(".csv", "_real.csv")
    
    ee.save_orders_csv(real_orders, Path(real_orders_path))
    ee.save_plan_csv(real_plan, Path(real_plan_path))
    
    print(f"[M2] Original: {len(orders)} orders -> {out_orders}")
    print(f"[M2] Real: {len(real_orders)} orders -> {real_orders_path}")
    print(f"[M2] Rebalance plan -> {out_plan}")
    print(f"[M2] Real plan -> {real_plan_path}")

if __name__ == "__main__":
    main()
