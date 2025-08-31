#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run basic unit tests for M4 (alp_execution.py).

- Creates fixtures in ./data/outputs
- Runs 4 cases against src/alp_execution.py
- Saves a snapshot for EACH case: data/outputs/execution_log_case{1..4}.csv
- Prints JSON summary; exits non-zero if any test fails

Usage:
  python tests/run_m4_tests.py --config ./config/config.yaml
  python tests/run_m4_tests.py --config ./config/test_config.yaml
"""
import argparse, csv, json, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONF = ROOT / "config"
OUT = ROOT / "data" / "outputs"

def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def run_case(args):
    proc = subprocess.run(args, cwd=str(ROOT), text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr

def safe_copy(src: Path, dst: Path):
    if src.exists():
        shutil.copyfile(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONF/"config.yaml"), help="Config path (default: ./config/config.yaml)")
    args = ap.parse_args()

    cfg_path = Path(args.config)

    # If using default config path, write a safe baseline config
    if str(cfg_path) == str(CONF/"config.yaml"):
        base_config = """
execution:
  out_orders: ./data/outputs/planned_orders.csv
  out_plan: ./data/outputs/rebalance_plan.csv
  time_in_force: DAY
  tp_pct: 0.05
  sl_pct: 0.03

alp_execution:
  max_orders: 100
  poll_fills: 1

risk:
  max_share_per_order: 100000
  max_notional_per_order: 1.0e9
  max_total_notional: 1.0e12
  price_tolerance_pct: 1.0
  min_cash_buffer_pct: 0.0
  reject_if_price_missing: false

idempotency:
  enable_client_order_id: true
  client_order_id_prefix: TEST

alpaca:
  trading_endpoint: https://paper-api.alpaca.markets/v2
  data_endpoint: https://data.alpaca.markets/v2
"""
        write(cfg_path, base_config)

    runner = SRC / "alp_execution.py"
    if not runner.exists():
        print("ERROR: src/alp_execution.py not found.", file=sys.stderr)
        sys.exit(2)

    summary = {}
    failures = 0
    log_path = OUT / "execution_log.csv"

    # ---------- Case 1 ----------
    log_path.unlink(missing_ok=True)
    write(OUT/"planned_orders.csv",
          "asset_id,symbol,side,qty,order_type,tp_price,sl_price\n"
          "1111,AAPL,BUY,10,BRACKET,210.0,190.0\n"
          "2222,MSFT,SELL,5,MARKET,,\n")
    write(OUT/"rebalance_plan.csv",
          "asset_id,symbol,price,action,planned_qty\n"
          "1111,AAPL,200.0,BUY,10\n"
          "2222,MSFT,400.0,SELL,5\n")
    rc1, out1, err1 = run_case([sys.executable, str(runner), "--mode","dry","--config",str(cfg_path)])
    rows1 = read_csv(log_path)
    ok1 = rc1==0 and len(rows1)==2 and all(r["status"]=="SIMULATED" for r in rows1)
    if not ok1: failures += 1
    summary["case1"] = {"pass": ok1, "rows": len(rows1)}
    safe_copy(log_path, OUT/"execution_log_case1.csv")

    # ---------- Case 2 ----------
    rc2, out2, err2 = run_case([sys.executable, str(runner), "--mode","dry","--config",str(cfg_path)])
    rows2 = read_csv(log_path)
    ok2 = rc2==0 and len(rows2)==2
    if not ok2: failures += 1
    summary["case2"] = {"pass": ok2, "rows": len(rows2)}
    safe_copy(log_path, OUT/"execution_log_case2.csv")

    # ---------- Case 3 ----------
   # log_path.unlink(misfsing_ok=True)
    print(f"{OUT}/planned_orders.csv")
    write(OUT/"planned_orders.csv",
          "asset_id,symbol,side,qty,order_type,tp_price,sl_price\n"
          "1111,AAPL,BUY,10,BRACKET,210.0,190.0\n"
          "2222,MSFT,BUY,8,BRACKET,410.0,380.0\n"
          "3333,GOOG,SELL,3,MARKET,,\n")
    cfg_text = cfg_path.read_text(encoding="utf-8").replace("max_orders: 100","max_orders: 1")
    write(cfg_path, cfg_text)
    rc3, out3, err3 = run_case([sys.executable, str(runner), "--mode","dry","--config",str(cfg_path)])
    rows3 = read_csv(log_path)
    ok3 = rc3==0 and len(rows3)==1
    if not ok3: failures += 1
    summary["case3"] = {"pass": ok3, "rows": len(rows3)}
    safe_copy(log_path, OUT/"execution_log_case3.csv")

    # ---------- Case 4 ----------
   # log_path.unlink(missing_ok=True)
    cfg_text2 = cfg_text.replace("max_orders: 1","max_orders: 100").replace("max_share_per_order: 100000","max_share_per_order: 5")
    write(cfg_path, cfg_text2)
    rc4, out4, err4 = run_case([sys.executable, str(runner), "--mode","dry","--config",str(cfg_path)])
    rows4 = read_csv(log_path)
    ok4 = rc4==0 and len(rows4)==1 and rows4[0].get("symbol")=="GOOG"
    if not ok4: failures += 1
    summary["case4"] = {"pass": ok4, "rows": len(rows4), "first_symbol": (rows4[0]["symbol"] if rows4 else None)}
    safe_copy(log_path, OUT/"execution_log_case4.csv")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
