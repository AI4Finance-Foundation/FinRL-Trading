#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fallback test: M2 -> M4 only
- directly construct simple targets file, bypass M1
- verify M2 products and M4 (dry) execution written
"""
import sys, csv, json
from pathlib import Path
from datetime import datetime
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONF = ROOT / "config" / "config.yaml"
OUT = ROOT / "data" / "outputs"
INTM = ROOT / "data" / "intermediate"

def write_targets(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    # minimal targets: trade_date, asset_id, target_weight
    today = datetime.now().strftime("%Y-%m-%d")
    rows = [
        {"trade_date": today, "asset_id": "1111", "target_weight": 0.5},
        {"trade_date": today, "asset_id": "2222", "target_weight": 0.5},
    ]
    # for compatibility with run_execution, your implementation usually supports parquet/csv; here we write csv first
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(str(p), index=False)

def main():
    targets = INTM / f"targets_consolidated_{datetime.now():%Y%m%d}.parquet"
    write_targets(targets)

    # M2
    cmd_m2 = [
        sys.executable, str(SRC/"run_execution.py"),
        "--config", str(CONF),
        "--targets", str(targets),
        "--equity", "10000",
        "--out-orders", str(OUT/"planned_orders.csv"),
        "--out-plan", str(OUT/"rebalance_plan.csv"),
    ]
    print("RUN M2:", " ".join(cmd_m2))
    r2 = subprocess.run(cmd_m2, cwd=str(ROOT), text=True, capture_output=True)
    print(r2.stdout)
    if r2.returncode != 0: 
        print(r2.stderr); raise SystemExit(r2.returncode)

    # M4 (dry)
    cmd_m4 = [
        sys.executable, str(SRC/"alp_execution.py"),
        "--config", str(CONF),
        "--mode", "dry",
        "--orders", str(OUT/"planned_orders.csv"),
        "--plan",   str(OUT/"rebalance_plan.csv"),
    ]
    print("RUN M4:", " ".join(cmd_m4))
    r4 = subprocess.run(cmd_m4, cwd=str(ROOT), text=True, capture_output=True)
    print(r4.stdout)
    if r4.returncode != 0:
        print(r4.stderr); raise SystemExit(r4.returncode)

    assert (OUT/"execution_log.csv").exists(), "M4 execution_log.csv not found"
    print("OK: execution_log.csv written")

if __name__ == "__main__":
    main()