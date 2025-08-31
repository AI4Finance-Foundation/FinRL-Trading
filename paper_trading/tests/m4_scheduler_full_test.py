#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full test: M1 -> M2 -> M4 (once)
- no changes to production config, only assert existence and product
- requires scheduler.m4.mode=dry
"""

import sys, json
from pathlib import Path
from datetime import datetime
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONF = ROOT / "config" / "config.yaml"
LOGS = ROOT / "logs"
OUT = ROOT / "data" / "outputs"
INTM = ROOT / "data" / "intermediate"

def must_exist(p: Path, kind="file"):
    if kind == "file" and not p.is_file():
        raise SystemExit(f"missing file: {p}")
    if kind == "dir" and not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)

def main():
    # T0: env check
    must_exist(SRC/"scheduler.py")
    must_exist(SRC/"run_signal_bus.py")
    must_exist(SRC/"run_execution.py")
    must_exist(SRC/"alp_execution.py")
    must_exist(CONF)
    must_exist(ROOT/"config"/"strategy_manifest.yaml")

    must_exist(LOGS, "dir")
    must_exist(OUT, "dir")
    must_exist(INTM, "dir")

    # T1: run once
    today = datetime.now().strftime("%Y-%m-%d")
    cmd = [sys.executable, str(SRC/"scheduler.py"),
           "--config", str(CONF),
           "--once",
           "--date", today,
           "--debug"]
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise SystemExit(f"scheduler returned {proc.returncode}")

    # Assert artifacts
    inter = list(INTM.glob("targets_consolidated_*.parquet"))
    if not inter:
        raise SystemExit("M1 output parquet not found in data/intermediate")
    if not (OUT/"planned_orders.csv").exists():
        raise SystemExit("M2 output planned_orders.csv not found")
    if not (OUT/"rebalance_plan.csv").exists():
        raise SystemExit("M2 output rebalance_plan.csv not found")
    if not (OUT/"execution_log.csv").exists():
        raise SystemExit("M4 output execution_log.csv not found (dry should still log)")

    result = {
        "full": True,
        "parquet": [p.name for p in inter][-1],
        "orders_rows": sum(1 for _ in open(OUT/"planned_orders.csv", "r", encoding="utf-8")),
        "plan_rows":   sum(1 for _ in open(OUT/"rebalance_plan.csv", "r", encoding="utf-8")),
        "exec_rows":   sum(1 for _ in open(OUT/"execution_log.csv", "r", encoding="utf-8")),
        "logs": {
            "scheduler": (ROOT/"logs"/"scheduler.log").exists(),
            "m4":        (ROOT/"logs"/"m4_execution.log").exists()
        }
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()