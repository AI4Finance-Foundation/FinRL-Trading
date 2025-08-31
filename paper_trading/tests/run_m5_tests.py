#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for M5 reconciliation (log-based)
- Case 1: perfect match BUY/SELL -> delta=0
- Case 2: partial mismatch -> delta!=0
- Case 3: empty exec_log -> actual=0
"""
import sys, json
from pathlib import Path
import subprocess
import pandas as pd
import datetime as dt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "data" / "outputs"
CONF = ROOT / "config" / "config.yaml"

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

def run_m5(date: str):
    cmd = [sys.executable, str(SRC/"m5_reconcile.py"),
           "--config", str(CONF),
           "--date", date,
           "--source", "logs",
           "--debug"]
    print("RUN:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(r.returncode)

def read_csv(path: Path):
    import csv
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def main():
    today = dt.datetime.now().strftime("%Y-%m-%d")
    ymd = today.replace("-","")

    # Case 1: match
    write_csv(OUT/"planned_orders.csv", [
        {"symbol":"AAPL","side":"BUY","qty":10,"order_type":"MARKET"},
        {"symbol":"MSFT","side":"SELL","qty":5,"order_type":"MARKET"},
    ])
    write_csv(OUT/"execution_log.csv", [
        {"symbol":"AAPL","side":"BUY","qty":10,"status":"SIMULATED"},
        {"symbol":"MSFT","side":"SELL","qty":5,"status":"SIMULATED"},
    ])
    write_csv(OUT/"positions_bod.csv", [
        {"symbol":"AAPL","qty":0},
        {"symbol":"MSFT","qty":10},
    ])
    run_m5(today)
    recon1 = read_csv(OUT/f"recon_report_{ymd}.csv")
    eod1 = read_csv(OUT/f"positions_eod_{ymd}.csv")
    assert all(r["status"]=="OK" for r in recon1), "Case1: expect all OK"
    # EOD: AAPL 0+10=10, MSFT 10-5=5
    qty_map = {r["symbol"]: float(r["qty"]) for r in eod1}
    assert abs(qty_map.get("AAPL",0)-10) < 1e-9 and abs(qty_map.get("MSFT",0)-5) < 1e-9

    # Case 2: mismatch
    write_csv(OUT/"planned_orders.csv", [
        {"symbol":"AAPL","side":"BUY","qty":10},
    ])
    write_csv(OUT/"execution_log.csv", [
        {"symbol":"AAPL","side":"BUY","qty":6,"status":"SIMULATED"},
    ])
    run_m5(today)
    recon2 = read_csv(OUT/f"recon_report_{ymd}.csv")
    aapl = [r for r in recon2 if r["symbol"]=="AAPL"][0]
    assert aapl["status"]=="MISMATCH", "Case2: expect mismatch"
    assert abs(float(aapl["delta_qty"]) - 4.0) < 1e-9

    # Case 3: empty exec_log -> actual=0
    write_csv(OUT/"planned_orders.csv", [
        {"symbol":"TSLA","side":"SELL","qty":3},
    ])
    # exec_log.csv empty
    (OUT/"execution_log.csv").unlink(missing_ok=True)
    run_m5(today)
    recon3 = read_csv(OUT/f"recon_report_{ymd}.csv")
    tsla = [r for r in recon3 if r["symbol"]=="TSLA"][0]
    assert abs(float(tsla["planned_qty"]) + 3.0) < 1e-9  # SELL -> planned -3
    assert abs(float(tsla["actual_qty"]) - 0.0) < 1e-9

    print(json.dumps({"ok": True, "date": today, "recon": OUT.as_posix()}, ensure_ascii=False))

if __name__ == "__main__":
    main()
