
"""
M3 Scheduler/Orchestrator
- serial execution: M1 -> M2 -> M4 (minimum changes to integrate M4)
- CLI, logging, daemon, etc. behavior remains consistent with the original version
"""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time
import yaml
import sys
import logging
import csv

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "scheduler.log"

def setup_logger(debug: bool=False):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scheduler")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # avoid duplicate handlers if re-run
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5

def next_run_after(now: datetime, run_time_hhmm: str) -> datetime:
    hh, mm = [int(x) for x in run_time_hhmm.split(":")]
    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return candidate

def build_m1_cmd(project_root: Path, cfg: dict, run_date: str) -> list:
    manifest = cfg.get("scheduler", {}).get("m1_manifest", "./config/strategy_manifest.yaml")
    m1_mode = cfg.get("scheduler", {}).get("m1", {}).get("mode", "select")
    m1_pick = cfg.get("scheduler", {}).get("m1", {}).get("pick", None)
    out_name = f"targets_consolidated_{run_date.replace('-','')}.parquet"
    out_path = cfg.get("scheduler", {}).get("m1_out", f"./data/intermediate/{out_name}")
    cmd = [
        sys.executable, str(project_root / "src" / "run_signal_bus.py"),
        "--manifest", str(project_root / manifest),
        "--date", run_date,
        "--mode", m1_mode,
        "--out", str(project_root / out_path)
    ]
    if m1_mode == "select" and m1_pick:
        cmd += ["--pick", m1_pick]
    blend = cfg.get("scheduler", {}).get("m1", {}).get("blend", None)
    if m1_mode == "blend" and blend:
        blend_str = ",".join([f"{k}={v}" for k,v in blend.items()])
        cmd += ["--blend", blend_str]
    return cmd, out_path

def build_m2_cmd(project_root: Path, cfg: dict, targets_path: str, run_date: str = None) -> list:
    cfg_path = cfg.get("scheduler", {}).get("config_path", "./config/config.yaml")
    equity = cfg.get("scheduler", {}).get("m2", {}).get("equity", None)
    if equity is None:
        raise SystemExit("scheduler.m2.equity is required in config to run M2")
    
    # Price fetching options
    scheduler_cfg = cfg.get("scheduler", {})
    m2_cfg = scheduler_cfg.get("m2", {})
    
    # Get execution mode
    execution_mode = m2_cfg.get("execution_mode", "dryrun")
    
    cmd = [
        sys.executable, str(project_root / "src" / "run_execution.py"),
        "--config", str(project_root / cfg_path),
        "--targets", str(project_root / targets_path),
        "--equity", str(equity),
        "--execution-mode", execution_mode
    ]
    
    # New: Pass timestamp parameter
    if run_date:
        # Convert YYYY-MM-DD to YYYYMMDD_HHMM format
        # For daily rebalancing, use 09:00 as default time
        timestamp = run_date.replace("-", "") + "_0900"
        cmd += ["--timestamp", timestamp]
    
    # Debug mode
    if m2_cfg.get("debug", False):
        cmd += ["--debug"]
    
    # Detailed debug information
    print(f"[DEBUG] Full scheduler config: {scheduler_cfg}")
    print(f"[DEBUG] m2_cfg: {m2_cfg}")
    print(f"[DEBUG] Final M2 command: {' '.join(cmd)}")
    
    return cmd

    # ===== new: M4 command construction (minimum changes to integrate) =====
def build_m4_cmd(project_root: Path, cfg: dict) -> list:
    """
    construct M4 (alp_execution) command:
    - read config.scheduler.m4.mode (dry|real, default dry)
    - optional: explicitly pass M2's output paths (--orders/--plan) to avoid relative path ambiguity
    - optional: append extra_args (e.g., --debug)
    """
    sched = cfg.get("scheduler", {}) or {}
    cfg_path = sched.get("config_path", "./config/config.yaml")

    m4_cfg = sched.get("m4", {}) or {}
    mode = m4_cfg.get("mode", "dry")
    pass_orders_plan = bool(m4_cfg.get("pass_orders_plan", True))
    extra = m4_cfg.get("extra_args", "").strip()

    out_orders = cfg.get("execution", {}).get("out_orders", "./data/outputs/planned_orders.csv")
    out_plan   = cfg.get("execution", {}).get("out_plan",   "./data/outputs/rebalance_plan.csv")

    cmd = [
        sys.executable, str(project_root / "src" / "alp_execution.py"),
        "--config", str(project_root / cfg_path),
        "--mode", mode
    ]
    if pass_orders_plan:
        cmd += ["--orders", str(project_root / out_orders),
                "--plan",   str(project_root / out_plan)]
    if extra:
        cmd += extra.split()
    return cmd



# === new: M5 command construction ===
def build_m5_cmd(project_root: Path, cfg: dict, run_date: str) -> list:
    sched = cfg.get("scheduler", {}) or {}
    cfg_path = sched.get("config_path", "./config/config.yaml")

    m5_cfg = sched.get("m5", {}) or {}
    source = m5_cfg.get("source", "logs")   # logs | alpaca
    extra  = (m5_cfg.get("extra_args") or "").strip()
    outdir = m5_cfg.get("out_dir", "./data/outputs")

    cmd = [
        sys.executable, str(project_root / "src" / "m5_reconcile.py"),
        "--config", str(project_root / cfg_path),
        "--date", run_date,
        "--source", source,
        "--out-dir", str(project_root / outdir),
    ]
    if extra:
        cmd += extra.split()
    return cmd

# ===  check if csv has any rows for M4 ===
def _csv_has_data_rows(p: Path) -> bool:
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            first_data = next(reader, None)
            return bool(first_data)  # Only header/empty files are considered as no data
    except Exception:
        return False

# === check if csv has any rows for M5 ===
def _csv_has_any_rows(p: Path) -> bool:
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            import csv
            reader = csv.reader(f)
            next(reader, None)         # header
            return next(reader, None) is not None
    except Exception:
        return False


def run_once(project_root: Path, cfg: dict, run_date: str, logger: logging.Logger):
    m1_cmd, out_path = build_m1_cmd(project_root, cfg, run_date)
    m2_cmd = build_m2_cmd(project_root, cfg, out_path, run_date)

    logger.info("M1 CMD: %s", " ".join(m1_cmd))
    r1 = subprocess.run(m1_cmd, capture_output=True, text=True)
    logger.info("== M1 stdout ==\n%s", r1.stdout)
    if r1.stderr:
        logger.warning("== M1 stderr ==\n%s", r1.stderr)
    if r1.returncode != 0:
        logger.error("M1 failed with code %s", r1.returncode)
        raise SystemExit(f"M1 failed with code {r1.returncode}")

    logger.info("M2 CMD: %s", " ".join(m2_cmd))
    r2 = subprocess.run(m2_cmd, capture_output=True, text=True)
    logger.info("== M2 stdout ==\n%s", r2.stdout)
    if r2.stderr:
        logger.warning("== M2 stderr ==\n%s", r2.stderr)
    if r2.returncode != 0:
        logger.error("M2 failed with code %s", r2.returncode)
        raise SystemExit(f"M2 failed with code {r2.returncode}")


# === new: call M4, with no-order check ===

    out_orders = cfg.get("execution", {}).get("out_orders", "./data/outputs/planned_orders.csv")
    orders_csv = (project_root / out_orders)
    if not _csv_has_data_rows(orders_csv):
        logger.info("M4 skipped: no orders found in %s", orders_csv)
        return

    m4_cfg = cfg.get("scheduler", {}).get("m4", {}) or {}
    if m4_cfg.get("run", True):
        m4_cmd = build_m4_cmd(project_root, cfg)
        logger.info("M4 CMD: %s", " ".join(m4_cmd))
        r4 = subprocess.run(m4_cmd, capture_output=True, text=True)
        logger.info("== M4 stdout ==\n%s", r4.stdout)
        if r4.stderr:
            logger.warning("== M4 stderr ==\n%s", r4.stderr)
        if r4.returncode != 0:
            logger.error("M4 failed with code %s", r4.returncode)
            raise SystemExit(f"M4 failed with code {r4.returncode}")
    else:
        logger.info("M4 skipped by config (scheduler.m4.run=false)")

 # === new: M5 reconciliation ===
    m5_cfg = cfg.get("scheduler", {}).get("m5", {}) or {}
    if m5_cfg.get("run", True):

        # if there is no material (no planned/execution_log rows), skip M5
        out_orders = cfg.get("execution", {}).get("out_orders", "./data/outputs/planned_orders.csv")
        orders_csv = (project_root / out_orders)
        exec_log   = (project_root / "data" / "outputs" / "execution_log.csv")

        if not (_csv_has_any_rows(orders_csv) or _csv_has_any_rows(exec_log)):
            logger.info("M5 skipped: nothing to reconcile (no planned / exec log rows)")
            return

        m5_cmd = build_m5_cmd(project_root, cfg, run_date)
        logger.info("M5 CMD: %s", " ".join(m5_cmd))
        r5 = subprocess.run(m5_cmd, capture_output=True, text=True)
        logger.info("== M5 stdout ==\n%s", r5.stdout)
        if r5.stderr:
            logger.warning("== M5 stderr ==\n%s", r5.stderr)
        if r5.returncode != 0:
            logger.error("M5 failed with code %s", r5.returncode)
            raise SystemExit(f"M5 failed with code {r5.returncode}")
    else:
        logger.info("M5 skipped by config (scheduler.m5.run=false)")



def main():
    ap = argparse.ArgumentParser(description="M3 Scheduler/Orchestrator")
    ap.add_argument("--config", required=True, help="Path to project config.yaml")
    ap.add_argument("--date", help="Trade date (YYYY-MM-DD). Default: today")
    ap.add_argument("--once", action="store_true", help="Run once and exit (no waiting)")
    ap.add_argument("--daemon", action="store_true", help="Run forever; sleep until next scheduled time")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    ap.add_argument("--skip-weekends", action="store_true", help="Skip Sat/Sun in daemon mode")
    args = ap.parse_args()

    logger = setup_logger(debug=args.debug)
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(Path(args.config))

    if args.once:
        run_date = args.date or datetime.now().strftime("%Y-%m-%d")
        logger.info("Running once for date=%s", run_date)
        run_once(project_root, cfg, run_date, logger=logger)
        logger.info("Done for %s", run_date)
        return

    if not args.daemon:
        logger.info("Specify either --once or --daemon")
        return

    sched = cfg.get("scheduler", {})
    run_time = sched.get("time", "09:00")
    logger.info("[scheduler] starting daemon. target time=%s (local clock). Ctrl+C to stop.", run_time)
    while True:
        now = datetime.now()
        nxt = next_run_after(now, run_time)
        wait_sec = (nxt - now).total_seconds()
        logger.info("[scheduler] waiting %ss until %s", int(wait_sec), nxt)
        try:
            time.sleep(max(0, wait_sec))
        except KeyboardInterrupt:
            logger.info("exiting scheduler.")
            return
        run_date = datetime.now().strftime("%Y-%m-%d")
        if args.skip_weekends and is_weekend(datetime.now()):
            logger.info("[scheduler] %s is weekend; skip.", run_date)
            continue
        try:
            logger.info("[scheduler] starting run for %s", run_date)
            run_once(project_root, cfg, run_date, logger=logger)
            logger.info("[scheduler] Done for %s", run_date)
        except SystemExit as e:
            logger.error("[scheduler] Run failed: %s", e)

if __name__ == "__main__":
    main()
