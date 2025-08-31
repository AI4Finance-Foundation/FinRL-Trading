
import argparse
from pathlib import Path
import sys
import importlib.util

# load local signal_bus module dynamically
spec = importlib.util.spec_from_file_location("signal_bus", str(Path(__file__).parent / "signal_bus.py"))
sb = importlib.util.module_from_spec(spec)
sys.modules["signal_bus"] = sb
spec.loader.exec_module(sb)

def parse_blend(arg):
    if not arg:
        return None
    out = {}
    for part in arg.split(","):
        k,v = part.split("=")
        out[k.strip()] = float(v)
    return out

def main():
    ap = argparse.ArgumentParser(description="Run Signal Bus to produce targets for a given date.")
    ap.add_argument("--manifest", required=True, help="Path to strategy_manifest.yaml")
    ap.add_argument("--date", required=True, help="Trade date, e.g., 2018-03-01")
    ap.add_argument("--mode", choices=["select","blend"], default="select")
    ap.add_argument("--pick", help="Strategy id to select when mode=select")
    ap.add_argument("--blend", help="Strategy blend string like 'drl=0.5,minvar=0.5' when mode=blend")
    ap.add_argument("--out", required=True, help="Output file path (.csv or .parquet)")
    args = ap.parse_args()

    specs = sb.load_manifest(Path(args.manifest))
    df_all = sb.load_all_strategies(specs)
    latest = sb.pick_latest_per_strategy(df_all)

    if args.mode == "select":
        if args.pick:
            merged = sb.merge_for_date(latest, args.date, mode="select", blend={args.pick:1.0})
        else:
            merged = sb.merge_for_date(latest, args.date, mode="select", blend=None)
    else:
        blend = parse_blend(args.blend)
        merged = sb.merge_for_date(latest, args.date, mode="blend", blend=blend)

    final = sb.enforce_constraints_and_normalize(merged, allow_short=False, allow_cash=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sb.save_targets(final, out_path)
    print(f"Saved targets: {out_path}  rows={len(final)}")

if __name__ == "__main__":
    main()
