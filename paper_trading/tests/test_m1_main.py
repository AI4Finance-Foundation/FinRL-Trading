
from pathlib import Path
import importlib.util, sys
import pandas as pd

# load module
spec = importlib.util.spec_from_file_location("signal_bus", str(Path(__file__).parents[1] / "src" / "signal_bus.py"))
sb = importlib.util.module_from_spec(spec)
sys.modules["signal_bus"] = sb
spec.loader.exec_module(sb)

def test_T1_overlapping():
    specs = sb.load_manifest(Path("./paper_trading/config/strategy_manifest.yaml"))
    df_all = sb.load_all_strategies(specs)
    latest = sb.pick_latest_per_strategy(df_all)
    merged = sb.merge_for_date(latest, "2018-03-01", mode="select", blend={"drl":1.0})
    final = sb.enforce_constraints_and_normalize(merged, allow_short=False, allow_cash=False)
    assert (final['target_weight'] >= -1e-12).all()
    assert abs(final['target_weight'].sum() - 1.0) < 1e-8
    print("T1 OK:", final.to_dict("records"))

def test_T2_negative():
    tmp = Path("./paper_trading/data/raw_signals/tmp_neg.csv")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "trade_date":["2018-03-01"], "gvkey":["Z"], "weights":[-0.1]
    }).to_csv(tmp, index=False)
    bad_manifest = Path("./paper_trading/config/manifest_bad.yaml")
    bad_manifest.write_text("""strategies:
  - id: bad
    path: ./paper_trading/data/raw_signals/tmp_neg.csv
    format: csv
    column_mapping: { trade_date: trade_date, gvkey: asset_id, weights: target_weight }
""", encoding="utf-8")
    specs = sb.load_manifest(bad_manifest)
    df_all = sb.load_all_strategies(specs)
    latest = sb.pick_latest_per_strategy(df_all)
    merged = sb.merge_for_date(latest, "2018-03-01", mode="select", blend={"bad":1.0})
    try:
        sb.enforce_constraints_and_normalize(merged, allow_short=False, allow_cash=False)
    except sb.SignalValidationError as e:
        print("T2 OK (caught):", str(e)[:120])
        return
    raise AssertionError("T2 failed: negative weights not caught")

if __name__ == "__main__":
    test_T1_overlapping()
    test_T2_negative()
    print("All M1 tests passed.")
