
from pathlib import Path
import importlib.util, sys

spec = importlib.util.spec_from_file_location("scheduler", str(Path(__file__).parents[1] / "src" / "scheduler.py"))
sch = importlib.util.module_from_spec(spec)
sys.modules["scheduler"] = sch
spec.loader.exec_module(sch)

def test_build_cmds():
    cfg = {
        "scheduler": {
            "m1_manifest": "./config/strategy_manifest.yaml",
            "m1": {"mode":"select","pick":"drl"},
            "m2": {"equity": 10000},
            "config_path": "./config/config.yaml",
        },
        "execution": {
            "out_orders": "./data/outputs/planned_orders.csv",
            "out_plan": "./data/outputs/rebalance_plan.csv"
        }
    }
    root = Path(__file__).parents[1]
    m1_cmd, out_path = sch.build_m1_cmd(root, cfg, "2025-03-01")
    m2_cmd = sch.build_m2_cmd(root, cfg, out_path)
    assert "--date" in m1_cmd and "2025-03-01" in m1_cmd
    assert "--config" in m2_cmd and "--targets" in m2_cmd
    print("M3 build cmds OK")

if __name__ == "__main__":
    test_build_cmds()
    print("All M3 tests passed.")
