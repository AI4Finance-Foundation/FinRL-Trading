from pathlib import Path
import importlib.util, sys, pandas as pd

spec = importlib.util.spec_from_file_location("execution_engine", str(Path(__file__).parents[1]/"src"/"execution_engine.py"))
ee = importlib.util.module_from_spec(spec)
sys.modules["execution_engine"]=ee
spec.loader.exec_module(ee)

def test_orders():
    targets=pd.DataFrame({"trade_date":["2025-03-01","2025-03-01"],
                          "asset_id":["A","B"],"target_weight":[0.6,0.4],"symbol":["AAPL","MSFT"]})
    prices={"AAPL":100.0,"MSFT":200.0}
    positions=[ee.Position(asset_id="A",symbol="AAPL",qty=0.0),ee.Position(asset_id="B",symbol="MSFT",qty=10.0)]
    orders,plan=ee.plan_orders(targets,prices,positions,equity=10000.0)
    assert any(o.side=="BUY" and o.order_type=="BRACKET" for o in orders)
    assert any(o.side=="SELL" or o.side=="BUY" for o in orders)
    print("M2 tests passed")

if __name__=="__main__":
    test_orders()
