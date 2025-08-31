
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

@dataclass
class ExecOrder:
    asset_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    client_order_id: Optional[str] = None

def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load_config(path: Optional[str]) -> dict:
    cfg_path = Path(path or "./config/config.yaml")
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def read_planned_orders(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["asset_id", "symbol", "side", "qty", "order_type"]:
        if c not in df.columns:
            raise SystemExit(f"planned_orders.csv missing required column: {c}")
    if "tp_price" not in df.columns:
        df["tp_price"] = None
    if "sl_price" not in df.columns:
        df["sl_price"] = None
    return df

def derive_orders_from_plan(plan_path: Path) -> pd.DataFrame:
    if not plan_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(plan_path)
    for c in ["asset_id", "symbol", "action", "planned_qty"]:
        if c not in df.columns:
            raise SystemExit(f"rebalance_plan.csv missing required column: {c}")
    rows = []
    for r in df.itertuples(index=False):
        action = str(getattr(r, "action")).upper()
        qty = float(getattr(r, "planned_qty"))
        if action == "HOLD" or abs(qty) < 1e-12:
            continue
        elif action == "BUY":
            side = "BUY"
            # Check if fractional shares (decimal part not 0)
            qty_abs = abs(qty)
            is_fractional = (qty_abs % 1) != 0
            
            if is_fractional:
                # Fractional shares use simple order types
                order_type = "MARKET"
                tp_price, sl_price = None, None
            else:
                # Whole shares use BRACKET orders
                order_type = "BRACKET"
                tp_price = getattr(r, "price", None)
                sl_price = getattr(r, "price", None)
                if tp_price is not None and pd.notna(tp_price):
                    tp_price = round(float(tp_price) * 1.0, 2)
                else:
                    tp_price = None
                if sl_price is not None and pd.notna(sl_price):
                    sl_price = round(float(sl_price) * 1.0, 2)
                else:
                    sl_price = None
        elif action == "SELL":
            side = "SELL"
            order_type = "MARKET"
            tp_price, sl_price = None, None
        else:
            continue
        rows.append({
            "asset_id": str(getattr(r, "asset_id")),
            "symbol": str(getattr(r, "symbol")),
            "side": side,
            "qty": abs(qty),
            "order_type": order_type,
            "tp_price": tp_price,
            "sl_price": sl_price,
        })
    return pd.DataFrame(rows, columns=["asset_id","symbol","side","qty","order_type","tp_price","sl_price"])

def df_to_orders(df: pd.DataFrame) -> List[ExecOrder]:
    out: List[ExecOrder] = []
    for r in df.itertuples(index=False):
        out.append(ExecOrder(
            asset_id=str(getattr(r, "asset_id")),
            symbol=str(getattr(r, "symbol")),
            side=str(getattr(r, "side")).upper(),
            qty=float(getattr(r, "qty")),
            order_type=str(getattr(r, "order_type")).upper(),
            tp_price=(None if pd.isna(getattr(r,"tp_price", None)) else float(getattr(r, "tp_price", None))),
            sl_price=(None if pd.isna(getattr(r,"sl_price", None)) else float(getattr(r, "sl_price", None))),
        ))
    return out

def load_prices_from_plan_or_none(plan_path: Path) -> Dict[str, float]:
    if not plan_path.exists():
        return {}
    df = pd.read_csv(plan_path)
    if "symbol" in df.columns and "price" in df.columns:
        m = df.dropna(subset=["symbol","price"])
        return dict(zip(m["symbol"].astype(str), pd.to_numeric(m["price"], errors="coerce").fillna(0.0)))
    return {}

def make_client_order_id(prefix: str, trade_date: str, o: ExecOrder) -> str:
    basis = f"{trade_date}|{o.symbol}|{o.side}|{o.qty:.8f}|{o.order_type}|{o.tp_price}|{o.sl_price}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{trade_date.replace('-','')}_{o.symbol}_{o.side}_{digest}"

def read_existing_ids_from_log(log_path: Path) -> set:
    if not log_path.exists():
        return set()
    seen = set()
    with log_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            cid = row.get("client_order_id")
            status = row.get("status")
            if cid and status in ("SUBMITTED", "FILLED", "ACCEPTED", "SIMULATED"):
                seen.add(cid)
    return seen

def risk_check_orders(orders: List[ExecOrder], prices: Dict[str, float], acct: Optional[dict], risk: dict) -> Tuple[List[ExecOrder], List[str]]:
    rej, kept = [], []
    max_share = float(risk.get("max_share_per_order", float("inf")))
    max_notional = float(risk.get("max_notional_per_order", float("inf")))
    max_total = float(risk.get("max_total_notional", float("inf")))
    tol_pct = float(risk.get("price_tolerance_pct", 1.0))
    min_cash_buf = float(risk.get("min_cash_buffer_pct", 0.0))
    reject_if_missing = bool(risk.get("reject_if_price_missing", False))

    batch_notional = 0.0
    for o in orders:
        if o.qty <= 0:
            rej.append(f"[{o.symbol}] reject: qty<=0"); continue
        if o.qty > max_share:
            rej.append(f"[{o.symbol}] reject: qty {o.qty} > max_share_per_order {max_share}"); continue
        px = prices.get(o.symbol)
        if px is None:
            if reject_if_missing:
                rej.append(f"[{o.symbol}] reject: missing price and reject_if_price_missing=true"); continue
            else:
                px = 0.0
        notional = o.qty * float(px or 0.0)
        if px and notional > max_notional:
            rej.append(f"[{o.symbol}] reject: notional {notional:.2f} > max_notional_per_order {max_notional}"); continue
        batch_notional += notional
        kept.append(o)

    if batch_notional > max_total:
        rej.append(f"[BATCH] reject: total notional {batch_notional:.2f} > max_total_notional {max_total}")

    if acct and min_cash_buf > 0:
        try:
            equity = float(acct.get("equity") or 0.0)
            cash = float(acct.get("cash") or acct.get("buying_power") or 0.0)
            if cash < equity * min_cash_buf:
                rej.append(f"[ACCOUNT] reject: cash/buying_power {cash:.2f} < min_cash_buffer {min_cash_buf*100:.1f}% of equity {equity:.2f}")
        except Exception:
            pass

    return kept, rej

class AlpacaClient:
    def __init__(self, trading_base: str, data_base: str, key_id: Optional[str], secret: Optional[str]):
        self.trading = trading_base.rstrip("/")
        self.data = data_base.rstrip("/")
        self.sess = requests.Session()
        if key_id and secret:
            self.sess.headers.update({
                "APCA-API-KEY-ID": key_id,
                "APCA-API-SECRET-KEY": secret,
                "Content-Type": "application/json"
            })

    def get_account(self) -> Optional[dict]:
        """获取账户信息"""
        try:
            response = self.sess.get(f"{self.trading}/account", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting account: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Exception getting account: {e}")
            return None

    def get_positions(self) -> List[dict]:
        """获取当前持仓"""
        try:
            response = self.sess.get(f"{self.trading}/positions", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting positions: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Exception getting positions: {e}")
            return []

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest trading prices"""
        if not symbols:
            return {}
        
        batch_size = 200  # Alpaca API limit
        all_prices = {}
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Fix: Use correct API parameter format
            params = {
                'symbols': ','.join(batch)
                # Remove 'feed': 'sip' parameter as it may cause permission issues
            }
            
            try:
                response = self.sess.get(
                    f"{self.data}/stocks/latest/trades",
                    params=params,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for trade in data.get('trades', {}).values():
                        symbol = trade.get('S')
                        price = trade.get('p')
                        if symbol and price:
                            all_prices[symbol] = float(price)
                elif response.status_code == 403:
                    print(f"Warning: API access denied for latest trades. Status: {response.status_code}")
                    print(f"Response: {response.text}")
                else:
                    print(f"Warning: Failed to fetch latest trades. Status: {response.status_code}")
                    print(f"Response: {response.text}")
                
                time.sleep(0.1)  # Avoid API rate limiting
                
            except Exception as e:
                print(f"Error fetching prices for batch {batch[:5]}: {e}")
                continue
        
        return all_prices

    def get_snapshot_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get snapshot prices (with more market information)"""
        if not symbols:
            return {}
        
        batch_size = 200
        all_prices = {}
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Fix: Remove parameters that may cause permission issues
            params = {
                'symbols': ','.join(batch)
                # Remove 'feed': 'sip' parameter
            }
            
            try:
                print(f"Fetching snapshot for batch {i//batch_size + 1}: {len(batch)} symbols")
                print(f"Sample symbols in batch: {batch[:5]}")
                
                response = self.sess.get(
                    f"{self.data}/stocks/snapshots",
                    params=params,
                    timeout=10
                )
                
                print(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Response text: {response.text}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    # Fix: Iterate directly over data key-value pairs instead of data.values()
                    for symbol, snapshot in data.items():
                        if isinstance(snapshot, dict):
                            latest_trade = snapshot.get('latestTrade', {})
                            price = latest_trade.get('p')
                            if price:
                                all_prices[symbol] = float(price)
                                print(f"  Found price for {symbol}: ${price}")
                        else:
                            print(f"  Skipping {symbol}: not a dict")
                    
                    # Debug: Check the structure of the first snapshot
                    if data and isinstance(data, dict):
                        first_symbol = list(data.keys())[0]
                        first_snapshot = data[first_symbol]
                        print(f"Sample snapshot for {first_symbol}: {first_snapshot}")
                elif response.status_code == 403:
                    print(f"Warning: API access denied for snapshots. Status: {response.status_code}")
                    print(f"Response: {response.text}")
                else:
                    print(f"Warning: Failed to fetch snapshots. Status: {response.status_code}")
                    print(f"Response: {response.text}")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching snapshots for batch {batch[:5]}: {e}")
                continue
        
        print(f"Total prices fetched: {len(all_prices)}")
        return all_prices

    def submit_order(self, o: ExecOrder, tif: str, client_order_id: Optional[str]):
        side = "buy" if o.side.upper()=="BUY" else "sell"
        tif = (tif or "day").lower()  # Ensure time_in_force is lowercase
        payload = {
            "symbol": o.symbol, "qty": o.qty, "side": side, "type": "market", "time_in_force": tif
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id
        if o.side.upper()=="BUY" and o.order_type.upper()=="BRACKET":
            payload["order_class"] = "bracket"
            if o.tp_price: payload["take_profit"] = {"limit_price": float(o.tp_price)}
            if o.sl_price: payload["stop_loss"] = {"stop_price": float(o.sl_price)}
        try:
            r = self.sess.post(f"{self.trading}/orders", data=json.dumps(payload), timeout=10)
            if r.ok:
                j = r.json(); return ("SUBMITTED", j.get("id"), None)
            else:
                return ("FAILED", None, f"{r.status_code} {r.text}")
        except Exception as e:
            return ("FAILED", None, str(e))

    def poll_fills_once(self):
        try:
            r = self.sess.get(f"{self.trading}/orders?status=all&limit=200&nested=true", timeout=10)
            if r.ok:
                data = r.json(); return [o for o in data if o.get("filled_at")]
        except Exception:
            return []
        return []

def append_execution_log(path: Path, rows: List[dict]):
    _ensure_dir(path)
    existed = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts","mode","symbol","side","qty","order_type","tp_price","sl_price",
            "time_in_force","client_order_id","price_used","notional","batch_id","status","order_id","error"
        ])
        if not existed: w.writeheader()
        for r in rows: w.writerow(r)

def write_filled_orders(path: Path, fills: List[dict]):
    _ensure_dir(path)
    cols = ["id","client_order_id","symbol","qty","filled_qty","filled_at","status","side","type","limit_price","stop_price"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for o in fills:
            w.writerow({k: o.get(k) for k in cols})

def main():
    ap = argparse.ArgumentParser(description="Alpaca Execution Runner (M4)")
    ap.add_argument("--config", default="./config/config.yaml")
    ap.add_argument("--mode", choices=["dry","real"], default="dry")
    ap.add_argument("--orders")
    ap.add_argument("--plan")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    
            # Generate batch ID
    batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[M4] Execution batch ID: {batch_id}")

    cfg = load_config(args.config)
    exec_cfg = cfg.get("execution", {})
            # Prioritize reading real version of order files
    default_orders = exec_cfg.get("out_orders", "./data/outputs/planned_orders.csv")
    if default_orders.endswith(".csv"):
        default_orders = default_orders.replace(".csv", "_real.csv")
    orders_path = Path(args.orders or default_orders)
    plan_path = Path(args.plan or exec_cfg.get("out_plan", "./data/outputs/rebalance_plan.csv"))
    alp_cfg = cfg.get("alp_execution", {})
    max_orders = int(alp_cfg.get("max_orders", 100))
    poll_fills = int(alp_cfg.get("poll_fills", 1))
    risk_cfg = cfg.get("risk", {})
    tif = exec_cfg.get("time_in_force", "DAY")
    idem_cfg = cfg.get("idempotency", {})
    idem_enable = bool(idem_cfg.get("enable_client_order_id", True))
    idem_prefix = str(idem_cfg.get("client_order_id_prefix", "ORD"))
    outputs_dir = Path("./data/outputs")
    execution_log_path = outputs_dir / "execution_log.csv"
    filled_orders_path = outputs_dir / "filled_orders.csv"

    df_orders = read_planned_orders(orders_path)
    if df_orders.empty:
        if args.debug:
            print(f"[INFO] planned_orders not found at {orders_path}, derive from {plan_path}")
        df_orders = derive_orders_from_plan(plan_path)
        if df_orders.empty:
            raise SystemExit("No orders available")

    if len(df_orders) > max_orders:
        if args.debug:
            print(f"[INFO] Truncating orders {len(df_orders)} -> {max_orders}")
        df_orders = df_orders.head(max_orders).copy()

    orders = df_to_orders(df_orders)
    
    # Sort by execution order: execute SELL orders first, then BUY orders
    def order_priority(order):
        # SELL orders have priority 0 (execute first), BUY orders have priority 1 (execute later)
        return 0 if order.side == "SELL" else 1
    
    orders.sort(key=order_priority)
    
    if args.debug:
        sell_count = sum(1 for o in orders if o.side == "SELL")
        buy_count = sum(1 for o in orders if o.side == "BUY")
        print(f"[M4] Execution order: {sell_count} SELL orders first, then {buy_count} BUY orders")
    
    prices = load_prices_from_plan_or_none(plan_path)

    alp = None; acct = None
    if args.mode == "real":
        alp_conf = cfg.get("alpaca", {})
        key_id = os.getenv("APCA_API_KEY_ID") or alp_conf.get("api_key")
        secret = os.getenv("APCA_API_SECRET_KEY") or alp_conf.get("api_secret")
        trading_base = alp_conf.get("trading_endpoint", "").rstrip("/")
        data_base = alp_conf.get("data_endpoint", "").rstrip("/")
        if not (key_id and secret and trading_base):
            raise SystemExit("Alpaca credentials missing for real mode.")
        alp = AlpacaClient(trading_base, data_base or trading_base, key_id, secret)
        acct = alp.get_account()

    kept, rejected_msgs = risk_check_orders(orders, prices, acct, risk_cfg)
    if args.debug:
        print("[RISK] kept:", len(kept), "rejected:", len(rejected_msgs))
        for m in rejected_msgs[:6]: print("  -", m)

    trade_date = datetime.now().strftime("%Y-%m-%d")
    if idem_enable:
        seen = read_existing_ids_from_log(execution_log_path)
        for o in kept:
            o.client_order_id = make_client_order_id(idem_prefix, trade_date, o)
        before = len(kept)
        kept = [o for o in kept if o.client_order_id not in seen]
        if args.debug and before != len(kept):
            print(f"[IDEMPOTENCY] skipped {before-len(kept)} duplicates")

    rows_for_log = []
    for o in kept:
        px = prices.get(o.symbol)
        notional = float(o.qty) * float(px or 0.0)
        base_row = {
            "ts": _now_iso(),
            "mode": args.mode.upper(),
            "symbol": o.symbol,
            "side": o.side,
            "qty": f"{o.qty:.0f}" if abs(o.qty - int(o.qty)) < 1e-9 else f"{o.qty:.6f}",
            "order_type": o.order_type,
            "tp_price": ("" if o.tp_price is None else f"{o.tp_price:.2f}"),
            "sl_price": ("" if o.sl_price is None else f"{o.sl_price:.2f}"),
            "time_in_force": tif,
            "client_order_id": o.client_order_id or "",
            "price_used": ("" if px is None else f"{px:.4f}"),
            "notional": f"{notional:.2f}",
            "batch_id": batch_id,  # Add batch ID
        }
        if args.mode == "dry":
            row = dict(base_row); row.update({"status": "SIMULATED", "order_id": "", "error": ""})
        else:
            status, oid, error = alp.submit_order(o, tif, o.client_order_id)
            row = dict(base_row); row.update({"status": status, "order_id": oid or "", "error": error or ""})
        rows_for_log.append(row)

    if rows_for_log:
        append_execution_log(execution_log_path, rows_for_log)
        print(f"[M4] wrote {len(rows_for_log)} rows to {execution_log_path}")

    if args.mode == "real" and poll_fills > 0 and alp is not None:
        fills_all = []
        try:
            for _ in range(int(poll_fills)):
                fills = alp.poll_fills_once()
                if fills:
                    fills_all = fills; break
        except Exception:
            fills_all = []
        if fills_all:
            write_filled_orders(filled_orders_path, fills_all)
            print(f"[M4] wrote filled orders -> {filled_orders_path} (n={len(fills_all)})")
    elif args.mode == "dry" and poll_fills > 0 and args.debug:
        print("[M4] dry mode: poll_fills ignored")

    total = len(orders); kept_n = len(kept); skipped = total - kept_n
    print(f"[M4] total_orders={total}, kept={kept_n}, skipped_local_idem={skipped}, rejected={len(rejected_msgs)}")

def fetch_prices_for_targets(cfg: dict, targets_df: pd.DataFrame, mapping_df: pd.DataFrame, 
                           price_source: str = "latest", debug: bool = False) -> Dict[str, float]:
    """
            Get prices for target weights
            
            Args:
                cfg: configuration file
                targets_df: target weights dataframe
                mapping_df: mapping dataframe
                price_source: price source ("latest" or "snapshot")
                debug: whether to print debug information
            
            Returns:
                price dictionary {symbol: price}
    """
            # Create asset_id to symbol mapping
    id_to_symbol = dict(zip(mapping_df['gvkey'], mapping_df['tic']))
    
            # Get required stock symbols
    needed_symbols = []
    for asset_id in targets_df['asset_id'].unique():
        if asset_id in id_to_symbol:
            needed_symbols.append(id_to_symbol[asset_id])
    
    if debug:
        print(f"Fetching prices for {len(needed_symbols)} symbols...")
        print(f"Sample symbols: {needed_symbols[:5]}")
    
            # Create Alpaca client
    alp_conf = cfg.get("alpaca", {})
    key_id = os.getenv("APCA_API_KEY_ID") or alp_conf.get("api_key")
    secret = os.getenv("APCA_API_SECRET_KEY") or alp_conf.get("api_secret")
    trading_base = alp_conf.get("trading_endpoint", "").rstrip("/")
    data_base = alp_conf.get("data_endpoint", "").rstrip("/")
    
    if debug:
        print(f"API Key: {key_id[:10]}...")
        print(f"Trading base: {trading_base}")
        print(f"Data base: {data_base}")
    
    if not (key_id and secret and data_base):
        print("Warning: Alpaca credentials missing, using fallback prices")
        return {}
    
            # Fix: Use correct trading_base
    alp = AlpacaClient(trading_base, data_base, key_id, secret)
    
            # Get prices
    if price_source == "snapshot":
        if debug:
            print(f"Using snapshot API for {len(needed_symbols)} symbols")
        prices = alp.get_snapshot_prices(needed_symbols)
    else:
        if debug:
            print(f"Using latest API for {len(needed_symbols)} symbols")
        prices = alp.get_latest_prices(needed_symbols)
    
    if debug:
        print(f"Successfully fetched {len(prices)} prices")
        if prices:
            print(f"Sample prices: {dict(list(prices.items())[:5])}")
        missing = set(needed_symbols) - set(prices.keys())
        if missing:
            print(f"Missing prices for {len(missing)} symbols: {list(missing)[:10]}")
    
    return prices

def save_prices_to_csv(prices: Dict[str, float], output_path: str):
    """Save prices to CSV file"""
    if not prices:
        return
    
    df = pd.DataFrame([
        {'symbol': symbol, 'price': price}
        for symbol, price in prices.items()
    ])
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(prices)} prices to {output_path}")

if __name__ == "__main__":
    main()
