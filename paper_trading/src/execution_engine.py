from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yaml

@dataclass
class Position:
    asset_id: str
    symbol: str
    qty: float

@dataclass
class Order:
    asset_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None

def load_targets(path: Path) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df

def load_mapping(path: Path, id_col: str, sym_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={id_col: "asset_id", sym_col: "symbol"})[["asset_id", "symbol"]]

def map_targets_to_symbols(targets: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    return targets.merge(mapping, on="asset_id", how="left").dropna(subset=["symbol"])

def load_prices_from_csv(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["symbol"], df["price"]))

def load_positions_from_csv(path: Path) -> List[Position]:
    df = pd.read_csv(path)
    
    # Check file format and process
    if 'asset_id' in df.columns and 'symbol' in df.columns and 'qty' in df.columns:
        # Standard format: asset_id, symbol, qty
        return [Position(asset_id=str(r.asset_id), symbol=str(r.symbol), qty=float(r.qty)) for r in df.itertuples(index=False)]
    elif 'symbol' in df.columns and 'qty' in df.columns:
        # Simplified format: symbol, qty (need to get asset_id from mapping file)
        print(f"Warning: Position file {path} has simplified format (symbol,qty). Asset IDs will be set to 'UNKNOWN'")
        return [Position(asset_id="UNKNOWN", symbol=str(r.symbol), qty=float(r.qty)) for r in df.itertuples(index=False)]
    else:
        print(f"Error: Position file {path} has unexpected format. Expected columns: asset_id,symbol,qty or symbol,qty")
        return []

def plan_orders(targets_df: pd.DataFrame, prices: Dict[str, float], positions: List[Position], equity: float,
                tp_pct: float=0.05, sl_pct: float=0.03) -> Tuple[List[Order], pd.DataFrame]:
    # Improved position matching logic
    cur_qty = {}
    for p in positions:
        if p.asset_id != "UNKNOWN":
            # Standard format: use asset_id matching
            cur_qty[p.asset_id] = p.qty
        else:
            # Simplified format: use symbol matching
            cur_qty[p.symbol] = p.qty
    
    targets_df = targets_df.copy()
    targets_df["price"] = targets_df["symbol"].map(prices)
    targets_df = targets_df.dropna(subset=["price"]).copy()
    targets_df["target_value"] = targets_df["target_weight"] * equity
    targets_df["target_qty"] = targets_df["target_value"] / targets_df["price"]

    orders: List[Order] = []
    plan_rows = []
    for _, r in targets_df.iterrows():
        sym = r["symbol"]
        asset = r["asset_id"]
        tgt_qty = r["target_qty"]
        
        # Try multiple matching methods
        prev_qty = 0.0
        if asset in cur_qty:
            prev_qty = cur_qty[asset]  # Priority: use asset_id matching
        elif sym in cur_qty:
            prev_qty = cur_qty[sym]    # Fallback: use symbol matching
        
        delta = tgt_qty - prev_qty
        if abs(delta) < 1e-8:
            action = "HOLD"; qty=0.0
        elif delta > 0:
            # Target position > current position, need to buy
            if prev_qty < 0:
                # Current is short position, need to close short first then buy
                # 1. Close short position first (BUY back to return)
                close_short_qty = abs(prev_qty)
                orders.append(Order(asset_id=asset,symbol=sym,side="BUY",qty=close_short_qty,order_type="MARKET"))
                
                # 2. Then buy target position
                buy_qty = tgt_qty
                base = r["price"]
                
                # Check if fractional shares
                is_fractional = (buy_qty % 1) != 0
                
                if is_fractional:
                                    # Fractional shares use simple order types
                    orders.append(Order(asset_id=asset,symbol=sym,side="BUY",qty=buy_qty,order_type="MARKET",
                                        tp_price=None,sl_price=None))
                else:
                                    # Whole shares use BRACKET orders
                    orders.append(Order(asset_id=asset,symbol=sym,side="BUY",qty=buy_qty,order_type="BRACKET",
                                        tp_price=round(base*(1+tp_pct),2),sl_price=round(base*(1-sl_pct),2)))
                
                action = "CLOSE_SHORT_AND_BUY"
                qty = buy_qty
            else:
                # Current is long or zero position, direct buy
                action="BUY"; qty=delta
                base = r["price"]
                
                # Check if fractional shares (decimal part not 0)
                is_fractional = (qty % 1) != 0
                
                if is_fractional:
                    # Fractional shares use simple order types
                    orders.append(Order(asset_id=asset,symbol=sym,side="BUY",qty=qty,order_type="MARKET",
                                        tp_price=None,sl_price=None))
                else:
                    # Whole shares use BRACKET orders
                    orders.append(Order(asset_id=asset,symbol=sym,side="BUY",qty=qty,order_type="BRACKET",
                                        tp_price=round(base*(1+tp_pct),2),sl_price=round(base*(1-sl_pct),2)))
        else:
            # Target position < current position, need to sell
            if prev_qty > 0:
                # Current is long position, need to sell
                action="SELL"; qty=-delta
                orders.append(Order(asset_id=asset,symbol=sym,side="SELL",qty=qty,order_type="MARKET"))
            else:
                # Current is short or zero position, need to sell short
                action="SHORT"; qty=-delta
                orders.append(Order(asset_id=asset,symbol=sym,side="SELL",qty=qty,order_type="MARKET"))
        
        plan_rows.append({"asset_id":asset,"symbol":sym,"price":r["price"],
                          "target_weight":r["target_weight"],"target_qty":tgt_qty,
                          "current_qty":prev_qty,"delta":delta,"action":action,"planned_qty":qty})
    plan_df = pd.DataFrame(plan_rows)
    return orders, plan_df

def save_orders_csv(orders: List[Order], path: Path):
    pd.DataFrame([asdict(o) for o in orders]).to_csv(path,index=False)

def save_plan_csv(plan_df: pd.DataFrame, path: Path):
    if str(path).endswith(".parquet"):
        plan_df.to_parquet(path,index=False)
    else:
        plan_df.to_csv(path,index=False)

def generate_real_orders(orders: List[Order], plan_df: pd.DataFrame, 
                        min_order_notional: float = 2.0,
                        quantity_precision: int = 2,
                        max_total_weight: float = 1.0) -> Tuple[List[Order], pd.DataFrame]:
    """
    Generate actual executable orders, handling fractional shares and minimum order value
    
    Args:
        orders: Original order list
        plan_df: Rebalance plan dataframe
        min_order_notional: Minimum order value
        quantity_precision: Quantity precision
        max_total_weight: Maximum total weight
    
    Returns:
        Tuple[List[Order], pd.DataFrame]: Actual order list and adjusted plan
    """
    # 1. Filter orders with less than minimum order value
    valid_orders = []
    filtered_plan_rows = []
    
    for order in orders:
        # Calculate order value (need to get price from plan_df)
        order_plan = plan_df[plan_df['symbol'] == order.symbol].iloc[0]
        price = order_plan['price']
        notional = order.qty * price
        
        if notional >= min_order_notional:
            valid_orders.append(order)
            filtered_plan_rows.append(order_plan)
        else:
            print(f"[M2] Skipping {order.symbol}: notional ${notional:.2f} < min ${min_order_notional}")
    
    if not valid_orders:
        return [], pd.DataFrame()
    
    # 2. Adjust quantity precision (round to specified decimal places)
    adjusted_orders = []
    for order in valid_orders:
        adjusted_qty = round(order.qty, quantity_precision)
        adjusted_order = Order(
            asset_id=order.asset_id,
            symbol=order.symbol,
            side=order.side,
            qty=adjusted_qty,
            order_type=order.order_type,
            tp_price=order.tp_price,
            sl_price=order.sl_price
        )
        adjusted_orders.append(adjusted_order)
    
    # 3. Check and adjust total weight overflow
    adjusted_plan_df = pd.DataFrame(filtered_plan_rows)
    
    # Calculate adjusted weights
    total_adjusted_weight = 0.0
    for _, row in adjusted_plan_df.iterrows():
        symbol = row['symbol']
        # Find corresponding adjusted order
        adjusted_order = next((o for o in adjusted_orders if o.symbol == symbol), None)
        if adjusted_order:
            # Recalculate weights
            adjusted_weight = (adjusted_order.qty * row['price']) / (row['target_qty'] * row['price'] / row['target_weight'])
            adjusted_plan_df.loc[adjusted_plan_df['symbol'] == symbol, 'adjusted_weight'] = adjusted_weight
            total_adjusted_weight += adjusted_weight
    
    # If total weight exceeds limit, adjust proportionally
    if total_adjusted_weight > max_total_weight and len(adjusted_orders) > 0:
        print(f"[M2] Total weight {total_adjusted_weight:.4f} > {max_total_weight}, adjusting...")
        
        # Adjust last order proportionally by weight
        adjustment_factor = max_total_weight / total_adjusted_weight
        last_order = adjusted_orders[-1]
        last_symbol = last_order.symbol
        
        # Adjust quantity of last order
        original_qty = last_order.qty
        adjusted_qty = round(original_qty * adjustment_factor, quantity_precision)
        
        # Update order
        adjusted_orders[-1] = Order(
            asset_id=last_order.asset_id,
            symbol=last_order.symbol,
            side=last_order.side,
            qty=adjusted_qty,
            order_type=last_order.order_type,
            tp_price=last_order.tp_price,
            sl_price=last_order.sl_price
        )
        
        # Update plan data
        adjusted_plan_df.loc[adjusted_plan_df['symbol'] == last_symbol, 'planned_qty'] = adjusted_qty
        print(f"[M2] Adjusted {last_symbol}: {original_qty:.4f} -> {adjusted_qty:.4f}")
    
    return adjusted_orders, adjusted_plan_df
