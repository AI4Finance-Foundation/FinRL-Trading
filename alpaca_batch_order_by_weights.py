import os
import sys
import math
import argparse
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import pandas as pd
import requests

# 复用已实现的 Alpaca 请求与账户查询/鉴权
from alpaca_paper_trade import request_alpaca, get_account, get_auth_headers  # type: ignore


def read_latest_weights(weights_csv: str) -> List[Tuple[int, float, str]]:
    """
    读取 drl_weight.csv，取最后一个 trade_date 的所有 (gvkey, weight)。

    返回列表项为 (gvkey, weight, trade_date_str)
    """
    df = pd.read_csv(weights_csv, usecols=["trade_date", "gvkey", "weights"])  # 忽略可能的无名索引列
    if df.empty:
        raise RuntimeError("drl_weight.csv 为空")
    # 找到最后一个交易日
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    last_date = df["trade_date"].max()
    if pd.isna(last_date):
        raise RuntimeError("无法解析 trade_date 列")
    df_last = df[df["trade_date"] == last_date].copy()
    if df_last.empty:
        raise RuntimeError("未找到最后交易日的数据")
    # 转为期望类型
    df_last["gvkey"] = pd.to_numeric(df_last["gvkey"], errors="coerce").astype("Int64")
    df_last["weights"] = pd.to_numeric(df_last["weights"], errors="coerce")
    df_last = df_last.dropna(subset=["gvkey", "weights"])  # type: ignore[arg-type]
    result: List[Tuple[int, float, str]] = [
        (int(row.gvkey), float(row.weights), last_date.strftime("%Y-%m-%d"))
        for row in df_last.itertuples(index=False)
        if float(row.weights) is not None
    ]
    if not result:
        raise RuntimeError("最后交易日没有有效权重记录")
    return result


def build_gvkey_to_tic_map(final_ratios_csv: str, gvkeys: List[int]) -> Dict[int, str]:
    """
    从 final_ratios_20250712.csv 构建 gvkey -> 最新 tic 映射。
    只读取必要列并过滤到给定 gvkeys。
    """
    usecols = ["date", "gvkey", "tic"]
    df = pd.read_csv(final_ratios_csv, usecols=usecols)
    if df.empty:
        raise RuntimeError("final_ratios 文件为空或列名不匹配")
    df = df[df["gvkey"].isin(gvkeys)].copy()
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # 仅保留有日期的记录
    # 取每个 gvkey 最新日期的记录来决定 tic
    df = df.sort_values(["gvkey", "date"]).drop_duplicates(subset=["gvkey"], keep="last")
    mapping: Dict[int, str] = {int(row.gvkey): str(row.tic) for row in df.itertuples(index=False)}
    return mapping


def request_alpaca_data(path: str, *, params: Optional[Dict[str, object]] = None, timeout: int = 30) -> Dict[str, object]:
    base = os.environ.get("APCA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
    url = f"{base}{path}"
    headers = get_auth_headers()
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    if resp.status_code >= 400:
        try:
            info = resp.json()
        except Exception:
            info = {"message": resp.text}
        raise RuntimeError(f"数据API错误: {resp.status_code} {info}")
    return resp.json()  # type: ignore[return-value]


def get_asset(symbol: str) -> Dict[str, object]:
    return request_alpaca("GET", f"/v2/assets/{symbol.upper()}")


def get_latest_price(symbol: str) -> Optional[float]:
    # 优先使用最新成交价
    try:
        data = request_alpaca_data(f"/v2/stocks/{symbol.upper()}/trades/latest")
        trade = data.get("trade") if isinstance(data, dict) else None
        if isinstance(trade, dict) and trade.get("p") is not None:
            return float(trade["p"])  # 成交价
    except Exception:
        pass
    # 回退到最新报价的买一/卖一中点
    try:
        data = request_alpaca_data(f"/v2/stocks/{symbol.upper()}/quotes/latest")
        quote = data.get("quote") if isinstance(data, dict) else None
        if isinstance(quote, dict) and quote.get("ap") is not None and quote.get("bp") is not None:
            ask = float(quote["ap"])  # 卖一
            bid = float(quote["bp"])  # 买一
            if ask > 0 and bid > 0:
                return (ask + bid) / 2.0
    except Exception:
        pass
    return None


def fetch_cash() -> float:
    account = get_account()
    cash_str = account.get("cash")
    if cash_str is None:
        raise RuntimeError("账户信息缺少 cash 字段")
    try:
        return float(cash_str)
    except Exception as exc:
        raise RuntimeError(f"无法解析账户现金: {cash_str}") from exc


def normalize_positive_weights(items: List[Tuple[int, float, str]]) -> List[Tuple[int, float, str]]:
    """仅对正权重进行归一化，非正权重将被忽略。"""
    positives = [(g, w, d) for (g, w, d) in items if w is not None and w > 0]
    total = sum(w for (_, w, _) in positives)
    if total <= 0:
        raise RuntimeError("正权重之和 <= 0，无法归一化")
    return [(g, w / total, d) for (g, w, d) in positives]


def place_market_notional(symbol: str, notional: float, side: str = "buy", tif: str = "day", extended_hours: bool = False) -> Dict[str, object]:
    body = {
        "symbol": symbol.upper(),
        "notional": round(float(notional), 2),  # 美股支持 0.01 精度
        "side": side,
        "type": "market",
        "time_in_force": tif,
        "extended_hours": bool(extended_hours),
    }
    return request_alpaca("POST", "/v2/orders", json_body=body)


def place_market_qty(symbol: str, qty: int, side: str = "buy", tif: str = "day", extended_hours: bool = False) -> Dict[str, object]:
    body = {
        "symbol": symbol.upper(),
        "qty": int(qty),
        "side": side,
        "type": "market",
        "time_in_force": tif,
        "extended_hours": bool(extended_hours),
    }
    return request_alpaca("POST", "/v2/orders", json_body=body)


def run(
    weights_csv: str,
    final_ratios_csv: str,
    *,
    min_notional: float = 1.0,
    dry_run: bool = False,
    time_in_force: str = "day",
    extended_hours: bool = False,
    log_csv: Optional[str] = None,
) -> int:
    # 1) 读取最后交易日的权重
    latest = read_latest_weights(weights_csv)
    gvkeys = [g for (g, _, _) in latest]

    # 2) gvkey -> tic 映射（以 final_ratios 中该 gvkey 的最新日期为准）
    gvkey_to_tic = build_gvkey_to_tic_map(final_ratios_csv, gvkeys)

    # 3) 查询现金
    cash = fetch_cash()
    if cash <= 0:
        raise RuntimeError("账户现金为 0，无法下单")

    # 4) 归一化并生成下单计划
    norm_items = normalize_positive_weights(latest)

    planned_notional: List[Tuple[str, float, int, int]] = []  # (tic, notional, gvkey, log_idx)
    planned_qty: List[Tuple[str, int, int, int]] = []        # (tic, qty, gvkey, log_idx)
    skipped_no_tic: List[int] = []
    skipped_inactive: List[int] = []
    log_records: List[Dict[str, Any]] = []
    for gvkey, w, _ in norm_items:
        tic = gvkey_to_tic.get(gvkey)
        if not tic:
            skipped_no_tic.append(gvkey)
            log_records.append({
                "status": "skipped",
                "reason": "no_tic_mapping",
                "symbol": None,
                "gvkey": gvkey,
                "weight": w,
                "allocation_notional": 0.0,
                "order_kind": None,
                "qty": None,
                "price_used": None,
                "tif": time_in_force,
                "extended_hours": extended_hours,
                "fractionable": None,
                "tradable": None,
                "order_id": None,
                "error": None,
                "dry_run": dry_run,
            })
            continue
        notional = w * cash
        if notional < min_notional:
            log_records.append({
                "status": "skipped",
                "reason": "below_min_notional",
                "symbol": tic,
                "gvkey": gvkey,
                "weight": w,
                "allocation_notional": round(float(notional), 2),
                "order_kind": None,
                "qty": None,
                "price_used": None,
                "tif": time_in_force,
                "extended_hours": extended_hours,
                "fractionable": None,
                "tradable": None,
                "order_id": None,
                "error": None,
                "dry_run": dry_run,
            })
            continue
        # 资产预检查
        try:
            asset = get_asset(tic)
            status = str(asset.get("status", "")).lower()
            tradable = bool(asset.get("tradable", False))
            fractionable = bool(asset.get("fractionable", False))
            if status != "active" or not tradable:
                skipped_inactive.append(gvkey)
                log_records.append({
                    "status": "skipped",
                    "reason": "inactive_or_not_tradable",
                    "symbol": tic,
                    "gvkey": gvkey,
                    "weight": w,
                    "allocation_notional": round(float(notional), 2),
                    "order_kind": None,
                    "qty": None,
                    "price_used": None,
                    "tif": time_in_force,
                    "extended_hours": extended_hours,
                    "fractionable": fractionable,
                    "tradable": tradable,
                    "order_id": None,
                    "error": None,
                    "dry_run": dry_run,
                })
                continue
            if fractionable:
                log_idx = len(log_records)
                log_records.append({
                    "status": "planned",
                    "reason": None,
                    "symbol": tic,
                    "gvkey": gvkey,
                    "weight": w,
                    "allocation_notional": round(float(notional), 2),
                    "order_kind": "notional",
                    "qty": None,
                    "price_used": None,
                    "tif": time_in_force,
                    "extended_hours": extended_hours,
                    "fractionable": True,
                    "tradable": tradable,
                    "order_id": None,
                    "error": None,
                    "dry_run": dry_run,
                })
                planned_notional.append((tic, notional, gvkey, log_idx))
            else:
                price = get_latest_price(tic)
                if price is None or price <= 0:
                    skipped_inactive.append(gvkey)
                    log_records.append({
                        "status": "skipped",
                        "reason": "price_unavailable",
                        "symbol": tic,
                        "gvkey": gvkey,
                        "weight": w,
                        "allocation_notional": round(float(notional), 2),
                        "order_kind": None,
                        "qty": None,
                        "price_used": None,
                        "tif": time_in_force,
                        "extended_hours": extended_hours,
                        "fractionable": False,
                        "tradable": tradable,
                        "order_id": None,
                        "error": None,
                        "dry_run": dry_run,
                    })
                    continue
                qty = int(math.floor(notional / price))
                if qty < 1:
                    log_records.append({
                        "status": "skipped",
                        "reason": "qty_below_1",
                        "symbol": tic,
                        "gvkey": gvkey,
                        "weight": w,
                        "allocation_notional": round(float(notional), 2),
                        "order_kind": "qty",
                        "qty": qty,
                        "price_used": float(price),
                        "tif": time_in_force,
                        "extended_hours": extended_hours,
                        "fractionable": False,
                        "tradable": tradable,
                        "order_id": None,
                        "error": None,
                        "dry_run": dry_run,
                    })
                    continue
                log_idx = len(log_records)
                log_records.append({
                    "status": "planned",
                    "reason": None,
                    "symbol": tic,
                    "gvkey": gvkey,
                    "weight": w,
                    "allocation_notional": round(float(notional), 2),
                    "order_kind": "qty",
                    "qty": int(qty),
                    "price_used": float(price),
                    "tif": time_in_force,
                    "extended_hours": extended_hours,
                    "fractionable": False,
                    "tradable": tradable,
                    "order_id": None,
                    "error": None,
                    "dry_run": dry_run,
                })
                planned_qty.append((tic, qty, gvkey, log_idx))
        except Exception:
            # 预检查失败则跳过，避免下单错误
            skipped_inactive.append(gvkey)
            log_records.append({
                "status": "skipped",
                "reason": "precheck_failed",
                "symbol": tic if 'tic' in locals() else None,
                "gvkey": gvkey,
                "weight": w,
                "allocation_notional": round(float(notional), 2),
                "order_kind": None,
                "qty": None,
                "price_used": None,
                "tif": time_in_force,
                "extended_hours": extended_hours,
                "fractionable": None,
                "tradable": None,
                "order_id": None,
                "error": None,
                "dry_run": dry_run,
            })
            continue

    total_planned = len(planned_notional) + len(planned_qty)
    if total_planned == 0:
        print("无可执行下单（可能全部无映射或金额过小）")
        return 0

    # 下单（不等待结果）
    successes: List[Tuple[str, float, str]] = []  # 统一记录金额，qty单用 qty*price 近似
    failures: List[Tuple[str, float, str]] = []

    # 先下 notional（碎股）
    for tic, notional, _, log_idx in planned_notional:
        if dry_run:
            successes.append((tic, notional, "dry-run"))
            log_records[log_idx]["status"] = "dry_run"
            log_records[log_idx]["order_id"] = "dry-run"
            continue
        try:
            resp = place_market_notional(tic, notional, side="buy", tif=time_in_force, extended_hours=extended_hours)
            order_id = str(resp.get("id"))
            successes.append((tic, notional, order_id))
            log_records[log_idx]["status"] = "submitted"
            log_records[log_idx]["order_id"] = order_id
        except Exception as exc:
            failures.append((tic, notional, str(exc)))
            log_records[log_idx]["status"] = "failed"
            log_records[log_idx]["error"] = str(exc)

    # 再下整股 qty（非碎股）
    for tic, qty, _, log_idx in planned_qty:
        approx_notional = float(qty)  # 仅占位，稍后用价格近似
        if dry_run:
            successes.append((tic, approx_notional, f"dry-run-qty:{qty}"))
            log_records[log_idx]["status"] = "dry_run"
            log_records[log_idx]["order_id"] = f"dry-run-qty:{qty}"
            continue
        try:
            resp = place_market_qty(tic, qty, side="buy", tif=time_in_force, extended_hours=extended_hours)
            order_id = str(resp.get("id"))
            successes.append((tic, approx_notional, f"{order_id}|qty:{qty}"))
            log_records[log_idx]["status"] = "submitted"
            log_records[log_idx]["order_id"] = order_id
        except Exception as exc:
            failures.append((tic, approx_notional, f"{exc}|qty:{qty}"))
            log_records[log_idx]["status"] = "failed"
            log_records[log_idx]["error"] = f"{exc}|qty:{qty}"

    # 输出摘要
    print(
        f"计划下单: {total_planned} 支；成功: {len(successes)}；失败: {len(failures)}；无 tic 映射: {len(skipped_no_tic)}；不活跃/不可交易: {len(skipped_inactive)}"
    )
    if skipped_no_tic:
        print("无映射 gvkey 示例(最多10个):", skipped_no_tic[:10])
    if successes:
        print("成功订单(前10):")
        for tic, notional, oid in successes[:10]:
            print(f"  {tic}: ${round(notional, 2)} -> {oid}")
    if failures:
        print("失败订单(前10):")
        for tic, notional, err in failures[:10]:
            print(f"  {tic}: ${round(notional, 2)} -> {err}")

    # 写日志
    if log_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_csv = f"orders_{ts}.csv"
    try:
        pd.DataFrame(log_records).to_csv(log_csv, index=False)
        print(f"日志已写入: {log_csv}")
    except Exception as exc:
        print(f"写入日志失败: {exc}")

    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="按权重批量下单（Paper Trading）")
    p.add_argument("--weights", default="drl_weight.csv", help="权重CSV路径，含 trade_date/gvkey/weights")
    p.add_argument("--ratios", default="final_ratios_20250712.csv", help="final_ratios CSV 路径，含 gvkey/tic")
    p.add_argument("--min-notional", type=float, default=1.0, help="最小下单金额，低于此值跳过")
    p.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际下单")
    p.add_argument("--tif", default="day", choices=["day", "gtc", "opg", "cls", "ioc", "fok"], help="time_in_force")
    p.add_argument("--extended", action="store_true", help="允许盘前/盘后成交（若标的支持）")
    p.add_argument("--log-csv", default=None, help="日志CSV路径，不传则自动按时间戳生成")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(
            weights_csv=args.weights,
            final_ratios_csv=args.ratios,
            min_notional=args.min_notional,
            dry_run=args.dry_run,
            time_in_force=args.tif,
            extended_hours=args.extended,
            log_csv=args.log_csv,
        )
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


