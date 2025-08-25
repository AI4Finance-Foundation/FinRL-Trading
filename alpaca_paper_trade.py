import os
import sys
import time
import argparse
import json
from typing import Any, Dict, Optional
from dotenv import load_dotenv

import requests

load_dotenv()


def get_base_url() -> str:
    base_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    return base_url.rstrip("/")


def get_auth_headers() -> Dict[str, str]:
    api_key_id = os.environ.get("APCA_API_KEY_ID")
    api_secret_key = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key_id or not api_secret_key:
        raise RuntimeError(
            "未找到环境变量 APCA_API_KEY_ID 或 APCA_API_SECRET_KEY，请先设置 Alpaca API 密钥"
        )
    return {
        "APCA-API-KEY-ID": api_key_id,
        "APCA-API-SECRET-KEY": api_secret_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def request_alpaca(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    url = f"{get_base_url()}{path}"
    headers = get_auth_headers()
    response = requests.request(method, url, headers=headers, json=json_body, params=params, timeout=timeout)
    if response.status_code >= 400:
        try:
            error_info = response.json()
        except Exception:
            error_info = {"message": response.text}
        raise RuntimeError(f"Alpaca API 错误: {response.status_code} {error_info}")
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(f"解析响应失败: {exc}; 原始响应: {response.text}") from exc


def place_order(
    *,
    symbol: str,
    quantity: str,
    side: str,
    order_type: str,
    time_in_force: str,
    limit_price: Optional[str],
    stop_price: Optional[str],
    extended_hours: bool,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "qty": quantity,
        "side": side.lower(),
        "type": order_type.lower(),
        "time_in_force": time_in_force.lower(),
        "extended_hours": extended_hours,
    }
    if limit_price is not None:
        body["limit_price"] = limit_price
    if stop_price is not None:
        body["stop_price"] = stop_price
    return request_alpaca("POST", "/v2/orders", json_body=body)


def get_order(order_id: str) -> Dict[str, Any]:
    return request_alpaca("GET", f"/v2/orders/{order_id}")


def cancel_order(order_id: str) -> Dict[str, Any]:
    # Alpaca 对取消订单返回 204 无内容；这里转换为统一结构
    url = f"{get_base_url()}/v2/orders/{order_id}"
    headers = get_auth_headers()
    response = requests.request("DELETE", url, headers=headers, timeout=30)
    if response.status_code != 204:
        try:
            error_info = response.json()
        except Exception:
            error_info = {"message": response.text}
        raise RuntimeError(f"取消订单失败: {response.status_code} {error_info}")
    return {"status": "cancellation_requested", "order_id": order_id}


def cancel_all_orders() -> Dict[str, Any]:
    url = f"{get_base_url()}/v2/orders"
    headers = get_auth_headers()
    response = requests.request("DELETE", url, headers=headers, timeout=30)
    if response.status_code != 207:
        try:
            error_info = response.json()
        except Exception:
            error_info = {"message": response.text}
        raise RuntimeError(f"取消全部订单失败: {response.status_code} {error_info}")
    return {"status": "all_orders_cancellation_requested"}


def get_account() -> Dict[str, Any]:
    return request_alpaca("GET", "/v2/account")


def get_positions(symbol: Optional[str] = None) -> Any:
    if symbol:
        return request_alpaca("GET", f"/v2/positions/{symbol.upper()}")
    return request_alpaca("GET", "/v2/positions")


def wait_for_final_state(order_id: str, *, timeout_seconds: int = 180, poll_interval_seconds: float = 2.0) -> Dict[str, Any]:
    terminal_statuses = {"filled", "canceled", "rejected", "expired"}
    start_ts = time.time()
    last_status = None
    while True:
        order = get_order(order_id)
        status = str(order.get("status", "")).lower()
        if status != last_status:
            print(f"订单状态: {status}")
            last_status = status
        if status in terminal_statuses:
            return order
        if time.time() - start_ts > timeout_seconds:
            raise TimeoutError(f"等待订单最终状态超时，order_id={order_id}")
        time.sleep(poll_interval_seconds)


def pretty_print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 Alpaca API 进行 Paper Trading 的实用脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 下单
    order_parser = subparsers.add_parser("order", help="创建订单")
    order_parser.add_argument("symbol", type=str, help="股票代码，例如 AAPL")
    order_parser.add_argument("qty", type=str, help="下单数量（整数或支持的字符串）")
    order_parser.add_argument("--side", choices=["buy", "sell"], default="buy", help="买入/卖出")
    order_parser.add_argument(
        "--type",
        dest="order_type",
        choices=["market", "limit", "stop", "stop_limit", "trailing_stop"],
        default="market",
        help="订单类型",
    )
    order_parser.add_argument("--tif", dest="time_in_force", choices=["day", "gtc", "opg", "cls", "ioc", "fok"], default="day", help="有效期")
    order_parser.add_argument("--limit", dest="limit_price", type=str, default=None, help="限价")
    order_parser.add_argument("--stop", dest="stop_price", type=str, default=None, help="止损价/触发价")
    order_parser.add_argument("--extended", dest="extended_hours", action="store_true", help="允许盘前/盘后交易")
    order_parser.add_argument("--wait", dest="wait", action="store_true", help="下单后等待至订单到达最终状态")
    order_parser.add_argument("--wait-timeout", dest="wait_timeout", type=int, default=180, help="等待最终状态的超时时间(秒)")

    # 查询订单
    status_parser = subparsers.add_parser("status", help="查询订单状态")
    status_parser.add_argument("order_id", type=str, help="订单ID")

    # 取消订单
    cancel_parser = subparsers.add_parser("cancel", help="取消订单")
    cancel_parser.add_argument("order_id", type=str, help="订单ID")

    # 取消全部订单
    subparsers.add_parser("cancel_all", help="取消全部未完成订单")

    # 持仓
    positions_parser = subparsers.add_parser("positions", help="查看持仓")
    positions_parser.add_argument("--symbol", type=str, default=None, help="可选：指定股票代码")

    # 账户
    subparsers.add_parser("account", help="查看账户信息")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "order":
            order = place_order(
                symbol=args.symbol,
                quantity=args.qty,
                side=args.side,
                order_type=args.order_type,
                time_in_force=args.time_in_force,
                limit_price=args.limit_price,
                stop_price=args.stop_price,
                extended_hours=args.extended_hours,
            )
            print("下单成功：")
            pretty_print(order)
            if args.wait:
                print("等待订单达到最终状态...")
                final_order = wait_for_final_state(order_id=order["id"], timeout_seconds=args.wait_timeout)
                print("最终状态：")
                pretty_print(final_order)
            return 0

        if args.command == "status":
            result = get_order(args.order_id)
            pretty_print(result)
            return 0

        if args.command == "cancel":
            result = cancel_order(args.order_id)
            pretty_print(result)
            return 0

        if args.command == "cancel_all":
            result = cancel_all_orders()
            pretty_print(result)
            return 0

        if args.command == "positions":
            result = get_positions(args.symbol)
            pretty_print(result)
            return 0

        if args.command == "account":
            result = get_account()
            pretty_print(result)
            return 0

        parser.print_help()
        return 2
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


