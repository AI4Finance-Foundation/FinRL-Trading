#!/usr/bin/env python3
"""
测试 Alpaca API 连接和价格获取功能
"""

import requests
import time
import os
from typing import Dict, List

def test_alpaca_api():
    """Test Alpaca API connection"""
    
    # Get API keys from environment variables or config file
    api_key = os.getenv("APCA_API_KEY_ID", "PKPWCZPOOYKF0YF2G68Z")
    api_secret = os.getenv("APCA_API_SECRET_KEY", "aRW9f6XvlYckqHtib0vOgtkxfYDsQ5qkQUIDDbF2")
    data_endpoint = "https://data.alpaca.markets/v2"
    
    print(f"Testing Alpaca API with key: {api_key[:10]}...")
    print(f"Data endpoint: {data_endpoint}")
    
    # Create session
    session = requests.Session()
    session.headers.update({
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Content-Type": "application/json"
    })
    
    # Test 1: Get account information
    print("\n=== Test 1: Account Info ===")
    try:
        trading_endpoint = "https://paper-api.alpaca.markets/v2"
        response = session.get(f"{trading_endpoint}/account", timeout=10)
        if response.status_code == 200:
            account = response.json()
            print(f"✅ Account test passed")
            print(f"   Account ID: {account.get('id', 'N/A')}")
            print(f"   Status: {account.get('status', 'N/A')}")
            print(f"   Currency: {account.get('currency', 'N/A')}")
        else:
            print(f"❌ Account test failed: {response.status_code} {response.text}")
    except Exception as e:
        print(f"❌ Account test error: {e}")
    
    # Test 2: Get latest prices
    print("\n=== Test 2: Latest Prices ===")
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    try:
        # Fix: Use correct API parameter format
        # According to Alpaca API documentation, should use 'symbols' as query parameter
        params = {
            'symbols': ','.join(test_symbols)
        }
        #response = session.get(f"{data_endpoint}/stocks/latest/trades", params=params, timeout=10)
        response = session.get(f"{data_endpoint}/stocks/trades/latest", params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            trades = data.get('trades', {})
            print(f"✅ Latest prices test passed")
            for symbol in test_symbols:
                if symbol in trades:
                    trade = trades[symbol]
                    price = trade.get('p', 'N/A')
                    timestamp = trade.get('t', 'N/A')
                    print(f"   {symbol}: ${price} at {timestamp}")
                else:
                    print(f"   {symbol}: No data")
        else:
            print(f"❌ Latest prices test failed: {response.status_code} {response.text}")
            # Try alternative parameter format
            print("Trying alternative parameter format...")
            params = {
                'symbol': ','.join(test_symbols)
            }
            response2 = session.get(f"{data_endpoint}/stocks/latest/trades", params=params, timeout=10)
            print(f"Alternative format result: {response2.status_code} {response2.text}")
    except Exception as e:
        print(f"❌ Latest prices test error: {e}")
    
    # Test 3: Get snapshot prices
    print("\n=== Test 3: Snapshot Prices ===")
    try:
        # Fix: Use correct API parameter format
        params = {
            'symbols': ','.join(test_symbols)
        }
        response = session.get(f"{data_endpoint}/stocks/snapshots", params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Snapshot prices test passed")
            for symbol in test_symbols:
                if symbol in data:
                    snapshot = data[symbol]
                    latest_trade = snapshot.get('latestTrade', {})
                    price = latest_trade.get('p', 'N/A')
                    timestamp = latest_trade.get('t', 'N/A')
                    print(f"   {symbol}: ${price} at {timestamp}")
                else:
                    print(f"   {symbol}: No data")
        else:
            print(f"❌ Snapshot prices test failed: {response.status_code} {response.text}")
            # Try alternative parameter format
            print("Trying alternative parameter format...")
            params = {
                'symbol': ','.join(test_symbols)
            }
            response2 = session.get(f"{data_endpoint}/stocks/snapshots", params=params, timeout=10)
            print(f"Alternative format result: {response2.status_code} {response2.text}")
    except Exception as e:
        print(f"❌ Snapshot prices test error: {e}")
    
    # Test 4: Batch price fetch (simulate actual usage scenario)
    print("\n=== Test 4: Batch Price Fetch ===")
    batch_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX", "AMD", "INTC"]
    try:
        all_prices = {}
        batch_size = 5
        
        for i in range(0, len(batch_symbols), batch_size):
            batch = batch_symbols[i:i + batch_size]
            params = {
                'symbols': ','.join(batch)
            }
            
            # 使用与 Test2 相同的正确端点
            response = session.get(f"{data_endpoint}/stocks/trades/latest", params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                trades = data.get('trades', {})
                for symbol in batch:
                    if symbol in trades:
                        trade = trades[symbol]
                        price = trade.get('p', 'N/A')
                        if price != 'N/A':
                            all_prices[symbol] = float(price)
            
            time.sleep(0.1)  # 避免 API 限制
        
        print(f"✅ Batch fetch test passed")
        print(f"   Successfully fetched {len(all_prices)} prices")
        for symbol, price in all_prices.items():
            print(f"   {symbol}: ${price}")
    except Exception as e:
        print(f"❌ Batch fetch test error: {e}")

if __name__ == "__main__":
    test_alpaca_api()
