# Automated Paper Trading System - Functional Specification (v1.1)

## 1. Overview

This system separates **strategy generation**, **signal exchange**, and
**execution** for Alpaca Paper Trading.\
It ensures modularity, support for multiple strategies, and easy
backtesting & live trading integration.

------------------------------------------------------------------------

## 2. System Modules

### M1: Signal Bus

**Purpose:**\
Intermediate layer using CSV/Excel files as the single source of truth
between strategy output and execution engine.

**Responsibilities:** - Accept signals from multiple strategy models
(e.g., DRL, equally weighted, mean-variance).\
- Append new strategy files or update existing files based on
timestamps.\
- Validate signals:\
- sum(weights) == 1.0 (no cash positions)\
- weights \>= 0 (no short selling)

**Input:**\
- Strategy weight files (CSV/XLSX) with schema: \[timestamp, ticker,
weight\].

**Output:**\
- Unified signal file sorted by timestamp.

**Test Case Example:**\
- Input: three weight files with overlapping timestamps.\
- Expected: merged single file, strictly normalized weights, sorted
timestamps.

------------------------------------------------------------------------

### M2: Execution Engine

**Purpose:**\
Reads validated signals from M1 and performs real-time trading on Alpaca
Paper Trading.

**Responsibilities:** - For each timestamp batch:\
1. Fetch market prices from Alpaca snapshots (single source).\
2. Cancel open orders if needed.\
3. Submit bracket orders for each asset with stop-loss & take-profit.\
- Default: no client_order_id idempotency; retry logic optional.

**Input:**\
- Unified signal file from M1.\
- Alpaca API credentials.

**Output:**\
- Order submission receipts, stored locally for audit.

**Test Case Example:**\
- Input: one batch of signals, market open.\
- Expected: all bracket orders submitted, positions match target
weights.

------------------------------------------------------------------------

### M3: Scheduler / Trigger

**Purpose:**\
Controls **when** the execution runs.

**Modes:** 1. **Automatic (Service Mode)**:\
- System runs continuously, triggers on schedule (e.g., daily 15:50).\
- Needs daemon process.

2.  **External Trigger Mode (Recommended)**:
    -   Triggered by CI/CD or cron job at fixed times.\
    -   One-shot execution → exit.

**Test Case Example:**\
- Simulate cron trigger at 15:50 → only one batch executed, then exit.

------------------------------------------------------------------------

## 3. Data Flow

    [Strategy Models] → (CSV/XLSX) → [M1: Signal Bus] → [M2: Execution Engine] → [Alpaca Paper Trading]
                                                               ↑
                                                         [M3: Scheduler]

------------------------------------------------------------------------

## 4. Constraints (Addendum v1.1)

1.  **No Cash Positions**:
    -   sum(weights) must be exactly 1.0.
2.  **No Short Selling**:
    -   weights \>= 0 only.
3.  **Price Source**:
    -   Always use Alpaca snapshots.
4.  **Order Type**:
    -   New positions via Bracket orders.
5.  **Idempotency**:
    -   client_order_id support optional, default disabled.
6.  **Scheduling**:
    -   External trigger recommended.

------------------------------------------------------------------------

## 5. Class & Function Outline

### M1: SignalBus

-   `load_signals(files: List[str]) -> pd.DataFrame`
-   `validate_signals(df: pd.DataFrame) -> pd.DataFrame`
-   `merge_signals(dfs: List[pd.DataFrame]) -> pd.DataFrame`
-   `save_unified(df: pd.DataFrame, path: str)`

### M2: ExecutionEngine

-   `load_unified(path: str) -> pd.DataFrame`
-   `fetch_prices(tickers: List[str]) -> Dict`
-   `submit_orders(signals: pd.DataFrame)`
-   `record_receipts(orders: List)`

### M3: Scheduler

-   `run_once()`: one-shot mode\
-   `run_service()`: loop mode with sleep + trigger

------------------------------------------------------------------------

## 6. Example Test Matrix

  ------------------------------------------------------------------------
  Module    Test   Scenario                  Expected Result
            ID                               
  --------- ------ ------------------------- -----------------------------
  M1        T1     Overlapping timestamps    Merged, sorted, normalized

  M1        T2     Negative weights          Validation error

  M2        T3     Market closed             Orders rejected, log warning

  M2        T4     Bracket order submission  All orders acknowledged

  M3        T5     Cron trigger at 15:50     One-shot run, exit after
                                             execution

  M3        T6     Service mode 2 days       Runs daily at fixed time
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 7. Future Extensions

-   Multiple price sources fallback.\
-   Strategy PnL attribution per batch.\
-   Parallel order submission with async API calls.

------------------------------------------------------------------------
