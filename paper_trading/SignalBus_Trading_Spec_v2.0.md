# Signal-Bus Modular Quantitative Trading System (Alpaca Paper) — Functional Specification v2.0

---

## 0. System Overview

### 0.1 Design Objectives
- **Modular Architecture**: Strategy Production → Signal Bus → Execution Engine → Trading Execution → Reconciliation Analysis → Performance Monitoring
- **Decoupled Design**: Each module operates independently, communicating through file interfaces
- **Multi-Strategy Support**: Single selection (select) and fusion (blend) modes
- **Risk Control**: No short selling, no cash positions, weight normalization
- **Real-time Execution**: Support for Alpaca Paper/Live environments
- **Complete Audit Trail**: End-to-end tracking from signals to performance

### 0.2 System Architecture
```
Strategy Files → M1(Signal Bus) → M2(Execution Engine) → M3(Scheduler) → M4(Alpaca Execution) → M5(Reconciliation) → M6(Analytics)
```

---

## 1. Directory Structure

```text
paper_trading/
├── config/
│   ├── config.yaml                  # Main system configuration
│   └── strategy_manifest.yaml       # Strategy manifest configuration
├── data/
│   ├── raw_signals/                 # Raw strategy signals
│   │   ├── drl_weight.csv          # DRL strategy weights
│   │   ├── equally_weighted.xlsx   # Equal weight strategy
│   │   ├── mean_weighted.xlsx      # Mean strategy
│   │   └── minimum_weighted.xlsx   # Minimum variance strategy
│   ├── mapping/
│   │   └── tic_gvkey_mapping_2025.csv  # Asset mapping table
│   ├── intermediate/               # Intermediate files
│   │   └── targets_consolidated_YYYYMMDD.parquet
│   └── outputs/                    # Output files
│       ├── planned_orders.csv      # M2 planned orders
│       ├── rebalance_plan.csv      # M2 rebalancing plan
│       ├── execution_log.csv       # M4 execution log
│       ├── positions_eod_YYYYMMDD.csv  # M5 end-of-day positions
│       ├── recon_report_YYYYMMDD.csv   # M5 reconciliation report
│       ├── pnl_daily_YYYYMMDD.csv      # M6 daily P&L
│       └── risk_snapshot_YYYYMMDD.csv  # M6 risk snapshot
├── src/
│   ├── signal_bus.py               # M1 core logic
│   ├── run_signal_bus.py           # M1 execution entry
│   ├── execution_engine.py         # M2 core logic
│   ├── run_execution.py            # M2 execution entry
│   ├── scheduler.py                # M3 scheduling orchestration
│   ├── alp_execution.py            # M4 trading execution
│   ├── m5_reconcile.py             # M5 reconciliation analysis
│   └── m6_analytics.py             # M6 performance analysis (planned)
├── logs/
│   └── scheduler.log               # Scheduler logs
└── tests/                          # Test files
```

---

## 2. Module Detailed Design

### 2.1 M1 — Signal Bus

#### **Business Logic**
- **Strategy Signal Reading**: Read weight signals from multiple strategy files
- **Signal Standardization**: Unify column names, data types, formats
- **Strategy Fusion**: Support single selection (select) and weighted fusion (blend)
- **Constraint Checking**: Weight normalization, no short selling, no cash positions

#### **Implementation Objectives**
- Provide unified strategy signal interface
- Support multiple strategies coexistence and switching
- Ensure output signal executability

#### **Inputs/Outputs**
**Inputs:**
- `strategy_manifest.yaml`: Strategy configuration file
- Strategy signal files: `drl_weight.csv`, `equally_weighted.xlsx`, etc.
- Runtime parameters: `--date`, `--mode`, `--pick`, `--blend`

**Outputs:**
- `targets_consolidated_YYYYMMDD.parquet`: Standardized target weights

#### **Core Functions**
```python
# signal_bus.py
def load_manifest(path: Path) -> List[StrategySpec]
def load_all_strategies(specs: List[StrategySpec]) -> pd.DataFrame
def pick_latest_per_strategy(df: pd.DataFrame) -> pd.DataFrame
def merge_for_date(df: pd.DataFrame, date: str, mode: str, blend: dict) -> pd.DataFrame
def enforce_constraints_and_normalize(df: pd.DataFrame, allow_short: bool, allow_cash: bool) -> pd.DataFrame
def save_targets(df: pd.DataFrame, path: Path)
```

#### **Configuration Example**
```yaml
# strategy_manifest.yaml
strategies:
  - id: drl
    path: ./data/raw_signals/drl_weight.csv
    format: csv
    column_mapping:
      trade_date: trade_date
      gvkey: asset_id
      weights: target_weight
```

---

### 2.2 M2 — Execution Engine

#### **Business Logic**
- **Target Weight Parsing**: Read target weights output from M1
- **Position Comparison**: Compare with current positions, calculate differences
- **Order Generation**: Generate buy/sell orders based on differences
- **Price Acquisition**: Support real-time prices and sample prices
- **Risk Control**: Order size limits, price checks

#### **Implementation Objectives**
- Convert target weights to executable orders
- Support real-time price acquisition
- Provide flexible risk control

#### **Inputs/Outputs**
**Inputs:**
- `targets_consolidated_YYYYMMDD.parquet`: Target weights
- `tic_gvkey_mapping_2025.csv`: Asset mapping table
- `positions_eod_YYYYMMDD.csv`: Current positions (optional)
- Price data: Real-time API or sample files

**Outputs:**
- `planned_orders.csv`: Planned order list
- `rebalance_plan.csv`: Rebalancing plan details

#### **Core Functions**
```python
# execution_engine.py
def load_targets(path: Path) -> pd.DataFrame
def load_mapping(path: Path, id_col: str, sym_col: str) -> pd.DataFrame
def map_targets_to_symbols(targets: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame
def plan_orders(targets_df: pd.DataFrame, prices: Dict[str, float], positions: List[Position], equity: float) -> Tuple[List[Order], pd.DataFrame]
def save_orders_csv(orders: List[Order], path: Path)
def save_plan_csv(plan_df: pd.DataFrame, path: Path)
```

#### **Price Acquisition Functions**
```python
# run_execution.py
def find_latest_positions_file(outputs_dir: Path, current_timestamp: str = None) -> Path
def fetch_prices_for_targets(cfg: dict, targets_df: pd.DataFrame, mapping_df: pd.DataFrame, price_source: str, debug: bool) -> Dict[str, float]
```

---

### 2.3 M3 — Scheduler

#### **Business Logic**
- **Module Orchestration**: Execute M1→M2→M4→M5→M6 in sequence
- **Time Scheduling**: Support daily, weekly, monthly scheduling
- **Dependency Checking**: Ensure successful execution of prerequisite modules
- **Error Handling**: Degradation strategies when modules fail

#### **Implementation Objectives**
- Provide unified execution entry point
- Support scheduled and manual execution
- Ensure correct dependencies between modules

#### **Inputs/Outputs**
**Inputs:**
- `config.yaml`: System configuration
- Command line arguments: `--date`, `--once`, `--daemon`

**Outputs:**
- Execution results from each module
- Scheduler logs: `logs/scheduler.log`

#### **Core Functions**
```python
# scheduler.py
def build_m1_cmd(project_root: Path, cfg: dict, run_date: str) -> list
def build_m2_cmd(project_root: Path, cfg: dict, targets_path: str, run_date: str) -> list
def build_m4_cmd(project_root: Path, cfg: dict) -> list
def build_m5_cmd(project_root: Path, cfg: dict, run_date: str) -> list
def run_once(project_root: Path, cfg: dict, run_date: str, logger: logging.Logger)
def next_run_after(now: datetime, run_time_hhmm: str) -> datetime
```

#### **Configuration Example**
```yaml
# config.yaml
scheduler:
  time: "09:00"
  m1:
    mode: "select"
    pick: "drl"
  m2:
    equity: 100000
    fetch_real_prices: true
    price_source: "snapshot"
  m4:
    mode: "dry"
    run: true
  m5:
    source: "logs"
    run: true
```

---

### 2.4 M4 — Alpaca Execution

#### **Business Logic**
- **Order Parsing**: Read planned orders generated by M2
- **Risk Control**: Order size, price deviation, cash buffer
- **API Interaction**: Communicate with Alpaca Trading API
- **Execution Monitoring**: Order status tracking, fill reports
- **Idempotency Control**: Prevent duplicate submissions

#### **Implementation Objectives**
- Safely and reliably execute trading orders
- Provide complete execution audit trail
- Support dry-run and live modes

#### **Inputs/Outputs**
**Inputs:**
- `planned_orders.csv`: Planned orders
- `rebalance_plan.csv`: Rebalancing plan (optional)
- Alpaca API configuration

**Outputs:**
- `execution_log.csv`: Execution log
- `filled_orders.csv`: Fill reports (optional)

#### **Core Functions**
```python
# alp_execution.py
def read_planned_orders(path: Path) -> pd.DataFrame
def derive_orders_from_plan(plan_path: Path) -> pd.DataFrame
def risk_check_orders(orders: List[ExecOrder], prices: Dict[str, float], acct: dict, risk: dict) -> Tuple[List[ExecOrder], List[str]]
def make_client_order_id(prefix: str, trade_date: str, o: ExecOrder) -> str
def append_execution_log(path: Path, rows: List[dict])

class AlpacaClient:
    def get_account(self) -> Optional[dict]
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]
    def get_snapshot_prices(self, symbols: List[str]) -> Dict[str, float]
    def submit_order(self, o: ExecOrder, tif: str, client_order_id: str) -> Tuple[str, str, str]
    def poll_fills_once(self) -> List[dict]
```

#### **Price Acquisition Functions**
```python
def fetch_prices_for_targets(cfg: dict, targets_df: pd.DataFrame, mapping_df: pd.DataFrame, price_source: str, debug: bool) -> Dict[str, float]
def save_prices_to_csv(prices: Dict[str, float], output_path: str)
```

---

### 2.5 M5 — Reconciliation

#### **Business Logic**
- **Dual Checking**:
  - Single execution reconciliation: `planned_qty` vs `actual_qty` (check M4 execution)
  - Final position reconciliation: `target_qty` vs `final_actual_qty` (check final results)
- **Position Calculation**: Calculate actual positions based on execution logs
- **Weight Analysis**: Calculate deviations between current weights and target weights
- **Report Generation**: Generate detailed reconciliation reports

#### **Implementation Objectives**
- Verify execution accuracy
- Provide position transparency
- Support weight deviation analysis

#### **Inputs/Outputs**
**Inputs:**
- `planned_orders.csv`: Planned orders
- `execution_log.csv`: Execution log
- `positions_bod.csv`: Beginning-of-day positions (optional)
- `rebalance_plan.csv`: Rebalancing plan

**Outputs:**
- `positions_eod_YYYYMMDD.csv`: End-of-day positions
- `recon_report_YYYYMMDD.csv`: Reconciliation report

#### **Core Functions**
```python
# m5_reconcile.py
def reconcile_from_logs(planned_path: Path, exec_log_path: Path, bod_positions_path: Path, rebalance_plan_path: Path, run_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]
def build_eod_positions(bod_positions: pd.DataFrame, execution_log: pd.DataFrame) -> pd.DataFrame
def calculate_weight_deviations(target_weights: pd.DataFrame, actual_positions: pd.DataFrame, prices: Dict[str, float], total_equity: float) -> pd.DataFrame
def save_positions_eod(positions: pd.DataFrame, run_date: str, out_dir: Path)
def save_recon_report(recon_df: pd.DataFrame, run_date: str, out_dir: Path)
```

#### **Weight Percentage Calculation**
```python
def calculate_weight_percentages(positions: pd.DataFrame, prices: Dict[str, float], total_equity: float) -> pd.DataFrame:
    """
    Calculate position weight percentages
    """
    weight_data = []
    for _, row in positions.iterrows():
        symbol = row['symbol']
        qty = row['actual_qty']
        price = prices.get(symbol, 0.0)
        market_value = qty * price
        weight_pct = (market_value / total_equity) * 100 if total_equity > 0 else 0.0
        
        weight_data.append({
            'symbol': symbol,
            'actual_qty': qty,
            'price': price,
            'market_value': market_value,
            'weight_percentage': weight_pct
        })
    
    return pd.DataFrame(weight_data)
```

---

### 2.6 M6 — Analytics & Monitoring

#### **Business Logic**
- **Return Calculation**: Daily returns, cumulative returns, excess returns
- **Risk Analysis**: Volatility, maximum drawdown, Sharpe ratio
- **Position Analysis**: Weight distribution, turnover, concentration
- **Performance Attribution**: Strategy contribution, sector contribution, stock contribution
- **Monitoring Alerts**: Abnormal trading, risk indicator thresholds

#### **Implementation Objectives**
- Provide comprehensive performance analysis
- Support risk monitoring and alerts
- Generate visualization reports

#### **Inputs/Outputs**
**Inputs:**
- `positions_eod_YYYYMMDD.csv`: Historical position data
- `execution_log.csv`: Execution log
- `recon_report_YYYYMMDD.csv`: Reconciliation report
- Benchmark data: Market indices, risk-free rate

**Outputs:**
- `pnl_daily_YYYYMMDD.csv`: Daily P&L data
- `risk_snapshot_YYYYMMDD.csv`: Risk snapshot
- `summary_dashboard.html`: Summary dashboard
- Alert notifications: Email, SMS, API

#### **Core Functions** (Planned Implementation)
```python
# m6_analytics.py
def calculate_daily_pnl(positions_history: pd.DataFrame, prices_history: pd.DataFrame) -> pd.DataFrame
def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]
def calculate_drawdown(equity_curve: pd.Series) -> pd.DataFrame
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float
def calculate_turnover(positions_history: pd.DataFrame) -> pd.DataFrame
def generate_performance_report(start_date: str, end_date: str, config: dict) -> Dict[str, pd.DataFrame]
def check_risk_alerts(risk_metrics: Dict[str, float], thresholds: Dict[str, float]) -> List[str]
def create_dashboard_html(performance_data: Dict[str, pd.DataFrame], output_path: str)
```

#### **Risk Metrics Calculation**
```python
def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate risk metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = returns.mean() * 252
    metrics['volatility'] = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # Other metrics
    metrics['var_95'] = returns.quantile(0.05)
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    
    return metrics
```

#### **Configuration Example**
```yaml
# config.yaml
analytics:
  benchmark: "SPY"  # Benchmark index
  risk_free_rate: 0.02  # Risk-free rate
  risk_thresholds:
    max_drawdown: -0.20
    volatility: 0.25
    var_95: -0.03
  alerts:
    email: "trader@example.com"
    slack_webhook: "https://hooks.slack.com/..."
```

---

## 3. Data Flow and File Interfaces

### 3.1 Data Flow Diagram
```
Strategy Files → M1 → targets_consolidated.parquet
                       ↓
Mapping Files → M2 → planned_orders.csv + rebalance_plan.csv
                       ↓
Alpaca API → M4 → execution_log.csv
                   ↓
Execution Log → M5 → positions_eod.csv + recon_report.csv
                       ↓
Historical Data → M6 → pnl_daily.csv + risk_snapshot.csv + dashboard.html
```

### 3.2 File Format Specifications

#### **Target Weight File** (M1 Output)
```csv
trade_date,asset_id,target_weight,asof,strategy_id
2025-07-28,1690,0.0047,2025-07-28T09:00:00Z,drl
2025-07-28,12141,0.0047,2025-07-28T09:00:00Z,drl
```

#### **Planned Orders File** (M2 Output)
```csv
asset_id,symbol,side,qty,order_type,tp_price,sl_price
1690,AAPL,BUY,5.17,BRACKET,150.50,145.20
12141,MSFT,BUY,2.34,BRACKET,420.30,405.60
```

#### **Execution Log File** (M4 Output)
```csv
ts,mode,symbol,side,qty,order_type,tp_price,sl_price,time_in_force,client_order_id,price_used,notional,status,order_id,error
2025-07-28T09:00:00Z,DRY,AAPL,BUY,5.17,BRACKET,150.50,145.20,DAY,ORD_20250728_AAPL_BUY_abc123,148.30,766.71,SIMULATED,,
```

#### **Reconciliation Report File** (M5 Output)
```csv
symbol,planned_qty,actual_qty,delta_qty,status,target_weight,actual_weight,weight_deviation
AAPL,5.17,5.17,0.00,OK,0.0047,0.0047,0.0000
MSFT,2.34,2.34,0.00,OK,0.0047,0.0047,0.0000
```

---

## 4. Configuration Management

### 4.1 Main Configuration File (config.yaml)
```yaml
# Alpaca API Configuration
alpaca:
  key_id: "${APCA_API_KEY_ID}"
  secret: "${APCA_API_SECRET_KEY}"
  trading_endpoint: "https://paper-api.alpaca.markets"
  data_endpoint: "https://data.alpaca.markets"

# Scheduler Configuration
scheduler:
  time: "09:00"
  m1:
    mode: "select"
    pick: "drl"
  m2:
    equity: 100000
    fetch_real_prices: true
    price_source: "snapshot"
  m4:
    mode: "dry"
    run: true
  m5:
    source: "logs"
    run: true

# Execution Configuration
execution:
  out_orders: "./data/outputs/planned_orders.csv"
  out_plan: "./data/outputs/rebalance_plan.csv"
  time_in_force: "DAY"

# Risk Control
risk:
  max_share_per_order: 1000
  max_notional_per_order: 50000
  max_total_notional: 1000000
  price_tolerance_pct: 1.0
  min_cash_buffer_pct: 0.05
  reject_if_price_missing: false

# Idempotency Control
idempotency:
  enable_client_order_id: true
  client_order_id_prefix: "ORD"

# Analytics Configuration
analytics:
  benchmark: "SPY"
  risk_free_rate: 0.02
  risk_thresholds:
    max_drawdown: -0.20
    volatility: 0.25
```

### 4.2 Strategy Manifest Configuration (strategy_manifest.yaml)
```yaml
strategies:
  - id: drl
    path: "./data/raw_signals/drl_weight.csv"
    format: csv
    column_mapping:
      trade_date: trade_date
      gvkey: asset_id
      weights: target_weight
  - id: equal
    path: "./data/raw_signals/equally_weighted.xlsx"
    format: xlsx
    column_mapping:
      trade_date: trade_date
      gvkey: asset_id
      weights: target_weight
```

---

## 5. Typical Workflows

### 5.1 Daily Rebalancing Workflow
```bash
# 1. Manual single execution
python src/scheduler.py --config config/config.yaml --date 2025-07-28 --once --debug

# 2. Check module outputs
ls -la data/outputs/
# targets_consolidated_20250728.parquet
# planned_orders.csv
# rebalance_plan.csv
# execution_log.csv
# positions_eod_20250728.csv
# recon_report_20250728.csv
```

### 5.2 Scheduled Workflow
```bash
# Start scheduled scheduler
python src/scheduler.py --config config/config.yaml --daemon --skip-weekends
```

### 5.3 Independent Module Execution
```bash
# M1: Signal processing
python src/run_signal_bus.py --manifest config/strategy_manifest.yaml --date 2025-07-28 --mode select --pick drl --out data/intermediate/targets_consolidated_20250728.parquet

# M2: Execution planning
python src/run_execution.py --config config/config.yaml --targets data/intermediate/targets_consolidated_20250728.parquet --equity 100000 --out-orders data/outputs/planned_orders.csv --out-plan data/outputs/rebalance_plan.csv --fetch-real-prices --price-source snapshot

# M4: Trading execution
python src/alp_execution.py --config config/config.yaml --mode dry --debug

# M5: Reconciliation analysis
python src/m5_reconcile.py --config config/config.yaml --date 2025-07-28 --source logs --out-dir data/outputs
```

---

## 6. Testing and Validation

### 6.1 Unit Testing
```bash
# Test M1 signal processing
python -m pytest tests/test_signal_bus.py

# Test M2 execution engine
python -m pytest tests/test_execution_engine.py

# Test M4 trading execution
python tests/run_m4_tests.py --config config/test_config.yaml
```

### 6.2 Integration Testing
```bash
# End-to-end testing
python tests/test_m4_m5_m6.py --config config/test_config.yaml
```

### 6.3 Performance Testing
```bash
# Stress testing
python tests/performance_test.py --symbols 1000 --iterations 100
```

---

## 7. Monitoring and Alerting

### 7.1 System Monitoring
- **Module Execution Status**: Success/failure/timeout
- **API Call Statistics**: Success rate, response time, error rate
- **File Integrity**: File size, format, content validation

### 7.2 Business Monitoring
- **Trading Execution Quality**: Fill rate, slippage, latency
- **Risk Indicators**: Position concentration, turnover, drawdown
- **Performance Indicators**: Return rate, Sharpe ratio, information ratio

### 7.3 Alert Mechanisms
- **Real-time Alerts**: Trading failures, risk threshold breaches, system anomalies
- **Periodic Reports**: Daily, weekly, monthly reports
- **Notification Methods**: Email, SMS, Slack, API

---

## 8. Deployment and Maintenance

### 8.1 Environment Requirements
- Python 3.10+
- Dependencies: pandas, numpy, requests, pyyaml, alpaca-py
- Operating System: Linux/Windows/macOS
- Network: Stable internet connection

### 8.2 Deployment Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment variables
export APCA_API_KEY_ID="your_key_id"
export APCA_API_SECRET_KEY="your_secret_key"

# 3. Create necessary directories
mkdir -p data/{raw_signals,intermediate,outputs} logs

# 4. Configure strategy files
# Place strategy files in data/raw_signals/

# 5. Test configuration
python src/scheduler.py --config config/config.yaml --date 2025-07-28 --once --debug
```

### 8.3 Maintenance Tasks
- **Log Cleanup**: Regularly clean old log files
- **Data Backup**: Backup important configuration and output files
- **Performance Optimization**: Monitor system performance, optimize bottlenecks
- **Security Updates**: Regularly update dependencies and security patches

---

## 9. Troubleshooting

### 9.1 Common Issues
1. **M1 returns rows=0**: Check if strategy files contain data for the specified date
2. **M2 plans 0 orders**: Check if price data is complete
3. **M4 execution fails**: Check Alpaca API configuration and network connection
4. **M5 shows MISMATCH**: Check execution logs and position files

### 9.2 Debugging Tips
```bash
# Enable detailed logging
python src/scheduler.py --config config/config.yaml --date 2025-07-28 --once --debug

# Check file contents
head -10 data/outputs/planned_orders.csv
tail -10 data/outputs/execution_log.csv

# Verify API connection
python test_alpaca_api.py
```

---

## 10. Future Extensions

### 10.1 Functional Extensions
- **Multi-Asset Support**: Futures, options, cryptocurrencies
- **Advanced Risk Management**: VaR, stress testing, scenario analysis
- **Machine Learning**: Automatic parameter optimization, anomaly detection
- **Real-time Monitoring**: Web interface, real-time dashboard

### 10.2 Architectural Extensions
- **Microservices**: Independent deployment of modules
- **Containerization**: Docker container deployment
- **Cloud-Native**: Kubernetes orchestration
- **Event-Driven**: Message queues, event bus

---

**Version**: v2.0  
**Date**: 2025-01-XX  
**Status**: M1-M5 implemented, M6 planned  
**Maintainer**: Simon Liao
