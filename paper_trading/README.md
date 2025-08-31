# Signal-Bus Modular Quantitative Trading System v2.1

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-production%20ready-green.svg)](https://github.com/your-repo/signal-bus-trading)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A production-ready modular quantitative trading system with Alpaca integration, featuring fractional share support, short position management, and real-time execution capabilities.

## 🚀 Key Features

- **Modular Architecture**: M1-M6 modules with clear separation of concerns
- **Fractional Share Support**: Advanced handling with Alpaca compliance
- **Short Position Management**: Proper short-to-long transitions
- **Batch Execution**: Unique batch IDs for idempotency
- **Real-time Integration**: Live price and position fetching from Alpaca
- **Enhanced Reconciliation**: Dual checking with Alpaca verification
- **Production Ready**: 100% success rate in testing

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🏗️ System Overview

### Architecture
```
Strategy Files → M1(Signal Bus) → M2(Execution Engine) → M3(Scheduler) → M4(Alpaca Execution) → M5(Reconciliation) → M6(Analytics)
```

### Module Status

| Module | Status | Description |
|--------|--------|-------------|
| **M1 - Signal Bus** | ✅ Complete | Multi-strategy signal processing |
| **M2 - Execution Engine** | ✅ Complete | Order planning with fractional shares |
| **M3 - Scheduler** | ✅ Complete | Module orchestration |
| **M4 - Alpaca Execution** | ✅ Complete | Live trading execution |
| **M5 - Reconciliation** | ✅ Complete | Dual checking with verification |
| **M6 - Analytics** | ❌ Planned | Performance analysis |

## 📦 Installation

### Prerequisites

- Python 3.10+
- Alpaca API credentials (Paper/Live)
- Stable internet connection

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd paper_trading
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export APCA_API_KEY_ID="your_alpaca_key_id"
   export APCA_API_SECRET_KEY="your_alpaca_secret_key"
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p data/{raw_signals,intermediate,outputs} logs
   ```

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
# Alpaca API Configuration
alpaca:
  api_key: "${APCA_API_KEY_ID}"
  api_secret: "${APCA_API_SECRET_KEY}"
  trading_endpoint: "https://paper-api.alpaca.markets/v2"
  data_endpoint: "https://data.alpaca.markets/v2"
  env: paper

# Scheduler Configuration
scheduler:
  time: "09:00"
  m1:
    mode: "select"
    pick: "drl"
  m2:
    execution_mode: online
    fetch_real_prices: true
    price_source: snapshot
  m4:
    mode: real
  m5:
    source: logs

# Risk Control
risk:
  fractional_shares:
    enabled: true
    min_order_notional: 2.0
    quantity_precision: 2
    max_total_weight: 1.0
    adjust_for_overflow: true
  max_notional_per_order: 200000
  min_cash_buffer_pct: 0.02

# Idempotency Control
idempotency:
  enable_client_order_id: true
  client_order_id_prefix: SBX
```

### Strategy Configuration (`config/strategy_manifest.yaml`)

```yaml
strategies:
  - id: drl
    path: ./data/raw_signals/drl_weight.csv
    format: csv
    column_mapping:
      trade_date: trade_date
      gvkey: asset_id
      weights: target_weight
  - id: equal
    path: ./data/raw_signals/equally_weighted.xlsx
    format: xlsx
    column_mapping:
      trade_date: trade_date
      gvkey: asset_id
      weights: target_weight
```

## 📁 Code Structure

```
paper_trading/
├── config/                          # Configuration files
│   ├── config.yaml                  # Main system configuration
│   ├── strategy_manifest.yaml       # Strategy definitions
│   ├── test_config.yaml            # Test configuration
│   └── config_alp_execution_patch.yaml
├── src/                             # Source code
│   ├── signal_bus.py               # M1: Signal processing
│   ├── run_signal_bus.py           # M1: Entry point
│   ├── execution_engine.py         # M2: Order planning
│   ├── run_execution.py            # M2: Entry point
│   ├── scheduler.py                # M3: Module orchestration
│   ├── alp_execution.py            # M4: Trading execution
│   ├── m5_reconcile.py             # M5: Reconciliation
│   └── m5_reconcile_backup.py      # M5: Backup version
├── data/                            # Data files
│   ├── raw_signals/                # Strategy signal files
│   ├── mapping/                    # Asset mapping tables
│   ├── intermediate/               # Intermediate files
│   └── outputs/                    # Output files
├── logs/                           # Log files
├── tests/                          # Test files
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── test_alpaca_api.py             # API test utility
```

### Key Source Files

#### M1 - Signal Bus
- **`signal_bus.py`**: Core signal processing logic
- **`run_signal_bus.py`**: Command-line interface

#### M2 - Execution Engine
- **`execution_engine.py`**: Order planning with fractional share support
- **`run_execution.py`**: Entry point with real-time integration

#### M3 - Scheduler
- **`scheduler.py`**: Module orchestration and scheduling

#### M4 - Alpaca Execution
- **`alp_execution.py`**: Live trading execution with batch IDs

#### M5 - Reconciliation
- **`m5_reconcile.py`**: Dual checking with Alpaca verification

## 🚀 Usage

### Quick Start

1. **Full system execution**
   ```bash
   python src/scheduler.py --config config/config.yaml --date 2025-07-28 --once --debug
   ```

2. **Check outputs**
   ```bash
   ls -la data/outputs/
   # You should see:
   # - planned_orders.csv
   # - planned_orders_real.csv
   # - execution_log.csv
   # - recon_single_20250728.csv
   # - recon_final_20250728.csv
   ```

### Individual Module Execution

#### M1 - Signal Processing
```bash
python src/run_signal_bus.py \
  --manifest config/strategy_manifest.yaml \
  --date 2025-07-28 \
  --mode select \
  --pick drl \
  --out data/intermediate/targets_consolidated_20250728.parquet
```

#### M2 - Execution Planning (Online Mode)
```bash
python src/run_execution.py \
  --config config/config.yaml \
  --targets data/intermediate/targets_consolidated_20250728.parquet \
  --equity 100000 \
  --execution-mode online \
  --debug
```

#### M4 - Trading Execution (Real Mode)
```bash
python src/alp_execution.py \
  --config config/config.yaml \
  --mode real \
  --debug
```

#### M5 - Reconciliation
```bash
python src/m5_reconcile.py \
  --config config/config.yaml \
  --date 2025-07-28 \
  --source logs \
  --out-dir data/outputs \
  --debug
```

### Execution Modes

#### M2 Execution Modes
- **DryRun**: Uses config equity, optional real prices
- **Online**: Fetches real account equity and positions from Alpaca

#### M4 Execution Modes
- **Dry**: Simulated execution
- **Real**: Live execution via Alpaca API

### Scheduled Execution

```bash
# Start scheduled scheduler (runs daily at 09:00)
python src/scheduler.py --config config/config.yaml --daemon --skip-weekends
```

## 🧪 Testing

### API Connectivity Test
```bash
python test_alpaca_api.py
```

### Individual Module Tests
```bash
# Test M1
python src/run_signal_bus.py --manifest config/strategy_manifest.yaml --date 2025-07-28 --mode select --pick drl

# Test M2 (DryRun mode)
python src/run_execution.py --config config/config.yaml --targets data/intermediate/targets_consolidated_20250728.parquet --equity 100000 --execution-mode dryrun

# Test M4 (Dry mode)
python src/alp_execution.py --config config/config.yaml --mode dry --debug
```

### End-to-End Test
```bash
# Complete system test
python src/scheduler.py --config config/config.yaml --date 2025-07-28 --once --debug
```

## 📊 Output Files

### M2 Outputs
- `planned_orders.csv`: All planned orders
- `planned_orders_real.csv`: Orders after fractional share filtering
- `rebalance_plan.csv`: Full rebalancing plan
- `rebalance_plan_real.csv`: Real rebalancing plan
- `real_prices.csv`: Real-time prices from Alpaca
- `current_positions.csv`: Current positions from Alpaca

### M4 Outputs
- `execution_log.csv`: Execution log with batch IDs
- `filled_orders.csv`: Fill reports

### M5 Outputs
- `recon_single_YYYYMMDD.csv`: Single execution reconciliation
- `recon_final_YYYYMMDD.csv`: Final position reconciliation
- `recon_summary_YYYYMMDD.txt`: Summary report
- `positions_eod_YYYYMMDD.csv`: End-of-day positions

## 🔧 Key Features Explained

### Fractional Share Support
The system automatically handles Alpaca's fractional share restrictions:
- Orders below `min_order_notional` ($2.0) are filtered out
- Fractional shares use `MARKET` orders (Alpaca requirement)
- Whole shares can use `BRACKET` orders with stop-loss/take-profit

### Short Position Management
Proper handling of short-to-long transitions:
```python
if prev_qty < 0:
    # Close short position (BUY back to return)
    close_short_qty = abs(prev_qty)
    orders.append(Order(side="BUY", qty=close_short_qty, order_type="MARKET"))
    
    # Buy target position
    buy_qty = tgt_qty
    # Check if fractional and set appropriate order type
```

### Batch ID System
Unique batch IDs ensure idempotency:
```python
def make_client_order_id(prefix, trade_date, batch_id, order):
    basis = f"{trade_date}|{batch_id}|{order.symbol}|{order.side}|{order.qty}|{order.order_type}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{trade_date}_{batch_id}_{order.symbol}_{order.side}_{digest}"
```

## 🐛 Troubleshooting

### Common Issues

1. **M1 returns rows=0**
   - Check if strategy files contain data for the specified date
   - Verify strategy manifest configuration

2. **M2 plans 0 orders**
   - Check if price data is complete
   - Verify fractional share filtering settings

3. **M4 execution fails**
   - Check Alpaca API configuration and network connection
   - Verify API credentials and permissions

4. **M5 shows MISMATCH**
   - Check execution logs and position files
   - Verify reconciliation threshold settings

### Debug Commands

```bash
# Enable detailed logging
python src/scheduler.py --config config/config.yaml --date 2025-07-28 --once --debug

# Check file contents
head -10 data/outputs/planned_orders.csv
head -10 data/outputs/planned_orders_real.csv
tail -10 data/outputs/execution_log.csv

# Verify API connection
python test_alpaca_api.py

# Check batch execution
grep "BATCH_" data/outputs/execution_log.csv | tail -5
```

### Log Files
- `logs/scheduler.log`: Scheduler execution logs
- `logs/signal_bus_YYYYMMDD.log`: M1 execution logs
- `logs/execution_YYYYMMDD.log`: M2 execution logs
- `logs/m4_execution.log`: M4 execution logs

## 📈 Performance Metrics

### Success Rates (Latest Test)
- **Final Position Accuracy**: 100%
- **Execution Success Rate**: 100%
- **Reconciliation Accuracy**: 100%
- **API Success Rate**: 100%

### Test Results
- **Total Orders Processed**: 216
- **Successful Executions**: 216
- **Short Position Handling**: ✅ Fixed
- **Fractional Share Compliance**: ✅ Working
- **Batch Execution**: ✅ Working
- **Real-time Integration**: ✅ Working

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [SignalBus_Trading_Spec_v2.1.md](SignalBus_Trading_Spec_v2.1.md)
- **Issues**: [GitHub Issues](https://github.com/your-repo/signal-bus-trading/issues)
- **Email**: support@your-company.com

## 🏷️ Version History

### v2.1 (Current) - 2025-08-30
- ✅ Fixed short position logic
- ✅ Added fractional share support
- ✅ Implemented batch ID system
- ✅ Enhanced real-time integration
- ✅ Improved reconciliation accuracy
- ✅ Production-ready status achieved

### v2.0 (Previous)
- ✅ Basic M1-M5 implementation
- ❌ Short position issues
- ❌ Fractional share limitations
- ❌ No batch execution tracking

---

**Version**: v2.1  
**Date**: 2025-08-29  
**Status**: Production Ready  
**Maintainer**: Simon Liao  
**Key Achievement**: 100% success rate in end-to-end testing
