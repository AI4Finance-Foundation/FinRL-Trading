
# Signal-Bus 模块化量化交易系统（Alpaca Paper）— 功能说明书 v1.5

> 在 v1.0–v1.4 的基础上重构，统一目录结构、模块顺序、配置规范，便于后续开发与测试。  
> 根目录：`./paper_trading`

---
## 0. 目标与范围
- **解耦**：策略生产 → Signal Bus → 执行（Rebalance→Alpaca 下单）  
- **多策略**：支持单选（select）与融合（blend）  
- **约束**：禁止现金头寸（权重和=1），禁止做空（权重≥0）  
- **执行**：新开仓/加仓用 **Bracket** 保护，减仓/清仓用市价  
- **价格**：统一使用 **Alpaca Snapshot**，缺失回退最近 Bar  
- **调度**：推荐外部触发（CI/cron），可选服务常驻  

---
## 1. 目录结构

```text
paper_trading/
├── config/
│   ├── config.yaml                  # 系统参数：执行/风控/调度/幂等等
│   └── strategy_manifest.yaml       # 策略文件路径 + 列映射
├── data/
│   ├── raw_signals/                 # 各策略原始权重
│   ├── mapping/
│   │   └── instrument_map.csv       # 资产映射（gvkey→symbol）
│   ├── intermediate/                # M1 输出：当日目标权重
│   └── outputs/                     # M2/M4/M5 输出：订单、回执、持仓
├── logs/
│   ├── signal_bus_YYYYMMDD.log      # M1 运行日志
│   ├── execution_YYYYMMDD.log       # M2/M4 执行日志
│   └── scheduler.log                # M3 调度日志
├── src/
│   ├── signal_bus.py                # M1 模块
│   ├── run_signal_bus.py            # M1 CLI
│   ├── execution_engine.py          # M2 模块
│   ├── m4_execution.py              # M4 模块
│   ├── m5_reconcile.py              # M5 模块
│   ├── m6_analytics.py              # M6 模块
│   ├── scheduler.py                 # M3 调度器
│   └── utils/                       # 通用工具
├── tests/
│   ├── test_m1_main.py              
│   ├── test_m2_main.py              
│   ├── test_m3_main.py              
│   └── test_m4_m5_m6.py             
└── notebooks/
    └── exploration.ipynb            
```

---
## 2. 数据契约

### 2.1 策略信号文件（输入）
- **位置**：`./data/raw_signals/`
- **列**：`trade_date, asset_id, target_weight`（最小集）  
- **格式**：CSV/XLSX/Parquet（推荐 Parquet）

### 2.2 策略清单 Manifest
```yaml
strategies:
  - id: drl
    path: ./data/raw_signals/drl_weight.csv
    format: csv
    column_mapping: { trade_date: trade_date, gvkey: asset_id, weights: target_weight }
```

### 2.3 M1 输出
- `./data/intermediate/targets_consolidated_YYYYMMDD.parquet`
- 列：`trade_date, asset_id, target_weight`

### 2.4 资产映射
- 文件：`./data/mapping/instrument_map.csv`
- 列：`gvkey,symbol,...`

### 2.5 执行与回执
- 订单：`planned_orders.csv`  
- 调仓计划：`rebalance_plan.csv`  
- 回执：`execution_log.csv`、`filled_orders.csv`  
- 对账：`positions_eod_YYYYMMDD.csv`、`recon_report_YYYYMMDD.csv`  

---
## 3. 模块定义

### 3.1 M1 — Signal Bus
- 读取多策略信号 → 对齐、融合、约束检查 → 输出当日目标权重  
- 禁止做空，禁止现金，归一化到 1  
- 输出：`targets_consolidated_YYYYMMDD.parquet`

### 3.2 M2 — Execution Engine
- 输入：目标权重 + 价格 + 当前持仓  
- 计算：目标股数、差额、调仓计划、订单草案  
- 输出：`rebalance_plan.csv`、`planned_orders.csv`  

### 3.3 M3 — Scheduler
- 外部触发或服务常驻  
- 串行执行 M1→M2→M4→M5→M6  
- 支持依赖检查、失败重跑  


### 3.4 M4 — Execution Runner（交易执行）

**1. 输入**
- M2 生成的 `planned_orders.csv`（优先）或 `rebalance_plan.csv`
- `config/config.yaml` 中的 Alpaca 账户/环境配置（Paper/Live）
- 执行模式：`--mode dry`（默认）或 `--mode real`，支持 `--debug`

**2. 功能**
- **解析订单**：加载 `symbol/side/qty/order_type`（不足时从 rebalance_plan 推导）
- **风控检查**：
  - 单票股数上限：`risk.max_share_per_order`
  - 名义金额上限：`risk.max_notional_per_order`
  - 总名义金额上限：`risk.max_total_notional`
  - 价格容忍：`risk.price_tolerance_pct`（相对快照价）
  - 现金余量：`risk.min_cash_buffer_pct`
  - 缺价策略：`reject_if_price_missing`
- **下单执行**（`--mode real` 时）：
  - BUY：`MARKET` 主单 + `BRACKET`（TP/SL）子单
  - SELL：`MARKET` 卖出
  - `time_in_force=DAY`（默认，可配）
  - 可选幂等：`idempotency.enable_client_order_id`
- **回执记录**：
  - `execution_log.csv` 记录每笔结果（成功/失败、order_id、错误信息）
  - 可选轮询填充 `filled_orders.csv`

**3. 输出**
- `data/outputs/execution_log.csv`（append）
- `data/outputs/filled_orders.csv`（可选）
- 运行日志：`logs/m4_execution.log`

**4. CLI**

```bash
# 干跑（默认）
python ./src/m4_execution.py --config ./config/config.yaml --mode dry --debug

# 真实执行
python ./src/m4_execution.py --config ./config/config.yaml --mode real
```

# test script

python tests/run_m4_tests.py --config ./config/test_config.yaml

### 3.5 M5 — Reconciliation
- 拉取 Alpaca 订单、成交、持仓  
- 匹配 planned vs actual  
- 生成对账报告、收盘持仓快照  
- 输出：`positions_eod_YYYYMMDD.csv`、`recon_report_YYYYMMDD.csv`  

### 3.6 M6 — Analytics & Monitoring
- 计算 PnL、超额收益、回撤、风险敞口  
- 生成报表：`pnl_daily_YYYYMMDD.csv`、`risk_snapshot_YYYYMMDD.csv`、`summary_dashboard.html`  
- 告警：成交偏差、下单失败、回撤超限  

---
## 4. 配置与默认值
- 调仓频率：`frequency: daily|weekly|monthly`
- 风控参数：`tp_pct=0.05`, `sl_pct=0.03`
- 定价模式：`pricing: snapshot|bar|fill-price`
- 幂等：`idempotency: false`（默认关闭）

---
## 5. 测试计划

| 模块 | 用例 | 场景 | 期望 |
|---|---|---|---|
| M1 | T1 | Overlapping timestamps | 归一化=1 |
| M1 | T2 | Negative weights | 报错 |
| M2 | T3 | Market closed | 拒单 |
| M2 | T4 | Bracket submission | 成功 |
| M3 | T5 | Cron 触发 | 单次执行 |
| M4 | T6 | Dry-run vs Real | 日志/回执生成 |

---
## 6. 典型工作流示例

```bash
# M1: 生成目标权重
python ./src/run_signal_bus.py --manifest ./config/strategy_manifest.yaml --date 2025-03-01 --mode select --pick equal --out ./data/intermediate/targets_consolidated_20250301.parquet

# M2: 计算调仓计划
python ./src/run_execution.py --config ./config/config.yaml --targets ./data/intermediate/targets_consolidated_20250301.parquet --equity 10000 --out-orders ./data/outputs/planned_orders.csv --out-plan ./data/outputs/rebalance_plan.csv

# M3: 调度编排
python ./src/scheduler.py --config ./config/config.yaml --date 2025-03-01 --once --debug

# M4: 执行交易
python ./src/m4_execution.py --config ./config/config.yaml --mode real

# M5: 成交回报与对账
python ./src/m5_reconcile.py --config ./config/config.yaml --date 2025-03-01

# M6: 绩效与风险分析
python ./src/m6_analytics.py --config ./config/config.yaml --date 2025-03-01
```

---
## 7. 开放问题
- gvkey→symbol 映射更新频率与失配处理  
- Bracket 缺省 TP/SL 参数配置  
- 调度串并行与依赖策略  
