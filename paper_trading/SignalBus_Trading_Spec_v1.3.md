# Signal-Bus 模块化量化交易系统（Alpaca Paper）— 功能说明书 v1.2

> 本文在 v1.0 与 v1.1 的基础上合并整理，并纳入统一的项目目录结构（相对路径根：`./paper_trading`）。

- 参考：v1.0《Signal-Bus 模块化量化交易系统（Alpaca Paper）软件功能说明书》
- 参考：v1.1《Automated Paper Trading System - Functional Specification (v1.1)》

---

## 0. 目标与范围（沿用 + 精炼）

- **完全解耦**：策略生产（多模型输出中间文件）↔ Signal Bus（读取/归并/治理）↔ 执行（Rebalance→Alpaca 下单）。
- **多策略**：支持单选（select）与线性融合（blend）。
- **约束**：不允许现金头寸（权重和=1）、不允许做空（权重≥0）。
- **执行**：新开仓/加仓使用 **Bracket** 保护；减仓/清仓用市价。
- **价格**：统一使用 **Alpaca Snapshot**（必要时回退最近 Bar）。
- **调度**：默认推荐**外部触发**（CI/cron），分钟级可用服务常驻。

---
## Overall

M1: signal_bus.py  
    多策略信号 -> 目标权重文件 targets_consolidated_xxx.parquet

M2: execution_engine.py
    目标权重 + 价格/持仓 -> 调仓计划 planned_orders.csv

M4: m4_execution.py（即将开发）
    调仓计划 + Alpaca API -> 真正下单（Paper/Live）

M1：信号总线模块

文件：signal_bus.py（我们后来把CLI封装成run_signal_bus.py）

主要职责

读取信号源

从不同的输出策略文件（CSV / Parquet）读取当日目标权重

这些策略可能

归并与标准化

查看日期、资产分析

选择单一（策略--pick） 策略融合（--blend）

当日输出目标权重

输出到data/intermediate/targets_consolidated_YYYYMMDD.parquet

每个符号一个target_weight，总和 = 1_

产

一个标准化文件：当日目标权重

这是调仓的目标状态，但还没有涉及实际下单

M2：执行引擎模块

文件：execution_engine.py（CLI 是run_execution.py）

主要

读取当日目标权重（M1的输出）

targets_consolidated_YYYYMMDD.parquet

获取价格+持仓（目前我们用静态价格，后续M4会实时接羊驼）

计算目标市值

与当前市值/仓位做差

计算调仓计划

生成planned_orders.csv：需要买/卖多少股

输出结果

planned_orders.csv：单笔调仓指令（symbol、qty、side）

rebalance_plan.csv：调仓、互换持仓、差额、权重等统计信息

特点

M2本身不下单，只负

真正发到羊驼的动作会在M4实现

M4 Execution Runner — 设计与规范（v0.1）
目标

把M2增量的调仓计划变成可审计、可回放的真实交易执行流程，支持纸质/现场环境，默认Dry-Run，逐步切换到真实下单。

1. 输入
1.1 调仓计划文件（从目标目录自动发现）

优先顺序：

./data/outputs/planned_orders.csv（M2输出）

回退：./data/outputs/rebalance_plan.csv（含明细时可解析出订单）

若两者都存在，默认读取planned_orders.csv。

文件格式见§5.1。

1.2 配置（./config/config.yaml）
1.3 执行模式（CLI）

--mode dry（默认）或--mode real

--debug（打印更多诊断）

任选：--prices ./data/outputs/price_template.csv（risk.price_source=file当时）

2. 功能
2.1 加载与解析

读取planned_orders.csv（或回退读取rebalance_plan.csv并提取action/ planned_qty）

过滤掉planned_qty <= 0的行

合法性检查：

字段齐全：（symbol/side/qty/order_type或预备计划推导）

符合约束策略（允许空头、不允许现金权重由M1/M2已保证）

2.2 账户与行情拉取

读取账户：/v2/account：现金、买力（buying_power）

读取持仓：/v2/positions：当前symbol → qty

读取快照：（/v2/stocks/{symbol}/snapshot或批量端点），获得last.price作为风控参考

2.3 风控（下单前检查）

逐单计算并校验：

单笔股数不超过risk.max_share_per_order

单笔金额（qty * ref_price）不超过risk.max_notional_per_order

若risk.price_source=alpaca_snapshot：

买：ref_price <= last * (1 + price_tolerance_pct)

卖出：ref_price >= last * (1 - price_tolerance_pct)
（注：市价单仅做参考拦截；括号的TP/SL也需满足羊驼规则：TP ≥ 基准 + $0.01，SL ≤ 基准 − $0.01。）

总计金额不超过risk.max_total_notional

办理后现金需≥ min_cash_buffer_pct * equity（约）

缺少价格时：若reject_if_price_missing=true则拒单并记录失败原因

2.4 下单执行

买入（增仓/新开） → order_type=BRACKET：

主单：MARKET BUY（或可配置为LIMIT-IOC/Day，首版默认市价）

子单：

take_profit.limit_price = round(base*(1+tp_pct), 2)

stop_loss.stop_price = round(base*(1-sl_pct), 2)

卖出（减仓/清仓） →MARKET SELL

time_in_force: DAY（默认，可配置）

client_order_id：idempotency.enable_client_order_id=true当时生成：
${prefix}-${trade_date}-${symbol}-${nonce}

注：羊驼对 BACKET 有价格方向约束，若tp <= base或sl >= base，需自动调节或拒单（记录原因）。

2.5 订单追踪（默认开启）

保存每笔请求与响应（含id/status/submitted_at）到execution_log.csv（append）

轮询征求意见，把发起成交明细（首版：下单即记录，不强制轮询，其次与外部OMS共存/v2/orders/{id}）filled/canceledfilled_orders.csv

3. 输出
3.1 执行日志（CSV）

路径：./data/outputs/execution_log.csv
字段（建议）：datetime, env, mode, symbol, side, qty, order_type, tp_price, sl_price,ref_price, notional, client_order_id, status, order_id, error

3.2 成交明细（CSV）

路径：./data/outputs/filled_orders.csv

追加模式
字段（建议）：
datetime, env, symbol, side, filled_qty, avg_fill_price, order_id, status

M5 和 M6 概览
M5：成交返回与对账（Post-Trade & Reconciliation）

目标：把羊驼的实际成交与我们计划的订单形成，“真实的持仓与流水”，并带动后续绩效统计与风险监控。

M4 职责

拉取订单/成交：

/v2/orders（含子单：止盈/止损）
>
/v2/positions（目前真实持仓）

（任选）/v2/account/activities（DIV、FEE、JNLC/JNCP 等）

对齐计划与事实：

将planned_orders.csv与orders/fills进行匹配（按symbol/side/qty，若启用client_order_id则精确映射）

识别部分成交、剩余未成交、被拒绝、被取消等状态

更新本地台账（仅附加）：

./data/outputs/filled_orders.csv（成交明细）

./data/outputs/execution_log.csv（补充：含API回执、错误码）

./data/outputs/positions_eod_YYYYMMDD.csv（收盘持仓快照）

./data/outputs/cash_ledger.csv（现金/费用/分红流水，交易）

输入

planned_orders.csv（来自M2/M4）

Alpaca API：订单、仓位、活动

输出

事实成交：filled_orders.csv

收盘持仓：positions_eod_YYYYMMDD.csv

账面差异：recon_report_YYYYMMDD.csv（例如：计划 100 股，成交 95 股；未成交 5 股，原因：超时）

触发

M3调度在交易日收盘后或执行完成后触发一次（可配置“T+0收盘后/T+1开盘前”）

M5 和 M6 概览
M5：成交返回与对账（Post-Trade & Reconciliation）

目标：把羊驼的实际成交与我们计划的订单形成，“真实的持仓与流水”，并带动后续绩效统计与风险监控。

职责

拉取订单/成交：

/v2/orders（含子单：止盈/止损）

/v2/positions（目前真实持仓）

（任选）/v2/account/activities（DIV、FEE、JNLC/JNCP 等）

对齐计划与事实：

将planned_orders.csv与orders/fills进行匹配（按symbol/side/qty，若启用client_order_id则精确映射）

识别部分成交、剩余未成交、被拒绝、被取消等状态

更新本地台账（仅附加）：

./data/outputs/filled_orders.csv（成交明细）

./data/outputs/execution_log.csv（补充：含API回执、错误码）

./data/outputs/positions_eod_YYYYMMDD.csv（收盘持仓快照）

./data/outputs/cash_ledger.csv（现金/费用/分红流水，交易）

输入

planned_orders.csv（来自M2/M4）

Alpaca API：订单、仓位、活动

输出

事实成交：filled_orders.csv

收盘持仓：positions_eod_YYYYMMDD.csv

账面差异：recon_report_YYYYMMDD.csv（例如：计划 100 股，成交 95 股；未成交 5 股，原因：超时）

触发

M3调度在交易日收盘后或执行完成后触发一次（可配置“T+0收盘后/T+1开盘前”）

M6：绩效评估、风险监控与另外（分析与监控）

目标：把M5的事实数据转成可视化指标与另外，支撑策略检视与日常运维。

职责

成效：

日/周/月度累计净值（权益曲线）

相对基准（SPY等）的超额、Alpha/Beta、回撤（MDD）

交易层面KPI：命中率、盈亏比、换手率、滑点（若有基准价）

风险监控：

集中度（Top N 权重、行业/风格）

杠杆/现金阈值检查

单票/总票据风险暴露监控

另有机制（可接企业微信/飞书/Slack/Email）：

下单失败/重复下单

成交偏差大（计划 vs 实际）

现金不足/融资活动超限

当日回撤超阈值（如-3%）

报表统计：

./data/reports/pnl_daily_YYYYMMDD.csv

./data/reports/risk_snapshot_YYYYMMDD.csv

./data/reports/summary_dashboard.html（静态报表，相当）

输入

M5的filled_orders.csv、、positions_eod_*.csvcash_ledger.csv

市场基准（SPY）行情（同 Alpaca 或外部文件）

输出

统计报表 CSV/HTML

颈部日志./logs/alerts.log，以及外部通道（任选）

触发

每笔交易日收盘后（或T+1开盘前）运行

约可盘中按分钟/小时运行（任选）


### 流程图
flowchart LR
    subgraph M1[ M1 Signal Bus ]
      A1[读取多策略信号\nCSV/Parquet]
      A2[对齐/归一化/融合]
      A3[输出目标权重\n targets_consolidated_YYYYMMDD.parquet]
      A1-->A2-->A3
    end

    subgraph M2[ M2 Execution Engine ]
      B1[读取目标权重 + 映射表]
      B2[加载价格与持仓\n(本地或快照)]
      B3[计算目标股数/差额]
      B4[生成调仓计划\n rebalance_plan.csv]
      B5[生成订单草案\n planned_orders.csv]
      B1-->B2-->B3-->B4-->B5
    end

    subgraph M3[ M3 Scheduler ]
      C1[外部触发/定时触发]
      C2[串行执行 M1→M2→M4]
      C1-->C2
    end

    subgraph M4[ M4 Execution Runner ]
      D1[读取 planned_orders.csv]
      D2[风控校验\n(额度/价格容忍/现金余量)]
      D3{dry-run\n或 real?}
      D4[调用 Alpaca /v2/orders 下单\nBUY=BRACKET, SELL=MARKET]
      D5[execution_log.csv]
      D1-->D2-->D3
      D3-- real -->D4-->D5
      D3-- dry -->D5
    end

    subgraph M5[ M5 Reconciliation ]
      E1[拉取 orders/fills/positions]
      E2[对齐计划与成交\n识别差异]
      E3[filled_orders.csv]
      E4[positions_eod_YYYYMMDD.csv]
      E5[recon_report_YYYYMMDD.csv]
      E1-->E2-->E3
      E2-->E4
      E2-->E5
    end

    subgraph M6[ M6 Analytics & Monitoring ]
      F1[汇总 PnL/净值/回撤/暴露]
      F2[风控监控与告警]
      F3[pnl_daily_*.csv / risk_snapshot_*.csv]
      F4[summary_dashboard.html]
      F1-->F3
      F1-->F4
      F2-->F3
    end

    A3-->B1
    B5-->D1
    M3-->M1
    M3-->M2
    M3-->M4
    D5-->E1
    E3-->F1
    E4-->F1


## 1. 目录结构（相对路径，根：`./paper_trading`）

```text
paper_trading/
├── config/
│   ├── config.yaml                  # 系统参数：执行/风控/调度/幂等等
│   └── strategy_manifest.yaml       # 策略文件路径 + 列映射
├── data/
│   ├── raw_signals/                 # 各策略原始权重 (CSV/XLSX/Parquet)
│   ├── mapping/
│   │   └── instrument_map.csv       # 资产映射（gvkey→symbol 等）
│   ├── intermediate/                # M1 产出的目标权重快照（当日）
│   └── outputs/                     # M2 订单、回执、持仓快照等
├── logs/
│   ├── signal_bus_YYYYMMDD.log      # M1 运行日志
│   ├── execution_YYYYMMDD.log       # M2 执行日志
│   └── scheduler.log                # M3 调度日志
├── src/
│   ├── signal_bus.py                # M1：读取→归并→治理→快照
│   ├── run_signal_bus.py            # M1 CLI 运行器
│   ├── execution_engine.py          # M2：执行引擎
│   ├── scheduler.py                 # M3：调度器
│   └── utils/                       # 通用工具（映射/日志/时间）
├── tests/
│   ├── test_m1_main.py              # M1 单元测试（T1/T2 等）
│   ├── test_m2_main.py              # M2 单测（未来）
│   └── test_m3_main.py              # M3 单测（未来）
└── notebooks/
    └── exploration.ipynb            # 可选：调研/实验
```

> 所有读写路径均使用**相对路径**，便于迁移与部署。

---

## 2. 数据契约（文件接口）

### 2.1 策略信号文件（输入）
- **位置**：`./paper_trading/data/raw_signals/`
- **格式**：CSV / XLSX / Parquet（推荐 Parquet）
- **最小列**：`trade_date`（date）、`asset_id`（gvkey）、`target_weight`（float ≥0）
- **推荐列**：`strategy_id`、`asof`（无则用文件 mtime）`confidence`、`predicted_return` 等

> 与现有文件兼容：如 `drl_weight.csv`（`trade_date|gvkey|weights`）等，统一在加载时映射到标准列。

### 2.2 策略清单（Strategy Manifest）
- **文件**：`./paper_trading/config/strategy_manifest.yaml`
- **作用**：注册策略文件路径、格式、列映射。
- **示例**：
```yaml
strategies:
  - id: drl
    path: ./paper_trading/data/raw_signals/drl_weight.csv
    format: csv
    column_mapping: {{ trade_date: trade_date, gvkey: asset_id, weights: target_weight }}

  - id: equal
    path: ./paper_trading/data/raw_signals/equally_weighted.xlsx
    format: xlsx
    column_mapping: {{ trade_date: trade_date, gvkey: asset_id, weights: target_weight }}
```

### 2.3 目标权重快照（Signal Bus 输出）
- **位置**：`./paper_trading/data/intermediate/targets_consolidated_YYYYMMDD.parquet`
- **列**：`trade_date, asset_id, target_weight, asof, strategy_set_id?, blend_mode?, run_id?`
- **语义**：用于 M2 执行的最终目标权重（已治理与归一化）。

### 2.4 资产映射
- **文件**：`./paper_trading/data/mapping/instrument_map.csv`
- **列**：`gvkey,symbol,...`
- **用途**：执行前将 `asset_id=gvkey` 映射为 Alpaca `symbol`；缺失映射的标的跳过并记录。

### 2.5 执行输出与回执
- **位置**：`./paper_trading/data/outputs/`
- **示例**：`orders_YYYYMMDD.csv`、`positions_YYYYMMDD.csv` 等。

---

## 3. 模块定义与接口

### 3.1 M1 — Signal Bus（读取→归并→治理→快照）
**核心职责**
1. 读取 manifest，加载各策略文件（支持 CSV/XLSX/Parquet）。  
2. `pick_latest_per_strategy`：同 `(strategy_id, trade_date, asset_id)` 取最新 `asof`。  
3. `merge_for_date`：按 `select`（单选）或 `blend`（线性融合）形成当日目标。  
4. `enforce_constraints_and_normalize`：**禁止做空、禁止现金**，严格归一化到 1。  
5. 写出 `targets_consolidated_YYYYMMDD.parquet`。

**关键函数**
- `load_manifest(path) -> List[StrategySpec]`
- `load_all_strategies(specs) -> pd.DataFrame`
- `pick_latest_per_strategy(df) -> pd.DataFrame`
- `merge_for_date(df, trade_date, mode, blend) -> pd.DataFrame`
- `enforce_constraints_and_normalize(df, allow_short=False, allow_cash=False) -> pd.DataFrame`
- `save_targets(df, out_path)`

**CLI 使用**
```bash
python ./paper_trading/src/run_signal_bus.py   --manifest ./paper_trading/config/strategy_manifest.yaml   --date 2018-03-01   --mode select --pick drl   --out ./paper_trading/data/intermediate/targets_consolidated_20180301.parquet
```

**测试（示例）**
- **T1**：Overlapping timestamps → 合并、排序并归一化到 1。  
- **T2**：Negative weights → 抛出校验错误。

### 3.2 M2 — Execution Engine（目标→订单→Alpaca）
**策略**：
- 新开仓/加仓 → **Bracket**（需满足 `TP≥base+0.01`、`SL≤base−0.01`）；
- 减仓/清仓 → 市价。
- 价格源 → **Alpaca Snapshot**（回退最近 Bar）。

**关键接口（预留）**
- `load_targets(path) -> pd.DataFrame`
- `map_asset_to_symbol(asset_id) -> symbol`
- `get_last_prices(symbols) -> Dict[str,float]`
- `get_equity() -> float`
- `rebalance_to_orders(targets, equity, current_positions, prices, ...) -> List[Order]`
- `submit_orders(orders) -> receipts`



### 3.3 M3 — Scheduler / Orchestrator
- **外部触发（推荐）**：CI/cron 先跑 M1 → 再跑 M2；清晰可审计。  
- **服务常驻**：分钟级低延迟，进程内缓存、重试与健康检查。

---

## 4. 约束与默认值（合并 v1.1）

- **No Cash**：权重和必须=1（不允许现金头寸）。  
- **No Short**：权重必须≥0。  
- **Pricing**：统一 Alpaca Snapshot；缺失回退最近 Bar；仍缺失则跳过并记录。  
- **Orders**：BUY/加仓→Bracket；SELL/减仓→市价。  
- **Idempotency**：支持 `client_order_id`，**默认关闭**；开启后重试复用同 ID。  
- **Scheduling**：日/季频推荐外部触发；分钟级可服务常驻或混合。

---

## 5. 测试计划（摘录）

| 模块 | 用例 | 场景 | 期望 |
|---|---|---|---|
| M1 | T1 | Overlapping timestamps | 合并、排序、归一化=1 |
| M1 | T2 | Negative weights | 报错（禁止做空） |
| M2 | T3 | Market closed | 拒单/告警 |
| M2 | T4 | Bracket submission | 成功回执 |
| M3 | T5 | Cron 触发 | 单次执行后退出 |
| M3 | T6 | 服务常驻两日 | 每日定时执行 |

---

## 6. 里程碑（保持不变）
1) **M1：Signal Bus**（完成）  
2) **M2：Rebalance + Alpaca Adapter + Execution Policy**  
3) **M3：Scheduler / Orchestrator**  
4) **M4：Observability**（日志/回执/告警）  
5) **M5：Risk 扩展**（黑名单、换手、净敞口、波动门控）  
6) **M6：E2E 验收**（以样例数据重放）

---

## 7. 开放问题
- gvkey→symbol 映射来源与更新频率（缓存与失配处理）。
- Bracket 的 TP/SL 具体比例缺省值（例如 TP=+5%，SL=−3%）。
- 外部调度的串并行策略与依赖（等待所有策略文件落地再执行）。

---
##8.运行样例 : go to ./paper_trading/
###M1: 
python ./src/run_signal_bus.py --manifest ./config/strategy_manifest.yaml --date 2025-03-01 --mode select --pick equal --out ./data/intermediate/targets_consolidated_20180301.parquet

###M2: 
一键干跑（无需 Alpaca，仅验算法）
python ./src/run_execution.py   --config ./config/config.yaml   --targets ./data/intermediate/targets_consolidated_20250301.parquet   --equity 10000 --debug

python ./src/run_execution.py   --config ./config/config.yaml   --targets ./data/intermediate/targets_consolidated_20250301.parquet   --equity 10000  --fill-price 100  --debug

M3
# 运行一次并记录日志（推荐先这样）
python ./src/scheduler.py --config ./config/config.yaml --date 2025-03-01 --once --debug

# 常驻模式（每天在 config.yaml 的 scheduler.time 触发），写入日志
python ./src/scheduler.py --config ./config/config.yaml --daemon --skip-weekends --debug

单元测试（M2）
python ./tests/test_m2_main.py


*编制人：Simon
*日期：* 2025-08-22
