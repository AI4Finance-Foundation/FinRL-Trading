# Signal-Bus 模块化量化交易系统（Alpaca Paper）— 功能说明书 v2.0

---

## 0. 系统概述

### 0.1 设计目标
- **模块化架构**：策略生产 → Signal Bus → 执行引擎 → 交易执行 （Alpaca 接口和下单/失败处理等等） → 对账分析 → 绩效监控
- **解耦设计**：各模块独立运行，通过文件接口通信
- **多策略支持**：单选（select）与融合（blend）模式
- **风险控制**：禁止做空、禁止现金头寸、权重归一化
- **实时执行**：支持 dry | Alpaca Paper()/Live 环境
- **完整审计**：从信号到绩效的端到端追踪

### 0.2 系统架构
```
策略文件 → M1(Signal Bus) → M2(Execution Engine) → M3(Scheduler) → M4(Alpaca Execution) → M5(Reconciliation) → M6(Analytics)
```

---

## 1. 目录结构

```text
paper_trading/
├── config/
│   ├── config.yaml                  # 系统主配置
│   └── strategy_manifest.yaml       # 策略清单配置
├── data/
│   ├── raw_signals/                 # 原始策略信号
│   │   ├── drl_weight.csv          # DRL 策略权重
│   │   ├── equally_weighted.xlsx   # 等权策略
│   │   ├── mean_weighted.xlsx      # 均值策略
│   │   └── minimum_weighted.xlsx   # 最小方差策略
│   ├── mapping/
│   │   └── tic_gvkey_mapping_2025.csv  # 资产映射表
│   ├── intermediate/               # 中间文件
│   │   └── targets_consolidated_YYYYMMDD.parquet
│   └── outputs/                    # 输出文件
│       ├── planned_orders.csv      # M2 计划订单
│       ├── rebalance_plan.csv      # M2 调仓计划
│       ├── execution_log.csv       # M4 执行日志
│       ├── positions_eod_YYYYMMDD.csv  # M5 收盘持仓
│       ├── recon_report_YYYYMMDD.csv   # M5 对账报告
│       ├── pnl_daily_YYYYMMDD.csv      # M6 日收益
│       └── risk_snapshot_YYYYMMDD.csv  # M6 风险快照
├── src/
│   ├── signal_bus.py               # M1 核心逻辑
│   ├── run_signal_bus.py           # M1 执行入口
│   ├── execution_engine.py         # M2 核心逻辑
│   ├── run_execution.py            # M2 执行入口
│   ├── scheduler.py                # M3 调度编排
│   ├── alp_execution.py            # M4 交易执行
│   ├── m5_reconcile.py             # M5 对账分析
│   └── m6_analytics.py             # M6 绩效分析（计划中）
├── logs/
│   └── scheduler.log               # 调度日志
└── tests/                          # 测试文件
```

---

## 2. 模块详细设计

### 2.1 M1 — Signal Bus（信号总线）

#### **业务逻辑**
- **策略信号读取**：从多个策略文件读取权重信号
- **信号标准化**：统一列名、数据类型、格式
- **策略融合**：支持单选（select）和加权融合（blend）
- **约束检查**：权重归一化、禁止做空、禁止现金头寸

#### **实现目标**
- 提供统一的策略信号接口
- 支持多策略并存和切换
- 确保输出信号的可执行性

#### **输入输出**
**输入：**
- `strategy_manifest.yaml`：策略配置文件
- 策略信号文件：`drl_weight.csv`、`equally_weighted.xlsx` 等
- 运行参数：`--date`、`--mode`、`--pick`、`--blend`

**输出：**
- `targets_consolidated_YYYYMMDD.parquet`：标准化目标权重

#### **核心函数**
```python
# signal_bus.py
def load_manifest(path: Path) -> List[StrategySpec]
def load_all_strategies(specs: List[StrategySpec]) -> pd.DataFrame
def pick_latest_per_strategy(df: pd.DataFrame) -> pd.DataFrame
def merge_for_date(df: pd.DataFrame, date: str, mode: str, blend: dict) -> pd.DataFrame
def enforce_constraints_and_normalize(df: pd.DataFrame, allow_short: bool, allow_cash: bool) -> pd.DataFrame
def save_targets(df: pd.DataFrame, path: Path)
```

#### **配置示例**
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

### 2.2 M2 — Execution Engine（执行引擎）

#### **业务逻辑**
- **目标权重解析**：读取 M1 输出的目标权重
- **持仓对比**：与当前持仓比较，计算差额
- **订单生成**：根据差额生成买卖订单
- **价格获取**：支持实时价格和样本价格
- **风险控制**：订单大小限制、价格检查

#### **实现目标**
- 将目标权重转换为可执行订单
- 支持实时价格获取
- 提供灵活的风险控制

#### **输入输出**
**输入：**
- `targets_consolidated_YYYYMMDD.parquet`：目标权重
- `tic_gvkey_mapping_2025.csv`：资产映射表
- `positions_eod_YYYYMMDD.csv`：当前持仓（可选）
- 价格数据：实时 API 或样本文件

**输出：**
- `planned_orders.csv`：计划订单列表
- `rebalance_plan.csv`：调仓计划详情

#### **核心函数**
```python
# execution_engine.py
def load_targets(path: Path) -> pd.DataFrame
def load_mapping(path: Path, id_col: str, sym_col: str) -> pd.DataFrame
def map_targets_to_symbols(targets: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame
def plan_orders(targets_df: pd.DataFrame, prices: Dict[str, float], positions: List[Position], equity: float) -> Tuple[List[Order], pd.DataFrame]
def save_orders_csv(orders: List[Order], path: Path)
def save_plan_csv(plan_df: pd.DataFrame, path: Path)
```

#### **价格获取功能**
```python
# run_execution.py
def find_latest_positions_file(outputs_dir: Path, current_timestamp: str = None) -> Path
def fetch_prices_for_targets(cfg: dict, targets_df: pd.DataFrame, mapping_df: pd.DataFrame, price_source: str, debug: bool) -> Dict[str, float]
```

---

### 2.3 M3 — Scheduler（调度编排）

#### **业务逻辑**
- **模块编排**：按顺序执行 M1→M2→M4→M5→M6
- **时间调度**：支持日频、周频、月频调度
- **依赖检查**：确保前置模块成功执行
- **错误处理**：模块失败时的降级策略

#### **实现目标**
- 提供统一的执行入口
- 支持定时和手动执行
- 确保模块间的正确依赖

#### **输入输出**
**输入：**
- `config.yaml`：系统配置
- 命令行参数：`--date`、`--once`、`--daemon`

**输出：**
- 各模块的执行结果
- 调度日志：`logs/scheduler.log`

#### **核心函数**
```python
# scheduler.py
def build_m1_cmd(project_root: Path, cfg: dict, run_date: str) -> list
def build_m2_cmd(project_root: Path, cfg: dict, targets_path: str, run_date: str) -> list
def build_m4_cmd(project_root: Path, cfg: dict) -> list
def build_m5_cmd(project_root: Path, cfg: dict, run_date: str) -> list
def run_once(project_root: Path, cfg: dict, run_date: str, logger: logging.Logger)
def next_run_after(now: datetime, run_time_hhmm: str) -> datetime
```

#### **配置示例**
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

### 2.4 M4 — Alpaca Execution（交易执行）

#### **业务逻辑**
- **订单解析**：读取 M2 生成的计划订单
- **风控检查**：订单大小、价格偏差、现金余量
- **API 交互**：与 Alpaca 交易 API 通信
- **执行监控**：订单状态跟踪、成交回报
- **幂等控制**：防止重复提交

#### **实现目标**
- 安全可靠地执行交易订单
- 提供完整的执行审计
- 支持干跑和实盘模式

#### **输入输出**
**输入：**
- `planned_orders.csv`：计划订单
- `rebalance_plan.csv`：调仓计划（可选）
- Alpaca API 配置

**输出：**
- `execution_log.csv`：执行日志
- `filled_orders.csv`：成交回报（可选）

#### **核心函数**
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

#### **价格获取功能**
```python
def fetch_prices_for_targets(cfg: dict, targets_df: pd.DataFrame, mapping_df: pd.DataFrame, price_source: str, debug: bool) -> Dict[str, float]
def save_prices_to_csv(prices: Dict[str, float], output_path: str)
```

---

### 2.5 M5 — Reconciliation（对账分析）

#### **业务逻辑**
- **双重检查**：
  - 单次执行对账：`planned_qty` vs `actual_qty`（检查 M4 执行情况）
  - 最终持仓对账：`target_qty` vs `final_actual_qty`（检查最终结果）
- **持仓计算**：基于执行日志计算实际持仓
- **权重分析**：计算当前权重与目标权重的偏差
- **报告生成**：生成详细的对账报告

#### **实现目标**
- 验证执行准确性
- 提供持仓透明度
- 支持权重偏差分析

#### **输入输出**
**输入：**
- `planned_orders.csv`：计划订单
- `execution_log.csv`：执行日志
- `positions_bod.csv`：期初持仓（可选）
- `rebalance_plan.csv`：调仓计划

**输出：**
- `positions_eod_YYYYMMDD.csv`：收盘持仓
- `recon_report_YYYYMMDD.csv`：对账报告

#### **核心函数**
```python
# m5_reconcile.py
def reconcile_from_logs(planned_path: Path, exec_log_path: Path, bod_positions_path: Path, rebalance_plan_path: Path, run_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]
def build_eod_positions(bod_positions: pd.DataFrame, execution_log: pd.DataFrame) -> pd.DataFrame
def calculate_weight_deviations(target_weights: pd.DataFrame, actual_positions: pd.DataFrame, prices: Dict[str, float], total_equity: float) -> pd.DataFrame
def save_positions_eod(positions: pd.DataFrame, run_date: str, out_dir: Path)
def save_recon_report(recon_df: pd.DataFrame, run_date: str, out_dir: Path)
```

#### **权重百分比计算**
```python
def calculate_weight_percentages(positions: pd.DataFrame, prices: Dict[str, float], total_equity: float) -> pd.DataFrame:
    """
    计算持仓权重百分比
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

### 2.6 M6 — Analytics & Monitoring（绩效分析）

#### **业务逻辑**
- **收益计算**：日收益、累计收益、超额收益
- **风险分析**：波动率、最大回撤、夏普比率
- **持仓分析**：权重分布、换手率、集中度
- **绩效归因**：策略贡献、行业贡献、个股贡献
- **监控告警**：异常交易、风险指标超限

#### **实现目标**
- 提供全面的绩效分析
- 支持风险监控和告警
- 生成可视化报告

#### **输入输出**
**输入：**
- `positions_eod_YYYYMMDD.csv`：历史持仓数据
- `execution_log.csv`：执行日志
- `recon_report_YYYYMMDD.csv`：对账报告
- 基准数据：市场指数、无风险利率

**输出：**
- `pnl_daily_YYYYMMDD.csv`：日收益数据
- `risk_snapshot_YYYYMMDD.csv`：风险快照
- `summary_dashboard.html`：汇总仪表板
- 告警通知：邮件、短信、API

#### **核心函数**（计划实现）
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

#### **风险指标计算**
```python
def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    计算风险指标
    """
    metrics = {}
    
    # 基础统计
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = returns.mean() * 252
    metrics['volatility'] = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
    
    # 回撤分析
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # 其他指标
    metrics['var_95'] = returns.quantile(0.05)
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    
    return metrics
```

#### **配置示例**
```yaml
# config.yaml
analytics:
  benchmark: "SPY"  # 基准指数
  risk_free_rate: 0.02  # 无风险利率
  risk_thresholds:
    max_drawdown: -0.20
    volatility: 0.25
    var_95: -0.03
  alerts:
    email: "trader@example.com"
    slack_webhook: "https://hooks.slack.com/..."
```

---

## 3. 数据流和文件接口

### 3.1 数据流图
```
策略文件 → M1 → targets_consolidated.parquet
                ↓
映射文件 → M2 → planned_orders.csv + rebalance_plan.csv
                ↓
Alpaca API → M4 → execution_log.csv
                ↓
执行日志 → M5 → positions_eod.csv + recon_report.csv
                ↓
历史数据 → M6 → pnl_daily.csv + risk_snapshot.csv + dashboard.html
```

### 3.2 文件格式规范

#### **目标权重文件**（M1 输出）
```csv
trade_date,asset_id,target_weight,asof,strategy_id
2025-07-28,1690,0.0047,2025-07-28T09:00:00Z,drl
2025-07-28,12141,0.0047,2025-07-28T09:00:00Z,drl
```

#### **计划订单文件**（M2 输出）
```csv
asset_id,symbol,side,qty,order_type,tp_price,sl_price
1690,AAPL,BUY,5.17,BRACKET,150.50,145.20
12141,MSFT,BUY,2.34,BRACKET,420.30,405.60
```

#### **执行日志文件**（M4 输出）
```csv
ts,mode,symbol,side,qty,order_type,tp_price,sl_price,time_in_force,client_order_id,price_used,notional,status,order_id,error
2025-07-28T09:00:00Z,DRY,AAPL,BUY,5.17,BRACKET,150.50,145.20,DAY,ORD_20250728_AAPL_BUY_abc123,148.30,766.71,SIMULATED,,
```

#### **对账报告文件**（M5 输出）
```csv
symbol,planned_qty,actual_qty,delta_qty,status,target_weight,actual_weight,weight_deviation
AAPL,5.17,5.17,0.00,OK,0.0047,0.0047,0.0000
MSFT,2.34,2.34,0.00,OK,0.0047,0.0047,0.0000
```

---

## 4. 配置管理

### 4.1 主配置文件（config.yaml）
```yaml
# Alpaca API 配置
alpaca:
  key_id: "${APCA_API_KEY_ID}"
  secret: "${APCA_API_SECRET_KEY}"
  trading_endpoint: "https://paper-api.alpaca.markets"
  data_endpoint: "https://data.alpaca.markets"

# 调度配置
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

# 执行配置
execution:
  out_orders: "./data/outputs/planned_orders.csv"
  out_plan: "./data/outputs/rebalance_plan.csv"
  time_in_force: "DAY"

# 风险控制
risk:
  max_share_per_order: 1000
  max_notional_per_order: 50000
  max_total_notional: 1000000
  price_tolerance_pct: 1.0
  min_cash_buffer_pct: 0.05
  reject_if_price_missing: false

# 幂等控制
idempotency:
  enable_client_order_id: true
  client_order_id_prefix: "ORD"

# 分析配置
analytics:
  benchmark: "SPY"
  risk_free_rate: 0.02
  risk_thresholds:
    max_drawdown: -0.20
    volatility: 0.25
```

### 4.2 策略清单配置（strategy_manifest.yaml）
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

## 5. 典型工作流

### 5.1 日频调仓流程
```bash
# 1. 手动执行单次调仓
python src/scheduler.py --config config/config.yaml --date 2025-07-28 --once --debug

# 2. 查看各模块输出
ls -la data/outputs/
# targets_consolidated_20250728.parquet
# planned_orders.csv
# rebalance_plan.csv
# execution_log.csv
# positions_eod_20250728.csv
# recon_report_20250728.csv
```

### 5.2 定时调度流程
```bash
# 启动定时调度器
python src/scheduler.py --config config/config.yaml --daemon --skip-weekends
```

### 5.3 模块独立执行
```bash
# M1: 信号处理
python src/run_signal_bus.py --manifest config/strategy_manifest.yaml --date 2025-07-28 --mode select --pick drl --out data/intermediate/targets_consolidated_20250728.parquet

# M2: 执行计划
python src/run_execution.py --config config/config.yaml --targets data/intermediate/targets_consolidated_20250728.parquet --equity 100000 --out-orders data/outputs/planned_orders.csv --out-plan data/outputs/rebalance_plan.csv --fetch-real-prices --price-source snapshot

# M4: 交易执行
python src/alp_execution.py --config config/config.yaml --mode dry --debug

# M5: 对账分析
python src/m5_reconcile.py --config config/config.yaml --date 2025-07-28 --source logs --out-dir data/outputs
```

---

## 6. 测试和验证

### 6.1 单元测试
```bash
# 测试 M1 信号处理
python -m pytest tests/test_signal_bus.py

# 测试 M2 执行引擎
python -m pytest tests/test_execution_engine.py

# 测试 M4 交易执行
python tests/run_m4_tests.py --config config/test_config.yaml
```

### 6.2 集成测试
```bash
# 端到端测试
python tests/test_m4_m5_m6.py --config config/test_config.yaml
```

### 6.3 性能测试
```bash
# 压力测试
python tests/performance_test.py --symbols 1000 --iterations 100
```

---

## 7. 监控和告警

### 7.1 系统监控
- **模块执行状态**：成功/失败/超时
- **API 调用统计**：成功率、响应时间、错误率
- **文件完整性**：文件大小、格式、内容验证

### 7.2 业务监控
- **交易执行质量**：成交率、滑点、延迟
- **风险指标**：持仓集中度、换手率、回撤
- **绩效指标**：收益率、夏普比率、信息比率

### 7.3 告警机制
- **实时告警**：交易失败、风险超限、系统异常
- **定期报告**：日报告、周报告、月报告
- **通知方式**：邮件、短信、Slack、API

---

## 8. 部署和维护

### 8.1 环境要求
- Python 3.10+
- 依赖包：pandas, numpy, requests, pyyaml, alpaca-py
- 操作系统：Linux/Windows/macOS
- 网络：稳定的互联网连接

### 8.2 部署步骤
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
export APCA_API_KEY_ID="your_key_id"
export APCA_API_SECRET_KEY="your_secret_key"

# 3. 创建必要目录
mkdir -p data/{raw_signals,intermediate,outputs} logs

# 4. 配置策略文件
# 将策略文件放入 data/raw_signals/

# 5. 测试配置
python src/scheduler.py --config config/config.yaml --date 2025-07-28 --once --debug
```

### 8.3 维护任务
- **日志清理**：定期清理旧日志文件
- **数据备份**：备份重要的配置和输出文件
- **性能优化**：监控系统性能，优化瓶颈
- **安全更新**：定期更新依赖包和安全补丁

---

## 9. 故障排除

### 9.1 常见问题
1. **M1 返回 rows=0**：检查策略文件是否存在对应日期的数据
2. **M2 计划 0 订单**：检查价格数据是否完整
3. **M4 执行失败**：检查 Alpaca API 配置和网络连接
4. **M5 出现 MISMATCH**：检查执行日志和持仓文件

### 9.2 调试技巧
```bash
# 启用详细日志
python src/scheduler.py --config config/config.yaml --date 2025-07-28 --once --debug

# 检查文件内容
head -10 data/outputs/planned_orders.csv
tail -10 data/outputs/execution_log.csv

# 验证 API 连接
python test_alpaca_api.py
```

---

## 10. 未来扩展

### 10.1 功能扩展
- **多资产支持**：期货、期权、加密货币
- **高级风控**：VaR、压力测试、情景分析
- **机器学习**：自动参数优化、异常检测
- **实时监控**：Web 界面、实时仪表板

### 10.2 架构扩展
- **微服务化**：各模块独立部署
- **容器化**：Docker 容器部署
- **云原生**：Kubernetes 编排
- **事件驱动**：消息队列、事件总线

---

**版本**：v2.0  
**日期**：2025-01-XX  
**状态**：M1-M5 已实现，M6 计划中  
**维护者**：Simon liao
