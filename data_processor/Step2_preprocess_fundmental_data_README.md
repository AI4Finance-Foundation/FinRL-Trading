# Step2_preprocess_fundmental_data.py 使用说明

## 功能描述

这个脚本用于预处理S&P 500股票的基本面数据并计算财务比率。它将季度基本面数据和日度价格数据整合，生成一个包含各种财务比率的综合数据集，用于机器学习应用。

## 主要功能

1. **数据加载与预处理**：加载基本面数据和价格数据
2. **交易日期调整**：将季度报告日期调整为交易日期
3. **股票代码匹配**：匹配基本面数据和价格数据中的股票代码和GVKEY
4. **财务比率计算**：计算多种财务比率（盈利能力、流动性、效率、杠杆等）
5. **数据清理**：处理缺失值和异常值
6. **结果输出**：生成主数据集和按行业分组的文件

## 依赖库

```bash
pip install pandas numpy openpyxl
```

### 必需库
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **openpyxl**: Excel文件读写（用于输出行业分组文件）

### 可选库
- **matplotlib**: 数据可视化（如果需要在脚本中添加图表功能）

## 使用方法

### 基本用法

```bash
python Step2_preprocess_fundmental_data.py --Stock_Index_fundation_file "sp500_tickers_fundamental_quarterly_20250712.csv" --Stock_Index_price_file "sp500_tickers_daily_price_20250712.csv"
```

### 指定输出目录

```bash
python Step2_preprocess_fundmental_data.py --Stock_Index_fundation_file "sp500_tickers_fundamental_quarterly_20250712.csv" --Stock_Index_price_file "sp500_tickers_daily_price_20250712.csv" --output_dir "my_outputs"
```

### 包含Sector 0记录

```bash
python Step2_preprocess_fundmental_data.py --Stock_Index_fundation_file "sp500_tickers_fundamental_quarterly_20250712.csv" --Stock_Index_price_file "sp500_tickers_daily_price_20250712.csv" --include_sector0
```

## 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--Stock_Index_fundation_file` | str | 是 | - | 基本面数据CSV文件路径 |
| `--Stock_Index_price_file` | str | 是 | - | 价格数据CSV文件路径 |
| `--output_dir` | str | 否 | "outputs" | 输出目录 |
| `--include_sector0` | flag | 否 | False | 在分Sector Excel文件中包含Sector 0记录 |

## 输入文件格式

### 基本面数据文件 (Stock_Index_fundation_file)
CSV格式，包含以下必需列：
- `datadate`: 报告日期
- `tic`: 股票代码
- `gvkey`: 公司唯一标识符
- `gsector`: GICS行业分类
- `prccq`: 季度收盘价
- `adjex`: 调整因子
- `epspxq`: 每股收益
- `revtq`: 季度收入
- `cshoq`: 流通股数
- `atq`: 总资产
- `ltq`: 总负债
- 其他财务指标列...

### 价格数据文件 (Stock_Index_price_file)
CSV格式，包含以下必需列：
- `gvkey`: 公司唯一标识符
- `tic`: 股票代码
- `datadate`: 交易日期
- `prccd`: 日度收盘价
- `ajexdi`: 日度调整因子

## 输出文件

### 主输出文件
- `final_ratios.csv`: 包含所有财务比率的主数据集

### 行业分组文件
- `sector10.xlsx`: 能源行业数据
- `sector15.xlsx`: 原材料行业数据
- `sector20.xlsx`: 工业行业数据
- `sector25.xlsx`: 非必需消费品行业数据
- `sector30.xlsx`: 必需消费品行业数据
- `sector35.xlsx`: 医疗保健行业数据
- `sector40.xlsx`: 金融行业数据
- `sector45.xlsx`: 信息技术行业数据
- `sector50.xlsx`: 通信服务行业数据
- `sector55.xlsx`: 公用事业行业数据
- `sector60.xlsx`: 房地产行业数据
- `sector0.xlsx`: Sector 0数据（仅在指定`--include_sector0`时生成）

## 输出数据列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| `date` | datetime | 交易日期 |
| `gvkey` | int | 公司唯一标识符 |
| `tic` | str | 股票代码 |
| `gsector` | float | GICS行业分类 |
| `adj_close_q` | float | 调整后季度收盘价 |
| `y_return` | float | 下一季度收益率 |
| `EPS` | float | 每股收益 |
| `BPS` | float | 每股账面价值 |
| `DPS` | float | 每股股息 |
| `cur_ratio` | float | 流动比率 |
| `quick_ratio` | float | 速动比率 |
| `cash_ratio` | float | 现金比率 |
| `acc_rec_turnover` | float | 应收账款周转率 |
| `debt_ratio` | float | 负债比率 |
| `debt_to_equity` | float | 负债权益比 |
| `pe` | float | 市盈率 |
| `ps` | float | 市销率 |
| `pb` | float | 市净率 |

## 内部函数说明

### 数据加载函数

#### `load_data(fundamental_file, price_file)`
- **功能**: 加载基本面数据和价格数据
- **输入**: 两个CSV文件路径
- **输出**: 两个DataFrame (fundamental_df, price_df)
- **特点**: 为节省内存，价格数据只加载必要列

### 数据处理函数

#### `adjust_trade_dates(fund_df)`
- **功能**: 将季度报告日期调整为交易日期
- **输入**: 基本面数据DataFrame
- **输出**: 包含调整后交易日期的DataFrame
- **逻辑**: 根据季度报告日期计算对应的交易日期

#### `calculate_adjusted_close(fund_df)`
- **功能**: 计算调整后收盘价
- **输入**: 基本面数据DataFrame
- **输出**: 包含调整后收盘价的DataFrame
- **公式**: `adj_close_q = prccq / adjex`

#### `match_tickers_and_gvkey(fund_df, df_daily_price)`
- **功能**: 匹配基本面数据和价格数据中的股票代码
- **输入**: 基本面数据DataFrame和价格数据DataFrame
- **输出**: 过滤后的基本面数据DataFrame
- **逻辑**: 只保留在价格数据中存在的股票

#### `calculate_next_quarter_returns(fund_df)`
- **功能**: 计算下一季度收益率
- **输入**: 基本面数据DataFrame
- **输出**: 包含下一季度收益率的DataFrame
- **公式**: `y_return = log(adj_close_q(t+1) / adj_close_q(t))`

### 财务比率计算函数

#### `calculate_basic_ratios(fund_df)`
- **功能**: 计算基本财务比率（PE、PS、PB）
- **输入**: 基本面数据DataFrame
- **输出**: 包含基本比率的DataFrame

#### `calculate_financial_ratios(fund_data)`
- **功能**: 计算综合财务比率
- **输入**: 处理后的基本面数据DataFrame
- **输出**: 包含所有财务比率的DataFrame
- **包含比率**:
  - 盈利能力比率：OPM、NPM、ROA、ROE
  - 流动性比率：流动比率、速动比率、现金比率
  - 效率比率：存货周转率、应收账款周转率、应付账款周转率
  - 杠杆比率：负债比率、负债权益比

### 数据清理函数

#### `handle_missing_values(ratios)`
- **功能**: 处理缺失值和异常值
- **输入**: 包含财务比率的DataFrame
- **输出**: 清理后的DataFrame
- **处理逻辑**:
  - 移除调整后收盘价为0的行
  - 将无效值（NaN、inf）替换为NaN
  - 移除包含过多无效值的列
  - 移除包含任何缺失值的行

### 结果保存函数

#### `save_results(final_ratios, output_dir)`
- **功能**: 保存处理结果到文件
- **输入**: 最终数据DataFrame和输出目录
- **输出**: 主输出文件路径
- **生成文件**:
  - 主CSV文件
  - 按行业分组的Excel文件

## 处理流程

1. **数据加载**: 读取基本面数据和价格数据文件
2. **日期调整**: 将报告日期调整为交易日期
3. **价格计算**: 计算调整后收盘价
4. **数据匹配**: 匹配基本面数据和价格数据
5. **收益率计算**: 计算下一季度收益率
6. **比率计算**: 计算各种财务比率
7. **数据清理**: 处理缺失值和异常值
8. **结果保存**: 保存到CSV和Excel文件

## 示例输出

```
================================================================================
S&P 500 Fundamental Data Preprocessing Tool
================================================================================
Fundamental file: sp500_tickers_fundamental_quarterly_20250712.csv
Price file: sp500_tickers_daily_price_20250712.csv
Output directory: outputs
--------------------------------------------------------------------------------
Loading data files...
Loading price data (only necessary columns)...
Fundamental data shape: (86175, 679)
Price data shape: (5320619, 5)
Unique tickers in fundamental data: 971
Unique tickers in price data: 1028
Adjusting trade dates...
Calculating adjusted close price...
Matching tickers and gvkey...
Original fundamental data shape: (86175, 681)
Filtered fundamental data shape: (86075, 681)
Unique gvkeys: 966
Calculating next quarter returns...
Data shape after calculating returns: (86058, 683)
Calculating basic financial ratios...
Selecting relevant columns...
Calculating comprehensive financial ratios...
  Calculating profitability ratios...
  Calculating liquidity ratios...
  Calculating efficiency ratios...
  Calculating leverage ratios...
Handling missing values...
Dropped columns: ['OPM', 'NPM', 'ROA', 'ROE', 'inv_turnover', 'acc_pay_turnover']
Final data shape: (81931, 18)
Saving results...
Main results saved to: outputs\final_ratios.csv
Saving sector-specific files...
  Sector 10: outputs\sector10.xlsx (3272 records)
  Sector 15: outputs\sector15.xlsx (3126 records)
  ...
================================================================================
Processing completed successfully!
Final dataset shape: (81931, 18)
Output saved to: outputs\final_ratios.csv
================================================================================
```

## 注意事项

1. **内存使用**: 脚本已优化内存使用，只加载价格数据的必要列
2. **数据质量**: 脚本会自动处理缺失值和异常值
3. **文件大小**: 输出文件可能较大，确保有足够的磁盘空间
4. **依赖库**: 确保安装了所有必需的Python库
5. **数据格式**: 确保输入文件格式正确，包含所有必需的列
6. **Sector 0处理**: 
   - 主CSV文件（final_ratios.csv）包含所有数据，包括Sector 0
   - 默认情况下，分Sector Excel文件不包含Sector 0（gsector缺失的股票）
   - 使用`--include_sector0`参数可在分Sector文件中包含Sector 0

## 错误处理

脚本包含完整的错误处理机制：
- 文件不存在检查
- 数据格式验证
- 内存不足处理
- 异常值处理

## 性能优化

- 使用`usecols`参数只加载必要的价格数据列
- 批量处理财务比率计算
- 高效的数据清理算法
- 内存友好的数据处理流程 