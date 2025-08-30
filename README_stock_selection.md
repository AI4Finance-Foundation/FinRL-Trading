# 股票选择系统使用说明

## 系统概述

本系统用于基于基本面数据的股票选择和投资组合构建。系统通过机器学习模型预测股票的未来收益率，并基于预测结果选择最优股票组合。

## 文件关系图

```
stock_selection.ipynb (主控制脚本)
    ↓
fundamental_run_model.py (模型运行脚本)
    ↓
ml_model.py (机器学习模型库)
    ↓
data_processor/Step2_preprocess_fundmental_data.py (数据预处理)
    ↓
final_ratios.csv + sector*.xlsx (输入数据)
```

## 核心文件说明

### 1. stock_selection.ipynb
**功能**: 主控制脚本，协调整个股票选择流程
**作用**: 
- 批量运行所有行业的股票选择模型
- 汇总各行业的预测结果
- 生成最终的股票选择列表

### 2. fundamental_run_model.py
**功能**: 单个行业的模型运行脚本
**作用**:
- 加载行业特定的数据
- 设置滚动窗口参数
- 调用机器学习模型
- 保存预测结果

### 3. ml_model.py
**功能**: 机器学习模型库
**作用**:
- 实现多种机器学习算法
- 处理滚动窗口训练和测试
- 计算模型评估指标
- 生成股票选择策略

## 输入文件要求

### 必需文件

#### 1. final_ratios.csv
**来源**: `data_processor/Step2_preprocess_fundmental_data.py`的输出
**格式**: CSV文件
**必需列**:
- `date`: 交易日期 (YYYY-MM-DD格式)
- `gvkey`: 公司唯一标识符
- `tic`: 股票代码
- `gsector`: GICS行业分类
- `y_return`: 下一季度收益率（目标变量）
- 财务比率列: `EPS`, `BPS`, `DPS`, `cur_ratio`, `quick_ratio`, `cash_ratio`, `acc_rec_turnover`, `debt_ratio`, `debt_to_equity`, `pe`, `ps`, `pb`

#### 2. sector*.xlsx 文件
**来源**: `data_processor/Step2_preprocess_fundmental_data.py`的分行业输出
**格式**: Excel文件
**内容**: 特定行业的股票数据
**命名**: `sector10.xlsx`, `sector15.xlsx`, ..., `sector60.xlsx`

### 数据质量要求
- 无缺失值（NaN）
- 数值型数据
- 时间序列完整性
- 股票代码一致性

## 外部依赖库

### 核心库
```bash
pip install pandas numpy scikit-learn
```

### 机器学习库
```bash
pip install xgboost lightgbm
```

### 数据处理库
```bash
pip install openpyxl
```

### 完整依赖列表
```python
# 数据处理
pandas
numpy

# 机器学习
scikit-learn
xgboost
lightgbm

# 文件处理
openpyxl

# 系统
multiprocessing
time
os
```

## 内部函数详解

### ml_model.py 核心函数

#### 1. 数据准备函数

##### `prepare_rolling_train()`
```python
def prepare_rolling_train(df, features_column, label_column, date_column, 
                         unique_datetime, testing_windows, first_trade_date_index, 
                         max_rolling_window_index, current_index)
```
**功能**: 准备滚动窗口的训练数据
**输入**: 
- `df`: 数据框
- `features_column`: 特征列名列表
- `label_column`: 目标变量列名
- `date_column`: 日期列名
- `unique_datetime`: 唯一日期列表
- `testing_windows`: 测试窗口大小
- `first_trade_date_index`: 第一个交易日期索引
- `max_rolling_window_index`: 最大滚动窗口索引
- `current_index`: 当前索引
**输出**: `X_train`, `y_train`

##### `prepare_rolling_test()`
```python
def prepare_rolling_test(df, features_column, label_column, date_column, 
                        unique_datetime, testing_windows, first_trade_date_index, 
                        current_index)
```
**功能**: 准备滚动窗口的测试数据
**输出**: `X_test`, `y_test`

##### `prepare_trade_data()`
```python
def prepare_trade_data(df, features_column, label_column, date_column, 
                      tic_column, unique_datetime, testing_windows, 
                      first_trade_date_index, current_index)
```
**功能**: 准备交易数据
**输出**: `X_trade`, `y_trade`, `trade_tic`

#### 2. 模型训练函数

##### `train_linear_regression()`
```python
def train_linear_regression(X_train, y_train)
```
**功能**: 训练线性回归模型
**算法**: LinearRegression

##### `train_lasso()`
```python
def train_lasso(X_train, y_train)
```
**功能**: 训练Lasso回归模型
**算法**: Lasso with GridSearchCV
**参数优化**: alpha值网格搜索

##### `train_ridge()`
```python
def train_ridge(X_train, y_train)
```
**功能**: 训练Ridge回归模型
**算法**: Ridge with GridSearchCV

##### `train_random_forest()`
```python
def train_random_forest(X_train, y_train)
```
**功能**: 训练随机森林模型
**算法**: RandomForestRegressor with RandomizedSearchCV

##### `train_xgb()`
```python
def train_xgb(X_train, y_train)
```
**功能**: 训练XGBoost模型
**算法**: XGBRegressor with RandomizedSearchCV

##### `train_lightgbm()`
```python
def train_lightgbm(X_train, y_train)
```
**功能**: 训练LightGBM模型
**算法**: LGBMRegressor with RandomizedSearchCV

#### 3. 模型评估函数

##### `evaluate_model()`
```python
def evaluate_model(model, X_test, y_test)
```
**功能**: 评估模型性能
**指标**: R², MAE, MSE, RMSE

#### 4. 主要运行函数

##### `run_4model()`
```python
def run_4model(df, features_column, label_column, date_column, tic_column,
               unique_ticker, unique_datetime, trade_date, 
               first_trade_date_index=20, testing_windows=4, 
               max_rolling_window_index=44)
```
**功能**: 运行4个机器学习模型
**模型**: Linear Regression, Lasso, Random Forest, XGBoost
**流程**:
1. 滚动窗口数据准备
2. 模型训练
3. 预测生成
4. 结果汇总

#### 5. 股票选择函数

##### `pick_stocks_based_on_quantiles()`
```python
def pick_stocks_based_on_quantiles(df_predict_best)
```
**功能**: 基于分位数选择股票
**策略**: 选择预测收益率最高的25%股票

##### `long_only_strategy_daily()`
```python
def long_only_strategy_daily(df_predict_return, daily_return, trade_month_plus1, 
                           top_quantile_threshold=0.75)
```
**功能**: 日度多头策略
**策略**: 选择预测收益率最高的股票进行多头投资

##### `long_only_strategy_monthly()`
```python
def long_only_strategy_monthly(df_predict_return, tic_monthly_return, trade_month, 
                             top_quantile_threshold=0.7)
```
**功能**: 月度多头策略
**策略**: 月度调仓的多头投资策略

#### 6. 结果保存函数

##### `save_model_result()`
```python
def save_model_result(sector_result, sector_name)
```
**功能**: 保存模型结果
**输出文件**:
- `df_predict_best.csv`: 最佳预测结果
- `model_evaluation.csv`: 模型评估指标
- `portfolio_return.csv`: 投资组合收益率

### fundamental_run_model.py 函数

#### 主函数流程
1. **参数解析**: 解析命令行参数
2. **数据加载**: 加载基本面数据和行业数据
3. **特征选择**: 自动选择数值型特征列
4. **模型运行**: 调用`ml_model.run_4model()`
5. **结果保存**: 保存模型结果

### stock_selection.ipynb 流程

#### 主要步骤
1. **批量运行**: 循环运行所有行业的模型
2. **结果汇总**: 收集各行业的预测结果
3. **股票筛选**: 基于75%分位数筛选股票
4. **最终输出**: 生成`stock_selected.csv`

## 使用方法

### 1. 数据准备
```bash
# 运行数据预处理
cd data_processor
python Step2_preprocess_fundmental_data.py --Stock_Index_fundation_file "sp500_tickers_fundamental_quarterly_20250712.csv" --Stock_Index_price_file "sp500_tickers_daily_price_20250712.csv"
```

### 2. 单个行业模型运行
```bash
# 运行特定行业的模型
python fundamental_run_model.py -sector_name sector10 -fundamental final_ratios.csv -sector sector10.xlsx
```

### 3. 批量股票选择
```python
# 在Jupyter notebook中运行
# 运行stock_selection.ipynb中的所有单元格
```

## 输出文件说明

### 1. 模型结果文件
**位置**: `results/sector{sector_number}/`
**文件**:
- `df_predict_best.csv`: 最佳预测结果
- `model_evaluation.csv`: 模型评估指标
- `portfolio_return.csv`: 投资组合收益率

### 2. 最终股票选择文件
**文件**: `stock_selected.csv`
**列**:
- `gvkey`: 公司唯一标识符
- `predicted_return`: 预测收益率
- `trade_date`: 交易日期

## 参数配置

### 滚动窗口参数
- `first_trade_index`: 第一个交易日期索引 (默认: 20)
- `testing_window`: 测试窗口大小 (默认: 4)
- `max_rolling_window_index`: 最大滚动窗口索引 (默认: 44)

### 模型参数
- `label_column`: 目标变量列名 (默认: 'y_return')
- `date_column`: 日期列名 (默认: 'date')
- `tic_column`: 股票代码列名 (默认: 'tic')

### 股票选择参数
- `top_quantile_threshold`: 股票选择分位数阈值 (默认: 0.75)

## 性能优化

### 1. 并行处理
- 使用多进程处理不同行业
- 利用CPU多核性能

### 2. 内存优化
- 只加载必要的列
- 及时释放内存

### 3. 计算优化
- 使用高效的机器学习库
- 优化超参数搜索

## 注意事项

### 1. 数据质量
- 确保输入数据无缺失值
- 检查数据时间序列完整性
- 验证股票代码一致性

### 2. 模型选择
- 不同行业可能需要不同的模型
- 定期评估模型性能
- 考虑模型过拟合问题

### 3. 风险控制
- 股票选择结果仅供参考
- 建议结合其他分析方法
- 注意投资风险

### 4. 系统要求
- 确保有足够的内存
- 建议使用SSD硬盘
- 多核CPU可提高处理速度

## 故障排除

### 常见问题
1. **内存不足**: 减少同时处理的行业数量
2. **文件路径错误**: 检查输入文件路径
3. **依赖库缺失**: 安装所需的Python库
4. **数据格式错误**: 检查CSV/Excel文件格式

### 调试建议
1. 先运行单个行业测试
2. 检查中间输出文件
3. 查看错误日志
4. 验证数据质量 