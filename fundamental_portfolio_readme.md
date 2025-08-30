# Fundamental Portfolio Optimization

## 概述

`fundamental_portfolio.ipynb` 是一个基于基本面分析的股票投资组合优化工具。该脚本使用机器学习预测的股票收益率和现代投资组合理论来构建最优投资组合。

## 主要功能

1. **数据预处理**: 读取股票价格数据和选股结果
2. **收益率计算**: 计算选定股票的历史日收益率
3. **投资组合优化**: 使用PyPortfolioOpt库进行投资组合优化
4. **权重分配**: 生成三种不同策略的投资组合权重

## 依赖库

```python
# 核心数据处理库
import pandas as pd
import numpy as np

# 投资组合优化库
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import expected_returns
from pypfopt import objective_functions

# 时间处理
from datetime import datetime
from pandas.tseries.offsets import BDay

# 其他工具
import time
import pickle
import warnings
```

## 输入文件

1. **`sp500_price_19960101_20221021.csv`**: S&P 500股票历史价格数据
   - 包含字段: `gvkey`, `datadate`, `prccd`, `ajexdi`
   - 时间范围: 1996-01-01 到 2022-10-21

2. **`stock_selected.csv`**: 选股结果文件
   - 包含字段: `gvkey`, `predicted_return`, `trade_date`
   - 时间范围: 2018-03-01 之后

## 处理流程

### 1. 数据读取和预处理
- 读取股票价格数据并计算调整后价格
- 读取选股结果，筛选2018-03-01之后的数据
- 提取交易日期和选定的股票

### 2. 历史收益率计算
- 对每个交易期间，获取选定股票过去一年的历史价格数据
- 计算日收益率
- 生成收益率数据表

### 3. 投资组合优化
使用PyPortfolioOpt库进行投资组合优化，实现三种策略：

#### a) 均值-方差优化 (Mean-Variance Optimization)
- 使用预测收益率作为期望收益
- 使用历史收益率计算协方差矩阵
- 最大化夏普比率

#### b) 最小方差优化 (Minimum Variance Optimization)
- 最小化投资组合方差
- 不考虑期望收益

#### c) 等权重策略 (Equal Weight)
- 所有股票权重相等

### 4. 权重约束
- 单只股票最大权重: 5% (0.05)
- 权重下限: 0%

## 输出文件

脚本生成以下输出文件：

1. **`mean_weighted.xlsx`**: 均值-方差优化权重
   - 字段: `trade_date`, `gvkey`, `weights`, `predicted_return`

2. **`minimum_weighted.xlsx`**: 最小方差优化权重
   - 字段: `trade_date`, `gvkey`, `weights`, `predicted_return`

3. **`equally_weighted.xlsx`**: 等权重策略权重
   - 字段: `trade_date`, `gvkey`, `weights`, `predicted_return`

4. **中间文件** (可选保存):
   - `all_return_table.pickle`: 历史收益率数据
   - `all_stocks_info.pickle`: 股票信息数据

## 性能特点

- **处理时间**: 约1.2分钟计算历史收益率，5分钟内完成投资组合优化
- **股票数量**: 处理644只不同的股票
- **交易期间**: 19个交易期间
- **数据规模**: 每个期间约180只股票，一年历史数据

## 使用说明

1. 确保输入文件存在于正确路径
2. 运行所有单元格
3. 检查输出文件是否正确生成
4. 可根据需要调整权重约束和优化目标

## 注意事项

- 脚本会忽略警告信息以提高执行效率
- 使用pickle格式保存中间结果以提高后续运行速度
- 确保有足够的内存处理大量股票数据
- 建议在运行前备份重要数据

## 技术细节

- **协方差估计**: 使用样本协方差矩阵
- **优化算法**: 非凸目标函数优化
- **权重清理**: 自动清理和标准化权重
- **数据对齐**: 自动处理股票代码和日期对齐问题 