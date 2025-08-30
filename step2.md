# Step2: 基本面数据预处理分析文档

## 📊 **文件概述**

`Step2_preprocess_fundmental_data.ipynb` 是一个**基本面数据预处理**的Jupyter Notebook，用于将原始的财务数据和价格数据转换为机器学习可用的特征数据。这是整个投资组合管理系统的数据预处理核心模块。

## 🔄 **与其他项目文件的关系**

### **输入文件（来自其他项目文件）**
| 输入文件 | 来源项目 | 文件类型 | 描述 |
|---------|---------|---------|------|
| `sp500_tickers_fundamental_quarterly_20250712.csv` | 外部数据源 | CSV | S&P 500股票季度财务数据 |
| `sp500_tickers_daily_price_20250712.csv` | 外部数据源 | CSV | S&P 500股票日度价格数据 |

### **输出文件**
| 输出文件 | 文件类型 | 描述 |
|---------|---------|------|
| `final_ratios.csv` | CSV | 处理后的财务比率数据（74,017行 × 18列） |
| `sector*.xlsx` | Excel | 按行业分类的数据文件 |

## 🏗️ **主要功能模块**

### **1. 数据加载和基础处理**

#### **输入数据格式**：
- **Fundamentals CSV**: 包含季度财务报告数据，列包括 `tic`, `datadate`, `gvkey`, `prccq`, `adjex`, `epspxq`, `revtq`, `cshoq`, `atq`, `ltq` 等
- **Daily Price CSV**: 包含日度价格数据，列包括 `tic`, `datadate`, `gvkey` 等

#### **数据规模**：
- **Fundamentals**: 77,585行 × 654列
- **Daily Price**: 1,843个唯一ticker
- **最终输出**: 74,017行 × 18列

### **2. 核心处理步骤**

#### **步骤1：交易日期转换**
```python
# 将季度报告日期转换为交易日期
times = list(fund_df['datadate']) # 季度报告日期
for i in range(len(times)):
    quarter = (times[i] - int(times[i]/10000)*10000)
    if 1201 < quarter:
        times[i] = int(times[i]/10000 + 1)*10000 + 301
    if quarter <= 301:
        times[i] = int(times[i]/10000)*10000 + 301
    if 301 < quarter <= 601:
        times[i] = int(times[i]/10000)*10000 + 601
    if 601 < quarter <= 901:
        times[i] = int(times[i]/10000)*10000 + 901
    if 901 < quarter <= 1201:
        times[i] = int(times[i]/10000)*10000 + 1201
```

**功能**：将财务报告的季度日期转换为对应的交易日期，确保数据时间对齐

#### **步骤2：调整收盘价计算**
```python
fund_df['adj_close_q'] = fund_df.prccq/fund_df.adjex
```

**功能**：计算调整后的季度收盘价，消除股票分割、股息等因素影响

#### **步骤3：Ticker和GVKEY匹配**
```python
tic_to_gvkey = {}
for tic, df_ in df_daily_groups:
    tic_to_gvkey[tic] = df_.gvkey.iloc[0]

fund_df['gvkey'] = [tic_to_gvkey[x] for x in fund_df['tic']]
```

**功能**：建立ticker和gvkey的映射关系，确保财务数据和价格数据的一致性

#### **步骤4：下季度收益率计算**
```python
for tic,df in l_df:
    df.reset_index(inplace=True, drop=True)
    df.sort_values('date')
    # 预测下季度收益率
    df['y_return'] = np.log(df['adj_close_q'].shift(-1) / df['adj_close_q'])
```

**功能**：计算每个季度的下季度对数收益率作为机器学习模型的预测目标

### **3. 财务比率计算**

#### **估值比率**：
```python
fund_df['pe'] = fund_df.prccq / fund_df.epspxq  # 市盈率
fund_df['ps'] = fund_df.prccq / (fund_df.revtq/fund_df.cshoq)  # 市销率
fund_df['pb'] = fund_df.prccq / ((fund_df.atq-fund_df.ltq)/fund_df.cshoq)  # 市净率
```

#### **盈利能力比率**：
```python
# 营业利润率 (OPM) - 使用过去4个季度的滚动数据
OPM[i] = np.sum(fund_data['op_inc_q'].iloc[i-3:i])/np.sum(fund_data['rev_q'].iloc[i-3:i])

# 净利润率 (NPM)
NPM[i] = np.sum(fund_data['net_inc_q'].iloc[i-3:i])/np.sum(fund_data['rev_q'].iloc[i-3:i])

# 资产收益率 (ROA)
ROA[i] = np.sum(fund_data['net_inc_q'].iloc[i-3:i])/fund_data['tot_assets'].iloc[i]

# 权益收益率 (ROE)
ROE[i] = np.sum(fund_data['net_inc_q'].iloc[i-3:i])/fund_data['sh_equity'].iloc[i]
```

#### **流动性比率**：
```python
cur_ratio = (fund_data['cur_assets']/fund_data['cur_liabilities'])  # 流动比率
quick_ratio = ((fund_data['cash_eq'] + fund_data['receivables'])/fund_data['cur_liabilities'])  # 速动比率
cash_ratio = (fund_data['cash_eq']/fund_data['cur_liabilities'])  # 现金比率
```

#### **效率比率**：
```python
inv_turnover = np.sum(fund_data['cogs_q'].iloc[i-3:i])/fund_data['inventories'].iloc[i]  # 存货周转率
acc_rec_turnover = np.sum(fund_data['rev_q'].iloc[i-3:i])/fund_data['receivables'].iloc[i]  # 应收账款周转率
acc_pay_turnover = np.sum(fund_data['cogs_q'].iloc[i-3:i])/fund_data['payables'].iloc[i]  # 应付账款周转率
```

#### **杠杆比率**：
```python
debt_ratio = (fund_data['tot_liabilities']/fund_data['tot_assets'])  # 资产负债率
debt_to_equity = (fund_data['tot_liabilities']/fund_data['sh_equity'])  # 负债权益比
```

#### **每股指标**：
```python
EPS = fund_data['eps_incl_ex']  # 每股收益
BPS = (fund_data['com_eq']/fund_data['sh_outstanding'])  # 每股净资产
DPS = fund_data['div_per_sh']  # 每股股息
```

### **4. 数据清理和标准化**

#### **NaN和无穷值处理**：
```python
def handle_nan(df, features_column_financial):
    # 删除调整收盘价为0的记录
    df = df.drop(list(df[df.adj_close_q==0].index)).reset_index(drop=True)
    
    # 将收益率和财务比率中的NaN和无穷值替换为NaN
    df['y_return'].replace([np.nan,np.inf,-np.inf], np.nan, inplace=True)
    df[features_column_financial].replace([np.nan, np.inf, -np.inf], np.nan, inplace=True)
    
    # 删除包含NaN的行
    df.dropna(axis=0, inplace=True)
    return df
```

**功能**：
- 删除调整收盘价为0的记录
- 将NaN和无穷值替换为NaN
- 删除包含NaN的行，确保数据质量

### **5. 行业分类输出**

```python
for sec, df_ in list(final_ratios.groupby('gsector')):
    df_.to_excel(f"sector{int(sec)}.xlsx")
```

**功能**：按GICS行业分类将数据分别保存到不同的Excel文件中，便于后续按行业进行投资组合管理

## 📈 **最终输出数据结构**

### **主要特征列**：
| 列名 | 类型 | 描述 |
|------|------|------|
| `date` | string | 日期 (YYYY-MM-DD格式) |
| `gvkey` | int | 公司唯一标识符 |
| `tic` | string | 股票代码 |
| `gsector` | float | GICS行业分类 |
| `adj_close_q` | float | 调整后季度收盘价 |
| `y_return` | float | 下季度收益率（预测目标） |

### **财务比率特征**：
| 特征类别 | 特征名称 | 描述 |
|---------|---------|------|
| **盈利能力** | `OPM`, `NPM`, `ROA`, `ROE` | 营业利润率、净利润率、资产收益率、权益收益率 |
| **每股指标** | `EPS`, `BPS`, `DPS` | 每股收益、每股净资产、每股股息 |
| **流动性** | `cur_ratio`, `quick_ratio`, `cash_ratio` | 流动比率、速动比率、现金比率 |
| **效率** | `inv_turnover`, `acc_rec_turnover`, `acc_pay_turnover` | 存货周转率、应收账款周转率、应付账款周转率 |
| **杠杆** | `debt_ratio`, `debt_to_equity` | 资产负债率、负债权益比 |
| **估值** | `pe`, `ps`, `pb` | 市盈率、市销率、市净率 |

## 🔄 **数据流程总结**

```
原始财务数据 (77,585行 × 654列)
    ↓
原始价格数据 (1,843个ticker)
    ↓
交易日期转换
    ↓
Ticker-GVKEY匹配
    ↓
下季度收益率计算
    ↓
财务比率计算（18个特征）
    ↓
数据清理（删除NaN/无穷值）
    ↓
最终输出：final_ratios.csv (74,017行 × 18列)
    ↓
按行业分类：sector*.xlsx
```

## 🎯 **关键特性**

### **1. 时间序列处理**
- 使用滚动窗口计算财务比率（过去4个季度）
- 确保预测目标（下季度收益率）的时间对齐

### **2. 数据质量保证**
- 严格的NaN和无穷值处理
- 删除无效数据（如调整收盘价为0）
- 确保所有特征都是数值型

### **3. 特征工程**
- 18个标准化财务比率特征
- 涵盖盈利能力、流动性、效率、杠杆、估值等多个维度
- 为机器学习模型提供丰富的特征信息

### **4. 行业分类**
- 按GICS标准进行行业分类
- 便于后续按行业进行投资组合管理
- 支持行业轮动策略

## 📊 **数据统计**

### **处理前后对比**：
- **输入数据量**: 77,585行财务数据
- **输出数据量**: 74,017行特征数据
- **数据保留率**: 95.4%
- **特征数量**: 18个标准化财务比率

### **行业分布**：
- 按GICS行业标准分类
- 生成多个sector*.xlsx文件
- 每个文件包含该行业的所有股票数据

## 🔗 **后续应用**

这个预处理后的数据将用于：
1. **机器学习模型训练** - 为DRL模型提供特征数据
2. **投资组合优化** - 基于财务比率进行股票选择
3. **行业分析** - 按行业进行投资组合管理
4. **回测分析** - 验证投资策略的有效性

---

*本文档详细记录了Step2数据预处理模块的完整功能、实现逻辑和数据流程，为后续的模型训练和投资策略开发提供了重要的数据基础。* 