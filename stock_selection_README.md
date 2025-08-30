# 股票选择脚本 (stock_selection.py)

## 概述

`stock_selection.py` 是将 `stock_selection.ipynb` Jupyter notebook 转换为命令行脚本的版本。该脚本可以批量运行所有行业的股票选择模型，并生成综合的股票选择结果。

## 功能特点

- **批量处理**: 自动处理所有11个行业 (sector10, sector15, ..., sector60)
- **灵活路径**: 支持自定义输入和输出目录
- **错误处理**: 包含文件存在性检查和异常处理
- **进度显示**: 详细的处理进度和统计信息
- **自动建目录**: 如果输出目录不存在会自动创建

## 使用方法

### 基本用法

```bash
# 使用默认输出目录
python stock_selection.py --data_path "./data_processor/my_outputs"

# 指定输出目录
python stock_selection.py --data_path "./data_processor/my_outputs" --output_path "./result"

# 查看帮助信息
python stock_selection.py --help
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--data_path` | str | 是 | - | 输入sector文件的目录路径 |
| `--output_path` | str | 否 | `./result` | 输出目录路径 |

### 示例用法

```bash
# 示例1: 使用my_outputs目录，输出到result目录
python stock_selection.py --data_path "./data_processor/my_outputs" --output_path "./result"

# 示例2: 使用其他数据目录
python stock_selection.py --data_path "./data_processor/outputs" --output_path "./my_results"

# 示例3: 使用绝对路径
python stock_selection.py --data_path "D:/data/sectors" --output_path "D:/results"
```

## 输入文件要求

### 必需文件

1. **final_ratios.csv**: 基本面数据文件
   - 位置: `{data_path}/final_ratios.csv`
   - 包含所有股票的基本面比率数据

2. **sector*.xlsx**: 行业数据文件
   - 位置: `{data_path}/sector10.xlsx`, `{data_path}/sector15.xlsx`, ..., `{data_path}/sector60.xlsx`
   - 每个文件包含对应行业的股票数据

### 文件结构示例

```
data_processor/my_outputs/
├── final_ratios.csv
├── sector10.xlsx
├── sector15.xlsx
├── sector20.xlsx
├── ...
└── sector60.xlsx
```

## 输出文件

### 主要输出

- **stock_selected.csv**: 股票选择结果文件
  - 位置: `{output_path}/stock_selected.csv`
  - 包含所有行业的股票选择结果

### 输出文件格式

| 列名 | 类型 | 说明 |
|------|------|------|
| `gvkey` | int | 股票唯一标识符 |
| `predicted_return` | float | 预测收益率 |
| `trade_date` | str | 交易日期 |

### 中间文件

脚本运行过程中会在 `results/` 目录下生成每个行业的模型结果：
```
results/
├── sector10/
│   └── df_predict_best.csv
├── sector15/
│   └── df_predict_best.csv
├── ...
└── sector60/
    └── df_predict_best.csv
```

## 处理流程

1. **参数验证**: 检查输入参数和文件存在性
2. **目录创建**: 如果输出目录不存在则自动创建
3. **行业循环**: 对每个行业 (10, 15, 20, ..., 60) 执行：
   - 检查sector文件是否存在
   - 运行 `fundamental_run_model.py` 进行模型训练
   - 读取预测结果文件
   - 选择前25%的股票 (quantile 0.75)
   - 收集股票选择结果
4. **结果汇总**: 将所有行业的结果合并
5. **保存输出**: 保存到指定的输出目录

## 错误处理

脚本包含以下错误处理机制：

- **文件不存在**: 如果输入文件不存在，会显示错误信息并退出
- **模型训练失败**: 如果某个行业的模型训练失败，会跳过该行业继续处理其他行业
- **结果文件缺失**: 如果预测结果文件不存在，会跳过该行业
- **数据处理异常**: 如果处理某个行业数据时出现异常，会记录错误并继续处理

## 依赖库

- `pandas`: 数据处理
- `argparse`: 命令行参数解析
- `pathlib`: 路径处理
- `os`: 操作系统接口
- `time`: 时间计算
- `sys`: 系统相关功能

## 注意事项

1. **Python版本**: 建议使用Python 3.7或更高版本
2. **依赖脚本**: 需要确保 `fundamental_run_model.py` 在同一目录下
3. **内存使用**: 处理大量数据时可能需要较多内存
4. **运行时间**: 完整运行所有行业可能需要较长时间
5. **文件权限**: 确保有足够的权限创建输出目录和文件

## 与Jupyter Notebook的对比

| 特性 | Jupyter Notebook | Python脚本 |
|------|------------------|------------|
| 交互性 | 高 | 低 |
| 自动化 | 低 | 高 |
| 参数化 | 硬编码 | 命令行参数 |
| 错误处理 | 手动 | 自动 |
| 批量运行 | 困难 | 简单 |
| 部署 | 复杂 | 简单 |

## 故障排除

### 常见问题

1. **"基本面数据文件不存在"**
   - 检查 `--data_path` 参数是否正确
   - 确认 `final_ratios.csv` 文件存在

2. **"sector文件不存在"**
   - 检查sector*.xlsx文件是否在指定目录中
   - 确认文件名格式正确

3. **"模型训练失败"**
   - 检查 `fundamental_run_model.py` 是否存在
   - 确认输入数据格式正确

4. **"预测结果文件不存在"**
   - 检查模型训练是否成功完成
   - 确认 `results/` 目录结构正确

### 调试建议

- 使用 `--help` 查看参数说明
- 检查输入文件的存在性和格式
- 查看详细的错误信息
- 确认所有依赖库已安装 