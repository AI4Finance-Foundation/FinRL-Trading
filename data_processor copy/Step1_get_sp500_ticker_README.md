# Step1_get_sp500_ticker.py 使用说明

## 功能描述

这个脚本用于从S&P 500历史成分股数据中提取唯一的股票代码，生成可用于WRDS查询的文本文件。

## 使用方法

### 基本用法

```bash
python Step1_get_sp500_ticker.py --Stock_Index_His_file "S&P 500 Historical Components & Changes(08-12-2022).csv"
```

### 指定输出文件名

```bash
python Step1_get_sp500_ticker.py --Stock_Index_His_file "S&P 500 Historical Components & Changes(08-12-2022).csv" --output_filename "my_tickers"
```

## 参数说明

- `--Stock_Index_His_file`: 必需参数，指定S&P 500历史成分股CSV文件的路径
- `--output_filename`: 可选参数，指定输出文件的基础名称（默认为"sp500_tickers"）

## 输出文件

脚本会生成一个`.txt`文件，包含所有唯一的股票代码，每行一个代码。文件名格式为：`{output_filename}.txt`

## 示例

### 使用默认输出文件名
```bash
python Step1_get_sp500_ticker.py --Stock_Index_His_file "S&P 500 Historical Components & Changes(08-12-2022).csv"
```
输出文件：`sp500_tickers.txt`

### 使用自定义输出文件名
```bash
python Step1_get_sp500_ticker.py --Stock_Index_His_file "S&P 500 Historical Components & Changes(08-12-2022).csv" --output_filename "custom_tickers"
```
输出文件：`custom_tickers.txt`

## 处理流程

1. 读取CSV文件（包含日期和股票代码列表）
2. 将股票代码字符串转换为列表
3. 移除股票代码中的日期后缀（如SYMBOL-yyyymm → SYMBOL）
4. 合并所有日期的股票代码
5. 提取唯一值并排序
6. 保存到文本文件

## 预期输出

脚本会显示处理进度信息，包括：
- 数据时间范围
- 总记录数
- 总股票代码条目数
- 唯一股票代码数量
- 输出文件路径

## 注意事项

- 确保输入CSV文件包含`date`和`tickers`列
- 输出文件会覆盖同名文件
- 股票代码按字母顺序排序 