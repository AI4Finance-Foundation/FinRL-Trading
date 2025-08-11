#!/usr/bin/env python3
"""
深入分析Sector 0的产生原因
"""

import pandas as pd
import numpy as np

def analyze_sector0_issue():
    """分析Sector 0的产生原因"""
    
    print("=" * 80)
    print("深入分析Sector 0的产生原因")
    print("=" * 80)
    
    # 1. 检查原始数据中这些股票的情况
    print("1. 检查原始数据中Sector 0股票的情况...")
    
    # 读取原始数据中的相关列
    df_original = pd.read_csv('sp500_tickers_fundamental_quarterly_20250712.csv', 
                             usecols=['gvkey', 'tic', 'gsector', 'datadate', 'prccq', 'adjex'])
    
    # 获取Sector 0的股票列表
    df_final = pd.read_csv('test_outputs/final_ratios.csv')
    sector0_tics = df_final[df_final['gsector'] == 0]['tic'].unique()
    
    print(f"Sector 0中的股票: {sorted(sector0_tics)}")
    
    # 检查这些股票在原始数据中的gsector值
    sector0_in_original = df_original[df_original['tic'].isin(sector0_tics)]
    
    print(f"\n这些股票在原始数据中的记录数: {len(sector0_in_original)}")
    
    if len(sector0_in_original) > 0:
        print("原始数据中这些股票的gsector分布:")
        sector_dist = sector0_in_original['gsector'].value_counts().sort_index()
        for sector, count in sector_dist.items():
            print(f"  Sector {sector}: {count} 条记录")
        
        # 检查是否有缺失的gsector值
        missing_gsector = sector0_in_original['gsector'].isna().sum()
        print(f"缺失gsector值的记录数: {missing_gsector}")
        
        if missing_gsector > 0:
            print("发现缺失的gsector值！这可能是Sector 0产生的原因。")
            
            # 检查缺失gsector的股票
            missing_gsector_data = sector0_in_original[sector0_in_original['gsector'].isna()]
            print(f"缺失gsector的股票: {sorted(missing_gsector_data['tic'].unique())}")
    
    # 2. 检查数据处理过程中的问题
    print("\n2. 检查数据处理过程中的问题...")
    
    # 检查是否有gsector为NaN的记录在最终结果中变成了0
    print("检查最终结果中gsector为0的记录...")
    sector0_final = df_final[df_final['gsector'] == 0]
    
    # 检查这些记录的财务比率是否大部分为0
    financial_cols = ['EPS', 'BPS', 'DPS', 'cur_ratio', 'quick_ratio', 'cash_ratio', 
                     'acc_rec_turnover', 'debt_ratio', 'debt_to_equity', 'pe', 'ps', 'pb']
    
    zero_counts = {}
    for col in financial_cols:
        if col in sector0_final.columns:
            zero_count = (sector0_final[col] == 0).sum()
            zero_counts[col] = zero_count
    
    print("Sector 0中财务比率为0的记录数:")
    for col, count in zero_counts.items():
        percentage = (count / len(sector0_final)) * 100
        print(f"  {col}: {count}/{len(sector0_final)} ({percentage:.1f}%)")
    
    # 3. 检查是否是数据填充问题
    print("\n3. 检查数据填充问题...")
    
    # 检查原始数据中这些股票的prccq和adjex
    if len(sector0_in_original) > 0:
        print("检查原始数据中这些股票的关键字段:")
        print(f"  prccq为0的记录数: {(sector0_in_original['prccq'] == 0).sum()}")
        print(f"  adjex为0的记录数: {(sector0_in_original['adjex'] == 0).sum()}")
        print(f"  prccq为NaN的记录数: {sector0_in_original['prccq'].isna().sum()}")
        print(f"  adjex为NaN的记录数: {sector0_in_original['adjex'].isna().sum()}")
    
    # 4. 检查是否是数据类型转换问题
    print("\n4. 检查数据类型转换问题...")
    
    print(f"原始数据中gsector的数据类型: {df_original['gsector'].dtype}")
    print(f"最终数据中gsector的数据类型: {df_final['gsector'].dtype}")
    
    # 检查是否有NaN值在转换过程中变成了0
    original_nan_count = df_original['gsector'].isna().sum()
    print(f"原始数据中gsector的NaN值数量: {original_nan_count}")
    
    if original_nan_count > 0:
        print("原始数据中存在gsector的NaN值！")
        print("这可能是在数据处理过程中，NaN值被填充为0导致的。")
    
    # 5. 总结分析结果
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)
    
    print("Sector 0产生的最可能原因:")
    print("1. 原始数据中存在gsector字段的缺失值（NaN）")
    print("2. 在数据处理过程中，这些NaN值被填充或转换为0")
    print("3. 这些股票可能是:")
    print("   - 已退市的公司")
    print("   - 数据不完整的公司")
    print("   - 特殊类型的公司（如ETF、基金等）")
    print("   - 数据源中的分类错误")
    
    print("\n建议:")
    print("1. 在数据处理脚本中添加对gsector缺失值的特殊处理")
    print("2. 考虑将这些记录单独分类或排除")
    print("3. 检查原始数据源，确认这些股票的gsector信息")

if __name__ == "__main__":
    analyze_sector0_issue() 