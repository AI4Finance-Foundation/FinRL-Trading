#!/usr/bin/env python3
"""
检查原始数据中gsector字段的值分布，了解为什么会有sector 0
"""

import pandas as pd
import numpy as np

def check_gsector_distribution():
    """检查gsector字段的值分布"""
    
    print("=" * 80)
    print("检查原始数据中gsector字段的值分布")
    print("=" * 80)
    
    # 读取最终结果文件
    try:
        df = pd.read_csv('test_outputs/final_ratios.csv')
        
        print(f"数据总行数: {len(df)}")
        print(f"数据总列数: {len(df.columns)}")
        
        # 检查gsector字段的唯一值
        unique_sectors = sorted(df['gsector'].unique())
        print(f"\n唯一的gsector值: {unique_sectors}")
        
        # 统计每个sector的记录数
        print(f"\n各sector的记录数:")
        sector_counts = df['gsector'].value_counts().sort_index()
        for sector, count in sector_counts.items():
            print(f"  Sector {sector}: {count} 条记录")
        
        # 检查sector 0的详细信息
        sector0_data = df[df['gsector'] == 0]
        print(f"\nSector 0 详细信息:")
        print(f"  记录数: {len(sector0_data)}")
        print(f"  唯一股票数: {len(sector0_data['tic'].unique())}")
        print(f"  股票代码: {sorted(sector0_data['tic'].unique())}")
        
        # 检查sector 0的日期范围
        sector0_data['date'] = pd.to_datetime(sector0_data['date'])
        print(f"  日期范围: {sector0_data['date'].min()} 到 {sector0_data['date'].max()}")
        
        # 检查是否有缺失的gsector值
        missing_gsector = df['gsector'].isna().sum()
        print(f"\n缺失的gsector值数量: {missing_gsector}")
        
        # 检查gsector的数据类型
        print(f"gsector数据类型: {df['gsector'].dtype}")
        
        # 检查gsector值的范围
        print(f"gsector值范围: {df['gsector'].min()} 到 {df['gsector'].max()}")
        
        # 检查是否有非标准的GICS sector值
        standard_gics_sectors = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        non_standard_sectors = [s for s in unique_sectors if s not in standard_gics_sectors]
        print(f"\n非标准GICS sector值: {non_standard_sectors}")
        
        if 0 in non_standard_sectors:
            print("  Sector 0 不是标准的GICS sector值！")
            print("  可能的原因:")
            print("    1. 数据缺失或错误")
            print("    2. 公司已退市或重组")
            print("    3. 数据源中的分类错误")
            print("    4. 临时或特殊分类")
        
    except Exception as e:
        print(f"读取文件时出错: {e}")

def check_original_data():
    """检查原始基本面数据中的gsector分布"""
    
    print("\n" + "=" * 80)
    print("检查原始基本面数据中的gsector分布")
    print("=" * 80)
    
    try:
        # 读取原始基本面数据（只读取必要的列以节省内存）
        print("正在读取原始基本面数据...")
        df_original = pd.read_csv('sp500_tickers_fundamental_quarterly_20250712.csv', 
                                 usecols=['gvkey', 'tic', 'gsector', 'datadate'])
        
        print(f"原始数据总行数: {len(df_original)}")
        
        # 检查原始数据中gsector的分布
        original_sector_counts = df_original['gsector'].value_counts().sort_index()
        print(f"\n原始数据中各sector的记录数:")
        for sector, count in original_sector_counts.items():
            print(f"  Sector {sector}: {count} 条记录")
        
        # 检查sector 0在原始数据中的情况
        sector0_original = df_original[df_original['gsector'] == 0]
        print(f"\n原始数据中Sector 0的情况:")
        print(f"  记录数: {len(sector0_original)}")
        print(f"  唯一股票数: {len(sector0_original['tic'].unique())}")
        print(f"  股票代码: {sorted(sector0_original['tic'].unique())}")
        
        # 检查这些股票在其他sector中是否也有记录
        sector0_tics = set(sector0_original['tic'].unique())
        other_sectors = df_original[df_original['gsector'] != 0]
        other_sector_tics = set(other_sectors['tic'].unique())
        
        overlap_tics = sector0_tics.intersection(other_sector_tics)
        print(f"\nSector 0的股票在其他sector中也有记录的数量: {len(overlap_tics)}")
        if overlap_tics:
            print(f"  重叠的股票: {sorted(overlap_tics)}")
        
    except Exception as e:
        print(f"读取原始数据时出错: {e}")

if __name__ == "__main__":
    check_gsector_distribution()
    check_original_data() 