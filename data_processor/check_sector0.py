#!/usr/bin/env python3
"""
检查sector0.xlsx文件内容，了解sector 0是什么类别
"""

import pandas as pd
import os

def check_sector0():
    """检查sector0.xlsx文件的内容"""
    
    sector0_file = "test_outputs/sector0.xlsx"
    
    if not os.path.exists(sector0_file):
        print(f"文件 {sector0_file} 不存在")
        return
    
    try:
        # 读取sector0.xlsx文件
        df = pd.read_excel(sector0_file)
        
        print("=" * 60)
        print("SECTOR 0 文件分析")
        print("=" * 60)
        
        print(f"文件大小: {os.path.getsize(sector0_file) / 1024:.1f} KB")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        print(f"\n前5行数据:")
        print(df.head().to_string())
        
        print(f"\n唯一股票代码 (tic):")
        unique_tics = df['tic'].unique()
        print(f"数量: {len(unique_tics)}")
        print(f"股票代码: {list(unique_tics)}")
        
        print(f"\n唯一GVKEY:")
        unique_gvkeys = df['gvkey'].unique()
        print(f"数量: {len(unique_gvkeys)}")
        print(f"GVKEY: {list(unique_gvkeys)}")
        
        print(f"\n日期范围:")
        df['date'] = pd.to_datetime(df['date'])
        print(f"最早日期: {df['date'].min()}")
        print(f"最晚日期: {df['date'].max()}")
        
        print(f"\n数据统计:")
        print(df.describe())
        
        # 检查是否有其他sector文件
        print(f"\n检查其他sector文件:")
        test_outputs_dir = "test_outputs"
        sector_files = [f for f in os.listdir(test_outputs_dir) if f.startswith('sector') and f.endswith('.xlsx')]
        sector_files.sort()
        
        print(f"找到的sector文件: {sector_files}")
        
        # 检查所有sector的股票分布
        print(f"\n各sector的股票分布:")
        for sector_file in sector_files:
            if sector_file != 'sector0.xlsx':
                try:
                    sector_df = pd.read_excel(os.path.join(test_outputs_dir, sector_file))
                    unique_tic_count = len(sector_df['tic'].unique())
                    print(f"{sector_file}: {unique_tic_count} 只股票")
                except Exception as e:
                    print(f"{sector_file}: 读取错误 - {e}")
        
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    check_sector0() 