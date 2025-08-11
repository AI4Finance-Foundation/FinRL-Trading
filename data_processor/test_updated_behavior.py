#!/usr/bin/env python3
"""
测试更新后的行为：主CSV包含所有数据，分Sector文件默认排除Sector 0
"""

import pandas as pd
import os

def test_updated_behavior():
    """测试更新后的行为"""
    
    print("=" * 80)
    print("测试更新后的行为")
    print("=" * 80)
    
    # 检查test_outputs目录
    test_outputs_dir = "test_outputs"
    
    if not os.path.exists(test_outputs_dir):
        print(f"目录 {test_outputs_dir} 不存在")
        return
    
    # 检查final_ratios.csv
    final_ratios_file = os.path.join(test_outputs_dir, 'final_ratios.csv')
    if os.path.exists(final_ratios_file):
        print("1. 检查final_ratios.csv（主CSV文件）...")
        df = pd.read_csv(final_ratios_file)
        
        print(f"   总记录数: {len(df)}")
        
        # 检查sector分布
        sector_counts = df['gsector'].value_counts().sort_index()
        print(f"   各sector记录数:")
        for sector, count in sector_counts.items():
            print(f"     Sector {sector}: {count} 条记录")
        
        # 检查是否包含sector 0
        has_sector0 = 0 in df['gsector'].values
        print(f"   包含Sector 0: {has_sector0}")
        
        if has_sector0:
            sector0_count = len(df[df['gsector'] == 0])
            print(f"   Sector 0记录数: {sector0_count}")
            print("   ✓ 主CSV文件包含所有数据（包括Sector 0）")
        else:
            print("   ✗ 主CSV文件不包含Sector 0")
    
    # 检查sector文件
    print("\n2. 检查分Sector Excel文件...")
    sector_files = [f for f in os.listdir(test_outputs_dir) if f.startswith('sector') and f.endswith('.xlsx')]
    sector_files.sort()
    
    print(f"   找到的sector文件: {sector_files}")
    
    # 检查是否有sector0.xlsx
    has_sector0_file = 'sector0.xlsx' in sector_files
    print(f"   包含sector0.xlsx: {has_sector0_file}")
    
    if has_sector0_file:
        print("   ✗ 当前包含Sector 0文件，更新后应该排除")
    else:
        print("   ✓ 当前不包含Sector 0文件，符合预期")
    
    # 检查其他sector文件
    other_sectors = [f for f in sector_files if f != 'sector0.xlsx']
    print(f"   其他sector文件数量: {len(other_sectors)}")
    
    print("\n" + "=" * 80)
    print("预期行为总结:")
    print("=" * 80)
    print("✓ 主CSV文件（final_ratios.csv）应该包含所有数据，包括Sector 0")
    print("✓ 分Sector Excel文件默认应该排除Sector 0")
    print("✓ 使用--include_sector0参数可以包含Sector 0在分Sector文件中")
    print("=" * 80)

if __name__ == "__main__":
    test_updated_behavior() 