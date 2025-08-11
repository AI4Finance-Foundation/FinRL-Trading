#!/usr/bin/env python3
"""
测试Sector 0排除功能
"""

import pandas as pd
import os

def test_sector0_exclusion():
    """测试Sector 0排除功能"""
    
    print("=" * 60)
    print("测试Sector 0排除功能")
    print("=" * 60)
    
    # 检查test_outputs目录中的文件
    test_outputs_dir = "test_outputs"
    
    if not os.path.exists(test_outputs_dir):
        print(f"目录 {test_outputs_dir} 不存在")
        return
    
    # 列出所有sector文件
    sector_files = [f for f in os.listdir(test_outputs_dir) if f.startswith('sector') and f.endswith('.xlsx')]
    sector_files.sort()
    
    print(f"找到的sector文件: {sector_files}")
    
    # 检查是否有sector0.xlsx
    has_sector0 = 'sector0.xlsx' in sector_files
    print(f"包含sector0.xlsx: {has_sector0}")
    
    if has_sector0:
        print("\n当前输出包含Sector 0文件")
        print("更新后的脚本应该排除Sector 0")
    else:
        print("\n当前输出不包含Sector 0文件")
        print("这符合预期行为")
    
    # 检查其他sector文件
    other_sectors = [f for f in sector_files if f != 'sector0.xlsx']
    print(f"\n其他sector文件数量: {len(other_sectors)}")
    
    # 检查最终结果文件中的sector分布
    final_ratios_file = os.path.join(test_outputs_dir, 'final_ratios.csv')
    if os.path.exists(final_ratios_file):
        print(f"\n检查 {final_ratios_file} 中的sector分布...")
        df = pd.read_csv(final_ratios_file)
        
        sector_counts = df['gsector'].value_counts().sort_index()
        print("各sector的记录数:")
        for sector, count in sector_counts.items():
            print(f"  Sector {sector}: {count} 条记录")
        
        # 检查是否有sector 0
        has_sector0_in_data = 0 in df['gsector'].values
        print(f"\n数据中包含sector 0: {has_sector0_in_data}")
        
        if has_sector0_in_data:
            sector0_count = len(df[df['gsector'] == 0])
            print(f"Sector 0记录数: {sector0_count}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_sector0_exclusion() 