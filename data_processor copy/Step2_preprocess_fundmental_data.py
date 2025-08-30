#!/usr/bin/env python3
"""
Step2: 基本面数据预处理
"""

import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def print_step(step_num, step_name, description=""):
    print(f"\n{'='*60}")
    print(f"步骤 {step_num}: {step_name}")
    print(f"{'='*60}")
    if description:
        print(f"描述: {description}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def print_complete(step_num, step_name, info=""):
    print(f"步骤 {step_num} 完成: {step_name}")
    if info:
        print(f"详细信息: {info}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='Step2: 基本面数据预处理')
    parser.add_argument('--fundamentals-csv', required=True, help='财务数据CSV文件路径')
    parser.add_argument('--daily-price-csv', required=True, help='价格数据CSV文件路径')
    parser.add_argument('--output-dir', default='./myoutput', help='输出目录路径 (默认: ./myoutput)')
    
    args = parser.parse_args()
    
    print("Step2: 基本面数据预处理")
    print("=" * 60)
    print(f"输入文件: {args.fundamentals_csv}, {args.daily_price_csv}")
    print(f"输出目录: {args.output_dir}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 步骤1: 数据加载
        print_step(1, "数据加载和基础处理", "加载财务数据和价格数据")
        fund_df = pd.read_csv(args.fundamentals_csv)
        df_daily_price = pd.read_csv(args.daily_price_csv)
        print_complete(1, "数据加载和基础处理", f"财务数据: {fund_df.shape}, 价格数据: {len(df_daily_price.tic.unique())}个ticker")
        print("DEBUG datadate样例与类型：", fund_df['datadate'].head().tolist(), type(fund_df['datadate'].iloc[0]))

        # 步骤2: 交易日期转换
        print_step(2, "交易日期转换", "将季度报告日期转换为交易日期")
# —— 新的、鲁棒的交易日期转换 —— 
# 1) 直接以 datetime 方式解析，兼容 int、'YYYYMMDD'、'YYYY-MM-DD' 等
        raw_dt = pd.to_datetime(fund_df['datadate'], errors='coerce', utc=False)

        before_drop = fund_df.shape[0]
        fund_df = fund_df[raw_dt.notna()].copy()
        raw_dt = raw_dt[raw_dt.notna()]

        # 2) 把报告日映射到 3/1, 6/1, 9/1, 12/1（同你原逻辑目标）
        #    做法：先取“报告所属季度”的季度末，再映射到本季度固定的月/日
        q_end = raw_dt.dt.to_period('Q').dt.to_timestamp(how='end')

        #q_end = raw_dt.dt.to_period('Q').dt.end_time.tz_localize(None)  # 季度末日期（Timestamp）
        # 根据季度末月份映射到固定的“交易基准日”
        map_month_to_md = {3: (3,1), 6: (6,1), 9: (9,1), 12: (12,1)}
        mapped_dates = []
        for dt in q_end:
            m = dt.month
            mm, dd = map_month_to_md[m]
            mapped_dates.append(pd.Timestamp(year=dt.year, month=mm, day=dd))

        fund_df['tradedate'] = pd.to_datetime(mapped_dates)

        after_drop = fund_df['tradedate'].notna().sum()
        print_complete(2, "交易日期转换", f"成功转换 {after_drop}/{before_drop} 行日期")

        # 将 datadate 安全转换为整数，保持与 notebook 相同的逻辑
        #fund_df['datadate_int'] = pd.to_numeric(fund_df['datadate'], errors='coerce').astype('Int64')
        #before_drop = fund_df.shape[0]
        #fund_df = fund_df[fund_df['datadate_int'].notna()].copy()

        #times = fund_df['datadate_int'].astype(int).tolist()
        # 将季度报告日期映射到 3/1 6/1 9/1 12/1（与 notebook 一致）
        #for i in range(len(times)):
        #    year = times[i] // 10000
        #    quarter = times[i] - year * 10000  # 得到 301 601 901 1201 等
        #    if quarter > 1201:
        #        times[i] = (year + 1) * 10000 + 301
        #    elif quarter <= 301:
        #        times[i] = year * 10000 + 301
        #    elif quarter <= 601:
        #        times[i] = year * 10000 + 601
        #    elif quarter <= 901:
        #        times[i] = year * 10000 + 901
       #     else:
      #          times[i] = year * 10000 + 1201

        # 格式化为字符串再解析为日期
        #times_str = [f"{t:08d}" for t in times]
        #tradedates = pd.to_datetime(times_str, format='%Y%m%d', errors='coerce')
        #fund_df['tradedate'] = tradedates

        #after_drop = fund_df['tradedate'].notna().sum()
        #fund_df.dropna(subset=['tradedate'], inplace=True)
        #print_complete(2, "交易日期转换", f"成功转换 {after_drop}/{before_drop} 行日期")
        # 步骤3: 调整收盘价计算
        print_step(3, "调整收盘价计算", "计算调整后的季度收盘价")
        fund_df['adj_close_q'] = fund_df.prccq / fund_df.adjex
        print_complete(3, "调整收盘价计算", "完成调整收盘价计算")
        
        # 步骤4: Ticker和GVKEY匹配
        print_step(4, "Ticker和GVKEY匹配", "建立ticker和gvkey的映射关系")
        tic_to_gvkey = {}
        for tic, df_ in df_daily_price.groupby('tic'):
            tic_to_gvkey[tic] = df_.gvkey.iloc[0]
        
        fund_df = fund_df[np.isin(fund_df.tic, list(tic_to_gvkey.keys()))]
        fund_df['gvkey'] = [tic_to_gvkey[x] for x in fund_df['tic']]
        print_complete(4, "Ticker和GVKEY匹配", f"匹配完成，剩余 {len(fund_df.gvkey.unique())} 个唯一gvkey")
        
        # 步骤5: 下季度收益率计算
        print_step(5, "下季度收益率计算", "计算每个季度的下季度对数收益率")
        fund_df['date'] = fund_df["tradedate"]
        fund_df['date'] = pd.to_datetime(fund_df['date'], format="%Y%m%d")
        fund_df.drop_duplicates(["date", "gvkey"], keep='last', inplace=True)
        
        l_df = list(fund_df.groupby('gvkey'))
        if not l_df:
            raise ValueError("在 gvkey 分组后没有任何数据可用于收益率计算，请检查前面步骤中的过滤逻辑（可能所有行都被过滤掉）。")

        for tic, df in l_df:
            df.reset_index(inplace=True, drop=True)
            df.sort_values('date')
            df['y_return'] = np.log(df['adj_close_q'].shift(-1) / df['adj_close_q'])
        
        fund_df = pd.concat([x[1] for x in l_df])
        print_complete(5, "下季度收益率计算", f"处理完成，数据形状: {fund_df.shape}")
        
        # 步骤6: 财务比率计算
        print_step(6, "财务比率计算", "计算18个标准化财务比率特征")
        
        # 重命名列名
        fund_df = fund_df.rename(columns={
            'oiadpq': 'op_inc_q', 'revtq': 'rev_q', 'niq': 'net_inc_q',
            'atq': 'tot_assets', 'teqq': 'sh_equity', 'epspiy': 'eps_incl_ex',
            'ceqq': 'com_eq', 'cshoq': 'sh_outstanding', 'dvpspq': 'div_per_sh',
            'actq': 'cur_assets', 'lctq': 'cur_liabilities', 'cheq': 'cash_eq',
            'rectq': 'receivables', 'cogsq': 'cogs_q', 'invtq': 'inventories',
            'apq': 'payables', 'dlttq': 'long_debt', 'dlcq': 'short_debt',
            'ltq': 'tot_liabilities',
        })
        
        # 计算估值比率
        fund_df['pe'] = fund_df.prccq / fund_df.epspxq
        # 市销率：使用已经重命名的季度收入列 rev_q
        fund_df['ps'] = fund_df['prccq'] / (fund_df['rev_q'] / fund_df['sh_outstanding'])
        fund_df['pb'] = fund_df.prccq / ((fund_df.tot_assets - fund_df.tot_liabilities) / fund_df.sh_outstanding)
        
        # 选择需要的列
        items = [
            'date', 'gvkey', 'tic', 'gsector', 'adj_close_q', 'y_return',
            'op_inc_q', 'rev_q', 'net_inc_q', 'tot_assets', 'sh_equity',
            'eps_incl_ex', 'com_eq', 'sh_outstanding', 'div_per_sh',
            'cur_assets', 'cur_liabilities', 'cash_eq', 'receivables',
            'cogs_q', 'inventories', 'payables', 'long_debt', 'short_debt',
            'tot_liabilities', 'pe', 'ps', 'pb'
        ]
        
        fund_data = fund_df[items]
        
        # 计算财务比率
        print("正在计算财务比率...")
        # —— 统一索引 —— 
        fund_data = fund_data.reset_index(drop=True).copy()
        idx = fund_data.index  # 共享的唯一 RangeIndex
        
        # 盈利能力比率
        OPM = pd.Series(np.empty(fund_data.shape[0], dtype='float64'), name='OPM')
        NPM = pd.Series(np.empty(fund_data.shape[0], dtype='float64'), name='NPM')
        ROA = pd.Series(np.empty(fund_data.shape[0], dtype='float64'), name='ROA')
        ROE = pd.Series(np.empty(fund_data.shape[0], dtype='float64'), name='ROE')
        
        for i in range(0, fund_data.shape[0]):
            # 需要同一公司且有完整的前三个历史季度
            if i-3 < 0 or fund_data.iloc[i, 1] != fund_data.iloc[i-3, 1]:
                OPM[i] = NPM[i] = ROA[i] = ROE[i] = np.nan
            else:
                OPM[i] = np.sum(fund_data['op_inc_q'].iloc[i-3:i]) / np.sum(fund_data['rev_q'].iloc[i-3:i])
                NPM[i] = np.sum(fund_data['net_inc_q'].iloc[i-3:i]) / np.sum(fund_data['rev_q'].iloc[i-3:i])
                ROA[i] = np.sum(fund_data['net_inc_q'].iloc[i-3:i]) / fund_data['tot_assets'].iloc[i]
                ROE[i] = np.sum(fund_data['net_inc_q'].iloc[i-3:i]) / fund_data['sh_equity'].iloc[i]
        
        # 每股指标
        EPS = fund_data['eps_incl_ex'].to_frame('EPS')
        BPS = (fund_data['com_eq'] / fund_data['sh_outstanding']).to_frame('BPS')
        DPS = fund_data['div_per_sh'].to_frame('DPS')
        

        # 流动性比率
        cur_ratio = (fund_data['cur_assets'] / fund_data['cur_liabilities']).to_frame('cur_ratio')
        quick_ratio = ((fund_data['cash_eq'] + fund_data['receivables']) / fund_data['cur_liabilities']).to_frame('quick_ratio')
        cash_ratio = (fund_data['cash_eq'] / fund_data['cur_liabilities']).to_frame('cash_ratio')
        
        # 效率比率
        inv_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='inv_turnover')
        acc_rec_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='acc_rec_turnover')
        acc_pay_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='acc_pay_turnover')
        
        for i in range(0, fund_data.shape[0]):
            if i-3 < 0 or fund_data.iloc[i, 1] != fund_data.iloc[i-3, 1]:
                inv_turnover[i] = acc_rec_turnover[i] = acc_pay_turnover[i] = np.nan
            else:
                inv_turnover[i] = np.sum(fund_data['cogs_q'].iloc[i-3:i]) / fund_data['inventories'].iloc[i]
                acc_rec_turnover[i] = np.sum(fund_data['rev_q'].iloc[i-3:i]) / fund_data['receivables'].iloc[i]
                acc_pay_turnover[i] = np.sum(fund_data['cogs_q'].iloc[i-3:i]) / fund_data['payables'].iloc[i]
        
        # 杠杆比率
        debt_ratio = (fund_data['tot_liabilities'] / fund_data['tot_assets']).to_frame('debt_ratio')
        debt_to_equity = (fund_data['tot_liabilities'] / fund_data['sh_equity']).to_frame('debt_to_equity')
        # —— 统一这些块的索引到 RangeIndex —— 
        for df_ in [EPS, BPS, DPS, cur_ratio, quick_ratio, cash_ratio, debt_ratio, debt_to_equity]:
            df_.reset_index(drop=True, inplace=True)

        # 三个效率比率是 Series：绑定同一索引
        inv_turnover.index = idx
        acc_rec_turnover.index = idx
        acc_pay_turnover.index = idx

        # 合并所有比率
        _parts = [
            fund_data[['date', 'gvkey', 'tic', 'gsector', 'adj_close_q', 'y_return']],
            OPM.to_frame(), NPM.to_frame(), ROA.to_frame(), ROE.to_frame(),
            EPS, BPS, DPS,
            cur_ratio, quick_ratio, cash_ratio,
            inv_turnover.to_frame(), acc_rec_turnover.to_frame(), acc_pay_turnover.to_frame(),
            debt_ratio, debt_to_equity,
            fund_data[['pe', 'ps', 'pb']]
        ]
        assert all(getattr(p, 'index', pd.RangeIndex(len(p))).is_unique for p in _parts), "拼接块存在非唯一索引"
        ratios = pd.concat([p.reset_index(drop=True) for p in _parts], axis=1)  # ← 关键：统一索引后再拼
        
        print_complete(6, "财务比率计算", f"计算完成，数据形状: {ratios.shape}")
        
        # 步骤7: 数据清理
        print_step(7, "数据清理和标准化", "处理NaN和无穷值")
        
        features_column_financial = [
            'OPM', 'NPM', 'ROA', 'ROE', 'EPS', 'BPS', 'DPS', 'cur_ratio',
            'quick_ratio', 'cash_ratio', 'inv_turnover', 'acc_rec_turnover',
            'acc_pay_turnover', 'debt_ratio', 'debt_to_equity', 'pe', 'ps', 'pb'
        ]
        
        # 删除调整收盘价为0的记录
        ratios = ratios.drop(list(ratios[ratios.adj_close_q == 0].index)).reset_index(drop=True)
        
        # 处理NaN和无穷值
        ratios['y_return'] = pd.to_numeric(ratios['y_return'], errors='coerce')
        for col in features_column_financial:
            ratios[col] = pd.to_numeric(ratios[col], errors='coerce')
        
        ratios['y_return'].replace([np.nan, np.inf, -np.inf], np.nan, inplace=True)
        ratios[features_column_financial].replace([np.nan, np.inf, -np.inf], np.nan, inplace=True)
        
        # 删除包含NaN的行
        ratios.dropna(axis=0, inplace=True)
        ratios = ratios.reset_index(drop=True)
        
        # 转换日期格式
        ratios.date = ratios.date.apply(lambda x: x.strftime('%Y-%m-%d'))
        
        print_complete(7, "数据清理和标准化", f"清理完成，最终数据形状: {ratios.shape}")
        
        # 步骤8: 保存输出
        print_step(8, "保存输出文件", "保存处理后的数据和行业分类文件")
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 保存主要输出文件
        main_output_path = os.path.join(args.output_dir, 'final_ratios.csv')
        ratios.to_csv(main_output_path, index=False)
        print(f"主要数据已保存到: {main_output_path}")
        
        # 按行业分类保存
        sector_count = 0
        for sec, df_ in list(ratios.groupby('gsector')):
            sector_output_path = os.path.join(args.output_dir, f"sector{int(sec)}.xlsx")
            df_.to_excel(sector_output_path, index=False)
            sector_count += 1
            print(f"行业 {int(sec)} 数据已保存到: {sector_output_path} ({len(df_)}行)")
        
        print_complete(8, "保存输出文件", f"保存完成，主文件: final_ratios.csv, 行业文件: {sector_count}个")
        
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"最终输出文件: {main_output_path}")
        print(f"数据形状: {ratios.shape}")
        print(f"特征数量: {len(ratios.columns) - 6}个财务比率")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 