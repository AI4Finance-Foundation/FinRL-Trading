import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import expected_returns
from datetime import datetime
from pandas.tseries.offsets import BDay

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split
from finrl import config
import pickle

from rl_model import run_models

df_price = pd.read_csv("/home/wenbiaolin/20221117/sp500_price_19960101_20221021.csv")

df_price['adjcp'] = df_price['prccd'] / df_price['ajexdi']

df_price['date'] = df_price['datadate']
df_price['open'] = df_price['prcod']
df_price['close'] = df_price['prccd']
df_price['high'] = df_price['prchd']
df_price['low'] = df_price['prcld']
df_price['volume'] =df_price['cshtrd']

df = df_price[['date', 'open', 'close', 'high', 'low','adjcp','volume', 'gvkey']]

df['tic'] = df_price['gvkey']

df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['day'] = [x.weekday() for x in df['date']]
df.drop_duplicates(['gvkey', 'date'], inplace=True)
selected_stock = pd.read_csv("stock_selected_rf.csv")

trade_date=selected_stock.trade_date.unique()

with open('all_return_table.pickle', 'rb') as handle:
    all_return_table = pickle.load(handle)

with open('all_stocks_info.pickle', 'rb') as handle:
    all_stocks_info = pickle.load(handle)


df_dict = {'trade_date':[], 'gvkey':[], 'weights':[]}
testing_window = pd.Timedelta(np.timedelta64(1,'Y'))
max_rolling_window = pd.Timedelta(np.timedelta64(10, 'Y'))


for idx in range(1, len(trade_date)):
    p1_alldata=all_stocks_info[trade_date[idx-1]]
    p1_alldata=p1_alldata.sort_values('gvkey')
    p1_alldata = p1_alldata.reset_index()
    del p1_alldata['index']
    p1_stock = p1_alldata.gvkey

    earliest_date = pd.to_datetime(trade_date[idx-1]) - max_rolling_window

    df_ = df[df['tic'].isin(p1_stock) & (df['date'] >= earliest_date) & (df['date'] < trade_date[idx])]
    print(df_)
    fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

    df_ = fe.preprocess_data(df_)

    df_=df_.sort_values(['date','tic'],ignore_index=True)
    df_.index = df_.date.factorize()[0]

    cov_list = []
    return_list = []

# look back is one year
    lookback=252
    for i in range(lookback,len(df_.index.unique())):
        data_lookback = df_.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)

  
    df_cov = pd.DataFrame({'date':df_.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    df_ = df_.merge(df_cov, on='date')
    df_ = df_.sort_values(['date','tic']).reset_index(drop=True)

    stock_dimension = len(df_.tic.unique())
    state_space = stock_dimension
    env_kwargs = {
    "hmax": 100, 
    "initial_amount": 1000000, 
    "transaction_cost_pct": 0.001, 
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.INDICATORS, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4
    
    }

    
    a2c_model,ppo_model,ddpg_model,td3_model,sac_model,best_model = run_models(df_, "date", pd.to_datetime(trade_date[idx-1]), env_kwargs,testing_window, max_rolling_window)
    
    trade = data_split(df_, pd.to_datetime(trade_date[idx-1]), pd.to_datetime(trade_date[idx]))
    e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
    df_daily_return, df_actions = DRLAgent.DRL_prediction(
    model=a2c_model, environment=e_trade_gym
    )

    
    for i in range(len(df_actions)):
        for j in df_actions.columns:
            df_dict['trade_date'].append(df_actions.index[i])
            df_dict['gvkey'].append(j)
            df_dict['weights'].append(df_actions.loc[df_actions.index[i], j])


df_rl = pd.DataFrame(df_dict)
df_rl.to_csv("drl_weight.csv")