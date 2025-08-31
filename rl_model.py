import pandas as pd
import numpy as np
import traceback

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = (DEVICE == "cuda")

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV,RandomizedSearchCV

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.preprocessor.preprocessors import data_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
import os
import errno

from multiprocessing import cpu_count

n_cpus = cpu_count() - 1

import numpy as np
import pandas as pd

# Try to import gymnasium instead of gym for compatibility
try:
    import gymnasium as gym
    from gymnasium.utils import seeding
    from gymnasium import spaces
except ImportError:
    import gym
    from gym.utils import seeding
    from gym import spaces

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

# ==== ADD：Fix for crash at end of vec env ====
def _safe_DRL_prediction(model, environment, deterministic=True):
    """
    Run a test episode and ALWAYS return (account_value_df, actions_df),
    even if the vec env ends early.
    """
    test_env, test_obs = environment.get_sb_env()
    test_env.reset()
    n_steps = len(environment.df.index.unique())
    max_steps = n_steps - 1

    account_memory = None
    actions_memory = None

    for i in range(n_steps):
        action, _ = model.predict(test_obs, deterministic=deterministic)
        test_obs, rewards, dones, info = test_env.step(action)

        # Fetch memories either at last step or when done happens early
        if (i == max_steps) or dones[0]:
            account_memory = test_env.env_method("save_asset_memory")
            actions_memory = test_env.env_method("save_action_memory")
            if dones[0]:
                print("hit end!")
            break

    # Fallback: if for any reason memories weren't fetched in the loop
    if account_memory is None:
        account_memory = test_env.env_method("save_asset_memory")
    if actions_memory is None:
        actions_memory = test_env.env_method("save_action_memory")

    # env_method returns list-of-envs; take the first
    return account_memory[0], actions_memory[0]

# Apply the patch
DRLAgent.DRL_prediction = staticmethod(_safe_DRL_prediction)

def prepare_rolling_train(df, date_column, testing_window, max_rolling_window, trade_date):
    print(trade_date-max_rolling_window, trade_date-testing_window)
    # ensure using correct column name - data_split expects 'date' column
    if 'datadate' in df.columns and 'date' not in df.columns:
        df_temp = df.rename(columns={'datadate': 'date'})
    else:
        df_temp = df
    train = data_split(df_temp, trade_date-max_rolling_window, trade_date-testing_window)
    #print(train)
    return train

def prepare_rolling_test(df, date_column, testing_window, max_rolling_window, trade_date):
    # ensure using correct column name - data_split expects 'date' column
    if 'datadate' in df.columns and 'date' not in df.columns:
        df_temp = df.rename(columns={'datadate': 'date'})
    else:
        df_temp = df
    test=data_split(df_temp, trade_date-testing_window, trade_date)
        
    X_test=test.reset_index()
    return test

def prepare_trade_data(df,features_column,label_column,date_column,tic_column,unique_datetime,testing_windows,fist_trade_date_index, current_index):
    trade  = df[df[date_column] == unique_datetime[current_index]]
    X_trade = trade[features_column]
    y_trade = trade[label_column]
    trade_tic = trade[tic_column].values
    return X_trade,y_trade,trade_tic


def train_a2c(agent,USE_GPU ):
    #add GPU support 
    #A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
    if USE_GPU:
        A2C_PARAMS = {"n_steps": 1024, "ent_coef": 0.005, "learning_rate": 0.0002, "device": DEVICE}
    else:
        A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
    model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)
    trained_a2c = agent.train_model(model=model_a2c, 
                                tb_log_name='a2c',
                                total_timesteps=50000)
    
    return trained_a2c

def train_ppo(agent,USE_GPU ):
    if USE_GPU:
        PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.005,
        "learning_rate": 0.0001,
        "batch_size": 1024,
        "device": DEVICE}
    else:

        PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.005,
        "learning_rate": 0.0001,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=80000)

    return trained_ppo

def train_ddpg(agent,USE_GPU ):
    if USE_GPU:
        DDPG_PARAMS = {"batch_size": 1024, "buffer_size": 100000, "learning_rate": 0.001, "device": DEVICE}
    else:
        DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}

    model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS) 

    trained_ddpg = agent.train_model(model=model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=50000)

    return trained_ddpg

def train_td3(agent):
    TD3_PARAMS = {"batch_size": 100, 
              "buffer_size": 1000000, 
              "learning_rate": 0.001}

    model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)

    trained_td3 = agent.train_model(model=model_td3, 
                             tb_log_name='td3',
                             total_timesteps=30000)

    return trained_td3

def train_sac(agent):
    SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0003,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

    model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)

    trained_sac = agent.train_model(model=model_sac, 
                             tb_log_name='sac',
                             total_timesteps=50000)
    return trained_sac

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error
    #from sklearn.metrics import mean_squared_log_error

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import r2_score
    y_predict = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_predict)
    

    mse = mean_squared_error(y_test, y_predict)
    #msle = mean_squared_log_error(y_test, y_predict)

    explained_variance = explained_variance_score(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    return mse


def append_return_table(df_predict, unique_datetime, y_trade_return, trade_tic, current_index):
    tmp_table = pd.DataFrame(columns=trade_tic)
    tmp_table = tmp_table.append(pd.Series(y_trade_return, index=trade_tic), ignore_index=True)
    df_predict.loc[unique_datetime[current_index]][tmp_table.columns] = tmp_table.loc[0]

# remove td3 and sac model
def run_models(df,date_column, trade_date, env_kwargs, 
              testing_window=4,
              max_rolling_window=44):
    print(f"=== run_models DEBUG ===")
    print(f"Input df columns: {list(df.columns)}")
    print(f"Input date_column parameter: {date_column}")
    print(f"Input df has 'date' column: {'date' in df.columns}")
    print(f"Input df has 'datadate' column: {'datadate' in df.columns}")
    
    ## initialize all the result tables
    ## need date as index and unique tic name as columns
    evaluation_record = {}
    # first trade date is 1995-06-01
    # fist_trade_date_index = 20
    # testing_windows = 6
    
    # make sure right DataFrame
    df_ = df.copy()
    #print(f"After copy - df_ columns: {list(df_.columns)}")
    
    X_train = prepare_rolling_train(df_, date_column, testing_window, max_rolling_window, trade_date)
    print(f"After prepare_rolling_train - X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'No shape'}")

    # prepare testing data
    X_test = prepare_rolling_test(df_, date_column, testing_window, max_rolling_window, trade_date)
    print(f"After prepare_rolling_test - X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'No shape'}")
    
    e_train_gym = StockPortfolioEnv(df = X_train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env = env_train)

    a2c_model = train_a2c(agent,USE_GPU )
    ppo_model = train_ppo(agent,USE_GPU )
    ddpg_model = train_ddpg(agent,USE_GPU )
    #td3_model = train_td3(agent)
    #sac_model = train_sac(agent)
    
    best_model = None
    max_return = -np.inf
    e_trade_gym = StockPortfolioEnv(df = X_test, **env_kwargs)

    df_daily_return, df_actions = DRLAgent.DRL_prediction(
    model=a2c_model, environment=e_trade_gym
)
    a2c_return =list((df_daily_return.daily_return+1).cumprod())[-1] 
    if a2c_return > max_return:
        max_return = a2c_return
        best_model = a2c_model
    
    df_daily_return, df_actions = DRLAgent.DRL_prediction(
    model=ppo_model, environment=e_trade_gym
)
    ppo_return =list((df_daily_return.daily_return+1).cumprod())[-1] 
    if ppo_return > max_return:
        max_return = ppo_return
        best_model = ppo_model

    df_daily_return, df_actions = DRLAgent.DRL_prediction(
    model=ddpg_model, environment=e_trade_gym
)
    ddpg_return =list((df_daily_return.daily_return+1).cumprod())[-1] 
    if ddpg_return > max_return:
        max_return = ddpg_return
        best_model = ddpg_model

#    df_daily_return, df_actions = DRLAgent.DRL_prediction(
#    model=ppo_model, environment=e_trade_gym
#)
    #td3_return =list((df_daily_return.daily_return+1).cumprod())[-1] 
    #if td3_return > max_return:
    #    max_return = td3_return
    #    best_model = td3_model
    
   # df_daily_return, df_actions = DRLAgent.DRL_prediction(
   # model=sac_model, environment=e_trade_gym
#)
    #sac_return =list((df_daily_return.daily_return+1).cumprod())[-1] 
   # if sac_return > max_return:
   #     max_return = sac_return
    #    best_model = sac_model
    
    return a2c_model,ppo_model,ddpg_model,best_model
def get_model_evaluation_table(evaluation_record,trade_date):
    evaluation_list = []
    for d in trade_date:
        try:
            evaluation_list.append(evaluation_record[d]['model_eval'].values)
        except:
            print('error')
    df_evaluation = pd.DataFrame(evaluation_list,columns = ['rf', 'xgb', 'gbm'])
    df_evaluation.index = trade_date
    return df_evaluation

def save_model_result(sector_result,sector_name):
    df_predict_rf = sector_result[0].astype(np.float64)
    df_predict_gbm = sector_result[1].astype(np.float64)
    df_predict_xgb = sector_result[2].astype(np.float64)
    df_predict_best = sector_result[3].astype(np.float64)

    df_best_model_name = sector_result[4]
    df_evaluation_score = sector_result[5]
    df_model_score = sector_result[6]
    


    filename = 'results/'+sector_name+'/'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    
    df_predict_rf.to_csv('results/'+sector_name+'/df_predict_rf.csv')
    df_predict_gbm.to_csv('results/'+sector_name+'/df_predict_gbm.csv')
    df_predict_xgb.to_csv('results/'+sector_name+'/df_predict_xgb.csv')
    df_predict_best.to_csv('results/'+sector_name+'/df_predict_best.csv')
    df_best_model_name.to_csv('results/'+sector_name+'/df_best_model_name.csv')
    #df_evaluation_score.to_csv('results/'+sector_name+'/df_evaluation_score.csv')
    df_model_score.to_csv('results/'+sector_name+'/df_model_score.csv')



def calculate_sector_daily_return(daily_price, unique_ticker,trade_date):
    daily_price_pivot = pd.pivot_table(daily_price, values='adj_price', index=['datadate'],
                       columns=['tic'], aggfunc=np.mean)
    daily_price_pivot=daily_price_pivot[unique_ticker]
    
    daily_return=daily_price_pivot.pct_change()
    daily_return = daily_return[daily_return.index>=trade_date[0]]
    return daily_return

def calculate_sector_quarterly_return(daily_price, unique_ticker,trade_date_plus1):
    daily_price_pivot = pd.pivot_table(daily_price, values='adj_price', index=['datadate'],
                       columns=['tic'], aggfunc=np.mean)
    daily_price_pivot=daily_price_pivot[unique_ticker]
    quarterly_price_pivot=daily_price_pivot.ix[trade_date_plus1]
    
    quarterly_return=quarterly_price_pivot.pct_change()
    quarterly_return = quarterly_return[quarterly_return.index>trade_date_plus1[0]]
    
    return quarterly_return

def pick_stocks_based_on_quantiles_old(df_predict_best):

    quantile_0_25 = {}
    quantile_25_50 = {}
    quantile_50_75 = {}
    quantile_75_100 = {}


    for i in range(df_predict_best.shape[0]):
        q_25=df_predict_best.iloc[i].quantile(0.25)
        q_50=df_predict_best.iloc[i].quantile(0.5)
        q_75=df_predict_best.iloc[i].quantile(0.75)
        q_100=df_predict_best.iloc[i].quantile(1)

        quantile_0_25[df_predict_best.index[i]] = df_predict_best.iloc[i][df_predict_best.iloc[i] <= q_25]
        quantile_25_50[df_predict_best.index[i]] = df_predict_best.iloc[i][(df_predict_best.iloc[i] > q_25) & \
                                                                             (df_predict_best.iloc[i] <= q_50)]
        quantile_50_75[df_predict_best.index[i]] = df_predict_best.iloc[i][(df_predict_best.iloc[i] > q_50) & \
                                                                               (df_predict_best.iloc[i] <= q_75)]
        quantile_75_100[df_predict_best.index[i]] = df_predict_best.iloc[i][(df_predict_best.iloc[i] > q_75)]
    return (quantile_0_25, quantile_25_50, quantile_50_75, quantile_75_100)        

def pick_stocks_based_on_quantiles(df_predict_best):

    quantile_0_30 = {}

    quantile_70_100 = {}


    for i in range(df_predict_best.shape[0]):
        q_30=df_predict_best.iloc[i].quantile(0.3)
        q_70=df_predict_best.iloc[i].quantile(0.7)

        quantile_0_30[df_predict_best.index[i]] = df_predict_best.iloc[i][df_predict_best.iloc[i] <= q_30]
                                                                             

        quantile_70_100[df_predict_best.index[i]] = df_predict_best.iloc[i][(df_predict_best.iloc[i] >= q_70)]
    return (quantile_0_30, quantile_70_100)   

def calculate_portfolio_return(daily_return,trade_date_plus1,long_dict,frequency_date):
    df_portfolio_return = pd.DataFrame(columns=['portfolio_return'])

    for i in range(len(trade_date_plus1) - 1):
        # for long only
        #equally weight
        #long_normalize_weight = 1/long_dict[trade_date_plus1[i]].shape[0]

        # map date and tic
        long_tic_return_daily = \
            daily_return[(daily_return.index >= trade_date_plus1[i]) &\
                         (daily_return.index < trade_date_plus1[i + 1])][long_dict[trade_date_plus1[i]].index]
        # return * weight
        long_daily_return = long_tic_return_daily 
        df_temp = long_daily_return.mean(axis=1)
        df_temp = pd.DataFrame(df_temp, columns=['daily_return'])
        df_portfolio_return = df_portfolio_return.append(df_temp)
    return df_portfolio_return    

def calculate_portfolio_quarterly_return(quarterly_return,trade_date_plus1,long_dict):
    df_portfolio_return = pd.DataFrame(columns=['portfolio_return'])

    for i in range(len(trade_date_plus1) - 1):
        # for long only
        #equally weight
        #long_normalize_weight = 1/long_dict[trade_date_plus1[i]].shape[0]

        # map date and tic
        long_tic_return = quarterly_return[quarterly_return.index == trade_date_plus1[i + 1]][long_dict[trade_date_plus1[i]].index]

        df_temp = long_tic_return.mean(axis=1)
        df_temp = pd.DataFrame(df_temp, columns=['portfolio_return'])
        df_portfolio_return = df_portfolio_return.append(df_temp)
    return df_portfolio_return    

def long_only_strategy_daily(df_predict_return, daily_return, trade_month_plus1, top_quantile_threshold=0.75):
    long_dict = {}
    for i in range(df_predict_return.shape[0]):
        top_q = df_predict_return.iloc[i].quantile(top_quantile_threshold)
        # low_q=df_predict_return.iloc[i].quantile(0.2)
        # Select all stocks
        # long_dict[df_predict_return.index[i]] = df_predict_return.iloc[i][~np.isnan(df_predict_return.iloc[i])]
        # Select Top 30% Stocks
        long_dict[df_predict_return.index[i]] = df_predict_return.iloc[i][df_predict_return.iloc[i] >= top_q]
        # short_dict[df_predict_return.index[i]] = df_predict_return.iloc[i][df_predict_return.iloc[i]<=low_q]

    df_portfolio_return_daily = pd.DataFrame(columns=['daily_return'])
    for i in range(len(trade_month_plus1) - 1):
        # for long only
        #equally weight
        long_normalize_weight = 1/long_dict[trade_month_plus1[i]].shape[0]
        
        # calculate weight based on predicted return
        #long_normalize_weight = \
        #long_dict[trade_month_plus1[i]] / sum(long_dict[trade_month_plus1[i]].values)
        # map date and tic
        long_tic_return_daily = \
        daily_return[(daily_return.index >= trade_month_plus1[i]) & (daily_return.index < trade_month_plus1[i + 1])][
            long_dict[trade_month_plus1[i]].index]
        # return * weight
        long_daily_return = long_tic_return_daily * long_normalize_weight
        df_temp = long_daily_return.sum(axis=1)
        df_temp = pd.DataFrame(df_temp, columns=['daily_return'])
        df_portfolio_return_daily = df_portfolio_return_daily.append(df_temp)

        # for short only
        # short_normalize_weight=short_dict[trade_month[i]]/sum(short_dict[trade_month[i]].values)
        # short_tic_return=tic_monthly_return[tic_monthly_return.index==trade_month[i]][short_dict[trade_month[i]].index]
        # short_return_table=short_tic_return
        # portfolio_return_dic[trade_month[i]] = long_return_table.values.sum() + short_return_table.values.sum()

    return df_portfolio_return_daily


def long_only_strategy_monthly(df_predict_return, tic_monthly_return, trade_month, top_quantile_threshold=0.7):
    long_dict = {}
    short_dict = {}
    for i in range(df_predict_return.shape[0]):
        top_q = df_predict_return.iloc[i].quantile(top_quantile_threshold)
        # low_q=df_predict_return.iloc[i].quantile(0.2)
        # Select all stocks
        # long_dict[df_predict_return.index[i]] = df_predict_return.iloc[i][~np.isnan(df_predict_return.iloc[i])]
        # Select Top 30% Stocks
        long_dict[df_predict_return.index[i]] = df_predict_return.iloc[i][df_predict_return.iloc[i] >= top_q]
        # short_dict[df_predict_return.index[i]] = df_predict_return.iloc[i][df_predict_return.iloc[i]<=low_q]

    portfolio_return_dic = {}
    for i in range(len(trade_month)):
        # for longX_train_rf only
        # calculate weight based on predicted return
        long_normalize_weight = long_dict[trade_month[i]] / sum(long_dict[trade_month[i]].values)
        # map date and tic
        long_tic_return = tic_monthly_return[tic_monthly_return.index == trade_month[i]][
            long_dict[trade_month[i]].index]
        # return * weight
        long_return_table = long_tic_return * long_normalize_weight
        portfolio_return_dic[trade_month[i]] = long_return_table.values.sum()

        # for short only
        # short_normalize_weight=short_dict[trade_month[i]]/sum(short_dict[trade_month[i]].values)
        # short_tic_return=tic_monthly_return[tic_monthly_return.index==trade_month[i]][short_dict[trade_month[i]].index]
        # short_return_table=short_tic_return
        # portfolio_return_dic[trade_month[i]] = long_return_table.values.sum() + short_return_table.values.sum()

    df_portfolio_return = pd.DataFrame.from_dict(portfolio_return_dic, orient='index')
    df_portfolio_return = df_portfolio_return.reset_index()
    df_portfolio_return.columns = ['trade_month', 'monthly_return']
    df_portfolio_return.index = df_portfolio_return.trade_month
    df_portfolio_return = df_portfolio_return['monthly_return']
    return df_portfolio_return





def plot_predict_return_distribution(df_predict_best,sector_name,out_path):
    import matplotlib.pyplot as plt

    for i in range(df_predict_best.shape[0]):
        fig=plt.figure(figsize=(8,5))
        df_predict_best.iloc[i].hist()
        plt.xlabel("predicted return",size=15)
        plt.ylabel("frequency",size=15)

        plt.title(sector_name+": trade date - "+str(df_predict_best.index[i]),size=15)
    plt.savefig(out_path+str(df_predict_best.index[i])+".png")










