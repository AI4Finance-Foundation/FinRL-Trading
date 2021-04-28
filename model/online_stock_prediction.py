import pandas as pd
import numpy as np
import datetime
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.env.env_stocks import StockEnv
from finrl.env.env_onlinestocktrading import OnlineStockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline




class OnlineStockPrediction:

    def __init__(self, e_trade_gym, model):
        self.e_trade_gym = e_trade_gym
        self.env_trade, self.cur_obs = self.e_trade_gym.get_sb_env()
        self.model = model
        #self.env_trade.reset()
    

    # def _get_numerical_data(self,all=True,col_list=['date','open', 'high', 'low', 'close', 'volume', 'tic','day']):
    #     if not all:
    #         window = 30
    #         if len(self.e_trade_gym.df.index.unique() > window):
    #             indices = self.e_trade_gym.df.unique()[-window:]
    #             prev_numerical_df = self.e_trade_gym.df.loc[indices][col_list]
    #     else:
    #         prev_numerical_df = e_trade_gym.df[col_list]
    #     return prev_numerical_df

    # def process_data(self,numerical_df,sentiment_df):
    #     full_numerical_df = self.compute_technical_indicators(numerical_df)
    #     new_df = full_numerical_df.merge(sentiment_df,on=['date','tic'])
    #     self.add_data(new_df)
    #     return new_df

    # def compute_technical_indicators(self,numerical_df):
    #     prev_numerical_df = self._get_numerical_data(all=False)
    #     full_df = prev_numerical_df.append(numerical_df)
    #     full_df = self.feature_engineer.preprocess_data(full_df)
    #     #TODO Might need to just take the indices corresponding to the new numerical data
    #     return full_df

    def add_data(self,df):
        self.e_trade_gym._update_data(df)


    def predict(self):
        #print("CURRENT OBSERVATION:" , self.cur_obs)
        action, states = self.model.predict(self.cur_obs)
        next_obs, rewards, done, info = self.e_trade_gym.step(action.squeeze())
        #print("NEXT OBSERVATION:", next_obs)
        self.cur_obs = next_obs
        return action,states, next_obs, rewards

    def run(self):
        pass




def generate_sentiment_scores(start_date,end_date,tickers=config.DOW_30_TICKER,time_fmt="%Y-%m-%d"):
    dates = pd.date_range(start_date,end_date).to_pydatetime()
    dates = np.array([datetime.datetime.strftime(r,time_fmt) for r in dates])
    data = np.array(np.meshgrid(dates,tickers)).T.reshape(-1,2)
    scores = np.random.uniform(low=-1.0,high=1.0,size=(len(data),1))
    df = pd.DataFrame(data,columns=['date','tic'])
    df['sentiment'] = scores
    return df

def get_initial_data(numerical_df,sentiment_df,use_turbulence=False):
    fe = FeatureEngineer(use_turbulence=use_turbulence)
    numerical_df = fe.preprocess_data(numerical_df)
    df = numerical_df.merge(sentiment_df,on=["date","tic"],how="left")
    df.fillna(0)
    return df

def main():
    start_date = '2020-01-01'
    trade_start_date='2020-12-01'
    end_date='2021-01-01'
    ticker_list=config.DOW_30_TICKER
    numerical_df = YahooDownloader(start_date=start_date,end_date=end_date,ticker_list=ticker_list).fetch_data()
    sentiment_df = generate_sentiment_scores(start_date,end_date)
    initial_data = get_initial_data(numerical_df,sentiment_df)
    train_data = data_split(initial_data,start_date,trade_start_date)
    trade_data = data_split(initial_data,trade_start_date,end_date)
    indicator_list = config.TECHNICAL_INDICATORS_LIST + ['sentiment']
    stock_dimension = len(trade_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(indicator_list)*stock_dimension
    env_kwargs = {
    "hmax": 100, 
    "initial_amount": 1000000, 
    "buy_cost_pct": 0.001, 
    "sell_cost_pct": 0.001, 
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": indicator_list, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4,
    "print_verbosity":5
    }
    e_train_gym = StockTradingEnv(df = train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    # print(train_data.index)
    # print(trade_data.index)
    # print(trade_data.loc[0])
    e_trade_gym = OnlineStockTradingEnv(trade_data.loc[0], **env_kwargs)
    training_agent = DRLAgent(env=env_train)
    model_a2c = training_agent.get_model("a2c")
    # print(train_data.index)
    # print(trade_data.index)
    #trained_a2c = agent.train_model(model=model_a2c, tb_log_name='a2c',total_timesteps=10000)
    feature_engineer = FeatureEngineer()
    online_stock_pred = OnlineStockPrediction(e_trade_gym,model_a2c,feature_engineer)
    for i in range(1,trade_data.index.unique().max()):
        print(trade_data.loc[i])
        online_stock_pred.add_data(trade_data.loc[i])
        action,states, next_obs, rewards = online_stock_pred.predict()
        print("Action:" ,action)
        print("States: ", states)
        print("Next observation: ", next_obs)
        print("Rewards: ", rewards)

if __name__ == "__main__":
    main()