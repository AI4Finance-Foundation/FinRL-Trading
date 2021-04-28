from __future__ import division, absolute_import, print_function
import numpy as np
import pandas as pd
import datetime
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split


class DataProcessor:

    def __init__(self,feature_engineer, initial_data, buffer_size=30,numerical_cols = ['date','open', 'high', 'low', 'close', 'volume', 'tic','day']):
        self.numerical_cols = numerical_cols
        self.fe = feature_engineer
        self.buffer_size = buffer_size
        initial_indices = initial_data.index.unique()
        initial_data.index = initial_data.date.factorize()[0]
        self.numerical_data_history = initial_data[self.numerical_cols]
        if len(initial_indices) > buffer_size:
            self.numerical_data_history = self.numerical_data_history.loc[initial_indices[-buffer_size:]]
    
    def _add_new(self,df):
        if len(self.numerical_data_history.index.unique()) >= self.buffer_size:
            self.numerical_data_history.drop(self.numerical_data_history.index[0],inplace=True)
        df.set_index(pd.Index(np.full((len(df),),self.numerical_data_history.index[-1]+1)),inplace=True)
        self.numerical_data_history = self.numerical_data_history.append(df)
        return self.numerical_data_history

    def process_data(self,numerical_df,sentiment_df):
        new_feature_df = self.compute_technical_indicators(numerical_df)
        new_df = new_feature_df.reset_index().merge(sentiment_df,on=['date','tic']).set_index('index')
        return new_df

    def compute_technical_indicators(self,numerical_df):
        full_df = self._add_new(numerical_df)
        feature_df = self.fe.preprocess_data(full_df)
        feature_df.index = feature_df.date.factorize()[0]
        new_feature_df = feature_df.loc[feature_df.index[-1]]
        return new_feature_df

    def save_to_database(self):
        pass



def get_initial_data(numerical_df,sentiment_df,use_turbulence=False):
    fe = FeatureEngineer(use_turbulence=use_turbulence)
    numerical_df = fe.preprocess_data(numerical_df)
    df = numerical_df.merge(sentiment_df,on=["date","tic"],how="left")
    df.fillna(0)
    return df

def generate_sentiment_scores(start_date,end_date,tickers=config.DOW_30_TICKER,time_fmt="%Y-%m-%d"):
    dates = pd.date_range(start_date,end_date).to_pydatetime()
    dates = np.array([datetime.datetime.strftime(r,time_fmt) for r in dates])
    data = np.array(np.meshgrid(dates,tickers)).T.reshape(-1,2)
    scores = np.random.uniform(low=-1.0,high=1.0,size=(len(data),1))
    df = pd.DataFrame(data,columns=['date','tic'])
    df['sentiment'] = scores
    return df

def test_process_data():
    start_date = '2020-11-01'
    end_date='2021-01-01'
    ticker_list=config.DOW_30_TICKER
    numerical_df = YahooDownloader(start_date=start_date,end_date=end_date,ticker_list=ticker_list).fetch_data()
    sentiment_df = generate_sentiment_scores(start_date,end_date)
    initial_data = get_initial_data(numerical_df,sentiment_df)
    trade_data = data_split(initial_data,start_date,'2020-12-01')
    numerical_feed_data = numerical_df[numerical_df.date > '2020-12-01']
    sentiment_feed_data = sentiment_df[sentiment_df.date > '2020-12-01']
    data_processor = DataProcessor(FeatureEngineer(),trade_data)
    for date in numerical_feed_data.date.unique():
        
        new_numerical = numerical_feed_data[numerical_feed_data.date==date]
        new_sentiment = sentiment_feed_data.loc[sentiment_feed_data.date==date]
        new_df=data_processor.process_data(new_numerical,new_sentiment)
        print(new_df)

if __name__ == "__main__":
    test_process_data()