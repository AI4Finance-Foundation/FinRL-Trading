# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models import *


def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    data = preprocess_data()
    data = add_turbulence(data)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63
    
    ## Ensemble Strategy
    run_ensemble_strategy(df=data, 
                          unique_trade_date= unique_trade_date,
                          rebalance_window = rebalance_window,
                          validation_window=validation_window)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
