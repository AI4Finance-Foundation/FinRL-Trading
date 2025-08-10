import pandas as pd
import numpy as np
import traceback

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



from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
import os
import errno

from multiprocessing import cpu_count

n_cpus = cpu_count() - 1


def prepare_rolling_train(df,features_column,label_column,date_column,unique_datetime,testing_windows,first_trade_date_index, max_rolling_window_index,current_index):
    if current_index <=max_rolling_window_index:
        train=df[(df[date_column] >= unique_datetime[0]) \
                & (df[date_column] < unique_datetime[current_index-testing_windows])]
    else:
        train=df[(df[date_column] >= unique_datetime[current_index-max_rolling_window_index]) \
                & (df[date_column] < unique_datetime[current_index-testing_windows])]
        
    X_train=train[features_column]
    y_train=train[label_column]
    return X_train,y_train

def prepare_rolling_test(df,features_column,label_column,date_column,unique_datetime,testing_windows,fist_trade_date_index, current_index):
    test=df[(df[date_column] >= unique_datetime[current_index-testing_windows]) \
            & (df[date_column] < unique_datetime[current_index])]
    X_test=test[features_column]
    y_test=test[label_column]
    return X_test,y_test

def prepare_trade_data(df,features_column,label_column,date_column,tic_column,unique_datetime,testing_windows,fist_trade_date_index, current_index):
    trade  = df[df[date_column] == unique_datetime[current_index]]
    X_trade = trade[features_column]
    y_trade = trade[label_column]
    trade_tic = trade[tic_column].values
    return X_trade,y_trade,trade_tic


def train_linear_regression(X_train,y_train):

    lr_regressor = LinearRegression()
    model = lr_regressor.fit(X_train, y_train)
    
    return model

def train_recursive_feature_elimination(X_train,y_train):

    lr_regressor = LinearRegression(random_state = 42)
    model = RFE(lr_regressor)
    
    return model

def train_lasso(X_train, y_train):
    # lasso_regressor = Lasso()
    # model = lasso_regressor.fit(X_train, y_train)

    lasso = Lasso(random_state = 42)
    # scoring_method = 'r2'
    # scoring_method = 'explained_variance'
    # scoring_method = 'neg_mean_absolute_error'
    scoring_method = 'neg_mean_squared_error'
    #scoring_method = 'neg_mean_squared_log_error'
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    # my_cv_lasso = TimeSeriesSplit(n_splits=3).split(X_train_advanced)
    lasso_regressor = GridSearchCV(lasso, parameters, scoring=scoring_method, cv=3)
    lasso_regressor.fit(X_train, y_train)

    model = lasso_regressor.best_estimator_
    return model

def train_ridge(X_train, y_train):
    # lasso_regressor = Lasso()
    # model = lasso_regressor.fit(X_train, y_train)

    ridge = Ridge(random_state = 42)
    # scoring_method = 'r2'
    # scoring_method = 'explained_variance'
    # scoring_method = 'neg_mean_absolute_error'
    scoring_method = 'neg_mean_squared_error'
    #scoring_method = 'neg_mean_squared_log_error'
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    # my_cv_lasso = TimeSeriesSplit(n_splits=3).split(X_train_advanced)
    ridge_regressor = GridSearchCV(ridge, parameters, scoring=scoring_method, cv=3)
    ridge_regressor.fit(X_train, y_train)

    model = ridge_regressor.best_estimator_
    return model

def train_random_forest(X_train, y_train):
    
    random_grid = {
                   #'max_depth': [10, 20, 40, 80, 100, None],
                   'max_features': ['sqrt'],
                   'min_samples_leaf': [0.05,0.1,0.2],
                   'min_samples_split': np.linspace(0.1, 1, 10, endpoint=True),
                   'n_estimators': [75,100,200]}
    # scoring_method = 'r2'
    # scoring_method = 'explained_variance'
    # scoring_method = 'neg_mean_absolute_error'
    scoring_method = 'neg_mean_squared_error'
    #scoring_method = 'neg_mean_squared_log_error'
    n_models = 1
    for key, val in random_grid.items():
        n_models *= len(val)
    n_jobs_per_model = min(max(1, n_cpus//n_models), n_cpus)
    # my_cv_rf = TimeSeriesSplit(n_splits=5).split(X_train_rf)
    rf = RandomForestRegressor(random_state=42, n_jobs= n_jobs_per_model)
    #RandomizedSearchCV
    #randomforest_regressor = RandomizedSearchCV(estimator=rf, 
    #                                            param_distributions=random_grid,
    #                                            n_iter = 100,
    #                                            cv=3, 
    #                                            n_jobs=-1, 
    #                                            scoring=scoring_method, 
    #                                            verbose=0)
    #GridSearchCV
    randomforest_regressor = GridSearchCV(estimator=rf, 
                                          param_grid=random_grid,
                                          cv=3, 
                                          n_jobs=n_cpus // n_jobs_per_model,
                                          scoring=scoring_method, 
                                          verbose=0)  
    
    randomforest_regressor.fit(X_train, y_train)
    #print(randomforest_regressor.best_params_ )
    model = randomforest_regressor.best_estimator_
    '''
    randomforest_regressor = RandomForestRegressor(random_state = 42,n_estimators = 400, max_features='auto')
    #randomforest_regressor = RandomForestRegressor(random_state = 42,n_estimators = 300)

    model = randomforest_regressor.fit(X_train, y_train)
    '''
    return model


def train_svm(X_train, y_train):
    svr = SVR(kernel = 'rbf')

    param_grid_svm = {'C':[0.001, 0.1, 1],'gamma': [1e-7,0.1]}
    #param_grid_svm = {'kernel': ('linear', 'rbf','poly'), 'C':[0.001, 0.01, 0.1, 1, 10],'gamma': [1e-7, 1e-4,0.001,0.1],'epsilon':[0.1,0.2,0.5,0.3]}

    # scoring_method = 'r2'
    # scoring_method = 'explained_variance'
    # scoring_method = 'neg_mean_absolute_error'
    scoring_method = 'neg_mean_squared_error'
    #scoring_method = 'neg_mean_squared_log_error'
    
    svm_regressor = GridSearchCV(estimator=svr, param_grid =param_grid_svm, cv=3, n_jobs=-1, scoring=scoring_method, verbose=0)
    
    svm_regressor.fit(X_train, y_train)
    model = svm_regressor.best_estimator_
    #estimator = svm_regressor.best_estimator_
    #selector = RFE(estimator, 5, step=1)
    #model = selector.fit(X, y)

    return model


def train_lightgbm(X_train, y_train):

    
    # model = gbm.fit(X_train, y_train)

    param_grid_gbm = {'learning_rate': [0.1,  0.01, 0.001], 'n_estimators': [100, 250, 500,1000]}
    n_models = 1
    for key, val in param_grid_gbm.items():
        n_models *= len(val)
    n_jobs_per_model = min(max(1, n_cpus//n_models), n_cpus)
    lightgbm = LGBMRegressor(random_state = 42, n_jobs=n_jobs_per_model)
    # scoring_method = 'r2'
    # scoring_method = 'explained_variance'
    # scoring_method = 'neg_mean_absolute_error'
    scoring_method = 'neg_mean_squared_error'
    #scoring_method = 'neg_mean_squared_log_error'
    gbm_regressor = GridSearchCV(estimator=lightgbm, param_grid=param_grid_gbm,
                                       cv=3, n_jobs=n_cpus // n_jobs_per_model, scoring=scoring_method, verbose=0)

    gbm_regressor.fit(X_train, y_train)
    model = gbm_regressor.best_estimator_
    '''
    
    gbm_regressor = GradientBoostingRegressor()
    model = gbm_regressor.fit(X_train, y_train)
    '''
    return model



def train_xgb(X_train, y_train):
    xgb = XGBRegressor(random_state = 42, n_jobs=10)

    param_grid_gbm = {'learning_rate': [0.1,  0.01, 0.001], 'n_estimators': [100, 250, 500,1000]}
    n_models = 1
    for key, val in param_grid_gbm.items():
        n_models *= len(val)
    n_jobs_per_model = min(max(1, n_cpus//n_models), n_cpus)
    xgb = XGBRegressor(random_state = 42, n_jobs=n_jobs_per_model)
    # scoring_method = 'r2'
    # scoring_method = 'explained_variance'
    # scoring_method = 'neg_mean_absolute_error'
    scoring_method = 'neg_mean_squared_error'
    #scoring_method = 'neg_mean_squared_log_error'
    xgb_regressor = GridSearchCV(estimator=xgb, param_grid=param_grid_gbm,
                                       cv=3, n_jobs=n_cpus // n_jobs_per_model, scoring=scoring_method, verbose=0)

    xgb_regressor.fit(X_train, y_train)
    model = xgb_regressor.best_estimator_
    '''
    
    gbm_regressor = GradientBoostingRegressor()
    model = gbm_regressor.fit(X_train, y_train)
    '''
    return model
def train_ada(X_train, y_train):
    ada = AdaBoostRegressor()

    # model = ada.fit(X_train, y_train)

    param_grid_ada = {'n_estimators': [20, 100],
                      'learning_rate': [0.01, 0.05, 1]}
    # scoring_method = 'r2'
    # scoring_method = 'explained_variance'
    # scoring_method = 'neg_mean_absolute_error'
    # scoring_method = 'neg_mean_squared_error'
    #scoring_method = 'neg_mean_squared_log_error'

    ada_regressor = GridSearchCV(estimator=ada, param_distributions=param_grid_ada,
                                       cv=3, n_jobs=-1, scoring=scoring_method, verbose=0)

    ada_regressor.fit(X_train, y_train)
    model = ada_regressor.best_estimator_
    '''
    ada_regressor = AdaBoostRegressor()
    model = ada_regressor.fit(X_train, y_train)
    '''
    return model


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
    # In versions of pandas (>= 2.0), the append() method has been deprecated 
    #tmp_table = tmp_table.append(pd.Series(y_trade_return, index=trade_tic), ignore_index=True)
    tmp_table = pd.concat([tmp_table, pd.Series(y_trade_return, index=trade_tic).to_frame().T], ignore_index=True)

    df_predict.loc[unique_datetime[current_index]][tmp_table.columns] = tmp_table.loc[0]


def run_4model(df,features_column, label_column,date_column,tic_column,
              unique_ticker, unique_datetime, trade_date, 
              first_trade_date_index=20,
              testing_windows=4,
              max_rolling_window_index=44):
    ## initialize all the result tables
    ## need date as index and unique tic name as columns
    df_predict_rf = pd.DataFrame(columns=unique_ticker, index=trade_date)
    df_predict_gbm = pd.DataFrame(columns=unique_ticker, index=trade_date)
    df_predict_xgb = pd.DataFrame(columns=unique_ticker, index=trade_date)
    df_predict_best = pd.DataFrame(columns=unique_ticker, index=trade_date)
    df_best_model_name = pd.DataFrame(columns=['model_name'], index=trade_date)
    evaluation_record = {}
    # first trade date is 1995-06-01
    # fist_trade_date_index = 20
    # testing_windows = 6
    import re
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    for i in range(first_trade_date_index, len(unique_datetime)):
        try:
            # prepare training data
            X_train, y_train = prepare_rolling_train(df, 
                                                     features_column,
                                                     label_column,
                                                     date_column, 
                                                     unique_datetime, 
                                                     testing_windows, 
                                                     first_trade_date_index, 
                                                     max_rolling_window_index,
                                                     current_index=i
                                                     )

            # prepare testing data
            X_test, y_test = prepare_rolling_test(df, 
                                                  features_column,
                                                  label_column,
                                                  date_column, 
                                                  unique_datetime, 
                                                  testing_windows, 
                                                  first_trade_date_index,
                                                  current_index=i)

            # prepare trade data
            X_trade, y_trade, trade_tic = prepare_trade_data(df,
                                                             features_column,
                                                             label_column,
                                                             date_column,
                                                             tic_column, 
                                                             unique_datetime, 
                                                             testing_windows, 
                                                             first_trade_date_index, 
                                                             current_index=i)

            # Training
         #   lr_model = train_linear_regression(X_train, y_train)
          
            t = time.perf_counter()
            xgb_model = train_xgb(X_train, y_train)
            print(f"xgb:{time.perf_counter() - t}s")
            t = time.perf_counter()
            gbm_model = train_lightgbm(X_train, y_train)
            print(f"gbm:{time.perf_counter() - t}s")
            t =time.perf_counter()
            rf_model = train_random_forest(X_train, y_train)
            print(f"rf:{time.perf_counter() - t}s")
         #   ridge_model = train_ridge(X_train, y_train)
            


            # Validation
            rf_eval = evaluate_model(rf_model, X_test, y_test)
            xgb_eval = evaluate_model(xgb_model, X_test, y_test)
            gbm_eval = evaluate_model(gbm_model, X_test ,y_test)
            # Trading
            
            y_trade_rf = rf_model.predict(X_trade)
            y_trade_xgb = xgb_model.predict(X_trade)
            y_trade_gbm = gbm_model.predict(X_trade)
            # Decide the best model
            eval_data = [
                         [rf_eval, y_trade_rf] ,
                         [xgb_eval, y_trade_xgb],
                         [gbm_eval, y_trade_gbm]
                                ]
            eval_table = pd.DataFrame(eval_data, columns=['model_eval', 'model_predict_return'],
                                              index=['rf', 'xgb', 'gbm'])        


            evaluation_record[unique_datetime[i]]=eval_table

            # lowest error score model
            y_trade_best = eval_table.model_predict_return.values[eval_table.model_eval == eval_table.model_eval.min()][0]
            best_model_name = eval_table.index.values[eval_table.model_eval == eval_table.model_eval.min()][0]

            # Highest Explained Variance
            # y_trade_best = eval_table.model_predict_return.values[eval_table.model_eval==eval_table.model_eval.max()][0]
            # best_model_name = eval_table.index.values[eval_table.model_eval==eval_table.model_eval.max()][0]

            df_best_model_name.loc[unique_datetime[i]] = best_model_name

            # Prepare Predicted Return table
            append_return_table(df_predict_rf, unique_datetime, y_trade_rf, trade_tic, current_index=i)
            append_return_table(df_predict_xgb, unique_datetime, y_trade_xgb, trade_tic, current_index=i)
            append_return_table(df_predict_gbm, unique_datetime, y_trade_gbm, trade_tic, current_index=i)
            append_return_table(df_predict_best, unique_datetime, y_trade_best, trade_tic, current_index=i)

            print('Trade Date: ', unique_datetime[i])

        except Exception:
            traceback.print_exc()
    df_evaluation = get_model_evaluation_table(evaluation_record,trade_date)
    return (
            df_predict_rf, 
            df_predict_gbm,
            df_predict_xgb,
            df_predict_best,
            df_best_model_name, 
            evaluation_record,
            df_evaluation)


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










