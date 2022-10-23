import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
import traceback
import sys
sys.path.append('code')
import ml_model




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    #sector name
    parser.add_argument('-sector_name','--sector_name_input', type=str,  required=True,help='sector name: i.e. sector10')

    # file name
    parser.add_argument('-fundamental','--fundamental_input', type=str,  required=True,help='inputfile name for fundamental table')
    parser.add_argument('-sector','--sector_input', type=str,  required=True,help='inputfile name for individual sector')
    
    # rolling window variables
    parser.add_argument("-first_trade_index", default=20, type=int)
    parser.add_argument("-testing_window", default=4, type=int)
    
    # column name
    parser.add_argument("-label_column", default='y_return', type=str)
    parser.add_argument("-date_column", default='date', type=str)
    parser.add_argument("-tic_column", default='tic', type=str)
    parser.add_argument("-no_feature_column_names", default = ['gvkey', 'tic', 'datadate', 'rdq', 'datadate', 'fyearq', 'fqtr',
       'conm', 'datacqtr', 'datafqtr', 'gsector','y_return'], type=list,help='column names that are not fundamental features')

    

    args = parser.parse_args()
    #load fundamental table
    inputfile_fundamental = args.fundamental_input
    
    fundamental_total=pd.read_csv(inputfile_fundamental)
   # fundamental_total=fundamental_total[fundamental_total['tradedate'] < 20170901]
    #get all unique quarterly date
    # load sector data
    inputfile_sector = args.sector_input
  
    sector_data=pd.read_excel(inputfile_sector)
    unique_datetime = sorted(sector_data.date.unique())


    #get sector unique ticker
    unique_ticker=sorted(sector_data[args.tic_column].unique())

    #set rolling window
    # train: 4 years = 16 quarters
    # test: 1 year = 4 quarters
    # so first trade date = #20 quarter
    #first trade date is 1995-06-01
    first_trade_date_index=args.first_trade_index

    #testing window
    testing_windows = args.testing_window

    #get all backtesting period trade dates
    trade_date=unique_datetime[first_trade_date_index:]
    #variable column name
    label_column = args.label_column
    date_column = args.date_column
    tic_column = args.tic_column
    
    # features column: different base on sectors
    no_feature_column_names = args.no_feature_column_names
    features_column = [x for x in sector_data.columns.values if (x not in no_feature_column_names) and (np.issubdtype(sector_data[x].dtype, np.number) and(not np.any(np.isnan(sector_data[x]))))]
    
    #sector name
    sector_name = args.sector_name_input
    
    try:
        start = time.time()
        model_result=ml_model.run_4model(sector_data,
                                            features_column, 
                                            label_column, 
                                            date_column,
                                            tic_column,
                                            unique_ticker, 
                                            unique_datetime, 
                                            trade_date,
                                            first_trade_date_index,
                                            testing_windows)
        end = time.time()
        print('Time Spent: ',(end-start)/60,' minutes')
        ml_model.save_model_result(model_result,sector_name)

    except e:
        print(e)

    

# python3 fundamental_run_model.py -sector_name sector10 -fundamental Data/fundamental_final_table.xlsx -sector Data/1-focasting_data/sector10_clean.xlsx 
