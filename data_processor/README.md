# FinRL Live trading

## Step 1: get all historical S&P 500 components 
* Get [Historical Lists of S&P 500 components since 1996](https://github.com/fja05680/sp500) 
* Input: S&P 500 Historical Components & Changes(MM-DD-YYYY).csv
* Output: sp500_tickers.txt
* Preprocess and get all unique tickers, save it into a txt file for WRDS

## Step 2: get fundamental data from WRDS
* Go to WRDS-Fundamentals Quarterly, start from 1996/01 to the most current date and do the query
* Input: sp500_tickers.txt
* output: raw csv file contains all fundamental data for the tickers in sp500_tickers, about 200MB

## Step 3: get daily price from WRDS
* Go to WRDS-Security Daily, start from 1996/01/01 to the most current date and do the query
* Input: sp500_tickers.txt
* output: raw csv file contains all daily price for the tickers in sp500_tickers, about 1GB




**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
