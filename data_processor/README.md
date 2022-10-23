# FinRL Live trading

# Step 1: Query data from WRDS
## 1.1 Get all historical S&P 500 components 
* Get [Historical Lists of S&P 500 components since 1996](https://github.com/fja05680/sp500) 
* Input: S&P 500 Historical Components & Changes(MM-DD-YYYY).csv
* Output: sp500_tickers.txt
* Preprocess and get all unique tickers, save it into a txt file for WRDS

## 1.2 Get fundamental data from WRDS
* Go to WRDS-Fundamentals Quarterly, start from 1996/01 to the most current date and do the query
* Input: sp500_tickers.txt
* output: raw csv file contains all fundamental data for the tickers in sp500_tickers, about 200MB

## 1.3 Get daily price from WRDS
* Go to WRDS-Security Daily, start from 1996/01/01 to the most current date and do the query
* Input: sp500_tickers.txt
* output: raw csv file contains all daily price for the tickers in sp500_tickers, about 1GB

# Step 2: Preprocess fundamental data
## 1.1 Use Trade date instead of quarterly report date
* We also extend the trade date by two months lag beyond the standard quarter end date in case some companies have a non-standard quarter end date, e.g. Apple released its earnings report on 2010/07/20 for the second quarter of year 2010. Thus for the quarter between 04/01 and 06/30, our trade date is adjusted to 09/01 (same method for other
three quarters).
## 1.2 Get next quarter's return
* Our goal is to predict S&P 500 forward quarter log-return; At a given time T of the financial horizon, the 1-quarter forward log-returns of a certain stock price S are defined as:


<img align="center" width="390" alt="image" src="https://user-images.githubusercontent.com/31713746/197394617-a99436da-d20a-42c2-a63b-6af9d1a999de.png">

## 1.3 Get next quarter's return
## 1.4 Calculate Financial Ratios
## 1.5 Split the financial ratios by sector



**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
