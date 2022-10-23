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
<div align="center">
<img align="center" width="390" alt="image" src="https://user-images.githubusercontent.com/31713746/197394617-a99436da-d20a-42c2-a63b-6af9d1a999de.png">
</div>

## 1.3 Calculate Financial Ratios (Welcome to add more)
* Profitability ratios: PE (Price–to-Earnings Ratio), PS (Price-to-Sales Ratio), PB (Price-to-Book Ratio), OPM (Operating Margin), NPM (Net Profit Margin), ROA (Return On Assets), ROE (Return on Equity), EPS (Earnings Per Share), BPS (Book Per Share), DPS (Dividend Per Share)
* Liquidity ratios: Current ratio, Quick ratio, Cash ratio
* Efficiency ratios: Inventory turnover ratio, Receivables turnover ratio, Payable turnover ratio, 
* Leverage ratios: Debt ratio, Debt to Equity ratio

## 1.4 Split the financial ratios by sector
* In order to build a sector-neutral portfolio, we split the dataset by the Global Industry Classification Standard (GICS) sectors (total 11 sectors). 
* 10-Energy, 15-Materials, 20-Industrials, 25-Consumer Discretionary, 30-Consumer Staples, 35-Health Care, 40-Financials, 45-Information Technology, 50-Communication Services, 55-Utilities, 60-Real Estate
* We handle missing data separately by sector: if one factor has more than 5% missing data, we delete this factor; if a certain stock generates the most missing data, we delete this stock.
* **Output**: final_ratios.csv and ratios by sector

**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
