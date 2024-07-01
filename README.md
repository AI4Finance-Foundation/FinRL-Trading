<div align="center">
<img align="center" width="30%" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139">
</div>

# FinRL for trading

![Visitors](https://api.visitorbadge.io/api/VisitorHit?user=AI4Finance-Foundation&repo=FinRL-Trading&countColor=%23B17A)

Purpose: Based on FinRL (https://github.com/AI4Finance-Foundation/FinRL), develop an AI stock-selection and trading strategy using Supervised Learning (SL) and Deep Reinforcement Learning (DRL), and deploy it to an online trading platform.

### Phase I: Financial Data Processing and Technical Indicators

1. Download Dow-30, NASDAQ-100, or S&P 500 data, including Open, High, Low, Close prices, and Volume (OHLCV) and fundamental indicators.

2. Obtain technical indicators and perform feature engineering: technical indicators, such as MACD, RSI; and fundamental indicators, such as EPS, ROI, ROE, P/E, P/S.

### Phase II: Stock Selection and Portfolio Allocation with Backtesting Results

1. Stock Selection: Perform supervised machine learning using classic machine learning algorithms (LSTM, Random Forest, SVM, Linear Regression, Lasso, Ridge) to select stocks based on fundamental multi-factor data, and select the top 25% of stocks every quarter; 
• Reference paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3302088
• GitHub Code: https://github.com/AI4Finance-Foundation/Machine-Learning-for-Stock-Recommendation-IEEE-2018

2. Portfolio Allocation: Use DRL Ensemble strategy (including PPO, DDPG, A2C, SAC, and TD3) in FinRL for asset allocation of the selected stocks, trade with daily data, and output positions; 
• Reference paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996
• GitHub Code: https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/1-Introduction/FinRL_PortfolioAllocation_NeurIPS_2020.ipynb

### Phase III: Deploy a DRL agent to an online trading platform

1. Deployment: Deploy strategies to online trading platforms such as Alpaca for paper trading

• GitHub Code: https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/3-Practical/FinRL_PaperTrading_Demo.ipynb


**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
