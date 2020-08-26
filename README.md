# Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy
This repository refers to the codes for ICAIF 2020 paper.


## Abstract
Stock trading strategies play a critical role in investment. However, it is challenging to design a profitable strategy in a complex and dynamic stock market. In this paper, we propose a deep ensemble reinforcement learning scheme that automatically learns a stock trading strategy by maximizing investment return. We train a deep reinforcement learning agent and obtain an ensemble trading strategy using the three actor-critic based algorithms: Proximal Policy Optimization (PPO), Advantage Actor Critic (A2C), and Deep Deterministic Policy Gradient (DDPG). The ensemble strategy inherits and integrates the best features of the three algorithms, thereby robustly adjusting to different market conditions. In order to avoid the large memory consumption in training networks with continuous action space, we employ a load-on-demand approach for processing very large data. We test our algorithms on the 30 Dow Jones stocks which have adequate liquidity. The performance of the trading agent with different reinforcement learning algorithms is evaluated and compared with both the Dow Jones Industrial Average index and the traditional min-variance portfolio allocation strategy. The proposed deep ensemble scheme is shown to outperform the three individual algorithms and the two baselines in terms of the risk-adjusted return measured by the Sharpe ratio.

<img src=figs/stock_trading.PNG width="500">

## Reference
Hongyang Yang, Xiao-Yang Liu, Shan Zhong, and Anwar Walid. 2020. Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. In ICAIF ’20: ACM International Conference on AI in Finance, Oct. 15–16, 2020, Manhattan, NY. ACM, New York, NY, USA.
## Medium Blog: https://medium.com/@ai4finance/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02

## Data

### Alternative Data & Preprocessing

The alternative data we use is from the [Microsoft Academic Graph database](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/), which is an open resource database with records of publications, including papers, journals, conferences, books etc. It provides the demographics of the publications like public date, citations, authors and affiliated institutes.

* First, we collect and combine every component stock from several AI related indexes such as Vanguard Information Technology Exchange-Traded Fund (ETF) and Global X Robotics & Artificial Intelligence Thematic ETF [12]. This gives us a basic company pool to select from.
* Then we get a list of companies including companies in non-US markets or do not have publications and patents
* We remove the stocks which are not in the US stock market and only keep the companies that have at least one publication or patent record during our backtesting period 2009-2018
* We obtain our investment universe that contains 115 publicly trade companies.
* Finally, we extracted 40 scholar-data-driven features including conference publications, journal publications, patents or books

### Financial Data

The daily price of stocks we use is pulled from [Compustat database via Wharton Research Data Services](https://wrds-web.wharton.upenn.edu/wrds/ds/compd/fundq).

## Predictive Model

### Ensemble

* PPO
* A2C
* DDPG

### Rolling window backtesting

## Performance
