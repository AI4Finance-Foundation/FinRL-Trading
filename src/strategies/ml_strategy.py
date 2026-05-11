"""
Machine Learning Strategy Module
===============================

Implements ML-based stock selection strategies:
- Supervised learning for stock selection
- Feature engineering
- Model training and prediction
- Sector-neutral portfolio construction
- Multiple weight allocation methods:
  * Equal weight: 等权重分配（默认）
  * Min variance: 最小方差权重分配（自动使用基本面数据中的 adj_close_q）

  Usage:
      # 使用等权重
      result = strategy.generate_weights(
          data_dict,
          prediction_mode='single',
          weight_method='equal'
      )
      
      # 使用最小方差权重（自动使用基本面数据中的 adj_close_q 列计算）
      data_dict = {
          'fundamentals': fundamentals_df  # 必须包含 adj_close_q 列
      }
      result = strategy.generate_weights(
          data_dict,
          prediction_mode='single',
          weight_method='min_variance',
          lookback_periods=8  # 回溯季度数（默认8，即2年）
      )
      
      # 也可以使用日度价格数据
      data_dict = {
          'fundamentals': fundamentals_df,
          'prices': prices_df  # 包含 ['date', 'tic', 'close']
      }
      result = strategy.generate_weights(
          data_dict,
          prediction_mode='single',
          weight_method='min_variance',
          lookback_periods=252  # 回溯交易日数
      )
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from src.data.data_fetcher import fetch_sp500_tickers, fetch_fundamental_data

logger = logging.getLogger(__name__)

# TODO: trade_date is not necessary to be the last quarter date, we should use the next quarter date instead of the last quarter date
class MLStockSelectionStrategy(BaseStrategy):
    """Machine learning based stock selection strategy."""

    def __init__(self, config: StrategyConfig):
        """
        Initialize ML strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _compute_min_variance_weights(self, 
                                     selected_gvkeys: List[str], 
                                     price_data: pd.DataFrame,
                                     lookback_periods: int = 8) -> pd.DataFrame:
        """
        计算最小方差权重。
        
        Args:
            selected_gvkeys: 选中的股票列表
            price_data: 价格数据，支持以下格式：
                - 日度数据：包含 ['date', 'tic'/'gvkey', 'close'] 列
                - 季度数据（基本面）：包含 ['datadate', 'gvkey'/'tic', 'adj_close_q'] 列
            lookback_periods: 回溯期数用于计算协方差矩阵
                - 如果是日度数据，建议设置为交易日数（如252）
                - 如果是季度数据，建议设置为季度数（如8，即2年）
            
        Returns:
            包含 ['gvkey', 'weight'] 的 DataFrame
        """
        try:
            # 确定股票标识列
            ticker_col = 'gvkey' if 'gvkey' in price_data.columns else 'tic'
            
            # 确定日期列和价格列
            if 'datadate' in price_data.columns and 'adj_close_q' in price_data.columns:
                # 季度基本面数据
                date_col = 'datadate'
                price_col = 'adj_close_q'
                self.logger.info("使用基本面数据中的季度价格（adj_close_q）计算最小方差权重")
            elif 'date' in price_data.columns:
                # 日度价格数据
                date_col = 'date'
                price_col = 'close' if 'close' in price_data.columns else 'adj_close'
                self.logger.info(f"使用日度价格数据（{price_col}）计算最小方差权重")
            else:
                self.logger.warning("价格数据格式不符合要求，回退到等权重")
                return self._compute_equal_weights(selected_gvkeys)
            
            # 筛选选中的股票
            selected_prices = price_data[price_data[ticker_col].isin(selected_gvkeys)].copy()
            
            if len(selected_prices) == 0:
                self.logger.warning("无价格数据，回退到等权重")
                return self._compute_equal_weights(selected_gvkeys)
            
            # 确保日期列是datetime类型
            selected_prices[date_col] = pd.to_datetime(selected_prices[date_col])
            selected_prices = selected_prices.sort_values(date_col)
            
            # 取最近的 lookback_periods 期
            # 对于季度数据，这意味着最近N个季度；对于日度数据，意味着最近N天
            unique_dates = selected_prices[date_col].unique()
            if len(unique_dates) > lookback_periods:
                cutoff_date = sorted(unique_dates)[-lookback_periods]
                selected_prices = selected_prices[selected_prices[date_col] >= cutoff_date]
            
            # 透视为宽表格式：date x ticker
            pivot_prices = selected_prices.pivot_table(
                index=date_col, 
                columns=ticker_col, 
                values=price_col, 
                aggfunc='last'
            )
            
            # 计算收益率
            returns = pivot_prices.pct_change().dropna()
            
            # 最小数据点要求：至少3个观测期（可以是3个季度或3个交易日）
            min_periods = 3
            if len(returns) < min_periods:
                self.logger.warning(f"收益率数据不足（{len(returns)}期），至少需要{min_periods}期，回退到等权重")
                return self._compute_equal_weights(selected_gvkeys)
            
            # 处理缺失值：只保留数据完整的股票
            valid_cols = returns.columns[returns.notna().all()]
            if len(valid_cols) == 0:
                self.logger.warning("无完整数据的股票，回退到等权重")
                return self._compute_equal_weights(selected_gvkeys)
            
            returns = returns[valid_cols]
            
            # 计算协方差矩阵
            cov_matrix = returns.cov().values
            n_assets = len(valid_cols)
            
            # 优化目标：最小化组合方差
            def portfolio_variance(weights):
                return weights.T @ cov_matrix @ weights
            
            # 约束：权重和为1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            
            # 边界：权重在 [0, 1] 之间（不允许做空）
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # 初始猜测：等权重
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # 优化求解
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                self.logger.warning(f"最小方差优化失败: {result.message}，回退到等权重")
                return self._compute_equal_weights(selected_gvkeys)
            
            # 构建权重 DataFrame
            weights_df = pd.DataFrame({
                'gvkey': valid_cols.tolist(),
                'weight': result.x
            })
            
            # 对于没有价格数据的股票，分配0权重
            missing_gvkeys = set(selected_gvkeys) - set(valid_cols)
            if missing_gvkeys:
                self.logger.info(f"以下股票无足够价格数据，权重为0: {missing_gvkeys}")
                missing_df = pd.DataFrame({
                    'gvkey': list(missing_gvkeys),
                    'weight': [0.0] * len(missing_gvkeys)
                })
                weights_df = pd.concat([weights_df, missing_df], ignore_index=True)
            
            # 重新归一化以确保和为1
            total = weights_df['weight'].sum()
            if total > 0:
                weights_df['weight'] = weights_df['weight'] / total
            
            self.logger.info(f"最小方差权重计算成功，组合方差: {result.fun:.6f}")
            return weights_df
            
        except Exception as e:
            self.logger.error(f"最小方差权重计算失败: {e}，回退到等权重")
            return self._compute_equal_weights(selected_gvkeys)
    
    def _compute_equal_weights(self, selected_gvkeys: List[str]) -> pd.DataFrame:
        """
        计算等权重。
        
        Args:
            selected_gvkeys: 选中的股票列表
            
        Returns:
            包含 ['gvkey', 'weight'] 的 DataFrame
        """
        n = len(selected_gvkeys)
        if n == 0:
            return pd.DataFrame(columns=['gvkey', 'weight'])
        
        weight = 1.0 / n
        return pd.DataFrame({
            'gvkey': selected_gvkeys,
            'weight': [weight] * n
        })
    
    def allocate_weights(self, 
                        selected_stocks: pd.DataFrame,
                        method: str = 'equal',
                        price_data: Optional[pd.DataFrame] = None,
                        fundamentals: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.DataFrame:
        """
        为选中的股票分配权重。
        
        Args:
            selected_stocks: 选中的股票 DataFrame，必须包含 'gvkey' 列，可能包含 'predicted_return' 等列
            method: 权重分配方法，可选 'equal'（等权重）或 'min_variance'（最小方差）
            price_data: 日度价格数据（用于 min_variance 方法），需包含 ['date', 'tic'/'gvkey', 'close']
            fundamentals: 基本面数据（用于 min_variance 方法，优先级高于 price_data），
                         需包含 ['datadate', 'gvkey'/'tic', 'adj_close_q']
            **kwargs: 其他参数
                - lookback_periods: 最小方差方法的回溯期数
                    * 使用季度数据时，默认8（2年）
                    * 使用日度数据时，默认252（1年）
        
        Returns:
            包含 ['gvkey', 'weight', ...] 的 DataFrame，保留 selected_stocks 中的其他列
        """
        if len(selected_stocks) == 0:
            return selected_stocks.copy()
        
        selected_gvkeys = selected_stocks['gvkey'].tolist()
        
        # 根据方法分配权重
        if method == 'min_variance':
            # 优先使用基本面数据中的季度价格
            if fundamentals is not None and len(fundamentals) > 0:
                if 'adj_close_q' in fundamentals.columns:
                    # 使用季度数据，默认回溯8个季度（2年）
                    lookback_periods = kwargs.get('lookback_periods', 8)
                    weights_df = self._compute_min_variance_weights(
                        selected_gvkeys, 
                        fundamentals, 
                        lookback_periods
                    )
                else:
                    self.logger.warning("基本面数据中缺少 adj_close_q 列，尝试使用 price_data")
                    if price_data is None or len(price_data) == 0:
                        self.logger.warning("无可用价格数据，回退到等权重")
                        weights_df = self._compute_equal_weights(selected_gvkeys)
                    else:
                        # 使用日度数据，默认回溯252天（1年）
                        lookback_periods = kwargs.get('lookback_periods', 252)
                        weights_df = self._compute_min_variance_weights(
                            selected_gvkeys, 
                            price_data, 
                            lookback_periods
                        )
            elif price_data is not None and len(price_data) > 0:
                # 使用日度数据，默认回溯252天（1年）
                lookback_periods = kwargs.get('lookback_periods', 252)
                weights_df = self._compute_min_variance_weights(
                    selected_gvkeys, 
                    price_data, 
                    lookback_periods
                )
            else:
                self.logger.warning("min_variance 方法需要价格数据（fundamentals或price_data），回退到等权重")
                weights_df = self._compute_equal_weights(selected_gvkeys)
        elif method == 'equal':
            weights_df = self._compute_equal_weights(selected_gvkeys)
        else:
            self.logger.warning(f"未知的权重分配方法: {method}，使用等权重")
            weights_df = self._compute_equal_weights(selected_gvkeys)
        
        # 合并权重到原始 DataFrame
        result = selected_stocks.copy()
        result = result.merge(weights_df[['gvkey', 'weight']], on='gvkey', how='left', suffixes=('_old', ''))
        
        # 如果有旧的 weight 列，删除它
        if 'weight_old' in result.columns:
            result = result.drop(columns=['weight_old'])
        
        # 填充缺失权重为0
        result['weight'] = result['weight'].fillna(0.0)
        
        return result


    def _build_candidate_models(self) -> Dict[str, Any]:
        """Build candidate models similar to ml_model.py (rf/gbm, optional xgb/lgbm)."""
        candidates: Dict[str, Any] = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),
        }

        # Optional LightGBM
        try:
            from lightgbm import LGBMRegressor  # type: ignore
            candidates['lgbm'] = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        except Exception:
            logger.warning("LightGBM not installed, skipping. Please install LightGBM to use this model.")
            pass

        # Optional XGBoost
        try:
            from xgboost import XGBRegressor  # type: ignore
            candidates['xgb'] = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=0
            )
        except Exception:
            logger.warning("XGBoost not installed, skipping. Please install XGBoost to use this model.")
            pass

        return candidates

    def _infer_price_schema(self, price_data: pd.DataFrame) -> Tuple[str, str, str]:
        """
        根据传入的日度价格数据推断(date_col, ticker_col, px_col)。
        优先使用 adj_close，其次 close/prccd。
        """
        df = price_data
        date_col = 'date' if 'date' in df.columns else ('datadate' if 'datadate' in df.columns else None)
        if date_col is None:
            raise ValueError("price_data 缺少日期列 ['date' 或 'datadate']")

        ticker_col = 'gvkey' if 'gvkey' in df.columns else ('tic' if 'tic' in df.columns else None)
        if ticker_col is None:
            raise ValueError("price_data 缺少股票列 ['gvkey' 或 'tic']")

        if 'adj_close' in df.columns:
            px_col = 'adj_close'
        elif 'close' in df.columns:
            px_col = 'close'
        elif 'prccd' in df.columns:
            px_col = 'prccd'
        else:
            raise ValueError("price_data 缺少价格列 ['adj_close' 或 'close' 或 'prccd']")

        return date_col, ticker_col, px_col

    def _adjust_predictions_by_same_day_gap(
        self,
        pred_df: pd.DataFrame,
        price_data: Optional[pd.DataFrame],
        execution_date: Optional[Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        当日确认模式：用执行日相对前一交易日的已实现收益，来修正 predicted_return。

        new_predicted = predicted_return_raw - realized_return_since_prev

        Args:
            pred_df: 含 ['gvkey','predicted_return'] 的 DataFrame
            price_data: 日度价格数据（需含日期、股票、价格列）
            execution_date: 下单日期（可为 str/datetime）；若为空则不调整

        Returns:
            (adjusted_pred_df, meta)
        """
        meta: Dict[str, Any] = {}
        if price_data is None or len(price_data) == 0:
            self.logger.info("当日模式：无 price_data，跳过调整")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        if execution_date is None:
            self.logger.info("当日模式：未提供 execution_date，跳过调整")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        try:
            date_col, ticker_col, px_col = self._infer_price_schema(price_data)
        except Exception as e:
            self.logger.warning(f"当日模式：价格数据结构无效，跳过调整: {e}")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        # 规范化日期
        prices = price_data[[date_col, ticker_col, px_col]].copy()
        prices[date_col] = pd.to_datetime(prices[date_col])
        exec_dt = pd.to_datetime(execution_date)

        # 找到执行日（或最近不晚于执行日的交易日）与其前一交易日
        all_trade_dates = np.array(sorted(prices[date_col].unique()))
        if len(all_trade_dates) < 2:
            self.logger.info("当日模式：价格日期不足，跳过调整")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        # 最近不晚于执行日的交易日索引
        idx = np.searchsorted(all_trade_dates, exec_dt, side='right') - 1
        if idx < 0:
            # 全部价格都晚于执行日
            self.logger.info("当日模式：执行日前无历史价格，跳过调整")
            meta['confirm_mode'] = 'none'
            return pred_df, meta
        exec_trade_dt = all_trade_dates[idx]
        prev_idx = max(idx - 1, -1)
        if prev_idx < 0:
            self.logger.info("当日模式：无上一个交易日，跳过调整")
            meta['confirm_mode'] = 'none'
            return pred_df, meta
        prev_trade_dt = all_trade_dates[prev_idx]

        # 仅取目标两日与候选股票
        gvkeys = pred_df['gvkey'].astype(str).unique().tolist()
        sub = prices[prices[ticker_col].astype(str).isin(gvkeys)]
        sub = sub[sub[date_col].isin([prev_trade_dt, exec_trade_dt])]

        if sub.empty:
            self.logger.info("当日模式：目标股票在两日内无价格，跳过调整")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        # 透视两日价格，计算当日相对前一交易日收益
        pivot = sub.pivot_table(index=date_col, columns=ticker_col, values=px_col, aggfunc='last')
        # 确保两行齐备
        missing_rows = [d for d in [prev_trade_dt, exec_trade_dt] if d not in pivot.index]
        if missing_rows:
            # 缺任一日则无法计算 gap，跳过
            self.logger.info(f"当日模式：缺少价格日期 {missing_rows}，跳过调整")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        try:
            prev_prices = pivot.loc[prev_trade_dt]
            exec_prices = pivot.loc[exec_trade_dt]
            realized = (exec_prices / prev_prices - 1.0).replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.info(f"当日模式：计算当日收益失败，跳过调整: {e}")
            meta['confirm_mode'] = 'none'
            return pred_df, meta

        # 合并回预测，并做差得到 gap
        out = pred_df.copy()
        out['predicted_return_raw'] = out['predicted_return']
        # map realized return by gvkey/tic
        realized_by_gvkey = realized
        if 'gvkey' not in pivot.columns.names:
            # columns 是 ticker_col；若 ticker_col 不是 gvkey，需要映射名称
            pass  # 我们直接用 ticker_col 名称
        # 无论 ticker_col 为何，都转成字典映射
        realized_map = {str(k): float(v) if pd.notna(v) else np.nan for k, v in realized.items()}
        out['realized_return_since_prev'] = out['gvkey'].astype(str).map(realized_map)
        out['realized_return_since_prev'] = out['realized_return_since_prev'].fillna(0.0)
        out['predicted_return'] = out['predicted_return_raw'] - out['realized_return_since_prev']

        meta.update({
            'confirm_mode': 'today',
            'execution_date_input': str(pd.to_datetime(execution_date).date()),
            'execution_trade_date': str(pd.to_datetime(exec_trade_dt).date()),
            'prev_trade_date': str(pd.to_datetime(prev_trade_dt).date()),
            'price_field_used': px_col,
            'n_adjusted': int(out['realized_return_since_prev'].ne(0.0).sum()),
        })

        self.logger.info(
            f"当日模式：以 {exec_trade_dt.date()} 对比 {prev_trade_dt.date()} 调整预测，修正 {meta['n_adjusted']} 只"
        )

        return out, meta

    def _prepare_supervised_dataset(self, fundamentals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare X, y, and date index using real fundamentals with forward returns.

        Expects 'y_return' column already computed by data_fetcher.
        Returns X, y, and a pd.Series of dates aligned with X/y (datetime64).
        """
        if 'y_return' not in fundamentals.columns:
            raise ValueError("'y_return' 缺失，请使用 data_fetcher 获取的真实基本面数据")

        # Select numeric feature columns (exclude ids and label/date)
        exclude_cols = {'gvkey', 'tic', 'gsector', 'datadate', 'y_return', 'prccd', 'ajexdi', 'adj_close', 'adj_close_q'}
        numeric_cols: List[str] = []
        for col in fundamentals.columns:
            if col in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(fundamentals[col]):
                numeric_cols.append(col)

        if not numeric_cols:
            raise ValueError("未找到可用的数值型特征列")

        df = fundamentals.copy()
        # Ensure datetime
        df['datadate'] = pd.to_datetime(df['datadate'])

        # Handle missing
        X = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        # Drop constant columns
        X = X.loc[:, X.nunique() > 1]

        y = df['y_return'].astype(float)
        dates = df['datadate']

        # Align by dropping rows with NaN in X only (ignore y_return nan check)
        valid_mask = ~X.isna().any(axis=1)

        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        dates = dates.loc[valid_mask]

        return X, y, dates

    def _rolling_train_single_date(self,
                                   fundamentals: pd.DataFrame,
                                   test_quarters: int = 4) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """单次切分（不滚动）在指定 trade_date 进行训练/验证/预测。

        约定：
        - trade_date 默认为 fundamentals 中最后一个季度日期；
        - 训练集：从最早季度起，到 trade_date 往前 test_quarters 的季度（不含该边界）；
        - 验证集：从 trade_date 往前 test_quarters 个季度至 trade_date 之前的所有季度；
        - 预测：trade_date 当季。

        返回 (pred_df, metadata)，其中 pred_df 至少包含 ['gvkey', 'predicted_return']。
        """
        X, y, dates = self._prepare_supervised_dataset(fundamentals)

        # Add helper index
        df_xy = pd.DataFrame({'datadate': dates.values})
        df_xy = df_xy.join(X.reset_index(drop=True))
        df_xy['y_return'] = y.values
        df_xy['row_idx'] = np.arange(len(df_xy))

        # Unique quarterly dates (sorted)
        unique_dates = sorted(pd.to_datetime(fundamentals['datadate'].dropna().unique()))
        if len(unique_dates) == 0:
            raise ValueError("无可用季度日期")

        # 使用传入数据中的最后一个日期作为 trade_date
        trade_date = unique_dates[-1]

        i = unique_dates.index(trade_date)
        # 需要至少 test_quarters 个验证窗口，且训练窗口至少包含一个季度
        if i - test_quarters < 1:
            raise ValueError("历史季度不足：需要至少 1 个训练季度与 test_quarters 个验证季度")

        # 固定窗口：全历史训练 + 最近 test_quarters 验证 + 当前季度预测
        train_start = unique_dates[0]
        train_end_exclusive = unique_dates[i - test_quarters]
        test_start = unique_dates[i - test_quarters]
        test_end_exclusive = unique_dates[i]

        # Build boolean masks on df_xy by date
        masks_date = pd.to_datetime(df_xy['datadate'])
        train_mask = (masks_date >= train_start) & (masks_date < train_end_exclusive)
        test_mask = (masks_date >= test_start) & (masks_date < test_end_exclusive)
        trade_mask = (masks_date == trade_date)

        # y_return may be missing for some rows (especially near the latest quarter).
        # Keep trade rows for inference, but restrict supervised train/test rows.
        y_valid_mask = df_xy['y_return'].notna()
        train_mask = train_mask & y_valid_mask
        test_mask = test_mask & y_valid_mask

        # 去掉"Unnamed: 0"这类列
        feature_cols = [col for col in X.columns if not col.startswith('Unnamed:')]
        X_train = df_xy.loc[train_mask, feature_cols]
        y_train = df_xy.loc[train_mask, 'y_return']

        X_test = df_xy.loc[test_mask, feature_cols]
        y_test = df_xy.loc[test_mask, 'y_return']

        X_trade = df_xy.loc[trade_mask, feature_cols]
        # Align gvkey with the same valid rows used for X/y to avoid mismatch
        gvkey_series_all = fundamentals.loc[X.index, 'gvkey'].reset_index(drop=True)
        gvkey_trade = gvkey_series_all.loc[trade_mask].reset_index(drop=True)

        if len(X_train) < 10 or len(X_trade) == 0:
            raise ValueError("训练或交易样本不足")

        # Fit multiple candidates and choose best by validation MSE
        candidates = self._build_candidate_models()
        metrics: Dict[str, float] = {}
        preds_trade: Dict[str, np.ndarray] = {}

        # Fit a per-date scaler
        scaler = StandardScaler()
        X_train_scaled = np.asarray(scaler.fit_transform(X_train), dtype=float)
        X_test_scaled = np.asarray(scaler.transform(X_test), dtype=float) if len(X_test) > 0 else None
        X_trade_scaled = np.asarray(scaler.transform(X_trade), dtype=float)

        best_name = None
        best_mse = np.inf
        best_model = None

        for name, model in candidates.items():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"X does not have valid feature names, but .* was fitted with feature names",
                        category=UserWarning,
                    )
                    model.fit(X_train_scaled, y_train)
                    if X_test_scaled is not None and len(y_test) > 0:
                        y_hat = model.predict(X_test_scaled)
                        mse = mean_squared_error(y_test, y_hat)
                    else:
                        # If no test window, evaluate on train (discouraged but fallback)
                        y_hat = model.predict(X_train_scaled)
                        mse = mean_squared_error(y_train, y_hat)
                    metrics[name] = mse

                    trade_pred = model.predict(X_trade_scaled)
                    preds_trade[name] = trade_pred

                    if mse < best_mse:
                        best_mse = mse
                        best_name = name
                        best_model = model
            except Exception as e:
                logger.warning(f"候选模型 {name} 训练失败: {e}")
                continue

        if best_model is None or best_name is None:
            raise RuntimeError("无可用候选模型完成训练")

        predicted_return = preds_trade[best_name]
        pred_df = pd.DataFrame({
            'gvkey': gvkey_trade.values,
            'predicted_return': predicted_return
        })

        meta = {
            'trade_date': trade_date,
            'test_window_quarters': test_quarters,
            'model_name': best_name,
            'model_mse': best_mse,
            'all_model_mse': metrics,
        }

        return pred_df, meta

    def _rolling_train_all_date(self,
                                         fundamentals: pd.DataFrame,
                                         train_quarters: int = 16,
                                         test_quarters: int = 4) -> pd.DataFrame:
        """按季度滚动：对每个 trade_date，调用 `_rolling_train_single_date`。

        窗口定义：对第 i 个季度（trade date）进行预测时，
        - 训练集：在窗口 [i-(train_quarters+test_quarters), i-test_quarters) 内的所有季度；
        - 验证集：窗口 [i-test_quarters, i) 的最近 `test_quarters` 个季度；
        - 预测：第 i 个季度。

        Args:
            fundamentals: 含有真实基本面与标签 `y_return` 的数据框
            train_quarters: 训练窗口跨度（季度数）
            test_quarters: 验证窗口跨度（季度数）

        Returns:
            拼接后的全部预测结果 DataFrame，至少包含列 ['date', 'gvkey', 'predicted_return']
        """
        df = fundamentals.copy()
        if 'datadate' not in df.columns:
            raise ValueError("fundamentals 缺少 datadate 列")
        if 'y_return' not in df.columns:
            raise ValueError("fundamentals 缺少 y_return 列，请先构造标签")

        df['datadate'] = pd.to_datetime(df['datadate'])
        unique_dates = sorted(pd.to_datetime(df['datadate'].dropna().unique()))
        if len(unique_dates) == 0:
            raise ValueError("无可用季度日期")

        # 需要：每个 trade_date 至少留出 train_quarters + test_quarters 个季度作为历史
        start_i = max(train_quarters + test_quarters, 1)
        if start_i >= len(unique_dates):
            raise ValueError("历史季度不足以进行滚动训练，请增大数据量或减小 test_quarters")

        all_preds: List[pd.DataFrame] = []
        all_metas: List[Dict[str, Any]] = []

        for i in range(start_i, len(unique_dates)):
            trade_date = unique_dates[i]
            try:
                # 构造窗口子集数据 [i-(train_quarters+test_quarters), i]
                window_start = unique_dates[i - (train_quarters + test_quarters)]
                sub_df = df[(df['datadate'] >= window_start) & (df['datadate'] <= trade_date)].copy()
                pred_df, meta = self._rolling_train_single_date(
                    fundamentals=sub_df,
                    test_quarters=test_quarters,
                )
            except Exception as e:
                logger.warning(f"trade_date={trade_date.date()} 滚动训练失败: {e}")
                continue

            if pred_df is None or len(pred_df) == 0:
                continue

            pred_df = pred_df.copy()
            pred_df['date'] = pd.to_datetime(trade_date)
            # 可选添加评估信息，便于后续分析
            pred_df['model_name'] = meta.get('model_name')
            pred_df['model_mse'] = meta.get('model_mse')

            # 统一列顺序
            cols = ['date', 'gvkey', 'predicted_return', 'model_name', 'model_mse']
            pred_df = pred_df[[c for c in cols if c in pred_df.columns]]

            all_preds.append(pred_df)
            all_metas.append(meta)

        if not all_preds:
            logger.warning("无任何 trade date 生成有效预测结果")
            return pd.DataFrame(columns=['date', 'gvkey', 'predicted_return'])

        result_df = pd.concat(all_preds, ignore_index=True)

        return result_df

    def generate_weights(self, data: Dict[str, pd.DataFrame],
                        **kwargs) -> StrategyResult:
        """
        Generate portfolio weights using ML predictions.

        Args:
            data: Dictionary containing fundamentals and price data
            **kwargs: Additional parameters

        Returns:
            StrategyResult with ML-based weights
        """
        self.logger.info("Generating ML-based weights")

        # 1) 获取真实基本面数据
        if 'fundamentals' not in data or len(data['fundamentals']) == 0:
            raise ValueError("必须在 data['fundamentals'] 中提供真实基本面数据（需包含 y_return）")
        fundamentals = data['fundamentals'].copy()

        if len(fundamentals) == 0:
            self.logger.warning("无可用基本面数据")
            return StrategyResult(
                strategy_name=self.config.name,
                weights=pd.DataFrame(columns=['gvkey', 'weight']),
                metadata={'error': 'no_fundamentals'}
            )

        # 2) 预测模式：single（默认）或 rolling（滚动所有日期）
        prediction_mode = str(kwargs.get('prediction_mode', 'single')).lower()
        test_quarters = int(kwargs.get('test_quarters', 4))
        train_quarters = int(kwargs.get('train_quarters', 16))

        # 提前获取日度价格数据（供当日模式使用）
        price_data = data.get('prices', None)

        try:
            if prediction_mode == 'rolling':
                preds_all = self._rolling_train_all_date(
                    fundamentals=fundamentals,
                    train_quarters=train_quarters,
                    test_quarters=test_quarters,
                )
                if preds_all is None or len(preds_all) == 0:
                    raise ValueError('rolling 模式未产生任何预测')
            else:
                # 单次（使用数据最后日期）
                pred_df, meta = self._rolling_train_single_date(
                    fundamentals=fundamentals,
                    test_quarters=test_quarters,
                )
                meta['mode'] = 'single'
                # 当日模式：在选股阈值之前，基于当日/最近交易日价格对 predicted_return 做 gap 调整
                confirm_mode = str(kwargs.get('confirm_mode', 'none')).lower()
                execution_date = kwargs.get('execution_date', None)
                if confirm_mode in ('today', 'same_day'):
                    try:
                        pred_df, confirm_meta = self._adjust_predictions_by_same_day_gap(
                            pred_df=pred_df,
                            price_data=price_data,
                            execution_date=execution_date,
                        )
                        meta.update(confirm_meta)
                    except Exception as e:
                        self.logger.warning(f"当日模式调整失败，将使用原始预测: {e}")
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            return StrategyResult(
                strategy_name=self.config.name,
                weights=pd.DataFrame(columns=['gvkey', 'weight']),
                metadata={'error': f'fit_failed: {e}'}
            )

        # 4) 选股与权重分配
        top_quantile = float(kwargs.get('top_quantile', 0.75))
        weight_method = str(kwargs.get('weight_method', 'equal')).lower()

        if prediction_mode == 'rolling':
            # 对每个 trade date 独立选股与组内归一化，输出所有日期
            grouped = preds_all.groupby('date')
            weights_list: List[pd.DataFrame] = []
            per_date_thresholds: Dict[pd.Timestamp, float] = {}

            for dt, g in grouped:
                if g.empty:
                    continue
                thr_raw = g['predicted_return'].quantile(top_quantile)
                thr = max(float(thr_raw), 0.0) if pd.notna(thr_raw) else float('nan')
                per_date_thresholds[dt] = thr
                sel = g[(g['predicted_return'] >= thr) & (g['predicted_return'] > 0)][['gvkey', 'predicted_return']].copy()
                if len(sel) == 0:
                    continue
                
                # 使用新的权重分配函数（优先使用基本面数据中的价格）
                sel = self.allocate_weights(
                    selected_stocks=sel,
                    method=weight_method,
                    fundamentals=fundamentals,
                    price_data=price_data,
                    **kwargs
                )
                sel['date'] = dt
                # 组内应用风控并归一
                sel = self.apply_risk_limits(sel[['gvkey', 'weight', 'predicted_return', 'date']])
                weights_list.append(sel)

            if not weights_list:
                self.logger.warning("rolling 模式下所有日期均无选股")
                return StrategyResult(
                    strategy_name=self.config.name,
                    weights=pd.DataFrame(columns=['gvkey', 'weight', 'predicted_return', 'date']),
                    metadata={'error': 'no_predictions_all_dates', 'mode': 'rolling', 'top_quantile': top_quantile}
                )

            weights_df = pd.concat(weights_list, ignore_index=True)
            meta_out = {
                'mode': 'rolling',
                'test_window_quarters': test_quarters,
                'n_trade_dates': int(weights_df['date'].nunique()),
                'top_quantile': top_quantile,
                'per_date_thresholds': {str(k.date() if hasattr(k, 'date') else k): v for k, v in per_date_thresholds.items()}
            }

            result = StrategyResult(
                strategy_name=self.config.name,
                weights=weights_df[['gvkey', 'weight', 'predicted_return', 'date']],
                metadata=meta_out
            )

            self.logger.info(f"生成 ML 滚动权重，共 {len(weights_df)} 条、覆盖 {meta_out['n_trade_dates']} 个日期")
            return result
        else:
            # 单一日期逻辑（保持原行为）
            if pred_df.empty:
                self.logger.warning("目标日期无可预测股票")
                return StrategyResult(
                    strategy_name=self.config.name,
                    weights=pd.DataFrame(columns=['gvkey', 'weight']),
                    metadata={'error': 'no_predictions', **meta}
                )

            threshold_raw = pred_df['predicted_return'].quantile(top_quantile)
            threshold = max(float(threshold_raw), 0.0) if pd.notna(threshold_raw) else float('nan')
            selected = pred_df[(pred_df['predicted_return'] >= threshold) & (pred_df['predicted_return'] > 0)][['gvkey', 'predicted_return']].copy()

            if len(selected) == 0:
                self.logger.warning("没有股票达到阈值")
                weights_df = pd.DataFrame(columns=['gvkey', 'weight', 'predicted_return'])
            else:
                # 使用新的权重分配函数（优先使用基本面数据中的价格）
                weights_df = self.allocate_weights(
                    selected_stocks=selected,
                    method=weight_method,
                    fundamentals=fundamentals,
                    price_data=price_data,
                    **kwargs
                )
                weights_df['date'] = meta.get('trade_date')

                # 应用风控限制与归一化
                weights_df = self.apply_risk_limits(weights_df)

            result = StrategyResult(
                strategy_name=self.config.name,
                weights=weights_df,
                metadata={
                    'n_selected_stocks': len(weights_df),
                    'selection_threshold': threshold if len(pred_df) > 0 else None,
                    'top_quantile': top_quantile,
                    **meta,
                }
            )

            self.logger.info(f"生成 ML 权重 {len(weights_df)} 只股票")
            return result



class SectorNeutralMLStrategy(MLStockSelectionStrategy):
    """Sector-neutral ML strategy that balances sector exposures."""

    def generate_weights(self, data: Dict[str, pd.DataFrame],
                        target_date: str = None, **kwargs) -> StrategyResult:
        """
        Per-sector training and selection, then merge selections.

        Args:
            data: contains 'fundamentals' with columns including y_return and sector info
            target_date: target quarter for prediction
            **kwargs: train_quarters, test_quarters, top_quantile

        Returns:
            StrategyResult with merged per-sector selections
        """
        self.logger.info("Generating sector ML weights with per-sector training")

        # Validate fundamentals
        if 'fundamentals' not in data or len(data['fundamentals']) == 0:
            raise ValueError("必须在 data['fundamentals'] 中提供真实基本面数据（需包含 y_return）")
        fundamentals = data['fundamentals'].copy()

        # Sector column
        sector_col = 'sector' if 'sector' in fundamentals.columns else ('gsector' if 'gsector' in fundamentals.columns else None)
        if sector_col is None:
            self.logger.warning("No sector/gsector column found. Fallback to base ML selection")
            return super().generate_weights(data, target_date, **kwargs)

        # Target date: 如果传入 target_date，则各行业子集仅保留该日期及之前
        if target_date is not None:
            td = pd.to_datetime(target_date)
            fundamentals = fundamentals[fundamentals['datadate'] <= td].copy()

        # 模式和窗口
        prediction_mode = str(kwargs.get('prediction_mode', 'single')).lower()
        test_quarters = int(kwargs.get('test_quarters', 4))
        train_quarters = int(kwargs.get('train_quarters', 16))
        top_quantile = float(kwargs.get('top_quantile', 0.75))
        weight_method = str(kwargs.get('weight_method', 'equal')).lower()
        price_data = data.get('prices', None)  # 获取日度价格数据（用于 min_variance与当日模式）

        selected_all = []
        per_sector_meta: Dict[str, Any] = {}

        sectors = [s for s in fundamentals[sector_col].dropna().unique().tolist()]
        for s in sectors:
            sector_df = fundamentals[fundamentals[sector_col] == s].copy()
            if len(sector_df) < 20:
                self.logger.warning(f"Sector {s}: insufficient samples, skip")
                continue
            try:
                if prediction_mode == 'rolling':
                    preds_all = self._rolling_train_all_date(
                        fundamentals=sector_df,
                        train_quarters=train_quarters,
                        test_quarters=test_quarters,
                    )
                    if preds_all is None or len(preds_all) == 0:
                        continue
                    # 在行业维度下保留所有日期，稍后统一到跨行业合并逻辑
                    pred_df = preds_all[['date', 'gvkey', 'predicted_return']].copy()
                    pred_df['sector'] = s
                    meta = {
                        'mode': 'rolling',
                        'test_window_quarters': test_quarters,
                    }
                else:
                    pred_df, meta = self._rolling_train_single_date(
                        fundamentals=sector_df,
                        test_quarters=test_quarters,
                    )
                    meta['mode'] = 'single'
                    # 当日模式（行业单次）：在阈值筛选之前调整 predicted_return
                    confirm_mode = str(kwargs.get('confirm_mode', 'none')).lower()
                    execution_date = kwargs.get('execution_date', None)
                    if confirm_mode in ('today', 'same_day'):
                        try:
                            pred_df, confirm_meta = self._adjust_predictions_by_same_day_gap(
                                pred_df=pred_df,
                                price_data=price_data,
                                execution_date=execution_date,
                            )
                            # 记录至 meta（注意：此 meta 为行业内局部，可用于汇总）
                            meta.update(confirm_meta)
                        except Exception as e:
                            self.logger.warning(f"当日模式（行业）调整失败，使用原始预测: {e}")
            except Exception as e:
                self.logger.warning(f"Sector {s}: rolling training failed: {e}")
                continue

            if pred_df.empty:
                continue

            if prediction_mode == 'rolling':
                # 逐日按行业选股
                grouped_sec = pred_df.groupby('date')
                for dt, g in grouped_sec:
                    thr = g['predicted_return'].quantile(top_quantile)
                    sel = g[g['predicted_return'] >= thr].copy()
                    if len(sel) == 0:
                        continue
                    sel['sector'] = s
                    sel['date'] = dt
                    selected_all.append(sel)
                # 记录行业层面候选与平均阈值等信息（可选）
                per_sector_meta[s] = {
                    'n_candidates_total': int(len(pred_df)),
                    'top_quantile': top_quantile,
                }
            else:
                pred_df['sector'] = s
                thr = pred_df['predicted_return'].quantile(top_quantile)
                selected = pred_df[pred_df['predicted_return'] >= thr].copy()
                if len(selected) == 0:
                    continue
                selected_all.append(selected)
                per_sector_meta[s] = {
                    'model_name': meta.get('model_name'),
                    'model_mse': meta.get('model_mse'),
                    'n_candidates': len(pred_df),
                    'n_selected': len(selected),
                    'threshold': thr,
                }

        if not selected_all:
            self.logger.warning("No sector produced selections")
            return StrategyResult(
                strategy_name=self.config.name,
                weights=pd.DataFrame(columns=['gvkey', 'weight']),
                metadata={'error': 'no_sector_selection'}
            )

        merged = pd.concat(selected_all, ignore_index=True)

        if prediction_mode == 'rolling':
            # 按日合并跨行业的选股，组内等权并风控归一
            weights_list = []
            for dt, g in merged.groupby('date'):
                if g.empty:
                    continue
                g = g[['gvkey', 'predicted_return', 'sector']].copy()
                
                # 使用新的权重分配函数（优先使用基本面数据中的价格）
                g = self.allocate_weights(
                    selected_stocks=g,
                    method=weight_method,
                    fundamentals=fundamentals,
                    price_data=price_data,
                    **kwargs
                )
                g['date'] = dt
                g = self.apply_risk_limits(g[['gvkey', 'weight', 'predicted_return', 'sector', 'date']])
                weights_list.append(g)

            if not weights_list:
                return StrategyResult(
                    strategy_name=self.config.name,
                    weights=pd.DataFrame(columns=['gvkey', 'weight', 'predicted_return', 'sector', 'date']),
                    metadata={'error': 'no_sector_selection_all_dates', 'mode': 'rolling'}
                )

            weights_df = pd.concat(weights_list, ignore_index=True)
            result = StrategyResult(
                strategy_name=self.config.name,
                weights=weights_df[['gvkey', 'weight', 'predicted_return', 'sector', 'date']].copy(),
                metadata={
                    'mode': 'rolling',
                    'top_quantile': top_quantile,
                    'n_trade_dates': int(weights_df['date'].nunique()),
                    'per_sector': per_sector_meta,
                }
            )
            self.logger.info(f"Generated sector-merged rolling weights: {len(weights_df)} rows across {result.metadata['n_trade_dates']} dates")
            return result
        else:
            # 单一日期：全局等权并风控归一（保持原逻辑）
            weights_df = merged[['gvkey', 'predicted_return', 'sector']].copy()
            
            # 使用新的权重分配函数（优先使用基本面数据中的价格）
            weights_df = self.allocate_weights(
                selected_stocks=weights_df,
                method=weight_method,
                fundamentals=fundamentals,
                price_data=price_data,
                **kwargs
            )
            weights_df['date'] = pred_df['date'].iloc[0] if 'date' in pred_df.columns and len(pred_df) > 0 else meta.get('trade_date')
            weights_df = weights_df[['gvkey', 'weight', 'predicted_return', 'sector', 'date']]

            weights_df = self.apply_risk_limits(weights_df)

            result = StrategyResult(
                strategy_name=self.config.name,
                weights=weights_df,
                metadata={
                    'n_selected_stocks': len(weights_df),
                    'top_quantile': top_quantile,
                    'trade_date': weights_df['date'].iloc[0] if 'date' in weights_df.columns and len(weights_df) > 0 else None,
                    'per_sector': per_sector_meta,
                }
            )
            self.logger.info(f"Generated sector-merged weights for {len(weights_df)} stocks across {len(per_sector_meta)} sectors")
            return result


if __name__ == "__main__":
    # 使用真实数据进行示例测试
    logging.basicConfig(level=logging.INFO)
    os.makedirs('./data', exist_ok=True)

    # manager = get_data_manager()
    # tickers = fetch_sp500_tickers()
    # # 为了运行速度，仅取前若干只股票
    # # tickers = sorted(tickers)[:10]
    # tickers = tickers[:100]

    # start_date = "2019-01-01"
    # end_date = datetime.now().strftime('%Y-%m-%d')
    # fundamentals = fetch_fundamental_data(tickers, start_date, end_date)

    # fundamentals = pd.read_csv('./data/fundamentals.csv')
    fundamentals = pd.read_csv(r'D:\Projects\FinRL-Trading-old\output_20250712\output_20250712\final_ratios.csv')
    fundamentals['datadate'] = fundamentals['date']
    fundamentals['gvkey'] = fundamentals['tic']

    fundamentals = fundamentals[(fundamentals['datadate'] <= '2025-03-01') & (fundamentals['datadate'] >= '2019-03-01')]
    # 仅保留包含 y_return 的样本
    fundamentals = fundamentals[fundamentals.get('y_return').notna()] if 'y_return' in fundamentals.columns else fundamentals
    
    # fundamentals.to_csv('./data/cache/fundamentals.csv', index=False)

    data_dict = { 'fundamentals': fundamentals }

    # 跨行业版本
    # 单次模式
    config = StrategyConfig(
        name="ML Stock Selection",
        description="Machine learning based stock selection"
    )

    # 以末期季度为目标日期进行训练与预测
    target_date = sorted(pd.to_datetime(fundamentals['datadate']).unique())[-1] if len(fundamentals) else None

    strategy = MLStockSelectionStrategy(config)
    # 单次模式 - 等权重
    result_single = strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='single',
        weight_method='equal'
    )
    print(f"Strategy(single-equal): {result_single.strategy_name}")
    print(f"Selected {len(result_single.weights)} stocks (single)")
    print(result_single.weights.head())
    result_single.weights.to_csv(r'.\data\ml_weights_single.csv', index=False)

    # 单次模式 - 最小方差（使用基本面数据中的 adj_close_q）
    # 注意：基本面数据已包含 adj_close_q 列，无需额外提供价格数据
    result_single_mv = strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='single',
        weight_method='min_variance',
        lookback_periods=8  # 回溯8个季度（2年）
    )
    print(f"\nStrategy(single-min_variance): {result_single_mv.strategy_name}")
    print(f"Selected {len(result_single_mv.weights)} stocks")
    print(result_single_mv.weights.head())
    print(result_single_mv.weights[['gvkey', 'weight']].head(10))
    result_single_mv.weights.to_csv(r'.\data\ml_weights_single_mv.csv', index=False)

    # exit()

    # 滚动模式（多日期）- 等权重
    result_rolling = strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='rolling',
        weight_method='equal'
    )
    print(f"\nStrategy(rolling-equal): {result_rolling.strategy_name}")
    print(f"Rows: {len(result_rolling.weights)}, dates: {result_rolling.weights['date'].nunique() if 'date' in result_rolling.weights.columns else 0}")
    print(result_rolling.weights.head())
    result_rolling.weights.to_csv(r'.\data\ml_weights.csv', index=False)


    # 行业版本（如果有 sector/gsector 信息）
    sector_config = StrategyConfig(
        name="Sector Neutral ML",
        description="Sector-neutral ML strategy"
    )
    sector_strategy = SectorNeutralMLStrategy(sector_config)

    # 行业-单次 - 等权重
    sector_single = sector_strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='single',
        weight_method='equal'
    )
    if len(sector_single.weights) > 0 and ('sector' in sector_single.weights.columns):
        print(f"\nSector-neutral(single-equal) selected {len(sector_single.weights)} stocks")
        print(sector_single.weights.groupby('sector')['weight'].sum())
        sector_single.weights.to_csv(r'.\data\ml_weights_sector_single.csv', index=False)

    # 行业-滚动（多日期）- 等权重
    sector_rolling = sector_strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='rolling',
        weight_method='equal'
    )
    if len(sector_rolling.weights) > 0 and ('sector' in sector_rolling.weights.columns):
        print(f"\nSector-neutral(rolling-equal) rows {len(sector_rolling.weights)}, dates {sector_rolling.weights['date'].nunique() if 'date' in sector_rolling.weights.columns else 0}")
        # 展示每日每行业权重和应为1（按日内归一）
        try:
            print(sector_rolling.weights.groupby(['date','sector'])['weight'].sum().head())
        except Exception:
            pass
        sector_rolling.weights.to_csv(r'.\data\ml_weights_sector.csv', index=False)
    
    # 行业-滚动（多日期）- 最小方差（使用基本面数据中的 adj_close_q）
    sector_rolling_mv = sector_strategy.generate_weights(
        data_dict,
        test_quarters=4,
        top_quantile=0.75,
        prediction_mode='rolling',
        weight_method='min_variance',
        lookback_periods=8  # 回溯8个季度
    )
    if len(sector_rolling_mv.weights) > 0:
        print(f"\nSector-neutral(rolling-min_variance) rows {len(sector_rolling_mv.weights)}, dates {sector_rolling_mv.weights['date'].nunique() if 'date' in sector_rolling_mv.weights.columns else 0}")
        # 展示前几个日期的权重分布
        try:
            first_date = sorted(sector_rolling_mv.weights['date'].unique())[0]
            print(f"\n第一个交易日 {first_date} 的权重分布（前10只）:")
            print(sector_rolling_mv.weights[sector_rolling_mv.weights['date']==first_date][['gvkey', 'weight', 'sector']].head(10))
        except Exception:
            pass
        sector_rolling_mv.weights.to_csv(r'.\data\ml_weights_sector_mv.csv', index=False)
