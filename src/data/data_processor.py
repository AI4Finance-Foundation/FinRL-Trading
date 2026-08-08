"""

Data Processor Module

====================

Handles data preprocessing and feature engineering:

- Fundamental data processing

- Price data processing

- Feature engineering for ML models

- Data quality checks and cleaning

"""

import logging

from typing import Dict, List, Optional, Tuple

from datetime import datetime, timedelta

from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class DataProcessor:

    """Data processor for fundamental and price data."""

    def __init__(self, data_dir: str = "./data"):

        """

        Initialize data processor.

        Args:

            data_dir: Base directory for data files

        """

        self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def process_fundamental_data(self, raw_fundamentals_path: str,

                               processed_path: str = None) -> pd.DataFrame:

        """

        Process raw fundamental data into ML-ready format.

        Args:

            raw_fundamentals_path: Path to raw fundamental data

            processed_path: Path to save processed data (optional)

        Returns:

            Processed fundamental data DataFrame

        """

        logger.info(f"Processing fundamental data from {raw_fundamentals_path}")

        # Load raw data

        df = pd.read_csv(raw_fundamentals_path)

        logger.info(f"Loaded {len(df)} raw fundamental records")

        # Basic data cleaning

        df = self._clean_fundamental_data(df)

        # Feature engineering

        df = self._engineer_fundamental_features(df)

        # Handle missing values

        df = self._handle_missing_values(df)

        # Save processed data

        if processed_path:

            processed_path = Path(processed_path)

            processed_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(processed_path, index=False)

            logger.info(f"Saved processed data to {processed_path}")

        logger.info(f"Processed {len(df)} fundamental records")

        return df

    def _clean_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:

        """Clean fundamental data."""

        # Remove duplicates

        df = df.drop_duplicates(subset=['gvkey', 'datadate'])

        # Convert date column

        df['datadate'] = pd.to_datetime(df['datadate'])

        # Filter out invalid data

        df = df[df['prccd'] > 0]  # Valid prices

        df = df[df['ajexdi'] > 0]  # Valid adjustment factors

        # Create adjusted price

        df['adj_price'] = df['prccd'] / df['ajexdi']

        return df

    def _engineer_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """Engineer fundamental features for ML models."""

        # Basic profitability ratios

        if 'revenue' in df.columns and 'net_income' in df.columns:

            df['profit_margin'] = df['net_income'] / df['revenue']

        # Growth rates (quarterly)

        df = df.sort_values(['gvkey', 'datadate'])

        df['price_growth_qtr'] = df.groupby('gvkey')['adj_price'].pct_change()

        # Rolling statistics

        df['price_volatility_4q'] = df.groupby('gvkey')['adj_price'].rolling(4).std().reset_index(0, drop=True)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:

        """Handle missing values in fundamental data."""

        # Drop columns with too many missing values

        missing_threshold = 0.5

        missing_ratios = df.isnull().mean()

        columns_to_drop = missing_ratios[missing_ratios > missing_threshold].index

        df = df.drop(columns=columns_to_drop)

        logger.info(f"Dropped {len(columns_to_drop)} columns with >{missing_threshold*100}% missing values")

        # Fill remaining missing values with median by sector (if sector column exists)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if "sector" in df.columns:
            df[numeric_columns] = df.groupby("sector")[numeric_columns].transform(
                lambda x: x.fillna(x.median())
            )
        else:
            # Fallback: fill numeric columns with global median
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        return df

    def process_price_data(self, raw_prices_path: str,

                          processed_path: str = None) -> pd.DataFrame:

        """

        Process raw price data into ML-ready format.

        Args:

            raw_prices_path: Path to raw price data

            processed_path: Path to save processed data (optional)

        Returns:

            Processed price data DataFrame

        """

        logger.info(f"Processing price data from {raw_prices_path}")

        # Load raw data

        df = pd.read_csv(raw_prices_path)

        logger.info(f"Loaded {len(df)} raw price records")

        # Basic data cleaning

        df = self._clean_price_data(df)

        # Feature engineering

        df = self._engineer_price_features(df)

        # Save processed data

        if processed_path:

            processed_path = Path(processed_path)

            processed_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(processed_path, index=False)

            logger.info(f"Saved processed data to {processed_path}")

        logger.info(f"Processed {len(df)} price records")

        return df

    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:

        """Clean price data."""

        # Remove duplicates

        df = df.drop_duplicates(subset=['gvkey', 'datadate'])

        # Convert date column

        df['datadate'] = pd.to_datetime(df['datadate'])

        # Filter out invalid data

        df = df[df['prccd'] > 0]  # Valid prices

        df = df[df['ajexdi'] > 0]  # Valid adjustment factors

        # Create adjusted price

        df['adj_close'] = df['prccd'] / df['ajexdi']

        df['adj_open'] = df['prcod'] / df['ajexdi'] if 'prcod' in df.columns else df['adj_close']

        df['adj_high'] = df['prchd'] / df['ajexdi'] if 'prchd' in df.columns else df['adj_close']

        df['adj_low'] = df['prcld'] / df['ajexdi'] if 'prcld' in df.columns else df['adj_close']

        return df

    def _engineer_price_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """Engineer price-based features."""

        # Daily returns

        df = df.sort_values(['gvkey', 'datadate'])

        df['daily_return'] = df.groupby('gvkey')['adj_close'].pct_change()

        # Technical indicators

        df = self._add_technical_indicators(df)

        # Volatility measures

        df['volatility_20d'] = df.groupby('gvkey')['daily_return'].rolling(20).std().reset_index(0, drop=True)

        df['volatility_60d'] = df.groupby('gvkey')['daily_return'].rolling(60).std().reset_index(0, drop=True)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:

        """Add technical indicators to price data."""

        # Simple moving averages

        for period in [5, 10, 20, 50, 200]:

            df[f'sma_{period}'] = df.groupby('gvkey')['adj_close'].rolling(period).mean().reset_index(0, drop=True)

        # RSI (Relative Strength Index)

        df = self._calculate_rsi(df)

        # MACD

        df = self._calculate_macd(df)

        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:

        """Calculate RSI indicator."""

        def rsi_calc(group):

            delta = group['adj_close'].diff()

            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()

            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss

            return 100 - (100 / (1 + rs))

        df['rsi_14'] = df.groupby('gvkey').apply(rsi_calc).reset_index(level=0, drop=True)

        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:

        """Calculate MACD indicator."""

        def macd_calc(group):

            ema_12 = group['adj_close'].ewm(span=12).mean()

            ema_26 = group['adj_close'].ewm(span=26).mean()

            macd = ema_12 - ema_26

            signal = macd.ewm(span=9).mean()

            return macd, signal

        macd_results = df.groupby('gvkey').apply(macd_calc)

        df['macd'] = macd_results.apply(lambda x: x[0]).reset_index(level=0, drop=True)

        df['macd_signal'] = macd_results.apply(lambda x: x[1]).reset_index(level=0, drop=True)

        return df

    def create_ml_dataset(self, fundamentals_path: str, prices_path: str,

                         target_period: int = 63) -> Tuple[pd.DataFrame, pd.Series]:

        """

        Create ML-ready dataset by combining fundamentals and price data.

        Args:

            fundamentals_path: Path to processed fundamental data

            prices_path: Path to processed price data

            target_period: Days to look ahead for target variable

        Returns:

            Tuple of (features DataFrame, target Series)

        """

        logger.info("Creating ML dataset...")

        # Load processed data

        fundamentals = pd.read_csv(fundamentals_path)

        prices = pd.read_csv(prices_path)

        # Merge data

        fundamentals['datadate'] = pd.to_datetime(fundamentals['datadate'])

        prices['datadate'] = pd.to_datetime(prices['datadate'])

        # Create target variable (future returns)

        prices = prices.sort_values(['gvkey', 'datadate'])

        prices['future_return'] = prices.groupby('gvkey')['adj_close'].shift(-target_period) / prices['adj_close'] - 1

        # Merge with fundamentals

        merged = pd.merge_asof(

            fundamentals.sort_values('datadate'),

            prices[['gvkey', 'datadate', 'future_return']].sort_values('datadate'),

            on='datadate',

            by='gvkey',

            direction='backward'

        )

        # Select features

        feature_columns = [col for col in merged.columns

                          if col not in ['gvkey', 'datadate', 'future_return', 'tic']]

        # Clean dataset

        merged = merged.dropna(subset=['future_return'])

        merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

        X = merged[feature_columns]

        y = merged['future_return']

        logger.info(f"Created ML dataset with {len(X)} samples and {len(feature_columns)} features")

        return X, y

    def split_by_sector(self, df: pd.DataFrame, sector_column: str = 'sector',

                       output_dir: str = "./data/processed/sectors") -> Dict[str, pd.DataFrame]:

        """

        Split data by sector for sector-neutral strategies.

        Args:

            df: Input DataFrame

            sector_column: Column name for sector information

            output_dir: Directory to save sector files

        Returns:

            Dictionary of sector DataFrames

        """

        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        sector_data = {}

        for sector, group in df.groupby(sector_column):

            sector_file = output_dir / f"sector_{sector}.csv"

            group.to_csv(sector_file, index=False)

            sector_data[sector] = group

            logger.info(f"Saved sector {sector} with {len(group)} records")

        return sector_data

# Convenience functions

def process_fundamentals(input_path: str, output_path: str = None) -> pd.DataFrame:

    """Process fundamental data."""

    processor = DataProcessor()

    return processor.process_fundamental_data(input_path, output_path)

def process_prices(input_path: str, output_path: str = None) -> pd.DataFrame:

    """Process price data."""

    processor = DataProcessor()

    return processor.process_price_data(input_path, output_path)

def create_ml_dataset(fundamentals_path: str, prices_path: str,

                     target_period: int = 63) -> Tuple[pd.DataFrame, pd.Series]:

    """Create ML-ready dataset."""

    processor = DataProcessor()

    return processor.create_ml_dataset(fundamentals_path, prices_path, target_period)

def main():
    """CLI entry point for data processing."""
    import argparse

    parser = argparse.ArgumentParser(description="Process financial data")
    parser.add_argument('--fetch', action='store_true',
                        help='Fetch data from source')
    parser.add_argument('--process', type=str, default=None,
                        help='Process data file (path to CSV)')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                        help='Symbols to process')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    processor = DataProcessor()

    if args.fetch:
        print(f"Fetching data for symbols: {args.symbols or 'default'}")
        print("Data fetch operation requested")
        print("Use data_fetcher module for data fetching")

    if args.process:
        print(f"Processing data from: {args.process}")
        if 'fundamental' in args.process.lower():
            result = processor.process_fundamental_data(args.process)
        else:
            result = processor.process_price_data(args.process)
        print(f"Processed {len(result)} records")
        print(f"Columns: {list(result.columns)}")

    if not args.fetch and not args.process:
        print("No operation specified. Use --fetch or --process.")
        print("Examples:")
        print("  python src/main.py data --process ./data/fundamentals.csv")
        print("  python src/main.py data --fetch --symbols AAPL MSFT")

if __name__ == "__main__":
    main()
