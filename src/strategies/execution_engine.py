import logging
import pandas as pd
from typing import Dict, Optional
import pandas_market_calendars as mcal
import random 
import numpy as np
from strategies.strategylogger import StrategyLogger
from strategies.universe_manager import UniverseManager
from strategies.base_signal import BaseSignalEngine


class ExecutionManager:
    def __init__(
        self,
        universe_mgr,
        max_positions: int = 20,
        max_weight: float = 0.20,
        min_weight: float = 0.05,
        weight_step: float = 0.05,
        allow_short: bool = True,
        gross_leverage: float = 1.0,
        cooling_days: int = 0,
        rebalance_freq: str = "D",  # "D" (daily) or "M" (monthly), "W" (weekly), extensible
        logger: Optional[object] = None,
        ratio: float = 1.0,           # Maximum available capital ratio
        seed: int = 42                # Random seed for reproducibility
    ):
        """
        Parameters
        ----------
        universe_mgr : UniverseManager
            Initialized UniverseManager for determining stock universe membership
        max_positions : int
            Maximum number of positions (counted by |weight| > 0)
        max_weight : float
            Maximum absolute weight per stock (e.g., 0.2 = 20%)
        min_weight : float
            Minimum non-zero absolute weight per stock (e.g., 0.05 = 5%);
            values below this threshold are treated as 0
        weight_step : float
            Step size for weight adjustments (e.g., 0.05)
        allow_short : bool
            Whether to allow short positions (weight < 0)
        gross_leverage : float
            Maximum total |weight| sum (e.g., 1.0 = 100%)
        cooling_days : int
            Cooling-off period: number of trading days to wait after closing
            a position before re-opening
        rebalance_freq : str
            Rebalance frequency:
              - "D": Daily rebalance
              - "M": Monthly rebalance (uses the 2nd trading day of the month)
              - "W": Weekly rebalance (uses the 1st trading day of the week)
              - Extensible to "intraday" etc.
        logger : object or None
            Optional logger implementing log_signal(...)
        """
        self.universe_mgr = universe_mgr
        self.max_positions = int(max_positions)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.weight_step = float(weight_step)
        self.allow_short = allow_short
        self.gross_leverage = float(gross_leverage)
        self.cooling_days = int(cooling_days)
        self.rebalance_freq = rebalance_freq.upper()
        self.logger = logger
        self.ratio = ratio
        random.seed(seed)

        # Current target weights: tic_name -> weight (can be negative for short)
        self.current_weights: Dict[str, float] = {}

        # Cooldown counter: tic_name -> remaining cooldown days
        self.cooldown: Dict[str, int] = {}

        # Previous date, used for close-only determination
        self.prev_date: Optional[pd.Timestamp] = None

    def set_rebalance_frequency(self, freq: str):
        """
        freq: 'D' / 'W' / 'M'
        """
        self.rebalance_freq = freq.upper()
    
    # =========================================================
    # Main public entry: generate full historical weight matrix from signal_df
    # =========================================================
    def generate_weight_matrix(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a weight matrix (index=date, columns=tic_name, value=weight)
        based on the daily signal_df (index=date, columns=tic_name, value=-1/0/1)

        Parameters
        ----------
        signal_df : pd.DataFrame
            index: Date (DatetimeIndex or convertible to Timestamp)
            columns: tic_name
            values: -1 / 0 / 1

        Returns
        -------
        weights_df : pd.DataFrame
            index: Date
            columns: tic_name
            values: Weight (float)
        """
        dates = sorted(pd.to_datetime(signal_df.index.unique()))
        all_tics = sorted(signal_df.columns.unique())

        records = []

        for dt in dates:
            # Get the signal for the day
            row = signal_df.loc[dt]
            if isinstance(row, pd.DataFrame):
                # If there are multiple rows (uncommon), take the first row
                signal_series = row.iloc[0]
            else:
                signal_series = row

            self.step(dt, signal_series)

            # Record the weights for the day
            row_weights = {tic: self.current_weights.get(tic, 0.0) for tic in all_tics}
            row_weights["date"] = pd.Timestamp(dt)
            records.append(row_weights)

        # Calculate the target weight matrix
        weights_df = pd.DataFrame(records).set_index("date").sort_index()

        if hasattr(self, "_compute_target_weights"):
            try:
                target_df = self._compute_target_weights(signal_df)
                # Align the index and columns (take the intersection to avoid column inconsistency)
                target_df = target_df.reindex_like(weights_df).fillna(0.0)
                # Use the target weights to override the current weights matrix
                weights_df.update(target_df)
                if self.logger:
                    self.logger.log_info("[ExecutionManager] Applied _compute_target_weights successfully.")
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"[ExecutionManager] _compute_target_weights failed: {e}")
                else:
                    print(f"[WARN] _compute_target_weights failed: {e}")

        return weights_df

    # Frequency control of rebalance
    def _should_rebalance(self, date: pd.Timestamp) -> bool:
        """
        Based on the rebalance_freq, determine if the current date needs rebalancing.
        Currently supported:
          - "D": Day
          - "M": Month
        """
        date = pd.Timestamp(date)

        if self.rebalance_freq == "D":
            return True

        if self.rebalance_freq == "W":
            cal = self.universe_mgr.trading_calendar
            # Find the trading days of the week
            week_dates = [d for d in cal
                          if d.isocalendar()[1] == date.isocalendar()[1]
                          and d.year == date.year]
            if not week_dates:
                return False
            week_dates = sorted(week_dates)
            return date.normalize() == week_dates[0].normalize()

        if self.rebalance_freq == "M":
            cal = self.universe_mgr.trading_calendar
            month_dates = [d for d in cal if d.year == date.year and d.month == date.month]
            if not month_dates:
                return False
            month_dates = sorted(month_dates)
            # 2nd trading day and last trading day of the month
            second_day = month_dates[1] if len(month_dates) >= 2 else month_dates[0]
            last_day = month_dates[-1]
            return date.normalize() in [
                pd.Timestamp(second_day).normalize(),
                pd.Timestamp(last_day).normalize()
            ]

    # Daily execution logic: update self.current_weights
    def step(self, date, signal_series: pd.Series):
        """
        Single day execution logic:
          1. Decrement cooldown period for each stock.
          2. Check if today is a rebalance day based on the strategy's settings.
          3. If today is a rebalance day:
             - Adjust weights according to Universe membership, signals, close-only rule, and cooldown status
             - Apply portfolio constraints such as max_positions and gross_leverage
        """
        date = pd.Timestamp(date)
        signals = signal_series.to_dict()  # tic -> -1/0/1

        # Decrement the cooldown period for each stock
        for tic in list(self.cooldown.keys()):
            if self.cooldown[tic] > 0:
                self.cooldown[tic] -= 1

        # Update prev_date (for close-only judgment)
        prev_date = self.prev_date
        self.prev_date = date

        # If not a rebalance day, do not change the weights (cooldown period still decrements)
        if not self._should_rebalance(date):
            return

        # The universe of stocks that are allowed to open new positions today
        today_universe = self.universe_mgr.get_universe(date)

        current_positions = {tic for tic, w in self.current_weights.items() if abs(w) > 0}

        # All stocks that need to be considered: have signal or have positions
        all_tics = sorted(set(signals.keys()) | current_positions)

        new_weights = self.current_weights.copy()

        for tic in all_tics:
            old_w = float(self.current_weights.get(tic, 0.0))
            sig = int(signals.get(tic, 0))

            # Cooldown status
            cd = int(self.cooldown.get(tic, 0))
            has_pos = abs(old_w) > 0

            in_uni_today = tic in today_universe
            in_uni_yday = False
            if prev_date is not None:
                in_uni_yday = self.universe_mgr.is_in_universe(tic, prev_date)

            # Close-only: yesterday in the pool & today not in the pool & still have positions
            close_only = in_uni_yday and (not in_uni_today) and has_pos

            # If no positions and in cooldown period, do not open new positions (regardless of the signal)
            if (not has_pos) and cd > 0:
                effective_sig = 0
            else:
                effective_sig = sig

            # Decide the target direction today (0/+1/-1)
            if effective_sig == 0:
                # Signal is 0: immediately close
                new_w = 0.0

            elif close_only:
                # Close-only: do not open new positions; only keep the original position
                # If the signal turns to 0, close (already covered above)
                new_w = old_w
                logging.getLogger(f"{__name__}.{self.__class__.__name__}").info(f"[CLOSE-ONLY] {tic} keep position {old_w:.2f} (still have positions in the pool)")

            elif effective_sig > 0 and in_uni_today:
                target_sign = 1
                new_w = self._update_weight_one_name(
                    old_weight=old_w, target_sign=target_sign, close_only=False, target_weight=self.max_weight, 
                )

            elif effective_sig < 0 and in_uni_today and self.allow_short:
                target_sign = -1
                new_w = self._update_weight_one_name(
                    old_weight=old_w, target_sign=target_sign, close_only=False, target_weight=self.max_weight, 
                )

            else:
                # Not in the universe today & no positions -> force 0
                new_w = 0.0

            # Update the weight for the day
            new_weights[tic] = new_w

            # If the position changes from non-zero to 0 -> start the cooldown period
            if (abs(old_w) > 0) and (abs(new_w) == 0) and (self.cooling_days > 0):
                self.cooldown[tic] = self.cooling_days

            # Log
            if self.logger is not None:
                if abs(old_w - new_w) > 1e-8:
                    action = "HOLD"
                    if old_w == 0 and new_w != 0:
                        action = "OPEN_LONG" if new_w > 0 else "OPEN_SHORT"
                    elif old_w != 0 and new_w == 0:
                        action = "CLOSE"
                    elif old_w * new_w < 0:
                        action = "FLIP"
                    else:
                        action = "ADJUST"

                    self.logger.log_signal(
                        date=date,
                        symbol=tic,
                        signal=effective_sig,
                        action=action,
                        old_weight=old_w,
                        new_weight=new_w,
                        close_only=close_only,
                        cooldown_left=self.cooldown.get(tic, 0),
                    )

        # ========= Portfolio level constraints =========

        # 1) Limit the number of positions
        nz = [(tic, w) for tic, w in new_weights.items() if abs(w) > 0]
        if len(nz) > self.max_positions:
            # Sort by |weight| in descending order, keep the top max_positions
            nz_sorted = sorted(nz, key=lambda x: abs(x[1]), reverse=True)
            keep = {tic for tic, _ in nz_sorted[: self.max_positions]}
            for tic, w in nz:
                if tic not in keep:
                    new_weights[tic] = 0.0

        # 2) Limit the total leverage (by the sum of absolute values)
        gross = sum(abs(w) for w in new_weights.values())
        if gross > 0 and gross > self.gross_leverage:
            scale = self.gross_leverage / gross
            for tic in new_weights:
                new_weights[tic] *= scale

        self.current_weights = new_weights

    # Single stock weight adjustment logic
    def _update_weight_one_name(
        self,
        old_weight: float,
        target_sign: int,
        close_only: bool,
        target_weight: float,
    ) -> float:
        w = float(old_weight)

        # In close-only mode, only reduce the position
        if close_only and target_sign != 0:
            return w  # Do not open new positions

        if target_sign == 0:
            # Immediately close
            return 0.0

        # Open new positions or add positions or flip positions: directly set the target position
        return target_sign * target_weight

    def _apply_min_weight_threshold(self, w: float) -> float:
        """
        When the absolute value is less than min_weight, directly treat it as 0 (close),
        to prevent "dirty positions" like 0.01.
        """
        if abs(w) < self.min_weight:
            return 0.0
        return w

    def _compute_target_weights(self, signal_df: pd.DataFrame, 
                                max_weight: Optional[float] = None,
                                min_weight: Optional[float] = None,
                                ratio: Optional[float] = None,
                                max_positions: Optional[int] = None,
                                seed: Optional[int] = None) -> pd.DataFrame:
        """
        Compute the target weight matrix from signal_df, subject to the following constraints:
        1. Single stock max position <= max_weight (default 20%)
        2. Single stock min position >= min_weight (default 2%)
        3. Short positions count toward total position (absolute value)
        4. ratio controls total available position ratio (default 1.0)
        5. Max positions limit (default 20, excess randomly sampled)
        6. New stocks only use remaining capacity, don't adjust existing positions
        7. Last trading day of each month: equal-weight rebalance

        Parameters
        ----------
        signal_df : pd.DataFrame
            Signal DataFrame with index=date, columns=tic_name, values=-1/0/1
        max_weight : float, optional
            Override for max_weight per stock
        min_weight : float, optional
            Override for min_weight per stock
        ratio : float, optional
            Override for available capital ratio
        max_positions : int, optional
            Override for max positions
        seed : int, optional
            Random seed for reproducibility (default uses self seed)

        Returns
        -------
        weights_target : pd.DataFrame
            Target weight matrix with index=date, columns=tic_name, values=weights
        """
        # Use instance defaults if not overridden
        if max_weight is None:
            max_weight = self.max_weight
        if min_weight is None:
            min_weight = self.min_weight
        if ratio is None:
            ratio = getattr(self, "ratio", 1.0)
        if max_positions is None:
            max_positions = self.max_positions
        if seed is not None:
            random.seed(seed)

        dates = sorted(pd.to_datetime(signal_df.index.unique()))
        all_tics = signal_df.columns
        weights_target = pd.DataFrame(index=dates, columns=all_tics, dtype=float).fillna(0.0)

        current_holdings = set()
        last_weights = pd.Series(0.0, index=all_tics)

        for date in dates:
            cal = self.universe_mgr.trading_calendar
            month_dates = [d for d in cal if d.year == date.year and d.month == date.month]
            if not month_dates:
                continue
            month_dates = sorted(month_dates)

            # 2nd trading day and last day of the month
            signal_day = month_dates[1] if len(month_dates) >= 2 else month_dates[0]
            last_day = month_dates[-1]
            is_signal_day = date.normalize() == pd.Timestamp(signal_day).normalize()
            is_month_end = date.normalize() == pd.Timestamp(last_day).normalize()

            # --- 2nd trading day of each month: build positions based on signals ---
            if is_signal_day:
                row = signal_df.loc[date] if date in signal_df.index else None
                if row is None:
                    weights_target.loc[date] = last_weights
                    continue

                active_tics = [tic for tic, sig in row.items() if sig != 0]
                if not active_tics:
                    weights_target.loc[date] = last_weights
                    continue

                if len(active_tics) > max_positions:
                    active_tics = random.sample(active_tics, max_positions)

                equal_w = min(max_weight, max(min_weight, ratio / len(active_tics)))
                new_weights = pd.Series(0.0, index=all_tics)
                for tic in active_tics:
                    new_weights[tic] = row[tic] * equal_w

                last_weights = new_weights.copy()
                weights_target.loc[date] = last_weights
                current_holdings = set(active_tics)

            # --- End of month Rebalance ---
            elif is_month_end and len(current_holdings) > 0:
                equal_w = min(max_weight, max(min_weight, ratio / len(current_holdings)))
                new_weights = pd.Series(0.0, index=all_tics)
                for tic in current_holdings:
                    sig = 1 if last_weights.get(tic, 0) >= 0 else -1
                    new_weights[tic] = sig * equal_w

                last_weights = new_weights.copy()
                weights_target.loc[date] = last_weights

            # --- Other dates: carry forward last positions ---
            else:
                weights_target.loc[date] = last_weights

        # Final cleanup
        weights_target = weights_target.fillna(0.0)
        return weights_target