"""
Raja Banks Trading Agent - Data Connector
Yahoo Finance (GC=F) M15 bar fetching with retry logic and caching.
Production-ready for continuous 24/5 operation.
"""
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class DataConnector:
    """
    Fetches XAUUSD (GC=F) M15 bars from Yahoo Finance.
    Includes retry logic for network failures and smart caching.
    """

    def __init__(self, symbol: str = "GC=F", timeframe: str = "15m",
                 cache_minutes: int = 1):
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_minutes = cache_minutes
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None
        self._ticker = yf.Ticker(symbol)
        self._consecutive_failures = 0

    def fetch_bars(self, lookback_days: int = 7,
                   min_bars: int = 0,
                   use_cache: bool = True) -> pd.DataFrame:
        """Fetch M15 bars with retry logic."""
        # Check cache
        if use_cache and self._cache is not None and self._cache_time:
            age = (datetime.now(timezone.utc) - self._cache_time).total_seconds()
            if age < self.cache_minutes * 60:
                return self._cache.copy()

        lookback_days = min(lookback_days, 59)

        for attempt in range(MAX_RETRIES):
            try:
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=lookback_days)

                df = self._ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=self.timeframe,
                    auto_adjust=True,
                )

                if df.empty:
                    logger.warning("No data from Yahoo Finance (attempt %d/%d)",
                                   attempt + 1, MAX_RETRIES)
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    if self._cache is not None:
                        logger.info("Returning stale cache")
                        return self._cache.copy()
                    return pd.DataFrame()

                # Ensure UTC timezone
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")

                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.dropna(inplace=True)

                # Add helper columns
                df["body"] = abs(df["Close"] - df["Open"])
                df["range"] = df["High"] - df["Low"]
                df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
                df["body_ratio"] = df["body_ratio"].fillna(0)
                df["bullish"] = df["Close"] > df["Open"]

                # Cache
                self._cache = df.copy()
                self._cache_time = datetime.now(timezone.utc)
                self._consecutive_failures = 0

                logger.info("Fetched %d %s bars for %s",
                            len(df), self.timeframe, self.symbol)
                return df

            except Exception as e:
                self._consecutive_failures += 1
                logger.error("Data fetch error (attempt %d/%d): %s",
                             attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        # All retries failed
        if self._cache is not None:
            logger.warning("All retries failed, returning stale cache "
                           "(%d consecutive failures)", self._consecutive_failures)
            return self._cache.copy()
        return pd.DataFrame()

    def get_latest_bar(self) -> Optional[pd.Series]:
        df = self.fetch_bars(lookback_days=2)
        if df.empty:
            return None
        if len(df) >= 2:
            return df.iloc[-2]
        return df.iloc[-1]

    def get_current_price(self) -> Optional[float]:
        df = self.fetch_bars(lookback_days=1)
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])

    def estimate_spread(self, df: Optional[pd.DataFrame] = None,
                        window: int = 20) -> float:
        if df is None:
            df = self.fetch_bars(lookback_days=2)
        if df.empty or len(df) < window:
            return 2.0
        recent = df.tail(window)
        wick_sizes = recent["range"] - recent["body"]
        avg_wick = wick_sizes.mean()
        spread_dollars = avg_wick * 0.25
        spread_pips = spread_dollars / 0.10
        return max(0.5, min(spread_pips, 10.0))

    def get_volume_sma(self, df: Optional[pd.DataFrame] = None,
                       period: int = 20) -> float:
        if df is None:
            df = self.fetch_bars(lookback_days=5)
        if df.empty or len(df) < period:
            return 0.0
        return float(df["Volume"].tail(period).mean())

    def is_market_open(self) -> bool:
        """Check if gold futures market is likely open."""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        if weekday == 5:  # Saturday
            return False
        if weekday == 6 and hour < 23:  # Sunday before 11pm UTC
            return False
        if weekday == 4 and hour >= 22:  # Friday after 10pm UTC
            return False
        if hour == 21:  # Daily maintenance
            return False
        return True

    def fetch_htf_bars(self, timeframe: str = "1h",
                       lookback_days: int = 30) -> pd.DataFrame:
        """Fetch higher timeframe bars for HTF zone stacking."""
        for attempt in range(MAX_RETRIES):
            try:
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=min(lookback_days, 59))
                df = self._ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=timeframe,
                    auto_adjust=True,
                )
                if df.empty:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    return pd.DataFrame()

                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")

                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.dropna(inplace=True)
                return df

            except Exception as e:
                logger.error("HTF fetch error (attempt %d): %s", attempt + 1, e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        return pd.DataFrame()

    def preload_bars(self, min_bars: int = 400) -> pd.DataFrame:
        """
        Preload enough historical M15 data to guarantee at least `min_bars` bars.
        Gold trades ~22h/day, 5 days/week = ~440 M15 bars per week.
        We fetch progressively more days until we have enough bars.
        """
        # Start with 8 days (~400-440 bars), escalate if needed
        days_attempts = [8, 14, 21, 30, 45, 59]

        for days in days_attempts:
            logger.info("Preloading historical data: trying %d days for %d+ bars...",
                        days, min_bars)
            df = self.fetch_bars(lookback_days=days, use_cache=False)
            if not df.empty and len(df) >= min_bars:
                logger.info("Preloaded %d bars (%d days) — ready for analysis",
                            len(df), days)
                return df
            elif not df.empty:
                logger.info("Got %d bars from %d days, need %d — trying more...",
                            len(df), days, min_bars)

        # Return whatever we got
        if not df.empty:
            logger.warning("Could only preload %d bars (target: %d) — "
                           "proceeding with available data", len(df), min_bars)
        else:
            logger.error("Failed to preload any historical data!")
        return df

    @property
    def is_healthy(self) -> bool:
        """Check if data connector is working."""
        return self._consecutive_failures < MAX_RETRIES * 2
