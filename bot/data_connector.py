"""
Raja Banks Trading Agent - Data Connector
Twelve Data API (XAU/USD) M15 bar fetching with retry logic and caching.
Production-ready for continuous 24/5 operation.

Free tier: 800 API calls/day, 8 calls/minute.
"""
import os
import time
import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
BASE_URL = "https://api.twelvedata.com"

# Twelve Data interval mapping
INTERVAL_MAP = {
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}


class DataConnector:
    """
    Fetches XAUUSD (XAU/USD) M15 bars from Twelve Data API.
    Includes retry logic for network failures and smart caching.
    """

    def __init__(self, symbol: str = "GC=F", timeframe: str = "15m",
                 cache_minutes: int = 1):
        # Map Yahoo symbol to Twelve Data symbol
        self.yahoo_symbol = symbol
        self.symbol = "XAU/USD"  # Twelve Data uses forex pair notation
        self.timeframe = timeframe
        self.td_interval = INTERVAL_MAP.get(timeframe, "15min")
        self.cache_minutes = cache_minutes
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None
        self._consecutive_failures = 0

        # API key from environment
        self.api_key = os.getenv("TWELVEDATA_API_KEY", "")
        if not self.api_key:
            logger.warning("TWELVEDATA_API_KEY not set! Data fetching will fail.")

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "RajaTradingBot/1.0"})

    def _fetch_from_api(self, interval: str, outputsize: int = 500,
                        start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Core fetch from Twelve Data time_series endpoint.
        Returns DataFrame with OHLCV data indexed by UTC datetime.
        """
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": min(outputsize, 5000),
            "timezone": "UTC",
            "dp": 2,  # decimal places
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        url = f"{BASE_URL}/time_series"
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Check for API errors
        if data.get("status") == "error":
            raise ValueError(f"Twelve Data API error: {data.get('message', 'unknown')}")

        values = data.get("values", [])
        if not values:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)

        # Convert string columns to float
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        # Volume may not be present for forex pairs
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(float)
        else:
            df["volume"] = 0.0

        # Rename to standard OHLCV format
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }, inplace=True)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Twelve Data returns newest first, reverse to chronological order
        df.sort_index(inplace=True)

        return df

    def fetch_bars(self, lookback_days: int = 7,
                   min_bars: int = 0,
                   use_cache: bool = True) -> pd.DataFrame:
        """Fetch M15 bars with retry logic."""
        # Check cache
        if use_cache and self._cache is not None and self._cache_time:
            age = (datetime.now(timezone.utc) - self._cache_time).total_seconds()
            if age < self.cache_minutes * 60:
                return self._cache.copy()

        # Calculate how many bars we need
        # M15: ~4 bars/hour, ~88 bars/day (22h trading)
        bars_needed = max(min_bars, int(lookback_days * 88))
        bars_needed = min(bars_needed, 5000)  # Twelve Data max

        for attempt in range(MAX_RETRIES):
            try:
                df = self._fetch_from_api(
                    interval=self.td_interval,
                    outputsize=bars_needed,
                )

                if df.empty:
                    logger.warning("No data from Twelve Data (attempt %d/%d)",
                                   attempt + 1, MAX_RETRIES)
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    if self._cache is not None:
                        logger.info("Returning stale cache")
                        return self._cache.copy()
                    return pd.DataFrame()

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
        """Get latest price from Twelve Data /price endpoint (lightweight)."""
        try:
            url = f"{BASE_URL}/price"
            resp = self._session.get(url, params={
                "symbol": self.symbol,
                "apikey": self.api_key,
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if "price" in data:
                return float(data["price"])
        except Exception as e:
            logger.warning("Price fetch failed: %s", e)

        # Fallback to cached data
        if self._cache is not None and not self._cache.empty:
            return float(self._cache["Close"].iloc[-1])
        return None

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
        td_interval = INTERVAL_MAP.get(timeframe, "1h")

        # HTF bars needed: ~22 bars/day for 1h
        bars_needed = min(lookback_days * 22, 5000)

        for attempt in range(MAX_RETRIES):
            try:
                df = self._fetch_from_api(
                    interval=td_interval,
                    outputsize=bars_needed,
                )

                if df.empty:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    return pd.DataFrame()

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
        Twelve Data supports up to 5000 bars in a single request.
        """
        logger.info("Preloading %d+ M15 bars from Twelve Data...", min_bars)

        # Request more than needed to ensure we have enough after cleanup
        request_size = min(min_bars + 100, 5000)

        df = self._fetch_from_api(
            interval=self.td_interval,
            outputsize=request_size,
        )

        if df.empty:
            logger.error("Failed to preload any historical data!")
            return df

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

        if len(df) >= min_bars:
            logger.info("Preloaded %d bars — ready for analysis", len(df))
        else:
            logger.warning("Could only preload %d bars (target: %d) — "
                           "proceeding with available data", len(df), min_bars)
        return df

    @property
    def is_healthy(self) -> bool:
        """Check if data connector is working."""
        return self._consecutive_failures < MAX_RETRIES * 2
