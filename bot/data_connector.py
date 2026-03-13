"""
Raja Banks Trading Agent - Multi-Source Data Connector
Fetches XAU/USD M15 bars with automatic failover between data sources.

Priority order:
1. Twelve Data (primary) — Free tier: 800 calls/day, 8 calls/min
2. Yahoo Finance (fallback) — Free, no API key needed
3. Alpha Vantage (backup) — Free tier: 25 calls/day

If one source fails, the bot seamlessly switches to the next.
"""
import os
import time
import logging
import threading
import numpy as np
import pandas as pd
import requests
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# ============================================================
# DATA SOURCE IMPLEMENTATIONS
# ============================================================

class TwelveDataSource:
    """Twelve Data API — primary source for XAU/USD."""

    BASE_URL = "https://api.twelvedata.com"
    INTERVAL_MAP = {
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.symbol = "XAU/USD"
        self.name = "TwelveData"
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "RajaTradingBot/1.0"})

        # Rate limiter: max 8 calls per 60 seconds
        self._rate_limit_max = 8
        self._rate_limit_window = 60
        self._call_timestamps: deque = deque()
        self._rate_lock = threading.Lock()

        # Daily usage tracking
        self._api_calls_today = 0
        self._api_day = datetime.now(timezone.utc).date()

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _wait_for_rate_limit(self):
        """Block until within 8 calls/60s rate limit."""
        with self._rate_lock:
            now = time.time()
            while self._call_timestamps and self._call_timestamps[0] < now - self._rate_limit_window:
                self._call_timestamps.popleft()
            if len(self._call_timestamps) >= self._rate_limit_max:
                sleep_time = self._call_timestamps[0] - (now - self._rate_limit_window) + 0.5
                if sleep_time > 0:
                    logger.info("[TwelveData] Rate limit: sleeping %.1fs", sleep_time)
                    time.sleep(sleep_time)
            self._call_timestamps.append(time.time())

    def fetch_bars(self, interval: str, outputsize: int = 500) -> pd.DataFrame:
        """Fetch OHLCV bars from Twelve Data."""
        if not self.api_key:
            raise ValueError("TWELVEDATA_API_KEY not set")

        td_interval = self.INTERVAL_MAP.get(interval, "15min")

        # Track daily usage
        today = datetime.now(timezone.utc).date()
        if today != self._api_day:
            self._api_calls_today = 0
            self._api_day = today
        self._api_calls_today += 1

        if self._api_calls_today > 750:
            logger.warning("[TwelveData] Approaching daily limit: %d/800 calls",
                           self._api_calls_today)

        self._wait_for_rate_limit()

        params = {
            "symbol": self.symbol,
            "interval": td_interval,
            "apikey": self.api_key,
            "outputsize": min(outputsize, 5000),
            "timezone": "UTC",
            "dp": 2,
        }

        resp = self._session.get(f"{self.BASE_URL}/time_series",
                                 params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "error":
            raise ValueError(f"Twelve Data error: {data.get('message', 'unknown')}")

        values = data.get("values", [])
        if not values:
            return pd.DataFrame()

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(float)
        else:
            df["volume"] = 0.0

        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }, inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.sort_index(inplace=True)
        return df

    def get_current_price(self) -> Optional[float]:
        """Lightweight price check."""
        if not self.api_key:
            return None
        try:
            self._wait_for_rate_limit()
            resp = self._session.get(f"{self.BASE_URL}/price", params={
                "symbol": self.symbol, "apikey": self.api_key,
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if "price" in data:
                return float(data["price"])
        except Exception as e:
            logger.warning("[TwelveData] Price fetch failed: %s", e)
        return None


class YahooFinanceSource:
    """Yahoo Finance via yfinance — free, no API key needed."""

    INTERVAL_MAP = {
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",  # yfinance doesn't support 4h natively
        "1d": "1d",
    }

    def __init__(self):
        self.symbol = "GC=F"  # Gold Futures
        self.name = "YahooFinance"
        self._yf = None
        self._yf_available = None

    @property
    def is_available(self) -> bool:
        if self._yf_available is None:
            try:
                import yfinance
                self._yf = yfinance
                self._yf_available = True
            except ImportError:
                self._yf_available = False
                logger.warning("[Yahoo] yfinance not installed — source unavailable")
        return self._yf_available

    def fetch_bars(self, interval: str, outputsize: int = 500) -> pd.DataFrame:
        """Fetch OHLCV bars from Yahoo Finance."""
        if not self.is_available:
            raise ValueError("yfinance not installed")

        yf_interval = self.INTERVAL_MAP.get(interval, "15m")

        # Yahoo limits: 15m data only goes back ~60 days, 1m/5m only 30 days
        if interval == "15m":
            lookback_days = min(outputsize // 88 + 2, 59)  # ~88 bars/day for M15
        elif interval == "1h":
            lookback_days = min(outputsize // 22 + 2, 730)
        else:
            lookback_days = min(outputsize // 1 + 2, 365)

        ticker = self._yf.Ticker(self.symbol)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)

        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=yf_interval,
        )

        if df.empty:
            return pd.DataFrame()

        # yfinance returns columns: Open, High, Low, Close, Volume
        # Index is already datetime
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Keep only OHLCV columns
        keep_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in keep_cols:
            if col not in df.columns:
                df[col] = 0.0

        df = df[keep_cols].copy()
        df.sort_index(inplace=True)

        # Limit to requested size
        if len(df) > outputsize:
            df = df.tail(outputsize)

        return df

    def get_current_price(self) -> Optional[float]:
        if not self.is_available:
            return None
        try:
            ticker = self._yf.Ticker(self.symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception as e:
            logger.warning("[Yahoo] Price fetch failed: %s", e)
        return None


class AlphaVantageSource:
    """Alpha Vantage API — backup source (25 free calls/day)."""

    BASE_URL = "https://www.alphavantage.co/query"
    INTERVAL_MAP = {
        "15m": "15min",
        "1h": "60min",
        "1d": "daily",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.symbol = "XAUUSD"
        self.name = "AlphaVantage"
        self._session = requests.Session()
        self._api_calls_today = 0
        self._api_day = datetime.now(timezone.utc).date()

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def fetch_bars(self, interval: str, outputsize: int = 500) -> pd.DataFrame:
        """Fetch OHLCV bars from Alpha Vantage."""
        if not self.api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY not set")

        # Track daily usage (25/day free tier)
        today = datetime.now(timezone.utc).date()
        if today != self._api_day:
            self._api_calls_today = 0
            self._api_day = today
        self._api_calls_today += 1

        if self._api_calls_today > 23:
            logger.warning("[AlphaVantage] Approaching daily limit: %d/25",
                           self._api_calls_today)

        av_interval = self.INTERVAL_MAP.get(interval, "15min")

        if interval == "1d":
            function = "FX_DAILY"
            params = {
                "function": function,
                "from_symbol": "XAU",
                "to_symbol": "USD",
                "outputsize": "full" if outputsize > 100 else "compact",
                "apikey": self.api_key,
            }
            time_key = "Time Series FX (Daily)"
        else:
            function = "FX_INTRADAY"
            params = {
                "function": function,
                "from_symbol": "XAU",
                "to_symbol": "USD",
                "interval": av_interval,
                "outputsize": "full" if outputsize > 100 else "compact",
                "apikey": self.api_key,
            }
            time_key = f"Time Series FX (Intraday)"

        resp = self._session.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Alpha Vantage error handling
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")

        # Find the time series key (varies by endpoint)
        ts_data = None
        for key in data:
            if "Time Series" in key:
                ts_data = data[key]
                break

        if not ts_data:
            return pd.DataFrame()

        # Convert to DataFrame
        rows = []
        for dt_str, values in ts_data.items():
            rows.append({
                "datetime": dt_str,
                "Open": float(values.get("1. open", 0)),
                "High": float(values.get("2. high", 0)),
                "Low": float(values.get("3. low", 0)),
                "Close": float(values.get("4. close", 0)),
                "Volume": 0.0,  # Forex doesn't have volume on AV
            })

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        if len(df) > outputsize:
            df = df.tail(outputsize)

        return df

    def get_current_price(self) -> Optional[float]:
        if not self.api_key:
            return None
        try:
            resp = self._session.get(self.BASE_URL, params={
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": "XAU",
                "to_currency": "USD",
                "apikey": self.api_key,
            }, timeout=10)
            data = resp.json()
            rate_data = data.get("Realtime Currency Exchange Rate", {})
            price_str = rate_data.get("5. Exchange Rate")
            if price_str:
                return float(price_str)
        except Exception as e:
            logger.warning("[AlphaVantage] Price fetch failed: %s", e)
        return None


# ============================================================
# MULTI-SOURCE DATA CONNECTOR (main interface)
# ============================================================

class DataConnector:
    """
    Multi-source data connector with automatic failover.
    Tries sources in priority order: TwelveData → Yahoo → AlphaVantage.

    Simple 2-minute cache: Fetches fresh data every 2 minutes for
    up-to-date prices while keeping API usage reasonable (~720 calls/day).
    """

    # Map timeframe string to candle duration in minutes
    TIMEFRAME_MINUTES = {
        "15m": 15, "1h": 60, "4h": 240, "1d": 1440,
    }

    def __init__(self, symbol: str = "GC=F", timeframe: str = "15m",
                 cache_minutes: float = 0.33):
        self.yahoo_symbol = symbol
        self.symbol = "XAU/USD"
        self.timeframe = timeframe
        self.cache_minutes = cache_minutes
        self._candle_minutes = self.TIMEFRAME_MINUTES.get(timeframe, 15)

        # Caches
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None
        self._htf_cache: Optional[pd.DataFrame] = None
        self._htf_cache_time: Optional[datetime] = None
        self._htf_cache_minutes = 15
        self._consecutive_failures = 0

        # Initialize data sources
        td_key = os.getenv("TWELVEDATA_API_KEY", "")
        av_key = os.getenv("ALPHAVANTAGE_API_KEY", "")

        self._twelve_data = TwelveDataSource(td_key)
        self._yahoo = YahooFinanceSource()
        self._alpha_vantage = AlphaVantageSource(av_key)

        # Build source priority list (only available sources)
        self._sources: List = []
        self._active_source_name = "none"
        self._rebuild_source_list()

        # Log available sources
        available = [s.name for s in self._sources]
        logger.info("Data sources available: %s", ", ".join(available) if available else "NONE!")
        if not available:
            logger.critical(
                "NO DATA SOURCES AVAILABLE! Set TWELVEDATA_API_KEY or "
                "install yfinance (pip install yfinance) or set ALPHAVANTAGE_API_KEY"
            )

    def _rebuild_source_list(self):
        """Rebuild ordered list of available sources."""
        self._sources = []
        if self._twelve_data.is_available:
            self._sources.append(self._twelve_data)
        if self._yahoo.is_available:
            self._sources.append(self._yahoo)
        if self._alpha_vantage.is_available:
            self._sources.append(self._alpha_vantage)

    def _fetch_with_failover(self, interval: str, outputsize: int) -> Tuple[pd.DataFrame, str]:
        """
        Try each data source in priority order.
        Returns (DataFrame, source_name) on success.
        Raises last exception if all fail.
        """
        last_error = None

        for source in self._sources:
            try:
                df = source.fetch_bars(interval, outputsize)
                if not df.empty:
                    if self._active_source_name != source.name:
                        logger.info("Data source: switched to %s", source.name)
                        self._active_source_name = source.name
                    return df, source.name
                else:
                    logger.warning("[%s] Returned empty data, trying next source",
                                   source.name)
            except Exception as e:
                last_error = e
                logger.warning("[%s] Failed: %s — trying next source",
                               source.name, e)

        if last_error:
            raise last_error
        raise ValueError("No data sources available")

    def _should_fetch_fresh(self) -> bool:
        """
        Simple 2-minute cache: fetch fresh data every 2 minutes.
        This gives the bot up-to-date price data for trade management
        while keeping API usage reasonable (~720 calls/day max).
        """
        if self._cache is None or self._cache_time is None:
            return True  # No cache, must fetch

        age_seconds = (datetime.now(timezone.utc) - self._cache_time).total_seconds()
        if age_seconds >= 120:  # 2 minutes
            return True

        return False

    def fetch_bars(self, lookback_days: int = 7,
                   min_bars: int = 0,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch M15 bars with failover and 2-minute caching.

        Fetches fresh data every 2 minutes for up-to-date prices.
        Between fetches, returns cached data for trade management.
        """
        # Simple 2-minute cache: reuse cached data if fresh enough
        if use_cache and self._cache is not None and self._cache_time:
            if not self._should_fetch_fresh():
                age = (datetime.now(timezone.utc) - self._cache_time).total_seconds()
                logger.debug("Using cached data (age: %.0fs, next fetch in %.0fs)",
                             age, 120 - age)
                return self._cache.copy()

        logger.info("Fetching fresh data from API (2-min interval)")

        # Calculate bars needed (~88 M15 bars per trading day)
        bars_needed = max(min_bars, int(lookback_days * 88))
        bars_needed = min(bars_needed, 5000)

        for attempt in range(MAX_RETRIES):
            try:
                df, source = self._fetch_with_failover(
                    self.timeframe, bars_needed
                )

                if df.empty:
                    logger.warning("All sources returned empty (attempt %d/%d)",
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

                logger.info("Fetched %d %s bars from %s",
                            len(df), self.timeframe, source)
                return df

            except Exception as e:
                self._consecutive_failures += 1
                logger.error("Data fetch error (attempt %d/%d): %s",
                             attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        # All retries exhausted
        if self._cache is not None:
            logger.warning("All retries failed, returning stale cache "
                           "(%d consecutive failures)",
                           self._consecutive_failures)
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
        """Get latest price — tries each source in order."""
        for source in self._sources:
            try:
                price = source.get_current_price()
                if price and price > 0:
                    return price
            except Exception:
                continue

        # Fallback to cached close
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
        """
        Check if gold futures market is likely open.
        Gold trades Sun 22:00 UTC through Fri 21:00 UTC.
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        if weekday == 5:  # Saturday
            return False
        if weekday == 6 and hour < 22:  # Sunday before 10pm UTC
            return False
        if weekday == 4 and hour >= 22:  # Friday after 10pm UTC
            return False
        return True

    def fetch_htf_bars(self, timeframe: str = "1h",
                       lookback_days: int = 30) -> pd.DataFrame:
        """Fetch higher timeframe bars with failover and caching."""
        # Check HTF cache
        if self._htf_cache is not None and self._htf_cache_time:
            age = (datetime.now(timezone.utc) - self._htf_cache_time).total_seconds()
            if age < self._htf_cache_minutes * 60:
                return self._htf_cache.copy()

        bars_needed = min(lookback_days * 22, 5000)

        for attempt in range(MAX_RETRIES):
            try:
                df, source = self._fetch_with_failover(timeframe, bars_needed)

                if df.empty:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    if self._htf_cache is not None:
                        return self._htf_cache.copy()
                    return pd.DataFrame()

                df.dropna(inplace=True)
                self._htf_cache = df.copy()
                self._htf_cache_time = datetime.now(timezone.utc)
                return df

            except Exception as e:
                logger.error("HTF fetch error (attempt %d): %s", attempt + 1, e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        if self._htf_cache is not None:
            return self._htf_cache.copy()
        return pd.DataFrame()

    def preload_bars(self, min_bars: int = 400) -> pd.DataFrame:
        """Preload historical bars for S/R zone detection."""
        logger.info("Preloading %d+ M15 bars...", min_bars)

        request_size = min(min_bars + 100, 5000)

        try:
            df, source = self._fetch_with_failover(self.timeframe, request_size)
        except Exception as e:
            logger.error("Preload failed from all sources: %s", e)
            return pd.DataFrame()

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
            logger.info("Preloaded %d bars from %s — ready", len(df), source)
        else:
            logger.warning("Only preloaded %d bars from %s (target: %d)",
                           len(df), source, min_bars)
        return df

    @property
    def is_healthy(self) -> bool:
        return self._consecutive_failures < MAX_RETRIES * 2

    @property
    def active_source(self) -> str:
        return self._active_source_name

    @property
    def available_sources(self) -> List[str]:
        return [s.name for s in self._sources]
