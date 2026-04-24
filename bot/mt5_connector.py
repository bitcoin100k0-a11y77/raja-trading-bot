# mt5_connector.py — MT5 Data Connector for ANi's FX Bot
"""
MT5 data feed — replaces TwelveData API.
Free unlimited data, real bid/ask spread, no rate limits.

Produces DataFrames identical to the old DataConnector:
  Columns: Open, High, Low, Close, Volume
  Index: DatetimeIndex (UTC)
"""
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Will fail gracefully with clear error

logger = logging.getLogger(__name__)

# MT5 timeframe mapping
TIMEFRAME_MAP = {
    "15m": "TIMEFRAME_M15",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
    "5m": "TIMEFRAME_M5",
    "30m": "TIMEFRAME_M30",
}

TIMEFRAME_MINUTES = {
    "15m": 15, "1h": 60, "4h": 240, "1d": 1440,
    "5m": 5, "30m": 30,
}

MAX_RECONNECT_ATTEMPTS = 10
BASE_RECONNECT_DELAY = 5  # seconds


class MT5Connector:
    """
    MT5 data connector — drop-in replacement for DataConnector.

    Same public interface:
      fetch_bars(), fetch_htf_bars(), preload_bars(),
      get_current_price(), is_market_open(), estimate_spread(),
      get_volume_sma(), is_healthy, active_source, available_sources
    """

    def __init__(self, mt5_config, strategy_config, telegram=None):
        """
        Args:
            mt5_config: MT5Config from config.py
            strategy_config: StrategyConfig from config.py
            telegram: TelegramNotifier instance (optional, for alerts)
        """
        if mt5 is None:
            raise RuntimeError(
                "MetaTrader5 package not installed! "
                "Run: pip install MetaTrader5"
            )

        self.mt5_cfg = mt5_config
        self.strategy_cfg = strategy_config
        self.telegram = telegram
        self.symbol = mt5_config.symbol
        self.timeframe = strategy_config.timeframe

        # Cache
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None
        self._htf_cache: Optional[pd.DataFrame] = None
        self._htf_cache_time: Optional[datetime] = None
        self._htf_cache_minutes = 15

        # Health tracking
        self._connected = False
        self._consecutive_failures = 0
        self._last_reconnect_attempt = 0

        # Symbol info (populated on connect)
        self._symbol_info = None
        self._point = 0.01  # default for XAUUSD
        self._digits = 2

    # ============================================================
    # CONNECTION MANAGEMENT
    # ============================================================

    def connect(self) -> bool:
        """
        🔁 RECONCILE — Initialize MT5 and log in.
        Must be called once on startup before any data fetching.
        """
        logger.info("Connecting to MT5...")

        # Build initialization kwargs
        init_kwargs = {"timeout": self.mt5_cfg.connect_timeout}
        if self.mt5_cfg.mt5_path:
            init_kwargs["path"] = self.mt5_cfg.mt5_path

        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            logger.error("MT5 initialize failed: %s", error)
            self._alert(f"❌ MT5 initialize FAILED: {error}")
            return False

        # Login if credentials provided
        if self.mt5_cfg.login and self.mt5_cfg.password:
            if not mt5.login(
                login=self.mt5_cfg.login,
                password=self.mt5_cfg.password,
                server=self.mt5_cfg.server,
            ):
                error = mt5.last_error()
                logger.error("MT5 login failed: %s", error)
                self._alert(f"❌ MT5 login FAILED: {error}")
                mt5.shutdown()
                return False

        # Validate symbol exists
        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.error("Symbol %s not found in MT5! Check MT5_SYMBOL env var.", self.symbol)
            self._alert(f"❌ Symbol {self.symbol} not found in MT5!")
            mt5.shutdown()
            return False

        # Ensure symbol is visible in Market Watch
        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error("Failed to enable %s in Market Watch", self.symbol)
                mt5.shutdown()
                return False

        # Store symbol info
        self._symbol_info = info
        self._point = info.point
        self._digits = info.digits

        # Get account info
        account = mt5.account_info()
        if account is None:
            logger.error("Failed to get account info")
            mt5.shutdown()
            return False

        self._connected = True
        self._consecutive_failures = 0

        logger.info("✅ MT5 connected successfully")
        logger.info("  Account: %d (%s)", account.login, account.company)
        logger.info("  Balance: $%.2f | Equity: $%.2f", account.balance, account.equity)
        logger.info("  Leverage: 1:%d", account.leverage)
        logger.info("  Symbol: %s (point=%.5f, digits=%d)",
                     self.symbol, self._point, self._digits)
        logger.info("  Min lot: %.2f | Step: %.2f | Max lot: %.2f",
                     info.volume_min, info.volume_step, info.volume_max)

        # 📲 ALERT — startup connection success
        self._alert(
            f"✅ MT5 Connected\n"
            f"Account: {account.login} ({account.company})\n"
            f"Balance: ${account.balance:.2f}\n"
            f"Symbol: {self.symbol}"
        )

        return True

    def disconnect(self):
        """Shutdown MT5 connection."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def ensure_connected(self) -> bool:
        """
        Check connection and auto-reconnect if needed.
        Uses exponential backoff: 5s, 10s, 20s, 40s, max 60s.
        """
        if self._connected:
            # Quick health check — try to get account info
            try:
                account = mt5.account_info()
                if account is not None:
                    return True
            except Exception:
                pass

            # Connection lost
            self._connected = False
            logger.warning("MT5 connection lost!")
            self._alert("⚠️ MT5 connection LOST — attempting reconnect...")  # 📲 ALERT

        # Exponential backoff
        now = time.time()
        delay = min(BASE_RECONNECT_DELAY * (2 ** self._consecutive_failures), 60)
        if now - self._last_reconnect_attempt < delay:
            return False

        self._last_reconnect_attempt = now
        self._consecutive_failures += 1

        if self._consecutive_failures > MAX_RECONNECT_ATTEMPTS:
            logger.critical("MT5 reconnect failed %d times — giving up",
                          self._consecutive_failures)
            self._alert(
                f"🔴 MT5 RECONNECT FAILED {self._consecutive_failures} times!\n"
                "Bot is PAUSED. Check VPS and MT5 terminal."
            )  # 📲 ALERT
            return False

        logger.info("Reconnect attempt %d/%d (backoff: %.0fs)...",
                     self._consecutive_failures, MAX_RECONNECT_ATTEMPTS, delay)

        # Shutdown and reinitialize
        try:
            mt5.shutdown()
        except Exception:
            pass

        if self.connect():
            logger.info("✅ MT5 reconnected after %d attempts", self._consecutive_failures)
            self._alert("✅ MT5 reconnected successfully")  # 📲 ALERT
            self._consecutive_failures = 0
            return True

        return False

    # ============================================================
    # DATA FETCHING (same interface as DataConnector)
    # ============================================================

    def _get_mt5_timeframe(self, tf_string: str):
        """Convert string timeframe to MT5 constant."""
        mapping = {
            "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
            "5m": mt5.TIMEFRAME_M5,
            "30m": mt5.TIMEFRAME_M30,
        }
        return mapping.get(tf_string, mt5.TIMEFRAME_M15)

    def _rates_to_dataframe(self, rates) -> pd.DataFrame:
        """
        Convert MT5 rates array to DataFrame matching DataConnector format.
        Output: columns [Open, High, Low, Close, Volume], DatetimeIndex (UTC).
        """
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)

        # MT5 returns 'time' as epoch seconds
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)

        # Rename to match existing bot's expected column names
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        }, inplace=True)

        # Keep only the columns the bot expects
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Convert to float
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        df["Volume"] = df["Volume"].astype(float)

        df.sort_index(inplace=True)
        return df

    def _add_helper_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add body, range, body_ratio, bullish columns (same as DataConnector)."""
        df["body"] = abs(df["Close"] - df["Open"])
        df["range"] = df["High"] - df["Low"]
        df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
        df["body_ratio"] = df["body_ratio"].fillna(0)
        df["bullish"] = df["Close"] > df["Open"]
        return df

    def fetch_bars(self, lookback_days: int = 7,
                   min_bars: int = 0,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch M15 bars from MT5 with 2-minute caching.
        Same interface as DataConnector.fetch_bars().
        """
        # Check cache (same 2-minute logic as DataConnector)
        if use_cache and self._cache is not None and self._cache_time:
            age = (datetime.now(timezone.utc) - self._cache_time).total_seconds()
            if age < 1:  # 1 second — fresh data every tick (tick loop is 30s, so ~1 call/30s)
                logger.debug("Using cached MT5 data (age: %.0fs)", age)
                return self._cache.copy()

        if not self.ensure_connected():
            logger.warning("MT5 not connected — returning stale cache")
            if self._cache is not None:
                return self._cache.copy()
            return pd.DataFrame()

        # Calculate bars needed — MT5's own formula (full 24h coverage)
        candle_minutes = TIMEFRAME_MINUTES.get(self.timeframe, 15)
        bars_needed = max(min_bars, int(lookback_days * 24 * 60 / candle_minutes))
        bars_needed = min(bars_needed, 10000)  # MT5 can handle more than TwelveData

        logger.info("Fetching %d %s bars from MT5...", bars_needed, self.timeframe)

        try:
            mt5_tf = self._get_mt5_timeframe(self.timeframe)
            rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, bars_needed)

            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logger.error("MT5 fetch_bars failed: %s", error)
                self._consecutive_failures += 1
                if self._cache is not None:
                    logger.info("Returning stale cache")
                    return self._cache.copy()
                return pd.DataFrame()

            df = self._rates_to_dataframe(rates)
            df.dropna(inplace=True)
            df = self._add_helper_columns(df)

            # Update cache
            self._cache = df.copy()
            self._cache_time = datetime.now(timezone.utc)
            self._consecutive_failures = 0

            logger.info("Fetched %d %s bars from MT5", len(df), self.timeframe)
            return df

        except Exception as e:
            logger.error("MT5 fetch_bars exception: %s", e, exc_info=True)
            self._consecutive_failures += 1
            if self._cache is not None:
                return self._cache.copy()
            return pd.DataFrame()

    def fetch_htf_bars(self, timeframe: str = "1h",
                       lookback_days: int = 30) -> pd.DataFrame:
        """Fetch higher timeframe bars (e.g., 1H for zone stacking)."""
        # Check HTF cache
        if self._htf_cache is not None and self._htf_cache_time:
            age = (datetime.now(timezone.utc) - self._htf_cache_time).total_seconds()
            if age < self._htf_cache_minutes * 60:
                return self._htf_cache.copy()

        if not self.ensure_connected():
            if self._htf_cache is not None:
                return self._htf_cache.copy()
            return pd.DataFrame()

        candle_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
        bars_needed = min(int(lookback_days * 24 * 60 / candle_minutes), 10000)

        try:
            mt5_tf = self._get_mt5_timeframe(timeframe)
            rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, bars_needed)

            if rates is None or len(rates) == 0:
                if self._htf_cache is not None:
                    return self._htf_cache.copy()
                return pd.DataFrame()

            df = self._rates_to_dataframe(rates)
            df.dropna(inplace=True)

            self._htf_cache = df.copy()
            self._htf_cache_time = datetime.now(timezone.utc)
            return df

        except Exception as e:
            logger.error("MT5 fetch_htf_bars exception: %s", e)
            if self._htf_cache is not None:
                return self._htf_cache.copy()
            return pd.DataFrame()

    def preload_bars(self, min_bars: int = 400) -> pd.DataFrame:
        """
        🔁 RECONCILE — Preload historical bars on startup for S/R zones.
        """
        logger.info("Preloading %d+ M15 bars from MT5...", min_bars)

        if not self.ensure_connected():
            logger.error("Cannot preload — MT5 not connected!")
            return pd.DataFrame()

        request_size = min(min_bars + 100, 10000)

        try:
            mt5_tf = self._get_mt5_timeframe(self.timeframe)
            rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, request_size)

            if rates is None or len(rates) == 0:
                logger.error("Preload FAILED — no data from MT5!")
                return pd.DataFrame()

            df = self._rates_to_dataframe(rates)
            df.dropna(inplace=True)
            df = self._add_helper_columns(df)

            # Cache
            self._cache = df.copy()
            self._cache_time = datetime.now(timezone.utc)
            self._consecutive_failures = 0

            if len(df) >= min_bars:
                logger.info("✅ Preloaded %d bars from MT5 — ready", len(df))
            else:
                logger.warning("Only preloaded %d bars (target: %d)", len(df), min_bars)

            return df

        except Exception as e:
            logger.error("Preload exception: %s", e, exc_info=True)
            return pd.DataFrame()

    def get_current_price(self) -> Optional[float]:
        """Get current price from MT5 tick data."""
        if not self.ensure_connected():
            # Fallback to cached close
            if self._cache is not None and not self._cache.empty:
                return float(self._cache["Close"].iloc[-1])
            return None

        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is not None:
                # Use bid for current price (sell side, conservative)
                return float(tick.bid)
        except Exception as e:
            logger.warning("MT5 price fetch failed: %s", e)

        # Fallback to cached close
        if self._cache is not None and not self._cache.empty:
            return float(self._cache["Close"].iloc[-1])
        return None

    def get_bid_ask(self) -> Tuple[float, float]:
        """Get current bid/ask prices from MT5."""
        if not self.ensure_connected():
            return (0.0, 0.0)

        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is not None:
                return (float(tick.bid), float(tick.ask))
        except Exception as e:
            logger.warning("MT5 bid/ask fetch failed: %s", e)
        return (0.0, 0.0)

    def get_latest_bar(self) -> Optional[pd.Series]:
        """Get the latest completed bar."""
        df = self.fetch_bars(lookback_days=2)
        if df.empty:
            return None
        if len(df) >= 2:
            return df.iloc[-2]
        return df.iloc[-1]

    def estimate_spread(self, df: Optional[pd.DataFrame] = None,
                        window: int = 20) -> float:
        """
        Get real spread from MT5 tick data (much better than TwelveData estimation).
        """
        if self._connected:
            try:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is not None:
                    spread_price = tick.ask - tick.bid
                    spread_pips = spread_price / 0.10  # gold pip = $0.10
                    return max(0.1, min(spread_pips, 20.0))
            except Exception:
                pass

        # Fallback: estimate from wicks (same as DataConnector)
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
        """Get volume SMA."""
        if df is None:
            df = self.fetch_bars(lookback_days=5)
        if df.empty or len(df) < period:
            return 0.0
        return float(df["Volume"].tail(period).mean())

    def is_market_open(self) -> bool:
        """
        Check if market is open using MT5 symbol trade mode.
        More accurate than time-based check.
        """
        if not self._connected:
            # Fallback to time-based check (same as DataConnector)
            return self._time_based_market_check()

        try:
            info = mt5.symbol_info(self.symbol)
            if info is not None:
                # trade_mode: 0 = disabled, 4 = full
                return info.trade_mode != 0
        except Exception:
            pass

        return self._time_based_market_check()

    def _time_based_market_check(self) -> bool:
        """Fallback market open check (same logic as DataConnector)."""
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

    def get_symbol_info(self) -> dict:
        """Get MT5 symbol info for lot size validation."""
        if self._symbol_info is None and self._connected:
            self._symbol_info = mt5.symbol_info(self.symbol)

        if self._symbol_info:
            return {
                "point": self._symbol_info.point,
                "digits": self._symbol_info.digits,
                "stops_level": self._symbol_info.trade_stops_level,
                "freeze_level": self._symbol_info.trade_freeze_level,
                "volume_min": self._symbol_info.volume_min,
                "volume_max": self._symbol_info.volume_max,
                "volume_step": self._symbol_info.volume_step,
                "trade_mode": self._symbol_info.trade_mode,
            }
        return {
            "point": 0.01, "digits": 2,
            "stops_level": 0, "freeze_level": 0,
            "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
            "trade_mode": 0,
        }

    # ============================================================
    # PROPERTIES (same as DataConnector)
    # ============================================================

    @property
    def is_healthy(self) -> bool:
        return self._connected and self._consecutive_failures < 6

    @property
    def active_source(self) -> str:
        return "MT5" if self._connected else "none"

    @property
    def available_sources(self) -> List[str]:
        if self._connected:
            return ["MT5"]
        return []

    # ============================================================
    # HELPERS
    # ============================================================

    def _alert(self, message: str):
        """📲 ALERT — Send Telegram notification."""
        if self.telegram:
            try:
                self.telegram.notify_error(message)
            except Exception as e:
                logger.warning("Telegram alert failed: %s", e)
