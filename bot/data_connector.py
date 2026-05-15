# bot/data_connector.py
"""
Backtester data adapter — wraps MT5Connector to expose the API surface that
backtester.py expects: fetch_bars(lookback_days, min_bars, use_cache),
fetch_htf_bars(timeframe, lookback_days), and active_source attribute.

This shim exists so `python backtester.py` runs on the Windows VPS where the
MetaTrader5 module is importable. The live bot (main.py) uses MT5Connector
directly — there is no double-import or runtime overhead in production.

⚠️ ENV REQUIRED: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER must be set before
running the backtester. Same env vars as RUN_BOT.ps1.
"""
import logging
import pandas as pd
from typing import Optional

from mt5_connector import MT5Connector

logger = logging.getLogger(__name__)


class DataConnector:
    """Thin adapter wrapping MT5Connector for backtester.py."""

    def __init__(self, symbol: str, timeframe: str):
        # Reuse BotConfig.from_env() so MT5 credentials + symbol/timeframe
        # match the live bot exactly. Single source of truth for connection.
        from config import BotConfig
        cfg = BotConfig.from_env()
        cfg.mt5.symbol = symbol
        cfg.strategy.symbol = symbol
        cfg.strategy.timeframe = timeframe

        self._mt5 = MT5Connector(cfg.mt5, cfg.strategy, telegram=None)
        if not self._mt5.connect():
            raise RuntimeError(
                "DataConnector: MT5 connect failed — "
                "verify MT5_LOGIN/MT5_PASSWORD/MT5_SERVER env vars."
            )
        self.active_source = "MT5"
        logger.info("DataConnector ready: symbol=%s timeframe=%s source=MT5",
                    symbol, timeframe)

    def fetch_bars(self, lookback_days: int = 7, min_bars: int = 672,
                   use_cache: bool = False) -> pd.DataFrame:
        """Delegate to MT5Connector.fetch_bars (M15 by default)."""
        return self._mt5.fetch_bars(
            lookback_days=lookback_days,
            min_bars=min_bars,
            use_cache=use_cache,
        )

    def fetch_htf_bars(self, timeframe: str = "1h",
                       lookback_days: int = 30) -> pd.DataFrame:
        """Delegate to MT5Connector.fetch_htf_bars for higher-TF zone stacking."""
        return self._mt5.fetch_htf_bars(
            timeframe=timeframe,
            lookback_days=lookback_days,
        )

    def disconnect(self):
        """Clean shutdown — call at end of backtest run."""
        try:
            self._mt5.disconnect()
        except Exception as e:
            logger.warning("DataConnector disconnect error: %s", e)
