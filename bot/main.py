"""
Raja Banks Trading Bot - Main Engine
Production-ready trading loop with health checks for Railway deployment.

Features:
- M15 bar-driven trading loop
- Health check HTTP server (Railway requires this)
- Graceful shutdown with trade cleanup
- Daily summary at session end
- Persistent learning across restarts
- Auto-recovery from data outages
"""
import os
import sys
import time
import signal
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from typing import Optional

from config import BotConfig
from database import TradingDatabase
from data_connector import DataConnector
from market_analyzer import MarketAnalyzer
from strategy import Strategy
from risk_manager import RiskManager
from trade_manager import TradeManager
from paper_trader import PaperTrader
from learning_agent import LearningAgent
from telegram_notifier import TelegramNotifier
from utils import get_session, is_tradeable_session, body_ratio, utc_now


# === LOGGING SETUP ===

def setup_logging(level: str = "INFO", data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(data_dir, "bot.log"), mode="a"),
    ]

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=log_format, handlers=handlers)

    # Suppress noisy libraries
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


logger = logging.getLogger("TradingBot")


# === HEALTH CHECK SERVER ===

class HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for Railway health checks."""
    bot_ref = None

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            status = self._get_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            import json
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _get_status(self):
        if HealthHandler.bot_ref:
            bot = HealthHandler.bot_ref
            return {
                "status": "running" if bot.running else "stopped",
                "uptime_seconds": int(time.time() - bot._start_time),
                "balance": round(bot.paper.balance, 2),
                "total_trades": bot.db.get_total_trades(),
                "trades_today": bot.db.get_trades_today(),
                "win_rate": round(bot.db.get_win_rate() * 100, 1),
                "open_trades": bot.trade_mgr.get_active_trade_count(),
                "data_healthy": bot.data.is_healthy,
                "last_bar": bot._last_bar_time.isoformat() if bot._last_bar_time else None,
                "patterns_blocked": len(bot.db.get_blocked_patterns()),
            }
        return {"status": "initializing"}

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def start_health_server(port: int, bot):
    """Start health check HTTP server in background thread."""
    HealthHandler.bot_ref = bot
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Health check server started on port %d", port)
    return server


# === MAIN BOT ===

class TradingBot:
    """
    Main bot orchestrator implementing Raja Banks Market Fluidity Strategy.

    Loop every 30s:
    1. Fetch M15 bars from Yahoo Finance
    2. On new bar: run full analysis pipeline
    3. Manage active trades (SL stages, TP, exits)
    4. Generate new signals with learning agent filter
    5. Execute in paper trading mode
    6. Log everything, learn from results
    """

    def __init__(self, config: BotConfig = None):
        self.cfg = config or BotConfig.from_env()
        self.running = False
        self._start_time = time.time()

        # Ensure data directory exists
        os.makedirs(self.cfg.data_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Initializing Raja Banks Trading Bot v1.0")
        logger.info("Mode: %s", "PAPER TRADING" if self.cfg.dry_run else "LIVE")
        logger.info("Symbol: %s | Timeframe: %s",
                     self.cfg.strategy.symbol, self.cfg.strategy.timeframe)
        logger.info("=" * 60)

        # Initialize components
        self.db = TradingDatabase(
            db_path=os.path.join(self.cfg.data_dir, "trading_bot.db")
        )
        self.data = DataConnector(
            symbol=self.cfg.strategy.symbol,
            timeframe=self.cfg.strategy.timeframe,
        )
        self.analyzer = MarketAnalyzer(self.cfg.strategy)
        self.strategy = Strategy(self.cfg.strategy, self.analyzer)
        self.risk_mgr = RiskManager(self.cfg.risk, self.cfg.strategy, self.db)
        self.trade_mgr = TradeManager(self.cfg.strategy)
        self.paper = PaperTrader(self.cfg, self.db, self.risk_mgr, self.trade_mgr)
        self.learner = LearningAgent(self.cfg.learning, self.db)
        self.telegram = TelegramNotifier(self.cfg.telegram)

        self._last_bar_time: Optional[datetime] = None
        self._status_interval = self.cfg.telegram.status_interval_minutes * 60
        self._last_status_time = 0
        self._last_daily_summary_date = None
        self._health_server = None

        logger.info("Balance: $%.2f | Risk: %.1f%%",
                     self.paper.balance, self.cfg.risk.risk_percent)
        logger.info("Learning agent: %d patterns tracked, %d blocked",
                     len(self.db.get_all_patterns()),
                     len(self.db.get_blocked_patterns()))

    def start(self):
        """Start the main trading loop with health check server."""
        self.running = True
        self._start_time = time.time()

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Start health check server
        try:
            self._health_server = start_health_server(
                self.cfg.health_check_port, self
            )
        except OSError as e:
            logger.warning("Could not start health server: %s", e)

        # === PRELOAD 400 M15 BARS FOR S/R ZONES ===
        self._preload_market_data()

        # Startup notification
        learning = self.db.get_learning_summary()
        self.telegram.notify_startup({
            "symbol": self.cfg.strategy.symbol,
            "dry_run": self.cfg.dry_run,
            "balance": self.paper.balance,
            "risk_pct": self.cfg.risk.risk_percent,
            "max_trades": self.cfg.risk.max_trades_per_day,
            "dd_limit": self.cfg.risk.daily_loss_limit,
            "patterns": learning["patterns_tracked"],
            "blocked": learning["patterns_blocked"],
        })

        logger.info("Bot STARTED - entering main loop")

        # Main loop
        while self.running:
            try:
                self._tick()
            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                self.telegram.notify_error(str(e))

            # Periodic hourly status — runs regardless of market state or new bar
            try:
                self._check_periodic_status()
            except Exception as e:
                logger.error("Error in periodic status: %s", e)

            time.sleep(self.cfg.update_interval_seconds)

        logger.info("Bot stopped.")

    def _preload_market_data(self):
        """
        Preload 400 M15 bars on startup so the bot can immediately
        detect S/R zones, trend structure, and be ready for trades.
        """
        logger.info("=" * 50)
        logger.info("PRELOADING MARKET DATA (400 M15 bars)...")
        logger.info("=" * 50)

        # Fetch 400+ M15 bars
        df = self.data.preload_bars(min_bars=400)
        if df.empty:
            logger.error("PRELOAD FAILED - no historical data available!")
            self.telegram.notify_error(
                "Startup preload failed: no historical data. "
                "Bot will try to build zones on next tick."
            )
            return

        price = float(df["Close"].iloc[-1])

        # Fetch HTF (1H) data for zone stacking
        htf_df = self.data.fetch_htf_bars(timeframe="1h", lookback_days=30)

        # Run full market analysis on preloaded data
        market = self.analyzer.analyze(df, htf_df, price)

        # Store last bar time so _tick knows where we left off
        self._last_bar_time = df.index[-1]

        # Log the preloaded market state
        logger.info("=" * 50)
        logger.info("PRELOAD COMPLETE")
        logger.info("Bars loaded: %d | Price: $%.2f", len(df), price)
        logger.info("Trend: %s | Session: %s | Volume: %s",
                     market.trend.value, market.session.value,
                     market.current_volume_level.value)
        logger.info("Support zones: %d | Resistance zones: %d",
                     len(market.support_levels), len(market.resistance_levels))

        for i, s in enumerate(market.support_levels):
            logger.info("  SUP %d: $%.2f (touches=%d, htf=%s, str=%.1f)",
                         i + 1, s.price, s.touches, s.htf_aligned, s.strength)
        for i, r in enumerate(market.resistance_levels):
            logger.info("  RES %d: $%.2f (touches=%d, htf=%s, str=%.1f)",
                         i + 1, r.price, r.touches, r.htf_aligned, r.strength)

        logger.info("Swing highs: %s", [f"${h:.2f}" for h in market.swing_highs])
        logger.info("Swing lows: %s", [f"${l:.2f}" for l in market.swing_lows])
        logger.info("Volatility: %.1f pips | Avg body: $%.2f",
                     market.volatility_pips, market.avg_body_size)
        logger.info("=" * 50)

        # Send Telegram notification with preloaded zones
        sup_text = "\n".join(
            f"  ${s.price:.2f} ({s.touches}t"
            f"{' HTF' if s.htf_aligned else ''}"
            f" str={s.strength:.1f})"
            for s in market.support_levels
        ) or "  None found"

        res_text = "\n".join(
            f"  ${r.price:.2f} ({r.touches}t"
            f"{' HTF' if r.htf_aligned else ''}"
            f" str={r.strength:.1f})"
            for r in market.resistance_levels
        ) or "  None found"

        self.telegram.notify_status({
            "preload": True,
            "bars_loaded": len(df),
            "price": price,
            "trend": market.trend.value,
            "session": market.session.value,
            "volume": market.current_volume_level.value,
            "support_count": len(market.support_levels),
            "resistance_count": len(market.resistance_levels),
            "support_text": sup_text,
            "resistance_text": res_text,
            "volatility_pips": market.volatility_pips,
        })

    def _tick(self):
        """Single iteration of the main loop."""

        # Skip if market closed
        if not self.data.is_market_open():
            self._check_daily_summary()
            return

        # 1. Fetch data (10 days to maintain 400+ bars for S/R)
        df = self.data.fetch_bars(lookback_days=10)
        if df.empty:
            logger.warning("No data available, skipping tick")
            return

        # Check for new bar
        latest_time = df.index[-1]
        if self._last_bar_time and latest_time <= self._last_bar_time:
            # Same bar — only update active trades with latest price
            price = float(df["Close"].iloc[-1])
            self._manage_active_trades(price)
            return

        self._last_bar_time = latest_time
        utc_hour = latest_time.hour if hasattr(latest_time, 'hour') else 12
        price = float(df["Close"].iloc[-1])

        logger.info("--- New bar: %s | Price: $%.2f ---",
                     latest_time.strftime("%Y-%m-%d %H:%M"), price)

        # 2. Fetch HTF data
        htf_df = self.data.fetch_htf_bars(timeframe="1h", lookback_days=30)

        # 3. Market analysis
        market = self.analyzer.analyze(df, htf_df, price)

        logger.info("Market: trend=%s, session=%s, vol=%s | S/R: %d sup, %d res",
                     market.trend.value, market.session.value,
                     market.current_volume_level.value,
                     len(market.support_levels), len(market.resistance_levels))

        # 4. Manage active trades
        exit_events = self._manage_active_trades(price)

        # Check opposite close emergency exit
        if self.trade_mgr.get_active_trade_count() > 0 and len(df) >= 1:
            bar = df.iloc[-1]
            opp_exits = self.trade_mgr.check_opposite_close(
                float(bar["Open"]), float(bar["Close"]), price
            )
            if opp_exits:
                exit_events.extend(opp_exits)

        # Process exits
        if exit_events:
            self.paper.process_exits(exit_events)
            for event in exit_events:
                if event["full_close"]:
                    trade = self.db.get_trade(event["trade_id"])
                    if trade:
                        context = self.db.get_context(event["trade_id"])
                        self.learner.on_trade_closed(trade, context)
                        self.telegram.notify_trade_closed(trade)

        # 5. Generate new signals
        if is_tradeable_session(utc_hour, self.cfg.strategy.session_start,
                                self.cfg.strategy.session_end):

            blocked = [p["pattern_key"] for p in self.db.get_blocked_patterns()]
            signals = self.strategy.generate_signals(df, market, blocked)

            # 6. Filter through learning agent
            for sig in signals:
                trade_ctx = {
                    "entry_type": sig.entry_type.value,
                    "session_type": market.session.value,
                    "trend_type": market.trend.value,
                    "sr_touches": sig.sr_level.touches,
                    "c1_body_ratio": sig.context.get("c1_body_ratio", 0),
                    "impulse_body_ratio": sig.context.get("impulse_body_ratio", 0),
                }

                should_trade, confidence, reason = \
                    self.learner.should_take_trade(trade_ctx)

                if not should_trade:
                    logger.info("Learning agent rejected: %s (%.2f)",
                                reason, confidence)
                    continue

                sig.confidence = confidence

                # Notify signal
                self.telegram.notify_signal({
                    "direction": sig.direction.value,
                    "entry_type": sig.entry_type.value,
                    "entry_price": sig.entry_price,
                    "sl_price": sig.sl_price,
                    "tp1_price": sig.tp1_price,
                    "tp2_price": sig.tp2_price,
                    "risk_pips": sig.risk_pips,
                    "confidence": confidence,
                })

                # 7. Execute
                trade_id = self.paper.execute_signal(sig)
                if trade_id:
                    self.telegram.notify_trade_opened({
                        "trade_id": trade_id,
                        "direction": sig.direction.value,
                        "lot_size": self.risk_mgr.calculate_position_size(
                            self.paper.balance, sig.risk_pips
                        ),
                        "entry_price": sig.entry_price,
                        "risk_percent": self.risk_mgr.current_risk_percent,
                    })

        # Session end: close all
        if not is_tradeable_session(utc_hour, self.cfg.strategy.session_start,
                                    self.cfg.strategy.session_end):
            session_exits = self.trade_mgr.session_end_close_all(price)
            if session_exits:
                self.paper.process_exits(session_exits)
                for event in session_exits:
                    if event["full_close"]:
                        trade = self.db.get_trade(event["trade_id"])
                        if trade:
                            self.learner.on_trade_closed(trade)

        # Daily summary check
        self._check_daily_summary()

    def _check_periodic_status(self):
        """
        Send hourly Telegram status update — runs every tick regardless
        of market state or whether a new bar has arrived.
        """
        now = time.time()
        if now - self._last_status_time < self._status_interval:
            return

        self._last_status_time = now

        # Get bot performance status
        status = self.paper.get_status()

        # Try to add live market context
        try:
            price = self.data.get_current_price()
            if price:
                status["current_price"] = price

            # Quick market analysis for the status message
            df = self.data.fetch_bars(lookback_days=10)
            if not df.empty and len(df) >= 20:
                htf_df = self.data.fetch_htf_bars(timeframe="1h", lookback_days=30)
                current_price = float(df["Close"].iloc[-1])
                market = self.analyzer.analyze(df, htf_df, current_price)
                status["current_price"] = current_price
                status["trend"] = market.trend.value
                status["session"] = market.session.value
                status["volume"] = market.current_volume_level.value
                status["support_count"] = len(market.support_levels)
                status["resistance_count"] = len(market.resistance_levels)
                status["nearest_support"] = (
                    market.support_levels[0].price
                    if market.support_levels else None
                )
                status["nearest_resistance"] = (
                    market.resistance_levels[0].price
                    if market.resistance_levels else None
                )
        except Exception as e:
            logger.warning("Could not fetch market context for status: %s", e)

        # Send via Telegram
        self.telegram.notify_status(status)
        self._log_status(status)

        # Learning report if we have data
        report = self.learner.get_learning_report()
        if report["trades_analyzed"] > 0:
            self.telegram.notify_learning_report(report)

    def _manage_active_trades(self, price: float) -> list:
        """Update all active trades with current price."""
        return self.trade_mgr.update_trades(price)

    def _check_daily_summary(self):
        """Send daily summary at end of trading session."""
        now = utc_now()
        today = now.strftime("%Y-%m-%d")

        if self._last_daily_summary_date == today:
            return

        # Only send at session end (21:00+ UTC)
        if now.hour < self.cfg.strategy.session_end:
            return

        self._last_daily_summary_date = today

        # Gather daily stats
        trades_today = self.db.get_trades_today()
        daily_pnl = self.db.get_daily_pnl()

        if trades_today > 0:
            self.telegram.notify_daily_summary({
                "date": today,
                "trades": trades_today,
                "wins": 0,  # Could query from DB
                "losses": 0,
                "pnl_dollars": daily_pnl,
                "balance": self.paper.balance,
            })

            # Save daily stats
            self.db.upsert_daily_stats(today, {
                "trades_taken": trades_today,
                "pnl_dollars": daily_pnl,
                "balance_end": self.paper.balance,
            })

    def _log_status(self, status: dict):
        """Log current bot status."""
        logger.info("=" * 50)
        logger.info("STATUS: $%.2f (%.1f%%) | WR %.1f%% | Trades %d | Open %d",
                     status["balance"], status["total_pnl_pct"],
                     status["win_rate"], status["total_trades"],
                     status["open_trades"])
        risk = status["risk_status"]
        logger.info("Risk: %.2f%% | Losses: %d | Cooldown: %s",
                     risk["current_risk_pct"], risk["consecutive_losses"],
                     "YES" if risk["cooldown_active"] else "NO")
        logger.info("=" * 50)

    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler."""
        logger.info("Shutdown signal received...")
        self.running = False

        # Close all trades
        price = self.data.get_current_price()
        if price:
            exits = self.trade_mgr.session_end_close_all(price)
            if exits:
                self.paper.process_exits(exits)

        # Final status
        status = self.paper.get_status()
        self._log_status(status)
        self.telegram.notify_status(status)

        # Learning report
        report = self.learner.get_learning_report()
        self.telegram.notify_learning_report(report)

        self.telegram.notify_shutdown("Signal received")
        self.db.close()


def main():
    """Entry point for the trading bot."""
    config = BotConfig.from_env()
    setup_logging(config.log_level, config.data_dir)

    print("\n" + "=" * 60)
    print("  Raja Banks Trading Bot - Market Fluidity Strategy")
    print("  " + ("PAPER TRADING" if config.dry_run else "LIVE TRADING"))
    print(f"  Symbol: {config.strategy.symbol}")
    print(f"  Balance: ${config.initial_balance:,.2f}")
    print(f"  Risk: {config.risk.risk_percent}% per trade")
    print(f"  Max Trades/Day: {config.risk.max_trades_per_day}")
    print(f"  Health Check: port {config.health_check_port}")
    print("=" * 60 + "\n")

    bot = TradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()
