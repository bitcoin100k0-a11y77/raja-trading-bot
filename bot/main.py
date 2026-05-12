# main.py — ANi's FX Bot (MT5 Version)
"""
Raja Banks Trading Bot — MT5 Live/Demo Deployment
Windows VPS version using MetaTrader5 Python API.

Same strategy, same logic, different execution layer.
Replaces: TwelveData → MT5 data feed
Replaces: PaperTrader → MT5Executor (real orders)
Adds: BrokerReconciler (startup safety)

⚠️  ENV REQUIRED: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
📵 KILL SWITCH: LIVE_MODE=false (default) — must be explicitly "true"
"""
import os
import sys
import time
import signal
import logging
from datetime import datetime, timezone
from typing import Optional

from config import BotConfig
from database import TradingDatabase
from mt5_connector import MT5Connector
from market_analyzer import MarketAnalyzer
from strategy import Strategy
from risk_manager import RiskManager
from trade_manager import TradeManager
from mt5_executor import MT5Executor
from broker_reconciler import BrokerReconciler, ReconcileReport
from learning_agent import LearningAgent
from telegram_notifier import TelegramNotifier
from constants import TradeDirection, SLStage, ExitReason
from utils import get_session, is_tradeable_session, body_ratio, utc_now, price_to_pips


# === LOGGING SETUP ===

def setup_logging(level: str = "INFO", data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers = [
        logging.StreamHandler(sys.stdout),
        # encoding='utf-8' required on Windows — default cp1252 crashes on emoji
        logging.FileHandler(
            os.path.join(data_dir, "bot.log"), mode="a", encoding="utf-8"
        ),
    ]

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=log_format, handlers=handlers)

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


logger = logging.getLogger("TradingBot")


# === MAIN BOT ===

class TradingBot:
    """
    Main bot orchestrator — MT5 version.

    Same Raja Banks Market Fluidity Strategy.
    Same 30-second loop. Same SL staging, TP logic, learning agent.
    Different execution: MT5 API instead of paper simulation.

    📵 KILL SWITCH: LIVE_MODE env var controls real vs shadow execution.
    """

    def __init__(self, config: BotConfig = None):
        self.cfg = config or BotConfig.from_env()
        self.running = False
        self._start_time = time.time()

        # Ensure data directory exists
        os.makedirs(self.cfg.data_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Initializing ANi's FX Bot v2.0 (MT5)")
        logger.info("Mode: %s", "🔴 LIVE" if self.cfg.live_mode else "🔒 SHADOW (no real orders)")
        logger.info("DRY_RUN: %s", self.cfg.dry_run)
        logger.info("Symbol: %s | Timeframe: %s",
                     self.cfg.mt5.symbol, self.cfg.strategy.timeframe)
        logger.info("=" * 60)

        # === VALIDATE MT5 CREDENTIALS ===
        if not self.cfg.mt5.login or not self.cfg.mt5.password:
            logger.critical(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.critical(
                "MT5_LOGIN and MT5_PASSWORD are NOT SET!")
            logger.critical(
                "Set them as environment variables on this VPS.")
            logger.critical(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Don't exit — let user see the error in logs

        # Initialize components
        self.db = TradingDatabase(
            db_path=os.path.join(self.cfg.data_dir, "trading_bot.db")
        )

        # Telegram (initialize early so other components can use it)
        self.telegram = TelegramNotifier(self.cfg.telegram)

        # MT5 Data Connector (replaces TwelveData)
        self.data = MT5Connector(
            mt5_config=self.cfg.mt5,
            strategy_config=self.cfg.strategy,
            telegram=self.telegram,
        )

        self.analyzer = MarketAnalyzer(self.cfg.strategy)
        self.strategy = Strategy(self.cfg.strategy, self.analyzer)
        self.risk_mgr = RiskManager(self.cfg.risk, self.cfg.strategy, self.db)
        self.trade_mgr = TradeManager(self.cfg.strategy)

        # MT5 Executor (replaces PaperTrader)
        self.executor = MT5Executor(
            config=self.cfg,
            db=self.db,
            risk_mgr=self.risk_mgr,
            trade_mgr=self.trade_mgr,
            mt5_connector=self.data,
            telegram=self.telegram,
        )

        self.learner = LearningAgent(self.cfg.learning, self.db)

        self._last_bar_time: Optional[datetime] = None
        self._status_interval = self.cfg.telegram.status_interval_minutes * 60
        self._last_status_time = 0
        self._last_daily_summary_date = None

        # 🔴 NEW ENTRY MODEL — pending STOP orders at C1 extreme.
        # ticket → {"signal": Signal, "bars_waiting": int, "placed_at": datetime,
        #           "expiry_bars": int, "lot_size": float}
        # Cancelled if not filled within `expiry_bars` (1 bar = 15 min).
        # On fill: broker converts pending → position, reconciler detects the new
        # position and removes the ticket from this dict.
        self.pending_stop_orders: Dict[int, Dict] = {}

        # Reconciler — created in start() after MT5 connects; used each tick
        # to detect filled stop orders.
        self.reconciler: Optional[object] = None

        logger.info("Balance: $%.2f | Risk: %.1f%%",
                     self.executor.balance, self.cfg.risk.risk_percent)
        logger.info("Learning agent: %d patterns tracked, %d blocked",
                     len(self.db.get_all_patterns()),
                     len(self.db.get_blocked_patterns()))

    def start(self):
        """Start the main trading loop."""
        self.running = True
        self._start_time = time.time()

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # === CONNECT TO MT5 ===
        logger.info("=" * 50)
        logger.info("CONNECTING TO MT5...")
        logger.info("=" * 50)

        if not self.data.connect():
            logger.critical("FATAL: Cannot connect to MT5! Check:")
            logger.critical("  1. MT5 terminal is running")
            logger.critical("  2. MT5_LOGIN, MT5_PASSWORD, MT5_SERVER are correct")
            logger.critical("  3. 'Allow algorithmic trading' is enabled in MT5")
            self.telegram.notify_error(
                "❌ FATAL: Cannot connect to MT5!\n"
                "Bot cannot start. Check VPS and MT5 terminal."
            )
            return

        # === 🔁 RECONCILE — Compare DB vs MT5 positions ===
        logger.info("=" * 50)
        logger.info("RECONCILING POSITIONS...")
        logger.info("=" * 50)

        self.reconciler = BrokerReconciler(
            magic_number=self.cfg.mt5.magic_number,
            symbol=self.cfg.mt5.symbol,
        )
        report = self.reconciler.reconcile(
            db=self.db,
            trade_mgr=self.trade_mgr,
            ticket_map=self.executor._ticket_map,
        )

        # Send reconciliation report via Telegram
        self.telegram.notify_error(report.summary())  # 📲 ALERT

        if report.mt5_only:
            logger.warning("⚠️ %d MT5 positions not in DB — check manually!",
                          len(report.mt5_only))

        # === PRELOAD MARKET DATA ===
        self._preload_market_data()

        # Startup notification
        learning = self.db.get_learning_summary()
        self.telegram.notify_startup({
            "symbol": self.cfg.mt5.symbol,
            "dry_run": not self.cfg.live_mode,
            "balance": self.executor.balance,
            "risk_pct": self.cfg.risk.risk_percent,
            "max_trades": self.cfg.risk.max_trades_per_day,
            "dd_limit": self.cfg.risk.daily_loss_limit,
            "patterns": learning["patterns_tracked"],
            "blocked": learning["patterns_blocked"],
        })

        mode_str = "🔴 LIVE MODE" if self.cfg.live_mode else "🔒 SHADOW MODE"
        logger.info("Bot STARTED — %s — entering main loop", mode_str)

        # Main loop
        while self.running:
            try:
                self._tick()
            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                self.telegram.notify_error(f"⚠️ Loop error: {str(e)}")

            # Periodic hourly status
            try:
                self._check_periodic_status()
            except Exception as e:
                logger.error("Error in periodic status: %s", e)

            time.sleep(self.cfg.update_interval_seconds)

        logger.info("Bot stopped.")
        self.data.disconnect()

    def _preload_market_data(self):
        """Preload 400 M15 bars for S/R zone detection."""
        logger.info("=" * 50)
        logger.info("PRELOADING MARKET DATA (400 M15 bars)...")
        logger.info("=" * 50)

        df = self.data.preload_bars(min_bars=400)
        if df.empty:
            logger.error("PRELOAD FAILED — no data from MT5!")
            self.telegram.notify_error(
                "⚠️ Startup preload failed: no historical data from MT5."
            )
            return

        price = float(df["Close"].iloc[-1])

        # Fetch HTF (1H) data for zone stacking
        htf_df = self.data.fetch_htf_bars(timeframe="1h", lookback_days=30)

        # Run full market analysis
        market = self.analyzer.analyze(df, htf_df, price)

        # Store last bar time
        self._last_bar_time = df.index[-1]

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
        logger.info("=" * 50)

        # Telegram notification
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

        # Ensure MT5 connected
        if not self.data.ensure_connected():
            logger.warning("MT5 disconnected — skipping tick")
            return

        # Skip if market closed
        if not self.data.is_market_open():
            logger.debug("Market closed, skipping tick")
            self._check_daily_summary()
            return

        # Record today's starting balance for daily loss limit calculation.
        # Must run before can_trade() is called so the denominator is stable.
        self.risk_mgr.record_daily_start(self.executor.balance)

        # 1. Fetch data (10 days for 400+ bars)
        df = self.data.fetch_bars(lookback_days=10)
        if df.empty:
            logger.warning("NO DATA from MT5")
            return

        # Check for new bar
        latest_time = df.index[-1]
        if self._last_bar_time and latest_time <= self._last_bar_time:
            # Same bar — update active trades with latest price
            price = float(df["Close"].iloc[-1])

            # 🔁 RECONCILE on every tick — detect stop fills within current bar.
            # Stop fills server-side within an M15 bar (up to 14.5 min between bars).
            # Without this, filled trade has no ActiveTrade until next bar close.
            self._register_pending_fills()

            exit_events = self._manage_active_trades(price)
            if exit_events:
                self.executor.process_exits(exit_events)
                for event in exit_events:
                    if event["full_close"]:
                        trade = self.db.get_trade(event["trade_id"])
                        if trade:
                            context = self.db.get_context(event["trade_id"])
                            self.learner.on_trade_closed(trade, context)
                            self.telegram.notify_trade_closed(trade)
            return

        self._last_bar_time = latest_time
        # Use wall-clock UTC for session checks, not bar timestamp.
        # ICMarkets bar timestamps are EET (UTC+3) — using .hour directly
        # would make the session window 3 hours early.
        utc_hour = datetime.now(timezone.utc).hour
        price = float(df["Close"].iloc[-1])

        logger.info("--- New bar: %s | Price: $%.2f | Bars: %d ---",
                     latest_time.strftime("%Y-%m-%d %H:%M"), price, len(df))

        # 2. Fetch HTF data
        htf_df = self.data.fetch_htf_bars(timeframe="1h", lookback_days=30)

        # 3. Market analysis
        market = self.analyzer.analyze(df, htf_df, price)

        logger.info("Market: trend=%s, session=%s, vol=%s | S/R: %d sup, %d res",
                     market.trend.value, market.session.value,
                     market.current_volume_level.value,
                     len(market.support_levels), len(market.resistance_levels))

        # Log nearest S/R
        if market.support_levels:
            nearest_sup = market.support_levels[0]
            logger.info("  Nearest SUP: $%.2f (touches=%d, dist=%.1f pips)",
                         nearest_sup.price, nearest_sup.touches,
                         price_to_pips(abs(price - nearest_sup.price)))
        if market.resistance_levels:
            nearest_res = market.resistance_levels[0]
            logger.info("  Nearest RES: $%.2f (touches=%d, dist=%.1f pips)",
                         nearest_res.price, nearest_res.touches,
                         price_to_pips(abs(price - nearest_res.price)))

        # 4. Manage active trades
        exit_events = self._manage_active_trades(price)

        # 🔴 NEW ENTRY MODEL — Expire stale pending stop orders (once per bar).
        # If C0 validated but C1 breakout never came within expiry_bars (1 bar),
        # cancel the pending to keep MT5 Orders tab clean.
        self._expire_pending_stops()

        # 🔁 RECONCILE (runtime) — detect pending stop orders filled/canceled by broker.
        self._register_pending_fills()

        # 🔴 STRUCTURAL INVALIDATION — cancel any pending stop whose C1 wick
        # was breached by the forming C0 bar (or current tick). Must run on
        # every 30s tick, not only on bar close, so a dead setup is cancelled
        # before the broker fills it on a whipsaw.
        self._check_stop_invalidation(price, df)

        # Check opposite close emergency exit (on COMPLETED bar, not forming bar).
        # df.iloc[-1] = forming bar (body ratio ~ 0 at bar open — would never trigger).
        # df.iloc[-2] = completed bar with final OHLC — correct for detecting strong
        # opposite candles that warrant emergency exit.
        if self.trade_mgr.get_active_trade_count() > 0 and len(df) >= 2:
            bar = df.iloc[-2]
            bar_body_ratio = body_ratio(
                float(bar["Open"]), float(bar["Close"]),
                float(bar["High"]), float(bar["Low"])
            )
            opp_exits = self.trade_mgr.check_opposite_close(
                float(bar["Open"]), float(bar["Close"]), price,
                body_ratio_val=bar_body_ratio
            )
            if opp_exits:
                exit_events.extend(opp_exits)

        # Process exits
        if exit_events:
            self.executor.process_exits(exit_events)
            for event in exit_events:
                if event["full_close"]:
                    trade = self.db.get_trade(event["trade_id"])
                    if trade:
                        context = self.db.get_context(event["trade_id"])
                        self.learner.on_trade_closed(trade, context)
                        self.telegram.notify_trade_closed(trade)

        # 5. Generate new signals
        in_session = is_tradeable_session(utc_hour, self.cfg.strategy.session_start,
                                          self.cfg.strategy.session_end)
        if not in_session:
            logger.info("Outside trading session (hour=%d, window=%d-%d)",
                         utc_hour, self.cfg.strategy.session_start,
                         self.cfg.strategy.session_end)
            # 📵 KILL SWITCH — strategy no longer queues pending at its layer. MT5
            # pending stops are the only pending state; _expire_pending_stops on the
            # bar-close branch and _check_stop_invalidation on every tick govern
            # their lifecycle. Nothing to clear here.

        if in_session:
            blocked = [p["pattern_key"] for p in self.db.get_blocked_patterns()]
            signals = self.strategy.generate_signals(df, market, blocked)
            logger.info("Signal generation: %d signals, %d MT5 pending stops",
                         len(signals), len(self.pending_stop_orders))

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

                # 7. Route to STOP ORDER (new model) or market entry (legacy fallback)
                if sig.context.get("is_stop_order"):
                    # 🔴 NEW ENTRY MODEL — Place MT5 pending stop order at C1 extreme.
                    lot_size = self.risk_mgr.calculate_position_size(
                        self.executor.balance, sig.risk_pips
                    )
                    ticket = self.executor.place_stop_order(sig, lot_size)
                    if ticket:
                        self.pending_stop_orders[ticket] = {
                            "signal": sig,
                            "bars_waiting": 0,
                            "placed_at": datetime.now(timezone.utc),
                            "expiry_bars": sig.context.get("stop_expiry_bars", self.cfg.strategy.pending_expiry_bars),
                            "lot_size": lot_size,
                            "risk_percent": self.risk_mgr.current_risk_percent,
                            "balance_before": self.executor.balance,
                        }
                        self.telegram.notify_pending_stop({
                            "ticket": ticket,
                            "direction": sig.direction.value,
                            "trigger_price": sig.entry_price,
                            "sl_price": sig.sl_price,
                            "tp1_price": sig.tp1_price,
                            "risk_pips": sig.risk_pips,
                            "stop_buffer_pips": sig.context.get("stop_buffer_pips", 2.0),
                            "expiry_bars": sig.context.get("stop_expiry_bars", self.cfg.strategy.pending_expiry_bars),
                            "sr_level": sig.sr_level.price,
                        })
                        logger.info("📲 STOP pending alert sent: ticket=%d", ticket)
                else:
                    # Legacy market entry (non-C1/C0 paths, e.g. impulse/retest)
                    trade_id = self.executor.execute_signal(sig)
                    if trade_id:
                        lot_size = self.risk_mgr.calculate_position_size(
                            self.executor.balance, sig.risk_pips
                        )
                        risk_dollars = sig.risk_pips * self.cfg.strategy.pip_value_per_lot * lot_size
                        self.telegram.notify_trade_opened({
                            "trade_id": trade_id,
                            "direction": sig.direction.value,
                            "entry_type": sig.entry_type.value,
                            "lot_size": lot_size,
                            "entry_price": sig.entry_price,
                            "sl_price": sig.sl_price,
                            "tp1_price": sig.tp1_price,
                            "tp2_price": sig.tp2_price,
                            "risk_pips": sig.risk_pips,
                            "risk_dollars": risk_dollars,
                            "risk_percent": self.risk_mgr.current_risk_percent,
                            "balance": self.executor.balance,
                            "sr_level": sig.sr_level.price,
                            "sr_touches": sig.sr_level.touches,
                        })

        # Session end: close all
        if not is_tradeable_session(utc_hour, self.cfg.strategy.session_start,
                                    self.cfg.strategy.session_end):
            session_exits = self.trade_mgr.session_end_close_all(price)
            if session_exits:
                self.executor.process_exits(session_exits)
                for event in session_exits:
                    if event["full_close"]:
                        trade = self.db.get_trade(event["trade_id"])
                        if trade:
                            self.learner.on_trade_closed(trade)

        # Daily summary
        self._check_daily_summary()

    def _check_periodic_status(self):
        """Send 2-hour Telegram status.

        🔴 Timer is advanced before the send so a failing send doesn't spam.
        executor.get_status() and the market-context fetch are wrapped in a
        broad try/except so an MT5 blip doesn't silently swallow the alert
        and leave a 4-hour gap (timer advanced + exception caught by caller).
        """
        now = time.time()
        if now - self._last_status_time < self._status_interval:
            return

        self._last_status_time = now

        try:
            status = self.executor.get_status()
        except Exception as e:
            logger.error("Periodic status: get_status() failed — skipping alert: %s", e)
            return

        # Add live market context
        try:
            df = self.data.fetch_bars(lookback_days=10, use_cache=True)
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
                    market.support_levels[0].price if market.support_levels else None
                )
                status["nearest_resistance"] = (
                    market.resistance_levels[0].price if market.resistance_levels else None
                )

                # Unrealized P&L on open trades
                open_trade_details = []
                unrealized_pnl_total = 0.0
                for tid, trade in self.trade_mgr.active_trades.items():
                    if trade.direction == TradeDirection.BUY:
                        pnl_pips = price_to_pips(current_price - trade.entry_price)
                    else:
                        pnl_pips = price_to_pips(trade.entry_price - current_price)
                    pnl_dollars = pnl_pips * self.cfg.strategy.pip_value_per_lot * trade.remaining_lots
                    unrealized_pnl_total += pnl_dollars
                    open_trade_details.append({
                        "trade_id": tid,
                        "direction": trade.direction.value,
                        "entry_price": trade.entry_price,
                        "current_pnl_pips": round(pnl_pips, 1),
                        "current_pnl_dollars": round(pnl_dollars, 2),
                        "sl_price": trade.sl_price,
                        "sl_stage": trade.sl_stage.value,
                        "tp1_hit": trade.tp1_hit,
                        "remaining_lots": trade.remaining_lots,
                    })
                status["open_trade_details"] = open_trade_details
                status["unrealized_pnl"] = round(unrealized_pnl_total, 2)

        except Exception as e:
            logger.warning("Could not fetch market context: %s", e)

        self.telegram.notify_status(status)
        self._log_status(status)

        # Learning report
        report = self.learner.get_learning_report()
        if report["trades_analyzed"] > 0:
            self.telegram.notify_learning_report(report)

    def _sync_broker_closes(self) -> list:
        """
        🔁 RECONCILE (runtime) — Check if broker closed any active trades
        server-side (SL/TP hit, margin call, manual close) during live trading.

        The 30-second tick loop can miss broker closes when price briefly dips
        below SL then recovers before the next poll. Without this check the
        ghost trade stays in active_trades indefinitely.

        Only runs in live mode (no MT5 API in shadow mode).
        """
        if not self.cfg.live_mode:
            return []

        exits = []
        for trade_id, trade in list(self.trade_mgr.active_trades.items()):
            ticket = self.executor._ticket_map.get(trade_id)
            if ticket is None:
                continue  # no MT5 ticket — paper/shadow trade or pre-reconcile

            try:
                import MetaTrader5 as _mt5
                positions = _mt5.positions_get(ticket=ticket)
            except Exception as e:
                logger.warning("Broker close check failed for %s: %s", trade_id, e)
                continue

            if positions is None or len(positions) == 0:
                # Position gone on broker side — was closed (SL/TP/manual)
                logger.warning(
                    "BROKER CLOSE DETECTED: %s (ticket=%d) — not in MT5 positions",
                    trade_id, ticket
                )
                # Try to get actual close price from deal history
                actual_exit = self.executor._get_deal_close_price(ticket)
                if actual_exit is None:
                    # Fallback: use current price
                    actual_exit = float(
                        self.data.get_current_price() or trade.sl_price
                    )

                if trade.direction.value == "BUY":
                    pnl_pips = price_to_pips(actual_exit - trade.entry_price)
                else:
                    pnl_pips = price_to_pips(trade.entry_price - actual_exit)

                # 🔴 LIVE RISK — Determine close reason from actual SL stage + exit price.
                # Old code hardcoded SL_STAGE1 regardless of what happened.
                # A trailing SL exit at +165 pips was incorrectly labelled "SL Stage 1".
                _stage_to_reason = {
                    SLStage.STAGE_1: ExitReason.SL_STAGE1,
                    SLStage.STAGE_2: ExitReason.SL_STAGE2,
                    SLStage.STAGE_3: ExitReason.SL_STAGE3,
                }
                from utils import pips_to_price as _p2p
                _tp2_tol = _p2p(5.0)  # 5-pip tolerance for TP2 match
                _sl_tol  = _p2p(8.0)  # 8-pip tolerance for SL match
                if trade.tp1_hit and abs(actual_exit - trade.tp2_price) <= _tp2_tol:
                    broker_reason = ExitReason.TP2_HIT
                elif abs(actual_exit - trade.sl_price) <= _sl_tol:
                    # Exit matches current SL level — use actual stage
                    broker_reason = _stage_to_reason.get(trade.sl_stage, ExitReason.SL_STAGE1)
                else:
                    # Exit price not near SL or TP2 (trailing SL hit between ticks,
                    # manual close, or slippage) — label with current SL stage.
                    broker_reason = _stage_to_reason.get(trade.sl_stage, ExitReason.SL_STAGE1)

                exits.append({
                    "trade_id": trade_id,
                    "reason": broker_reason,
                    "exit_price": actual_exit,
                    "pnl_pips": pnl_pips,
                    "lots_closed": trade.remaining_lots,
                    "full_close": True,
                })

                # Remove from active_trades immediately
                del self.trade_mgr.active_trades[trade_id]

                # 📲 ALERT — use notify_broker_alert (not notify_error) so a
                # profitable broker close doesn't appear as "⚠️ BOT ERROR".
                reason_str = broker_reason.value if hasattr(broker_reason, "value") else str(broker_reason)
                self.telegram.notify_broker_alert(
                    trade_id=trade_id,
                    ticket=ticket,
                    actual_exit=actual_exit,
                    pnl_pips=pnl_pips,
                    reason=reason_str,
                )  # 📲 ALERT

        return exits

    def _manage_active_trades(self, price: float) -> list:
        """Update active trades and sync SL changes to MT5."""

        # 🔁 RECONCILE (runtime) — catch broker-side closes FIRST so we don't
        # try to manage positions the broker already closed.
        broker_close_exits = self._sync_broker_closes()

        # Snapshot SL stages before update
        sl_before = {
            tid: (t.sl_price, t.sl_stage.value)
            for tid, t in self.trade_mgr.active_trades.items()
        }

        exits = self.trade_mgr.update_trades(price)

        # Merge broker-side closes into exit list
        if broker_close_exits:
            exits = broker_close_exits + exits

        # Persist max_favorable_pips to DB so it survives a restart.
        # Give-back exit uses this as the high-water mark for pullback % calculation.
        # Without persistence, it resets to 0 on restart and give-back fires too early
        # on trades that were already deep in profit before the restart.
        for tid, trade in self.trade_mgr.active_trades.items():
            if trade.max_favorable_pips > 0:
                try:
                    self.db.update_trade(
                        tid, {"max_favorable_pips": trade.max_favorable_pips}
                    )
                except Exception:
                    pass  # Non-critical — will retry next tick

        # Check for SL stage changes → modify on MT5
        for tid, trade in self.trade_mgr.active_trades.items():
            if tid in sl_before:
                old_sl, old_stage = sl_before[tid]
                if trade.sl_stage.value != old_stage:
                    # 🔴 LIVE RISK — SL stage changed: push new SL to broker.
                    # Returns actual SL set (may be clamped vs target if price
                    # is near C0 extreme), or None if modify failed completely.
                    actual_sl = self.executor.modify_sl(tid, trade.sl_price)
                    if actual_sl is not None:
                        trade.sl_price = actual_sl  # use actual (clamped if needed)
                        self.telegram.notify_sl_update(
                            tid, old_sl, trade.sl_price, trade.sl_stage.value
                        )
                        # Persist to DB
                        try:
                            self.db.update_trade(tid, {
                                "sl_current": trade.sl_price,
                                "sl_stage": trade.sl_stage.value,
                            })
                        except Exception as e:
                            logger.error("Failed to persist SL: %s", e)
                    else:
                        # 🔴 LIVE RISK — broker rejected even clamped SL.
                        # Revert TradeManager so next tick re-attempts.
                        trade.sl_price = old_sl
                        trade.sl_stage = SLStage(old_stage)
                        logger.warning(
                            "SL stage advance failed for %s — reverted to "
                            "stage=%d, sl=%.2f. Will retry next tick.",
                            tid, old_stage, old_sl
                        )

                elif abs(trade.sl_price - old_sl) > 0.001:
                    # Trailing stop moved SL (same stage, different price)
                    actual_sl = self.executor.modify_sl(tid, trade.sl_price)
                    if actual_sl is not None:
                        trade.sl_price = actual_sl  # use actual (clamped if needed)
                        try:
                            self.db.update_trade(tid, {
                                "sl_current": trade.sl_price,
                            })
                        except Exception as e:
                            logger.error("Failed to persist trailing SL: %s", e)
                    else:
                        # 🔴 LIVE RISK — trailing SL rejected: revert for retry
                        trade.sl_price = old_sl
                        logger.warning(
                            "Trailing SL update rejected for %s — reverted to "
                            "%.2f. Will retry next tick.", tid, old_sl
                        )

        # Notify partial closes
        for event in exits:
            if not event["full_close"]:
                self.telegram.notify_trade_update({
                    "trade_id": event["trade_id"],
                    "event": event["reason"].value if hasattr(event["reason"], "value") else str(event["reason"]),
                    "exit_price": event["exit_price"],
                    "pnl_pips": event["pnl_pips"],
                    "lots_closed": event["lots_closed"],
                })

        return exits

    def _check_stop_invalidation(self, current_tick_price: float, df) -> None:
        """
        🔴 STRUCTURAL INVALIDATION — cancel pending stop if C1 wick breached intra-bar.

        Per Raja: a BUY setup dies the instant price prints below C1_low; a SELL setup
        dies when price prints above C1_high. Time-based expiry alone is too slow — a
        15-min forming bar can break the wick within seconds and a stale pending will
        still fire if price whips back.

        Runs EVERY 30s tick (not only bar close). Uses forming-bar running high/low
        (df.iloc[-1]) plus the current tick price. Cancels the MT5 pending and removes
        it from the tracking dict so it cannot fill.
        """
        if not self.pending_stop_orders or df is None or len(df) < 1:
            return

        forming = df.iloc[-1]
        forming_low  = min(float(forming["Low"]),  current_tick_price)
        forming_high = max(float(forming["High"]), current_tick_price)

        invalidated: list[int] = []
        for ticket, info in self.pending_stop_orders.items():
            sig = info["signal"]
            inv_price = sig.context.get("invalidation_price")
            if inv_price is None:
                continue  # legacy signal (old C0-gate pendings) — skip

            if sig.direction == TradeDirection.BUY and forming_low < inv_price:
                reason = (
                    f"C1_low {inv_price:.2f} broken "
                    f"(tick_low={forming_low:.2f})"
                )
            elif sig.direction == TradeDirection.SELL and forming_high > inv_price:
                reason = (
                    f"C1_high {inv_price:.2f} broken "
                    f"(tick_high={forming_high:.2f})"
                )
            else:
                continue

            if self.executor.cancel_stop_order(ticket):
                invalidated.append(ticket)
                logger.info(
                    "🔴 STOP INVALIDATED: ticket=%d %s — %s",
                    ticket, sig.direction.value, reason
                )
            else:
                logger.warning(
                    "Stop %d invalidation cancel failed — will retry next tick", ticket
                )

        for t in invalidated:
            del self.pending_stop_orders[t]

    def _register_pending_fills(self) -> None:
        """
        🔁 RECONCILE (every tick) — detect pending stop orders filled by broker,
        register them as active trades, and send Telegram alert.
        Also cleans canceled stops (gone from broker with no matching position).
        """
        if not self.reconciler or not self.pending_stop_orders:
            return

        filled, canceled = self.reconciler.register_filled_stops(
            self.pending_stop_orders, self.db, self.trade_mgr,
            self.executor._ticket_map,
        )

        for trade_id, _position_ticket in filled:
            # Grab pending data before removing (needed for Telegram alert)
            pdata = next(
                (v for v in self.pending_stop_orders.values()
                 if v["signal"].trade_id == trade_id),
                None,
            )
            # Remove from pending dict
            for pticket in list(self.pending_stop_orders.keys()):
                if self.pending_stop_orders[pticket]["signal"].trade_id == trade_id:
                    del self.pending_stop_orders[pticket]
                    break
            # 📲 ALERT — stop order filled, trade is now live
            active = self.trade_mgr.active_trades.get(trade_id)
            if active:
                lot_size = pdata["lot_size"] if pdata else active.lot_size
                risk_pct = pdata["risk_percent"] if pdata else 0.0
                bal = pdata["balance_before"] if pdata else self.executor.balance
                risk_pips = price_to_pips(abs(active.entry_price - active.sl_price))
                risk_dollars = risk_pips * self.cfg.strategy.pip_value_per_lot * lot_size
                self.telegram.notify_trade_opened({
                    "trade_id": trade_id,
                    "direction": active.direction.value,
                    "entry_type": "c1_c0_rejection",
                    "lot_size": lot_size,
                    "entry_price": active.entry_price,
                    "sl_price": active.sl_price,
                    "tp1_price": active.tp1_price,
                    "tp2_price": active.tp2_price,
                    "risk_pips": risk_pips,
                    "risk_dollars": risk_dollars,
                    "risk_percent": risk_pct,
                    "balance": bal,
                    "sr_level": pdata["signal"].sr_level.price if pdata else 0.0,
                    "sr_touches": pdata["signal"].sr_level.touches if pdata else 0,
                })
                logger.info("📲 Trade-opened alert sent for filled stop: %s", trade_id)

        for ticket in canceled:
            del self.pending_stop_orders[ticket]
            logger.info("🔁 Stop %d canceled (no position found)", ticket)

    def _expire_pending_stops(self) -> None:
        """
        🔴 NEW ENTRY MODEL — Cancel pending stop orders that exceeded expiry window.

        Call on new-bar branch (once per M15 bar close), NOT every 30s tick.
        Each bar increments `bars_waiting`. If >= `expiry_bars` (default 2),
        cancel via MT5 and remove from tracking dict. If cancel fails, keep
        entry so next bar retries.

        Reconciler handles the FILL case separately (pending → position).
        """
        if not self.pending_stop_orders:
            return

        expired: list[int] = []
        for ticket, info in self.pending_stop_orders.items():
            info["bars_waiting"] += 1
            if info["bars_waiting"] >= info["expiry_bars"]:
                if self.executor.cancel_stop_order(ticket):
                    expired.append(ticket)
                    sig = info["signal"]
                    logger.info(
                        "Stop order EXPIRED: ticket=%d %s trigger=%.2f "
                        "(bars_waited=%d >= expiry=%d)",
                        ticket, sig.direction.value, sig.entry_price,
                        info["bars_waiting"], info["expiry_bars"]
                    )
                else:
                    # Cancel failed — leave in dict, retry next bar
                    logger.warning(
                        "Stop %d cancel failed — will retry next bar", ticket
                    )

        for t in expired:
            del self.pending_stop_orders[t]

    def _check_daily_summary(self):
        """Send daily summary at end of trading session."""
        now = utc_now()
        today = now.strftime("%Y-%m-%d")

        if self._last_daily_summary_date == today:
            return

        if now.hour < self.cfg.strategy.session_end:
            return

        self._last_daily_summary_date = today

        trades_today = self.db.get_trades_today()
        daily_pnl = self.db.get_daily_pnl()

        wins_losses = self.db.conn.execute(
            "SELECT "
            "  SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as wins, "
            "  SUM(CASE WHEN pnl_dollars <= 0 THEN 1 ELSE 0 END) as losses "
            "FROM trades WHERE exit_time LIKE ? AND is_open = 0",
            (f"{today}%",)
        ).fetchone()
        wins = wins_losses["wins"] or 0 if wins_losses else 0
        losses = wins_losses["losses"] or 0 if wins_losses else 0

        if trades_today > 0:
            self.telegram.notify_daily_summary({
                "date": today,
                "trades": trades_today,
                "wins": wins,
                "losses": losses,
                "pnl_dollars": daily_pnl,
                "balance": self.executor.balance,
            })

            self.db.upsert_daily_stats(today, {
                "trades_taken": trades_today,
                "wins": wins,
                "losses": losses,
                "pnl_dollars": daily_pnl,
                "balance_end": self.executor.balance,
            })

    def _log_status(self, status: dict):
        """Log current bot status."""
        logger.info("=" * 50)
        logger.info("STATUS [%s]: $%.2f (%.1f%%) | WR %.1f%% | Trades %d | Open %d",
                     status.get("mode", "?"),
                     status["balance"], status["total_pnl_pct"],
                     status["win_rate"], status["total_trades"],
                     status["open_trades"])
        risk = status["risk_status"]
        logger.info("Risk: %.2f%% | Losses: %d | Cooldown: %s",
                     risk["current_risk_pct"], risk["consecutive_losses"],
                     "YES" if risk["cooldown_active"] else "NO")
        logger.info("=" * 50)

    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown — preserve trades, disconnect MT5."""
        sig_name = signal.Signals(signum).name if signum else "unknown"
        logger.info("Shutdown signal received: %s", sig_name)
        self.running = False

        open_count = self.trade_mgr.get_active_trade_count()
        if open_count > 0:
            logger.info("Preserving %d open trade(s) for restart recovery", open_count)
            self.telegram.notify_status({
                "shutdown": True,
                "open_trades_preserved": open_count,
                "signal": sig_name,
            })

        status = self.executor.get_status()
        self._log_status(status)

        report = self.learner.get_learning_report()
        if report["trades_analyzed"] > 0:
            self.telegram.notify_learning_report(report)

        self.telegram.notify_shutdown(f"{sig_name} — {open_count} trades preserved")
        self.data.disconnect()
        self.db.close()


def main():
    """Entry point for ANi's FX Bot (MT5 version)."""
    config = BotConfig.from_env()
    setup_logging(config.log_level, config.data_dir)

    print("\n" + "=" * 60)
    print("  ANi's FX Bot v2.0 — MT5 Edition")
    print("  Raja Banks Market Fluidity Strategy")
    print("  " + ("🔴 LIVE MODE" if config.live_mode else "🔒 SHADOW MODE"))
    print(f"  Symbol: {config.mt5.symbol}")
    print(f"  Account: {config.mt5.login}")
    print(f"  Balance: ${config.initial_balance:,.2f}")
    print(f"  Risk: {config.risk.risk_percent}% per trade")
    print(f"  Max Trades/Day: {config.risk.max_trades_per_day}")
    print("=" * 60 + "\n")

    bot = TradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()
