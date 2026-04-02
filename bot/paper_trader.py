"""
Raja Banks Trading Agent - Paper Trader
Simulated execution, balance tracking, position management.
"""
import logging
from typing import Optional, Dict, List
from datetime import datetime, timezone

from config import BotConfig
from constants import TradeDirection, SLStage, ExitReason
from strategy import Signal
from trade_manager import TradeManager, ActiveTrade
from risk_manager import RiskManager
from database import TradingDatabase
from utils import price_to_pips, pips_to_price, get_session, get_4h_candle_number, utc_now

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Simulated paper trading engine.
    Tracks balance, executes signals, manages positions via TradeManager.
    """

    def __init__(self, config: BotConfig, db: TradingDatabase,
                 risk_mgr: RiskManager, trade_mgr: TradeManager):
        self.cfg = config
        self.db = db
        self.risk_mgr = risk_mgr
        self.trade_mgr = trade_mgr
        self.balance = config.initial_balance
        self._partial_pnl: Dict[str, float] = {}  # Track partial close P&L per trade
        self._load_balance()

    def _load_balance(self):
        """
        🔁 RECONCILE — Restore balance from database on startup.
        Three-layer fallback:
          1. Last CLOSED trade's balance_after (most accurate)
          2. initial_balance + SUM of all closed trade P&L (recalculated)
          3. Fall back to initial_balance only if DB has zero closed trades
        """
        try:
            # Layer 1: Last closed trade with balance_after
            closed = self.db.conn.execute(
                "SELECT balance_after FROM trades "
                "WHERE is_open = 0 AND balance_after IS NOT NULL "
                "ORDER BY exit_time DESC LIMIT 1"
            ).fetchone()
            if closed and closed[0]:
                self.balance = closed[0]
                logger.info("🔁 Balance restored from last closed trade: $%.2f", self.balance)
                return

            # Layer 2: Recalculate from all closed trade P&L
            total_pnl = self.db.conn.execute(
                "SELECT COALESCE(SUM(pnl_dollars), 0) FROM trades WHERE is_open = 0"
            ).fetchone()[0]
            if total_pnl != 0:
                self.balance = self.cfg.initial_balance + total_pnl
                logger.info("🔁 Balance recalculated from P&L sum: $%.2f (initial $%.2f + P&L $%.2f)",
                           self.balance, self.cfg.initial_balance, total_pnl)
                return

            # Layer 3: No closed trades — keep initial balance
            logger.info("🔁 No closed trades found — starting with initial balance: $%.2f",
                       self.balance)

        except Exception as e:
            logger.error("⚠️ Balance restore FAILED: %s — keeping initial balance $%.2f",
                        str(e), self.balance)

    def execute_signal(self, signal: Signal) -> Optional[str]:
        """
        Execute a trading signal in paper mode.
        Returns trade_id if executed, None if rejected.
        """
        # Check risk limits
        can_trade, reason = self.risk_mgr.can_trade(self.balance)
        if not can_trade:
            logger.info("Signal rejected by risk manager: %s", reason)
            return None

        # Calculate position size
        lot_size = self.risk_mgr.calculate_position_size(
            self.balance, signal.risk_pips
        )

        # Calculate dollar risk
        risk_dollars = signal.risk_pips * self.cfg.strategy.pip_value_per_lot * lot_size

        now = utc_now()
        utc_hour = now.hour

        # Create active trade for TradeManager
        active = ActiveTrade(
            trade_id=signal.trade_id,
            direction=signal.direction,
            entry_price=signal.entry_price,
            lot_size=lot_size,
            sl_price=signal.sl_price,
            sl_stage=SLStage.STAGE_1,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            c0_extreme=signal.context.get("c1_low") if signal.direction == TradeDirection.BUY
                       else signal.context.get("c1_high"),
        )
        self.trade_mgr.open_trade(active)

        # Store in database
        trade_record = {
            "trade_id": signal.trade_id,
            "symbol": self.cfg.strategy.symbol,
            "direction": signal.direction.value,
            "entry_type": signal.entry_type.value,
            "entry_price": signal.entry_price,
            "lot_size": lot_size,
            "sl_initial": signal.sl_price,
            "sl_current": signal.sl_price,
            "sl_stage": 1,
            "tp1_price": signal.tp1_price,
            "tp2_price": signal.tp2_price,
            "risk_pips": signal.risk_pips,
            "risk_percent": self.risk_mgr.current_risk_percent,
            "balance_before": self.balance,
            "entry_time": now.isoformat(),
            "session_type": get_session(utc_hour).value,
            "four_h_candle": get_4h_candle_number(utc_hour),
        }
        self.db.insert_trade(trade_record)

        # Store context
        context_record = {
            "trade_id": signal.trade_id,
            "sr_level_price": signal.sr_level.price,
            "sr_level_type": signal.sr_level.zone_type.value,
            "sr_touches": signal.sr_level.touches,
            "confidence_score": signal.confidence,
        }
        # Merge signal context
        ctx = signal.context
        if "c1_body_ratio" in ctx:
            context_record["c1_body_ratio"] = ctx["c1_body_ratio"]
            context_record["c1_range_pips"] = ctx.get("c1_range_pips", 0)
            context_record["c1_is_wickless"] = 1 if ctx.get("c1_is_wickless") else 0
        if "impulse_body_ratio" in ctx:
            context_record["impulse_body_ratio"] = ctx["impulse_body_ratio"]
            context_record["impulse_volume_ratio"] = ctx.get("impulse_volume_ratio", 0)
            context_record["clean_range_bars"] = ctx.get("clean_range_bars", 0)
        if "break_level_price" in ctx:
            context_record["break_level_price"] = ctx["break_level_price"]
            context_record["retest_bars_waited"] = ctx.get("retest_bars_waited", 0)
        if "trend" in ctx:
            context_record["trend_type"] = ctx["trend"]
        if "htf_aligned" in ctx:
            context_record["htf_zone_aligned"] = 1 if ctx["htf_aligned"] else 0

        self.db.insert_context(context_record)

        logger.info(
            "PAPER TRADE: %s %s %.2f lots at %.2f | SL %.2f | TP1 %.2f | TP2 %.2f | Risk $%.2f (%.1f%%)",
            signal.direction.value, self.cfg.strategy.symbol, lot_size,
            signal.entry_price, signal.sl_price, signal.tp1_price,
            signal.tp2_price, risk_dollars, self.risk_mgr.current_risk_percent
        )

        return signal.trade_id

    def process_exits(self, exit_events: List[Dict]):
        """Process trade exit events from TradeManager."""
        for event in exit_events:
            self._process_exit(event)

    def _process_exit(self, event: Dict):
        """Process a single trade exit."""
        trade_id = event["trade_id"]
        trade = self.db.get_trade(trade_id)
        if not trade:
            logger.error("Trade not found for exit: %s", trade_id)
            return

        exit_price = event["exit_price"]
        lots_closed = event["lots_closed"]
        reason = event["reason"]
        full_close = event["full_close"]

        # Calculate P&L
        direction = TradeDirection(trade["direction"])
        if direction == TradeDirection.BUY:
            pnl_pips = price_to_pips(exit_price - trade["entry_price"])
        else:
            pnl_pips = price_to_pips(trade["entry_price"] - exit_price)

        # Adjust sign for losses
        if (direction == TradeDirection.BUY and exit_price < trade["entry_price"]) or \
           (direction == TradeDirection.SELL and exit_price > trade["entry_price"]):
            pnl_pips = -abs(pnl_pips)

        pnl_dollars = pnl_pips * self.cfg.strategy.pip_value_per_lot * lots_closed
        rr = pnl_pips / trade["risk_pips"] if trade["risk_pips"] > 0 else 0

        # Update balance
        self.balance += pnl_dollars

        if full_close:
            # Include any partial close P&L in the total
            total_pnl_dollars = pnl_dollars + self._partial_pnl.pop(trade_id, 0.0)

            self.db.close_trade(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=reason.value if isinstance(reason, ExitReason) else reason,
                pnl_pips=pnl_pips,
                pnl_dollars=total_pnl_dollars,
                rr_achieved=rr,
                balance_after=self.balance,
            )

            is_win = total_pnl_dollars > 0
            self.risk_mgr.update_after_trade(is_win)

            logger.info(
                "TRADE CLOSED: %s | %s | PnL: %.1f pips ($%.2f total) | RR: %.2f | Balance: $%.2f",
                trade_id, reason.value if isinstance(reason, ExitReason) else reason,
                pnl_pips, total_pnl_dollars, rr, self.balance
            )
        else:
            # Partial close (TP1) - track partial P&L
            self._partial_pnl[trade_id] = self._partial_pnl.get(trade_id, 0.0) + pnl_dollars
            self.db.update_trade(trade_id, {
                "partial_close_pct": event.get("partial_close_pct", 0.5),
            })
            logger.info(
                "PARTIAL CLOSE: %s | TP1 | %.2f lots closed | PnL: +%.1f pips ($%.2f)",
                trade_id, lots_closed, pnl_pips, pnl_dollars
            )

    def get_status(self) -> Dict:
        """Get current paper trading status."""
        return {
            "balance": round(self.balance, 2),
            "initial_balance": self.cfg.initial_balance,
            "total_pnl": round(self.balance - self.cfg.initial_balance, 2),
            "total_pnl_pct": round((self.balance / self.cfg.initial_balance - 1) * 100, 2),
            "total_trades": self.db.get_total_trades(),
            "trades_today": self.db.get_trades_today(),
            "daily_pnl": round(self.db.get_daily_pnl(), 2),
            "win_rate": round(self.db.get_win_rate() * 100, 1),
            "open_trades": self.trade_mgr.get_active_trade_count(),
            "risk_status": self.risk_mgr.get_risk_status(self.balance),
        }
