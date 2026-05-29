# mt5_executor.py — Live MT5 Order Execution for ANi's FX Bot
"""
Live order execution via MetaTrader5 API.
Replaces PaperTrader for real/demo trading.

Every order has:
  - SL attached at creation (NEVER naked)
  - Magic number for identification
  - Pre/post execution logging
  - Error handling with Telegram alerts
"""
import logging
import time
from typing import Optional, Dict, List
from datetime import datetime, timezone

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from config import BotConfig
from constants import TradeDirection, SLStage, ExitReason, EntryType
from strategy import Signal
from trade_manager import TradeManager, ActiveTrade, reconcile_close_price
from risk_manager import RiskManager
from database import TradingDatabase
from utils import price_to_pips, pips_to_price, get_session, get_4h_candle_number, utc_now

# 🔴 LIVE RISK — Max pips beyond C0 close we will still accept entry.
# The bot detects C0 on the next 30-second tick after the bar closes.
# Gold can move 10-20 pips in that window. If price ran MORE than this
# from C0 close, the lot size calculated from C0 wick distance is now wrong
# (actual SL distance is larger → we'd be overleveraged). Skip the trade.
# Set MAX_ENTRY_SLIPPAGE_PIPS env var to override. 20 pips handles normal
# 30-second XAUUSD movement without skipping valid setups.
MAX_ENTRY_SLIPPAGE_PIPS = 20.0  # 20 pips = ~$2.00 on XAUUSD — override via env

logger = logging.getLogger(__name__)

# MT5 return codes
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_REJECT = 10006
TRADE_RETCODE_CANCEL = 10007
TRADE_RETCODE_INVALID = 10013
TRADE_RETCODE_INVALID_VOLUME = 10014
TRADE_RETCODE_INVALID_STOPS = 10016  # SL/TP too close to market price
TRADE_RETCODE_NO_MONEY = 10019

MAX_ORDER_RETRIES = 3
REQUOTE_DELAY = 0.5  # seconds between retries


def clamp_sl_to_valid(
    is_buy: bool,
    target_sl: float,
    current_sl: float,
    ask: float,
    bid: float,
    min_distance: float,
    digits: int = 2,
) -> Optional[float]:
    """
    🔴 LIVE RISK — Clamp a desired SL to a broker-valid, never-looser level.

    A stop closes the position against the OPPOSITE quote, so validity is about
    the correct side of the *triggering* price, not raw distance from mid:
      BUY  position closes at the Bid → SL must sit BELOW bid by >= min_distance.
      SELL position closes at the Ask → SL must sit ABOVE ask by >= min_distance.

    Returns the SL to send (clamped into the valid band and rounded), or None
    when no valid level exists that is ALSO tighter than the current SL. None
    means "can't improve right now" — never loosen protection, never spam the
    broker with a level it will reject (the INVALID_STOPS revert loop).
    """
    eps = (10.0 ** (-digits)) / 2.0
    if is_buy:
        # SL must be below bid by min_distance; tighter = higher.
        max_valid = bid - min_distance
        candidate = round(min(target_sl, max_valid), digits)
        if candidate <= current_sl + eps:      # not strictly tighter than now
            return None
        return candidate
    # SELL: SL must be above ask by min_distance; tighter = lower.
    min_valid = ask + min_distance
    candidate = round(max(target_sl, min_valid), digits)
    if candidate >= current_sl - eps:           # not strictly tighter than now
        return None
    return candidate


def trade_total_pips(
    total_pnl_dollars: float,
    pip_value_per_lot: float,
    original_lots: float,
    fallback_pips: float,
) -> float:
    """
    Blended pips for a WHOLE trade, derived from total realised dollars over the
    ORIGINAL position size.

    With a TP1 partial, the final leg often closes at breakeven (0 pips) while
    TP1 already banked profit (held in _partial_pnl). Reporting only the final
    leg produced a misleading "0 pips / R:R 0.00" next to a positive dollar PnL.
    Deriving pips from the summed dollars keeps pips, $ and R:R consistent.
    Falls back to the leg pips when sizing data is missing (e.g. paper edge).
    """
    if pip_value_per_lot > 0 and original_lots and original_lots > 0:
        return total_pnl_dollars / (pip_value_per_lot * original_lots)
    return fallback_pips


class MT5Executor:
    """
    Live MT5 order execution engine.
    Same interface as PaperTrader: execute_signal(), process_exits(), get_status()

    🔴 LIVE RISK — This module sends REAL orders to the broker.
    """

    def __init__(self, config: BotConfig, db: TradingDatabase,
                 risk_mgr: RiskManager, trade_mgr: TradeManager,
                 mt5_connector, telegram=None):
        self.cfg = config
        self.db = db
        self.risk_mgr = risk_mgr
        self.trade_mgr = trade_mgr
        self.mt5_conn = mt5_connector
        self.telegram = telegram
        self.balance = config.initial_balance
        self._partial_pnl: Dict[str, float] = {}

        # Map trade_id -> MT5 ticket number
        self._ticket_map: Dict[str, int] = {}

        # Load balance from broker (ground truth)
        self._load_balance()

    def _load_balance(self):
        """
        🔁 RECONCILE — Load balance from MT5 account (ground truth).
        Falls back to DB if MT5 not connected.
        """
        try:
            account = mt5.account_info()
            if account is not None:
                self.balance = float(account.balance)
                logger.info("🔁 Balance loaded from MT5: $%.2f", self.balance)
                return
        except Exception as e:
            logger.warning("Could not get MT5 balance: %s", e)

        # Fallback to DB (same as PaperTrader)
        try:
            closed = self.db.conn.execute(
                "SELECT balance_after FROM trades "
                "WHERE is_open = 0 AND balance_after IS NOT NULL "
                "ORDER BY exit_time DESC LIMIT 1"
            ).fetchone()
            if closed and closed[0]:
                self.balance = closed[0]
                logger.info("🔁 Balance restored from DB: $%.2f", self.balance)
                return

            total_pnl = self.db.conn.execute(
                "SELECT COALESCE(SUM(pnl_dollars), 0) FROM trades WHERE is_open = 0"
            ).fetchone()[0]
            if total_pnl != 0:
                self.balance = self.cfg.initial_balance + total_pnl
                logger.info("🔁 Balance recalculated: $%.2f", self.balance)
                return

            logger.info("🔁 Using initial balance: $%.2f", self.balance)

        except Exception as e:
            logger.error("⚠️ Balance restore FAILED: %s", e)

    def _refresh_balance(self):
        """Get current balance from MT5 (call after trade closes)."""
        try:
            account = mt5.account_info()
            if account is not None:
                self.balance = float(account.balance)
        except Exception:
            pass

    def _validate_lot_size(self, lot_size: float) -> float:
        """
        🔴 LIVE RISK — Validate and round lot size to broker constraints.
        """
        info = self.mt5_conn.get_symbol_info()
        vol_min = info["volume_min"]
        vol_max = info["volume_max"]
        vol_step = info["volume_step"]

        # Round to volume_step
        if vol_step > 0:
            lot_size = round(round(lot_size / vol_step) * vol_step, 8)

        # Clamp to min/max
        lot_size = max(vol_min, min(lot_size, vol_max))

        return round(lot_size, 2)

    def _get_filling_mode(self):
        """Get the correct filling mode for the broker."""
        mode = self.cfg.mt5.filling_mode.upper()
        if mode == "FOK":
            return mt5.ORDER_FILLING_FOK
        elif mode == "RETURN":
            return mt5.ORDER_FILLING_RETURN
        return mt5.ORDER_FILLING_IOC  # default

    def execute_signal(self, signal: Signal) -> Optional[str]:
        """
        🔴 LIVE RISK — Execute a trading signal on MT5.
        Returns trade_id if executed, None if rejected.

        Safety: Every order has SL+TP attached at creation.
        """
        # Check risk limits (same as PaperTrader)
        can_trade, reason = self.risk_mgr.can_trade(self.balance)
        if not can_trade:
            logger.info("Signal rejected by risk manager: %s", reason)
            return None

        # Calculate position size
        lot_size = self.risk_mgr.calculate_position_size(
            self.balance, signal.risk_pips
        )
        lot_size = self._validate_lot_size(lot_size)

        # 📵 KILL SWITCH — if not live mode, log but don't execute
        if not self.cfg.live_mode:
            logger.info(
                "🔒 SHADOW MODE: Would %s %.2f lots at %.2f | SL %.2f | TP1 %.2f",
                signal.direction.value, lot_size, signal.entry_price,
                signal.sl_price, signal.tp1_price
            )
            # Still track in paper mode via trade manager
            return self._execute_paper(signal, lot_size)

        # Get current bid/ask
        bid, ask = self.mt5_conn.get_bid_ask()
        if bid <= 0 or ask <= 0:
            logger.error("Invalid bid/ask: bid=%.2f, ask=%.2f", bid, ask)
            self._alert("❌ Order rejected: invalid bid/ask prices")
            return None

        # Determine order type and price
        if signal.direction == TradeDirection.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = ask  # Buy at ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = bid  # Sell at bid

        # 🔴 LIVE RISK — ENTRY SLIPPAGE GATE
        # Reject trade if price has run too far past where we expected to fill.
        #
        # For C1/C0 entries (new Raja Banks model):
        #   signal.entry_price = C0 open (theoretical entry — where wick broke open)
        #   slippage_ref       = C0 close (the bar we just detected — execution reference)
        #   The distance from C0 open to C0 close is the C0 BODY, NOT slippage.
        #   Using C0 open as reference would flag every valid C0 as "too much slippage".
        #
        # For other entry types: slippage_ref = entry_price (unchanged behaviour).
        max_slip = pips_to_price(
            float(getattr(self.cfg.strategy, 'max_entry_slippage_pips',
                          MAX_ENTRY_SLIPPAGE_PIPS))
        )
        slippage_ref = signal.entry_price  # default for non-C0 entries
        if signal.entry_type == EntryType.C1_C0_REJECTION:
            # Use C0 close as slippage reference — ask should be very close to
            # C0 close since we detect it on the next 30-second tick.
            slippage_ref = signal.context.get("c0_close", signal.entry_price)

        if signal.direction == TradeDirection.BUY:
            overshoot = ask - slippage_ref  # positive = price chased above reference
            if overshoot > max_slip:
                slippage_pips = price_to_pips(overshoot)
                logger.warning(
                    "ENTRY SKIPPED (slippage): ask=%.2f is %.1f pips above "
                    "ref=%.2f (max allowed: %.1f pips)",
                    ask, slippage_pips, slippage_ref, price_to_pips(max_slip)
                )
                self._alert(
                    f"Trade skipped: price chased {slippage_pips:.1f} pips past C0 close\n"
                    f"Ref: ${slippage_ref:.2f} | Ask: ${ask:.2f}"
                )  # 📲 ALERT
                return None
        else:
            overshoot = slippage_ref - bid  # positive = price chased below reference
            if overshoot > max_slip:
                slippage_pips = price_to_pips(overshoot)
                logger.warning(
                    "ENTRY SKIPPED (slippage): bid=%.2f is %.1f pips below "
                    "ref=%.2f (max allowed: %.1f pips)",
                    bid, slippage_pips, slippage_ref, price_to_pips(max_slip)
                )
                self._alert(
                    f"Trade skipped: price chased {slippage_pips:.1f} pips past C0 close\n"
                    f"Ref: ${slippage_ref:.2f} | Bid: ${bid:.2f}"
                )  # 📲 ALERT
                return None

        logger.info(
            "Entry slippage OK: ref=%.2f, actual=%.2f, overshoot=%.2f pips",
            slippage_ref, price, price_to_pips(abs(price - slippage_ref))
        )

        # 🔴 LIVE RISK — Recalculate lot size from ACTUAL current price → SL.
        # signal.risk_pips was estimated from C0 open → C0 wick at signal creation.
        # If price moved between C0 close and this execution attempt, the real
        # fill price → SL distance may differ. Resize to keep exact RISK_PERCENT.
        # Only applies to C1/C0 entries; other types already use current price.
        if signal.entry_type == EntryType.C1_C0_REJECTION:
            if signal.direction == TradeDirection.BUY:
                actual_risk_pips_now = price_to_pips(abs(ask - signal.sl_price))
            else:
                actual_risk_pips_now = price_to_pips(abs(bid - signal.sl_price))

            if actual_risk_pips_now > 0 and abs(actual_risk_pips_now - signal.risk_pips) > 2.0:
                logger.info(
                    "Lot resize: C0 risk %.1f pips → fill risk %.1f pips "
                    "(price moved since C0 close). Resizing to maintain RISK_PERCENT.",
                    signal.risk_pips, actual_risk_pips_now
                )
                lot_size = self.risk_mgr.calculate_position_size(
                    self.balance, actual_risk_pips_now
                )
                lot_size = self._validate_lot_size(lot_size)

        # 🔴 LIVE RISK — Validate stops meet broker minimum distance.
        # ICMarkets reports trade_stops_level=0 for XAUUSD (dynamic minimum).
        # When 0, fall back to spread×3 + 5-pip buffer (same logic as modify_sl).
        info = mt5.symbol_info(self.cfg.mt5.symbol)
        if info is not None:
            stops_level = info.trade_stops_level  # minimum distance in points
            point = info.point  # smallest price increment

            if stops_level > 0:
                min_distance = stops_level * point
            else:
                # Dynamic minimum (broker like ICMarkets): spread * 3, min 5 pips
                spread_price = info.spread * point
                min_distance = max(spread_price * 3.0, pips_to_price(5.0))

            sl_distance = abs(price - signal.sl_price)
            tp_distance = abs(signal.tp1_price - price)

            if sl_distance < min_distance:
                logger.warning(
                    "SL too close at entry: %.2f (distance=%.2f, min=%.2f). Adjusting.",
                    signal.sl_price, sl_distance, min_distance
                )
                # Push SL further out to meet minimum
                if signal.direction == TradeDirection.BUY:
                    signal.sl_price = round(price - min_distance - point * 10,
                                            self.mt5_conn._digits)
                else:
                    signal.sl_price = round(price + min_distance + point * 10,
                                            self.mt5_conn._digits)

            if tp_distance < min_distance:
                logger.warning(
                    "TP too close at entry: %.2f (distance=%.2f, min=%.2f). Adjusting.",
                    signal.tp1_price, tp_distance, min_distance
                )
                # Push TP further out to meet minimum
                if signal.direction == TradeDirection.BUY:
                    signal.tp1_price = round(price + min_distance + point * 10,
                                             self.mt5_conn._digits)
                else:
                    signal.tp1_price = round(price - min_distance - point * 10,
                                             self.mt5_conn._digits)

            logger.info("Broker stops_level=%d points (%.2f price), SL dist=%.2f, TP dist=%.2f",
                        stops_level, min_distance, sl_distance, tp_distance)

        # Build order request
        # 🔴 LIVE RISK — SL and TP attached at creation (NEVER naked).
        #
        # TP is set to TP2 (NOT TP1) as a safety net.
        # Rationale: MT5's native TP mechanism closes 100% of the position when
        # price hits the TP level.  If we set TP = TP1, MT5 closes the FULL
        # position at TP1 before the bot's 30-second poll can do the 50% partial.
        # By setting TP = TP2, MT5 acts as a fallback that closes the remaining
        # position at TP2 only if the bot misses the level entirely (e.g. crash).
        # The bot manages TP1 partial close entirely in software via partial_close().
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.mt5.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": round(signal.sl_price, self.mt5_conn._digits),
            "tp": round(signal.tp2_price, self.mt5_conn._digits),  # 🔴 TP2 as safety net
            "magic": self.cfg.mt5.magic_number,
            "comment": f"ANi_{signal.trade_id[:8]}",
            "type_filling": self._get_filling_mode(),
            "type_time": mt5.ORDER_TIME_GTC,
            "deviation": self.cfg.mt5.max_slippage,
        }

        # === PRE-EXECUTION LOG ===
        logger.info(
            "SENDING ORDER: %s %s %.2f lots at %.2f | SL %.2f | TP1 %.2f | TP2(MT5) %.2f | Magic %d",
            signal.direction.value, self.cfg.mt5.symbol, lot_size,
            price, signal.sl_price, signal.tp1_price, signal.tp2_price, self.cfg.mt5.magic_number
        )

        # === SEND ORDER WITH RETRY ===
        result = None
        for attempt in range(MAX_ORDER_RETRIES):
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                logger.error("order_send returned None: %s", error)
                self._alert(f"❌ Order FAILED (None): {error}")
                return None

            if result.retcode == TRADE_RETCODE_DONE:
                break  # Success!

            if result.retcode == TRADE_RETCODE_REQUOTE:
                logger.warning("Requote (attempt %d/%d), retrying...",
                             attempt + 1, MAX_ORDER_RETRIES)
                time.sleep(REQUOTE_DELAY)
                # Refresh price
                bid, ask = self.mt5_conn.get_bid_ask()
                if signal.direction == TradeDirection.BUY:
                    request["price"] = ask
                else:
                    request["price"] = bid
                continue

            # Non-retryable error
            logger.error(
                "❌ ORDER REJECTED: retcode=%d, comment=%s",
                result.retcode, result.comment
            )
            self._alert(
                f"❌ Order REJECTED\n"
                f"Code: {result.retcode}\n"
                f"Comment: {result.comment}\n"
                f"{signal.direction.value} {lot_size} lots"
            )  # 📲 ALERT
            return None

        # Check final result
        if result.retcode != TRADE_RETCODE_DONE:
            logger.error("Order failed after %d retries: %d - %s",
                        MAX_ORDER_RETRIES, result.retcode, result.comment)
            self._alert(f"❌ Order FAILED after retries: {result.comment}")
            return None

        # === POST-EXECUTION LOG ===
        ticket = result.order
        actual_price = result.price
        actual_volume = result.volume

        logger.info(
            "✅ ORDER FILLED: ticket=%d | %s %.2f lots at %.2f (requested %.2f) | SL %.2f",
            ticket, signal.direction.value, actual_volume, actual_price,
            price, signal.sl_price
        )

        # Map trade_id to MT5 ticket
        self._ticket_map[signal.trade_id] = ticket

        # 🔴 LIVE RISK — Recalculate risk_pips from ACTUAL fill price, not C0 open.
        # signal.risk_pips was calculated from C0 open to SL (pre-fill estimate).
        # After fill we know the real entry — actual risk includes spread + slippage.
        actual_risk_pips = price_to_pips(abs(actual_price - signal.sl_price))
        if actual_risk_pips <= 0:
            actual_risk_pips = signal.risk_pips  # fallback — shouldn't happen
        risk_dollars = actual_risk_pips * self.cfg.strategy.pip_value_per_lot * actual_volume

        logger.info(
            "Risk recalculated: estimated=%.1f pips → actual=%.1f pips "
            "(entry drift: %.1f pips)",
            signal.risk_pips, actual_risk_pips,
            price_to_pips(abs(actual_price - signal.entry_price))
        )

        now = utc_now()
        utc_hour = now.hour

        # 🔴 LIVE RISK — c0_extreme: Stage-2 SL anchor (used by TradeManager._update_sl_stage).
        #
        # New entry model: entry = C0 open, SL = C0 wick tip.
        # Stage-2 target = C0 open (the entry level = near-breakeven).
        #   BUY:  at +20 pips, SL advances from C0 low → C0 open  (entry)
        #   SELL: at +20 pips, SL advances from C0 high → C0 open (entry)
        # Stage-3 target = actual fill price (true breakeven, may differ from C0 open
        # by spread + execution slip — handled by TradeManager).
        #
        # For non-C1/C0 signals: c0_extreme is None (no Stage-2 advance).
        if signal.entry_type.value == "c1_c0_rejection":
            c0_extreme = signal.entry_price   # C0 open = near-breakeven target
        else:
            c0_extreme = None

        # Register in TradeManager (for SL staging / TP management)
        active = ActiveTrade(
            trade_id=signal.trade_id,
            direction=signal.direction,
            entry_price=actual_price,
            lot_size=actual_volume,
            sl_price=signal.sl_price,
            sl_stage=SLStage.STAGE_1,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            c0_extreme=c0_extreme,
        )
        self.trade_mgr.open_trade(active)

        # Store in database (same as PaperTrader)
        trade_record = {
            "trade_id": signal.trade_id,
            "symbol": self.cfg.strategy.symbol,
            "direction": signal.direction.value,
            "entry_type": signal.entry_type.value,
            "entry_price": actual_price,
            "lot_size": actual_volume,
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

        # Store context (same as PaperTrader)
        context_record = {
            "trade_id": signal.trade_id,
            "sr_level_price": signal.sr_level.price,
            "sr_level_type": signal.sr_level.zone_type.value,
            "sr_touches": signal.sr_level.touches,
            "confidence_score": signal.confidence,
        }
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

        # NOTE: Trade-opened Telegram notification is sent by main.py via
        # telegram.notify_trade_opened() after this function returns.
        # Do NOT send via _alert() here — that routes through notify_error()
        # which adds "⚠️ BOT ERROR" prefix and duplicates the message.
        logger.info(
            "TRADE OPENED (MT5): ticket=%d | %s %.2f lots at %.2f | SL %.2f | TP2 %.2f",
            ticket, signal.direction.value, actual_volume, actual_price,
            signal.sl_price, signal.tp2_price
        )

        return signal.trade_id

    def _execute_paper(self, signal: Signal, lot_size: float) -> Optional[str]:
        """Execute in paper/shadow mode (no real orders)."""
        now = utc_now()
        utc_hour = now.hour
        risk_dollars = signal.risk_pips * self.cfg.strategy.pip_value_per_lot * lot_size

        # 🔴 LIVE RISK — c0_extreme: Stage-2 SL anchor.
        # Must match live path: C0 open (entry level = near-breakeven) for C1C0 entries.
        # Old code used c0_low/c0_high (the wick tip = Stage-1 SL) — that was a no-op.
        if signal.entry_type.value == "c1_c0_rejection":
            c0_extreme = signal.entry_price   # C0 open = near-breakeven target
        else:
            c0_extreme = None

        active = ActiveTrade(
            trade_id=signal.trade_id,
            direction=signal.direction,
            entry_price=signal.entry_price,
            lot_size=lot_size,
            sl_price=signal.sl_price,
            sl_stage=SLStage.STAGE_1,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            c0_extreme=c0_extreme,
        )
        self.trade_mgr.open_trade(active)

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

        context_record = {
            "trade_id": signal.trade_id,
            "sr_level_price": signal.sr_level.price,
            "sr_level_type": signal.sr_level.zone_type.value,
            "sr_touches": signal.sr_level.touches,
            "confidence_score": signal.confidence,
        }
        ctx = signal.context
        if "c1_body_ratio" in ctx:
            context_record["c1_body_ratio"] = ctx["c1_body_ratio"]
            context_record["c1_range_pips"] = ctx.get("c1_range_pips", 0)
            context_record["c1_is_wickless"] = 1 if ctx.get("c1_is_wickless") else 0
        if "trend" in ctx:
            context_record["trend_type"] = ctx["trend"]
        if "htf_aligned" in ctx:
            context_record["htf_zone_aligned"] = 1 if ctx["htf_aligned"] else 0
        self.db.insert_context(context_record)

        logger.info(
            "🔒 SHADOW TRADE: %s %s %.2f lots at %.2f | SL %.2f | TP1 %.2f | Risk $%.2f",
            signal.direction.value, self.cfg.strategy.symbol, lot_size,
            signal.entry_price, signal.sl_price, signal.tp1_price, risk_dollars
        )
        return signal.trade_id

    # ============================================================
    # STOP ORDER (C1 break-of-rejection entry — NEW MODEL)
    # ============================================================

    def _validate_stops_distance(self, entry: float, sl: float, tp: float,
                                 direction: TradeDirection) -> tuple:
        """
        🔴 LIVE RISK — Ensure SL/TP meet broker minimum distance from entry.
        Auto-push SL/TP outward if too close (prevents Error 10016).
        ICMarkets reports trade_stops_level=0 (dynamic min). Fall back to
        spread×3 or 5-pip floor. Returns (sl_adj, tp_adj).
        """
        info = mt5.symbol_info(self.cfg.mt5.symbol)
        if info is None:
            return sl, tp

        if info.trade_stops_level > 0:
            min_distance = info.trade_stops_level * info.point
        else:
            spread_price = info.spread * info.point
            min_distance = max(spread_price * 3.0, pips_to_price(5.0))

        point = info.point
        digits = self.mt5_conn._digits
        buffer = point * 10  # extra cushion past the exact minimum

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)

        sl_adj = sl
        tp_adj = tp

        if sl_dist < min_distance:
            logger.warning(
                "Stop order SL too close: %.2f (dist=%.2f, min=%.2f). Pushing out.",
                sl, sl_dist, min_distance
            )
            if direction == TradeDirection.BUY:
                sl_adj = round(entry - min_distance - buffer, digits)
            else:
                sl_adj = round(entry + min_distance + buffer, digits)

        if tp_dist < min_distance:
            logger.warning(
                "Stop order TP too close: %.2f (dist=%.2f, min=%.2f). Pushing out.",
                tp, tp_dist, min_distance
            )
            if direction == TradeDirection.BUY:
                tp_adj = round(entry + min_distance + buffer, digits)
            else:
                tp_adj = round(entry - min_distance - buffer, digits)

        logger.info(
            "Stop order stops check: stops_level=%d pts, min=%.2f, SL dist=%.2f, TP dist=%.2f",
            info.trade_stops_level, min_distance, sl_dist, tp_dist
        )
        return sl_adj, tp_adj

    def place_stop_order(self, signal: Signal, lots: float) -> Optional[int]:
        """
        🔴 LIVE RISK — Place MT5 pending stop order at signal.entry_price.
        BUY_STOP fires when market rises to trigger price (C1 high + buffer).
        SELL_STOP fires when market falls to trigger price (C1 low - buffer).

        Reuses stops-distance validation to auto-push SL/TP outward if broker
        minimum not met.

        Returns MT5 pending order ticket (NOT a position ticket) or None.
        The broker converts pending → position on trigger; reconciler picks up.
        """
        if not self.cfg.live_mode:
            # Shadow mode — log intended stop order, no real order placed
            logger.info(
                "🔒 SHADOW STOP ORDER: %s %.2f lots trigger=%.2f SL=%.2f TP1=%.2f",
                signal.direction.value, lots, signal.entry_price,
                signal.sl_price, signal.tp1_price
            )
            return None

        if mt5 is None:
            logger.error("MetaTrader5 module not available")
            return None

        # Validate lot size
        lots = self._validate_lot_size(lots)
        if lots <= 0:
            logger.error("Invalid lot size for stop order: %.4f", lots)
            return None

        # Check symbol
        symbol_info = mt5.symbol_info(self.cfg.mt5.symbol)
        if symbol_info is None:
            logger.error("Symbol %s not found for stop order", self.cfg.mt5.symbol)
            return None

        # Pick stop order type
        if signal.direction == TradeDirection.BUY:
            order_type = mt5.ORDER_TYPE_BUY_STOP
        else:
            order_type = mt5.ORDER_TYPE_SELL_STOP

        # 🔴 LIVE RISK — Send TP=TP2 to broker (not TP1).
        # Bot handles TP1 partial close (50%) internally via trade_manager on every tick.
        # Broker holds TP2 as the safety net: if bot crashes after fill, the
        # position still closes at full target instead of running forever.
        sl_adj, tp_adj = self._validate_stops_distance(
            signal.entry_price, signal.sl_price, signal.tp2_price,
            signal.direction,
        )
        signal.sl_price = sl_adj  # keep Signal in sync

        # 🔴 LIVE RISK — Compute broker-side expiry capability.
        # Bot's _expire_pending_stops is the primary canceler; broker-side expiration is
        # defense-in-depth for the case where bot crashes mid-bar.
        # ICMarkets and many retail brokers do NOT advertise SYMBOL_EXPIRATION_SPECIFIED
        # in their bitmask — sending ORDER_TIME_SPECIFIED then triggers retcode 10022
        # ("Invalid expiration"). Fall back to GTC when SPECIFIED unsupported.
        expiry_bars = signal.context.get("stop_expiry_bars", self.cfg.strategy.pending_expiry_bars)
        expiration_ts = int(time.time()) + (expiry_bars * 15 * 60) + 60

        # Bitmask: 1=GTC, 2=DAY, 4=SPECIFIED, 8=SPECIFIED_DAY
        # Use literal 4 in case the named constant is missing on older mt5 module versions.
        exp_mode = getattr(symbol_info, "expiration_mode", 0) or 0
        SYMBOL_EXP_SPECIFIED = getattr(mt5, "SYMBOL_EXPIRATION_SPECIFIED", 4)
        supports_specified = bool(exp_mode & SYMBOL_EXP_SPECIFIED)

        digits = self.mt5_conn._digits
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.cfg.mt5.symbol,
            "volume": lots,
            "type": order_type,
            "price": round(signal.entry_price, digits),  # Stop trigger
            "sl": round(sl_adj, digits),
            "tp": round(tp_adj, digits),
            "deviation": self.cfg.mt5.max_slippage,
            "magic": self.cfg.mt5.magic_number,
            "comment": f"ANiSTOP_{signal.trade_id[:8]}",
            "type_filling": self._get_filling_mode(),
        }

        if supports_specified:
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = expiration_ts
            expiry_log = f"expires={expiration_ts - int(time.time())}s (broker)"
        else:
            request["type_time"] = mt5.ORDER_TIME_GTC
            expiry_log = f"GTC (broker mode={exp_mode}, bot cancels in {expiry_bars} bars)"

        logger.info(
            "🔴 PLACING STOP: %s %.2f lots | trigger=%.2f SL=%.2f TP=%.2f | %s | Magic=%d",
            signal.direction.value, lots, signal.entry_price,
            sl_adj, tp_adj, expiry_log, self.cfg.mt5.magic_number
        )

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            logger.error("Stop order order_send returned None: %s", err)
            self._alert(f"❌ STOP ORDER FAILED (None): {err}")
            return None

        # 🔴 LIVE RISK — Retcode 10022 = TRADE_RETCODE_INVALID_EXPIRATION.
        # Some brokers advertise SYMBOL_EXPIRATION_SPECIFIED in the bitmask but
        # still reject the actual timestamp (per-instrument or per-account quirks).
        # Auto-retry once with GTC fallback so the trade still fires.
        if result.retcode == 10022 and request.get("type_time") == mt5.ORDER_TIME_SPECIFIED:
            logger.warning(
                "Stop order rejected with 10022 (Invalid expiration) despite "
                "broker advertising SPECIFIED support. Retrying with GTC..."
            )
            request.pop("expiration", None)
            request["type_time"] = mt5.ORDER_TIME_GTC
            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                logger.error("Stop order GTC retry returned None: %s", err)
                self._alert(f"❌ STOP ORDER FAILED (GTC retry, None): {err}")
                return None

        if result.retcode != TRADE_RETCODE_DONE:
            logger.error(
                "❌ STOP ORDER REJECTED: retcode=%d comment=%s signal=%s",
                result.retcode, result.comment, signal.trade_id
            )
            self._alert(
                f"❌ Stop order REJECTED\n"
                f"Code: {result.retcode}\n"
                f"Comment: {result.comment}\n"
                f"{signal.direction.value} trigger={signal.entry_price:.2f}"
            )
            return None

        ticket = result.order
        self._ticket_map[signal.trade_id] = ticket
        logger.info(
            "✅ STOP ORDER PLACED: ticket=%d %s %.2f lots @ trigger=%.2f SL=%.2f TP=%.2f",
            ticket, signal.direction.value, lots,
            signal.entry_price, sl_adj, tp_adj
        )
        return ticket

    def cancel_stop_order(self, ticket: int) -> bool:
        """
        Cancel pending stop order by ticket. Returns True on success.
        Called on bar-expiry (C0 validated but C1 breakout never came within 1 bar).
        """
        if not self.cfg.live_mode:
            logger.info("🔒 SHADOW: Would cancel stop order ticket=%d", ticket)
            return True

        if mt5 is None:
            return False

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }
        result = mt5.order_send(request)
        if result is None:
            logger.warning("cancel_stop_order order_send None: %s", mt5.last_error())
            return False

        if result.retcode != TRADE_RETCODE_DONE:
            logger.warning(
                "Stop order cancel FAILED: ticket=%d retcode=%d comment=%s",
                ticket, result.retcode, result.comment
            )
            return False

        logger.info("Stop order %d CANCELED (expired)", ticket)
        return True

    # ============================================================
    # SL MODIFICATION (for 3-stage SL system)
    # ============================================================

    def modify_sl(self, trade_id: str, new_sl: float) -> Optional[float]:
        """
        🔴 LIVE RISK — Modify stop loss on an existing MT5 position.
        Used by the 3-stage SL system.

        Returns the actual SL price set on the broker (may be clamped if the
        target was within broker minimum distance), or None if the modify failed
        completely (ticket not found, position gone, or unexpected retcode).
        Caller must update trade.sl_price with the returned value.
        """
        if not self.cfg.live_mode:
            logger.info("🔒 SHADOW: Would modify SL for %s to %.2f", trade_id, new_sl)
            return new_sl

        ticket = self._ticket_map.get(trade_id)
        if ticket is None:
            logger.warning("No MT5 ticket found for trade %s", trade_id)
            return None

        # Get current position to preserve TP
        positions = mt5.positions_get(ticket=ticket)
        if not positions or len(positions) == 0:
            logger.warning("Position %d not found in MT5", ticket)
            return None

        pos = positions[0]

        # 🔴 LIVE RISK — Clamp the requested SL to a broker-valid, never-looser
        # level using a FRESH tick. The old logic measured distance from a
        # single (possibly stale) pos.price_current and only nudged the SL when
        # it was too CLOSE; it could not fix a target that landed on the wrong
        # side of the live Ask/Bid, so a valid breakeven move was rejected with
        # INVALID_STOPS every tick and the stop never advanced past Stage 1.
        info = mt5.symbol_info(self.cfg.mt5.symbol)
        if info is None:
            logger.warning("SL modify: no symbol_info for %s (ticket=%d)",
                           self.cfg.mt5.symbol, ticket)
            return None

        if info.trade_stops_level > 0:
            min_distance = info.trade_stops_level * info.point
        else:
            # ICMarkets reports trade_stops_level=0 (dynamic minimum) for XAUUSD.
            # Fall back to spread×3, floor 5 pips ($0.50).
            spread_price = info.spread * info.point
            min_distance = max(spread_price * 3.0, pips_to_price(5.0))

        is_buy = (pos.type == getattr(mt5, "POSITION_TYPE_BUY", 0))
        target_sl = new_sl
        digits = self.mt5_conn._digits

        def _attempt(min_dist: float) -> Optional[float]:
            # Re-read the tick each attempt so the clamp uses live Ask/Bid.
            tk = mt5.symbol_info_tick(self.cfg.mt5.symbol)
            if tk is None:
                logger.warning("SL modify: no tick for %s (ticket=%d)",
                               self.cfg.mt5.symbol, ticket)
                return None
            sl = clamp_sl_to_valid(is_buy, target_sl, pos.sl,
                                   tk.ask, tk.bid, min_dist, digits)
            if sl is None:
                logger.info(
                    "SL modify skipped: no broker-valid SL tighter than current "
                    "%.2f (target=%.2f, ask=%.2f, bid=%.2f, min_dist=%.2f, %s)",
                    pos.sl, target_sl, tk.ask, tk.bid, min_dist,
                    "BUY" if is_buy else "SELL",
                )
                return None
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.cfg.mt5.symbol,
                "position": ticket,
                "sl": sl,
                "tp": pos.tp,  # Keep existing TP
            }
            logger.info(
                "Modifying SL: ticket=%d, old=%.2f -> new=%.2f "
                "(target=%.2f, ask=%.2f, bid=%.2f, min_dist=%.2f)",
                ticket, pos.sl, sl, target_sl, tk.ask, tk.bid, min_dist,
            )
            res = mt5.order_send(request)
            if res is not None and res.retcode == TRADE_RETCODE_DONE:
                logger.info("SL modified OK: ticket=%d -> %.2f", ticket, sl)
                return sl
            rc = res.retcode if res else -1
            err = res.comment if res else str(mt5.last_error())
            if rc == TRADE_RETCODE_INVALID_STOPS:
                logger.warning(
                    "SL modify INVALID_STOPS: ticket=%d sl=%.2f (min_dist=%.2f) "
                    "— retrying wider", ticket, sl, min_dist,
                )
            else:
                logger.error("SL modify FAILED: ticket=%d retcode=%d error=%s",
                             ticket, rc, err)
            return None

        # Attempt at broker minimum; on rejection retry once with a wider buffer
        # (covers brokers whose effective minimum exceeds the reported value,
        # e.g. spread blowout during news).
        result_sl = _attempt(min_distance)
        if result_sl is None:
            result_sl = _attempt(min_distance * 1.5)
        if result_sl is None:
            logger.warning(
                "SL modify could not place a valid tighter SL for %s "
                "(target=%.2f, current=%.2f). Caller will revert/retry next tick.",
                trade_id, target_sl, pos.sl,
            )
        return result_sl

    def modify_tp(self, trade_id: str, new_tp: float) -> bool:
        """Modify take profit on an existing MT5 position (for TP2 after TP1)."""
        if not self.cfg.live_mode:
            logger.info("🔒 SHADOW: Would modify TP for %s to %.2f", trade_id, new_tp)
            return True

        ticket = self._ticket_map.get(trade_id)
        if ticket is None:
            return False

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        pos = positions[0]

        if abs(pos.tp - new_tp) < 0.01:
            logger.debug("TP already at %.2f for ticket %d — skip modify", new_tp, ticket)
            return True

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.cfg.mt5.symbol,
            "position": ticket,
            "sl": pos.sl,
            "tp": round(new_tp, self.mt5_conn._digits),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != TRADE_RETCODE_DONE:
            error = result.comment if result else mt5.last_error()
            logger.error("TP modify FAILED for ticket %d: %s", ticket, error)
            return False

        logger.info("✅ TP modified: ticket=%d -> %.2f", ticket, new_tp)
        return True

    # ============================================================
    # PARTIAL CLOSE (for TP1 — close 50%)
    # ============================================================

    def partial_close(self, trade_id: str, lots_to_close: float,
                      reason: str = "TP1") -> bool:
        """
        🔴 LIVE RISK — Partially close a position (e.g., 50% at TP1).
        """
        if not self.cfg.live_mode:
            logger.info("🔒 SHADOW: Would partial close %s — %.2f lots (%s)",
                        trade_id, lots_to_close, reason)
            return True

        ticket = self._ticket_map.get(trade_id)
        if ticket is None:
            logger.warning("No MT5 ticket for partial close: %s", trade_id)
            return False

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning("Position %d not found for partial close", ticket)
            return False

        pos = positions[0]
        lots_to_close = self._validate_lot_size(lots_to_close)

        # Opposite order type to close
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.cfg.mt5.symbol)
        close_price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.mt5.symbol,
            "volume": lots_to_close,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "magic": self.cfg.mt5.magic_number,
            "comment": f"ANi_{reason}",
            "type_filling": self._get_filling_mode(),
            "deviation": self.cfg.mt5.max_slippage,
        }

        logger.info("📤 PARTIAL CLOSE: ticket=%d, %.2f lots (%s)",
                     ticket, lots_to_close, reason)

        result = mt5.order_send(request)
        if result is None or result.retcode != TRADE_RETCODE_DONE:
            error = result.comment if result else mt5.last_error()
            logger.error("Partial close FAILED: ticket=%d, %s", ticket, error)
            self._alert(f"❌ Partial close FAILED: ticket {ticket} — {error}")
            return False

        logger.info("✅ PARTIAL CLOSE: ticket=%d, %.2f lots closed at %.2f",
                     ticket, lots_to_close, result.price)
        return True

    # ============================================================
    # FULL CLOSE
    # ============================================================

    def close_position(self, trade_id: str, reason: str = "signal") -> bool:
        """
        🔴 LIVE RISK — Fully close an MT5 position.
        """
        if not self.cfg.live_mode:
            logger.info("🔒 SHADOW: Would close %s (%s)", trade_id, reason)
            return True

        ticket = self._ticket_map.get(trade_id)
        if ticket is None:
            logger.warning("No MT5 ticket for close: %s", trade_id)
            return False

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.info("Position %d already closed", ticket)
            return True  # Already closed

        pos = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.cfg.mt5.symbol)
        close_price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.mt5.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "magic": self.cfg.mt5.magic_number,
            "comment": f"ANi_{reason}",
            "type_filling": self._get_filling_mode(),
            "deviation": self.cfg.mt5.max_slippage,
        }

        logger.info("📤 CLOSING: ticket=%d, %.2f lots (%s)",
                     ticket, pos.volume, reason)

        result = mt5.order_send(request)
        if result is None or result.retcode != TRADE_RETCODE_DONE:
            error = result.comment if result else mt5.last_error()
            logger.error("Close FAILED: ticket=%d, %s", ticket, error)
            self._alert(f"❌ Close FAILED: ticket {ticket} — {error}")
            return False

        logger.info("✅ CLOSED: ticket=%d at %.2f", ticket, result.price)
        self._refresh_balance()
        return True

    # ============================================================
    # PROCESS EXITS (same interface as PaperTrader)
    # ============================================================

    def process_exits(self, exit_events: List[Dict]):
        """Process trade exit events from TradeManager."""
        for event in exit_events:
            self._process_exit(event)

    def _process_exit(self, event: Dict):
        """Process a single trade exit — execute on MT5 if live."""
        trade_id = event["trade_id"]
        trade = self.db.get_trade(trade_id)
        if not trade:
            logger.error("Trade not found for exit: %s", trade_id)
            return

        exit_price = event["exit_price"]
        lots_closed = event["lots_closed"]
        reason = event["reason"]
        full_close = event["full_close"]

        # For live mode: check if broker already closed it (e.g., SL hit by broker)
        if self.cfg.live_mode:
            ticket = self._ticket_map.get(trade_id)
            if ticket:
                positions = mt5.positions_get(ticket=ticket)
                if not positions:
                    # Broker already closed it (SL/TP hit server-side)
                    logger.info("Position %d already closed by broker", ticket)
                    # Get actual close price from deal history, then validate it
                    # against the trade's own SL/TP levels so a wrong/stale deal
                    # lookup cannot overwrite a clean exit with a fabricated
                    # price (the breakeven-reported-as-loss bug).
                    raw_close = self._get_deal_close_price(ticket)
                    validated, trusted = reconcile_close_price(
                        raw_close,
                        trade.get("sl_current") or trade.get("sl_initial") or exit_price,
                        trade.get("tp1_price") or 0.0,
                        trade.get("tp2_price") or 0.0,
                        default_price=exit_price,
                    )
                    if validated and validated > 0:
                        exit_price = validated
                    if not trusted and raw_close is not None:
                        logger.warning(
                            "Close price for %s snapped to %.2f (raw=%s) — "
                            "deal lookup inconsistent with SL/TP levels",
                            trade_id, exit_price, raw_close,
                        )
                elif full_close:
                    # We need to close it ourselves (giveback, session end, etc.)
                    self.close_position(trade_id,
                                       reason.value if hasattr(reason, "value") else str(reason))
                elif not full_close:
                    # Partial close (TP1) — only proceed if partial_close succeeds.
                    # If lot rounding forces full size (< broker min lot), skip partial
                    # so trade continues on full size rather than closing 100% unintentionally.
                    partial_ok = self.partial_close(trade_id, lots_closed, "TP1")
                    if not partial_ok:
                        # Revert tp1_hit so trade_manager re-checks next tick
                        active = self.trade_mgr.active_trades.get(trade_id)
                        if active:
                            active.tp1_hit = False
                        logger.warning(
                            "TP1 partial close failed for %s — tp1_hit reverted, "
                            "trade continues full size toward TP2", trade_id
                        )
                        return
                    # TP already set to TP2 on broker (placed via place_stop_order with TP2).
                    # modify_tp is a no-op if already correct, but call anyway to guard
                    # against broker-side TP drift (slippage, manual edits, etc.).
                    active = self.trade_mgr.active_trades.get(trade_id)
                    if active and active.tp2_price:
                        self.modify_tp(trade_id, active.tp2_price)

        # Calculate P&L (same as PaperTrader)
        direction = TradeDirection(trade["direction"])
        if direction == TradeDirection.BUY:
            pnl_pips = price_to_pips(exit_price - trade["entry_price"])
        else:
            pnl_pips = price_to_pips(trade["entry_price"] - exit_price)

        if (direction == TradeDirection.BUY and exit_price < trade["entry_price"]) or \
           (direction == TradeDirection.SELL and exit_price > trade["entry_price"]):
            pnl_pips = -abs(pnl_pips)

        pnl_dollars = pnl_pips * self.cfg.strategy.pip_value_per_lot * lots_closed
        rr = pnl_pips / trade["risk_pips"] if trade["risk_pips"] > 0 else 0

        # Update balance
        if self.cfg.live_mode:
            self._refresh_balance()  # Get real balance from MT5
        else:
            self.balance += pnl_dollars  # Shadow mode: track locally

        if full_close:
            computed_total = pnl_dollars + self._partial_pnl.pop(trade_id, 0.0)

            # 🔴 MT5 TRUTH — prefer the broker's ACTUAL realized P/L (sum of the
            # position's OUT deals: profit+commission+swap) over the price-model
            # estimate. The price model drifts from MT5 (real fills, commission,
            # swap) and the close-price reconcile can snap to BE when the real
            # close was in profit — under-reporting the trade (e.g. bot $58 vs
            # MT5 $75.50). Live: use MT5 realized + actual final close price.
            # Fall back to the computed total when deals are unavailable (shadow).
            ticket = self._ticket_map.get(trade_id)
            total_pnl_dollars = computed_total
            if self.cfg.live_mode and ticket:
                realized, mt5_exit = self._get_position_realized(ticket)
                if realized is not None:
                    total_pnl_dollars = realized
                    if mt5_exit:
                        exit_price = mt5_exit  # actual final close (not snapped/target)

            # Blended pips/RR over the ORIGINAL lot size so pips, $ and R:R agree.
            orig_lots = trade.get("lot_size") or lots_closed
            total_pnl_pips = trade_total_pips(
                total_pnl_dollars, self.cfg.strategy.pip_value_per_lot,
                orig_lots, pnl_pips,
            )
            total_rr = (total_pnl_pips / trade["risk_pips"]
                        if trade.get("risk_pips", 0) and trade["risk_pips"] > 0 else 0.0)

            self.db.close_trade(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=reason.value if isinstance(reason, ExitReason) else reason,
                pnl_pips=total_pnl_pips,
                pnl_dollars=total_pnl_dollars,
                rr_achieved=total_rr,
                balance_after=self.balance,
            )

            is_win = total_pnl_dollars > 0
            self.risk_mgr.update_after_trade(is_win)

            logger.info(
                "TRADE CLOSED: %s | %s | PnL: %.1f pips ($%.2f) | RR: %.2f | Balance: $%.2f",
                trade_id, reason.value if isinstance(reason, ExitReason) else reason,
                total_pnl_pips, total_pnl_dollars, total_rr, self.balance
            )
            # NOTE: Telegram notification for trade close is sent by main.py
            # via telegram.notify_trade_closed() after process_exits() returns.
            # Do NOT send via _alert() here — duplicates the message and routes
            # it through notify_error() which adds "⚠️ BOT ERROR" prefix.

            # Clean up ticket map
            self._ticket_map.pop(trade_id, None)
        else:
            self._partial_pnl[trade_id] = self._partial_pnl.get(trade_id, 0.0) + pnl_dollars
            self.db.update_trade(trade_id, {
                "partial_close_pct": event.get("partial_close_pct", 0.5),
                "tp1_hit": 1,   # persist so a restart after TP1 doesn't re-fire a 2nd partial
            })
            logger.info(
                "PARTIAL CLOSE: %s | TP1 | %.2f lots | PnL: +%.1f pips ($%.2f)",
                trade_id, lots_closed, pnl_pips, pnl_dollars
            )

    def _get_position_realized(self, ticket: int):
        """
        🔴 MT5 TRUTH — realized P/L for a closed position from deal history.

        Returns (total_dollars, final_exit_price), summing profit+commission+swap
        across the position's OUT deals (entry==1), or (None, None) if no deals.
        This is the broker's ACTUAL realized P/L in account currency — accounts
        for real fills, partial closes, commission and swap — so the report can
        equal the MT5 terminal. Price-model PnL only approximates it.
        """
        try:
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            deals = mt5.history_deals_get(now - timedelta(days=7), now,
                                          position=ticket)
            if not deals:
                return None, None
            out = [d for d in deals if getattr(d, "entry", 0) == 1]  # DEAL_ENTRY_OUT
            if not out:
                return None, None
            total = sum(float(d.profit) + float(d.commission) + float(d.swap)
                        for d in out)
            final_price = float(out[-1].price)  # last OUT deal = final remainder close
            return round(total, 2), final_price
        except Exception as e:
            logger.warning("Could not read MT5 realized P/L for ticket %d: %s",
                           ticket, e)
            return None, None

    def _get_deal_close_price(self, ticket: int) -> Optional[float]:
        """Get actual close price from MT5 deal history."""
        try:
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            deals = mt5.history_deals_get(
                now - timedelta(days=7), now,
                position=ticket
            )
            if deals:
                # Last deal for this position is the closing deal
                for deal in reversed(deals):
                    if deal.entry == 1:  # 1 = out (closing)
                        return float(deal.price)
        except Exception as e:
            logger.warning("Could not get deal history for ticket %d: %s", ticket, e)
        return None

    # ============================================================
    # STATUS (same interface as PaperTrader)
    # ============================================================

    def get_status(self) -> Dict:
        """Get current trading status."""
        # Refresh balance from MT5 if connected
        if self.cfg.live_mode:
            self._refresh_balance()

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
            "mode": "LIVE" if self.cfg.live_mode else "SHADOW",
        }

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
