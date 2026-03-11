"""
Raja Banks Trading Agent - Trade Manager
3-stage SL, TP1/TP2, give-back exit, emergency exit, risk add-back after TP1.
"""
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

from config import StrategyConfig
from constants import TradeDirection, SLStage, ExitReason
from utils import price_to_pips, pips_to_price

logger = logging.getLogger(__name__)


@dataclass
class ActiveTrade:
    """In-memory representation of an active trade."""
    trade_id: str
    direction: TradeDirection
    entry_price: float
    lot_size: float
    sl_price: float
    sl_stage: SLStage
    tp1_price: float
    tp2_price: float
    tp1_hit: bool = False
    partial_close_pct: float = 0.0
    max_favorable_pips: float = 0.0  # Tracks best price reached
    c0_extreme: Optional[float] = None  # For Stage 2 SL

    @property
    def remaining_lots(self) -> float:
        return round(self.lot_size * (1.0 - self.partial_close_pct), 2)


class TradeManager:
    """
    Manages active trades: SL staging, TP hits, give-back, emergency exits.

    Raja's 3-Stage Stop Loss:
    Stage 1: Initial SL at C1 extreme (wick tip)
    Stage 2: After +20 pips, move SL to C0 extreme
    Stage 3: After +40 pips, move SL to breakeven

    Give-back Exit: If price goes +30 pips favorable then pulls back 50%,
    close remaining position.
    """

    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.active_trades: Dict[str, ActiveTrade] = {}

    def open_trade(self, trade: ActiveTrade):
        """Register a new active trade."""
        self.active_trades[trade.trade_id] = trade
        logger.info("Trade opened: %s %s at %.2f, SL %.2f, TP1 %.2f, TP2 %.2f",
                     trade.trade_id, trade.direction.value,
                     trade.entry_price, trade.sl_price,
                     trade.tp1_price, trade.tp2_price)

    def update_trades(self, current_price: float) -> List[Dict]:
        """
        Update all active trades with current price.
        Returns list of exit events: {trade_id, reason, exit_price, ...}
        """
        exits = []
        for trade_id, trade in list(self.active_trades.items()):
            exit_event = self._update_single_trade(trade, current_price)
            if exit_event:
                exits.append(exit_event)
                del self.active_trades[trade_id]
        return exits

    def update_trades_ohlc(self, open_p: float, high: float,
                           low: float, close: float) -> List[Dict]:
        """
        Update trades using full OHLC bar data for accurate backtest.
        Checks SL against adverse extreme, TP against favorable extreme.
        For BUY: SL checked against low, TP checked against high.
        For SELL: SL checked against high, TP checked against low.
        """
        exits = []
        for trade_id, trade in list(self.active_trades.items()):
            # Determine which prices to check
            if trade.direction == TradeDirection.BUY:
                sl_check_price = low    # SL hit by low
                tp_check_price = high   # TP hit by high
            else:
                sl_check_price = high   # SL hit by high
                tp_check_price = low    # TP hit by low

            # Check SL first (worst case within bar)
            sl_event = self._check_sl(trade, sl_check_price)
            if sl_event:
                # Conservative: when SL and TP both hit in same bar, SL wins
                # (we can't know intra-bar order from OHLC data)
                exits.append(sl_event)
                del self.active_trades[trade_id]
                continue

            # Update favorable distance using best price in bar
            if trade.direction == TradeDirection.BUY:
                favorable_pips = price_to_pips(high - trade.entry_price)
            else:
                favorable_pips = price_to_pips(trade.entry_price - low)
            if favorable_pips > trade.max_favorable_pips:
                trade.max_favorable_pips = favorable_pips

            # Check TP1 with favorable extreme
            tp1_event = self._check_tp1(trade, tp_check_price)
            if tp1_event:
                exits.append(tp1_event)
                # Don't delete — trade continues with remaining lots
                # Advance SL stages based on favorable movement
                self._update_sl_stage(trade, favorable_pips)
                # Check TP2
                tp2_event = self._check_tp2(trade, tp_check_price)
                if tp2_event:
                    exits.append(tp2_event)
                    del self.active_trades[trade_id]
                continue

            # Check TP2
            tp2_event = self._check_tp2(trade, tp_check_price)
            if tp2_event:
                exits.append(tp2_event)
                del self.active_trades[trade_id]
                continue

            # Update SL stages
            self._update_sl_stage(trade, favorable_pips)

            # Give-back check using close (end-of-bar position)
            if trade.direction == TradeDirection.BUY:
                close_favorable = price_to_pips(close - trade.entry_price)
            else:
                close_favorable = price_to_pips(trade.entry_price - close)
            giveback = self._check_giveback(trade, close_favorable)
            if giveback:
                exits.append(giveback)
                del self.active_trades[trade_id]

        return exits

    def _update_single_trade(self, trade: ActiveTrade,
                             price: float) -> Optional[Dict]:
        """Process a single trade against current price."""

        # Calculate current favorable distance
        if trade.direction == TradeDirection.BUY:
            favorable_pips = price_to_pips(price - trade.entry_price)
            adverse_pips = price_to_pips(trade.entry_price - price)
        else:
            favorable_pips = price_to_pips(trade.entry_price - price)
            adverse_pips = price_to_pips(price - trade.entry_price)

        # Track max favorable
        if favorable_pips > trade.max_favorable_pips:
            trade.max_favorable_pips = favorable_pips

        # === CHECK STOP LOSS ===
        sl_hit = self._check_sl(trade, price)
        if sl_hit:
            return sl_hit

        # === CHECK TP1 ===
        tp1_event = self._check_tp1(trade, price)
        if tp1_event:
            return tp1_event  # Partial close, trade continues

        # === CHECK TP2 ===
        tp2_event = self._check_tp2(trade, price)
        if tp2_event:
            return tp2_event

        # === UPDATE SL STAGES ===
        self._update_sl_stage(trade, favorable_pips)

        # === GIVE-BACK EXIT ===
        giveback = self._check_giveback(trade, favorable_pips)
        if giveback:
            return giveback

        # === EMERGENCY EXIT (small body cluster) ===
        # Note: This is checked per-bar in the main loop with candle data

        return None

    def _check_sl(self, trade: ActiveTrade, price: float) -> Optional[Dict]:
        """Check if stop loss is hit."""
        hit = False
        if trade.direction == TradeDirection.BUY:
            hit = price <= trade.sl_price
        else:
            hit = price >= trade.sl_price

        if hit:
            reason_map = {
                SLStage.STAGE_1: ExitReason.SL_STAGE1,
                SLStage.STAGE_2: ExitReason.SL_STAGE2,
                SLStage.STAGE_3: ExitReason.SL_STAGE3,
            }
            reason = reason_map.get(trade.sl_stage, ExitReason.SL_STAGE1)
            pnl_pips = self._calc_pnl_pips(trade, trade.sl_price)

            logger.info("SL hit (%s): %s at %.2f, PnL %.1f pips",
                         trade.sl_stage.name, trade.trade_id,
                         trade.sl_price, pnl_pips)

            return {
                "trade_id": trade.trade_id,
                "reason": reason,
                "exit_price": trade.sl_price,
                "pnl_pips": pnl_pips,
                "lots_closed": trade.remaining_lots,
                "full_close": True,
            }
        return None

    def _check_tp1(self, trade: ActiveTrade, price: float) -> Optional[Dict]:
        """Check if TP1 is hit (partial close)."""
        if trade.tp1_hit:
            return None

        hit = False
        if trade.direction == TradeDirection.BUY:
            hit = price >= trade.tp1_price
        else:
            hit = price <= trade.tp1_price

        if hit:
            trade.tp1_hit = True
            close_lots = round(trade.lot_size * self.cfg.tp1_close_pct, 2)
            trade.partial_close_pct = self.cfg.tp1_close_pct

            pnl_pips = price_to_pips(abs(trade.tp1_price - trade.entry_price))

            logger.info("TP1 hit: %s at %.2f, closing %.2f lots (%.0f%%)",
                         trade.trade_id, price, close_lots,
                         self.cfg.tp1_close_pct * 100)

            return {
                "trade_id": trade.trade_id,
                "reason": ExitReason.TP1_HIT,
                "exit_price": trade.tp1_price,
                "pnl_pips": pnl_pips,
                "lots_closed": close_lots,
                "full_close": False,
            }
        return None

    def _check_tp2(self, trade: ActiveTrade, price: float) -> Optional[Dict]:
        """Check if TP2 is hit (full close of remaining)."""
        if not trade.tp1_hit:
            return None  # Must hit TP1 first

        hit = False
        if trade.direction == TradeDirection.BUY:
            hit = price >= trade.tp2_price
        else:
            hit = price <= trade.tp2_price

        if hit:
            pnl_pips = price_to_pips(abs(trade.tp2_price - trade.entry_price))

            logger.info("TP2 hit: %s at %.2f, full close",
                         trade.trade_id, price)

            return {
                "trade_id": trade.trade_id,
                "reason": ExitReason.TP2_HIT,
                "exit_price": trade.tp2_price,
                "pnl_pips": pnl_pips,
                "lots_closed": trade.remaining_lots,
                "full_close": True,
            }
        return None

    def _update_sl_stage(self, trade: ActiveTrade, favorable_pips: float):
        """
        Progress SL through Raja's 3 stages.
        Stage 1 -> Stage 2 at +20 pips (move to C0 extreme)
        Stage 2 -> Stage 3 at +40 pips (move to breakeven)
        """
        if trade.sl_stage == SLStage.STAGE_1 and \
           favorable_pips >= self.cfg.sl_stage2_pips:
            # Move to Stage 2: SL at C0 extreme
            if trade.c0_extreme is not None:
                old_sl = trade.sl_price
                trade.sl_price = trade.c0_extreme
                trade.sl_stage = SLStage.STAGE_2
                logger.info("SL Stage 2: %s moved SL %.2f -> %.2f (C0 extreme)",
                             trade.trade_id, old_sl, trade.sl_price)
            else:
                # If no C0 extreme stored, use a conservative move
                buffer = pips_to_price(self.cfg.sl_stage2_pips / 2)
                old_sl = trade.sl_price
                if trade.direction == TradeDirection.BUY:
                    trade.sl_price = max(trade.sl_price, trade.entry_price - buffer)
                else:
                    trade.sl_price = min(trade.sl_price, trade.entry_price + buffer)
                trade.sl_stage = SLStage.STAGE_2
                logger.info("SL Stage 2: %s moved SL %.2f -> %.2f (conservative)",
                             trade.trade_id, old_sl, trade.sl_price)

        elif trade.sl_stage == SLStage.STAGE_2 and \
             favorable_pips >= self.cfg.sl_stage3_pips:
            # Move to Stage 3: Breakeven
            old_sl = trade.sl_price
            trade.sl_price = trade.entry_price
            trade.sl_stage = SLStage.STAGE_3
            logger.info("SL Stage 3 (BE): %s moved SL %.2f -> %.2f",
                         trade.trade_id, old_sl, trade.sl_price)

        elif trade.sl_stage == SLStage.STAGE_3 and \
             getattr(self.cfg, 'trailing_enabled', True) and \
             favorable_pips >= getattr(self.cfg, 'trailing_activation_pips', 50.0):
            # Trailing stop: lock in profits by trailing SL behind price
            trail_dist = pips_to_price(getattr(self.cfg, 'trailing_distance_pips', 20.0))
            if trade.direction == TradeDirection.BUY:
                # For BUY: trail SL below the current high-water mark
                new_sl = trade.entry_price + pips_to_price(favorable_pips) - trail_dist
                if new_sl > trade.sl_price:
                    old_sl = trade.sl_price
                    trade.sl_price = round(new_sl, 2)
                    logger.info("Trailing SL: %s moved SL %.2f -> %.2f (+%.1f pips locked)",
                                trade.trade_id, old_sl, trade.sl_price,
                                price_to_pips(trade.sl_price - trade.entry_price))
            else:
                # For SELL: trail SL above the current low-water mark
                new_sl = trade.entry_price - pips_to_price(favorable_pips) + trail_dist
                if new_sl < trade.sl_price:
                    old_sl = trade.sl_price
                    trade.sl_price = round(new_sl, 2)
                    logger.info("Trailing SL: %s moved SL %.2f -> %.2f (+%.1f pips locked)",
                                trade.trade_id, old_sl, trade.sl_price,
                                price_to_pips(trade.entry_price - trade.sl_price))

    def _check_giveback(self, trade: ActiveTrade,
                        favorable_pips: float) -> Optional[Dict]:
        """
        Give-back exit: If price was +30 pips favorable and pulls back 50%,
        close remaining position.
        """
        if trade.max_favorable_pips < self.cfg.giveback_trigger_pips:
            return None

        pullback = trade.max_favorable_pips - favorable_pips
        pullback_pct = pullback / trade.max_favorable_pips if trade.max_favorable_pips > 0 else 0

        if pullback_pct >= self.cfg.giveback_close_pct:
            # Calculate exit price
            if trade.direction == TradeDirection.BUY:
                exit_price = trade.entry_price + pips_to_price(favorable_pips)
            else:
                exit_price = trade.entry_price - pips_to_price(favorable_pips)

            logger.info("Give-back exit: %s, max +%.1f pips, now +%.1f pips (%.0f%% pullback)",
                         trade.trade_id, trade.max_favorable_pips,
                         favorable_pips, pullback_pct * 100)

            return {
                "trade_id": trade.trade_id,
                "reason": ExitReason.GIVEBACK_EXIT,
                "exit_price": exit_price,
                "pnl_pips": favorable_pips,
                "lots_closed": trade.remaining_lots,
                "full_close": True,
            }
        return None

    def check_opposite_close(self, open_p: float, close_p: float,
                             current_price: float,
                             body_ratio_val: float = 0.0) -> List[Dict]:
        """
        Emergency exit on strong opposite candle — only when ALL conditions met:
        1. Trade is still in Stage 1 (hasn't moved in our favor yet)
        2. Opposite candle has strong body (>= 0.60 body ratio)
        3. Trade is currently in the red

        This prevents exiting on normal pullback candles which are part of
        healthy price action. The 3-stage SL handles normal adverse movement.
        """
        exits = []
        for trade_id, trade in list(self.active_trades.items()):
            opposite = False
            if trade.direction == TradeDirection.BUY and close_p < open_p:
                opposite = True
            elif trade.direction == TradeDirection.SELL and close_p > open_p:
                opposite = True

            if not opposite:
                continue

            # Only emergency exit in Stage 1 — once SL has been tightened,
            # let the SL system do its job
            if trade.sl_stage != SLStage.STAGE_1:
                continue

            # Only exit on STRONG opposite candles (body ratio >= 0.60)
            if body_ratio_val < 0.60:
                continue

            # Only exit if trade is currently losing
            pnl_pips = self._calc_pnl_pips(trade, current_price)
            if pnl_pips >= 0:
                continue

            logger.warning("Opposite close exit: %s at %.2f (strong opposite candle, body=%.2f, pnl=%.1f pips)",
                            trade_id, current_price, body_ratio_val, pnl_pips)

            exits.append({
                "trade_id": trade_id,
                "reason": ExitReason.EMERGENCY_EXIT,
                "exit_price": current_price,
                "pnl_pips": pnl_pips,
                "lots_closed": trade.remaining_lots,
                "full_close": True,
            })
            del self.active_trades[trade_id]

        return exits

    def _calc_pnl_pips(self, trade: ActiveTrade, price: float) -> float:
        """
        Calculate signed PnL in pips for a trade at a given price.
        Positive = profitable, negative = losing.
        """
        if trade.direction == TradeDirection.BUY:
            return price_to_pips(price - trade.entry_price) * (
                1 if price >= trade.entry_price else -1
            )
        else:
            return price_to_pips(trade.entry_price - price) * (
                1 if price <= trade.entry_price else -1
            )

    def emergency_exit(self, trade_id: str, current_price: float,
                       reason: str = "opposite_close") -> Optional[Dict]:
        """Force close a trade for emergency reasons."""
        trade = self.active_trades.get(trade_id)
        if not trade:
            return None

        pnl_pips = self._calc_pnl_pips(trade, current_price)

        logger.warning("Emergency exit: %s at %.2f (%s)",
                        trade_id, current_price, reason)

        del self.active_trades[trade_id]

        return {
            "trade_id": trade_id,
            "reason": ExitReason.EMERGENCY_EXIT,
            "exit_price": current_price,
            "pnl_pips": pnl_pips,
            "lots_closed": trade.remaining_lots,
            "full_close": True,
        }

    def session_end_close_all(self, current_price: float) -> List[Dict]:
        """Close all trades at session end."""
        exits = []
        for trade_id, trade in list(self.active_trades.items()):
            pnl_pips = self._calc_pnl_pips(trade, current_price)

            exits.append({
                "trade_id": trade_id,
                "reason": ExitReason.SESSION_END,
                "exit_price": current_price,
                "pnl_pips": pnl_pips,
                "lots_closed": trade.remaining_lots,
                "full_close": True,
            })
            del self.active_trades[trade_id]

        if exits:
            logger.info("Session end: closed %d trades", len(exits))
        return exits

    def restore_trades_from_db(self, open_trades: List[Dict]):
        """
        Restore active trades from database rows after bot restart.
        Each row should come from TradingDatabase.get_open_trades().
        """
        restored = 0
        for row in open_trades:
            try:
                direction = TradeDirection(row["direction"])
                sl_stage_val = row.get("sl_stage", 1)
                sl_stage = SLStage(sl_stage_val) if sl_stage_val in (1, 2, 3) else SLStage.STAGE_1

                trade = ActiveTrade(
                    trade_id=row["trade_id"],
                    direction=direction,
                    entry_price=row["entry_price"],
                    lot_size=row["lot_size"],
                    sl_price=row["sl_current"],
                    sl_stage=sl_stage,
                    tp1_price=row.get("tp1_price") or 0.0,
                    tp2_price=row.get("tp2_price") or 0.0,
                    tp1_hit=bool(row.get("tp1_hit", 0)),
                    partial_close_pct=row.get("partial_close_pct", 0.0),
                    max_favorable_pips=0.0,  # Will be re-calculated on next tick
                    c0_extreme=None,  # Lost on restart — conservative SL fallback used
                )
                self.active_trades[trade.trade_id] = trade
                restored += 1
                logger.info("Restored trade: %s %s at %.2f (SL %.2f, stage %s)",
                            trade.trade_id, trade.direction.value,
                            trade.entry_price, trade.sl_price, trade.sl_stage.name)
            except Exception as e:
                logger.error("Failed to restore trade %s: %s",
                             row.get("trade_id", "?"), e)

        if restored:
            logger.info("Restored %d open trade(s) from database", restored)
        return restored

    def get_active_trade_count(self) -> int:
        return len(self.active_trades)

    def get_active_trades(self) -> List[ActiveTrade]:
        return list(self.active_trades.values())
