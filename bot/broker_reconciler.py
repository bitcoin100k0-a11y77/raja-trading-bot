# broker_reconciler.py — Startup Position Reconciliation
"""
🔁 RECONCILE — Runs on every bot startup.

Compares local database open trades with MT5's actual open positions.
Prevents the #1 live bot killer: restarting and double-entering
because local state thinks no positions are open.

Flow:
1. Query MT5 for all positions with our magic number
2. Query local DB for all trades marked is_open=1
3. Match by trade_id in MT5 comment field
4. Fix discrepancies:
   - MT5 has position not in DB → WARNING + alert
   - DB has trade not in MT5 → mark as closed
   - SL/TP differ → sync from MT5 (broker is truth)
5. Return report for Telegram
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from database import TradingDatabase
from trade_manager import TradeManager, ActiveTrade
from constants import TradeDirection, SLStage

logger = logging.getLogger(__name__)


@dataclass
class ReconcileReport:
    """Report from reconciliation process."""
    mt5_positions: int = 0
    db_open_trades: int = 0
    matched: int = 0
    mt5_only: List[dict] = field(default_factory=list)    # In MT5 but not DB
    db_only: List[str] = field(default_factory=list)       # In DB but not MT5
    sl_synced: int = 0
    tp_synced: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "🔁 RECONCILIATION REPORT",
            f"MT5 positions: {self.mt5_positions}",
            f"DB open trades: {self.db_open_trades}",
            f"Matched: {self.matched}",
        ]
        if self.mt5_only:
            lines.append(f"⚠️ MT5-only (unknown): {len(self.mt5_only)}")
            for p in self.mt5_only:
                lines.append(f"  ticket={p['ticket']}, {p['volume']} lots at {p['price']}")
        if self.db_only:
            lines.append(f"⚠️ DB-only (closed by broker): {len(self.db_only)}")
            for tid in self.db_only:
                lines.append(f"  {tid}")
        if self.sl_synced > 0:
            lines.append(f"SL synced from MT5: {self.sl_synced}")
        if self.tp_synced > 0:
            lines.append(f"TP synced from MT5: {self.tp_synced}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  {e}")
        if not self.mt5_only and not self.db_only and not self.errors:
            lines.append("✅ All positions reconciled — safe to trade")
        return "\n".join(lines)


class BrokerReconciler:
    """
    🔁 RECONCILE — Startup safety check.

    Must run BEFORE any new trades are placed.
    Ensures local state matches broker reality.
    """

    def __init__(self, magic_number: int, symbol: str):
        self.magic_number = magic_number
        self.symbol = symbol

    def reconcile(self, db: TradingDatabase,
                  trade_mgr: TradeManager,
                  ticket_map: Dict[str, int]) -> ReconcileReport:
        """
        🔁 RECONCILE — Main reconciliation entry point.

        Args:
            db: Trading database
            trade_mgr: Trade manager (will restore matched trades)
            ticket_map: Dict to populate with trade_id -> MT5 ticket mapping

        Returns:
            ReconcileReport with all findings
        """
        report = ReconcileReport()

        # Step 1: Get MT5 positions with our magic number
        mt5_positions = self._get_mt5_positions()
        report.mt5_positions = len(mt5_positions)

        # Step 2: Get DB open trades
        db_open = db.get_open_trades()
        report.db_open_trades = len(db_open)

        logger.info("🔁 Reconciling: %d MT5 positions, %d DB open trades",
                     len(mt5_positions), len(db_open))

        # Step 3: Build lookup maps
        # MT5 positions indexed by trade_id extracted from comment
        mt5_by_trade_id: Dict[str, dict] = {}
        mt5_unmatched: List[dict] = []

        for pos in mt5_positions:
            trade_id = self._extract_trade_id(pos.get("comment", ""))
            if trade_id:
                mt5_by_trade_id[trade_id] = pos
            else:
                mt5_unmatched.append(pos)

        # DB trades indexed by trade_id
        db_by_trade_id: Dict[str, dict] = {}
        for row in db_open:
            db_by_trade_id[row["trade_id"]] = row

        # Step 4: Match and reconcile
        matched_trade_ids = set()

        for trade_id, db_trade in db_by_trade_id.items():
            if trade_id[:8] in [tid[:8] for tid in mt5_by_trade_id]:
                # Find the matching MT5 position
                mt5_trade_id = None
                for tid in mt5_by_trade_id:
                    if tid[:8] == trade_id[:8]:
                        mt5_trade_id = tid
                        break

                if mt5_trade_id is None:
                    continue

                mt5_pos = mt5_by_trade_id[mt5_trade_id]
                matched_trade_ids.add(trade_id)
                report.matched += 1

                # Map trade_id to MT5 ticket
                ticket_map[trade_id] = mt5_pos["ticket"]

                # Sync SL if different
                if abs(mt5_pos["sl"] - db_trade.get("sl_current", 0)) > 0.01:
                    logger.info("SL sync: %s DB=%.2f -> MT5=%.2f",
                               trade_id, db_trade.get("sl_current", 0), mt5_pos["sl"])
                    db.update_trade(trade_id, {"sl_current": mt5_pos["sl"]})
                    report.sl_synced += 1

                # Restore into TradeManager
                try:
                    direction = TradeDirection(db_trade["direction"])
                    sl_stage_val = db_trade.get("sl_stage", 1)
                    sl_stage = SLStage(sl_stage_val) if sl_stage_val in (1, 2, 3) else SLStage.STAGE_1

                    active = ActiveTrade(
                        trade_id=trade_id,
                        direction=direction,
                        entry_price=mt5_pos["price_open"],
                        lot_size=mt5_pos["volume"],
                        sl_price=mt5_pos["sl"],
                        sl_stage=sl_stage,
                        tp1_price=db_trade.get("tp1_price") or 0.0,
                        tp2_price=db_trade.get("tp2_price") or 0.0,
                        tp1_hit=bool(db_trade.get("tp1_hit", 0)),
                        partial_close_pct=db_trade.get("partial_close_pct", 0.0),
                        # Restore high-water mark so give-back pullback % is calculated
                        # from the true all-time peak, not from 0 (post-restart floor).
                        max_favorable_pips=db_trade.get("max_favorable_pips") or 0.0,
                    )
                    trade_mgr.open_trade(active)
                    logger.info("✅ Restored: %s (ticket=%d, %s %.2f lots)",
                               trade_id, mt5_pos["ticket"], direction.value, mt5_pos["volume"])
                except Exception as e:
                    report.errors.append(f"Failed to restore {trade_id}: {e}")
                    logger.error("Restore failed for %s: %s", trade_id, e)

        # Step 5: DB trades not in MT5 (broker closed them)
        for trade_id in db_by_trade_id:
            if trade_id not in matched_trade_ids:
                report.db_only.append(trade_id)
                logger.warning("⚠️ DB trade %s not found in MT5 — marking as closed", trade_id)
                try:
                    db.close_trade(
                        trade_id=trade_id,
                        exit_price=db_by_trade_id[trade_id].get("entry_price", 0),
                        exit_reason="broker_closed",
                        pnl_pips=0,
                        pnl_dollars=0,
                        rr_achieved=0,
                        balance_after=0,
                    )
                except Exception as e:
                    report.errors.append(f"Failed to close {trade_id}: {e}")

        # Step 6: MT5 positions not in DB (manual or unknown)
        for pos in mt5_unmatched:
            report.mt5_only.append(pos)
            logger.warning(
                "⚠️ MT5 position ticket=%d not in DB (manual trade? comment='%s')",
                pos["ticket"], pos.get("comment", "")
            )

        # Also check mt5_by_trade_id for unmatched
        for trade_id, pos in mt5_by_trade_id.items():
            matched = False
            for db_tid in db_by_trade_id:
                if db_tid[:8] == trade_id[:8]:
                    matched = True
                    break
            if not matched:
                report.mt5_only.append(pos)
                logger.warning("⚠️ MT5 position %s (ticket=%d) not in DB",
                             trade_id, pos["ticket"])

        logger.info("🔁 Reconciliation complete: %s", report.summary())
        return report

    def _get_mt5_positions(self) -> List[dict]:
        """Get all MT5 positions with our magic number."""
        positions = []
        try:
            all_pos = mt5.positions_get(symbol=self.symbol)
            if all_pos is None:
                return positions

            for pos in all_pos:
                if pos.magic == self.magic_number:
                    positions.append({
                        "ticket": pos.ticket,
                        "symbol": pos.symbol,
                        "type": pos.type,  # 0=BUY, 1=SELL
                        "volume": pos.volume,
                        "price_open": pos.price_open,
                        "sl": pos.sl,
                        "tp": pos.tp,
                        "profit": pos.profit,
                        "comment": pos.comment,
                        "magic": pos.magic,
                        "time": pos.time,
                    })
        except Exception as e:
            logger.error("Failed to get MT5 positions: %s", e)

        return positions

    def _extract_trade_id(self, comment: str) -> Optional[str]:
        """Extract trade_id from MT5 order comment.
        Supports two formats:
          - ANi_xxxxxxxx       → market entry (legacy)
          - ANiSTOP_xxxxxxxx   → filled pending stop order (new model)
        Both return the 8-char trade_id prefix.
        """
        if not comment:
            return None
        if comment.startswith("ANiSTOP_"):
            return comment[len("ANiSTOP_"):]
        if comment.startswith("ANi_"):
            return comment[len("ANi_"):]
        return None

    def check_stop_fills(self, pending_stop_orders: Dict[int, dict]) -> List[int]:
        """
        🔁 RECONCILE (runtime) — detect pending stop orders no longer live on broker.

        A ticket in pending_stop_orders that is NOT in mt5.orders_get() means one of:
          (a) Broker triggered the stop → position now open (trade_manager will
              pick it up once reconciler sees the new position).
          (b) User canceled manually via MT5 terminal.
          (c) Bot's own cancel_stop_order succeeded (already removed upstream —
              shouldn't reach here but harmless).

        Returns list of tickets main.py should remove from its pending_stop_orders
        dict. Called each tick BEFORE signal generation so state stays in sync.
        """
        if not pending_stop_orders:
            return []
        if mt5 is None:
            return []

        gone: List[int] = []
        try:
            live = mt5.orders_get(symbol=self.symbol)
            if live is None:
                logger.warning("orders_get returned None for %s", self.symbol)
                return []
            live_tickets = {o.ticket for o in live if o.magic == self.magic_number}
            for ticket in list(pending_stop_orders.keys()):
                if ticket not in live_tickets:
                    gone.append(ticket)
                    logger.info(
                        "🔁 Stop %d no longer pending on broker "
                        "(filled → position, or manually canceled)", ticket
                    )
        except Exception as e:
            logger.error("check_stop_fills failed: %s", e)
        return gone
