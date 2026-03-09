"""
Raja Banks Trading Agent - Learning Agent
Trade journal, 10 loss categories, pattern analysis every 5 trades,
blocking patterns with >70% loss rate, confidence scoring.
"""
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone

from config import LearningConfig
from constants import (
    EntryType, TradeDirection, ExitReason, LossCategory,
    TrendType, SessionType
)
from database import TradingDatabase

logger = logging.getLogger(__name__)


class LearningAgent:
    """
    Adaptive learning system that learns from every trade.

    Key behaviors:
    1. Categorizes every loss into one of 10 categories
    2. Builds pattern keys from trade context
    3. Analyzes patterns every N trades
    4. Blocks patterns with >70% loss rate (min 5 trades)
    5. Scores signal confidence based on blocked pattern matches
    6. Immediately rejects patterns with >80% loss rate
    """

    def __init__(self, config: LearningConfig, db: TradingDatabase):
        self.cfg = config
        self.db = db
        self._trade_counter = 0

    # === LOSS CATEGORIZATION ===

    def categorize_loss(self, trade: Dict, context: Optional[Dict] = None) -> List[str]:
        """
        Assign loss categories to a losing trade.
        Raja's 10 loss categories:
        1. false_breakout - Price faked through S/R then reversed
        2. early_reversal - Reversal came before TP1
        3. stage2_pullback - Hit SL after Stage 2 move
        4. breakeven_loss - Stopped out at breakeven (Stage 3)
        5. giveback_loss - Gave back profits
        6. session_quality - Bad session/time for trading
        7. high_spread - Spread was too wide
        8. tight_sl - SL was too close
        9. wide_range - Candle range was too large
        10. overcrowded_sr - Too many levels near entry
        """
        categories = []

        if not context:
            context = self.db.get_context(trade["trade_id"]) or {}

        exit_reason = trade.get("exit_reason", "")
        risk_pips = trade.get("risk_pips", 0)
        pnl_pips = trade.get("pnl_pips", 0)
        entry_type = trade.get("entry_type", "")

        # False breakout: SL hit at Stage 1 with small adverse move
        if exit_reason in (ExitReason.SL_STAGE1.value, "sl_stage1"):
            if abs(pnl_pips) <= risk_pips * 1.1:
                categories.append(LossCategory.FALSE_BREAKOUT.value)

        # Early reversal: Lost before any SL stage advancement
        if exit_reason in (ExitReason.SL_STAGE1.value, "sl_stage1"):
            categories.append(LossCategory.EARLY_REVERSAL.value)

        # Stage 2 pullback
        if exit_reason in (ExitReason.SL_STAGE2.value, "sl_stage2"):
            categories.append(LossCategory.STAGE2_PULLBACK.value)

        # Breakeven loss
        if exit_reason in (ExitReason.SL_STAGE3.value, "sl_stage3"):
            categories.append(LossCategory.BREAKEVEN_LOSS.value)

        # Give-back
        if exit_reason in (ExitReason.GIVEBACK_EXIT.value, "giveback_exit"):
            if pnl_pips < 0:
                categories.append(LossCategory.GIVEBACK_LOSS.value)

        # Session quality
        session = trade.get("session_type", "")
        if session in ("pre_asian", "end_of_day", "off_session"):
            categories.append(LossCategory.SESSION_QUALITY.value)

        # High spread
        spread = context.get("spread_pips", 0)
        if spread and spread > 4.0:
            categories.append(LossCategory.HIGH_SPREAD.value)

        # Tight SL
        if risk_pips > 0 and risk_pips < 8:
            categories.append(LossCategory.TIGHT_SL.value)

        # Wide range
        c1_range = context.get("c1_range_pips", 0)
        if c1_range and c1_range > 50:
            categories.append(LossCategory.WIDE_RANGE.value)

        # Overcrowded S/R
        room = context.get("room_to_opposing_pips", 999)
        if room and room < 15:
            categories.append(LossCategory.OVERCROWDED_SR.value)

        # Default if no category assigned
        if not categories:
            categories.append(LossCategory.FALSE_BREAKOUT.value)

        # Store in database
        for cat in categories:
            self.db.assign_loss_category(trade["trade_id"], cat)

        logger.info("Loss categorized (%s): %s",
                     trade["trade_id"], ", ".join(categories))

        return categories

    # === PATTERN KEY GENERATION ===

    def generate_pattern_key(self, trade: Dict,
                             context: Optional[Dict] = None) -> str:
        """
        Generate a pattern key from trade context for matching.
        Format: {entry_type}_{session}_{trend}_{sr_touches}_{body_strength}
        """
        if not context:
            context = self.db.get_context(trade["trade_id"]) or {}

        parts = []

        # Entry type
        parts.append(trade.get("entry_type") or "unknown")

        # Session
        session = trade.get("session_type") or "unknown"
        parts.append(session)

        # Trend
        trend = context.get("trend_type") or "unknown"
        parts.append(trend)

        # S/R touch count (bucketed)
        touches = context.get("sr_touches", 0) or 0
        if touches <= 1:
            parts.append("weak_sr")
        elif touches <= 3:
            parts.append("moderate_sr")
        else:
            parts.append("strong_sr")

        # Body strength
        c1_br = context.get("c1_body_ratio", 0)
        imp_br = context.get("impulse_body_ratio", 0)
        br = c1_br or imp_br or 0
        if br < 0.35:
            parts.append("weak_body")
        elif br < 0.60:
            parts.append("normal_body")
        else:
            parts.append("strong_body")

        return "_".join(parts)

    # === TRADE ANALYSIS ===

    def on_trade_closed(self, trade: Dict, context: Optional[Dict] = None):
        """Called after every trade closes. Handles categorization and learning."""
        is_win = trade.get("pnl_dollars", 0) > 0

        # Categorize losses
        if not is_win:
            self.categorize_loss(trade, context)

        # Generate and update pattern
        pattern_key = self.generate_pattern_key(trade, context)
        self.db.upsert_pattern(
            pattern_key=pattern_key,
            is_win=is_win,
            description=f"{trade.get('entry_type', '')} at {trade.get('session_type', '')}"
        )

        self._trade_counter += 1

        # Periodic full analysis
        if self._trade_counter % self.cfg.analysis_interval == 0:
            self._run_pattern_analysis()

        logger.info("Trade logged for learning: %s (pattern: %s, %s)",
                     trade["trade_id"], pattern_key,
                     "WIN" if is_win else "LOSS")

    def _run_pattern_analysis(self):
        """
        Analyze all patterns and block those with high loss rates.
        Called every N trades (analysis_interval).
        """
        logger.info("Running pattern analysis (every %d trades)...",
                     self.cfg.analysis_interval)

        patterns = self.db.get_all_patterns()
        blocked_count = 0
        unblocked_count = 0

        for p in patterns:
            total = p["total_trades"]
            loss_rate = p["loss_rate"]

            # Block if loss rate > threshold AND sufficient trades
            if total >= self.cfg.min_trades_for_blocking:
                if loss_rate >= self.cfg.loss_ratio_threshold and not p["is_blocked"]:
                    self.db.block_pattern(p["pattern_key"])
                    blocked_count += 1
                    logger.warning("BLOCKED pattern '%s': %.0f%% loss rate (%d trades)",
                                    p["pattern_key"], loss_rate * 100, total)

                # Unblock if performance improved
                elif loss_rate < self.cfg.loss_ratio_threshold * 0.8 and p["is_blocked"]:
                    self.db.unblock_pattern(p["pattern_key"])
                    unblocked_count += 1
                    logger.info("UNBLOCKED pattern '%s': loss rate improved to %.0f%%",
                                 p["pattern_key"], loss_rate * 100)

        logger.info("Pattern analysis complete: %d blocked, %d unblocked",
                     blocked_count, unblocked_count)

    # === CONFIDENCE SCORING ===

    def score_confidence(self, trade_context: Dict) -> Tuple[float, List[str]]:
        """
        Score signal confidence based on blocked pattern matches.

        - Start at base confidence (1.0)
        - Subtract penalty per blocked pattern match
        - Reject if below min_confidence (0.50)
        - Immediately reject if any pattern >80% loss rate

        Returns: (confidence_score, list_of_matched_blocked_patterns)
        """
        blocked = self.db.get_blocked_patterns()
        if not blocked:
            return self.cfg.confidence_base, []

        # Build a pseudo pattern key from the trade context
        parts = []
        parts.append(trade_context.get("entry_type", "unknown"))
        parts.append(trade_context.get("session_type", "unknown"))
        parts.append(trade_context.get("trend_type", "unknown"))

        touches = trade_context.get("sr_touches", 0) or 0
        if touches <= 1:
            parts.append("weak_sr")
        elif touches <= 3:
            parts.append("moderate_sr")
        else:
            parts.append("strong_sr")

        br = trade_context.get("c1_body_ratio", 0) or \
             trade_context.get("impulse_body_ratio", 0) or 0
        if br < 0.35:
            parts.append("weak_body")
        elif br < 0.60:
            parts.append("normal_body")
        else:
            parts.append("strong_body")

        test_key = "_".join(parts)

        # Check matches
        matched = []
        confidence = self.cfg.confidence_base

        for bp in blocked:
            # Exact match only — partial matching was too aggressive
            if bp["pattern_key"] == test_key:
                matched.append(bp["pattern_key"])
                confidence -= self.cfg.confidence_penalty

                # Immediate reject for very high loss rate
                if bp["loss_rate"] >= self.cfg.immediate_reject_rate:
                    logger.warning("Immediate reject: pattern '%s' has %.0f%% loss rate",
                                    bp["pattern_key"], bp["loss_rate"] * 100)
                    return 0.0, matched

        confidence = max(0.0, confidence)

        if matched:
            logger.info("Confidence score: %.2f (matched %d blocked patterns: %s)",
                         confidence, len(matched), ", ".join(matched))

        return confidence, matched

    def should_take_trade(self, trade_context: Dict) -> Tuple[bool, float, str]:
        """
        Final decision: should we take this trade?
        Returns (should_trade, confidence, reason)
        """
        confidence, matched = self.score_confidence(trade_context)

        if confidence <= 0:
            return False, 0.0, f"Immediate reject: matched {matched}"

        if confidence < self.cfg.min_confidence:
            return False, confidence, \
                f"Below min confidence ({confidence:.2f} < {self.cfg.min_confidence})"

        return True, confidence, "OK"

    # === REPORTING ===

    def get_learning_report(self) -> Dict:
        """Generate a summary of what the agent has learned."""
        patterns = self.db.get_all_patterns()
        blocked = [p for p in patterns if p["is_blocked"]]
        cat_stats = self.db.get_category_stats()
        entry_stats = self.db.get_entry_type_stats()
        session_stats = self.db.get_session_stats()

        return {
            "total_patterns_tracked": len(patterns),
            "blocked_patterns": len(blocked),
            "blocked_pattern_details": [
                {"key": p["pattern_key"], "loss_rate": p["loss_rate"],
                 "trades": p["total_trades"]}
                for p in blocked
            ],
            "loss_category_breakdown": [
                {"category": c["category"], "count": c["count"]}
                for c in cat_stats
            ],
            "entry_type_performance": [
                {"type": e["entry_type"], "total": e["total"],
                 "wins": e["wins"], "avg_rr": round(e["avg_rr"] or 0, 2),
                 "total_pnl": round(e["total_pnl"] or 0, 2)}
                for e in entry_stats
            ],
            "session_performance": [
                {"session": s["session_type"], "total": s["total"],
                 "wins": s["wins"], "total_pnl": round(s["total_pnl"] or 0, 2)}
                for s in session_stats
            ],
            "trades_analyzed": self._trade_counter,
        }
