"""
Raja Banks Trading Bot - Telegram Notifier
Production-ready async Telegram notifications with retry logic.
Sends: signals, trade opens/closes, SL updates, status, learning reports, daily summaries.
"""
import logging
import asyncio
import traceback
from datetime import datetime, timezone
from typing import Dict, Optional

from config import TelegramConfig

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    HAS_TELEGRAM = True
except ImportError:
    try:
        from telegram import Bot
        HAS_TELEGRAM = True
        ParseMode = None
    except ImportError:
        HAS_TELEGRAM = False
        logger.warning("python-telegram-bot not installed")


class TelegramNotifier:
    """Production Telegram notifier with retry and message queue."""

    def __init__(self, config: TelegramConfig):
        self.cfg = config
        self.bot = None
        self._enabled = config.enabled and HAS_TELEGRAM and config.bot_token
        self._loop = None

        if self._enabled:
            try:
                self.bot = Bot(token=config.bot_token)
                logger.info("Telegram notifier initialized (chat: %s)", config.chat_id)
            except Exception as e:
                logger.error("Failed to init Telegram bot: %s", e)
                self._enabled = False
        else:
            if config.enabled and not HAS_TELEGRAM:
                logger.warning("Telegram enabled but python-telegram-bot not installed")
            elif config.enabled and not config.bot_token:
                logger.warning("Telegram enabled but no bot token configured")

    def _get_loop(self):
        """Get or create event loop for async operations."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def _send(self, text: str, retries: int = 2):
        """Send message with retry logic."""
        if not self._enabled or not self.bot:
            return

        parse_mode = "HTML"

        for attempt in range(retries + 1):
            try:
                await self.bot.send_message(
                    chat_id=self.cfg.chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
                return
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(1)
                else:
                    logger.error("Telegram send failed after %d attempts: %s",
                                 retries + 1, e)

    def send_sync(self, text: str):
        """Synchronous send wrapper."""
        if not self._enabled:
            return
        try:
            loop = self._get_loop()
            if loop.is_running():
                asyncio.ensure_future(self._send(text))
            else:
                loop.run_until_complete(self._send(text))
        except Exception as e:
            logger.error("Telegram sync send error: %s", e)

    # === NOTIFICATION TYPES ===

    def notify_startup(self, config_summary: Dict):
        """Bot startup notification."""
        text = (
            "🚀 <b>RAJA BOT STARTED</b>\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"Symbol: {config_summary.get('symbol', 'XAUUSD')}\n"
            f"Mode: {'PAPER' if config_summary.get('dry_run', True) else 'LIVE'}\n"
            f"Balance: ${config_summary.get('balance', 0):,.2f}\n"
            f"Risk: {config_summary.get('risk_pct', 1.0)}%\n"
            f"Max Trades/Day: {config_summary.get('max_trades', 3)}\n"
            f"Daily DD Limit: {config_summary.get('dd_limit', 3.0)}%\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"Learning Agent: Active\n"
            f"Patterns Tracked: {config_summary.get('patterns', 0)}\n"
            f"Patterns Blocked: {config_summary.get('blocked', 0)}"
        )
        self.send_sync(text)

    def notify_signal(self, signal_info: Dict):
        """New trading signal detected."""
        direction = signal_info.get("direction", "?")
        emoji = "🟢" if direction == "BUY" else "🔴"
        entry = signal_info.get("entry_price", 0)
        sl = signal_info.get("sl_price", 0)
        tp1 = signal_info.get("tp1_price", 0)
        tp2 = signal_info.get("tp2_price", 0)
        risk = signal_info.get("risk_pips", 0)
        conf = signal_info.get("confidence", 1.0)

        text = (
            f"{emoji} <b>SIGNAL: {direction}</b>\n"
            f"Type: {signal_info.get('entry_type', '?')}\n"
            f"Entry: ${entry:.2f}\n"
            f"SL: ${sl:.2f} ({risk:.0f}p)\n"
            f"TP1: ${tp1:.2f} | TP2: ${tp2:.2f}\n"
            f"Confidence: {conf:.0%}"
        )
        self.send_sync(text)

    def notify_trade_opened(self, trade_info: Dict):
        """Trade executed — immediate notification with full details."""
        direction = trade_info.get("direction", "?")
        emoji = "🟢" if direction == "BUY" else "🔴"
        entry = trade_info.get("entry_price", 0)
        sl = trade_info.get("sl_price", 0)
        tp1 = trade_info.get("tp1_price", 0)
        tp2 = trade_info.get("tp2_price", 0)
        risk_pips = trade_info.get("risk_pips", 0)

        text = (
            f"{emoji} <b>TRADE OPENED — {direction}</b>\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"ID: <code>{trade_info.get('trade_id', '?')}</code>\n"
            f"Type: {trade_info.get('entry_type', 'C1/C0')}\n"
            f"Lots: {trade_info.get('lot_size', 0):.2f}\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"Entry: ${entry:.2f}\n"
            f"SL: ${sl:.2f} ({risk_pips:.0f} pips)\n"
            f"TP1: ${tp1:.2f}\n"
            f"TP2: ${tp2:.2f}\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"Risk: {trade_info.get('risk_percent', 0):.1f}% (${trade_info.get('risk_dollars', 0):.2f})\n"
            f"S/R Level: ${trade_info.get('sr_level', 0):.2f} ({trade_info.get('sr_touches', 0)} touches)\n"
            f"Balance: ${trade_info.get('balance', 0):,.2f}"
        )
        self.send_sync(text)

    def notify_trade_closed(self, trade_info: Dict):
        """Trade closed with full P&L summary."""
        pnl = trade_info.get("pnl_dollars", 0)
        emoji = "✅" if pnl >= 0 else "❌"
        rr = trade_info.get("rr_achieved", 0)
        exit_reason = trade_info.get("exit_reason", "?")

        # Determine exit reason label
        reason_labels = {
            "SL_STAGE1": "SL Hit (Stage 1)",
            "SL_STAGE2": "SL Hit (C0 Level)",
            "SL_STAGE3": "SL Hit (Breakeven)",
            "TP1_HIT": "TP1 Hit (Partial)",
            "TP2_HIT": "TP2 Hit (Full Target)",
            "GIVEBACK_EXIT": "Give-back Exit",
            "EMERGENCY_EXIT": "Emergency Exit",
            "SESSION_END": "Session End Close",
        }
        reason_text = reason_labels.get(exit_reason, exit_reason)

        text = (
            f"{emoji} <b>TRADE CLOSED</b>\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"ID: <code>{trade_info.get('trade_id', '?')}</code>\n"
            f"Direction: {trade_info.get('direction', '?')}\n"
            f"Entry: ${trade_info.get('entry_price', 0):.2f}\n"
            f"Exit: ${trade_info.get('exit_price', 0):.2f}\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"Reason: {reason_text}\n"
            f"PnL: {trade_info.get('pnl_pips', 0):.0f} pips (${pnl:+,.2f})\n"
            f"R:R Achieved: {rr:.2f}\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"Balance: ${trade_info.get('balance_after', 0):,.2f}"
        )
        self.send_sync(text)

    def notify_sl_update(self, trade_id: str, old_sl: float,
                         new_sl: float, stage: int):
        """Stop loss stage change."""
        stage_names = {1: "Initial", 2: "C0 Extreme", 3: "Breakeven"}
        text = (
            f"🔄 <b>SL → Stage {stage}</b> ({stage_names.get(stage, '?')})\n"
            f"Trade: <code>{trade_id}</code>\n"
            f"SL: ${old_sl:.2f} → ${new_sl:.2f}"
        )
        self.send_sync(text)

    def notify_trade_update(self, info: Dict):
        """Partial close — TP1 hit, 50% closed, remaining running to TP2."""
        text = (
            f"🎯 <b>TP1 HIT — 50% CLOSED</b>\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"ID: <code>{info.get('trade_id', '?')}</code>\n"
            f"TP1 Price: ${info.get('exit_price', 0):.2f}\n"
            f"PnL (closed): +{info.get('pnl_pips', 0):.0f} pips\n"
            f"Lots closed: {info.get('lots_closed', 0):.2f}\n"
            "━━━━━━━━━━━━━━━━━\n"
            "⏳ Remaining 50% running to TP2...\n"
            "SL advancing to protect profits"
        )
        self.send_sync(text)

    def notify_status(self, status: Dict):
        """Periodic status update or preload market snapshot."""

        # Preload market data notification
        if status.get("preload"):
            text = (
                "📊 <b>MARKET DATA PRELOADED</b>\n"
                "━━━━━━━━━━━━━━━━━\n"
                f"Bars: {status.get('bars_loaded', 0)} x M15\n"
                f"Price: ${status.get('price', 0):,.2f}\n"
                f"Trend: {status.get('trend', '?')}\n"
                f"Session: {status.get('session', '?')}\n"
                f"Volume: {status.get('volume', '?')}\n"
                f"Volatility: {status.get('volatility_pips', 0):.1f} pips\n"
                "━━━━━━━━━━━━━━━━━\n"
                f"<b>Support Zones ({status.get('support_count', 0)}):</b>\n"
                f"{status.get('support_text', 'None')}\n\n"
                f"<b>Resistance Zones ({status.get('resistance_count', 0)}):</b>\n"
                f"{status.get('resistance_text', 'None')}\n"
                "━━━━━━━━━━━━━━━━━\n"
                "✅ Ready for trades"
            )
            self.send_sync(text)
            return

        # Normal periodic status
        now = datetime.now(timezone.utc)
        risk = status.get("risk_status", {})

        text = (
            "📈 <b>STATUS UPDATE</b>\n"
            f"🕐 {now.strftime('%Y-%m-%d %H:%M')} UTC\n"
            "━━━━━━━━━━━━━━━━━\n"
        )

        # Market context if available
        price = status.get("current_price")
        if price:
            text += f"💰 Price: ${price:,.2f}\n"
        trend = status.get("trend")
        if trend:
            text += f"📊 Trend: {trend}\n"
        session = status.get("session")
        if session:
            text += f"🕐 Session: {session}\n"
        volume = status.get("volume")
        if volume:
            text += f"📶 Volume: {volume}\n"

        # S/R zones
        sup = status.get("nearest_support")
        res = status.get("nearest_resistance")
        if sup or res:
            text += "━━━━━━━━━━━━━━━━━\n"
            sc = status.get("support_count", 0)
            rc = status.get("resistance_count", 0)
            if sup:
                text += f"🟢 Nearest Support: ${sup:,.2f} ({sc} zones)\n"
            if res:
                text += f"🔴 Nearest Resistance: ${res:,.2f} ({rc} zones)\n"

        # Performance
        unrealized = status.get("unrealized_pnl", 0)
        realized = status.get("total_pnl", 0)
        total_equity = status.get("balance", 0) + unrealized

        text += (
            "━━━━━━━━━━━━━━━━━\n"
            f"💵 Balance: ${status.get('balance', 0):,.2f}\n"
            f"📊 Realized PnL: ${realized:+,.2f} "
            f"({status.get('total_pnl_pct', 0):+.1f}%)\n"
        )

        if unrealized != 0:
            text += f"📈 Unrealized PnL: ${unrealized:+,.2f}\n"
            text += f"💰 Total Equity: ${total_equity:,.2f}\n"

        text += (
            f"🎯 Win Rate: {status.get('win_rate', 0):.1f}%\n"
            f"📋 Today: {status.get('trades_today', 0)} trades "
            f"(${status.get('daily_pnl', 0):+.2f})\n"
            f"⚠️ Risk: {risk.get('current_risk_pct', 1.0):.2f}%"
        )

        # Open trade details with live P&L
        open_details = status.get("open_trade_details", [])
        if open_details:
            text += "\n━━━━━━━━━━━━━━━━━\n"
            text += f"<b>OPEN TRADES ({len(open_details)}):</b>\n"
            for t in open_details:
                emoji = "🟢" if t["direction"] == "BUY" else "🔴"
                pnl_emoji = "✅" if t["current_pnl_dollars"] >= 0 else "❌"
                tp1_tag = " [TP1 ✓]" if t["tp1_hit"] else ""
                stage_names = {1: "S1", 2: "S2-C0", 3: "BE"}
                sl_tag = stage_names.get(t["sl_stage"], "S1")
                text += (
                    f"\n{emoji} {t['trade_id'][:8]}..{tp1_tag}\n"
                    f"  Entry: ${t['entry_price']:.2f} | SL: ${t['sl_price']:.2f} ({sl_tag})\n"
                    f"  {pnl_emoji} PnL: {t['current_pnl_pips']:+.1f}p (${t['current_pnl_dollars']:+.2f})\n"
                )

        self.send_sync(text)

    def notify_learning_report(self, report: Dict):
        """Learning agent summary."""
        blocked = report.get("blocked_patterns", 0)
        total = report.get("total_patterns_tracked", 0)

        text = (
            "🧠 <b>LEARNING REPORT</b>\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"Patterns: {total} tracked, {blocked} blocked\n"
            f"Trades Analyzed: {report.get('trades_analyzed', 0)}"
        )

        # Top loss categories
        cats = report.get("loss_category_breakdown", [])[:5]
        if cats:
            text += "\n\n<b>Top Losses:</b>"
            for c in cats:
                text += f"\n  • {c['category']}: {c['count']}"

        # Blocked patterns
        blocked_details = report.get("blocked_pattern_details", [])[:5]
        if blocked_details:
            text += "\n\n<b>Blocked Patterns:</b>"
            for b in blocked_details:
                text += f"\n  🚫 {b['key']} ({b['loss_rate']:.0%}, {b['trades']} trades)"

        self.send_sync(text)

    def notify_daily_summary(self, summary: Dict):
        """End-of-day summary."""
        pnl = summary.get("pnl_dollars", 0)
        emoji = "📗" if pnl >= 0 else "📕"

        text = (
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"Date: {summary.get('date', 'N/A')}\n"
            f"Trades: {summary.get('trades', 0)}\n"
            f"Wins: {summary.get('wins', 0)} | "
            f"Losses: {summary.get('losses', 0)}\n"
            f"PnL: ${pnl:+,.2f}\n"
            f"Balance: ${summary.get('balance', 0):,.2f}\n"
        )

        # Learning updates
        new_blocked = summary.get("new_blocked", [])
        if new_blocked:
            text += "\n🧠 <b>New Blocked Patterns:</b>"
            for p in new_blocked:
                text += f"\n  🚫 {p}"

        self.send_sync(text)

    def notify_error(self, error_msg: str):
        """Error notification."""
        text = f"⚠️ <b>BOT ERROR</b>\n<code>{error_msg[:500]}</code>"
        self.send_sync(text)

    def notify_shutdown(self, reason: str = "Manual"):
        """Shutdown notification."""
        text = (
            "🔴 <b>BOT STOPPED</b>\n"
            f"Reason: {reason}"
        )
        self.send_sync(text)
