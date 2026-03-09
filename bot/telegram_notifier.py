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
        """Trade executed."""
        text = (
            f"📊 <b>TRADE OPENED</b>\n"
            f"ID: <code>{trade_info.get('trade_id', '?')}</code>\n"
            f"Direction: {trade_info.get('direction', '?')}\n"
            f"Lots: {trade_info.get('lot_size', 0):.2f}\n"
            f"Entry: ${trade_info.get('entry_price', 0):.2f}\n"
            f"Risk: {trade_info.get('risk_percent', 0):.1f}%"
        )
        self.send_sync(text)

    def notify_trade_closed(self, trade_info: Dict):
        """Trade closed with P&L."""
        pnl = trade_info.get("pnl_dollars", 0)
        emoji = "✅" if pnl >= 0 else "❌"
        rr = trade_info.get("rr_achieved", 0)

        text = (
            f"{emoji} <b>TRADE CLOSED</b>\n"
            f"ID: <code>{trade_info.get('trade_id', '?')}</code>\n"
            f"Reason: {trade_info.get('exit_reason', '?')}\n"
            f"PnL: {trade_info.get('pnl_pips', 0):.0f}p (${pnl:+.2f})\n"
            f"RR: {rr:.2f}\n"
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
        risk = status.get("risk_status", {})
        text = (
            "📈 <b>STATUS UPDATE</b>\n"
            "━━━━━━━━━━━━━━━━━\n"
            f"Balance: ${status.get('balance', 0):,.2f}\n"
            f"PnL: ${status.get('total_pnl', 0):+,.2f} "
            f"({status.get('total_pnl_pct', 0):+.1f}%)\n"
            f"Win Rate: {status.get('win_rate', 0):.1f}%\n"
            f"Today: {status.get('trades_today', 0)} trades "
            f"(${status.get('daily_pnl', 0):+.2f})\n"
            f"Open: {status.get('open_trades', 0)}\n"
            f"Risk: {risk.get('current_risk_pct', 1.0):.2f}%"
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
