"""
Raja Banks Trading Agent - Risk Manager
Static/Dynamic risk (2/0.5 system), position sizing, daily limits, cooldown.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from config import RiskConfig, StrategyConfig
from database import TradingDatabase
from utils import calculate_lot_size, bars_since, utc_now

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages position sizing and risk controls.

    Raja's Dynamic Risk System (2/0.5):
    - Start at base risk (1%)
    - After 2 consecutive losses: reduce by 0.5%
    - After 3 consecutive losses: reduce by another 0.5%
    - On recovery wins: add back 0.25% per win
    - Never go below 0.25% or above base risk
    """

    def __init__(self, risk_config: RiskConfig, strategy_config: StrategyConfig,
                 db: TradingDatabase):
        self.cfg = risk_config
        self.strat_cfg = strategy_config
        self.db = db
        self._current_risk_pct = risk_config.risk_percent
        self._last_loss_time: Optional[datetime] = None
        self._cooldown_active = False
        # Start-of-day balance for daily loss % calculation.
        # Using current balance as denominator grows the % as losses accumulate,
        # causing the circuit breaker to trip earlier than the configured limit.
        self._daily_start_balance: Optional[float] = None
        self._daily_start_date: Optional[str] = None
        # 🔴 REAL-MONEY SAFETY — also key on the MT5 account login. If the terminal
        # switches accounts mid-session (demo → real, or wrong real account), the
        # daily-loss denominator must reset. Otherwise an account with $1,000 keeps
        # using yesterday's $11,000 demo balance as the 4% denominator, so the
        # circuit breaker won't fire until 46% of the real account is gone.
        self._daily_start_account: Optional[str] = None

    @property
    def current_risk_percent(self) -> float:
        return self._current_risk_pct

    def record_daily_start(self, balance: float,
                           account_id: Optional[str] = None) -> None:
        """
        Record today's starting balance (call once per tick before can_trade).
        Used as the denominator for daily loss % so the limit is stable all day.

        Resets when the DATE changes OR the MT5 ACCOUNT changes (terminal swap
        from demo to real, or wrong real account). Without the account check the
        loss-limit denominator silently keeps yesterday's other-account balance —
        catastrophic for a smaller real account (4% of $11k demo = 46% of $1k real).
        """
        today = utc_now().strftime("%Y-%m-%d")
        if self._daily_start_date != today or self._daily_start_account != account_id:
            if (self._daily_start_account is not None
                    and self._daily_start_account != account_id):
                logger.warning(
                    "🔴 MT5 account changed (%s → %s) — daily-start balance reset "
                    "from $%.2f to $%.2f (loss-limit denominator was stale).",
                    self._daily_start_account, account_id,
                    self._daily_start_balance or 0.0, balance,
                )
            self._daily_start_date = today
            self._daily_start_account = account_id
            self._daily_start_balance = balance
            logger.info("Daily start balance recorded: $%.2f (account=%s)",
                        balance, account_id)

    def can_trade(self, balance: float) -> tuple:
        """
        Check if trading is allowed right now.
        Returns (allowed: bool, reason: str)
        """
        # Daily trade limit
        trades_today = self.db.get_trades_today()
        if trades_today >= self.cfg.max_trades_per_day:
            return False, f"Daily trade limit reached ({trades_today}/{self.cfg.max_trades_per_day})"

        # Daily loss limit — use start-of-day balance as denominator so the
        # circuit-breaker threshold is stable across the session. Using current
        # balance (which shrinks with each loss) would make the % grow faster
        # than intended and trip the limit before the configured threshold.
        daily_pnl = self.db.get_daily_pnl()
        base_balance = self._daily_start_balance or balance
        daily_loss_pct = abs(daily_pnl) / base_balance * 100 if base_balance > 0 else 0
        if daily_pnl < 0 and daily_loss_pct >= self.cfg.daily_loss_limit:
            return False, f"Daily loss limit reached ({daily_loss_pct:.1f}% >= {self.cfg.daily_loss_limit}%)"

        # Post-loss cooldown
        if self._cooldown_active and self._last_loss_time:
            bars = bars_since(self._last_loss_time, 15)
            if bars < self.cfg.cooldown_bars:
                remaining = self.cfg.cooldown_bars - bars
                return False, f"Post-loss cooldown: {remaining} bars remaining"
            else:
                self._cooldown_active = False
                logger.info("Post-loss cooldown ended")

        # Minimum balance check
        if balance < 100:
            return False, "Balance too low"

        return True, "OK"

    def calculate_position_size(self, balance: float,
                                risk_pips: float) -> float:
        """
        Calculate lot size based on current risk level.
        lot_size = (balance * risk%) / (risk_pips * pip_value_per_lot)
        """
        return calculate_lot_size(
            balance=balance,
            risk_pct=self._current_risk_pct,
            risk_pips=risk_pips,
            pip_value=self.strat_cfg.pip_value_per_lot,
            min_lots=self.cfg.min_lots,
            max_lots=self.cfg.max_lots,
        )

    def update_after_trade(self, is_win: bool):
        """
        Update dynamic risk after a trade closes.
        Raja's 2/0.5 system.
        """
        consecutive_losses = self.db.get_consecutive_losses()

        if not is_win:
            self._last_loss_time = utc_now()
            self._cooldown_active = True
            logger.info("Loss recorded. Cooldown activated for %d bars",
                        self.cfg.cooldown_bars)

            # Dynamic risk reduction
            if consecutive_losses >= self.cfg.consecutive_loss_threshold:
                steps = consecutive_losses - self.cfg.consecutive_loss_threshold + 1
                reduction = steps * self.cfg.risk_reduction_increment
                new_risk = max(0.25, self.cfg.risk_percent - reduction)
                if new_risk != self._current_risk_pct:
                    logger.info("Dynamic risk: %.2f%% -> %.2f%% after %d consecutive losses",
                                self._current_risk_pct, new_risk, consecutive_losses)
                    self._current_risk_pct = new_risk
        else:
            # Recovery: add back risk on wins
            if self._current_risk_pct < self.cfg.risk_percent:
                new_risk = min(
                    self.cfg.risk_percent,
                    self._current_risk_pct + self.cfg.risk_recovery_increment
                )
                logger.info("Dynamic risk recovery: %.2f%% -> %.2f%%",
                            self._current_risk_pct, new_risk)
                self._current_risk_pct = new_risk

    def get_second_chance_risk(self) -> float:
        """Reduced risk for second chance / re-entry trades."""
        return min(self._current_risk_pct, self.cfg.second_chance_risk_pct)

    def get_risk_status(self, balance: float) -> dict:
        """Get current risk status for logging/display."""
        return {
            "current_risk_pct": self._current_risk_pct,
            "base_risk_pct": self.cfg.risk_percent,
            "consecutive_losses": self.db.get_consecutive_losses(),
            "trades_today": self.db.get_trades_today(),
            "daily_pnl": self.db.get_daily_pnl(),
            "cooldown_active": self._cooldown_active,
            "can_trade": self.can_trade(balance)[0],
        }
