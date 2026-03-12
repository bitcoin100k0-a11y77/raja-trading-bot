"""
Raja Banks Trading Bot - Backtester
Replays historical M15 bars through the full strategy pipeline.

Usage:
    python backtester.py                          # Default: 30 days
    python backtester.py --days 90                # Custom lookback
    python backtester.py --days 60 --balance 50000
    python backtester.py --start 2025-01-01 --end 2025-03-01
    python backtester.py --verbose                # Print every signal + trade

Produces:
- Total P&L, win rate, max drawdown, avg RR
- Monthly/weekly breakdown
- Entry type performance
- Session performance
- Equity curve CSV
"""
import os
import sys
import argparse
import logging
import json
import csv
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Add bot directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BotConfig, StrategyConfig, RiskConfig, LearningConfig
from constants import (
    EntryType, TradeDirection, TrendType, SessionType,
    VolumeLevel, SLStage, ExitReason
)
from market_analyzer import MarketAnalyzer, MarketState, SRLevel
from strategy import Strategy, Signal
from trade_manager import TradeManager, ActiveTrade
from risk_manager import RiskManager
from database import TradingDatabase
from learning_agent import LearningAgent
from data_connector import DataConnector
from utils import (
    price_to_pips, pips_to_price, body_ratio, is_bullish, is_bearish,
    is_tradeable_session, get_session, get_4h_candle_number,
    set_simulated_time, utc_now
)


# ============================================================
# BACKTEST RESULT TRACKING
# ============================================================

@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    trade_id: str
    direction: str
    entry_type: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    lot_size: float
    risk_pips: float
    pnl_pips: float
    pnl_dollars: float
    rr_achieved: float
    exit_reason: str
    entry_time: str
    exit_time: str
    session: str
    trend: str
    bars_held: int = 0
    max_favorable_pips: float = 0.0
    max_adverse_pips: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    total_bars: int = 0
    start_date: str = ""
    end_date: str = ""

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl_dollars > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl_dollars <= 0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.wins / len(self.trades) * 100

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_dollars for t in self.trades)

    @property
    def total_pnl_pips(self) -> float:
        return sum(t.pnl_pips for t in self.trades)

    @property
    def avg_win(self) -> float:
        winning = [t.pnl_dollars for t in self.trades if t.pnl_dollars > 0]
        return np.mean(winning) if winning else 0.0

    @property
    def avg_loss(self) -> float:
        losing = [t.pnl_dollars for t in self.trades if t.pnl_dollars <= 0]
        return np.mean(losing) if losing else 0.0

    @property
    def avg_rr(self) -> float:
        rrs = [t.rr_achieved for t in self.trades if t.pnl_dollars > 0]
        return np.mean(rrs) if rrs else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_dollars for t in self.trades if t.pnl_dollars > 0)
        gross_loss = abs(sum(t.pnl_dollars for t in self.trades if t.pnl_dollars <= 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> Tuple[float, float]:
        """Returns (max_drawdown_dollars, max_drawdown_pct)."""
        if not self.equity_curve:
            return 0.0, 0.0
        peak = self.initial_balance
        max_dd = 0.0
        max_dd_pct = 0.0
        for point in self.equity_curve:
            bal = point["balance"]
            if bal > peak:
                peak = bal
            dd = peak - bal
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        return max_dd, max_dd_pct

    @property
    def max_consecutive_losses(self) -> int:
        max_streak = 0
        current = 0
        for t in self.trades:
            if t.pnl_dollars <= 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @property
    def max_consecutive_wins(self) -> int:
        max_streak = 0
        current = 0
        for t in self.trades:
            if t.pnl_dollars > 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio from trade returns."""
        if len(self.trades) < 2:
            return 0.0
        returns = [t.pnl_dollars for t in self.trades]
        avg = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return 0.0
        # Approximate annualization: assume ~250 trading days, ~2 trades/day
        trades_per_year = 250 * 2
        return (avg / std) * np.sqrt(trades_per_year)


# ============================================================
# BACKTESTER ENGINE
# ============================================================

class Backtester:
    """
    Bar-by-bar backtest engine.

    Replays historical data through the full strategy pipeline:
    1. For each new M15 bar, run market analysis
    2. Generate signals via Strategy
    3. Execute signals (paper)
    4. Manage active trades using OHLC (not just close)
    5. Track all results
    """

    def __init__(self, config: BotConfig = None, verbose: bool = False):
        self.cfg = config or BotConfig.from_env()
        self.verbose = verbose

        # Use in-memory DB for backtesting
        self.db = TradingDatabase(db_path=":memory:")
        self.analyzer = MarketAnalyzer(self.cfg.strategy)
        self.strategy = Strategy(self.cfg.strategy, self.analyzer)
        self.risk_mgr = RiskManager(self.cfg.risk, self.cfg.strategy, self.db)
        self.trade_mgr = TradeManager(self.cfg.strategy)
        self.learner = LearningAgent(self.cfg.learning, self.db)

        self.balance = self.cfg.initial_balance
        self.result = BacktestResult(initial_balance=self.balance)
        self._partial_pnl: Dict[str, float] = {}
        self._bar_count = 0
        self._trade_entry_bars: Dict[str, int] = {}

    def run(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Run backtest on provided DataFrame.

        Args:
            df: M15 OHLCV DataFrame with DatetimeIndex (UTC)
            htf_df: Optional 1H DataFrame for HTF zone stacking
        """
        if df.empty or len(df) < 50:
            print("ERROR: Need at least 50 bars for backtesting")
            return self.result

        # Add helper columns if missing
        if "body" not in df.columns:
            df["body"] = abs(df["Close"] - df["Open"])
            df["range"] = df["High"] - df["Low"]
            df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
            df["body_ratio"] = df["body_ratio"].fillna(0)
            df["bullish"] = df["Close"] > df["Open"]

        self.result.start_date = str(df.index[50])
        self.result.end_date = str(df.index[-1])
        self.result.total_bars = len(df) - 50

        # Initial equity point
        self.result.equity_curve.append({
            "time": str(df.index[50]),
            "balance": self.balance,
            "bar": 0,
        })

        print(f"\nBacktest: {self.result.start_date} → {self.result.end_date}")
        print(f"Bars: {self.result.total_bars} | Balance: ${self.balance:,.2f}")
        print(f"Entry types: C1/C0={True}, Impulse={self.cfg.strategy.impulse_enabled}, "
              f"B&R={self.cfg.strategy.break_retest_enabled}")
        print(f"Session: {self.cfg.strategy.session_start}:00-{self.cfg.strategy.session_end}:00 UTC")
        blackout = self.cfg.strategy.session_blackout_start
        if blackout:
            print(f"Blackout: {blackout}:00-{self.cfg.strategy.session_blackout_end}:00 UTC")
        else:
            print("Blackout: None")
        print("-" * 60)

        # Bar-by-bar replay (start at bar 50 to have enough lookback)
        for i in range(50, len(df)):
            self._bar_count = i - 50

            # Slice data up to current bar (the strategy only sees past data)
            current_df = df.iloc[:i + 1].copy()
            bar = df.iloc[i]

            # Set simulated time for utils.utc_now()
            bar_time = df.index[i]
            if hasattr(bar_time, 'to_pydatetime'):
                set_simulated_time(bar_time.to_pydatetime())
            else:
                set_simulated_time(bar_time)

            o = float(bar["Open"])
            h = float(bar["High"])
            l = float(bar["Low"])
            c = float(bar["Close"])
            utc_hour = bar_time.hour if hasattr(bar_time, 'hour') else 12

            # 1. Manage active trades with OHLC (more accurate than close-only)
            exit_events = self.trade_mgr.update_trades_ohlc(o, h, l, c)

            # Also check opposite close (strong candles only)
            if self.trade_mgr.get_active_trade_count() > 0:
                br = body_ratio(o, c, h, l)
                opp_exits = self.trade_mgr.check_opposite_close(o, c, c, body_ratio_val=br)
                if opp_exits:
                    exit_events.extend(opp_exits)

            # Process exits
            for event in exit_events:
                self._process_exit(event, bar_time)

            # 2. Skip signal generation outside session
            if not is_tradeable_session(utc_hour, self.cfg.strategy.session_start,
                                        self.cfg.strategy.session_end):
                # Session end close
                session_exits = self.trade_mgr.session_end_close_all(c)
                for event in session_exits:
                    self._process_exit(event, bar_time)
                continue

            # 3. Market analysis
            market = self.analyzer.analyze(current_df, htf_df, c)

            # 4. Generate signals
            signals = self.strategy.generate_signals(current_df, market)

            # 5. Execute signals
            for sig in signals:
                # Risk check
                can_trade, reason = self.risk_mgr.can_trade(self.balance)
                if not can_trade:
                    if self.verbose:
                        print(f"  [{bar_time}] Risk rejected: {reason}")
                    continue

                # Learning agent check
                trade_ctx = {
                    "entry_type": sig.entry_type.value,
                    "session_type": market.session.value,
                    "trend_type": market.trend.value,
                    "sr_touches": sig.sr_level.touches,
                    "c1_body_ratio": sig.context.get("c1_body_ratio", 0),
                    "impulse_body_ratio": sig.context.get("impulse_body_ratio", 0),
                }
                should_trade, confidence, reason = self.learner.should_take_trade(trade_ctx)
                if not should_trade:
                    if self.verbose:
                        print(f"  [{bar_time}] Learning rejected: {reason}")
                    continue

                # Calculate lot size
                lot_size = self.risk_mgr.calculate_position_size(
                    self.balance, sig.risk_pips
                )

                # Create active trade
                active = ActiveTrade(
                    trade_id=sig.trade_id,
                    direction=sig.direction,
                    entry_price=sig.entry_price,
                    lot_size=lot_size,
                    sl_price=sig.sl_price,
                    sl_stage=SLStage.STAGE_1,
                    tp1_price=sig.tp1_price,
                    tp2_price=sig.tp2_price,
                    c0_extreme=sig.context.get("c1_low") if sig.direction == TradeDirection.BUY
                               else sig.context.get("c1_high"),
                )
                self.trade_mgr.open_trade(active)
                self._trade_entry_bars[sig.trade_id] = self._bar_count

                # Store in DB for risk manager
                trade_record = {
                    "trade_id": sig.trade_id,
                    "symbol": self.cfg.strategy.symbol,
                    "direction": sig.direction.value,
                    "entry_type": sig.entry_type.value,
                    "entry_price": sig.entry_price,
                    "lot_size": lot_size,
                    "sl_initial": sig.sl_price,
                    "sl_current": sig.sl_price,
                    "sl_stage": 1,
                    "tp1_price": sig.tp1_price,
                    "tp2_price": sig.tp2_price,
                    "risk_pips": sig.risk_pips,
                    "risk_percent": self.risk_mgr.current_risk_percent,
                    "balance_before": self.balance,
                    "entry_time": str(bar_time),
                    "session_type": market.session.value,
                    "four_h_candle": get_4h_candle_number(utc_hour),
                }
                self.db.insert_trade(trade_record)

                # Store context for learning
                context_record = {
                    "trade_id": sig.trade_id,
                    "sr_level_price": sig.sr_level.price,
                    "sr_level_type": sig.sr_level.zone_type.value,
                    "sr_touches": sig.sr_level.touches,
                    "confidence_score": confidence,
                }
                ctx = sig.context
                if "c1_body_ratio" in ctx:
                    context_record["c1_body_ratio"] = ctx["c1_body_ratio"]
                    context_record["c1_range_pips"] = ctx.get("c1_range_pips", 0)
                if "impulse_body_ratio" in ctx:
                    context_record["impulse_body_ratio"] = ctx["impulse_body_ratio"]
                if "break_level_price" in ctx:
                    context_record["break_level_price"] = ctx["break_level_price"]
                if "trend" in ctx:
                    context_record["trend_type"] = ctx["trend"]
                self.db.insert_context(context_record)

                if self.verbose:
                    print(f"  [{bar_time}] OPEN {sig.direction.value} "
                          f"{sig.entry_type.value} @ {sig.entry_price:.2f} "
                          f"SL={sig.sl_price:.2f} TP1={sig.tp1_price:.2f} "
                          f"risk={sig.risk_pips:.1f} pips, conf={confidence:.2f}")

            # Progress indicator
            if self._bar_count % 500 == 0 and self._bar_count > 0:
                pct = self._bar_count / self.result.total_bars * 100
                print(f"  Progress: {pct:.0f}% ({self._bar_count}/{self.result.total_bars} bars) "
                      f"| Trades: {len(self.result.trades)} | Balance: ${self.balance:,.2f}")

        # Close any remaining open trades at last price
        last_price = float(df["Close"].iloc[-1])
        remaining = self.trade_mgr.session_end_close_all(last_price)
        for event in remaining:
            self._process_exit(event, df.index[-1])

        # Reset simulated time
        set_simulated_time(None)

        self.result.final_balance = self.balance
        return self.result

    def _process_exit(self, event: Dict, exit_time):
        """Process a trade exit event."""
        trade_id = event["trade_id"]
        trade = self.db.get_trade(trade_id)
        if not trade:
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

        # Adjust sign
        if (direction == TradeDirection.BUY and exit_price < trade["entry_price"]) or \
           (direction == TradeDirection.SELL and exit_price > trade["entry_price"]):
            pnl_pips = -abs(pnl_pips)

        pnl_dollars = pnl_pips * self.cfg.strategy.pip_value_per_lot * lots_closed
        rr = pnl_pips / trade["risk_pips"] if trade["risk_pips"] > 0 else 0

        self.balance += pnl_dollars

        if full_close:
            total_pnl = pnl_dollars + self._partial_pnl.pop(trade_id, 0.0)

            # Close in DB
            reason_str = reason.value if isinstance(reason, ExitReason) else str(reason)
            self.db.close_trade(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=reason_str,
                pnl_pips=pnl_pips,
                pnl_dollars=total_pnl,
                rr_achieved=rr,
                balance_after=self.balance,
            )

            # Update risk manager
            is_win = total_pnl > 0
            self.risk_mgr.update_after_trade(is_win)

            # Update learning agent
            closed_trade = self.db.get_trade(trade_id)
            if closed_trade:
                context = self.db.get_context(trade_id)
                self.learner.on_trade_closed(closed_trade, context)

            # Calculate bars held
            entry_bar = self._trade_entry_bars.pop(trade_id, self._bar_count)
            bars_held = self._bar_count - entry_bar

            # Record result
            bt_trade = BacktestTrade(
                trade_id=trade_id,
                direction=trade["direction"],
                entry_type=trade["entry_type"],
                entry_price=trade["entry_price"],
                exit_price=exit_price,
                sl_price=trade["sl_initial"],
                tp1_price=trade.get("tp1_price", 0),
                tp2_price=trade.get("tp2_price", 0),
                lot_size=trade["lot_size"],
                risk_pips=trade["risk_pips"],
                pnl_pips=pnl_pips,
                pnl_dollars=total_pnl,
                rr_achieved=rr,
                exit_reason=reason_str,
                entry_time=trade["entry_time"],
                exit_time=str(exit_time),
                session=trade.get("session_type", ""),
                trend=trade.get("four_h_candle", ""),
                bars_held=bars_held,
            )
            self.result.trades.append(bt_trade)

            # Equity curve
            self.result.equity_curve.append({
                "time": str(exit_time),
                "balance": round(self.balance, 2),
                "bar": self._bar_count,
                "trade_id": trade_id,
                "pnl": round(total_pnl, 2),
            })

            if self.verbose:
                emoji = "✅" if is_win else "❌"
                print(f"  {emoji} CLOSE {trade_id} | {reason_str} | "
                      f"PnL: {pnl_pips:+.1f} pips (${total_pnl:+.2f}) | "
                      f"RR: {rr:.2f} | Bal: ${self.balance:,.2f} | "
                      f"Held: {bars_held} bars")
        else:
            # Partial close (TP1)
            self._partial_pnl[trade_id] = self._partial_pnl.get(trade_id, 0.0) + pnl_dollars
            if self.verbose:
                print(f"  TP1 partial: {trade_id} | +{pnl_pips:.1f} pips (${pnl_dollars:+.2f})")


# ============================================================
# REPORTING
# ============================================================

def print_report(result: BacktestResult):
    """Print detailed backtest report."""
    print("\n" + "=" * 70)
    print("  BACKTEST REPORT — Raja Banks Market Fluidity Strategy")
    print("=" * 70)

    print(f"\n  Period: {result.start_date} → {result.end_date}")
    print(f"  Bars:   {result.total_bars} M15 bars")

    # Overall performance
    print(f"\n--- PERFORMANCE ---")
    print(f"  Initial Balance:     ${result.initial_balance:>12,.2f}")
    print(f"  Final Balance:       ${result.final_balance:>12,.2f}")
    print(f"  Total P&L:           ${result.total_pnl:>+12,.2f} "
          f"({(result.final_balance / result.initial_balance - 1) * 100:+.1f}%)")
    print(f"  Total P&L (pips):    {result.total_pnl_pips:>+12.1f}")

    print(f"\n--- TRADES ---")
    print(f"  Total Trades:        {result.total_trades:>6}")
    print(f"  Wins:                {result.wins:>6} ({result.win_rate:.1f}%)")
    print(f"  Losses:              {result.losses:>6}")
    print(f"  Avg Win:             ${result.avg_win:>+12,.2f}")
    print(f"  Avg Loss:            ${result.avg_loss:>+12,.2f}")
    print(f"  Avg Win RR:          {result.avg_rr:>8.2f}")
    print(f"  Profit Factor:       {result.profit_factor:>8.2f}")

    dd_dollars, dd_pct = result.max_drawdown
    print(f"\n--- RISK ---")
    print(f"  Max Drawdown:        ${dd_dollars:>12,.2f} ({dd_pct:.1f}%)")
    print(f"  Max Consec Losses:   {result.max_consecutive_losses:>6}")
    print(f"  Max Consec Wins:     {result.max_consecutive_wins:>6}")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio:>8.2f}")

    # By entry type
    if result.trades:
        print(f"\n--- BY ENTRY TYPE ---")
        entry_types = {}
        for t in result.trades:
            et = t.entry_type
            if et not in entry_types:
                entry_types[et] = {"wins": 0, "losses": 0, "pnl": 0.0, "pips": 0.0}
            if t.pnl_dollars > 0:
                entry_types[et]["wins"] += 1
            else:
                entry_types[et]["losses"] += 1
            entry_types[et]["pnl"] += t.pnl_dollars
            entry_types[et]["pips"] += t.pnl_pips

        for et, stats in sorted(entry_types.items()):
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total * 100 if total > 0 else 0
            print(f"  {et:20s} | {total:3d} trades | WR {wr:5.1f}% | "
                  f"PnL ${stats['pnl']:+10,.2f} ({stats['pips']:+.1f} pips)")

    # By session
    if result.trades:
        print(f"\n--- BY SESSION ---")
        sessions = {}
        for t in result.trades:
            s = t.session or "unknown"
            if s not in sessions:
                sessions[s] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if t.pnl_dollars > 0:
                sessions[s]["wins"] += 1
            else:
                sessions[s]["losses"] += 1
            sessions[s]["pnl"] += t.pnl_dollars

        for s, stats in sorted(sessions.items()):
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total * 100 if total > 0 else 0
            print(f"  {s:20s} | {total:3d} trades | WR {wr:5.1f}% | "
                  f"PnL ${stats['pnl']:+10,.2f}")

    # By exit reason
    if result.trades:
        print(f"\n--- BY EXIT REASON ---")
        reasons = {}
        for t in result.trades:
            r = t.exit_reason
            if r not in reasons:
                reasons[r] = {"count": 0, "pnl": 0.0}
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t.pnl_dollars

        for r, stats in sorted(reasons.items(), key=lambda x: x[1]["count"], reverse=True):
            print(f"  {r:20s} | {stats['count']:3d} trades | "
                  f"PnL ${stats['pnl']:+10,.2f}")

    # Monthly breakdown
    if result.trades:
        print(f"\n--- MONTHLY BREAKDOWN ---")
        months = {}
        for t in result.trades:
            month = t.entry_time[:7]  # "YYYY-MM"
            if month not in months:
                months[month] = {"trades": 0, "wins": 0, "pnl": 0.0}
            months[month]["trades"] += 1
            if t.pnl_dollars > 0:
                months[month]["wins"] += 1
            months[month]["pnl"] += t.pnl_dollars

        for m, stats in sorted(months.items()):
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
            print(f"  {m} | {stats['trades']:3d} trades | WR {wr:5.1f}% | "
                  f"PnL ${stats['pnl']:+10,.2f}")

    # Average bars held
    if result.trades:
        bars = [t.bars_held for t in result.trades]
        print(f"\n--- TIMING ---")
        print(f"  Avg bars held:       {np.mean(bars):>8.1f} ({np.mean(bars) * 15:.0f} min)")
        print(f"  Min bars held:       {min(bars):>8}")
        print(f"  Max bars held:       {max(bars):>8}")
        print(f"  Trades per day:      {len(result.trades) / max(1, result.total_bars / 88):>8.1f}")

    print("\n" + "=" * 70)


def save_equity_csv(result: BacktestResult, filepath: str):
    """Save equity curve to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "balance", "bar", "trade_id", "pnl"])
        writer.writeheader()
        for point in result.equity_curve:
            writer.writerow(point)
    print(f"Equity curve saved: {filepath}")


def save_trades_csv(result: BacktestResult, filepath: str):
    """Save all trades to CSV."""
    if not result.trades:
        return
    fields = [
        "trade_id", "direction", "entry_type", "entry_price", "exit_price",
        "sl_price", "tp1_price", "tp2_price", "lot_size", "risk_pips",
        "pnl_pips", "pnl_dollars", "rr_achieved", "exit_reason",
        "entry_time", "exit_time", "session", "bars_held"
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in result.trades:
            writer.writerow({k: getattr(t, k) for k in fields})
    print(f"Trades saved: {filepath}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Raja Banks Strategy Backtester")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to backtest (default: 30)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Starting balance (default: 10000)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print every signal and trade")
    parser.add_argument("--output", "-o", type=str, default="data",
                        help="Output directory for CSV files")
    parser.add_argument("--risk", type=float, default=None,
                        help="Risk percent per trade (default: from config)")
    parser.add_argument("--no-impulse", action="store_true",
                        help="Disable impulse entry")
    parser.add_argument("--no-breakretest", action="store_true",
                        help="Disable break & retest entry")
    parser.add_argument("--log-level", type=str, default="WARNING",
                        help="Log level (default: WARNING)")
    args = parser.parse_args()

    # Setup logging (quiet by default for backtest)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    # Build config
    config = BotConfig.from_env()
    config.initial_balance = args.balance

    if args.risk is not None:
        config.risk.risk_percent = args.risk

    if args.no_impulse:
        config.strategy.impulse_enabled = False
    if args.no_breakretest:
        config.strategy.break_retest_enabled = False

    # Fetch data
    print("\nFetching historical data...")
    data = DataConnector(
        symbol=config.strategy.symbol,
        timeframe=config.strategy.timeframe,
    )

    if args.start and args.end:
        # Custom date range — fetch enough bars
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 10  # buffer
    else:
        days = args.days + 5  # buffer for lookback

    # Fetch M15 bars
    df = data.fetch_bars(lookback_days=days, min_bars=days * 88, use_cache=False)
    if df.empty:
        print("ERROR: Could not fetch historical data. Check your API keys.")
        sys.exit(1)

    # Trim to date range if specified
    if args.start:
        start_dt = pd.Timestamp(args.start, tz="UTC")
        df = df[df.index >= start_dt]
    if args.end:
        end_dt = pd.Timestamp(args.end, tz="UTC") + timedelta(days=1)
        df = df[df.index < end_dt]

    print(f"Loaded {len(df)} M15 bars from {data.active_source}")

    # Fetch HTF (1H) for zone stacking
    print("Fetching 1H data for HTF zone stacking...")
    htf_df = data.fetch_htf_bars(timeframe="1h", lookback_days=max(days, 30))
    if not htf_df.empty:
        print(f"Loaded {len(htf_df)} 1H bars")
    else:
        print("WARNING: No HTF data — zone stacking disabled")

    # Run backtest
    bt = Backtester(config, verbose=args.verbose)
    result = bt.run(df, htf_df)

    # Print report
    print_report(result)

    # Save outputs
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_equity_csv(result, os.path.join(args.output, f"equity_{timestamp}.csv"))
    save_trades_csv(result, os.path.join(args.output, f"trades_{timestamp}.csv"))

    # Save summary JSON
    dd_dollars, dd_pct = result.max_drawdown
    summary = {
        "period": f"{result.start_date} to {result.end_date}",
        "total_bars": result.total_bars,
        "initial_balance": result.initial_balance,
        "final_balance": result.final_balance,
        "total_pnl": round(result.total_pnl, 2),
        "total_pnl_pct": round((result.final_balance / result.initial_balance - 1) * 100, 2),
        "total_trades": result.total_trades,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": round(result.win_rate, 1),
        "profit_factor": round(result.profit_factor, 2),
        "max_drawdown_dollars": round(dd_dollars, 2),
        "max_drawdown_pct": round(dd_pct, 1),
        "max_consecutive_losses": result.max_consecutive_losses,
        "sharpe_ratio": round(result.sharpe_ratio, 2),
        "avg_win_rr": round(result.avg_rr, 2),
        "config": {
            "risk_percent": config.risk.risk_percent,
            "min_touches": config.strategy.min_touches,
            "impulse_enabled": config.strategy.impulse_enabled,
            "break_retest_enabled": config.strategy.break_retest_enabled,
            "session_start": config.strategy.session_start,
            "session_end": config.strategy.session_end,
            "blackout_start": config.strategy.session_blackout_start,
            "blackout_end": config.strategy.session_blackout_end,
            "min_sl_pips": config.strategy.min_sl_pips,
            "max_risk_pips": config.strategy.max_risk_pips,
            "min_room_pips": config.strategy.min_room_pips,
            "trailing_enabled": config.strategy.trailing_enabled,
        }
    }
    summary_path = os.path.join(args.output, f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
