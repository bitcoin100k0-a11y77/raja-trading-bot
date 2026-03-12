#!/usr/bin/env python3
"""
Run backtest on local XAUUSDM15.csv data (5 years of gold M15 bars).
Uses the new LonesomeTheBlue S/R channel detection.

Usage:
    python run_backtest_local.py                  # Full 5 year run
    python run_backtest_local.py --year 2024      # Just 2024
    python run_backtest_local.py --year 2025 -v   # 2025 verbose
"""
import os
import sys
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BotConfig
from backtester import Backtester, print_report, save_equity_csv, save_trades_csv


def load_local_data(csv_path: str, year: int = None) -> pd.DataFrame:
    """Load XAUUSDM15.csv and convert to standard DataFrame format."""
    df = pd.read_csv(csv_path, encoding='utf-16', header=None,
                     names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread'])

    # Parse datetime: "2021.12.08 16:15"
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df.index = df.index.tz_localize('UTC')

    # Drop Spread column
    df = df.drop(columns=['Spread'], errors='ignore')

    # Filter by year if specified
    if year:
        df = df[df.index.year == year]

    # Add helper columns
    df['body'] = abs(df['Close'] - df['Open'])
    df['range'] = df['High'] - df['Low']
    df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
    df['body_ratio'] = df['body_ratio'].fillna(0)
    df['bullish'] = df['Close'] > df['Open']

    return df


def main():
    parser = argparse.ArgumentParser(description="Backtest on local gold M15 data")
    parser.add_argument("--year", type=int, default=None,
                        help="Filter to specific year (2021-2026)")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Starting balance (default: 10000)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print every signal and trade")
    parser.add_argument("--risk", type=float, default=None,
                        help="Risk percent per trade")
    parser.add_argument("--no-impulse", action="store_true")
    parser.add_argument("--no-breakretest", action="store_true")
    parser.add_argument("--log-level", type=str, default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    # Find CSV
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "XAUUSDM15.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "XAUUSDM15.csv")
    if not os.path.exists(csv_path):
        print("ERROR: Cannot find XAUUSDM15.csv")
        sys.exit(1)

    print("Loading data from", csv_path)
    df = load_local_data(csv_path, year=args.year)
    print("Loaded %d M15 bars (%s to %s)" % (len(df), df.index[0], df.index[-1]))

    # Config
    config = BotConfig.from_env()
    config.initial_balance = args.balance
    if args.risk is not None:
        config.risk.risk_percent = args.risk
    if args.no_impulse:
        config.strategy.impulse_enabled = False
    if args.no_breakretest:
        config.strategy.break_retest_enabled = False

    # Run
    bt = Backtester(config, verbose=args.verbose)
    result = bt.run(df)

    # Report
    print_report(result)

    # Save outputs
    os.makedirs("data", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_equity_csv(result, "data/equity_%s.csv" % ts)
    save_trades_csv(result, "data/trades_%s.csv" % ts)

    import json
    dd_d, dd_p = result.max_drawdown
    summary = {
        "period": "%s to %s" % (result.start_date, result.end_date),
        "total_bars": result.total_bars,
        "initial_balance": result.initial_balance,
        "final_balance": round(result.final_balance, 2),
        "total_pnl": round(result.total_pnl, 2),
        "total_trades": result.total_trades,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": round(result.win_rate, 1),
        "profit_factor": round(result.profit_factor, 2),
        "max_drawdown_pct": round(dd_p, 1),
        "sharpe_ratio": round(result.sharpe_ratio, 2),
    }
    with open("data/summary_%s.json" % ts, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved: data/summary_%s.json" % ts)


if __name__ == "__main__":
    main()
