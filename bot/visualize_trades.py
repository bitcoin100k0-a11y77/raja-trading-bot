#!/usr/bin/env python3
"""
Generate visual charts for individual trades from backtest results.
Shows price action with entry/exit markers, SL/TP lines, and context bars.

Usage:
    python visualize_trades.py                           # Latest trades CSV, 50 charts
    python visualize_trades.py --trades-csv data/trades_xxx.csv
    python visualize_trades.py --count 20 --spread even  # 20 evenly spread trades
"""
import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_m15_data(csv_path: str) -> pd.DataFrame:
    """Load XAUUSDM15.csv."""
    df = pd.read_csv(csv_path, encoding='utf-16', header=None,
                     names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df.index = df.index.tz_localize('UTC')
    df = df.drop(columns=['Spread'], errors='ignore')
    return df


def draw_candlestick(ax, df_slice, alpha=1.0):
    """Draw candlestick chart on given axes."""
    width = 0.006  # For M15 timeframe
    width2 = width * 0.3

    for i in range(len(df_slice)):
        dt = mdates.date2num(df_slice.index[i])
        o = df_slice['Open'].iloc[i]
        h = df_slice['High'].iloc[i]
        l = df_slice['Low'].iloc[i]
        c = df_slice['Close'].iloc[i]

        if c >= o:
            color = '#26a69a'  # Green / bullish
            body_bottom = o
            body_height = c - o
        else:
            color = '#ef5350'  # Red / bearish
            body_bottom = c
            body_height = o - c

        # Wick
        ax.plot([dt, dt], [l, h], color=color, linewidth=0.7, alpha=alpha)
        # Body
        ax.add_patch(Rectangle(
            (dt - width / 2, body_bottom), width, max(body_height, 0.01),
            facecolor=color, edgecolor=color, linewidth=0.5, alpha=alpha
        ))


def plot_trade(trade, m15_df, output_dir, trade_num):
    """Generate a single trade chart."""
    entry_time = pd.Timestamp(trade['entry_time'])
    exit_time = pd.Timestamp(trade['exit_time'])

    # Show context: 20 bars before entry, 10 bars after exit
    context_before = 25
    context_after = 15

    # Find entry bar index
    try:
        entry_idx = m15_df.index.get_indexer([entry_time], method='nearest')[0]
    except:
        return False

    try:
        exit_idx = m15_df.index.get_indexer([exit_time], method='nearest')[0]
    except:
        exit_idx = entry_idx + 5

    start_idx = max(0, entry_idx - context_before)
    end_idx = min(len(m15_df), exit_idx + context_after)

    df_slice = m15_df.iloc[start_idx:end_idx]
    if len(df_slice) < 5:
        return False

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Draw candlesticks
    draw_candlestick(ax, df_slice)

    # Trade info
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    sl_price = trade['sl_price']
    tp1_price = trade['tp1_price']
    tp2_price = trade['tp2_price']
    direction = trade['direction']
    is_win = trade['pnl_dollars'] > 0

    entry_dt = mdates.date2num(entry_time)
    exit_dt = mdates.date2num(exit_time)

    # Time range for horizontal lines
    line_start = mdates.date2num(df_slice.index[0])
    line_end = mdates.date2num(df_slice.index[-1])

    # Entry line
    ax.axhline(y=entry_price, color='#ffffff', linewidth=1.0, linestyle='--', alpha=0.5)

    # SL line (red dashed)
    ax.axhline(y=sl_price, color='#ff4444', linewidth=1.0, linestyle=':', alpha=0.6,
               label=f'SL: {sl_price:.2f}')

    # TP1 line (blue dashed)
    if tp1_price > 0:
        ax.axhline(y=tp1_price, color='#42a5f5', linewidth=1.0, linestyle=':', alpha=0.6,
                   label=f'TP1: {tp1_price:.2f}')

    # TP2 line (cyan dashed)
    if tp2_price > 0:
        ax.axhline(y=tp2_price, color='#00e5ff', linewidth=1.0, linestyle=':', alpha=0.6,
                   label=f'TP2: {tp2_price:.2f}')

    # Entry marker
    entry_color = '#00e676' if direction == 'BUY' else '#ff5252'
    entry_marker = '^' if direction == 'BUY' else 'v'
    ax.scatter([entry_dt], [entry_price], color=entry_color, marker=entry_marker,
              s=200, zorder=5, edgecolors='white', linewidths=1.5)

    # Exit marker
    exit_color = '#00e676' if is_win else '#ff5252'
    ax.scatter([exit_dt], [exit_price], color=exit_color, marker='x',
              s=200, zorder=5, linewidths=2.5)

    # Draw arrow from entry to exit
    arrow_color = '#00e676' if is_win else '#ff5252'
    ax.annotate('', xy=(exit_dt, exit_price), xytext=(entry_dt, entry_price),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, alpha=0.7))

    # Shade SL zone (red)
    if direction == 'BUY':
        ax.axhspan(sl_price, entry_price, alpha=0.05, color='red')
    else:
        ax.axhspan(entry_price, sl_price, alpha=0.05, color='red')

    # Shade TP zone (green) to TP2
    if tp2_price > 0:
        if direction == 'BUY':
            ax.axhspan(entry_price, tp2_price, alpha=0.05, color='green')
        else:
            ax.axhspan(tp2_price, entry_price, alpha=0.05, color='green')

    # Title and labels
    pnl = trade['pnl_dollars']
    pnl_pips = trade['pnl_pips']
    rr = trade['rr_achieved']
    result_emoji = 'WIN' if is_win else 'LOSS'
    result_color = '#00e676' if is_win else '#ff5252'

    title = (f"Trade #{trade_num} — {direction} {trade['entry_type']} | "
             f"{result_emoji} | PnL: ${pnl:+.2f} ({pnl_pips:+.1f} pips) | "
             f"RR: {rr:.2f} | Exit: {trade['exit_reason']}")
    ax.set_title(title, color=result_color, fontsize=11, fontweight='bold', pad=12)

    subtitle = (f"Entry: {entry_time.strftime('%Y-%m-%d %H:%M')} @ {entry_price:.2f} | "
                f"Exit: {exit_time.strftime('%Y-%m-%d %H:%M')} @ {exit_price:.2f} | "
                f"Risk: {trade['risk_pips']:.1f} pips | Bars held: {trade['bars_held']}")
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, fontsize=8,
            color='#aaaaaa', ha='center', va='bottom')

    # Format axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=30, colors='#cccccc', labelsize=8)
    ax.tick_params(axis='y', colors='#cccccc', labelsize=9)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # Grid
    ax.grid(True, alpha=0.15, color='#ffffff')

    # Legend
    ax.legend(loc='upper left', fontsize=8, facecolor='#16213e',
              edgecolor='#333333', labelcolor='#cccccc')

    # Spine colors
    for spine in ax.spines.values():
        spine.set_color('#333333')

    plt.tight_layout()

    # Save
    filename = f"trade_{trade_num:03d}_{direction}_{trade['entry_type']}_{result_emoji}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize backtest trades")
    parser.add_argument("--trades-csv", type=str, default=None,
                        help="Path to trades CSV (default: latest in data/)")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of trade charts to generate (default: 50)")
    parser.add_argument("--spread", type=str, default="even",
                        choices=["even", "recent", "random", "best", "worst"],
                        help="How to select trades: even=evenly spread, recent=latest, "
                             "random=random, best=top winners, worst=top losers")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for charts")
    args = parser.parse_args()

    # Find trades CSV
    if args.trades_csv:
        trades_csv = args.trades_csv
    else:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        csvs = sorted(glob.glob(os.path.join(data_dir, "trades_*.csv")))
        if not csvs:
            print("ERROR: No trades CSV found in data/")
            sys.exit(1)
        trades_csv = csvs[-1]  # Latest

    print(f"Loading trades from: {trades_csv}")
    trades_df = pd.read_csv(trades_csv)
    print(f"Total trades: {len(trades_df)}")

    # Find M15 data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "..", "XAUUSDM15.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "XAUUSDM15.csv")
    if not os.path.exists(csv_path):
        print("ERROR: Cannot find XAUUSDM15.csv")
        sys.exit(1)

    print(f"Loading M15 data from: {csv_path}")
    m15_df = load_m15_data(csv_path)
    print(f"Loaded {len(m15_df)} M15 bars")

    # Select trades
    count = min(args.count, len(trades_df))
    if args.spread == "even":
        # Evenly spaced across all trades
        indices = np.linspace(0, len(trades_df) - 1, count, dtype=int)
    elif args.spread == "recent":
        indices = range(len(trades_df) - count, len(trades_df))
    elif args.spread == "random":
        indices = np.random.choice(len(trades_df), count, replace=False)
        indices.sort()
    elif args.spread == "best":
        sorted_idx = trades_df['pnl_dollars'].argsort()[::-1]
        indices = sorted(sorted_idx[:count])
    elif args.spread == "worst":
        sorted_idx = trades_df['pnl_dollars'].argsort()
        indices = sorted(sorted_idx[:count])

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(base_dir, "data", "trade_charts")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating {count} trade charts ({args.spread} spread)...")
    print(f"Output: {output_dir}")

    generated = 0
    for num, idx in enumerate(indices, 1):
        trade = trades_df.iloc[idx]
        try:
            success = plot_trade(trade, m15_df, output_dir, num)
            if success:
                generated += 1
                if generated % 10 == 0:
                    print(f"  Generated {generated}/{count} charts...")
        except Exception as e:
            print(f"  Warning: Failed to plot trade #{num}: {e}")

    print(f"\nDone! Generated {generated} trade charts in {output_dir}")

    # Generate summary HTML page
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html><head><title>Trade Visualizations — Raja Banks Bot</title>
<style>
body { background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; margin: 20px; }
h1 { color: #00e5ff; text-align: center; }
.stats { text-align: center; margin: 20px 0; color: #aaa; font-size: 14px; }
.grid { display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 1200px; margin: 0 auto; }
.trade-card { background: #16213e; border-radius: 8px; overflow: hidden; border: 1px solid #333; }
.trade-card img { width: 100%; display: block; }
.trade-info { padding: 10px 15px; font-size: 13px; }
.win { border-left: 4px solid #00e676; }
.loss { border-left: 4px solid #ff5252; }
</style></head><body>
<h1>Trade Visualizations — Raja Banks Market Fluidity</h1>
<div class="stats">""")
        wins = sum(1 for idx in indices if trades_df.iloc[idx]['pnl_dollars'] > 0)
        total_pnl = sum(trades_df.iloc[idx]['pnl_dollars'] for idx in indices)
        f.write(f"Showing {count} trades | Wins: {wins} | Losses: {count-wins} | "
                f"Total PnL: ${total_pnl:+,.2f}")
        f.write('</div><div class="grid">')

        for num, idx in enumerate(indices, 1):
            trade = trades_df.iloc[idx]
            is_win = trade['pnl_dollars'] > 0
            result = "WIN" if is_win else "LOSS"
            css_class = "win" if is_win else "loss"
            filename = f"trade_{num:03d}_{trade['direction']}_{trade['entry_type']}_{result}.png"
            if os.path.exists(os.path.join(output_dir, filename)):
                f.write(f'<div class="trade-card {css_class}">')
                f.write(f'<img src="{filename}" alt="Trade #{num}">')
                f.write(f'<div class="trade-info">'
                        f'{trade["direction"]} {trade["entry_type"]} | '
                        f'PnL: ${trade["pnl_dollars"]:+.2f} | '
                        f'RR: {trade["rr_achieved"]:.2f} | '
                        f'{trade["exit_reason"]}</div></div>')

        f.write('</div></body></html>')

    print(f"Summary page: {html_path}")


if __name__ == "__main__":
    main()
