# Raja Banks Trading Bot

XAUUSD M15 trading bot implementing the Market Fluidity Strategy. The bot learns from its mistakes by tracking trade patterns and automatically blocking losing setups.

## Features

- **C1/C0 Rejection Entry** at S/R zones with pending stop orders
- **3-Stage Stop Loss** management (C1 extreme → C0 extreme → breakeven)
- **Learning Agent** that categorizes losses, tracks patterns, and blocks setups with >75% loss rate
- **Dynamic Risk** (Raja's 2/0.5 system) - reduces risk after consecutive losses
- **Telegram Alerts** for signals, trades, status updates, and daily summaries
- **Health Check API** for Railway deployment monitoring

## Backtest Results (Dec 2021 - Mar 2026)

| Metric | Value |
|--------|-------|
| Total Return | +609% |
| Win Rate | 55% |
| Profit Factor | 1.53 |
| Max Drawdown | 7.6% |
| Red Months | 7 of 52 |

## Quick Start

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/raja-trading-bot.git
cd raja-trading-bot
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Telegram bot token
python bot/main.py
```

### Deploy to Railway

1. Push to GitHub
2. Connect repo in [Railway](https://railway.app)
3. Add environment variables from `.env.example`
4. Add a **Volume** mounted at `/app/data` for persistent storage
5. Deploy

### Telegram Setup

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → get your token
2. Message [@userinfobot](https://t.me/userinfobot) → get your chat ID
3. Set `TELEGRAM_ENABLED=true`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID`

## How the Learning Agent Works

Every closed trade goes through this pipeline:

1. **Loss Categorization** - Each loss gets tagged (false breakout, early reversal, tight SL, etc.)
2. **Pattern Key** - Trade context encoded as: `{entry_type}_{session}_{trend}_{sr_strength}_{body_strength}`
3. **Analysis** - Every 10 trades, all patterns are re-evaluated
4. **Blocking** - Patterns with >75% loss rate (15+ trades) get blocked
5. **Confidence Scoring** - New signals checked against blocked patterns, rejected if low confidence
6. **Recovery** - Blocked patterns automatically unblock if performance improves

The learning persists in SQLite, so the bot keeps learning across restarts.

## Health Check

Railway monitors `GET /health` which returns:

```json
{
  "status": "running",
  "balance": 10000.0,
  "total_trades": 0,
  "win_rate": 0.0,
  "data_healthy": true,
  "patterns_blocked": 0
}
```

## Environment Variables

See `.env.example` for all configuration options.

## Project Structure

```
raja-trading-bot/
├── bot/
│   ├── main.py              # Main loop + health check server
│   ├── config.py             # All config from env vars
│   ├── constants.py          # Enums and constants
│   ├── utils.py              # Pip calc, candle analysis, time helpers
│   ├── database.py           # SQLite persistence layer
│   ├── data_connector.py     # Yahoo Finance data with retries
│   ├── market_analyzer.py    # Trend, S/R detection, volume
│   ├── strategy.py           # Signal generation (C1/C0, filters)
│   ├── trade_manager.py      # 3-stage SL, TP, exits
│   ├── paper_trader.py       # Simulated execution + P&L
│   ├── risk_manager.py       # Position sizing, daily limits
│   ├── learning_agent.py     # Pattern tracking + blocking
│   └── telegram_notifier.py  # Telegram alerts
├── Dockerfile
├── railway.toml
├── Procfile
├── requirements.txt
├── .env.example
└── .gitignore
```
