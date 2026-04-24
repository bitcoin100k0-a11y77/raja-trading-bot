# ANi's FX Bot — CLAUDE.md

**Raja Banks Market Fluidity Strategy — MT5 Live Trading**
Single codebase. Windows VPS only. MetaTrader5 Python API.

---

## What This Bot Does

Automated XAUUSD trading on M15 timeframe using the Raja Banks Market Fluidity Strategy.
Detects S/R zones (LonesomeTheBlue method), waits for C1/C0 wick rejection pattern at zone edge,
enters with stop-limit logic, manages trade through 3-stage SL system + TP1/TP2.

**Status:** 🔴 ACTIVE — Live/demo deployment on Windows VPS via RDP.
Paper trading: PASSED (4.2Y backtest +1,983%, 60D live paper +37.8%).

---

## Module Architecture (code-review-graph — 13 files, 215 nodes, 1722 edges)

Graph analysis found 12 communities. Arranged bottom-up by dependency layer:

```
LAYER 0 — Foundation (no internal deps)
  constants.py     [cohesion=0.00]  Enums only: EntryType, TradeDirection, SLStage, ExitReason
  utils.py         [cohesion=0.04]  ★ ARCHITECTURAL HUB — 64 incoming edges from 5 modules
                                    price_to_pips / pips_to_price called from EVERYWHERE

LAYER 1 — Config (self-contained)
  config.py        [cohesion=0.71]  Highest cohesion. BotConfig.from_env() loads all params.
                                    Safe to change in isolation.

LAYER 2 — Persistence
  database.py      [cohesion=0.22]  30 nodes. TradingDatabase (SQLite). Calls utils.utc_now.

LAYER 3 — Broker
  mt5_connector.py                  MT5 data feed: fetch_bars(), fetch_htf_bars()
  mt5_executor.py  🔴 LIVE RISK     order_send(), modify_sl(), close_position()
  broker_reconciler.py 🔁 RECONCILE Startup position check vs MT5 actual

LAYER 4 — Analysis (calls utils heavily)
  market_analyzer.py [cohesion=0.24] 19 nodes. S/R zone builder. 11 edges → utils.
                                     _detect_pivots_body() is active method.

LAYER 5 — Strategy (highest cross-module coupling)
  strategy.py      [cohesion=0.16]  14 nodes. Signal + PendingC1. 33 edges → utils.
                                    generate_signals() is most complex flow (21 nodes, 3 files).

LAYER 6 — Risk + Trade
  risk_manager.py  [cohesion=0.28]  9 nodes. Position sizing. Daily loss circuit breaker.
  trade_manager.py [cohesion=0.31]  20 nodes. ActiveTrade state. SL staging + TP1/TP2.
                                    20 edges → utils.

LAYER 7 — Intelligence
  learning_agent.py [cohesion=0.12] 9 nodes. Pattern blocker. Calls database.

LAYER 8 — Notification
  telegram_notifier.py [cohesion=0.22] 18 nodes. All 10 alert types. Async via event loop.

LAYER 9 — Orchestrator
  main.py          [cohesion=0.09]  15 nodes. TradingBot. 30s loop. Fans out to all layers.
                                    TradingBot.__init__ is highest-criticality flow (0.66).

LAYER 10 — Offline Only
  backtester.py    [cohesion=0.10]  24 nodes. Not used live. Mirrors main.py fan-out pattern.
```

### Coupling Warnings (from graph)

| Source | Target | Edges | Risk |
|---|---|---|---|
| strategy.py | utils.py | 33 | Breaking utils = breaks signal detection |
| trade_manager.py | utils.py | 20 | Breaking utils = breaks SL staging |
| market_analyzer.py | utils.py | 11 | Breaking utils = breaks S/R zones |

**Before changing utils.py**: run backtester to verify strategy output unchanged.

### Top Execution Flows by Criticality

| Rank | Flow | Criticality | Nodes | Files |
|---|---|---|---|---|
| 1 | TradingBot.__init__ | 0.66 | 8 | 8 |
| 2 | backtester.main | 0.62 | 15 | 4 |
| 3 | generate_signals | 0.58 | 21 | 3 |
| 4 | MarketAnalyzer.analyze | 0.47 | 19 | 2 |
| 5 | TradeManager.update_trades | 0.46 | 10 | 2 |

---

## Project Structure

```
ani-fx-bot/
├── RUN_BOT.ps1             # ⚠️  PRIMARY LAUNCHER — set all 8 env vars here
├── setup.ps1               # One-time VPS setup (pip install, dirs)
├── quick_start.ps1         # Quick re-launch (uses existing env)
├── start_bot.bat           # Shortcut → calls RUN_BOT.ps1
├── stop_bot.bat            # Kills Python process
├── .gitignore
├── data/                   # Runtime data — NOT committed to git
│   ├── trading_bot.db      # SQLite: all trades, patterns, equity
│   └── bot.log             # Rolling log
├── docs/                   # Strategy & deployment reference docs
└── bot/                    # All Python source — 13 modules
    ├── constants.py        # Layer 0: Enums only. No dependencies.
    ├── utils.py            # Layer 0: ★ HUB. Candle math. Called by 5+ modules.
    ├── config.py           # Layer 1: BotConfig. All params loaded from env vars.
    ├── database.py         # Layer 2: SQLite via raw SQL. Trade/pattern/equity tables.
    ├── mt5_connector.py    # Layer 3: MT5 data feed: fetch_bars(), fetch_htf_bars()
    ├── mt5_executor.py     # Layer 3: 🔴 LIVE RISK — order_send(), modify_sl(), close_position()
    ├── broker_reconciler.py # Layer 3: 🔁 RECONCILE — startup position check vs MT5
    ├── market_analyzer.py  # Layer 4: S/R zone builder (LonesomeTheBlue). MarketState.
    ├── strategy.py         # Layer 5: Signal + PendingC1. C1/C0 detection pipeline.
    ├── risk_manager.py     # Layer 6: Position sizing (% balance). Daily loss circuit breaker.
    ├── trade_manager.py    # Layer 6: ActiveTrade state. SL staging. TP1/TP2/trailing.
    ├── learning_agent.py   # Layer 7: Pattern blocker: skips losing entry combos.
    ├── telegram_notifier.py # Layer 8: 📲 All Telegram alerts.
    ├── main.py             # Layer 9: Entry point. TradingBot class. 30s loop.
    └── backtester.py       # Layer 10: Offline backtest engine (not used live).
```

---

## Commands (on Windows VPS)

```powershell
# First-time setup (run once)
.\setup.ps1

# Start bot (set env vars at top of RUN_BOT.ps1 first)
.\RUN_BOT.ps1

# Stop bot
.\stop_bot.bat

# Check recent logs
Get-Content data\bot.log -Tail 100

# Check if bot is running
Get-Process python
```

---

## Signal Pipeline (C1 Stop-Order Model)

```
30s tick
  ↓ mt5_connector.fetch_bars()       — M15 OHLCV from MT5 (pos=0 = forming bar included)
  ↓ new bar detected?                — if same bar, only manage trades + return
  ↓ _expire_pending_stops()          — cancel stop orders > 1 bar old (not filled)
  ↓ reconciler.check_stop_fills()    — clean stops filled by broker (pending→position)
  ↓ market_analyzer.analyze()        — build S/R zones + MarketState
  ↓ strategy.generate_signals()      — C1/C0 on df.iloc[-2] (COMPLETED bar, not forming!)
      C1 bar: wick pokes zone (±$8), body closes AWAY → PendingC1 stored
      C0 bar: 3 gates — distance (≤50p), wick (≥5p), direction (bull/bear)
              all pass → Signal with is_stop_order=True, entry=C1_extreme±2p
  ↓ learning_agent.should_block()    — skip if pattern has >75% loss rate
  ↓ risk_manager.calculate_size()    — lots = risk% of balance / SL pips
  ↓ executor.place_stop_order()      — 🔴 BUY_STOP/SELL_STOP at C1 extreme ± buffer
  ↓ pending_stop_orders dict         — tracked per-ticket, expires after 1 bar (15 min)
  ↓ broker triggers stop → position  — reconciler detects fill on next tick
  ↓ trade_manager + process_exits()  — SL staging, TP1 partial, trailing (unchanged)
  ↓ telegram_notifier                — 📲 alert: STOP PENDING, then TRADE OPENED on fill
```

### Entry Model — C1 + C0 Validation → BUY_STOP / SELL_STOP at C1 Extreme

1. **C1 bar** (df.iloc[-2]): Wick pokes INTO zone (±$8 tolerance). Body closes AWAY from zone (rejection). → Creates PendingC1 with c1_high, c1_low, direction.
2. **C0 bar** (next bar, df.iloc[-2]): 3 gates ALL must pass:
   - Gate 1 (Distance): `|C0_open - zone_price| ≤ 50 pips`
   - Gate 2 (Wick): C0 wick toward zone ≥ 5 pips
   - Gate 3 (Direction): Bullish C0 for BUY setup, bearish C0 for SELL setup
3. **Stop order placed** (all gates pass):
   - BUY: `entry = C1_high + 2 pips` (BUY_STOP). `SL = C1_low`
   - SELL: `entry = C1_low − 2 pips` (SELL_STOP). `SL = C1_high`
   - TP1 = 1:1 RR from entry (floor 30 pips). TP2 = 2.5R or opposing S/R.
4. **Expiry**: stop canceled if not filled within **1 bar (15 min)** after C0 validates.
5. **Fill**: broker converts pending → position. Reconciler detects on next tick. trade_manager takes over (SL stages + TP1/TP2 + trailing unchanged).

---

## Strategy Parameters (XAUUSD M15)

| Parameter | Value | Notes |
|---|---|---|
| S/R scan range | 290 bars | LonesomeTheBlue method |
| Pivot period | 10 bars | Lookback + lookforward |
| Zone tolerance | $8 | Wick must poke this deep |
| Max channels | 6 | Best 6 S/R zones only |
| SL Stage 2 | +20 pips profit | Move SL to C0 bar extreme (actual C0 high/low) |
| SL Stage 3 | +40 pips profit | Move SL to breakeven (entry price) |
| Trailing activation | +50 pips | Ratchet trailing begins |
| Trailing distance | 20 pips | Trails 20 pips behind high-water mark (never regresses) |
| TP1 | 1:1 RR | Close 50% position |
| TP2 | 2.5 RR max | Or opposing S/R |
| Min TP1 | 30 pips | Floor regardless of SL |
| Session | 08:00–22:00 UTC | London open to NY close |
| Blackout | 11:00–14:00 UTC | Skip NY open whipsaw |
| Max trades/day | 3 | Hard limit |
| Daily loss limit | 4% | 📵 Circuit breaker |

---

## Environment Variables (Set in RUN_BOT.ps1)

```powershell
# ⚠️ ENV REQUIRED — fill these 8 values before running
$env:MT5_LOGIN          = "YOUR_ACCOUNT_NUMBER"  # ⚠️ MT5 account
$env:MT5_PASSWORD       = "YOUR_PASSWORD"         # ⚠️ MT5 password
$env:MT5_SERVER         = "YOUR_BROKER_SERVER"    # ⚠️ e.g. "ICMarkets-Demo"
$env:MT5_SYMBOL         = "XAUUSD"
$env:MT5_MAGIC          = "20260402"
$env:LIVE_MODE          = "true"                  # 📵 RUN_BOT.ps1 default = REAL MT5 ORDERS (safe on demo account)
$env:DRY_RUN            = "false"                 # Shadow mode: "true" = full logic, no orders placed at all
$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"        # 📲 ALERT
$env:TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID"          # 📲 ALERT
```

Optional tuning (safe defaults apply if omitted):
```
RISK_PERCENT=1.0           # % of balance risked per trade
DAILY_LOSS_LIMIT=4.0       # % max drawdown before shutdown
MAX_TRADES_PER_DAY=3
MIN_SL_PIPS=20.0           # 🔴 LIVE RISK
MAX_ENTRY_SLIPPAGE_PIPS=20.0  # 🔴 LIVE RISK — raised from 10; 30s window on XAUUSD M15
C0_MIN_WICK_PIPS=5.0          # Min C0 wick depth — rejects noise wicks
C0_MAX_DISTANCE_FROM_ZONE_PIPS=30.0  # Max distance from S/R zone to C0 open
SR_MIN_STRENGTH=2              # Min 2 pivots per S/R zone (was 1 = too loose)
SR_MIN_BODY_TOUCHES=3          # Min bar bodies touching zone
REJECTION_TOLERANCE=8.0
TRAILING_ENABLED=true
```

---

## Critical Code Rules — DO NOT CHANGE

### strategy.py — COMPLETED BAR RULE (most important rule in the bot)
- **C1 and C0 MUST use `df.iloc[-2]` (completed bar), NOT `df.iloc[-1]` (forming bar).**
  The forming bar's OHLC changes every tick. Detecting C1 on it gives wrong entry/SL levels
  and C0 direction checks are meaningless on an unclosed bar. Do NOT revert to df.iloc[-1].
- **C0 is a GATE, not the entry bar** — After the new stop-order model, C0 validates whether to place a pending stop. Entry price = C1 extreme ± 2 pips. SL = C1 opposite wick. The old "enter at C0 open" logic is replaced. Do not revert.
- **C0 direction check is required** — BUY requires `is_bullish(c0_o, c0_c)`, SELL requires `is_bearish(c0_o, c0_c)`. A bearish C0 on BUY setup = gates fail, no stop placed. Do not remove this check.
- **Same-bar C0 guard (`c1_bar_time`)** — `_check_pending_c0` skips if current bar IS the C1 bar. C0 can only fire on a bar strictly newer than C1. Without this, bot enters within 30 seconds of C1 close (wrong OHLC, price moved). Do not remove this guard.
- **HTF alignment pre-filter REMOVED** — The block that rejected signals with `not htf_aligned and strength < 10` was deleted. Body-touch filter + pivot strength upstream already gate quality. Re-adding this filter silences valid reactions at orphan zones.

### market_analyzer.py
- **S/R uses body (max/min of Open,Close), NOT wicks** — `_detect_pivots_body()` is the active method. The original wick-based `_detect_pivots()` is kept as legacy but is NOT called. Do not revert to wick-based detection.

### utils.py — ARCHITECTURAL HUB (change with extreme care)
- Called by 5 modules with 64 total edges. Any broken function cascades to strategy, trade_manager, market_analyzer simultaneously.
- `price_to_pips` / `pips_to_price` are the most-called functions in the entire codebase.
- After any change: run backtester end-to-end to verify pip math unchanged.

### mt5_connector.py
- **`pos=0` is intentional for data fetching** — includes forming bar in the DataFrame. S/R zone detection uses 290 bars; one forming bar has minimal impact. But C1/C0 entry logic skips the forming bar (uses df.iloc[-2]). Do not change pos.
- **Bar count = 672** — formula: `lookback_days * 24 * 60 / 15`. This is correct. Do not change.

### mt5_executor.py
- **Stops-level validation (~lines 190–227)** — prevents Error 10016 by auto-pushing SL/TP outward to meet broker minimum. Keep this.
- **`modify_sl()` stops check (~lines 486–495)** — same guard for SL stage moves. Keep this.
- **SL on every `order_send()`** — every call includes `sl=` and `tp=`. Never a naked order.

### RUN_BOT.ps1
- **`Read-Host` OUTSIDE the try/catch** — window stays open on crash. Do not move it inside.
- **Telegram env vars before Python** — alerts fire even on startup crashes.

### strategy.py / market_analyzer.py
- **S/R uses wicks (High/Low), NOT bodies** — this is correct. Do not change pivot detection.
- **`impulse_enabled=False`, `break_retest_enabled=False`** — do not enable without explicit review.

---

## Startup Sequence (Every Restart)

1. Config loaded from env vars
2. MT5 connection established
3. 🔁 `BrokerReconciler.reconcile()` — DB open trades vs MT5 actual positions
4. Discrepancies resolved before any new orders placed
5. 📲 "🟢 BOT STARTED" Telegram alert with reconcile report
6. 30s main loop begins

---

## Telegram Alerts (📲)

| Alert | Trigger |
|---|---|
| 🟢 BOT STARTED | Startup + reconcile complete |
| 🔁 RECONCILE REPORT | Startup, shows position diff |
| 📈/📉 ENTRY PENDING | Before order_send() |
| ✅ ORDER CONFIRMED | MT5 confirms fill |
| 🎯 TP1 HIT | Partial close at 1:1 RR |
| 🏁 TRADE CLOSED | Full close + PnL |
| ⚠️ MT5 ERROR | Any API error |
| 🚨 DAILY LIMIT | Circuit breaker, bot paused |
| 💀 BOT CRASHED | Unhandled exception |

---

## Known Broker Quirks (XAUUSD / ICMarkets)

- `trade_stops_level` can be 0–50 points on XAUUSD. Error 10016 fix handles this automatically.
- Spread widens during news events — no news filter yet (planned enhancement).
- Magic number `20260402` tags all bot orders. Do not use same magic for manual trades.
- Some brokers use `GOLD` or `XAU/USD` — set `MT5_SYMBOL` accordingly.
- Filling mode `IOC` works on ICMarkets. If orders reject, try `MT5_FILLING=RETURN`.

---

## Live Go-Live Checklist

Before `LIVE_MODE=true`:
- [ ] Shadow mode ran cleanly for at least 1 full trading day
- [ ] Reconciler showing 0 discrepancies on startup
- [ ] Telegram alerts confirmed (entry + exit + error)
- [ ] Daily loss limit ≤ 4%
- [ ] Starting lots = 0.01 (minimum)
- [ ] No Error 10016 in recent logs
- [ ] MT5 account has sufficient margin for 0.01 lot XAUUSD
- [ ] PowerShell window stays open on crash (Read-Host working)

---

## What NOT to Do

- ❌ Change `pos=0` → `pos=1` in mt5_connector.py
- ❌ Enable `impulse_enabled` or `break_retest_enabled` without strategy review
- ❌ Remove stops_level validation from mt5_executor.py
- ❌ Place any order without SL in the same `order_send()` call
- ❌ Increase RISK_PERCENT before 30 days of live results
- ❌ Change strategy parameters mid-session
- ❌ Delete `data/trading_bot.db` — contains all trade history and learned patterns
- ❌ Use `df.iloc[-1]` for C1/C0 detection — forming bar OHLC is incomplete. Use `df.iloc[-2]` (completed bar). This was THE root-cause bug for wrong live entries.
- ❌ Revert S/R detection to wick-based — body-based (`_detect_pivots_body`) is active and tested
- ❌ Remove the C0 direction check in `strategy.py` — bearish C0 on BUY setup = false signal
- ❌ Set `SESSION_START` below 8 — 06:00 UTC is Asia session; bot was live-tested and was over-trading
- ❌ Change utils.py without running backtester — it's the hub with 64 incoming edges across 5 modules
