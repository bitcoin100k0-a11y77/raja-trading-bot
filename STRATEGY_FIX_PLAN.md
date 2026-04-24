# Strategy Fix Plan — "Random Trades" Diagnosis & Remediation

**Date:** 2026-04-13
**Problem:** Bot is taking trades that don't match the Raja Banks Market Fluidity strategy.
**Root cause:** The C1/C0 signal pipeline qualification criteria are too permissive at multiple stages, allowing nearly any candle near any zone to generate a signal.

---

## Executive Summary

The bot's safety gates (C0 direction check, same-bar guard, slippage gate, SL staging, session filter) are all correct and working. The problem is upstream: **what qualifies as a valid C1 and C0 is far too loose**, and **S/R zones are too weak/numerous**. The result is that the bot fires on noise rather than high-conviction rejection patterns.

7 bugs identified across 3 files. Ordered by impact.

---

## Bug #1 — CRITICAL: C0 wick validation is non-existent

**File:** `bot/strategy.py` — `_check_pending_c0()` ~lines 351-386
**Impact:** Nearly every M15 bar qualifies as C0

### Current behavior
```python
# BUY C0 check:
c0_has_wick = c0_l < c0_o  # any lower wick below C0 open
```
A bar with O=3200, L=3199.90, C=3201 passes — the "wick" is $0.10 (1 pip). On XAUUSD M15, virtually every bar has at least a tiny wick below open.

### What Raja Banks requires
C0 must create a **meaningful wick** that shows a genuine attempt to push into the zone followed by rejection. A 1-pip wick is market noise, not a rejection pattern.

### Fix
Add a minimum C0 wick depth in pips. Suggested: **5 pips minimum** ($0.50 on XAUUSD). This filters noise wicks while allowing genuine rejection patterns through.

```python
# Add to StrategyConfig:
c0_min_wick_pips: float = 5.0  # Minimum C0 wick depth for valid rejection

# In _check_pending_c0 BUY:
c0_wick_depth_pips = price_to_pips(c0_o - c0_l)
c0_has_wick = c0_wick_depth_pips >= self.cfg.c0_min_wick_pips

# In _check_pending_c0 SELL:
c0_wick_depth_pips = price_to_pips(c0_h - c0_o)
c0_has_wick = c0_wick_depth_pips >= self.cfg.c0_min_wick_pips
```

### Validation
Log every C0 rejection with wick depth. After fix, "C0 wick too small" should be the most common rejection reason. Verify no valid signals from backtest are lost at 5 pip threshold (tune if needed).

---

## Bug #2 — CRITICAL: S/R zones formed on single pivots (too weak)

**File:** `bot/market_analyzer.py` — `_detect_sr_levels()` ~lines 214-218
**Impact:** Chart covered in meaningless zones; C1 fires at random levels

### Current behavior
```python
min_score = self.cfg.sr_min_strength * 20   # = 1 * 20 = 20
```
Each individual pivot adds +20 to the score. So a channel containing a **single pivot** with zero additional bar-touches has score=20 and PASSES. With `pivot_period=10` and 290 bars scanned, there are dozens of pivots, creating zones everywhere.

The 6 strongest channels are selected, but "strongest" among many weak zones is still weak.

### What Raja Banks requires
S/R zones should be areas where price has **repeatedly** reacted — multiple touches, visible on the chart as clear horizontal bands of price congestion.

### Fix
Raise `sr_min_strength` from 1 to **2** (minimum 2 pivot touches to form a zone). This means `min_score = 2 * 20 = 40`, requiring at least 2 pivots in the channel window.

Also add a minimum bar-body-touch count to filter channels that only have distant pivots but no actual price interaction:

```python
# In config.py StrategyConfig:
sr_min_strength: int = 2        # Require at least 2 pivot touches
sr_min_body_touches: int = 5    # Minimum bar bodies overlapping the zone

# In market_analyzer.py after bar-touch scoring (~line 210-212):
# Add filter: reject channels with fewer than sr_min_body_touches bar overlaps
body_touch_count = int(np.sum(body_top_in | body_bot_in))
if body_touch_count < self.cfg.sr_min_body_touches:
    supres[x][0] = -1  # disqualify
else:
    supres[x][0] += body_touch_count
```

### Validation
Log zone count before and after. Before fix: expect 6 zones nearly always. After fix: expect 2-4 zones on average, all with meaningful multi-touch history. Compare detected zones against chart visually.

---

## Bug #3 — HIGH: C1 body position check doesn't enforce "close away from zone"

**File:** `bot/strategy.py` — `_scan_c1_rejection()` ~lines 224-228, 262-265
**Impact:** Candles closing INSIDE the zone pass as C1 rejection candles

### Current behavior (BUY at support)
```python
body_bottom = min(o, c)
body_ok = body_bottom > level.price - tol * 0.5   # body_bottom > zone - $4
```
A candle whose entire body is $3 below the zone midpoint passes. That's a candle closing INTO the zone, not away from it. The strategy explicitly requires the body to close AWAY (above support, below resistance).

### What Raja Banks requires
- **BUY at support:** C1 wick dips INTO zone, body closes ABOVE zone level
- **SELL at resistance:** C1 wick pokes INTO zone, body closes BELOW zone level

### Fix
Tighten the body check to require the body to be on the correct side of the zone:

```python
# BUY at support — body must close ABOVE the zone level (rejection away)
body_bottom = min(o, c)
body_ok = body_bottom >= level.price  # body entirely above zone midpoint

# SELL at resistance — body must close BELOW the zone level (rejection away)
body_top = max(o, c)
body_ok = body_top <= level.price  # body entirely below zone midpoint
```

If this is too strict (filtering too many valid C1s), allow a small tolerance:
```python
body_ok = body_bottom >= level.price - tol * 0.15  # max $1.20 below zone
```

### Validation
Log every C1 scan with body position relative to zone. Compare against the 4.2Y backtest — verify win rate doesn't collapse.

---

## Bug #4 — HIGH: C1 wick tolerance is asymmetric and too wide

**File:** `bot/strategy.py` — `_scan_c1_rejection()` ~lines 220-221, 258-259
**Impact:** Candles far from the zone qualify as C1

### Current behavior (BUY at support)
```python
wick_ok = l <= level.price + tol and l >= level.price - tol * 2
# tol = $8 → window = [zone - $16, zone + $8] = $24 total
```
A $24 window is enormous on XAUUSD M15 where typical candle range is $5-15. A candle low that's $15 below the zone still passes. This is NOT "wick dipping into zone."

For SELL:
```python
wick_ok = h >= level.price - tol and h <= level.price + tol * 2
# window = [zone - $8, zone + $16] = $24 total
```

### What Raja Banks requires
The C1 wick should penetrate INTO the zone by a meaningful amount (the $8 tolerance). The strategy document says "±$8 tolerance." The asymmetric 2x multiplier on the far side isn't part of the strategy.

### Fix
Make the tolerance symmetric and tighten:

```python
# BUY at support — wick dips INTO zone from above:
# C1 low should be near zone level (below it or slightly above)
wick_ok = l <= level.price + tol and l >= level.price - tol
# window = [zone - $8, zone + $8] = $16 total (symmetric)

# SELL at resistance — wick pokes INTO zone from below:
wick_ok = h >= level.price - tol and h <= level.price + tol
```

### Validation
Log wick distance from zone for every C1 candidate. Verify that rejected C1s are genuinely far from the zone.

---

## Bug #5 — MEDIUM: No C0 proximity check to C1 / S/R zone

**File:** `bot/strategy.py` — `_check_pending_c0()` ~lines 341-408
**Impact:** C0 can fire $50 away from the zone; entry has no relationship to the S/R level

### Current behavior
C0 only needs: (1) wick exists, (2) correct direction close. There's NO check that C0 is anywhere near the C1 level or the S/R zone. If C1 rejects at support $3100, and C0 opens at $3150 (50 pips away), that's still a valid C0.

### What Raja Banks requires
C0 is the "confirmation candle" immediately following C1. It should form near the zone — not 50 pips away in a completely different price region.

### Fix
Add a maximum distance check between C0 and the C1 entry reference:

```python
# Add to StrategyConfig:
c0_max_distance_from_c1_pips: float = 30.0  # C0 must form within 30 pips of C1 level

# In _check_pending_c0, before the trigger logic:
c0_distance_from_c1 = price_to_pips(abs(c0_o - pending.entry_price))
if c0_distance_from_c1 > self.cfg.c0_max_distance_from_c1_pips:
    logger.debug(
        "C0 skipped: too far from C1 (%.1f pips, max=%.1f)",
        c0_distance_from_c1, self.cfg.c0_max_distance_from_c1_pips
    )
    still_pending.append(pending)
    continue
```

### Validation
Log distance for every C0 evaluation. Tune threshold using backtest data.

---

## Bug #6 — MEDIUM: Duplicate pending orders accumulate

**File:** `bot/strategy.py` — `_scan_c1_rejection()` ~lines 247, 285
**Impact:** Multiple pending orders for same zone → multiple entries from one setup

### Current behavior
Every bar scan can create new pending C0 orders. If consecutive bars each qualify as C1 at the same zone, multiple pendings accumulate. When C0 fires, it can trigger multiple signals simultaneously.

### Fix
Before adding a new pending, check if one already exists for the same zone + direction:

```python
# At the top of _scan_c1_rejection:
existing_directions = {
    (p.sr_level.price, p.direction) for p in self.pending_orders
}

# Before self.pending_orders.append(pending):
if (level.price, TradeDirection.BUY) not in existing_directions:
    self.pending_orders.append(pending)
else:
    logger.debug("Skipping duplicate C1 pending for zone %.2f BUY", level.price)
```

### Validation
Log when duplicates are skipped. Monitor that only one pending exists per zone/direction combo.

---

## Bug #7 — LOW: `is_tradeable_session()` default parameter is stale

**File:** `bot/utils.py` — `is_tradeable_session()` ~line 154
**Impact:** LOW — all callers currently pass explicit values, but any future caller using the default would use session_start=6 (Asia session)

### Current behavior
```python
def is_tradeable_session(utc_hour: int, session_start: int = 6,
                         session_end: int = 22) -> bool:
```
Default `session_start=6` is stale (should be 8). All current callers pass explicit config values, so this doesn't cause bugs TODAY. But it's a landmine.

### Fix
Change default to match the actual strategy:
```python
def is_tradeable_session(utc_hour: int, session_start: int = 8,
                         session_end: int = 22) -> bool:
```

---

## Implementation Order

| Priority | Bug | File | Risk Level | Estimated Complexity |
|----------|-----|------|------------|---------------------|
| 1 | #1 C0 min wick | strategy.py | Low (additive filter) | Simple — add 1 config param + 2 line change per direction |
| 2 | #2 S/R min strength | market_analyzer.py + config.py | Medium (changes zone detection) | Moderate — add config params + filter logic |
| 3 | #3 C1 body close-away | strategy.py | Medium (tightens C1 filter) | Simple — change 2 comparison lines |
| 4 | #4 C1 symmetric tolerance | strategy.py | Medium (tightens C1 filter) | Simple — remove `* 2` multiplier |
| 5 | #5 C0 proximity to C1 | strategy.py | Low (additive filter) | Simple — add 1 config param + distance check |
| 6 | #6 Duplicate pendings | strategy.py | Low (edge case) | Simple — add set check before append |
| 7 | #7 Stale default | utils.py | Zero (defensive) | Trivial — change default value |

### Deployment strategy
- Fix #1 + #7 first (highest impact + trivial). Test on VPS shadow mode.
- Fix #2 next (zone quality). Compare zones visually against chart.
- Fix #3 + #4 together (C1 tightening). These are related.
- Fix #5 + #6 last (C0 proximity + dedup). Lower priority.
- After each fix: run at least 1 full session in shadow mode before enabling live.

---

## Files to Modify

| File | Bugs | Changes |
|------|------|---------|
| `bot/config.py` | #1, #2, #5 | Add `c0_min_wick_pips`, `sr_min_body_touches`, `c0_max_distance_from_c1_pips` to StrategyConfig. Update `sr_min_strength` default from 1 → 2. |
| `bot/strategy.py` | #1, #3, #4, #5, #6 | Tighten C0 wick check, C1 body check, C1 tolerance, add C0 proximity, add pending dedup |
| `bot/market_analyzer.py` | #2 | Add min body-touch filter to `_detect_sr_levels()` |
| `bot/utils.py` | #7 | Change `is_tradeable_session` default from 6 → 8 |

---

## Verification Checklist (After All Fixes)

- [ ] Bot runs 1 full session (08:00-22:00 UTC) in shadow mode with zero crashes
- [ ] S/R zones: typically 2-4 per analysis (was: always 6)
- [ ] C1 candidates per session: reduced (log count before/after)
- [ ] C0 fires: reduced (log count before/after)
- [ ] Signals that pass all filters: 0-2 per day (was: potentially many more)
- [ ] Every signal that fires has: (a) multi-touch S/R zone, (b) body closing away from zone, (c) meaningful C0 wick, (d) C0 near C1 level
- [ ] TP2 is always beyond TP1 (existing fix confirmed still works)
- [ ] Backtest with new parameters: compare win rate vs old (should improve, not collapse)
- [ ] Daily trade count: max 1-3 per day (per strategy intent)

---

## What NOT to Change

These are CORRECT and WORKING — do not touch:
- C0 direction check (bullish for BUY, bearish for SELL) — `strategy.py`
- Same-bar C0 guard (`c1_bar_time`) — `strategy.py`
- Slippage gate (20 pips) — `mt5_executor.py`
- SL staging system (3-stage) — `trade_manager.py`
- Trailing stop ratchet (high-water mark) — `trade_manager.py`
- Session filter (08:00-22:00 UTC) — `config.py`
- Blackout filter (11:00-14:00 UTC) — `strategy.py`
- Broker reconciliation — `broker_reconciler.py`
- TP2 S/R guard (must be beyond TP1) — `strategy.py`
- Body-based S/R detection method (`_detect_pivots_body`) — `market_analyzer.py`
- `pos=0` for data fetch — `mt5_connector.py`
- `df.iloc[-2]` completed bar rule — `strategy.py`

---

## Reference: Raja Banks C1/C0 Pattern (From Course Notes)

**C1 (Rejection Candle):**
- Forms at S/R zone
- Wick penetrates INTO the zone (meaningful depth, not noise)
- Body closes AWAY from zone (body is on the exit side)
- Body ratio >= 0.30

**C0 (Confirmation Candle):**
- Forms 1-3 bars after C1
- Creates a wick in the same direction as C1
- Body closes in the trade direction (bullish for BUY, bearish for SELL)
- Near the C1 level / S/R zone (not far away)

**S/R Zones:**
- Are ZONES (bands), not lines (~10 pip thickness)
- Multiple touches / price congestion history
- Visible on chart as clear horizontal levels
- Typical range between zones: 25-45 pips
