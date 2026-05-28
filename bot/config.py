"""
MT5 Trading Bot - Configuration
All parameters loaded from environment variables with sensible defaults.
Optimized values from 4+ years of XAUUSD M15 backtesting (2021-2026).
Live MT5 trading with hard safety switches.
"""
import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class StrategyConfig:
    # === INSTRUMENT ===
    symbol: str = "GC=F"
    timeframe: str = "15m"
    pip_size: float = 0.10
    pip_value_per_lot: float = 10.0

    # === S/R CHANNEL DETECTION (LonesomeTheBlue method) ===
    pivot_period: int = 10          # Lookback/lookforward bars for pivot detection
    channel_width_pct: float = 5.0  # Max channel width as % of 300-bar range
    sr_min_strength: int = 2        # Min pivot count per channel (2 = at least 2 pivots must cluster)
    sr_min_body_touches: int = 2    # Min bar bodies overlapping zone (lowered 3→2: catch 2-touch reactions)
    sr_max_channels: int = 6        # Max S/R channels to detect
    sr_loopback: int = 290          # Bars to scan for pivots

    # Legacy aliases (kept for backward compat with existing code paths)
    swing_lookback: int = 10
    sr_zone_tolerance: float = 3.0
    sr_scan_range: int = 290
    min_touches: int = 1
    min_level_spacing: float = 0.0
    max_levels_per_side: int = 6

    # === C1/C0 REJECTION ENTRY ===
    c1_body_min: float = 0.30
    c1_min_range_pips: float = 0.0
    rejection_tolerance: float = 8.0   # $8 tolerance — gold M15 candles range $5-15
    pending_expiry_bars: int = 2       # 🔴 MT5 pending-order expiry: 2 bars (30 min) after C1 close.
                                       # Strategy no longer queues pending — MT5 holds the stop directly.
                                       # Structural invalidation (main._check_stop_invalidation) cancels
                                       # early if the C1 wick is breached intra-bar.
    c1_stop_buffer_pips: float = 2.0   # Buffer above C1 high (BUY) / below C1 low (SELL) for stop trigger

    # === IMPULSE ENTRY ===
    impulse_enabled: bool = False
    impulse_min_body_ratio: float = 0.50
    impulse_volume_multiplier: float = 1.5
    impulse_clean_range_lookback: int = 5

    # === BREAK & RETEST ===
    break_retest_enabled: bool = False
    break_confirmation_pips: float = 5.0
    retest_max_bars: int = 8

    # === C0 CONFIRMATION CANDLE QUALITY ===
    # C0 is now a VALIDATION GATE (not final entry bar). Actual entry is a STOP order
    # placed at C1 extreme + c1_stop_buffer_pips. These knobs gate when the stop gets placed.
    # Minimum wick depth for a valid C0 (avoids 1-pip noise wicks opening stop orders)
    c0_min_wick_pips: float = 5.0          # ⚠️ ENV: C0_MIN_WICK_PIPS  (5 pips: C0 is gate, not entry — stop order at C1 filters false rejections)
    # Maximum distance (pips) between C0 open and the S/R zone that generated C1.
    # C0 must form near the zone, not far off in random price space.
    c0_max_distance_from_zone_pips: float = 50.0  # ⚠️ ENV: C0_MAX_DISTANCE_FROM_ZONE_PIPS  (loosened 30→50)

    # === ENTRY SLIPPAGE GATE ===
    # 🔴 LIVE RISK — max pips from C0 close we still accept a market order.
    # The bot detects C0 on the next 30-second tick after the bar closes.
    # XAUUSD M15 can move 10-20 pips in 30 seconds. 20 pips handles normal
    # post-C0-close movement without skipping valid setups.
    max_entry_slippage_pips: float = 20.0  # ⚠️ ENV: MAX_ENTRY_SLIPPAGE_PIPS

    # === PRE-ENTRY FILTERS ===
    min_room_pips: float = 15.0     # $1.5 room to opposing S/R (was $3, too tight)
    min_sr_gap_pips: float = 20.0
    max_mid_range_pct: float = 0.30  # 30% dead zone: blocks 35%-65% only.
                                     # 0.50 killed valid near-S/R London setups at
                                     # 0.27-0.34 (near support) and 0.66-0.72 (near
                                     # resistance). ⚠️ ENV: MAX_MID_RANGE_PCT
    min_sl_pips: float = 10.0         # 🔴 LIVE RISK — stop-order model uses C1 wick tip SL (tighter than old C0-market-entry SL).
                                      # 20p floor inherited from C0-market-entry model; was blocking most valid C1 setups silently.
    max_risk_pips: float = 300.0
    max_spread_pips: float = 10.0

    # === STOP LOSS RULES ===
    # 🔴 LIVE RISK — Initial SL at C1 wick tip (BUY=c1_low, SELL=c1_high).
    # When C1 candle range exceeds sl_midpoint_cap_pips, cap SL to the candle
    # midpoint. Bounds risk_pips at ~range/2, lets sizing produce larger lots
    # at the same 1% account risk on high-volatility setups.
    sl_midpoint_cap_pips: float = 50.0   # ⚠️ ENV: SL_MIDPOINT_CAP_PIPS

    # Favorable distance that triggers SL → breakeven jump (single-jump model).
    # Replaces the legacy 3-stage ladder (+20 → C0, +40 → BE). After BE,
    # trade runs to TP1/TP2/BE-stop with no further SL movement (unless
    # protect_to_tp2=False which re-enables trailing).
    sl_breakeven_pips: float = 30.0      # ⚠️ ENV: SL_BREAKEVEN_PIPS

    # DEPRECATED — kept for backward compat with old code paths and
    # legacy DB rows (sl_stage column accepts 1/2/3). STAGE_2 is now
    # unreachable for new trades; old trades restored from DB with
    # sl_stage=2 still deserialize and advance to STAGE_3 at +30p.
    sl_stage2_pips: float = 20.0
    sl_stage3_pips: float = 40.0

    # === TRAILING STOP (after Stage 3 / breakeven) ===
    trailing_enabled: bool = True
    trailing_activation_pips: float = 50.0  # Start trailing after +50 pips
    trailing_distance_pips: float = 20.0  # Trail 20 pips behind price

    # === TAKE PROFIT ===
    tp1_rr: float = 1.0
    tp1_close_pct: float = 0.50
    tp2_max_rr: float = 2.5
    min_tp1_pips: float = 30.0        # 🔴 LIVE RISK — TP1 floor: if SL < 30p, TP1 = 30p minimum

    # === SPECIAL EXITS ===
    giveback_trigger_pips: float = 30.0
    giveback_close_pct: float = 0.50
    small_body_threshold: float = 0.35

    # === TRADE PROTECTION ===
    # When True, lock trade to TP1 partial → TP2 full close, OR SL hit, OR session-end.
    # Suppresses ALL three early-exit paths:
    #   • check_opposite_close (emergency reversal exit)
    #   • _check_giveback (pullback-after-peak exit)
    #   • _update_sl_stage trailing branch (SL ratchet behind peak)
    # SL stages 1→2→3 still advance (those move SL toward profit, never close).
    # 🔴 LIVE RISK: setting False reverts to legacy management — strong opposite
    # candles can close 100% pre-TP1, and post-TP1 trailing/giveback can cut the
    # remaining 50% short of TP2.
    protect_to_tp2: bool = True   # ⚠️ ENV: PROTECT_TO_TP2

    # === SESSION FILTER (UTC hours) ===
    # London open = 08:00 UTC, NY close = 22:00 UTC
    # Do NOT change session_start below 8 — 06:00 UTC is still Asia session
    session_start: int = 8   # ⚠️ ENV: SESSION_START — London open
    session_end: int = 22    # ⚠️ ENV: SESSION_END   — NY close
    session_blackout_start: int = 11
    session_blackout_end: int = 14


@dataclass
class RiskConfig:
    risk_percent: float = 1.0
    min_lots: float = 0.01
    max_lots: float = 100.0
    consecutive_loss_threshold: int = 2
    risk_reduction_increment: float = 0.0   # 🔴 LIVE RISK — 0.0 disables consecutive-loss reduction.
                                            # Bot uses flat risk_percent always. Daily loss limit (4%)
                                            # and max-trades-per-day (3) still active. Set 0.5 to
                                            # re-enable Raja 2/0.5 safety system.
    risk_recovery_increment: float = 0.25
    max_trades_per_day: int = 3
    max_concurrent_trades: int = 1   # 🔴 LIVE RISK — one trade at a time. Blocks
                                     # duplicate/same-zone entries (a position OR a
                                     # live pending stop counts against this).
    daily_loss_limit: float = 4.0
    cooldown_bars: int = 4
    second_chance_risk_pct: float = 1.2


@dataclass
class LearningConfig:
    analysis_interval: int = 10
    loss_ratio_threshold: float = 0.75
    min_trades_for_blocking: int = 15
    confidence_base: float = 1.0
    confidence_penalty: float = 0.05
    min_confidence: float = 0.40
    immediate_reject_rate: float = 0.90
    db_path: str = "data/trading_bot.db"


@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    status_interval_minutes: int = 120


@dataclass
class MT5Config:
    """MT5 Connection Configuration — ⚠️ ENV REQUIRED for live trading"""
    login: int = 0               # ⚠️ ENV: MT5_LOGIN
    password: str = ""           # ⚠️ ENV: MT5_PASSWORD
    server: str = ""             # ⚠️ ENV: MT5_SERVER
    mt5_path: str = ""           # ⚠️ ENV: MT5_PATH (optional, auto-detect on Windows)
    symbol: str = "XAUUSD"       # ⚠️ ENV: MT5_SYMBOL
    magic_number: int = 20260402 # ⚠️ ENV: MT5_MAGIC
    max_slippage: int = 20       # ⚠️ ENV: MT5_SLIPPAGE (points)
    filling_mode: str = "IOC"    # ⚠️ ENV: MT5_FILLING
    connect_timeout: int = 60000 # milliseconds


@dataclass
class BotConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    mt5: MT5Config = field(default_factory=MT5Config)

    # === PAPER TRADING ===
    initial_balance: float = 10000.0
    dry_run: bool = True

    # === LIVE MODE ===
    # 📵 KILL SWITCH — hard env var, default OFF
    # Must be explicitly set to "true" for live execution
    live_mode: bool = False

    # === BOT SETTINGS ===
    update_interval_seconds: int = 30
    log_level: str = "INFO"

    # === DEPLOYMENT ===
    health_check_port: int = 8080
    data_dir: str = "data"

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load all config from environment variables."""
        cfg = cls()

        # Bot settings
        cfg.initial_balance = _env_float("INITIAL_BALANCE", 10000.0)
        cfg.dry_run = _env_bool("DRY_RUN", True)
        cfg.update_interval_seconds = _env_int("UPDATE_INTERVAL", 30)
        cfg.log_level = _env("LOG_LEVEL", "INFO")
        cfg.health_check_port = _env_int("PORT", 8080)
        cfg.data_dir = _env("DATA_DIR", "data")

        # Strategy
        cfg.strategy.symbol = _env("SYMBOL", "GC=F")
        cfg.strategy.timeframe = _env("TIMEFRAME", "15m")
        cfg.strategy.max_risk_pips = _env_float("MAX_RISK_PIPS", 300.0)
        cfg.strategy.session_start = _env_int("SESSION_START", 8)   # 08:00 UTC = London open
        cfg.strategy.session_end = _env_int("SESSION_END", 22)     # 22:00 UTC = NY close

        # S/R Channel Detection
        cfg.strategy.pivot_period = _env_int("PIVOT_PERIOD", 10)
        cfg.strategy.channel_width_pct = _env_float("CHANNEL_WIDTH_PCT", 5.0)
        cfg.strategy.sr_max_channels = _env_int("SR_MAX_CHANNELS", 6)
        cfg.strategy.sr_loopback = _env_int("SR_LOOPBACK", 290)
        cfg.strategy.session_blackout_start = _env_int("BLACKOUT_START", 11)
        cfg.strategy.session_blackout_end = _env_int("BLACKOUT_END", 14)

        # SL / TP params
        cfg.strategy.min_sl_pips = _env_float("MIN_SL_PIPS", 10.0)
        cfg.strategy.min_tp1_pips = _env_float("MIN_TP1_PIPS", 30.0)
        cfg.strategy.sl_midpoint_cap_pips = _env_float("SL_MIDPOINT_CAP_PIPS", 50.0)
        cfg.strategy.sl_breakeven_pips    = _env_float("SL_BREAKEVEN_PIPS", 30.0)

        # Entry params
        cfg.strategy.rejection_tolerance = _env_float("REJECTION_TOLERANCE", 8.0)
        cfg.strategy.min_room_pips = _env_float("MIN_ROOM_PIPS", 15.0)
        cfg.strategy.max_mid_range_pct = _env_float("MAX_MID_RANGE_PCT", 0.50)
        cfg.strategy.max_entry_slippage_pips = _env_float("MAX_ENTRY_SLIPPAGE_PIPS", 20.0)  # ⚠️ ENV REQUIRED for live
        cfg.strategy.c0_min_wick_pips = _env_float("C0_MIN_WICK_PIPS", 5.0)
        cfg.strategy.c0_max_distance_from_zone_pips = _env_float("C0_MAX_DISTANCE_FROM_ZONE_PIPS", 50.0)
        cfg.strategy.sr_min_strength = _env_int("SR_MIN_STRENGTH", 2)
        cfg.strategy.sr_min_body_touches = _env_int("SR_MIN_BODY_TOUCHES", 2)
        cfg.strategy.pending_expiry_bars = _env_int("PENDING_EXPIRY_BARS", 2)
        cfg.strategy.c1_stop_buffer_pips = _env_float("C1_STOP_BUFFER_PIPS", 2.0)

        # Entry types
        cfg.strategy.impulse_enabled = _env_bool("IMPULSE_ENABLED", False)
        cfg.strategy.break_retest_enabled = _env_bool("BREAK_RETEST_ENABLED", False)

        # Trailing stop
        cfg.strategy.trailing_enabled = _env_bool("TRAILING_ENABLED", True)
        cfg.strategy.trailing_activation_pips = _env_float("TRAILING_ACTIVATION_PIPS", 50.0)
        cfg.strategy.trailing_distance_pips = _env_float("TRAILING_DISTANCE_PIPS", 20.0)

        # Trade protection (lock to TP1 partial → TP2 full, or SL/session-end only)
        cfg.strategy.protect_to_tp2 = _env_bool("PROTECT_TO_TP2", True)

        # Risk
        cfg.risk.risk_percent = _env_float("RISK_PERCENT", 1.0)
        cfg.risk.max_trades_per_day = _env_int("MAX_TRADES_PER_DAY", 3)
        cfg.risk.max_concurrent_trades = _env_int("MAX_CONCURRENT_TRADES", 1)
        cfg.risk.daily_loss_limit = _env_float("DAILY_LOSS_LIMIT", 4.0)
        cfg.risk.risk_reduction_increment = _env_float("RISK_REDUCTION_INCREMENT", 0.0)  # 🔴 0.0 = disabled

        # Telegram
        cfg.telegram.enabled = _env_bool("TELEGRAM_ENABLED", False)
        cfg.telegram.bot_token = _env("TELEGRAM_BOT_TOKEN", "")
        cfg.telegram.chat_id = _env("TELEGRAM_CHAT_ID", "")
        cfg.telegram.status_interval_minutes = _env_int("STATUS_INTERVAL_MIN", 120)

        # Learning
        cfg.learning.db_path = os.path.join(cfg.data_dir, "trading_bot.db")
        cfg.learning.analysis_interval = _env_int("LEARNING_INTERVAL", 10)
        cfg.learning.min_trades_for_blocking = _env_int("MIN_TRADES_BLOCK", 15)

        # MT5 Connection  ⚠️ ENV REQUIRED for live trading
        cfg.mt5.login = _env_int("MT5_LOGIN", 0)
        cfg.mt5.password = _env("MT5_PASSWORD", "")
        cfg.mt5.server = _env("MT5_SERVER", "")
        cfg.mt5.mt5_path = _env("MT5_PATH", "")
        cfg.mt5.symbol = _env("MT5_SYMBOL", "XAUUSD")
        cfg.mt5.magic_number = _env_int("MT5_MAGIC", 20260402)
        cfg.mt5.max_slippage = _env_int("MT5_SLIPPAGE", 20)
        cfg.mt5.filling_mode = _env("MT5_FILLING", "IOC")

        # 📵 KILL SWITCH — must be explicitly set to "true" for live execution
        cfg.live_mode = _env_bool("LIVE_MODE", False)

        return cfg
