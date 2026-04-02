"""
Raja Banks Trading Bot - Configuration
All parameters loaded from environment variables with sensible defaults.
Optimized values from 4+ years of XAUUSD M15 backtesting (2021-2026).
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
    sr_min_strength: int = 1        # Min pivot count per channel
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
    pending_expiry_bars: int = 3

    # === IMPULSE ENTRY ===
    impulse_enabled: bool = False
    impulse_min_body_ratio: float = 0.50
    impulse_volume_multiplier: float = 1.5
    impulse_clean_range_lookback: int = 5

    # === BREAK & RETEST ===
    break_retest_enabled: bool = False
    break_confirmation_pips: float = 5.0
    retest_max_bars: int = 8

    # === PRE-ENTRY FILTERS ===
    min_room_pips: float = 15.0     # $1.5 room to opposing S/R (was $3, too tight)
    min_sr_gap_pips: float = 20.0
    max_mid_range_pct: float = 0.50  # 50% dead zone (was 35%, too restrictive)
    min_sl_pips: float = 20.0         # 🔴 LIVE RISK — minimum 20 pip SL for conviction (was 5.0)
    max_risk_pips: float = 300.0
    max_spread_pips: float = 10.0

    # === STOP LOSS STAGES ===
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

    # === SESSION FILTER (UTC hours) ===
    session_start: int = 6
    session_end: int = 22
    session_blackout_start: int = 11
    session_blackout_end: int = 14


@dataclass
class RiskConfig:
    risk_percent: float = 1.0
    min_lots: float = 0.01
    max_lots: float = 100.0
    consecutive_loss_threshold: int = 2
    risk_reduction_increment: float = 0.5
    risk_recovery_increment: float = 0.25
    max_trades_per_day: int = 3
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
class BotConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    # === PAPER TRADING ===
    initial_balance: float = 10000.0
    dry_run: bool = True

    # === BOT SETTINGS ===
    update_interval_seconds: int = 30
    log_level: str = "INFO"

    # === DEPLOYMENT ===
    health_check_port: int = 8080
    data_dir: str = "data"

    @classmethod
    def from_env(cls):
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
        cfg.strategy.session_start = _env_int("SESSION_START", 6)
        cfg.strategy.session_end = _env_int("SESSION_END", 22)

        # S/R Channel Detection
        cfg.strategy.pivot_period = _env_int("PIVOT_PERIOD", 10)
        cfg.strategy.channel_width_pct = _env_float("CHANNEL_WIDTH_PCT", 5.0)
        cfg.strategy.sr_min_strength = _env_int("SR_MIN_STRENGTH", 1)
        cfg.strategy.sr_max_channels = _env_int("SR_MAX_CHANNELS", 6)
        cfg.strategy.sr_loopback = _env_int("SR_LOOPBACK", 290)
        cfg.strategy.session_blackout_start = _env_int("BLACKOUT_START", 11)
        cfg.strategy.session_blackout_end = _env_int("BLACKOUT_END", 14)

        # SL / TP params
        cfg.strategy.min_sl_pips = _env_float("MIN_SL_PIPS", 20.0)
        cfg.strategy.min_tp1_pips = _env_float("MIN_TP1_PIPS", 30.0)

        # Entry params
        cfg.strategy.rejection_tolerance = _env_float("REJECTION_TOLERANCE", 8.0)
        cfg.strategy.min_room_pips = _env_float("MIN_ROOM_PIPS", 15.0)
        cfg.strategy.max_mid_range_pct = _env_float("MAX_MID_RANGE_PCT", 0.50)

        # Entry types
        cfg.strategy.impulse_enabled = _env_bool("IMPULSE_ENABLED", False)
        cfg.strategy.break_retest_enabled = _env_bool("BREAK_RETEST_ENABLED", False)

        # Trailing stop
        cfg.strategy.trailing_enabled = _env_bool("TRAILING_ENABLED", True)
        cfg.strategy.trailing_activation_pips = _env_float("TRAILING_ACTIVATION_PIPS", 50.0)
        cfg.strategy.trailing_distance_pips = _env_float("TRAILING_DISTANCE_PIPS", 20.0)

        # Risk
        cfg.risk.risk_percent = _env_float("RISK_PERCENT", 1.0)
        cfg.risk.max_trades_per_day = _env_int("MAX_TRADES_PER_DAY", 3)
        cfg.risk.daily_loss_limit = _env_float("DAILY_LOSS_LIMIT", 4.0)

        # Telegram
        cfg.telegram.enabled = _env_bool("TELEGRAM_ENABLED", False)
        cfg.telegram.bot_token = _env("TELEGRAM_BOT_TOKEN", "")
        cfg.telegram.chat_id = _env("TELEGRAM_CHAT_ID", "")
        cfg.telegram.status_interval_minutes = _env_int("STATUS_INTERVAL_MIN", 120)

        # Learning
        cfg.learning.db_path = os.path.join(cfg.data_dir, "trading_bot.db")
        cfg.learning.analysis_interval = _env_int("LEARNING_INTERVAL", 10)
        cfg.learning.min_trades_for_blocking = _env_int("MIN_TRADES_BLOCK", 15)

        return cfg
