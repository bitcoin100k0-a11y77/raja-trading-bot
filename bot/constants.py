"""
Raja Banks Trading Agent - Constants & Enumerations
"""
from enum import Enum


class EntryType(Enum):
    C1_C0_REJECTION = "c1_c0_rejection"
    IMPULSE = "impulse"
    BREAK_RETEST = "break_retest"


class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


class TrendType(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    CONSOLIDATION = "consolidation"


class SessionType(Enum):
    PRE_ASIAN = "pre_asian"
    ASIAN = "asian"
    PRE_LONDON = "pre_london"
    LONDON = "london"
    NY_LONDON_OVERLAP = "ny_london_overlap"
    END_OF_DAY = "end_of_day"
    OFF_SESSION = "off_session"


class VolumeLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SLStage(Enum):
    STAGE_1 = 1  # Initial: C1 extreme
    STAGE_2 = 2  # +25 pips: C0 extreme
    STAGE_3 = 3  # +40 pips: Breakeven


class ExitReason(Enum):
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    SL_STAGE1 = "sl_stage1"
    SL_STAGE2 = "sl_stage2"
    SL_STAGE3 = "sl_stage3"
    GIVEBACK_EXIT = "giveback_exit"
    EMERGENCY_EXIT = "emergency_exit"
    SESSION_END = "session_end"
    MANUAL = "manual"


class LossCategory(Enum):
    FALSE_BREAKOUT = "false_breakout"
    EARLY_REVERSAL = "early_reversal"
    STAGE2_PULLBACK = "stage2_pullback"
    BREAKEVEN_LOSS = "breakeven_loss"
    GIVEBACK_LOSS = "giveback_loss"
    SESSION_QUALITY = "session_quality"
    HIGH_SPREAD = "high_spread"
    TIGHT_SL = "tight_sl"
    WIDE_RANGE = "wide_range"
    OVERCROWDED_SR = "overcrowded_sr"


class ZoneType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


class WickType(Enum):
    RANGE = "range"
    LIQUIDITY = "liquidity"
