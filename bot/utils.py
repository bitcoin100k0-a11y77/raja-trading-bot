"""
Raja Banks Trading Agent - Utility Functions
Pip calculations, candle analysis, time helpers.
"""
import numpy as np
from datetime import datetime, timezone
from constants import WickType, TrendType, SessionType


# === PIP CALCULATIONS (XAUUSD) ===

def price_to_pips(price_diff: float, pip_size: float = 0.10) -> float:
    return abs(price_diff) / pip_size


def pips_to_price(pips: float, pip_size: float = 0.10) -> float:
    return pips * pip_size


def calculate_lot_size(balance: float, risk_pct: float, risk_pips: float,
                       pip_value: float = 10.0, min_lots: float = 0.01,
                       max_lots: float = 100.0) -> float:
    if risk_pips <= 0 or pip_value <= 0:
        return min_lots
    risk_dollars = balance * risk_pct / 100.0
    lots = risk_dollars / (risk_pips * pip_value)
    return round(max(min_lots, min(max_lots, lots)), 2)


# === CANDLE ANALYSIS ===

def body_ratio(open_p: float, close_p: float, high: float, low: float) -> float:
    candle_range = high - low
    if candle_range <= 0:
        return 0.0
    return abs(close_p - open_p) / candle_range


def is_bullish(open_p: float, close_p: float) -> bool:
    return close_p > open_p


def is_bearish(open_p: float, close_p: float) -> bool:
    return close_p < open_p


def candle_body_size(open_p: float, close_p: float) -> float:
    return abs(close_p - open_p)


def get_body_top(open_p: float, close_p: float) -> float:
    return max(open_p, close_p)


def get_body_bottom(open_p: float, close_p: float) -> float:
    return min(open_p, close_p)


def classify_wick(open_p: float, close_p: float, high: float, low: float):
    body_top = get_body_top(open_p, close_p)
    body_bot = get_body_bottom(open_p, close_p)
    upper_size = high - body_top
    lower_size = body_bot - low
    if is_bullish(open_p, close_p):
        return WickType.RANGE, WickType.LIQUIDITY, upper_size, lower_size
    else:
        return WickType.LIQUIDITY, WickType.RANGE, upper_size, lower_size


def is_wickless(open_p: float, close_p: float, high: float, low: float,
                threshold: float = 0.05) -> bool:
    rng = high - low
    if rng <= 0:
        return True
    body_top = get_body_top(open_p, close_p)
    body_bot = get_body_bottom(open_p, close_p)
    upper_wick_ratio = (high - body_top) / rng
    lower_wick_ratio = (body_bot - low) / rng
    return upper_wick_ratio < threshold and lower_wick_ratio < threshold


def candle_volume_strength(body_size: float, avg_body_size: float) -> str:
    if avg_body_size <= 0:
        return "unknown"
    ratio = body_size / avg_body_size
    if ratio > 1.5:
        return "strong"
    elif ratio > 0.8:
        return "normal"
    else:
        return "weak"


# === TREND HELPERS ===

def detect_swing_highs(highs: np.ndarray, window: int = 5) -> list:
    swings = []
    for i in range(window, len(highs) - window):
        left = highs[i - window:i]
        right = highs[i + 1:i + window + 1]
        if highs[i] >= max(left) and highs[i] >= max(right):
            swings.append(i)
    return swings


def detect_swing_lows(lows: np.ndarray, window: int = 5) -> list:
    swings = []
    for i in range(window, len(lows) - window):
        left = lows[i - window:i]
        right = lows[i + 1:i + window + 1]
        if lows[i] <= min(left) and lows[i] <= min(right):
            swings.append(i)
    return swings


def identify_hh_hl(highs: list, lows: list) -> bool:
    if len(highs) < 2 or len(lows) < 2:
        return False
    hh = all(highs[i] > highs[i - 1] for i in range(1, len(highs)))
    hl = all(lows[i] > lows[i - 1] for i in range(1, len(lows)))
    return hh and hl


def identify_ll_lh(highs: list, lows: list) -> bool:
    if len(highs) < 2 or len(lows) < 2:
        return False
    ll = all(lows[i] < lows[i - 1] for i in range(1, len(lows)))
    lh = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))
    return ll and lh


# === SESSION / TIME HELPERS ===

def get_session(utc_hour: int) -> SessionType:
    if 22 <= utc_hour or utc_hour < 2:
        return SessionType.PRE_ASIAN
    elif 2 <= utc_hour < 6:
        return SessionType.ASIAN
    elif 6 <= utc_hour < 8:
        return SessionType.PRE_LONDON
    elif 8 <= utc_hour < 13:
        return SessionType.LONDON
    elif 13 <= utc_hour < 17:
        return SessionType.NY_LONDON_OVERLAP
    elif 17 <= utc_hour < 22:
        return SessionType.END_OF_DAY
    return SessionType.OFF_SESSION


def is_high_volume_session(session: SessionType) -> bool:
    return session in (SessionType.LONDON, SessionType.NY_LONDON_OVERLAP)


def is_tradeable_session(utc_hour: int, session_start: int = 6,
                         session_end: int = 22) -> bool:
    return session_start <= utc_hour < session_end


def get_4h_candle_number(utc_hour: int) -> int:
    if 22 <= utc_hour or utc_hour < 2:
        return 1
    elif 2 <= utc_hour < 6:
        return 2
    elif 6 <= utc_hour < 10:
        return 3
    elif 10 <= utc_hour < 14:
        return 4
    elif 14 <= utc_hour < 18:
        return 5
    else:
        return 6


_simulated_time = None


def set_simulated_time(t: datetime):
    global _simulated_time
    _simulated_time = t


def utc_now() -> datetime:
    if _simulated_time is not None:
        return _simulated_time
    return datetime.now(timezone.utc)


def bars_since(last_time: datetime, bar_interval_minutes: int = 15) -> int:
    if last_time is None:
        return 999
    diff = (utc_now() - last_time).total_seconds()
    return int(diff / (bar_interval_minutes * 60))
