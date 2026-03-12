"""
Raja Banks Trading Agent - Market Analyzer
Trend detection (HH/HL, LL/LH), HTF zone stacking, session mapping, volume/volatility.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from config import StrategyConfig
from constants import TrendType, SessionType, VolumeLevel, ZoneType
from utils import (
    detect_swing_highs, detect_swing_lows, identify_hh_hl, identify_ll_lh,
    get_session, is_high_volume_session, get_4h_candle_number, price_to_pips
)

logger = logging.getLogger(__name__)


@dataclass
class SRLevel:
    """Support/Resistance level with metadata."""
    price: float
    zone_type: ZoneType
    touches: int = 1
    htf_aligned: bool = False
    strength: float = 1.0  # Higher = stronger level

    @property
    def is_support(self) -> bool:
        return self.zone_type == ZoneType.SUPPORT

    @property
    def is_resistance(self) -> bool:
        return self.zone_type == ZoneType.RESISTANCE


@dataclass
class MarketState:
    """Complete market analysis snapshot."""
    trend: TrendType
    session: SessionType
    four_h_candle: int
    is_high_volume: bool
    current_price: float
    support_levels: List[SRLevel]
    resistance_levels: List[SRLevel]
    avg_body_size: float
    avg_volume: float
    current_volume_level: VolumeLevel
    volatility_pips: float
    swing_highs: List[float]
    swing_lows: List[float]


class MarketAnalyzer:
    """Analyzes market structure, S/R levels, trend, and conditions."""

    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def analyze(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None,
                current_price: Optional[float] = None) -> MarketState:
        """
        Full market analysis pipeline.
        Returns a MarketState with trend, S/R levels, session, volume info.
        """
        if df.empty or len(df) < 20:
            return self._empty_state(current_price or 0.0)

        price = current_price or float(df["Close"].iloc[-1])
        utc_hour = df.index[-1].hour if hasattr(df.index[-1], 'hour') else 12

        # Trend analysis
        highs = df["High"].values
        lows = df["Low"].values
        trend, swing_hi_vals, swing_lo_vals = self._detect_trend(highs, lows)

        # S/R detection pipeline
        support, resistance = self._detect_sr_levels(
            df, price, htf_df
        )

        # Session info
        session = get_session(utc_hour)
        four_h = get_4h_candle_number(utc_hour)
        high_vol = is_high_volume_session(session)

        # Volume analysis — XAU/USD on Twelve Data often has 0 volume
        # Default to MEDIUM when no real volume data exists
        avg_vol = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0
        current_vol = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
        if avg_vol == 0 and current_vol == 0:
            # No real volume data (common for forex/commodities)
            vol_level = VolumeLevel.MEDIUM
        else:
            vol_level = self._classify_volume(current_vol, avg_vol)

        # Body/volatility
        avg_body = float(df["body"].tail(20).mean()) if "body" in df else 1.0
        volatility = self._calculate_volatility(df)

        return MarketState(
            trend=trend,
            session=session,
            four_h_candle=four_h,
            is_high_volume=high_vol,
            current_price=price,
            support_levels=support,
            resistance_levels=resistance,
            avg_body_size=avg_body,
            avg_volume=avg_vol,
            current_volume_level=vol_level,
            volatility_pips=volatility,
            swing_highs=swing_hi_vals,
            swing_lows=swing_lo_vals,
        )

    # === TREND DETECTION ===

    def _detect_trend(self, highs: np.ndarray,
                      lows: np.ndarray) -> Tuple[TrendType, List[float], List[float]]:
        """
        Detect market trend using swing high/low structure.
        HH + HL = Uptrend, LL + LH = Downtrend, else Consolidation.
        """
        window = self.cfg.swing_lookback
        sh_indices = detect_swing_highs(highs, window)
        sl_indices = detect_swing_lows(lows, window)

        # Get last 4 swing values for structure analysis
        recent_highs = [float(highs[i]) for i in sh_indices[-4:]]
        recent_lows = [float(lows[i]) for i in sl_indices[-4:]]

        if identify_hh_hl(recent_highs, recent_lows):
            trend = TrendType.UPTREND
        elif identify_ll_lh(recent_highs, recent_lows):
            trend = TrendType.DOWNTREND
        else:
            trend = TrendType.CONSOLIDATION

        return trend, recent_highs, recent_lows

    # === S/R CHANNEL DETECTION (LonesomeTheBlue method) ===

    def _detect_sr_levels(self, df: pd.DataFrame, current_price: float,
                          htf_df: Optional[pd.DataFrame] = None
                          ) -> Tuple[List[SRLevel], List[SRLevel]]:
        """
        S/R Channel Detection (LonesomeTheBlue method).
        Replaces old swing-clustering with institutional-style channel zones:
        1. Detect pivot highs/lows with configurable lookback
        2. Group nearby pivots into channels within max width % of price range
        3. Score each channel by bar touches (strength)
        4. Greedy strongest-first selection up to max_channels
        5. Classify as support/resistance relative to current price
        + HTF zone stacking bonus
        """
        scan = min(self.cfg.sr_loopback, len(df))
        if scan < self.cfg.pivot_period * 2 + 1:
            return [], []

        recent = df.tail(scan)
        highs = recent["High"].values
        lows = recent["Low"].values
        closes = recent["Close"].values
        n = len(highs)

        # Step 1: Detect pivots
        pivot_vals = self._detect_pivots(highs, lows, self.cfg.pivot_period)

        if len(pivot_vals) == 0:
            return [], []

        # Calculate max channel width (% of 300-bar range)
        range_bars = min(300, n)
        prdhighest = float(highs[-range_bars:].max())
        prdlowest = float(lows[-range_bars:].min())
        cwidth = (prdhighest - prdlowest) * self.cfg.channel_width_pct / 100.0

        if cwidth <= 0:
            return [], []

        # Step 2: Build channel for each pivot
        supres = []
        for x in range(len(pivot_vals)):
            hi, lo, strength = self._build_sr_channel(x, pivot_vals, cwidth)
            supres.append([float(strength), float(hi), float(lo)])

        # Step 3: Add bar-touch strength (vectorized for speed)
        supres_arr = np.array(supres)  # shape: (num_pivots, 3) = [strength, hi, lo]
        for x in range(len(supres)):
            h_val = supres[x][1]
            l_val = supres[x][2]
            # Count bars whose high or low falls within [lo, hi] channel
            high_in = (highs <= h_val) & (highs >= l_val)
            low_in = (lows <= h_val) & (lows >= l_val)
            s = int(np.sum(high_in | low_in))
            supres[x][0] += s

        # Step 4: Greedy strongest-first selection
        sr_zones = []
        min_score = self.cfg.sr_min_strength * 20

        for _ in range(min(10, self.cfg.sr_max_channels)):
            best_score = -1.0
            best_idx = -1
            for y in range(len(supres)):
                if supres[y][0] > best_score and supres[y][0] >= min_score:
                    best_score = supres[y][0]
                    best_idx = y

            if best_idx < 0:
                break

            hh = supres[best_idx][1]
            ll = supres[best_idx][2]
            sr_zones.append((hh, ll, best_score))

            # Zero out overlapping channels
            for y in range(len(supres)):
                if (supres[y][1] <= hh and supres[y][1] >= ll) or \
                   (supres[y][2] <= hh and supres[y][2] >= ll):
                    supres[y][0] = -1

        # Step 5: Sort by strength and classify
        sr_zones.sort(key=lambda z: z[2], reverse=True)
        sr_zones = sr_zones[:self.cfg.sr_max_channels]

        # HTF zone stacking
        htf_levels = []
        if htf_df is not None and not htf_df.empty:
            htf_levels = self._get_htf_levels(htf_df)

        support = []
        resistance = []

        for hi, lo, score in sr_zones:
            mid = (hi + lo) / 2.0
            touches = max(1, int(score // 20))  # Approximate pivot count

            # Check HTF alignment
            htf_aligned = any(
                abs(mid - hl) <= self.cfg.sr_zone_tolerance
                for hl in htf_levels
            )
            strength = score + (40.0 if htf_aligned else 0.0)

            if hi < current_price and lo < current_price:
                support.append(SRLevel(
                    price=mid,
                    zone_type=ZoneType.SUPPORT,
                    touches=touches,
                    htf_aligned=htf_aligned,
                    strength=strength,
                ))
            elif hi > current_price and lo > current_price:
                resistance.append(SRLevel(
                    price=mid,
                    zone_type=ZoneType.RESISTANCE,
                    touches=touches,
                    htf_aligned=htf_aligned,
                    strength=strength,
                ))
            else:
                # Price inside zone — classify by midpoint
                if mid > current_price:
                    resistance.append(SRLevel(
                        price=mid,
                        zone_type=ZoneType.RESISTANCE,
                        touches=touches,
                        htf_aligned=htf_aligned,
                        strength=strength,
                    ))
                else:
                    support.append(SRLevel(
                        price=mid,
                        zone_type=ZoneType.SUPPORT,
                        touches=touches,
                        htf_aligned=htf_aligned,
                        strength=strength,
                    ))

        support.sort(key=lambda x: x.price, reverse=True)   # Nearest first
        resistance.sort(key=lambda x: x.price)               # Nearest first

        return support, resistance

    @staticmethod
    def _detect_pivots(highs: np.ndarray, lows: np.ndarray,
                       prd: int) -> List[float]:
        """
        Detect pivot high/low values (equivalent to ta.pivothigh/ta.pivotlow).
        A pivot high at bar i requires high[i] to be highest of bars [i-prd, i+prd].
        A pivot low at bar i requires low[i] to be lowest of bars [i-prd, i+prd].
        """
        n = len(highs)
        pivots = []
        for i in range(prd, n - prd):
            window_h = highs[i - prd:i + prd + 1]
            if highs[i] >= window_h.max():
                pivots.append(float(highs[i]))

            window_l = lows[i - prd:i + prd + 1]
            if lows[i] <= window_l.min():
                pivots.append(float(lows[i]))

        return pivots

    @staticmethod
    def _build_sr_channel(pivot_idx: int, pivot_vals: List[float],
                          max_channel_width: float) -> Tuple[float, float, int]:
        """
        Build an S/R channel around a pivot by absorbing nearby pivots.
        Returns (channel_hi, channel_lo, pivot_count_score).
        """
        lo = pivot_vals[pivot_idx]
        hi = lo
        numpp = 0

        for y in range(len(pivot_vals)):
            cpp = pivot_vals[y]
            wdth = (hi - cpp) if cpp <= hi else (cpp - lo)
            if wdth <= max_channel_width:
                if cpp <= hi:
                    lo = min(lo, cpp)
                else:
                    hi = max(hi, cpp)
                numpp += 20

        return hi, lo, numpp

    def _get_htf_levels(self, htf_df: pd.DataFrame) -> List[float]:
        """Extract key levels from higher timeframe data."""
        if htf_df.empty:
            return []

        highs = htf_df["High"].values
        lows = htf_df["Low"].values

        sh_idx = detect_swing_highs(highs, 3)
        sl_idx = detect_swing_lows(lows, 3)

        levels = [float(highs[i]) for i in sh_idx[-10:]]
        levels += [float(lows[i]) for i in sl_idx[-10:]]
        return levels

    # === VOLUME / VOLATILITY ===

    def _classify_volume(self, current: float, avg: float) -> VolumeLevel:
        if avg <= 0:
            return VolumeLevel.MEDIUM
        ratio = current / avg
        if ratio > 1.5:
            return VolumeLevel.HIGH
        elif ratio > 0.7:
            return VolumeLevel.MEDIUM
        return VolumeLevel.LOW

    def _calculate_volatility(self, df: pd.DataFrame,
                              window: int = 20) -> float:
        """Average True Range in pips over recent bars."""
        if len(df) < window + 1:
            return 0.0

        recent = df.tail(window + 1)
        tr_values = []
        for i in range(1, len(recent)):
            high = recent["High"].iloc[i]
            low = recent["Low"].iloc[i]
            prev_close = recent["Close"].iloc[i - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        atr = np.mean(tr_values)
        return price_to_pips(atr)

    # === HELPERS ===

    def get_nearest_support(self, levels: List[SRLevel],
                            price: float) -> Optional[SRLevel]:
        """Get nearest support level below current price."""
        below = [l for l in levels if l.price < price]
        if not below:
            return None
        return max(below, key=lambda x: x.price)

    def get_nearest_resistance(self, levels: List[SRLevel],
                               price: float) -> Optional[SRLevel]:
        """Get nearest resistance level above current price."""
        above = [l for l in levels if l.price > price]
        if not above:
            return None
        return min(above, key=lambda x: x.price)

    def get_room_to_opposing(self, direction: str, price: float,
                             support: List[SRLevel],
                             resistance: List[SRLevel]) -> float:
        """
        Get distance in pips to nearest opposing S/R level.
        For BUY: room to nearest resistance.
        For SELL: room to nearest support.
        """
        if direction == "BUY":
            nearest = self.get_nearest_resistance(resistance, price)
        else:
            nearest = self.get_nearest_support(support, price)

        if nearest is None:
            return 999.0  # No opposing level found
        return price_to_pips(abs(nearest.price - price))

    def _empty_state(self, price: float) -> MarketState:
        return MarketState(
            trend=TrendType.CONSOLIDATION,
            session=SessionType.OFF_SESSION,
            four_h_candle=0,
            is_high_volume=False,
            current_price=price,
            support_levels=[],
            resistance_levels=[],
            avg_body_size=1.0,
            avg_volume=0.0,
            current_volume_level=VolumeLevel.LOW,
            volatility_pips=0.0,
            swing_highs=[],
            swing_lows=[],
        )
