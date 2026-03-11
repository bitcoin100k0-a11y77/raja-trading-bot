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

    # === S/R DETECTION PIPELINE (Raja's 5 steps) ===

    def _detect_sr_levels(self, df: pd.DataFrame, current_price: float,
                          htf_df: Optional[pd.DataFrame] = None
                          ) -> Tuple[List[SRLevel], List[SRLevel]]:
        """
        Raja Banks S/R Detection Pipeline:
        1. Find swing highs/lows
        2. Cluster nearby swings ($3 tolerance)
        3. Filter by touch count
        4. Enforce minimum spacing
        5. Cap max levels per side
        + HTF zone stacking bonus
        """
        scan = min(self.cfg.sr_scan_range, len(df))
        recent = df.tail(scan)
        highs = recent["High"].values
        lows = recent["Low"].values

        # Step 1: Find swing points
        sh_idx = detect_swing_highs(highs, self.cfg.swing_lookback)
        sl_idx = detect_swing_lows(lows, self.cfg.swing_lookback)

        swing_highs = [float(highs[i]) for i in sh_idx]
        swing_lows = [float(lows[i]) for i in sl_idx]

        all_swings = swing_highs + swing_lows

        if not all_swings:
            return [], []

        # Step 2: Cluster nearby swings
        clusters = self._cluster_levels(all_swings, self.cfg.sr_zone_tolerance)

        # Step 3: Filter by touch count
        filtered = [c for c in clusters if c["touches"] >= self.cfg.min_touches]

        # HTF zone stacking
        htf_levels = []
        if htf_df is not None and not htf_df.empty:
            htf_levels = self._get_htf_levels(htf_df)

        # Split into support / resistance relative to current price
        support = []
        resistance = []

        for c in filtered:
            level_price = c["price"]
            touches = c["touches"]

            # Check HTF alignment
            htf_aligned = any(
                abs(level_price - hl) <= self.cfg.sr_zone_tolerance
                for hl in htf_levels
            )

            strength = touches + (2.0 if htf_aligned else 0.0)

            if level_price < current_price:
                support.append(SRLevel(
                    price=level_price,
                    zone_type=ZoneType.SUPPORT,
                    touches=touches,
                    htf_aligned=htf_aligned,
                    strength=strength,
                ))
            else:
                resistance.append(SRLevel(
                    price=level_price,
                    zone_type=ZoneType.RESISTANCE,
                    touches=touches,
                    htf_aligned=htf_aligned,
                    strength=strength,
                ))

        # Step 4: Enforce minimum spacing
        support = self._enforce_spacing(support, ascending=False)
        resistance = self._enforce_spacing(resistance, ascending=True)

        # Step 5: Cap max levels per side
        support.sort(key=lambda x: x.price, reverse=True)  # Nearest first
        resistance.sort(key=lambda x: x.price)  # Nearest first
        support = support[:self.cfg.max_levels_per_side]
        resistance = resistance[:self.cfg.max_levels_per_side]

        return support, resistance

    def _cluster_levels(self, prices: List[float],
                        tolerance: float) -> List[Dict]:
        """
        Cluster nearby price levels within $tolerance.
        Returns list of {price: avg_price, touches: count}.
        """
        if not prices:
            return []

        sorted_prices = sorted(prices)
        clusters = []
        current_cluster = [sorted_prices[0]]

        for p in sorted_prices[1:]:
            if p - current_cluster[-1] <= tolerance:
                current_cluster.append(p)
            else:
                clusters.append({
                    "price": round(np.mean(current_cluster), 2),
                    "touches": len(current_cluster),
                })
                current_cluster = [p]

        # Don't forget last cluster
        clusters.append({
            "price": round(np.mean(current_cluster), 2),
            "touches": len(current_cluster),
        })

        return clusters

    def _enforce_spacing(self, levels: List[SRLevel],
                         ascending: bool = True) -> List[SRLevel]:
        """Remove levels that are too close together, keeping stronger ones."""
        if len(levels) <= 1:
            return levels

        # Sort by strength (strongest first), then filter by spacing
        levels.sort(key=lambda x: x.strength, reverse=True)
        filtered = [levels[0]]

        for lvl in levels[1:]:
            too_close = any(
                abs(lvl.price - kept.price) < self.cfg.min_level_spacing
                for kept in filtered
            )
            if not too_close:
                filtered.append(lvl)

        return filtered

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
