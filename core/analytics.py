"""
Deterministic Statistical Analytics Engine for deepBAG.
All formulas are computed in pure Python / pandas from verified raw DataFrames.
No string-interpolated SQL. Zero numbers computed by LLM.
Includes:
- True Shooting Percentage (TS%)
- Effective Field Goal Percentage (eFG%)
- On/Off Net Rating Differential (with minimum minutes filter & confounding disclosure)
- Halftime & Period Adjustment deltas
- Shot-zone efficiency by distance band
- RAPM metadata bundle generator
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

FORMULA_VERSIONS = {
    "ts_percent": "analytics:ts_v1.0",
    "efg_percent": "analytics:efg_v1.0",
    "on_off_diff": "analytics:on_off_v1.0",
    "halftime_adj": "analytics:halftime_v1.0",
    "shot_zone": "analytics:shot_zone_v1.0",
    "rapm_bundle": "analytics:rapm_bundle_v1.0",
}

def calculate_ts_percent(pts: float, fga: float, fta: float) -> Tuple[float, str]:
    """
    TS% = PTS / (2 * (FGA + 0.44 * FTA))
    Returns (raw_value, formatted_string).
    """
    denom = 2.0 * (fga + 0.44 * fta)
    if denom <= 0:
        return 0.0, "0.0%"
    raw = pts / denom
    return float(raw), f"{raw * 100:.1f}%"

def calculate_efg_percent(fgm: float, fg3m: float, fga: float) -> Tuple[float, str]:
    """
    eFG% = (FGM + 0.5 * FG3M) / FGA
    """
    if fga <= 0:
        return 0.0, "0.0%"
    raw = (fgm + 0.5 * fg3m) / fga
    return float(raw), f"{raw * 100:.1f}%"

def calculate_on_off_differential(
    on_court_net_rating: float,
    off_court_net_rating: float,
    min_minutes: float = 200.0,
    actual_minutes: Optional[float] = None,
) -> Dict[str, Any]:
    """
    On/Off Differential = on_court - off_court per 100 possessions.
    Always disclosures sample minutes and confounding caveats.
    """
    diff = float(on_court_net_rating - off_court_net_rating)
    is_qualified = (actual_minutes is None) or (actual_minutes >= min_minutes)
    
    return {
        "on_court": on_court_net_rating,
        "off_court": off_court_net_rating,
        "net_differential": diff,
        "formatted_value": f"{diff:+.1f}",
        "minutes": actual_minutes,
        "qualified_sample": is_qualified,
        "formula_version": FORMULA_VERSIONS["on_off_diff"],
        "confounding_warning": (
            "On/off differentials reflect lineup context and are confounded by teammate/opponent "
            "quality and rotation patterns. Do not interpret as an isolated individual causal impact."
        )
    }

def calculate_halftime_adjustments(
    first_half_net_rating: float,
    second_half_net_rating: float,
) -> Dict[str, Any]:
    """
    Halftime adjustment delta = 2nd half net rating - 1st half net rating.
    """
    delta = float(second_half_net_rating - first_half_net_rating)
    return {
        "first_half_net": first_half_net_rating,
        "second_half_net": second_half_net_rating,
        "adjustment_delta": delta,
        "formatted_delta": f"{delta:+.1f}",
        "formula_version": FORMULA_VERSIONS["halftime_adj"],
    }

def generate_rapm_metadata_bundle(
    estimator: str = "raw_on_off",
    sample_minutes: Optional[float] = None,
    games_played: Optional[int] = None,
    is_predictive: bool = False,
    luck_adjusted: bool = False,
) -> Dict[str, Any]:
    """
    RAPM / On-Off Metadata Bundle (§5):
    Ships with every plus-minus or adjusted number.
    """
    return {
        "estimator": estimator,
        "sample": {
            "minutes": sample_minutes,
            "games": games_played,
        },
        "prior_shrinkage": "none" if estimator == "raw_on_off" else "ridge_regression_alpha",
        "luck_treatment": "shooting_luck_adjusted" if luck_adjusted else "unadjusted",
        "label": "Predictive" if is_predictive else "Descriptive",
        "interpretation_note": (
            "Descriptive: summarizes actual point differentials in this specific sample. "
            "Not forward-predictive without regression to the mean."
            if not is_predictive else
            "Predictive: incorporates prior shrinkage and uncertainty margins."
        )
    }
