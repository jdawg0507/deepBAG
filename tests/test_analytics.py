"""
Unit tests for deterministic analytics formulas (Milestone 2 & 4).
"""
import pytest
from core.analytics import (
    calculate_ts_percent,
    calculate_efg_percent,
    calculate_on_off_differential,
    calculate_halftime_adjustments,
    generate_rapm_metadata_bundle,
)

def test_ts_percent_calculation():
    # Example: 30 PTS on 20 FGA, 5 FTA
    # TS% = 30 / (2 * (20 + 0.44 * 5)) = 30 / (2 * 22.2) = 30 / 44.4 = 0.6756... -> 67.6%
    raw, formatted = calculate_ts_percent(pts=30, fga=20, fta=5)
    assert abs(raw - 0.67567) < 0.001
    assert formatted == "67.6%"

def test_efg_percent_calculation():
    # Example: 10 FGM, 4 FG3M on 20 FGA
    # eFG% = (10 + 0.5 * 4) / 20 = 12 / 20 = 0.60 -> 60.0%
    raw, formatted = calculate_efg_percent(fgm=10, fg3m=4, fga=20)
    assert raw == 0.6
    assert formatted == "60.0%"

def test_on_off_differential():
    result = calculate_on_off_differential(
        on_court_net_rating=8.5,
        off_court_net_rating=-2.1,
        min_minutes=200,
        actual_minutes=250
    )
    assert result["net_differential"] == 10.6
    assert result["formatted_value"] == "+10.6"
    assert result["qualified_sample"] is True
    assert "confounding_warning" in result

def test_halftime_adjustments():
    result = calculate_halftime_adjustments(
        first_half_net_rating=-4.0,
        second_half_net_rating=6.5
    )
    assert result["adjustment_delta"] == 10.5
    assert result["formatted_delta"] == "+10.5"

def test_rapm_metadata_bundle():
    bundle = generate_rapm_metadata_bundle(
        estimator="raw_on_off",
        sample_minutes=1450,
        games_played=55,
        is_predictive=False,
        luck_adjusted=False
    )
    assert bundle["estimator"] == "raw_on_off"
    assert bundle["label"] == "Descriptive"
    assert bundle["prior_shrinkage"] == "none"
    assert "Descriptive" in bundle["interpretation_note"]
