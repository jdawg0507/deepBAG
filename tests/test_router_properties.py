"""
Hypothesis Property-Based Testing Suite for deepBAG Semantic Router.
Tests core invariants across randomized player, season, and comparison draws:
1. Identity Invariant: (player, season) strictly resolves to matching player_id and season.
2. Cardinality Invariant: N distinct players emit N calls or 1 filtered league call.
3. Era Invariant: Seasons < min_season resolve to HISTORICAL_DB or UNSUPPORTED_ERA.
4. No-Silent-Default Invariant: Missing season prompts must record an explicit assumption in ScopeReceipt.
"""
import pytest
from hypothesis import given, settings, strategies as st
from typing import Tuple, Dict, Any, List

from nba_api.stats.static import players
from catalogue.schema import EndpointSpec, PlayerMeta, resolve_data_source, DataSource, CareerBoundsError
from router.scope_receipt import TemporalResolver, ScopeReceipt
from entities.resolver import EntityResolver, normalize_text
from router.verifier import CodeVerifier

# Sample of real players with known career bounds
SAMPLE_PLAYERS = [
    PlayerMeta(id=1630162, full_name="Anthony Edwards", from_year=2020, to_year=2024),
    PlayerMeta(id=2544, full_name="LeBron James", from_year=2003, to_year=2024),
    PlayerMeta(id=201939, full_name="Stephen Curry", from_year=2009, to_year=2024),
    PlayerMeta(id=203999, full_name="Nikola Jokic", from_year=2015, to_year=2024),
    PlayerMeta(id=1641705, full_name="Victor Wembanyama", from_year=2023, to_year=2024),
    PlayerMeta(id=893, full_name="Michael Jordan", from_year=1984, to_year=2003),
    PlayerMeta(id=201566, full_name="Russell Westbrook", from_year=2008, to_year=2024),
    PlayerMeta(id=201142, full_name="Kevin Durant", from_year=2007, to_year=2024),
    PlayerMeta(id=203507, full_name="Giannis Antetokounmpo", from_year=2013, to_year=2024),
    PlayerMeta(id=1628369, full_name="Jayson Tatum", from_year=2017, to_year=2024),
]

@st.composite
def valid_player_and_season(draw) -> Tuple[PlayerMeta, str]:
    p = draw(st.sampled_from(SAMPLE_PLAYERS))
    yr = draw(st.integers(min_value=p.from_year, max_value=min(p.to_year, 2024)))
    season_str = f"{yr}-{str(yr + 1)[-2:]}"
    return p, season_str

@st.composite
def pre_tracking_era_season(draw) -> str:
    yr = draw(st.integers(min_value=1950, max_value=1995))
    return f"{yr}-{str(yr + 1)[-2:]}"

# ----------------------------------------------------------------------
# 1. Identity Invariant
# ----------------------------------------------------------------------
@settings(max_examples=50)
@given(valid_player_and_season())
def test_identity_invariant(player_season_pair):
    player, season = player_season_pair
    resolver = EntityResolver()
    resolved = resolver.resolve(player.full_name, season_hint=season)
    
    assert resolved is not None, f"Failed to resolve {player.full_name}"
    assert resolved.id == player.id, f"Player ID mismatch: got {resolved.id}, expected {player.id}"
    
    # Temporal parsing invariant
    query = f"How many points did {player.full_name} score in {season}?"
    temp_info = TemporalResolver.resolve_temporal_ref(query, player_from_year=player.from_year)
    assert temp_info["season"] == season, f"Season mismatch: got {temp_info['season']}, expected {season}"
    assert temp_info["is_assumed"] is False

# ----------------------------------------------------------------------
# 2. Cardinality Invariant
# ----------------------------------------------------------------------
@settings(max_examples=30)
@given(st.lists(st.sampled_from(SAMPLE_PLAYERS), min_size=2, max_size=4, unique_by=lambda p: p.id))
def test_cardinality_invariant(player_list):
    names = [p.full_name for p in player_list]
    resolver = EntityResolver()
    
    resolved_ids = []
    for name in names:
        res = resolver.resolve(name)
        if res:
            resolved_ids.append(res.id)
            
    assert len(resolved_ids) == len(player_list), "Entity resolver collapsed or dropped players in comparison"
    assert len(set(resolved_ids)) == len(player_list), "Duplicate IDs detected across distinct players"

# ----------------------------------------------------------------------
# 3. Era Invariant (No False API Calls on Pre-Tracking Era)
# ----------------------------------------------------------------------
@settings(max_examples=30)
@given(pre_tracking_era_season())
def test_era_invariant(historical_season):
    # Tracking endpoint ShotChartDetail has min_season 1996-97 with no historical box score equivalent
    shot_chart_spec = EndpointSpec(
        endpoint_name="ShotChartDetail",
        min_season="1996-97",
        entity_grain="player",
        capability_card="Shot chart tracking coordinates",
        has_historical_equivalent=False
    )
    
    ds = resolve_data_source(player=None, season=historical_season, endpoint=shot_chart_spec)
    assert ds == DataSource.UNSUPPORTED_ERA, f"Expected UNSUPPORTED_ERA for pre-1996 shot chart, got {ds}"

    # Box score endpoint PlayerGameLog has min_season 1946-47 with historical equivalent
    box_score_spec = EndpointSpec(
        endpoint_name="PlayerGameLog",
        min_season="1996-97",
        entity_grain="player",
        capability_card="Game log box scores",
        has_historical_equivalent=True
    )
    ds_box = resolve_data_source(player=None, season=historical_season, endpoint=box_score_spec)
    assert ds_box == DataSource.HISTORICAL_DB, f"Expected HISTORICAL_DB for pre-1996 box scores, got {ds_box}"

# ----------------------------------------------------------------------
# 4. No-Silent-Default Invariant
# ----------------------------------------------------------------------
@settings(max_examples=30)
@given(st.sampled_from(SAMPLE_PLAYERS))
def test_no_silent_default_invariant(player):
    # Query without explicit season
    query = f"What are {player.full_name}'s points per game?"
    temp_info = TemporalResolver.resolve_temporal_ref(query, player_from_year=player.from_year)
    
    assert temp_info["is_assumed"] is True, "Season should be flagged as assumed when not specified"
    assert temp_info["assumption_text"] is not None, "Missing explanation for assumed season"
    assert "current season" in temp_info["assumption_text"].lower()

    # Scope receipt must capture assumption
    receipt = ScopeReceipt(
        season=temp_info["season"],
        entities=[{"name": player.full_name, "id": player.id}],
        assumptions=[temp_info["assumption_text"]]
    )
    assert len(receipt.assumptions) > 0
    assert "Assumed" in receipt.format_markdown()
