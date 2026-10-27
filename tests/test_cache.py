"""
Unit tests for distributed RedisQueryCache with fakeredis fallback.
"""
import pytest
import time
from core.cache import RedisQueryCache

def test_cache_sync_get_set():
    cache = RedisQueryCache(default_ttl=3600)
    query = "LeBron James stats last game"
    response = '{"points": 28, "rebounds": 8, "assists": 12}'

    # Initial miss
    assert cache.get(query) is None
    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0

    # Set response
    assert cache.set(query, response) is True
    assert cache.get_stats()["sets"] == 1

    # Cache hit
    cached = cache.get(query)
    assert cached == response
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["hit_ratio"] == 0.5

def test_cache_key_normalization():
    cache = RedisQueryCache(default_ttl=3600)
    q1 = "   STEPHEN CURRY 3-POINTERS   "
    q2 = "stephen   curry   3-pointers"
    
    assert cache.set(q1, "Curry response") is True
    assert cache.get(q2) == "Curry response"

@pytest.mark.asyncio
async def test_cache_async_get_set():
    cache = RedisQueryCache(default_ttl=3600)
    query = "Nikola Jokic triple doubles"
    response = "Jokic had 25 triple doubles"

    assert await cache.aget(query) is None
    assert await cache.aset(query, response) is True
    assert await cache.aget(query) == response

def test_cache_ttl_expiration():
    cache = RedisQueryCache(default_ttl=1)
    query = "Quick expiring query"
    cache.set(query, "Temporary data", ttl=1)
    assert cache.get(query) == "Temporary data"
    time.sleep(1.2)
    assert cache.get(query) is None

def test_canonical_key_generation():
    key1 = RedisQueryCache.make_canonical_key(
        endpoint="PlayerGameLog",
        params={"PlayerID": 2544, "Season": "2023-24", "SeasonType": "Regular Season"},
    )
    # Different param order should produce the exact same key
    key2 = RedisQueryCache.make_canonical_key(
        endpoint="playergamelog",
        params={"SeasonType": "Regular Season", "Season": "2023-24", "PlayerID": "2544"},
    )
    assert key1 == key2
    assert key1.startswith("deepbag:api:playergamelog:")

    # Different param value should produce different key
    key3 = RedisQueryCache.make_canonical_key(
        endpoint="PlayerGameLog",
        params={"PlayerID": 201939, "Season": "2023-24"},
    )
    assert key1 != key3

def test_season_state_ttl():
    # Historical seasons should get 30-day TTL (2592000s)
    assert RedisQueryCache.get_season_ttl("2022-23") == 30 * 86400
    assert RedisQueryCache.get_season_ttl("2015-16") == 30 * 86400

    # Current / active seasons should get 12-hour TTL (43200s)
    assert RedisQueryCache.get_season_ttl("2024-25") == 12 * 3600
    assert RedisQueryCache.get_season_ttl("2025-26") == 12 * 3600
    assert RedisQueryCache.get_season_ttl(None) == 12 * 3600

def test_canonical_api_data_caching():
    cache = RedisQueryCache()
    endpoint = "PlayerGameLog"
    params = {"PlayerID": 2544, "Season": "2022-23"}
    data = [{"GAME_DATE": "2023-01-15", "PTS": 35}]

    # Initial miss
    assert cache.get_api_data(endpoint, params) is None

    # Set API data
    assert cache.set_api_data(endpoint, params, data) is True

    # Cache hit
    cached = cache.get_api_data(endpoint, params)
    assert cached == data

@pytest.mark.asyncio
async def test_async_canonical_api_data_caching():
    cache = RedisQueryCache()
    endpoint = "CommonPlayerInfo"
    params = {"PlayerID": 201939}
    data = {"DISPLAY_FIRST_LAST": "Stephen Curry", "TEAM_NAME": "Warriors"}

    assert await cache.aget_api_data(endpoint, params) is None
    assert await cache.aset_api_data(endpoint, params, data) is True
    assert await cache.aget_api_data(endpoint, params) == data


def test_negative_caching_exclusion_for_fail_closed():
    """Verify that transient API timeout and fail-closed responses are not saved to cache"""
    from unittest.mock import MagicMock
    from nba_nlp_analyzer_modular import NBAQueryAnalyzer
    from core.schemas import IntentContract, QuestionType
    
    analyzer = NBAQueryAnalyzer.__new__(NBAQueryAnalyzer)
    analyzer.cache = RedisQueryCache()
    analyzer.synthesizer = MagicMock()
    analyzer.caesura_planner = MagicMock()
    analyzer.client = MagicMock()
    
    analyzer.caesura_planner.extract_intent.return_value = IntentContract(
        question_type=QuestionType.SINGLE,
        entities=[],
        season="2023-24",
        interpretation="Test"
    )
    analyzer.caesura_planner.build_logical_plan.return_value = MagicMock(steps=[])
    analyzer.caesura_planner.validate_plan_schema.return_value = (True, [])
    from core.evidence import EvidenceTable
    analyzer.caesura_planner.execute_plan.return_value = (EvidenceTable(), {})
    
    # Synthesizer yields Data Unavailable fail-closed payload
    fail_payload = (
        "📋 **Scope Receipt:** Season: **2023-24**\n\n"
        "### ⚠️ Data Unavailable (Fail-Closed Enforcement)\n"
        "deepBAG was unable to retrieve verified data from the NBA API..."
    )
    analyzer.synthesizer.synthesize_research.return_value = [
        {"type": "final_answer", "content": fail_payload}
    ]
    
    query = "Steph Curry points tonight"
    events = list(analyzer._process_research_flow(query))
    
    # Must NOT be stored in cache
    assert analyzer.cache.get(query) is None

