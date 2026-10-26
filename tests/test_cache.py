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
