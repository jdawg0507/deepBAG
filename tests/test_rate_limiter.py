"""
Unit tests for atomic sliding-window token bucket rate limiter.
"""
import pytest
import time
from core.rate_limiter import RedisSlidingWindowRateLimiter, RateLimitExceededException

def test_rate_limiter_burst_capacity():
    limiter = RedisSlidingWindowRateLimiter(
        capacity=3.0,
        requests_per_minute=60.0,  # 1 token/sec
        key="deepbag:test:ratelimit:burst",
    )
    # Burst 3 tokens sequentially
    assert limiter.acquire_sync(tokens=1, timeout=0.5) is True
    assert limiter.acquire_sync(tokens=1, timeout=0.5) is True
    assert limiter.acquire_sync(tokens=1, timeout=0.5) is True

    # 4th token should fail if timeout is near zero
    with pytest.raises(RateLimitExceededException):
        limiter.acquire_sync(tokens=1, timeout=0.1)

def test_rate_limiter_refill():
    limiter = RedisSlidingWindowRateLimiter(
        capacity=2.0,
        requests_per_minute=120.0,  # 2 tokens/sec
        key="deepbag:test:ratelimit:refill",
    )
    assert limiter.acquire_sync(tokens=2, timeout=0.5) is True
    
    # Wait 0.6 seconds to replenish at least 1 token
    time.sleep(0.6)
    assert limiter.acquire_sync(tokens=1, timeout=0.5) is True

@pytest.mark.asyncio
async def test_rate_limiter_async_acquire():
    limiter = RedisSlidingWindowRateLimiter(
        capacity=2.0,
        requests_per_minute=60.0,
        key="deepbag:test:ratelimit:async",
    )
    assert await limiter.acquire_async(tokens=1, timeout=1.0) is True
    assert await limiter.acquire_async(tokens=1, timeout=1.0) is True
