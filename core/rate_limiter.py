"""
Distributed Atomic Sliding-Window Token Bucket Rate Limiter using Redis Lua scripts.
Protects upstream NBA API (stats.nba.com) from rate limits and HTTP 429 / ReadTimeout bans.
"""
import os
import time
import asyncio
import logging
from typing import Optional, Tuple

try:
    import redis
    import redis.asyncio as aioredis
except ImportError:
    redis = None
    aioredis = None

try:
    import fakeredis
    import fakeredis.aioredis as fake_aioredis
except ImportError:
    fakeredis = None
    fake_aioredis = None

logger = logging.getLogger("deepbag.rate_limiter")

LUA_TOKEN_BUCKET_SCRIPT = """
-- KEYS[1]: rate limit key
-- ARGV[1]: bucket capacity (max tokens)
-- ARGV[2]: fill rate (tokens per second)
-- ARGV[3]: current timestamp in seconds (float)
-- ARGV[4]: requested tokens

local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local fill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local data = redis.call("HMGET", key, "tokens", "last_update")
local tokens = tonumber(data[1])
local last_update = tonumber(data[2])

if not tokens then
    tokens = capacity
    last_update = now
else
    local delta = math.max(0, now - last_update)
    tokens = math.min(capacity, tokens + delta * fill_rate)
    last_update = now
end

if tokens >= requested then
    tokens = tokens - requested
    redis.call("HMSET", key, "tokens", tokens, "last_update", last_update)
    local ttl = math.ceil(capacity / fill_rate) * 2
    redis.call("EXPIRE", key, math.max(60, ttl))
    return {1, tokens, 0} -- {success (1/0), remaining_tokens, wait_time_seconds}
else
    local wait_time = (requested - tokens) / fill_rate
    redis.call("HMSET", key, "tokens", tokens, "last_update", last_update)
    return {0, tokens, wait_time}
end
"""

class RateLimitExceededException(Exception):
    """Raised when request cannot acquire an NBA API rate limit token within timeout."""
    pass

class RedisSlidingWindowRateLimiter:
    """
    Distributed token bucket rate limiter backed by an atomic Redis Lua script.
    Default: 20 requests per 60-second window (0.333 tokens/sec) with a burst capacity of 5.
    """
    def __init__(
        self,
        redis_url: Optional[str] = None,
        capacity: float = 5.0,
        requests_per_minute: float = 20.0,
        key: str = "deepbag:ratelimit:nba_api",
    ):
        self.redis_url = redis_url or os.environ.get("REDIS_URL") or os.environ.get("UPSTASH_REDIS_URL")
        self.capacity = float(capacity)
        self.fill_rate = float(requests_per_minute) / 60.0  # tokens per second
        self.key = key
        
        self._sync_client = self._init_sync_client()
        self._sync_script = self._sync_client.register_script(LUA_TOKEN_BUCKET_SCRIPT)
        self._async_client = None
        self._async_script = None

    def _init_sync_client(self):
        if self.redis_url:
            try:
                client = redis.from_url(self.redis_url, decode_responses=True)
                client.ping()
                return client
            except Exception as e:
                logger.warning(f"Rate limiter falling back to fakeredis: {e}")
        return fakeredis.FakeStrictRedis(decode_responses=True)

    async def _get_async_client(self):
        if self._async_client is not None:
            return self._async_client, self._async_script
            
        if self.redis_url:
            try:
                client = aioredis.from_url(self.redis_url, decode_responses=True)
                await client.ping()
                self._async_client = client
                self._async_script = self._async_client.register_script(LUA_TOKEN_BUCKET_SCRIPT)
                return self._async_client, self._async_script
            except Exception as e:
                logger.warning(f"Async rate limiter falling back to fake_aioredis: {e}")

        self._async_client = fake_aioredis.FakeRedis(decode_responses=True)
        self._async_script = self._async_client.register_script(LUA_TOKEN_BUCKET_SCRIPT)
        return self._async_client, self._async_script

    def acquire_sync(self, tokens: int = 1, timeout: float = 10.0) -> bool:
        """
        Synchronously acquire token(s) from bucket. Queues and sleeps if necessary up to timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            now = time.time()
            result = self._sync_script(
                keys=[self.key],
                args=[self.capacity, self.fill_rate, now, tokens],
            )
            success, remaining, wait_time = result[0], float(result[1]), float(result[2])
            if success == 1:
                return True
            
            sleep_duration = min(wait_time, deadline - time.time())
            if sleep_duration <= 0:
                break
            time.sleep(sleep_duration)
            
        raise RateLimitExceededException(
            f"Rate limit exceeded: could not acquire {tokens} NBA API token(s) within {timeout}s timeout."
        )

    async def acquire_async(self, tokens: int = 1, timeout: float = 10.0) -> bool:
        """
        Asynchronously acquire token(s). Awaits sleep if necessary up to timeout.
        """
        client, script = await self._get_async_client()
        deadline = time.time() + timeout
        while time.time() < deadline:
            now = time.time()
            result = await script(
                keys=[self.key],
                args=[self.capacity, self.fill_rate, now, tokens],
            )
            success, remaining, wait_time = result[0], float(result[1]), float(result[2])
            if success == 1:
                return True
            
            sleep_duration = min(wait_time, deadline - time.time())
            if sleep_duration <= 0:
                break
            await asyncio.sleep(sleep_duration)
            
        raise RateLimitExceededException(
            f"Rate limit exceeded: could not acquire {tokens} NBA API token(s) within {timeout}s timeout."
        )
