"""
Distributed Redis & Upstash caching layer with automatic fakeredis fallback.
Replaces legacy local SQLite query_cache.db.
"""
import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, Any

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

logger = logging.getLogger("deepbag.cache")

class RedisQueryCache:
    """
    Production-grade distributed query cache for deepBAG.
    Supports Upstash Redis, Redis Cloud, local Redis, or in-memory fakeredis fallback.
    """
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 86400,
        key_prefix: str = "deepbag:cache:",
    ):
        self.redis_url = redis_url or os.environ.get("REDIS_URL") or os.environ.get("UPSTASH_REDIS_URL")
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.stats = {"hits": 0, "misses": 0, "sets": 0}
        
        # Initialize sync client
        self._sync_client = self._init_sync_client()
        # Async client is initialized lazily
        self._async_client = None

    def _init_sync_client(self):
        if self.redis_url:
            try:
                client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )
                client.ping()
                logger.info(f"Connected to remote Redis instance: {self.redis_url.split('@')[-1]}")
                return client
            except Exception as e:
                logger.warning(f"Failed to connect to Redis at {self.redis_url}: {e}. Falling back to fakeredis.")
        
        if fakeredis:
            logger.info("Initializing in-memory fakeredis cache client.")
            return fakeredis.FakeStrictRedis(decode_responses=True)
        else:
            raise RuntimeError("Neither live Redis connection nor fakeredis is available.")

    async def _get_async_client(self):
        if self._async_client is not None:
            return self._async_client
            
        if self.redis_url:
            try:
                client = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )
                await client.ping()
                self._async_client = client
                return self._async_client
            except Exception as e:
                logger.warning(f"Async Redis connection failed: {e}. Using fake_aioredis.")

        if fake_aioredis:
            self._async_client = fake_aioredis.FakeRedis(decode_responses=True)
            return self._async_client
        else:
            raise RuntimeError("Neither async Redis nor fake_aioredis is available.")

    def _generate_key(self, query: str) -> str:
        """Create a deterministic SHA-256 cache key from normalized query text"""
        normalized = " ".join(query.strip().lower().split())
        query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}{query_hash}"

    @staticmethod
    def get_season_ttl(season_str: Optional[str] = None) -> int:
        """
        Calculates TTL based on whether the season is completed/historical or active.
        Historical seasons (<= 2023-24): 30 days (immutable).
        Active/current seasons (>= 2024-25 or None): 12 hours.
        """
        import re
        if not season_str:
            return 12 * 3600  # 12 hours for current/unspecified

        match = re.search(r"(\d{4})", str(season_str))
        if match:
            start_year = int(match.group(1))
            # Current seasons are 2024-25 and newer
            if start_year < 2024:
                return 30 * 86400  # 30 days
        return 12 * 3600

    @classmethod
    def make_canonical_key(
        cls,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        season: Optional[str] = None,
        prefix: str = "deepbag:api:",
    ) -> str:
        """
        Generate a canonical cache key for an API call based on endpoint and sorted params.
        """
        cleaned_params = {}
        if params:
            for k, v in params.items():
                if k not in ("proxy", "headers", "timeout", "get_request"):
                    cleaned_params[k] = str(v).strip()
        
        # If season not explicit, extract from params if present
        if not season:
            for key in ("Season", "season", "season_id", "SeasonId"):
                if key in cleaned_params:
                    season = cleaned_params[key]
                    break
        
        sorted_items = sorted(cleaned_params.items())
        param_payload = json.dumps({"params": sorted_items, "season": season}, sort_keys=True)
        param_hash = hashlib.sha256(param_payload.encode("utf-8")).hexdigest()
        norm_endpoint = endpoint.strip().lower()
        return f"{prefix}{norm_endpoint}:{param_hash}"

    # Tier 1: Canonical API Data Caching
    def get_api_data(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        season: Optional[str] = None,
    ) -> Optional[Any]:
        cache_key = self.make_canonical_key(endpoint, params, season)
        try:
            val = self._sync_client.get(cache_key)
            if val is not None:
                self.stats["hits"] += 1
                try:
                    payload = json.loads(val)
                    return payload.get("data")
                except json.JSONDecodeError:
                    return val
            self.stats["misses"] += 1
            return None
        except Exception as e:
            logger.error(f"Error reading API data from sync cache: {e}")
            self.stats["misses"] += 1
            return None

    def set_api_data(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]],
        data: Any,
        season: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        cache_key = self.make_canonical_key(endpoint, params, season)
        if ttl is None:
            # Detect season
            s = season
            if not s and params:
                for key in ("Season", "season", "season_id", "SeasonId"):
                    if key in params:
                        s = str(params[key])
                        break
            ttl = self.get_season_ttl(s)

        from datetime import timezone
        payload = {
            "endpoint": endpoint,
            "params": params,
            "data": data,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._sync_client.set(cache_key, json.dumps(payload), ex=ttl)
            self.stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Error writing API data to sync cache: {e}")
            return False

    async def aget_api_data(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        season: Optional[str] = None,
    ) -> Optional[Any]:
        cache_key = self.make_canonical_key(endpoint, params, season)
        client = await self._get_async_client()
        try:
            val = await client.get(cache_key)
            if val is not None:
                self.stats["hits"] += 1
                try:
                    payload = json.loads(val)
                    return payload.get("data")
                except json.JSONDecodeError:
                    return val
            self.stats["misses"] += 1
            return None
        except Exception as e:
            logger.error(f"Error reading API data from async cache: {e}")
            self.stats["misses"] += 1
            return None

    async def aset_api_data(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]],
        data: Any,
        season: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        cache_key = self.make_canonical_key(endpoint, params, season)
        if ttl is None:
            s = season
            if not s and params:
                for key in ("Season", "season", "season_id", "SeasonId"):
                    if key in params:
                        s = str(params[key])
                        break
            ttl = self.get_season_ttl(s)

        from datetime import timezone
        payload = {
            "endpoint": endpoint,
            "params": params,
            "data": data,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        client = await self._get_async_client()
        try:
            await client.set(cache_key, json.dumps(payload), ex=ttl)
            self.stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Error writing API data to async cache: {e}")
            return False

    # Tier 2: Natural Language Query Caching (Sync)
    def get(self, query: str) -> Optional[str]:
        cache_key = self._generate_key(query)
        try:
            val = self._sync_client.get(cache_key)
            if val is not None:
                self.stats["hits"] += 1
                try:
                    payload = json.loads(val)
                    return payload.get("response")
                except json.JSONDecodeError:
                    return val
            self.stats["misses"] += 1
            return None
        except Exception as e:
            logger.error(f"Error reading from sync cache: {e}")
            self.stats["misses"] += 1
            return None

    def set(self, query: str, response: str, ttl: Optional[int] = None) -> bool:
        cache_key = self._generate_key(query)
        ttl = ttl if ttl is not None else self.default_ttl
        from datetime import timezone
        payload = {
            "query": query,
            "response": response,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._sync_client.set(cache_key, json.dumps(payload), ex=ttl)
            self.stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Error writing to sync cache: {e}")
            return False

    # Async Interface (for FastAPI async request handlers)
    async def aget(self, query: str) -> Optional[str]:
        cache_key = self._generate_key(query)
        client = await self._get_async_client()
        try:
            val = await client.get(cache_key)
            if val is not None:
                self.stats["hits"] += 1
                try:
                    payload = json.loads(val)
                    return payload.get("response")
                except json.JSONDecodeError:
                    return val
            self.stats["misses"] += 1
            return None
        except Exception as e:
            logger.error(f"Error reading from async cache: {e}")
            self.stats["misses"] += 1
            return None

    async def aset(self, query: str, response: str, ttl: Optional[int] = None) -> bool:
        cache_key = self._generate_key(query)
        ttl = ttl if ttl is not None else self.default_ttl
        from datetime import timezone
        payload = {
            "query": query,
            "response": response,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        client = await self._get_async_client()
        try:
            await client.set(cache_key, json.dumps(payload), ex=ttl)
            self.stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Error writing to async cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        hit_ratio = 0.0
        total = self.stats["hits"] + self.stats["misses"]
        if total > 0:
            hit_ratio = self.stats["hits"] / total
        return {
            **self.stats,
            "total_lookups": total,
            "hit_ratio": round(hit_ratio, 4),
            "backend": "fakeredis" if isinstance(self._sync_client, getattr(fakeredis, "FakeStrictRedis", ())) else "redis",
        }
