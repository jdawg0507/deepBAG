"""
Core NBA API Client.
Single unified call site for executing nba_api stats requests with:
- Canonical Redis query caching (core.cache.RedisQueryCache)
- Atomic Sliding-Window Token Bucket Rate Limiting (core.rate_limiter.TokenBucketRateLimiter)
- Standardized custom headers to bypass stats.nba.com bot protections
- Automatic exponential backoff on HTTP 429 and ReadTimeout errors
- Schema and dataset parsing into pandas DataFrames and JSON dicts
"""
import os
import time
import json
import logging
import inspect
import importlib
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

from core.cache import RedisQueryCache
from core.rate_limiter import RedisSlidingWindowRateLimiter
from core.telemetry import TelemetryManager
from nba_api.stats.library.http import STATS_HEADERS

logger = logging.getLogger("deepbag.nba_client")

# Production browser headers from nba_api matching current Chrome signatures
DEFAULT_HEADERS = dict(STATS_HEADERS)

class NBAClient:
    def __init__(
        self,
        cache: Optional[RedisQueryCache] = None,
        rate_limiter: Optional[RedisSlidingWindowRateLimiter] = None,
        timeout: int = 15,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        self.cache = cache or RedisQueryCache()
        self.rate_limiter = rate_limiter or RedisSlidingWindowRateLimiter()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._endpoint_classes: Dict[str, Any] = {}

    def _resolve_endpoint_class(self, endpoint_name: str) -> Optional[Any]:
        if endpoint_name in self._endpoint_classes:
            return self._endpoint_classes[endpoint_name]

        try:
            # Direct import try
            mod = importlib.import_module(f"nba_api.stats.endpoints.{endpoint_name.lower()}")
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if name.lower() == endpoint_name.lower():
                    self._endpoint_classes[endpoint_name] = obj
                    return obj
        except Exception:
            pass

        # Try finding across endpoints
        import pkgutil
        import nba_api.stats.endpoints as ep_pkg
        for _, mod_name, _ in pkgutil.iter_modules(ep_pkg.__path__):
            if mod_name.lower() == endpoint_name.lower():
                try:
                    mod = importlib.import_module(f"nba_api.stats.endpoints.{mod_name}")
                    for name, obj in inspect.getmembers(mod, inspect.isclass):
                        if name.lower() == endpoint_name.lower():
                            self._endpoint_classes[endpoint_name] = obj
                            return obj
                except Exception:
                    pass

        return None

    def call(
        self,
        endpoint_name: str,
        params: Dict[str, Any],
        use_cache: bool = True,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute an NBA API call with caching, rate limiting, and backoff retries.
        Returns a dict:
        {
            "endpoint": endpoint_name,
            "params": params,
            "cached": bool,
            "datasets": { "DatasetName": [rows_dict] },
            "status": "success" | "error",
            "error": str or None,
            "latency_ms": float
        }
        """
        start_time = time.time()
        
        # 1. Canonical Cache Check
        if use_cache:
            cached_data = self.cache.get_api_data(endpoint_name, params)
            if cached_data is not None:
                latency = (time.time() - start_time) * 1000
                TelemetryManager.record_cache_event(hit=True, query=f"{endpoint_name}:{sorted(params.items())}")
                return {
                    "endpoint": endpoint_name,
                    "params": params,
                    "cached": True,
                    "datasets": cached_data.get("datasets", {}),
                    "status": "success",
                    "error": None,
                    "latency_ms": latency,
                }
            TelemetryManager.record_cache_event(hit=False, query=f"{endpoint_name}:{sorted(params.items())}")

        # 2. Resolve endpoint class
        endpoint_cls = self._resolve_endpoint_class(endpoint_name)
        if not endpoint_cls:
            return {
                "endpoint": endpoint_name,
                "params": params,
                "cached": False,
                "datasets": {},
                "status": "error",
                "error": f"Endpoint class '{endpoint_name}' not found in nba_api.",
                "latency_ms": (time.time() - start_time) * 1000,
            }

        # 3. Rate limiting and execution loop with exponential backoff
        effective_timeout = timeout or self.timeout
        effective_retries = max_retries or self.max_retries
        last_exception = None
        for attempt in range(1, effective_retries + 1):
            try:
                # Rate limit token wait
                self.rate_limiter.acquire_sync(tokens=1, timeout=effective_timeout)
                
                # Execute request
                instance = endpoint_cls(
                    headers=DEFAULT_HEADERS,
                    timeout=effective_timeout,
                    **params
                )
                
                # Extract all datasets as dicts
                data_frames = instance.get_data_frames()
                dataset_names = getattr(instance, 'dataset_names', []) or [f"dataset_{i}" for i in range(len(data_frames))]
                
                datasets_dict = {}
                for name, df in zip(dataset_names, data_frames):
                    datasets_dict[name] = df.to_dict(orient="records")
                    
                result_payload = {
                    "endpoint": endpoint_name,
                    "params": params,
                    "cached": False,
                    "datasets": datasets_dict,
                    "status": "success",
                    "error": None,
                    "latency_ms": (time.time() - start_time) * 1000,
                }
                
                # Save to cache
                if use_cache:
                    season = params.get("season") or params.get("season_nullable") or "2023-24"
                    self.cache.set_api_data(endpoint_name, params, result_payload, season=season)
                    
                return result_payload

            except Exception as e:
                last_exception = e
                err_str = str(e)
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed for {endpoint_name}: {err_str}")
                
                # If rate limited (429) or timed out, back off
                if "429" in err_str or "ReadTimeout" in err_str or "timeout" in err_str.lower():
                    sleep_time = self.backoff_factor ** attempt
                    time.sleep(sleep_time)
                else:
                    # Non-transient error (e.g. invalid parameter), don't retry endlessly
                    break

        return {
            "endpoint": endpoint_name,
            "params": params,
            "cached": False,
            "datasets": {},
            "status": "error",
            "error": str(last_exception),
            "latency_ms": (time.time() - start_time) * 1000,
        }
