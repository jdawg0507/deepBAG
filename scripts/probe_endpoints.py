"""
Automated probe script for all NBA API endpoints.
Reads data/endpoint_catalogue.json, constructs benchmark test parameters for:
- Player: LeBron James (ID: 2544)
- Team: New York Knicks (ID: 1610612752)
- Season: 2023-24
- SeasonType: Regular Season
Executes calls via core.nba_client.NBAClient with canonical caching & token bucket rate limits.
Saves verification results to data/endpoint_probe_results.json.
"""
import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core.nba_client import NBAClient
from core.cache import RedisQueryCache
from core.rate_limiter import RedisSlidingWindowRateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("probe_endpoints")

BENCHMARK_VALUES = {
    'player_id': 2544, # LeBron James
    'player_id_nullable': 2544,
    'team_id': 1610612752, # New York Knicks
    'team_id_nullable': 1610612752,
    'season': '2023-24',
    'season_nullable': '2023-24',
    'season_id': '22023',
    'season_year_nullable': '2023-24',
    'season_type_all_star': 'Regular Season',
    'season_type_playoffs': 'Regular Season',
    'season_type_nullable': 'Regular Season',
    'league_id': '00',
    'league_id_nullable': '00',
    'per_mode_detailed': 'PerGame',
    'per_mode_simple': 'PerGame',
    'per_mode_nullable': 'PerGame',
    'measure_type_detailed_defense': 'Base',
    'measure_type_simple': 'Base',
    'measure_type_base': 'Base',
    'measure_type_player_defense': 'Base',
    'game_segment_nullable': 'First Half',
    'period': 0,
    'last_n_games': 0,
    'last_n_games_nullable': 0,
    'game_id': '0022300001',
    'game_id_nullable': '0022300001',
}

def generate_probe_params(endpoint_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate viable parameters for probing based on endpoint schema requirements."""
    params = {}
    required = endpoint_info.get("required_parameters", [])
    optional = endpoint_info.get("optional_parameters", [])
    
    # 1. Fill all required parameters
    for p in required:
        p_lower = p.lower()
        if p in BENCHMARK_VALUES:
            params[p] = BENCHMARK_VALUES[p]
        elif p_lower in BENCHMARK_VALUES:
            params[p] = BENCHMARK_VALUES[p_lower]
        elif 'player' in p_lower and 'id' in p_lower:
            params[p] = 2544
        elif 'team' in p_lower and 'id' in p_lower:
            params[p] = 1610612752
        elif 'season' in p_lower:
            params[p] = '2023-24'
        elif 'game' in p_lower and 'id' in p_lower:
            params[p] = '0022300001'
        elif 'league' in p_lower:
            params[p] = '00'
        else:
            # Cannot infer required parameter safely
            return None
            
    # 2. Add standard scoping parameters from optional if supported
    for p in optional:
        p_lower = p.lower()
        if p in ['season', 'season_nullable'] and p not in params:
            params[p] = '2023-24'
        elif p in ['season_type_all_star', 'season_type_playoffs', 'season_type_nullable'] and p not in params:
            params[p] = 'Regular Season'
        elif p in ['per_mode_detailed', 'per_mode_simple', 'per_mode_nullable'] and p not in params:
            params[p] = 'PerGame'
            
    return params

def run_probe(limit: Optional[int] = None, timeout: int = 2, max_retries: int = 1):
    catalogue_path = os.path.join(repo_root, "data", "endpoint_catalogue.json")
    if not os.path.exists(catalogue_path):
        logger.error(f"Catalogue not found at {catalogue_path}. Run build_catalogue.py first.")
        return
        
    with open(catalogue_path, "r", encoding="utf-8") as f:
        catalogue = json.load(f)
        
    logger.info(f"Loaded {len(catalogue)} endpoints from catalogue.")
    client = NBAClient()
    
    results = {}
    endpoints_to_probe = list(catalogue.keys())
    if limit:
        endpoints_to_probe = endpoints_to_probe[:limit]
        
    logger.info(f"Starting probe of {len(endpoints_to_probe)} endpoints (timeout={timeout}s, retries={max_retries})...")
    
    healthy_count = 0
    error_count = 0
    skipped_count = 0
    
    for i, ep_name in enumerate(endpoints_to_probe, 1):
        ep_info = catalogue[ep_name]
        probe_params = generate_probe_params(ep_info)
        
        if probe_params is None:
            results[ep_name] = {
                "status": "skipped_untestable",
                "reason": f"Could not satisfy required params: {ep_info.get('required_parameters')}",
                "datasets_found": [],
                "total_rows": 0,
                "latency_ms": 0,
            }
            skipped_count += 1
            logger.info(f"[{i}/{len(endpoints_to_probe)}] {ep_name}: SKIPPED (untestable required params)")
            continue
            
        logger.info(f"[{i}/{len(endpoints_to_probe)}] Probing {ep_name} with params: {probe_params}...")
        res = client.call(ep_name, probe_params, use_cache=True, timeout=timeout, max_retries=max_retries)
        
        total_rows = sum(len(rows) for rows in res.get("datasets", {}).values())
        ds_names = list(res.get("datasets", {}).keys())
        
        status = "healthy" if res.get("status") == "success" and total_rows > 0 else (
            "empty" if res.get("status") == "success" else "error"
        )
        
        if status in ("healthy", "empty"):
            healthy_count += 1
        else:
            error_count += 1
            
        results[ep_name] = {
            "status": status,
            "params_used": probe_params,
            "cached": res.get("cached", False),
            "datasets_found": ds_names,
            "total_rows": total_rows,
            "error": res.get("error"),
            "latency_ms": res.get("latency_ms", 0),
        }
        logger.info(f"  -> Result: status={status}, rows={total_rows}, latency={res.get('latency_ms', 0):.1f}ms")
        
    output_path = os.path.join(repo_root, "data", "endpoint_probe_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        
    logger.info("=" * 60)
    logger.info(f"Probe completed! Total: {len(endpoints_to_probe)}")
    logger.info(f"Healthy/Empty: {healthy_count} | Errors: {error_count} | Skipped: {skipped_count}")
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of endpoints to probe")
    parser.add_argument("--timeout", type=int, default=2, help="Timeout in seconds for probe calls")
    parser.add_argument("--retries", type=int, default=1, help="Max retries for probe calls")
    args = parser.parse_args()
    run_probe(limit=args.limit, timeout=args.timeout, max_retries=args.retries)
