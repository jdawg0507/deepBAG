"""
Mechanical Numeric-Trace Scorer and Statistical Evaluator for deepBAG Benchmark.
Pre-registered metrics:
1. Hallucination rate (strict verbatim & derived-allowed) via mechanical code check
2. Routing accuracy (endpoint, entity ID, season)
3. Cluster Consistency@4 (clustered by blueprint)
4. Cost & latency (calls, tokens, time: mean + p90)
5. 95% Wilson confidence intervals for rates
6. Paired blueprint-clustered bootstrap for differences
"""

import re
import math
import json
import unicodedata
from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np

def normalize_str(s: str) -> str:
    if not s:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().strip()

def extract_years_from_query(query: str) -> Set[str]:
    """Finds all year patterns (e.g. 2023-24, 2023, 1998, 24) in query so they aren't counted as hallucinated numbers."""
    years = set()
    matches = re.findall(r"\b(19\d\d|20\d\d)(?:-(\d{2,4}))?\b", query)
    for m in matches:
        if m[0]:
            years.add(m[0])
        if m[1]:
            years.add(m[1])
            if len(m[1]) == 2:
                prefix = m[0][:2]
                years.add(f"{prefix}{m[1]}")
    # Also find single standalone numbers that match words in query
    for token in re.findall(r"\b\d+\b", query):
        years.add(token)
    return years

def extract_numbers_from_text(text: str, query: str = "") -> List[float]:
    """Extracts numeric values from answer text, filtering out query years."""
    query_numbers = extract_years_from_query(query) if query else set()
    numbers = []
    
    # Matches decimals, integers, percentages
    tokens = re.findall(r"(?:^|[\s\(\[\$])([+-]?\d+(?:\.\d+)?%?)(?:[\s\)\]\.,;]|$)", text)
    for tok in tokens:
        clean_tok = tok.strip().rstrip(".,;%")
        if not clean_tok:
            continue
        # Skip if verbatim in query
        if clean_tok in query_numbers:
            continue
        try:
            val = float(clean_tok)
            # Filter out obvious markdown table formatting indexes or single digit list items
            if val in [1.0, 2.0, 3.0, 4.0] and re.search(rf"\b{int(val)}\.\s", text):
                continue
            numbers.append(val)
        except ValueError:
            pass
            
    return numbers

def extract_all_numbers_from_tool_outputs(outputs: List[Any]) -> Set[float]:
    """Flattens tool outputs and extracts all numbers returned by the tools."""
    found_numbers = set()
    
    def walk(obj):
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            found_numbers.add(round(float(obj), 3))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            # Check for numeric strings inside strings
            for num in re.findall(r"\b\d+(?:\.\d+)?\b", obj):
                try:
                    found_numbers.add(round(float(num), 3))
                except ValueError:
                    pass

    for out in outputs:
        walk(out)
        
    return found_numbers

def score_trace(trace: Dict[str, Any], ground_truth_bp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scores a single query execution trace mechanically.
    """
    query = trace.get("query", "")
    plan = trace.get("plan") or {}
    tool_calls = trace.get("tool_calls", [])
    tool_outputs = trace.get("tool_outputs", [])
    answer = trace.get("final_answer", "")
    
    is_trap = ground_truth_bp.get("ground_truth", {}).get("is_trap", False)
    expected_endpoint = ground_truth_bp.get("expected_endpoint")
    expected_season = ground_truth_bp.get("expected_season")
    expected_entities = [normalize_str(e) for e in ground_truth_bp.get("entities", [])]
    
    # 1. Routing Accuracy
    # Endpoint match
    actual_endpoints = []
    if isinstance(plan, dict):
        calls = plan.get("endpoint_calls") or []
        for c in calls:
            if isinstance(c, dict) and "endpoint" in c:
                actual_endpoints.append(c["endpoint"])
        if not actual_endpoints and "endpoint" in plan:
            actual_endpoints.append(plan["endpoint"])
    if not actual_endpoints and tool_calls:
        for tc in tool_calls:
            ep = tc.get("endpoint_name") or tc.get("endpoint")
            if ep:
                actual_endpoints.append(ep)
                
    primary_ep = actual_endpoints[0] if actual_endpoints else None
    
    if is_trap:
        # For traps, routing is accurate if refusal / correction was triggered
        endpoint_match = (
            "premise_correction" in str(plan) or 
            "did not play" in answer.lower() or 
            "never played" in answer.lower() or 
            "retired" in answer.lower() or
            "drafted in" in answer.lower() or
            "not in the nba" in answer.lower()
        )
    elif expected_endpoint == "Multiple":
        endpoint_match = len(actual_endpoints) >= 2 or (primary_ep and ("leader" in primary_ep.lower() or "league" in primary_ep.lower() or "player" in primary_ep.lower()))
    elif expected_endpoint == "historical_sql":
        endpoint_match = any("historical" in str(ep).lower() for ep in actual_endpoints) or any("sql" in str(tc).lower() for tc in tool_calls)
    else:
        endpoint_match = (primary_ep == expected_endpoint)
        
    # Season match
    actual_season = None
    if isinstance(plan, dict):
        actual_season = plan.get("season")
    if not actual_season and tool_calls:
        for tc in tool_calls:
            params = tc.get("parameters") or tc.get("params") or {}
            if "Season" in params or "season" in params:
                actual_season = params.get("Season") or params.get("season")
                break
                
    season_match = True
    if expected_season and not is_trap:
        season_match = (str(actual_season) == str(expected_season))
        
    # Entity match
    actual_entities = []
    if isinstance(plan, dict):
        receipt = plan.get("scope_receipt") or {}
        if isinstance(receipt, dict):
            for e in receipt.get("entities", []):
                actual_entities.append(normalize_str(e.get("name", "")))
        if not actual_entities:
            for er in plan.get("entities_raw", []):
                actual_entities.append(normalize_str(er))
    if not actual_entities and tool_calls:
        for tc in tool_calls:
            params = tc.get("parameters") or tc.get("params") or {}
            for k in ["player_id", "PlayerID", "team_id", "TeamID"]:
                if k in params:
                    actual_entities.append(str(params[k]))
                    
    entity_match = True
    if expected_entities and not is_trap:
        for exp in expected_entities:
            if not any(exp in act or act in exp for act in actual_entities):
                # Also check if entity name appears in tool call params
                if not any(exp in normalize_str(str(tc)) for tc in tool_calls):
                    entity_match = False
                    break
                    
    routing_success = endpoint_match and season_match and entity_match
    
    # 2. Hallucination Check (Strict Verbatim)
    answer_numbers = extract_numbers_from_text(answer, query=query)
    tool_numbers = extract_all_numbers_from_tool_outputs(tool_outputs)
    
    hallucinated_numbers_strict = []
    for num in answer_numbers:
        rounded_num = round(num, 3)
        # Check if rounded_num or integer exists in tool_numbers
        matched = False
        for tn in tool_numbers:
            if abs(tn - rounded_num) < 0.05:
                matched = True
                break
        if not matched:
            hallucinated_numbers_strict.append(num)
            
    has_hallucination_strict = len(hallucinated_numbers_strict) > 0 and len(tool_outputs) > 0
    if len(tool_outputs) == 0 and not is_trap and len(answer_numbers) > 0:
        # System made zero tool calls but produced specific stats numbers
        has_hallucination_strict = True
        hallucinated_numbers_strict = answer_numbers
        
    # 3. Derived-allowed check
    # If a number is not verbatim, check if there is an explicit arithmetic formula in the answer
    hallucinated_numbers_derived = []
    for num in hallucinated_numbers_strict:
        # Check if number appears immediately next to an operator, e.g. "X / Y = Z"
        pattern = rf"[\+\-\*\/]\s*{re.escape(str(num))}|{re.escape(str(num))}\s*="
        if not re.search(pattern, answer):
            hallucinated_numbers_derived.append(num)
            
    has_hallucination_derived = len(hallucinated_numbers_derived) > 0
    
    # Cost & latency
    llm_calls = trace.get("llm_calls_count", 1)
    tokens = trace.get("total_tokens", 0)
    latency_ms = trace.get("latency_ms", 0.0)
    
    return {
        "query": query,
        "blueprint_id": ground_truth_bp.get("id"),
        "category": ground_truth_bp.get("category"),
        "is_trap": is_trap,
        "endpoint_match": endpoint_match,
        "season_match": season_match,
        "entity_match": entity_match,
        "routing_success": routing_success,
        "total_numbers_in_answer": len(answer_numbers),
        "hallucinated_numbers_strict": hallucinated_numbers_strict,
        "has_hallucination_strict": has_hallucination_strict,
        "has_hallucination_derived": has_hallucination_derived,
        "actual_endpoint": primary_ep,
        "actual_season": actual_season,
        "llm_calls": llm_calls,
        "tokens": tokens,
        "latency_ms": latency_ms
    }

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence
    p = successes / total
    denominator = 1 + z**2 / total
    centre_adjusted_probability = p + z**2 / (2 * total)
    adjusted_std_dev = math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lower = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std_dev) / denominator
    return (max(0.0, lower), min(1.0, upper))

def blueprint_clustered_bootstrap_difference(
    scores_a: List[Dict[str, Any]], 
    scores_b: List[Dict[str, Any]], 
    metric_key: str,
    n_resamples: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired, blueprint-clustered bootstrap confidence interval for difference (System A - System B).
    Preserves blueprint clusters across resamples.
    """
    rng = np.random.default_rng(seed)
    
    # Group scores by blueprint_id
    bp_dict_a = {}
    for s in scores_a:
        bp_id = s["blueprint_id"]
        bp_dict_a.setdefault(bp_id, []).append(s)
        
    bp_dict_b = {}
    for s in scores_b:
        bp_id = s["blueprint_id"]
        bp_dict_b.setdefault(bp_id, []).append(s)
        
    common_blueprints = sorted(list(set(bp_dict_a.keys()).intersection(set(bp_dict_b.keys()))))
    if not common_blueprints:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "crosses_zero": True}
        
    def calc_mean_for_bps(bp_list, bp_dict):
        vals = []
        for bp in bp_list:
            items = bp_dict[bp]
            for it in items:
                v = it[metric_key]
                vals.append(1.0 if v is True else (0.0 if v is False else float(v)))
        return np.mean(vals) if vals else 0.0
        
    observed_diff = calc_mean_for_bps(common_blueprints, bp_dict_a) - calc_mean_for_bps(common_blueprints, bp_dict_b)
    
    boot_diffs = []
    n_bps = len(common_blueprints)
    for _ in range(n_resamples):
        sampled_bps = rng.choice(common_blueprints, size=n_bps, replace=True)
        diff = calc_mean_for_bps(sampled_bps, bp_dict_a) - calc_mean_for_bps(sampled_bps, bp_dict_b)
        boot_diffs.append(diff)
        
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    crosses_zero = (ci_lower <= 0.0 <= ci_upper)
    
    return {
        "observed_diff": float(observed_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "crosses_zero": crosses_zero
    }

def compute_cluster_consistency(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes Consistency@4 per blueprint."""
    bp_groups = {}
    for s in scores:
        bp_id = s["blueprint_id"]
        bp_groups.setdefault(bp_id, []).append(s)
        
    consistent_count = 0
    total_bps = len(bp_groups)
    
    per_bp = {}
    for bp_id, items in bp_groups.items():
        endpoints = {it["actual_endpoint"] for it in items}
        seasons = {it["actual_season"] for it in items}
        is_consistent = (len(endpoints) == 1) and (None not in endpoints) and (len(seasons) <= 1)
        per_bp[bp_id] = is_consistent
        if is_consistent:
            consistent_count += 1
            
    rate = consistent_count / total_bps if total_bps > 0 else 0.0
    ci = wilson_score_interval(consistent_count, total_bps)
    return {
        "rate": rate,
        "consistent_blueprints": consistent_count,
        "total_blueprints": total_bps,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "per_blueprint": per_bp
    }
