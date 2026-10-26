"""
Evaluation Runner for Router Semantic Evolution.
Measures Accuracy and Cluster Consistency@4 across generated evaluation benchmarks.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List
from tabulate import tabulate

from router.intent_parser import IntentParser

import unicodedata

def normalize_str(s: str) -> str:
    if not s:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()

def evaluate_item(parser: IntentParser, item: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    expected_endpoint = item.get("expected_endpoint")
    expected_season = item.get("expected_season")
    expected_entities = [normalize_str(e) for e in item.get("entities", [])]
    paraphrases = item.get("paraphrases", [])
    
    paraphrase_results = []
    
    for p in paraphrases:
        try:
            plan = parser.parse_intent(p)
            receipt = plan.scope_receipt
            actual_endpoint = plan.primary_endpoint
            actual_season = plan.season or plan.parameters.get("season")
            actual_entities = [normalize_str(e.get("name", "")) for e in receipt.entities] if (receipt and receipt.entities) else []
            
            # Check endpoint
            if expected_endpoint == "Multiple":
                endpoint_match = (
                    len(plan.endpoint_calls) >= 2 or
                    (actual_endpoint and ("leader" in actual_endpoint.lower() or "league" in actual_endpoint.lower() or "player" in actual_endpoint.lower()))
                )
            elif expected_endpoint == "historical_sql":
                endpoint_match = (
                    any(c.tool == "query_historical_database" or "historical" in (c.endpoint or "").lower() for c in plan.endpoint_calls) or
                    (int(str(actual_season)[:4]) < 1996 if actual_season and str(actual_season)[:4].isdigit() else False)
                )
            else:
                endpoint_match = (actual_endpoint == expected_endpoint)
            
            # Check entities
            entities_match = True
            for exp in expected_entities:
                if not any(exp in act or act in exp for act in actual_entities):
                    entities_match = False
                    break
            
            # Check season if specified
            season_match = True
            if expected_season:
                season_match = (actual_season == expected_season)
                
            is_accurate = endpoint_match and entities_match and season_match
            
            paraphrase_results.append({
                "query": p,
                "endpoint": actual_endpoint,
                "season": actual_season,
                "entities": actual_entities,
                "endpoint_match": endpoint_match,
                "entities_match": entities_match,
                "season_match": season_match,
                "is_accurate": is_accurate
            })
            
            if verbose:
                status = "✓" if is_accurate else "✗"
                print(f"    [{status}] {p} -> {actual_endpoint} ({actual_season}) entities={actual_entities}")
                
        except Exception as e:
            if verbose:
                print(f"    [ERR] {p} -> Exception: {e}")
            paraphrase_results.append({
                "query": p,
                "error": str(e),
                "is_accurate": False,
                "endpoint": None,
                "season": None,
                "entities": []
            })
            
    # Compute Cluster Consistency@4:
    # All paraphrases must produce the exact same endpoint and season
    endpoints = {r.get("endpoint") for r in paraphrase_results}
    seasons = {r.get("season") for r in paraphrase_results}
    consistent = len(endpoints) == 1 and (None not in endpoints) and (len(seasons) <= 1)
    
    all_accurate = all(r.get("is_accurate", False) for r in paraphrase_results)
    
    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "canonical_query": item.get("canonical_query"),
        "paraphrase_results": paraphrase_results,
        "is_consistent": consistent,
        "all_accurate": all_accurate,
        "accuracy_count": sum(1 for r in paraphrase_results if r.get("is_accurate", False)),
        "total_count": len(paraphrases)
    }

def run_eval(dataset_path: str, limit: int = None, threshold: float = 0.85, output_file: str = None, verbose: bool = False):
    print(f"=== DeepBAG Semantic Router Evaluation Harness ===")
    print(f"Loading dataset: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if limit:
        dataset = dataset[:limit]
        print(f"Running on subset of {len(dataset)} blueprints")
    else:
        print(f"Running on all {len(dataset)} blueprints ({len(dataset) * 4} queries)")
        
    parser = IntentParser()
    
    category_stats = {}
    item_results = []
    
    total_paraphrases = 0
    total_accurate = 0
    total_consistent_clusters = 0
    
    for i, item in enumerate(dataset):
        cat = item.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {
                "blueprints": 0,
                "consistent_clusters": 0,
                "paraphrases": 0,
                "accurate_paraphrases": 0
            }
            
        print(f"[{i+1}/{len(dataset)}] ({cat}) {item.get('canonical_query')}")
        res = evaluate_item(parser, item, verbose=verbose)
        item_results.append(res)
        
        category_stats[cat]["blueprints"] += 1
        category_stats[cat]["paraphrases"] += res["total_count"]
        category_stats[cat]["accurate_paraphrases"] += res["accuracy_count"]
        if res["is_consistent"]:
            category_stats[cat]["consistent_clusters"] += 1
            total_consistent_clusters += 1
            
        total_paraphrases += res["total_count"]
        total_accurate += res["accuracy_count"]
        
    # Build report table
    table_rows = []
    for cat, stats in category_stats.items():
        acc_pct = (stats["accurate_paraphrases"] / stats["paraphrases"]) * 100 if stats["paraphrases"] > 0 else 0
        cons_pct = (stats["consistent_clusters"] / stats["blueprints"]) * 100 if stats["blueprints"] > 0 else 0
        table_rows.append([
            cat,
            stats["blueprints"],
            stats["paraphrases"],
            f"{stats['accurate_paraphrases']}/{stats['paraphrases']} ({acc_pct:.1f}%)",
            f"{stats['consistent_clusters']}/{stats['blueprints']} ({cons_pct:.1f}%)"
        ])
        
    overall_acc = (total_accurate / total_paraphrases) * 100 if total_paraphrases > 0 else 0
    overall_cons = (total_consistent_clusters / len(dataset)) * 100 if len(dataset) > 0 else 0
    
    table_rows.append([
        "TOTAL / OVERALL",
        len(dataset),
        total_paraphrases,
        f"{total_accurate}/{total_paraphrases} ({overall_acc:.1f}%)",
        f"{total_consistent_clusters}/{len(dataset)} ({overall_cons:.1f}%)"
    ])
    
    headers = ["Category", "Blueprints", "Queries", "Accuracy Rate", "Consistency@4 Rate"]
    report_table = tabulate(table_rows, headers=headers, tablefmt="github")
    print("\n" + report_table + "\n")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        summary = {
            "dataset": dataset_path,
            "blueprints_evaluated": len(dataset),
            "total_queries": total_paraphrases,
            "overall_accuracy_pct": overall_acc,
            "overall_consistency_pct": overall_cons,
            "category_breakdown": category_stats,
            "item_results": item_results
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved evaluation report to {output_file}")
        
    passed = (overall_acc / 100.0 >= threshold) and (overall_cons / 100.0 >= threshold)
    if passed:
        print(f"PASSED: Threshold {threshold * 100:.0f}% met!")
        return 0
    else:
        print(f"FAILED: Failed to achieve threshold {threshold * 100:.0f}%")
        return 1

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="DeepBAG Router Evaluation Runner")
    cli_parser.add_argument("--dataset", default="evals/dev_seed.json", help="Path to evaluation JSON dataset")
    cli_parser.add_argument("--limit", type=int, default=None, help="Limit number of blueprints to run")
    cli_parser.add_argument("--threshold", type=float, default=0.80, help="Required consistency threshold (0-1.0)")
    cli_parser.add_argument("--output", default="evals/dev_results.json", help="Path to write JSON results")
    cli_parser.add_argument("--verbose", action="store_true", help="Print verbose paraphrase results")
    
    args = cli_parser.parse_args()
    sys.exit(run_eval(
        dataset_path=args.dataset,
        limit=args.limit,
        threshold=args.threshold,
        output_file=args.output,
        verbose=args.verbose
    ))
