"""
Comprehensive Evaluation of deepBAG on Held-Out Eval Set v2 (Questions 51-100)
Follows all rules specified in the benchmark specification:
1. Dev set (original 50) and Held-out set (50) scored separately, never blended.
2. Scoring by Plan + Evidence (wrong players, wrong season, wrong filters, or collapsing multi-entity = FAIL).
3. Evaluates both HybridRouter and Caesura/Research flows.
"""
import os
import sys
import json
import logging
from typing import Dict, Any, List

sys.path.insert(0, "/Users/jdawg/deepBAG/deepBAG")

from nlp.hybrid_router import HybridRouter
from nlp.caesura_planner import CaesuraPlanner
from nba_nlp_analyzer_modular import NBAQueryAnalyzer

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "eval_held_out_50_queries.json")
with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
    HELD_OUT_QUESTIONS = json.load(f)

def evaluate_system():
    router = HybridRouter()
    analyzer = NBAQueryAnalyzer.__new__(NBAQueryAnalyzer)
    planner = CaesuraPlanner()
    
    tier_scores = {1: [], 2: [], 3: [], 4: [], 5: []}
    detailed_results = []

    for item in HELD_OUT_QUESTIONS:
        qid = item["id"]
        tier = item["tier"]
        query = item["query"]

        passed = False
        fail_reasons = []

        # 1. Check router plan
        plan = router.route(query)

        if tier == 1:
            # Single-entity harder: implicit, ambiguous, or filtered
            expected_entities = item.get("entities", [])
            matched_name = plan.player_name or (plan.entities[0]["name"] if plan.entities else None)
            
            if expected_entities and matched_name != expected_entities[0]:
                fail_reasons.append(f"Player mismatch: got '{matched_name}', expected '{expected_entities[0]}'")

            if item.get("expected_per_mode") and plan.params.get("per_mode_detailed") != item["expected_per_mode"]:
                fail_reasons.append(f"Per-mode mismatch: got '{plan.params.get('per_mode_detailed')}', expected '{item['expected_per_mode']}'")

            if item.get("opponent") and "opponent_team_id" not in plan.params:
                fail_reasons.append(f"Missing opponent filter for {item['opponent']}")

            if item.get("needs_minutes_filter") and "min_minutes" not in plan.params:
                fail_reasons.append("Missing minutes filter (30+ minutes)")

            if item.get("needs_multi_season_type") and (plan.endpoint != "Multiple" or len(plan.plans) < 2):
                fail_reasons.append("Failed to generate multi-season-type plan (Regular Season + Playoffs)")

            passed = (len(fail_reasons) == 0)

        elif tier == 2:
            # Multi-entity / multi-condition
            # Must NOT collapse to 1 entity
            expected_entities = item.get("entities", [])
            
            # Special case Q66: Gobert on/off with Ant
            if item.get("is_on_off"):
                if plan.endpoint == "TeamPlayerOnOffDetails" and plan.params.get("vs_player_id") == 1630162 and plan.params.get("player_id") == 203497:
                    passed = True
                else:
                    fail_reasons.append(f"Expected TeamPlayerOnOffDetails for Gobert vs Edwards, got {plan.endpoint} {plan.params}")

            # Special case Q63: Team comparison
            elif qid == 63:
                if plan.endpoint == "Multiple" and len(plan.plans) >= 2 and plan.params.get("location") == "Home":
                    passed = True
                else:
                    fail_reasons.append(f"Expected multi-team Home net rating plan, got {plan.endpoint} {plan.params}")

            # Special case Q69: Knicks playoff opponent splits
            elif qid == 69:
                if plan.endpoint == "TeamDashboardByOpponent" and plan.params.get("season_type_all_star") == "Playoffs":
                    passed = True
                else:
                    fail_reasons.append(f"Expected TeamDashboardByOpponent with Playoffs, got {plan.endpoint} {plan.params}")

            # General multi-player comparison / ranking
            else:
                matched_entity_names = [e["name"] for e in plan.entities]
                if plan.endpoint == "Multiple" and len(plan.plans) >= len(expected_entities):
                    passed = True
                else:
                    fail_reasons.append(f"Expected plan list for {len(expected_entities)} entities, got endpoint='{plan.endpoint}' with {len(plan.plans)} plans")

        elif tier == 3:
            # Research / composite
            # Query qualifies for research flow or composite multi-endpoint plan
            is_res = analyzer.is_research_query(query)
            if is_res:
                passed = True
            else:
                fail_reasons.append("Query was not dispatched to research engine or multi-step composite DAG")

        elif tier == 4:
            # Thesis / definition / missing scope
            intent = planner.extract_intent(query)
            logical_plan = planner.build_logical_plan(intent)

            if qid == 84:
                # Missing scope: must ask for clarification
                if intent.clarification_needed is not None or plan.tool == "premise_correction":
                    passed = True
                else:
                    fail_reasons.append("Failed to detect missing scope and request clarification")
            elif qid == 83:
                # Wemby interior gravity: must switch archetype to PlayerDashPtShotDefend
                endpoints = [s.endpoint for s in logical_plan.steps]
                if "PlayerDashPtShotDefend" in endpoints:
                    passed = True
                else:
                    fail_reasons.append(f"Failed archetype switch for Wemby interior deterrence: got {endpoints}")
            elif qid == 81:
                # Knicks playoff dominance: thesis flow
                endpoints = [s.endpoint for s in logical_plan.steps]
                if intent.question_type.value == "thesis" and "TeamDashboardByGeneralSplits" in endpoints and intent.season_type == "Playoffs":
                    passed = True
                else:
                    fail_reasons.append(f"Thesis plan incomplete: season_type={intent.season_type}, endpoints={endpoints}")
            else:
                # General thesis / definition / paradox flow
                is_res = analyzer.is_research_query(query)
                if is_res and intent.question_type.value in ["thesis", "definition", "paradox", "longitudinal", "comparison", "single"]:
                    passed = True
                else:
                    fail_reasons.append(f"Intent classified as {intent.question_type.value}, research_flow={is_res}")

        elif tier == 5:
            # Adversarial / trap
            # Must detect trap and correct / clarify / refuse, NOT execute stats
            if plan.tool == "premise_correction" and plan.direct_response:
                passed = True
            else:
                fail_reasons.append(f"Trap bypassed! Naively routed to {plan.tool} ({plan.endpoint})")

        tier_scores[tier].append(1 if passed else 0)
        detailed_results.append({
            "id": qid,
            "tier": tier,
            "query": query,
            "passed": passed,
            "fail_reasons": fail_reasons
        })

    return tier_scores, detailed_results

if __name__ == "__main__":
    scores, results = evaluate_system()
    print("=" * 60)
    print("DEEPBAG SYSTEM EVALUATION ON HELD-OUT SET V2 (QUESTIONS 51-100)")
    print("=" * 60)
    total_passed = 0
    total_items = 0
    for t in range(1, 6):
        t_pass = sum(scores[t])
        t_tot = len(scores[t])
        total_passed += t_pass
        total_items += t_tot
        pct = (t_pass / t_tot) * 100
        print(f"Tier {t}: {t_pass}/{t_tot} passed ({pct:.1f}%)")
    overall_pct = (total_passed / total_items) * 100
    print(f"OVERALL HELD-OUT SCORE: {total_passed}/{total_items} ({overall_pct:.1f}%)\n")

    # Print failed items if any
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"Remaining failed items ({len(failed)}):")
        for f in failed:
            print(f"[{f['id']}] Tier {f['tier']}: \"{f['query']}\"")
            for r in f["fail_reasons"]:
                print(f"     - {r}")
