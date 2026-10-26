"""
DeepEval & PyTest Evaluation Suite for deepBAG Held-Out Benchmark (Questions 51-100).
Rules:
1. Dev set (1-50) and Held-out set (51-100) are scored strictly separately, never blended.
2. Scoring by Plan + Evidence: wrong player, missing filter, wrong season, or collapsing multi-entity = FAIL.
3. Tests each tier individually and enforces a >= 0.85 pass threshold (target 1.00).
"""
import os
import json
import pytest
from typing import Dict, Any, List

from nlp.hybrid_router import HybridRouter
from nlp.caesura_planner import CaesuraPlanner
from nba_nlp_analyzer_modular import NBAQueryAnalyzer

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "fixtures", "eval_held_out_50_queries.json")

def load_held_out_queries() -> List[Dict[str, Any]]:
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def router():
    return HybridRouter()

@pytest.fixture(scope="module")
def planner():
    return CaesuraPlanner()

@pytest.fixture(scope="module")
def analyzer():
    return NBAQueryAnalyzer.__new__(NBAQueryAnalyzer)

@pytest.fixture(scope="module")
def held_out_data():
    return load_held_out_queries()


def test_tier_1_harder_single_entity(router, held_out_data):
    """
    Tier 1: Single-entity queries with implicit, ambiguous, or contextual filters.
    Requires exact entity matching, per-mode resolution, opponent filters, and minutes filters.
    """
    tier1_items = [q for q in held_out_data if q["tier"] == 1]
    assert len(tier1_items) == 10, f"Expected 10 Tier 1 items, got {len(tier1_items)}"

    passed = 0
    failures = []
    for item in tier1_items:
        plan = router.route(item["query"])
        expected_entities = item.get("entities", [])
        matched_name = plan.player_name or (plan.entities[0]["name"] if plan.entities else None)

        item_fails = []
        if expected_entities and matched_name != expected_entities[0]:
            item_fails.append(f"Entity mismatch: got '{matched_name}', expected '{expected_entities[0]}'")

        if item.get("expected_per_mode") and plan.params.get("per_mode_detailed") != item["expected_per_mode"]:
            item_fails.append(f"Per-mode mismatch: got '{plan.params.get('per_mode_detailed')}', expected '{item['expected_per_mode']}'")

        if item.get("opponent") and "opponent_team_id" not in plan.params:
            item_fails.append(f"Missing opponent filter for {item['opponent']}")

        if item.get("needs_minutes_filter") and "min_minutes" not in plan.params:
            item_fails.append("Missing minutes filter (30+ minutes)")

        if item.get("needs_multi_season_type") and (plan.endpoint != "Multiple" or len(plan.plans) < 2):
            item_fails.append("Failed to generate multi-season-type plan (Regular Season + Playoffs)")

        if not item_fails:
            passed += 1
        else:
            failures.append(f"Q{item['id']} ('{item['query']}'): {'; '.join(item_fails)}")

    rate = passed / len(tier1_items)
    print(f"\n[HELD-OUT TIER 1] {passed}/{len(tier1_items)} passed ({rate * 100:.1f}%)")
    assert rate >= 0.85, f"Tier 1 below 85% threshold ({rate:.2f}):\n" + "\n".join(failures)


def test_tier_2_multi_entity_multi_condition(router, held_out_data):
    """
    Tier 2: Multi-entity, comparisons, head-to-head, and rankings.
    Must NOT collapse to a single entity.
    """
    tier2_items = [q for q in held_out_data if q["tier"] == 2]
    assert len(tier2_items) == 10, f"Expected 10 Tier 2 items, got {len(tier2_items)}"

    passed = 0
    failures = []
    for item in tier2_items:
        qid = item["id"]
        plan = router.route(item["query"])
        expected_entities = item.get("entities", [])

        item_passed = False
        fail_reasons = []

        if item.get("is_on_off"):
            if plan.endpoint == "TeamPlayerOnOffDetails" and plan.params.get("player_id") == 203497 and plan.params.get("vs_player_id") == 1630162:
                item_passed = True
            else:
                fail_reasons.append(f"Expected TeamPlayerOnOffDetails for Gobert/Edwards, got {plan.endpoint} {plan.params}")
        elif qid == 63:
            if plan.endpoint == "Multiple" and len(plan.plans) >= 2 and plan.params.get("location") == "Home":
                item_passed = True
            else:
                fail_reasons.append(f"Expected multi-team Home net rating plan, got {plan.endpoint} {plan.params}")
        elif qid == 69:
            if plan.endpoint == "TeamDashboardByOpponent" and plan.params.get("season_type_all_star") == "Playoffs":
                item_passed = True
            else:
                fail_reasons.append(f"Expected TeamDashboardByOpponent with Playoffs, got {plan.endpoint} {plan.params}")
        else:
            if plan.endpoint == "Multiple" and len(plan.plans) >= len(expected_entities):
                item_passed = True
            else:
                fail_reasons.append(f"Expected plan list for {len(expected_entities)} entities, got endpoint='{plan.endpoint}' with {len(plan.plans)} plans")

        if item_passed:
            passed += 1
        else:
            failures.append(f"Q{item['id']} ('{item['query']}'): {'; '.join(fail_reasons)}")

    rate = passed / len(tier2_items)
    print(f"\n[HELD-OUT TIER 2] {passed}/{len(tier2_items)} passed ({rate * 100:.1f}%)")
    assert rate >= 0.85, f"Tier 2 below 85% threshold ({rate:.2f}):\n" + "\n".join(failures)


def test_tier_3_research_composite(analyzer, held_out_data):
    """
    Tier 3: Research engine queries requiring multi-step aggregation, pick-and-roll, rim deterrence, pace.
    """
    tier3_items = [q for q in held_out_data if q["tier"] == 3]
    assert len(tier3_items) == 10, f"Expected 10 Tier 3 items, got {len(tier3_items)}"

    passed = 0
    failures = []
    for item in tier3_items:
        is_res = analyzer.is_research_query(item["query"])
        if is_res:
            passed += 1
        else:
            failures.append(f"Q{item['id']} ('{item['query']}'): Not routed to research engine")

    rate = passed / len(tier3_items)
    print(f"\n[HELD-OUT TIER 3] {passed}/{len(tier3_items)} passed ({rate * 100:.1f}%)")
    assert rate >= 0.85, f"Tier 3 below 85% threshold ({rate:.2f}):\n" + "\n".join(failures)


def test_tier_4_thesis_and_scope(router, planner, analyzer, held_out_data):
    """
    Tier 4: Thesis verification, definition queries, and missing scope clarification.
    """
    tier4_items = [q for q in held_out_data if q["tier"] == 4]
    assert len(tier4_items) == 10, f"Expected 10 Tier 4 items, got {len(tier4_items)}"

    passed = 0
    failures = []
    for item in tier4_items:
        qid = item["id"]
        query = item["query"]
        plan = router.route(query)
        intent = planner.extract_intent(query)
        logical_plan = planner.build_logical_plan(intent)

        item_passed = False
        fail_reasons = []

        if qid == 84:
            if intent.clarification_needed is not None or plan.tool == "premise_correction":
                item_passed = True
            else:
                fail_reasons.append("Failed to detect missing scope and request clarification")
        elif qid == 83:
            endpoints = [s.endpoint for s in logical_plan.steps]
            if "PlayerDashPtShotDefend" in endpoints:
                item_passed = True
            else:
                fail_reasons.append(f"Failed archetype switch for Wemby interior deterrence: got {endpoints}")
        elif qid == 81:
            endpoints = [s.endpoint for s in logical_plan.steps]
            if intent.question_type.value == "thesis" and "TeamDashboardByGeneralSplits" in endpoints and intent.season_type == "Playoffs":
                item_passed = True
            else:
                fail_reasons.append(f"Thesis plan incomplete: endpoints={endpoints}, season_type={intent.season_type}")
        else:
            is_res = analyzer.is_research_query(query)
            if is_res and intent.question_type.value in ["thesis", "definition", "paradox", "longitudinal", "comparison", "single"]:
                item_passed = True
            else:
                fail_reasons.append(f"Intent classified as {intent.question_type.value}, research={is_res}")

        if item_passed:
            passed += 1
        else:
            failures.append(f"Q{item['id']} ('{item['query']}'): {'; '.join(fail_reasons)}")

    rate = passed / len(tier4_items)
    print(f"\n[HELD-OUT TIER 4] {passed}/{len(tier4_items)} passed ({rate * 100:.1f}%)")
    assert rate >= 0.85, f"Tier 4 below 85% threshold ({rate:.2f}):\n" + "\n".join(failures)


def test_tier_5_adversarial_traps(router, held_out_data):
    """
    Tier 5: Adversarial traps, false premises, impossible career bounds, 0-attempt divisions, future data.
    Must intercept and clarify/correct rather than naively querying APIs.
    """
    tier5_items = [q for q in held_out_data if q["tier"] == 5]
    assert len(tier5_items) == 10, f"Expected 10 Tier 5 items, got {len(tier5_items)}"

    passed = 0
    failures = []
    for item in tier5_items:
        plan = router.route(item["query"])
        if plan.tool == "premise_correction" and plan.direct_response:
            passed += 1
        else:
            failures.append(f"Q{item['id']} ('{item['query']}'): Bypassed trap, routed to {plan.tool} ({plan.endpoint})")

    rate = passed / len(tier5_items)
    print(f"\n[HELD-OUT TIER 5] {passed}/{len(tier5_items)} passed ({rate * 100:.1f}%)")
    assert rate >= 0.85, f"Tier 5 below 85% threshold ({rate:.2f}):\n" + "\n".join(failures)
