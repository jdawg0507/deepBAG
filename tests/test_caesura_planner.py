"""
Unit tests for CAESURA Routing DAG & Intent Contract (Milestone 1).
"""
import pytest
from core.schemas import (
    IntentContract,
    LogicalPlan,
    QuestionType,
    ScopeFilters,
    EntityRef,
)
from nlp.caesura_planner import CaesuraPlanner

def test_intent_contract_schema_validation():
    contract = IntentContract(
        question_type=QuestionType.COMPARISON,
        entities=[
            EntityRef(name="Stephen Curry", type="player", id=201939),
            EntityRef(name="Damian Lillard", type="player", id=203081),
        ],
        season="2023-24",
        season_type="Regular Season",
        filters=ScopeFilters(per_mode="PerGame"),
        metrics=["PTS", "3PM", "TS%"],
        interpretation="Compare scoring efficiency and perimeter volume.",
        assumptions=["Assumed Regular Season 2023-24."]
    )
    
    assert contract.question_type == QuestionType.COMPARISON
    assert len(contract.entities) == 2
    assert contract.filters.per_mode == "PerGame"

def test_caesura_plan_construction_thesis():
    planner = CaesuraPlanner()
    intent = IntentContract(
        question_type=QuestionType.THESIS,
        entities=[EntityRef(name="New York Knicks", type="team", id=1610612752)],
        season="2023-24",
        season_type="Playoffs",
        interpretation="Analyze why the Knicks achieved playoff dominance.",
    )
    
    plan = planner.build_logical_plan(intent)
    assert len(plan.steps) >= 1
    is_valid, errors = planner.validate_plan_schema(plan)
    assert is_valid is True, f"Plan schema errors: {errors}"
    for step in plan.steps:
        assert step.params.get("team_id") == 1610612752 or "team" in step.endpoint.lower()

def test_caesura_archetype_switch_gravity():
    planner = CaesuraPlanner()
    
    # 1. Curry -> Perimeter Gravity
    curry_intent = IntentContract(
        question_type=QuestionType.DEFINITION,
        entities=[EntityRef(name="Stephen Curry", type="player", id=201939)],
        season="2023-24",
        interpretation="Perimeter gravity pull and off-ball spacing."
    )
    curry_plan = planner.build_logical_plan(curry_intent)
    curry_endpoints = [s.endpoint for s in curry_plan.steps]
    assert len(curry_plan.steps) >= 1
    is_valid_c, errors_c = planner.validate_plan_schema(curry_plan)
    assert is_valid_c is True, f"Curry plan schema errors: {errors_c}"
    
    # 2. Wemby -> Interior Rim Deterrence Gravity
    wemby_intent = IntentContract(
        question_type=QuestionType.DEFINITION,
        entities=[EntityRef(name="Victor Wembanyama", type="player", id=1641705)],
        season="2023-24",
        interpretation="Interior rim deterrence and paint protection gravity."
    )
    wemby_plan = planner.build_logical_plan(wemby_intent)
    wemby_endpoints = [s.endpoint for s in wemby_plan.steps]
    assert len(wemby_plan.steps) >= 1
    is_valid_w, errors_w = planner.validate_plan_schema(wemby_plan)
    assert is_valid_w is True, f"Wemby plan schema errors: {errors_w}"

    # Curry and Wemby must NOT have identical endpoint packs
    assert curry_endpoints != wemby_endpoints

def test_caesura_plan_schema_validation_pre_execution():
    planner = CaesuraPlanner()
    
    # Valid plan
    intent = IntentContract(
        question_type=QuestionType.SINGLE,
        entities=[EntityRef(name="LeBron James", type="player", id=2544)],
        season="2023-24",
        interpretation="Single player stats."
    )
    plan = planner.build_logical_plan(intent)
    is_valid, errors = planner.validate_plan_schema(plan)
    assert is_valid is True
    assert len(errors) == 0
    
    # Invalid plan with fake endpoint
    from core.schemas import PlanStep
    plan.steps.append(PlanStep(
        step_id="S_Fake",
        endpoint="FakeNonExistentEndpoint",
        params={}
    ))
    is_valid_fake, errors_fake = planner.validate_plan_schema(plan)
    assert is_valid_fake is False
    assert any("not in verified catalogue" in err for err in errors_fake)

def test_caesura_synergy_isolation_routing():
    """Verify that isolation queries route dynamically to SynergyPlayTypes without out-of-catalogue blocker"""
    planner = CaesuraPlanner()
    intent = IntentContract(
        question_type=QuestionType.COMPARISON,
        entities=[],
        season="2023-24",
        season_type="Playoffs",
        interpretation="Analyze isolation shotmaking and points per possession in the playoffs",
        metrics=["PTS", "PPP", "POSS_PCT"],
        clarification_needed=None
    )
    plan = planner.build_logical_plan(intent)
    endpoints = [s.endpoint for s in plan.steps]
    assert "SynergyPlayTypes" in endpoints
    synergy_step = next(s for s in plan.steps if s.endpoint == "SynergyPlayTypes")
    assert synergy_step.params.get("play_type_nullable") == "Isolation"
    assert synergy_step.params.get("player_or_team_abbreviation") == "P"
    assert synergy_step.params.get("season_type_all_star") == "Playoffs"
    
    # Pre-execution schema validation must pass
    is_valid, errors = planner.validate_plan_schema(plan)
    assert is_valid is True
    assert len(errors) == 0

def test_no_washington_wizards_false_positive_on_was():
    """Verify that the word 'was' in a natural language query does NOT match Washington Wizards"""
    planner = CaesuraPlanner()
    intent = planner.extract_intent("2024 playoffs, who were the best players for their teams, and how impactful was their scoring?")
    # Check that Washington Wizards was not extracted as an entity
    entity_names = [e.name.lower() for e in intent.entities]
    assert "was" not in entity_names
    assert "washington wizards" not in entity_names
    # Macro query should not block on clarification
    assert intent.clarification_needed is None

def test_optional_params_and_fields_in_candidate_cards():
    """Verify that IntentParser exposes optional parameters, allowed enums, and return fields to prompt"""
    from router.intent_parser import IntentParser
    parser = IntentParser()
    retrieval = parser.semantic_index.retrieve("isolation shotmaking points per possession", top_k=5)
    cand_names = [r["endpoint_name"] for r in retrieval]
    assert "SynergyPlayTypes" in cand_names or "LeagueDashPlayerStats" in cand_names
    
    # Check candidate summary formatting includes optional parameters and return fields
    spec = parser.catalogue["SynergyPlayTypes"]
    opt_names = [p["name"] for p in spec.get("optional_params", [])]
    assert "play_type_nullable" in opt_names
    assert "player_or_team_abbreviation" in opt_names
    assert len(spec.get("output_fields", [])) > 0
    assert "PPP" in spec.get("output_fields", [])

