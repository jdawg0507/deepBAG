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
    assert len(plan.steps) >= 3
    endpoints = [s.endpoint for s in plan.steps]
    assert "TeamDashboardByGeneralSplits" in endpoints
    assert "TeamPlayerOnOffSummary" in endpoints
    assert "TeamDashboardByGeneralSplits" in endpoints

def test_caesura_archetype_switch_gravity():
    planner = CaesuraPlanner()
    
    # 1. Curry -> Perimeter Gravity
    curry_intent = IntentContract(
        question_type=QuestionType.DEFINITION,
        entities=[EntityRef(name="Stephen Curry", type="player", id=201939)],
        interpretation="Perimeter gravity pull and off-ball spacing."
    )
    curry_plan = planner.build_logical_plan(curry_intent)
    curry_endpoints = [s.endpoint for s in curry_plan.steps]
    assert "PlayerDashPtShots" in curry_endpoints
    
    # 2. Wemby -> Interior Rim Deterrence Gravity
    wemby_intent = IntentContract(
        question_type=QuestionType.DEFINITION,
        entities=[EntityRef(name="Victor Wembanyama", type="player", id=1641705)],
        interpretation="Interior rim deterrence and paint protection gravity."
    )
    wemby_plan = planner.build_logical_plan(wemby_intent)
    wemby_endpoints = [s.endpoint for s in wemby_plan.steps]
    assert "PlayerDashPtShotDefend" in wemby_endpoints
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
