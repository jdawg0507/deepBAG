"""
Unit tests for CAESURA Routing DAG & Intent Contract (Milestone 1).
"""
import pytest
from unittest.mock import MagicMock
from core.schemas import (
    IntentContract,
    LogicalPlan,
    PlanStep,
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


def test_no_silent_scope_substitution_on_empty_playoffs():
    """Verify that empty playoff results fail closed and do not silently substitute Regular Season"""
    from unittest.mock import MagicMock
    from core.schemas import LogicalPlan, PlanStep, IntentContract, QuestionType
    
    planner = CaesuraPlanner()
    intent = IntentContract(
        question_type=QuestionType.SINGLE,
        entities=[],
        season="2023-24",
        season_type="Playoffs",
        interpretation="Playoff test"
    )
    plan = LogicalPlan(
        intent=intent,
        steps=[
            PlanStep(
                step_id="S1_Playoffs",
                endpoint="PlayerDashboardByGeneralSplits",
                params={"player_id": 201939, "season_type_playoffs": "Playoffs"},
                join_keys=["player_id"],
                required=True
            )
        ]
    )
    
    mock_client = MagicMock()
    mock_client.call.return_value = {"status": "success", "datasets": {"PlayerDashboard": []}}
    
    table, results = planner.execute_plan(plan, mock_client)
    
    assert "error" in results
    assert "Required gate failed" in results["error"]
    assert plan.steps[0].params["season_type_playoffs"] == "Playoffs"
    # Verify mock_client was never called with Regular Season
    for call_args in mock_client.call.call_args_list:
        args, kwargs = call_args
        called_params = args[1] if len(args) > 1 else kwargs.get("params", {})
        assert called_params.get("season_type_playoffs") != "Regular Season"


def test_caesura_dag_backtracking_on_endpoint_failure():
    """Verify that when a primary step returns 0 rows or errors, the planner backtracks to fallback candidate."""
    planner = CaesuraPlanner()
    intent = IntentContract(
        question_type=QuestionType.SINGLE,
        entities=[EntityRef(name="Stephen Curry", type="player", id=201939)],
        season="2023-24",
        interpretation="Curry stats with fallback recovery.",
    )
    plan = LogicalPlan(
        intent=intent,
        steps=[
            PlanStep(
                step_id="S1_Primary",
                endpoint="PlayerDashboardByGeneralSplits",
                params={"player_id": 201939, "season": "2023-24"},
                join_keys=["player_id"],
                required=True
            )
        ],
        fallback_candidates=["PlayerCareerStats"]
    )
    
    mock_client = MagicMock()
    # First call (Primary) fails; second call (Fallback PlayerCareerStats) succeeds
    mock_client.call.side_effect = [
        {"status": "error", "error": "Upstream timeout", "datasets": {}},
        {
            "status": "success",
            "datasets": {
                "CareerTotalsRegularSeason": [
                    {"PLAYER_ID": 201939, "PTS": 23668, "AST": 6119, "GP": 956}
                ]
            }
        }
    ]
    
    table, results = planner.execute_plan(plan, mock_client)
    
    assert "error" not in results
    assert "Fallback_PlayerCareerStats" in results
    # Check that fallback was invoked
    assert mock_client.call.call_count == 2
    second_call = mock_client.call.call_args_list[1]
    assert second_call[0][0] == "PlayerCareerStats"
    # Evidence claims should be registered from fallback
    assert len(table.all_claims()) > 0
    # Provenance assumption should be noted
    assert any("dynamically backtracked" in a for a in intent.assumptions)


