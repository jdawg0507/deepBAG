"""
CAESURA Routing Loop & Ponder Gate Tree Execution Planner for deepBAG.
Implements:
1. Intent-First JSON Contract extraction via UnifiedLLMClient
2. PremiseGuard adversarial and trap detection
3. Schema Validation BEFORE execution against data/endpoint_catalogue.json
4. Logical Plan DAG Generation (endpoints, parameters, join keys)
5. Incremental Step Execution with deterministic verification gates
6. Backtracking / leaf retry on empty results
"""
import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from core.schemas import (
    IntentContract,
    LogicalPlan,
    PlanStep,
    QuestionType,
    ScopeFilters,
    EntityRef,
)
from core.llm_client import UnifiedLLMClient
from core.nba_client import NBAClient
from core.evidence import EvidenceTable
from core.premise_guard import PremiseGuard
from api.parameter_utils import get_player_id_from_name
from entities.resolver import EntityResolver
from router.semantic_index import SemanticEndpointIndex
from router.intent_parser import IntentParser

logger = logging.getLogger("deepbag.caesura")

INTENT_SYSTEM_PROMPT = """You are the deepBAG Natural Language Intent Classifier & Stathead Scope Triage Engine.
Translate the user's NBA query into a strict JSON contract.

JSON Schema fields:
- "question_type": one of "single", "comparison", "thesis", "definition", "longitudinal", "paradox", "unknown".
- "entities": list of objects [{"name": "<Full Name or Team Name>", "type": "player"|"team"}]. List ALL mentioned players and teams! If the user query is a general/macro leaderboard question asking "who was the best", "which players were top", leave "entities": []!
- "season": "YYYY-YY" string (e.g. "2023-24") or null. If query mentions rookie year or prior season, resolve appropriately. For iconic player seasons (e.g. Westbrook's triple-double MVP season = 2016-17, Curry's unanimous MVP = 2015-16), resolve the milestone season.
- "season_type": "Regular Season" or "Playoffs". If the query mentions playoffs, postseason, or Finals, you MUST set this to "Playoffs"!
- "metrics": list of requested metric names (e.g. ["PTS", "AST", "TS%", "PPP", "POSS_PCT", "3P%"]).
- "interpretation": WORDS ONLY: what vague or cultural terms ("gravity", "clutch", "dominant", "midrange paradox", "iso shotmaking") mean in this context. DO NOT include any numerical claims in "interpretation".
- "filters": object with keys: location, opponent_team_id, outcome, game_segment, period, per_mode, measure_type, last_n_games, date_from, date_to (or null).
- "clarification_needed": string question ONLY if critical scope is completely missing and cannot be inferred. For general comparison or leaderboard questions ("who were the best players"), DO NOT ask for clarification; make reasonable assumptions in "assumptions" and set "clarification_needed": null.
- "assumptions": list of transparent scope decisions made for the user (e.g. ["Assumed Regular Season 2023-24"]).

Return ONLY a valid JSON object matching the contract schema.
"""

class CaesuraPlanner:
    def __init__(
        self,
        llm_client: Optional[UnifiedLLMClient] = None,
        catalogue: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm_client or UnifiedLLMClient()
        self.catalogue = catalogue or self._load_catalogue()
        self.premise_guard = PremiseGuard()
        self.entity_resolver = EntityResolver()
        self.semantic_index = SemanticEndpointIndex(catalogue_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "endpoint_catalogue_v2.json"
        ))
        self.intent_parser = IntentParser(
            catalogue=self.catalogue,
            entity_resolver=self.entity_resolver,
            llm_client=self.llm
        )

    def _load_catalogue(self) -> Dict[str, Any]:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cat_v2_path = os.path.join(repo_root, "data", "endpoint_catalogue_v2.json")
        if os.path.exists(cat_v2_path):
            with open(cat_v2_path, "r", encoding="utf-8") as f:
                return json.load(f)
        cat_path = os.path.join(repo_root, "data", "endpoint_catalogue.json")
        if os.path.exists(cat_path):
            with open(cat_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def extract_intent(self, query: str) -> IntentContract:
        """Extract schema-validated IntentContract from query using LLM and EntityResolver"""
        # 1. PremiseGuard check against adversarial traps
        premise_res = self.premise_guard.check_query(query)
        if not premise_res.is_valid:
            return IntentContract(
                question_type=QuestionType.UNKNOWN,
                entities=[],
                interpretation=f"Premise check: {premise_res.trap_type}",
                clarification_needed=premise_res.clarification_needed or premise_res.correction,
                assumptions=[]
            )

        user_prompt = f"User Query: {query}\n\nEmit the exact IntentContract JSON object."
        
        for attempt in range(2):
            try:
                raw_json = self.llm.generate_json(
                    user_prompt=user_prompt,
                    system_prompt=INTENT_SYSTEM_PROMPT,
                    temperature=0.1
                )

                # Resolve extracted entities via EntityResolver
                resolved_entities_list = []
                for ent in raw_json.get("entities", []):
                    if isinstance(ent, dict):
                        ent_name = ent.get("name", "").strip()
                        resolved = self.entity_resolver.resolve(ent_name, season_hint=raw_json.get("season"))
                        if resolved:
                            resolved_entities_list.append({
                                "name": resolved.name,
                                "type": resolved.type,
                                "id": resolved.id
                            })
                        else:
                            resolved_entities_list.append(ent)
                raw_json["entities"] = resolved_entities_list

                # Schema validate using Pydantic
                return IntentContract.model_validate(raw_json)
                
            except Exception as e:
                logger.warning(f"Intent extraction attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    user_prompt += f"\n\nPrevious attempt failed schema validation: {e}. Return strict JSON."
                else:
                    return IntentContract(
                        question_type=QuestionType.UNKNOWN,
                        entities=[],
                        interpretation="Failed to parse query intent into a valid contract.",
                        clarification_needed=f"Query intent parsing failed schema validation: {e}. Please specify player/team and metric scope.",
                        assumptions=[]
                    )

    def build_logical_plan(self, intent: IntentContract, query: Optional[str] = None) -> LogicalPlan:
        """
        Dynamically constructs an executable CAESURA LogicalPlan using dense vector retrieval
        and schema-driven LLM compilation. ZERO hardcoded if/elif endpoint switches.
        """
        search_query = query or intent.interpretation or "NBA statistics"
        if intent.metrics:
            search_query += " " + " ".join(intent.metrics)
        if intent.entities:
            search_query += " " + " ".join([e.name for e in intent.entities])

        # 1. Pure dense vector candidate retrieval
        retrieval = self.semantic_index.retrieve(search_query, top_k=8)
        candidate_names = [r["endpoint_name"] for r in retrieval]

        # 2. Compile query plan via LLM using candidate tool schemas
        query_plan = self.intent_parser.parse_intent(
            query=search_query,
            candidate_endpoints=candidate_names
        )

        steps = []
        if query_plan and query_plan.endpoint_calls:
            for idx, call in enumerate(query_plan.endpoint_calls, 1):
                spec = self.catalogue.get(call.endpoint, {})
                join_keys = spec.get("join_keys", ["season"])
                steps.append(PlanStep(
                    step_id=f"S{idx}_{call.endpoint}",
                    endpoint=call.endpoint,
                    params=call.params,
                    join_keys=join_keys,
                    description=f"Execute {call.endpoint} for {call.entity_label or 'league'}"
                ))

        return LogicalPlan(
            intent=intent,
            steps=steps,
            target_claims=intent.metrics
        )

    def validate_plan_schema(self, plan: LogicalPlan) -> Tuple[bool, List[str]]:
        """
        Schema validation BEFORE execution:
        Every endpoint exists in catalogue, and required parameters are present.
        """
        errors = []
        for step in plan.steps:
            if step.endpoint not in self.catalogue:
                errors.append(f"Step '{step.step_id}': Endpoint '{step.endpoint}' not in verified catalogue.")
                continue
                
            ep_meta = self.catalogue[step.endpoint]
            req_params = ep_meta.get("required_parameters", [])
            for req_p in req_params:
                if req_p not in step.params and req_p.lower() not in [k.lower() for k in step.params.keys()]:
                    errors.append(
                        f"Step '{step.step_id}' ({step.endpoint}): Missing required parameter '{req_p}'."
                    )
                    
        return (len(errors) == 0), errors

    def execute_plan(
        self,
        plan: LogicalPlan,
        client: NBAClient,
        evidence_table: Optional[EvidenceTable] = None,
    ) -> Tuple[EvidenceTable, Dict[str, Any]]:
        """
        Ponder Gate Tree Incremental Execution.
        Executes step-by-step; verifies leaf nodes, backtracks on empty results.
        """
        table = evidence_table or EvidenceTable()
        execution_results = {}
        
        # Validate schema first
        is_valid, validation_errors = self.validate_plan_schema(plan)
        if not is_valid:
            logger.error(f"Plan failed pre-execution validation: {validation_errors}")
            return table, {"error": "Pre-execution plan validation failed", "details": validation_errors}

        for step in plan.steps:
            logger.info(f"Executing step {step.step_id} -> {step.endpoint} with params {step.params}")
            res = client.call(step.endpoint, step.params, use_cache=True)
            
            # Check gate: rows returned?
            total_rows = sum(len(rows) for rows in res.get("datasets", {}).values())
            
            if res.get("status") != "success" or total_rows == 0:
                logger.warning(f"Step {step.step_id} produced 0 rows or error: {res.get('error')}")
                # Leaf backtracking / retry with widened filter
                if "season_type_playoffs" in step.params or "season_type_all_star" in step.params:
                    logger.info(f"Backtracking: retry step {step.step_id} with Regular Season...")
                    fallback_params = dict(step.params)
                    for k in ("season_type_playoffs", "season_type_all_star"):
                        if k in fallback_params:
                            fallback_params[k] = "Regular Season"
                    res = client.call(step.endpoint, fallback_params, use_cache=True)
                    total_rows = sum(len(rows) for rows in res.get("datasets", {}).values())
                    
                if total_rows == 0 and step.required:
                    logger.error(f"Required gate failed for step {step.step_id}. Failing fast.")
                    execution_results[step.step_id] = {"status": "gate_failed", "step": step.model_dump()}
                    return table, {"error": f"Required gate failed at {step.step_id}", "step": step.step_id}

            execution_results[step.step_id] = res

        return table, execution_results
