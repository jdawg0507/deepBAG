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
- "season": "YYYY-YY" string (e.g. "2025-26") or null. If no specific season, date, or milestone year is mentioned, you MUST set "season": null to query career totals and career profiles!
- "season_type": "Regular Season" or "Playoffs". If the query mentions playoffs, postseason, or Finals, you MUST set this to "Playoffs"!
- "metrics": list of requested metric names (e.g. ["PTS", "AST", "TS%", "PPP", "POSS_PCT", "3P%"]).
- "interpretation": WORDS ONLY: what vague or cultural terms ("gravity", "clutch", "dominant", "midrange paradox", "iso shotmaking") mean in this context. DO NOT include any numerical claims in "interpretation".
- "filters": object with keys: location, opponent_team_id, outcome, game_segment, period, per_mode, measure_type, last_n_games, date_from, date_to (or null).
- "clarification_needed": string question ONLY if critical scope is completely missing and cannot be inferred. For general comparison or leaderboard questions ("who were the best players"), DO NOT ask for clarification; make reasonable assumptions in "assumptions" and set "clarification_needed": null.
- "assumptions": list of transparent scope decisions (e.g. ["Defaulted to career statistics as no season was specified"]). Do NOT assume a specific single season (such as 2023-24) if none was mentioned!

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
        if intent.season is None and any(e.type == "player" for e in intent.entities):
            search_query += " career stats profile totals all-time"

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

        # Model's best-guess fallbacks from the candidate pool
        fallback_candidates = []
        if query_plan and query_plan.fallback_candidates:
            fallback_candidates = [fb for fb in query_plan.fallback_candidates if fb in candidate_names]
        if not fallback_candidates:
            primary_eps = {s.endpoint for s in steps}
            fallback_candidates = [c for c in candidate_names if c not in primary_eps]

        return LogicalPlan(
            intent=intent,
            steps=steps,
            target_claims=intent.metrics,
            fallback_candidates=fallback_candidates
        )

    def get_fallback_step(
        self,
        failed_step: PlanStep,
        plan: LogicalPlan,
        tried_endpoints: Set[str]
    ) -> Optional[PlanStep]:
        """
        Dynamically finds an alternative candidate endpoint from fallback_candidates
        that matches the entity grain and parameter requirements.
        """
        player_id = failed_step.params.get("player_id")
        if not player_id and plan.intent.entities:
            p_ent = next((e for e in plan.intent.entities if e.type == "player"), None)
            if p_ent:
                player_id = p_ent.id

        team_id = failed_step.params.get("team_id")
        if not team_id and plan.intent.entities:
            t_ent = next((e for e in plan.intent.entities if e.type == "team"), None)
            if t_ent:
                team_id = t_ent.id

        season = failed_step.params.get("season") or plan.intent.season
        season_type = failed_step.params.get("season_type_all_star") or failed_step.params.get("season_type_playoffs") or plan.intent.season_type or "Regular Season"

        for cand_name in plan.fallback_candidates:
            if cand_name in tried_endpoints or cand_name == failed_step.endpoint:
                continue
            spec = self.catalogue.get(cand_name)
            if not spec:
                continue

            raw_req = spec.get("required_params", []) or spec.get("required_parameters", [])
            raw_opt = spec.get("optional_params", []) or spec.get("optional_parameters", [])
            req_param_names = [p.get("name") if isinstance(p, dict) else p for p in raw_req]
            opt_param_names = [p.get("name") if isinstance(p, dict) else p for p in raw_opt]
            all_param_names = req_param_names + opt_param_names

            # Entity grain constraint
            if player_id and "player_id" not in all_param_names:
                continue
            if "player_id" in req_param_names and not player_id:
                continue

            if team_id and not player_id and "team_id" not in all_param_names:
                continue
            if "team_id" in req_param_names and not team_id:
                continue

            params = {}
            if player_id and "player_id" in all_param_names:
                params["player_id"] = player_id
            if team_id and "team_id" in all_param_names:
                params["team_id"] = team_id

            for s_key in ("season", "season_nullable"):
                if s_key in all_param_names:
                    if season:
                        params[s_key] = season
                    elif s_key in req_param_names:
                        params[s_key] = "2026-27"

            for st_key in ("season_type_all_star", "season_type_playoffs", "season_type_nullable"):
                if st_key in all_param_names:
                    params[st_key] = season_type

            # Check per_mode if present in original params
            if "per_mode" in failed_step.params or "per_mode_detailed" in failed_step.params:
                val = failed_step.params.get("per_mode") or failed_step.params.get("per_mode_detailed")
                if "per_mode" in all_param_names:
                    params["per_mode"] = val
                elif "per_mode_detailed" in all_param_names:
                    params["per_mode_detailed"] = val

            # Check all required params are satisfied
            if all(req_p in params for req_p in req_param_names):
                join_keys = spec.get("join_keys", ["player_id"] if player_id else ["team_id"])
                return PlanStep(
                    step_id=f"Fallback_{cand_name}",
                    endpoint=cand_name,
                    params=params,
                    join_keys=join_keys,
                    description=f"Fallback execution of {cand_name} replacing {failed_step.endpoint}"
                )
        return None

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
            raw_req = ep_meta.get("required_params", []) or ep_meta.get("required_parameters", [])
            req_params = [p.get("name") if isinstance(p, dict) else p for p in raw_req]
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
            total_rows = sum(len(rows) for rows in res.get("datasets", {}).values()) if res.get("datasets") else 0
            
            if res.get("status") != "success" or total_rows == 0:
                logger.warning(f"Step {step.step_id} produced 0 rows or error: {res.get('error')}. Backtracking DAG to fallbacks...")
                tried = {step.endpoint}
                while True:
                    fallback_step = self.get_fallback_step(step, plan, tried)
                    if not fallback_step:
                        break
                    tried.add(fallback_step.endpoint)
                    logger.info(f"Traversing DAG fallback: trying {fallback_step.endpoint} with params {fallback_step.params}")
                    fb_res = client.call(fallback_step.endpoint, fallback_step.params, use_cache=True)
                    fb_datasets = fb_res.get("datasets", {})
                    fb_rows = sum(len(rows) for rows in fb_datasets.values()) if fb_datasets else 0
                    if fb_res.get("status") == "success" and fb_rows > 0:
                        logger.info(f"Fallback {fallback_step.endpoint} succeeded with {fb_rows} rows.")
                        res = fb_res
                        step = fallback_step
                        total_rows = fb_rows
                        if plan.intent and hasattr(plan.intent, "assumptions"):
                            plan.intent.assumptions.append(
                                f"Primary endpoint was unavailable; dynamically backtracked to {fallback_step.endpoint}."
                            )
                        break

                if (res.get("status") != "success" or total_rows == 0) and step.required:
                    logger.error(f"Required gate failed for step {step.step_id}. Failing fast.")
                    execution_results[step.step_id] = {
                        "status": "gate_failed",
                        "step": step.model_dump(),
                        "reason": f"No data found matching step parameters {step.params}"
                    }
                    return table, {"error": f"Required gate failed at {step.step_id}", "step": step.step_id}

            execution_results[step.step_id] = res
            if total_rows > 0 and res.get("datasets"):
                step_entity = step.step_id.split("_", 1)[1] if "_" in step.step_id else None
                table.register_dataset_claims(
                    endpoint=step.endpoint,
                    params=step.params,
                    datasets=res.get("datasets", {}),
                    entity_hint=step_entity
                )

        return table, execution_results
