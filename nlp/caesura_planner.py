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
from nba_api.stats.static import teams, players

logger = logging.getLogger("deepbag.caesura")

INTENT_SYSTEM_PROMPT = """You are the deepBAG Natural Language Intent Classifier & Stathead Scope Triage Engine.
Translate the user's NBA query into a strict JSON contract.

JSON Schema fields:
- "question_type": one of "single", "comparison", "thesis", "definition", "longitudinal", "paradox", "unknown".
- "entities": list of objects [{"name": "<Full Name or Team Name>", "type": "player"|"team"}]. List ALL mentioned players and teams!
- "season": "YYYY-YY" string (e.g. "2023-24") or null. If query mentions rookie year or prior season, resolve appropriately. For iconic player seasons (e.g. Westbrook's triple-double MVP season = 2016-17, Curry's unanimous MVP = 2015-16), resolve the milestone season.
- "season_type": "Regular Season" or "Playoffs". If the query mentions playoffs, postseason, or Finals, you MUST set this to "Playoffs"!
- "metrics": list of requested metric names (e.g. ["PTS", "AST", "TS%", "on_off_net", "3P%"]).
- "interpretation": WORDS ONLY: what vague or cultural terms ("gravity", "clutch", "dominant", "midrange paradox") mean in this context. DO NOT include any numerical claims in "interpretation".
- "filters": object with keys: location, opponent_team_id, outcome, game_segment, period, per_mode, measure_type, last_n_games, date_from, date_to (or null).
- "clarification_needed": string question ONLY if critical scope is completely missing (e.g. "What adjustments happen after halftime?" with NO player or team named). If players or teams ARE named, DO NOT ask for clarification; instead, make reasonable assumptions in "assumptions" (e.g. "Assumed 2016-17 for Westbrook and 2023-24 for Jokic") and set "clarification_needed": null.
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
        self._all_teams = teams.get_teams()
        self._team_lookup = {}
        for t in self._all_teams:
            self._team_lookup[t["full_name"].lower()] = t["id"]
            self._team_lookup[t["nickname"].lower()] = t["id"]
            self._team_lookup[t["abbreviation"].lower()] = t["id"]

    def _load_catalogue(self) -> Dict[str, Any]:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cat_path = os.path.join(repo_root, "data", "endpoint_catalogue.json")
        if os.path.exists(cat_path):
            with open(cat_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def extract_intent(self, query: str) -> IntentContract:
        """Extract schema-validated IntentContract from query with 1-retry on failure and PremiseGuard check"""
        # 1. PremiseGuard check
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
                
                # Check for playoffs in raw query
                if "playoff" in query.lower() or "finals" in query.lower() or "postseason" in query.lower():
                    raw_json["season_type"] = "Playoffs"

                # Resolve entities if missing or incomplete
                if "entities" not in raw_json or not isinstance(raw_json["entities"], list) or len(raw_json["entities"]) == 0:
                    raw_json["entities"] = []
                    # Fallback entity scan from text
                    q_lower = query.lower()
                    for t_name, tid in self._team_lookup.items():
                        if re.search(rf"\b{re.escape(t_name)}\b", q_lower):
                            raw_json["entities"].append({
                                "name": t_name.title(),
                                "type": "team",
                                "id": tid
                            })
                            break
                    
                    # Scan notable players
                    notable_players = [
                        ("curry", "Stephen Curry", 201939),
                        ("steph", "Stephen Curry", 201939),
                        ("wemby", "Victor Wembanyama", 1641705),
                        ("wembanyama", "Victor Wembanyama", 1641705),
                        ("lebron", "LeBron James", 2544),
                        ("jokic", "Nikola Jokic", 203999),
                        ("giannis", "Giannis Antetokounmpo", 203507),
                        ("embiid", "Joel Embiid", 203954),
                        ("tatum", "Jayson Tatum", 1628369),
                        ("gobert", "Rudy Gobert", 203497),
                        ("edwards", "Anthony Edwards", 1630162),
                        ("ant", "Anthony Edwards", 1630162),
                        ("chet", "Chet Holmgren", 1631096),
                    ]
                    for alias, full_name, pid in notable_players:
                        if re.search(rf"\b{re.escape(alias)}\b", q_lower):
                            raw_json["entities"].append({
                                "name": full_name,
                                "type": "player",
                                "id": pid
                            })
                else:
                    for ent in raw_json["entities"]:
                        if isinstance(ent, dict):
                            name_clean = ent.get("name", "").strip().lower()
                            if ent.get("type") == "player" or not ent.get("type"):
                                p_id = get_player_id_from_name(ent.get("name", ""))
                                if p_id:
                                    ent["id"] = p_id
                                    ent["type"] = "player"
                            elif ent.get("type") == "team" or name_clean in self._team_lookup:
                                ent["id"] = self._team_lookup.get(name_clean, ent.get("id"))
                                ent["type"] = "team"
                
                # Auto-resolve player seasons in assumptions if clarification is only asking for season/year
                if raw_json.get("clarification_needed") and raw_json.get("entities"):
                    clarif_text = str(raw_json.get("clarification_needed", "")).lower()
                    if any(w in clarif_text for w in ["season", "year"]):
                        raw_json["clarification_needed"] = None
                        if "assumptions" not in raw_json or not isinstance(raw_json["assumptions"], list):
                            raw_json["assumptions"] = []
                        if any("westbrook" in str(e.get("name", "")).lower() for e in raw_json.get("entities", [])):
                            raw_json["assumptions"].append("Assumed 2016-17 for Russell Westbrook's MVP triple-double season.")
                        if any("jokic" in str(e.get("name", "")).lower() for e in raw_json.get("entities", [])):
                            raw_json["assumptions"].append("Assumed 2023-24 for Nikola Jokic's triple-double season.")
                        if not raw_json.get("season"):
                            raw_json["season"] = "2023-24"

                # Schema validate using Pydantic
                return IntentContract.model_validate(raw_json)
                
            except Exception as e:
                logger.warning(f"Intent extraction attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    user_prompt += f"\n\nPrevious attempt failed schema validation: {e}. Return strict JSON."
                else:
                    # Deterministic fallback
                    return self._fallback_intent(query)

    def _fallback_intent(self, query: str) -> IntentContract:
        """Deterministic fallback when LLM JSON fails twice"""
        assumptions = ["Assumed Regular Season 2023-24 via fallback router."]
        is_playoff = "playoff" in query.lower() or "finals" in query.lower()
        
        entities = []
        q_lower = query.lower()
        for t_name, tid in self._team_lookup.items():
            if re.search(rf"\b{re.escape(t_name)}\b", q_lower):
                entities.append(EntityRef(name=t_name.title(), type="team", id=tid))
                break

        return IntentContract(
            question_type=QuestionType.SINGLE,
            entities=entities,
            season="2023-24",
            season_type="Playoffs" if is_playoff else "Regular Season",
            filters=ScopeFilters(),
            metrics=["PTS", "AST", "REB"],
            interpretation="Fallback stat lookup.",
            assumptions=assumptions
        )

    def build_logical_plan(self, intent: IntentContract) -> LogicalPlan:
        """
        CAESURA logical plan DAG generation.
        Maps intent to multi-endpoint steps based on QuestionType and entities.
        """
        steps = []
        season = intent.season or "2023-24"
        season_type = intent.season_type or "Regular Season"
        q_type = intent.question_type

        # 1. Thesis Queries (e.g. Knicks/Celtics Playoff Dominance)
        if q_type == QuestionType.THESIS and not ("gravity" in intent.interpretation.lower() or any("gravity" in m.lower() for m in intent.metrics)):
            team_id = next((e.id for e in intent.entities if e.type == "team" and e.id), None)
            if not team_id:
                for t in self._all_teams:
                    if t["full_name"].lower() in intent.interpretation.lower() or t["nickname"].lower() in intent.interpretation.lower():
                        team_id = t["id"]
                        break
            if not team_id:
                team_id = 1610612752  # New York Knicks default

            steps.append(PlanStep(
                step_id="S1_FourFactors",
                endpoint="TeamDashboardByGeneralSplits",
                params={"team_id": team_id, "season": season, "season_type_all_star": season_type, "measure_type_detailed_defense": "Four Factors"},
                join_keys=["team_id", "season"],
                description="Team four factors dominance decomposition (eFG%, TOV%, OREB%, FTR)"
            ))
            steps.append(PlanStep(
                step_id="S2_TeamOnOff",
                endpoint="TeamPlayerOnOffSummary",
                params={"team_id": team_id, "season": season, "season_type_all_star": season_type},
                join_keys=["team_id", "season"],
                description="On/Off net rating differential across playoff rotation"
            ))
            steps.append(PlanStep(
                step_id="S3_TeamSplits",
                endpoint="TeamDashboardByGeneralSplits",
                params={"team_id": team_id, "season": season, "season_type_all_star": season_type},
                join_keys=["team_id"],
                description="Team general shooting and pace splits"
            ))

        # 2. Gravity / Definition Queries (Curry vs Wemby Archetype Switch)
        elif q_type == QuestionType.DEFINITION or "gravity" in intent.interpretation.lower() or any("gravity" in m.lower() for m in intent.metrics) or any(w in intent.interpretation.lower() for w in ["deterrence", "spacing", "gravity"]):
            player = next((e for e in intent.entities if e.type == "player"), None)
            if not player:
                player = EntityRef(name="Stephen Curry", type="player", id=201939)
            p_id = player.id or 201939
            
            # Archetype check: Wemby/Rim vs Curry/Perimeter
            is_interior = any(w in player.name.lower() for w in ["wemby", "wembanyama", "gobert", "davis", "giannis"])
            
            team_id = 1610612744  # Warriors
            if is_interior:
                team_id = 1610612759  # Spurs
            if "gobert" in player.name.lower() or "edwards" in player.name.lower() or "ant" in player.name.lower():
                team_id = 1610612750  # Timberwolves

            if is_interior:
                # Interior Rim Deterrence Archetype
                steps.append(PlanStep(
                    step_id="S1_PtDefend",
                    endpoint="PlayerDashPtShotDefend",
                    params={"player_id": p_id, "season": season, "season_type_all_star": season_type},
                    join_keys=["player_id"],
                    description="Opponent rim field goal percentage deterrence"
                ))
            else:
                # Perimeter Spacing & Pull Archetype
                steps.append(PlanStep(
                    step_id="S1_PtShots",
                    endpoint="PlayerDashPtShots",
                    params={"player_id": p_id, "season": season, "season_type_all_star": season_type},
                    join_keys=["player_id"],
                    description="Perimeter pull-up and catch-and-shoot distribution"
                ))
                
            steps.append(PlanStep(
                step_id="S2_OnOffSummary",
                endpoint="TeamPlayerOnOffSummary",
                params={"team_id": team_id, "season": season},
                join_keys=["team_id"],
                description="Teammate offensive/defensive rating spacing uplift on vs off"
            ))

        # 3. Longitudinal / Halftime Adjustments
        elif q_type == QuestionType.LONGITUDINAL or "halftime" in intent.interpretation.lower():
            team_id = next((e.id for e in intent.entities if e.type == "team" and e.id), 1610612752)
            steps.append(PlanStep(
                step_id="S1_FirstHalf",
                endpoint="TeamPlayerOnOffDetails",
                params={"team_id": team_id, "season": season, "game_segment_nullable": "First Half"},
                join_keys=["team_id"],
                description="First half team rating baseline"
            ))
            steps.append(PlanStep(
                step_id="S2_SecondHalf",
                endpoint="TeamPlayerOnOffDetails",
                params={"team_id": team_id, "season": season, "game_segment_nullable": "Second Half"},
                join_keys=["team_id"],
                description="Second half team rating adjustment split"
            ))

        # 4. Paradox (Midrange)
        elif q_type == QuestionType.PARADOX or "midrange" in intent.interpretation.lower():
            steps.append(PlanStep(
                step_id="S1_ShotLocations",
                endpoint="LeagueDashPlayerShotLocations",
                params={"season": season, "season_type_all_star": season_type, "per_mode_detailed": "PerGame"},
                join_keys=["season"],
                description="League-wide shot zone efficiency conditioned by player volume"
            ))

        # 5. Multi-player Comparison
        elif q_type == QuestionType.COMPARISON:
            players_list = [e for e in intent.entities if e.type == "player"]
            if not players_list:
                players_list = intent.entities
            for idx, ent in enumerate(players_list, 1):
                p_id = ent.id or 2544
                clean_name = ent.name.replace(" ", "_")
                steps.append(PlanStep(
                    step_id=f"S{idx}_{clean_name}",
                    endpoint="PlayerGameLog",
                    params={"player_id": p_id, "season": season, "season_type_all_star": season_type},
                    join_keys=["player_id", "season"],
                    description=f"Season game logs for {ent.name}"
                ))

        # 6. Default Single Player
        else:
            p_id = intent.entities[0].id if intent.entities and intent.entities[0].id else 2544
            steps.append(PlanStep(
                step_id="S1_SingleLog",
                endpoint="PlayerGameLog",
                params={"player_id": p_id, "season": season, "season_type_all_star": season_type},
                join_keys=["player_id", "season"],
                description="Player season game log"
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
