"""
Constrained LLM Intent Parser & Router Engine.
Stage 2 of the two-stage semantic router:
1. Receives Top-5 candidate endpoint schemas from SemanticEndpointIndex.
2. Prompts Nebius Nemotron 120B to generate structured Pydantic QueryPlan.
3. Validates against CodeVerifier. In case of verification error, executes 1 self-correction pass.
"""
import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from core.llm_client import UnifiedLLMClient
from catalogue.schema import PlayerMeta
from router.verifier import CodeVerifier, VerificationResult
from router.scope_receipt import ScopeReceipt, TemporalResolver
from entities.resolver import EntityResolver, ResolvedEntity

logger = logging.getLogger("deepbag.intent_parser")

class EndpointCallPlan(BaseModel):
    endpoint: str
    params: Dict[str, Any] = Field(default_factory=dict)
    tool: str = "call_nba_api"
    entity_label: Optional[str] = None

class QueryPlan(BaseModel):
    entities_raw: List[str] = Field(default_factory=list)
    temporal_expression: Optional[str] = None
    season: str = "2023-24"
    season_type: str = "Regular Season"
    endpoint_calls: List[EndpointCallPlan] = Field(default_factory=list)
    reasoning: str
    scope_receipt: Optional[ScopeReceipt] = None

    @property
    def primary_endpoint(self) -> Optional[str]:
        return self.endpoint_calls[0].endpoint if self.endpoint_calls else None

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.endpoint_calls[0].params if self.endpoint_calls else {}

STAGE2_SYSTEM_PROMPT = """You are the deepBAG Constrained Query Plan Compiler.
You must construct an executable NBA statistics QueryPlan using ONLY the provided candidate endpoints.

JSON Output Schema:
{
  "entities_raw": ["<Raw Entity Name 1>", ...],
  "temporal_expression": "<Extracted temporal phrase, e.g. 'rookie year', 'last 5 games', '2023-24', or null>",
  "season": "<YYYY-YY format, e.g. '2023-24'>",
  "season_type": "'Regular Season' or 'Playoffs'",
  "endpoint_calls": [
    {
      "endpoint": "<Candidate Endpoint Name>",
      "params": {
        "<param_name>": "<param_value>"
      },
      "entity_label": "<Player or Team Name>"
    }
  ],
  "reasoning": "<Short analytical rationale for endpoint selection>"
}

RULES:
1. You may ONLY select endpoints from the Candidate Endpoints list provided in the prompt.
2. For multi-player comparisons, emit an endpoint call for EACH player (or a single league-wide call).
3. Set season_type to "Playoffs" if query mentions playoffs, postseason, or Finals.
4. When a query asks about a player's performance against an opponent team (e.g. 'vs Celtics'), query the player's endpoint (e.g. PlayerGameLog) with the player; do NOT emit an unjoinable team endpoint call.
5. Prefer PlayerGameLog for individual player game performances, matchups, and game-level splits.
6. Output valid JSON only.
"""

class IntentParser:
    def __init__(
        self,
        catalogue: Optional[Dict[str, Dict[str, Any]]] = None,
        entity_resolver: Optional[EntityResolver] = None,
        llm_client: Optional[UnifiedLLMClient] = None
    ):
        if catalogue is None:
            cat_path = os.path.join(os.path.dirname(__file__), "..", "data", "endpoint_catalogue_v2.json")
            if os.path.exists(cat_path):
                with open(cat_path, "r", encoding="utf-8") as f:
                    self.catalogue = json.load(f)
            else:
                self.catalogue = {}
        else:
            self.catalogue = catalogue

        self.entity_resolver = entity_resolver or EntityResolver()
        self.llm = llm_client or UnifiedLLMClient(provider="nebius")
        self.verifier = CodeVerifier(catalogue=self.catalogue)
        self._semantic_index = None

    @property
    def semantic_index(self):
        if self._semantic_index is None:
            from router.semantic_index import SemanticEndpointIndex
            self._semantic_index = SemanticEndpointIndex()
        return self._semantic_index

    def parse_intent(
        self,
        query: str,
        candidate_endpoints: Optional[List[str]] = None
    ) -> QueryPlan:
        if not candidate_endpoints:
            retrieval = self.semantic_index.retrieve(query, top_k=5)
            candidate_endpoints = [r["endpoint_name"] for r in retrieval]
        # 1. Format candidate endpoint summaries
        cand_summaries = []
        for name in candidate_endpoints:
            spec = self.catalogue.get(name, {})
            req_params = [p.get("name") for p in spec.get("required_params", [])]
            cand_summaries.append(
                f"- Endpoint: {name}\n"
                f"  Capability: {spec.get('capability_card', '')}\n"
                f"  Required Params: {req_params}"
            )

        candidate_block = "\n\n".join(cand_summaries)
        user_prompt = f"""User Query: "{query}"

Candidate Endpoints (You MUST select from this pool):
{candidate_block}
"""

        # 2. First attempt with Nebius Nemotron
        plan_dict = self._call_llm_json(user_prompt)
        resolved_plan = self._enrich_and_verify(query, plan_dict)

        # 3. Check verification; if compiler error, run 1 self-correction pass
        if not resolved_plan.get("is_valid", False):
            compiler_err = resolved_plan.get("error_message", "Unknown verification error")
            logger.warning(f"Verification failed on attempt 1: {compiler_err}. Executing self-correction pass...")
            retry_prompt = user_prompt + f"\n\nCOMPILER ERROR IN PREVIOUS ATTEMPT: {compiler_err}\nPlease correct the endpoint calls and parameters to resolve this error."
            plan_dict = self._call_llm_json(retry_prompt)
            resolved_plan = self._enrich_and_verify(query, plan_dict)

        return resolved_plan.get("plan")

    def _call_llm_json(self, prompt: str) -> Dict[str, Any]:
        resp = self.llm.generate(
            user_prompt=prompt,
            system_prompt=STAGE2_SYSTEM_PROMPT,
            temperature=0.1
        )
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}

    def _enrich_and_verify(self, query: str, raw_plan: Dict[str, Any]) -> Dict[str, Any]:
        # Resolve entities
        raw_entities = raw_plan.get("entities_raw", [])
        resolved_entities: List[ResolvedEntity] = []
        players_meta: List[PlayerMeta] = []

        # Also extract any entities found directly from query if LLM missed them
        for raw_ent in raw_entities:
            res = self.entity_resolver.resolve(raw_ent, season_hint=raw_plan.get("season"))
            if res:
                resolved_entities.append(res)
                if res.type == "player":
                    players_meta.append(PlayerMeta(
                        id=res.id,
                        full_name=res.name,
                        from_year=res.from_year or 1946,
                        to_year=res.to_year or 2024
                    ))

        # Temporal Resolution
        first_player_from = players_meta[0].from_year if players_meta else None
        temporal_info = TemporalResolver.resolve_temporal_ref(
            query=raw_plan.get("temporal_expression") or query,
            player_from_year=first_player_from
        )
        resolved_season = temporal_info["season"]
        season_type = raw_plan.get("season_type", "Regular Season")

        # Scope Receipt construction
        assumptions = []
        if temporal_info.get("is_assumed") and temporal_info.get("assumption_text"):
            assumptions.append(temporal_info["assumption_text"])

        scope_receipt = ScopeReceipt(
            season=resolved_season,
            season_type=season_type,
            entities=[{"name": e.name, "id": e.id, "type": e.type} for e in resolved_entities],
            assumptions=assumptions
        )

        # Build endpoint calls
        calls: List[EndpointCallPlan] = []
        for c in raw_plan.get("endpoint_calls", []):
            ep_name = c.get("endpoint")
            params = c.get("params", {})
            if "season" not in params:
                params["season"] = resolved_season
            if "season_type_all_star" not in params and "season_type" in c:
                params["season_type_all_star"] = season_type

            # Inject resolved player_id / team_id if missing
            if resolved_entities:
                for ent in resolved_entities:
                    if ent.type == "player" and "player_id" not in params:
                        params["player_id"] = ent.id
                    elif ent.type == "team" and "team_id" not in params:
                        params["team_id"] = ent.id

            calls.append(EndpointCallPlan(
                endpoint=ep_name,
                params=params,
                entity_label=c.get("entity_label")
            ))

        # Run CodeVerifier
        v_result: VerificationResult = self.verifier.verify_plan(
            endpoint_calls=[c.model_dump() for c in calls],
            players_meta=players_meta,
            season=resolved_season
        )

        # Build final QueryPlan
        plan = QueryPlan(
            entities_raw=raw_entities,
            temporal_expression=raw_plan.get("temporal_expression"),
            season=resolved_season,
            season_type=season_type,
            endpoint_calls=calls,
            reasoning=raw_plan.get("reasoning", "Synthesized via Nebius Nemotron"),
            scope_receipt=scope_receipt
        )

        return {
            "is_valid": v_result.is_valid,
            "error_message": v_result.error_message,
            "plan": plan
        }
