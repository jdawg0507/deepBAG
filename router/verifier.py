"""
Deterministic Code Verifier for deepBAG.
Validates QueryPlan constraints:
1. Endpoint presence in Catalogue
2. Parameter types and enum constraints
3. Multi-tool joinability (join_keys intersection)
4. Player career bounds & tracking era enforcement via resolve_data_source()
"""
from typing import Dict, Any, List, Optional, Tuple, Set
from pydantic import BaseModel, Field

from catalogue.schema import EndpointSpec, PlayerMeta, resolve_data_source, CareerBoundsError, DataSource

class VerificationResult(BaseModel):
    is_valid: bool
    error_message: Optional[str] = None
    data_sources: Dict[str, DataSource] = Field(default_factory=dict)

class CodeVerifier:
    def __init__(self, catalogue: Dict[str, Dict[str, Any]]):
        self.catalogue = catalogue

    def verify_plan(
        self,
        endpoint_calls: List[Dict[str, Any]],
        players_meta: Optional[List[PlayerMeta]] = None,
        season: Optional[str] = None
    ) -> VerificationResult:
        if not endpoint_calls:
            return VerificationResult(is_valid=False, error_message="QueryPlan contains zero endpoint calls.")

        data_sources = {}
        join_keys_list: List[Set[str]] = []

        # 1. Validate each endpoint individually
        for call in endpoint_calls:
            ep_name = call.get("endpoint")
            if not ep_name or ep_name not in self.catalogue:
                return VerificationResult(
                    is_valid=False,
                    error_message=f"Endpoint '{ep_name}' is not registered in the Endpoint Catalogue."
                )

            spec_dict = self.catalogue[ep_name]
            spec = EndpointSpec.model_validate(spec_dict)
            join_keys_list.append(set(spec.join_keys))

            params = call.get("params", {})
            
            # Check required parameters
            for req in spec.required_params:
                if req.name not in params and req.default is None:
                    return VerificationResult(
                        is_valid=False,
                        error_message=f"Endpoint '{ep_name}' missing required parameter '{req.name}'."
                    )

            # Check parameter allowed values if constrained
            for opt in spec.optional_params:
                if opt.name in params and opt.allowed_values:
                    val = params[opt.name]
                    if val not in opt.allowed_values:
                        return VerificationResult(
                            is_valid=False,
                            error_message=f"Parameter '{opt.name}' value '{val}' not in allowed values: {opt.allowed_values}."
                        )

            # Check era and career bounds
            call_season = params.get("season") or season or "2023-24"
            target_player = None
            if players_meta:
                # Find corresponding player if call has player_id
                pid = params.get("player_id")
                if pid:
                    for p in players_meta:
                        if p.id == pid:
                            target_player = p
                            break
                if not target_player and len(players_meta) == 1:
                    target_player = players_meta[0]

            try:
                ds = resolve_data_source(target_player, call_season, spec)
                data_sources[ep_name] = ds
                if ds == DataSource.UNSUPPORTED_ERA:
                    return VerificationResult(
                        is_valid=False,
                        error_message=f"Query requests season '{call_season}' on '{ep_name}' which only supports seasons >= '{spec.min_season}' with no historical database fallback."
                    )
            except CareerBoundsError as cbe:
                return VerificationResult(
                    is_valid=False,
                    error_message=str(cbe)
                )

        # 2. Multi-endpoint joinability check
        if len(endpoint_calls) > 1:
            common_join_keys = set.intersection(*join_keys_list) if join_keys_list else set()
            if not common_join_keys:
                ep_names = [c.get("endpoint") for c in endpoint_calls]
                return VerificationResult(
                    is_valid=False,
                    error_message=f"Multi-tool plan endpoints {ep_names} do not share common join keys: {[list(s) for s in join_keys_list]}."
                )

        return VerificationResult(is_valid=True, data_sources=data_sources)
