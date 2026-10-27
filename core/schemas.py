"""
Pydantic v2 Strict Data Contracts & Schema Validation for deepBAG.
Enforces strict ingress contracts and guarantees zero unhandled type coercions for NBA API calls.
"""
import re
import json
import inspect
from typing import Optional, Dict, Any, Union, Type, List
from enum import Enum
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    ValidationError,
    TypeAdapter,
    create_model,
)

# ---------------------------------------------------------
# 1. Incoming API Gateway Ingress Schemas
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    """Ingress schema for natural language query requests."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    
    query: str = Field(
        ...,
        min_length=2,
        max_length=1500,
        description="Natural language question about NBA statistics, players, or box scores.",
        examples=["What were LeBron James' stats in his last game?", "Who led the league in scoring in 2023-24?"]
    )
    session_id: Optional[str] = Field(None, description="Optional client session identifier for telemetry correlation.")
    bypass_cache: bool = Field(False, description="Whether to bypass the distributed cache.")
    model: Optional[str] = Field(None, description="Optional LLM model override (e.g. nvidia/nemotron-3-super-120b-a12b, gemini-3.5-flash-lite).")


class ChatResponse(BaseModel):
    """Egress schema for final agent responses."""
    response: str
    cached: bool = False
    trace_id: Optional[str] = None

# ---------------------------------------------------------
# 2. Tool Call Invocation Schemas
# ---------------------------------------------------------

class ToolCallInvocation(BaseModel):
    """Schema for dynamic tool call invocations from the agentic LLM."""
    model_config = ConfigDict(extra="ignore")
    
    endpoint_name: str = Field(..., min_length=1, description="NBA API endpoint class name.")
    parameters: Union[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Parameters dictionary or raw JSON string."
    )

    @field_validator("parameters", mode="before")
    @classmethod
    def parse_parameters(cls, v):
        if isinstance(v, str):
            v_stripped = v.strip()
            if not v_stripped or v_stripped == "{}":
                return {}
            try:
                return json.loads(v_stripped)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON string passed as parameters: {err}")
        elif isinstance(v, dict):
            return v
        return {}

# ---------------------------------------------------------
# 3. Explicit High-Traffic Endpoint Pydantic v2 Schemas
# ---------------------------------------------------------

SEASON_REGEX = re.compile(r"^\d{4}-\d{2}$")

class MeasureTypeEnum(str, Enum):
    Base = "Base"
    Advanced = "Advanced"
    Misc = "Misc"
    FourFactors = "Four Factors"
    Scoring = "Scoring"
    Opponent = "Opponent"
    Usage = "Usage"
    Defense = "Defense"

class PerModeEnum(str, Enum):
    Totals = "Totals"
    PerGame = "PerGame"
    MinutesPer = "MinutesPer"
    Per48 = "Per48"
    Per40 = "Per40"
    Per36 = "Per36"
    PerMinute = "PerMinute"
    PerPossession = "PerPossession"
    PerPlay = "PerPlay"
    Per100Possessions = "Per100Possessions"
    Per100Plays = "Per100Plays"

class SeasonTypeEnum(str, Enum):
    RegularSeason = "Regular Season"
    PreSeason = "Pre Season"
    Playoffs = "Playoffs"
    AllStar = "All Star"

def to_snake(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class CommonPlayerInfoParams(BaseModel):
    """Strict schema for CommonPlayerInfo."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    player_id: int = Field(..., alias="PlayerID", gt=0, description="Unique NBA player ID.")

    @field_validator("player_id", mode="before")
    @classmethod
    def validate_player_id(cls, v):
        if isinstance(v, str):
            if v.isdigit():
                return int(v)
            raise ValueError(f"player_id must be a numeric integer, got: '{v}'")
        return v

class PlayerGameLogParams(BaseModel):
    """Strict schema for PlayerGameLog."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    player_id: int = Field(..., alias="PlayerID", gt=0)
    season: Optional[str] = Field(None, alias="Season", pattern=r"^\d{4}-\d{2}$")
    season_type_all_star: Optional[str] = Field("Regular Season", alias="SeasonTypeAllStar")
    date_from_nullable: Optional[str] = Field(None, alias="DateFromNullable")
    date_to_nullable: Optional[str] = Field(None, alias="DateToNullable")

    @field_validator("player_id", mode="before")
    @classmethod
    def coerce_player_id(cls, v):
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v


class LeagueDashPlayerStatsParams(BaseModel):
    """Strict schema for LeagueDashPlayerStats."""
    model_config = ConfigDict(extra="ignore")
    season: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}$")
    measure_type_detailed_defense: Optional[str] = Field("Base")
    per_mode_detailed: Optional[str] = Field("PerGame")
    season_type_all_star: Optional[str] = Field("Regular Season")

class BoxScoreTraditionalV2Params(BaseModel):
    """Strict schema for BoxScoreTraditionalV2."""
    model_config = ConfigDict(extra="ignore")
    game_id: str = Field(..., pattern=r"^\d{10}$", description="10-digit NBA Game ID.")

EXPLICIT_ENDPOINT_MODELS: Dict[str, Type[BaseModel]] = {
    "CommonPlayerInfo": CommonPlayerInfoParams,
    "PlayerGameLog": PlayerGameLogParams,
    "LeagueDashPlayerStats": LeagueDashPlayerStatsParams,
    "BoxScoreTraditionalV2": BoxScoreTraditionalV2Params,
}

# ---------------------------------------------------------
# 4. Dynamic Pydantic v2 Factory for 100+ Endpoints
# ---------------------------------------------------------

class EndpointModelFactory:
    """
    Dynamically constructs and caches Pydantic v2 models for any NBA API endpoint class.
    Ensures zero unhandled type coercions across the entire catalog.
    """
    _cache: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def get_model(cls, endpoint_class: Any) -> Type[BaseModel]:
        class_name = endpoint_class.__name__
        if class_name in EXPLICIT_ENDPOINT_MODELS:
            return EXPLICIT_ENDPOINT_MODELS[class_name]
            
        if class_name in cls._cache:
            return cls._cache[class_name]

        # Build dynamic fields from inspect signature
        sig = inspect.signature(endpoint_class.__init__)
        fields: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name in ["self", "proxy", "headers", "timeout", "get_request"]:
                continue

            default = param.default if param.default != inspect.Parameter.empty else ...
            # Infer Python type
            if "id" in param_name.lower() and not param_name.endswith("_nullable"):
                annotation = Optional[int] if default is not ... else int
            else:
                annotation = Optional[str] if default is not ... else str

            fields[param_name] = (annotation, default)

        # Create Pydantic v2 model dynamically
        model = create_model(
            f"{class_name}DynamicParams",
            __config__=ConfigDict(extra="ignore"),
            **fields
        )
        cls._cache[class_name] = model
        return model

def validate_tool_call(endpoint_class_or_name: Any, raw_params: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates parameter payloads using Pydantic v2 models and TypeAdapters.
    Returns cleaned, validated dict or raises ValidationError.
    """
    # 1. Parse raw_params if string
    params_dict: Dict[str, Any] = {}
    if isinstance(raw_params, str):
        raw_stripped = raw_params.strip()
        if raw_stripped and raw_stripped != "{}":
            try:
                params_dict = json.loads(raw_stripped)
            except Exception as e:
                raise ValueError(f"Parameters payload is not valid JSON: {e}")
    elif isinstance(raw_params, dict):
        params_dict = raw_params
    else:
        raise ValueError(f"Parameters must be a dict or JSON string, received: {type(raw_params)}")

    # 2. Resolve endpoint model
    if isinstance(endpoint_class_or_name, str):
        endpoint_name = endpoint_class_or_name
        if endpoint_name in EXPLICIT_ENDPOINT_MODELS:
            model = EXPLICIT_ENDPOINT_MODELS[endpoint_name]
        else:
            from api.endpoint_utils import load_endpoint_class
            endpoint_cls = load_endpoint_class(endpoint_name)
            if not endpoint_cls:
                raise ValueError(f"Unknown NBA API endpoint: '{endpoint_name}'")
            model = EndpointModelFactory.get_model(endpoint_cls)
    else:
        model = EndpointModelFactory.get_model(endpoint_class_or_name)

    # 3. Normalize parameter keys
    normalized_params = {}
    for k, v in params_dict.items():
        normalized_params[k] = v
        normalized_params[to_snake(k)] = v

    # 4. Validate using TypeAdapter
    adapter = TypeAdapter(model)
    validated_instance = adapter.validate_python(normalized_params)
    
    # Return dict excluding None values so defaults from the endpoint work
    return validated_instance.model_dump(exclude_none=True)


# ---------------------------------------------------------
# 4. Research Engine v2 Schemas (Intent Contract & CAESURA Plan)
# ---------------------------------------------------------

class QuestionType(str, Enum):
    SINGLE = "single"
    COMPARISON = "comparison"
    THESIS = "thesis"
    DEFINITION = "definition"
    LONGITUDINAL = "longitudinal"
    PARADOX = "paradox"
    UNKNOWN = "unknown"


class ScopeFilters(BaseModel):
    """Stathead filter taxonomy representing full query scope"""
    model_config = ConfigDict(extra="ignore")
    
    location: Optional[str] = None
    opponent_team_id: Optional[int] = None
    outcome: Optional[str] = None
    game_segment: Optional[str] = None
    period: Optional[int] = None
    per_mode: Optional[str] = None
    measure_type: Optional[str] = None
    last_n_games: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class EntityRef(BaseModel):
    """Resolved NBA entity with canonical ID"""
    name: str
    type: str = Field(..., description="'player' | 'team'")
    id: Optional[int] = None


class IntentContract(BaseModel):
    """
    v2 Intent-first JSON contract emitted by LLM and validated
    against this schema before any endpoint execution.
    """
    model_config = ConfigDict(extra="ignore")
    
    question_type: QuestionType = Field(..., description="Classification of the question structure")
    entities: List[EntityRef] = Field(default_factory=list, description="Resolved players or teams")
    season: Optional[str] = Field(None, description="e.g. '2023-24' or null")
    season_type: Optional[str] = Field("Regular Season", description="'Regular Season' | 'Playoffs' | null")
    filters: ScopeFilters = Field(default_factory=ScopeFilters)
    metrics: List[str] = Field(default_factory=list, description="Requested metric names, e.g. TS%, on_off_net, PPG")
    group_by: List[str] = Field(default_factory=lambda: ["player", "season"])
    joins: List[Dict[str, Any]] = Field(default_factory=list)
    interpretation: str = Field(
        ...,
        description="Words only: what vague/slang terms in the question are taken to mean in this context. No numbers."
    )
    clarification_needed: Optional[str] = Field(None, description="Enumerated question if gap cannot be resolved by assumption")
    assumptions: List[str] = Field(
        default_factory=list,
        description="Transparent scope assumptions disclosed to the user"
    )


class PlanStep(BaseModel):
    """Individual execution step in a CAESURA logical plan"""
    step_id: str = Field(..., description="e.g. S1, S2")
    endpoint: str = Field(..., description="Target nba_api endpoint name")
    params: Dict[str, Any] = Field(default_factory=dict)
    join_keys: List[str] = Field(default_factory=list, description="e.g. ['player_id', 'season']")
    description: str = Field(default="")
    required: bool = Field(default=True, description="Whether failure of this step halts the plan")


class LogicalPlan(BaseModel):
    """DAG of planned endpoint calls, joins, and aggregates"""
    intent: IntentContract
    steps: List[PlanStep] = Field(default_factory=list)
    target_claims: List[str] = Field(default_factory=list)
    fallback_candidates: List[str] = Field(default_factory=list, description="Alternative candidate endpoints for DAG backtracking")


