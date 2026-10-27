"""
Catalogue & Temporal Metadata Schema for deepBAG.
Defines strongly-typed schemas for NBA API endpoints, parameter contracts,
entity grains, and historical era routing rules.
"""
from typing import Dict, Any, List, Set, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

class DataSource(str, Enum):
    NBA_API_LIVE = "NBA_API_LIVE"
    HISTORICAL_DB = "HISTORICAL_DB"
    UNSUPPORTED_ERA = "UNSUPPORTED_ERA"

class CareerBoundsError(Exception):
    """Raised when a player query falls outside their recorded active career bounds."""
    pass

class PlayerMeta(BaseModel):
    id: int
    full_name: str
    from_year: int
    to_year: int

class ParamSpec(BaseModel):
    name: str
    type: str = "str"
    required: bool = False
    default: Any = None
    allowed_values: Optional[List[str]] = None
    description: Optional[str] = None

class EndpointSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    endpoint_name: str
    min_season: str = "1946-47"
    entity_grain: Literal["player", "team", "league", "game"]
    required_params: List[ParamSpec] = Field(default_factory=list)
    optional_params: List[ParamSpec] = Field(default_factory=list)
    join_keys: List[str] = Field(default_factory=list)
    capability_card: str
    has_historical_equivalent: bool = False

def resolve_data_source(player: Optional[PlayerMeta], season: Optional[str], endpoint: EndpointSpec) -> DataSource:
    """
    Computes the appropriate data source based on player career bounds,
    endpoint tracking era limitations, and historical database availability.
    """
    if season is None:
        return DataSource.NBA_API_LIVE

    try:
        target_year = int(season[:4])
    except (ValueError, TypeError, IndexError):
        target_year = 2023

    if player:
        if target_year < player.from_year or target_year > player.to_year:
            raise CareerBoundsError(f"{player.full_name} active {player.from_year}-{player.to_year}, got {season}")

    if endpoint.min_season and season < endpoint.min_season:
        return DataSource.HISTORICAL_DB if endpoint.has_historical_equivalent else DataSource.UNSUPPORTED_ERA

    return DataSource.NBA_API_LIVE
