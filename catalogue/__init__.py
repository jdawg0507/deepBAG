"""
Catalogue package for deepBAG.
"""
from catalogue.schema import (
    DataSource,
    CareerBoundsError,
    PlayerMeta,
    ParamSpec,
    EndpointSpec,
    resolve_data_source,
)

__all__ = [
    "DataSource",
    "CareerBoundsError",
    "PlayerMeta",
    "ParamSpec",
    "EndpointSpec",
    "resolve_data_source",
]
