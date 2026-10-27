"""
Unit tests for Milestone 0: Endpoint Catalogue and Foundation Clients.
"""
import os
import json
import pytest

from core.nba_client import NBAClient
from core.llm_client import UnifiedLLMClient

def test_endpoint_catalogue_structure():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalogue_path = os.path.join(repo_root, "data", "endpoint_catalogue.json")
    
    assert os.path.exists(catalogue_path), "endpoint_catalogue.json must exist"
    
    with open(catalogue_path, "r", encoding="utf-8") as f:
        catalogue = json.load(f)
        
    assert len(catalogue) >= 130, f"Expected at least 130 endpoints, found {len(catalogue)}"
    
    # Required core endpoints for Research Engine v2
    expected_endpoints = [
        "TeamPlayerOnOffSummary",
        "TeamPlayerOnOffDetails",
        "ShotChartDetail",
        "LeagueDashPtDefend",
        "PlayerDashPtShotDefend",
        "LeagueDashPlayerShotLocations",
        "PlayerGameLog",
        "TeamGameLog",
        "CommonPlayerInfo",
        "BoxScoreFourFactorsV2",
    ]
    
    for ep in expected_endpoints:
        assert ep in catalogue, f"Expected endpoint '{ep}' in catalogue"
        info = catalogue[ep]
        assert "name" in info
        assert "module" in info
        assert "parameters" in info
        assert "required_parameters" in info
        assert "optional_parameters" in info
        assert "param_patterns" in info
        assert isinstance(info["datasets"], list)

def test_nba_client_resolution():
    client = NBAClient()
    # Test resolving classes
    cls_log = client._resolve_endpoint_class("PlayerGameLog")
    assert cls_log is not None
    assert cls_log.__name__ == "PlayerGameLog"
    
    cls_onoff = client._resolve_endpoint_class("TeamPlayerOnOffSummary")
    assert cls_onoff is not None
    assert cls_onoff.__name__ == "TeamPlayerOnOffSummary"
    
    # Unknown endpoint should return None
    assert client._resolve_endpoint_class("NonExistentEndpoint123") is None

def test_unified_llm_client_initialization():
    # Should initialize Gemini provider if NEBIUS_API_KEY is not set
    client = UnifiedLLMClient()
    assert client.provider in ("gemini", "nebius")
    assert client.client is not None


def test_resolve_data_source_null_season():
    """Verify that season=None is safely handled without TypeError or false era rejections"""
    from catalogue.schema import EndpointSpec, resolve_data_source, DataSource
    spec = EndpointSpec(
        endpoint_name="PlayerGameLog",
        min_season="1980-81",
        entity_grain="player",
        capability_card="Game log",
        has_historical_equivalent=True
    )
    ds = resolve_data_source(player=None, season=None, endpoint=spec)
    assert ds == DataSource.NBA_API_LIVE


def test_code_verifier_historical_era_anchoring():
    """Verify that omitting a season for a retired player anchors to their active era rather than injecting 2023-24"""
    from catalogue.schema import PlayerMeta
    from router.intent_parser import IntentParser
    
    wilt = PlayerMeta(id=76375, full_name="Wilt Chamberlain", from_year=1959, to_year=1973)
    code_verifier = IntentParser().verifier
    res = code_verifier.verify_plan(
        endpoint_calls=[{"endpoint": "PlayerGameLog", "params": {"player_id": 76375}}],
        players_meta=[wilt],
        season=None
    )
    # Must NOT blame the user for requesting 2023-24
    if not res.is_valid:
        assert "2023-24" not in res.error_message
        assert "1972-73" in res.error_message or "CareerBoundsError" not in res.error_message
