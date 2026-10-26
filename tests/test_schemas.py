"""
Unit tests for Pydantic v2 strict schemas and dynamic TypeAdapter factory.
"""
import pytest
from pydantic import ValidationError
from core.schemas import (
    ChatRequest,
    ToolCallInvocation,
    validate_tool_call,
    CommonPlayerInfoParams,
    PlayerGameLogParams,
    LeagueDashPlayerStatsParams,
)

def test_chat_request_valid():
    req = ChatRequest(query="  Who scored 70 points?  ", session_id="sess_123")
    assert req.query == "Who scored 70 points?"
    assert req.session_id == "sess_123"

def test_chat_request_invalid_empty():
    with pytest.raises(ValidationError):
        ChatRequest(query=" ")

def test_chat_request_invalid_extra_field():
    with pytest.raises(ValidationError):
        ChatRequest(query="valid query", extra_garbage="forbidden")

def test_tool_call_invocation_json_parsing():
    inv = ToolCallInvocation(
        endpoint_name="CommonPlayerInfo",
        parameters='{"PlayerID": 2544}'
    )
    assert inv.endpoint_name == "CommonPlayerInfo"
    assert inv.parameters == {"PlayerID": 2544}

def test_tool_call_invocation_invalid_json():
    with pytest.raises(ValidationError):
        ToolCallInvocation(
            endpoint_name="CommonPlayerInfo",
            parameters='{invalid_json_format'
        )

def test_validate_tool_call_common_player_info():
    # Valid call
    cleaned = validate_tool_call("CommonPlayerInfo", {"PlayerID": 2544})
    assert cleaned["player_id"] == 2544

    # String coercion if digits
    cleaned2 = validate_tool_call("CommonPlayerInfo", {"PlayerID": "2544"})
    assert cleaned2["player_id"] == 2544

    # Rejection on invalid non-numeric string
    with pytest.raises(Exception):
        validate_tool_call("CommonPlayerInfo", {"PlayerID": "not_an_id"})

def test_validate_tool_call_player_game_log():
    cleaned = validate_tool_call(
        "PlayerGameLog",
        {"PlayerID": 201939, "Season": "2023-24", "SeasonTypeAllStar": "Regular Season"}
    )
    assert cleaned["player_id"] == 201939
    assert cleaned["season"] == "2023-24"

    # Invalid season format rejection
    with pytest.raises(ValidationError):
        validate_tool_call("PlayerGameLog", {"PlayerID": 201939, "Season": "invalid_season_2023"})

def test_validate_tool_call_dynamic_fallback():
    # Test an endpoint that is dynamically registered and validated
    cleaned = validate_tool_call(
        "LeagueLeaders",
        {"Season": "2023-24", "StatCategory": "PTS"}
    )
    assert "Season" in cleaned or "season" in cleaned
