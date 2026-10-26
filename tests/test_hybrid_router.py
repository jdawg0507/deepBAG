"""
Unit tests for HybridRouter.
Verifies deterministic local entity resolution, parameter extraction, and route selection.
"""
import pytest
from nlp.hybrid_router import HybridRouter

@pytest.fixture
def router():
    return HybridRouter()

def test_extract_season(router):
    assert router.extract_season("How many points in 2023-24?") == "2023-24"
    assert router.extract_season("Stats from 1995-96") == "1995-96"
    assert router.extract_season("Games in 2021") == "2021-22"
    assert router.extract_season("Random query without year") is None

def test_match_player_entity(router):
    name, pid = router.match_player_entity("What were LeBron James' stats last game?")
    assert name == "LeBron James"
    assert pid == 2544

    name, pid = router.match_player_entity("How many 3s did Curry make?")
    assert name == "Stephen Curry"
    assert pid == 201939

    name, pid = router.match_player_entity("Nikola Jokic triple doubles")
    assert "Joki" in name
    assert pid == 203999

def test_route_documentation(router):
    plan = router.route("Which NBA API endpoints allow fetching player shot tracking data?")
    assert plan.tool == "search_api_docs"
    assert plan.endpoint == "documentation"
    assert "keywords" in plan.params

def test_route_historical_sql(router):
    plan = router.route("Michael Jordan game logs during the 1995-96 regular season.")
    assert plan.tool == "query_historical_database"
    assert plan.endpoint == "historical_sql"
    assert plan.player_name == "Michael Jordan"
    assert plan.sql_query is not None
    assert "playerstatistics" in plan.sql_query.lower()

def test_route_player_game_log(router):
    plan = router.route("What were LeBron James' points and assists in his most recent game?")
    assert plan.tool == "call_nba_api"
    assert plan.endpoint == "PlayerGameLog"
    assert plan.player_id == 2544
    assert plan.params["player_id"] == 2544

def test_route_common_player_info(router):
    plan = router.route("Give me the biographical info and draft position for Nikola Jokic.")
    assert plan.tool == "call_nba_api"
    assert plan.endpoint == "CommonPlayerInfo"
    assert plan.player_id == 203999
    assert plan.params["player_id"] == 203999

def test_route_league_dash_player_stats(router):
    plan = router.route("Show me the advanced stats leaders across the league in 2023-24.")
    assert plan.tool == "call_nba_api"
    assert plan.endpoint == "LeagueDashPlayerStats"
    assert plan.params["season"] == "2023-24"
    assert plan.params["measure_type_detailed_defense"] == "Advanced"

    plan_def = router.route("Find the defensive rating leaders in the 2023-24 regular season.")
    assert plan_def.tool == "call_nba_api"
    assert plan_def.endpoint == "LeagueDashPlayerStats"
    assert plan_def.params["measure_type_detailed_defense"] == "Defense"
