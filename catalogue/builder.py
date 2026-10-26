"""
Endpoint Catalogue Builder.
Compiles all 135 nba_api endpoints into strongly-typed EndpointSpec records
with min_season, entity_grain, required_params, join_keys, and capability_cards.
Saves the artifact to data/endpoint_catalogue_v2.json.
"""
import os
import json
import inspect
import pkgutil
import importlib
from typing import Dict, Any, List

from catalogue.schema import EndpointSpec, ParamSpec
import nba_api.stats.endpoints as ep_pkg

# Known tracking and advanced era boundary years
TRACKING_ENDPOINTS = {
    "ShotChartDetail": ("1996-97", False),
    "ShotChartLineupDetail": ("1996-97", False),
    "PlayerDashPtShotDefend": ("2013-14", False),
    "PlayerDashPtShots": ("2013-14", False),
    "PlayerDashPtReboundLogs": ("2013-14", False),
    "PlayerDashPtPass": ("2013-14", False),
    "LeagueHustleStatsPlayer": ("2016-17", False),
    "LeagueHustleStatsTeam": ("2016-17", False),
    "TeamDashPtShots": ("2013-14", False),
    "TeamDashPtPass": ("2013-14", False),
}

BOX_SCORE_HISTORICAL_EQUIVALENTS = {
    "PlayerGameLog": ("1946-47", True),
    "PlayerGameLogs": ("1946-47", True),
    "CommonPlayerInfo": ("1946-47", True),
    "TeamGameLog": ("1946-47", True),
    "LeagueDashPlayerStats": ("1996-97", True),
    "LeagueDashTeamStats": ("1996-97", True),
    "TeamPlayerOnOffDetails": ("1996-97", False),
    "TeamPlayerOnOffSummary": ("1996-97", False),
    "TeamDashboardByGeneralSplits": ("1996-97", False),
    "TeamDashboardByOpponent": ("1996-97", False),
}

def build_catalogue_v2() -> Dict[str, Any]:
    catalogue: Dict[str, Any] = {}
    
    for _, mod_name, _ in pkgutil.iter_modules(ep_pkg.__path__):
        try:
            mod = importlib.import_module(f"nba_api.stats.endpoints.{mod_name}")
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if name.lower() == mod_name.lower() and hasattr(obj, "__init__"):
                    endpoint_name = obj.__name__
                    
                    # 1. Inspect parameters
                    sig = inspect.signature(obj.__init__)
                    req_params: List[ParamSpec] = []
                    opt_params: List[ParamSpec] = []
                    join_keys = set()
                    
                    for p_name, param in sig.parameters.items():
                        if p_name in ["self", "proxy", "headers", "timeout", "get_request"]:
                            continue
                            
                        # Infer parameter type
                        p_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "str"
                        is_req = (param.default == inspect.Parameter.empty)
                        
                        spec = ParamSpec(
                            name=p_name,
                            type=p_type,
                            required=is_req,
                            default=None if is_req else (str(param.default) if param.default is not None else None)
                        )
                        
                        if is_req:
                            req_params.append(spec)
                        else:
                            opt_params.append(spec)
                            
                        p_lower = p_name.lower()
                        if "player_id" in p_lower:
                            join_keys.add("PLAYER_ID")
                        if "team_id" in p_lower:
                            join_keys.add("TEAM_ID")
                        if "game_id" in p_lower:
                            join_keys.add("GAME_ID")

                    # 2. Determine Entity Grain
                    ep_lower = endpoint_name.lower()
                    if "player" in ep_lower or "PLAYER_ID" in join_keys:
                        grain = "player"
                    elif "team" in ep_lower or "TEAM_ID" in join_keys:
                        grain = "team"
                    elif "game" in ep_lower or "boxscore" in ep_lower:
                        grain = "game"
                    else:
                        grain = "league"

                    # 3. Determine era boundary & historical equivalent
                    if endpoint_name in TRACKING_ENDPOINTS:
                        min_season, has_hist = TRACKING_ENDPOINTS[endpoint_name]
                    elif endpoint_name in BOX_SCORE_HISTORICAL_EQUIVALENTS:
                        min_season, has_hist = BOX_SCORE_HISTORICAL_EQUIVALENTS[endpoint_name]
                    elif "pt" in ep_lower or "hustle" in ep_lower or "tracking" in ep_lower:
                        min_season, has_hist = "2013-14", False
                    elif "dash" in ep_lower or "split" in ep_lower:
                        min_season, has_hist = "1996-97", False
                    else:
                        min_season, has_hist = "1946-47", True

                    # 4. Generate Capability Card
                    doc = inspect.getdoc(obj) or f"NBA API endpoint {endpoint_name}"
                    first_line = doc.split("\n")[0] if doc else endpoint_name
                    req_names = [p.name for p in req_params]

                    # Enrich with domain query patterns
                    extra_intents = ""
                    if endpoint_name == "PlayerGameLog":
                        extra_intents = "Individual player game-by-game statistics, points, rebounds, assists, 3-pointers, steals, blocks, game log, single game box score, stats against specific opponent (vs team, against Celtics, vs Lakers), regular season and playoff games, minutes played, 30+ minutes games, single player performances, player stats in specific season."
                    elif endpoint_name == "PlayerDashboardByOpponent":
                        extra_intents = "Player career and season splits against specific opponent teams, matchup averages vs opponents, head-to-head vs team."
                    elif endpoint_name == "PlayerDashboardByYearOverYear":
                        extra_intents = "Player year-by-year season averages, career trajectory, season-level stats progression."
                    elif endpoint_name == "LeagueDashPlayerStats":
                        extra_intents = "League-wide player stats, multi-player comparisons, player rankings, scoring leaders, true shooting (TS%), effective field goal (eFG%), points, rebounds, assists per game leaders, comparison between multiple players."
                    elif endpoint_name == "PlayerCareerStats":
                        extra_intents = "Player career totals, year by year career averages, all seasons summary, rookie season stats, career points, career rebounds, career assists."
                    elif endpoint_name == "CommonPlayerInfo":
                        extra_intents = "Player biographical info, draft position, height, weight, college, draft year, team roster status."
                    elif endpoint_name == "TeamPlayerOnOffDetails":
                        extra_intents = "On off splits, with and without teammate, on the floor vs on the bench, net rating with teammate out, lineup differential."
                    elif endpoint_name == "ShotChartDetail":
                        extra_intents = "Shot chart, shooting locations, shot zones, shot attempts on court, rim deterrence, midrange shots."
                    elif endpoint_name == "TeamDashboardByGeneralSplits":
                        extra_intents = "Team home road splits, monthly splits, win loss splits."

                    card = (
                        f"Endpoint: {endpoint_name}. Grain: {grain}. "
                        f"Description: {first_line}. {extra_intents} "
                        f"Required parameters: {', '.join(req_names) if req_names else 'None'}. "
                        f"Join keys: {', '.join(sorted(join_keys)) if join_keys else 'None'}. "
                        f"Valid era: seasons >= {min_season}."
                    )

                    spec_obj = EndpointSpec(
                        endpoint_name=endpoint_name,
                        min_season=min_season,
                        entity_grain=grain,
                        required_params=req_params,
                        optional_params=opt_params,
                        join_keys=sorted(list(join_keys)),
                        capability_card=card,
                        has_historical_equivalent=has_hist
                    )
                    catalogue[endpoint_name] = spec_obj.model_dump()
        except Exception as e:
            continue

    return catalogue

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo_root, "data", "endpoint_catalogue_v2.json")
    cat = build_catalogue_v2()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=2)
    print(f"Successfully generated Endpoint Catalogue v2 with {len(cat)} endpoints at {out_path}")
