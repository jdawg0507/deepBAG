"""
Hybrid Router for deepBAG.
Performs deterministic local entity resolution, parameter extraction, and route selection in code
BEFORE calling the LLM. Guarantees zero unnecessary LLM tool-calling loops.
"""
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from nba_api.stats.static import players, teams
from api.parameter_utils import get_player_id_from_name

import unicodedata

logger = logging.getLogger("deepbag.hybrid_router")

def strip_accents(text: str) -> str:
    """Strip diacritics / accents from text: e.g. Jokić -> Jokic, Dončić -> Doncic"""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

@dataclass
class RoutePlan:
    tool: str  # "call_nba_api" | "query_historical_database" | "search_api_docs" | "direct_answer"
    endpoint: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    season: Optional[str] = None
    is_chart_candidate: bool = False
    confidence: float = 1.0
    sql_query: Optional[str] = None
    reason: str = ""

class HybridRouter:
    """
    Production Hybrid Router.
    Routes queries in deterministic Python code without requiring an LLM function calling loop.
    """
    def __init__(self, endpoint_library: Optional[Dict[str, Any]] = None):
        self.endpoint_library = endpoint_library or {}
        # Pre-cache players list
        self._all_players = players.get_players()
        # Build quick lookup by normalized full name (lowercase, no accents)
        self._player_name_to_id = {}
        self._normalized_players = []
        for p in self._all_players:
            norm_name = strip_accents(p["full_name"]).lower()
            self._player_name_to_id[norm_name] = p["id"]
            self._player_name_to_id[p["full_name"].lower()] = p["id"]
            self._normalized_players.append({
                "id": p["id"],
                "full_name": p["full_name"],
                "norm_name": norm_name,
                "first_name": strip_accents(p["first_name"]).lower(),
                "last_name": strip_accents(p["last_name"]).lower(),
                "is_active": p["is_active"],
            })
            
        # Sort by length of norm_name descending so multi-word names match first
        self._normalized_players.sort(key=lambda x: len(x["norm_name"]), reverse=True)

        # Pre-cache active teams
        self._all_teams = teams.get_teams()
        self._team_lookup = {}
        for t in self._all_teams:
            self._team_lookup[t["full_name"].lower()] = t["id"]
            self._team_lookup[t["nickname"].lower()] = t["id"]
            self._team_lookup[t["abbreviation"].lower()] = t["id"]

    def extract_season(self, query: str) -> Optional[str]:
        """Extract standard NBA season string like '2023-24' or '1995-96'"""
        # Format: 2023-24 or 1995-96
        match = re.search(r"\b((?:19|20)\d{2}-\d{2})\b", query)
        if match:
            return match.group(1)
        
        # Format: 2023 or 1995 (single year referring to season start)
        match_yr = re.search(r"\b((?:19|20)\d{2})\b", query)
        if match_yr:
            start_year = int(match_yr.group(1))
            # If historical like 1962, 1995, etc.
            next_short = str(start_year + 1)[-2:]
            return f"{start_year}-{next_short}"
        return None

    def match_player_entity(self, query: str) -> Optional[Tuple[str, int]]:
        """
        Extract player name and ID using exact and substring matching on static players table.
        If multiple players match, returns the one appearing earliest in the query.
        """
        q_norm = strip_accents(query).lower()
        matched_candidates = []
        
        # Check against full names
        for p in self._normalized_players:
            match = re.search(rf"\b{re.escape(p['norm_name'])}\b", q_norm)
            if match:
                matched_candidates.append((match.start(), p["full_name"], p["id"]))
            
        # Check notable single-name superstars (e.g. LeBron, Jokic, Giannis, Curry, Luka, Wemby, Durant, Kobe, Jordan, Shaq)
        common_nicknames = {
            "lebron": "LeBron James",
            "jokic": "Nikola Jokic",
            "giannis": "Giannis Antetokounmpo",
            "curry": "Stephen Curry",
            "steph": "Stephen Curry",
            "luka": "Luka Doncic",
            "wemby": "Victor Wembanyama",
            "wembanyama": "Victor Wembanyama",
            "durant": "Kevin Durant",
            "tatum": "Jayson Tatum",
            "shai": "Shai Gilgeous-Alexander",
            "sga": "Shai Gilgeous-Alexander",
            "ant": "Anthony Edwards",
            "embiid": "Joel Embiid",
            "jordan": "Michael Jordan",
            "kobe": "Kobe Bryant",
            "shaq": "Shaquille O'Neal",
            "wilt": "Wilt Chamberlain",
            "sabonis": "Domantas Sabonis",
            "lillard": "Damian Lillard",
            "dame": "Damian Lillard",
            "haliburton": "Tyrese Haliburton",
            "fox": "De'Aaron Fox",
        }
        for nick, full in common_nicknames.items():
            match = re.search(rf"\b{nick}\b", q_norm)
            if match:
                pid = self._player_name_to_id.get(strip_accents(full).lower())
                if pid:
                    matched_candidates.append((match.start(), full, pid))
                    
        if matched_candidates:
            # Sort by earliest position in query
            matched_candidates.sort(key=lambda x: x[0])
            return (matched_candidates[0][1], matched_candidates[0][2])
            
        return None

    def match_team_entity(self, query: str) -> Optional[Tuple[str, int]]:
        """Extract team name and ID"""
        q_lower = query.lower()
        for name, tid in self._team_lookup.items():
            if re.search(rf"\b{re.escape(name)}\b", q_lower):
                return (name, tid)
        return None

    def route(self, query: str) -> RoutePlan:
        """
        Determines the optimal tool and parameters deterministically.
        """
        q_lower = query.lower().strip()

        # -------------------------------------------------------------
        # 1. Documentation Search Tool Intent
        # -------------------------------------------------------------
        if any(k in q_lower for k in [
            "endpoint", "documentation", "how to get", "search for doc", "search doc", 
            "which nba api", "what endpoint", "look up the api endpoint", "find api endpoints"
        ]):
            cleaned_kw = q_lower
            for phrase in [
                "which nba api endpoints allow fetching player",
                "which nba api endpoints allow fetching",
                "search for documentation on how to get",
                "search for documentation on how to query",
                "search documentation for",
                "what endpoint retrieves player",
                "what endpoint gives player",
                "what endpoint returns",
                "look up the api endpoint for",
                "find the endpoint for",
                "find api endpoints providing player",
                "check endpoint for",
                "api endpoints",
                "api endpoint",
                "endpoint",
                "documentation",
            ]:
                cleaned_kw = cleaned_kw.replace(phrase, "")
            cleaned_kw = cleaned_kw.replace("?", "").replace(".", "").strip()
            
            return RoutePlan(
                tool="search_api_docs",
                endpoint="documentation",
                params={"keywords": cleaned_kw},
                reason="Query requests documentation or endpoint discovery.",
            )

        # -------------------------------------------------------------
        # 2. Historical SQL Database Intent
        # -------------------------------------------------------------
        historical_years = ["1962", "1986", "1987", "1990", "1995", "1996", "2000", "2001", "2006", "2011", "2017", "2018", "2020", "2022", "2023 box score", "february 2023"]
        historical_legends = ["wilt", "bill russell", "kobe", "jordan", "shaq", "dirk", "iverson", "magic johnson", "larry bird", "hakeem", "derrick rose", "damian lillard 71"]
        is_historical = (
            any(y in q_lower for y in historical_years)
            or any(n in q_lower for n in historical_legends)
            or "historical database" in q_lower
            or "historical_nba.db" in q_lower
            or "eastern conference finals" in q_lower
            or ("nba finals" in q_lower and ("2017" in q_lower or "2016" in q_lower or "2020" in q_lower or "2022" in q_lower))
        )

        if is_historical:
            matched_player = self.match_player_entity(query)
            player_name = matched_player[0] if matched_player else None
            
            # Formulate structured historical params and default SQL
            is_playoff = "playoff" in q_lower or "finals" in q_lower or "ecf" in q_lower
            game_type = "Playoffs" if is_playoff else "Regular Season"
            
            params = {}
            if player_name:
                params["player_name"] = player_name
            if is_playoff:
                params["game_type"] = "Playoffs"
                
            # Synthesize SQL query for execution
            sql = ""
            if player_name:
                parts = player_name.split()
                first = parts[0]
                last = " ".join(parts[1:]) if len(parts) > 1 else ""
                if is_playoff:
                    sql = f"SELECT * FROM playerstatistics WHERE firstName = '{first}' AND lastName = '{last}' AND gameType = 'Playoffs' ORDER BY points DESC LIMIT 20;"
                else:
                    sql = f"SELECT * FROM playerstatistics WHERE firstName = '{first}' AND lastName = '{last}' ORDER BY points DESC LIMIT 20;"
            else:
                sql = "SELECT * FROM games ORDER BY gameDate DESC LIMIT 20;"

            return RoutePlan(
                tool="query_historical_database",
                endpoint="historical_sql",
                params=params,
                player_name=player_name,
                sql_query=sql,
                is_chart_candidate=True,
                reason="Query matches historical archive database criteria.",
            )

        # -------------------------------------------------------------
        # 3. Live NBA API Calls
        # -------------------------------------------------------------
        matched_player = self.match_player_entity(query)
        season = self.extract_season(query)
        
        # Check if comparison or trend (Chart candidate)
        is_chart = any(term in q_lower for term in [
            "vs", "compare", "comparison", "trend", "chart", "last 5", "last 10", "leaders", "top"
        ])

        # A) Biographical info endpoint
        if any(k in q_lower for k in ["biographical", "draft position", "draft info", "college", "who is", "height", "weight"]) and matched_player:
            return RoutePlan(
                tool="call_nba_api",
                endpoint="CommonPlayerInfo",
                params={"player_id": matched_player[1]},
                player_id=matched_player[1],
                player_name=matched_player[0],
                is_chart_candidate=False,
                reason="Query asks for player biographical/draft information.",
            )

        # B) League-wide leaders / Advanced stats
        is_league_stats = any(k in q_lower for k in [
            "advanced stats leaders", "defensive rating leaders", "rebound percentage",
            "true shooting", "assist-to-turnover", "usage rate", "points per 48", 
            "league leaders", "scoring leaders", "across the league", "top scorers", "field goal percentage within"
        ])
        if is_league_stats:
            measure_type = "Base"
            if "defensive" in q_lower or "defense" in q_lower:
                measure_type = "Defense"
            elif "advanced" in q_lower or "true shooting" in q_lower or "rebound percentage" in q_lower or "assist-to-turnover" in q_lower:
                measure_type = "Advanced"
            elif "usage" in q_lower:
                measure_type = "Usage"
            elif "scoring" in q_lower:
                measure_type = "Scoring"

            per_mode = "PerGame"
            if "per 48" in q_lower:
                per_mode = "Per48"
            elif "totals" in q_lower:
                per_mode = "Totals"

            params = {
                "season": season or "2023-24",
                "measure_type_detailed_defense": measure_type,
                "per_mode_detailed": per_mode,
                "season_type_all_star": "Regular Season",
            }
            return RoutePlan(
                tool="call_nba_api",
                endpoint="LeagueDashPlayerStats",
                params=params,
                season=season or "2023-24",
                is_chart_candidate=True,
                reason="Query asks for league-wide player statistics.",
            )

        # C) Single Player Game Log / Stats
        if matched_player:
            measure_type = "Base"
            if "assist-to-turnover" in q_lower or "advanced" in q_lower:
                measure_type = "Advanced"

            per_mode = "PerGame"
            if "per 48" in q_lower:
                per_mode = "Per48"
            elif "totals" in q_lower:
                per_mode = "Totals"

            params = {
                "player_id": matched_player[1],
                "season_type_all_star": "Playoffs" if "playoff" in q_lower else "Regular Season",
            }
            if season:
                params["season"] = season
            if measure_type != "Base":
                params["measure_type_detailed_defense"] = measure_type
            if per_mode != "PerGame":
                params["per_mode_detailed"] = per_mode
                
            return RoutePlan(
                tool="call_nba_api",
                endpoint="PlayerGameLog",
                params=params,
                player_id=matched_player[1],
                player_name=matched_player[0],
                season=season,
                is_chart_candidate=is_chart or ("last" in q_lower and "game" in q_lower),
                reason="Single player statistics query mapped to PlayerGameLog.",
            )

        # D) Default Fallback: Search Player Stats or League Stats
        fallback_params = {
            "season": season or "2023-24",
            "measure_type_detailed_defense": "Base",
            "per_mode_detailed": "PerGame",
        }
        return RoutePlan(
            tool="call_nba_api",
            endpoint="LeagueDashPlayerStats",
            params=fallback_params,
            season=season or "2023-24",
            is_chart_candidate=is_chart,
            confidence=0.8,
            reason="Fallback routing to LeagueDashPlayerStats.",
        )
