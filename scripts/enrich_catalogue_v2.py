"""
Comprehensive catalogue capability card enricher for deepBAG.
Feeds rich domain metadata into data/endpoint_catalogue_v2.json so dense embeddings
and LLM function compilation can find and use endpoints dynamically without hardcoded switches.
"""
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enrich_catalogue")

ENRICHMENTS = {
    "TeamDashboardByGeneralSplits": {
        "capability_card": (
            "Endpoint: TeamDashboardByGeneralSplits. Grain: team. "
            "Description: Team performance profile, playoff dominance analysis, and general splits. "
            "Includes Four Factors decomposition (eFG%, TOV%, OREB%, FTR) via measure_type_detailed_defense='Four Factors', "
            "advanced ratings (Offensive Rating, Defensive Rating, Net Rating, Pace), and wins/losses/location splits. "
            "Required parameters: team_id. Valid era: seasons >= 1996-97."
        ),
        "measure_type_allowed": ["Base", "Advanced", "Misc", "Four Factors", "Scoring", "Opponent"],
    },
    "TeamPlayerOnOffSummary": {
        "capability_card": (
            "Endpoint: TeamPlayerOnOffSummary. Grain: player/team. "
            "Description: On/Off net rating differential across team rotation. Measures team and teammate performance, "
            "offensive/defensive efficiency, spacing uplift, and point differential when specific players are ON the court versus OFF the court. "
            "Required parameters: team_id. Valid era: seasons >= 1996-97."
        ),
        "join_keys": ["team_id", "season"]
    },
    "TeamPlayerOnOffDetails": {
        "capability_card": (
            "Endpoint: TeamPlayerOnOffDetails. Grain: player/team. "
            "Description: Detailed On/Off lineup splits and game segment adjustments (First Half vs Second Half halftime adjustments, quarters). "
            "Measures tactical adjustments, clutch rating, and player lineup net ratings with game_segment_nullable ('First Half', 'Second Half'). "
            "Required parameters: team_id. Valid era: seasons >= 1996-97."
        ),
        "game_segment_allowed": ["First Half", "Second Half", "Overtime"]
    },
    "PlayerDashPtShots": {
        "capability_card": (
            "Endpoint: PlayerDashPtShots. Grain: player. "
            "Description: Player tracking shooting profile, perimeter gravity, spacing, off-ball pull, and shot creation. "
            "Includes catch-and-shoot volume and efficiency, pull-up jump shots, touch time, dribble numbers before shot, "
            "defender distance (tight, open, wide open), and shot distance. "
            "Required parameters: player_id (or team_id). Valid era: seasons >= 2013-14."
        ),
        "join_keys": ["player_id", "season"]
    },
    "PlayerDashPtShotDefend": {
        "capability_card": (
            "Endpoint: PlayerDashPtShotDefend. Grain: player. "
            "Description: Player defensive tracking metrics, paint protection, interior rim deterrence, and defensive gravity. "
            "Measures opponent field goal percentage at the rim (Less than 6ft, Less than 10ft), contested shots, "
            "and defensive field goal percentage differential (DIFF_PCT) when player contests shots. "
            "Required parameters: player_id (or team_id). Valid era: seasons >= 2013-14."
        ),
        "join_keys": ["player_id", "season"]
    },
    "SynergyPlayTypes": {
        "capability_card": (
            "Endpoint: SynergyPlayTypes. Grain: league/player/team. "
            "Description: Synergy Sports play-type tracking data for offensive and defensive play-types including "
            "Isolation (iso shotmaking, iso scoring, one-on-one shot creation, unassisted scoring), "
            "Pick and Roll Ball Handler (PRBallHandler, p&r ball handler), Pick and Roll Roll Man (PRRollman), "
            "Post Up (post scoring), Spot Up (catch and shoot spotup), Handoff, Cut, Off Screen, and Transition. "
            "Metrics include Points Per Possession (PPP), Possession Percentage (POSS_PCT), Field Goal Percentage (FG_PCT), "
            "Effective Field Goal Percentage (EFG_PCT), Points (PTS), Percentile, and turnover frequency. "
            "Accepts play_type_nullable ('Isolation', 'PRBallHandler', 'Spotup', etc.) and player_or_team_abbreviation "
            "('P' for player, 'T' for team). Valid era: seasons >= 2015-16."
        ),
        "join_keys": ["player_id", "season", "team_id"],
        "play_type_allowed": [
            "Isolation", "PRBallHandler", "PRRollman", "Postup", "Spotup",
            "Handoff", "Cut", "OffScreen", "Transition", "Misc"
        ]
    },
    "LeagueDashPlayerShotLocations": {
        "capability_card": (
            "Endpoint: LeagueDashPlayerShotLocations. Grain: player/league. "
            "Description: League-wide player shooting efficiency broken down by specific shot zones on the floor. "
            "Includes Restricted Area, In The Paint (Non-RA), Mid-Range (midrange paradox), Corner 3, and Above the Break 3. "
            "Required parameters: None (league-wide). Valid era: seasons >= 1996-97."
        ),
        "join_keys": ["player_id", "season"]
    },
    "LeagueDashPlayerStats": {
        "capability_card": (
            "Endpoint: LeagueDashPlayerStats. Grain: player/league. "
            "Description: Comprehensive league-wide player leaderboard and multi-player comparisons. "
            "Scoring leaders, rebounds, assists, steals, blocks, minutes, True Shooting (TS%), Effective Field Goal (eFG%), "
            "usage percentage (USG%), and net ratings across regular season and playoffs. "
            "Required parameters: None (league-wide). Valid era: seasons >= 1996-97."
        ),
        "join_keys": ["player_id", "season"]
    },
    "PlayerGameLog": {
        "capability_card": (
            "Endpoint: PlayerGameLog. Grain: player/game. "
            "Description: Individual player game-by-game logs, game performances, recent scoring trends, "
            "single-game box scores, and head-to-head opponent matchups. "
            "Required parameters: player_id. Valid era: seasons >= 1946-47."
        ),
        "join_keys": ["player_id", "season"]
    },
    "PlayerCareerStats": {
        "capability_card": (
            "Endpoint: PlayerCareerStats. Grain: player. "
            "Description: Career statistics profile, career totals, multi-season year-by-year progression, "
            "historical player comparisons, all-time scoring totals, and career per-game averages. "
            "Required parameters: player_id. Valid era: seasons >= 1946-47."
        ),
        "join_keys": ["player_id"]
    }
}

def enrich():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cat_v2_path = os.path.join(repo_root, "data", "endpoint_catalogue_v2.json")
    v1_path = os.path.join(repo_root, "data", "endpoint_catalogue.json")

    with open(cat_v2_path, "r", encoding="utf-8") as f:
        v2_cat = json.load(f)

    with open(v1_path, "r", encoding="utf-8") as f:
        v1_cat = json.load(f)

    # Hydrate output fields
    for ep_name, spec in v2_cat.items():
        v1_entry = v1_cat.get(ep_name, {})
        fields_by_ds = v1_entry.get("fields_by_dataset", {})
        all_fields = []
        for ds_name, fields in fields_by_ds.items():
            for fld in fields:
                if fld not in all_fields:
                    all_fields.append(fld)
        spec["output_fields"] = all_fields

    # Apply rich capability cards and allowed enums
    for ep_name, data in ENRICHMENTS.items():
        if ep_name in v2_cat:
            spec = v2_cat[ep_name]
            spec["capability_card"] = data["capability_card"]
            if "join_keys" in data:
                spec["join_keys"] = data["join_keys"]

            for opt in spec.get("optional_params", []):
                if opt["name"] == "play_type_nullable" and "play_type_allowed" in data:
                    opt["allowed_values"] = data["play_type_allowed"]
                elif opt["name"] == "game_segment_nullable" and "game_segment_allowed" in data:
                    opt["allowed_values"] = data["game_segment_allowed"]
                elif opt["name"] == "measure_type_detailed_defense" and "measure_type_allowed" in data:
                    opt["allowed_values"] = data["measure_type_allowed"]
                elif opt["name"] == "player_or_team_abbreviation":
                    opt["allowed_values"] = ["P", "T"]
                elif opt["name"] == "season_type_all_star":
                    opt["allowed_values"] = ["Regular Season", "Playoffs"]

    with open(cat_v2_path, "w", encoding="utf-8") as f:
        json.dump(v2_cat, f, indent=2)

    logger.info("Successfully updated endpoint_catalogue_v2.json with rich capability cards.")

    # Rebuild vector embeddings
    from router.semantic_index import SemanticEndpointIndex
    idx = SemanticEndpointIndex(catalogue_path=cat_v2_path)
    idx.build_index()
    logger.info("Dense vector index successfully rebuilt.")

if __name__ == "__main__":
    enrich()
