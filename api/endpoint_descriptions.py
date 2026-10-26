"""
Enhanced descriptions for NBA API endpoints.
"""

ENHANCED_DESCRIPTIONS = {
    "LeagueDashPlayerStats": "Provides season averages for all players including points, rebounds, assists per game. Good for player season averages and rankings.",
    
    "PlayerGameLogs": "Contains game-by-game stats for a specific player. Good for analyzing individual game performance, recent trends, and game-specific stats.",
    
    "PlayerCareerStats": "Provides career totals and averages for a player. Good for career statistics, season-by-season breakdowns, and historical performance.",
    
    "PlayerDashboardByYearOverYear": "Shows player stats broken down by season. Good for comparing performance across seasons and tracking player development.",
    
    "CommonPlayerInfo": "Contains biographical information about a player including height, weight, birthdate, draft position, etc.",
    
    "PlayerDashPtShots": "Contains detailed shooting statistics for a player including shot locations, distances, and efficiency.",
    
    "BoxScoreTraditionalV2": "Contains detailed box score stats for a specific game including points, rebounds, assists, etc. Requires game_id.",
    
    "BoxScoreAdvancedV2": "Contains advanced box score stats for a specific game including offensive/defensive rating, usage rate, etc. Requires game_id.",
    
    "TeamGameLogs": "Contains game-by-game stats for a specific team. Good for analyzing team performance over time.",
    
    "TeamDashboardByGeneralSplits": "Contains team stats broken down by various splits like home/away, wins/losses, etc.",
    
    "TeamPlayerDashboard": "Contains stats for all players on a team. Good for comparing players within a team.",
    
    "TeamYearByYearStats": "Contains team stats broken down by season. Good for historical team performance analysis.",
    
    "LeagueGameFinder": "Helps find games based on various criteria like teams, players, dates, etc. Good for finding specific games.",
    
    "PlayerDashboardByLastNGames": "Shows player stats broken down by the last N games. Good for analyzing recent performance trends.",
    
    "PlayerDashboardByOpponent": "Shows player stats broken down by opponent. Good for matchup analysis.",
    
    "ShotChartDetail": "Provides detailed shot chart data for visualizing shooting patterns and efficiency from different locations."
}

def enhance_endpoint_descriptions(endpoint_library):
    """Add enhanced descriptions to the endpoint library"""
    for endpoint_name, description in ENHANCED_DESCRIPTIONS.items():
        if endpoint_name in endpoint_library:
            endpoint_library[endpoint_name]["description"] = description
    
    return endpoint_library