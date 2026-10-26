"""
Utility functions for working with NBA API parameters.
"""
import inspect
from nba_api.stats.library import parameters
from nba_api.stats.static import players

def build_parameter_library():
    """Extract and organize NBA API parameters into a structured format"""
    param_library = {}
    
    # Get all parameter classes from the NBA API
    for name, obj in inspect.getmembers(parameters):
        if inspect.isclass(obj) and not name.startswith('_'):
            param_values = {}
            for attr in dir(obj):
                if not attr.startswith('_') and attr != 'default':
                    value = getattr(obj, attr)
                    if isinstance(value, str):
                        param_values[attr] = value
            
            if param_values:
                param_library[name] = param_values
    
    return param_library

def get_player_id_from_name(player_name):
    """Convert a player name to their NBA API player ID"""
    if not player_name:
        return None
        
    # Try exact match on full name first
    player_list = players.find_players_by_full_name(f"^{player_name}$")
    
    # If no exact match, try partial match on full name
    if not player_list:
        player_list = players.find_players_by_full_name(player_name)
    
    # If still no match, try last name
    if not player_list:
        player_list = players.find_players_by_last_name(player_name)
    
    # If multiple matches, prioritize active players
    if len(player_list) > 1:
        active_players = [p for p in player_list if p['is_active']]
        if active_players:
            player_list = active_players
    
    # Return the ID of the first match, or None if no matches
    return player_list[0]['id'] if player_list else None

def validate_parameters(params, endpoint_info):
    """Validate parameters against endpoint requirements"""
    valid_params = list(endpoint_info.get("parameters", {}).keys())
    required_params = endpoint_info.get("required", [])
    
    # Filter out invalid parameters
    filtered_params = {k: v for k, v in params.items() if k in valid_params}
    
    # Check for missing required parameters
    missing_params = [param for param in required_params if param not in filtered_params]
    
    return filtered_params, missing_params