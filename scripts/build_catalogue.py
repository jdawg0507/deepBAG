"""
Script to build the verified machine-readable endpoint catalogue from nba_api.
Introspects all endpoint modules in nba_api.stats.endpoints, extracting:
- Endpoint class name & module
- URL & docstrings
- Parameters (required, optional, defaults, type annotations)
- Dataset names and field signatures
- Scope capabilities (season, season_type, per_mode, measure_type, game_segment, period)
Outputs data/endpoint_catalogue.json.
"""
import os
import json
import inspect
import importlib
import pkgutil
from typing import Dict, Any, List
import nba_api.stats.endpoints as endpoints_pkg

IGNORE_PARAMS = {'self', 'proxy', 'headers', 'timeout', 'get_request'}

KNOWN_PARAM_PATTERNS = {
    'season': ['season', 'season_nullable', 'season_id'],
    'season_type': ['season_type_all_star', 'season_type_playoffs', 'season_type_nullable'],
    'per_mode': ['per_mode_detailed', 'per_mode_simple', 'per_mode_nullable'],
    'measure_type': ['measure_type_detailed_defense', 'measure_type_simple', 'measure_type_base'],
    'game_segment': ['game_segment_nullable'],
    'period': ['period'],
    'player_id': ['player_id', 'player_id_nullable'],
    'team_id': ['team_id', 'team_id_nullable'],
    'opponent_team_id': ['opponent_team_id', 'opponent_team_id_nullable'],
    'location': ['location_nullable'],
    'outcome': ['outcome_nullable'],
    'last_n_games': ['last_n_games_nullable', 'last_n_games'],
    'date_from': ['date_from_nullable'],
    'date_to': ['date_to_nullable'],
}

def analyze_endpoint_class(class_obj, module_name: str) -> Dict[str, Any]:
    name = class_obj.__name__
    doc = getattr(class_obj, '__doc__', '') or ''
    url = getattr(class_obj, 'endpoint_url', '') or ''
    
    # Inspect parameters from __init__
    init_sig = inspect.signature(class_obj.__init__)
    params_info = {}
    required_params = []
    optional_params = []
    
    param_patterns = []
    
    for param_name, param in init_sig.parameters.items():
        if param_name in IGNORE_PARAMS:
            continue
            
        is_required = (param.default == inspect.Parameter.empty)
        default_val = None if is_required else (
            repr(param.default) if not isinstance(param.default, (str, int, float, bool, type(None)))
            else param.default
        )
        
        # Check matching pattern category
        for pattern_cat, pattern_keys in KNOWN_PARAM_PATTERNS.items():
            if param_name.lower() in [k.lower() for k in pattern_keys]:
                if pattern_cat not in param_patterns:
                    param_patterns.append(pattern_cat)
        
        param_data = {
            "name": param_name,
            "required": is_required,
            "default": default_val,
        }
        
        params_info[param_name] = param_data
        if is_required:
            required_params.append(param_name)
        else:
            optional_params.append(param_name)
            
    # Inspect datasets and expected output fields
    expected_data = getattr(class_obj, 'expected_data', {}) or {}
    dataset_names = []
    fields_by_dataset = {}
    
    if isinstance(expected_data, dict):
        for ds_name, fields in expected_data.items():
            dataset_names.append(ds_name)
            if isinstance(fields, (list, tuple)):
                fields_by_dataset[ds_name] = list(fields)
            elif isinstance(fields, dict):
                fields_by_dataset[ds_name] = list(fields.keys())
    
    return {
        "name": name,
        "module": module_name,
        "url": url,
        "description": doc.strip().split('\n')[0] if doc else f"NBA stats endpoint for {name}",
        "full_doc": doc.strip() if doc else "",
        "parameters": params_info,
        "required_parameters": required_params,
        "optional_parameters": optional_params,
        "param_patterns": sorted(param_patterns),
        "datasets": dataset_names,
        "fields_by_dataset": fields_by_dataset,
    }

def build_catalogue() -> Dict[str, Any]:
    catalogue = {}
    
    for _, module_name, _ in pkgutil.iter_modules(endpoints_pkg.__path__):
        try:
            mod = importlib.import_module(f"nba_api.stats.endpoints.{module_name}")
            for item_name, item_obj in inspect.getmembers(mod, inspect.isclass):
                # Ensure the class belongs to this module
                if item_obj.__module__ == mod.__name__ and hasattr(item_obj, '__init__'):
                    info = analyze_endpoint_class(item_obj, module_name)
                    catalogue[info["name"]] = info
        except Exception as e:
            print(f"Error inspecting module {module_name}: {e}")
            
    return catalogue

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "endpoint_catalogue.json")
    
    print("Building endpoint catalogue from nba_api...")
    catalogue = build_catalogue()
    print(f"Discovered {len(catalogue)} endpoint classes.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalogue, f, indent=2, sort_keys=True)
        
    print(f"Catalogue written to: {output_path}")

if __name__ == "__main__":
    main()
