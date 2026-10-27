"""
Utility functions for working with NBA API endpoints.
"""
import inspect
import importlib
import pkgutil
import os
import json
from nba_api.stats import endpoints

def load_endpoint_class(name):
    """Load an endpoint class by name"""
    try:
        # First try direct import with the exact name
        try:
            module = importlib.import_module(f'nba_api.stats.endpoints.{name.lower()}')
        except ImportError:
            # If that fails, try to find the module by iterating through all endpoints
            for _, module_name, _ in pkgutil.iter_modules(endpoints.__path__):
                if module_name.lower() == name.lower():
                    module = importlib.import_module(f'nba_api.stats.endpoints.{module_name}')
                    break
            else:
                print(f"Could not find module for endpoint: {name}")
                return None
        
        # Find the class in the module
        for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
            # Check if the class name matches (case-insensitive)
            if class_name.lower() == name.lower():
                return class_obj
            # Also check if the class name matches the module name (common pattern)
            elif module.__name__.split('.')[-1].lower() == class_name.lower():
                return class_obj
        
        print(f"Could not find class in module for endpoint: {name}")
        return None
    except Exception as e:
        print(f"Error loading endpoint class {name}: {str(e)}")
        return None

def inspect_endpoint_class(class_obj):
    """Extract information about an endpoint class"""
    if class_obj is None:
        return None
        
    endpoint_info = {
        "url": getattr(class_obj, 'endpoint_url', ''),
        "description": getattr(class_obj, '__doc__', '') or f"NBA API endpoint for {class_obj.__name__}",
        "parameters": {},
        "required": [],
        "expected_output": {}  # New field for expected output
    }
    
    # Extract parameter information
    init_params = inspect.signature(class_obj.__init__).parameters
    for param_name, param in init_params.items():
        if param_name not in ['self', 'proxy', 'headers', 'timeout', 'get_request']:
            param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'str'
            
            endpoint_info["parameters"][param_name] = {
                "type": "string",
                "description": f"Parameter {param_name}",
                "enum": get_parameter_values(param.annotation) if param.annotation != inspect.Parameter.empty else None
            }
            
            if param.default == inspect.Parameter.empty:
                endpoint_info["required"].append(param_name)
    
    # Extract expected output fields - check for expected_data attribute first
    if hasattr(class_obj, 'expected_data'):
        endpoint_info["expected_output"] = class_obj.expected_data
    else:
        # Fall back to default fields for this endpoint
        endpoint_info["expected_output"] = get_default_output_fields(class_obj.__name__)
    
    return endpoint_info

def get_default_output_fields(endpoint_name):
    """Provide default output fields for common endpoints"""
    default_fields = {
        "PlayerGameLogs": {
            "GameLogs": [
                "SEASON_ID", "PLAYER_ID", "GAME_DATE", "MATCHUP", "WL",
                "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
                "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"
            ]
        },
        "PlayerFantasyProfile": {
            "OverallPlayerDashboard": [
                "GROUP_SET", "GROUP_VALUE", "GP", "W", "L",
                "MIN", "FGM", "FGA", "FG_PCT", "FG3M",
                "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
                "OREB", "DREB", "REB", "AST", "TOV",
                "STL", "BLK", "PTS", "PLUS_MINUS"
            ]
        },
        # Add more default fields for other common endpoints
    }
    return default_fields.get(endpoint_name, {"DefaultSet": ["No default fields available"]})

def get_parameter_values(param_class):
    """Extract valid values from NBA API parameter classes"""
    if not param_class:
        return None
        
    values = []
    for attr in dir(param_class):
        if not attr.startswith('_') and attr != 'default':
            value = getattr(param_class, attr)
            if isinstance(value, str):
                values.append(value)
    return values if values else None

def build_endpoint_library():
    """Extract and organize NBA API endpoints into a structured format"""
    endpoint_library = {}
    
    for _, name, _ in pkgutil.iter_modules(endpoints.__path__):
        if not name.startswith('_'):
            endpoint_class = load_endpoint_class(name)
            if endpoint_class:
                info = inspect_endpoint_class(endpoint_class)
                if info:
                    endpoint_library[endpoint_class.__name__] = info
    
    return endpoint_library

def execute_api_call(endpoint_name, params):
    """Execute an API call to a specific NBA API endpoint with strict Pydantic validation & sliding-window rate limiting."""
    from core.schemas import validate_tool_call
    from core.rate_limiter import RedisSlidingWindowRateLimiter
    from core.telemetry import TelemetryManager
    import time
    
    print(f"  Looking for endpoint class: {endpoint_name}")
    endpoint_class = load_endpoint_class(endpoint_name)
    
    if not endpoint_class:
        return {"error": f"Unknown endpoint: {endpoint_name}"}
    
    print(f"  Found endpoint class: {endpoint_name}")
    
    # 1. Enforce strict Pydantic v2 validation before making network calls
    try:
        valid_params = validate_tool_call(endpoint_class, params or {})
    except Exception as ve:
        err_msg = f"Data Contract Violation: Parameter validation failed for '{endpoint_name}': {ve}"
        print(f"  [CONTRACT REJECTION] {err_msg}")
        return {"error": err_msg}

    print(f"  Validated parameters: {json.dumps(valid_params, indent=2)}")

    # 2. Acquire token from distributed sliding-window token bucket
    try:
        rate_limiter = RedisSlidingWindowRateLimiter()
        rate_limiter.acquire_sync(tokens=1, timeout=10.0)
    except Exception as rle:
        err_msg = f"Rate Limit Exceeded: Upstream NBA API token bucket depleted: {rle}"
        print(f"  [RATE LIMIT] {err_msg}")
        return {"error": err_msg}
    
    # 3. Distributed tracing span for upstream network call
    with TelemetryManager.start_span("deepbag.nba_api_call", {"endpoint": endpoint_name, "params": json.dumps(valid_params)}) as span:
        start_time = time.time()
        try:
            from core.nba_client import DEFAULT_HEADERS
            timeout_sec = int(os.environ.get("NBA_API_TIMEOUT", "15"))
            try:
                endpoint_instance = endpoint_class(headers=DEFAULT_HEADERS, timeout=timeout_sec, **valid_params)
            except TypeError:
                endpoint_instance = endpoint_class(**valid_params)
            print(f"  Successfully created endpoint instance")
        except Exception as e:
            print(f"  ERROR creating endpoint instance: {str(e)}")
            TelemetryManager.record_api_call(endpoint_name, 500, time.time() - start_time, str(e))
            return {"error": f"Failed to create endpoint instance: {str(e)}"}
        
        from utils.data_utils import process_api_dataframes
        
        # 4. Fetch data from upstream API
        try:
            print(f"  Fetching data from NBA API...")
            data = endpoint_instance.get_data_frames()
            duration = time.time() - start_time
            print(f"  Successfully fetched data in {duration:.2f}s. Got {len(data)} dataframes")
            TelemetryManager.record_api_call(endpoint_name, 200, duration)
        except Exception as e:
            duration = time.time() - start_time
            print(f"  ERROR fetching data: {str(e)}")
            TelemetryManager.record_api_call(endpoint_name, 502, duration, str(e))
            return {"error": f"Error fetching data: {str(e)}"}
        
        # 5. Convert dataframes
        try:
            result = process_api_dataframes(data)
            print(f"  Successfully processed all dataframes")
            return result
        except Exception as e:
            print(f"  ERROR processing dataframes: {str(e)}")
            return {"error": f"Error processing dataframes: {str(e)}"}