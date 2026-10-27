"""
Module for selecting and preparing NBA API endpoints using Gemini.
"""
import json
from utils.data_utils import clean_json_response  # Changed from relative import
from api.parameter_utils import validate_parameters  # Changed from relative import

class EndpointSelector:
    def __init__(self, model, endpoint_library):
        self.model = model
        self.endpoint_library = endpoint_library
        self.explored_data = {}  # Store data we've found during exploration
    
    def identify_endpoints(self, query, extracted_params, depth=0, max_depth=3):
        """Recursively identify and explore endpoints to answer the query"""
        if depth >= max_depth:
            print("Max exploration depth reached")
            return []
        
        exploration_prompt = f"""
        Analyze this NBA query: "{query}"
        
        Current parameters:
        {json.dumps(extracted_params, indent=2)}
        
        Available endpoints:
        {json.dumps(self.endpoint_library, indent=2)}
        
        Return a JSON object with:
        {{
            "primary_endpoints": ["endpoints that directly answer the query"],
            "fallback_endpoints": ["endpoints that might provide useful context"],
            "needed_parameters": ["parameters we should look for in other endpoints"]
        }}
        """
        
        response = self.model.generate_content(
            exploration_prompt,
            generation_config={'temperature': 0.2}
        )
        
        try:
            plan = json.loads(clean_json_response(response.text))
            
            # Try primary endpoints first
            primary_endpoints = [ep for ep in plan.get("primary_endpoints", [])
                           if self._can_use_endpoint(ep, extracted_params)]
            
            if primary_endpoints:
                return primary_endpoints
                
            # If primary endpoints aren't available, look for data in fallback endpoints
            fallback_endpoints = [ep for ep in plan.get("fallback_endpoints", [])
                            if self._can_use_endpoint(ep, extracted_params)]
            
            if fallback_endpoints:
                # Try to get additional parameters from fallback endpoints
                new_params = extracted_params.copy()
                for endpoint in fallback_endpoints:
                    params = self.prepare_api_call(endpoint, extracted_params, query)
                    if params:
                        response = execute_api_call(endpoint, params)
                        if response:
                            # Extract potential parameters from response
                            for param in plan.get("needed_parameters", []):
                                if param in response:
                                    new_params[param] = response[param]
                
                # If we found new parameters, try again with updated params
                if new_params != extracted_params:
                    print("Found new parameters, exploring further...")
                    return self.identify_endpoints(query, new_params, depth + 1, max_depth)
                
            return fallback_endpoints
            
        except Exception as e:
            print(f"Error in endpoint exploration: {str(e)}")
            return []
    
    def _can_use_endpoint(self, endpoint, params):
        """Check if we can use an endpoint with current parameters"""
        endpoint_info = self.endpoint_library.get(endpoint, {})
        required_params = endpoint_info.get("required", [])
        return all(param in params for param in required_params)
    
    def prepare_api_call(self, endpoint, extracted_params, query):
        """Prepare parameters for an API call to a specific endpoint"""
        print(f"Preparing API call for {endpoint}...")
        
        context_prompt = f"""
        For this NBA query: "{query}"
        Using endpoint: {endpoint}
        
        Available parameters:
        {json.dumps(extracted_params, indent=2)}
        
        Previously collected data:
        {json.dumps(self.explored_data, indent=2)}
        
        Endpoint requirements:
        {json.dumps(self.endpoint_library.get(endpoint, {}), indent=2)}
        
        Return a JSON object with only the parameters needed for this specific endpoint call.
        """
        
        response = self.model.generate_content(
            context_prompt,
            generation_config={'temperature': 0.1}
        )
        
        try:
            params = json.loads(clean_json_response(response.text))
            return {k: v for k, v in params.items() if v is not None}
        except Exception as e:
            print(f"Error preparing parameters: {str(e)}")
            return None

    def _handle_missing_parameters(self, endpoint, missing_params, current_params, extracted_params, query, endpoint_info):
        """Handle missing required parameters for an endpoint"""
        print(f"Handling missing parameters for {endpoint}: {missing_params}")
        
        # Initialize filtered_params with current parameters
        filtered_params = current_params.copy()
        
        # Handle player_id specifically if it's missing
        if "player_id" in missing_params:
            from entities.resolver import EntityResolver
            resolver = EntityResolver()
            player_name = extracted_params.get("Player")
            
            if player_name:
                resolved = resolver.resolve(player_name)
                if resolved and resolved.type == "player":
                    filtered_params["player_id"] = resolved.id
                    print(f"  Added player_id {resolved.id} for {resolved.name}")
                    missing_params.remove("player_id")
            else:
                # Dynamically resolve any player mentioned in query without hardcoded name lists
                all_entities = resolver.resolve_all(query)
                for ent in all_entities:
                    if ent.type == "player":
                        filtered_params["player_id"] = ent.id
                        print(f"  Added player_id {ent.id} for {ent.name}")
                        missing_params.remove("player_id")
                        break
        
        # Only proceed with LLM if we still have missing parameters
        if missing_params:
            # Create a prompt to determine values for missing parameters
            prompt = f"""
            I need to determine values for missing required parameters for the NBA API endpoint: {endpoint}
            
            This endpoint provides: {endpoint_info.get("description", "")}
            Missing required parameters: {missing_params}
            Current parameters: {json.dumps(current_params, indent=2)}
            Expected output: {json.dumps(endpoint_info.get("expected_output", {}), indent=2)}
            
            The query is: "{query}"
            
            I've extracted these parameters:
            {json.dumps(extracted_params, indent=2)}
            
            For each missing parameter, determine the most appropriate value based on the query context.
            
            Return a JSON object with parameter names and values for ALL parameters (both current and missing).
            """
        
            missing_response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.1}
            )
            
            try:
                missing_text = clean_json_response(missing_response.text)
                missing_params_values = json.loads(missing_text)
                
                # Add the missing parameters
                for param, value in missing_params_values.items():
                    if param in missing_params:
                        filtered_params[param] = value
                        print(f"  Added missing parameter {param}: {value}")
            except Exception as e:
                print(f"  ERROR getting missing parameters: {str(e)}")
                # Use default values for common parameters
                for param in missing_params:
                    if param == "player_id" and "player_id" in extracted_params:
                        filtered_params[param] = extracted_params["player_id"]
                    elif param == "season":
                        filtered_params[param] = "2023-24"
                    elif param == "season_type_all_star" or param == "season_type":
                        filtered_params[param] = "Regular Season"
                    else:
                        print(f"  ERROR: No default value for required parameter: {param}")
            
            return filtered_params