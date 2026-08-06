import google.generativeai as genai
import json
from nba_api.stats import endpoints
from nba_api.stats.library import parameters
import inspect
import importlib
import pkgutil
import pandas as pd
from nba_api.stats.static import players
from .api.endpoint_utils import execute_api_call

class NBAQueryAnalyzer:
    def __init__(self, api_key):
        self._setup_model(api_key)
        self.parameter_library = self._build_parameter_library()
        self.endpoint_library = self._build_endpoint_library()

    def _setup_model(self, api_key):
        """Initialize the Gemini model with the provided API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b')

    def _build_parameter_library(self):
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

    def _build_endpoint_library(self):
        """Extract and organize NBA API endpoints into a structured format"""
        endpoint_library = {}
        
        for _, name, _ in pkgutil.iter_modules(endpoints.__path__):
            if not name.startswith('_'):
                endpoint_class = self._load_endpoint_class(name)
                if endpoint_class:
                    info = self._inspect_endpoint_class(endpoint_class)
                    if info:
                        endpoint_library[endpoint_class.__name__] = info
        
        return endpoint_library

    def _load_endpoint_class(self, name):
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

    def _inspect_endpoint_class(self, class_obj):
        """Extract information about an endpoint class"""
        if class_obj is None:
            return None
            
        endpoint_info = {
            "url": getattr(class_obj, 'endpoint_url', ''),
            "description": getattr(class_obj, '__doc__', '') or f"NBA API endpoint for {class_obj.__name__}",
            "parameters": {},
            "required": []
        }
        
        init_params = inspect.signature(class_obj.__init__).parameters
        for param_name, param in init_params.items():
            if param_name not in ['self', 'proxy', 'headers', 'timeout', 'get_request']:
                param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'str'
                
                endpoint_info["parameters"][param_name] = {
                    "type": "string",
                    "description": f"Parameter {param_name}",
                    "enum": self._get_parameter_values(param.annotation) if param.annotation != inspect.Parameter.empty else None
                }
                
                if param.default == inspect.Parameter.empty:
                    endpoint_info["required"].append(param_name)
        
        return endpoint_info

    def _get_parameter_values(self, param_class):
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

    def extract_parameters(self, query):
        """Extract parameters from the user query using Gemini"""
        # Format parameters for Gemini
        param_summary = []
        for class_name, values in self.parameter_library.items():
            param_summary.append(f"class {class_name}:")
            # Only include a few example values to keep the prompt size manageable
            example_values = list(values.items())[:5]  # Take up to 5 examples
            for attr_name, attr_value in example_values:
                param_summary.append(f"    {attr_name} = \"{attr_value}\"")
            if len(values) > 5:
                param_summary.append(f"    # ... and {len(values) - 5} more values")
            param_summary.append("")
        
        param_prompt = f"""
        Given this NBA stats query: "{query}"
        
        I need to extract specific NBA API parameters that directly relate to this query.
        
        For example, if the query mentions:
        - A player name → Extract as "player_name": "Player Full Name"
        - A specific season → Use "Season": "YYYY-YY" format (like "2023-24")
        - Recent games → Use "LastNGames": "N" (like "1" for last game)
        - Regular season vs playoffs → Use "SeasonType": "Type" (like "Regular Season" or "Playoffs")
        
        Here are the available parameter classes in the NBA API (showing examples):
        {"\n".join(param_summary)}
        
        Based on the query "{query}", identify the specific parameters needed.
        
        Return a JSON object with parameter names as keys and their values from the query.
        For example: {{"player_name": "LeBron James", "LastNGames": "1", "Season": "2023-24", "SeasonType": "Regular Season"}}
        
        Return ONLY the JSON object without any markdown formatting or code blocks.
        """
        
        response = self.model.generate_content(
            param_prompt,
            generation_config={'temperature': 0.1}
        )
        
        try:
            # Clean the response text to handle potential markdown formatting
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            response_text = response_text.strip()
            
            extracted_params = json.loads(response_text)
            
            # Process player names to get player IDs
            if "player_name" in extracted_params:
                player_id = self.get_player_id_from_name(extracted_params["player_name"])
                if player_id:
                    extracted_params["player_id"] = player_id
            
            return extracted_params
        except json.JSONDecodeError as e:
            print("Error parsing JSON response from Gemini. Raw response:")
            print(response.text)
            print(f"JSON error: {str(e)}")
            return {
                "player_name": None,
                "season": None,
                "stat_type": None,
                "filters": {}
            }
    
    def get_player_id_from_name(self, player_name):
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

    def identify_endpoints(self, query, extracted_params):
        """Identify the most relevant endpoints based on query and extracted parameters"""
        # Create a summary of available endpoints
        endpoint_summary = {}
        for name, info in self.endpoint_library.items():
            endpoint_summary[name] = {
                "description": info["description"],
                "required_params": info["required"]
            }
        
        # Add information about what parameters we've extracted
        param_info = []
        for key, value in extracted_params.items():
            param_info.append(f"- {key}: {value}")
        
        endpoint_prompt = f"""
        The user asked: "{query}"
        
        Given these parameters extracted from the query:
        {json.dumps(extracted_params, indent=2)}
        
        Parameters summary:
        {chr(10).join(param_info)}
        
        You need to identify which NBA API endpoints would be most relevant to answer this query.
        
        Think creatively about how to answer this query. Consider:
        1. What data would directly answer the question?
        2. If the perfect endpoint isn't available, what combination of endpoints could provide the data needed?
        3. What calculations might be needed on the raw data to derive the answer?
        
        Here's a summary of available endpoints:
        {json.dumps(endpoint_summary, indent=2)}
        
        Return a JSON array with the names of up to 3 most relevant endpoints that together could answer the query.
        Return ONLY the JSON array without any markdown formatting or code blocks.
        """
        
        response = self.model.generate_content(
            endpoint_prompt,
            generation_config={'temperature': 0.3}
        )
        
        try:
            # Clean the response text to handle potential markdown formatting
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            response_text = response_text.strip()
            
            endpoints = json.loads(response_text)
            return endpoints if isinstance(endpoints, list) else [endpoints]
        except json.JSONDecodeError as e:
            print("Error parsing JSON response from Gemini. Raw response:")
            print(response.text)
            print(f"JSON error: {str(e)}")
            return ["PlayerGameLogs"]

    def prepare_api_call(self, endpoint_name, extracted_params):
        """Prepare parameters for an API call to a specific endpoint"""
        endpoint_info = self.endpoint_library.get(endpoint_name, {})
        
        # Print the required parameters for this endpoint
        if "required" in endpoint_info:
            print(f"  Required parameters for {endpoint_name}: {', '.join(endpoint_info['required'])}")
        
        # Get the valid parameters for this endpoint
        valid_params = list(endpoint_info.get("parameters", {}).keys())
        print(f"  Valid parameters for {endpoint_name}: {', '.join(valid_params)}")
        
        call_prompt = f"""
        I need to call the NBA API endpoint "{endpoint_name}" with these parameters:
        {json.dumps(extracted_params, indent=2)}
        
        The endpoint accepts only these parameters:
        {', '.join(valid_params)}
        
        Required parameters are:
        {', '.join(endpoint_info.get('required', []))}
        
        Return a JSON object with parameters for this API call.
        Include only parameters that are valid for this endpoint.
        Map the extracted parameters to the correct parameter names for this endpoint.
        For required parameters that aren't in the extracted parameters, use reasonable defaults.
        Return ONLY the JSON object without any markdown formatting or code blocks.
        """
        
        response = self.model.generate_content(
            call_prompt,
            generation_config={'temperature': 0.1}
        )
        
        try:
            # Clean the response text to handle potential markdown formatting
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            response_text = response_text.strip()
            
            call_params = json.loads(response_text)
            
            # Filter out any invalid parameters
            filtered_params = {}
            for param, value in call_params.items():
                if param in valid_params:
                    filtered_params[param] = value
                else:
                    print(f"  WARNING: Removing invalid parameter: {param}")
            
            # Check for missing required parameters
            if "required" in endpoint_info:
                missing_params = [param for param in endpoint_info["required"] if param not in filtered_params]
                if missing_params:
                    print(f"  WARNING: Missing required parameters for {endpoint_name}: {', '.join(missing_params)}")
                    
                    # Let Gemini try again with more specific instructions about missing parameters
                    missing_prompt = f"""
                    I need to call the NBA API endpoint "{endpoint_name}" but I'm missing these required parameters:
                    {', '.join(missing_params)}
                    
                    Based on this query: "{query}"
                    And these extracted parameters:
                    {json.dumps(extracted_params, indent=2)}
                    
                    Return a JSON object with ONLY the missing parameters and appropriate values for them.
                    Return ONLY the JSON object without any markdown formatting or code blocks.
                    """
                    
                    missing_response = self.model.generate_content(
                        missing_prompt,
                        generation_config={'temperature': 0.1}
                    )
                    
                    try:
                        missing_text = missing_response.text.strip()
                        if missing_text.startswith("```json"):
                            missing_text = missing_text.replace("```json", "", 1)
                        if missing_text.endswith("```"):
                            missing_text = missing_text.rsplit("```", 1)[0]
                        missing_text = missing_text.strip()
                        
                        missing_params_values = json.loads(missing_text)
                        
                        # Add the missing parameters
                        for param, value in missing_params_values.items():
                            if param in missing_params:
                                filtered_params[param] = value
                                print(f"  Added missing parameter {param}: {value}")
                    except Exception as e:
                        print(f"  ERROR getting missing parameters: {str(e)}")
                        # If we still have missing parameters, use some common defaults
                        for param in missing_params:
                            if param == "player_id" and "player_id" in extracted_params:
                                filtered_params[param] = extracted_params["player_id"]
                            elif param == "season":
                                filtered_params[param] = "2023-24"
                            elif param == "season_type_all_star" or param == "season_type":
                                filtered_params[param] = "Regular Season"
                            elif param == "game_id" and endpoint_name.lower().startswith("boxscore"):
                                filtered_params[param] = "0022300001"  # Default to first game of 2023-24 season
                            else:
                                print(f"  ERROR: No default value for required parameter: {param}")
            
            return filtered_params
        except json.JSONDecodeError as e:
            print("Error parsing JSON response from Gemini. Raw response:")
            print(response.text)
            print(f"JSON error: {str(e)}")
            return {}
        # Add this method after the prepare_api_call method
    def execute_api_call(self, endpoint_name, call_params):
        """Execute an NBA API call with the given endpoint and parameters"""
        try:
            print(f"  Looking for endpoint class: {endpoint_name}")
            # Find the endpoint class
            endpoint_class = self._load_endpoint_class(endpoint_name)
            if not endpoint_class:
                print(f"  ERROR: Endpoint {endpoint_name} not found")
                return {"error": f"Endpoint {endpoint_name} not found"}
            
            print(f"  Found endpoint class: {endpoint_class.__name__}")
            print(f"  Creating endpoint instance with parameters: {json.dumps(call_params, indent=2)}")
            
            # Create an instance of the endpoint class with the provided parameters
            try:
                endpoint_instance = endpoint_class(**call_params)
                print(f"  Successfully created endpoint instance")
            except Exception as e:
                print(f"  ERROR creating endpoint instance: {str(e)}")
                return {"error": f"Error creating endpoint instance: {str(e)}"}
            
            # Get the data from the API
            try:
                print(f"  Fetching data from API...")
                data = endpoint_instance.get_data_frames()
                print(f"  Successfully fetched data. Got {len(data)} dataframes")
            except Exception as e:
                print(f"  ERROR fetching data: {str(e)}")
                return {"error": f"Error fetching data: {str(e)}"}
            
            # Convert to a more manageable format
            try:
                result = {}
                for i, df in enumerate(data):
                    print(f"  Processing dataframe {i} with {len(df)} rows and {len(df.columns)} columns")
                    if len(df) > 0:
                        print(f"  Column names: {', '.join(df.columns.tolist())}")
                    result[f"table_{i}"] = json.loads(df.to_json(orient="records"))
                
                print(f"  Successfully processed all dataframes")
                return result
            except Exception as e:
                print(f"  ERROR processing dataframes: {str(e)}")
                return {"error": f"Error processing dataframes: {str(e)}"}
            
        except Exception as e:
            print(f"  ERROR executing API call to {endpoint_name}: {str(e)}")
            return {"error": str(e)}
    
    # Also add this method for formatting the response
    def format_response(self, query, api_results, extracted_params):
        """Format the API results into a user-friendly response"""
        # Process the data to extract relevant information
        processed_data = {}
        
        # Check if we have any successful API calls
        has_successful_call = False
        for endpoint, result in api_results.items():
            if "error" not in result:
                has_successful_call = True
                break
        
        # Extract player name for reference
        player_name = extracted_params.get("player_name", "the player")
        
        format_prompt = f"""
        The user asked: "{query}"
        
        Based on the query, we extracted these parameters:
        {json.dumps(extracted_params, indent=2)}
        
        Here are the results from the NBA API:
        {json.dumps(api_results, indent=2)}
        
        I need you to analyze this data thoroughly and answer the user's question.
        
        Important instructions:
        1. Be creative and resourceful with the data. If the answer isn't directly available, find ways to derive it.
        
        2. You may need to:
           - Perform calculations on the data (sums, averages, percentages, etc.)
           - Combine data from multiple endpoints
           - Filter data based on criteria in the query
           - Look for patterns or trends in the data
        
        3. If you need to make calculations:
           - Clearly explain your methodology
           - Show the key numbers that led to your conclusion
        
        4. If the data is incomplete or doesn't fully answer the query:
           - Provide the best answer possible with the available data
           - Explain what additional data would be needed for a complete answer
        
        5. Format your response in a user-friendly way with relevant statistics.
        
        6. Don't just say there's no data - try to extract whatever insights you can from what's available.
        """
        
        response = self.model.generate_content(
            format_prompt,
            generation_config={'temperature': 0.7}
        )
        
        return response.text

    def process_query(self, query):
        """Process a user query and return a response"""
        try:
            print("\n=== Query ===")
            print(query)
            
            # Step 1: Extract parameters from the query
            extracted_params = self.extract_parameters(query)
            print("\n=== Extracted Parameters ===")
            print(json.dumps(extracted_params, indent=2))
            
            # Step 2: Identify relevant endpoints
            relevant_endpoints = self.identify_endpoints(query, extracted_params)
            print("\n=== Relevant Endpoints ===")
            print(json.dumps(relevant_endpoints, indent=2))
            
            # Step 3: Execute API calls
            api_results = {}
            for endpoint in relevant_endpoints:
                print(f"\n=== Preparing API Call for {endpoint} ===")
                call_params = self.prepare_api_call(endpoint, extracted_params)
                print(f"Parameters: {json.dumps(call_params, indent=2)}")
                
                print(f"\n=== Executing API Call for {endpoint} ===")
                result = self.execute_api_call(endpoint, call_params)
                
                # Check if there was an error
                if "error" in result:
                    print(f"  API call failed: {result['error']}")
                else:
                    print(f"  API call successful")
                    # Print a sample of the data
                    for table_name, table_data in result.items():
                        print(f"  {table_name}: {len(table_data)} records")
                        if len(table_data) > 0:
                            print(f"  Sample data: {json.dumps(table_data[0], indent=2)[:200]}...")
                
                api_results[endpoint] = result
            
            # Step 4: Format the response
            print("\n=== Formatting Response ===")
            response = self.format_response(query, api_results, extracted_params)
            
            print("\n=== Final Response ===")
            print(response)
            return response
                
        except Exception as e:
            import traceback
            print(f"\n=== Error Details ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    analyzer = NBAQueryAnalyzer(api_key="AIzaSyDGvmWXOZRTyqWl_MhcYEAU2jvKV5mS8vA")
    result = analyzer.process_query("what was lebrons statline averages the past 5 games?")
    print("\n=== Analysis Result ===")
    print(json.dumps(result, indent=2) if isinstance(result, dict) else result)