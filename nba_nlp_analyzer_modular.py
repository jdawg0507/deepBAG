"""
NBA NLP Analyzer (Modular) - Main module for processing natural language queries about NBA statistics.
"""
import json
import traceback
import google.generativeai as genai

from api.endpoint_utils import build_endpoint_library, execute_api_call
from api.parameter_utils import build_parameter_library, get_player_id_from_name
from api.endpoint_descriptions import enhance_endpoint_descriptions
from utils.data_utils import clean_json_response
from nlp.parameter_extractor import ParameterExtractor
from nlp.endpoint_selector import EndpointSelector
from nlp.response_formatter import ResponseFormatter
from nlp.tool_retriever import EndpointToolRetriever
from nlp.tool_builder import EndpointToolBuilder
from nlp.agent import NBAAgent

from query_cache import QueryCache

class NBAQueryAnalyzer:
    def __init__(self, api_key):
        """Initialize the NBA Query Analyzer with the Gemini API key"""
        # Configure Gemini API
        self._setup_model(api_key)
        
        # Build libraries of NBA API endpoints and parameters
        print("Building NBA API endpoint library...")
        self.endpoint_library = build_endpoint_library()
        # Enhance endpoint descriptions
        self.endpoint_library = enhance_endpoint_descriptions(self.endpoint_library)
        print(f"Found {len(self.endpoint_library)} endpoints")
        
        print("Building NBA API parameter library...")
        self.parameter_library = build_parameter_library()
        print(f"Found {len(self.parameter_library)} parameter classes")
        
        # Initialize NLP components
        self.parameter_extractor = ParameterExtractor(self.model, self.parameter_library)
        self.endpoint_selector = EndpointSelector(self.model, self.endpoint_library)
        self.response_formatter = ResponseFormatter(self.model)
        
        # Initialize Agentic components
        print("Initializing Agentic components...")
        self.tool_retriever = EndpointToolRetriever(self.endpoint_library)
        self.tool_builder = EndpointToolBuilder(self.endpoint_library)
        self.agent = NBAAgent()
        self.cache = QueryCache()
    
    def process_query_agentic(self, query):
        """Process an NBA stats query using the new Gemini Function Calling Agent"""
        print(f"\nProcessing query (Agentic): {query}")
        
        from nlp.meta_tools import get_meta_tools_declarations
        from nlp.sql_tool import get_sql_tool_declaration
        
        # 1. Check Cache first
        cached_response = self.cache.get(query)
        if cached_response:
            print("Found response in cache!")
            yield {"type": "thought", "content": "Found exact match in local query cache. Skipping API calls..."}
            yield {"type": "final_answer", "content": cached_response}
            return
            
        try:
            # 2. Build the static list of tools (Meta-Tools + SQL Tool)
            tools = get_meta_tools_declarations()
            tools.append(get_sql_tool_declaration())
                
            # 3. Run the Agentic Loop (Streaming)
            final_text = ""
            for event in self.agent.run_stream(query, tools, self.endpoint_library):
                yield event
                if event.get("type") == "final_answer":
                    final_text = event.get("content", "")
            
            # Save final response to cache if valid
            if final_text and not final_text.startswith("Error"):
                self.cache.set(query, final_text)
            
        except Exception as e:
            print(f"Error in agentic processing: {str(e)}")
            import traceback
            traceback.print_exc()
            yield {"type": "final_answer", "content": f"Error: {str(e)}"}
            
    def _setup_model(self, api_key):
        """Initialize the Gemini model with the provided API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    

    def process_query_with_planning(self, query):
        """Process an NBA stats query with a planning phase"""
        try:
            # Extract initial parameters
            extracted_params = self.parameter_extractor.extract_parameters(query)
            print("\nExtracted parameters:")
            print(json.dumps(extracted_params, indent=2))
            
            # Initialize data collection
            collected_data = {}
            
            # First attempt: try with initial parameters
            endpoints = self.endpoint_selector.identify_endpoints(query, extracted_params)
            
            for endpoint in endpoints:
                params = self.endpoint_selector.prepare_api_call(endpoint, extracted_params, query)
                if params:
                    response = execute_api_call(endpoint, params)
                    if response:
                        collected_data[endpoint] = response
            
            # Check if we can answer the query with collected data
            answer_check_prompt = f"""
            Given this NBA stats query: "{query}"
            
            Data collected:
            {json.dumps(collected_data, indent=2)}
            
            Can we answer the query with this data? Return JSON:
            {{
                "can_answer": true/false,
                "missing_data": ["list of missing data points if any"]
            }}
            """
            
            check_response = self.model.generate_content(
                answer_check_prompt,
                generation_config={'temperature': 0.1}
            )
            
            check_result = json.loads(clean_json_response(check_response.text))
            
            if check_result["can_answer"]:
                final_answer = self.response_formatter.format_response(query, collected_data, extracted_params)
            else:
                print("\nNeed more data. Planning additional queries...")
                # Update parameters with collected data
                for endpoint_data in collected_data.values():
                    extracted_params.update(self._extract_new_params(endpoint_data))
                
                # Try another round of endpoint calls with updated parameters
                new_endpoints = self.endpoint_selector.identify_endpoints(
                    query, 
                    extracted_params,
                    depth=1
                )
                
                for endpoint in new_endpoints:
                    if endpoint not in collected_data:
                        params = self.endpoint_selector.prepare_api_call(
                            endpoint, 
                            extracted_params, 
                            query
                        )
                        if params:
                            response = execute_api_call(endpoint, params)
                            if response:
                                collected_data[endpoint] = response
            
                final_answer = self.response_formatter.format_response(query, collected_data, extracted_params)
            
            return final_answer
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            traceback.print_exc()  # Added for better error tracking
            return "I encountered an error processing your query. Please try rephrasing it."
    
    def _extract_new_params(self, endpoint_data):
        """Extract potential new parameters from endpoint response data"""
        new_params = {}
        try:
            # Let the LLM identify useful parameters from the response
            extraction_prompt = f"""
            Analyze this NBA API response data:
            {json.dumps(endpoint_data, indent=2)}
            
            Identify any values that could be used as parameters for other API calls.
            Return them as a JSON object mapping parameter names to values.
            """
            
            response = self.model.generate_content(
                extraction_prompt,
                generation_config={'temperature': 0.1}
            )
            
            new_params = json.loads(clean_json_response(response.text))
        except Exception as e:
            print(f"Error extracting new parameters: {str(e)}")
        
        return new_params

    def _create_data_summary(self, api_results):
        """Create a summary of API results for analysis"""
        summary = []
        
        for endpoint, result in api_results.items():
            endpoint_summary = f"Data from {endpoint}:\n"
            
            if "error" in result:
                endpoint_summary += f"  Error: {result['error']}\n"
            else:
                for table_name, table_data in result.items():
                    row_count = len(table_data)
                    endpoint_summary += f"  {table_name}: {row_count} records\n"
                    
                    if row_count > 0:
                        # Get column names from first row
                        columns = list(table_data[0].keys())
                        endpoint_summary += f"  Columns: {', '.join(columns)}\n"
                        
                        # For analysis, include more data
                        endpoint_summary += f"  Full data (first 20 rows):\n"
                        for i in range(min(20, row_count)):
                            endpoint_summary += f"    {json.dumps(table_data[i])}\n"
            
            summary.append(endpoint_summary)
        
        return "\n".join(summary)

    # Example usage