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
from core.cache import RedisQueryCache
from core.telemetry import TelemetryManager

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
        
        # Initialize Agentic & Hybrid components
        print("Initializing Hybrid Architecture components...")
        self.tool_retriever = EndpointToolRetriever(self.endpoint_library)
        self.tool_builder = EndpointToolBuilder(self.endpoint_library)
        self.agent = NBAAgent()
        self.cache = RedisQueryCache()
        from nlp.hybrid_router import HybridRouter
        from nlp.synthesizer import Synthesizer
        self.hybrid_router = HybridRouter(self.endpoint_library)
        self.synthesizer = Synthesizer()
    
    def process_query_agentic(self, query, model_name=None):
        """Process an NBA stats query using the hybrid direct-code router + 1-turn synthesis architecture"""
        print(f"\nProcessing query (Hybrid, model={model_name or 'default'}): {query}")
        
        # 1. Distributed Cache Lookup with Telemetry
        with TelemetryManager.start_span("deepbag.cache_lookup", {"query": query}) as cache_span:
            cached_response = self.cache.get(query)
            if cached_response:
                print("Found response in distributed cache!")
                TelemetryManager.record_cache_event(hit=True, query=query)
                yield {"type": "thought", "content": "Found exact match in distributed cache. Skipping API calls..."}
                yield {"type": "final_answer", "content": cached_response}
                return
            TelemetryManager.record_cache_event(hit=False, query=query)
            
        try:
            # 2. Local Route Resolution in Python code (0 LLM calls!)
            with TelemetryManager.start_span("deepbag.local_routing", {"query": query}) as route_span:
                plan = self.hybrid_router.route(query)
                yield {"type": "thought", "content": f"Resolved route deterministically in code -> {plan.tool} ({plan.endpoint}). Reason: {plan.reason}"}

            final_text = ""
            
            # 3. Direct Code Dispatch
            if plan.tool == "search_api_docs":
                yield {"type": "tool_call", "tool": "search_api_docs", "params": plan.params}
                from nlp.meta_tools import MetaToolExecutor
                executor = MetaToolExecutor(self.endpoint_library)
                data = executor.search_api_docs(plan.params.get("keywords", ""))
                yield {"type": "tool_result", "tool": "search_api_docs", "result": data}
                
                # 4. 1-Turn Synthesis
                for event in self.synthesizer.synthesize(query, data, endpoint_name="Documentation", model_name=model_name):
                    yield event
                    if event.get("type") == "final_answer":
                        final_text = event.get("content", "")

            elif plan.tool == "query_historical_database":
                yield {"type": "tool_call", "tool": "query_historical_database", "params": {"sql_query": plan.sql_query}}
                from nlp.sql_tool import HistoricalSQLTool
                sql_tool = HistoricalSQLTool()
                data = sql_tool.execute_sql(plan.sql_query or "")
                yield {"type": "tool_result", "tool": "query_historical_database", "result": data}
                
                # 4. 1-Turn Synthesis
                for event in self.synthesizer.synthesize(query, data, endpoint_name="Historical SQLite DB", model_name=model_name):
                    yield event
                    if event.get("type") == "final_answer":
                        final_text = event.get("content", "")

            elif plan.tool == "call_nba_api":
                # Check Tier 1 Canonical Cache
                with TelemetryManager.start_span("deepbag.canonical_cache", {"endpoint": plan.endpoint, "params": json.dumps(plan.params)}) as cache_span:
                    cached_data = self.cache.get_api_data(plan.endpoint, plan.params, season=plan.season)
                
                if cached_data is not None:
                    yield {"type": "thought", "content": f"Found canonical cache hit for {plan.endpoint}. Skipping upstream NBA API call."}
                    data = cached_data
                else:
                    yield {"type": "tool_call", "tool": "call_nba_api", "params": {"endpoint_name": plan.endpoint, "parameters": plan.params}}
                    with TelemetryManager.start_span("deepbag.api_dispatch", {"endpoint": plan.endpoint}) as dispatch_span:
                        data = execute_api_call(plan.endpoint, plan.params)
                    if not (isinstance(data, dict) and "error" in data):
                        self.cache.set_api_data(plan.endpoint, plan.params, data, season=plan.season)
                    yield {"type": "tool_result", "tool": "call_nba_api", "result": self.agent.summarize_result(data) if isinstance(data, dict) else data}
                
                # 4. 1-Turn Synthesis
                for event in self.synthesizer.synthesize(query, data, endpoint_name=plan.endpoint, model_name=model_name):
                    yield event
                    if event.get("type") == "final_answer":
                        final_text = event.get("content", "")

            else:
                # Agent Fallback
                from nlp.meta_tools import get_meta_tools_declarations
                from nlp.sql_tool import get_sql_tool_declaration
                tools = get_meta_tools_declarations()
                tools.append(get_sql_tool_declaration())
                for event in self.agent.run_stream(query, tools, self.endpoint_library, model_name=model_name):
                    yield event
                    if event.get("type") == "final_answer":
                        final_text = event.get("content", "")

            # Save synthesized response to Tier 2 Cache
            if final_text and not final_text.startswith("Error"):
                self.cache.set(query, final_text)

        except Exception as e:
            print(f"Error in hybrid processing: {str(e)}")
            import traceback
            traceback.print_exc()
            yield {"type": "final_answer", "content": f"Error: {str(e)}"}

            
    def _setup_model(self, api_key):
        """Initialize the Gemini model with the provided API key"""
        import os
        genai.configure(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.model = genai.GenerativeModel(model_name)



    
    

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