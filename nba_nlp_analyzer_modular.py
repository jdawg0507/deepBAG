"""
NBA NLP Analyzer (Modular) - Main module for processing natural language queries about NBA statistics.
"""
import json
import traceback
from typing import Optional, Dict, Any, List
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
        self.cache = RedisQueryCache()
        from nlp.hybrid_router import HybridRouter
        from nlp.synthesizer import Synthesizer
        from nlp.caesura_planner import CaesuraPlanner
        from core.nba_client import NBAClient
        from router.semantic_index import SemanticEndpointIndex
        from router.intent_parser import IntentParser
        from entities.resolver import EntityResolver

        self.hybrid_router = HybridRouter(self.endpoint_library)
        self.synthesizer = Synthesizer()
        self.caesura_planner = CaesuraPlanner()
        self.nba_client = NBAClient(cache=self.cache)
        self.entity_resolver = EntityResolver()
        self.semantic_index = SemanticEndpointIndex()
        self.semantic_intent_parser = IntentParser(
            catalogue=None,
            entity_resolver=self.entity_resolver
        )

    def is_research_query(self, query: str) -> bool:
        """
        Determines whether a query routes to the unified CAESURA Research Engine.
        In deepBAG v2, all analytical, comparative, thesis, and tracking queries
        are processed through the CAESURA Research Engine.
        """
        return True
    
    def _process_research_flow(self, query: str, model_name: Optional[str] = None):
        """Executes the Research Engine v2 CAESURA loop with mechanical number lock."""
        from core.evidence import EvidenceTable
        from core.theses import ACHMatrix
        from core.analytics import calculate_ts_percent, calculate_efg_percent, calculate_on_off_differential

        # 1. Intent JSON Contract Triage
        yield {"type": "thought", "content": "Extracting schema-validated Intent JSON contract..."}
        intent = self.caesura_planner.extract_intent(query)
        yield {"type": "intent", "content": intent.model_dump()}

        if intent.clarification_needed:
            yield {"type": "final_answer", "content": f"Clarification needed: {intent.clarification_needed}"}
            return

        # 2. CAESURA Logical Plan DAG
        plan = self.caesura_planner.build_logical_plan(intent)
        yield {"type": "thought", "content": f"Constructed CAESURA DAG with {len(plan.steps)} execution steps. Pre-validating schema..."}

        is_valid, validation_errors = self.caesura_planner.validate_plan_schema(plan)
        if not is_valid:
            yield {"type": "thought", "content": f"Plan schema validation failed: {validation_errors}. Backtracking..."}

        # 3. Step Execution with Ponder Verification Gates
        evidence_table = EvidenceTable()
        ach = None
        if intent.question_type == "thesis":
            ach = ACHMatrix(query)
            ach.add_hypotheses_from_context([
                {"label": "Lineup Depth & Rebounding", "description": "Offensive glass control and secondary scoring"},
                {"label": "Isolation Shotmaking", "description": "Late-clock individual shot generation"}
            ], include_null=True)

        for step in plan.steps:
            yield {"type": "tool_call", "tool": "caesura_step", "params": {"step_id": step.step_id, "endpoint": step.endpoint, "params": step.params}}
            res = self.nba_client.call(step.endpoint, step.params, use_cache=True, timeout=12, max_retries=2)
            yield {
                "type": "tool_result",
                "tool": "caesura_step",
                "result": {
                    "step_id": step.step_id,
                    "status": res.get("status"),
                    "cached": res.get("cached"),
                    "datasets": list(res.get("datasets", {}).keys()),
                    "latency_ms": res.get("latency_ms", 0)
                }
            }

            # 4. Deterministic Analytics Extraction & Evidence Claim Registration
            datasets = res.get("datasets", {})
            total_rows = sum(len(rows) for rows in datasets.values()) if datasets else 0

            # Dynamic DAG Traversal / Backtracking if endpoint timed out, errored, or returned 0 rows
            if res.get("status") != "success" or total_rows == 0:
                tried_endpoints = {step.endpoint}
                while True:
                    fallback_step = self.caesura_planner.get_fallback_step(step, plan, tried_endpoints)
                    if not fallback_step:
                        break
                    tried_endpoints.add(fallback_step.endpoint)
                    yield {
                        "type": "thought",
                        "content": f"Primary endpoint `{step.endpoint}` returned no data or timed out. Dynamically traversing CAESURA DAG to fallback candidate `{fallback_step.endpoint}`..."
                    }
                    yield {
                        "type": "tool_call",
                        "tool": "caesura_step",
                        "params": {"step_id": fallback_step.step_id, "endpoint": fallback_step.endpoint, "params": fallback_step.params}
                    }
                    fb_res = self.nba_client.call(fallback_step.endpoint, fallback_step.params, use_cache=True, timeout=12, max_retries=2)
                    fb_datasets = fb_res.get("datasets", {})
                    fb_rows = sum(len(rows) for rows in fb_datasets.values()) if fb_datasets else 0
                    yield {
                        "type": "tool_result",
                        "tool": "caesura_step",
                        "result": {
                            "step_id": fallback_step.step_id,
                            "status": fb_res.get("status"),
                            "cached": fb_res.get("cached"),
                            "datasets": list(fb_datasets.keys()),
                            "latency_ms": fb_res.get("latency_ms", 0)
                        }
                    }
                    if fb_res.get("status") == "success" and fb_rows > 0:
                        res = fb_res
                        datasets = fb_datasets
                        step = fallback_step
                        total_rows = fb_rows
                        intent.assumptions.append(
                            f"Primary endpoint was unavailable; dynamically backtracked to {fallback_step.endpoint}."
                        )
                        break

            if datasets:
                step_entity = step.step_id.split("_", 1)[1] if "_" in step.step_id else None
                evidence_table.register_dataset_claims(
                    endpoint=step.endpoint,
                    params=step.params,
                    datasets=datasets,
                    entity_hint=step_entity,
                    ach=ach
                )

        ach_md = ach.render_markdown_matrix() if ach else None

        # 5. Mechanical Number-Locked Synthesis (Fails closed if 0 claims)
        final_text = ""
        for event in self.synthesizer.synthesize_research(query, evidence_table, intent, ach_matrix_md=ach_md, model_name=model_name):
            yield event
            if event.get("type") == "final_answer":
                final_text = event.get("content", "")

        # Only cache responses that are successful and not transient errors/fail-closed states
        is_unsuitable_for_cache = (
            not final_text or
            final_text.startswith("Error") or
            "Data Unavailable" in final_text or
            "Synthesis Blocked" in final_text or
            "Number Lock: FAILED" in final_text or
            "Number Lock: FAIL-CLOSED" in final_text or
            len(evidence_table.all_claims()) == 0
        )
        if not is_unsuitable_for_cache:
            self.cache.set(query, final_text)

    def process_query_agentic(self, query, model_name=None):
        """Process an NBA stats query using the unified Research Engine v2 (CAESURA DAG + PAL Number Lock)"""
        print(f"\nProcessing query (Unified Research Engine, model={model_name or 'default'}): {query}")
        
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

        # 2. Check for direct documentation or fast premise correction
        legacy_route = self.hybrid_router.route(query)
        if legacy_route.tool == "premise_correction" or legacy_route.direct_response:
            yield {"type": "thought", "content": f"Premise check triggered: {legacy_route.reason}"}
            yield {"type": "final_answer", "content": legacy_route.direct_response}
            return

        if legacy_route.tool == "search_api_docs":
            yield {"type": "tool_call", "tool": "search_api_docs", "params": legacy_route.params}
            from nlp.meta_tools import MetaToolExecutor
            executor = MetaToolExecutor(self.endpoint_library)
            data = executor.search_api_docs(legacy_route.params.get("keywords", ""))
            yield {"type": "tool_result", "tool": "search_api_docs", "result": data}
            for event in self.synthesizer.synthesize(query, data, endpoint_name="Documentation", model_name=model_name):
                yield event
            return

        # 3. Always dispatch to unified deepBAG Research Engine v2 (CAESURA loop + PAL Number Lock)
        yield {"type": "thought", "content": "Routing to unified deepBAG Research Engine (CAESURA loop + PAL Number Lock)..."}
        try:
            for event in self._process_research_flow(query, model_name=model_name):
                yield event
        except Exception as e:
            print(f"Error in research processing: {str(e)}")
            import traceback
            traceback.print_exc()
            yield {"type": "final_answer", "content": f"Error: {str(e)}"}

            
    def _setup_model(self, api_key):
        """Initialize the Gemini model with the provided API key or fallback"""
        import os
        try:
            if api_key and not api_key.startswith("v1."):
                genai.configure(api_key=api_key)
                model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
                self.model = genai.GenerativeModel(model_name)
            else:
                self.model = None
        except Exception as e:
            print(f"Warning: Gemini model setup fallback: {e}")
            self.model = None



    
    

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
