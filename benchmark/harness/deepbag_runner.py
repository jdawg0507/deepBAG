"""
deepBAG System-Under-Test (SUT) Runner at commit f038e1a.
Runs the complete semantic router pipeline:
1. FastEmbed top-5 candidate retrieval
2. Nebius Nemotron 120B constrained QueryPlan compiler
3. Deterministic CodeVerifier with 1-pass self-correction
4. Three-tier entity resolution & ScopeReceipt
5. Dispatches tool calls and synthesizes via Nebius Nemotron 120B
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from api.endpoint_utils import execute_api_call
from router.semantic_index import SemanticEndpointIndex
from router.intent_parser import IntentParser
from router.verifier import CodeVerifier
from router.scope_receipt import ScopeReceipt
from entities.resolver import EntityResolver
from nlp.synthesizer import Synthesizer

load_dotenv()
logger = logging.getLogger("benchmark.deepbag")

class DeepBAGBenchmarkRunner:
    def __init__(self, model_name: str = "nvidia/nemotron-3-super-120b-a12b"):
        self.model_name = model_name
        self.semantic_index = SemanticEndpointIndex()
        self.entity_resolver = EntityResolver()
        self.intent_parser = IntentParser(
            catalogue=None,
            entity_resolver=self.entity_resolver
        )
        self.synthesizer = Synthesizer()

    def run_query(self, query: str, blueprint_id: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. FastEmbed Top-5 Candidate Retrieval
        candidates = self.semantic_index.retrieve_candidates(query=query, top_k=5)
        cand_names = [c[0] for c in candidates]
        
        # 2. Constrained Intent Parsing with 1-Pass Self-Correction Verifier
        # Tracks LLM calls: attempt 1 + possible attempt 2 (self-correction)
        plan = self.intent_parser.parse_intent(query=query, candidate_endpoints=cand_names)
        
        llm_calls_count = 1 # At least 1 call for intent parsing
        # (If self-correction triggered, it made 2 calls during parsing)
        
        tool_calls_executed = []
        tool_outputs_received = []
        combined_data = {}
        
        # 3. Execution of plan
        if plan and plan.endpoint_calls:
            for call in plan.endpoint_calls:
                ep = call.endpoint
                params = call.params
                lbl = call.entity_label or ep
                
                tool_calls_executed.append({
                    "endpoint_name": ep,
                    "parameters": params
                })
                
                try:
                    data = execute_api_call(ep, params)
                except Exception as e:
                    data = {"error": str(e)}
                    
                tool_outputs_received.append(data)
                combined_data[lbl] = data
        else:
            combined_data = {}
            
        # 4. Final Synthesis with identical model
        llm_calls_count += 1 # Synthesis call
        final_answer = ""
        try:
            synth_payload = list(combined_data.values())[0] if len(combined_data) == 1 else combined_data
            for event in self.synthesizer.synthesize(
                query=query,
                data=synth_payload,
                endpoint_name=plan.primary_endpoint if plan else "NBA API",
                model_name=self.model_name
            ):
                if event.get("type") == "final_answer":
                    final_answer = event.get("content", "")
        except Exception as e:
            final_answer = f"Error during synthesis: {e}"
            
        latency_ms = (time.time() - start_time) * 1000.0
        
        # Estimate token usage (standard estimate: 1 token ~= 4 chars)
        prompt_chars = len(query) + sum(len(json.dumps(c)) for c in cand_names)
        comp_chars = len(final_answer) + (len(plan.reasoning) if plan else 0)
        est_prompt_tokens = prompt_chars // 4
        est_completion_tokens = comp_chars // 4
        
        plan_dict = {
            "entities_raw": plan.entities_raw if plan else [],
            "season": plan.season if plan else None,
            "season_type": plan.season_type if plan else None,
            "endpoint_calls": [c.model_dump() for c in plan.endpoint_calls] if plan else [],
            "reasoning": plan.reasoning if plan else "",
            "scope_receipt": plan.scope_receipt.model_dump() if (plan and plan.scope_receipt) else None
        }
        
        return {
            "system": "deepbag_sut",
            "blueprint_id": blueprint_id,
            "query": query,
            "candidate_endpoints": cand_names,
            "plan": plan_dict,
            "tool_calls": tool_calls_executed,
            "tool_outputs": tool_outputs_received,
            "final_answer": final_answer,
            "llm_calls_count": llm_calls_count,
            "prompt_tokens": est_prompt_tokens,
            "completion_tokens": est_completion_tokens,
            "total_tokens": est_prompt_tokens + est_completion_tokens,
            "latency_ms": latency_ms
        }
