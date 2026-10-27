"""
Generic Tool-Use Agent (MCP-Equivalent Baseline Runner).
Exposes raw nba_api endpoints as tools with raw parameter schemas.
- No catalogue
- No capability cards
- No deterministic CodeVerifier
- No ScopeReceipt
- No EntityResolver
- Free tool-calling loop (up to 8 iterations) using Nebius Nemotron 120B
- Final synthesis using Nebius Nemotron 120B (identical synthesis model as deepBAG)
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

from api.endpoint_utils import execute_api_call

load_dotenv()
logger = logging.getLogger("benchmark.mcp_baseline")

MCP_TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "call_nba_api_endpoint",
            "description": "Call any raw NBA stats API endpoint with raw parameters. Available endpoints include: playergamelog, playercareerstats, leaguedashplayerstats, commonplayerinfo, teamplayeronoffdetails, boxscoretraditionalv2, shotchartdetail, draftroundpickhistory, leaguegamefinder, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint_name": {
                        "type": "string",
                        "description": "The exact name of the nba_api endpoint to call (e.g. 'PlayerGameLog', 'LeagueDashPlayerStats', 'PlayerCareerStats')"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Key-value dictionary of raw parameters to pass to the endpoint (e.g. {'PlayerID': 1630169, 'Season': '2023-24', 'SeasonType': 'Regular Season'})"
                    }
                },
                "required": ["endpoint_name", "parameters"]
            }
        }
    }
]

MCP_SYSTEM_PROMPT = """You are an NBA statistics research assistant equipped with direct access to the NBA stats API.
You can query NBA endpoints to retrieve official data for players, teams, games, and historical seasons.
Call endpoints as needed to gather data. When you have collected sufficient data, provide a comprehensive, direct, and factual answer to the user's question.
If the question is based on a false premise (e.g. a player never played for a team, or was not active in that season), state the correction clearly."""

class MCPBaselineAgent:
    def __init__(self, model_name: str = "nvidia/nemotron-3-super-120b-a12b", max_iterations: int = 8):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.client = OpenAI(
            base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
            api_key=os.environ.get("NEBIUS_API_KEY")
        )

    def run_query(self, query: str, blueprint_id: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        messages = [
            {"role": "system", "content": MCP_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        
        tool_calls_executed = []
        tool_outputs_received = []
        llm_calls_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        final_answer = ""
        plan_recorded = {"endpoint_calls": []}
        
        for iteration in range(self.max_iterations):
            llm_calls_count += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=MCP_TOOL_DECLARATIONS,
                    tool_choice="auto",
                    temperature=0.1
                )
            except Exception as e:
                logger.error(f"Error calling LLM in MCP baseline: {e}")
                final_answer = f"Error during model generation: {e}"
                break
                
            choice = response.choices[0]
            msg = choice.message
            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens or 0
                total_completion_tokens += response.usage.completion_tokens or 0
                
            # Check if model made tool calls
            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                        
                    ep_name = args.get("endpoint_name", "")
                    params = args.get("parameters", {})
                    
                    tool_calls_executed.append({
                        "endpoint_name": ep_name,
                        "parameters": params
                    })
                    plan_recorded["endpoint_calls"].append({
                        "endpoint": ep_name,
                        "params": params
                    })
                    
                    # Execute API call via raw execute_api_call
                    try:
                        result = execute_api_call(ep_name, params)
                    except Exception as ex:
                        result = {"error": str(ex)}
                        
                    tool_outputs_received.append(result)
                    
                    # Truncate result if excessively long to avoid context blowup
                    result_str = json.dumps(result)
                    if len(result_str) > 8000:
                        result_str = result_str[:8000] + "... [truncated]"
                        
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": result_str
                    })
            else:
                # No more tool calls; model emitted final answer
                final_answer = msg.content or ""
                break
                
        # If loop ended without explicit content, prompt one last synthesis turn
        if not final_answer and messages:
            llm_calls_count += 1
            messages.append({"role": "user", "content": "Based on the tool outputs above, provide the final answer to the original question."})
            try:
                synth_resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2
                )
                final_answer = synth_resp.choices[0].message.content or ""
                if synth_resp.usage:
                    total_prompt_tokens += synth_resp.usage.prompt_tokens or 0
                    total_completion_tokens += synth_resp.usage.completion_tokens or 0
            except Exception as e:
                final_answer = f"Error during synthesis: {e}"
                
        latency_ms = (time.time() - start_time) * 1000.0
        
        return {
            "system": "generic_mcp_baseline",
            "blueprint_id": blueprint_id,
            "query": query,
            "plan": plan_recorded,
            "tool_calls": tool_calls_executed,
            "tool_outputs": tool_outputs_received,
            "final_answer": final_answer,
            "llm_calls_count": llm_calls_count,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "latency_ms": latency_ms
        }
