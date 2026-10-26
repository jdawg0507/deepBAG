import json
import google.generativeai as genai
from api.endpoint_utils import execute_api_call
from utils.data_utils import clean_json_response

class NBAAgent:
    def __init__(self, model_name=None):
        import os
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key, transport="rest")
        self.fallback_models = [
            "gemini-3.1-flash-lite-preview",
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-flash-latest",
            "gemini-pro-latest",
        ]

    def summarize_result(self, result_dict, max_rows=3):
        """Summarize or truncate the DataFrame dict to prevent blowing up the context window"""
        if "error" in result_dict:
            return result_dict
            
        summary = {}
        for table_name, data_list in result_dict.items():
            if not isinstance(data_list, list):
                summary[table_name] = data_list
                continue
                
            total_rows = len(data_list)
            if total_rows == 0:
                summary[table_name] = {"rows": 0, "data": []}
            else:
                # Truncate to first 'max_rows' to save tokens
                truncated = data_list[:max_rows]
                columns = list(data_list[0].keys()) if len(data_list) > 0 else []
                summary[table_name] = {
                    "total_rows_available": total_rows,
                    "columns": columns,
                    "sample_data": truncated,
                    "note": f"Data truncated to {max_rows} rows. If you need a specific row, ask to filter in the API call."
                }
        return summary

    def run_stream(self, query, tools, endpoint_library, model_name=None):
        """Runs the agentic loop as a generator, yielding events"""
        yield {"type": "thought", "content": f"Initializing agent with {len(tools)} meta-tools..."}
        
        from datetime import datetime
        now = datetime.now()
        current_year = now.year if now.month >= 9 else now.year - 1
        current_season = f"{current_year}-{str(current_year + 1)[-2:]}"
        
        system_instruction = f"""
        You are an expert NBA data analyst. You have access to TWO dynamic Meta-Tools to fetch live NBA API data, AND a local SQLite database via 'query_historical_database'.
        
        CRITICAL RULES:
        1. DYNAMIC API FETCHING: If you need live or current season data, you MUST use 'search_api_docs' to find the exact endpoint name and required parameters. Once you find it, use 'call_nba_api' to execute it.
        2. DATA-BACKED OPINIONS: You MUST NOT write theoretical essays without data.
        3. ADVANCED STATS: If an endpoint (like LeagueDashPlayerStats) supports it, set `MeasureType` to 'Advanced' or 'Defense'.
        4. PARAMETER FIXING: The current active NBA season is {current_season}. If an endpoint requires a 'Season' parameter, it usually expects '{current_season}'.
        5. ERROR RECOVERY: If a tool returns 0 rows, it may be a network timeout (e.g. ReadTimeout). You can try again or search for a different endpoint.
        6. HISTORICAL QUERIES: If the user asks for historical data (anything before {current_season}), you MUST use 'query_historical_database' against the local Kaggle dataset.
        7. INTERACTIVE CHARTS: When answering queries that involve comparisons between players/teams, last N game scoring/rebounding trends, or multi-stat profiles, include an interactive Chart.js visualization in your markdown using a ```chart code block with valid JSON:
        ```chart
        {{
          "type": "bar" | "line" | "radar",
          "title": "Descriptive Chart Title",
          "data": {{
            "labels": ["Label 1", "Label 2"],
            "datasets": [
              {{ "label": "Series Name", "data": [10, 20] }}
            ]
          }}
        }}
        ```
        Omit chart blocks for simple single-fact lookups.
        """

        primary_model = model_name or self.model_name
        models_to_try = [primary_model] + [m for m in self.fallback_models if m != primary_model]
        from nlp.meta_tools import MetaToolExecutor
        meta_executor = MetaToolExecutor(endpoint_library)
        
        # Loop through fallback models at the top level
        for model_idx, current_model in enumerate(models_to_try):
            try:
                yield {"type": "thought", "content": f"Analyzing query using {current_model}..."}

                
                model = genai.GenerativeModel(
                    model_name=current_model,
                    tools=tools,
                    system_instruction=system_instruction
                )
                chat = model.start_chat()
                response = chat.send_message(query)
                
                # Allow up to 8 iterations to accommodate doc searching + execution
                for _ in range(8):
                    function_call_part = None
                    text_thoughts = []
                    if hasattr(response, 'parts'):
                        for part in response.parts:
                            if hasattr(part, "text") and part.text:
                                text_thoughts.append(part.text)
                            if hasattr(part, "function_call") and part.function_call:
                                function_call_part = part.function_call

                    if function_call_part:
                        for thought in text_thoughts:
                            yield {"type": "thought", "content": thought}
                            
                        fc = function_call_part
                        function_name = fc.name
                        
                        import json
                        args = {}
                        for key in fc.args:
                            args[key] = fc.args[key]
                            
                        yield {
                            "type": "tool_call", 
                            "content": f"Calling Tool: {function_name}",
                            "details": json.dumps(args)
                        }
                        
                        # Execute the actual API call
                        from core.telemetry import TelemetryManager
                        with TelemetryManager.start_span("deepbag.tool_execution", {"tool.name": function_name}) as tool_span:
                            try:
                                if function_name == "query_historical_database":
                                    from nlp.sql_tool import HistoricalSQLTool
                                    sql_tool = HistoricalSQLTool()
                                    raw_result = sql_tool.execute_sql(args.get("sql_query", ""))
                                    summary = raw_result
                                elif function_name == "search_api_docs":
                                    summary = meta_executor.search_api_docs(args.get("keywords", ""))
                                elif function_name == "call_nba_api":
                                    from core.schemas import ToolCallInvocation
                                    invocation = ToolCallInvocation(
                                        endpoint_name=args.get("endpoint_name", ""),
                                        parameters=args.get("parameters", {})
                                    )
                                    raw_result = execute_api_call(invocation.endpoint_name, invocation.parameters)
                                    summary = self.summarize_result(raw_result)
                                else:
                                    summary = {"error": f"Unknown function {function_name}"}

                                row_count = 0
                                if isinstance(summary, dict) and "error" not in summary:
                                    if function_name == "search_api_docs" and "matching_endpoints" in summary:
                                        row_count = len(summary["matching_endpoints"])
                                    else:
                                        for k, v in summary.items():
                                            if isinstance(v, dict) and "total_rows_available" in v:
                                                row_count = max(row_count, v["total_rows_available"])
                                            
                                yield {
                                    "type": "tool_result", 
                                    "content": f"Received {row_count} rows from API."
                                }
                                
                                # Explicitly build Protobuf Struct to avoid whichOneof dict inference errors
                                from google.protobuf.struct_pb2 import Struct
                                from google.protobuf import json_format
                                
                                clean_summary = json.loads(json.dumps(summary))
                                
                                s = Struct()
                                try:
                                    json_format.ParseDict({"result": clean_summary}, s)
                                except Exception as parse_e:
                                    yield {"type": "thought", "content": f"Protobuf Parse Error: {parse_e}"}
                                    s = Struct()
                                    json_format.ParseDict({"error": str(parse_e)}, s)

                                part = genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=function_name,
                                        response=s
                                    )
                                )
                                
                                try:
                                    response = chat.send_message(part)
                                except Exception as send_err:
                                    # Fallback if SDK or model requires thought_signature on function_response
                                    if "thought_signature" in str(send_err).lower() or "400" in str(send_err):
                                        if len(chat.history) > 2:
                                            chat.history.pop()
                                        fallback_prompt = (
                                            f"Tool '{function_name}' returned the following result:\n"
                                            f"{json.dumps(clean_summary)}\n\n"
                                            "Based on this data, please provide your final data-backed answer and analysis for the user."
                                        )
                                        response = chat.send_message(fallback_prompt)
                                    else:
                                        raise send_err
                                
                            except Exception as e:
                                yield {"type": "thought", "content": f"Tool Execution Error: {e}"}
                                from google.protobuf.struct_pb2 import Struct
                                from google.protobuf import json_format
                                s = Struct()
                                json_format.ParseDict({"error": str(e)}, s)
                                
                                error_part = genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=function_name,
                                        response=s
                                    )
                                )
                                
                                try:
                                    response = chat.send_message(error_part)
                                except Exception as err_send_err:
                                    if "thought_signature" in str(err_send_err).lower() or "400" in str(err_send_err):
                                        if len(chat.history) > 2:
                                            chat.history.pop()
                                        fallback_prompt = f"Tool '{function_name}' returned an error: {e}. Please answer the user with the best available data."
                                        response = chat.send_message(fallback_prompt)
                                    else:
                                        raise err_send_err

                    else:
                        # Model returned text, we are done
                        try:
                            text_content = ""
                            if hasattr(response, "parts") and response.parts:
                                for part in response.parts:
                                    if hasattr(part, "text") and part.text:
                                        text_content += part.text
                            if not text_content:
                                text_content = "Empty response returned by the model. This might be due to safety filters or lack of data."
                            yield {"type": "final_answer", "content": text_content}
                        except Exception as parse_e:
                            yield {"type": "final_answer", "content": f"Failed to parse text: {parse_e}"}
                        return
                        
                yield {"type": "final_answer", "content": "Agent reached maximum tool call iterations without a final answer."}
                return # Successful execution (even if it hit max iterations)
                
            except Exception as e:
                import google.api_core.exceptions
                err_str = str(e).lower()
                is_recoverable = (
                    isinstance(e, (
                        google.api_core.exceptions.ResourceExhausted,
                        google.api_core.exceptions.ServiceUnavailable,
                        google.api_core.exceptions.NotFound
                    ))
                    or any(k in err_str for k in [
                        "429", "503", "404", "quota", "exhausted",
                        "high demand", "unavailable", "temporarily", "rate limit", "no longer available"
                    ])
                )
                if is_recoverable and model_idx < len(models_to_try) - 1:
                    next_model = models_to_try[model_idx + 1]
                    short_err = str(e).splitlines()[0][:90]
                    yield {
                        "type": "thought",
                        "content": f"⚠️ {current_model} unavailable ({short_err}...). Auto-switching to fallback model: {next_model}..."
                    }
                    continue # Try next model in the top-level loop
                
                # If we get here, it's either not a recoverable error, or we ran out of fallback models
                import traceback
                traceback.print_exc()
                yield {"type": "final_answer", "content": f"Error processing query via agent: {str(e)}"}
                return
