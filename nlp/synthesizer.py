"""
Synthesizer for deepBAG.
Receives structured data fetched directly via Python code execution and synthesizes
concise, analytical responses in a single LLM turn.
Dynamically injects Chart.js specifications ONLY when the query or data warrants a chart.
"""
import os
import json
import logging
from typing import Any, Dict, Generator, Optional
import google.generativeai as genai
from core.telemetry import TelemetryManager

logger = logging.getLogger("deepbag.synthesizer")

BASE_ANALYST_PROMPT = """You are an elite NBA statistical analyst.
Answer the user's question directly, accurately, and concisely using the provided NBA dataset.
Highlight key statistics and trends clearly with markdown bullet points or tables where helpful.
Never hallucinate or extrapolate numbers not present in the provided dataset.
If the data indicates an error or no records found, explain what occurred clearly and politely."""

CHARTJS_INSTRUCTION = """
CHART GENERATION REQUIREMENT:
Since this query involves comparisons, trends, or leaderboard rankings, include an embedded Chart.js visualization.
Embed a valid JSON specification inside a ```chartjs ``` code block formatted as:
```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["Player A", "Player B"],
    "datasets": [{
      "label": "Points Per Game",
      "data": [28.5, 26.2],
      "backgroundColor": ["rgba(59, 130, 246, 0.8)", "rgba(16, 185, 129, 0.8)"]
    }]
  },
  "options": {
    "responsive": true,
    "plugins": {
      "legend": { "position": "top" },
      "title": { "display": true, "text": "Scoring Comparison" }
    }
  }
}
```
Ensure the JSON is strictly valid. Always accompany the chart with analytical narrative text."""

def should_inject_chart(query: str, data: Any) -> bool:
    """
    Dual-trigger for Chart.js instruction injection:
    1. Query contains comparison, trend, or chart keywords.
    2. Data contains >= 3 comparable numeric records.
    """
    q_lower = query.lower()
    chart_keywords = [
        "vs", "compare", "comparison", "trend", "chart", "graph", "plot", 
        "leaders", "top", "leaderboard", "last 5", "last 10", "past 5", "past 10"
    ]
    if any(k in q_lower for k in chart_keywords):
        return True

    # Check data row count
    if isinstance(data, list) and len(data) >= 3:
        return True
    if isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list) and len(val) >= 3:
                return True
            if isinstance(val, dict) and "sample_data" in val:
                if len(val["sample_data"]) >= 3:
                    return True
    return False

def truncate_dataset_for_llm(data: Any, max_rows: int = 15) -> Any:
    """Truncates large datasets to prevent overflowing context while retaining critical stats."""
    if not data:
        return data
    if isinstance(data, list):
        return data[:max_rows]
    if isinstance(data, dict):
        truncated = {}
        for k, v in data.items():
            if isinstance(v, list):
                truncated[k] = v[:max_rows]
            else:
                truncated[k] = v
        return truncated
    return data

class Synthesizer:
    """
    Performs 1-turn synthesis from structured API/SQL data to user answers.
    """
    def __init__(self, default_model: Optional[str] = None):
        from dotenv import load_dotenv
        load_dotenv()
        self.default_model = default_model or os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key, transport="rest")
        self.fallback_models = [
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
            "gemini-flash-latest",
        ]

    def synthesize(
        self,
        query: str,
        data: Any,
        endpoint_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Synthesize the final answer using a single LLM call.
        """
        chosen_model = model_name or self.default_model
        inject_chart = should_inject_chart(query, data)
        
        system_instruction = BASE_ANALYST_PROMPT
        if inject_chart:
            system_instruction += "\n" + CHARTJS_INSTRUCTION
            yield {"type": "thought", "content": "Query/data qualifies for visual charting; injecting Chart.js spec."}

        trimmed_data = truncate_dataset_for_llm(data)
        user_prompt = f"""User Question: {query}

Source Endpoint: {endpoint_name or 'NBA Historical/Live Data'}
Retrieved Data:
{json.dumps(trimmed_data, indent=2, default=str)}

Provide a direct, complete, and insightful answer to the user's question."""

        yield {"type": "thought", "content": f"Synthesizing final analytical response with {chosen_model}..."}

        models_to_try = [chosen_model] + [m for m in self.fallback_models if m != chosen_model]
        last_error = None
        
        for model_candidate in models_to_try:
            try:
                with TelemetryManager.start_span("deepbag.llm_synthesis", {"model": model_candidate}) as span:
                    model = genai.GenerativeModel(
                        model_candidate,
                        system_instruction=system_instruction,
                    )
                    response = model.generate_content(
                        user_prompt,
                        generation_config={"temperature": 0.2},
                    )
                    final_text = response.text if response else "No response generated."
                    TelemetryManager.record_llm_generation(
                        name="synthesizer",
                        model=model_candidate,
                        prompt=user_prompt,
                        completion=final_text,
                    )
                    yield {"type": "final_answer", "content": final_text}
                    return
            except Exception as e:
                logger.warning(f"Synthesis failed with {model_candidate}: {e}")
                last_error = e
                continue

        error_message = f"Failed to synthesize response across models: {last_error}"
        yield {"type": "final_answer", "content": error_message}
