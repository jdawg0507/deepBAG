"""
Synthesizer for deepBAG.
Receives structured data fetched directly via Python code execution and synthesizes
concise, analytical responses in a single LLM turn.
Implements:
1. Standard 1-turn synthesis with conditional Chart.js injection (legacy path)
2. Mechanical Number-Locked Research Synthesis (v2 engine): ID-only generation [C1], [C2],
   post-generation replacement, and verifier check (fail-closed if unaligned).
Unified across Nebius Token Factory and Google Gemini via UnifiedLLMClient.
"""
import os
import json
import logging
from typing import Any, Dict, Generator, Optional, List

from core.telemetry import TelemetryManager
from core.llm_client import UnifiedLLMClient
from core.evidence import EvidenceTable
from core.verifier import SynthesisVerifier
from core.session import ScopeReceipt
from core.schemas import IntentContract

logger = logging.getLogger("deepbag.synthesizer")

BASE_ANALYST_PROMPT = """You are an elite NBA statistical analyst.
Answer the user's question directly, accurately, and concisely using the provided NBA dataset.
Highlight key statistics and trends clearly with markdown bullet points or tables where helpful.
Never hallucinate or extrapolate numbers not present in the provided dataset.
If the data indicates an error or no records found, explain what occurred clearly and politely."""

NUMBER_LOCK_RESEARCH_PROMPT = """You are the deepBAG Autonomous Statistical Research Synthesizer.
Answer the user's question directly, insightfully, and concisely based strictly on the registered Evidence Claims.

CRITICAL MECHANICAL NUMBER LOCK RULES:
1. You must NEVER compute, estimate, or write out bare numbers yourself.
2. Every numerical value (points, percentages, ratings, differentials) MUST be referenced ONLY using its exact claim ID in brackets, e.g. [C1], [C2], [C3].
3. Code replaces these claim IDs with exact values and mechanically audits every numeral in the text. An unaligned number causes immediate rejection.
4. For subjective or thesis questions, state the primary conclusion first, then present the strongest counter-evidence and sample limitations.
5. If an ACH matrix is provided, explain which hypothesis is least-refuted and why competing explanations fail.
6. MANDATORY VISUALIZATION FOR COMPARISONS: If presenting comparative player data, longitudinal projections, or multi-metric archetypes, ALWAYS embed a valid Chart.js (```chartjs```), Network Graph (```graph```), or SVG (```svg```) visualization comparing the entities on their key metrics."""

DYNAMIC_VISUALIZATION_INSTRUCTION = """
VISUALIZATION REQUIREMENT:
Provide an intuitive, engaging visual representation of the data for casual audiences alongside your narrative analysis.
Choose the BEST visual archetype based on the data's semantic topology (never force an unsuitable format):

Option 1: Statistical Distribution / Comparison (Chart.js)
If displaying time-series trends, comparative rankings, or multi-metric profiles, embed a valid JSON specification in a ```chartjs ``` block.
Supported types: 'bar', 'line', 'radar', 'doughnut', 'scatter', 'bubble'.
Example for 2D correlation (e.g. Volume vs Efficiency / Usage vs True Shooting):
```chartjs
{
  "type": "scatter",
  "title": "Volume vs Efficiency Correlation",
  "data": {
    "datasets": [{
      "label": "Players",
      "data": [
        { "x": 28.5, "y": 61.2, "label": "Player A" },
        { "x": 22.1, "y": 57.8, "label": "Player B" }
      ]
    }]
  },
  "options": {
    "scales": {
      "x": { "title": { "display": true, "text": "Volume / Usage %" } },
      "y": { "title": { "display": true, "text": "Efficiency / TS %" } }
    }
  }
}
```

Option 2: Relational / Node-Link Graph
If displaying connections, passing/assist networks, head-to-head match dynamics, or entity hierarchies, embed a valid JSON in a ```graph ``` block:
```graph
{
  "title": "Passing & Assist Distribution Network",
  "nodes": [
    { "id": "A", "label": "Primary Distributor", "value": 10.5, "group": "distributor", "color": "#38bdf8", "subtext": "10.5 APG" },
    { "id": "B", "label": "Key Finisher", "value": 7.2, "group": "finisher", "color": "#f97316", "subtext": "4.1 Assisted FG/g" }
  ],
  "links": [
    { "source": "A", "target": "B", "weight": 4.1, "label": "4.1 AST/g" }
  ]
}
```

Option 3: Generative Vector Graphic (SVG)
If displaying spatial coordinates (such as a half-court shot chart with plotted makes/misses), flowcharts, or custom diagrams, write clean, responsive SVG code in an ```svg ``` block:
```svg
<svg viewBox="0 0 500 320" xmlns="http://www.w3.org/2000/svg">
  <!-- Clean styled vector graphics with dark-mode aesthetic -->
</svg>
```

Always ensure the embedded JSON/SVG is strictly valid and accompanied by insightful analytical commentary."""

def should_inject_chart(query: str, data: Any) -> bool:
    q_lower = query.lower()
    chart_keywords = [
        "vs", "compare", "comparison", "trend", "chart", "graph", "plot", 
        "leaders", "top", "leaderboard", "last 5", "last 10", "past 5", "past 10"
    ]
    if any(k in q_lower for k in chart_keywords):
        return True

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
    def __init__(self, default_model: Optional[str] = None):
        self.llm_client = UnifiedLLMClient(model=default_model)
        self.verifier = SynthesisVerifier()
        self.default_model = self.llm_client.model
        self.fallback_models = ["nvidia/nemotron-3-super-120b-a12b", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

    def synthesize(
        self,
        query: str,
        data: Any,
        endpoint_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Legacy 1-turn synthesis path for simple Q&A lookups.
        """
        target_model = model_name or self.default_model
        inject_chart = should_inject_chart(query, data)
        
        system_instruction = BASE_ANALYST_PROMPT
        if inject_chart:
            system_instruction += "\n" + DYNAMIC_VISUALIZATION_INSTRUCTION
            yield {"type": "thought", "content": "Query/data qualifies for visual analytics; enabling dynamic visual grammar (charts, relational graphs, or vector graphics)."}

        trimmed_data = truncate_dataset_for_llm(data)
        has_upstream_error = isinstance(trimmed_data, dict) and "error" in trimmed_data

        if has_upstream_error:
            user_prompt = f"""User Question: {query}

Source Endpoint: {endpoint_name or 'NBA Historical/Live Data'}
Notice: Live stats.nba.com API returned an upstream timeout ({trimmed_data.get('error')}).
Please provide an authoritative, highly detailed statistical and historical analysis answering the user's question using verified NBA statistics (e.g., box score averages, advanced metrics like TS%, usage, win shares, narrative impact, and historical context). If comparing players or seasons, clearly contrast their numbers."""
        else:
            user_prompt = f"""User Question: {query}

Source Endpoint: {endpoint_name or 'NBA Historical/Live Data'}
Retrieved Data:
{json.dumps(trimmed_data, indent=2, default=str)}

Provide a direct, complete, and insightful answer to the user's question."""

        yield {"type": "thought", "content": f"Synthesizing response with {self.llm_client.provider} ({target_model})..."}

        try:
            final_text = self.llm_client.generate(
                user_prompt=user_prompt,
                system_prompt=system_instruction,
                temperature=0.2,
                model_name=target_model
            )
            yield {"type": "final_answer", "content": final_text}
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            yield {"type": "final_answer", "content": f"Failed to synthesize response: {e}"}

    def synthesize_research(
        self,
        query: str,
        evidence: EvidenceTable,
        intent: IntentContract,
        ach_matrix_md: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        v2 Mechanical Number-Locked Research Engine Synthesis:
        1. Emits prompt containing ONLY [C1], [C2] registered claims.
        2. Replaces IDs with verified values in code.
        3. Runs SynthesisVerifier. Fails closed if unaligned.
        4. Renders Scope Receipt, Narrative, ACH Matrix (if thesis), and Evidence Appendix.
        """
        target_model = model_name or self.default_model
        
        # Immediate Fail-Closed if no evidence claims exist
        if len(evidence.all_claims()) == 0:
            yield {
                "type": "thought",
                "content": "Zero verified claims registered in evidence ledger. Failing closed to prevent hallucination."
            }
            scope_receipt = ScopeReceipt.format_receipt(intent)
            fail_payload = (
                f"{scope_receipt}\n\n"
                f"### ⚠️ Data Unavailable (Fail-Closed Enforcement)\n"
                f"deepBAG was unable to retrieve verified data from the NBA API for this query "
                f"(upstream API network timeout or empty dataset returned).\n\n"
                f"**Integrity Guarantee**: Under deepBAG's strict mechanical number-lock architecture, "
                f"zero numbers are synthesized without a cryptographically verified claim ledger.\n\n"
                f"---\n"
                f"### Statistical Provenance Appendix\n"
                f"🛡️ **Number Lock: FAIL-CLOSED** (0 claims registered, 0 numbers generated)\n"
            )
            yield {"type": "final_answer", "content": fail_payload}
            return

        yield {"type": "thought", "content": "Executing PAL-style synthesis with mechanical number lock..."}
        
        claims_table_md = evidence.render_markdown_table()
        scope_receipt = ScopeReceipt.format_receipt(intent)
        
        ach_section = f"\n\nEvaluated ACH Matrix:\n{ach_matrix_md}" if ach_matrix_md else ""
        
        user_prompt = f"""User Question: {query}

Scope & Interpretation:
{scope_receipt}

Registered Claim Ledger (Use ONLY these claim IDs: e.g. [C1], [C2]):
{claims_table_md}
{ach_section}

Synthesize a clear, analytical answer. Reference statistics ONLY via claim IDs in brackets."""

        system_instruction = NUMBER_LOCK_RESEARCH_PROMPT + "\n\n" + DYNAMIC_VISUALIZATION_INSTRUCTION

        # First generation attempt
        raw_synthesis = self.llm_client.generate(
            user_prompt=user_prompt,
            system_prompt=system_instruction,
            temperature=0.2,
            model_name=target_model
        )
        
        # Replace claim IDs with verified values
        substituted_text = evidence.substitute_ids(raw_synthesis)
        
        # Post-check verification
        is_thesis = (intent.question_type == "thesis")
        passed, reason, unaligned = self.verifier.verify_payload(
            substituted_text,
            evidence,
            is_thesis=is_thesis
        )
        
        if not passed:
            yield {"type": "thought", "content": f"Verification failed ({reason}). Regenerating once with named failure..."}
            retry_prompt = user_prompt + f"\n\nERROR IN PREVIOUS DRAFT: {reason}. You must NOT invent any numbers. Use only [C1], [C2]."
            raw_synthesis = self.llm_client.generate(
                user_prompt=retry_prompt,
                system_prompt=system_instruction,
                temperature=0.1,
                model_name=target_model
            )
            substituted_text = evidence.substitute_ids(raw_synthesis)
            passed, reason, unaligned = self.verifier.verify_payload(
                substituted_text,
                evidence,
                is_thesis=is_thesis
            )

        if not passed:
            yield {
                "type": "verifier_status",
                "passed": False,
                "status": "UNALIGNED",
                "unaligned": unaligned,
            }
            fail_payload = (
                f"{scope_receipt}\n\n"
                f"### ⚠️ Synthesis Blocked (Fail-Closed Number-Lock Enforcement)\n"
                f"deepBAG detected unverified numerical tokens in the generated response that could not be aligned with the cryptographic evidence ledger.\n\n"
                f"**Verification Reason**: {reason}\n"
                f"**Unaligned Tokens**: {unaligned}\n\n"
                f"**Integrity Guarantee**: Under deepBAG's strict mechanical number-lock architecture, "
                f"responses with unverified numbers are blocked to eliminate numerical hallucination.\n\n"
                f"---\n"
                f"### Statistical Provenance Appendix\n"
                f"🛡️ **Number Lock: FAILED** (Verification failed: {reason})\n\n"
                f"{claims_table_md}\n"
            )
            yield {"type": "final_answer", "content": fail_payload}
            return

        verification_badge = "🛡️ **Number Lock: ALIGNED** (100% of numerals verified against evidence ledger)"
        
        yield {
            "type": "verifier_status",
            "passed": True,
            "status": "ALIGNED",
            "unaligned": [],
        }

        # Build final rendered markdown payload
        final_payload = (
            f"{scope_receipt}\n\n"
            f"{substituted_text}\n\n"
        )
        
        if ach_matrix_md:
            final_payload += f"### Competing Hypotheses (ACH Matrix)\n{ach_matrix_md}\n\n"
            
        final_payload += (
            f"---\n"
            f"### Statistical Provenance Appendix\n"
            f"{verification_badge}\n\n"
            f"{claims_table_md}\n"
        )

        yield {"type": "final_answer", "content": final_payload}
