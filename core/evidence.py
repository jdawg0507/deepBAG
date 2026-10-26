"""
Evidence Table & Claim Ledger for deepBAG.
Implements the PAL-style deterministic number-lock contract:
Every numerical value cited in downstream analysis originates from a code-computed
EvidenceItem with a unique claim ID ([C1], [C2], ...), provenanced by endpoint,
parameter hash, season, sample size, and formula version.
"""
import re
import hashlib
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field

class EvidenceItem(BaseModel):
    claim_id: str = Field(..., description="Unique claim identifier, e.g. C1, C2")
    metric_name: str = Field(..., description="Name of the statistic or metric, e.g. TS%, PPG, Net_Rating")
    value: str = Field(..., description="Formatted string representation of the value, e.g. '25.7', '58.4%'")
    raw_value: float = Field(..., description="Raw computed numerical value")
    endpoint: str = Field(..., description="NBA API endpoint or data source origin")
    params_hash: str = Field(..., description="MD5 hash of sorted input parameters")
    formula_version: str = Field(default="raw", description="Version identifier of calculation formula")
    sample_size: Optional[str] = Field(default=None, description="Sample size disclosure, e.g. 71 GP, 2504 MIN")
    description: Optional[str] = Field(default=None, description="Plain language explanation of metric")

class EvidenceTable:
    """
    Collection of registered claims and provenanced statistics for a query session.
    """
    def __init__(self):
        self._claims: Dict[str, EvidenceItem] = {}
        self._counter: int = 1

    def add_claim(
        self,
        metric_name: str,
        raw_value: float,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        formatted_value: Optional[str] = None,
        formula_version: str = "raw",
        sample_size: Optional[str] = None,
        description: Optional[str] = None,
    ) -> EvidenceItem:
        claim_id = f"C{self._counter}"
        self._counter += 1
        
        # Calculate canonical parameter hash
        sorted_params = json_canonical_sort(params) if params else ""
        p_hash = hashlib.md5(sorted_params.encode('utf-8')).hexdigest()[:8]
        
        if formatted_value is None:
            if isinstance(raw_value, float):
                formatted_value = f"{raw_value:.1f}" if abs(raw_value) >= 1 else f"{raw_value:.3f}"
            else:
                formatted_value = str(raw_value)
                
        item = EvidenceItem(
            claim_id=claim_id,
            metric_name=metric_name,
            value=formatted_value,
            raw_value=float(raw_value),
            endpoint=endpoint,
            params_hash=p_hash,
            formula_version=formula_version,
            sample_size=sample_size,
            description=description,
        )
        self._claims[claim_id] = item
        return item

    def get_claim(self, claim_id: str) -> Optional[EvidenceItem]:
        clean_id = claim_id.strip("[]")
        return self._claims.get(clean_id)

    def all_claims(self) -> List[EvidenceItem]:
        return list(self._claims.values())

    def render_markdown_table(self) -> str:
        """Render markdown table for inclusion in LLM prompt or answer appendix"""
        if not self._claims:
            return "_No evidence claims recorded._"
            
        lines = [
            "| ID | Metric | Value | Sample | Source Endpoint | Formula Version |",
            "|:---|:---|:---|:---|:---|:---|",
        ]
        for item in self._claims.values():
            sample = item.sample_size or "N/A"
            lines.append(
                f"| `[{item.claim_id}]` | **{item.metric_name}** | {item.value} | {sample} | `{item.endpoint}` | `{item.formula_version}` |"
            )
        return "\n".join(lines)

    def substitute_ids(self, text: str) -> str:
        """Replace all [C1], [C2] tags in text with their formatted numerical values"""
        def repl(match):
            cid = match.group(1)
            item = self._claims.get(cid)
            if item:
                return item.value
            return match.group(0) # Keep verbatim if not found
            
        return re.sub(r'\[(C\d+)\]', repl, text)

    def extract_expected_numerals(self) -> Set[str]:
        """Extract all valid numeral representations from the registered claims"""
        numerals = set()
        for item in self._claims.values():
            # Add formatted value
            val_clean = item.value.replace("%", "").strip()
            numerals.add(val_clean)
            # Add raw value formatted with various precision
            numerals.add(str(item.raw_value))
            numerals.add(f"{item.raw_value:.1f}")
            numerals.add(f"{item.raw_value:.2f}")
            numerals.add(f"{item.raw_value:.3f}")
            numerals.add(str(int(round(item.raw_value))))
        return numerals

def json_canonical_sort(obj: Any) -> str:
    import json
    return json.dumps(obj, sort_keys=True)
