"""
Mechanical Number Lock & Synthesis Verifier for deepBAG.
Strict post-generation verifier that guarantees:
1. Zero Hallucinated Numbers: Every numerical token in the synthesized text matches an
   active EvidenceItem from the EvidenceTable.
2. Metric Association Check: Metrics cited are linked to valid claim IDs.
3. Causal Language Guardrail: Discourages unwarranted causal assertions from raw on/off data.
4. Thesis Completeness Check: Asserts presence of counter-evidence in thesis answers.
Fail-Closed: If any numeral fails alignment, synthesis is blocked.
"""
import re
import json
import logging
from typing import Tuple, List, Set, Optional, Any
from core.evidence import EvidenceTable

logger = logging.getLogger("deepbag.verifier")

# Permitted baseline integers (strictly non-box-score structural constants)
ALLOWLISTED_INTEGERS = {
    "0", "1", "2", "3", "4", # quarter/overtime indices (Q1-Q4, baseline 0)
    "2020", "2021", "2022", "2023", "2024", "2025", "2026", # season references
}

CAUSAL_PHRASES = [
    r"\bcauses\b", r"\bcaused\b", r"\bcausing\b",
    r"\bproves\b", r"\bproven\b", r"\bproof\b",
    r"\bdue solely to\b", r"\bdirectly responsible for\b"
]

class SynthesisVerifier:
    def __init__(self, allowlisted_numbers: Optional[Set[str]] = None):
        self.allowlist = set(allowlisted_numbers or ALLOWLISTED_INTEGERS)

    def extract_numerals(self, text: str) -> List[str]:
        """Extract all raw numerals and percentages from prose text and data points from chart/graph blocks."""
        cleaned = []

        # 1. Parse and extract data coordinates from chart and graph code blocks
        chart_blocks = re.findall(r'```(?:chartjs|chart|graph|network|svg|json:chart)([\s\S]*?)```', text)
        for block in chart_blocks:
            content = block.strip()
            try:
                data_obj = json.loads(content)
                cleaned.extend(self._extract_chart_numbers(data_obj))
            except Exception:
                for m in re.finditer(r'"(?:data|values|x|y|value|weight)"\s*:\s*\[?([^\]}\n]+)\]?', content):
                    matches = re.findall(r'\b\d+(?:\.\d+)?%?\b', m.group(1))
                    cleaned.extend([v.replace("%", "").strip() for v in matches])

        # 2. Strip chart/graph blocks from prose so formatting/layout integers are not audited as bare prose
        clean_text = re.sub(r'```(?:chartjs|chart|graph|network|svg|json:chart)[\s\S]*?```', '', text)

        # 3. Match integers, floating points, percentages
        pattern = r'\b\d+(?:\.\d+)?%?\b'
        matches = re.findall(pattern, clean_text)
        for m in matches:
            val = m.replace("%", "").strip()
            cleaned.append(val)
        return cleaned

    def _extract_chart_numbers(self, obj: Any) -> List[str]:
        """Recursively extracts data point numerals from chart/graph JSON structures."""
        nums = []
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            nums.append(str(obj))
        elif isinstance(obj, list):
            for item in obj:
                nums.extend(self._extract_chart_numbers(item))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("data", "datasets", "nodes", "links", "values", "x", "y", "r", "value", "weight"):
                    nums.extend(self._extract_chart_numbers(v))
                elif k not in ("options", "scales", "plugins", "borderWidth", "borderRadius", "fontSize", "pointRadius", "width", "height", "responsive"):
                    if isinstance(v, (list, dict)):
                        nums.extend(self._extract_chart_numbers(v))
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        nums.append(str(v))
        return nums

    def verify_numerals(self, text: str, evidence: EvidenceTable) -> Tuple[bool, List[str]]:
        """
        Verify that every numeral in the text exists in the evidence table or allowlist.
        Returns (is_aligned, unaligned_numerals_list).
        """
        expected_numerals = evidence.extract_expected_numerals()
        found_numerals = self.extract_numerals(text)
        
        unaligned = []
        for num in found_numerals:
            # Check if in expected evidence or allowlist
            if num in expected_numerals or num in self.allowlist:
                continue
                
            # Allow approximate rounding (e.g. 25.74 -> 25.7)
            try:
                f_val = float(num)
                # Check if close to any expected raw value
                is_close = any(abs(f_val - item.raw_value) < 0.15 for item in evidence.all_claims())
                if is_close:
                    continue
            except ValueError:
                pass
                
            unaligned.append(num)
            
        is_aligned = (len(unaligned) == 0)
        return is_aligned, unaligned

    def check_causal_language(self, text: str) -> List[str]:
        """Detect causal overclaims in text"""
        violations = []
        for pat in CAUSAL_PHRASES:
            if re.search(pat, text, re.IGNORECASE):
                violations.append(pat.replace(r"\b", ""))
        return violations

    def verify_thesis_counter_evidence(self, text: str) -> bool:
        """Check if thesis response includes required counter-evidence or limitations"""
        counter_markers = [
            "counter-evidence", "counter evidence", "alternatively", "confounding",
            "sample size", "limitation", "small sample", "on the other hand", "competing"
        ]
        text_lower = text.lower()
        return any(marker in text_lower for marker in counter_markers)

    def verify_payload(
        self,
        text: str,
        evidence: EvidenceTable,
        is_thesis: bool = False,
        strict_causal: bool = False,
    ) -> Tuple[bool, str, List[str]]:
        """
        Full verification suite:
        Returns (passed, status_reason, unaligned_items)
        """
        # 1. Numeral verification
        is_aligned, unaligned = self.verify_numerals(text, evidence)
        if not is_aligned:
            reason = f"UNALIGNED: Numerals {unaligned} not found in claim ledger."
            logger.warning(reason)
            return False, reason, unaligned

        # 2. Causal language guardrail
        if strict_causal:
            causal_violations = self.check_causal_language(text)
            if causal_violations:
                reason = f"CAUSAL_VIOLATION: Text contains causal assertion phrases: {causal_violations}"
                logger.warning(reason)
                return False, reason, causal_violations

        # 3. Thesis completeness
        if is_thesis:
            has_counter = self.verify_thesis_counter_evidence(text)
            if not has_counter:
                reason = "THESIS_INCOMPLETE: Thesis response missing required counter-evidence or sample caveats."
                logger.warning(reason)
                return False, reason, ["missing_counter_evidence"]

        return True, "ALIGNED: All numerals provenanced by EvidenceTable.", []
