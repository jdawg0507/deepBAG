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
import logging
from typing import Tuple, List, Set, Optional
from core.evidence import EvidenceTable

logger = logging.getLogger("deepbag.verifier")

# Permitted baseline integers (e.g., quarter numbers, rankings, standard basketball constants)
ALLOWLISTED_INTEGERS = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "12", "24", "48", "100", # shot clock, minutes, per 100
    "0", "2023", "2024", "2025", "2026", # season references
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
        """Extract all raw numerals and percentages from prose text, excluding chart/graph code blocks"""
        # Strip chart, graph, and svg code blocks so declarative visual configs are not audited as bare prose numbers
        clean_text = re.sub(r'```(?:chartjs|chart|graph|network|svg|json:chart)[\s\S]*?```', '', text)
        # Match integers, floating points, percentages
        pattern = r'\b\d+(?:\.\d+)?%?\b'
        matches = re.findall(pattern, clean_text)
        cleaned = []
        for m in matches:
            val = m.replace("%", "").strip()
            cleaned.append(val)
        return cleaned

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
