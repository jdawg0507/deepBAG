"""
Unit tests for Mechanical Number Lock Synthesis Verifier (Milestone 2).
"""
import pytest
from core.evidence import EvidenceTable
from core.verifier import SynthesisVerifier

def test_verifier_aligned_numerals():
    table = EvidenceTable()
    table.add_claim("PTS", 32.0, "PlayerGameLog", formatted_value="32")
    table.add_claim("AST", 8.0, "PlayerGameLog", formatted_value="8")
    
    verifier = SynthesisVerifier()
    
    # 32 and 8 are in table; 1 and 4 are allowlisted integers
    text = "Curry scored 32 points with 8 assists in quarter 1 and 4."
    passed, reason, unaligned = verifier.verify_payload(text, table)
    
    assert passed is True
    assert len(unaligned) == 0

def test_verifier_blocks_unaligned_hallucinated_number():
    table = EvidenceTable()
    table.add_claim("PTS", 28.0, "PlayerGameLog", formatted_value="28")
    
    verifier = SynthesisVerifier()
    
    # 54 is not in table and not in allowlist
    hallucinated_text = "Curry scored 28 points on 54.3% shooting."
    passed, reason, unaligned = verifier.verify_payload(hallucinated_text, table)
    
    assert passed is False
    assert "UNALIGNED" in reason
    assert "54.3" in unaligned

def test_verifier_causal_language_guardrail():
    table = EvidenceTable()
    table.add_claim("NET_RATING", 10.5, "TeamPlayerOnOffSummary", formatted_value="+10.5")
    
    verifier = SynthesisVerifier()
    text = "Curry's gravity caused the Warriors to score +10.5 points."
    
    passed, reason, violations = verifier.verify_payload(text, table, strict_causal=True)
    assert passed is False
    assert "CAUSAL_VIOLATION" in reason

def test_verifier_thesis_counter_evidence_check():
    table = EvidenceTable()
    table.add_claim("NET_RATING", 8.2, "TeamPlayerOnOffSummary", formatted_value="+8.2")
    
    verifier = SynthesisVerifier()
    # Incomplete thesis
    incomplete_thesis = "The Knicks were dominant with a +8.2 net rating."
    passed, reason, _ = verifier.verify_payload(incomplete_thesis, table, is_thesis=True)
    assert passed is False
    assert "THESIS_INCOMPLETE" in reason
    
    # Complete thesis with counter-evidence
    complete_thesis = "The Knicks were dominant (+8.2), however counter-evidence shows a small sample size in the second round."
    passed, reason, _ = verifier.verify_payload(complete_thesis, table, is_thesis=True)
    assert passed is True


def test_verifier_blocks_unprovenanced_boxscore_integers():
    """Verify that common basketball integers (10, 12, 24, 48) are not allowlisted and must be in evidence"""
    table = EvidenceTable()
    verifier = SynthesisVerifier()
    
    text = "LeBron James scored 48 points on 10 of 12 shooting in 24 minutes."
    is_aligned, unaligned = verifier.verify_numerals(text, table)
    
    assert is_aligned is False
    assert "48" in unaligned
    assert "10" in unaligned
    assert "12" in unaligned
    assert "24" in unaligned


def test_verifier_audits_chartjs_data_points():
    """Verify that hallucinated data coordinates in chart blocks are audited and caught"""
    table = EvidenceTable()
    verifier = SynthesisVerifier()
    
    chart_code = "```chartjs\n{\"data\": [9999.9, 8888.8]}\n```"
    is_aligned, unaligned = verifier.verify_numerals(chart_code, table)
    assert is_aligned is False
    assert "9999.9" in unaligned
    assert "8888.8" in unaligned
    
    # Valid chart with evidence
    table.add_claim("PTS", 32.5, "PlayerGameLog", formatted_value="32.5")
    valid_chart = "```chartjs\n{\"data\": {\"datasets\": [{\"data\": [32.5]}]}}\n```"
    is_aligned_valid, unaligned_valid = verifier.verify_numerals(valid_chart, table)
    assert is_aligned_valid is True
    assert len(unaligned_valid) == 0
