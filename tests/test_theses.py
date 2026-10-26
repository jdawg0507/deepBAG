"""
Unit tests for Analysis of Competing Hypotheses (ACH) Matrix (Milestone 5).
"""
import pytest
from core.theses import ACHMatrix, Hypothesis
from core.evidence import EvidenceItem

def test_ach_matrix_hypothesis_generation():
    ach = ACHMatrix("What made the Knicks dominant in the playoffs?")
    ach.add_hypotheses_from_context([
        {"label": "Defensive Rebounding Dominance", "description": "Offensive and defensive glass control"},
        {"label": "Brunson Isolation Scoring", "description": "High-volume clutch shotmaking"}
    ], include_null=True)
    
    assert len(ach.hypotheses) == 3
    # Ensure H_NULL is present
    null_h = next((h for h in ach.hypotheses if h.is_null), None)
    assert null_h is not None
    assert "Shooting Variance" in null_h.label

def test_ach_least_refuted_ranking():
    ach = ACHMatrix("Test Query")
    ach.add_hypotheses_from_context([
        {"label": "Scheme Advantage", "description": "Defensive rotation scheme"},
        {"label": "Pure Shooting Variance", "description": "Unsustainable hot shooting"}
    ], include_null=False)
    
    # H1 has 1 support, 0 contradictions
    # H2 has 3 supports, but 1 fatal contradiction
    ev1 = EvidenceItem(claim_id="C1", metric_name="OREB%", value="34%", raw_value=34.0, endpoint="BoxScore", params_hash="123")
    ev2 = EvidenceItem(claim_id="C2", metric_name="3P%", value="31%", raw_value=31.0, endpoint="BoxScore", params_hash="456")
    
    ach.evaluate_evidence(ev1, {"H1": "S", "H2": "S"})
    ach.evaluate_evidence(ev2, {"H1": "S", "H2": "C"}) # H2 contradicted!
    
    ranked = ach.compute_least_refuted_ranking()
    # H1 must rank first because it has 0 contradictions, even if H2 had more support
    assert ranked[0].id == "H1"
    assert ranked[0].inconsistency_count == 0
    assert ranked[1].id == "H2"
    assert ranked[1].inconsistency_count == 1

def test_ach_markdown_matrix_rendering():
    ach = ACHMatrix("Knicks thesis")
    ach.add_hypotheses_from_context([{"label": "Rebounding", "description": "Glass"}], include_null=True)
    ev = EvidenceItem(claim_id="C1", metric_name="OREB%", value="35%", raw_value=35.0, endpoint="BoxScore", params_hash="abc")
    ach.evaluate_evidence(ev, {"H1": "S", "H_NULL": "C"})
    
    md = ach.render_markdown_matrix()
    assert "| `[C1]` OREB% (35%)" in md
    assert "✅ Supports" in md
    assert "❌ Contradicts" in md
