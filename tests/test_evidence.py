"""
Unit tests for Evidence Table & Claim Ledger (Milestone 2).
"""
import pytest
from core.evidence import EvidenceTable, EvidenceItem

def test_evidence_table_add_and_retrieve():
    table = EvidenceTable()
    item1 = table.add_claim(
        metric_name="TS%",
        raw_value=0.612,
        endpoint="PlayerGameLog",
        params={"player_id": 2544, "season": "2023-24"},
        formatted_value="61.2%",
        formula_version="analytics:ts_v1.0",
        sample_size="71 GP",
        description="True shooting percentage"
    )
    
    assert item1.claim_id == "C1"
    assert item1.metric_name == "TS%"
    assert item1.value == "61.2%"
    assert item1.raw_value == 0.612
    assert len(item1.params_hash) == 8
    
    # Second claim gets C2
    item2 = table.add_claim(
        metric_name="PPG",
        raw_value=25.7,
        endpoint="PlayerGameLog",
        formatted_value="25.7"
    )
    assert item2.claim_id == "C2"
    assert table.get_claim("C1") == item1
    assert table.get_claim("[C2]") == item2

def test_evidence_id_substitution():
    table = EvidenceTable()
    table.add_claim("TS%", 0.612, "PlayerGameLog", formatted_value="61.2%")
    table.add_claim("PPG", 25.7, "PlayerGameLog", formatted_value="25.7")
    
    text = "LeBron posted a true shooting of [C1] while averaging [C2] points."
    substituted = table.substitute_ids(text)
    
    assert substituted == "LeBron posted a true shooting of 61.2% while averaging 25.7 points."

def test_numeral_extraction():
    table = EvidenceTable()
    table.add_claim("TS%", 0.612, "PlayerGameLog", formatted_value="61.2%")
    table.add_claim("PPG", 25.7, "PlayerGameLog", formatted_value="25.7")
    
    expected = table.extract_expected_numerals()
    assert "61.2" in expected
    assert "25.7" in expected
