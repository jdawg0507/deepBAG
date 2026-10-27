"""
Unit tests for Synthesizer and dynamic Chart.js injection.
"""
import pytest
from nlp.synthesizer import should_inject_chart, truncate_dataset_for_llm, Synthesizer

def test_should_inject_chart_keywords():
    assert should_inject_chart("Who scored more points, LeBron vs Curry?", []) is True
    assert should_inject_chart("Compare Jokic and Embiid", []) is True
    assert should_inject_chart("Show scoring leaders in 2023-24", []) is True
    assert should_inject_chart("What is LeBron James' height?", [{"HEIGHT": "6-9"}]) is False

def test_should_inject_chart_data_size():
    small_data = [{"game": 1, "pts": 20}, {"game": 2, "pts": 25}]
    assert should_inject_chart("LeBron stats", small_data) is False

    large_data = [
        {"game": 1, "pts": 20},
        {"game": 2, "pts": 25},
        {"game": 3, "pts": 30},
    ]
    assert should_inject_chart("LeBron stats", large_data) is True

def test_truncate_dataset_for_llm():
    items = [{"id": i} for i in range(50)]
    truncated = truncate_dataset_for_llm(items, max_rows=10)
    assert len(truncated) == 10

    dict_data = {"PlayerStats": [{"id": i} for i in range(30)], "Meta": "info"}
    truncated_dict = truncate_dataset_for_llm(dict_data, max_rows=5)
    assert len(truncated_dict["PlayerStats"]) == 5
    assert truncated_dict["Meta"] == "info"

def test_synthesizer_init():
    synth = Synthesizer()
    assert synth.default_model is not None
    assert len(synth.fallback_models) >= 2


def test_synthesize_research_fails_closed_on_unaligned_numerals():
    """Verify that hallucinated numerals cause synthesis to fail closed without emitting hallucinated text"""
    from unittest.mock import MagicMock
    from core.evidence import EvidenceTable
    from core.schemas import IntentContract, QuestionType
    
    synth = Synthesizer()
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "LeBron James scored 77 points and had 33 assists [C1]."
    synth.llm_client = mock_llm
    
    evidence = EvidenceTable()
    evidence.add_claim("PTS", 30.0, "PlayerGameLog", formatted_value="30")
    
    intent = IntentContract(
        question_type=QuestionType.SINGLE,
        entities=[],
        season="2023-24",
        interpretation="Test intent"
    )
    
    events = list(synth.synthesize_research("How many points did LeBron score?", evidence, intent))
    final_events = [e for e in events if e.get("type") == "final_answer"]
    assert len(final_events) == 1
    final_content = final_events[0]["content"]
    
    assert "Synthesis Blocked (Fail-Closed Number-Lock Enforcement)" in final_content
    assert "Number Lock: FAILED" in final_content
    assert "LeBron James scored 77 points" not in final_content
