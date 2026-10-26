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
