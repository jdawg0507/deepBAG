"""
DeepEval Automated Regression & CI Evaluation Suite for deepBAG.
Evaluates:
1. Tool Selection F1-Score across 50 ambiguous edge-case queries.
2. Parameter Extraction Accuracy (player ID, date ranges, stat filters).
Blocks CI on drift if metrics fall below threshold (0.85).
"""
import os
import json
import pytest
from typing import Dict, Any, List
from collections import defaultdict
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "fixtures", "eval_50_legacy.json")

def load_benchmark_queries() -> List[Dict[str, Any]]:
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------
# 1. Custom DeepEval Metric: Tool Selection F1-Score
# ---------------------------------------------------------

class ToolSelectionF1Metric(BaseMetric):
    """
    Evaluates multi-class Tool Selection Precision, Recall, and F1-Score
    across call_nba_api, search_api_docs, and query_historical_database.
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        # Expects JSON metadata in test_case.context containing { "predicted_tools": [...], "expected_tools": [...] }
        eval_data = json.loads(test_case.actual_output)
        predictions = eval_data.get("predicted_tools", [])
        ground_truth = eval_data.get("expected_tools", [])

        classes = ["call_nba_api", "search_api_docs", "query_historical_database"]
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        for pred, gt in zip(predictions, ground_truth):
            if pred == gt:
                tp[pred] += 1
            else:
                fp[pred] += 1
                fn[gt] += 1

        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())

        self.precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        self.recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        self.f1 = (
            (2 * self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0.0
        )

        self.score = round(self.f1, 4)
        self.success = self.score >= self.threshold
        self.reason = (
            f"Tool Selection F1-Score: {self.score:.4f} (Precision: {self.precision:.4f}, Recall: {self.recall:.4f}). "
            f"Threshold is {self.threshold:.2f}."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold

# ---------------------------------------------------------
# 2. Custom DeepEval Metric: Parameter Extraction Accuracy
# ---------------------------------------------------------

class ParameterExtractionAccuracyMetric(BaseMetric):
    """
    Evaluates whether the router accurately extracted key parameters
    (Player ID, Season / Date filters, MeasureType / Stat filters).
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        eval_data = json.loads(test_case.actual_output)
        comparisons = eval_data.get("param_comparisons", [])

        if not comparisons:
            self.score = 1.0
            self.success = True
            return self.score

        matches = sum(1 for c in comparisons if c.get("match") is True)
        self.score = round(matches / len(comparisons), 4)
        self.success = self.score >= self.threshold
        self.reason = (
            f"Parameter Extraction Accuracy: {self.score:.4f} ({matches}/{len(comparisons)} parameters matched). "
            f"Threshold is {self.threshold:.2f}."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold

# ---------------------------------------------------------
# 3. Predictor / Hybrid Router Harness
# ---------------------------------------------------------
from nlp.hybrid_router import HybridRouter
hybrid_router = HybridRouter()

def route_query_semantic(query_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Directly delegates routing to deepBAG HybridRouter without circular overwrites.
    """
    plan = hybrid_router.route(query_item["query"])
    return {
        "tool": plan.tool,
        "endpoint": plan.endpoint,
        "params": dict(plan.params),
    }


# ---------------------------------------------------------
# 4. DeepEval Test Cases & Assertions
# ---------------------------------------------------------

def test_deepbag_50_queries_benchmark():
    """
    Runs the DeepEval benchmark over 50 edge-case queries.
    Blocks CI on performance regression or drift.
    """
    benchmark_data = load_benchmark_queries()
    assert len(benchmark_data) == 50, f"Expected 50 queries, found {len(benchmark_data)}"

    predicted_tools = []
    expected_tools = []
    param_comparisons = []

    for item in benchmark_data:
        routed = route_query_semantic(item)
        pred_tool = routed["tool"]
        exp_tool = item["expected_tool"]

        predicted_tools.append(pred_tool)
        expected_tools.append(exp_tool)

        # Check parameter accuracy
        exp_params = item.get("expected_params", {})
        pred_params = routed.get("params", {})

        for p_key, p_val in exp_params.items():
            if p_key == "keywords" and isinstance(p_val, str) and isinstance(pred_params.get(p_key), str):
                exp_tokens = set(p_val.lower().split())
                pred_tokens = set(pred_params.get(p_key, "").lower().split())
                overlap = len(exp_tokens & pred_tokens)
                param_match = (overlap / len(exp_tokens) >= 0.5) if exp_tokens else True
            else:
                param_match = pred_params.get(p_key) == p_val
            param_comparisons.append({
                "query_id": item["id"],
                "param": p_key,
                "expected": p_val,
                "predicted": pred_params.get(p_key),
                "match": param_match,
            })

    eval_payload = {
        "predicted_tools": predicted_tools,
        "expected_tools": expected_tools,
        "param_comparisons": param_comparisons,
    }

    test_case = LLMTestCase(
        input="NBA API 50 Edge-Case Semantic Router Benchmark",
        actual_output=json.dumps(eval_payload),
    )

    # 1. Evaluate Tool Selection F1-Score
    f1_metric = ToolSelectionF1Metric(threshold=0.85)
    f1_metric.measure(test_case)
    print(f"\n[DEEPEVAL GATE] {f1_metric.reason}")

    # 2. Evaluate Parameter Extraction Accuracy
    param_metric = ParameterExtractionAccuracyMetric(threshold=0.85)
    param_metric.measure(test_case)
    print(f"[DEEPEVAL GATE] {param_metric.reason}")

    # Enforce CI Gate Thresholds
    assert f1_metric.is_successful(), f"Tool Selection F1 regression! F1: {f1_metric.score} < {f1_metric.threshold}"
    assert param_metric.is_successful(), f"Parameter Extraction Accuracy regression! Score: {param_metric.score} < {param_metric.threshold}"
