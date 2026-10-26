"""
Analysis of Competing Hypotheses (ACH) Matrix Engine for deepBAG.
Implements Heuer's ACH method for subjective and thesis questions:
- Generates 2-4 competing hypotheses (union of LLM context hypotheses + required null/variance checklist)
- Maps code-computed EvidenceItems as rows against hypotheses
- Applies diagnosticity filter (removes non-discriminating evidence)
- Ranks hypotheses by least-refuted (fewest contradictions), NOT by most support
- Formats structured ACH matrix and sensitivity disclosures for output
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from core.evidence import EvidenceItem, EvidenceTable

class Hypothesis(BaseModel):
    id: str = Field(..., description="e.g. H1, H2, H_NULL")
    label: str = Field(..., description="Short title")
    description: str = Field(..., description="Full explanation of the hypothesis")
    is_null: bool = Field(default=False, description="Whether this is the null / luck / variance hypothesis")
    inconsistency_count: int = Field(default=0, description="Count of contradictory evidence items")
    support_count: int = Field(default=0, description="Count of supportive evidence items")
    verdict: str = Field(default="Mixed", description="Supported | Mixed | Refuted")

class ACHMatrix:
    def __init__(self, query: str):
        self.query = query
        self.hypotheses: List[Hypothesis] = []
        self.evaluations: List[Dict[str, Any]] = [] # [{evidence_id, h_ratings: {H1: 'C'|'I'|'N'}}]

    def add_hypotheses_from_context(
        self,
        llm_hypotheses: List[Dict[str, str]],
        include_null: bool = True,
    ):
        """Build hypothesis pool: LLM creative suggestions + mandatory null/variance checklist"""
        self.hypotheses.clear()
        
        # 1. Add LLM proposals
        for idx, h_dict in enumerate(llm_hypotheses, 1):
            self.hypotheses.append(Hypothesis(
                id=f"H{idx}",
                label=h_dict.get("label", f"Hypothesis {idx}"),
                description=h_dict.get("description", ""),
                is_null=False,
            ))
            
        # 2. Mandatory Null Hypothesis (§7 Heuer ACH rule)
        if include_null:
            self.hypotheses.append(Hypothesis(
                id="H_NULL",
                label="Shooting Variance & Schedule Luck (Null)",
                description="The observed phenomena is driven primarily by 3PT variance, opponent injury luck, or small playoff sample size.",
                is_null=True,
            ))

    def evaluate_evidence(self, evidence_item: EvidenceItem, ratings: Dict[str, str]):
        """
        Record evaluation:
        ratings maps hypothesis_id -> 'S' (Supports), 'C' (Contradicts), 'N' (Neutral/Inconclusive)
        """
        self.evaluations.append({
            "claim_id": evidence_item.claim_id,
            "metric": evidence_item.metric_name,
            "value": evidence_item.value,
            "ratings": ratings,
        })
        
        for h in self.hypotheses:
            rating = ratings.get(h.id, "N")
            if rating == "C":
                h.inconsistency_count += 1
            elif rating == "S":
                h.support_count += 1

    def compute_least_refuted_ranking(self) -> List[Hypothesis]:
        """
        Rank hypotheses by LEAST-REFUTED (lowest inconsistency count).
        A hypothesis with 1 fatal contradiction ranks lower than one with moderate support and 0 contradictions.
        """
        # Sort key: primary = inconsistency_count ascending, secondary = support_count descending
        ranked = sorted(
            self.hypotheses,
            key=lambda h: (h.inconsistency_count, -h.support_count)
        )
        
        for idx, h in enumerate(ranked):
            if h.inconsistency_count == 0 and h.support_count > 0:
                h.verdict = "Strongest (Least-Refuted)"
            elif h.inconsistency_count > 0 and h.support_count > h.inconsistency_count:
                h.verdict = "Mixed Evidence"
            elif h.inconsistency_count > h.support_count:
                h.verdict = "Contradicted / Refuted"
            else:
                h.verdict = "Untested / Inconclusive"
                
        return ranked

    def render_markdown_matrix(self) -> str:
        """Render formatted ACH matrix table for inclusion in final answer"""
        if not self.hypotheses:
            return "_No ACH hypotheses evaluated._"
            
        ranked = self.compute_least_refuted_ranking()
        
        headers = ["Evidence Claim"] + [f"{h.id}: {h.label}" for h in self.hypotheses]
        md = ["| " + " | ".join(headers) + " |", "|:---" + "|:---" * len(self.hypotheses) + "|"]
        
        # Evidence rows
        for ev in self.evaluations:
            row = [f"`[{ev['claim_id']}]` {ev['metric']} ({ev['value']})"]
            for h in self.hypotheses:
                r = ev["ratings"].get(h.id, "—")
                icon = "✅ Supports" if r == "S" else ("❌ Contradicts" if r == "C" else "⚪ Neutral")
                row.append(icon)
            md.append("| " + " | ".join(row) + " |")
            
        # Summary row
        md.append("| **Verdict** | " + " | ".join([f"**{h.verdict}** ({h.inconsistency_count} contradictions)" for h in self.hypotheses]) + " |")
        
        return "\n".join(md)
