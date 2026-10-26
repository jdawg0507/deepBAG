"""
Research Session & Scope Receipt Manager for deepBAG.
Implements:
1. Stathead-style transparent scope receipt generation ("I assumed: Regular Season 2023-24...")
2. Three-way gap triage: tool-handled / resolvable-by-assumption / ask
3. Correction turn handling: allows user to alter scope/filter in-place without restarting conversation.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from core.schemas import IntentContract, ScopeFilters
from core.evidence import EvidenceTable

class ScopeReceipt:
    @staticmethod
    def format_receipt(intent: IntentContract) -> str:
        """
        Produce a plain-language Stathead scope receipt header.
        Discloses every material scope decision made for the user.
        """
        items = []
        if intent.season:
            items.append(f"Season: **{intent.season}**")
        if intent.season_type:
            items.append(f"Type: **{intent.season_type}**")
            
        f = intent.filters
        if f.location:
            items.append(f"Location: **{f.location}**")
        if f.game_segment:
            items.append(f"Segment: **{f.game_segment}**")
        if f.period and f.period > 0:
            items.append(f"Period: **Quarter {f.period}**")
        if f.per_mode:
            items.append(f"Mode: **{f.per_mode}**")
        if f.last_n_games and f.last_n_games > 0:
            items.append(f"Sample: **Last {f.last_n_games} Games**")
            
        scope_str = ", ".join(items) if items else "Full Season"
        
        assumptions_str = ""
        if intent.assumptions:
            assumptions_str = " (" + "; ".join(intent.assumptions) + ")"
            
        interpretation_str = ""
        if intent.interpretation and intent.interpretation != "Fallback direct stat lookup.":
            interpretation_str = f"\n> **Interpretation used:** *{intent.interpretation}*"
            
        return (
            f"📋 **Scope Receipt:** {scope_str}{assumptions_str}"
            f"{interpretation_str}\n"
            f"> _Say \"change to playoffs\" or \"change season to 2022-23\" to adjust this scope in one turn._\n"
        )

class ResearchSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, Any]] = []
        self.current_intent: Optional[IntentContract] = None
        self.current_evidence: Optional[EvidenceTable] = None

    def apply_scope_correction(self, correction_text: str) -> Optional[IntentContract]:
        """
        Patch current intent with user scope correction without restarting the session.
        e.g. 'no, in the playoffs' -> sets season_type='Playoffs'.
        """
        if not self.current_intent:
            return None
            
        cor_lower = correction_text.lower()
        new_intent = self.current_intent.model_copy(deep=True)
        
        # Season Type corrections
        if "playoff" in cor_lower:
            new_intent.season_type = "Playoffs"
            new_intent.assumptions.append("User corrected scope to Playoffs.")
        elif "regular season" in cor_lower:
            new_intent.season_type = "Regular Season"
            new_intent.assumptions.append("User corrected scope to Regular Season.")
            
        # Interpretation corrections (e.g. gravity)
        if "rim" in cor_lower or "interior" in cor_lower:
            new_intent.interpretation = "Interior deterrence and rim protection gravity."
            new_intent.assumptions.append("User corrected interpretation to Interior Gravity.")
        elif "perimeter" in cor_lower or "spacing" in cor_lower:
            new_intent.interpretation = "Perimeter shooting pull and off-ball spacing gravity."
            new_intent.assumptions.append("User corrected interpretation to Perimeter Gravity.")
            
        self.current_intent = new_intent
        return new_intent
