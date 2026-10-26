"""
Temporal Resolution & Scope Receipts for deepBAG.
Replaces silent defaults with explicit date logic and transparent scope receipts.
"""
import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ScopeAssumption(BaseModel):
    category: str
    assumed_value: str
    reason: str

class ScopeReceipt(BaseModel):
    season: str
    season_type: str = "Regular Season"
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)

    def format_markdown(self) -> str:
        assumptions_str = "; ".join(self.assumptions) if self.assumptions else "Exact user scope"
        entities_str = ", ".join(e.get("name", "Unknown") for e in self.entities) or "League-wide"
        return (
            f"📋 **Scope Receipt:** Season: **{self.season}** · SeasonType: **{self.season_type}** · Entities: **{entities_str}**\n"
            f"> *Assumptions:* {assumptions_str}\n"
            f"> *Adjust:* Reply *\"make it playoffs\"* or *\"change season to 2022-23\"* to refine."
        )

class TemporalResolver:
    @staticmethod
    def current_season(today: Optional[datetime.date] = None) -> str:
        """Computes current NBA season based on the NBA calendar (Oct-June vs Offseason)."""
        d = today or datetime.date.today()
        # NBA season starts in October
        if d.month >= 10:
            return f"{d.year}-{str(d.year + 1)[-2:]}"
        else:
            return f"{d.year - 1}-{str(d.year)[-2:]}"

    @staticmethod
    def prior_season(today: Optional[datetime.date] = None) -> str:
        """Computes prior completed NBA season."""
        curr = TemporalResolver.current_season(today)
        start_year = int(curr[:4]) - 1
        return f"{start_year}-{str(start_year + 1)[-2:]}"

    @staticmethod
    def resolve_temporal_ref(
        query: str,
        player_from_year: Optional[int] = None,
        today: Optional[datetime.date] = None
    ) -> Dict[str, Any]:
        """
        Resolves relative temporal references ('rookie year', 'this season', 'last season', etc.).
        Returns dict with: season, season_type, is_assumed, assumption_text
        """
        q_lower = query.lower()
        curr_season = TemporalResolver.current_season(today)
        
        # Check explicit season pattern (e.g. 2023-24, 2016-17)
        import re
        m = re.search(r"\b(19\d{2}|20\d{2})[-/](\d{2})\b", q_lower)
        if m:
            century = m.group(1)[:2]
            y2 = int(m.group(2))
            full_y2 = f"{century}{m.group(2)}"
            return {
                "season": f"{m.group(1)}-{m.group(2)}",
                "is_assumed": False,
                "assumption_text": None
            }

        # Check rookie year
        if any(k in q_lower for k in ["rookie year", "rookie season", "debut year", "debut season"]):
            if player_from_year:
                rookie_season = f"{player_from_year}-{str(player_from_year + 1)[-2:]}"
                return {
                    "season": rookie_season,
                    "is_assumed": False,
                    "assumption_text": f"Resolved rookie season ({rookie_season}) from player career start."
                }

        # Check last season / prior season
        if any(k in q_lower for k in ["last season", "prior season", "previous season"]):
            prior = TemporalResolver.prior_season(today)
            return {
                "season": prior,
                "is_assumed": False,
                "assumption_text": f"Resolved to prior season ({prior})."
            }

        # Check this season / current season
        if any(k in q_lower for k in ["this season", "current season", "today"]):
            return {
                "season": curr_season,
                "is_assumed": False,
                "assumption_text": None
            }

        # Default fallback: Unspecified season defaults to current season ONLY with explicit assumption
        return {
            "season": curr_season,
            "is_assumed": True,
            "assumption_text": f"Assumed current season ({curr_season}) as no season was specified."
        }
