"""
Temporal Resolution & Scope Receipts for deepBAG.
Replaces silent defaults with explicit date logic and transparent scope receipts.
"""
import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class ScopeAssumption(BaseModel):
    category: str
    assumed_value: str
    reason: str

class ScopeReceipt(BaseModel):
    """
    Universal Scope Receipt.
    Fully abstractable beyond basketball: supports seasons, specific days,
    date ranges, arbitrary entities (players, teams, stocks, servers), and dynamic dimensions.
    """
    model_config = ConfigDict(extra="allow")

    season: Optional[str] = None
    season_type: Optional[str] = None
    date: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    timeframe: Optional[str] = None
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)

    def format_markdown(self) -> str:
        parts = []
        if self.date:
            parts.append(f"Date: **{self.date}**")
        elif self.date_from and self.date_to:
            parts.append(f"Dates: **{self.date_from}** to **{self.date_to}**")
        elif self.season:
            parts.append(f"Season: **{self.season}**")

        if self.season_type:
            parts.append(f"Scope: **{self.season_type}**")
        elif self.timeframe:
            parts.append(f"Timeframe: **{self.timeframe}**")

        if self.entities:
            ent_names = [e.get("name") or e.get("label") or str(e.get("id")) for e in self.entities if e]
            if ent_names:
                parts.append(f"Entities: **{', '.join(ent_names)}**")

        for k, v in self.dimensions.items():
            if v:
                clean_k = k.replace("_", " ").title()
                parts.append(f"{clean_k}: **{v}**")

        scope_summary = " · ".join(parts) if parts else "Universal Scope"
        assumptions_str = "; ".join(self.assumptions) if self.assumptions else "Exact user scope"

        return (
            f"📋 **Scope Receipt:** {scope_summary}\n"
            f"> *Assumptions:* {assumptions_str}\n"
            f"> *Adjust:* Reply to alter date, season, entity, or filters in one turn."
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
        
        # 1. Check explicit specific date (e.g. March 2, 1962, 1962-03-02, Dec 25, 2023)
        import re
        date_pattern = r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(19\d{2}|20\d{2})\b"
        date_match = re.search(date_pattern, q_lower)
        if date_match:
            month_str, day_str, year_str = date_match.groups()
            year_int = int(year_str)
            # Normalize month
            months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
            m_prefix = month_str[:3]
            month_int = months.get(m_prefix, 1)
            # In NBA, Oct-Dec belongs to start year, Jan-July belongs to prior year's season
            season_start = year_int if month_int >= 10 else year_int - 1
            calculated_season = f"{season_start}-{str(season_start + 1)[-2:]}"
            formatted_date = f"{month_str.capitalize()} {day_str}, {year_str}"
            return {
                "date": formatted_date,
                "season": calculated_season,
                "is_assumed": False,
                "assumption_text": f"Resolved date {formatted_date} to {calculated_season} season."
            }

        # Check ISO date (YYYY-MM-DD)
        iso_match = re.search(r"\b(19\d{2}|20\d{2})-(\d{2})-(\d{2})\b", q_lower)
        if iso_match:
            y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
            season_start = y if m >= 10 else y - 1
            calculated_season = f"{season_start}-{str(season_start + 1)[-2:]}"
            iso_date = f"{y:04d}-{m:02d}-{d:02d}"
            return {
                "date": iso_date,
                "season": calculated_season,
                "is_assumed": False,
                "assumption_text": f"Resolved date {iso_date} to {calculated_season} season."
            }

        # 2. Check explicit season pattern (e.g. 2023-24, 2016-17)
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

        # 3. Check rookie year
        if any(k in q_lower for k in ["rookie year", "rookie season", "debut year", "debut season"]):
            if player_from_year:
                rookie_season = f"{player_from_year}-{str(player_from_year + 1)[-2:]}"
                return {
                    "season": rookie_season,
                    "is_assumed": False,
                    "assumption_text": f"Resolved rookie season ({rookie_season}) from player career start."
                }

        # 4. Check last season / prior season
        if any(k in q_lower for k in ["last season", "prior season", "previous season"]):
            prior = TemporalResolver.prior_season(today)
            return {
                "season": prior,
                "is_assumed": False,
                "assumption_text": f"Resolved to prior season ({prior})."
            }

        # 5. Check this season / current season
        if any(k in q_lower for k in ["this season", "current season", "today"]):
            return {
                "season": curr_season,
                "is_assumed": False,
                "assumption_text": None
            }

        # 6. Check standalone 4-digit calendar year (e.g. "in 1998", "1962 game")
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", q_lower)
        if year_match:
            y_val = int(year_match.group(1))
            # Standard mapping for single year: prior-calendar to calendar (e.g. 1998 -> 1997-98)
            inferred_season = f"{y_val - 1}-{str(y_val)[-2:]}"
            return {
                "season": inferred_season,
                "timeframe": str(y_val),
                "is_assumed": False,
                "assumption_text": f"Mapped calendar year {y_val} to {inferred_season} season."
            }

        # Default fallback: Unspecified season defaults to current season ONLY with explicit assumption
        return {
            "season": curr_season,
            "is_assumed": True,
            "assumption_text": f"Assumed current season ({curr_season}) as no season or date was specified."
        }
