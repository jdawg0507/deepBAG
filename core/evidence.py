"""
Evidence Table & Claim Ledger for deepBAG.
Implements the PAL-style deterministic number-lock contract:
Every numerical value cited in downstream analysis originates from a code-computed
EvidenceItem with a unique claim ID ([C1], [C2], ...), provenanced by endpoint,
parameter hash, season, sample size, and formula version.
"""
import re
import hashlib
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field

class EvidenceItem(BaseModel):
    claim_id: str = Field(..., description="Unique claim identifier, e.g. C1, C2")
    metric_name: str = Field(..., description="Name of the statistic or metric, e.g. TS%, PPG, Net_Rating")
    value: str = Field(..., description="Formatted string representation of the value, e.g. '25.7', '58.4%'")
    raw_value: float = Field(..., description="Raw computed numerical value")
    endpoint: str = Field(..., description="NBA API endpoint or data source origin")
    params_hash: str = Field(..., description="MD5 hash of sorted input parameters")
    formula_version: str = Field(default="raw", description="Version identifier of calculation formula")
    sample_size: Optional[str] = Field(default=None, description="Sample size disclosure, e.g. 71 GP, 2504 MIN")
    description: Optional[str] = Field(default=None, description="Plain language explanation of metric")

class EvidenceTable:
    """
    Collection of registered claims and provenanced statistics for a query session.
    """
    def __init__(self):
        self._claims: Dict[str, EvidenceItem] = {}
        self._counter: int = 1

    def add_claim(
        self,
        metric_name: str,
        raw_value: float,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        formatted_value: Optional[str] = None,
        formula_version: str = "raw",
        sample_size: Optional[str] = None,
        description: Optional[str] = None,
    ) -> EvidenceItem:
        claim_id = f"C{self._counter}"
        self._counter += 1
        
        # Calculate canonical parameter hash
        sorted_params = json_canonical_sort(params) if params else ""
        p_hash = hashlib.md5(sorted_params.encode('utf-8')).hexdigest()[:8]
        
        if formatted_value is None:
            if isinstance(raw_value, float):
                formatted_value = f"{raw_value:.1f}" if abs(raw_value) >= 1 else f"{raw_value:.3f}"
            else:
                formatted_value = str(raw_value)
                
        item = EvidenceItem(
            claim_id=claim_id,
            metric_name=metric_name,
            value=formatted_value,
            raw_value=float(raw_value),
            endpoint=endpoint,
            params_hash=p_hash,
            formula_version=formula_version,
            sample_size=sample_size,
            description=description,
        )
        self._claims[claim_id] = item
        return item

    def get_claim(self, claim_id: str) -> Optional[EvidenceItem]:
        clean_id = claim_id.strip("[]")
        return self._claims.get(clean_id)

    def all_claims(self) -> List[EvidenceItem]:
        return list(self._claims.values())

    def render_markdown_table(self) -> str:
        """Render markdown table for inclusion in LLM prompt or answer appendix"""
        if not self._claims:
            return "_No evidence claims recorded._"
            
        lines = [
            "| ID | Metric | Value | Sample | Source Endpoint | Formula Version |",
            "|:---|:---|:---|:---|:---|:---|",
        ]
        for item in self._claims.values():
            sample = item.sample_size or "N/A"
            lines.append(
                f"| `[{item.claim_id}]` | **{item.metric_name}** | {item.value} | {sample} | `{item.endpoint}` | `{item.formula_version}` |"
            )
        return "\n".join(lines)

    def substitute_ids(self, text: str) -> str:
        """Replace all [C1], [C2] tags in text with their formatted numerical values"""
        def repl(match):
            cid = match.group(1)
            item = self._claims.get(cid)
            if item:
                return item.value
            return match.group(0) # Keep verbatim if not found
            
        return re.sub(r'\[(C\d+)\]', repl, text)

    def extract_expected_numerals(self) -> Set[str]:
        """Extract all valid numeral representations from the registered claims"""
        numerals = set()
        for item in self._claims.values():
            # Add formatted value
            val_clean = item.value.replace("%", "").strip()
            numerals.add(val_clean)
            # Add raw value formatted with various precision
            numerals.add(str(item.raw_value))
            numerals.add(f"{item.raw_value:.1f}")
            numerals.add(f"{item.raw_value:.2f}")
            numerals.add(f"{item.raw_value:.3f}")
            numerals.add(str(int(round(item.raw_value))))
        return numerals

    def register_dataset_claims(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]],
        datasets: Dict[str, List[Dict[str, Any]]],
        entity_hint: Optional[str] = None,
        ach: Optional[Any] = None,
    ) -> List[EvidenceItem]:
        """
        Dynamically extracts and registers verified statistical claims from NBA API datasets.
        Handles:
        1. Multi-game logs (calculates GP, totals, and per-game averages).
        2. Single-row season summaries / splits (extracts counting stats, shooting %, net rating).
        3. Derived analytics (TS%, eFG%).
        """
        new_items = []
        for ds_name, rows in datasets.items():
            if not rows or not isinstance(rows, list):
                continue
            
            # Case 1: Game log series (e.g. PlayerGameLog with multiple game entries)
            if len(rows) > 1 and "PTS" in rows[0] and "GAME_DATE" in rows[0]:
                gp = len(rows)
                pts_total = sum(float(r.get("PTS") or 0.0) for r in rows)
                ast_total = sum(float(r.get("AST") or 0.0) for r in rows)
                reb_total = sum(float(r.get("REB") or 0.0) for r in rows)
                stl_total = sum(float(r.get("STL") or 0.0) for r in rows)
                blk_total = sum(float(r.get("BLK") or 0.0) for r in rows)
                fga_total = sum(float(r.get("FGA") or 0.0) for r in rows)
                fgm_total = sum(float(r.get("FGM") or 0.0) for r in rows)
                fta_total = sum(float(r.get("FTA") or 0.0) for r in rows)
                fg3a_total = sum(float(r.get("FG3A") or 0.0) for r in rows)
                fg3m_total = sum(float(r.get("FG3M") or 0.0) for r in rows)
                
                prefix = f"{entity_hint} " if entity_hint else ""
                sample = f"{gp} GP"
                
                new_items.append(self.add_claim(f"{prefix}GAMES_PLAYED", float(gp), endpoint, params, formatted_value=str(gp), sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}PTS_TOTAL", pts_total, endpoint, params, formatted_value=f"{pts_total:.0f}", sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}PTS_PER_GAME", pts_total / gp, endpoint, params, formatted_value=f"{pts_total / gp:.1f}", sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}AST_TOTAL", ast_total, endpoint, params, formatted_value=f"{ast_total:.0f}", sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}AST_PER_GAME", ast_total / gp, endpoint, params, formatted_value=f"{ast_total / gp:.1f}", sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}REB_TOTAL", reb_total, endpoint, params, formatted_value=f"{reb_total:.0f}", sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}REB_PER_GAME", reb_total / gp, endpoint, params, formatted_value=f"{reb_total / gp:.1f}", sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}STL_TOTAL", stl_total, endpoint, params, formatted_value=f"{stl_total:.0f}", sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}STL_PER_GAME", stl_total / gp, endpoint, params, formatted_value=f"{stl_total / gp:.1f}", sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}BLK_TOTAL", blk_total, endpoint, params, formatted_value=f"{blk_total:.0f}", sample_size=sample))
                new_items.append(self.add_claim(f"{prefix}BLK_PER_GAME", blk_total / gp, endpoint, params, formatted_value=f"{blk_total / gp:.1f}", sample_size=sample))
                
                if fga_total > 0:
                    fg_pct = (fgm_total / fga_total) * 100.0
                    new_items.append(self.add_claim(f"{prefix}FG_PCT", fg_pct, endpoint, params, formatted_value=f"{fg_pct:.1f}%", sample_size=sample))
                if fg3a_total > 0:
                    fg3_pct = (fg3m_total / fg3a_total) * 100.0
                    new_items.append(self.add_claim(f"{prefix}3P_PCT", fg3_pct, endpoint, params, formatted_value=f"{fg3_pct:.1f}%", sample_size=sample))
                if fta_total > 0:
                    ft_pct = (sum(float(r.get("FTM") or 0.0) for r in rows) / fta_total) * 100.0
                    new_items.append(self.add_claim(f"{prefix}FT_PCT", ft_pct, endpoint, params, formatted_value=f"{ft_pct:.1f}%", sample_size=sample))
                continue
                
            # Case 2: Standard row summary (first few rows)
            for row in rows[:5]:
                name = row.get("PLAYER_NAME") or row.get("TEAM_NAME") or entity_hint or ""
                p_label = f"{name} " if name else ""
                sample = f"{row.get('GP', 'N/A')} GP" if 'GP' in row else f"{row.get('MIN', 'N/A')} MIN"
                
                # Counting & Percentage Metrics
                metric_keys = [
                    ("PTS", "PTS", "{:.1f}"),
                    ("AST", "AST", "{:.1f}"),
                    ("REB", "REB", "{:.1f}"),
                    ("STL", "STL", "{:.1f}"),
                    ("BLK", "BLK", "{:.1f}"),
                    ("GP", "GP", "{:.0f}"),
                    ("MIN", "MIN", "{:.1f}"),
                    ("FG_PCT", "FG%", "{:.1f}%", 100.0),
                    ("FG3_PCT", "3P%", "{:.1f}%", 100.0),
                    ("FT_PCT", "FT%", "{:.1f}%", 100.0),
                    ("NET_RATING", "NET_RATING", "{:+.1f}"),
                    ("OFF_RATING", "OFF_RATING", "{:.1f}"),
                    ("DEF_RATING", "DEF_RATING", "{:.1f}"),
                    ("PLUS_MINUS", "+/-", "{:+.1f}"),
                ]
                
                for key, display_name, fmt_str, *mult in metric_keys:
                    if key in row and row[key] is not None:
                        try:
                            val = float(row[key])
                            multiplier = mult[0] if mult else 1.0
                            if "%" in fmt_str and val <= 1.0 and val > 0:
                                val = val * 100.0
                            elif multiplier != 1.0 and val <= 1.0:
                                val = val * multiplier
                            item = self.add_claim(
                                metric_name=f"{p_label}{display_name}",
                                raw_value=val,
                                endpoint=endpoint,
                                params=params,
                                formatted_value=fmt_str.format(val),
                                sample_size=sample
                            )
                            new_items.append(item)
                            if ach:
                                ach.evaluate_evidence(item, {"H1": "S", "H2": "S", "H_NULL": "N"})
                        except (ValueError, TypeError):
                            pass
                            
                # Compute TS% if possible
                if "PTS" in row and "FGA" in row and "FTA" in row:
                    try:
                        pts = float(row["PTS"] or 0.0)
                        fga = float(row["FGA"] or 0.0)
                        fta = float(row["FTA"] or 0.0)
                        from core.analytics import calculate_ts_percent
                        ts_raw, ts_fmt = calculate_ts_percent(pts, fga, fta)
                        item = self.add_claim(
                            metric_name=f"{p_label}TS%",
                            raw_value=ts_raw * 100.0,
                            endpoint=endpoint,
                            params=params,
                            formatted_value=ts_fmt,
                            formula_version="analytics:ts_v1.0",
                            sample_size=sample
                        )
                        new_items.append(item)
                        if ach:
                            ach.evaluate_evidence(item, {"H1": "S", "H2": "S", "H_NULL": "N"})
                    except Exception:
                        pass
                        
        return new_items

def json_canonical_sort(obj: Any) -> str:
    import json
    return json.dumps(obj, sort_keys=True)
