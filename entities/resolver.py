"""
Three-Tier Data-Driven Entity Resolver for deepBAG.
Replaces static nickname dictionaries with:
Tier 1: Exact & Diacritic Normalization
Tier 2: Trigram / Jaro-Winkler Fuzzy Matching with Career Bounds Filtering
Tier 3: Nebius LLM Disambiguator + SQLite Persistent Context Cache
"""
import os
import re
import json
import sqlite3
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from nba_api.stats.static import players, teams
from catalogue.schema import PlayerMeta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "disambiguation_cache.db")

class ResolvedEntity(BaseModel):
    name: str
    id: int
    type: str = "player"  # "player" or "team"
    from_year: Optional[int] = None
    to_year: Optional[int] = None
    source: str = "exact"  # "exact", "fuzzy_temporal", "llm_inferred", "cache"
    confidence: float = 1.0

class ClarificationRequest(BaseModel):
    ambiguous_mention: str
    candidates: List[Dict[str, Any]]
    message: str

def normalize_text(text: str) -> str:
    """Strips accents and normalizes to lower ascii."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Computes Jaro-Winkler string similarity."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = ((matches / len1) + (matches / len2) + ((matches - transpositions / 2) / matches)) / 3.0
    
    # Winkler bonus
    prefix = 0
    for i in range(min(4, min(len1, len2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1.0 - jaro)

class EntityResolver:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
        self._load_static_entities()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS disambiguation_cache (
                surface_form TEXT PRIMARY KEY,
                decade_or_season_bucket TEXT,
                resolved_player_id INTEGER,
                resolved_name TEXT,
                source TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            # Pre-seed popular monikers
            seeds = [
                ("joker", "2020s", 203999, "Nikola Jokic", "seed", 1.0),
                ("jokic", "2020s", 203999, "Nikola Jokic", "seed", 1.0),
                ("ant", "2020s", 1630162, "Anthony Edwards", "seed", 1.0),
                ("ant edwards", "2020s", 1630162, "Anthony Edwards", "seed", 1.0),
                ("wemby", "2020s", 1641705, "Victor Wembanyama", "seed", 1.0),
                ("chet", "2020s", 1631096, "Chet Holmgren", "seed", 1.0),
                ("steph", "2010s_2020s", 201939, "Stephen Curry", "seed", 1.0),
                ("curry", "2010s_2020s", 201939, "Stephen Curry", "seed", 1.0),
                ("lebron", "2000s_2020s", 2544, "LeBron James", "seed", 1.0),
                ("the beard", "2010s_2020s", 201935, "James Harden", "seed", 1.0),
                ("greek freak", "2010s_2020s", 203507, "Giannis Antetokounmpo", "seed", 1.0),
                ("giannis", "2010s_2020s", 203507, "Giannis Antetokounmpo", "seed", 1.0),
                ("spida", "2020s", 1628378, "Donovan Mitchell", "seed", 1.0),
                ("dame", "2010s_2020s", 203081, "Damian Lillard", "seed", 1.0),
                ("book", "2010s_2020s", 1626164, "Devin Booker", "seed", 1.0),
                ("sga", "2020s", 1628983, "Shai Gilgeous-Alexander", "seed", 1.0),
                ("shai", "2020s", 1628983, "Shai Gilgeous-Alexander", "seed", 1.0),
                ("tatum", "2010s_2020s", 1628369, "Jayson Tatum", "seed", 1.0),
                ("kd", "2000s_2020s", 201142, "Kevin Durant", "seed", 1.0),
                ("mj", "1980s_1990s", 893, "Michael Jordan", "seed", 1.0),
                ("air jordan", "1980s_1990s", 893, "Michael Jordan", "seed", 1.0),
                ("magic", "1980s_1990s", 77142, "Earvin Magic Johnson", "seed", 1.0),
            ]
            conn.executemany("""
            INSERT OR IGNORE INTO disambiguation_cache 
            (surface_form, decade_or_season_bucket, resolved_player_id, resolved_name, source, confidence)
            VALUES (?, ?, ?, ?, ?, ?);
            """, seeds)
            conn.commit()

    def _load_static_entities(self):
        self.all_players = players.get_players()
        self.all_teams = teams.get_teams()
        
        # Load career bounds
        cb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "player_career_bounds.json")
        self.career_bounds = {}
        if os.path.exists(cb_path):
            with open(cb_path, "r", encoding="utf-8") as f:
                raw_cb = json.load(f)
                self.career_bounds = {int(k): tuple(v) for k, v in raw_cb.items()}

        # Exact Normalized Lookups
        self.player_by_norm = {}
        for p in self.all_players:
            pid = p["id"]
            if pid in self.career_bounds:
                p["from_year"], p["to_year"] = self.career_bounds[pid]
            else:
                p["from_year"] = p.get("from_year", 1946)
                p["to_year"] = p.get("to_year", 2024)

            norm = normalize_text(p["full_name"])
            self.player_by_norm[norm] = p
            
        self.team_by_norm = {}
        for t in self.all_teams:
            self.team_by_norm[normalize_text(t["full_name"])] = t
            self.team_by_norm[normalize_text(t["nickname"])] = t
            self.team_by_norm[normalize_text(t["abbreviation"])] = t

    def resolve(
        self,
        surface_form: str,
        season_hint: Optional[str] = None,
        llm_client: Optional[Any] = None
    ) -> Optional[ResolvedEntity]:
        norm = normalize_text(surface_form)
        if not norm:
            return None

        # ---------------------------------------------------------
        # Tier 1: Exact & Diacritic Normalization
        # ---------------------------------------------------------
        if norm in self.team_by_norm:
            t = self.team_by_norm[norm]
            return ResolvedEntity(
                name=t["full_name"],
                id=t["id"],
                type="team",
                source="exact",
                confidence=1.0
            )

        if norm in self.player_by_norm:
            p = self.player_by_norm[norm]
            return ResolvedEntity(
                name=p["full_name"],
                id=p["id"],
                type="player",
                from_year=p.get("from_year", 1946),
                to_year=p.get("to_year", 2024),
                source="exact",
                confidence=1.0
            )

        # ---------------------------------------------------------
        # Tier 3 Cache Check (Disambiguation SQLite)
        # ---------------------------------------------------------
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT resolved_player_id, resolved_name, source, confidence FROM disambiguation_cache WHERE surface_form = ?", (norm,))
            row = cur.fetchone()
            if row:
                pid = row[0]
                fy, ty = self.career_bounds.get(pid, (1946, 2024))
                return ResolvedEntity(
                    id=pid,
                    name=row[1],
                    type="player",
                    from_year=fy,
                    to_year=ty,
                    source=f"cache_{row[2]}",
                    confidence=row[3]
                )

        # ---------------------------------------------------------
        # Tier 2: Trigram / Jaro-Winkler + Career Year Filtering
        # ---------------------------------------------------------
        target_year = int(season_hint[:4]) if season_hint and len(season_hint) >= 4 else 2023
        
        # Filter candidate pool to players active in the era if hint exists
        candidate_pool = self.all_players
        if season_hint:
            filtered = [
                p for p in self.all_players
                if (p.get("from_year", 1946) <= target_year + 1 and p.get("to_year", 2024) >= target_year - 1)
            ]
            if filtered:
                candidate_pool = filtered

        scores = []
        for p in candidate_pool:
            p_norm = normalize_text(p["full_name"])
            sim = jaro_winkler_similarity(norm, p_norm)
            scores.append((sim, p))

        scores.sort(key=lambda x: x[0], reverse=True)
        if scores and scores[0][0] >= 0.88:
            top1_score, top1_player = scores[0]
            top2_score = scores[1][0] if len(scores) > 1 else 0.0
            
            # Check margin
            if (top1_score - top2_score) >= 0.10 or top1_score >= 0.95:
                return ResolvedEntity(
                    name=top1_player["full_name"],
                    id=top1_player["id"],
                    type="player",
                    from_year=top1_player.get("from_year"),
                    to_year=top1_player.get("to_year"),
                    source="fuzzy_temporal",
                    confidence=top1_score
                )

        # ---------------------------------------------------------
        # Tier 3: LLM Disambiguator (Nebius Nemotron)
        # ---------------------------------------------------------
        if llm_client:
            try:
                prompt = f"""You are an NBA entity resolver. Disambiguate the following basketball moniker or player reference:
Mention: "{surface_form}"
Season/Era Hint: {season_hint or 'Modern Era'}

Return JSON:
{{
  "full_name": "<Full official player name>",
  "player_id": <int or null>,
  "confidence": <float 0.0-1.0>
}}
"""
                resp_text = llm_client.generate(prompt, temperature=0.1)
                import json
                m_json = re.search(r"\{.*\}", resp_text, re.DOTALL)
                if m_json:
                    data = json.loads(m_json.group(0))
                    full_name = data.get("full_name")
                    p_id = data.get("player_id")
                    conf = float(data.get("confidence", 0.8))
                    
                    if not p_id and full_name:
                        matched = players.find_players_by_full_name(full_name)
                        if matched:
                            p_id = matched[0]["id"]
                            full_name = matched[0]["full_name"]
                            
                    if p_id and conf >= 0.75:
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute("""
                            INSERT OR REPLACE INTO disambiguation_cache 
                            (surface_form, decade_or_season_bucket, resolved_player_id, resolved_name, source, confidence)
                            VALUES (?, ?, ?, ?, 'llm_inferred', ?);
                            """, (norm, season_hint or "current", p_id, full_name, conf))
                            conn.commit()

                        return ResolvedEntity(
                            name=full_name,
                            id=p_id,
                            type="player",
                            source="llm_inferred",
                            confidence=conf
                        )
            except Exception:
                pass

        return None
