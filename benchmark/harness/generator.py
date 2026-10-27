"""
Held-Out Benchmark Dataset Generator for deepBAG vs Generic MCP.
Strictly disjunct from evals/dev_seed.json:
- Different random seed (20261026)
- Different player population and historical seasons
- Includes 5 explicit false-premise traps
- Records ground truth: intended endpoint, entity IDs, season, expected metric
"""

import os
import json
import random
from typing import Dict, Any, List

HELD_OUT_PILOT_BLUEPRINTS = [
    # Category 1: Single-Entity Filtered (4 blueprints)
    {
        "id": "pilot_01",
        "category": "single_filtered",
        "player": "Tyrese Haliburton",
        "player_id": 1630169, # Note: Haliburton is 1630169
        "season": "2023-24",
        "metric": "AST",
        "filter": "In-Season Tournament",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "Tyrese Haliburton assists during the 2023-24 In-Season Tournament",
        "ground_truth": {"metric": "AST", "filter": "Tournament", "is_trap": False}
    },
    {
        "id": "pilot_02",
        "category": "single_filtered",
        "player": "Jalen Brunson",
        "player_id": 1628973,
        "season": "2023-24",
        "metric": "PTS",
        "filter": "Playoffs",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "Jalen Brunson playoff points per game in 2023-24",
        "ground_truth": {"metric": "PTS", "filter": "Playoffs", "is_trap": False}
    },
    {
        "id": "pilot_03",
        "category": "single_filtered",
        "player": "Bam Adebayo",
        "player_id": 1628389,
        "season": "2022-23",
        "metric": "REB",
        "filter": "vs Celtics",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "Bam Adebayo rebounds against the Boston Celtics in 2022-23",
        "ground_truth": {"metric": "REB", "filter": "vs Celtics", "is_trap": False}
    },
    {
        "id": "pilot_04",
        "category": "single_filtered",
        "player": "De'Aaron Fox",
        "player_id": 1628368,
        "season": "2023-24",
        "metric": "STL",
        "filter": "Clutch",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "De'Aaron Fox clutch steals in 2023-24",
        "ground_truth": {"metric": "STL", "filter": "Clutch", "is_trap": False}
    },

    # Category 2: Multi-Player Comparison (4 blueprints)
    {
        "id": "pilot_05",
        "category": "multi_comparison",
        "players": ["Domantas Sabonis", "Anthony Davis"],
        "player_ids": [1627734, 203076],
        "season": "2023-24",
        "metric": "REB",
        "expected_endpoint": "Multiple",
        "canonical": "Compare Domantas Sabonis and Anthony Davis rebounds in 2023-24",
        "ground_truth": {"metric": "REB", "is_trap": False}
    },
    {
        "id": "pilot_06",
        "category": "multi_comparison",
        "players": ["Trae Young", "Dejounte Murray"],
        "player_ids": [1629027, 1627749],
        "season": "2022-23",
        "metric": "AST",
        "expected_endpoint": "Multiple",
        "canonical": "Compare Trae Young and Dejounte Murray assists per game in 2022-23",
        "ground_truth": {"metric": "AST", "is_trap": False}
    },
    {
        "id": "pilot_07",
        "category": "multi_comparison",
        "players": ["Kyrie Irving", "Luka Doncic"],
        "player_ids": [202681, 1629029],
        "season": "2023-24",
        "metric": "3P%",
        "expected_endpoint": "Multiple",
        "canonical": "Kyrie Irving vs Luka Doncic 3P% in 2023-24",
        "ground_truth": {"metric": "3P%", "is_trap": False}
    },
    {
        "id": "pilot_08",
        "category": "multi_comparison",
        "players": ["Paolo Banchero", "Franz Wagner"],
        "player_ids": [1631094, 1630532],
        "season": "2023-24",
        "metric": "PTS",
        "expected_endpoint": "Multiple",
        "canonical": "Paolo Banchero vs Franz Wagner scoring in 2023-24",
        "ground_truth": {"metric": "PTS", "is_trap": False}
    },

    # Category 3: Relative Temporal (4 blueprints)
    {
        "id": "pilot_09",
        "category": "relative_temporal",
        "player": "Paolo Banchero",
        "player_id": 1631094,
        "relative": "rookie year",
        "expected_season": "2022-23",
        "metric": "PTS",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "Paolo Banchero points in his rookie year",
        "ground_truth": {"expected_season": "2022-23", "is_trap": False}
    },
    {
        "id": "pilot_10",
        "category": "relative_temporal",
        "player": "Chet Holmgren",
        "player_id": 1631096,
        "relative": "rookie season",
        "expected_season": "2023-24",
        "metric": "BLK",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "Chet Holmgren blocks in his rookie season",
        "ground_truth": {"expected_season": "2023-24", "is_trap": False}
    },
    {
        "id": "pilot_11",
        "category": "relative_temporal",
        "player": "Scottie Barnes",
        "player_id": 1630567,
        "relative": "rookie year",
        "expected_season": "2021-22",
        "metric": "REB",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "Scottie Barnes rebounds in his rookie year",
        "ground_truth": {"expected_season": "2021-22", "is_trap": False}
    },
    {
        "id": "pilot_12",
        "category": "relative_temporal",
        "player": "Tyrese Maxey",
        "player_id": 1630178,
        "relative": "rookie year",
        "expected_season": "2020-21",
        "metric": "PTS",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "Tyrese Maxey scoring in his rookie year",
        "ground_truth": {"expected_season": "2020-21", "is_trap": False}
    },

    # Category 4: Historical Era & Tracking Boundary (4 blueprints)
    {
        "id": "pilot_13",
        "category": "historical_era",
        "player": "Tim Duncan",
        "player_id": 1495,
        "season": "2002-03",
        "metric": "BLK",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "Tim Duncan playoff blocks in 2002-03",
        "ground_truth": {"season": "2002-03", "is_trap": False}
    },
    {
        "id": "pilot_14",
        "category": "historical_era",
        "player": "Patrick Ewing",
        "player_id": 335,
        "season": "1989-90",
        "metric": "PTS",
        "expected_endpoint": "historical_sql",
        "canonical": "Patrick Ewing points per game in 1989-90",
        "ground_truth": {"season": "1989-90", "is_trap": False}
    },
    {
        "id": "pilot_15",
        "category": "historical_era",
        "player": "Isiah Thomas",
        "player_id": 78318,
        "season": "1984-85",
        "metric": "AST",
        "expected_endpoint": "historical_sql",
        "canonical": "Isiah Thomas assists per game in 1984-85",
        "ground_truth": {"season": "1984-85", "is_trap": False}
    },
    {
        "id": "pilot_16",
        "category": "historical_era",
        "player": "Dirk Nowitzki",
        "player_id": 1717,
        "season": "2010-11",
        "metric": "PTS",
        "expected_endpoint": "PlayerGameLog",
        "canonical": "Dirk Nowitzki Finals scoring in 2010-11",
        "ground_truth": {"season": "2010-11", "is_trap": False}
    },

    # Category 5: False-Premise Traps (4 blueprints - TRAPS)
    {
        "id": "pilot_17",
        "category": "trap_premise",
        "player": "LeBron James",
        "player_id": 2544,
        "season": "2011-12",
        "fake_team": "Boston Celtics",
        "expected_endpoint": "premise_correction",
        "canonical": "How many points did LeBron James score for the Boston Celtics in 2011-12?",
        "ground_truth": {"is_trap": True, "trap_type": "wrong_team", "explanation": "LeBron James played for the Miami Heat in 2011-12, never for the Boston Celtics."}
    },
    {
        "id": "pilot_18",
        "category": "trap_premise",
        "player": "Stephen Curry",
        "player_id": 201939,
        "season": "1998-99",
        "expected_endpoint": "premise_correction",
        "canonical": "What were Stephen Curry's stats in the 1998-99 NBA season?",
        "ground_truth": {"is_trap": True, "trap_type": "pre_career", "explanation": "Stephen Curry was born in 1988 and entered the NBA in 2009-10; he did not play in 1998-99."}
    },
    {
        "id": "pilot_19",
        "category": "trap_premise",
        "player": "Michael Jordan",
        "player_id": 893,
        "season": "2015-16",
        "expected_endpoint": "premise_correction",
        "canonical": "Michael Jordan three-pointers made in the 2015-16 season",
        "ground_truth": {"is_trap": True, "trap_type": "post_career", "explanation": "Michael Jordan retired permanently after the 2002-03 season and did not play in 2015-16."}
    },
    {
        "id": "pilot_20",
        "category": "trap_premise",
        "player": "Victor Wembanyama",
        "player_id": 1641705,
        "season": "2021-22",
        "expected_endpoint": "premise_correction",
        "canonical": "Victor Wembanyama playoff stats in 2021-22",
        "ground_truth": {"is_trap": True, "trap_type": "pre_career", "explanation": "Victor Wembanyama was drafted in 2023 and played his rookie NBA season in 2023-24."}
    }
]

def generate_pilot_paraphrases(bp: Dict[str, Any]) -> List[str]:
    cat = bp["category"]
    if cat == "single_filtered":
        p, s, f, m = bp["player"], bp["season"], bp["filter"], bp["metric"]
        return [
            f"What were {p}'s {m} numbers during the {s} season ({f})?",
            f"Show me {p} {f} {m} stats back in {s}",
            f"{p} {s} {f} {m}",
            f"How did {p} perform in {s} when looking at {f} for {m}?"
        ]
    elif cat == "multi_comparison":
        p1, p2 = bp["players"]
        s, m = bp["season"], bp["metric"]
        return [
            f"Please provide a comparison of {p1} and {p2} in terms of {m} for {s}.",
            f"Who had better {m} in {s}, {p1} or {p2}?",
            f"{p1} vs {p2} {m} {s}",
            f"Compare {m} numbers between {p1} and {p2} throughout {s}."
        ]
    elif cat == "relative_temporal":
        p, rel, m = bp["player"], bp["relative"], bp["metric"]
        return [
            f"What were {p}'s {m} averages during his {rel}?",
            f"How many {m} did {p} put up as a rookie?",
            f"{p} {rel} {m} stats",
            f"Can you pull up {p}'s {m} production in his {rel} year?"
        ]
    elif cat == "historical_era":
        p, s, m = bp["player"], bp["season"], bp["metric"]
        return [
            f"What were {p}'s {m} stats in {s}?",
            f"Show me {p} {s} {m} numbers",
            f"{p} {s} {m} breakdown",
            f"How many {m} did {p} average in {s}?"
        ]
    elif cat == "trap_premise":
        p, s = bp["player"], bp["season"]
        if bp["ground_truth"]["trap_type"] == "wrong_team":
            team = bp["fake_team"]
            return [
                f"How many points did {p} score for the {team} in {s}?",
                f"What were {p}'s {s} stats while playing on the {team}?",
                f"{p} {team} stats {s}",
                f"Can you give me {p}'s {s} scoring breakdown with the {team}?"
            ]
        elif bp["ground_truth"]["trap_type"] == "pre_career":
            return [
                f"What were {p}'s stats in the {s} NBA season?",
                f"How many points did {p} average back in {s}?",
                f"{p} {s} regular season numbers",
                f"Pull up {p}'s NBA box score logs for {s}"
            ]
        elif bp["ground_truth"]["trap_type"] == "post_career":
            return [
                f"{p} three-pointers made in the {s} season",
                f"How many 3-pointers did {p} hit during {s}?",
                f"{p} {s} three point shooting stats",
                f"Can you find {p}'s three point percentage in {s}?"
            ]
    return [bp["canonical"]] * 4

def generate_pilot_dataset(output_path: str = "benchmark/harness/held_out_pilot_80.json"):
    random.seed(20261026)
    dataset = []
    
    for bp in HELD_OUT_PILOT_BLUEPRINTS:
        paraphrases = generate_pilot_paraphrases(bp)
        dataset.append({
            "id": bp["id"],
            "category": bp["category"],
            "canonical_query": bp["canonical"],
            "expected_endpoint": bp["expected_endpoint"],
            "expected_season": bp.get("season") or bp.get("expected_season"),
            "entities": bp.get("players") or ([bp["player"]] if "player" in bp else []),
            "ground_truth": bp.get("ground_truth", {}),
            "paraphrases": paraphrases
        })
        
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Generated held-out pilot dataset with {len(dataset)} blueprints ({len(dataset) * 4} queries) at {output_path}")

if __name__ == "__main__":
    generate_pilot_dataset()
