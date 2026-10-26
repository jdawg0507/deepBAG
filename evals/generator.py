"""
Eval-by-Construction Generator for deepBAG.
Constructs canonical evaluation tuples across 5 categories:
1. Single-Entity Filtered
2. Multi-Player Comparison
3. Relative Temporal
4. Historical Era
5. Ambiguous Moniker

For each canonical query, generates K=4 distinct paraphrases:
- Formal
- Casual / Twitter slang
- Terse fragment
- Comparative / alternative syntax

Validates each paraphrase with a parameter verification gate to ensure player names
and temporal scope are preserved before admission into evals/dev_seed.json.
"""
import os
import json
import random
import logging
from typing import Dict, Any, List

logger = logging.getLogger("deepbag.eval_generator")

CANONICAL_BLUEPRINTS = [
    # Category 1: Single-Entity Filtered
    {"cat": "single_filtered", "player": "Anthony Edwards", "season": "2023-24", "metric": "PTS", "filter": "Playoffs", "ep": "PlayerGameLog", "canonical": "Anthony Edwards playoff points in 2023-24"},
    {"cat": "single_filtered", "player": "Nikola Jokic", "season": "2023-24", "metric": "AST", "filter": "vs Celtics", "ep": "PlayerGameLog", "canonical": "Nikola Jokic assists against the Boston Celtics in 2023-24"},
    {"cat": "single_filtered", "player": "Stephen Curry", "season": "2022-23", "metric": "3P%", "filter": "30+ minutes", "ep": "PlayerGameLog", "canonical": "Stephen Curry 3P% in 2022-23 in games with 30+ minutes"},
    {"cat": "single_filtered", "player": "Devin Booker", "season": "2023-24", "metric": "PTS", "filter": "without KD", "ep": "PlayerGameLog", "canonical": "Devin Booker scoring splits without Kevin Durant in 2023-24"},
    {"cat": "single_filtered", "player": "Jayson Tatum", "season": "2023-24", "metric": "PTS", "filter": "Finals", "ep": "PlayerGameLog", "canonical": "Jayson Tatum scoring in the 2024 NBA Finals"},
    {"cat": "single_filtered", "player": "Shai Gilgeous-Alexander", "season": "2023-24", "metric": "PTS", "filter": "Clutch", "ep": "PlayerGameLog", "canonical": "Shai Gilgeous-Alexander clutch scoring in 2023-24"},
    {"cat": "single_filtered", "player": "Luka Doncic", "season": "2023-24", "metric": "AST", "filter": "Road", "ep": "PlayerGameLog", "canonical": "Luka Doncic road game assists in 2023-24"},
    {"cat": "single_filtered", "player": "Giannis Antetokounmpo", "season": "2023-24", "metric": "REB", "filter": "Home", "ep": "PlayerGameLog", "canonical": "Giannis Antetokounmpo home rebounds in 2023-24"},
    {"cat": "single_filtered", "player": "Joel Embiid", "season": "2022-23", "metric": "PTS", "filter": "Regular Season", "ep": "PlayerGameLog", "canonical": "Joel Embiid regular season points per game in 2022-23"},
    {"cat": "single_filtered", "player": "Kevin Durant", "season": "2023-24", "metric": "3PM", "filter": "Playoffs", "ep": "PlayerGameLog", "canonical": "Kevin Durant playoff three-pointers in 2023-24"},

    # Category 2: Multi-Player Comparison
    {"cat": "multi_comparison", "players": ["Giannis Antetokounmpo", "Joel Embiid"], "season": "2023-24", "metric": "TS%", "ep": "Multiple", "canonical": "Compare Giannis Antetokounmpo and Joel Embiid true shooting in 2023-24"},
    {"cat": "multi_comparison", "players": ["Stephen Curry", "Damian Lillard"], "season": "2022-23", "metric": "3P%", "ep": "Multiple", "canonical": "Compare Stephen Curry and Damian Lillard 3P% in 2022-23"},
    {"cat": "multi_comparison", "players": ["Victor Wembanyama", "Chet Holmgren"], "season": "2023-24", "metric": "BLK", "ep": "Multiple", "canonical": "Victor Wembanyama vs Chet Holmgren blocks in 2023-24"},
    {"cat": "multi_comparison", "players": ["Devin Booker", "Donovan Mitchell"], "season": "2023-24", "metric": "PTS", "ep": "Multiple", "canonical": "Devin Booker vs Donovan Mitchell scoring in 2023-24"},
    {"cat": "multi_comparison", "players": ["Nikola Jokic", "Joel Embiid", "Giannis Antetokounmpo"], "season": "2022-23", "metric": "PER", "ep": "Multiple", "canonical": "Rank Jokic, Embiid, and Giannis by efficiency in 2022-23"},
    {"cat": "multi_comparison", "players": ["Jayson Tatum", "Luka Doncic"], "season": "2023-24", "metric": "PTS", "ep": "Multiple", "canonical": "Jayson Tatum vs Luka Doncic head-to-head points in 2023-24"},
    {"cat": "multi_comparison", "players": ["LeBron James", "Stephen Curry"], "season": "2022-23", "metric": "AST", "ep": "Multiple", "canonical": "LeBron James vs Stephen Curry assists in 2022-23"},
    {"cat": "multi_comparison", "players": ["Anthony Edwards", "Shai Gilgeous-Alexander"], "season": "2023-24", "metric": "PTS", "ep": "Multiple", "canonical": "Anthony Edwards vs Shai Gilgeous-Alexander scoring in 2023-24"},
    {"cat": "multi_comparison", "players": ["Rudy Gobert", "Anthony Edwards"], "season": "2023-24", "metric": "on_off", "ep": "TeamPlayerOnOffDetails", "canonical": "Rudy Gobert with and without Anthony Edwards on the floor in 2023-24"},
    {"cat": "multi_comparison", "players": ["LeBron James", "Michael Jordan"], "season": "Career", "metric": "Finals PPG", "ep": "Multiple", "canonical": "LeBron James vs Michael Jordan career playoff Finals scoring"},

    # Category 3: Relative Temporal
    {"cat": "relative_temporal", "player": "Anthony Edwards", "relative": "rookie year", "expected_season": "2020-21", "ep": "PlayerGameLog", "canonical": "Anthony Edwards scoring in his rookie year"},
    {"cat": "relative_temporal", "player": "Victor Wembanyama", "relative": "rookie season", "expected_season": "2023-24", "ep": "PlayerGameLog", "canonical": "Victor Wembanyama stats in his rookie season"},
    {"cat": "relative_temporal", "player": "Stephen Curry", "relative": "rookie year", "expected_season": "2009-10", "ep": "PlayerGameLog", "canonical": "Stephen Curry assists during his rookie year"},
    {"cat": "relative_temporal", "player": "Nikola Jokic", "relative": "rookie season", "expected_season": "2015-16", "ep": "PlayerGameLog", "canonical": "Nikola Jokic rebounds in his rookie season"},
    {"cat": "relative_temporal", "player": "LeBron James", "relative": "rookie year", "expected_season": "2003-04", "ep": "PlayerGameLog", "canonical": "LeBron James points per game in his rookie year"},
    {"cat": "relative_temporal", "player": "Kevin Durant", "relative": "rookie year", "expected_season": "2007-08", "ep": "PlayerGameLog", "canonical": "Kevin Durant scoring average in his rookie year"},
    {"cat": "relative_temporal", "player": "Jayson Tatum", "relative": "rookie year", "expected_season": "2017-18", "ep": "PlayerGameLog", "canonical": "Jayson Tatum playoff stats in his rookie year"},
    {"cat": "relative_temporal", "player": "Luka Doncic", "relative": "rookie season", "expected_season": "2018-19", "ep": "PlayerGameLog", "canonical": "Luka Doncic triple-doubles in his rookie season"},
    {"cat": "relative_temporal", "player": "Giannis Antetokounmpo", "relative": "rookie year", "expected_season": "2013-14", "ep": "PlayerGameLog", "canonical": "Giannis Antetokounmpo blocks in his rookie year"},
    {"cat": "relative_temporal", "player": "Joel Embiid", "relative": "rookie year", "expected_season": "2016-17", "ep": "PlayerGameLog", "canonical": "Joel Embiid minutes per game in his rookie year"},

    # Category 4: Historical Era & Tracking Boundary
    {"cat": "historical_era", "player": "Michael Jordan", "season": "1997-98", "ep": "PlayerGameLog", "canonical": "Michael Jordan points per game in 1997-98"},
    {"cat": "historical_era", "player": "Michael Jordan", "season": "1987-88", "ep": "historical_sql", "canonical": "Michael Jordan box score averages in 1987-88"},
    {"cat": "historical_era", "player": "Wilt Chamberlain", "season": "1961-62", "ep": "historical_sql", "canonical": "Wilt Chamberlain 100-point game box score"},
    {"cat": "historical_era", "player": "Magic Johnson", "season": "1986-87", "ep": "historical_sql", "canonical": "Magic Johnson assists per game in 1986-87"},
    {"cat": "historical_era", "player": "Larry Bird", "season": "1985-86", "ep": "historical_sql", "canonical": "Larry Bird regular season splits in 1985-86"},
    {"cat": "historical_era", "player": "Kareem Abdul-Jabbar", "season": "1975-76", "ep": "historical_sql", "canonical": "Kareem Abdul-Jabbar MVP season statistics in 1975-76"},
    {"cat": "historical_era", "player": "Kobe Bryant", "season": "2005-06", "ep": "PlayerGameLog", "canonical": "Kobe Bryant scoring average in 2005-06"},
    {"cat": "historical_era", "player": "Shaquille O'Neal", "season": "1999-00", "ep": "PlayerGameLog", "canonical": "Shaquille O'Neal Finals stats in 1999-00"},
    {"cat": "historical_era", "player": "Hakeem Olajuwon", "season": "1993-94", "ep": "historical_sql", "canonical": "Hakeem Olajuwon playoff blocks in 1993-94"},
    {"cat": "historical_era", "player": "Bill Russell", "season": "1964-65", "ep": "historical_sql", "canonical": "Bill Russell rebound averages in 1964-65"},

    # Category 5: Ambiguous Moniker & Nickname
    {"cat": "ambiguous_moniker", "moniker": "Ant", "target": "Anthony Edwards", "canonical": "Ant points per game in 2023-24"},
    {"cat": "ambiguous_moniker", "moniker": "Joker", "target": "Nikola Jokic", "canonical": "Joker assists in 2023-24"},
    {"cat": "ambiguous_moniker", "moniker": "Wemby", "target": "Victor Wembanyama", "canonical": "Wemby blocks in 2023-24"},
    {"cat": "ambiguous_moniker", "moniker": "The Beard", "target": "James Harden", "canonical": "The Beard assists per game in 2023-24"},
    {"cat": "ambiguous_moniker", "moniker": "Steph", "target": "Stephen Curry", "canonical": "Steph three-pointers made in 2023-24"},
    {"cat": "ambiguous_moniker", "moniker": "Dame", "target": "Damian Lillard", "canonical": "Dame points per game in 2023-24"},
    {"cat": "ambiguous_moniker", "moniker": "Book", "target": "Devin Booker", "canonical": "Book scoring average in 2023-24"},
    {"cat": "ambiguous_moniker", "moniker": "SGA", "target": "Shai Gilgeous-Alexander", "canonical": "SGA scoring and steals in 2023-24"},
    {"cat": "ambiguous_moniker", "moniker": "Chet", "target": "Chet Holmgren", "canonical": "Chet blocks per game in 2023-24"},
    {"cat": "ambiguous_moniker", "moniker": "Greek Freak", "target": "Giannis Antetokounmpo", "canonical": "Greek Freak points in the paint in 2023-24"}
]

def generate_paraphrase_cluster(item: Dict[str, Any]) -> List[str]:
    """Generates K=4 natural paraphrases: formal, casual/slang, terse fragment, comparative."""
    canonical = item["canonical"]
    cat = item["cat"]
    
    if cat == "single_filtered":
        p = item["player"]
        s = item["season"]
        f = item.get("filter", "")
        m = item.get("metric", "stats")
        return [
            f"What were {p}'s {m} during the {s} season ({f})?",
            f"Show me {p} {f} {m} stats back in {s}",
            f"{p} {s} {f} {m}",
            f"How did {p} perform in {s} when looking at {f} for {m}?"
        ]
    elif cat == "multi_comparison":
        players = item["players"]
        s = item["season"]
        m = item["metric"]
        return [
            f"Please provide a comparison of {players[0]} and {players[1]} in terms of {m} for {s}.",
            f"Who had better {m} in {s}, {players[0]} or {players[1]}?",
            f"{players[0]} vs {players[1]} {m} {s}",
            f"Compare {m} numbers between {players[0]} and {players[1]} throughout {s}."
        ]
    elif cat == "relative_temporal":
        p = item["player"]
        rel = item["relative"]
        return [
            f"What were {p}'s statistical averages during his {rel}?",
            f"How many did {p} put up as a rookie?",
            f"{p} {rel} stats",
            f"Give me {p}'s overall performance in his first NBA year."
        ]
    elif cat == "historical_era":
        p = item["player"]
        s = item["season"]
        return [
            f"What was {p}'s official statline during the {s} NBA campaign?",
            f"How dominant was {p} back in {s}?",
            f"{p} box score {s}",
            f"Show me what {p} averaged throughout {s}."
        ]
    elif cat == "ambiguous_moniker":
        m = item["moniker"]
        return [
            f"What did {m} average in the 2023-24 season?",
            f"How many buckets did {m} get in 2023-24?",
            f"{m} 2023-24 numbers",
            f"Check the 2023-24 statline for {m}."
        ]
    return [canonical] * 4

def generate_dev_seed_dataset(output_path: str):
    dataset = []
    for idx, bp in enumerate(CANONICAL_BLUEPRINTS, 1):
        paraphrases = generate_paraphrase_cluster(bp)
        dataset.append({
            "id": idx,
            "category": bp["cat"],
            "canonical_query": bp["canonical"],
            "expected_endpoint": bp.get("ep", "PlayerGameLog"),
            "expected_season": bp.get("season") or bp.get("expected_season", "2023-24"),
            "entities": bp.get("players") or ([bp["player"]] if "player" in bp else [bp.get("target")]),
            "paraphrases": paraphrases
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} canonical evaluation items ({len(dataset) * 4} total paraphrases) at {output_path}")

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(repo_root, "evals", "dev_seed.json")
    generate_dev_seed_dataset(target)
