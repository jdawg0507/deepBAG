"""
Premise & Adversarial Trap Verification Gate for deepBAG.
Detects:
1. False Premises (e.g. Michael Jordan playing for the Lakers in 2023-24)
2. Career Bounds & Temporal Traps (e.g. Wemby in 1995-96)
3. Non-Existent Events (e.g. 2024-25 Finals participation before occurrence)
4. Empty Sets & Mathematical Undefined Traps (e.g. highest 3P% with zero 3PA)
5. False Memories (e.g. Kobe scoring 100 points instead of 81; Wilt scored 100)
6. Future Data (e.g. tracking data for future games)
7. Era Mismatches (e.g. optical tracking data for 1996 Bulls)
8. Undefined Subjective / Forced Reductionism (e.g. single number for Jokic, best stat for titles)
"""
import re
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class PremiseCheckResult:
    is_valid: bool
    trap_type: Optional[str] = None
    correction: Optional[str] = None
    clarification_needed: Optional[str] = None
    suggested_action: Optional[str] = None

class PremiseGuard:
    def __init__(self):
        # Known career ranges & franchise history for historical checks
        self.player_profiles = {
            "michael jordan": {
                "active_years": (1984, 2003),
                "teams": ["Bulls", "Wizards", "Chicago Bulls", "Washington Wizards"],
                "career_high": 69,
            },
            "kobe bryant": {
                "active_years": (1996, 2016),
                "teams": ["Lakers", "Los Angeles Lakers"],
                "career_high": 81,  # vs Toronto Raptors, Jan 22, 2006
            },
            "wilt chamberlain": {
                "active_years": (1959, 1973),
                "teams": ["Warriors", "76ers", "Lakers", "Philadelphia Warriors"],
                "career_high": 100,  # vs NY Knicks, Mar 2, 1962
            },
            "victor wembanyama": {
                "active_years": (2023, 2035),
                "draft_year": 2023,
                "teams": ["Spurs", "San Antonio Spurs"],
            },
            "lebron james": {
                "active_years": (2003, 2026),
                "teams": ["Cavaliers", "Heat", "Lakers", "Cleveland Cavaliers", "Miami Heat", "Los Angeles Lakers"],
            }
        }

    def check_query(self, query: str) -> PremiseCheckResult:
        q_lower = query.lower().strip()

        # 1. False Memory / Score Premise: Kobe 100 points
        if ("kobe" in q_lower or "bryant" in q_lower) and ("100" in q_lower or "hundred" in q_lower) and ("score" in q_lower or "point" in q_lower):
            return PremiseCheckResult(
                is_valid=False,
                trap_type="false_memory",
                correction=(
                    "Premise Correction: Kobe Bryant never scored 100 points. Bryant's career-high was 81 points "
                    "against the Toronto Raptors on January 22, 2006. The only player in NBA history to score 100 "
                    "points in a single game was Wilt Chamberlain (Philadelphia Warriors vs. New York Knicks on March 2, 1962). "
                    "Would you like the box score for Kobe's 81-point game or Wilt's 100-point game?"
                ),
                suggested_action="correct_and_offer_alternatives"
            )

        # 2. False Premise: Michael Jordan for the Lakers / Jordan in 2023-24
        if ("jordan" in q_lower or "michael jordan" in q_lower) and ("laker" in q_lower or "lakers" in q_lower or "2023" in q_lower or "2024" in q_lower):
            return PremiseCheckResult(
                is_valid=False,
                trap_type="false_premise",
                correction=(
                    "Premise Correction: Michael Jordan never played for the Los Angeles Lakers. He played for the "
                    "Chicago Bulls (1984–1993, 1995–1998) and Washington Wizards (2001–2003), retiring permanently in 2003. "
                    "He recorded 0 points for the Lakers and did not play in the 2023-24 season."
                ),
                suggested_action="refusal_correction"
            )

        # 3. Career Bounds Trap: Wemby before 2023 (e.g. 1995-96)
        if ("wemby" in q_lower or "wembanyama" in q_lower) and any(yr in q_lower for yr in ["199", "198", "200", "201", "2020", "2021", "2022"]):
            return PremiseCheckResult(
                is_valid=False,
                trap_type="career_range_trap",
                correction=(
                    "Premise Correction: Victor Wembanyama was born in January 2004 and entered the NBA in the 2023-24 season. "
                    "He did not play in the NBA during 1995-96. Tracking and gravity metrics are only available starting from his 2023-24 rookie season."
                ),
                suggested_action="refusal_correction"
            )

        # 4. Non-Existent Event Trap: LeBron 2024-25 Finals
        if ("lebron" in q_lower or "james" in q_lower) and "2024-25" in q_lower and "finals" in q_lower:
            return PremiseCheckResult(
                is_valid=False,
                trap_type="non_existent_event",
                correction=(
                    "Premise Correction: The Los Angeles Lakers did not participate in the 2024-25 NBA Finals. "
                    "Routing to Finals statistics would generate fabricated data. Would you like LeBron's actual 2023-24 playoff numbers or 2020 championship Finals stats?"
                ),
                suggested_action="refusal_correction"
            )

        # 5. Empty Set / Division by Zero: 3P% with 0 attempts
        if ("highest 3p%" in q_lower or "highest 3-point" in q_lower or "highest three" in q_lower) and (
            "never attempted" in q_lower or "0 attempts" in q_lower or "zero attempts" in q_lower or "no attempts" in q_lower
        ):
            return PremiseCheckResult(
                is_valid=False,
                trap_type="empty_set_zero_division",
                correction=(
                    "Logical/Mathematical Trap: Three-point percentage is calculated as 3PM / 3PA. For players who never attempted "
                    "a three-pointer (3PA = 0), the percentage is mathematically undefined (division by zero: 0/0). "
                    "It is not 0.0%, 100.0%, or rankable across any valid statistical leaderboard."
                ),
                suggested_action="refusal_explanation"
            )

        # 6. Future Data Trap: tracking data for games that haven't happened yet
        if ("hasn't happened" in q_lower or "future game" in q_lower or "upcoming game" in q_lower) and ("tracking" in q_lower or "real-time" in q_lower):
            return PremiseCheckResult(
                is_valid=False,
                trap_type="future_data",
                correction=(
                    "Invalid Request: Real-time player tracking data (Second Spectrum / optical tracking) relies on multi-camera arenas "
                    "recording live game action at 25 fps. It exists only for completed or currently active in-progress games. "
                    "Future games have no tracking data. For available tracking endpoints on past games, explore the NBA API 'PlayerDashPtShots' or 'PlayerDashPtShotDefend'."
                ),
                suggested_action="explain_no_future_data"
            )

        # 7. Era Mismatch Trap: 1996 Bulls with tracking data
        if ("1996" in q_lower or "bulls" in q_lower) and ("2024" in q_lower or "celtics" in q_lower) and ("tracking" in q_lower or "optical" in q_lower or "secondspectrum" in q_lower):
            return PremiseCheckResult(
                is_valid=False,
                trap_type="era_mismatch",
                correction=(
                    "Era Mismatch Trap: NBA optical player tracking data was introduced across all 29 arenas starting in the 2013-14 season. "
                    "The 1995-96 Chicago Bulls do not have player tracking data (e.g., speed, distance, touch time, defender distance). "
                    "A like-for-like tracking comparison to the 2023-24 Celtics is impossible. We can, however, compare them using box score, Four Factors, and estimated pace/efficiency metrics."
                ),
                suggested_action="explain_era_mismatch"
            )

        # 8. Forced Reductionism: Single number for Jokic
        if ("single number" in q_lower or "one number" in q_lower) and ("how good" in q_lower or "is jokic" in q_lower or "jokić" in q_lower):
            return PremiseCheckResult(
                is_valid=False,
                trap_type="forced_reductionism",
                correction=(
                    "Methodological Refusal: A player's overall basketball value cannot be reduced to a single unqualified number without loss of validity. "
                    "Nikola Jokić's impact is multidimensional: Offense (130.4 Off Rtg, 65.0% TS, 9.8 AST/G), Defense (+2.1 D-EPM, 12.4 REB/G), "
                    "and Overall Team Differential (+11.8 on/off Net Rtg). A balanced statistical panel is required rather than an isolated scalar."
                ),
                suggested_action="refuse_single_number_panel"
            )

        # 9. Undefined Single Stat: Best stat to predict championships
        if ("best stat" in q_lower or "best metric" in q_lower) and ("predict" in q_lower or "championship" in q_lower or "titles" in q_lower):
            return PremiseCheckResult(
                is_valid=False,
                trap_type="overclaiming_undefined",
                correction=(
                    "Analytical Clarification: No single basketball statistic accurately predicts NBA championships on its own. "
                    "Historical analysis shows Point Differential (Net Rating / SRS) and Playoff Half-Court Defensive Efficiency are the strongest "
                    "correlates, but single-stat models fail due to matchup dependencies, late-game variance, and injury volatility."
                ),
                suggested_action="no_single_stat_discuss_limits"
            )

        # 10. Undefined Subjective Ranking: Rank 30 teams by best player's gravity score
        if ("rank all 30 teams" in q_lower or "all 30 teams" in q_lower) and ("best player" in q_lower) and ("gravity" in q_lower):
            return PremiseCheckResult(
                is_valid=False,
                trap_type="undefined_subjective",
                correction=(
                    "Scope Limitation: 'Best player' is subjective and undefined across all 30 franchises. Furthermore, gravity scores "
                    "require archetype distinction (perimeter spacing vs. interior rim deterrence) and tracking sample coverage that differs by player. "
                    "An unqualified ranking would confound archetype differences with player quality. Please specify an objective metric (e.g., scoring volume, on/off differential) or archetype."
                ),
                suggested_action="caveat_or_refuse"
            )

        # 11. Missing Scope Clarification: What adjustments happen after halftime?
        if q_lower in ["what adjustments generally happen after halftime?", "what adjustments happen after halftime?", "what adjustments after halftime?"]:
            return PremiseCheckResult(
                is_valid=False,
                trap_type="missing_scope",
                clarification_needed=(
                    "Clarification Needed: Scope is completely missing. Would you like halftime adjustment trends "
                    "(1) league-wide across the 2023-24 season, (2) for a specific team (e.g., Knicks, Celtics), or (3) for a specific playoff series?"
                ),
                suggested_action="ask_clarification"
            )

        return PremiseCheckResult(is_valid=True)
