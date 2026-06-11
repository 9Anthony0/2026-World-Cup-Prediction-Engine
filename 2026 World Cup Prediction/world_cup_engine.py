"""
2026 FIFA World Cup Prediction Engine
======================================
A Monte Carlo simulation engine that predicts tournament outcomes using
Poisson-distributed goal scoring derived from World Football Elo ratings.

Tournament Format:
    - 48 teams in 12 groups of 4 (Groups A–L)
    - Top 2 from each group + 8 best third-place teams advance (32 total)
    - Single-elimination knockout: R32 → R16 → QF → SF → Final

Architecture:
    Modular function design for downstream Streamlit UI integration.
    All simulation state flows through function arguments — no hidden globals.

Usage:
    >>> from world_cup_engine import run_monte_carlo
    >>> df = run_monte_carlo(iterations=10_000, seed=42)
    >>> print(df.head(10))
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple, Set, Optional, Any
import time

# Lazy import to avoid circular dependency — loaded on first use
_TEAM_PROFILES = None
_DEFAULT_WEIGHTS = None

def _get_profiles():
    global _TEAM_PROFILES, _DEFAULT_WEIGHTS
    if _TEAM_PROFILES is None:
        from team_data import TEAM_PROFILES, DEFAULT_WEIGHTS
        _TEAM_PROFILES = TEAM_PROFILES
        _DEFAULT_WEIGHTS = DEFAULT_WEIGHTS
    return _TEAM_PROFILES, _DEFAULT_WEIGHTS


# ============================================================================
# 1. CONFIGURATION & TOURNAMENT DATA
# ============================================================================

# --- Elo Ratings ---
# World Football Elo ratings as of June 5, 2026 (eloratings.net).
# These are the pre-tournament ratings for all 48 qualified nations.
# Source: eloratings.net, footballratings.org, international-football.net

ELO_RATINGS: Dict[str, int] = {
    # Tier 1 — Elite (2000+)
    "Spain":         2155,
    "Argentina":     2113,
    "France":        2062,
    "England":       2020,
    "Brazil":        1988,
    "Portugal":      2000,
    # Tier 2 — Strong (1900–1999)
    "Colombia":      1977,
    "Netherlands":   1944,
    "Ecuador":       1933,
    "Germany":       1925,
    "Croatia":       1908,
    "Japan":         1906,
    "Turkiye":       1906,
    "Uruguay":       1892,
    "Belgium":       1888,
    "Senegal":       1879,
    # Tier 3 — Competitive (1800–1899)
    "Mexico":        1859,
    "Switzerland":   1845,
    "Paraguay":      1833,
    "Morocco":       1822,
    "Austria":       1805,
    "Canada":        1800,
    "Norway":        1795,
    # Tier 4 — Solid (1700–1799)
    "Algeria":       1743,
    "South Korea":   1752,
    "Scotland":      1735,
    "USA":           1721,
    "Iran":          1710,
    "Australia":     1700,
    "Egypt":         1689,
    "Ghana":         1680,
    "Ivory Coast":   1676,
    # Tier 5 — Qualifiers (1500–1699)
    "Saudi Arabia":  1592,
    "Tunisia":       1618,
    "South Africa":  1525,
    "Bosnia and Herzegovina": 1594,
    "Sweden":        1510,
    "Czech Republic":1506,
    "Panama":        1545,
    "Uzbekistan":    1461,
    "Qatar":         1520,
    "DR Congo":      1480,
    "Iraq":          1607,
    "Jordan":        1590,
    "New Zealand":   1485,
    "Cape Verde":    1466,
    "Haiti":         1391,
    "Curacao":       1394,
}

# Note: Italy (lost UEFA Path A final to Bosnia) and Denmark (lost UEFA Path D
# final to Czech Republic) did NOT qualify for the 2026 World Cup.

# --- Group Assignments ---
# Official 2026 FIFA World Cup draw (December 5, 2025, Washington D.C.)
# Playoff winners confirmed March 31, 2026:
#   UEFA Path A: Bosnia and Herzegovina | Path B: Sweden
#   UEFA Path C: Turkiye              | Path D: Czech Republic
#   Intercontinental 1: DR Congo      | Intercontinental 2: Iraq

GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico",      "South Africa",   "South Korea",    "Czech Republic"],
    "B": ["Canada",      "Bosnia and Herzegovina", "Qatar",  "Switzerland"],
    "C": ["Brazil",      "Morocco",        "Haiti",          "Scotland"],
    "D": ["USA",         "Paraguay",       "Australia",      "Turkiye"],
    "E": ["Germany",     "Curacao",        "Ivory Coast",    "Ecuador"],
    "F": ["Netherlands", "Japan",          "Sweden",         "Tunisia"],
    "G": ["Belgium",     "Egypt",          "Iran",           "New Zealand"],
    "H": ["Spain",       "Cape Verde",     "Saudi Arabia",   "Uruguay"],
    "I": ["France",      "Senegal",        "Iraq",           "Norway"],
    "J": ["Argentina",   "Algeria",        "Austria",        "Jordan"],
    "K": ["Portugal",    "DR Congo",       "Uzbekistan",     "Colombia"],
    "L": ["England",     "Croatia",        "Ghana",          "Panama"],
}

# --- Match Simulation Parameters ---
BASE_GOALS_PER_TEAM: float = 1.35      # Historical World Cup average goals per team per match
ELO_EXPONENT: float = 1.0              # Controls how aggressively Elo gaps map to goal expectancy
PENALTY_DAMPING: float = 0.25          # Dampens Elo advantage in penalty shootouts (penalties are more random)

# --- Knockout Bracket ---
# Defines the Round of 32 matchups for the 48-team format.
#
# Format: Each entry is ((position, id), (position, id))
#   position "1" = group winner,  id = group letter   → e.g., ("1", "A") = Group A winner
#   position "2" = group runner-up                     → e.g., ("2", "D") = Group D runner-up
#   position "3" = ranked third-place team, id = rank  → e.g., ("3", 0)  = best 3rd-place team
#
# The bracket tree works by adjacency:
#   Matches 0–1 winners meet in R16, matches 2–3 winners meet in R16, etc.
#   R16 pairs feed into QF, QF pairs feed into SF, SF winners play the Final.
#
#   LEFT HALF                                    RIGHT HALF
#   ┌─ M0:  1A vs T8 ─┐                         ┌─ M8:  1C vs T4 ─┐
#   │                  ├─ R16 ─┐                 │                  ├─ R16 ─┐
#   └─ M1:  2D vs 2E ─┘       │                 └─ M9:  2A vs 2B ─┘       │
#                              ├─ QF ─┐                                    ├─ QF ─┐
#   ┌─ M2:  1D vs T5 ─┐       │      │         ┌─ M10: 1F vs T1 ─┐       │      │
#   │                  ├─ R16 ─┘      │         │                  ├─ R16 ─┘      │
#   └─ M3:  1B vs 2I ─┘              │         └─ M11: 1H vs 2C ─┘              │
#                                     ├─ SF                                      ├─ SF
#   ┌─ M4:  1G vs T7 ─┐              │         ┌─ M12: 1I vs T3 ─┐              │
#   │                  ├─ R16 ─┐      │         │                  ├─ R16 ─┐      │
#   └─ M5:  2J vs 2K ─┘       │      │         └─ M13: 2G vs 2H ─┘       │      │
#                              ├─ QF ─┘                                    ├─ QF ─┘
#   ┌─ M6:  1J vs T6 ─┐       │                ┌─ M14: 1L vs T2 ─┐       │
#   │                  ├─ R16 ─┘                │                  ├─ R16 ─┘
#   └─ M7:  1E vs 2L ─┘                        └─ M15: 1K vs 2F ─┘
#
#                         FINAL: Left SF winner vs Right SF winner

R32_MATCHUPS: List[Tuple[Tuple[str, Any], Tuple[str, Any]]] = [
    # ---- LEFT HALF ----
    # Section 1 → feeds into QF-0
    (("1", "A"), ("3", 7)),    # Match 0:  1A vs 8th-best 3rd place
    (("2", "D"), ("2", "E")),  # Match 1:  2D vs 2E
    (("1", "D"), ("3", 4)),    # Match 2:  1D vs 5th-best 3rd place
    (("1", "B"), ("2", "I")),  # Match 3:  1B vs 2I

    # Section 2 → feeds into QF-1
    (("1", "G"), ("3", 6)),    # Match 4:  1G vs 7th-best 3rd place
    (("2", "J"), ("2", "K")),  # Match 5:  2J vs 2K
    (("1", "J"), ("3", 5)),    # Match 6:  1J vs 6th-best 3rd place
    (("1", "E"), ("2", "L")),  # Match 7:  1E vs 2L

    # ---- RIGHT HALF ----
    # Section 3 → feeds into QF-2
    (("1", "C"), ("3", 3)),    # Match 8:  1C vs 4th-best 3rd place
    (("2", "A"), ("2", "B")),  # Match 9:  2A vs 2B
    (("1", "F"), ("3", 0)),    # Match 10: 1F vs best 3rd place
    (("1", "H"), ("2", "C")),  # Match 11: 1H vs 2C

    # Section 4 → feeds into QF-3
    (("1", "I"), ("3", 2)),    # Match 12: 1I vs 3rd-best 3rd place
    (("2", "G"), ("2", "H")),  # Match 13: 2G vs 2H
    (("1", "L"), ("3", 1)),    # Match 14: 1L vs 2nd-best 3rd place
    (("1", "K"), ("2", "F")),  # Match 15: 1K vs 2F
]


# ============================================================================
# 2. CORE MATCH SIMULATION (Poisson Engine)
# ============================================================================

def calculate_expected_goals(
    elo_a: int,
    elo_b: int,
    base_rate: float = BASE_GOALS_PER_TEAM,
    exponent: float = ELO_EXPONENT,
) -> Tuple[float, float]:
    """
    Derive Poisson lambda (expected goals) for each team from Elo ratings.

    The model treats a team's Elo as a proxy for combined attacking and
    defensive quality. The ratio of ratings determines the goal expectancy:

        λ_A = base_rate × (Elo_A / Elo_B) ^ exponent
        λ_B = base_rate × (Elo_B / Elo_A) ^ exponent

    When two teams have equal Elo, both get λ = base_rate (≈ 1.35 goals).
    A higher exponent amplifies the advantage of the stronger team.

    Args:
        elo_a:     Elo rating of team A.
        elo_b:     Elo rating of team B.
        base_rate: Average goals per team per match (World Cup baseline ≈ 1.35).
        exponent:  Controls sensitivity to Elo gaps (1.0 = linear, >1 = amplified).

    Returns:
        (lambda_a, lambda_b) — Poisson expected goals for each team.
    """
    ratio = elo_a / elo_b
    lambda_a = base_rate * (ratio ** exponent)
    lambda_b = base_rate * ((1.0 / ratio) ** exponent)
    return lambda_a, lambda_b


def calculate_composite_strength(
    team_name: str,
    profiles: Dict[str, Dict],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a multi-factor composite strength score for a team.

    Combines 8 quantitative indicators into a single value using
    configurable weights. Each factor is normalized to a 0–1 scale
    before weighting.

    Factors:
        elo, star_player_rating, squad_depth, tournament_pedigree,
        recent_form, offensive_rating, defensive_rating,
        manager_experience, host_advantage

    Args:
        team_name: Name of the team.
        profiles:  Team profiles dict (from team_data.py).
        weights:   Optional custom weight dict. Uses DEFAULT_WEIGHTS if None.

    Returns:
        Composite strength score (typically in range 0.3–1.1).
    """
    if weights is None:
        _, weights = _get_profiles()
    p = profiles[team_name]

    # Klement Econometric Factors normalizations (log-scaled / climate distance)
    norm_gdp = np.log(max(p.get("gdp_pc", 1000.0), 10.0)) / np.log(90000.0)
    norm_pop = np.log(max(p.get("population", 1.0), 0.01)) / np.log(350.0)
    norm_temp = np.exp(-((p.get("avg_temp", 14.0) - 14.0) ** 2) / 100.0)

    # Ensure variables stay bounded in 0-1
    norm_gdp = min(max(norm_gdp, 0.0), 1.0)
    norm_pop = min(max(norm_pop, 0.0), 1.0)

    strength = (
        weights["elo"]                 * (p["elo"] / 2000.0)           +
        weights["star_player_rating"]   * (p["star_player_rating"] / 100.0) +
        weights["squad_depth"]          * (p["squad_depth"] / 100.0)    +
        weights["tournament_pedigree"]  * (p["tournament_pedigree"] / 100.0) +
        weights["recent_form"]          * p["recent_form"]              +
        weights["offensive_rating"]     * (p["offensive_rating"] / 100.0) +
        weights["defensive_rating"]     * (p["defensive_rating"] / 100.0) +
        weights["manager_experience"]   * (p["manager_experience"] / 100.0) +
        weights.get("gdp_pc", 0.0)     * norm_gdp                      +
        weights.get("population", 0.0) * norm_pop                      +
        weights.get("avg_temp", 0.0)   * norm_temp                     +
        weights["host_advantage"]       * (1.0 + p["host_advantage"])   # 1.0 base + bonus
    )
    return strength


def calculate_expected_goals_composite(
    team_a: str,
    team_b: str,
    profiles: Dict[str, Dict],
    weights: Optional[Dict[str, float]] = None,
    base_rate: float = BASE_GOALS_PER_TEAM,
) -> Tuple[float, float]:
    """
    Enhanced expected goals using multi-factor composite strength.

    Uses the composite strength ratio as the primary driver, then applies
    secondary offensive/defensive adjustments for asymmetric attack/defense.
    Also applies a dynamic multiplier to reward recent form and penalize poor form.

    Args:
        team_a:    Name of team A.
        team_b:    Name of team B.
        profiles:  Team profiles dict.
        weights:   Optional custom weight dict.
        base_rate: Average goals per team per match.

    Returns:
        (lambda_a, lambda_b) — Poisson expected goals for each team.
    """
    str_a = calculate_composite_strength(team_a, profiles, weights)
    str_b = calculate_composite_strength(team_b, profiles, weights)

    # Offensive/defensive asymmetry adjustment
    # Scale so that average team (rating ~75) gets a 1.0 multiplier
    off_a = profiles[team_a]["offensive_rating"] / 75.0
    def_b = 75.0 / max(profiles[team_b]["defensive_rating"], 30.0)
    off_b = profiles[team_b]["offensive_rating"] / 75.0
    def_a = 75.0 / max(profiles[team_a]["defensive_rating"], 30.0)

    # Blend: 70% composite ratio, 30% offense/defense adjustment
    ratio_a = str_a / str_b
    ratio_b = str_b / str_a
    adjust_a = (off_a * def_b) ** 0.5  # geometric mean
    adjust_b = (off_b * def_a) ** 0.5

    lambda_a = base_rate * (0.7 * ratio_a + 0.3 * adjust_a)
    lambda_b = base_rate * (0.7 * ratio_b + 0.3 * adjust_b)

    # --- Recent Form Amplification ---
    form_a = profiles[team_a].get("recent_form", 0.5)
    form_b = profiles[team_b].get("recent_form", 0.5)
    form_diff = form_a - form_b
    
    # A 0.2 form difference shifts goal expectations by 10%
    form_mult_a = 1.0 + (form_diff * 0.5)
    form_mult_b = 1.0 - (form_diff * 0.5)
    
    lambda_a = max(lambda_a * form_mult_a, 0.1)
    lambda_b = max(lambda_b * form_mult_b, 0.1)

    return lambda_a, lambda_b


def simulate_match(
    team1: str,
    team2: str,
    elo_ratings: Dict[str, int],
    allow_draw: bool = True,
    rng: Optional[np.random.Generator] = None,
    profiles: Optional[Dict[str, Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """
    Simulate a single match using Poisson-distributed goal scoring or expected goals.

    When `profiles` are provided, uses the multi-factor composite strength
    model. Otherwise, falls back to the original Elo-only model.

    Each team's goal count is independently drawn from a Poisson distribution.
    In knockout matches (allow_draw=False), drawn results are resolved via a
    penalty shootout probability mechanic.

    Args:
        team1:       Name of team 1.
        team2:       Name of team 2.
        elo_ratings: Dictionary mapping team names -> Elo ratings.
        allow_draw:  If False (knockout), resolves draws via penalty probability.
        rng:         Numpy random generator for reproducibility.
        profiles:    Optional team profiles for multi-factor model.
        weights:     Optional custom factor weights.
        deterministic: If True, uses expected values instead of random variables.

    Returns:
        dict with keys:
            'team1':     Name of team 1
            'team2':     Name of team 2
            'goals1':    Goals scored by team 1 (int)
            'goals2':    Goals scored by team 2 (int)
            'winner':    Winning team name, or None for a draw
            'penalties': True if the match was decided by penalties
    """
    if rng is None:
        rng = np.random.default_rng()

    elo1 = elo_ratings[team1]
    elo2 = elo_ratings[team2]

    # Calculate expected goals: multi-factor if profiles provided, else Elo-only
    if profiles is not None:
        lambda1, lambda2 = calculate_expected_goals_composite(
            team1, team2, profiles, weights
        )
    else:
        lambda1, lambda2 = calculate_expected_goals(elo1, elo2)

    # Draw goal counts from Poisson distributions (or use expected goals if deterministic)
    if deterministic:
        goals1 = int(round(lambda1))
        goals2 = int(round(lambda2))
    else:
        goals1 = int(rng.poisson(lambda1))
        goals2 = int(rng.poisson(lambda2))

    # Build result dict
    result: Dict[str, Any] = {
        "team1": team1,
        "team2": team2,
        "goals1": goals1,
        "goals2": goals2,
        "winner": None,
        "penalties": False,
    }

    if goals1 > goals2:
        result["winner"] = team1
    elif goals2 > goals1:
        result["winner"] = team2
    elif not allow_draw:
        # --- Penalty Shootout Mechanic ---
        if profiles is not None:
            # Scale composite strength diff to an ELO-like range
            str1 = calculate_composite_strength(team1, profiles, weights)
            str2 = calculate_composite_strength(team2, profiles, weights)
            if deterministic:
                # Deterministically, the higher composite strength wins
                result["winner"] = team1 if str1 >= str2 else team2
                result["penalties"] = True
                return result
            # A 0.2 difference in composite strength maps to a 200 Elo difference
            str_diff = (str1 - str2) * 1000.0
            base_prob = 1.0 / (1.0 + 10.0 ** (-str_diff / 400.0))
        else:
            if deterministic:
                result["winner"] = team1 if elo1 >= elo2 else team2
                result["penalties"] = True
                return result
            elo_diff = elo1 - elo2
            base_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        pen_prob = 0.5 + (base_prob - 0.5) * PENALTY_DAMPING

        result["penalties"] = True
        result["winner"] = team1 if rng.random() < pen_prob else team2

    return result


# ============================================================================
# 3. GROUP STAGE SIMULATION
# ============================================================================

def simulate_group(
    group_teams: List[str],
    elo_ratings: Dict[str, int],
    rng: Optional[np.random.Generator] = None,
    profiles: Optional[Dict[str, Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
    deterministic: bool = False,
) -> List[Dict[str, Any]]:
    """
    Simulate all round-robin matches within a single group.

    Each group of 4 teams plays C(4,2) = 6 matches. Standings are computed
    using FIFA World Cup rules:
        - 3 points for a win, 1 for a draw, 0 for a loss
        - Tiebreakers: Goal Difference -> Goals Scored -> random (lots)

    Args:
        group_teams: List of 4 team names in this group.
        elo_ratings: Dictionary mapping team names -> Elo ratings.
        rng:         Numpy random generator.
        profiles:    Optional team profiles for multi-factor model.
        weights:     Optional custom factor weights.
        deterministic: If True, resolves matches and tiebreakers deterministically.

    Returns:
        Sorted list of standing dicts (1st -> 4th), each containing:
            team, points, gd (goal difference), gs (goals scored), ga (goals allowed)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize standings
    standings = {
        team: {"team": team, "points": 0, "gd": 0, "gs": 0, "ga": 0}
        for team in group_teams
    }

    # Round-robin: every pair plays exactly once
    for team_a, team_b in combinations(group_teams, 2):
        result = simulate_match(team_a, team_b, elo_ratings, allow_draw=True, rng=rng,
                                profiles=profiles, weights=weights, deterministic=deterministic)
        ga, gb = result["goals1"], result["goals2"]

        # Update team A
        standings[team_a]["gs"] += ga
        standings[team_a]["ga"] += gb
        standings[team_a]["gd"] += ga - gb

        # Update team B
        standings[team_b]["gs"] += gb
        standings[team_b]["ga"] += ga
        standings[team_b]["gd"] += gb - ga

        # Award points
        if ga > gb:
            standings[team_a]["points"] += 3
        elif gb > ga:
            standings[team_b]["points"] += 3
        else:
            standings[team_a]["points"] += 1
            standings[team_b]["points"] += 1

    # Sort by: points (desc) → goal difference (desc) → goals scored (desc)
    # A tiny random jitter breaks any remaining ties (simulates "drawing of lots"), or Elo if deterministic
    sorted_standings = sorted(
        standings.values(),
        key=lambda x: (x["points"], x["gd"], x["gs"], elo_ratings[x["team"]] if deterministic else rng.random() * 0.001),
        reverse=True,
    )

    return sorted_standings


def simulate_group_stage(
    groups: Dict[str, List[str]],
    elo_ratings: Dict[str, int],
    rng: Optional[np.random.Generator] = None,
    profiles: Optional[Dict[str, Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """
    Simulate the full group stage across all 12 groups.

    Processes each group independently, extracts 1st/2nd/3rd place teams,
    and prepares the third-place data for the advancement bottleneck.

    Args:
        groups:      Dictionary mapping group letter -> list of 4 team names.
        elo_ratings: Dictionary mapping team names -> Elo ratings.
        rng:         Numpy random generator.
        profiles:    Optional team profiles for multi-factor model.
        weights:     Optional custom factor weights.
        deterministic: If True, simulates deterministically.

    Returns:
        dict with keys:
            'group_results': {group_letter: sorted standings list}
            'first_place':   {group_letter: team name}
            'second_place':  {group_letter: team name}
            'third_place':   list of 12 dicts (one per group) with
                             team, points, gd, gs, ga, group
    """
    if rng is None:
        rng = np.random.default_rng()

    group_results: Dict[str, List[Dict]] = {}
    first_place: Dict[str, str] = {}
    second_place: Dict[str, str] = {}
    third_place_teams: List[Dict[str, Any]] = []

    for group_letter in sorted(groups.keys()):
        teams = groups[group_letter]
        standings = simulate_group(teams, elo_ratings, rng=rng,
                                   profiles=profiles, weights=weights, deterministic=deterministic)

        group_results[group_letter] = standings
        first_place[group_letter] = standings[0]["team"]
        second_place[group_letter] = standings[1]["team"]

        # Capture third-place record with group origin (needed for bracket assignment)
        third = standings[2].copy()
        third["group"] = group_letter
        third_place_teams.append(third)

    return {
        "group_results": group_results,
        "first_place": first_place,
        "second_place": second_place,
        "third_place": third_place_teams,
    }


# ============================================================================
# 4. THIRD-PLACE RESOLUTION (The Bottleneck)
# ============================================================================

def resolve_third_place_teams(
    third_place_teams: List[Dict[str, Any]],
    rng: Optional[np.random.Generator] = None,
    deterministic: bool = False,
    elo_ratings: Optional[Dict[str, int]] = None,
) -> List[str]:
    """
    Determine the 8 best third-place teams from the 12 groups.

    This is the critical bottleneck of the 48-team format. Third-place finishers
    from different groups must be cross-compared despite facing different opponents.

    FIFA ranking criteria (applied in order):
        1. Points earned (descending)
        2. Goal difference (descending)
        3. Goals scored (descending)
        4. Drawing of lots — simulated as a random tiebreaker

    In the actual tournament, the identities of the qualifying third-place groups
    also determine bracket slot assignments via a FIFA lookup table. This engine
    simplifies by assigning bracket positions by overall third-place rank.

    Args:
        third_place_teams: List of 12 dicts, each with:
                           team, points, gd, gs, ga, group
        rng:               Numpy random generator (for tiebreaking).
        deterministic:     If True, resolves ties deterministically using Elo.
        elo_ratings:       Elo ratings for tiebreaking.

    Returns:
        List of 8 team names (best third-place teams), ordered best → worst.
    """
    if rng is None:
        rng = np.random.default_rng()
    if elo_ratings is None:
        elo_ratings = ELO_RATINGS

    # Sort by FIFA tiebreaker cascade + random lot/Elo for absolute ties
    sorted_thirds = sorted(
        third_place_teams,
        key=lambda x: (x["points"], x["gd"], x["gs"], elo_ratings[x["team"]] if deterministic else rng.random() * 0.001),
        reverse=True,
    )

    # Top 8 advance to the knockout stage
    advancing = sorted_thirds[:8]
    return [team["team"] for team in advancing]


# ============================================================================
# 5. KNOCKOUT STAGE SIMULATION
# ============================================================================

def _resolve_bracket_team(
    source: Tuple[str, Any],
    first_place: Dict[str, str],
    second_place: Dict[str, str],
    third_place_ranked: List[str],
) -> str:
    """
    Map a bracket source descriptor to an actual team name.

    Args:
        source:              ("1", "A") | ("2", "B") | ("3", 0)
        first_place:         {group_letter: team_name}
        second_place:        {group_letter: team_name}
        third_place_ranked:  [team_name, ...] ordered best → worst

    Returns:
        Team name string.
    """
    position, identifier = source
    if position == "1":
        return first_place[identifier]
    elif position == "2":
        return second_place[identifier]
    elif position == "3":
        return third_place_ranked[identifier]
    else:
        raise ValueError(f"Unknown bracket position type: {position}")


def simulate_knockouts(
    first_place: Dict[str, str],
    second_place: Dict[str, str],
    third_place_ranked: List[str],
    elo_ratings: Dict[str, int],
    rng: Optional[np.random.Generator] = None,
    profiles: Optional[Dict[str, Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Simulate the entire knockout stage: R32 -> R16 -> QF -> SF -> Final.

    Uses the predefined R32_MATCHUPS bracket structure. The bracket tree
    advances by adjacency -- winners of matches (2n, 2n+1) play each other
    in the next round.

    All knockout matches that end in a draw after 90 minutes are resolved
    via the penalty shootout probability mechanic.

    Args:
        first_place:         {group_letter: team name}   -- 12 group winners.
        second_place:        {group_letter: team name}   -- 12 runners-up.
        third_place_ranked:  List of 8 team names        -- ranked 3rd-place qualifiers.
        elo_ratings:         {team_name: elo_rating}
        rng:                 Numpy random generator.
        profiles:            Optional team profiles for multi-factor model.
        weights:             Optional custom factor weights.

    Returns:
        dict with keys:
            'r32_winners':    list of 16 team names (survived R32 -> play R16)
            'qf_teams':       list of 8 team names  (survived R16 -> play QF)
            'sf_teams':       list of 4 team names  (survived QF  -> play SF)
            'final_teams':    list of 2 team names  (survived SF  -> play Final)
            'champion':       winning team name
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Helper closures ---
    def play_round(matchups: List[Tuple[str, str]]) -> List[str]:
        """Simulate all matches in a knockout round, return list of winners."""
        return [
            simulate_match(t1, t2, elo_ratings, allow_draw=False, rng=rng,
                           profiles=profiles, weights=weights)["winner"]
            for t1, t2 in matchups
        ]

    def pair_adjacent(teams: List[str]) -> List[Tuple[str, str]]:
        """Pair adjacent teams for the next round: (0 vs 1), (2 vs 3), ..."""
        return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]

    # --- Resolve R32 matchups from bracket descriptors to team names ---
    r32_matchups = [
        (
            _resolve_bracket_team(src1, first_place, second_place, third_place_ranked),
            _resolve_bracket_team(src2, first_place, second_place, third_place_ranked),
        )
        for src1, src2 in R32_MATCHUPS
    ]

    # --- Play each round ---
    r32_winners = play_round(r32_matchups)           # 16 winners → R16
    qf_teams    = play_round(pair_adjacent(r32_winners))  # 8 winners → QF
    sf_teams    = play_round(pair_adjacent(qf_teams))     # 4 winners → SF
    final_teams = play_round(pair_adjacent(sf_teams))     # 2 winners → Final

    # Final match
    final_result = simulate_match(
        final_teams[0], final_teams[1], elo_ratings, allow_draw=False, rng=rng,
        profiles=profiles, weights=weights
    )

    return {
        "r32_winners": r32_winners,
        "qf_teams": qf_teams,
        "sf_teams": sf_teams,
        "final_teams": final_teams,
        "champion": final_result["winner"],
    }


# ============================================================================
# 6. FULL TOURNAMENT WRAPPER
# ============================================================================

def simulate_tournament(
    groups: Dict[str, List[str]],
    elo_ratings: Dict[str, int],
    rng: Optional[np.random.Generator] = None,
    profiles: Optional[Dict[str, Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Simulate one complete FIFA World Cup tournament (group + knockout).

    This is the single-iteration unit that the Monte Carlo engine calls
    repeatedly. It chains together the group stage, third-place resolution,
    and knockout bracket, then returns a lightweight dict of which teams
    reached each milestone.

    Args:
        groups:      {group_letter: [team1, team2, team3, team4]}
        elo_ratings: {team_name: elo_rating}
        rng:         Numpy random generator.
        profiles:    Optional team profiles for multi-factor model.
        weights:     Optional custom factor weights.

    Returns:
        dict with keys:
            'group_advance': set of 32 team names that advanced from groups
            'qf':            set of 8 team names that reached the quarterfinals
            'sf':            set of 4 team names that reached the semifinals
            'final':         set of 2 team names that reached the final
            'champion':      winning team name
    """
    if rng is None:
        rng = np.random.default_rng()

    # Phase 1: Group stage
    gs = simulate_group_stage(groups, elo_ratings, rng=rng,
                              profiles=profiles, weights=weights)

    # Phase 2: Resolve the 8 best third-place teams
    third_place_ranked = resolve_third_place_teams(gs["third_place"], rng=rng)

    # Phase 3: Build the set of all 32 advancing teams
    advancing: Set[str] = set()
    advancing.update(gs["first_place"].values())    # 12 group winners
    advancing.update(gs["second_place"].values())   # 12 runners-up
    advancing.update(third_place_ranked)             # 8 best third-place

    # Phase 4: Knockout stage
    ko = simulate_knockouts(
        gs["first_place"], gs["second_place"],
        third_place_ranked, elo_ratings, rng=rng,
        profiles=profiles, weights=weights,
    )

    return {
        "group_advance": advancing,                 # 32 teams
        "qf": set(ko["qf_teams"]),                  # 8 teams reached QF
        "sf": set(ko["sf_teams"]),                   # 4 teams reached SF
        "final": set(ko["final_teams"]),             # 2 teams reached Final
        "champion": ko["champion"],                  # 1 team
    }


def simulate_single_bracket(
    groups: Optional[Dict[str, List[str]]] = None,
    elo_ratings: Optional[Dict[str, int]] = None,
    profiles: Optional[Dict[str, Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """
    Simulate a single complete tournament and return the full bracket.

    Unlike simulate_tournament(), this returns detailed bracket data
    suitable for bracket visualization in the UI.

    Returns:
        dict with keys:
            'group_results':    {group_letter: sorted standings list}
            'r32_matchups':     list of 16 (team_a, team_b) tuples
            'r32_winners':      list of 16 team names
            'r16_matchups':     list of 8 (team_a, team_b) tuples
            'qf_teams':         list of 8 team names
            'qf_matchups':      list of 4 (team_a, team_b) tuples
            'sf_teams':         list of 4 team names
            'sf_matchups':      list of 2 (team_a, team_b) tuples
            'final_teams':      list of 2 team names
            'champion':         winning team name
    """
    if groups is None:
        groups = GROUPS
    if elo_ratings is None:
        elo_ratings = ELO_RATINGS
    if profiles is None:
        profiles, _ = _get_profiles()

    rng = np.random.default_rng(seed)

    # Group stage
    gs = simulate_group_stage(groups, elo_ratings, rng=rng,
                              profiles=profiles, weights=weights, deterministic=deterministic)
    third_place_ranked = resolve_third_place_teams(gs["third_place"], rng=rng,
                                                   deterministic=deterministic, elo_ratings=elo_ratings)

    # Build R32 matchups as actual team names
    r32_matchups = [
        (
            _resolve_bracket_team(src1, gs["first_place"], gs["second_place"], third_place_ranked),
            _resolve_bracket_team(src2, gs["first_place"], gs["second_place"], third_place_ranked),
        )
        for src1, src2 in R32_MATCHUPS
    ]

    def pair_adjacent(teams):
        return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]

    def play_round_detailed(matchups):
        results = []
        for t1, t2 in matchups:
            r = simulate_match(t1, t2, elo_ratings, allow_draw=False, rng=rng,
                               profiles=profiles, weights=weights, deterministic=deterministic)
            results.append(r)
        return results

    r32_results = play_round_detailed(r32_matchups)
    r32_winners = [r["winner"] for r in r32_results]

    r16_matchups = pair_adjacent(r32_winners)
    r16_results = play_round_detailed(r16_matchups)
    qf_teams = [r["winner"] for r in r16_results]

    qf_matchups = pair_adjacent(qf_teams)
    qf_results = play_round_detailed(qf_matchups)
    sf_teams = [r["winner"] for r in qf_results]

    sf_matchups = pair_adjacent(sf_teams)
    sf_results = play_round_detailed(sf_matchups)
    final_teams = [r["winner"] for r in sf_results]

    final_result = simulate_match(
        final_teams[0], final_teams[1], elo_ratings, allow_draw=False, rng=rng,
        profiles=profiles, weights=weights, deterministic=deterministic
    )

    return {
        "group_results": gs["group_results"],
        "r32_matchups": r32_matchups,
        "r32_results": r32_results,
        "r32_winners": r32_winners,
        "r16_matchups": r16_matchups,
        "r16_results": r16_results,
        "qf_teams": qf_teams,
        "qf_matchups": qf_matchups,
        "qf_results": qf_results,
        "sf_teams": sf_teams,
        "sf_matchups": sf_matchups,
        "sf_results": sf_results,
        "final_teams": final_teams,
        "final_result": final_result,
        "champion": final_result["winner"],
    }


# ============================================================================
# 7. MONTE CARLO ENGINE
# ============================================================================

def run_monte_carlo(
    iterations: int = 1_000,
    groups: Optional[Dict[str, List[str]]] = None,
    elo_ratings: Optional[Dict[str, int]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    profiles: Optional[Dict[str, Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Run the full Monte Carlo simulation of the 2026 FIFA World Cup.

    Simulates the entire tournament `iterations` times, aggregates results,
    and returns a Pandas DataFrame showing the probability (%) that each team
    reaches each stage:
        - Advance from Group Stage
        - Reach the Quarterfinals
        - Reach the Semifinals
        - Reach the Final
        - Win the World Cup

    Performance:
        ~1,000 iterations completes in seconds on modern hardware.
        ~10,000 iterations produces stable probabilities for top contenders.
        ~100,000 iterations yields publication-grade precision (< 0.1% variance).

    Args:
        iterations:  Number of full tournament simulations to run.
        groups:      Group assignments (defaults to GROUPS).
        elo_ratings: Elo rating dictionary (defaults to ELO_RATINGS).
        seed:        Random seed for reproducibility (None = non-deterministic).
        verbose:     If True, prints progress and timing information.
        profiles:    Optional team profiles for multi-factor model.
        weights:     Optional custom factor weights.

    Returns:
        pd.DataFrame sorted by "Win World Cup %" descending, with columns:
            Team, Group, Elo, Group Stage %, Quarterfinals %,
            Semifinals %, Final %, Win World Cup %
    """
    if groups is None:
        groups = GROUPS
    if elo_ratings is None:
        elo_ratings = ELO_RATINGS
    if profiles is None:
        profiles, _ = _get_profiles()

    rng = np.random.default_rng(seed)

    # Build team → group lookup for the output DataFrame
    team_to_group: Dict[str, str] = {}
    all_teams: List[str] = []
    for group_letter, team_list in sorted(groups.items()):
        for team in team_list:
            all_teams.append(team)
            team_to_group[team] = group_letter

    # Initialize advancement counters
    counters = {
        team: {
            "group_advance": 0,
            "qf": 0,
            "sf": 0,
            "final": 0,
            "champion": 0,
        }
        for team in all_teams
    }

    # --- Main simulation loop ---
    start_time = time.perf_counter()
    milestone = max(1, iterations // 10)  # Print progress every 10%

    for i in range(iterations):
        result = simulate_tournament(groups, elo_ratings, rng=rng,
                                      profiles=profiles, weights=weights)

        # Tally each team's advancement
        for team in result["group_advance"]:
            counters[team]["group_advance"] += 1
        for team in result["qf"]:
            counters[team]["qf"] += 1
        for team in result["sf"]:
            counters[team]["sf"] += 1
        for team in result["final"]:
            counters[team]["final"] += 1
        counters[result["champion"]]["champion"] += 1

        # Progress reporting
        if verbose and (i + 1) % milestone == 0:
            pct = (i + 1) / iterations * 100
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{pct:5.0f}%] complete  ({i + 1:,} / {iterations:,})  "
                  f"[{rate:,.0f} sims/sec]")

    elapsed = time.perf_counter() - start_time

    if verbose:
        print(f"\n  [DONE] Completed {iterations:,} simulations in {elapsed:.2f}s "
              f"({iterations / elapsed:,.0f} sims/sec)\n")

    # --- Build results DataFrame ---
    rows = []
    for team in all_teams:
        c = counters[team]
        rows.append({
            "Team":             team,
            "Group":            team_to_group[team],
            "Elo":              elo_ratings[team],
            "Group Stage %":    round(c["group_advance"] / iterations * 100, 1),
            "Quarterfinals %":  round(c["qf"]            / iterations * 100, 1),
            "Semifinals %":     round(c["sf"]            / iterations * 100, 1),
            "Final %":          round(c["final"]         / iterations * 100, 1),
            "Win World Cup %":  round(c["champion"]      / iterations * 100, 1),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Win World Cup %", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-indexed ranking
    df.index.name = "Rank"

    return df


# ============================================================================
# 8. STANDALONE EXECUTION
# ============================================================================

def print_results(df: pd.DataFrame, top_n: int = 20) -> None:
    """Pretty-print the top N teams from a Monte Carlo results DataFrame."""
    print("=" * 90)
    print("   2026 FIFA WORLD CUP -- MONTE CARLO PREDICTION ENGINE")
    print("=" * 90)
    print()

    # Header
    print(f"{'Rank':<6}{'Team':<20}{'Grp':<5}{'Elo':<6}"
          f"{'Grp %':>7}{'QF %':>7}{'SF %':>7}{'Final %':>9}{'Win %':>8}")
    print("-" * 90)

    # Rows
    for rank, row in df.head(top_n).iterrows():
        print(f"{rank:<6}{row['Team']:<20}{row['Group']:<5}{row['Elo']:<6}"
              f"{row['Group Stage %']:>7.1f}{row['Quarterfinals %']:>7.1f}"
              f"{row['Semifinals %']:>7.1f}{row['Final %']:>9.1f}"
              f"{row['Win World Cup %']:>8.1f}")

    print("-" * 90)

    # Quick sanity checks
    total_group = df["Group Stage %"].sum()
    total_win = df["Win World Cup %"].sum()
    print(f"\n  Sanity check:")
    print(f"     Sum of Group Stage %:  {total_group:>7.1f}  (expected ~ {32 / 48 * 100 * 48:.0f})")
    print(f"     Sum of Win World Cup %: {total_win:>7.1f}  (expected ~ 100.0)")
    print()


if __name__ == "__main__":
    print()
    print("  Launching 2026 FIFA World Cup Prediction Engine...")
    print(f"  Model: Poisson goals | Elo-derived lambda | Base rate: {BASE_GOALS_PER_TEAM}")
    print(f"  Simulations: 10,000 | Seed: 42 (reproducible)")
    print()

    # Run with a fixed seed for reproducible demo output
    results_df = run_monte_carlo(iterations=10_000, seed=42, verbose=True)

    # Display top 20 teams
    print_results(results_df, top_n=20)

    # Show the full table for all 48 teams
    print("\n  Full 48-team probability table:\n")
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(results_df.to_string())
    print()
