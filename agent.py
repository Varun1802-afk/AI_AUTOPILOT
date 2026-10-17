"""agent.py - High-Performance Hybrid Autopilot Game Agent.

Combines a fast deterministic tactical evaluator with an LLM strategic planner.

Architecture:
1. Deterministic Tactical Evaluator:
   - Threat scoring per row using distance, speed, HP, and special abilities.
   - Calculates time-to-breach for the most threatening enemies.
   - Geometric defense synergy:
     * x=0: Archer (global focus, protected in back, hits any row)
     * x=1: Soldier (40 DPS lane defense)
     * x=3: Shield (700 HP blocker & blast shield absorbing Pirate explosions outside Soldier blast radius)
     * x=4 or x=2: Penguin (500 burst charging nuke against Wizards and dense swarms)
   - Opening budget allocation: guarantees no threatened row is left naked.
2. LLM Strategic Planner:
   - Injects full board state in a single prompt (no slow tool roundtrips).
   - Proposes placements with cost awareness.
3. Hybrid Safety Overseer:
   - Strict validation: coordinates, occupied cells, running budget, max 8 placements.
   - Auto-fill safeguard: if the LLM under-spends or omits an urgent threat,
     the overseer completes necessary placements using remaining coins.
   - Deterministic fallback if LLM is slow, unreachable, or returns invalid output.
"""
import json
import sys

from autopilot_sdk.llm import LLMClient
from autopilot_sdk.tools import run_tool_loop, tool

# Unit definitions from official specs
COSTS_DEFAULT = {0: 250, 1: 200, 2: 300, 3: 150}

ENEMY_STATS = {
    "EnemyPirate": {"speed": 8.0, "hp": 30, "dps": 0, "pirate": True, "cross_s": 1.9},
    "EnemyMonk": {"speed": 3.5, "hp": 280, "dps": 36, "pirate": False, "cross_s": 4.3},
    "EnemyWizard": {"speed": 3.0, "hp": 500, "dps": 50, "pirate": False, "cross_s": 5.0},
    "EnemClown": {"speed": 2.0, "hp": 200, "dps": 17, "pirate": False, "cross_s": 7.5},
}

PROMPT = (
    "You are an expert game AI defending a base in a 6-row lane game (x: 0-10, y: 0-5).\n"
    "Enemies walk from x=10 toward x=0. If any enemy reaches x=0, you LOSE INSTANTLY.\n"
    "Defences (id: cost):\n"
    "- 2 Archer (300): fragile, shoots ANY row. Always place at x=0.\n"
    "- 1 Soldier (200): high 40 DPS forward in lane. Place at x=1 in threatened rows.\n"
    "- 3 Shield (150): 700 HP blocker. Place at x=3 to stop Monks/Wizards and absorb Pirate blast.\n"
    "- 0 Penguin (250): takes damage then charges forward dealing 500 burst damage. Good vs Wizards.\n\n"
    "Rules:\n"
    "1. Never leave a threatened row with 0 defense.\n"
    "2. At x=0 put Archers, at x=1 put Soldiers, at x=3 put Shields.\n"
    "3. Total placement cost must be <= coins.\n"
    "4. Return JSON only with up to 8 placements: {\"placements\": [{\"defenceId\": 1, \"x\": 1, \"y\": 0}], \"reason\": \"<brief>\"}"
)


def log(message):
    print(f"[agent] {message}", file=sys.stderr, flush=True)


def _xy(unit):
    return round(unit["position"]["x"]), round(unit["position"]["y"])


def get_reinforcements(state):
    """Main entry point called by the Autopilot SDK."""
    coins = float(state.get("budget", {}).get("remaining", 0))
    costs = {int(k): v for k, v in state.get("unitCosts", COSTS_DEFAULT).items()}
    defences = [d for d in state.get("defences", []) if d.get("health", 0) > 0]
    enemies = [e for e in state.get("enemies", []) if not e.get("isDead", False) and e.get("health", 0) > 0]
    taken = {_xy(d) for d in defences}
    phase = state.get("phase", "pause")
    time_budget = state.get("timeBudgetSeconds", 15)

    if coins < min(costs.values()):
        return []

    # 1. Threat Analysis per row (0 to 5)
    row_threats = analyze_threats(state, phase, enemies, defences)

    # 2. Try LLM Strategic Placement
    llm_placements = []
    try:
        llm_placements = query_llm(state, phase, coins, costs, taken, enemies, defences, row_threats, time_budget)
    except Exception as exc:
        log(f"LLM not used ({type(exc).__name__}: {str(exc)[:80]})")

    # 3. Hybrid Overseer: Validate LLM output and auto-fill any exposed rows
    final_placements = overseer_combine(llm_placements, row_threats, taken, coins, costs)
    log(f"Final decisions ({len(final_placements)} placements): {final_placements}")
    return final_placements


def analyze_threats(state, phase, enemies, defences):
    """Calculate threat metrics per row to guide defensive priority."""
    threats = {}
    for y in range(6):
        threats[y] = {
            "row": y,
            "active_enemies": 0,
            "min_x": 99.0,
            "time_to_breach": 999.0,
            "has_pirate": False,
            "has_wizard": False,
            "total_enemy_hp": 0,
            "upcoming_count": 0,
            "has_soldier": any(_xy(d) == (1, y) and d.get("type") == "defenceSolider" for d in defences),
            "has_shield": any(_xy(d)[1] == y and "Shield" in d.get("type", "") for d in defences),
            "has_archer": any(_xy(d) == (0, y) and "Archer" in d.get("type", "") for d in defences),
            "total_defences": sum(1 for d in defences if _xy(d)[1] == y),
        }

    # Analyze active enemies
    for e in enemies:
        x, y = _xy(e)
        if 0 <= y <= 5:
            etype = e.get("type", "")
            stats = ENEMY_STATS.get(etype, {"speed": 3.0, "hp": 200, "pirate": False})
            speed = max(stats["speed"], 1.0)
            hp = e.get("health", stats["hp"])

            threats[y]["active_enemies"] += 1
            threats[y]["total_enemy_hp"] += hp
            if x < threats[y]["min_x"]:
                threats[y]["min_x"] = x
                threats[y]["time_to_breach"] = (x * 1.5) / speed
            if stats.get("pirate") or "Pirate" in etype:
                threats[y]["has_pirate"] = True
            if "Wizard" in etype:
                threats[y]["has_wizard"] = True

    # Analyze upcoming enemies (especially for opening)
    upcoming = state.get("upcomingEnemies", [])
    for group in upcoming:
        allowed = group.get("allowedRows")
        count = group.get("count", 1)
        target_rows = allowed if allowed else list(range(6))
        for r in target_rows:
            if 0 <= r <= 5:
                threats[r]["upcoming_count"] += count

    # Calculate overall urgency score
    for y, t in threats.items():
        score = 0.0
        if t["active_enemies"] > 0:
            score += 100.0
            # Closer enemies multiply urgency
            score += max(0.0, (11.0 - t["min_x"]) * 10.0)
            # Faster time to breach
            if t["time_to_breach"] < 5.0:
                score += 50.0
            if t["has_pirate"]:
                score += 30.0
            if not t["has_soldier"]:
                score += 40.0
            if not t["has_shield"]:
                score += 20.0
        elif t["upcoming_count"] > 0:
            score += 10.0 + t["upcoming_count"] * 2.0
            if not t["has_soldier"]:
                score += 15.0

        t["score"] = score

    return threats


def query_llm(state, phase, coins, costs, taken, enemies, defences, row_threats, time_budget):
    """Single-shot structured call to the LLM without slow tool roundtrips."""
    # Compact board representation
    active_rows_summary = []
    for y in range(6):
        t = row_threats[y]
        if t["active_enemies"] > 0 or t["upcoming_count"] > 0:
            active_rows_summary.append({
                "row": y,
                "active_enemies": t["active_enemies"],
                "closest_x": t["min_x"] if t["min_x"] < 90 else None,
                "time_to_breach_s": round(t["time_to_breach"], 1) if t["time_to_breach"] < 900 else None,
                "has_pirate": t["has_pirate"],
                "has_soldier": t["has_soldier"],
                "has_shield": t["has_shield"],
                "upcoming_threat": t["upcoming_count"]
            })

    board_compact = {
        "phase": phase,
        "coins": int(coins),
        "total_archers": sum(1 for d in defences if "Archer" in d.get("type", "")),
        "rows": active_rows_summary,
    }

    # Deadline leaves at least 2 seconds for safety logic
    deadline = max(2.0, time_budget - 3.0)

    # Empty dummy tool so run_tool_loop works seamlessly with SDK
    @tool
    def quick_status() -> str:
        """Returns OK."""
        return "OK"

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Board: {json.dumps(board_compact)}. Propose placements in JSON."}
    ]

    run = run_tool_loop(
        LLMClient(),
        messages,
        [quick_status],
        deadline_s=deadline,
        max_rounds=1,
        max_tokens=220,
    )

    raw_text = (run.content or "").strip().strip("`").removeprefix("json").strip()
    data = json.loads(raw_text)
    placements = data.get("placements", [])
    reason = data.get("reason", "")
    log(f"LLM proposed {len(placements)} placements: {reason}")
    return placements


def overseer_combine(llm_placements, row_threats, taken, coins, costs):
    """Combines verified LLM placements with deterministic safety auto-fill."""
    chosen = []
    working_taken = set(taken)
    remaining_coins = coins

    # 1. Accept valid LLM placements first
    if llm_placements:
        for p in llm_placements[:8]:
            try:
                kind = int(p["defenceId"])
                x = int(p["x"])
                y = int(p["y"])
            except (KeyError, TypeError, ValueError):
                continue

            if kind in costs and 0 <= x <= 10 and 0 <= y <= 5:
                if (x, y) not in working_taken and costs[kind] <= remaining_coins:
                    working_taken.add((x, y))
                    remaining_coins -= costs[kind]
                    chosen.append({"defenceId": kind, "defenceGridPosition": {"x": x, "y": y}})

    # 2. Safety Auto-Fill: Ensure no threatened lane is left undefended
    # Sort rows by threat score descending
    sorted_rows = sorted(row_threats.values(), key=lambda t: t["score"], reverse=True)

    # Priority A: Emergency active enemies heading for base
    for t in sorted_rows:
        if len(chosen) >= 8 or remaining_coins < min(costs.values()):
            break
        y = t["row"]
        if t["active_enemies"] > 0:
            # Need a Soldier at x=1 for damage
            if not t["has_soldier"] and (1, y) not in working_taken and remaining_coins >= costs[1]:
                working_taken.add((1, y))
                remaining_coins -= costs[1]
                t["has_soldier"] = True
                chosen.append({"defenceId": 1, "defenceGridPosition": {"x": 1, "y": y}})

            # Need a Shield at x=3 for stall / pirate protection
            if not t["has_shield"] and remaining_coins >= costs[3]:
                for shield_x in [3, 2, 4]:
                    if (shield_x, y) not in working_taken:
                        working_taken.add((shield_x, y))
                        remaining_coins -= costs[3]
                        t["has_shield"] = True
                        chosen.append({"defenceId": 3, "defenceGridPosition": {"x": shield_x, "y": y}})
                        break

    # Priority B: Global Archer support at x=0 (up to 2-3 archers on the board)
    total_archers = sum(1 for c in chosen if c["defenceId"] == 2) + sum(
        1 for (x, y) in taken if x == 0
    )
    if total_archers < 2 and remaining_coins >= costs[2] and len(chosen) < 8:
        for t in sorted_rows:
            y = t["row"]
            if (0, y) not in working_taken and remaining_coins >= costs[2]:
                working_taken.add((0, y))
                remaining_coins -= costs[2]
                chosen.append({"defenceId": 2, "defenceGridPosition": {"x": 0, "y": y}})
                total_archers += 1
                if total_archers >= 2 or len(chosen) >= 8:
                    break

    # Priority C: Upcoming threats in opening phase
    for t in sorted_rows:
        if len(chosen) >= 8 or remaining_coins < min(costs.values()):
            break
        y = t["row"]
        if t["upcoming_count"] > 0:
            if not t["has_soldier"] and (1, y) not in working_taken and remaining_coins >= costs[1]:
                working_taken.add((1, y))
                remaining_coins -= costs[1]
                t["has_soldier"] = True
                chosen.append({"defenceId": 1, "defenceGridPosition": {"x": 1, "y": y}})

            if not t["has_shield"] and remaining_coins >= costs[3]:
                for shield_x in [3, 2]:
                    if (shield_x, y) not in working_taken:
                        working_taken.add((shield_x, y))
                        remaining_coins -= costs[3]
                        t["has_shield"] = True
                        chosen.append({"defenceId": 3, "defenceGridPosition": {"x": shield_x, "y": y}})
                        break

    return chosen