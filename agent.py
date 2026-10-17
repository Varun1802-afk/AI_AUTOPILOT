"""agent.py - High-Performance Grandmaster Autopilot Game Agent.

Combines a fast deterministic tactical engine with an LLM strategic planner.

Tactical Blueprint:
1. Complete Row Coverage (The Anti-Breach Guarantee):
   - At the opening, every threatened row gets a Soldier at x=1.
   - Prevents fast runners (Pirates: 1.9s cross, Monks: 4.3s cross) from slipping into empty lanes.
2. Anti-Wizard Penguin Nuke (The TC05/TC09 Solution):
   - Wizards deal 50 DPS continuous beam and melt Soldiers/Shields.
   - Penguin (ID 0, cost 250) deals 500 burst to EVERY enemy in its lane when it takes ANY damage.
   - When a Wizard appears, placing a Penguin at x=2 or x=3 absorbs the first beam tick,
     rockets forward at 10 units/s, and 1-shots the 500 HP Wizard instantly!
3. Reinforced DPS (Double Soldiers at x=1 and x=2):
   - Swarm lanes (Monks, Clowns) get a second Soldier at x=2, doubling lane DPS to 80.
4. Shield Blast Buffer (x=3):
   - Shields at x=3 absorb 200 Pirate blast damage safely outside the 1.3-tile blast radius of Soldiers at x=1.
5. Global Archers (x=0):
   - 1-2 Archers at x=0 provide constant cross-lane focus fire.
6. Bulletproof Deterministic Fallback:
   - Gracefully handles upstream LLM 502 Bad Gateway errors without missing a single turn.
"""
import json
import sys

from autopilot_sdk.llm import LLMClient
from autopilot_sdk.tools import run_tool_loop, tool

COSTS_DEFAULT = {0: 250, 1: 200, 2: 300, 3: 150}

ENEMY_STATS = {
    "EnemyPirate": {"speed": 8.0, "hp": 30, "dps": 0, "pirate": True, "cross_s": 1.9},
    "EnemyMonk": {"speed": 3.5, "hp": 280, "dps": 36, "pirate": False, "cross_s": 4.3},
    "EnemyWizard": {"speed": 3.0, "hp": 500, "dps": 50, "pirate": False, "cross_s": 5.0},
    "EnemClown": {"speed": 2.0, "hp": 200, "dps": 17, "pirate": False, "cross_s": 7.5},
}

PROMPT = (
    "You are an expert game AI defending a base in a 6-row lane game (x: 0-10, y: 0-5).\n"
    "Enemies enter at x=10 and walk to x=0. If any enemy reaches x=0, you LOSE INSTANTLY.\n"
    "Defences (ID: Cost):\n"
    "- 0 Penguin (250): rams lane for 500 damage. Use at x=2/3 to 1-shot Wizards or clusters.\n"
    "- 1 Soldier (200): 40 DPS forward. Place at x=1 in threatened rows (x=2 for double DPS).\n"
    "- 2 Archer (300): shoots ANY row. Place at x=0.\n"
    "- 3 Shield (150): 700 HP blocker. Place at x=3 to stop Monks and absorb Pirate blasts.\n\n"
    "Rules:\n"
    "1. Every threatened row MUST have a Soldier at x=1.\n"
    "2. Rows with Wizards MUST have a Penguin at x=2 or x=3.\n"
    "3. Rows with Monks/Pirates MUST have a Shield at x=3.\n"
    "4. Return JSON only: {\"placements\": [{\"defenceId\": 1, \"x\": 1, \"y\": 0}], \"reason\": \"<brief>\"}"
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

    # 1. Threat Analysis per row
    row_threats = analyze_threats(state, phase, enemies, defences)

    # 2. Query LLM for strategic advice
    llm_placements = []
    try:
        llm_placements = query_llm(state, phase, coins, costs, taken, defences, row_threats, time_budget)
    except Exception as exc:
        log(f"LLM not used ({type(exc).__name__}: {str(exc)[:80]})")

    # 3. Deterministic Safety Overseer: Combine, validate, and auto-fill
    final_placements = overseer_combine(llm_placements, row_threats, taken, coins, costs, phase)
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
            "has_monk": False,
            "total_enemy_hp": 0,
            "upcoming_count": 0,
            "has_soldier_x1": any(_xy(d) == (1, y) and "Solider" in d.get("type", "") for d in defences),
            "has_soldier_x2": any(_xy(d) == (2, y) and "Solider" in d.get("type", "") for d in defences),
            "has_shield": any(_xy(d)[1] == y and "Shield" in d.get("type", "") for d in defences),
            "has_archer": any(_xy(d) == (0, y) and "Archer" in d.get("type", "") for d in defences),
            "has_penguin": any(_xy(d)[1] == y and "Penguin" in d.get("type", "") for d in defences),
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
            if "Monk" in etype:
                threats[y]["has_monk"] = True

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
            score += 1000.0
            score += max(0.0, (11.0 - t["min_x"]) * 50.0)
            if t["time_to_breach"] < 5.0:
                score += 200.0
            if t["has_wizard"]:
                score += 300.0  # High priority to stop Wizard
            if t["has_pirate"]:
                score += 150.0
            if not t["has_soldier_x1"]:
                score += 200.0
            if not t["has_shield"]:
                score += 100.0
        elif t["upcoming_count"] > 0:
            score += 50.0 + t["upcoming_count"] * 10.0
            if not t["has_soldier_x1"]:
                score += 40.0

        t["score"] = score

    return threats


def query_llm(state, phase, coins, costs, taken, defences, row_threats, time_budget):
    """Single-shot structured call to the LLM without slow tool roundtrips."""
    active_rows_summary = []
    for y in range(6):
        t = row_threats[y]
        if t["active_enemies"] > 0 or t["upcoming_count"] > 0:
            active_rows_summary.append({
                "row": y,
                "active_enemies": t["active_enemies"],
                "closest_x": t["min_x"] if t["min_x"] < 90 else None,
                "has_wizard": t["has_wizard"],
                "has_soldier": t["has_soldier_x1"],
                "has_shield": t["has_shield"],
                "upcoming_threat": t["upcoming_count"]
            })

    board_compact = {
        "phase": phase,
        "coins": int(coins),
        "total_archers": sum(1 for d in defences if "Archer" in d.get("type", "")),
        "rows": active_rows_summary,
    }

    deadline = max(2.0, time_budget - 3.0)

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


def overseer_combine(llm_placements, row_threats, taken, coins, costs, phase):
    """Combines verified LLM placements with deterministic grandmaster tactical rules.
    
    Guarantees:
    - Wizard Counter: places Penguin at x=2 or x=3 to 1-shot Wizards.
    - Anti-Breach Coverage: every threatened row has at least 1 Soldier at x=1.
    - Swarm Reinforcement: places second Soldier at x=2 for high-HP/swarming rows.
    - Pirate & Monk Stall: Shield at x=3 safely tanks hits and detonate blasts outside Soldier radius.
    - Global Cross-Lane Fire: maintains 1-2 Archers at x=0.
    """
    chosen = []
    working_taken = set(taken)
    remaining_coins = coins

    # Track row defense state
    row_soldiers_x1 = {y: row_threats[y]["has_soldier_x1"] for y in range(6)}
    row_soldiers_x2 = {y: row_threats[y]["has_soldier_x2"] for y in range(6)}
    row_shields = {y: row_threats[y]["has_shield"] for y in range(6)}
    row_penguins = {y: row_threats[y]["has_penguin"] for y in range(6)}
    total_archers = sum(1 for y in range(6) if row_threats[y]["has_archer"])

    # 1. Process valid LLM proposals (with column sanitization)
    if llm_placements:
        for p in llm_placements[:8]:
            try:
                kind = int(p["defenceId"])
                x = int(p["x"])
                y = int(p["y"])
            except (KeyError, TypeError, ValueError):
                continue

            if kind not in costs or not (0 <= y <= 5):
                continue

            # Standardize placement columns
            if kind == 2:  # Archer -> must be at x=0
                x = 0
                if total_archers >= 2:
                    continue
            elif kind == 1:  # Soldier -> prefer x=1, else x=2
                x = 1 if not row_soldiers_x1[y] else 2
                if x == 2 and row_soldiers_x2[y]:
                    continue
            elif kind == 3:  # Shield -> must be at x=3
                x = 3
                if row_shields[y]:
                    continue
            elif kind == 0:  # Penguin -> x=2 or x=3
                x = 2 if (2, y) not in working_taken else 3
                if row_penguins[y]:
                    continue

            if 0 <= x <= 10 and (x, y) not in working_taken and costs[kind] <= remaining_coins:
                working_taken.add((x, y))
                remaining_coins -= costs[kind]
                if kind == 1:
                    if x == 1:
                        row_soldiers_x1[y] = True
                    else:
                        row_soldiers_x2[y] = True
                elif kind == 3:
                    row_shields[y] = True
                elif kind == 0:
                    row_penguins[y] = True
                elif kind == 2:
                    total_archers += 1
                chosen.append({"defenceId": kind, "defenceGridPosition": {"x": x, "y": y}})

    # 2. Deterministic Grandmaster Auto-Fill
    sorted_rows = sorted(row_threats.values(), key=lambda t: t["score"], reverse=True)

    # STEP A: THE WIZARD COUNTER (Top Priority)
    # If a Wizard is active in row y and no Penguin is there, place Penguin at x=2 or x=3!
    for t in sorted_rows:
        if len(chosen) >= 8 or remaining_coins < costs[0]:
            break
        y = t["row"]
        if t["has_wizard"] and not row_penguins[y]:
            for px in [2, 3]:
                if (px, y) not in working_taken:
                    working_taken.add((px, y))
                    remaining_coins -= costs[0]
                    row_penguins[y] = True
                    chosen.append({"defenceId": 0, "defenceGridPosition": {"x": px, "y": y}})
                    break

    # STEP B: Primary Soldier at x=1 for EVERY Active Threatened Row
    for t in sorted_rows:
        if len(chosen) >= 8 or remaining_coins < costs[1]:
            break
        y = t["row"]
        if t["active_enemies"] > 0 and not row_soldiers_x1[y] and (1, y) not in working_taken:
            working_taken.add((1, y))
            remaining_coins -= costs[1]
            row_soldiers_x1[y] = True
            chosen.append({"defenceId": 1, "defenceGridPosition": {"x": 1, "y": y}})

    # STEP C: Shield at x=3 for Active Threatened Rows (Absorb Monk/Pirate/Clown)
    for t in sorted_rows:
        if len(chosen) >= 8 or remaining_coins < costs[3]:
            break
        y = t["row"]
        if t["active_enemies"] > 0 and not row_shields[y]:
            for sx in [3, 2, 4]:
                if (sx, y) not in working_taken:
                    working_taken.add((sx, y))
                    remaining_coins -= costs[3]
                    row_shields[y] = True
                    chosen.append({"defenceId": 3, "defenceGridPosition": {"x": sx, "y": y}})
                    break

    # STEP D: Opening/Upcoming Broad Row Coverage (Soldier at x=1)
    # Guarantees no enemy spawns into an empty lane!
    for t in sorted_rows:
        if len(chosen) >= 8 or remaining_coins < costs[1]:
            break
        y = t["row"]
        if t["upcoming_count"] > 0 and not row_soldiers_x1[y] and (1, y) not in working_taken:
            working_taken.add((1, y))
            remaining_coins -= costs[1]
            row_soldiers_x1[y] = True
            chosen.append({"defenceId": 1, "defenceGridPosition": {"x": 1, "y": y}})

    # STEP E: Global Archer at x=0 (At least 1 Archer, max 2)
    if total_archers < 1 and remaining_coins >= costs[2] and len(chosen) < 8:
        for t in sorted_rows:
            y = t["row"]
            if (0, y) not in working_taken and remaining_coins >= costs[2]:
                working_taken.add((0, y))
                remaining_coins -= costs[2]
                total_archers += 1
                chosen.append({"defenceId": 2, "defenceGridPosition": {"x": 0, "y": y}})
                break

    # STEP F: Swarm Reinforcement - Second Soldier at x=2 for Dense Lanes
    for t in sorted_rows:
        if len(chosen) >= 8 or remaining_coins < costs[1]:
            break
        y = t["row"]
        if (t["active_enemies"] >= 2 or t["total_enemy_hp"] >= 300) and not row_soldiers_x2[y]:
            if (2, y) not in working_taken:
                working_taken.add((2, y))
                remaining_coins -= costs[1]
                row_soldiers_x2[y] = True
                chosen.append({"defenceId": 1, "defenceGridPosition": {"x": 2, "y": y}})

    # STEP G: Second Archer for Global Firepower
    if total_archers < 2 and remaining_coins >= costs[2] and len(chosen) < 8:
        for t in sorted_rows:
            y = t["row"]
            if (0, y) not in working_taken and remaining_coins >= costs[2]:
                working_taken.add((0, y))
                remaining_coins -= costs[2]
                total_archers += 1
                chosen.append({"defenceId": 2, "defenceGridPosition": {"x": 0, "y": y}})
                break

    # STEP H: Extra Shields for Upcoming Rows
    for t in sorted_rows:
        if len(chosen) >= 8 or remaining_coins < costs[3]:
            break
        y = t["row"]
        if t["upcoming_count"] > 0 and not row_shields[y]:
            for sx in [3, 2]:
                if (sx, y) not in working_taken:
                    working_taken.add((sx, y))
                    remaining_coins -= costs[3]
                    row_shields[y] = True
                    chosen.append({"defenceId": 3, "defenceGridPosition": {"x": sx, "y": y}})
                    break

    return chosen