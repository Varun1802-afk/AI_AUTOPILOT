"""agent.py - Grandmaster Tactical Game Agent (Team NeuralQuesters).

A state-aware, deterministic tactical engine engineered to win all test cases
and maximize the official competition score (Outcome, Kills, Survival, Speed, Economy).

Key Principles:
1. Complete Soldier-Shield Line (The Unbreachable Defense):
   - A naked Soldier at x=1 has only 150 HP and dies in 3s to Wizards, 4s to Monks,
     or instantly to 200 Pirate blast damage.
   - Pairing a Soldier at x=1 with a Shield at x=3 is completely impenetrable:
     * Shield has 700 HP (takes 20s for Monks to break, 14s for Wizards).
     * Shield at x=3 is 2 tiles ahead of Soldier at x=1 (distance = 3.0 units).
       Since Pirate blast radius is 2.0 units (1.3 tiles), a Pirate detonating at x=3
       deals ZERO damage to the Soldier at x=1!
     * 700 HP Shield absorbs 200 blast and still has 500 HP left!
2. Anti-Wizard Penguin Nuke:
   - Wizard has 500 HP (+15/s regen) and 50 DPS beam from 5.3 tiles range.
   - Penguin (ID 0, cost 250) deals 500 burst to EVERY enemy in its lane upon taking ANY damage.
   - Placing a Penguin at x=2 or x=3 absorbs the first beam tick, rockets forward at 10 units/s,
     and 1-shots the 500 HP Wizard instantly!
3. Dynamic Swarm Reinforcement:
   - High-HP lanes receive a second Soldier at x=2, doubling lane DPS to 80.
4. Global Cross-Lane Focus:
   - 1-2 Archers at x=0 provide constant cross-lane focus fire.
5. Zero Network / Zero 502 Vulnerability:
   - Evaluated in microseconds locally; completely immune to upstream LLM 502 Bad Gateway timeouts.
"""
import json
import sys

# Official Unit Stats
COSTS = {0: 250, 1: 200, 2: 300, 3: 150}
DEFENCE_HP = {0: 200, 1: 150, 2: 100, 3: 700}

ENEMY_STATS = {
    "EnemyPirate": {"speed": 8.0, "hp": 30.0, "dps": 0.0, "pirate": True, "wizard": False},
    "EnemyMonk": {"speed": 3.5, "hp": 280.0, "dps": 36.0, "pirate": False, "wizard": False},
    "EnemyWizard": {"speed": 3.0, "hp": 500.0, "dps": 50.0, "pirate": False, "wizard": True},
    "EnemClown": {"speed": 2.0, "hp": 200.0, "dps": 17.0, "pirate": False, "wizard": False},
}


def log(message):
    print(f"[agent] {message}", file=sys.stderr, flush=True)


def _xy(unit):
    pos = unit.get("position") or unit.get("defenceGridPosition") or {}
    try:
        x = int(round(float(pos.get("x", 0))))
        y = int(round(float(pos.get("y", 0))))
    except (TypeError, ValueError):
        return 0, 0
    return max(0, min(10, x)), max(0, min(5, y))


def _defence_id(d):
    did = d.get("defenceId")
    if isinstance(did, (int, float)) and int(did) in COSTS:
        return int(did)
    t = str(d.get("type", "")).lower()
    if "penguin" in t:
        return 0
    if "archer" in t:
        return 2
    if "shield" in t:
        return 3
    return 1  # soldier / "solider"


def _enemy_type(e):
    t = str(e.get("type", ""))
    for name in ENEMY_STATS:
        if name.lower() in t.lower():
            return name
    eid = e.get("enemyId")
    if eid == 0:
        return "EnemClown"
    if eid == 1:
        return "EnemyMonk"
    if eid == 2:
        return "EnemyWizard"
    if eid == 3:
        return "EnemyPirate"
    return "EnemClown"


def get_reinforcements(state):
    """Main entry point called by the Autopilot SDK."""
    try:
        return plan_defences(state)
    except Exception as exc:
        log(f"Error in plan_defences: {type(exc).__name__}: {exc}")
        return []


def plan_defences(state):
    phase = str(state.get("phase", "pause")).lower()
    budget = state.get("budget", {})
    coins = float(budget.get("remaining", 0) if isinstance(budget, dict) else budget)
    unit_costs = {int(k): v for k, v in state.get("unitCosts", COSTS).items()}

    defences = [d for d in state.get("defences", []) if d.get("health", 0) > 0]
    enemies = [e for e in state.get("enemies", []) if not e.get("isDead", False) and e.get("health", 0) > 0]

    taken = {_xy(d) for d in defences}

    if coins < min(unit_costs.values()):
        return []

    # 1. Analyze Active Grid and Threats
    row_data = {}
    for y in range(6):
        row_data[y] = {
            "row": y,
            "has_soldier_x1": any(_xy(d) == (1, y) and _defence_id(d) == 1 for d in defences),
            "has_soldier_x2": any(_xy(d) == (2, y) and _defence_id(d) == 1 for d in defences),
            "has_shield": any(_xy(d)[1] == y and _defence_id(d) == 3 for d in defences),
            "has_archer": any(_xy(d) == (0, y) and _defence_id(d) == 2 for d in defences),
            "has_penguin": any(_xy(d)[1] == y and _defence_id(d) == 0 for d in defences),
            "active_enemies": 0,
            "closest_x": 99.0,
            "time_to_breach": 999.0,
            "has_wizard": False,
            "has_pirate": False,
            "has_monk": False,
            "total_enemy_hp": 0.0,
            "upcoming_threat": 0,
            "expect_wizard": False,
            "expect_pirate": False,
            "expect_monk": False,
        }

    # Inspect active enemies
    for e in enemies:
        x, y = _xy(e)
        if 0 <= y <= 5:
            etype = _enemy_type(e)
            stats = ENEMY_STATS.get(etype, {"speed": 3.0, "hp": 200.0})
            hp = float(e.get("health", stats["hp"]))
            speed = max(stats["speed"], 1.0)

            row_data[y]["active_enemies"] += 1
            row_data[y]["total_enemy_hp"] += hp
            if x < row_data[y]["closest_x"]:
                row_data[y]["closest_x"] = x
                row_data[y]["time_to_breach"] = (x * 1.5) / speed
            if stats.get("wizard") or "wizard" in etype.lower():
                row_data[y]["has_wizard"] = True
            if stats.get("pirate") or "pirate" in etype.lower():
                row_data[y]["has_pirate"] = True
            if "monk" in etype.lower():
                row_data[y]["has_monk"] = True

    # Inspect upcoming wave configurations (especially at start)
    upcoming = state.get("upcomingEnemies", [])
    for u in upcoming:
        if not isinstance(u, dict):
            continue
        eid = u.get("enemyId")
        count = int(u.get("count", 1))
        allowed = u.get("allowedRows")
        target_rows = allowed if (isinstance(allowed, (list, tuple)) and allowed) else list(range(6))
        for r in target_rows:
            if 0 <= r <= 5:
                row_data[r]["upcoming_threat"] += count
                if eid == 2:
                    row_data[r]["expect_wizard"] = True
                elif eid == 3:
                    row_data[r]["expect_pirate"] = True
                elif eid == 1:
                    row_data[r]["expect_monk"] = True

    # Calculate urgency score
    for y, r in row_data.items():
        score = 0.0
        if r["active_enemies"] > 0:
            score += 2000.0
            score += max(0.0, (11.0 - r["closest_x"]) * 100.0)
            if r["time_to_breach"] < 5.0:
                score += 500.0
            if r["has_wizard"]:
                score += 800.0
            if r["has_pirate"]:
                score += 400.0
            if not r["has_soldier_x1"]:
                score += 600.0
            if not r["has_shield"]:
                score += 300.0
        elif r["upcoming_threat"] > 0:
            score += 100.0 + r["upcoming_threat"] * 10.0
            if r["expect_wizard"]:
                score += 50.0
            if r["expect_pirate"]:
                score += 40.0
        r["score"] = score

    # 2. Grandmaster Placement Allocator
    placements = []
    working_taken = set(taken)
    remaining_coins = coins
    total_archers = sum(1 for y in range(6) if row_data[y]["has_archer"])

    def can_afford(kind):
        return remaining_coins >= unit_costs[kind]

    def place(kind, x, y):
        nonlocal remaining_coins, total_archers
        if (x, y) in working_taken or remaining_coins < unit_costs[kind]:
            return False
        if not (0 <= x <= 10 and 0 <= y <= 5):
            return False
        working_taken.add((x, y))
        remaining_coins -= unit_costs[kind]
        if kind == 1:
            if x == 1:
                row_data[y]["has_soldier_x1"] = True
            elif x == 2:
                row_data[y]["has_soldier_x2"] = True
        elif kind == 3:
            row_data[y]["has_shield"] = True
        elif kind == 0:
            row_data[y]["has_penguin"] = True
        elif kind == 2:
            row_data[y]["has_archer"] = True
            total_archers += 1

        placements.append({"defenceId": kind, "defenceGridPosition": {"x": x, "y": y}})
        return True

    sorted_threats = sorted(row_data.values(), key=lambda r: r["score"], reverse=True)

    # =========================================================================
    # PHASE A: EMERGENCY ACTIVE COUNTERS (Active Wizard / Active Threat)
    # =========================================================================
    # 1. Anti-Wizard Penguin Nuke
    for r in sorted_threats:
        y = r["row"]
        if r["has_wizard"] and not r["has_penguin"] and can_afford(0):
            # Place Penguin at x=2 or x=3
            for px in [2, 3]:
                if (px, y) not in working_taken and place(0, px, y):
                    break

    # 2. Urgent Soldier at x=1 for any active lane without a Soldier
    for r in sorted_threats:
        y = r["row"]
        if r["active_enemies"] > 0 and not r["has_soldier_x1"] and can_afford(1):
            place(1, 1, y)

    # 3. Urgent Shield at x=3 for any active lane without a Shield
    for r in sorted_threats:
        y = r["row"]
        if r["active_enemies"] > 0 and not r["has_shield"] and can_afford(3):
            for sx in [3, 2, 4]:
                if (sx, y) not in working_taken and place(3, sx, y):
                    break

    # =========================================================================
    # PHASE B: OPENING / UPCOMING COMPLETE LANE COVERAGE
    # =========================================================================
    # Every threatened lane MUST have a Soldier at x=1 to prevent naked lane breaches
    for r in sorted_threats:
        y = r["row"]
        if (r["upcoming_threat"] > 0 or r["active_enemies"] > 0) and not r["has_soldier_x1"] and can_afford(1):
            place(1, 1, y)

    # Every threatened lane with Wizards, Pirates, or Monks MUST have a Shield at x=3
    for r in sorted_threats:
        y = r["row"]
        if (r["expect_wizard"] or r["expect_pirate"] or r["expect_monk"] or r["active_enemies"] > 0):
            if not r["has_shield"] and can_afford(3):
                for sx in [3, 2, 4]:
                    if (sx, y) not in working_taken and place(3, sx, y):
                        break

    # Even for generic upcoming threats, ensure Shield at x=3 if budget permits
    for r in sorted_threats:
        y = r["row"]
        if r["upcoming_threat"] > 0 and not r["has_shield"] and can_afford(3):
            for sx in [3, 2]:
                if (sx, y) not in working_taken and place(3, sx, y):
                    break

    # =========================================================================
    # PHASE C: GLOBAL CROSS-LANE FIREPOWER (Archers at x=0)
    # =========================================================================
    # Target 1-2 Archers at x=0
    target_archers = 2 if coins >= 1800 else 1
    if total_archers < target_archers and can_afford(2):
        # Place in quietest/safest row
        for r in sorted(row_data.values(), key=lambda r: (r["active_enemies"], r["upcoming_threat"])):
            y = r["row"]
            if (0, y) not in working_taken and place(2, 0, y):
                if total_archers >= target_archers:
                    break

    # =========================================================================
    # PHASE D: MID-GAME SWARM REINFORCEMENT (Double Soldier at x=2)
    # =========================================================================
    # If a row has multiple active enemies or high total HP, add second Soldier at x=2
    for r in sorted_threats:
        y = r["row"]
        if (r["active_enemies"] >= 2 or r["total_enemy_hp"] >= 250.0) and not r["has_soldier_x2"] and can_afford(1):
            if (2, y) not in working_taken and place(1, 2, y):
                pass

    # =========================================================================
    # PHASE E: SURPLUS BUDGET UTILIZATION (Additional Archers or Shields)
    # =========================================================================
    if remaining_coins >= 300 and total_archers < 2:
        for r in sorted(row_data.values(), key=lambda r: (r["active_enemies"], r["upcoming_threat"])):
            y = r["row"]
            if (0, y) not in working_taken and place(2, 0, y):
                break

    desc = ", ".join(f"{p['defenceId']}@({p['defenceGridPosition']['x']},{p['defenceGridPosition']['y']})" for p in placements)
    log(f"phase={phase} coins={int(coins)} placed={len(placements)} spent={int(coins - remaining_coins)} left={int(remaining_coins)} [{desc}]")
    return placements