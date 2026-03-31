"""
Checkpoint analysis script for Pokemon Blue PPO training run.
Samples 20 checkpoints to measure progress over training.
"""

import sys
import os
sys.path.insert(0, r'E:\Local\Documents\Local Projects\Pokemon AI')

from env import PokemonBlueEnv
from env.memory import read_party_xp_total, read_party_level_sum, read_badges, MAP_ID
from stable_baselines3 import PPO

ROM_PATH        = r'E:\Local\Documents\Local Projects\Pokemon AI\pokemon_blue.gb'
STATE_PATH      = r'E:\Local\Documents\Local Projects\Pokemon AI\init.state'
CHECKPOINT_DIR  = r'E:\Local\Documents\Local Projects\Pokemon AI\checkpoints'

CHECKPOINTS = [
    100_000, 300_000, 500_000, 700_000,
    1_000_000, 1_500_000, 2_000_000, 3_000_000,
    4_000_000, 5_000_000, 6_000_000, 7_000_000,
    8_000_000, 9_000_000, 10_000_000, 12_000_000,
    14_000_000, 16_000_000, 18_000_000, 19_100_000,
]

# Known map IDs for milestone detection
MAP_NAMES = {
    0:   "Pallet Town",
    1:   "Viridian City",
    2:   "Pewter City",
    3:   "Cerulean City",
    4:   "Lavender Town",
    5:   "Vermilion City",
    6:   "Celadon City",
    7:   "Fuchsia City",
    8:   "Cinnabar Island",
    9:   "Indigo Plateau",
    10:  "Saffron City",
    12:  "Route 1",
    13:  "Route 2",
    14:  "Route 3",
    15:  "Route 4",
    33:  "Viridian Forest",
    36:  "Mt. Moon",
    37:  "Mt. Moon B1F",
    38:  "Mt. Moon B2F",
}

def map_name(map_id):
    return MAP_NAMES.get(map_id, f"Map {map_id}")

results = []

header = (
    f"{'Steps':>12} | {'Reward':>8} | {'Maps':>5} | {'Tiles':>6} | "
    f"{'XP_Start':>9} | {'XP_End':>9} | {'XP_Gain':>8} | "
    f"{'LvlSum':>6} | {'Badges':>6} | {'Steps':>5}"
)
print("=" * len(header))
print("POKEMON BLUE PPO — CHECKPOINT ANALYSIS")
print("=" * len(header))
print(header)
print("-" * len(header))

for steps in CHECKPOINTS:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"pokemon_blue_ppo_{steps}_steps.zip")

    if not os.path.exists(ckpt_path):
        print(f"{steps:>12} | MISSING CHECKPOINT")
        continue

    try:
        # Create a brand-new env each time so global tile_visit_count is clean
        env = PokemonBlueEnv(
            rom_path=ROM_PATH,
            init_state_path=STATE_PATH,
            max_steps=3000,
            frame_skip=16,
        )

        model = PPO.load(ckpt_path, env=env, device='cpu')

        obs, _ = env.reset()

        xp_start     = read_party_xp_total(env.pyboy.memory)
        total_reward = 0.0
        maps_visited = set()
        steps_taken  = 0

        for step in range(3000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))

            total_reward += reward
            maps_visited.add(env.pyboy.memory[MAP_ID])
            steps_taken += 1

            if term or trunc:
                break

        xp_end    = read_party_xp_total(env.pyboy.memory)
        level_sum = read_party_level_sum(env.pyboy.memory)
        badges    = read_badges(env.pyboy.memory)
        xp_gained = xp_end - xp_start
        tiles     = info['visited_tiles']

        env.close()

        row = {
            "steps":      steps,
            "reward":     total_reward,
            "maps":       maps_visited,
            "tiles":      tiles,
            "xp_start":   xp_start,
            "xp_end":     xp_end,
            "xp_gained":  xp_gained,
            "level_sum":  level_sum,
            "badges":     badges,
            "ep_steps":   steps_taken,
        }
        results.append(row)

        # Print the sorted map set with names for readability
        sorted_maps = sorted(maps_visited)
        map_str = "{" + ", ".join(str(m) for m in sorted_maps) + "}"

        print(
            f"{steps:>12,} | {total_reward:>8.2f} | {len(maps_visited):>5} | {tiles:>6} | "
            f"{xp_start:>9} | {xp_end:>9} | {xp_gained:>8} | "
            f"{level_sum:>6} | {badges:>6} | {steps_taken:>5}"
        )
        print(f"             Maps visited: {map_str}")
        named = [map_name(m) for m in sorted_maps if m in MAP_NAMES]
        if named:
            print(f"             Named:        {', '.join(named)}")
        print()

    except Exception as e:
        print(f"{steps:>12,} | ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()

# -----------------------------------------------------------------------
# Summary analysis
# -----------------------------------------------------------------------
print("\n" + "=" * 80)
print("MILESTONE ANALYSIS")
print("=" * 80)

if results:
    # Find first time each milestone is hit
    first_positive_reward = None
    first_viridian        = None
    first_route1          = None
    first_xp              = None
    first_beyond_route1   = None  # any map > 12 that isn't Pallet/Route1/Viridian

    for r in results:
        s = r["steps"]
        if first_positive_reward is None and r["reward"] > 0:
            first_positive_reward = s
        if first_viridian is None and 1 in r["maps"]:
            first_viridian = s
        if first_route1 is None and 12 in r["maps"]:
            first_route1 = s
        if first_xp is None and r["xp_gained"] > 0:
            first_xp = s
        if first_beyond_route1 is None:
            beyond = r["maps"] - {0, 1, 12}
            if beyond:
                first_beyond_route1 = (s, sorted(beyond))

    print(f"1. Positive reward first seen:       step {first_positive_reward:,}" if first_positive_reward else "1. Positive reward NEVER seen")
    print(f"2. Viridian City (map 1) first seen: step {first_viridian:,}"        if first_viridian        else "2. Viridian City NEVER reached")
    print(f"3. Route 1 (map 12) first seen:      step {first_route1:,}"          if first_route1          else "3. Route 1 NEVER reached")
    print(f"4. XP gain first seen:               step {first_xp:,}"              if first_xp              else "4. XP gain NEVER seen")
    if first_beyond_route1:
        s, maps = first_beyond_route1
        names = [map_name(m) for m in maps]
        print(f"5. Beyond Pallet/Route1/Viridian:    step {s:,} — maps {maps} ({', '.join(names)})")
    else:
        print("5. Agent NEVER ventured beyond Pallet Town / Route 1 / Viridian City")

    print()
    print("TREND OVERVIEW (reward / tiles / maps / xp_gained)")
    print(f"{'Steps':>12} | {'Reward':>8} | {'Tiles':>6} | {'Maps':>5} | {'XP Gained':>10} | {'Lvl Sum':>7}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['steps']:>12,} | {r['reward']:>8.2f} | {r['tiles']:>6} | "
            f"{len(r['maps']):>5} | {r['xp_gained']:>10} | {r['level_sum']:>7}"
        )

print("\nDone.")
