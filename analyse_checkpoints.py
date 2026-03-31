"""
analyse_checkpoints.py
----------------------
Loads each PPO checkpoint (100k–1800k steps) and runs one episode (max 1000 steps,
deterministic=True) to measure agent progression.

Recorded per checkpoint:
  - max map_id reached
  - unique tiles visited
  - badges earned
  - level sum
  - total XP
  - set of map IDs visited
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Make sure the local env package is importable
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_DIR)

from stable_baselines3 import PPO
from env.pokemon_env import PokemonBlueEnv
from env.memory import (
    MAP_ID, PLAYER_X, PLAYER_Y,
    read_badges, read_party_level_sum, read_party_xp_total,
    count_bits, POKEDEX_OWNED_START,
)

ROM_PATH        = os.path.join(PROJ_DIR, "pokemon_blue.gb")
INIT_STATE_PATH = os.path.join(PROJ_DIR, "init.state")
CKPT_DIR        = os.path.join(PROJ_DIR, "checkpoints")
MAX_STEPS       = 1000

STEPS = list(range(100_000, 1_900_000, 100_000))  # 100k … 1800k


def run_episode(model, env):
    """Run one deterministic episode; return stats dict."""
    obs, _ = env.reset()

    max_map_id      = env.pyboy.memory[MAP_ID]
    visited_maps    = set()
    badges          = 0
    level_sum       = 0
    xp_total        = 0

    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        mem = env.pyboy.memory
        cur_map = mem[MAP_ID]
        visited_maps.add(cur_map)
        if cur_map > max_map_id:
            max_map_id = cur_map

        badges    = read_badges(mem)
        level_sum = read_party_level_sum(mem)
        xp_total  = read_party_xp_total(mem)

        if terminated or truncated:
            break

    unique_tiles = len(env.visited_tiles)

    return {
        "max_map_id":    max_map_id,
        "unique_tiles":  unique_tiles,
        "badges":        badges,
        "level_sum":     level_sum,
        "xp_total":      xp_total,
        "visited_maps":  visited_maps,
    }


def main():
    # Create a single env instance and reuse it across all checkpoints
    env = PokemonBlueEnv(
        rom_path=ROM_PATH,
        init_state_path=INIT_STATE_PATH,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )

    results = []

    for steps in STEPS:
        ckpt_name = f"pokemon_blue_ppo_{steps}_steps.zip"
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"[WARN] checkpoint not found: {ckpt_path}", flush=True)
            continue

        print(f"Loading {ckpt_name} ...", flush=True)
        model = PPO.load(ckpt_path, env=env, device="cpu")

        stats = run_episode(model, env)
        stats["steps"] = steps
        results.append(stats)

        print(
            f"  done | map_id={stats['max_map_id']:3d} "
            f"tiles={stats['unique_tiles']:5d} "
            f"badges={stats['badges']} "
            f"lvl_sum={stats['level_sum']:3d} "
            f"xp={stats['xp_total']:7d} "
            f"maps={sorted(stats['visited_maps'])}",
            flush=True,
        )

    env.close()

    # ------------------------------------------------------------------ #
    # Print summary table
    # ------------------------------------------------------------------ #
    print()
    print("=" * 100)
    print(
        f"{'Steps':>12} | {'MaxMapID':>8} | {'Tiles':>6} | {'Badges':>6} | "
        f"{'LvlSum':>6} | {'TotalXP':>9} | Maps Visited"
    )
    print("=" * 100)
    for r in results:
        maps_str = str(sorted(r["visited_maps"]))
        print(
            f"{r['steps']:>12,} | {r['max_map_id']:>8} | {r['unique_tiles']:>6} | "
            f"{r['badges']:>6} | {r['level_sum']:>6} | {r['xp_total']:>9,} | {maps_str}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
