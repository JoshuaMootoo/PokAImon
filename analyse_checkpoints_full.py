"""
analyse_checkpoints_full.py
---------------------------
Loads each PPO checkpoint (100k–5300k steps, 100k increments) and runs one
episode (max 2000 steps, deterministic=True) to measure agent progression.

Recorded per checkpoint:
  - max map_id reached
  - unique tiles visited
  - badges earned
  - level sum (end of episode)
  - total XP (end of episode, raw from memory)
  - XP gained during the episode (end XP minus start XP, not from save state)
  - set of map IDs visited

Also prints:
  - OFF_XP_0, OFF_XP_1, OFF_XP_2 values from env/memory.py
  - Raw XP bytes for slot 0 at episode start and end (to verify fix)
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_DIR)

from stable_baselines3 import PPO
from env.pokemon_env import PokemonBlueEnv
from env.memory import (
    MAP_ID, PLAYER_X, PLAYER_Y,
    read_badges, read_party_level_sum, read_party_xp_total,
    count_bits, POKEDEX_OWNED_START,
    PARTY_ADDRS, OFF_XP_0, OFF_XP_1, OFF_XP_2,
    PARTY_COUNT,
)

ROM_PATH        = os.path.join(PROJ_DIR, "pokemon_blue.gb")
INIT_STATE_PATH = os.path.join(PROJ_DIR, "init.state")
CKPT_DIR        = os.path.join(PROJ_DIR, "checkpoints")
MAX_STEPS       = 2000

STEPS = list(range(100_000, 5_400_000, 100_000))  # 100k … 5300k


def read_raw_xp_bytes(mem):
    """Return the raw XP bytes for each party slot as a list of (b0, b1, b2) tuples."""
    party_size = mem[PARTY_COUNT]
    result = []
    for i in range(min(party_size, 6)):
        base = PARTY_ADDRS[i]
        b0 = mem[base + OFF_XP_0]
        b1 = mem[base + OFF_XP_1]
        b2 = mem[base + OFF_XP_2]
        result.append((b0, b1, b2))
    return result


def run_episode(model, env):
    """Run one deterministic episode; return stats dict."""
    obs, _ = env.reset()

    mem = env.pyboy.memory

    # Capture XP at the very start (after reset, before any steps)
    xp_start        = read_party_xp_total(mem)
    raw_xp_start    = read_raw_xp_bytes(mem)
    level_sum_start = read_party_level_sum(mem)

    max_map_id   = mem[MAP_ID]
    visited_maps = set()
    badges       = 0
    level_sum    = level_sum_start
    xp_total     = xp_start

    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        cur_mem = env.pyboy.memory
        cur_map = cur_mem[MAP_ID]
        visited_maps.add(cur_map)
        if cur_map > max_map_id:
            max_map_id = cur_map

        badges    = read_badges(cur_mem)
        level_sum = read_party_level_sum(cur_mem)
        xp_total  = read_party_xp_total(cur_mem)

        if terminated or truncated:
            break

    raw_xp_end = read_raw_xp_bytes(env.pyboy.memory)
    unique_tiles = len(env.visited_tiles)
    xp_gained = xp_total - xp_start

    return {
        "max_map_id":     max_map_id,
        "unique_tiles":   unique_tiles,
        "badges":         badges,
        "level_sum":      level_sum,
        "level_sum_start": level_sum_start,
        "xp_total":       xp_total,
        "xp_start":       xp_start,
        "xp_gained":      xp_gained,
        "visited_maps":   visited_maps,
        "raw_xp_start":   raw_xp_start,
        "raw_xp_end":     raw_xp_end,
    }


def main():
    # Print OFF_XP offsets from memory.py
    print("=" * 60)
    print("memory.py XP offset values:")
    print(f"  OFF_XP_0 = 0x{OFF_XP_0:02X} ({OFF_XP_0})")
    print(f"  OFF_XP_1 = 0x{OFF_XP_1:02X} ({OFF_XP_1})")
    print(f"  OFF_XP_2 = 0x{OFF_XP_2:02X} ({OFF_XP_2})")
    print("=" * 60)
    print()

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
            f"lvl_sum={stats['level_sum']:3d} (start={stats['level_sum_start']}) "
            f"xp_total={stats['xp_total']:7d} xp_gained={stats['xp_gained']:7d} "
            f"maps={sorted(stats['visited_maps'])}",
            flush=True,
        )
        print(
            f"  raw XP bytes start={stats['raw_xp_start']}  "
            f"end={stats['raw_xp_end']}",
            flush=True,
        )

    env.close()

    # ------------------------------------------------------------------ #
    # Print summary table
    # ------------------------------------------------------------------ #
    print()
    print("=" * 130)
    print(
        f"{'Steps':>12} | {'MaxMapID':>8} | {'Tiles':>6} | {'Badges':>6} | "
        f"{'LvlStart':>8} | {'LvlEnd':>6} | {'XP_Start':>9} | {'XP_End':>9} | "
        f"{'XP_Gained':>9} | Maps Visited"
    )
    print("=" * 130)
    for r in results:
        maps_str = str(sorted(r["visited_maps"]))
        print(
            f"{r['steps']:>12,} | {r['max_map_id']:>8} | {r['unique_tiles']:>6} | "
            f"{r['badges']:>6} | {r['level_sum_start']:>8} | {r['level_sum']:>6} | "
            f"{r['xp_start']:>9,} | {r['xp_total']:>9,} | {r['xp_gained']:>9,} | "
            f"{maps_str}"
        )
    print("=" * 130)


if __name__ == "__main__":
    main()
