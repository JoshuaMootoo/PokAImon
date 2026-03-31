"""
Checkpoint analysis script.
Loads sampled checkpoints, runs ONE episode of 2000 steps (deterministic=True),
and records key metrics.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure the project root is on the path so 'env' package resolves
PROJECT_ROOT = r"E:\Local\Documents\Local Projects\Pokemon AI"
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from stable_baselines3 import PPO
from env.pokemon_env import PokemonBlueEnv
from env.memory import MAP_ID, BADGES, read_badges, read_party_xp_total, read_party_level_sum

ROM_PATH   = os.path.join(PROJECT_ROOT, "pokemon_blue.gb")
STATE_PATH = os.path.join(PROJECT_ROOT, "init.state")
CKPT_DIR   = os.path.join(PROJECT_ROOT, "checkpoints")

# Sampled steps to evaluate
STEPS_TO_CHECK = [
    100_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000,
    3_000_000, 3_500_000, 4_000_000, 4_500_000, 5_000_000, 5_500_000,
    6_000_000, 6_500_000, 7_000_000, 7_500_000, 8_000_000, 8_500_000,
    9_000_000, 9_500_000, 10_000_000, 10_500_000, 11_000_000, 11_500_000,
    12_000_000, 12_500_000, 13_000_000, 13_500_000, 14_000_000, 14_500_000,
    15_000_000, 15_500_000, 16_000_000, 16_500_000, 17_000_000, 17_100_000,
]

MAX_STEPS = 2000

def ckpt_path(steps):
    return os.path.join(CKPT_DIR, f"pokemon_blue_ppo_{steps}_steps.zip")

def run_episode(model, env):
    obs, _ = env.reset()
    # Reset XP baseline AFTER reset so we capture XP gained during this episode
    xp_start = read_party_xp_total(env.pyboy.memory)

    maps_visited = set()
    max_map_id = 0
    total_reward = 0.0

    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        mem = env.pyboy.memory
        map_id = mem[MAP_ID]
        maps_visited.add(map_id)
        max_map_id = max(max_map_id, map_id)
        total_reward += reward

        if terminated or truncated:
            break

    xp_end   = read_party_xp_total(env.pyboy.memory)
    xp_gained = xp_end - xp_start

    level_sum  = read_party_level_sum(env.pyboy.memory)
    badges     = read_badges(env.pyboy.memory)
    unique_tiles = len(env.visited_tiles)

    return {
        "max_map_id":    max_map_id,
        "unique_tiles":  unique_tiles,
        "xp_gained":     xp_gained,
        "level_sum":     level_sum,
        "maps_visited":  maps_visited,
        "badges":        badges,
        "total_reward":  total_reward,
    }

# ── Determine which checkpoints actually exist ──────────────────────────────
available = [(s, ckpt_path(s)) for s in STEPS_TO_CHECK if os.path.exists(ckpt_path(s))]
print(f"Found {len(available)} checkpoints to evaluate.\n")

# ── Header ───────────────────────────────────────────────────────────────────
HDR = (f"{'Steps':>12}  {'MaxMap':>6}  {'Tiles':>6}  "
       f"{'XPGain':>8}  {'LvlSum':>6}  {'Badges':>6}  "
       f"{'TotRew':>8}  Maps")
print(HDR)
print("-" * len(HDR))

# ── Env is created once and reused (state is reloaded in reset()) ─────────────
env = PokemonBlueEnv(
    rom_path=ROM_PATH,
    init_state_path=STATE_PATH,
    max_steps=MAX_STEPS,
    render_mode="rgb_array",
)

results = []
for steps, path in available:
    try:
        model = PPO.load(path, env=env, device="cpu")
        r = run_episode(model, env)
        results.append((steps, r))

        maps_str = "{" + ",".join(str(m) for m in sorted(r["maps_visited"])) + "}"
        print(
            f"{steps:>12,}  {r['max_map_id']:>6}  {r['unique_tiles']:>6}  "
            f"{r['xp_gained']:>8}  {r['level_sum']:>6}  {r['badges']:>6}  "
            f"{r['total_reward']:>8.1f}  {maps_str}"
        )
        sys.stdout.flush()
    except Exception as e:
        print(f"{steps:>12,}  ERROR: {e}")
        sys.stdout.flush()

env.close()

print("\nDone.")
