import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r'E:\Local\Documents\Local Projects\Pokemon AI')

from env import PokemonBlueEnv
from env.memory import read_party_xp_total, read_party_level_sum, read_badges, MAP_ID, BATTLE_FLAG
from stable_baselines3 import PPO

ROM_PATH   = r'E:\Local\Documents\Local Projects\Pokemon AI\pokemon_blue.gb'
STATE_PATH = r'E:\Local\Documents\Local Projects\Pokemon AI\init.state'
CHECKPOINT_DIR = r'E:\Local\Documents\Local Projects\Pokemon AI\checkpoints'

MAP_NAMES = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    3: "Cerulean City",
    12: "Route 1",
    13: "Route 2",
}

PRE_FIX  = [1000000, 5000000, 10000000, 15000000, 19100000]
POST_FIX = [19200000, 19500000, 20000000, 20500000, 21000000, 21500000, 22000000]
ALL_STEPS = PRE_FIX + POST_FIX

results = []

print("=" * 80)
print("POKEMON BLUE CHECKPOINT ANALYSIS")
print("Pre-fix: 1M, 5M, 10M, 15M, 19.1M | Post-fix: 19.2M-22M")
print("=" * 80)

for step_count in ALL_STEPS:
    ckpt_name = f"pokemon_blue_ppo_{step_count}_steps.zip"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

    label = "PRE " if step_count in PRE_FIX else "POST"
    tag   = "*** POST-FIX ***" if step_count in POST_FIX else "pre-fix"

    print(f"\n[{label}] Checkpoint {step_count:,} steps  ({tag})")
    print(f"  Loading: {ckpt_name}")

    if not os.path.exists(ckpt_path):
        print(f"  ERROR: file not found, skipping.")
        results.append({"steps": step_count, "error": True})
        continue

    env = PokemonBlueEnv(
        rom_path=ROM_PATH,
        init_state_path=STATE_PATH,
        max_steps=3000,
        frame_skip=16,
    )

    try:
        model = PPO.load(ckpt_path, env=env, device='cpu')
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        env.close()
        results.append({"steps": step_count, "error": True})
        continue

    obs, info = env.reset()
    xp_start = read_party_xp_total(env.pyboy.memory)

    total_reward  = 0.0
    visited_maps  = set()
    step_num      = 0
    battle_steps  = 0
    battles_entered  = 0
    battles_ended    = 0
    prev_in_battle   = False
    done = False
    truncated = False

    while not done and not truncated:
        cur_map = env.pyboy.memory[MAP_ID]
        visited_maps.add(cur_map)

        in_battle = env.pyboy.memory[BATTLE_FLAG] != 0
        if in_battle:
            battle_steps += 1
            if not prev_in_battle:
                battles_entered += 1
        else:
            if prev_in_battle:
                battles_ended += 1
        prev_in_battle = in_battle

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_num += 1

    xp_end     = read_party_xp_total(env.pyboy.memory)
    xp_gained  = xp_end - xp_start
    level_sum  = read_party_level_sum(env.pyboy.memory)
    badges     = read_badges(env.pyboy.memory)
    unique_tiles = len(env.tile_visit_count)

    named_maps = []
    for mid in sorted(visited_maps):
        name = MAP_NAMES.get(mid, f"Map#{mid}")
        named_maps.append(f"{mid}({name})")

    print(f"  Steps taken       : {step_num}")
    print(f"  Total reward      : {total_reward:.4f}")
    print(f"  Unique tiles      : {unique_tiles}")
    print(f"  Maps visited      : {named_maps}")
    print(f"  XP start          : {xp_start}")
    print(f"  XP end            : {xp_end}")
    print(f"  XP gained         : {xp_gained}")
    print(f"  Level sum         : {level_sum}")
    print(f"  Badges            : {badges}")
    print(f"  Battle steps      : {battle_steps} / {step_num} ({100*battle_steps/max(step_num,1):.1f}%)")
    print(f"  Battles entered   : {battles_entered}")
    print(f"  Battles ended     : {battles_ended}")

    reached_route1 = 12 in visited_maps
    reached_viridian = 1 in visited_maps
    print(f"  Reached Route 1   : {'YES' if reached_route1 else 'no'}")
    print(f"  Reached Viridian  : {'YES' if reached_viridian else 'no'}")

    env.close()

    results.append({
        "steps": step_count,
        "label": label,
        "step_num": step_num,
        "total_reward": total_reward,
        "unique_tiles": unique_tiles,
        "visited_maps": visited_maps,
        "xp_start": xp_start,
        "xp_end": xp_end,
        "xp_gained": xp_gained,
        "level_sum": level_sum,
        "badges": badges,
        "battle_steps": battle_steps,
        "battles_entered": battles_entered,
        "battles_ended": battles_ended,
        "reached_route1": reached_route1,
        "reached_viridian": reached_viridian,
        "error": False,
    })

print("\n")
print("=" * 115)
print("SUMMARY TABLE")
print("=" * 115)
print(f"{'Ckpt':>10} {'Tag':>5} | {'Reward':>9} {'Tiles':>6} {'#Maps':>5} {'XP':>8} {'Lvl':>5} {'Bdg':>4} | {'BtlStps':>8} {'Btl%':>6} {'Btls':>5} | {'Rt1':>4} {'Vrd':>4}")
print("-" * 115)

for r in results:
    if r.get("error"):
        print(f"  {r['steps']:>8,}  ERROR")
        continue
    maps_count = len(r["visited_maps"])
    tag = r["label"]
    rt1 = "Y" if r["reached_route1"] else "n"
    vrd = "Y" if r["reached_viridian"] else "n"
    print(
        f"  {r['steps']:>8,}  {tag} | "
        f"{r['total_reward']:>9.3f} {r['unique_tiles']:>6} {maps_count:>5} {r['xp_gained']:>8} {r['level_sum']:>5} {r['badges']:>4} | "
        f"{r['battle_steps']:>8} {100*r['battle_steps']/max(r['step_num'],1):>5.1f}% {r['battles_entered']:>5} | "
        f"{rt1:>4} {vrd:>4}"
    )

print("=" * 115)
print("\nKey:")
print("  Reward=total episode reward | Tiles=unique tiles visited | #Maps=distinct map IDs")
print("  XP=XP gained | Lvl=party level sum | Bdg=badges | BtlStps=steps in battle")
print("  Btl%=% of steps in battle | Btls=battles entered | Rt1=Route1 reached | Vrd=Viridian reached")
print("\nMap ID reference: 0=Pallet Town, 1=Viridian City, 2=Pewter, 12=Route 1, 13=Route 2")
print("\nDone.")
