"""stream.py — Live stream viewer for Pokemon Blue AI training.

Shows the AI playing at real game speed while train.py runs in the background.
Automatically reloads the model as training progresses — no restart needed.

The viewer prefers checkpoints/latest.zip (written by train.py every ~2000 steps)
over the numbered step-checkpoints (saved every 100k steps), so you see near-real-
time behaviour: the model reloads every 10 seconds by default.

Run in a second terminal while training:
    python train.py --no-render      # training without its own window
    python stream.py                 # live viewer in separate terminal

Usage:
    python stream.py                   # auto-loads latest model, normal speed
    python stream.py --speed 2         # 2x game speed
    python stream.py --check-interval 5   # reload check every 5 seconds
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from stable_baselines3 import PPO

from env.memory import (
    MAP_ID, PLAYER_X, PLAYER_Y,
    PARTY_COUNT, BATTLE_FLAG,
    POKEDEX_OWNED_START, POKEDEX_SEEN_START,
    read_badges, is_in_battle,
    read_party_hp_fraction, read_party_level_sum,
    read_party_pokemon, read_bcd, count_bits,
    MONEY_0,
)
from env.guide import GameGuide, MILESTONES, NUM_MILESTONES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROM_PATH   = os.path.join(SCRIPT_DIR, "pokemon_blue.gb")
STATE_PATH = os.path.join(SCRIPT_DIR, "init.state")
FRAME_SKIP = 16           # actions per game second (matches training)
TARGET_FPS = 60           # display frames per second
FRAME_TIME = 1.0 / TARGET_FPS

ACTIONS = ['up', 'down', 'left', 'right', 'a', 'b', 'start']

PRESS = {
    'up':    WindowEvent.PRESS_ARROW_UP,
    'down':  WindowEvent.PRESS_ARROW_DOWN,
    'left':  WindowEvent.PRESS_ARROW_LEFT,
    'right': WindowEvent.PRESS_ARROW_RIGHT,
    'a':     WindowEvent.PRESS_BUTTON_A,
    'b':     WindowEvent.PRESS_BUTTON_B,
    'start': WindowEvent.PRESS_BUTTON_START,
}
RELEASE = {
    'up':    WindowEvent.RELEASE_ARROW_UP,
    'down':  WindowEvent.RELEASE_ARROW_DOWN,
    'left':  WindowEvent.RELEASE_ARROW_LEFT,
    'right': WindowEvent.RELEASE_ARROW_RIGHT,
    'a':     WindowEvent.RELEASE_BUTTON_A,
    'b':     WindowEvent.RELEASE_BUTTON_B,
    'start': WindowEvent.RELEASE_BUTTON_START,
}

# ---------------------------------------------------------------------------
# Map names
# ---------------------------------------------------------------------------
MAP_NAMES = {
    0:  "Pallet Town",      1:  "Viridian City",    2:  "Pewter City",
    3:  "Cerulean City",    4:  "Lavender Town",    5:  "Vermilion City",
    6:  "Celadon City",     7:  "Fuchsia City",     8:  "Cinnabar Island",
    9:  "Indigo Plateau",   10: "Saffron City",
    12: "Route 1",          13: "Route 2",          14: "Route 3",
    15: "Route 4",          16: "Route 5",          17: "Route 6",
    18: "Route 7",          19: "Route 8",          20: "Route 9",
    21: "Route 10",         22: "Route 11",         23: "Route 12",
    24: "Route 13",         25: "Route 14",         26: "Route 15",
    27: "Route 16",         28: "Route 17",         29: "Route 18",
    33: "Route 22",         34: "Route 23",         35: "Route 24",
    36: "Route 25",         37: "Red's House 1F",   38: "Red's House 2F",
    39: "Blue's House",     40: "Oak's Lab",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint():
    """Return (path, key) for the most up-to-date checkpoint, or (None, 0).

    Priority:
      1. checkpoints/latest.zip  — written by LiveViewCallback every ~2000 steps;
         preferred when it exists and is newer than the last numbered checkpoint.
      2. Highest-step numbered checkpoint  (pokemon_blue_ppo_N_steps.zip)
      3. checkpoints/final_model.zip
    """
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints")

    # latest.zip — modification time used as the reload key so stream.py
    # detects every new write, not just when the filename changes.
    latest_path = os.path.join(ckpt_dir, "latest.zip")
    latest_mtime = os.path.getmtime(latest_path) if os.path.exists(latest_path) else 0

    # Highest numbered step checkpoint
    files = glob.glob(os.path.join(ckpt_dir, "pokemon_blue_ppo_*_steps.zip"))

    def step_count(path):
        try:
            return int(os.path.basename(path).split("_steps")[0].rsplit("_", 1)[-1])
        except Exception:
            return 0

    best_path, best_steps = None, 0
    if files:
        best_path  = max(files, key=step_count)
        best_steps = step_count(best_path)

    # Use latest.zip if it exists and is newer than the best numbered checkpoint
    if latest_mtime > 0:
        best_mtime = os.path.getmtime(best_path) if best_path else 0
        if latest_mtime >= best_mtime:
            return latest_path, latest_mtime   # key = mtime float

    if best_path:
        return best_path, best_steps

    final = os.path.join(ckpt_dir, "final_model.zip")
    if os.path.exists(final):
        return final, -1
    return None, 0


def build_obs(pyboy, visited_tiles, tile_visit_count, guide, visited_maps):
    """Build the same observation dict the training env produces (17 features)."""
    mem = pyboy.memory

    badges      = read_badges(mem)
    party_count = mem[PARTY_COUNT]
    hp_frac     = read_party_hp_fraction(mem)
    level_sum   = read_party_level_sum(mem)
    owned       = count_bits(mem, POKEDEX_OWNED_START, 19)
    seen        = count_bits(mem, POKEDEX_SEEN_START, 19)
    map_id      = mem[MAP_ID]
    px          = mem[PLAYER_X]
    py_         = mem[PLAYER_Y]
    in_battle   = float(is_in_battle(mem))
    unique      = len(visited_tiles)
    lead        = read_party_pokemon(mem, 0) if party_count > 0 else {}
    money       = read_bcd(mem, MONEY_0, 3)

    # Keep guide state in sync with current game state (no reward needed here)
    guide.update(mem[0xD356], visited_maps)

    features = np.array([
        badges / 8.0,
        party_count / 6.0,
        hp_frac,
        lead.get("level", 0) / 100.0,
        level_sum / 600.0,
        owned / 151.0,
        seen / 151.0,
        map_id / 255.0,
        px / 255.0,
        py_ / 255.0,
        in_battle,
        min(unique / 2000.0, 1.0),
        float(lead.get("status", 1) == 0),
        lead.get("species", 0) / 151.0,
        min(money / 999999.0, 1.0),
        guide.progress,                   # [15] guide milestone progress [0,1]
        guide.target_map_id / 255.0,      # [16] next target map hint
    ], dtype=np.float32)

    rgb    = np.array(pyboy.screen.image)
    gray   = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(gray, (40, 36), interpolation=cv2.INTER_AREA)[:, :, np.newaxis]

    return {"screen": screen, "memory_features": features}


def read_stats(pyboy, visited_tiles):
    mem = pyboy.memory
    return {
        "badges":    read_badges(mem),
        "level_sum": read_party_level_sum(mem),
        "owned":     count_bits(mem, POKEDEX_OWNED_START, 19),
        "tiles":     len(visited_tiles),
        "map_id":    mem[MAP_ID],
        "in_battle": is_in_battle(mem),
        "hp":        read_party_hp_fraction(mem),
    }


def build_heatmap(all_time_tile_count, current_map_id, player_x, player_y):
    """Build a 340×400 exploration heatmap panel for the current map.

    Uses all-time visit counts (accumulated across all episodes since stream
    started) so the map fills in gradually — giving a visual record of
    everywhere the AI has ever been.

    Colour scale (INFERNO): black = never/rarely visited → purple → red →
    orange → yellow/white = most visited.
    """
    PANEL_W = 340
    PANEL_H = 400
    MAP_PX   = 300   # pixel size of the heatmap square

    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    # Header: map name
    map_name = MAP_NAMES.get(current_map_id, f"Map ID {current_map_id}")
    cv2.putText(panel, map_name, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Filter tiles to current map
    map_tiles = {
        (x, y): c
        for (mid, x, y), c in all_time_tile_count.items()
        if mid == current_map_id
    }

    if not map_tiles:
        cv2.putText(panel, "Waiting for map data...", (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    else:
        xs = [x for x, y in map_tiles]
        ys = [y for x, y in map_tiles]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max(max_x - min_x + 1, 1)
        h = max(max_y - min_y + 1, 1)

        # Build count grid and apply log scale (so sparse tiles are visible)
        grid = np.zeros((h, w), dtype=np.float32)
        for (x, y), c in map_tiles.items():
            grid[y - min_y, x - min_x] = c
        grid = np.log1p(grid)

        norm = np.zeros_like(grid, dtype=np.uint8)
        if grid.max() > 0:
            norm = (grid / grid.max() * 255).astype(np.uint8)

        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

        # Scale to fit MAP_PX square, preserving aspect ratio
        scale = min(MAP_PX / w, MAP_PX / h)
        dw = max(int(w * scale), 1)
        dh = max(int(h * scale), 1)
        scaled = cv2.resize(colored, (dw, dh), interpolation=cv2.INTER_NEAREST)

        # Centre in panel below header
        x_off = (PANEL_W - dw) // 2
        y_off = 30 + (MAP_PX - dh) // 2
        panel[y_off:y_off + dh, x_off:x_off + dw] = scaled

        # White dot for current player position
        if min_x <= player_x <= max_x and min_y <= player_y <= max_y:
            dot_x = int((player_x - min_x) * scale) + x_off
            dot_y = int((player_y - min_y) * scale) + y_off
            dot_r = max(2, int(scale * 0.7))
            cv2.circle(panel, (dot_x, dot_y), dot_r + 1, (0, 0, 0), -1)    # outline
            cv2.circle(panel, (dot_x, dot_y), dot_r,     (255, 255, 255), -1)

    # Footer stats
    total_tiles = len(all_time_tile_count)
    total_maps  = len({mid for mid, x, y in all_time_tile_count})
    fy = MAP_PX + 38
    cv2.putText(panel, f"Tiles explored (all-time): {total_tiles:,}", (8, fy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(panel, f"Maps visited: {total_maps}", (8, fy + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    # Colourmap legend
    legend = np.linspace(0, 255, PANEL_W - 20, dtype=np.uint8).reshape(1, -1)
    legend_img = cv2.applyColorMap(legend, cv2.COLORMAP_INFERNO)[0]
    panel[fy + 30:fy + 44, 10:PANEL_W - 10] = legend_img
    cv2.putText(panel, "cold", (10, fy + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
    cv2.putText(panel, "hot", (PANEL_W - 34, fy + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)

    return panel


def build_frame(pyboy, stats, checkpoint_label, model_updated, action_name="",
                episode_step=0, goal_milestone=None):
    """Render the game + HUD into a display frame."""
    rgb     = np.array(pyboy.screen.image)          # (144, 160, 3)
    game    = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    game    = cv2.resize(game, (480, 432), interpolation=cv2.INTER_NEAREST)

    # --- header bar ---
    header = np.zeros((36, 480, 3), dtype=np.uint8)

    # Red live dot
    cv2.circle(header, (14, 18), 7, (0, 0, 220), -1)
    cv2.putText(header, "LIVE", (26, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Checkpoint label (right-aligned)
    ckpt_text = checkpoint_label
    (tw, _), _ = cv2.getTextSize(ckpt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(header, ckpt_text, (480 - tw - 8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 128) if model_updated else (180, 180, 180), 1)

    # Model updated flash
    if model_updated:
        cv2.putText(header, "MODEL UPDATED", (90, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

    # --- footer bar ---
    footer = np.zeros((66, 480, 3), dtype=np.uint8)

    badge_str = f"Badges: {stats['badges']}/8"
    level_str = f"Lv total: {stats['level_sum']}"
    dex_str   = f"Dex: {stats['owned']}"
    tile_str  = f"Tiles: {stats['tiles']}"
    map_str   = f"Map: {stats['map_id']}"
    status    = "BATTLE" if stats["in_battle"] else f"HP {int(stats['hp']*100)}%"

    cv2.putText(footer, f"{badge_str}   {level_str}   {dex_str}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    step_str = f"Step: {episode_step}/4096   Action: {action_name}"
    cv2.putText(footer, f"{tile_str}   {map_str}   {status}   {step_str}", (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 200, 255) if stats["in_battle"] else (180, 180, 180), 1)

    # Goal line
    if goal_milestone is not None:
        name, _, _ = MILESTONES[goal_milestone]
        goal_text = f"GOAL: {name.replace('_', ' ').title()}  ({goal_milestone + 1}/{NUM_MILESTONES})  [H=clear]"
        cv2.putText(footer, goal_text, (8, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
    else:
        cv2.putText(footer, "G=set goal   H=clear goal   Q=quit", (8, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)

    return np.vstack([header, game, footer])   # (534, 480, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (1=normal, 2=2x, 0=unlimited)")
    parser.add_argument("--check-interval", type=float, default=10.0,
                        help="Seconds between model reload checks (default 10)")
    args = parser.parse_args()

    # Wait for a checkpoint to exist
    print("Waiting for a checkpoint from train.py...")
    current_path, current_steps = find_latest_checkpoint()
    while current_path is None:
        time.sleep(5)
        current_path, current_steps = find_latest_checkpoint()

    print(f"Loading: {current_path}")
    model = PPO.load(current_path, device="cpu")

    # Set up PyBoy
    pyboy = PyBoy(ROM_PATH, window="null")
    pyboy.set_emulation_speed(0)   # we throttle manually for exact FPS control

    def reset_game():
        with open(STATE_PATH, "rb") as f:
            import io
            pyboy.load_state(io.BytesIO(f.read()))
        pyboy.tick(4, False)
        return set(), {}   # visited_tiles, tile_visit_count

    visited_tiles, tile_visit_count = reset_game()
    episode_step = 0
    MAX_EPISODE_STEPS = 4096   # match training episode length

    # Guide + visited_maps — mirror the env's global state so obs[15-16] match
    guide        = GameGuide()
    visited_maps = set()

    # All-time tile counts — never reset, accumulate across every episode
    # so the heatmap shows the AI's full exploration history.
    all_time_tile_count = {}

    # OpenCV windows
    cv2.namedWindow("Pokemon Blue AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pokemon Blue AI", 480, 534)
    cv2.namedWindow("Exploration Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Exploration Map", 340, 400)

    last_check   = time.time()
    model_flash  = 0.0   # timestamp of last model update (for flash effect)
    goal_milestone: int | None = None   # None = guide auto; 0-22 = user override

    def make_label(path, key):
        if "latest" in os.path.basename(path):
            return "latest  (live)"
        if isinstance(key, int) and key > 0:
            return f"{key:,} steps"
        return "final model"

    ckpt_label = make_label(current_path, current_steps)

    print(f"Streaming at {args.speed}x speed. Press Q in the window to quit.")
    print(f"Reloading model every {args.check_interval}s. (latest.zip updates every ~2k training steps)")
    print("Press G to cycle goal destination, H to clear goal.")

    speed_multiplier = max(args.speed, 0.1)

    try:
        while True:
            # --- Check for a newer checkpoint ---
            if time.time() - last_check > args.check_interval:
                last_check = time.time()
                new_path, new_steps = find_latest_checkpoint()
                if new_path and new_steps != current_steps:
                    print(f"Reloading model: {os.path.basename(new_path)}")
                    try:
                        model         = PPO.load(new_path, device="cpu")
                        current_path  = new_path
                        current_steps = new_steps
                        ckpt_label    = make_label(new_path, new_steps)
                        model_flash   = time.time()
                    except Exception as e:
                        print(f"  reload failed ({e}), keeping current model")

            model_updated = (time.time() - model_flash) < 3.0

            # --- Get AI action ---
            obs = build_obs(pyboy, visited_tiles, tile_visit_count, guide, visited_maps)
            # If user set a goal, override the guide hint features the model reads
            if goal_milestone is not None:
                obs["memory_features"][15] = goal_milestone / (NUM_MILESTONES - 1)
                _, target_map, _ = MILESTONES[goal_milestone]
                obs["memory_features"][16] = target_map / 255.0
            action, _ = model.predict(obs, deterministic=False)  # sample like training does
            btn       = ACTIONS[int(action)]
            stats        = read_stats(pyboy, visited_tiles)

            # Update visited tiles (episode + all-time) and global map set
            mem  = pyboy.memory
            tile = (mem[MAP_ID], mem[PLAYER_X], mem[PLAYER_Y])
            visited_tiles.add(tile)
            visited_maps.add(mem[MAP_ID])
            tile_visit_count[tile]      = tile_visit_count.get(tile, 0) + 1
            all_time_tile_count[tile]   = all_time_tile_count.get(tile, 0) + 1

            # --- Execute action one frame at a time for smooth display ---
            for frame_num in range(FRAME_SKIP):
                t_start = time.perf_counter()

                if frame_num == 0:
                    pyboy.send_input(PRESS[btn])
                if frame_num == 8:
                    pyboy.send_input(RELEASE[btn])

                pyboy.tick(1, True)   # render=True keeps screen buffer fresh

                display = build_frame(pyboy, stats, ckpt_label, model_updated, btn, episode_step, goal_milestone)
                cv2.imshow("Pokemon Blue AI", display)

                # Update the exploration map once per action (last frame only)
                if frame_num == FRAME_SKIP - 1:
                    heatmap = build_heatmap(
                        all_time_tile_count,
                        mem[MAP_ID], mem[PLAYER_X], mem[PLAYER_Y],
                    )
                    cv2.imshow("Exploration Map", heatmap)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):   # Q or ESC
                    raise KeyboardInterrupt
                elif key == ord('g'):
                    goal_milestone = 0 if goal_milestone is None else (goal_milestone + 1) % NUM_MILESTONES
                    name, _, _ = MILESTONES[goal_milestone]
                    print(f"Goal set: {name.replace('_', ' ').title()} (milestone {goal_milestone})")
                elif key == ord('h'):
                    goal_milestone = None
                    print("Goal cleared.")

                # Throttle to target FPS scaled by speed multiplier
                elapsed = time.perf_counter() - t_start
                sleep   = (FRAME_TIME / speed_multiplier) - elapsed
                if sleep > 0:
                    time.sleep(sleep)

            episode_step += 1
            if episode_step >= MAX_EPISODE_STEPS:
                visited_tiles, tile_visit_count = reset_game()
                episode_step = 0

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    pyboy.stop()
    print("Stream ended.")


if __name__ == "__main__":
    main()
