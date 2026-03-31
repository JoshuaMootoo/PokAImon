"""Interactive live viewer — watch the AI play and take control any time.

Controls:
  Arrow keys / WASD  — take control (move)
  Z or comma (,)     — A button
  X or period (.)    — B button
  Enter              — Start
  Q / ESC            — quit

Hold a key to take over. Release it and the AI resumes immediately.

Usage:
    python watch.py                        # auto-loads latest checkpoint
    python watch.py --model checkpoints/pokemon_blue_ppo_5000000_steps
    python watch.py --speed 2             # run at 2x Game Boy speed
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
    MAP_ID, PLAYER_X, PLAYER_Y, PARTY_COUNT, MONEY_0,
    POKEDEX_OWNED_START, POKEDEX_SEEN_START,
    read_badges, is_in_battle, read_party_hp_fraction,
    read_party_level_sum, read_party_pokemon, read_bcd, count_bits,
)

ROM_PATH   = "pokemon_blue.gb"
STATE_PATH = "init.state"
ACTIONS    = ['up', 'down', 'left', 'right', 'a', 'b', 'start']

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

# OpenCV key codes → game button  (WASD + arrow keys + action keys)
KEY_MAP = {
    ord('w'): 'up',    ord('s'): 'down',
    ord('a'): 'left',  ord('d'): 'right',
    ord('z'): 'a',     ord(','): 'a',
    ord('x'): 'b',     ord('.'): 'b',
    13:       'start',          # Enter
    # Arrow keys on Windows via OpenCV
    2490368: 'up',    2621440: 'down',
    2424832: 'left',  2555904: 'right',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint() -> str | None:
    files = glob.glob("checkpoints/pokemon_blue_ppo_*_steps.zip")
    if files:
        def step_num(p):
            try:
                return int(p.replace('\\', '/').split('_steps')[0].split('_')[-1])
            except Exception:
                return 0
        return max(files, key=step_num)
    final = "checkpoints/final_model.zip"
    return final if os.path.exists(final) else None


def build_obs(pyboy, visited_tiles: set) -> dict:
    mem         = pyboy.memory
    party_count = mem[PARTY_COUNT]
    lead        = read_party_pokemon(mem, 0) if party_count > 0 else {}

    features = np.array([
        read_badges(mem)                              / 8.0,
        party_count                                   / 6.0,
        read_party_hp_fraction(mem),
        lead.get('level', 0)                          / 100.0,
        read_party_level_sum(mem)                     / 600.0,
        count_bits(mem, POKEDEX_OWNED_START, 19)      / 151.0,
        count_bits(mem, POKEDEX_SEEN_START, 19)       / 151.0,
        mem[MAP_ID]                                   / 255.0,
        mem[PLAYER_X]                                 / 255.0,
        mem[PLAYER_Y]                                 / 255.0,
        float(is_in_battle(mem)),
        min(len(visited_tiles) / 2000.0, 1.0),
        float(lead.get('status', 1) == 0),
        lead.get('species', 0)                        / 151.0,
        min(read_bcd(mem, MONEY_0, 3) / 999999.0, 1.0),
        0.0,
    ], dtype=np.float32)

    gray   = cv2.cvtColor(np.array(pyboy.screen.image), cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(gray, (40, 36), interpolation=cv2.INTER_AREA)[:, :, np.newaxis]
    return {'screen': screen, 'memory_features': features}


def draw_hud(display: np.ndarray, mode: str, btn: str, stats: dict) -> np.ndarray:
    """Add a HUD bar below the game frame."""
    hud = np.zeros((72, display.shape[1], 3), dtype=np.uint8)

    # Mode pill
    if mode == 'AI':
        pill_color = (30, 160, 30)
        label      = f'AI  > {btn}'
    else:
        pill_color = (30, 100, 220)
        label      = f'YOU > {btn}'

    cv2.rectangle(hud, (8, 6), (200, 30), pill_color, -1)
    cv2.putText(hud, label, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Stats row
    stat_line = (f"Badges {stats['badges']}/8    "
                 f"Levels {stats['levels']}    "
                 f"Pokedex {stats['pokedex']}/151    "
                 f"Tiles {stats['tiles']}")
    cv2.putText(hud, stat_line, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)

    # Controls hint
    hint = "Hold WASD/arrows=move  Z=A  X=B  Enter=Start  Q=quit"
    cv2.putText(hud, hint, (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1)

    return np.vstack([display, hud])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,   default=None)
    parser.add_argument('--speed',      type=float, default=1.0,
                        help='Emulation speed (1=normal, 2=2x, 0=unlimited)')
    parser.add_argument('--frame-skip', type=int,   default=24)
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if model_path is None:
        model_path = find_latest_checkpoint()
        if model_path is None:
            print('No checkpoint found in checkpoints/. Run train.py first.')
            return
        print(f'Auto-loaded latest checkpoint: {model_path}')
    if not model_path.endswith('.zip'):
        model_path += '.zip'
    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        return

    for path, name in [(ROM_PATH, 'ROM'), (STATE_PATH, 'save state')]:
        if not os.path.exists(path):
            print(f'ERROR: {name} not found at {path!r}')
            return

    print(f'Loading model …')
    model = PPO.load(model_path, device='cpu')

    pyboy = PyBoy(ROM_PATH, window='null')
    pyboy.set_emulation_speed(args.speed)
    with open(STATE_PATH, 'rb') as f:
        pyboy.load_state(f)
    pyboy.tick(4, False)

    visited_tiles: set = set()
    frame_skip         = args.frame_skip
    mode               = 'AI'
    btn                = 'none'
    stats              = {'badges': 0, 'levels': 0, 'pokedex': 0, 'tiles': 0}

    cv2.namedWindow('Pokemon Blue AI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pokemon Blue AI', 480, 504)

    print('\nWatching … hold a key to take control, release to hand back to AI.\n')

    try:
        while True:
            mem  = pyboy.memory
            tile = (mem[MAP_ID], mem[PLAYER_X], mem[PLAYER_Y])
            visited_tiles.add(tile)

            stats = {
                'badges':  read_badges(mem),
                'levels':  read_party_level_sum(mem),
                'pokedex': count_bits(mem, POKEDEX_OWNED_START, 19),
                'tiles':   len(visited_tiles),
            }

            # --- Decide action ---
            key        = cv2.waitKey(1)
            human_btn  = None
            if key != -1:
                lk = key & 0xFF
                if lk in (ord('q'), 27):   # Q or ESC
                    break
                human_btn = KEY_MAP.get(key) or KEY_MAP.get(lk)

            if human_btn is not None:
                mode = 'YOU'
                btn  = human_btn
            else:
                mode      = 'AI'
                obs       = build_obs(pyboy, visited_tiles)
                ai_action, _ = model.predict(obs, deterministic=True)
                btn       = ACTIONS[int(ai_action)]

            # --- Execute action — render every frame at target speed ---
            frame_duration = 1.0 / (60.0 * args.speed) if args.speed > 0 else 0.0
            pyboy.send_input(PRESS[btn])
            for i in range(frame_skip):
                t0 = time.perf_counter()
                pyboy.tick(1, True)          # update screen buffer every frame
                rgb     = np.array(pyboy.screen.image)
                bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                bgr     = cv2.resize(bgr, (480, 432), interpolation=cv2.INTER_NEAREST)
                display = draw_hud(bgr, mode, btn, stats)
                cv2.imshow('Pokemon Blue AI', display)
                cv2.waitKey(1)
                # throttle to target fps
                sleep_t = frame_duration - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
            pyboy.send_input(RELEASE[btn])

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    pyboy.stop()

    print(f'\nSession ended.')
    print(f'Badges: {stats["badges"]}/8  |  '
          f'Levels: {stats["levels"]}  |  '
          f'Pokedex: {stats["pokedex"]}/151  |  '
          f'Tiles: {stats["tiles"]}')


if __name__ == '__main__':
    main()
