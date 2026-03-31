"""Watch a trained PPO agent play Pokemon Blue in a live Game Boy window.

Usage:
    python evaluate.py --model checkpoints/pokemon_blue_ppo_5000000_steps
    python evaluate.py --model checkpoints/final_model --speed 2
    python evaluate.py --model checkpoints/final_model --speed 0.5
"""

import argparse
import os

import cv2
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent

PRESS = {
    'up':    WindowEvent.PRESS_ARROW_UP,    'down':  WindowEvent.PRESS_ARROW_DOWN,
    'left':  WindowEvent.PRESS_ARROW_LEFT,  'right': WindowEvent.PRESS_ARROW_RIGHT,
    'a':     WindowEvent.PRESS_BUTTON_A,    'b':     WindowEvent.PRESS_BUTTON_B,
    'start': WindowEvent.PRESS_BUTTON_START,
}
RELEASE = {
    'up':    WindowEvent.RELEASE_ARROW_UP,    'down':  WindowEvent.RELEASE_ARROW_DOWN,
    'left':  WindowEvent.RELEASE_ARROW_LEFT,  'right': WindowEvent.RELEASE_ARROW_RIGHT,
    'a':     WindowEvent.RELEASE_BUTTON_A,    'b':     WindowEvent.RELEASE_BUTTON_B,
    'start': WindowEvent.RELEASE_BUTTON_START,
}
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
SCREEN_SIZE = (36, 40)  # must match training


def build_obs(pyboy, visited_tiles: set) -> dict:
    """Build the same observation dict the model was trained on."""
    mem = pyboy.memory

    badges      = read_badges(mem)
    party_count = mem[PARTY_COUNT]
    hp_frac     = read_party_hp_fraction(mem)
    level_sum   = read_party_level_sum(mem)
    owned       = count_bits(mem, POKEDEX_OWNED_START, 19)
    seen        = count_bits(mem, POKEDEX_SEEN_START, 19)
    map_id      = mem[MAP_ID]
    px          = mem[PLAYER_X]
    py          = mem[PLAYER_Y]
    in_battle   = float(is_in_battle(mem))
    unique      = len(visited_tiles)
    lead        = read_party_pokemon(mem, 0) if party_count > 0 else {}
    money       = read_bcd(mem, MONEY_0, 3)

    features = np.array([
        badges                        / 8.0,
        party_count                   / 6.0,
        hp_frac,
        lead.get("level", 0)          / 100.0,
        level_sum                     / 600.0,
        owned                         / 151.0,
        seen                          / 151.0,
        map_id                        / 255.0,
        px                            / 255.0,
        py                            / 255.0,
        in_battle,
        min(unique / 2000.0, 1.0),
        float(lead.get("status", 1) == 0),
        lead.get("species", 0)        / 151.0,
        min(money / 999999.0, 1.0),
        0.0,
    ], dtype=np.float32)

    # Screen observation
    rgb    = np.array(pyboy.screen.image)
    gray   = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W   = SCREEN_SIZE
    screen = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)[:, :, np.newaxis]

    return {"screen": screen, "memory_features": features}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to trained model (.zip extension optional).")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Emulation speed multiplier (1=normal, 2=2x, 0=unlimited).")
    parser.add_argument("--frame-skip", type=int, default=24,
                        help="Frames advanced per action (should match training).")
    args = parser.parse_args()

    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path += ".zip"

    for path, name in [(ROM_PATH, "ROM"), (STATE_PATH, "save state"), (model_path, "model")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at '{path}'.")
            raise SystemExit(1)

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    print("Opening game window... (close window or press Ctrl+C to quit)")
    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(args.speed)

    with open(STATE_PATH, "rb") as f:
        pyboy.load_state(f)

    visited_tiles: set = set()
    step    = 0
    badges  = 0

    try:
        while True:
            mem  = pyboy.memory
            tile = (mem[MAP_ID], mem[PLAYER_X], mem[PLAYER_Y])
            visited_tiles.add(tile)

            obs            = build_obs(pyboy, visited_tiles)
            action, _state = model.predict(obs, deterministic=True)
            btn            = ACTIONS[int(action)]

            pyboy.send_input(PRESS[btn])
            pyboy.tick(8, True)
            pyboy.send_input(RELEASE[btn])
            pyboy.tick(args.frame_skip - 8, True)  # True = update screen buffer

            step += 1

            new_badges = read_badges(mem)
            if new_badges > badges:
                badges = new_badges
                print(f"  *** Badge obtained! Total: {badges}/8 ***")

            if step % 500 == 0:
                print(
                    f"Step {step:>6} | "
                    f"Tiles: {len(visited_tiles):>4} | "
                    f"Badges: {badges} | "
                    f"Levels: {read_party_level_sum(mem):>3} | "
                    f"Pokedex: {count_bits(mem, POKEDEX_OWNED_START, 19):>3}"
                )

    except KeyboardInterrupt:
        print("\nStopped.")

    pyboy.stop()


if __name__ == "__main__":
    main()
