"""Diagnostic script — run this to check if buttons and save state are working.

Usage:
    python test_env.py
"""

import time
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from env.memory import MAP_ID, PLAYER_X, PLAYER_Y, PARTY_COUNT, read_badges

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

ROM_PATH   = "pokemon_blue.gb"
STATE_PATH = "init.state"

ACTIONS = ['up', 'down', 'left', 'right', 'a', 'b', 'start']


def main():
    print("=== Pokemon Blue Environment Diagnostic ===\n")

    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1)

    with open(STATE_PATH, "rb") as f:
        pyboy.load_state(f)

    pyboy.tick(10, True)

    mem = pyboy.memory
    start_map = mem[MAP_ID]
    start_x   = mem[PLAYER_X]
    start_y   = mem[PLAYER_Y]
    party     = mem[PARTY_COUNT]
    badges    = read_badges(mem)

    print(f"Save state loaded:")
    print(f"  Map ID    : {start_map} (0=Pallet Town, 37=Bedroom, 38=Living Room)")
    print(f"  Position  : X={start_x}, Y={start_y}")
    print(f"  Party size: {party}  (should be > 0 — you need your starter!)")
    print(f"  Badges    : {badges}")
    print()

    if party == 0:
        print("WARNING: No Pokemon in party! Get your starter before saving the state.")
        print("Run 'python create_save_state.py' again.\n")

    if start_map == 37 or start_map == 38:
        print("WARNING: Save state is inside the house (map 37/38).")
        print("Re-run 'python create_save_state.py' and save while standing OUTSIDE.\n")

    # --- Button test ---
    print("Testing button presses (watch the window)...")
    print()

    results = {}
    for btn in ['right', 'left', 'down', 'up', 'a', 'b', 'start']:
        before_x = mem[PLAYER_X]
        before_y = mem[PLAYER_Y]

        pyboy.send_input(PRESS[btn])
        pyboy.tick(8, True)
        pyboy.send_input(RELEASE[btn])
        pyboy.tick(16, True)

        after_x = mem[PLAYER_X]
        after_y = mem[PLAYER_Y]

        moved = (before_x != after_x) or (before_y != after_y)
        results[btn] = moved
        status = "MOVED" if moved else "no movement"
        print(f"  Button '{btn:>6}': {status}  (X {before_x}->{after_x}, Y {before_y}->{after_y})")

        time.sleep(0.1)

    print()
    any_moved = any(results.values())
    if any_moved:
        print("Buttons are working correctly.")
    else:
        print("WARNING: No movement detected for any button!")
        print("The character may be blocked by a wall, in a menu, or button names are wrong.")
        print("Check: is the character facing an open area in the window?")

    print()
    print("Close the window to exit.")

    # Keep window open
    try:
        while True:
            pyboy.tick(1, True)
    except KeyboardInterrupt:
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()
