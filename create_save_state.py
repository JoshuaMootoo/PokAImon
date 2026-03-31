"""One-time setup script: play through the Pokemon Blue intro manually,
then press Enter to save the starting state used by the RL agent.

Run:
    python create_save_state.py

The saved state skips the unskippable Oak monologue on every training reset,
saving ~30 seconds per episode.  Save when you have control on the first map
(Pallet Town, just outside your house).
"""

import os
import sys
from pyboy import PyBoy

ROM_PATH   = "pokemon_blue.gb"
STATE_PATH = "init.state"


def main():
    if not os.path.exists(ROM_PATH):
        print(f"ERROR: ROM not found at '{ROM_PATH}'")
        print("Place your Pokemon Blue ROM in the project root and name it 'pokemon_blue.gb'.")
        sys.exit(1)

    print("Opening Pokemon Blue...")
    print("Play through the intro until you have control of your character.")
    print("Recommended save point: standing in Pallet Town, just outside your house.")
    print()
    print("Press Enter here (in this terminal) when you are ready to save the state.")

    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1)

    # Run the game loop until the user presses Enter in the terminal
    import threading

    save_now = threading.Event()

    def wait_for_enter():
        input()
        save_now.set()

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    while not save_now.is_set():
        pyboy.tick(1, True)

    # Save state
    with open(STATE_PATH, "wb") as f:
        pyboy.save_state(f)

    print(f"\nState saved to '{STATE_PATH}'.")
    print("You can now run 'python train.py' to start training.")

    pyboy.stop()


if __name__ == "__main__":
    main()
