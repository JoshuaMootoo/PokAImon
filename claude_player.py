"""claude_player.py — Claude AI plays Pokémon Blue in real time.

Claude reads the current game screen and memory state every action, narrates
its thinking out loud, and executes the chosen button press.  You can send
hints by typing in this terminal and pressing Enter.

Usage:
    python claude_player.py
    python claude_player.py --speed 1.0        # display speed (1=normal, 2=2x)
    python claude_player.py --frame-skip 16    # frames per action
    python claude_player.py --model claude-opus-4-6   # use a different model

Controls (OpenCV window):
    Q / ESC — quit

Hints:
    Type anything in the terminal and press Enter.
    Claude will read it on its next turn.
"""

import argparse
import base64
import io
import os
import sys
import threading
import time

import anthropic
import cv2
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from env.memory import (
    MAP_ID, PLAYER_X, PLAYER_Y, PARTY_COUNT, MONEY_0, BADGES,
    POKEDEX_OWNED_START, POKEDEX_SEEN_START,
    read_badges, is_in_battle, read_party_hp_fraction,
    read_party_level_sum, read_party_pokemon, read_bcd, count_bits,
)
from env.guide import MILESTONES, NUM_MILESTONES, GameGuide

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROM_PATH   = os.path.join(SCRIPT_DIR, "pokemon_blue.gb")
STATE_PATH = os.path.join(SCRIPT_DIR, "init.state")

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
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

MAP_NAMES = {
    0:   "Pallet Town",       1:   "Viridian City",     2:   "Pewter City",
    3:   "Cerulean City",     4:   "Lavender Town",     5:   "Vermilion City",
    6:   "Celadon City",      7:   "Fuchsia City",      8:   "Cinnabar Island",
    9:   "Indigo Plateau",    10:  "Saffron City",
    12:  "Route 1",           13:  "Route 2",           14:  "Route 3",
    15:  "Route 4",           16:  "Route 5",           17:  "Route 6",
    18:  "Route 7",           19:  "Route 8",           20:  "Route 9",
    21:  "Route 10",          22:  "Route 11",          23:  "Route 12",
    24:  "Route 13",          25:  "Route 14",          26:  "Route 15",
    27:  "Route 16",          28:  "Route 17",          29:  "Route 18",
    33:  "Route 22",          34:  "Route 23",          35:  "Route 24",
    36:  "Route 25",          37:  "Viridian Forest",   38:  "Mt. Moon 1F",
    39:  "Mt. Moon B1F",      40:  "Mt. Moon B2F",      57:  "Victory Road",
    64:  "S.S. Anne",         92:  "Silph Co.",          103: "Pokemon Mansion",
    107: "Safari Zone",       113: "Viridian Gym",       114: "Pewter Gym",
    115: "Cerulean Gym",      116: "Vermilion Gym",      117: "Celadon Gym",
    118: "Fuchsia Gym",       119: "Saffron Gym",        120: "Cinnabar Gym",
    121: "E4: Lorelei",       122: "E4: Bruno",          123: "E4: Agatha",
    124: "E4: Lance",         125: "Champion Room",      126: "Hall of Fame",
    127: "Oak's Lab",
}

WINDOW_TITLE = "Claude Plays Pokemon Blue"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an AI agent playing Pokémon Blue in real time.
Your primary goal is to learn how to play the game effectively through exploration, feedback, and iterative improvement.

SYSTEM BEHAVIOUR:

1. REAL-TIME LOOP
- Continuously observe the current game state (screen + memory data).
- After each action, briefly reflect on what happened and what your next intention is.

2. LIVE COMMENTARY MODE
- Output short, stream-like commentary of your thinking.
- Keep it concise, like a livestream narrator.

3. LEARNING
- Prefer actions that previously led to progress (winning battles, gaining XP, new areas).
- Avoid clearly bad actions (walking into walls, losing fights repeatedly).

4. HUMAN HINT SYSTEM (VERY IMPORTANT)
- You may receive a HINT at any time. Hints are suggestions, not commands.
- If useful, incorporate it into your strategy.
- If not, briefly say why you're ignoring it.

5. PRIORITY ORDER
1. Survival (don't black out)
2. Progress (badges, story)
3. Growth (XP, leveling)
4. Exploration

6. PERSONALITY
- Be slightly expressive and curious.
- Sound like someone learning, not a perfect bot.
- Occasionally question your own decisions.

OUTPUT FORMAT — respond with EXACTLY these three lines, nothing else:
ACTION: <one of: up, down, left, right, a, b, start>
THOUGHT: <one sentence of internal reasoning>
COMMENTARY: <one sentence of stream-style narration>
"""

# ---------------------------------------------------------------------------
# Hint queue  (background thread reads stdin)
# ---------------------------------------------------------------------------
_hint_queue: list[str] = []
_hint_lock  = threading.Lock()


def _hint_reader():
    """Background thread: collect hints typed in the terminal."""
    print("Hint mode: type a message and press Enter to guide Claude.", flush=True)
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            hint = line.strip()
            if hint:
                with _hint_lock:
                    _hint_queue.append(hint)
                print(f"  Hint queued: {hint!r}", flush=True)
        except Exception:
            break


def pop_hint() -> str | None:
    with _hint_lock:
        return _hint_queue.pop(0) if _hint_queue else None


# ---------------------------------------------------------------------------
# Game-state helpers
# ---------------------------------------------------------------------------

def capture_screen_b64(pyboy) -> str:
    """Return the current Game Boy screen as a base64-encoded PNG string."""
    rgb    = np.array(pyboy.screen.image)                         # (144, 160, 3) RGB
    bgr    = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    scaled = cv2.resize(bgr, (480, 432), interpolation=cv2.INTER_NEAREST)
    _, buf = cv2.imencode('.png', scaled)
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def build_state_text(mem, visited_tiles: set, guide: GameGuide,
                     last_action: str, step: int) -> str:
    """Build the structured GAME_STATE text sent to Claude."""
    badges      = read_badges(mem)
    party_count = mem[PARTY_COUNT]
    map_id      = mem[MAP_ID]
    px, py_     = mem[PLAYER_X], mem[PLAYER_Y]
    in_battle   = is_in_battle(mem)
    money       = read_bcd(mem, MONEY_0, 3)
    owned       = count_bits(mem, POKEDEX_OWNED_START, 19)
    seen_count  = count_bits(mem, POKEDEX_SEEN_START, 19)
    hp          = read_party_hp_fraction(mem)
    map_name    = MAP_NAMES.get(map_id, f"Unknown (Map ID {map_id})")

    party_lines = []
    for i in range(min(party_count, 6)):
        mon = read_party_pokemon(mem, i)
        if mon['species'] > 0:
            s = "fainted" if mon['cur_hp'] == 0 else ("OK" if mon['status'] == 0 else "status condition")
            party_lines.append(f"  [{i+1}] Lv.{mon['level']}  HP {mon['cur_hp']}/{mon['max_hp']}  [{s}]")

    milestone_name, target_map, _ = MILESTONES[guide.milestone_index]
    target_name = MAP_NAMES.get(target_map, f"Map {target_map}")

    party_block = "\n".join(party_lines) if party_lines else "  (empty)"
    return (
        f"GAME_STATE:\n"
        f"Step: {step}\n"
        f"Location: {map_name}\n"
        f"Position: ({px}, {py_})\n"
        f"In battle: {'YES' if in_battle else 'No'}\n"
        f"Badges: {badges}/8\n"
        f"Money: ${money:,}\n"
        f"Party HP: {int(hp * 100)}%  ({party_count} Pokémon)\n"
        f"{party_block}\n"
        f"Pokédex: {owned} owned / {seen_count} seen\n"
        f"Tiles explored: {len(visited_tiles)}\n"
        f"Last action: {last_action or 'none'}\n"
        f"Guide: {milestone_name} → heading to {target_name}"
    )


def read_stats(mem, visited_tiles: set) -> dict:
    return {
        "badges":    read_badges(mem),
        "levels":    read_party_level_sum(mem),
        "owned":     count_bits(mem, POKEDEX_OWNED_START, 19),
        "tiles":     len(visited_tiles),
        "map_id":    mem[MAP_ID],
        "in_battle": is_in_battle(mem),
        "hp":        read_party_hp_fraction(mem),
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(text: str) -> dict:
    result = {"action": "a", "thought": "", "commentary": ""}
    for line in text.strip().splitlines():
        upper = line.upper().lstrip()
        if upper.startswith("ACTION:"):
            raw = line.split(":", 1)[1].strip().lower()
            if raw in ACTIONS:
                result["action"] = raw
        elif upper.startswith("THOUGHT:"):
            result["thought"]    = line.split(":", 1)[1].strip()
        elif upper.startswith("COMMENTARY:"):
            result["commentary"] = line.split(":", 1)[1].strip()
    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 68) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).lstrip()
    if cur:
        lines.append(cur)
    return lines or [""]


def build_display(pyboy, stats: dict, commentary_lines: list[str],
                  last_thought: str, thinking: bool, active_hint: str | None) -> np.ndarray:
    """Compose the full display frame: header + game + stats + commentary + hint."""
    rgb  = np.array(pyboy.screen.image)
    game = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    game = cv2.resize(game, (480, 432), interpolation=cv2.INTER_NEAREST)

    # Header (36 px)
    header = np.zeros((36, 480, 3), dtype=np.uint8)
    dot_col = (40, 140, 255) if thinking else (30, 200, 30)
    cv2.circle(header, (14, 18), 7, dot_col, -1)
    label = "THINKING..." if thinking else "CLAUDE PLAYS"
    cv2.putText(header, label, (26, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    map_str = MAP_NAMES.get(stats["map_id"], f"Map {stats['map_id']}")
    (tw, _), _ = cv2.getTextSize(map_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(header, map_str, (480 - tw - 8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # Stats bar (28 px)
    stats_bar = np.zeros((28, 480, 3), dtype=np.uint8)
    status = "BATTLE" if stats["in_battle"] else f"HP {int(stats['hp'] * 100)}%"
    cv2.putText(stats_bar,
        f"Badges: {stats['badges']}/8   Lv: {stats['levels']}   "
        f"Dex: {stats['owned']}/151   Tiles: {stats['tiles']}   {status}",
        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Commentary panel — last 3 lines (66 px)
    comm = np.zeros((66, 480, 3), dtype=np.uint8)
    recent = commentary_lines[-3:]
    for i, line in enumerate(recent):
        brightness = int(140 + 115 * (i + 1) / max(len(recent), 1))
        colour = (brightness, brightness, 60)
        cv2.putText(comm, line[:70], (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)

    # Thought bar (24 px)
    thought_bar = np.zeros((24, 480, 3), dtype=np.uint8)
    if last_thought:
        cv2.putText(thought_bar, f"> {last_thought[:74]}", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 180, 255), 1)

    # Hint bar (22 px)
    hint_bar = np.zeros((22, 480, 3), dtype=np.uint8)
    if active_hint:
        cv2.putText(hint_bar, f"HINT: {active_hint[:66]}", (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
    else:
        cv2.putText(hint_bar, "Type a hint in this terminal and press Enter to guide Claude",
                    (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (55, 55, 55), 1)

    # Total: 36 + 432 + 28 + 66 + 24 + 22 = 608 px
    return np.vstack([header, game, stats_bar, comm, thought_bar, hint_bar])


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

def execute_action(pyboy, btn: str, frame_skip: int, speed: float,
                   display_callback) -> None:
    """Press and hold a button for 8 frames, release, then idle — rendering each frame."""
    frame_time = (1.0 / (60.0 * speed)) if speed > 0 else 0.0
    pyboy.send_input(PRESS[btn])
    for i in range(frame_skip):
        t0 = time.perf_counter()
        pyboy.tick(1, True)
        cv2.imshow(WINDOW_TITLE, display_callback())
        cv2.waitKey(1)
        if i == 7:
            pyboy.send_input(RELEASE[btn])
        sleep_t = frame_time - (time.perf_counter() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)
    if frame_skip <= 7:
        pyboy.send_input(RELEASE[btn])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Claude AI plays Pokémon Blue")
    parser.add_argument("--speed",      type=float, default=1.0,
                        help="Display speed multiplier (1=normal, 2=2x, 0=unlimited)")
    parser.add_argument("--frame-skip", type=int,   default=16,
                        help="Frames per action (default 16, matches training)")
    parser.add_argument("--model",      type=str,   default="claude-sonnet-4-6",
                        help="Claude model ID (default: claude-sonnet-4-6)")
    parser.add_argument("--history",    type=int,   default=6,
                        help="Number of past turns kept in context (default 6)")
    args = parser.parse_args()

    for path, name in [(ROM_PATH, "ROM"), (STATE_PATH, "save state")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path!r}")
            print("Run python create_save_state.py first to create init.state")
            return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Export it with: export ANTHROPIC_API_KEY=sk-ant-...")
        return

    # Start hint reader
    threading.Thread(target=_hint_reader, daemon=True).start()

    client = anthropic.Anthropic()
    print(f"\nClaude model : {args.model}")
    print(f"Speed        : {args.speed}x")
    print(f"Frame skip   : {args.frame_skip}")
    print(f"Window title : {WINDOW_TITLE!r}")
    print("\nStarting...\n")

    pyboy = PyBoy(ROM_PATH, window="null")
    pyboy.set_emulation_speed(0)   # throttled manually per-frame
    with open(STATE_PATH, "rb") as f:
        pyboy.load_state(io.BytesIO(f.read()))
    pyboy.tick(4, False)

    guide        = GameGuide()
    visited_tiles: set = set()
    visited_maps: set  = set()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 480, 608)

    # Rolling commentary shown in the window
    commentary_lines: list[str] = ["Booting up... let's see where we are."]
    last_thought = ""
    last_action  = ""
    active_hint: str | None = None
    step = 0

    # Conversation history — text only (no images) for past turns
    # to keep context manageable.  Current turn always gets the fresh image.
    message_history: list[dict] = []

    try:
        while True:
            # --- Update tracking state ---
            mem  = pyboy.memory
            tile = (mem[MAP_ID], mem[PLAYER_X], mem[PLAYER_Y])
            visited_tiles.add(tile)
            visited_maps.add(mem[MAP_ID])
            guide.update(mem[BADGES], visited_maps)   # raw bitmask

            stats = read_stats(mem, visited_tiles)

            # --- Show thinking indicator while API call runs ---
            thinking_frame = build_display(
                pyboy, stats, commentary_lines, last_thought, True, active_hint)
            cv2.imshow(WINDOW_TITLE, thinking_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("Quit.")
                break

            # --- Collect hint if one arrived ---
            hint = pop_hint()
            if hint:
                active_hint = hint

            # --- Build message for Claude ---
            state_text = build_state_text(mem, visited_tiles, guide, last_action, step)
            if active_hint:
                state_text += f"\nHINT: {active_hint}"

            screen_b64 = capture_screen_b64(pyboy)

            current_user_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screen_b64,
                        },
                    },
                    {"type": "text", "text": state_text},
                ],
            }

            # History (text-only) + current (with image)
            context = list(message_history[-(args.history * 2):]) + [current_user_msg]

            # --- Claude API call ---
            try:
                response = client.messages.create(
                    model=args.model,
                    max_tokens=200,
                    system=SYSTEM_PROMPT,
                    messages=context,
                )
                raw_text = response.content[0].text
            except Exception as e:
                print(f"API error: {e}")
                raw_text = "ACTION: a\nTHOUGHT: API error.\nCOMMENTARY: Hmm, something went wrong — pressing A for now."

            parsed     = parse_response(raw_text)
            btn        = parsed["action"]
            thought    = parsed["thought"]
            commentary = parsed["commentary"]

            # --- Print to terminal ---
            loc = MAP_NAMES.get(mem[MAP_ID], f"Map {mem[MAP_ID]}")
            print(f"[{step:>5}] {loc:<22}  {btn:<8}  {commentary}", flush=True)
            if active_hint:
                print(f"        HINT RESPONSE: {thought}", flush=True)

            # --- Update display state ---
            last_thought = thought
            last_action  = btn
            if commentary:
                commentary_lines.extend(_wrap(commentary))
                if len(commentary_lines) > 40:
                    commentary_lines = commentary_lines[-40:]

            active_hint = None   # consumed after one action

            # --- Save to text-only history ---
            message_history.append({"role": "user",      "content": state_text})
            message_history.append({"role": "assistant", "content": raw_text})
            if len(message_history) > (args.history * 2 + 4):
                message_history = message_history[-(args.history * 2):]

            # --- Execute action with live display each frame ---
            def _frame():
                s = read_stats(pyboy.memory, visited_tiles)
                return build_display(pyboy, s, commentary_lines, last_thought, False, None)

            execute_action(pyboy, btn, args.frame_skip, args.speed, _frame)
            step += 1

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    pyboy.stop()

    final = read_stats(pyboy.memory, visited_tiles)
    print(f"\nSession ended at step {step}.")
    print(f"Badges: {final['badges']}/8  |  Levels: {final['levels']}  |  "
          f"Dex: {final['owned']}/151  |  Tiles: {final['tiles']}")


if __name__ == "__main__":
    main()
