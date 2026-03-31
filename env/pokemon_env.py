import io
import numpy as np
import cv2
import gymnasium
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from .memory import (
    MAP_ID, PLAYER_X, PLAYER_Y,
    PARTY_COUNT, BATTLE_FLAG,
    POKEDEX_OWNED_START, POKEDEX_SEEN_START,
    read_badges, is_in_battle,
    read_party_hp_fraction, read_party_level_sum, read_party_xp_total,
    read_party_pokemon, read_bcd, count_bits,
    MONEY_0, PARTY_ADDRS, OFF_SPECIES, OFF_STATUS,
)
from .guide import GameGuide

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


class PokemonBlueEnv(gymnasium.Env):
    """Gymnasium environment wrapping PyBoy for Pokemon Blue.

    Observation space:
        "screen"         — (36, 40, 1) uint8 grayscale downscaled Game Boy screen
        "memory_features"— (17,) float32 normalised game-state vector
                           [0-14] game state, [15] guide progress, [16] target map

    Action space:
        Discrete(7): up / down / left / right / a / b / start
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        rom_path: str,
        init_state_path: str,
        max_steps: int = 2048,
        frame_skip: int = 24,
        screen_size: tuple = (36, 40),   # (height, width)
        render_mode: str = "rgb_array",
        instance_id: int = 0,
        exploration_reward_scale: float = 0.05,
        level_reward_scale: float = 1.0,
        badge_reward: float = 5.0,
        pokedex_reward_scale: float = 0.5,
        run_penalty: float = 0.0,
        battle_step_reward: float = 0.02,
        xp_reward_scale: float = 0.001,
        money_reward_scale: float = 0.0005,
    ):
        super().__init__()

        self.rom_path = rom_path
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.screen_size = screen_size  # (H, W)
        self.render_mode = render_mode
        self.instance_id = instance_id

        self.exploration_reward_scale = exploration_reward_scale
        self.level_reward_scale = level_reward_scale
        self.badge_reward = badge_reward
        self.pokedex_reward_scale = pokedex_reward_scale
        self.run_penalty = run_penalty
        self.battle_step_reward = battle_step_reward
        self.xp_reward_scale = xp_reward_scale
        self.money_reward_scale = money_reward_scale

        # Load initial save state bytes once — avoids filesystem access during rollouts
        with open(init_state_path, "rb") as f:
            self.init_state_bytes = f.read()

        # Create PyBoy instance (always headless; rendering done via screen.image)
        self.pyboy = PyBoy(rom_path, window="null")
        self.pyboy.set_emulation_speed(0)  # run as fast as possible

        # Spaces
        # memory_features is 17 elements:
        #   [0-14]  game-state scalars (unchanged)
        #   [15]    guide progress  — normalised milestone index from GameGuide
        #   [16]    guide target    — next target map ID / 255.0
        H, W = screen_size
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(H, W, 1), dtype=np.uint8),
            "memory_features": spaces.Box(low=0.0, high=1.0, shape=(17,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Guide agent — persists across episodes; tracks game progression
        # milestones and provides observation hints + milestone bonuses.
        self.guide = GameGuide()

        # Episode tracking (initialised properly in reset())
        self.visited_tiles: set = set()
        self.tile_visit_count: dict = {}   # tile -> visit count (for count-based reward)
        # visited_maps is intentionally NOT reset between episodes — each map_id
        # can only ever award the +2.0 new-map bonus ONCE across the entire
        # training run.  This prevents the agent from farming the same
        # Pallet Town → Route 1 → Viridian City loop every episode and forces
        # it to always push toward genuinely unexplored territory.
        self.visited_maps: set = set()
        self.current_step: int = 0
        self.steps_since_new_tile: int = 0
        self.prev_badges: int = 0
        self.prev_level_sum: int = 0
        self.prev_pokedex_count: int = 0
        self.prev_battle_flag: int = 0
        self.prev_xp_total: int = 0
        self.xp_at_battle_start: int = 0   # XP snapshot when battle began
        self.prev_money: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state_buf = io.BytesIO(self.init_state_bytes)
        self.pyboy.load_state(state_buf)

        # Tick a few frames to let the game stabilise after state load
        self.pyboy.tick(4, False)

        # Reset per-episode trackers.
        # NOTE: tile_visit_count, visited_maps, and guide state are intentionally
        # NOT reset — they accumulate globally so the agent is always pushed
        # toward genuinely unexplored tiles/maps/milestones.
        self.visited_tiles = set()
        # self.visited_maps  ← NOT reset; persists across episodes
        self.guide.reset()  # no-op by design; call kept for interface clarity
        self.current_step = 0
        self.steps_since_new_tile = 0
        self.prev_badges = read_badges(self.pyboy.memory)
        self.prev_level_sum = read_party_level_sum(self.pyboy.memory)
        self.prev_pokedex_count = count_bits(self.pyboy.memory, POKEDEX_OWNED_START, 19)
        self.prev_battle_flag = 0
        self.prev_xp_total = read_party_xp_total(self.pyboy.memory)
        self.xp_at_battle_start = self.prev_xp_total
        self.prev_money = read_bcd(self.pyboy.memory, MONEY_0, 3)

        return self._get_obs(), {}

    def step(self, action: int):
        self._press_action(action)

        obs = self._get_obs()
        reward = self._compute_reward()

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "visited_tiles": len(self.visited_tiles),
            "badges": self.prev_badges,
            "level_sum": self.prev_level_sum,
            "pokedex": self.prev_pokedex_count,
            "guide_milestone": self.guide.milestone_index,
            "guide_name": self.guide.milestone_name,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        rgb = np.array(self.pyboy.screen.image)  # (144, 160, 3)
        if self.render_mode == "rgb_array":
            return rgb
        # "human" mode: nothing extra — PyBoy window=null, caller must handle display
        return rgb

    def close(self):
        self.pyboy.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _press_action(self, action_idx: int):
        """Hold a button for 8 frames, release, then idle for the remainder.
        The final tick uses render=True so pyboy.screen.image is fresh for
        the next observation read.
        """
        btn = ACTIONS[action_idx]
        self.pyboy.send_input(PRESS[btn])
        self.pyboy.tick(8, False)
        self.pyboy.send_input(RELEASE[btn])
        self.pyboy.tick(self.frame_skip - 9, False)  # all but last frame headless
        self.pyboy.tick(1, True)                     # final frame updates screen buffer

    def _get_obs(self) -> dict:
        return {
            "screen": self._get_screen_obs(),
            "memory_features": self._get_memory_features(),
        }

    def _get_screen_obs(self) -> np.ndarray:
        """Capture Game Boy screen, convert to grayscale, resize to screen_size."""
        rgb = np.array(self.pyboy.screen.image)       # (144, 160, 3)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # (144, 160)
        H, W = self.screen_size
        resized = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)  # (H, W)
        return resized[:, :, np.newaxis]  # (H, W, 1)

    def _get_memory_features(self) -> np.ndarray:
        """Build a 16-element float32 observation vector from game memory."""
        mem = self.pyboy.memory

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
        unique_tiles = len(self.visited_tiles)

        # Lead Pokemon info (slot 0)
        lead = read_party_pokemon(mem, 0) if party_count > 0 else {}
        lead_level   = lead.get("level", 0)
        lead_healthy = float(lead.get("status", 1) == 0)
        lead_species = lead.get("species", 0)

        money = read_bcd(mem, MONEY_0, 3)

        features = np.array([
            badges       / 8.0,                       # 0  badge count
            party_count  / 6.0,                       # 1  team size
            hp_frac,                                  # 2  mean HP fraction
            lead_level   / 100.0,                     # 3  lead Pokemon level
            level_sum    / 600.0,                     # 4  total party levels
            owned        / 151.0,                     # 5  Pokedex owned
            seen         / 151.0,                     # 6  Pokedex seen
            map_id       / 255.0,                     # 7  current map
            px           / 255.0,                     # 8  player X tile
            py           / 255.0,                     # 9  player Y tile
            in_battle,                                # 10 battle flag
            min(unique_tiles / 2000.0, 1.0),          # 11 exploration count
            lead_healthy,                             # 12 lead status OK
            lead_species / 151.0,                     # 13 lead species
            min(money    / 999999.0, 1.0),            # 14 money
            self.guide.progress,                      # 15 guide: milestone progress [0,1]
            self.guide.target_map_id / 255.0,         # 16 guide: next target map
        ], dtype=np.float32)

        return features

    def _compute_reward(self) -> float:
        mem = self.pyboy.memory

        # Tiny per-step cost — just enough to avoid doing nothing being free.
        reward = -0.001

        # --- Count-based exploration reward ---
        # Reward = scale / sqrt(visit_count) so the first visit is worth the
        # full scale, the second is ~71%, tenth is ~32%, etc.  The reward never
        # hits zero, so the agent always has a gradient pushing it toward
        # less-visited tiles rather than a hard cliff that causes loops.
        tile = (mem[MAP_ID], mem[PLAYER_X], mem[PLAYER_Y])
        raw_count = self.tile_visit_count.get(tile, 0) + 1
        self.tile_visit_count[tile] = raw_count
        # Cap at 25 so reward floor = scale/sqrt(25) = 0.004, always above the
        # -0.001/step cost.  Without this cap, globally-accumulated counts decay
        # exploration reward to near-zero in familiar areas, making movement
        # indistinguishable from standing still and collapsing the policy.
        reward += self.exploration_reward_scale / np.sqrt(min(raw_count, 25))

        # Track truly new tiles (used for stuck detection only)
        if tile not in self.visited_tiles:
            self.visited_tiles.add(tile)
            self.steps_since_new_tile = 0
        else:
            self.steps_since_new_tile += 1

        # Stuck penalty — kicks in after stagnation and scales up the longer
        # the agent stays stuck, creating mounting pressure to explore.
        # The 200-step penalty is 5× stronger than before so it clearly
        # dominates the tile-floor reward (+0.01/step), forcing the agent
        # out of Pallet Town once all local tiles are exhausted.
        if self.steps_since_new_tile > 200:
            reward -= 0.05
        elif self.steps_since_new_tile > 100:
            reward -= 0.003

        # --- New map bonus ---
        # Strong one-off reward the first time the agent enters any map ID
        # (new route, building, cave).  visited_maps is global (never reset)
        # so this fires at most once per map across the whole training run,
        # permanently forcing the agent to push toward new territory.
        map_id = mem[MAP_ID]
        if map_id not in self.visited_maps:
            self.visited_maps.add(map_id)
            reward += 2.0

        # Read badges once; reused for guide update, badge delta, and run-penalty.
        badges = read_badges(mem)

        # --- Guide milestone bonus ---
        # The GameGuide checks whether the badge/map state matches the next
        # walkthrough milestone and fires a large one-time bonus reward when
        # a milestone is first reached.  This creates a clear reward signal
        # aligned with actual game progression (gyms, story gates, etc.).
        guide_bonus = self.guide.update(badges, self.visited_maps)
        if guide_bonus > 0:
            reward += guide_bonus

        # --- Battle participation reward ---
        # Small reward for each step the agent stays in a battle rather than
        # immediately fleeing.  This tips the risk/reward balance toward
        # engaging with encounters instead of avoiding them.
        if is_in_battle(mem):
            reward += self.battle_step_reward

        # --- Badges ---
        badge_delta = badges - self.prev_badges
        if badge_delta > 0:
            reward += badge_delta * self.badge_reward
            self.prev_badges = badges

        # --- Party level sum ---
        level_sum = read_party_level_sum(mem)
        level_delta = level_sum - self.prev_level_sum
        if level_delta > 0:
            reward += level_delta * self.level_reward_scale
            self.prev_level_sum = level_sum

        # --- XP gained ---
        # XP increases every time an enemy is defeated, not just on level-up.
        # This gives the agent tight, frequent feedback for actually winning
        # battles rather than running — the key signal for game progression.
        xp_total = read_party_xp_total(mem)
        xp_delta = xp_total - self.prev_xp_total
        if xp_delta > 0:
            reward += xp_delta * self.xp_reward_scale
            self.prev_xp_total = xp_total

        # --- Money gained ---
        # Money increases when defeating trainers.  Rewards engaging with
        # the overworld trainer fights that gate gym progression.
        money = read_bcd(mem, MONEY_0, 3)
        money_delta = money - self.prev_money
        if money_delta > 0:
            reward += money_delta * self.money_reward_scale
            self.prev_money = money

        # --- Pokedex (owned) ---
        owned = count_bits(mem, POKEDEX_OWNED_START, 19)
        pokedex_delta = owned - self.prev_pokedex_count
        if pokedex_delta > 0:
            reward += pokedex_delta * self.pokedex_reward_scale
            self.prev_pokedex_count = owned

        # --- Penalty: entire party fainted ---
        if read_party_hp_fraction(mem) == 0.0:
            reward -= 0.5

        # --- Battle win bonus + run penalty ---
        # xp_at_battle_start is set when battle begins (flag 0→nonzero) and
        # compared against xp_total when battle ends (flag nonzero→0).  This
        # spans the full battle regardless of how many steps the XP animation
        # takes, making it immune to frame-skip timing issues.
        cur_battle_flag = mem[BATTLE_FLAG]
        if self.prev_battle_flag == 0 and cur_battle_flag != 0:
            # Battle just started — snapshot XP so we can measure the gain.
            self.xp_at_battle_start = xp_total

        if self.prev_battle_flag != 0 and cur_battle_flag == 0:
            # Battle just ended.
            xp_gained_in_battle = xp_total > self.xp_at_battle_start
            no_level_gain = (read_party_level_sum(mem) == self.prev_level_sum)
            no_dex_gain   = (count_bits(mem, POKEDEX_OWNED_START, 19) == self.prev_pokedex_count)
            if xp_gained_in_battle:
                # Bonus for actually winning — on top of the per-XP reward.
                reward += 1.0
            elif no_level_gain and no_dex_gain:
                # No XP, no level, no catch → agent fled or was wiped.
                reward += self.run_penalty

        self.prev_battle_flag = cur_battle_flag

        return reward
