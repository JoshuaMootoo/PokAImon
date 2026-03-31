"""Train a PPO agent to play Pokemon Blue.

Usage:
    python train.py                          # fresh training run
    python train.py --resume checkpoints/pokemon_blue_ppo_10000000_steps.zip
    python train.py --envs 4                 # override number of parallel environments
    python train.py --no-render              # disable the live view window
"""

import argparse
import os

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

from env import PokemonBlueEnv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ROM_PATH       = os.path.join(SCRIPT_DIR, "pokemon_blue.gb")
STATE_PATH     = os.path.join(SCRIPT_DIR, "init.state")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
LOG_DIR        = os.path.join(SCRIPT_DIR, "logs")

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
NUM_ENVS        = 8
TOTAL_TIMESTEPS = 50_000_000
CHECKPOINT_FREQ = 100_000

PPO_KWARGS = dict(
    learning_rate=2.5e-4,
    n_steps=1024,        # larger rollout → better gradient estimates
    batch_size=512,      # matched to larger rollout
    n_epochs=4,
    gamma=0.998,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.15,       # raised from 0.05 — policy was too deterministic to consistently
                         # discover Route 1 exit; higher entropy accelerates exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        normalize_images=True,
    ),
)

WINDOW_TITLE = "Pokemon Blue - Training (env 0)"


# ---------------------------------------------------------------------------
# Live view callback
# ---------------------------------------------------------------------------

class LiveViewCallback(BaseCallback):
    """Shows a live Game Boy window with stats from the first training environment."""

    def __init__(self, render_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self._last_info: dict = {}

    def _on_step(self) -> bool:
        # Keep the latest info dict from env 0 so we always have fresh stats
        infos = self.locals.get("infos", [])
        if infos:
            self._last_info = infos[0]

        if self.n_calls % self.render_freq == 0:
            try:
                frames = self.training_env.env_method("render", indices=[0])
                if frames and frames[0] is not None:
                    frame = np.array(frames[0])          # (144, 160, 3) RGB
                    bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    bgr   = cv2.resize(bgr, (480, 432), interpolation=cv2.INTER_NEAREST)

                    # Build stats footer
                    info      = self._last_info
                    badges    = info.get("badges", 0)
                    levels    = info.get("level_sum", 0)
                    pokedex   = info.get("pokedex", 0)
                    tiles     = info.get("visited_tiles", 0)
                    milestone = info.get("guide_name", "—")
                    steps     = self.n_calls

                    footer = np.zeros((50, 480, 3), dtype=np.uint8)
                    cv2.putText(footer,
                        f"Badges: {badges}/8   Levels: {levels}   Dex: {pokedex}/151   Tiles: {tiles}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    cv2.putText(footer,
                        f"Goal: {milestone}   Steps: {steps:,}",
                        (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

                    display = np.vstack([bgr, footer])
                    cv2.imshow(WINDOW_TITLE, display)
                    cv2.waitKey(1)
            except Exception:
                pass  # never crash training due to a display error
        return True

    def _on_training_end(self) -> None:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(rank: int, seed: int = 0):
    def _init():
        set_random_seed(seed + rank)
        return PokemonBlueEnv(
            rom_path=ROM_PATH,
            init_state_path=STATE_PATH,
            instance_id=rank,
            max_steps=4096,   # longer episodes — agent needs time to reach Pewter City
            frame_skip=16,    # shorter skip → more responsive battle/menu control
        )
    return _init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint .zip to resume from.")
    parser.add_argument("--envs", type=int, default=NUM_ENVS,
                        help="Number of parallel environments.")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable the live view window.")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    for path, name in [(ROM_PATH, "ROM"), (STATE_PATH, "initial save state")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at '{path}'.")
            if name == "initial save state":
                print("Run 'python create_save_state.py' first.")
            raise SystemExit(1)

    n_envs = args.envs
    print(f"Starting training with {n_envs} parallel environments.")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    if not args.no_render:
        print(f"Live view window: '{WINDOW_TITLE}' (pass --no-render to disable)\n")

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        print(f"Resuming from: {args.resume}")
        try:
            model = PPO.load(args.resume, env=vec_env, **PPO_KWARGS)
            reset_timesteps = False
            print("Checkpoint loaded successfully.")
        except ValueError as e:
            # Observation space changed (e.g. memory_features expanded from 16→17
            # when the GameGuide was added).  Start fresh rather than crashing.
            print(f"WARNING: Could not load checkpoint — {e}")
            print("Observation space mismatch (guide system added?). Starting fresh.")
            model = PPO(
                policy="MultiInputPolicy",
                env=vec_env,
                tensorboard_log=LOG_DIR,
                verbose=1,
                **PPO_KWARGS,
            )
            reset_timesteps = True
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            tensorboard_log=LOG_DIR,
            verbose=1,
            **PPO_KWARGS,
        )
        reset_timesteps = True

    callbacks = [
        CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // n_envs, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="pokemon_blue_ppo",
            verbose=1,
        )
    ]
    if not args.no_render:
        callbacks.append(LiveViewCallback(render_freq=100))

    print("Monitor training with:")
    print(f"  tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList(callbacks),
        reset_num_timesteps=reset_timesteps,
    )

    final_path = os.path.join(CHECKPOINT_DIR, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to '{final_path}.zip'.")

    vec_env.close()


if __name__ == "__main__":
    main()
