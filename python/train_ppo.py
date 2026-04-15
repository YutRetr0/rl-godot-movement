from __future__ import annotations

import argparse
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_env import GodotGymEnv


def make_env(port: int, host: str):
    def _init() -> GodotGymEnv:
        return GodotGymEnv(host=host, port=port)

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Godot TCP RL environment")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=[9000],
        help="List of environment ports (one Godot instance per port)",
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--model-out", type=str, default="ppo_godot_movement")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_fns: list[Callable[[], GodotGymEnv]] = [make_env(port, args.host) for port in args.ports]

    if len(env_fns) == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        verbose=1,
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.model_out)
    vec_env.close()


if __name__ == "__main__":
    main()
