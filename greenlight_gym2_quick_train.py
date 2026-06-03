"""Quick standalone PPO training test — no W&B account needed."""
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "/home/user/GreenLight-Gym2")

import numpy as np
from torch.nn.modules.activation import SiLU
from torch.optim import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, DummyVecEnv
from gymnasium.wrappers import FlattenObservation

from gl_gym.environments.greenlight_env import GreenLightEnv
from gl_gym.environments.utils import load_weather_data
from RL.utils import load_env_params, build_env_kwargs

def make_env(rank, seed, env_kwargs):
    def _init():
        env = GreenLightEnv(**env_kwargs)
        env = FlattenObservation(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    env_kwargs = load_env_params("GreenLightEnv", "gl_gym/configs/envs/")
    env_kwargs, eval_scenarios = build_env_kwargs(env_kwargs)

    N_ENVS = 2
    TOTAL_STEPS = 10_000
    SEED = 42

    vec_env = SubprocVecEnv([make_env(i, SEED, env_kwargs) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10, gamma=0.96)

    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=SEED,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.96,
        learning_rate=2e-5,
        ent_coef=0.05,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[256, 256]),
            activation_fn=SiLU,
            optimizer_class=Adam,
        ),
        verbose=1,
        device="cpu",
    )

    print(f"\n=== PPO 훈련 시작 ({TOTAL_STEPS:,} steps, {N_ENVS} envs) ===\n")
    model.learn(total_timesteps=TOTAL_STEPS)

    os.makedirs("train_data/quick_test", exist_ok=True)
    model.save("train_data/quick_test/ppo_model")
    vec_env.save("train_data/quick_test/vec_normalize.pkl")
    vec_env.close()

    print("\n=== 훈련 완료 ===")
    print("모델 저장: train_data/quick_test/ppo_model.zip")
    print("정규화 통계: train_data/quick_test/vec_normalize.pkl")

if __name__ == "__main__":
    main()
