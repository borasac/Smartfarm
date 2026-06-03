"""
Full PPO training on GreenLightEnv — W&B 불필요, 로컬 저장.
하이퍼파라미터: configs/agents/ppo.yml 기준 (Bayes 최적화 결과)
"""
import os
import sys
import time

os.environ["WANDB_MODE"] = "offline"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, "/home/user/GreenLight-Gym2")

import numpy as np
import torch
from torch.nn.modules.activation import SiLU
from torch.optim import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from gymnasium.wrappers import FlattenObservation

from gl_gym.environments.greenlight_env import GreenLightEnv
from gl_gym.environments.utils import load_weather_data
from RL.utils import load_env_params, build_env_kwargs

# ── 하이퍼파라미터 (ppo.yml 기준) ──────────────────────────────────────────
TOTAL_TIMESTEPS = 2_000_000
N_ENVS          = 4
N_STEPS         = 2048
BATCH_SIZE      = 128
N_EPOCHS        = 8
GAMMA           = 0.9631
GAE_LAMBDA      = 0.9167
CLIP_RANGE      = 0.2
ENT_COEF        = 0.05434
VF_COEF         = 0.8225
MAX_GRAD_NORM   = 0.3
LEARNING_RATE   = 2e-5
SEED            = 666
SAVE_DIR        = "train_data/ppo_full"
N_EVALS         = 10   # 훈련 중 평가 횟수
# ────────────────────────────────────────────────────────────────────────────


class ProgressLogger(BaseCallback):
    """매 평가마다 보상 및 진행률을 출력합니다."""
    def __init__(self, total_steps, baseline_reward=3103.47, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.baseline = baseline_reward
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        return True


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

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/eval", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/best", exist_ok=True)

    # ── 훈련 환경 ──────────────────────────────────────────────────────────
    vec_norm_kwargs = dict(norm_obs=True, norm_reward=True, clip_obs=10, gamma=GAMMA)

    train_env = SubprocVecEnv([make_env(i, SEED, env_kwargs) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, **vec_norm_kwargs)

    # ── 평가 환경 (eval scenario 고정) ─────────────────────────────────────
    eval_scenario = eval_scenarios[0]
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs["weather_scenario_sampler"] = "fixed"
    eval_env_kwargs["weather_scenario_sampler_kwargs"] = eval_scenario

    def make_eval_env(rank, seed, kwargs):
        def _init():
            env = GreenLightEnv(**kwargs)
            env = FlattenObservation(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    eval_vec = SubprocVecEnv([make_eval_env(0, SEED, eval_env_kwargs)])
    eval_vec = VecMonitor(eval_vec)
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10, gamma=GAMMA, training=False)

    # ── 모델 ────────────────────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        normalize_advantage=True,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        learning_rate=LEARNING_RATE,
        use_sde=False,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], vf=[512, 512, 512]),
            activation_fn=SiLU,
            optimizer_class=Adam,
            optimizer_kwargs=dict(amsgrad=True),
            log_std_init=np.log(1),
        ),
        verbose=1,
        device="cpu",
        tensorboard_log=f"{SAVE_DIR}/tb_logs",
    )

    # ── 콜백 ────────────────────────────────────────────────────────────────
    eval_freq = TOTAL_TIMESTEPS // N_EVALS // N_ENVS

    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=f"{SAVE_DIR}/best",
        log_path=f"{SAVE_DIR}/eval",
        eval_freq=eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=f"{SAVE_DIR}/checkpoints",
        name_prefix="ppo",
        verbose=1,
    )
    progress_cb = ProgressLogger(total_steps=TOTAL_TIMESTEPS)

    # ── 훈련 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PPO 훈련 시작")
    print(f"  Total steps : {TOTAL_TIMESTEPS:,}")
    print(f"  Envs        : {N_ENVS}")
    print(f"  Eval freq   : 매 {eval_freq * N_ENVS:,} steps")
    print(f"  Baseline    : 3,103.47 (rule-based)")
    print(f"  Save dir    : {SAVE_DIR}/")
    print(f"{'='*60}\n")

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_cb, ckpt_cb, progress_cb],
        reset_num_timesteps=True,
    )
    elapsed = time.time() - start

    # ── 저장 ────────────────────────────────────────────────────────────────
    model.save(f"{SAVE_DIR}/final_model")
    train_env.save(f"{SAVE_DIR}/vec_normalize.pkl")
    train_env.close()
    eval_vec.close()

    print(f"\n{'='*60}")
    print(f"  훈련 완료! 소요 시간: {elapsed/60:.1f}분")
    print(f"  최종 모델 : {SAVE_DIR}/final_model.zip")
    print(f"  최선 모델 : {SAVE_DIR}/best/best_model.zip")
    print(f"  평가 로그 : {SAVE_DIR}/eval/evaluations.npz")
    print(f"{'='*60}\n")

    # ── 평가 결과 요약 ───────────────────────────────────────────────────────
    try:
        evals = np.load(f"{SAVE_DIR}/eval/evaluations.npz")
        rewards = evals["results"]
        timesteps = evals["timesteps"]
        print("[ 훈련 중 평가 보상 (에피소드 총합) ]")
        for t, r in zip(timesteps, rewards):
            print(f"  step {t:>10,} : {r.mean():>8.1f}  (baseline: 3103.5)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
