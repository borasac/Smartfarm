"""
GreenLight-Gym2 시뮬레이션 결과를 self-contained HTML 파일로 저장.
PPO vs Rule-Based 비교.
"""
import sys
sys.path.insert(0, "/home/user/GreenLight-Gym2")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import FlattenObservation

from gl_gym.environments.greenlight_env import GreenLightEnv
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext
from RL.utils import load_env_params, build_env_kwargs, load_model_hyperparams
from gl_gym.environments.utils import co2dens2ppm, vaporPres2rh

MODEL_PATH   = "train_data/ppo_full/best/best_model.zip"
VEC_NORM_PKG = "train_data/ppo_full/vec_normalize.pkl"
SCENARIO     = {"location": "Amsterdam", "growth_year": 2010, "start_day": 59}
SEED         = 0


def satVp(temp):
    return 610.78 * np.exp(17.2694 * temp / (temp + 238.3))


def build_step_context(env):
    return StepContext(
        t=env.timestep, dt=env.dt, Np=env.Np,
        x_prev=env.x_prev, x=env.x, u=env.u, p=env.p,
        d=env.weather_data,
        hour_of_day=env.hour_of_day, day_of_year=env.day_of_year,
    )


def record(env, u, reward, info):
    x = env.x
    temp = float(x[2])
    return {
        "step":      env.timestep,
        "day":       env.day_of_year - SCENARIO["start_day"],
        "temp":      temp,
        "co2_ppm":   float(co2dens2ppm(temp, x[0] * 1e-6)),
        "rh":        float(vaporPres2rh(temp, x[15])) * 100,
        "fruit_dm":  float(x[25]),
        "uBoil":     float(u[0]),
        "uCO2":      float(u[1]),
        "uThScr":    float(u[2]),
        "uVent":     float(u[3]),
        "uLamp":     float(u[4]),
        "uBlScr":    float(u[5]),
        "reward":    reward,
        "revenue":   float(info.get("revenue", 0)),
        "heat_cost": float(info.get("heat_cost", 0)),
        "co2_cost":  float(info.get("co2_cost", 0)),
        "elec_cost": float(info.get("elec_cost", 0)),
        "profit":    float(info.get("profit", 0)),
        "rh_viol":   float(info.get("rh_violation", 0)),
        "co2_viol":  float(info.get("co2_violation", 0)),
        "temp_viol": float(info.get("temp_violation", 0)),
        "lamp_pen":  float(info.get("lamp_penalty", 0)),
    }


def run_rb(env_kwargs, rb_params):
    print("  Rule-Based 에피소드 실행 중...")
    kwargs = env_kwargs.copy()
    kwargs["normalize_actions"] = False
    env = GreenLightEnv(**kwargs)
    controller = RuleBasedController(**rb_params)
    env.reset(seed=SEED, options={"scenario": SCENARIO})
    rows = []
    while True:
        ctx = build_step_context(env)
        u = controller.predict(ctx)
        _, rw, terminated, truncated, info = env.step(u.astype(np.float32))
        rows.append(record(env, u, float(rw), info))
        if terminated or truncated:
            break
    env.close()
    return pd.DataFrame(rows)


def run_ppo(env_kwargs):
    print("  PPO 에피소드 실행 중...")
    kwargs = env_kwargs.copy()

    def _make():
        e = GreenLightEnv(**kwargs)
        return FlattenObservation(e)

    vec_env = DummyVecEnv([_make])
    vec_env = VecNormalize.load(VEC_NORM_PKG, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(MODEL_PATH, env=vec_env)

    raw_env = vec_env.envs[0].env
    raw_env.reset(seed=SEED, options={"scenario": SCENARIO})
    obs = vec_env.reset()

    rows = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info_vec = vec_env.step(action)
        u = raw_env.u
        info = info_vec[0]
        rows.append(record(raw_env, u, float(info.get("reward", 0)), info))
        if done[0]:
            break
    vec_env.close()
    return pd.DataFrame(rows)


def metrics(df):
    return {
        "에피소드 총 보상":   df["reward"].sum(),
        "총 수익 (€/m²)":    df["revenue"].sum(),
        "난방비 (€/m²)":     df["heat_cost"].sum(),
        "CO2비 (€/m²)":      df["co2_cost"].sum(),
        "전기비 (€/m²)":     df["elec_cost"].sum(),
        "순이익 (€/m²)":     df["profit"].sum(),
        "수확량 (kg/m²)":    df["fruit_dm"].iloc[-1] * 1e-6 / 0.065,
        "온도 위반 (%)":      (df["temp_viol"] > 0).mean() * 100,
        "CO2 위반 (%)":       (df["co2_viol"] > 0).mean() * 100,
        "습도 위반 (%)":      (df["rh_viol"] > 0).mean() * 100,
        "야간 조명 위반 (%)": (df["lamp_pen"] > 0).mean() * 100,
    }


def build_figure(rb: pd.DataFrame, ppo: pd.DataFrame) -> go.Figure:
    RB_C  = "#FF5722"
    PPO_C = "#2196F3"
    x_rb  = rb["day"]
    x_ppo = ppo["day"]

    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            "온도 (°C)", "습도 (%RH)",
            "CO₂ (ppm)", "수확량 (kg/m²)",
            "제어: 난방 & 환기", "제어: 조명 & CO2",
            "누적 순이익 (€/m²)", "스텝별 보상",
            "비용 분해 — Rule-Based (€/m²)", "비용 분해 — PPO (€/m²)",
        ],
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
    )

    def add(row, col, y_rb, y_ppo, name_rb="Rule-Based", name_ppo="PPO",
            showlegend_rb=True, showlegend_ppo=True):
        fig.add_trace(go.Scatter(x=x_rb,  y=y_rb,  name=name_rb,
                                 line=dict(color=RB_C,  width=1.5),
                                 legendgroup="rb",  showlegend=showlegend_rb),  row=row, col=col)
        fig.add_trace(go.Scatter(x=x_ppo, y=y_ppo, name=name_ppo,
                                 line=dict(color=PPO_C, width=1.5),
                                 legendgroup="ppo", showlegend=showlegend_ppo), row=row, col=col)

    # Row 1: 온도, 습도
    add(1, 1, rb["temp"],   ppo["temp"])
    add(1, 2, rb["rh"],     ppo["rh"],     showlegend_rb=False, showlegend_ppo=False)
    fig.add_hline(y=85,   row=1, col=2, line_dash="dot", line_color="red",  annotation_text="한계 85%")
    fig.add_hline(y=19.5, row=1, col=1, line_dash="dot", line_color="gray", annotation_text="주간 설정점")

    # Row 2: CO2, 수확량
    add(2, 1, rb["co2_ppm"], ppo["co2_ppm"], showlegend_rb=False, showlegend_ppo=False)
    harvest_rb  = rb["fruit_dm"]  * 1e-6 / 0.065
    harvest_ppo = ppo["fruit_dm"] * 1e-6 / 0.065
    add(2, 2, harvest_rb, harvest_ppo, showlegend_rb=False, showlegend_ppo=False)
    fig.add_hline(y=800, row=2, col=1, line_dash="dot", line_color="green", annotation_text="CO₂ 목표")

    # Row 3: 제어값
    for (u, col, leg) in [("uBoil", 1, False), ("uVent", 1, False),
                           ("uLamp", 2, False), ("uCO2",  2, False)]:
        dash = "solid" if u in ("uBoil", "uLamp") else "dash"
        fig.add_trace(go.Scatter(x=x_rb,  y=rb[u],  name=f"RB {u}",
                                 line=dict(color=RB_C,  width=1, dash=dash),
                                 legendgroup="rb",  showlegend=leg), row=3, col=col)
        fig.add_trace(go.Scatter(x=x_ppo, y=ppo[u], name=f"PPO {u}",
                                 line=dict(color=PPO_C, width=1, dash=dash),
                                 legendgroup="ppo", showlegend=leg), row=3, col=col)

    # Row 4: 누적 이익, 보상
    add(4, 1, rb["profit"].cumsum(),  ppo["profit"].cumsum(),
        showlegend_rb=False, showlegend_ppo=False)
    add(4, 2, rb["reward"],           ppo["reward"],
        showlegend_rb=False, showlegend_ppo=False)
    fig.add_hline(y=0, row=4, col=1, line_dash="dash", line_color="black")

    # Row 5: 비용 분해 (누적 스택 바 → 누적 선)
    for col, df, c in [(1, rb, RB_C), (2, ppo, PPO_C)]:
        for cost_key, label, dash in [
            ("revenue",   "수익",   "solid"),
            ("heat_cost", "난방비", "dot"),
            ("elec_cost", "전기비", "dash"),
            ("co2_cost",  "CO2비",  "dashdot"),
        ]:
            sign = 1 if cost_key == "revenue" else -1
            fig.add_trace(go.Scatter(
                x=x_ppo if col == 2 else x_rb,
                y=(sign * df[cost_key]).cumsum(),
                name=label, line=dict(color=c, width=1.2, dash=dash),
                showlegend=False,
            ), row=5, col=col)

    fig.update_layout(
        height=1600,
        title_text=(
            "<b>GreenLight-Gym2 시뮬레이션 결과</b><br>"
            "<sub>Amsterdam · 2010 · Day 59 · 60일 에피소드</sub>"
        ),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="경과 일수 (day)")
    return fig


def main():
    print("리소스 로드 중...")
    env_kwargs = load_env_params("GreenLightEnv", "gl_gym/configs/envs/")
    env_kwargs, _ = build_env_kwargs(env_kwargs)
    rb_params = load_model_hyperparams("rule_based", "GreenLightEnv")

    print("에피소드 실행 중...")
    rb  = run_rb(env_kwargs, rb_params)
    ppo = run_ppo(env_kwargs)

    # 지표 출력
    print("\n[ 핵심 지표 비교 ]")
    m_rb  = metrics(rb)
    m_ppo = metrics(ppo)
    print(f"{'지표':<22} {'Rule-Based':>12} {'PPO':>12} {'차이':>10}")
    print("-" * 58)
    for k in m_rb:
        v_rb  = m_rb[k]
        v_ppo = m_ppo[k]
        diff  = v_ppo - v_rb
        print(f"{k:<22} {v_rb:>12.2f} {v_ppo:>12.2f} {diff:>+10.2f}")

    # HTML 저장
    print("\nHTML 생성 중...")
    fig = build_figure(rb, ppo)
    out = "simulation_report.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"저장 완료: {out}")


if __name__ == "__main__":
    main()
