"""
GreenLight-Gym2 시뮬레이션 대시보드
PPO vs Rule-Based 컨트롤러 비교 시각화
"""
import sys
sys.path.insert(0, "/home/user/GreenLight-Gym2")

import numpy as np
import pandas as pd
import streamlit as st
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

# ── 상수 ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "train_data/ppo_full/best/best_model.zip"
VEC_NORM_PKG = "train_data/ppo_full/vec_normalize.pkl"
CONTROL_NAMES = ["난방(uBoil)", "CO2(uCO2)", "차열스크린(uThScr)",
                 "환기(uVent)", "조명(uLamp)", "차광스크린(uBlScr)"]
CONTROL_COLORS = ["#e74c3c", "#2ecc71", "#f39c12", "#3498db", "#9b59b6", "#1abc9c"]

# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────
def satVp(temp):
    return 610.78 * np.exp(17.2694 * temp / (temp + 238.3))


def build_step_context(env):
    return StepContext(
        t=env.timestep, dt=env.dt, Np=env.Np,
        x_prev=env.x_prev, x=env.x, u=env.u, p=env.p,
        d=env.weather_data,
        hour_of_day=env.hour_of_day, day_of_year=env.day_of_year,
    )


@st.cache_resource
def load_resources():
    env_kwargs = load_env_params("GreenLightEnv", "gl_gym/configs/envs/")
    env_kwargs, scenarios = build_env_kwargs(env_kwargs)
    rb_params = load_model_hyperparams("rule_based", "GreenLightEnv")
    return env_kwargs, scenarios, rb_params


def run_episode_rb(env_kwargs, rb_params, scenario, seed):
    """Rule-based 컨트롤러로 에피소드 실행"""
    kwargs = env_kwargs.copy()
    kwargs["normalize_actions"] = False
    env = GreenLightEnv(**kwargs)
    controller = RuleBasedController(**rb_params)

    env.reset(seed=seed, options={"scenario": scenario})
    records = []
    while True:
        ctx = build_step_context(env)
        u = controller.predict(ctx)
        _, reward, terminated, truncated, info = env.step(u.astype(np.float32))
        records.append(_record(env, u, reward, info))
        if terminated or truncated:
            break
    env.close()
    return pd.DataFrame(records)


def run_episode_ppo(env_kwargs, scenario, seed):
    """PPO 모델로 에피소드 실행"""
    kwargs = env_kwargs.copy()

    def _make():
        e = GreenLightEnv(**kwargs)
        e = FlattenObservation(e)
        return e

    vec_env = DummyVecEnv([_make])
    vec_env = VecNormalize.load(VEC_NORM_PKG, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=vec_env)

    raw_env = vec_env.envs[0].env
    raw_env.reset(seed=seed, options={"scenario": scenario})
    obs = vec_env.reset()

    records = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info_vec = vec_env.step(action)

        u = raw_env.u
        reward_info = info_vec[0]
        reward = float(reward_info.get("reward", 0))
        records.append(_record(raw_env, u, reward, reward_info))
        if done[0]:
            break
    vec_env.close()
    return pd.DataFrame(records)


def _record(env, u, reward, info):
    x = env.x
    temp  = float(x[2])
    co2   = float(co2dens2ppm(temp, x[0] * 1e-6))
    rh    = float(vaporPres2rh(temp, x[15])) * 100
    fruit_dm = float(x[25])

    return {
        "step":       env.timestep,
        "hour":       env.hour_of_day,
        "day":        env.day_of_year,
        "temp":       temp,
        "co2_ppm":    co2,
        "rh":         rh,
        "fruit_dm":   fruit_dm,
        "uBoil":      float(u[0]),
        "uCO2":       float(u[1]),
        "uThScr":     float(u[2]),
        "uVent":      float(u[3]),
        "uLamp":      float(u[4]),
        "uBlScr":     float(u[5]),
        "reward":     reward,
        "revenue":    float(info.get("revenue", 0)),
        "heat_cost":  float(info.get("heat_cost", 0)),
        "co2_cost":   float(info.get("co2_cost", 0)),
        "elec_cost":  float(info.get("elec_cost", 0)),
        "profit":     float(info.get("profit", 0)),
        "temp_viol":  float(info.get("temp_violation", 0)),
        "co2_viol":   float(info.get("co2_violation", 0)),
        "rh_viol":    float(info.get("rh_violation", 0)),
        "lamp_pen":   float(info.get("lamp_penalty", 0)),
    }


def summary_metrics(df):
    N = len(df)
    return {
        "에피소드 총 보상":    df["reward"].sum(),
        "총 수익 (€/m²)":     df["revenue"].sum(),
        "난방비 (€/m²)":      df["heat_cost"].sum(),
        "CO2 비용 (€/m²)":    df["co2_cost"].sum(),
        "전기비 (€/m²)":      df["elec_cost"].sum(),
        "순이익 (€/m²)":      df["profit"].sum(),
        "총 수확량 (kg/m²)":  df["fruit_dm"].iloc[-1] * 1e-6 / 0.065,
        "온도 위반 (%)":       (df["temp_viol"] > 0).mean() * 100,
        "CO2 위반 (%)":        (df["co2_viol"] > 0).mean() * 100,
        "습도 위반 (%)":       (df["rh_viol"] > 0).mean() * 100,
        "야간 조명 위반 (%)":  (df["lamp_pen"] > 0).mean() * 100,
    }


# ── 차트 함수 ────────────────────────────────────────────────────────────────
def plot_climate(dfs: dict[str, pd.DataFrame]):
    colors = {"PPO": "#2196F3", "Rule-Based": "#FF5722"}
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=["실내 온도 (°C)", "실내 습도 (%RH)", "실내 CO₂ (ppm)"],
                        shared_xaxes=True, vertical_spacing=0.08)
    for name, df in dfs.items():
        c = colors.get(name, "#888")
        x = df["step"]
        fig.add_trace(go.Scatter(x=x, y=df["temp"],   name=name, line=dict(color=c, width=1.5),
                                 legendgroup=name, showlegend=True),  row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["rh"],     name=name, line=dict(color=c, width=1.5),
                                 legendgroup=name, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["co2_ppm"],name=name, line=dict(color=c, width=1.5),
                                 legendgroup=name, showlegend=False), row=3, col=1)

    # 기준선
    fig.add_hline(y=19.5, row=1, col=1, line_dash="dot", line_color="gray", annotation_text="주간 설정점")
    fig.add_hline(y=85,   row=2, col=1, line_dash="dot", line_color="red",  annotation_text="RH 한계")
    fig.add_hline(y=800,  row=3, col=1, line_dash="dot", line_color="green",annotation_text="CO₂ 목표")
    fig.update_layout(height=600, title_text="실내 기후 변수", hovermode="x unified")
    fig.update_xaxes(title_text="스텝", row=3)
    return fig


def plot_controls(dfs: dict[str, pd.DataFrame]):
    colors = {"PPO": "#2196F3", "Rule-Based": "#FF5722"}
    ukeys = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]
    labels = ["난방", "CO2", "차열스크린", "환기", "조명", "차광스크린"]

    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=labels,
                        shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.1)
    positions = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]

    for name, df in dfs.items():
        c = colors.get(name, "#888")
        for i, (key, (r, col)) in enumerate(zip(ukeys, positions)):
            fig.add_trace(
                go.Scatter(x=df["step"], y=df[key], name=name, line=dict(color=c, width=1),
                           legendgroup=name, showlegend=(i == 0)),
                row=r, col=col)

    fig.update_layout(height=600, title_text="제어 액션 (0~1)", hovermode="x unified")
    fig.update_yaxes(range=[-0.05, 1.05])
    return fig


def plot_economics(dfs: dict[str, pd.DataFrame]):
    colors = {"PPO": "#2196F3", "Rule-Based": "#FF5722"}
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["누적 순이익 (€/m²)", "스텝별 보상"],
                        shared_xaxes=True, vertical_spacing=0.1)
    for name, df in dfs.items():
        c = colors.get(name, "#888")
        cum_profit = df["profit"].cumsum()
        fig.add_trace(go.Scatter(x=df["step"], y=cum_profit, name=name,
                                 line=dict(color=c, width=2),
                                 legendgroup=name, showlegend=True),  row=1, col=1)
        fig.add_trace(go.Scatter(x=df["step"], y=df["reward"], name=name,
                                 line=dict(color=c, width=1, dash="dot"),
                                 legendgroup=name, showlegend=False), row=2, col=1)

    fig.add_hline(y=0, row=1, col=1, line_dash="dash", line_color="black", annotation_text="손익분기")
    fig.update_layout(height=500, title_text="경제 지표", hovermode="x unified")
    return fig


def plot_harvest(dfs: dict[str, pd.DataFrame]):
    colors = {"PPO": "#2196F3", "Rule-Based": "#FF5722"}
    fig = go.Figure()
    for name, df in dfs.items():
        c = colors.get(name, "#888")
        harvest_kg = df["fruit_dm"] * 1e-6 / 0.065
        fig.add_trace(go.Scatter(x=df["step"], y=harvest_kg, name=name,
                                 line=dict(color=c, width=2)))
    fig.update_layout(title="누적 수확량 (kg/m²)", xaxis_title="스텝",
                      yaxis_title="kg/m²", height=350, hovermode="x unified")
    return fig


# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="GreenLight 시뮬레이터", layout="wide", page_icon="🌿")
st.title("🌿 GreenLight-Gym2 온실 시뮬레이션 대시보드")
st.caption("PPO 강화학습 vs Rule-Based 컨트롤러 비교")

env_kwargs, default_scenarios, rb_params = load_resources()

# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 시뮬레이션 설정")

    controller_mode = st.radio(
        "컨트롤러",
        ["PPO만", "Rule-Based만", "둘 다 비교"],
        index=2,
    )

    st.subheader("날씨 시나리오")
    location   = st.selectbox("위치", ["Amsterdam"], index=0)
    growth_year = st.selectbox("연도", [2010, 2011, 2012], index=0)
    start_day  = st.slider("시작일 (day of year)", 1, 300, 59)
    seed       = st.number_input("랜덤 시드", value=0, min_value=0, step=1)

    scenario = {"location": location, "growth_year": growth_year, "start_day": start_day}

    run_btn = st.button("▶ 시뮬레이션 실행", type="primary", use_container_width=True)

# ── 실행 ─────────────────────────────────────────────────────────────────────
if run_btn:
    dfs = {}
    with st.spinner("시뮬레이션 실행 중... (60일 에피소드, 약 30~60초)"):
        if controller_mode in ("Rule-Based만", "둘 다 비교"):
            dfs["Rule-Based"] = run_episode_rb(env_kwargs, rb_params, scenario, int(seed))
        if controller_mode in ("PPO만", "둘 다 비교"):
            dfs["PPO"] = run_episode_ppo(env_kwargs, scenario, int(seed))

    st.success(f"완료! ({len(next(iter(dfs.values())))} 스텝)")

    # ── 핵심 지표 ─────────────────────────────────────────────────────────
    st.subheader("📊 핵심 지표 요약")
    cols = st.columns(len(dfs))
    for col, (name, df) in zip(cols, dfs.items()):
        m = summary_metrics(df)
        with col:
            st.markdown(f"### {name}")
            st.metric("에피소드 총 보상", f"{m['에피소드 총 보상']:,.0f}")
            st.metric("순이익 (€/m²)",   f"{m['순이익 (€/m²)']:.2f}")
            st.metric("총 수확량 (kg/m²)",f"{m['총 수확량 (kg/m²)']:.2f}")
            st.metric("습도 위반률",       f"{m['습도 위반 (%)']:.1f}%")
            st.metric("CO₂ 위반률",        f"{m['CO2 위반 (%)']:.1f}%")

    if len(dfs) == 2:
        names = list(dfs.keys())
        m0 = summary_metrics(dfs[names[0]])
        m1 = summary_metrics(dfs[names[1]])
        st.subheader("📈 PPO vs Rule-Based 상세 비교")
        rows = []
        for k in m0:
            rows.append({"지표": k, names[0]: m0[k], names[1]: m1[k],
                         "차이": m1[k] - m0[k] if isinstance(m1[k], float) else "-"})
        cmp_df = pd.DataFrame(rows).set_index("지표")
        st.dataframe(cmp_df.style.format(precision=2), use_container_width=True)

    # ── 차트 ──────────────────────────────────────────────────────────────
    st.subheader("🌡️ 실내 기후")
    st.plotly_chart(plot_climate(dfs), use_container_width=True)

    st.subheader("🎛️ 제어 액션")
    st.plotly_chart(plot_controls(dfs), use_container_width=True)

    st.subheader("💰 경제 지표")
    st.plotly_chart(plot_economics(dfs), use_container_width=True)

    st.subheader("🍅 수확량")
    st.plotly_chart(plot_harvest(dfs), use_container_width=True)

else:
    st.info("왼쪽 사이드바에서 설정을 선택한 후 **▶ 시뮬레이션 실행** 버튼을 누르세요.")
    st.markdown("""
    ### 대시보드 기능
    - **PPO vs Rule-Based** 컨트롤러 나란히 비교
    - **실내 기후**: 온도 / 습도 / CO₂ 시계열
    - **제어 액션**: 난방·조명·환기 등 6개 제어값
    - **경제 지표**: 누적 순이익, 수익·비용 분해
    - **수확량**: 60일간 토마토 누적 생산량 (kg/m²)
    """)
