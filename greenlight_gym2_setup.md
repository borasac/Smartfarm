# GreenLight-Gym2 설치 및 테스트 기록

## 레포지토리
- 원본: https://github.com/BartvLaatum/GreenLight-Gym2
- 클론 위치: `/home/user/GreenLight-Gym2`

## 프로젝트 개요

**GL-Gym** — Gymnasium 기반 강화학습 환경: 온실 토마토 재배 최적 제어

| 항목 | 내용 |
|------|------|
| 버전 | `gl-gym 0.3.1` |
| Python 요구사항 | >= 3.11 |
| 라이센스 | AGPL-3.0-or-later |
| 핵심 모델 | GreenLight (CasADi 기반) |

## 환경 스펙 (`gl_gym/GreenLightTomato-v0`)

| 항목 | 값 |
|------|----|
| 에피소드 길이 | 60일 (5,760 스텝) |
| 스텝 간격 | 900초 (15분) |
| 행동 공간 | `Box(-1, 1, (6,))` — 6개 제어 변수 |
| 관측 공간 | Dict (5개 카테고리) |
| 보상 | 수익 기반 + 실내 환경 위반 패널티 |

### 6개 제어 변수 (Action Space)
1. `uBoil` — 난방 보일러
2. `uCO2` — CO2 주입
3. `uThScr` — 열 스크린
4. `uVent` — 환기
5. `uLamp` — 보조 조명
6. `uBlScr` — 블랙아웃 스크린

### 5개 관측 카테고리 (Observation Space)
- `BasicCropObservations` — 작물 상태
- `ControlObservations` — 현재 제어값
- `IndoorClimateObservations` — 실내 기후
- `WeatherObservations` — 날씨 예보
- `TimeObservations` — 시간 정보

## 설치 과정

```bash
# 기본 설치
git clone https://github.com/BartvLaatum/GreenLight-Gym2.git
cd GreenLight-Gym2
pip install -e .

# 훈련 의존성 추가 설치 (PyTorch, SB3, wandb 등)
pip install -e ".[train]"
```

### 설치된 주요 패키지
- `casadi 3.7.2`
- `numpy 1.26.4`
- `scipy 1.17.1`
- `gymnasium 1.3.0`
- `pandas 3.0.3`
- `torch 2.12.0+cu130`
- `stable-baselines3 2.8.0`

## PPO 훈련 테스트 결과

`greenlight_gym2_quick_train.py` 실행 결과:

| 항목 | 값 |
|------|----|
| 총 스텝 | 10,000 |
| 환경 수 (n_envs) | 2 |
| 처리 속도 | ~340 fps (CPU) |
| 소요 시간 | ~30초 |
| explained_variance | 0.027 → 0.395 (학습 진행 확인) |

## 훈련 설정 파일

- **환경 설정**: `gl_gym/configs/envs/GreenLightEnv.yml`
- **PPO 하이퍼파라미터**: `configs/agents/ppo.yml`
- **SAC 하이퍼파라미터**: `configs/agents/sac.yml`
- **RecurrentPPO**: `configs/agents/recurrentppo.yml`

## 다음 단계

1. **본격 훈련** (W&B 계정 필요)
   ```bash
   bash run_scripts/train_rl.sh
   ```

2. **Rule-based 기준점 평가**
   ```bash
   python experiments/evaluate_baseline.py
   ```

3. **환경 커스터마이징** — `gl_gym/configs/envs/GreenLightEnv.yml` 수정
   - 날씨 데이터 위치, 시즌 길이, 보상 함수 파라미터 등 조정 가능
