# GreenLight-Gym2 코드 리뷰

> 대상 커밋: `BartvLaatum/GreenLight-Gym2` `master` (8350306)  
> 버전: `gl-gym 0.3.1`  
> 리뷰 범위: 전체 소스코드 (`gl_gym/`, `RL/`, `common/`, `experiments/`, `tests/`)

---

## 아키텍처 개요

```
GreenLightEnv (gymnasium.Env)
├── ODE (CasADi CVODES integrator)          ← 28-state 온실-작물 물리 모델
├── NamedControlActionScheme                 ← delta 액션 + 정규화
├── WeatherRepository                        ← 날씨 CSV 캐싱 로더
├── BaseWeatherSampler (Fixed/Random/Cycling)
├── BaseObservations (5가지 플러그인)
├── BaseReward (GreenhouseReward)
└── BaseParameterProvider (Fixed/Randomized/Set)
```

설계의 주요 장점:
- **Context 객체 패턴** (`StepContext`, `RewardContext`) — 환경 상태를 reward/observation 모듈에 전달 시 env 직접 참조 없이 처리
- **플러그인 레지스트리** (`OBSERVATION_MODULES`, `REWARDS_MODULES`, `WEATHER_SAMPLERS`, `PARAMETER_PROVIDERS`) — YAML 설정에서 string으로 모듈 선택 가능
- **`WeatherRepository` 캐시** — 동일 시나리오 재사용 시 CSV 재로딩 방지

---

## 버그 (수정 필요)

### BUG-1: `_control_violation` 항상 0 반환
**파일**: `gl_gym/components/rewards.py:182-198`

```python
def _control_violation(self, ctx):
    if ctx.hour_of_day >= 20:
        if ctx.u[4] > 0:
            self.lamp_violation = 1   # ← self.lamp_violation 설정
    lamp_violation = 0                # ← 지역변수 0으로 초기화
    return lamp_violation             # ← 지역변수 반환 (항상 0)
```

`self.lamp_violation = 1`로 인스턴스 변수를 설정하지만 반환하는 건 지역변수 `lamp_violation = 0`입니다. 야간 조명 페널티(`pen_lamp`)가 절대 적용되지 않습니다.

**수정안**:
```python
def _control_violation(self, ctx):
    if ctx.hour_of_day >= 20 and ctx.u[4] > 0:
        return 1
    return 0
```

---

### BUG-2: `_output_violations` 중간 변수 미사용
**파일**: `gl_gym/components/rewards.py:166-180`

```python
def _output_violations(self, ctx):
    ...
    co2_violation = lowerbound[0] + upperbound[0]   # ← 미사용
    temp_violation = lowerbound[1] + upperbound[1]  # ← 미사용
    rh_violation = lowerbound[2] + upperbound[2]    # ← 미사용
    return lowerbound + upperbound                   # 직접 반환
```

`co2_violation`, `temp_violation`, `rh_violation`이 계산되지만 반환값에 쓰이지 않습니다. 데드코드이며 혼란을 줄 수 있습니다.

---

### BUG-3: 테스트 전반적으로 깨진 상태
**파일**: `tests/env_test.py`

| 줄 | 문제 |
|----|------|
| 21, 28, 75 | `self.env.reward` → 속성 없음 (실제: `self.env.reward_fn`) |
| 82 | `self.env.action_to_control()` → 메서드 없음 (실제: `action_scheme.to_full_control_input()`) |
| 94, 100 | `self.env_base_params` → setUp에서 정의되지 않음 |

현재 테스트 파일은 실행하면 `AttributeError`로 즉시 실패합니다. API 변경 이후 테스트가 업데이트되지 않은 것으로 보입니다.

---

### BUG-4: `samplers.py` 존재하지 않는 타입 임포트
**파일**: `gl_gym/components/samplers.py:4`

```python
from gl_gym.core.types import Scenario  # ← Scenario 타입 없음
```

`types.py`에는 `WeatherScenario`만 존재하며 `Scenario`는 정의되어 있지 않습니다. 이 파일을 임포트하면 `ImportError`가 발생합니다. `samplers.py` 자체가 나머지 코드에서 임포트되지 않고 있어서 현재는 문제가 되지 않지만, 잔류 파일입니다.

---

## 코드 품질 문제

### Q-1: 넓은 `except:` 절 (안티패턴)
**파일**: `gl_gym/environments/greenlight_env.py:248`

```python
try:
    ...
    res = self.F(x0=..., u=..., p=...)
    self.x = res["xf"].full().flatten()
except:
    print("Error in ODE approximation")
    self.truncated = True
```

베어 `except:`는 `KeyboardInterrupt`, `SystemExit` 등 모든 예외를 잡습니다. 디버깅을 극도로 어렵게 만들며, 단순 `print`만 하고 상태를 `truncated = True`로만 설정하는 것은 근본 원인을 숨깁니다.

**수정안**:
```python
except Exception as e:
    print(f"ODE integration failed: {e}")
    self.truncated = True
```

---

### Q-2: `delta_u_max` 중복 계산 및 미사용
**파일**: `gl_gym/environments/greenlight_env.py:73`, `gl_gym/components/actions.py:31`

`GreenLightEnv.__init__`에서:
```python
self.delta_u_max = self.u_max * delta_u_max  # 계산 후 저장...
```

`NamedControlActionScheme` 생성 시:
```python
self.action_scheme = NamedControlActionScheme(
    nu=self.nu,
    controlled_inputs=controlled_inputs,
    low=self.u_min,
    high=self.u_max,
    normalize_actions=normalize_actions,
    # delta_u_max 전달 없음 → 기본값 0.1 사용
)
```

`GreenLightEnv`가 계산한 `self.delta_u_max`는 `NamedControlActionScheme`에 전달되지 않습니다. YAML에서 `delta_u_max: 0.1`이고 `u_max=[1,1,...]`이므로 우연히 값이 같지만, 다른 값으로 설정하면 무시됩니다.

---

### Q-3: `WeatherRepository` 캐시 키 불완전
**파일**: `gl_gym/components/weather.py:31`

```python
key = (location, growth_year, start_day)  # season_length, pred_horizon, dt, nd 제외
```

같은 `(location, growth_year, start_day)`에 다른 `season_length`, `dt`로 호출하면 처음 캐시된 배열을 반환합니다. 현재는 단일 env 설정 내에서 이 파라미터들이 변하지 않아서 문제 없지만, 향후 멀티-설정 실험에서 잠재적 버그입니다.

**수정안**: 캐시 키에 `season_length`, `pred_horizon`, `dt`, `nd` 추가.

---

### Q-4: `GreenhouseReward.__init__`의 더미 RewardContext
**파일**: `gl_gym/components/rewards.py:46-73`

```python
ctx = RewardContext(
    t=0, dt=dt, Np=0,
    x_prev=np.zeros(26), x=np.zeros(26), u=np.zeros(6),
    p=p, d=np.zeros(10), obs={}, day_of_year=0, hour_of_day=0,
)
self.max_profit = self.max_profit_reward(ctx)  # ctx.p, ctx.dt만 사용
self.min_profit = self.min_profit_reward(ctx)
```

`max_profit_reward`/`min_profit_reward`는 `ctx.p`와 `ctx.dt`만 사용합니다. 더미 `RewardContext`를 만드는 것보다 `p`와 `dt`를 직접 받는 게 더 명확합니다.

---

### Q-5: magic index로 파라미터 접근
**파일**: `gl_gym/components/rewards.py:93, 106-108, 134-136`

```python
max_gains = ctx.p[154] * ctx.dt * 1e-6 / self.dmfm * ...
max_heating = ctx.p[108] / ctx.p[46] * ...
elec_use = ctx.u[4] * ctx.p[172] * ...
```

208개 파라미터 벡터를 숫자 인덱스로만 접근합니다. `greenlight_parameters.py`에 `ParameterRegistry`가 이미 있으므로 이름 기반 접근으로 전환하면 가독성과 안전성이 크게 향상됩니다.

---

### Q-6: 날씨 예보 관측이 경계 검사 없음
**파일**: `gl_gym/components/observations.py:144-148`

```python
def compute_obs(self, ctx: StepContext) -> np.ndarray:
    forecast = []
    for i in range(1, self.Np+1):
        forecast.extend(ctx.d[ctx.t+i][0:5])  # ← ctx.t + Np가 배열 범위 초과 가능
```

에피소드 종료 근처에서 `ctx.t + Np > len(ctx.d)`일 경우 `IndexError`가 발생할 수 있습니다. `WeatherForecastObservations`는 기본 설정에서 활성화되어 있지 않지만, 사용 시 주의 필요합니다.

---

## 확장성 관찰

### E-1: `RuleBasedController` 파라미터 28개
**파일**: `gl_gym/components/rule_based.py`

생성자가 28개 개별 파라미터를 받습니다. dataclass나 config dict 패턴으로 전환하면 사용성이 크게 향상됩니다.

### E-2: `RL/utils.py`의 W&B 강제 의존
`RL/utils.py`에서 `import wandb`와 `from wandb.integration.sb3 import WandbCallback`이 최상위에 있어, W&B 없이 환경만 사용하려 해도 wandb 임포트가 강제됩니다. 기본 `gl_gym` 패키지 자체는 W&B 의존성이 없으므로 `RL/utils.py`를 분리하거나 lazy import가 바람직합니다.

### E-3: ODE 파라미터 매직 인덱스 (모델 코어)
**파일**: `gl_gym/models/GreenLight/ode.py`, `aux_states.py`

`p[122]`, `p[112]` 등 208개 파라미터 전체가 인덱스로만 접근됩니다. 도메인 지식 없이는 이해하기 매우 어렵습니다. 주석이나 Named Tuple이라도 있으면 유지보수성이 크게 향상됩니다.

---

## 요약

| 분류 | 항목 | 심각도 |
|------|------|--------|
| 버그 | `_control_violation` 항상 0 반환 (야간 조명 페널티 미적용) | **높음** |
| 버그 | 테스트 파일 전반적 API 불일치 (실행 즉시 실패) | **높음** |
| 버그 | `samplers.py` 존재하지 않는 타입 임포트 | 보통 |
| 버그 | `_output_violations` 데드코드 변수 | 낮음 |
| 품질 | 넓은 `except:` 절 | 보통 |
| 품질 | `delta_u_max` GreenLightEnv→ActionScheme 전달 누락 | 보통 |
| 품질 | `WeatherRepository` 캐시 키 불완전 | 낮음 |
| 품질 | magic index로 파라미터 접근 (rewards, ode) | 낮음 |
| 품질 | W&B 강제 의존 (`RL/utils.py`) | 낮음 |

**핵심 수정 권장**: BUG-1 (야간 조명 페널티), BUG-3 (테스트 수정), Q-1 (bare except).
