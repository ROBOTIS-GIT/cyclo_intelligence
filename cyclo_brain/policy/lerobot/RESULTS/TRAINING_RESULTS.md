# LeRobot Policy 학습 검증 결과

데이터셋: `Dongkkka/Task_0013_clean_cafe_table_paper_lerobot`
환경: Jetson Orin (ARM64), `dustynv/lerobot:r36.4-cu128-24.04` 베이스, robot_type=`ffw_sg2_rev1`
학습 명령: `lerobot-train --policy.type=<P> --tolerance_s=0.1 --steps=<N> --policy.push_to_hub=false --wandb.enable=false`

## 데이터셋 특성 (호환성에 영향 큰 항목)

- 50 에피소드 / 56,630 프레임 / 15 fps
- 4 카메라 (`rgb.cam_<side>_<part>`):
  - `rgb.cam_left_head`, `rgb.cam_right_head`: **720×1280 (ZED stereo)**
  - `rgb.cam_left_wrist`, `rgb.cam_right_wrist`: **640×480 (RealSense)**
- state 차원 = **36** (양팔 7+1×2 + 머리 2 + 리프트 1 + 베이스 3 + EE pose 7×2)
- action 차원 = 모달리티 5개 (`arm_left, arm_right, head, lift, mobile`)
- task 텍스트 1개 (`clean_cafe_table_paper`)
- v2.1 포맷 → v3.0 변환 필요 (변환 스크립트 사용)
- 변환 후 `task_index`에 고아 인덱스 [0,1,2] 존재 → 0으로 sanitize 필요

## 정책별 결과

| Policy | Steps | 결과 | 시간 | 비고 |
|---|---|---|---|---|
| **act** | 5,000 | ✅ **성공** | 3h 13m (2.31s/step) | 전 세션, 정상 체크포인트 생성 |
| **diffusion** | 500 | ❌ FAIL | 21s | 이미지 shape 동일 요구 |
| **multi_task_dit** | 500 | ❌ FAIL | 22s | 이미지 shape 동일 요구 |
| **wall_x** | 500 | ❌ FAIL | 21s | state dim 36 > max 20 |
| **xvla** | 500 | ❌ FAIL | 22s | from-scratch 미지원 (vision_config 필수) |
| **smolvla** | 500 | ❌ FAIL | 1m 17s | state dim 36 ≠ 32 |
| **pi0** | 500 | ❌ FAIL | 4m 16s | HF gated repo 401 (TOKEN 없음) |
| **sac** | 500 | ❌ FAIL | 22s | `SACPolicy.__init__()` API 불일치 |

## 실패 사유 분석

### 1. 카메라 이미지 shape mismatch (diffusion, multi_task_dit)

```
ValueError: Image 'observation.images.rgb.cam_left_wrist' shape (640, 480, 3)
         != 'observation.images.rgb.cam_left_head' shape (720, 1280, 3)
```

- 두 정책은 **모든 카메라가 동일 해상도**여야 함 (단일 인코더 공유).
- ACT는 카메라별 별도 ResNet18 인코더라 다양한 해상도 OK.
- **해결**: 데이터셋의 wrist 카메라를 720×1280으로 업샘플링하거나, 모두 같은 작은 해상도(예: 224×224)로 다운샘플 후 재업로드. 또는 정책 config에 `image_resize=(H,W)` 강제.

### 2. State dimension 초과 (wall_x, smolvla)

```
[wall_x]   ValueError: State dimension 36 exceeds max_state_dim 20.
[smolvla]  RuntimeError: tensor (32) must match (36) at dim 1.
```

- VLA 백본들이 token 수 또는 projection 차원에 hard cap이 있음 (wall_x=20, smolvla=32).
- ffw_sg2의 36차원 state(EE pose 14차원 포함)가 cap 초과.
- **해결**: ① state를 21~32차원으로 축소(EE pose 제외하면 22차원), ② 정책 config에서 `max_state_dim` 또는 동등 옵션 늘리기 (가능한 정책 한정).

### 3. 사전훈련 백본 의존 (xvla)

```
ValueError: vision_config is required
```

- xvla는 vision backbone config를 명시하거나 사전훈련 체크포인트(`--policy.path=<HF_id>`)를 base로 fine-tune하는 패턴.
- from-scratch 학습 미지원.
- **해결**: HF Hub의 xvla 사전훈련 ckpt를 베이스로 두고 fine-tune 모드로 재시도.

### 4. HF gated repo (pi0)

```
OSError: You are trying to access a gated repo.
401 Client Error.
```

- pi0의 사전훈련 weight (`lerobot/pi0_base` 등)는 access request 후 승인된 토큰 필요.
- **해결**: HF에서 access 신청 + `HF_TOKEN` env 컨테이너에 주입.

### 5. SAC API 불일치

```
TypeError: SACPolicy.__init__() got an unexpected keyword argument 'dataset_stats'
```

- RL 데이터 호환성 이전 단계, lerobot 내부 API 불일치로 초기화 자체 실패.
- 다른 정책들(ACT, Diffusion 등)은 통일된 시그니처(`dataset_stats=...`)로 호출되는데 SACPolicy만 미반영.
- **해결**: lerobot upstream 패치 또는 정책 클래스 직접 호출 (CLI 우회).

## 결론

**현 데이터셋(Dongkkka/Task_0013...)으로 from-scratch 학습 가능한 정책은 ACT뿐.** 다른 정책들은 다음 중 하나 이상 충족시켜야:

1. 데이터셋 카메라 해상도 통일 (4개 모두 같은 H×W)
2. state 차원 축소 (wall_x ≤ 20, smolvla ≤ 32, multi_task_dit는 어떨지 추가 확인 필요)
3. 사전훈련 ckpt 다운로드 + HF_TOKEN (pi0) — 또는 fine-tune 모드 (xvla)

## 향후 개선 항목 (Follow-up)

- [ ] 데이터셋 image shape 통일 후 diffusion/multi_task_dit 재시도
- [ ] state 축소 (EE pose 제거) 버전 데이터셋 + smolvla/wall_x
- [ ] HF_TOKEN 등록 후 pi0/pi05 fine-tune 시도
- [ ] xvla `policy.path` 옵션으로 사전훈련 base 받아서 fine-tune

---

## 2차 시도 (2026-05-09 새벽~오전)

각 실패 사유에 대한 fix 적용 후 재시도 (Phase A v2~v5).

### 진전된 결과

| Policy | 시도 | 결과 | 비고 |
|---|---|---|---|
| **smolvla** | `--policy.max_state_dim=36` | ✅ **성공** (500 step ~14분) | LOAD 검증도 통과, action_keys 5개 정확 |
| diffusion (v3) | `--policy.resize_shape 84 84` (space-form) | ❌ argparse error | draccus는 `=[a,b]` 형식 필요 |
| diffusion (v4) | `--policy.resize_shape=[84,84]` | ❌ shape mismatch | resize는 적용되지만 `validate_features()`가 그 전에 native shape 검증 |
| multi_task_dit (v4) | `--policy.image_resize_shape=[84,84]` | ❌ shape mismatch | 동일 사유 |
| wall_x (v3) | `--policy.max_state_dim=36` | ❌ action_dim 22 > 20 | 다음 fix로 진전 |
| wall_x (v3) | `--policy.max_state_dim=36 --policy.max_action_dim=22` | ❌ negative tensor dim | 내부 hardcoded 계산 깨짐 |
| **xvla** | `--policy.path=lerobot/xvla-base` (fine-tune) | ❌ exit 137 (OOM) | Jetson Orin 메모리 한계 |
| **pi0** | `--policy.path=lerobot/pi0_base` (fine-tune) | ❌ exit 124 (timeout 2h) | step rate 너무 느림 |
| **pi05** | `--policy.path=lerobot/pi05_base` (fine-tune) | ❌ exit 124 (timeout 2h) | 동일 |

### Phase v5 — 데이터셋 preprocessing

- 모든 카메라 240×320으로 통일된 새 데이터셋 빌드 (ffmpeg av1 인코딩, ~13분).
- 위치: `/root/.cache/huggingface/lerobot/Dongkkka/Task_0013_clean_cafe_table_paper_lerobot_uniform/`
- 학습 호출 시 `--dataset.root=$DATASET_ROOT` 추가.
- 진행 중: diffusion / multi_task_dit / wall_x (max=64/32) 재시도.

## 핵심 발견

### Jetson Orin은 VLA 학습에 부적합

- xvla / pi0 / pi05 모두 OOM 또는 timeout. Jetson Orin (64GB shared)가 VLA 백본(Gemma-2B 등)을 batch=8로 fine-tune하기엔 메모리/연산 모두 부족.
- 결론: VLA 학습은 별도 GPU 서버(A100/H100)에서. Jetson은 inference 전용으로.

### dustynv-build vs lerobot-build 미스매치

- dustynv/lerobot 베이스 이미지의 `/etc/pip.conf`가 dead 도메인 (`pypi.jetson-ai-lab.dev`) 가리킴 → `PIP_INDEX_URL=https://pypi.org/simple` 우회 (Dockerfile에 반영 완료).

### 데이터셋 자체 이슈

1. 카메라 이미지 shape 비균일 (head 720×1280 / wrist 640×480) → `validate_features()` 강제 검증하는 정책(diffusion, multi_task_dit, wall_x) 모두 거부.
2. v2.1 포맷 + v3.0 변환 후 `task_index`에 고아값 [0,1,2] → sanitize 필요.
3. State 36차원이 일부 VLA(smolvla 32, wall_x 20)의 hardcoded cap 초과 → `max_state_dim` 옵션으로 해결 가능.

---

_업데이트: 2026-05-09 새벽~오전. ACT (5000 step) + smolvla (500 step) 학습 성공. VLA 3개 (xvla/pi0/pi05) Jetson 한계로 fail. Phase v5 (uniform 데이터셋) 진행 중._

