# 학습된 LeRobot 모델 인벤토리

이번 검증 사이클에서 **학습 완료된 체크포인트**만 정리. 다른 6개 정책은 학습 단계에서 fail (자세한 사유는 `TRAINING_RESULTS.md`).

## 1. ACT — Action Chunking Transformer

| 항목 | 값 |
|---|---|
| **Policy type** | `act` |
| **Steps** | 5,000 |
| **Job name** | `act_task0013_5000` |
| **Dataset** | `Dongkkka/Task_0013_clean_cafe_table_paper_lerobot` (v3.0 변환 후) |
| **Robot type** | `ffw_sg2_rev1` |
| **학습 시간** | 3시간 13분 (Jetson Orin, 2.31s/step) |
| **batch_size** | 8 |
| **Learnable params** | 51,644,310 (~52M) |
| **Backbone** | ResNet18 (per-camera) |

### 호스트 경로

```
docker/workspace/training_outputs/act_task0013_5000/
└── checkpoints/
    ├── 005000/
    │   ├── pretrained_model/                                        (198 MB)
    │   │   ├── config.json
    │   │   ├── model.safetensors                                    207 MB
    │   │   ├── policy_preprocessor.json                             ← v3.0 신 포맷
    │   │   ├── policy_preprocessor_step_3_normalizer_processor.safetensors
    │   │   ├── policy_postprocessor.json
    │   │   ├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
    │   │   └── train_config.json
    │   └── training_state/                                          (395 MB)
    │       ├── optimizer_state.safetensors                          413 MB
    │       ├── optimizer_param_groups.json
    │       ├── rng_state.safetensors
    │       └── training_step.json
    └── last → 005000   (symlink)
```

### 컨테이너 안 경로 (LOAD 시 model_path)

```
/workspace/training_outputs/act_task0013_5000/checkpoints/last
또는
/workspace/training_outputs/act_task0013_5000/checkpoints/005000
또는
/workspace/training_outputs/act_task0013_5000/checkpoints/005000/pretrained_model
```

(엔진의 `_resolve_model_dir`가 `pretrained_model/` 자동 descend)

### Input/Output 스펙 (config.json 발췌)

**입력**:
- `observation.state` shape = (36,)
- `observation.images.rgb.cam_left_head`  shape = (720, 1280, 3)
- `observation.images.rgb.cam_right_head` shape = (720, 1280, 3)
- `observation.images.rgb.cam_left_wrist` shape = (640, 480, 3)
- `observation.images.rgb.cam_right_wrist` shape = (640, 480, 3)

**출력 모달리티 (action_keys)**:
- `arm_left`, `arm_right`, `head`, `lift`, `mobile`

### 디스크 사용량

- `pretrained_model/`만 배포한다면 198 MB
- `training_state/` 포함 (resume용): 593 MB

---

## 2. SmolVLA — Vision-Language-Action

| 항목 | 값 |
|---|---|
| **Policy type** | `smolvla` |
| **Steps** | 500 |
| **Job name** | `smolvla_500_v2` |
| **Dataset** | `Dongkkka/Task_0013_clean_cafe_table_paper_lerobot` (v3.0) |
| **Robot type** | `ffw_sg2_rev1` |
| **학습 시간** | 약 14분 (Jetson Orin) |
| **batch_size** | 8 |
| **추가 옵션** | `--policy.max_state_dim=36` (default 32 → 36으로 늘림) |
| **Backbone** | SmolVLM (Vision-Language base) |

### 호스트 경로

```
docker/workspace/training_outputs/smolvla_500_v2/
└── checkpoints/
    ├── 000500/
    │   ├── pretrained_model/                                        (1.2 GB)
    │   │   ├── config.json
    │   │   ├── model.safetensors
    │   │   ├── policy_preprocessor.json + .safetensors
    │   │   ├── policy_postprocessor.json + .safetensors
    │   │   └── train_config.json
    │   └── training_state/                                          (394 MB)
    └── last → 000500
```

### LOAD 검증

zenoh `/lerobot/inference_command` LOAD 호출 시 동일 action_keys 반환 (`arm_left, arm_right, head, lift, mobile`).

500 step은 imitation 품질 검증용으론 부족 (full training은 수만 step). 본 검증의 목적은 **학습 파이프라인이 ffw 데이터셋과 호환된다는 것 + LOAD 경로가 SmolVLA에서도 정상 동작한다는 것**.

---

## 다른 정책 — 학습 미완료 (요약)

| Policy | 시도 횟수 | 최종 사유 |
|---|---|---|
| diffusion | 3회 (v3, v4, v5) | 이미지 shape mismatch (모든 카메라 동일 shape 강제) — `validate_features()`가 native shape 기준 검증 |
| multi_task_dit | 3회 | 동일 |
| wall_x | 3회 | state_dim/action_dim cap 늘려도 내부 hardcoded 계산에 negative tensor dim 발생; 사전훈련 401 |
| xvla | 2회 (v4 fine-tune) | exit 137 (OOM); Jetson Orin 메모리 한계 |
| pi0 | 1회 | exit 124 (2h timeout); Jetson에서 step rate 너무 느림 |
| pi05 | 1회 | 동일 |
| sac | 1회 | `SACPolicy.__init__()` API 불일치 (lerobot 내부 버그) |

**Jetson Orin은 VLA 학습에 본질적으로 부적합** — VLA fine-tune은 별도 GPU 서버(A100/H100)에서.

---

_업데이트: 2026-05-09 오후. ACT 5000 step + smolvla 500 step의 2개 체크포인트 inventory._
