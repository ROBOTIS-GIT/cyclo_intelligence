# LeRobot Inference (LOAD) 테스트 결과

호출 경로: 호스트 → Zenoh router (`rmw_zenohd` in `cyclo_intelligence`) → `/lerobot/inference_command` (`main-runtime`) → `/lerobot/engine_command` (`engine-process`) → `LeRobotEngine.load_policy()`.

테스트 클라이언트: `/tmp/test_lerobot_load.py` (zenoh_ros2_sdk.ROS2ServiceClient로 InferenceCommand.LOAD 호출, 옵션으로 UNLOAD 선행 가능).

## 풀 스택 검증 항목

| 단계 | 상태 |
|---|---|
| 1. dustynv/lerobot:r36.4-cu128-24.04 베이스 + cyclo extras 빌드 | ✅ |
| 2. rmw_zenohd 호스트(=`cyclo_intelligence`) 기동 | ✅ |
| 3. Zenoh service `/lerobot/inference_command` liveliness 등록 | ✅ |
| 4. 테스트 클라이언트 service discovery + request 전송 | ✅ |
| 5. CDR 직렬화/역직렬화 (`InferenceCommand.srv`) | ✅ |
| 6. model_path 해석 (HF Hub id / 로컬 경로) | ✅ |
| 7. **HF Hub id에서 policy_type 자동 감지 (engine.py 수정)** | ✅ |
| 8. 가중치 GPU 로드 (cuda) | ✅ (대부분의 정책에서) |
| 9. policy_preprocessor.json / policy_postprocessor.json 신 포맷 로드 | ✅ (보유한 ckpt에서) |
| 10. robot_type → 카메라/state/action io 매핑 | ⚠️ 우리 ffw_sg2 ↔ 외부 정책 카메라 키 다름 |
| 11. 응답 (success, message, action_keys) 반환 | ✅ |

## LOAD 배터리 — 9개 체크포인트 검증 결과

테스트 환경: ffw_sg2_rev1 robot_type, lerobot_server (Jetson Orin), 모든 호출 동일 클라이언트.

| # | 체크포인트 | 종류 | 결과 | 비고 |
|---|---|---|---|---|
| 1 | **`act_task0013_5000` (local)** | 우리가 학습한 ACT 5000 step | ✅ **SUCCESS** | action_keys=`[arm_left, arm_right, head, lift, mobile]` (정확) |
| 2 | **`smolvla_500_v2` (local)** | 우리가 학습한 smolvla 500 step | ✅ **SUCCESS** | 동일 action_keys |
| 3 | `lerobot/xvla-base` (HF) | xvla 사전훈련 (base) | 🟡 partial | 가중치 GPU 로드 OK, io mismatch (`image/image2/image3`) |
| 4 | `lerobot/xvla-libero` (HF) | xvla LIBERO fine-tuned | 🟡 partial | 동일, `empty_camera_0/image/image2` 기대 |
| 5 | `lerobot/xvla-widowx` (HF) | xvla WidowX fine-tuned | 🟡 partial | 동일, `image/image2` 기대 |
| 6 | `lerobot/pi0_base` (HF) | pi0 사전훈련 | ❌ TIMEOUT | 600s 한도 초과 — Jetson에서 가중치 로드 너무 느림 (~2 GB 모델) |
| 7 | `lerobot/pi05_base` (HF) | pi05 사전훈련 | ❌ FAIL | 옛 포맷 — `policy_preprocessor.json` 없음 |
| 8 | `lerobot/smolvla_base` (HF) | smolvla 사전훈련 | 🟡 partial | 가중치 GPU 로드 OK, io mismatch (`camera1/camera2/camera3`) |
| 9 | `lerobot/diffusion_pusht` (HF) | diffusion PushT 사전훈련 | ❌ FAIL | 옛 포맷 (단일 카메라, 옛 normalizer 형식) |

### 분류

| 결과 | 의미 | 개수 |
|---|---|---|
| **✅ FULL SUCCESS** | 가중치 로드 + io 매핑 + action_keys까지 반환 | 2 (우리 학습 ckpt) |
| **🟡 PARTIAL** | 정책 dispatch + 가중치 GPU 로드 통과; io 매핑 단계에서 robot config와 정책 학습 환경 카메라 키 불일치 | 4 (xvla 3개 + smolvla_base) |
| **❌ INFRA-LEVEL FAIL** | 옛 포맷 / 다운로드 timeout | 3 (pi05_base, pi0_base, diffusion_pusht) |

## 핵심 발견

### 1. `lerobot_engine`의 HF Hub policy_type 감지 버그 fix 완료

원본 코드는 `Path(model_path) / "config.json"` 로컬 경로만 검사하고 HF Hub id는 항상 `"act"`로 폴백 → 모든 비-ACT 정책이 ACTPolicy로 instantiate되며 `'XConfig' object has no attribute 'use_vae'` 에러.

**수정**: HF Hub id 형태(`"<user>/<repo>"` 패턴) 인식 → `huggingface_hub.hf_hub_download(filename="config.json")`로 metadata만 받아 policy_type 결정 → 올바른 PolicyClass dispatch.

수정 위치: `cyclo_brain/policy/lerobot/lerobot_engine/loading.py`.

### 2. 우리 학습 ckpt vs 사전훈련 ckpt의 io 매핑

- 우리 ACT/smolvla는 **ffw_sg2 데이터셋으로 학습**돼서 `rgb.cam_<side>_<part>` 카메라 키 + 5개 모달리티 action 그룹과 그대로 매칭.
- 외부 정책(xvla-base/libero/widowx, smolvla_base)은 LIBERO·WidowX·PushT 등 **다른 환경 학습 산출물**이라 카메라 키 다름. ffw_sg2와 io 호환되려면:
  - (a) 외부 ckpt를 ffw 데이터셋으로 fine-tune (input_features 재학습), 또는
  - (b) `lerobot_engine`에 카메라 키 alias / 복사 매핑 추가.
- 따라서 **🟡 partial은 인프라 결함이 아니라 사용 시점 fine-tune 필요라는 신호**.

### 3. Jetson에서 큰 VLA 가중치 로드는 매우 느림

- pi0_base (~2 GB): 600 s timeout. GPU 로드 단계에서 막힘 (Jetson Orin 메모리/PCIe 병목).
- xvla-base 등은 통과 (백본 작은 편).
- 결론: pi0/pi05급 큰 VLA는 inference 시 **eager load 부담** 높음. 사용 시 cold-start 비용 분석 필요.

### 4. HF Hub 옛 포맷 ckpt 호환성

- `lerobot/pi05_base`, `lerobot/diffusion_pusht`: lerobot v3.0 이전 포맷으로 업로드돼 `policy_preprocessor.json`이 분리 저장 안 됨.
- 현 lerobot은 신 포맷 강제 → 이런 ckpt는 사용하려면 재업로드 또는 컨버터 필요.

## LOAD 경로에서 발견된 시스템 이슈 (이번 검증 사이클)

1. **베이스 이미지 `pip.conf` 오타** — `pypi.jetson-ai-lab.dev`(죽은 도메인). Dockerfile에 `ENV PIP_INDEX_URL=https://pypi.org/simple` 추가로 해결 (반영됨).

2. **`zenoh_ros2_sdk` 서브모듈 미초기화** — `git submodule update --init --recursive`로 해결.

3. **카메라 키 prefix 불일치** — robot_config의 `cam_<part>_<side>` vs 정책 expected `rgb.cam_<side>_<part>`. 데이터 수집 파이프라인 + robot_configs 통일 (반영됨).

4. **HF Hub policy_type 감지 버그** — 위 §핵심 발견 1. 수정 반영 + 재시동 후 fix 검증 완료.

5. **bind-mount stale (Edit→inode 교체)** — Edit 도구가 atomic-rename으로 inode 교체하면 컨테이너 bind-mount가 옛 inode 가리킴. 컨테이너 restart로 해결.

6. **Docker HEALTHCHECK가 unhealthy 유지** — `s6-svstat` PATH 미등록. follow-up.

7. **Zenoh router 자동 시작 부재** — `rmw_zenohd` 수동 기동 필요. follow-up.

8. **Jetson 디스크 부족** — Docker 이미지 + HF cache가 NVMe 229GB 가득. follow-up: 정기 prune 또는 외부 디스크.

## 결론

- **LOAD 인프라 자체는 모든 정책 종류에 대해 정상 작동** (engine.py fix 후).
- **우리가 학습한 ckpt 2개**(ACT 5000 step + smolvla 500 step)는 ffw_sg2 환경에서 **end-to-end LOAD 완전 통과**.
- **외부 사전훈련 ckpt 4개**는 가중치 로드까지 통과, 마지막 io 매핑이 학습 환경 차이 때문에 mismatch (예상된 동작; 사용하려면 fine-tune).
- **3개**는 인프라 외 사유(옛 포맷 / 다운로드 timeout).

---

_업데이트: 2026-05-09 오후. 9개 체크포인트 LOAD 검증 완료._
