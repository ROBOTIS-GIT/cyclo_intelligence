# LeRobot 정책 검증 결과 (2026-05-08 ~ 09)

이 디렉토리는 ffw_sg2 데이터셋(`Dongkkka/Task_0013_clean_cafe_table_paper_lerobot`)을 이용한 LeRobot 정책 학습/추론 검증 결과를 담음.

## 파일

| 파일 | 내용 |
|---|---|
| [TRAINING_RESULTS.md](TRAINING_RESULTS.md) | 7개 정책 학습 시도 결과 + 실패 사유 분석 (Phase v1~v5) |
| [TRAINED_MODELS.md](TRAINED_MODELS.md) | 학습 완료된 체크포인트 인벤토리 (ACT 5000 step + smolvla 500 step) |
| [INFERENCE_TEST_RESULTS.md](INFERENCE_TEST_RESULTS.md) | LOAD 9개 체크포인트 검증 (로컬 2 + HF 사전훈련 7) |

## TL;DR — 최종 결과 요약

### 학습된 모델 (2개)

| Policy | Steps | 결과 |
|---|---|---|
| **ACT** | 5,000 | ✅ 풀 학습 + LOAD 통과, action_keys 정확 |
| **SmolVLA** | 500 | ✅ 학습 + LOAD 통과 (단, 품질 검증용 step 수 부족) |

### LOAD 검증 (9개)

| 결과 | 케이스 |
|---|---|
| ✅ FULL SUCCESS (2) | ACT_5000, smolvla_500_v2 — 우리 학습 ckpt |
| 🟡 PARTIAL (4) | xvla-base, xvla-libero, xvla-widowx, smolvla_base — 가중치 OK + io 매핑 mismatch (ffw vs 정책 학습 환경 카메라 키 다름) |
| ❌ FAIL (3) | pi0_base (Jetson timeout), pi05_base (옛 포맷), diffusion_pusht (옛 포맷) |

### 핵심 인프라 fix들 (2026-05-08~09 사이클)

1. **dustynv/lerobot 베이스 이미지**로 전환 — Jetson cu126 cp312 휠 부재 우회.
2. **`PIP_INDEX_URL=pypi.org/simple`** Dockerfile에 추가 — 죽은 jetson-ai-lab.dev 도메인 우회.
3. **lerobot 서브모듈 + zenoh_ros2_sdk 서브모듈** 초기화 (LOAD 인프라 작동 전제).
4. **카메라 명명 통일** — 데이터 수집 + robot_configs를 `rgb.cam_<side>_<part>`로 일치 (7개 파일 변경).
5. **`lerobot_engine` policy_type 감지 버그 fix** — HF Hub id를 ACT로 폴백하던 버그 수정 → 모든 정책 종류 dispatch 가능.
6. **Dockerfile extras 확장** — `dataset, training, async, peft, diffusion, multi_task_dit, wallx, pi, smolvla, xvla, hilserl` (production은 학습 extras 제외).
7. **데이터셋 v2.1 → v3.0 변환** + 고아 task_index sanitize.

### Jetson Orin의 한계 (관찰)

- VLA 정책(xvla / pi0 / pi05) **from-scratch 또는 fine-tune 학습은 사실상 불가** (OOM 또는 timeout).
- pi0/pi05 같은 **큰 VLA의 inference cold-load도 timeout 가능** (2GB+ weight, 600s 한도 초과).
- ACT, smolvla 정도까지는 학습 + 추론 모두 가능.

## Follow-up 항목

- [ ] Docker `HEALTHCHECK` 수정 (s6-svstat 동적 경로)
- [ ] `rmw_zenohd` s6 service 자동 시작 (또는 talos `zenoh_daemon` 채택)
- [ ] 데이터셋 카메라 해상도 통일 후 diffusion / multi_task_dit 재시도 (Phase v5의 `_uniform` 데이터셋 활용 가능)
- [ ] xvla / pi0 fine-tune은 별도 GPU 서버(A100급)에서 진행 후 ckpt만 Jetson으로 가져와 inference
- [ ] 옛 포맷 HF ckpt(`pi05_base`, `diffusion_pusht`)는 `make_pre_post_processors` 호출 우회 또는 재업로드 필요
- [ ] Docker 디스크 관리 — 정기 prune 또는 외부 디스크 마운트
- [ ] `groot` extra (flash-attn ARM source build) 별도 빌드 검토
