> 과거 실험 기록입니다. 아래 _exports 경로의 데이터 사본은 삭제되었으며 재실행 명령은 현재 사용할 수 없습니다. 현재 저장 방식은 [데이터 규격](COLLECTION_DATA_SPEC_KO.md)을 따릅니다.

# 두 장면으로 Mission1 / Mission2 선택 학습

2026-09-09. 이미지 두 장의 정답을 구분하는 초기 테스트다. 독립된 검증/테스트 데이터는 아직 없다.

| 수집 장면 | 승인 정답 |
| --- | --- |
| 검은 스틱이 보이는 장면 | `start / Mission2` |
| 검은 스틱이 보이지 않는 장면 | `start / Mission1` |

사용자 확인에 따라 두 번째 프레임의 라벨을 `unknown`에서 `start / Mission1`로 수정하고 에피소드를 종료했다. Mission1의 설명과 실행 조건도 스틱이 보이지 않을 때로 변경했다(revision 42). 경로, BT, 완료/중단 조건은 그대로다.

## 데이터와 설정

- 원본 에피소드: `5d437df5-405d-4be1-8f03-0ecc3f551fff`, `6f4c9a81-06d8-43a5-82da-38960b152da8`.
- 원본 export: `docker/workspace/dataset/reasoning/gazebo_warehouse/_exports/4a3df569-0b9c-44b5-ba71-c41423969f0e`.
- 실제 학습 입력: `docker/workspace/dataset/reasoning/gazebo_warehouse/_exports/stick-presence-2scene-v1`.
- 수집 당시 카탈로그와 export는 보존했다. 학습 입력 사본에서 Mission1 카탈로그를 현재 조건으로 바꾸고, system prompt를 Operate의 영어 출력 프롬프트와 맞췄다. 변경 내역과 해시는 `manifest.json`에 기록했다.
- 두 샘플의 언어 입력은 같고 이미지와 정답만 다르다. 시나리오 설명이나 정답을 입력에 넣지 않는다.
- Train 2 / validation 0 / test 0. 기존 SAM2 마스크는 보존하며 이번 학습 loss에는 사용하지 않는다.
- 베이스: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- 베이스 모델에서 새 NF4 QLoRA 학습. 기존 한 장 학습 어댑터에 이어 학습하지 않는다.
- Rank 8 / alpha 16, learning rate `2e-4`, 120 steps(샘플마다 60회), seed 42.
- 출력: `docker/workspace/model/qwen/black-stick-2scene-lora-v1`.

## 확인 범위

학습 이미지 재입력으로 두 정답을 구분하는지 확인한다. 이는 새로운 시점·조명·가림에서도 스틱 유무를 구분한다는 성능 검증이 아니다. 두 장은 카메라 시점도 다르므로 스틱 자체 대신 배경을 외웠을 수 있다.

이번 두 정답은 실행 대기 상태의 `start` 학습이다. 실행 중 `continue`/`cancel`이나 언어 중단 지시의 학습 데이터는 추가하지 않았다. 실행 중 다른 `start`는 기존 실행 코드가 막는다.

## 결과와 적용

- 학습 120 steps 완료, 학습 루프 약 128초. 최대 PyTorch GPU reserved 약 14.75 GiB.
- 학습 후 두 샘플의 decision/mission ID/reason 재현 성공.
- 실제 `qwen_server`에 어댑터를 새로 로드한 뒤 대기 상태 두 이미지의 선택과 설명 모두 정답과 일치했다.
- 같은 이미지에 직전 미션 완료 이력을 추가한 두 검사에서도 선택 ID는 맞았다. 하지만 스틱 있음 이미지에서 ID는 Mission2인데 reason은 스틱 없음/Mission1로 출력되는 불일치가 있었다. 전체 출력이 모두 정확한 것은 아니다.
- 재로드 검사 4건의 decision/mission ID 일치: 4/4. reason 정답 문자열 일치: 3/4. 새로운 장면에 대한 성능 수치가 아니다.
- 새 어댑터를 `docker/.env`의 `QWEN_ADAPTER_PATH=/workspace/model/qwen/black-stick-2scene-lora-v1`로 설정하고 `qwen_server`를 재생성했다.
- Operate에서 맵/위치 추정을 확인한 후 Run을 누르면 사용한다. 학습/검증 과정에서 미션 이동을 실행하지 않았다.

모델 폴더의 `training_report.json`, `reloaded_evaluation.json`, `training_manifest.json`에서 설정과 출력을 확인한다. 이전 `black-stick-lora-smoke` 어댑터도 보존되어 있다.

## 재학습 명령

`cyclo_intelligence` 디렉터리에서 실행한다. 출력 이름은 아직 존재하지 않는 새 이름으로 바꾼다.

```bash
docker run --rm --gpus device=0 --shm-size 4g \
  -v "$PWD/docker/huggingface:/root/.cache/huggingface" \
  -v "$PWD/docker/workspace/dataset/reasoning/gazebo_warehouse/_exports/stick-presence-2scene-v1:/data:ro" \
  -v "$PWD/docker/workspace/model/qwen:/output" \
  -v "$PWD/cyclo_brain/policy/qwen/training:/training:ro" \
  mission-reasoning/qwen-training:0.1.0 /training/train_lora.py \
  --dataset /data --output /output/black-stick-2scene-lora-v2 \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --steps 120
```
