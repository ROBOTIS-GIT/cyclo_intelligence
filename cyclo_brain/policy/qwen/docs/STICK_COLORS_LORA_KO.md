> 과거 실험 기록입니다. 아래 _exports 경로의 데이터 사본은 삭제되었으며 재실행 명령은 현재 사용할 수 없습니다. 현재 저장 방식은 [데이터 규격](COLLECTION_DATA_SPEC_KO.md)을 따릅니다.

# 빨강 스틱 추가 학습

2026-09-09. `Qwen/Qwen3.5-4B` 베이스 모델에서 세 장의 이미지로 새 QLoRA 어댑터를 학습했다.

| 이미지 | 승인 정답 |
| --- | --- |
| 검정 스틱 있음 | start / Mission2 |
| 스틱 없음 | start / Mission1 |
| 빨강 스틱 있음 | start / Mission2 |

원본 Collect에는 빨강 스틱 에피소드만 남아 있어, 이전 두 장은 보존된 `stick-presence-2scene-v1` 학습 입력에서 가져왔다. 삭제된 에피소드를 Collect에 복원하지 않았다.

- 새 에피소드: `39e7c3b6-40de-4cab-b86e-447f1b7abd5a`.
- 빨강 원본 export: `docker/workspace/dataset/reasoning/gazebo_warehouse/_exports/f3811e25-eb6a-4f32-9431-39613c002807`.
- 실제 학습 입력: `docker/workspace/dataset/reasoning/gazebo_warehouse/_exports/stick-colors-3scene-v1`.
- `red_stick` 그룹은 자동 분할상 test였지만 사용자의 학습 요청에 따라 파생 학습 입력에서 train으로 사용했다. 원본 export는 그대로이며 별도 테스트 데이터로 재사용하지 않는다.
- Train 3 / validation 0 / test 0. 학습 이미지 재입력 검사는 일반화 평가가 아니다.
- Mission1/2의 설명과 조건을 “빨강 또는 검정 스틱이 보임/보이지 않음”으로 맞추었다. 원래 미션은 학습 입력의 `missions_before.json`에 보존했다. 경로와 BT는 변경하지 않았다.
- 세 샘플의 system/user 입력은 동일하며 이미지와 정답만 다르다. 이전 마스크는 원본 export에 남아 있고 이번 학습에 사용하지 않는다.
- 베이스 revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- NF4 QLoRA, rank 8 / alpha 16, learning rate `2e-4`, seed 42, 180 steps(각 샘플 60회).
- 저장 어댑터: `docker/workspace/model/qwen/stick-colors-3scene-lora-v1`.

실행 중 continue/cancel, 언어 중단 지시, 새로운 시점/조명/가림 상황에 대한 학습 데이터는 추가하지 않았다.

## 재학습 명령

`cyclo_intelligence`에서 실행하며 출력 이름은 아직 없는 새 이름으로 지정한다.

```bash
docker run --rm --gpus device=0 --shm-size 4g \
  -v "$PWD/docker/huggingface:/root/.cache/huggingface" \
  -v "$PWD/docker/workspace/dataset/reasoning/gazebo_warehouse/_exports/stick-colors-3scene-v1:/data:ro" \
  -v "$PWD/docker/workspace/model/qwen:/output" \
  -v "$PWD/cyclo_brain/policy/qwen/training:/training:ro" \
  mission-reasoning/qwen-training:0.1.0 /training/train_lora.py \
  --dataset /data --output /output/stick-colors-3scene-lora-v2 \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --steps 180
```

## 실제 서버 검증과 적용

- 학습 후 세 이미지의 승인 정답 JSON을 모두 재현했다.
- 새 프로세스의 `qwen_server`에 저장 어댑터를 로드한 뒤 대기 상태 이미지 재입력: decision/mission ID/reason 3/3 일치.
- 미션 완료 이력을 추가한 이미지 재입력: 2/3 일치. 검정 스틱 있음 이미지가 Mission2 대신 Mission1/스틱 없음으로 바뀌었다. 빨강 스틱과 스틱 없음은 정답과 일치했다.
- 합계 6개 요청 중 5개 정답 일치. 모두 학습 이미지 검사이며 독립 평가가 아니다. 완료 이력에 따른 판단 변화는 미해결 한계다.
- `docker/.env`: `QWEN_ADAPTER_PATH=/workspace/model/qwen/stick-colors-3scene-lora-v1`.
- 실제 서버 `/health`의 어댑터 경로와 ready 상태를 확인했다. Operate에서 Run을 누르면 이 모델을 사용한다.
- 기존 어댑터는 보존했다. 이번 학습/검사 과정에서는 미션 이동 명령을 보내지 않았다.

모델 폴더의 `training_report.json`, `reloaded_evaluation.json`, `training_manifest.json`에 결과 및 데이터 출처를 기록했다.
