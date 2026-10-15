> 과거 실험 기록입니다. 아래 _exports 경로의 데이터 사본은 삭제되었으며 재실행 명령은 현재 사용할 수 없습니다. 현재 저장 방식은 [데이터 규격](COLLECTION_DATA_SPEC_KO.md)을 따릅니다.

# Collect 데이터 → Qwen LoRA → Gazebo 미션 연결 테스트

2026-09-09, `gazebo_warehouse`에서 수집한 데이터로 실행한 초기 CLI 테스트 기록이다.
이후 추가한 UI 연결은 [Operate 사용법](OPERATE_KO.md)을 참고한다.
이번 도구는 **한 번의 판단으로 기존 미션 하나를 실행하는 연결 테스트**다.
Operate의 Run 버튼에 학습/미션 선택을 연결하거나, 실행 중 조건을 계속 감시하는 기능은 포함하지 않는다.

## 사용한 데이터와 모델

- 원본 에피소드: `5d437df5-405d-4be1-8f03-0ecc3f551fff` (종료 후 export)
- Export: `docker/workspace/dataset/reasoning/gazebo_warehouse/_exports/bd30df4b-8fdd-4c86-b8d6-d8286fb2def4`
- 승인된 이미지 1장, decision 1개, mask 1개. Train 1 / validation 0 / test 0.
- 모델: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- NF4 QLoRA, language backbone의 linear layers, rank 8 / alpha 16, 60 steps, learning rate 0.0002.
- 이미지와 미션 카탈로그를 입력하고 승인된 `decision / mission_id / reason` JSON만 loss 대상으로 학습한다.
- SAM2 마스크는 보존하지만 이 학습에서는 attention supervision으로 사용하지 않는다.

저장 위치: `docker/workspace/model/qwen/black-stick-lora-smoke/`

| 파일 | 의미 |
| --- | --- |
| `adapter_model.safetensors`, `adapter_config.json` | 약 62 MiB의 LoRA 가중치 및 설정. 베이스 모델이 별도로 필요하다. |
| `training_report.json` | 데이터 해시, 설정, loss, 학습 전후 출력 |
| `candidate_load.json`, `reloaded_replay.json` | 새 프로세스에서 저장된 어댑터를 로드한 결과 |
| `live-mission-test-2/observation.jpg`, `events.jsonl` | 실제 카메라 입력, Qwen 판단, Nav2/BT 실행 기록 |
| `live-mission-test-2/continuation.jsonl` | 빈 BT 처리 수정 후 마지막 구간을 수동으로 이어 실행한 기록 |
| `negative_scene.jpg`, `negative_scene_check.json` | 막대기가 보이지 않는 별도 장면 평가. 학습 라벨로 추가한 데이터는 아니다. |

## 확인 결과

학습 전 같은 이미지에서 `unknown`, 학습 후 및 어댑터 재로드 후에는 `start / Mission2`가 나왔다.
현재 카메라의 막대기가 보이는 장면에서도 같은 판단이 나왔고, 해당 추론은 약 3.1초였다.

Mission2의 ID는 `0caaa044-d093-4c23-bd01-7d3f988c4bdd`, revision은 48이다.
이 미션은 GR00T 호출이 아니라 Nav2 주행과 JointControl BT로 구성되어 있다.

1. 1~3번 지점: Nav2 `SUCCEEDED`, local BT `completed`.
2. 4번 `Move Through`: Nav2 `SUCCEEDED`. 테스트 도구가 비어 있는 BT를 실행하려 해 중단됨.
3. 기존 Mission Run과 같이 빈 BT를 이동 전용 지점으로 처리하도록 수정하고 회귀 테스트 추가.
4. 마지막 5번 시작 지점으로 **수동 continuation**: Nav2 `SUCCEEDED`, local BT `completed`.

따라서 Qwen의 선택이 실제 Gazebo 주행·BT 실행으로 연결되는 것은 확인했다.
첫 실행이 수정 없이 끝까지 성공한 것은 아니다. 전체 자동 재실행은 아직 수행하지 않았다.
BT의 성공 상태만으로 물체를 실제로 집었는지까지 검증한 것은 아니다.

## 확인된 오판과 다음 데이터

막대기가 보이지 않는 상자·벽 장면에서도 `start / Mission2`라고 답했다.
실제 출력 이유는 `A black stick is visible standing upright in the black compartment of the yellow box.`였다.
따라서 현재 어댑터를 지속 자율 실행에 연결하지 않았다. 검증용 컨테이너는 테스트 종료 후 정지했다.

다음 수집에서는 막대기가 보이는 장면의 `start`뿐 아니라, 없는 장면의 `wait`,
가려지거나 식별이 어려운 장면의 `unknown`을 함께 승인 라벨로 저장한다.
같은 장면의 연속 프레임을 무작위로 train/test에 나누지 말고, 독립된 scene group으로 평가한다.
기존 한 장의 SAM2 마스크를 저장했다는 것만으로 모델 attention이 해당 물체로 제한되는 것은 아니다.

## 다시 학습하기

아래 명령은 `cyclo_intelligence` 폴더에서 실행한다.
Collect에서 승인 라벨을 저장하고 에피소드를 종료한 뒤 Export training dataset을 실행한다.
새 export 경로와 **아직 존재하지 않는 모델 출력 이름**을 사용한다.

```bash
docker build -f cyclo_brain/policy/qwen/Dockerfile.training \
  -t mission-reasoning/qwen-training:0.1.0 cyclo_brain/policy/qwen

docker run --rm --gpus device=0 --shm-size 4g \
  -v "$PWD/docker/huggingface:/root/.cache/huggingface" \
  -v "$PWD/docker/workspace/dataset/reasoning/gazebo_warehouse/_exports/bd30df4b-8fdd-4c86-b8d6-d8286fb2def4:/data:ro" \
  -v "$PWD/docker/workspace/model/qwen:/output" \
  -v "$PWD/cyclo_brain/policy/qwen/training:/training:ro" \
  mission-reasoning/qwen-training:0.1.0 /training/train_lora.py \
  --dataset /data --output /output/experiment-02 \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --steps 60
```

`train.jsonl`만 학습한다. validation/test 데이터는 학습에 섞지 않는다.
현재 한 장으로 얻은 낮은 loss는 일반화 성능이 아니라 해당 예시를 외웠다는 증거에 가깝다.

## 저장된 어댑터로 추론하기

이번 검증용 컨테이너는 `qwen_lora_candidate`, API는 `127.0.0.1:7112`이다.
기존 UI의 `qwen_server:7110`과 별도다. 아래 환경으로 만들었다.

```bash
docker run -d --name qwen_lora_candidate --gpus device=0 --shm-size 2g \
  -p 127.0.0.1:7112:7110 \
  -e QWEN_MODEL_ID=Qwen/Qwen3.5-4B \
  -e QWEN_MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  -e QWEN_ADAPTER_PATH=/models/black-stick-lora-smoke \
  -v "$PWD/docker/huggingface:/root/.cache/huggingface" \
  -v "$PWD/docker/workspace/model/qwen:/models:ro" \
  -v "$PWD/cyclo_brain/policy/qwen/qwen_engine:/app/qwen_engine:ro" \
  -w /app mission-reasoning/qwen-training:0.1.0 \
  -m uvicorn qwen_engine.server:app --host 0.0.0.0 --port 7110
```

이미 생성된 컨테이너는 `docker start qwen_lora_candidate`로 시작한다.
그다음 `curl -X POST http://127.0.0.1:7112/load`를 호출한다.
`/health`에서 `adapter`와 `ready`를 확인한다.
종료는 `docker stop qwen_lora_candidate`다. 이 별도 컨테이너는 Operate의 Stop 버튼으로 종료되지 않는다.

## 한 번의 미션 실행 테스트

먼저 Gazebo를 실행하고 해당 맵으로 Nav2를 시작한 뒤 localization을 잡고 BT 서비스를 시작한다.
기본값은 **추론만 하는 dry run**이며 `--execute-simulation`을 붙여야 이동한다.
`--output`은 새 디렉터리여야 한다.

```bash
docker exec -e ROS_DOMAIN_ID=30 -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
  cyclo_intelligence bash -c '
    source /opt/ros/jazzy/setup.bash
    source /root/ros2_ws/install/setup.bash
    python3 /root/ros2_ws/src/cyclo_intelligence/cyclo_brain/policy/qwen/scripts/mission_smoke.py \
      --dataset /workspace/dataset/reasoning/gazebo_warehouse/_exports/bd30df4b-8fdd-4c86-b8d6-d8286fb2def4 \
      --output /workspace/model/qwen/black-stick-lora-smoke/check-03
  '
```

학습 export에서 system/user 입력 형식과 카탈로그만 재사용하고, 이미지는 카메라에서 새로 받는다.
정답 assistant 메시지는 추론 요청에 포함하지 않는다.
`start` 및 등록된 mission ID만 실행하며, `wait`/`unknown`은 실행하지 않는다.
잘린 응답, 잘못된 JSON, 변경된 미션 revision/조건, 다른 지도, 오래된 판단은 실행하지 않는다.
Gazebo clock과 controller의 simulation time을 확인하고 기존 미션이 실행 중이면 시작하지 않는다.
Nav2 완료 후 각 local BT를 실행하며 타임아웃/실패 시 중단한다.
UI와 달리 이동 구간을 하나씩 실행하는 도구이며, 연속 조건 판단·취소 판단·음성 개입·Mission1 자동 연계는 구현하지 않았다.

관련 테스트: Qwen API/미션 선택/빈 BT 처리 12개 통과.
구현 참고: [PEFT 양자화 학습](https://huggingface.co/docs/peft/developer_guides/quantization).
