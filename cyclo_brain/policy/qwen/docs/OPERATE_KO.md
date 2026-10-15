# Operate: Qwen 판단으로 미션 실행

Autonomy Studio의 **Autonomy → Operate**에서 사용한다. 화면을 새로고침하면 적용된 UI를 볼 수 있다.

1. **Load Map**: 환경 맵을 선택한다. 미션은 미리 선택하지 않는다.
2. **Localize**: 지도에서 로봇 위치와 방향을 지정하고 `Localized`가 될 때까지 기다린다.
3. 초록색 **Run autonomy**: 현재 맵의 `ObservationMode` 미션 실행 → 정상 완료 → `qwen_server` 시작 → 모델/LoRA 로드 → 헤드 카메라 추론 → 해당 맵의 미션 선택 → 기존 실행기로 주행 및 BT 실행 → 정상 완료 후 Qwen 재시작. Stop까지 반복한다.
4. **Current situation**: 추론에 사용한 이미지, AI 판단과 이유, 현재 미션, 이동/BT 진행률을 확인한다.
5. **Activity**: 판단, 미션 시작/완료/중단/실패 및 실행을 막은 이유를 확인한다.
6. **Manual intervention**: 미션 선택 전 `Send instruction`은 다음 추론에 언어 지시를 전달한다. `Stop current mission`은 실행 계층에 직접 취소를 요청한다.
7. 빨간 **Stop**: 판단 루프와 실행 중 미션을 중단하고 Qwen 컨테이너 및 navigation을 정지한다. 다시 실행하려면 맵 로드/위치 추정을 다시 확인한다.

Qwen의 상황 설명과 판단 이유는 영어로 직접 생성한다. 한국어로 번역하는 단계는 없다.

## Qwen3.5-0.8B·2B 시험

Operate의 **Reasoning model** 또는 Record의 **Observation → Reasoning model**에서
`Qwen3.5-0.8B (Base · Untuned)` 또는 `Qwen3.5-2B (Base · Untuned)`를 선택할 수 있다. 이 모델은 이미지와 텍스트를 받는
원본 체크포인트이며, 현재 맵의 정답으로 학습된 모델이 아니다. 선택 후 Reasoning을
시작하면 첫 로드 때 가중치를 받아 GPU에 올린다. 기존 4B 모델과 전환할 때는 실행을
Stop한 다음 다시 시작한다. 현재 사용 중인 4B LoRA는 0.8B나 2B에 적용할 수 없다.

Record의 **Dataset Annotation → Fine-tuning/Train**에는 `Base model` 선택이 있다.
0.8B 또는 2B를 선택해 새 모델명으로 학습한 뒤 Operate에서 해당 학습 모델을 선택할 수 있다.
기존 학습 데이터와 4B 모델은 유지된다. 모델 상태의 `base_model`과 `adapter`로
실제 로드된 체크포인트를 확인한다.

## 판단과 실행 규칙

- 모든 맵에 실행 가능한 `ObservationMode` 미션을 Design에서 저장한다. 해당 미션이 없거나 실패하면 초기 추론을 시작하지 않는다.

- `start`: 현재 맵에 저장된 mission ID만 허용한다. 미션의 revision과 ID를 다시 확인하고 파일 로딩이 끝난 후 실행한다.
- `wait`, `unknown`: 새 미션을 실행하지 않고 추론을 계속한다.
- 미션이 시작되면 Qwen 추론 요청만 일시정지하고 컨테이너·로드된 모델·Operate 세션은 유지한다. Fail Safe와 완료 판단은 VLA, Navigation, 기존 Nav2/BT 실행기가 담당한다.
- 미션 정상 완료 후 같은 메모리 상주 모델로 추론을 자동 재개한다. 명시적 Stop 시 컨테이너를 종료해 GPU 메모리를 해제한다. 같은 미션도 다시 선택되면 실행한다. `ObservationMode`는 Start를 누른 직후 한 번만 선행 실행한다. 이후 반복마다 추가 복귀 이동을 넣지는 않는다.
- 미션 실패·취소 또는 위치 추정 상실 시에는 자동 반복을 멈춘다. 원인을 확인한 뒤 다시 시작한다.
- `Stop current mission`은 현재 실행 계층에 직접 취소를 요청한다. 다시 Reasoning하려면 실행 종료 후 Run autonomy를 누른다.
- 잘못된 JSON, 알 수 없는 mission ID, 잘린 출력, 오래된 판단, 판단 도중 바뀐 미션 카탈로그는 실행하지 않는다.
- Stop, 맵 변경, 위치 추정 상실, 언어 지시 변경으로 무효화된 응답이나 미션 로딩 결과는 실행하지 않는다.
- 실행 관리는 현재 열린 Operate 페이지가 담당한다. 다른 탭으로 나가면 판단 루프와 이 페이지가 실행한 미션을 정지한다. 상시 백그라운드 데몬은 아니다.

## 현재 모델

이 PC의 `docker/.env`에는 다음 테스트 설정을 적용했다.

```dotenv
QWEN_MODEL_ID=Qwen/Qwen3.5-4B
QWEN_MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
QWEN_ADAPTER_PATH=/workspace/model/qwen/stick-colors-3scene-lora-v1
```

어댑터의 호스트 경로는 `docker/workspace/model/qwen/stick-colors-3scene-lora-v1`다.
`/reasoning/status`와 Qwen `/health`에서 실제 모델과 어댑터를 확인할 수 있다.
Record의 Qwen 관찰도 같은 모델 서비스를 사용한다. 모델을 바꾸려면 현재 Reasoning을 Stop하고 다른 모델을 선택해 다시 시작한다.

기존 v2 annotation 어댑터는 v3 목록에 표시하지 않는다. training report의 정답이 모두 `start / wait / unknown`인 legacy 어댑터는 계속 사용할 수 있으며, 새 Fine-tuning은 v3 계약으로 저장한다.
학습 데이터·적용 설정·검증 결과는 [빨강 스틱 추가 학습 기록](STICK_COLORS_LORA_KO.md)을 참고한다.

## 구성

- `docker/supervisor_api/reasoning.py`: `/reasoning/decide`, 현재 카메라와 Design 카탈로그를 사용한 판단/검증.
- `useAutonomyOperation.js`: 미션 선택/로딩과 실행 계층으로의 handoff.
- `useMissionRunner.js`: 기존 Nav2 → local BT → 다음 지점 실행과 취소. 이동 전용 지점의 빈 BT 처리도 재사용.
- `qwen_engine/engine.py`: 선택한 베이스 모델과 LoRA 로딩.

이번 구현에서 학습 데이터나 Mission1/Mission2의 조건·BT 내용은 수정하지 않았다.


## 1차 검증 기록 (2026-09-09)

- 판단 API 16개, 관련 UI/실행기 회귀 테스트 80개, Operate 페이지 통합 테스트 2개 통과.
- 실제 배포된 브라우저 화면에서 `gazebo_warehouse` 로드 및 위치 추정 후 Run을 실행했다.
- `qwen_server`가 저장된 4B LoRA를 로드하고 `start / Mission2`를 반환했다.
- UI가 실제 `/api/navigation/goal/wait` 주행 요청을 보냈고, 실행 중 다른 start는 차단했다.
- 이어 Stop을 눌러 미션 `cancelled`, Qwen `Stopped` 상태를 확인했다.
- 이번 UI 실동작 검증은 출발과 취소까지다. 미션 전체 완주와 모델의 상황 구분 정확도를 보장하는 시험은 아니다.

## 카메라 입력 안정화

- 추론/Collect 캡처는 `docker/supervisor_api/head_camera.py`의 단일 ROS 구독을 사용한다. JPEG를 재압축하거나 web_video_server의 `/snapshot`을 반복 호출하지 않는다.
- QoS는 best effort / volatile / depth 1이며, 메모리에 최신 JPEG 하나만 보관한다. ROS 구독은 첫 캡처 요청에서 시작해 API 종료 시 정리한다. Qwen의 시작/정지와 독립적이다.
- 마지막 새 타임스탬프의 수신 후 2초가 지난 프레임은 사용하지 않는다. 프레임을 최대 2초 기다린 후 없으면 503을 반환하고 기존 추론 루프가 재시도한다.
- 화면용 web_video_server는 ROS launch에서 종료 시 2초 후 자동 재시작한다. 현재 터미널에서 이미 실행한 bringup에는 launch 설정을 다시 읽도록 재실행해야 한다. 이 설정은 hang 감지가 아니라 프로세스 종료 후 복구다.
- 실카메라 연속 캡처 25/25 성공, 서로 다른 타임스탬프 25개, HTTP 호출 0회 확인. 별도 포트에서 web_video_server 종료 후 새 PID로 재시작 확인. 카메라/기존 reasoning 테스트 23개 통과.

## 미션 시작 연속 확인

같은 관찰 지점에서 동일 미션의 `start`가 **4회 연속** 확인되어야 실행합니다(최초 판단 + 추가 3회). 서로 다른 프레임만 계산하며, `wait`·`unknown`·오류·실행 불가 응답이나 미션/지시 변경 시 다시 확인합니다. 관찰 지점을 옮기거나 미션을 완료한 뒤에도 횟수는 새로 시작합니다.

Activity의 `Mission confirmation`에서 `1/4`부터 `4/4`까지 확인할 수 있습니다. 순회 관찰의 체류 시간 안에 확인을 마치지 못하면 다음 관찰 지점으로 이동합니다.
