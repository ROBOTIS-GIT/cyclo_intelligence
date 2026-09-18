# Inference Input Pipeline 구현과 검증

## 목적과 책임

사용자 YAML은 Cyclo의 추가 전처리와 등록 handler 선택만 담당한다.
Python adapter/handler는 기본 입력 포장과 내부 그래프 구성, 계산과 메모리를 담당하고,
Runtime은 실행 사실과 요청 시점을 담당한다.
공통 패키지는 Torch와 LeRobot에 의존하지 않는다. 이번 전환 대상은 LeRobot이며 별도 GR00T Worker는 기존 경로를 유지한다.

```text
Robot topics -> Worker RobotClient -> 필요한 최신값 / 수신 이력
                                            |
                                  Python-owned input graph
                                            |
                                 before 단계 -> saved processor
                                            |
                                 after 단계 -> model inference
                                            |
                                 saved postprocessor -> action chunk
                                            |               |
                                 result 단계 -> 상태 제안    +-> Cyclo Runtime
                                                           정렬 / 보간 / 발행
                                                                  |
Worker 상태 확정 <--- Engine context: 계획 채택 / 발행 / 종료 / 폐기 ---+
```

이미지와 feature를 Cyclo 컨테이너로 중계하지 않는다.
command 발행 성공은 물리적인 로봇 동작 완료를 의미하지 않는다.

## 파일 구성

| 위치 | 책임 |
| --- | --- |
| `policy/common/runtime/inference_inputs/graph.py` | LOAD 시 DAG 검증·컴파일, 단계별 평가 |
| `inference_inputs/operators.py` | 공통 선택, 슬라이스, stack/concat, 축·dtype 연산 |
| `inference_inputs/memory.py` | 상태 제안·확정·폐기, generation 초기화 |
| `inference_inputs/providers.py` | 등록된 외부 데이터 제공자의 수신·초기화·종료 계약 |
| `inference_inputs/resources.py` | history·feature·cache의 공유 보관 예산 |
| `inference_context/` | 기존 수신 이력, freshness, 실행 사실, protocol 계약 |
| `policy/lerobot/configs/inference_inputs/*.yaml` | 정책별 추가 전처리와 custom handler 선택 |
| `policy/lerobot/lerobot_engine/input_config.py` | 사용자 설정 검증과 내부 그래프 생성, 순차 이미지 연산 |
| `policy/lerobot/lerobot_engine/input_pipeline.py` | LeRobot binding, Torch 연산, 저장 processor 연결 |
| `policy/lerobot/lerobot_engine/adapters/` | 모델 공개 API와 실행 방식·자산·호환성 검증 |

이전 `configs/image_preprocessing/`와 `lerobot_engine/input_plan.py`는 제거했다.
체크포인트 가중치와 `config.json`, LeRobot 원본은 변경하지 않았다.

## 사용자 YAML과 내부 그래프

```yaml
preprocessing: identity
```

추가 변환이 없다면 이 한 줄이면 된다. 이미지 변환은 `preprocessing.images` 목록에
순서대로 선언하고, 전용 처리는 `preprocessing.custom.handler`와 `options`로 선택한다.
자세한 예시는 [사용자 설정 가이드](../policy/lerobot/configs/inference_inputs/README.md)에 있다.
사용자 YAML에 `sources`, `nodes`, `outputs`는 허용하지 않는다.

아래 그래프, memory, execution 예시는 **Python handler 개발용 내부 표현**이다.
그대로 모델별 YAML에 붙여 넣는 설정이 아니다. Handler는 모델 adapter의
`input_handlers`에 명시적으로 등록하고, YAML에서 선택됐을 때만 실행한다.
모듈의 옵션을 검증하는 builder는 모델 로드 전에 실행된다. 복잡한 모델 API 계산은
기존 `input_extensions`에서 연산으로 등록한다. 등록 이름으로만 연결하고 임의 import는 없다.

`sources`는 출처, `nodes`는 연산 연결, `outputs`는 단계별 최종 입력이다.
모든 선언 노드는 요청당 한 번 실행된다. 출력에 직접 연결되지 않은 상태 쓰기 노드도 실행된다.
사용하지 않는 source나 연산은 Python handler의 내부 그래프에 선언하지 않는다.

- `before`: 저장된 LeRobot processor 이전.
- `after`: processor 이후. `processed`는 실제 processor 출력이다.
- `result`: 모델과 postprocessor 이후. `model_action`과 `postprocessed_action`을 구분한다.
- 단계 간 역방향 참조, 순환, 미등록 연산, 잘못된 옵션은 거부한다.
- `identity`는 그 노드에서 Cyclo가 추가 변환하지 않는다는 뜻이다.
- 회전, RGB float32 변환, BCHW 배치, device 이동은 기본 Python 그래프가 담당한다.
- 사용자 이미지 연산은 목록 순서로 실행한다. Torch/OpenCV 수치 경로와 순서를 유지한다.
  Torch 뒤 OpenCV는 암묵적 양자화가 필요하므로 거부한다.
- 모델 내부 변환과 저장 processor는 그대로 실행한다. YAML이 이를 끄거나 대체하지 않는다.

설정은 Clear/UNLOAD 후 LOAD에서만 다시 읽는다. START/RESUME 중 파일 변경은 반영하지 않는다.
모델 종류별 YAML 하나를 사용하며 체크포인트별 선택 UI는 없다.

기본값의 의미:
- ACT, Pi 계열, SmolVLA, FastWAM, VLA-JEPA 등의 spatial identity는 내부 모델 동작을 유지한다.
- Diffusion의 OpenCV bilinear/checkpoint 크기는 기존 Cyclo 호환 기본값이다. 학습 데이터의 리사이즈 방식이 확인됐다는 뜻이 아니다.
- Multi-Task DiT의 기존 224x224 Torch bilinear/antialias 테스트 설정을 유지했다. 모든 학습 데이터의 보편적 기본값은 아니다.
- XVLA의 내부 padding 설정이 꺼져 있으면 서로 다른 카메라 크기를 자동으로 해결하지 않는다.
- 기존 state padding/truncation 호환 동작은 Python 기본 연결에 유지한다. robot action layout은 바꾸지 않았다.

## 과거 관측

Cyclo는 모델 config의 `n_obs_steps`나 `observation_delta_indices`를 해석해 history를 자동 구성하지 않는다.
명시적인 `size: checkpoint` 참조, 모델 로딩과 호환성 검증용 config 읽기는 유지한다.

등록된 Python handler가 구성할 내부 history 그래프 예시:

```yaml
sources:
  arm_history:
    source: joint:follower_arm_left
    frame_offsets: [-1, 0]
    fps: 15
    max_age_s: 0.04
nodes:
  history:
    op: stack
    inputs: [arm_history]
    options: {axis: 0, sequence: true}
outputs:
  before: {arm_history: history}
  after: {"*": processed}
startup: {missing: wait}
```

위 예시는 특정 모델의 완전한 설정이나 사용자 YAML이 아니라 내부 history 조립 예시다.
필요하면 `to_tensor`, batch 축, normalization 연산을 추가한다.
`offsets_s: [-0.1, 0]`처럼 초 단위 지정도 가능하다.
frame offset은 명시한 FPS로 변환하며 Control Hz나 Dataset FPS에서 자동 유추하지 않는다.

- 실제 토픽 callback 이력을 보관한다. 모델 호출마다 같은 최신값을 읽은 것을 새 frame으로 세지 않는다.
- 필요한 topic만 구독하고 과거값을 요구한 source만 history를 보관한다.
- 시각은 같은 호스트의 monotonic 수신 시간이다. 카메라 촬영 시각 정렬이나 장치 간 clock 동기화는 아니다.
- 누락·stale·서로 다른 시간 위치에 같은 sample이 선택되는 경우를 거부한다.
- `startup.missing`은 `wait` 또는 `error`다. 기다림은 관측 준비에 한정되며 미래 실행 피드백을 Worker 잠금 안에서 기다리지 않는다.
- 실제 source ID, 수신 시각, 계산 시각과 축·단위·좌표계·정규화 의미를 분리한다.
- Diffusion/Multi-Task DiT의 공개 `select_action`은 모델 내부 history를 사용한다. 기본 YAML은 최신 observation 하나를 전달하며 중복 history를 만들지 않는다.

## 상태와 실행 조건

상태는 먼저 제안하고 선언한 사건이 확인될 때만 확정한다.
`memory.slots`의 조건과 `execution.request_after`는 독립적이다.

| 사건 | 의미 |
| --- | --- |
| `prediction_success` | 유효한 예측 완료 |
| `plan_accepted` | Runtime 실행 계획 채택 |
| `first_publication` | 해당 계획의 첫 command 발행 |
| `published_count` | 해당 계획의 서로 다른 command N개 발행 |
| `plan_terminal` | 계획의 모든 command 발행 완료. 폐기·실패는 성공 아님 |

Runtime은 다음 요청 조건을 로컬 실행 장부로 검사한다.
Worker는 요청 처리 잠금 안에서 다음 command나 미래 피드백을 기다리지 않는다.
부분 실패, 전체 정렬 폐기, 불가능한 N개 조건은 성공으로 처리하지 않는다.
ZOH 반복은 원래 계획의 command 범위 밖이므로 계획 진행 횟수에서 제외한다.

`memory_read`와 `memory_write`는 등록된 상태 슬롯에만 접근한다.
초기 메모리는 명시적인 literal 또는 `initial: input`으로 bootstrap하며 없으면 실패한다.
모델 출력 기반 기억은 `result` 단계에서 공개 raw/postprocessed 결과를 사용한다.
모델 private attribute나 임의 Python 경로는 YAML로 접근할 수 없다.

세션·generation·prediction·command·event ID로 중복과 지연을 구분한다.
Stop/Clear, 새 generation, instruction/model 변경에서 세션 메모리와 cache를 초기화한다.
Initial Pose Sync와 일반 inference의 세대를 분리한다.
모델 내부 상태를 되돌릴 수 없는 실패는 같은 generation에서 재시도하지 않는다.

## 확장과 성능

복잡한 feature 계산은 Python 등록 모듈로 작성한다.
LeRobot adapter의 `input_extensions(registry, bindings, engine)`에서 연산을 등록한다.
`input_handlers`의 builder가 옵션을 내부 연산 구성으로 변환하며 YAML이 모듈을 import하지 않는다.

- operator compiler는 LOAD에서 실행되고 요청마다 호출할 작은 함수를 반환한다.
- 기본적으로 입력을 변경하지 않는다. 변경이 필요한 등록 연산은 `mutates_inputs=True`로 선언하여 전용 복사본을 받는다.
- 외부 source는 `Binding(..., provider=...)`과 `start(queries, budget)`, `resolve_samples`, `reset`, `close`를 구현한다.
- 제공자는 실제 sample ID/원본 수신 시각을 반환하고 자체 보관 버퍼도 전달받은 Budget에 등록해야 한다.
- 등록 모듈은 신뢰된 코드다. 임의로 잘못 작성된 Python 모듈을 sandbox로 통제하는 구조는 아니다.
- GPU 계산은 Torch 등록 모듈에서 실행하며 공통 핵심은 Torch에 의존하지 않는다.

요청 간 cache는 `cacheable=True`로 검토된 연산에만 허용한다.
Python handler의 내부 그래프에서 입력과 instruction 의존성을 모두 명시하고 실제 sample ID가 있어야 한다.
계산 설정과 encoder는 LOAD 수명에 고정되며 generation 변경 시 cache를 비운다.
모델 결과는 확률적일 수 있으므로 `result` 단계의 요청 간 cache는 금지한다.
메모리 읽기의 ID는 확정된 상태 버전을 포함해 같은 image에서도 변경을 구분한다.

기본 보관 한도는 CPU/GPU 합산 256 MiB이며 history·미확정/확정 feature·cache가 공유한다.
이는 보관된 수치 버퍼 예산이지 전체 프로세스 RSS 제한이 아니다.
모델 weights, Python 메타데이터, 임시 복사, encoder activation과 allocator 여유분은 별도다.
callback과 제어 잠금 안에서 encoder 계산을 실행하지 않는다.

## Protocol과 제한

Engine protocol은 3.1이다. 확장 상태 계약은 `feedback_schema=2`로 협상한다.
기존 schema 1의 JSON 필드·형식과 ROS 서비스 필드는 유지한다.
schema 2는 보간 전 prediction ID와 계획 command 범위를 구별한다.
호환되지 않는 Worker/Runtime 조합은 LOAD 계약에서 거부한다.

이 기반만으로 LingBot-VA, RTC/TT-RTC 등 모든 모델을 지원한다고 볼 수 없다.
각 모델의 공개 API, action 좌표계, memory 확정 시점과 bootstrap을 검토한 adapter가 여전히 필요하다.
대화형 추가 관측, 온라인 학습, 후보별 분기 메모리, 물리적 실행 완료 ACK,
원격 clock 동기화와 임의 대형 모델 feature API는 이번 구현의 지원 범위 밖이다.
지원되지 않는 옵션/연산/실행 계약은 LOAD에서 거부하며 모델 내부 요구를 자동 발견하지는 않는다.

## 간단한 YAML 전환 검증 (2026-09-17)

- LeRobot adapter/입력 테스트 171개, 이미지 연산 40개 통과.
- 공통 Runtime/Catalog/Dockerfile 테스트 410개 통과.
- 13개 기본 설정을 이전 경로와 rotation 0/270에서 수치 오차 없이 비교했다.
- YAML handler 선택, 옵션 거부, 모델별 등록 범위와 작은 실제 encoder의 메모리 결합을 검증했다.
- 3-camera CPU 입력 조립 p95: 기존 2.44 ms, 새 경로 2.52 ms. 기준 2.88 ms 이내.
- 두 경로 모두 Torch copy 9회, positive self allocation 합 10,948,620 bytes.
- 이번 전환에서 실제 모델 checkpoint/GPU/로봇과 ARM64 검증은 재실행하지 않았다.
- 아래 이전 검증 기록은 이번 전환 후 실제 모델 검증을 대신하지 않는다.
- 이번 사용자 설정 변경만 적용하려면 LeRobot Worker 코드 업데이트가 필요하다.
  기존 Worker에 새 YAML만 제공하면 LOAD가 실패한다. 기존 공통 pipeline 전환까지
  아직 배포하지 않았다면 아래 설명대로 Cyclo도 함께 업데이트해야 한다.

## 이전 파이프라인 검증 기록

- 기본 13개 profile의 이미지 수치·shape·회전 호환 검사.
- 작은 실제 Torch encoder의 과거/현재 feature 결합과 실행 피드백·초기화 검사. 모델 본체와 transport는 mock.
- 실제 pinned LeRobot Diffusion 공개 API: 관측 1/2/4개, action queue 반복과 reset 등 11개 테스트 통과.
- 실제 ACT 40k 체크포인트: 격리 CPU 환경에서 추론 6회, cached LOAD, UNLOAD 통과. 로봇 command 발행 없음.
- 최신 입력 조립 p95: 기존 2.52 ms, 새 그래프 2.59 ms. 기준 `기존*1.1+0.2ms` 이내.
- 같은 입력에서 Torch `copy_` 호출은 각각 9회, positive self allocation 합은 각각 10,948,620 bytes였다. allocation 합은 최대 상주 메모리가 아니다.
- tracemalloc peak는 기존 758,569 bytes, 새 그래프 761,897 bytes였다. Torch tensor storage는 이 수치에서 제외된다.
- 위 p95는 CPU 3-camera 입력 조립 벤치마크다. 전체 GPU 추론이나 실제 100 Hz 제어 jitter 측정이 아니다.
- UI 190개, Docker/Supervisor 77개, 격리된 ROS 환경 Initial Pose Sync 47개 테스트 통과.
- 공통 Runtime 341개, LeRobot 146개, 이미지 연산 40개, action 처리 47개, RobotClient 단위 50개, ROS CDR 4개, Catalog 36개 테스트 통과.
- 실제 pinned WALL-X/LeRobot GR00T import·저장 processor 6개 통과. 같은 파일의 ACT 테스트 1개는 경로 환경변수 미지정으로 skip됐으며, ACT는 별도 실제 체크포인트 smoke로 검증했다.
- ARM64 로봇 hostname `ffw-snpr48a1110.local`이 해석되지 않아 ARM 검증은 미완료.
- GPU peak/임시 복사 최대치, 실제 제어 jitter, 실제 로봇·모델 성능 검증은 미완료다.

실행 중인 서비스는 재시작하지 않았다. 버전 변경·commit·push·이미지 배포는 진행하지 않았다.
AMD64 검증은 기존 이미지에 새 소스를 read-only mount한 격리 실행이다. 후보 이미지 자체의 전체 빌드·기동 검증은 별도로 남아 있다.
적용 시 Cyclo와 LeRobot 이미지를 함께 재빌드·재생성해야 한다. 두 LeRobot Dockerfile에 새 공통 패키지 COPY를 추가했고 Compose의 YAML mount 경로도 변경했다.
