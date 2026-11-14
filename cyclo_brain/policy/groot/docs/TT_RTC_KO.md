# TT-RTC 모델을 Action Canvas에서 사용하기

`mission_reasoning/ttrtc_training`에서 학습한 GR00T-N1.7 체크포인트용 실행 경로입니다. 일반 GR00T 모델과 VLA-RAIL 실행은 기존 방식으로 사용할 수 있습니다.

## LOAD 노드 설정

Action Canvas의 `SendCommand`에서 다음을 선택합니다.

- Target: `INFERENCE`, Command: `LOAD`, Model: `groot:n17`
- Policy path: 학습이 저장한 `checkpoint-60000`, `checkpoint-80000` 또는 `checkpoint-100000` 폴더
- Action request mode: `async`
- `ttrtc_enabled`: 켬
- `ttrtc_prefix_steps`: 기본 `7`, 허용 범위 `0..8`

TT-RTC를 켜면 VLA-RAIL은 꺼지고, action 실행 주기는 15 Hz, acceleration은 PyTorch로 설정됩니다. `inference_hz=15`는 action 실행 주기입니다. 모델을 초당 15번 호출한다는 뜻은 아닙니다. 학습 체크포인트에는 processor, 정규화 통계, experiment manifest가 함께 있어야 합니다. 학습 중인 실행 폴더 자체나 `.safetensors` 파일 하나만 지정하지 마세요.

LOAD는 모델을 준비하며 로봇 동작을 시작하지 않습니다. 기존 Canvas의 simulation/robot 선택 및 RESUME/STOP 흐름을 사용합니다. ROS 메시지 정의도 변경되었으므로 구형 인터페이스나 컨테이너와 섞어 사용하면 안 됩니다.

## Gazebo용 목업 체크포인트

테스트 모델 경로는 `/workspace/model/groot/ttrtc_mock_n1.7`입니다. 호스트에서는 `docker/workspace/model/groot/ttrtc_mock_n1.7`에 해당합니다. 위 LOAD 설정을 그대로 사용하며, Gazebo에 관절 명령을 보내려면 Canvas에서 **robot 모드**를 선택합니다. 목업에 simulation 전용 제한은 없으며 기존 LOAD → START/RESUME → STOP 명령 경로를 사용합니다.

이 모델은 원본 GR00T-N1.7 전체 가중치에 `refilling_visual_llm15_kl_nopad_1000_20261110-vlm.safetensors`를 병합하고, 현재 15 Hz TT-RTC 학습의 config와 processor·정규화 통계를 복사한 것입니다. TT-RTC 학습 step은 **0**입니다. 로딩·추론·prefix 유지·postfix 실행 연결을 점검하는 용도이며, refilling 작업 성능을 검증하는 학습 모델은 아닙니다.

config와 experiment manifest 모두 `mock_checkpoint: true`를 기록하고 manifest에는 `trained: false`, `step: 0`, `complete: false`를 명시합니다. 로더는 두 mock 표시와 기존 TT-RTC action 계약을 검증하고 목업 로딩 로그를 남깁니다. mock 표시가 없는 미학습 체크포인트는 계속 거절합니다. 재생성은 `scripts/create_ttrtc_mock_checkpoint.py --help`를 참고하세요.

생성된 목업은 약 6.44 GiB이며, CPU에서 실제 전체 모델(32층 DiT, 16층 VLM)을 로드해 prefix 0·7·8 추론을 확인했습니다. 세 경우 모두 `[1,32,16]` 유한값 출력과 prefix 보존 검사를 통과했습니다. Engine의 checkpoint·processor 검사도 통과했습니다. 결과는 모델 폴더의 `cpu_mock_validation.json`, 재현 코드·명령·로그는 `validation/`에 있습니다. 관련 GR00T 테스트 53개와 TT-RTC runtime 테스트 49개도 통과했습니다. Gazebo 서비스에 연결한 실시간 구동 및 GPU 추론 시간은 별도로 확인해야 합니다.

## Prefix와 postfix

학습 모델은 logical action 32개를 반환합니다. 최대 prefix는 8개이며, runtime은 반환값 중 prefix 다음의 **새 action 24개만** 실행 큐에 추가합니다. 첫 추론은 prefix가 없는 `d=0`입니다. 이후에는 현재 큐의 남은 action이 설정한 prefix 수 이하가 되면 그 action들을 고정할 prefix로 전달합니다. prefix를 두 번 실행하지 않습니다.

prefix 7개는 15 Hz에서 약 0.467초, 8개는 약 0.533초입니다. 다음 결과가 그 시간 안에 준비되어야 실행 큐를 끊김 없이 이어갈 수 있습니다. 결과가 늦거나 이전 세션의 응답이면 버리고, 기존 큐를 소진한 후 마지막 목표를 유지하면서 prefix 0으로 다시 요청합니다. 실제로 끊김 없이 동작하는지는 모델 연산뿐 아니라 관측 준비 및 통신 지연을 포함해 측정해야 합니다.

학습에 사용한 min-max 통계로 physical prefix를 정규화하고, prefix token의 time을 1로 둡니다. 4회 flow-matching Euler update마다 prefix를 다시 고정합니다. 학습된 32-action horizon과 16차원 action 구조를 검증하며, TT-RTC 전용 체크포인트를 일반 실행으로 로드하거나 일반 체크포인트에 TT-RTC를 켜면 실패합니다.

## 적용 및 검증 범위

현재 소스 변경에는 UI, orchestrator, ROSIDL 메시지, SDK 스키마, Main runtime, Engine 및 GR00T model override가 포함됩니다. 배포 환경에서는 변경된 UI와 `interfaces`를 빌드하고 orchestrator/GR00T 서비스를 같은 버전으로 재시작해야 합니다. 배포·모델 로딩·실제 로봇 구동은 코드 작성과 별도로 수행합니다.

실행 로그의 `ttrtc_chunk_installed`는 새 action 24개 설치, `ttrtc_chunk_rejected`는 deadline/shape/오래된 요청 등의 거절 원인을 기록합니다. Gazebo에서는 robot 명령 발행 경로로 첫 prefix 0, 후속 prefix, 늦은 응답, STOP/RESUME 동작을 확인할 수 있습니다.

코드 검증 결과: orchestrator 83개, 공통 runtime 413개, GR00T 50개, UI 39개 테스트가 통과했고 UI production build가 완료되었습니다. GR00T 검증은 CPU에서 실제 작은 DiT/ActionHead 및 배포용 StateActionProcessor를 사용했습니다. 실제 학습 체크포인트의 GPU 추론 시간과 로봇 동작은 아직 검증하지 않았습니다.
