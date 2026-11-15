# RL Canvas 통합

`feature/rl_playground_simplify`의 데이터 취득·Replay Buffer·학습 기능을 Autonomy Studio의 **RL Canvas**에 통합한다. 기존 Action Canvas, Mission Canvas, Autonomy Canvas와 일반 Sync/Async 및 VLA-RAIL은 유지한다.

## 사용 흐름

1. Autonomy Studio에서 **RL Canvas**를 선택한다.
2. **Environment**에서 로봇·정책·체크포인트를 선택하고 추론 세션을 준비한다. 추론 중 RL Recording으로 데이터를 취득하고 성공/실패 결과를 저장한다.
3. **Replay Buffer**에서 원본 녹화와 변환 데이터를 확인하고 학습용 데이터를 선택한다.
4. **Training**에서 정책에 맞는 ACT TD3, MultiTaskDiT Flow-SDE PPO 또는 GR00T RLT 작업을 실행한다. 진행 상태·결과 및 배포 기능은 해당 학습 방식의 계약을 따른다.

모델과 데이터는 기존 `/workspace` 아래 저장소를 사용한다. 이번 코드 이식은 사용자 데이터·체크포인트를 복사하거나 삭제하지 않는다.

## Action Canvas의 TT-RTC

SendCommand의 **LOAD → action_request_mode → tt_rtc**를 선택한다. 모델은 TT-RTC 계약을 명시하는 GR00T 체크포인트여야 한다. `control_hz`는 명령 출력 주파수이고 모델 액션의 기준 주파수는 15 Hz다.

- 이번 요청의 지원 프로파일: **32×16**, prefix **0~8**.
- TT-RTC는 체크포인트의 논리 horizon을 사용하므로 `action_chunk_size=0`을 전달한다. 일반 추론의 청크 길이 선택은 유지한다.
- 기존 XML의 `ttrtc_enabled=true`는 새 `action_request_mode=tt_rtc`로 읽는다. 이전 고정 `ttrtc_prefix_steps` 옵션은 새 런타임에 전달하지 않는다.
- prefix 정책은 Playground의 `CYCLO_TT_RTC_PREFIX_MODE`를 따른다. 기본은 `fixed_first_6`이고 `fixed_first_8`, `latency_adaptive`도 지원한다. 체크포인트가 선언한 최소·최대 prefix 범위 내에서만 실행한다.
- 코드가 0~8을 지원하더라도 0~6으로 선언된 체크포인트를 0~8 학습 모델로 취급하지 않는다. 실제 가중치의 학습 범위와 config가 일치해야 한다.
- TT-RTC와 VLA-RAIL은 별도 실행 방식이다. 기존 일반 Async + VLA-RAIL 경로는 유지한다.

이전 `ttrtc_execution.py` 실행기는 제거하고 Playground의 source-clock timeline 및 prefix-conditioned 추론 경로로 교체했다. 자세한 모델 계약과 실행 과정은 [TT_RTC_KO.md](../cyclo_brain/policy/groot/docs/TT_RTC_KO.md)를 참고한다.

## 통신 및 배포

UI/ROS/정책 컨테이너가 같은 인터페이스 버전을 사용해야 한다. UI만 새로고침해서 적용되는 변경이 아니다.

- SendCommand: `LOAD_INFERENCE=25` 유지, `SET_ACTION_POLICY=26`.
- InferenceCommand: `CAPABILITIES=7`, `FAULT_STOP=8` 유지. RLT 명령은 20~28에 분리한다.
- `interfaces`와 관련 ROS 패키지 재빌드, UI 빌드, 변경된 Compose 마운트 반영이 필요하다.
- GR00T Dockerfile에 Playground의 선택적 arm smoother용 `ruckig==0.12.2`를 추가했다. LeRobot의 MultiTaskDiT 학습 의존성도 포함한다.
- Compose는 `cyclo_brain/algorithm`, `model`, `contracts`, `shared`를 정책 컨테이너에 읽기 전용으로 마운트한다.
- 중지 시 기존 feedback-confirmed hold와 fault latch를 유지한다. 모델 LOAD 자체는 로봇 동작을 시작하지 않는다.

## 외부 하드웨어 의존성

사람 개입 UI는 AI Worker Intervention Controller를 사용한다. 기본 주소는 `http://127.0.0.1:8779/api/intervention`이고 `AI_WORKER_INTERVENTION_URL`, `INTERVENTION_CONTROL_TOKEN`으로 설정한다. 컨트롤러가 없으면 UI에 서비스 연결 오류를 반환한다.

외부 명령 arbiter는 `CYCLO_COMMAND_ARBITER_ENABLED=1`일 때만 사용한다. 활성화하려면 AI Worker의 arbiter와 로봇 명령 토픽 라우팅을 함께 구성해야 한다. 기본값은 기존 직접 명령 경로를 유지한다. 이번 이식은 AI Worker의 브링업이나 로봇 토픽 설정을 변경하지 않는다.

## 통합 검증 (2026-11-14)

- UI 관련 회귀 테스트 및 production build/lint 통과. 기존 Autonomy Canvas 테스트 4건은 변경 전 브랜치에서도 같은 fixture 오류로 실패하며 이번 이식과 구분했다.
- API 418건, 데이터 처리 159건, ROS/BT 통합 150건, 공통 런타임 536건, SDK/LeRobot 93건, TT-RTC 계약 58건, RL 알고리즘 330건(7건 skip) 검증. 실제 배포 중 발견한 필수 응답 필드 누락은 추가 회귀 테스트로 수정했고 서비스 handler 52건을 재검증했다. 이 수치는 겹치는 테스트를 포함하므로 합산하지 않는다.
- 실제 저장소의 interfaces/shared/cyclo_data/orchestrator 빌드 완료. UI·API·ROS/BT 서비스 및 GR00T 재시작 반영 완료. RL API와 nginx 프록시, 실제 GR00T CAPABILITIES 응답 확인.
- 현 GR00T 컨테이너에는 ruckig 0.12.2 설치 완료. 새 이미지 재빌드 시에도 Dockerfile의 고정 버전을 사용한다.
- 브라우저 연결을 사용할 수 없어 실제 화면의 시각 검수는 미실행. 실제 모델 학습·가중치 로드·로봇 동작 검증도 별도 테스트가 필요하다.
- 기존 `RobotisSW/mock_test`는 `trained:false`, `mock-inference-only`의 구형 manifest이므로 새 TT-RTC LOAD 계약을 만족하지 않는다. 32×16 / prefix 0~8 지원 여부와 실제 체크포인트의 학습 완료 여부는 별개다.
