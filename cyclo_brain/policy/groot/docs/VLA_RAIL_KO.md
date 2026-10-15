# 일반 Async용 VLA-RAIL

## 사용

Action Canvas와 Mission Canvas의 SendCommand LOAD에서 다음과 같이 설정한다.

| 설정 | 값 |
|---|---|
| model | groot:n17 |
| policy_path | 기존 일반 GR00T 체크포인트 |
| action_request_mode | async |
| vla_rail_enabled | true |

별도 학습 없이 추론 출력에 적용한다. 기본값은 꺼짐이며, LOAD 이후 RESUME으로 실행한다. 서버와 UI는 일반 Async에서 RAIL을 선택하도록 검증한다. Sync에는 적용할 수 없다. 이미 로드한 모델에는 LOAD를 다시 해야 설정이 바뀐다.

## 구현

[VLA-RAIL 원문](https://arxiv.org/html/2512.24673v1)의 식 8–13을 참고하여 NumPy로 독립 구현했다.

1. 일반 비동기 추론으로 원래 액션 청크를 요청한다. 별도 prefix conditioning을 보내지 않는다.
2. 14개 팔 관절의 청크 전체를 관절별 3차 최소제곱 다항식으로 맞춘다.
3. 응답 지연을 기준으로 ±0.1초 범위를 탐색하여 이전 궤적 속도와 진행 방향이 일치하는 연결 시점을 선택한다.
4. 이전 궤적의 위치·속도·가속도에서 새 궤적으로 이중 5차 다항식으로 연결한다. 중간점은 양끝 위치·속도의 평균, 가속도 0이다.
5. 원래 시간축을 유지하며 control_hz(기본 100Hz)로 참조 궤적을 평가한다. 정책 간격은 inference_hz(기본 15Hz)를 사용한다. 출력 청크 길이를 강제 재샘플링하는 TARGET_CHUNK_SIZE는 사용할 수 없다.

연결 시간은 기본 0.2초이며 잔여 청크 길이의 1/3 이하로 줄인다. 그리퍼 두 축은 원래 청크의 이산 목표를 사용하고 다항식에 포함하지 않는다.

## 로컬 적용 차이와 한계

- 논문의 시간 가속 기능은 적용하지 않았다. 기본 비교에서는 시간 배율 1이다.
- 일반 추론에는 정확한 센서 취득 시각이 없으므로 요청 시작 시각을 정렬 기준으로 사용한다. 유효한 캐시 스냅샷 시각이 제공될 때만 그것을 사용한다. 하드웨어 취득 시각 동기화로 볼 수 없다.
- 첫 청크는 첫 원본 목표와 속도·가속도 0에서 시작한다. 현재 로봇 자세와 첫 목표의 차이를 자동으로 해결하는 기능은 아니다.
- 다음 응답이 늦으면 청크 끝에서 정지하도록 별도 이중 5차 꼬리 구간을 둔다. 끝 이후 외삽하지 않고 마지막 목표를 유지한다. 지연이 잔여 예측 범위를 넘긴 응답은 버리고 재요청한다.
- 참조 팔 궤적은 C2 연속이나, 실제 로봇 추종의 연속성이나 jerk·속도·가속도의 상한을 보장하지 않는다. 곡선 오버슈트나 작업 정확도 변화는 실기 비교가 필요하다. 별도 속도·가속도 제한기는 사용하지 않는다.
- Stop, CLEAR, 지시 변경 시 실행 세대를 갱신하여 이전 비동기 응답이 새 실행에 들어오지 않게 한다.

## 코드와 확인

- `common/runtime/main_runtime/rail_trajectory.py`: cubic, dual-quintic, 시간 정렬, 종단 정지.
- `common/runtime/main_runtime/rail_execution.py`: 단일 비동기 요청, 청크 교체, 세대 관리.
- `common/runtime/main_runtime/control_loop.py`: opt-in 실행 경로와 publish 연결.
- `common/runtime/tests/test_rail.py`: 수치 경계, C2 연결, 그리퍼, 만료·실패·초기화 검증.

추적 이벤트 `rail_chunk_installed`에 정렬 offset, 연결 시간, 잔여 horizon, timing을 기록한다. `rail_chunk_rejected`는 만료 응답이며 `request_completed`에는 실패 사유를 기록한다. `VLA_RAIL_CAPTURE_ACTIONS=1`을 서버 시작 환경에 지정하면 발행 액션도 기록한다.

실기에서 일반 Async와 RAIL Async를 동일 체크포인트·초기 자세·정책 주기로 비교해야 한다. 합성 테스트 통과가 실기 떨림 감소를 의미하지는 않는다.


## 현재 구성 (2026-10-08 정리)

유지하는 평활화 경로는 **일반 Async → cubic fitting → 시간 정렬 → dual-quintic 연결 → 발행**이다. 학습 기반 청크 조건화, 잔차 RL, spline 전처리, 별도 속도·가속도 제한 기능은 제거했다. 일반 GR00T Sync/Async와 기존 비전 튜닝 모델 로딩은 유지한다.

LOAD에서는 `action_request_mode=async`, `vla_rail_enabled=true`만 선택하면 된다. 기존 일반 GR00T 체크포인트를 사용한다. 예전 옵션이 켜진 XML은 그대로 실행하지 말고 Canvas에서 LOAD 노드를 다시 저장하여 폐기된 필드를 제거한다.

일반 추론 로그 경로는 `INFERENCE_TRACE_PATH`, RAIL 발행 명령 기록은 `VLA_RAIL_CAPTURE_ACTIONS=1`로 설정한다. 기존 학습 데이터와 체크포인트는 코드 정리 시 삭제하지 않았다.
