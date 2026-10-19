# ReturnToWaypoint 사용법

현재 미션의 특정 waypoint로 Navigation 이동하는 BT 액션입니다. 도착하면
다음 BT 노드로 이어집니다. 목적지의 local BT를 다시 실행하거나 미션 경로의
실행 순서를 바꾸지 않습니다. 같은 waypoint로 복귀해도 재귀 실행되지 않습니다.

## Mission Run / Record / Operate

local BT에서 `ReturnToWaypoint`를 추가하고 `waypoint`에 목적지 이름 또는 ID를
입력합니다. 예: `DockingStart`. 이름이 중복되면 ID로 지정해야 합니다.
`map_name`, `mission_name`은 비워 둡니다. 실행 시 불러온 미션의 전체 waypoint
좌표를 전달받으며, 목적지는 그 스냅샷에서 결정됩니다. 좌표를 노드에 복사할
필요가 없습니다. 경로에 포함되지 않은 미션 waypoint도 목적지로 선택할 수 있습니다.

```xml
<Sequence name="DockAndReturn">
  <ArucoDock marker_id="11" marker_size="0.30" target_distance="0.85"/>
  <ReturnToWaypoint waypoint="DockingStart" timeout="180"/>
</Sequence>
```

`house / ArUcoDockTest`에 추가된 `InitPosition`도 복귀 대상으로 지정할 수 있습니다.
예를 들어 도킹 노드 다음에 `<ReturnToWaypoint waypoint="InitPosition"/>`을
추가하면 그 위치로 돌아옵니다. 원하는 local BT 위치에 직접 배치합니다.

## Action Canvas 단독 실행

미션 실행 컨텍스트가 없으므로 `map_name`, `mission_name`을 함께 지정합니다.
해당 맵의 Navigation과 localization이 준비되어 있어야 합니다.

```xml
<ReturnToWaypoint waypoint="DockingStart" map_name="house"
                  mission_name="ArUcoDockTest" timeout="180"/>
```

노드가 시작할 때 공식 미션 저장 API에서 목적지를 읽어 고정합니다.
존재하지 않는 미션, 잘못된 waypoint, 중복된 이름은 이동 전에 실패합니다.
저장소 조회는 ROS tick을 막지 않는 읽기 전용 작업으로 실행됩니다.

## 동작

- Nav2 `NavigateToPose`로 저장된 위치·방향에 이동합니다.
- 성공은 Nav2 성공 응답 및 새로운 odometry의 정지 상태로 확인합니다.
- Pause는 자기 목표만 취소하고 정지를 확인합니다. Resume은 같은 목적지로 재전송합니다.
- Stop은 늦게 수락된 자기 목표까지 취소한 후 정리합니다.
- 다른 Navigation 목표가 실행 중이면 새 목표를 보내지 않습니다.
- 이동 실패·거부·시간 초과는 BT FAILURE로 전달합니다.
- `timeout`: 기본 180초, 범위 5–300초. 일시정지 시간은 제외합니다.
- 목적지 도착 허용 오차는 현재 Nav2 설정을 사용합니다. ArUco 정밀 도킹의
  3.5cm 오차 설정과 별개입니다.

## 코드

- `bt/actions/return_to_waypoint.py`: BT 파라미터·읽기 전용 목표 조회·수명주기
- `bt/waypoint_targets.py`: waypoint 이름/ID 검증 및 저장된 미션 조회
- `bt/waypoint_navigation.py`: Nav2 목표와 Pause/Resume/Stop 제어
- `utils/missionWaypointContext.js`: Run/Record local BT에 waypoint 스냅샷 전달
- `supervisor_api/autonomy_config.py`: Operate의 실행 스냅샷 전달

저장된 local BT XML은 실행 시 waypoint 스냅샷을 덧붙이는 방식으로 재사용합니다.
`LoadAndRunTree` ROS 서비스 형식은 변경하지 않습니다.
