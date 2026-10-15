# Qwen 미션 선택·실행 감독 데이터 규격

## 작업 흐름

Autonomy Canvas의 **Record → Annotate → Train → Operate** 네 탭을 사용한다. Record에서 이미지를 취득하고 Annotate에서 필요한 마스크와 미션 라벨을 작성한다. Train에서 승인 라벨로 학습한 뒤 Record의 Observation으로 판단을 확인하고 Operate에서 모델을 선택한다. 학습 완료가 실행 중 모델을 자동 교체하지는 않는다.

Annotate는 **Reasoning labels / Object masks** 도구와 **Manual / Auto Labeling** 작업 모드를 제공한다. 샘플 추출·수동 정답·자동 생성·검수는 여기서 수행한다. 학습 폼은 Train으로 모으며 **Training purpose**에서 **Auto-labeler**(Qwen 또는 SAM)와 **Reasoning model**을 구분한다. 탭 분리는 기존 데이터 형식·저장 경로·학습 API를 변경하지 않는다.

- **미션 선택:** 관찰 지점에 도착해 관찰 중 `start / wait / next_observation / unknown` 판단.
- **미션 감독:** 하위 미션 수행·일시중지 중 `continue / pause / cancel` 판단.
- **이동 감독:** Patrol 이동·도착 자세 설정·미션 종료 후 복귀 중 `continue / pause` 판단. 이때는 미션을 선택하거나 취소하지 않는다.
- **마스크:** 선택 사항이다. 미션 라벨과 독립적으로 저장하며, 저장만으로 Qwen attention에 자동 반영되지 않는다.

수기로 작성할 정답은 Expected decision, Mission, 영어 Reason이다. 실행 문맥이 확인된 감독 프레임에서는 Mission을 촬영 당시 실행 중이던 미션으로 고정한다. 모드와 실행 단계는 취득 문맥이며 정답을 보고 역으로 추정하지 않는다.

이 변경은 수집·라벨링·학습 입력·추론 계약을 연결한다. **기존 모델을 재학습하거나 기존 데이터의 라벨을 일괄 변경하지 않는다.** 감독 판단을 학습하려면 별도의 continue/pause/cancel 정답을 수집·승인해야 한다.

## 세 유형의 수집과 학습

**Record**의 지도 툴바는 Localize·이전 지점·순회 시작/일시정지/재개·다음 지점·Stop의 주행 버튼 다섯 개와, 구분선 뒤의 Create waypoint·Edit route·Undo·Redo로 구성한다. **Create waypoint → On Map / At Robot**으로 Observation waypoint 초안을 만들고 Save한다. **Edit route**에서 지도 지점을 선택하면 이름·체류 시간·경로 포함 여부·연결할 하위 Mission을 설정하는 팝업이 열린다. 지점 본체와 방향 핸들을 드래그해 위치·방향을 바꿀 수 있으며, 변경은 저장 전 초안으로 유지한다. 툴바 아래의 **Save / Set observation route order** 아이콘 두 개로 경로를 편집한다. 순서 지정 아이콘을 켜면 초안 경로를 새로 그리며, 지도 지점을 클릭한 순서대로 번호와 연결선을 표시한다. 두 지점 이상 선택한 뒤 첫 지점을 다시 클릭하면 Closed loop로 마치고, 순서 지정 아이콘을 끄면 One-way로 마친다. 순서 지정 중에는 지점 설정·드래그를 잠그며, 기존 지점을 다시 포함해도 체류 시간·연결 미션은 유지한다. Undo/Redo는 초안에 적용하고 **Save**는 지점과 경로 변경을 함께 저장한다. 저장 전까지 기존 경로는 유지되며, **Edit route**를 다시 눌러 변경 폐기를 확인하면 저장된 상태를 복원한다. 별도의 Cancel 버튼은 없다. 이 설정과 Operate의 **Observation plan**은 같은 맵의 `autonomy.json`을 사용한다. Operate에서도 계획과 체류 시간 등을 수정할 수 있으며 별도의 수집용 경로 사본은 만들지 않는다.

**주행과 녹화는 독립적이다.** 지도 툴바의 **Start Patrol**은 저장된 Observation 경로만 주행한다. 편집 중이 아닐 때 지도 지점을 선택한 뒤 **Go to selected observation**으로 해당 지점에 이동할 수 있다. Record는 Qwen 판단으로 하위 미션을 자동 실행하지 않는다. **Stop**은 현재 주행·BT를 중지하고 맵과 로컬라이제이션을 유지한다. 다른 맵으로 바꾸려면 별도의 **Unload map**을 사용한다.

**Start patrol** 버튼은 순회 중 **Pause patrol**, 일시정지 중 **Resume patrol**로 바뀐다. Pause는 현재 목적지와 남은 체류 시간을 보관하며 캡처를 유지한다. 지점 사이에서 멈추면 Patrol 감독 데이터를, 지점 도착·정렬 후 머무르면 Observation 데이터를 취득한다. **Stop**은 현재 실행과 미션 후 자동 순회 재개를 취소한다.

**Previous / Next observation** 아이콘은 체류 중에는 현재 지점, 이동 중에는 현재 목적지를 기준으로 이전·다음 지점으로 이동한다. 진행 중인 주행의 취소를 확인한 뒤 새 목표를 보내며 캡처는 유지된다. 미션·복귀 중 또는 지점이 하나뿐인 경로에서는 비활성화된다. 단방향 경로의 범위를 벗어나는 버튼은 비활성화되고, Closed loop에서는 처음과 끝을 연결한다.

One-way 경로는 마지막 지점에서 멈추고 Closed loop는 마지막 지점에서 처음으로 연결해 반복한다. Operate의 One-way는 마지막 지점에서도 관찰과 미션 선택을 유지하며 사용자 Stop 전까지 자율 세션을 종료하지 않는다.

지도 우측 **Capture** 패널은 지점 선택과 지도 편집 중에도 계속 표시되며 취득 세션과 컨트롤을 유지한다. 패널의 설정 아이콘에서 단일/연속 촬영과 간격을 정하고 **Start episode → Capture**로 수집을 켠다. Patrol은 수집 전후 어느 때나 시작할 수 있다. **Recording type**은 실제 실행 상태에 따라 Observation / Patrol / Mission으로 바뀌는 표시이며 직접 선택하지 않는다. 실행 전에 수집을 준비해도 실제 실행 문맥이 확인될 때까지 프레임 저장은 대기한다. **Capture/Pause**의 일시정지는 촬영만 멈추고, **Save episode**는 수집 세션의 현재 구간을 닫는다. 둘 다 주행·미션을 중단하지 않는다. 기존 에피소드를 열어도 로봇을 움직이지 않는다. Qwen 미리보기는 수집의 필수 조건이 아니다.

| 유형 | 실제 촬영 상태 | 라벨링 |
|---|---|---|
| Observation | 관찰 지점 도착·위치 정렬 후 정지 관찰 | start / wait / next_observation / unknown. start만 해당 지점의 후보 Mission 선택 |
| Patrol | 초기 접근, 지점 간 이동, 도착 자세 설정, 미션 복귀 또는 이동 일시중지 | continue / pause, Mission 선택 없음 |
| Mission | 하위 미션의 주행·BT 실행 또는 일시중지 | continue / pause / cancel. 촬영 당시 실행 미션 자동 연결 |

저장하는 프레임의 Capture type은 실제 실행 문맥과 일치해야 한다. 실행 전 Episode 준비에는 저장된 계획을 사용하며, 프레임에는 준비 상태를 실행 상태로 기록하지 않는다. 경로 중간에서 Stop했다고 Observation으로 간주하지 않으며, 선택만 해 둔 미션을 실행 중인 미션으로 기록하지 않는다. Patrol은 목적지가 하나뿐인 이동과 첫 지점까지의 접근도 취득한다. 초기 접근은 `phase: approaching`, `from_spot_id: null`로 남긴다.

하위 미션은 실제로 도착한 Observation에서 연결된 Mission을 선택해 명시적으로 실행한다. 순회 중 체류하거나 Pause한 상태에서 **Run child mission**을 누르면 순회 문맥을 보관한다. 실행기는 **작업 미션 → 출발 Observation 복귀 → 도착 자세 안정화 → 미션 시작 직전 남은 체류 시간 대기 → 다음 Observation으로 순회 재개**를 녹화 여부와 무관하게 수행한다. 복귀 완료 후 같은 지점으로 다시 이동하거나 도착 BT를 중복 실행하지 않는다. 한 지점 Closed loop는 제자리 관찰, One-way의 마지막 지점은 남은 체류 시간이 끝난 뒤 관찰로 남는다. 미션 실행 및 복귀 시간은 남은 체류 시간에서 차감하지 않는다. Patrol 없이 직접 실행한 미션은 복귀 후 새 순회를 임의로 시작하지 않는다.

미션 시작 때 확인한 출발 지점의 위치·방향·도착 자세 BT를 복귀까지 고정한다. 실패·취소·Stop·맵 변경 시 자동 순회 재개를 취소하며, 순회 재개 전 저장된 경로가 바뀌었으면 새 주행을 시작하지 않는다. 미션을 선택하지 않으면 설정된 체류 시간 뒤 다음 지점으로 이동한다. 실행하지 않았다는 이유로 정답 `wait`를 자동 생성하지 않으며 라벨은 이후 Annotate에서 지정한다.

녹화가 켜진 상태에서 `Patrol → Observation → Mission → 복귀 Patrol → 다음 Patrol / Observation`처럼 실제 유형·대상이 바뀌면 기존 에피소드를 닫고 다음 구간으로 연결한다. 한 에피소드 안에서 유형을 바꾸지 않는다. 녹화가 꺼져 있으면 실행 변화만으로 새 데이터나 에피소드를 만들지 않는다. **Capture/Pause**의 일시정지는 촬영만 멈추며, 주행이 끝나도 열린 녹화는 사용자가 저장하거나 촬영을 중지할 때까지 유지한다.

- 일반 전환은 `POST /collection/episodes/{id}/segment`를 사용한다. 이전 에피소드의 `next_episode_id`, 다음 에피소드의 `previous_episode_id`로 연결하며 `group_id`, 로봇·환경 정보를 공유한다. 같은 요청의 재시도는 같은 구간을 반환한다. 검증 실패 시 원래 에피소드는 열어 둔다.
- 촬영하던 Mission의 복귀는 `/return`으로 생성한다. Mission의 `return_observation_spot_id`, `return_observation_spot` 스냅샷을 유지하고 `return_episode_id` 및 일반 전후 링크를 기록한다.
- 연결된 복귀 Patrol에는 `capture_segment: mission_return`, `source_episode_id`, `return_mission_id/name`, 단일 목적지 `patrol_spot_ids`를 기록한다. 복귀 프레임은 `patrol_supervision`, `active_mission: null`, `phase: returning/settling`이다.
- 미션을 촬영하지 않았다가 복귀 도중 처음 Capture해도 일반 Patrol 에피소드로 수집할 수 있다. 이 경우 없는 Mission 에피소드를 만들어 연결하지 않으며 프레임의 `returning` 문맥으로 구분한다.
- 저장·전환 실패는 오류로 표시하고 이미 저장한 프레임은 유지한다. Save episode는 진행 중인 저장을 마친 뒤 현재 구간을 닫는다.

새 UI는 저장된 `plan_revision`을 보내며 서버가 최신 버전·지점 소속·미션 연결을 확인한 뒤 에피소드를 만든다. `observation_binding: observation_plan_v1`인 에피소드는 `recording_plan`, `candidate_mission_ids`, 카탈로그를 해당 구간 시작 당시 값으로 고정한다. Mission의 `source_observation_spot_id`와 복귀 지점은 같다. 이후 계획 수정은 기존 프레임의 후보 미션이나 정답 의미를 바꾸지 않는다. 같은 ID의 이름 변경은 화면 표시명에만 반영한다.

유형이 맞지 않거나 확인되지 않은 실행 문맥은 새 reasoning 프레임으로 저장하지 않는다. 이전 데이터는 원본을 유지하되 유효하지 않은 프레임을 reasoning 샘플링·자동 라벨링·학습에서 제외한다. 마스크 편집은 원본 이미지 기준으로 계속 가능하다.

**Train**의 Qwen 학습에서 **Training types**로 세 유형을 함께 또는 일부만 선택할 수 있다. **Auto-labeler**는 샘플 또는 검수한 승인 라벨로 자동 라벨 모델을, **Reasoning model**은 준비된 승인 라벨로 Operate 모델을 만든다. 유형마다 시스템 프롬프트·실행 문맥·허용 결정이 유지된다. 모델명·스텝·진행률은 Train에서 관리하며, 완료된 자동 라벨 모델은 **Use for auto labeling**으로 Annotate에 연결한다.

Train의 **Balance decisions per type**은 유형별 다수 결정만 두 번째로 많은 결정 수까지 줄인다. 더 적은 unknown/pause/cancel 라벨은 그대로 유지하고, 결정 종류가 하나뿐이면 경고와 함께 모두 사용한다. 원본 라벨은 바뀌지 않는다. 유형별 Available/Training 분포를 확인한 뒤 학습한다.

## 저장 구조

호스트: `docker/workspace/dataset/reasoning/` · 컨테이너: `/workspace/dataset/reasoning/`

```text
reasoning/<map_name>/
  observation/
    raw_episode_0001/  # 관찰 지점에서 미션 선택
  patrol/
    raw_episode_0002/  # 관찰 지점 간 이동·일시중지
  mission/
    raw_episode_0003/  # 미션 실행·일시중지

각 raw_episode 폴더 안:
  episode.json       # 맵, 취득 정보, catalog, 샘플 목록, 관찰 계획 스냅샷
  samples.jsonl      # 이미지별 observation + annotations/predictions/regions
  images/
  masks/
  train.jsonl        # 학습 준비 시 필요한 split만 생성
  val.jsonl
  test.jsonl
```

원본 파일은 `episode.json`과 `samples.jsonl`이다. `episode_type`은 에피소드 시작 시 observation/patrol/mission 중 하나로 고정한다. 맵 안의 세 유형 전체에서 비어 있는 가장 작은 양의 에피소드 순번을 사용하며, 내부 episode ID는 매번 새로 만든다. 별도 exports/training_inputs/_staging/.deleted 계층을 만들지 않는다. Delete episode는 해당 이미지·마스크·라벨·학습 파일을 직접 삭제한다.

Annotate 에피소드 목록에서 Observation/Patrol/Mission을 따로 선택한다. 승인 판단 라벨에는 `episode_type`과 프레임의 `reasoning_task`를 함께 기록한다. 폴더 유형은 촬영 목적이며 start/wait/unknown 또는 continue/pause/cancel 같은 선택지는 촬영된 프레임 문맥을 기준으로 한다. 자동 라벨링용 Qwen 모델은 같은 맵의 Observation/Patrol/Mission 에피소드를 함께 학습할 수 있다. 모델의 `episode_types`에 실제 학습에 포함된 유형을 기록하고, 자동 라벨링은 그 유형들에 적용한다. 기존 단일 유형 모델은 `episode_type`을 계속 지원한다. 최종 reasoning 모델 학습에서는 맵의 승인된 세 유형을 함께 사용할 수 있다.

새 에피소드는 `reasoning_episode_v4`, 새 observation은 `reasoning_raw_v2`로 저장한다. 이전 형식은 읽을 수 있으며 조회만으로 과거 판단 작업을 변환하지 않는다.

| episode 정보 | 용도 |
|---|---|
| `map_name`, `episode_type` | 맵 및 관찰/순회/미션 폴더 구분 |
| `capture_contract` | 새 Record는 `task_aligned_v1`. 유형·실제 실행 문맥 일치를 촬영/라벨링/학습에서 검사 |
| `observation_spot_id`, `patrol_spot_ids` | Observation 선택 지점 / Patrol 이동할 지점 순서 |
| `group_id` | 학습·평가 split 관리. 새 수집에서 생성하고 연결된 구간 전체가 공유 |
| `previous_episode_id`, `next_episode_id` | 같은 수집 세션의 이전·다음 유형 구간 |
| `capture_mode`, `capture_interval_ms` | 단일/연속 촬영 방식과 촬영 간격 |
| `robot_type`, `environment` | 취득 메타데이터. 미확인 값은 unknown |
| `recording_mission_id/name` | 취득에 사용한 미션. 정답 미션으로 대입하지 않음 |
| `catalog`, `catalog_snapshot_id` | 에피소드 시작 당시 미션 정의 |
| `observation_plans` | 서버에서 확인한 촬영 당시 Plan 스냅샷. 체크섬으로 참조 |

새 프레임은 자신의 `catalog`와 `catalog_snapshot_id`도 보존한다. 같은 에피소드 도중 미션 정의가 바뀌어도 다른 시점의 정의를 섞지 않기 위해서다. 서버 Plan을 취득한 경우 `context.plan_snapshot_id`가 `episode.observation_plans`를 참조한다.

## 프레임 문맥과 정답

`samples.jsonl` 한 줄은 이미지 하나다. 아래는 감독 프레임의 주요 필드만 표시한 예시다.

```json
{
  "observation": {
    "frame_id": "frame-001",
    "image": {"path": "images/frame-001.jpg", "sha256": "...", "width": 640, "height": 480},
    "catalog_snapshot_id": "...",
    "catalog": {"missions": [{"mission_id": "clean", "mission_name": "CleanTable", "revision": 1}]},
    "context": {
      "reasoning_task": "mission_supervision",
      "runtime_source": "autonomy",
      "context_version": 7,
      "active_mission": {"mission_id": "clean", "status": "paused", "step": 2, "action": "navigation"},
      "candidate_mission_ids": ["clean"],
      "observation": {"phase": "executing", "at_spot_id": null},
      "operator_instruction": null
    }
  },
  "annotations": [{
    "frame_ids": ["frame-001"],
    "reasoning_task": "mission_supervision",
    "target": {
      "decision": "pause",
      "mission_id": "clean",
      "reason": "A person is crossing the robot's route."
    },
    "annotator": "collect-ui",
    "review_status": "approved",
    "revision": 1
  }]
}
```

실제 파일에는 식별자·체크섬·촬영 시각 등 추적 정보도 저장된다. annotations는 수정 이력을 보존하고 최신 revision을 사용한다. predictions는 모델 예측이고, regions는 SAM 인스턴스 마스크다. 자동 라벨은 draft이며 사람이 승인하기 전에는 학습 정답에 포함하지 않는다.

`observation.context`의 의미는 다음과 같다.

| 필드 | 의미 |
|---|---|
| `reasoning_task` | `mission_selection`, `mission_supervision`, `patrol_supervision`. 누락된 과거 데이터는 selection |
| `active_mission` | 실제 실행 미션 ID, running/paused 상태, 단계와 동작. 선택·이동 감독에서는 null. 과거 returning 문맥은 읽기 호환 |
| `candidate_mission_ids` | 촬영 당시 후보 목록. 명시적인 빈 배열은 후보 없음 |
| `observation.phase` | approaching/moving/settling/observing/executing/returning 또는 확인 불가 시 unknown |
| `at_spot_id` | 도착한 관찰 위치. 이동 목적지와 구분 |
| `from_spot_id`, `target_spot_id` | 이동 출발지와 목적지 |
| `plan_revision`, `plan_snapshot_id` | 적용된 순회 계획 추적 |
| `runtime_source`, `context_version` | 실행 문맥의 출처와 버전 |

과거 호환용 `allowed_mission_ids`도 입력 필터로 읽는다. 미션 감독은 Patrol 후보 목록 대신 **active_mission 하나만** 대상으로 삼는다. 이동 감독은 후보 목록이 비어 있고 active_mission과 mission_id가 null이다.

## 촬영 문맥의 출처

**Autonomy 서버 이력:** `runtime_source=autonomy`이다. 카메라 프레임의 서버 수신 시각을 상태 이력의 `context_valid_from`/`context_valid_until`에 대조하고 `context_version`과 함께 저장한다. 이력에 맞지 않는 프레임은 현재 상태를 임의로 붙이지 않는다. 카메라 수신 시각은 센서 노출 시각과 동일하다고 보장하지 않는다.

**Record 실행 보고:** 브라우저가 실제 관찰 위치·Patrol 이동·미션 실행·일시중지 문맥을 `recording_execution`으로 전달한다. 선택만 한 미션을 실행 중이라고 기록하지 않는다. 저장 시 `runtime_source=record_runner_reported`, `context_confidence=client_reported`로 구분한다.

- 미션 ID·맵·running/paused 상태·단계·동작을 검증한다.
- `context_valid_from_ms`, `sampled_at_ms`, 클라이언트 버전과 촬영 수신 시각을 대조한다.
- 보고는 5초 이내여야 하며 프레임은 보고된 단계 시작 이후, 보고 시점의 최대 1.5초 뒤까지 허용한다. 클라이언트 시계 차이도 오차가 될 수 있다.
- 원본 보고와 `client_context_version`, `capture_report_delta_ms`를 보존한다. 양수 delta는 보고 이후 받은 프레임이라는 뜻이다.
- 이는 **클라이언트 보고에 기반한 근사 문맥**이며 서버가 해당 촬영 시점의 실행 상태를 확정했다는 의미가 아니다. 중요한 단계 전환 장면은 라벨링할 때 확인해야 한다.

Autonomy 서버 이력이 있으면 클라이언트 보고보다 우선한다. 검증할 수 없는 상태는 `runtime_source=unverified`, active_mission=null, 후보 목록=[]로 남기며 감독 정답을 임의 생성하지 않는다. 해당 미리보기는 새 프레임과 맞는 문맥을 기다린다. 실행이 없는 경우에만 명시적인 `preview_active_mission`을 가상 미리보기 문맥으로 사용할 수 있으며 출처를 별도로 기록한다.

## 판단 계약

| 작업 | 결정 | mission_id와 의미 |
|---|---|---|
| mission_selection | start | 후보 미션 ID 필수. 장면이 시작 조건에 부합한다는 **제안** |
| mission_selection | wait | null. 시작할 미션이 없음; Patrol 정지 명령이 아님 |
| mission_selection | unknown | null. 장면 판단 근거가 부족함 |
| mission_supervision | pause | active_mission ID와 일치. 현재 작업을 계속하기 어렵거나 근거가 부족함 |
| mission_supervision | continue | active_mission ID와 일치. 계속 실행·AI 중지 해제 제안 |
| mission_supervision | cancel | active_mission ID와 일치. 현재 미션을 포기하고 시작을 확정한 관찰 Spot으로 복귀 후 관찰·Patrol 재개 |
| patrol_supervision | pause | null. Patrol·복귀 이동 또는 도착 자세 설정을 일시중지 |
| patrol_supervision | continue | null. 이동 계속·AI 일시중지 해제 제안 |

Patrol과 복귀 이동 중에는 pause/continue만 허용한다. 다음 스팟 도착 후 새 이미지로 미션 선택을 진행한다. 미션 수행·일시중지·복귀 중에도 Qwen 추론은 계속되지만 판단 계약은 단계별로 바뀐다. continue와 cancel은 사용자 Hold/Stop이나 하위 제어기 오류를 해제하지 않는다. 미션 성공은 Mission Runner의 결과로 확인한다.

**pause는 재개 가능한 임시 중지**, **cancel은 현재 미션을 종료하고 복귀하는 판단**이다. 일시적인 사람·장애물이나 불확실성에는 pause를 쓰고, 작업 필요성이 사라지거나 더 이상 적절하지 않은 장면에 cancel을 붙인다. cancel로 다른 미션을 곧바로 실행하지 않으며 이미 끝낸 물리 동작을 되돌리지 않는다. 취소 확인 실패 시 복귀 주행을 시작하지 않는다.

예를 들어 CleanTable 수행 중 가족이 다시 식사를 시작해 청소가 불필요해졌다면 다음 정답을 사용한다.

```json
{"decision":"cancel","mission_id":"clean","reason":"The family has resumed eating, so clearing the table is no longer appropriate."}
```

이 정답은 reasoning_task=mission_supervision이고 active_mission.mission_id가 clean인 실행·일시중지 프레임에서만 유효하다. start/wait/unknown 선택 프레임과 이동·복귀 프레임에는 사용할 수 없다. **장면 판단 정답과 실행 허용 여부는 다르다.** Activity의 AI decision 이벤트는 task·phase·context_version을 포함하고 `execution_dispatched=false`로 기록된다. 실제 실행은 Controller의 별도 이벤트로 확인한다.

## 모델 입력과 저장된 미션 정의

`supervisor_api/mission_decisions.py`의 `task_for`, `decisions_for`, `system_for`, `candidates_for`, `build_input`을 실시간 판단·미리보기·자동 라벨링·학습에서 공유한다.

- 입력: 이미지 + 판단 작업/실행 문맥 + 그 문맥에서 허용되는 후보 미션.
- 출력: 정확히 `decision`, `mission_id`, `reason` 세 필드. Reason은 영어로 작성한다.
- Scenario, 그룹, 로봇 종류, 환경, pose와 관리용 시각은 프롬프트에 넣지 않는다.
- 정답 Reason은 assistant 메시지에만 들어간다. 정답 상황 설명을 user 입력에 넣지 않는다.

공식 Qwen/Transformers typed messages 형식을 유지한다.

```json
{"messages":[{"role":"system","content":[{"type":"text","text":"<system_for(context)>"}]},{"role":"user","content":[{"type":"image","image":"images/frame.jpg"},{"type":"text","text":"{\"context\":{\"reasoning_task\":\"mission_selection\",\"active_mission\":null,\"operator_instruction\":null,\"candidate_mission_ids\":[]},\"missions\":[]}"}]},{"role":"assistant","content":[{"type":"text","text":"{\"decision\":\"wait\",\"mission_id\":null,\"reason\":\"No candidate mission should start.\"}"}]}]}
```

`<system_for(context)>`는 설명용이며 실제 파일에는 해당 작업의 전체 프롬프트가 들어간다. `training/data_format.py`는 이미지 경로를 RGB PIL 이미지로 읽고 typed messages를 유지한다. AutoProcessor의 chat template을 사용하고 assistant 정답 토큰만 학습한다.

### Catalog 스냅샷 정책

- 서버 Autonomy가 활성화된 실시간 판단·미리보기는 검증된 실행 문맥의 `mission_catalog_snapshot`을 사용한다. 세션 도중 Design 문구가 바뀌어도 실행 중인 작업의 입력과 저장 프레임은 같은 정의를 유지한다. 현재 Design에서 삭제된 ID와 추론 도중의 목록 변경은 여전히 거부하며, 새 미션 실행 직전 revision 검증은 Controller가 수행한다.
- 에피소드·프레임의 `catalog`는 당시 미션 정의다. `current_catalog`는 UI 조회 시 현재 Design에서 읽는 목록이다.
- 학습은 **라벨의 catalog → 프레임 catalog → 에피소드 catalog** 순서로 저장된 정의를 사용한다. 자동 라벨링도 저장 프레임의 catalog를 사용한다. 오늘의 조건으로 과거 입력을 몰래 교체하지 않는다.
- 후보 범위가 기록된 프레임은 그 범위 안에서만 라벨링한다. 과거처럼 범위가 없는 프레임을 새 미션으로 수기 재라벨링하면 그때의 catalog를 라벨에 보존한다.
- 삭제한 미션 ID는 현재 학습 정답으로 사용할 수 없다. 새 미션 ID로 다시 라벨링해야 한다.
- UI 표시 이름은 안정된 mission ID로 최신 이름을 연결할 수 있지만 과거 이미지·정답·학습 스냅샷 자체는 변경하지 않는다.

## 이전 데이터와 모델

`reasoning_task`가 없는 샘플은 mission_selection으로 취급한다. active_mission이나 기존 continue 라벨이 있다는 이유로 supervision으로 승격하지 않는다.

- 기존 start/wait/unknown 정답은 문맥·후보 검증을 통과하면 사용할 수 있다.
- task 없이 저장된 cancel/continue/pause는 감독 정답으로 자동 변환하지 않는다. UI에서 미검토로 표시하고 호환되는 정답을 직접 선택해야 한다.
- 과거 mission_supervision의 returning 프레임은 pause/continue 정답과 active_mission ID를 유지해 읽고 학습할 수 있다. 이 문맥에서 cancel은 거부한다. 새 복귀 프레임은 patrol_supervision이며 active_mission과 mission_id는 null이다.
- 기존 wait는 pause와 의미가 다르므로 일괄 치환하지 않는다.
- 과거 Scenario는 원본 메타데이터로 남지만 필수 라벨이나 모델 입력·출력으로 쓰지 않는다.
- 예전 데이터에서 알 수 없는 관찰 스팟·실행 단계·시각 관계를 만들어 넣지 않는다.

기존 선택용 모델이 continue/pause/cancel 프롬프트에 응답할 수 있다는 사실만으로 감독 성능이 검증된 것은 아니다. 실행 중 문제 장면, AI 일시중지 상태에서 문제가 남은 장면, 해소되어 continue할 장면, 작업이 불필요해져 cancel할 장면을 따로 취득·검수해야 한다.

## 학습 준비와 모델 메타데이터

종료된 에피소드의 승인 라벨만 학습에 포함한다. 준비하는 train/val/test.jsonl은 group_id SHA-256 기반 80/10/10 split이며, 그룹 수가 적으면 빈 split이 생긴다. 비슷한 장면이 다른 에피소드에 반복될 때는 별도 평가 장면을 확보해야 한다. 이미지·마스크는 원본 상대 경로를 참조한다.

라벨·마스크 수정 시 생성된 학습 파일과 episode.training을 무효화한다. Qwen UI 학습은 실행 중 임시 입력을 만들고 종료 시 제거하며 별도 exports 계층을 남기지 않는다.

```bash
python3 cyclo_brain/policy/qwen/training/data_format.py \
  --dataset docker/workspace/dataset/reasoning/Home/observation/raw_episode_0001
```

start/wait 균형 조정은 선택 작업에 적용한다. 이 옵션에서 unknown은 제외하고, **감독용 continue/pause/cancel 샘플은 전부 유지**한다. 감독 데이터만 있는 학습도 가능하다.

episode·sample·typed messages 필드 구조, 모델 스키마 `qwen_mission_annotation_v3`, 기존 저장 경로는 유지한다. 출력 decision의 허용 범위만 작업 문맥에 맞게 확장하고 input_policy로 계약 버전을 기록한다. 추가 메타데이터는 실제 선택된 학습 행을 기준으로 계산한다.

| 필드 | 의미 |
|---|---|
| `input_policy` | `continuous_autonomy_v4` (미션 감독 cancel 추가) |
| `reasoning_tasks` | 실제 학습 샘플이 있는 작업 목록 |
| `task_sample_counts` | 선택/감독별 실제 학습 샘플 수. 없는 작업은 0 |
| `supervision_decision_counts` | 실제 continue/pause/cancel 정답 개수 (없으면 0) |
| `training_catalog_snapshot_ids` | 학습에 사용한 저장 catalog들의 체크섬 |
| `catalog_snapshot_id`, `mission_ids`, `mission_revisions` | 학습 시작 당시 현재 Design 정보 |

모델 목록의 `catalog_compatible`은 `training_catalog_snapshot_ids`가 있으면 그 모든 값과 현재 Design을 비교한다. 옛 모델은 기존 catalog_snapshot_id 비교로 처리한다. `supervision_trained`는 실제 감독 샘플·판단 개수가 양수일 때만 true이며, 정확도나 안전성을 보장하는 값은 아니다. 감독 샘플이 0개인 모델을 감독 학습 완료로 표시하지 않는다.

기존 모델과 데이터는 자동 재학습하지 않는다. 저장된 라벨로 학습 작업을 명시적으로 실행하고 Operate에서 새 모델을 선택해야 적용된다.


## 판단 기반 Observation 출발

- 관찰 지점의 `departure_policy` 기본값은 `reasoning`이다. 기존 `dwell_seconds`가 있어도 `timed`를 명시한 지점에서만 타이머로 이동한다.
- Record에서 `By decision` 지점은 계속 관찰한다. 수집은 유지한 채 Next observation 버튼으로 이동 시점을 직접 시연한다. 이동 버튼 클릭이 자동으로 정답을 저장하지는 않는다.
- Train의 Reasoning labels에서 출발 직전 관찰 프레임에 `next_observation`을 지정한다. `mission_id`는 null이다. 아직 지켜볼 필요가 있으면 `wait`, 시야가 가려졌거나 불확실하면 `unknown`이다. 기존 wait 라벨은 자동 변환하지 않는다.
- 프레임의 `context.observation`에 `can_advance`, `next_spot_id`, `departure_policy`가 저장된다. 고정 관찰, 단일 지점, 편도 마지막 지점은 이동을 실행하지 않는다.
- Operate는 유효한 최신 프레임의 `next_observation`을 받으면 다음 저장 지점으로 이동한다. 이동 중에는 기존 patrol supervision(`continue / pause`)이 적용된다. 미션 완료 후에는 원래 관찰 지점으로 돌아와 다시 판단한다.
- JSONL messages 구조와 세 가지 reasoning_task는 유지한다. 새 출발 판단은 `mission_selection`의 확장이고 prompt_version은 `continuous_autonomy_v5`다. 기존 모델은 새 출발 정답으로 추가 학습해야 한다.

```json
{"decision":"next_observation","mission_id":null,"reason":"The basket is empty and no further observation is needed here."}
```


### Qwen 학습 배치 기본값

자동 라벨러 Fine-tuning과 Train의 reasoning 학습은 실제 GPU 배치 4를 기본으로 사용한다. `batch_size`는 요청과 모델 메타데이터에 기록된다. CLI에서는 `--batch-size`로 변경할 수 있다. 비전 체크포인트를 비교하는 reasoning_tuning의 mission 학습도 기본 배치는 4다. 학습을 누르기 전 이 변경 자체로 새 학습을 시작하지 않는다.
