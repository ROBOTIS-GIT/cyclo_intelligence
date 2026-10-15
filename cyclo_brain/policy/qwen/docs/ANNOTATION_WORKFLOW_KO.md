# Autonomy Canvas: Record / Annotate / Train / Operate

SAM 마스크와 미션 정답은 별도 작업입니다. 마스크를 만들지 않아도 reasoning 라벨을 저장할 수 있습니다.

## 네 작업 탭

Autonomy Studio에서 **Autonomy Canvas**를 열고 좌측 탭으로 이동합니다.

| 탭 | 작업 |
|---|---|
| **Record** | 지도·Observation 경로·Episode 취득, 카메라와 학습 모델 판단 미리보기 |
| **Annotate** | 에피소드·프레임 탐색, 샘플 추출, 수동 라벨링, 자동 라벨링·검수 |
| **Train** | Auto-labeler 또는 Reasoning model 학습, 데이터·모델명·스텝 설정, 진행률 |
| **Operate** | 모델·Observation plan 선택 후 순회·판단·미션 실행 |

Annotate의 도구는 **Reasoning labels / Object masks (optional)**, 작업 모드는 **Manual / Auto Labeling**입니다.
샘플은 Annotate에서 추출하고 정답을 저장합니다. 모델 학습 폼은 Train에서만 표시합니다.
Train의 **Training purpose**는 **Auto-labeler / Reasoning model**입니다. Auto-labeler를 선택하면
**Model**에서 Qwen(Reasoning labels) 또는 SAM(Object masks)을 선택합니다.
탭을 전환해도 선택한 에피소드·프레임을 유지하며, 저장하지 않은 편집이나 수집·주행 중에는 전환을 막습니다.

Record의 Observation에서 해당 맵의 **Reasoning model**을 선택하고 **Start reasoning**을 누릅니다.
로드된 맵이 없으면 선택한 에피소드나 Train에서 넘긴 모델의 맵을 사용합니다.
미리보기는 라이브 카메라를 분석하며 미션 실행이나 저장된 정답 수정은 하지 않습니다.
Annotate 또는 Train으로 이동하면 실행 중인 미리보기를 중지합니다.
완료 모델의 **Test in Observation**은 Record로 돌아가 모델만 선택합니다. 추론 시작은 직접 누릅니다.
**Use for auto labeling**은 선택한 모델을 유지한 채 Annotate의 해당 도구·Auto Labeling으로 이동합니다.
학습은 백그라운드 작업이므로 다른 탭으로 이동해도 계속되고, Train으로 돌아오면 서버 진행률을 다시 조회합니다.

SAM 마스크와 Qwen 미션 라벨 학습은 현재 별개입니다. 현재 Qwen 학습은 언어 모델 LoRA이며,
마스크 기반 attention 학습이나 비전타워 튜닝을 수행하지 않습니다.

## Record → 유형별 취득

지도 좌측 상단 툴바에는 **Localize · 이전 지점 · 순회 시작/일시정지/재개 · 다음 지점 · Stop**의 주행 버튼 다섯 개가 있고, 구분선 뒤에 **Create waypoint · Edit route · Undo · Redo**가 이어진다. 지도 우측의 Capture 패널은 지점 선택과 지도 편집 중에도 계속 표시되며, 취득 세션과 컨트롤을 유지한다.

1. 툴바의 **Create waypoint → On Map / At Robot**으로 Observation waypoint를 만들고 초안을 **Save**한다. **Edit route**에서 지도 지점을 클릭하면 해당 지점 옆 팝업에서 이름·체류 시간·**Include in route**·연결할 하위 Mission을 편집한다. 본체와 방향 핸들을 드래그해 위치·방향을 바꾼다. 툴바 아래에는 **Save / Set observation route order** 아이콘 두 개만 표시된다. 순서 지정 아이콘을 켜면 초안 경로를 새로 그리며, 클릭한 지점에 방문 번호와 연결선이 표시된다. 두 지점 이상 선택한 뒤 첫 지점을 다시 클릭하면 Closed loop로 마치고, 순서 지정 아이콘을 끄면 One-way로 마친다. 순서 지정 중에는 설정 팝업과 위치·방향 편집을 잠근다. 기존 지점을 다시 포함하면 체류 시간·연결 미션을 유지한다. **Save** 전까지 저장된 경로는 바뀌지 않으며, 취소하려면 **Edit route**를 다시 눌러 변경 폐기를 확인한다. Operate의 **Observation plan**도 같은 계획을 읽고 수정한다.
2. **Localize** 후 **Start Patrol**로 저장된 관찰 경로만 주행한다. 편집 중이 아닐 때 지도 지점을 선택하고 **Go to selected observation**으로 이동할 수도 있다. One-way는 마지막 지점에서 정지하고 Closed loop는 반복한다.
3. 계속 표시되는 **Capture** 패널의 설정 아이콘에서 촬영 모드·간격을 정하고 **Start episode → Capture**로 수집한다. **Recording type**은 실제 상태에 따라 자동 표시된다. Observation은 지점 도착·정렬 후, Patrol은 초기 접근·지점 간 이동·복귀 중, Mission은 실제 하위 작업 실행 중이다. Qwen 실행은 필수가 아니다.
4. 관찰 지점에서 오래 수집하려면 순회 버튼의 **Pause patrol**을 누른다. **Resume patrol**로 남은 체류 시간부터 이어간다. 체류·일시정지 중 연결된 하위 Mission을 실행하면 작업 → 출발 Observation 복귀 → 남은 체류 시간 대기 → 다음 지점 순회로 이어진다. 녹화는 이 과정에서 유지되며, 녹화 없이 실행할 수도 있다. 미션을 선택하지 않으면 체류 시간 뒤 다음 지점으로 이동한다.
5. 연속 촬영 중 실제 유형·대상이 바뀌면 구간별 에피소드로 자동 분리하고 같은 `group_id` 및 전후 에피소드 ID로 연결한다. Mission 복귀도 Patrol 구간으로 취득한다. 녹화가 꺼져 있으면 새 구간을 만들지 않는다.
6. **Capture/Pause**의 일시정지는 촬영만, **Stop**은 주행·BT와 예약된 자동 순회 재개를 취소한다. **Save episode**는 현재 저장을 마친 뒤 에피소드를 닫으며 로봇을 멈추지 않는다. 맵·로컬라이제이션 종료는 더보기 메뉴의 **Unload map**을 사용한다.

Record의 Start patrol과 Run child mission은 Episode 및 Capture와 독립적으로 실행할 수 있다. 데이터를 수집하려면 Episode를 열고 Capture를 켠다. 수집 유형은 직접 고르지 않으며, 실행 전에 Capture를 켜면 검증된 실행 문맥이 생길 때까지 프레임 저장은 대기한다. 경로 중간에 멈춘 로봇은 Observation 도착으로 처리하지 않는다. 기존 에피소드를 열어도 이전 경로나 미션을 다시 실행하지 않는다. 초기 접근과 단일 목적지 이동도 Patrol로 수집할 수 있고, 미션을 촬영하지 않았어도 복귀 중부터 수집할 수 있다.

Annotate에서는 Observation / Patrol / Mission 목록에서 각 구간을 따로 라벨링한다. 같은 수집 세션의 구간은 학습·평가 분리에서도 같은 그룹을 유지한다. 촬영 당시 지점 후보 미션과 계획 스냅샷을 사용하므로 오늘의 계획 편집으로 과거 정답을 바꾸지 않는다.

## Annotate → Reasoning labels → Manual

이미지와 라벨 편집기를 함께 보며 샘플 추출·프레임 선택·정답 저장을 수행합니다. 학습 설정은 Train으로 분리합니다.

1. Observation / Patrol / Mission 중 유형을 고르고 종료된 에피소드를 선택합니다.
2. `Sample count`에 현재 에피소드의 프레임 수 이내에서 필요한 개수를 넣고 `Sample frames`를 누릅니다. 에피소드 전 구간에 분산해 무작위 추출합니다.
3. Observation 프레임은 `Expected decision`과 필요한 `Mission`, `Reason`을 작성합니다. Mission 감독 프레임은 `Mission action`(Continue / Pause / Cancel and return to patrol)과 `Reason`을 작성하며 실행 중 미션은 기록에서 자동 연결됩니다. Patrol 프레임은 `Movement action`(Continue / Pause)을 선택합니다. `Save`하면 다음 샘플로 넘어갑니다. Mission 에피소드 중 실행 문맥이 없는 시작 전 프레임은 감독 라벨을 저장할 수 없습니다.
SAM의 `sam_sample`과 별도로 `episode.json`의 `qwen_sample.frame_ids`에 선택 목록을 보관합니다.
기존 샘플/마스크/라벨은 재추출로 삭제하지 않습니다.

## Train → Auto-labeler

1. **Training purpose: Auto-labeler**에서 **Qwen · Reasoning labels** 또는 **SAM · Object masks**를 선택합니다.
2. 선택한 에피소드의 맵에서 **Training data**와 **Training scope**로 사용할 승인 데이터를 고릅니다.
   Qwen은 **Training types**로 Observation, Patrol, Mission 샘플을 함께 또는 일부만 학습합니다.
3. **Model name**과 **Training steps**를 입력하고 **Train Qwen / Train SAM**을 누릅니다.
   스텝 범위는 기존 서버 계약과 같은 2–2,000이며 기본값은 Qwen 200, SAM 1,000입니다.
4. 진행률과 완료 상태를 확인한 뒤 **Use for auto labeling**으로 Annotate에서 해당 모델을 사용합니다.

Qwen은 승인 프레임이 최소 2개 필요하며 미라벨·초안은 제외합니다. 프레임별 `reasoning_task`와 실행 문맥은 유지됩니다.
기본 Qwen 학습은 Qwen3.5-4B NF4 LoRA, learning rate 5e-5입니다. SAM 샘플 추출과 수동 마스크 저장은
Annotate → Object masks → Manual에서 수행합니다.

## Annotate → Reasoning labels → Auto Labeling

1. 같은 맵에서 학습한 이름 있는 Qwen 모델을 선택합니다.
2. 대상(현재 에피소드/동일 맵에서 모델이 지원하는 유형의 모든 종료 에피소드)을 선택하고 `Generate labels`를 누릅니다.
3. 미승인 프레임에 `decision`, `mission_id`, `reason`을 자동 생성합니다.
4. 완료 후 `Review labels`에서 수정하고 `Save`로 승인합니다. `Unreviewed labels only`로 미검수 프레임을 볼 수 있습니다.

- 수기로 저장/승인한 라벨은 생성 대상에서 제외합니다.
- 생성 중 사람이 수정한 라벨은 리비전 검사로 덮어쓰지 않습니다.
- `Scene group`, `Robot`, `Environment`는 이미지로 추측하지 않고 프레임의 기존 수집 정보를 유지합니다.
- 마스크는 수정하지 않습니다. 생성된 reasoning 라벨은 draft이며 검수 전 학습 정답에 포함되지 않습니다.
- Observation은 `start / wait / unknown`, Patrol은 `continue / pause`, Mission은 `continue / pause / cancel`을 촬영 당시 문맥·카탈로그와 검증합니다.
  이미지 인식이 틀리거나 출력 형식이 잘못되면 failed 수와 오류가 표시되므로 수기로 확인해야 합니다.

## Train → Reasoning model

독립된 **Train** 탭은 선택한 맵에 속한 모든 종료 에피소드에서 검수·승인된 미션 라벨만 모아
Operate용 Reasoning 모델을 학습합니다.

1. Annotate에서 자동 생성된 draft를 프레임별로 검토하고 `Save`로 승인한 뒤 Train의 **Training purpose: Reasoning model**을 선택합니다.
2. `Training map`과 `Training types`로 맵·유형을 선택하고 승인 라벨 수를 확인합니다. `Balance decisions per type`은 유형별로 다수 결정만 줄이고 소수 라벨은 보존합니다. Available/Training 분포를 비교할 수 있습니다.
3. `Training steps`와 최종 모델명을 입력하고 `Train reasoning model`을 누릅니다.
4. 설정한 step 기준의 학습 진행률과 완료 상태를 확인합니다.
5. 완료된 **Reasoning model**을 선택해 `Test in Observation`으로 라이브 장면에서 확인합니다.

샘플 수동 라벨로 만든 모델은 자동 라벨 생성용(`auto_labeling`), 전체 승인 데이터로 만든 모델은
최종 실행용(`reasoning`)으로 구분해 `model.json`에 저장합니다. 승인되지 않은 draft와 다른 맵의
에피소드는 최종 학습에 포함하지 않습니다.

## 데이터 및 모델

원본 저장 구조는 유지합니다:
`dataset/reasoning/<map>/<observation|patrol|mission>/raw_episode_####/episode.json`, `samples.jsonl`, `images/`, `masks/`. Annotate 목록에서 Observation, Patrol, Mission을 전환해 해당 유형만 라벨링한다. 프레임의 실행 문맥에 따라 판단 선택지가 달라지며, 폴더 이름만으로 기존 프레임의 문맥을 바꾸지 않는다.
수집 메타데이터는 observation.metadata, 정답은 annotations[].target에 저장합니다.
Scenario 입력은 없으며, 보이는 상황과 판단 근거를 Reason 한 칸에 작성합니다.
Scene group은 새 수집 시작 시 자동 생성하고 연결된 유형 구간이 공유합니다. Robot은 UI에서 선택한 로봇, Environment는 수집 설정을 사용합니다.
설정이 없으면 unknown으로 남기며 이미지나 맵 이름으로 추측하지 않습니다.

학습·자동 라벨링·Operate는 `mission_decisions.py`의 동일한 시스템 프롬프트와 문맥 변환을 사용합니다.
Scene group / Robot / Environment / 기존 Scenario는 모델 입력이나 출력에 넣지 않습니다.
과거 원본의 Scenario는 호환성을 위해 보존하고, 기존 reason 정답을 그대로 사용합니다.
새 annotation 모델 스키마는 `qwen_mission_annotation_v3`이며 기존 5단계 판단 모델은 재학습해야 선택할 수 있습니다.

학습 입력은 공식 Transformers typed `messages` 구조(system / user의 image+text / assistant의 text)를 사용합니다.
assistant 정답 JSON은 다음과 같습니다:

```json
{"decision":"start","mission_id":"<catalog mission ID>","reason":"The table is unoccupied and an empty plate is visible."}
```

학습 작업 중 임시 디렉터리에 입력 이미지와 train.jsonl을 준비하고 종료 시 제거합니다.
선택한 원본 프레임/정답 스냅샷은 `model/qwen/annotation/jobs/<job ID>/request.json`에 기록합니다.
자동 라벨 모델은 `model/qwen/annotation/<map>/<model name>/adapter/`, 최종 실행 모델은
`model/qwen/reasoning/<map>/<model name>/adapter/`에 분리하고 모델 정보는 각각 같은 폴더의 `model.json`에 보관합니다.
작업 상태는 모델 저장소의 `workflow.json`에 기록하며 API 재시작 시 같은 작업 컨테이너에 재연결합니다.

## 실행 환경

기존 `mission-reasoning/qwen-training:0.1.0` 이미지에 소스 디렉터리를 마운트한 `qwen_annotation_job` 컨테이너에서 실행합니다.
작업 컨테이너는 원본 데이터는 읽기 전용으로 보고 모델 저장소만 씁니다. 초안 저장과 정답 검증은 supervisor API가 담당합니다.
학습과 자동 라벨링은 Operate 실행 모델을 자동으로 교체하거나 미션 명령을 보내지 않습니다.

### Operate에서 튜닝 모델 사용

1. Operate에서 맵을 로드합니다.
2. Current situation의 **Reasoning model**에서 해당 맵의 학습 완료 모델을 선택합니다. 새로 학습한 모델은 새로고침 버튼으로 갱신합니다.
3. Localize 후 지도 상단의 Run을 누릅니다. `qwen_server`가 시작되고 선택한 베이스 모델·revision·LoRA를 로드한 뒤 판단 결과로 기존 미션 실행기를 제어합니다.
4. 실행 중에는 모델 선택이 잠깁니다. Stop 후 다른 모델을 선택할 수 있습니다. 선택값은 브라우저에 맵별로 기억합니다.

`GET /reasoning/models?map_name=...`는 완료된 `model.json`과 가중치를 확인합니다.
기존 CLI 학습 모델은 `training_report.json`의 정답 미션들이 현재 맵에 속할 때만 **Legacy**로 표시합니다. 이는 기존 모델을 불러오는 기능이며 새 5종 판단 학습이나 성능 검증을 의미하지 않습니다.
`POST /reasoning/start`에 `model_id`, `map_name`을 보내면 서버가 등록 경로를 확인합니다. 클라이언트가 임의의 모델 경로를 지정하지 않습니다.
로딩 또는 모델 일치 확인 실패 시 미션 판단을 실행하지 않습니다. 기존 모델로 자동 대체하지 않으며 `.env`도 변경하지 않습니다.
모델 전환과 Stop 이전의 오래된 판단은 세션 확인으로 차단합니다.

필요 시 supervisor API 환경변수 `QWEN_ANNOTATION_IMAGE`, `QWEN_ANNOTATION_MODEL_ID`, `QWEN_ANNOTATION_MODEL_REVISION`, `QWEN_GPU_DEVICE`로 실행 구성을 지정합니다.
적은 샘플 학습은 같은 에피소드 자동 라벨링을 위한 시작점이며 새로운 장면에 대한 성능은 별도로 검증해야 합니다.

### 검토 완료 마스크로 재학습

Train → **Auto-labeler → SAM · Object masks**의 **Training data**에서 **All reviewed frames**를 선택하면
무작위 샘플 외에 자동 생성 후 승인한 마스크도 학습에 포함됩니다.
**Choose episodes**로 같은 맵의 여러 에피소드를 선택하고 모델명을 입력한 뒤 Train SAM을 실행합니다.
초안이 남아 있는 프레임은 제외하며, 마스크 없는 검토 완료 프레임은 현재 학습 인스턴스로 사용하지 않습니다.
한 작업당 1,024프레임·4,096인스턴스 제한은 유지됩니다. 학습 스텝은 기본 1,000이며 Train에서 변경합니다.
이는 검토 데이터로 새 모델을 학습하는 기능이며 이전 모델 체크포인트에서 이어 학습하는 기능은 아닙니다.

Auto Labeling에서 새 모델과 **Regenerate unreviewed masks**를 선택하면 학습 프레임과
검토 완료 프레임을 제외하고 미검토 자동 초안을 교체합니다.
승인 마스크와 수동 초안은 보존합니다. 추론이 실패하면 이전 결과를 유지하고,
정상적인 빈 예측이면 이전 자동 초안을 제거하여 다시 검토할 수 있게 합니다.

### 검토 완료 Reasoning labels로 재학습

Train → **Auto-labeler → Qwen · Reasoning labels**의 Training data에서 **All reviewed frames**를 선택하고
**Choose episodes**로 같은 맵의 학습 에피소드를 지정합니다.
무작위 샘플 여부와 관계없이 승인된 라벨만 사용하여 새 자동 라벨링 모델을 학습합니다.
최소 2프레임이 필요하며 미검토 초안은 제외됩니다. 기존 체크포인트의 이어 학습은 아닙니다.

Auto Labeling에서 새 모델을 선택하고 **Regenerate unreviewed labels**로 자동 초안을 갱신하거나,
**Add missing labels**로 라벨 없는 프레임만 채울 수 있습니다.
승인 라벨과 수동 초안은 보존하며, 추론 실패 시 기존 라벨은 유지됩니다.
Operate용 최종 모델은 Train의 **Training purpose: Reasoning model**로 학습합니다.

SAM 학습은 이미지 특징을 임시 디스크 캐시에 저장하고 현재 프레임만 메모리로 읽습니다. 정상 종료 또는 예외 종료 시 캐시는 정리됩니다.

### Record 편집 되돌리기

지도 툴바의 **Undo / Redo**는 현재 편집 세션에서 Observation waypoint의 위치·방향·이름과 경로 포함 여부·순서·체류 시간·연결 미션·루프 설정 변경을 되돌린다. 드래그와 Undo/Redo 결과도 저장 전 초안이며, **Save**로 지점과 경로 변경을 함께 확정한다. 별도의 Cancel 버튼은 없으며, **Edit route**를 다시 눌러 변경 폐기를 확인하면 저장된 지점과 경로를 복원한다. 새 편집을 하면 Redo 이력이 초기화된다. 실행·캡처 중에는 지도 편집을 사용할 수 없다.
