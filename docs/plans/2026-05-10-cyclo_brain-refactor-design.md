# cyclo_brain refactor — design (2026-05-10)

## Background

이번 세션에서 lerobot 정책 컨테이너의 인프라(빌드/베이스 이미지/extras/디버그 패치/검증)를
풀-스택으로 정착시키는 과정에서, `cyclo_brain/` 안의 일부 파일이 책임이 섞여 큰 파일로 자랐고
README가 stale 상태가 되었다. groot 측 비대칭은 별도(Phase 2 대상)로 두고, 이번 정리는
**lerobot 측 + `policy/common/runtime/` 공유 코드**에 한정한다.

## Scope / Non-scope

| | 포함 | 제외 |
|---|---|---|
| 파일 | `policy/common/runtime/*.py`, `policy/lerobot/lerobot_engine.py`, `cyclo_brain/README.md` | `policy/groot/*` (별도 phase) |
| 변경 종류 | (a) 정리·문서화·디버그 잔재 제거, (b) 큰 파일을 mixin 단위 다중 파일로 분할 | 패키지 디렉토리 신설/이동/이름 변경 |
| 외부 contract | 보존 — 컴포즈 볼륨(1 변경만), Zenoh 토픽, InferenceCommand srv, s6 longrun 이름, `create_engine()` factory 모두 유지 | — |

## Inputs

- Codex `codex:codex-rescue` 검토 (agentId `a6755bb9589c16bf8`) — 우선순위 8개 작업 + 위험도.
- 이번 세션의 풀-스택 검증 (`verify_clean.sh`, fake_robot_publisher) — refactor 후 회귀 게이트로 재사용.
- 사용자 합의: ① 정리 + 기능별 파일 분할만, ② 패키지 디렉토리 구조는 그대로, ③ mixin 방식,
  ④ S1–S4를 한 PR, S5/S6/S7은 각각 별도 PR.

## 현재 구조 평가 요약

| 영역 | 평가 |
|---|---|
| `sdk/` ↔ `policy/` 분리 | 의도대로. lerobot은 `common/runtime` 공유 정착, groot만 자체 보유(Phase 2). |
| `policy/common/runtime/` 책임 분리 | engine ABC / Process A / Process B 디렉토리 단위로 깨끗. 그러나 각 .py가 큼(500~683 lines). |
| `lerobot_engine.py` 내부 책임 | 531 lines, 클러스터 4개(loading / io_mapping / observation / engine class). |
| 잔재 패치 | (a) `[CP-DBG]` stderr print 12 sites, (b) state padding hack(임시, root-cause 별도), (c) HF Hub `policy_type` 감지 fix(보존). |
| 숨은 risk | `/inference/trajectory_preview` 토픽, `CMD_UPDATE_INSTRUCTION=6` 컨트랙트가 명시 문서 없음. |

## Changes — Phase 1 (정리, 단일 PR)

| ID | 작업 | 파일 | 위험 |
|---|---|---|---|
| S1 | `[CP-DBG]` stderr print 12 sites 제거 | `policy/common/runtime/control_publisher.py` | low |
| S2 | README 업데이트 (common/runtime 정책 반영, per-backend `runtime/` 묘사 제거) | `cyclo_brain/README.md` | low |
| S3 | state padding 블록에 `# TODO(<owner>)` + expected/actual dim 로그 한 줄 | `policy/lerobot/lerobot_engine.py` | low |
| S4 | `/inference/trajectory_preview` 토픽, `CMD_UPDATE_INSTRUCTION=6` 컨트랙트 명시 docstring | `policy/common/runtime/control_publisher.py`, `policy/common/runtime/inference_server.py` | low |

이 4건은 **외부 동작 불변**. PR 1건으로 합쳐 검증 부담 최소화.

## Changes — Phase 2 (책임별 파일 분할, PR 3건)

### 분할 방식

**Mixin classes.** 책임별 mixin class를 신규 파일에 정의하고, 기존 클래스가 다중상속.
외부 시그니처(`ControlPublisher.configure(...)`, `InferenceServer.start_service(...)`,
`LeRobotEngine.load_policy(...)`)는 변경하지 않는다. 호출부 zero-touch.

### 파일명 규약

prefix `_<owner>_*.py` (`_cp_`, `_is_`, `_le_`) — 같은 owner끼리 묶이고 정렬에서 인접.
신규 디렉토리는 만들지 않는다. 평탄 구조.

### S5 — `control_publisher.py` 분할 (PR 2)

`policy/common/runtime/` (디렉토리 mount, **compose 변경 없음**):

```
control_publisher.py    # ControlPublisher core: __init__/setup/shutdown/run/_tick + main (~200 lines)
_cp_lifecycle.py        # LifecycleMixin: configure/deconfigure/_setup_robot_specific_locked/
                        #                  _teardown_robot_specific_locked/_on_configure/_on_lifecycle
_cp_pipeline.py         # PipelineMixin: _on_chunk/_send_trigger_locked
_cp_publishers.py       # PublishersMixin: _publish_action_locked/_publish_twist/
                        #                  _publish_joint_trajectory/_publish_trajectory_preview_locked
```

`class ControlPublisher(LifecycleMixin, PipelineMixin, PublishersMixin):` 다중상속.

### S6 — `inference_server.py` 분할 (PR 3)

`policy/common/runtime/`:

```
inference_server.py     # InferenceServer core: __init__/start_service/shutdown + _resolve_engine + main (~200 lines)
_is_commands.py         # CommandsMixin: _handle_command/_cmd_load/_cmd_start/_cmd_pause/_cmd_resume/
                        #                _cmd_stop/_cmd_unload/_cmd_update_instruction
_is_zenoh.py            # ZenohIOMixin: _publish_configure/_publish_lifecycle/_setup_zenoh_io/
                        #                _on_trigger/_publish_chunk/_teardown_runtime
```

### S7 — `lerobot_engine.py` 분할 (PR 4)

`policy/lerobot/` (단일 파일 mount → 4 파일 mount로 확장):

```
lerobot_engine.py       # LeRobotEngine core + create_engine (~200 lines)
_le_loading.py          # LoadingMixin: _resolve_model_dir/_load_policy_assets/_infer_image_resize
_le_io_mapping.py       # IoMappingMixin: _init_robot/_teardown_robot/_policy_image_keys
_le_observation.py      # ObservationMixin: _build_observation/_predict_chunk/_to_numpy_chunk
```

`docker/docker-compose.yml`의 `lerobot` 서비스 volumes에 3 라인 추가 (helper 파일 각각 mount).

## Verification gate

각 PR 후 `cyclo_intelligence` (외부 ROS2) 에서 동일 시퀀스:

1. `ros2 service call /lerobot/inference_command ... command:0` (LOAD) → response `success=True`
   + `action_keys=[arm_left, arm_right, head, lift, mobile]`
2. `ros2 service call ... command:1` (START)
3. fake_robot_publisher 가동 상태에서 `ros2 topic hz /cmd_vel` ≈ 100 Hz
4. `... command:4 / 5` (STOP/UNLOAD) 응답 `success=True`

핵심 게이트: `cmd_vel` 100 Hz가 모든 PR 후 유지. 회귀 시 git revert.

## Out of scope (이 design 종결 후 별도 phase)

- groot/runtime 의 common 마이그레이션 (Phase 2 → 별도 design)
- state padding의 root-cause (FK 또는 robot_config 스키마 확장)
- HF Hub `policy_type` 감지 fix의 lerobot upstream PR
- sdk/ 패키지 layout 일관화 (robot_client / post_processing pyproject 정리)

## Risks

| Risk | Mitigation |
|---|---|
| Mixin 다중상속 순서가 잘못되어 메서드 resolution 깨짐 | 모든 mixin이 `self`만 사용, 서로 호출 없음 — MRO 충돌 없음. PR마다 verify_clean.sh로 회귀 검증. |
| S7에서 compose mount 추가 후 helper 파일 안 보임 | 컨테이너 down/up + `ros2 service call` LOAD로 즉시 검증. mount 라인은 git diff에서 명확. |
| `/policy_runtime` 디렉토리 mount가 atomic-rename에 stale 될 수 있음 | 이번 세션에서 확인된 패턴. PR 적용 후 컨테이너 restart로 inode 갱신. |

## Next

`writing-plans` skill로 hand-off하여 PR별 step-by-step implementation plan 작성.
