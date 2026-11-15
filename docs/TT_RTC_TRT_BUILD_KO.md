# TT-RTC TensorRT 빌드 연결 수정 및 검증 (2026-11-14)

UI가 전달한 `action_request_mode`를 supervisor API와 변환 CLI가 누락하던 문제를 수정했다. 일반 엔진 `dit_model_bf16.trt`와 TT-RTC 엔진 `dit_model_bf16_ttrtc.trt`를 분리하고 TT-RTC 전용 exporter를 선택한다. 체크포인트·GPU/CUDA/TRT 버전·엔진 SHA-256 정보를 저장한다.

## 실제 검증

- 모델: `/workspace/model/groot/RobotisSW/mock`
- 엔진: `/workspace/model/groot/RobotisSW/mock/dit_model_bf16_ttrtc.trt`
- 파일 크기: 2,191,330,388 bytes
- 빌드 종료 코드: 0
- 체크포인트/device/hash 검증 및 실제 LOAD 성공
- prefix 0과 8 각각에서 유한한 32×16 액션 청크 생성 확인
- 합성 관측 사용, RobotClient 명령 publisher 0개. 로봇 동작 미실행.
- 이 검증은 실행/shape/유한성 검사다. PyTorch 대비 수치 동등성이나 실제 작업 성공률을 검증한 것은 아니다.

## 테스트 결과 및 제한

관련 supervisor API 및 신규 모드 전달 회귀 테스트 155개 통과.

API 디렉터리 전체 검사에서는 996개 통과, 아래 9개 실패가 관찰됐다. 변경 범위 밖의 reasoning 데이터 정책 버전 기대값, SAM 데이터 읽기 횟수 기대값, Docker mock 모듈 충돌에서 발생했다. 이번 작업에서 변경하지 않았다.

- `test_collection.py::CollectionTests::test_episode_and_export_need_no_clock_fields`
- `test_collection.py::ThreeDecisionCollectionTests::test_episode_and_export_need_no_clock_fields`
- `test_labeling_automation.py::WorkflowTests::test_auto_frame_reads_and_publishes_samples_once`
- `test_labeling_automation.py::SamAutomaticTests::test_empty_predictions_persist_generation_evidence_and_no_extra_dataset_read`
- `test_mission_cancel_data.py::test_cancel_round_trip_annotation_presets_export_and_balanced_training`
- `test_qwen_annotation_runtime.py::RuntimeTests::test_cancel_serializes_with_container_creation`
- `test_qwen_annotation_runtime.py::RuntimeTests::test_cancelled_queued_job_cannot_create_a_container`
- `test_qwen_annotation_runtime.py::RuntimeTests::test_new_worker_mounts_annotation_and_reasoning_model_roots_writable`
- `test_segmentation_workflow.py::WorkflowTests::test_auto_frame_reads_and_publishes_samples_once`
