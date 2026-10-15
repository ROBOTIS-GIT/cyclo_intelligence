# 전체 승인 데이터 Qwen 학습 (2026-09-14)

- 모델: Qwen/Qwen3.5-4B, revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
- 데이터: gazebo_warehouse 종료 에피소드 7개, 승인 이미지/정답 25개. Mission1 15개, Mission2 10개.
- 전체 승인 데이터를 학습에 사용했다. 별도 홀드아웃 평가가 아니며 원본 에피소드/라벨은 변경하지 않았다.
- NF4 4bit LoRA rank 8, alpha 16, 학습률 5e-5, 250 steps, 매 epoch shuffle(seed 42).
- 이미지+언어의 승인 미션 판단 JSON을 학습했다. SAM2 마스크는 이번 학습의 loss에 포함하지 않는다.
- 최초 고정 순서/2e-4/125-step 실험은 10/25 일치여서 배포하지 않았다.
- 최종 모델의 학습 이미지 재검증: decision와 mission_id 25/25 일치.
- 실제 추론 API의 대표 이미지 2장: 스틱 없음 Mission1, 빨간 스틱 있음 Mission2 확인. 실제 로봇 미션 실행 테스트는 하지 않았다.

모델: docker/workspace/model/qwen/all-approved-25frames-20260914-v2

training_report.json, training_sources.json, serving_verification.json에 상세 결과를 저장했다.
docker/.env의 QWEN_ADAPTER_PATH를 위 모델의 /workspace 경로로 지정했고 qwen_server에 로드했다.
다음 Operate 실행은 이 어댑터를 사용한다. 새로운 장면과 현재 미션 실행 상태에 대한 일반화 성능은 별도 확인이 필요하다.
