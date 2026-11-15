
Playground TT-RTC는 `action_request_mode="tt_rtc"`, 모델 horizon을 사용하는
`action_chunk_size=0`으로 요청합니다. 32×16 profile은 검증된 학습 범위에 따라
prefix0~8을 지원합니다. `RobotisSW/mock_test`는 모양이 일치해도
`trained:false`, `scope:mock-inference-only`와 boolean 학습 metadata 때문에
엄격한 source LOAD 계약에서 거절됩니다. 체크포인트는 수정하지 않습니다.
지원하는 metadata 예시와15Hz/ControlHz 동작은
[TT_RTC_KO.md](docs/TT_RTC_KO.md)를 참고하세요.
