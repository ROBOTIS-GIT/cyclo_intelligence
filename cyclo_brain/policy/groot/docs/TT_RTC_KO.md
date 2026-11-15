# Playground TT-RTC와 RLT 실행

TT-RTC는 InferenceCommand의 `action_request_mode="tt_rtc"`로 선택합니다. 이전 `ttrtc_enabled` 토글과 `ttrtc_execution.py` 실행 경로는 제거되었습니다. 일반 `sync`/`async`와 VLA-RAIL은 기존 실행 경로를 유지합니다. VLA-RAIL과 TT-RTC/RLT는 동시에 활성화할 수 없습니다.

32×16 체크포인트는 32개 시점, 16개 물리 액션 차원을 사용합니다. 모델의 원본 액션은 15 Hz이고 `control_hz`(최소15 Hz)는 이 액션을 로봇 명령으로 보간하는 주파수입니다. LOAD의 `action_chunk_size`는0을 보내며, 실제 horizon과 차원은 검증된 체크포인트 응답을 사용합니다.

런타임의32×16 prefix 지원 범위는0~8입니다. 단, 체크포인트 manifest와 모델 설정이 실제 학습한 범위를 증명해야 합니다. 기존 `max_delay_steps=6` 체크포인트는0~6만 허용하며7/8 요청은 거절합니다. `max_delay_steps=8` 및 일치하는 학습/모델 metadata가 있는 체크포인트에서만0~8을 실행합니다. UI 설정이나 사용자 선택으로 학습 범위를 변경하지 않습니다. 16×19(0~6),20×16(6~8) 소스 프로파일도 호환됩니다.

`CYCLO_TT_RTC_PREFIX_MODE`는 `fixed_first_6`(기본), `fixed_first_8`, `latency_adaptive`를 지원합니다. `fixed_first_8`은8까지 학습된 체크포인트가 필요합니다. 요청에는 이미 실행 큐에 확정된 물리 액션 prefix만 전달하며, 응답은 유지된 prefix 뒤의 연속 구간에 결합됩니다. source clock 기준 prefix와 Control Hz 기준 보간 명령을 구분합니다. 단순히 이전 구현의7개prefix/24개실행 구간으로 자르지 않습니다.

RLT는 LOAD 시 `rlt_enabled`와 검증된 `rlt_bundle_path`로 미리 로드합니다. `SET_ACTION_POLICY=20`은 base/RLT 출력을 안전한 경계에서 전환합니다. 상태/자동적용/적용/학습ON/OFF/dataset/save 명령은21~28이며, 기존 `CAPABILITIES=7`, `FAULT_STOP=8`과 충돌하지 않습니다. RLT는 원본 VLA 참조 구간과 실행 액션을 구분하여 trace에 기록합니다. 학습 실행, 정책 적용, 로봇 출력 전환은 별도 명령입니다.

STOP/PAUSE 및 실패 시의 자동 정지는 로봇에 hold를 보낸 뒤 최신 관측으로 실제 정지를 확인합니다. hold 실패는 재시도 가능한 미확인 상태로 남으며, 감독 fault latch가 설정되면 새로운 LOAD/START/RESUME와 정책 변경을 차단합니다. 실제 로봇 실행 전 체크포인트/processor/모델 계약과 deployment qualification을 검증해야 합니다.

## 체크포인트 metadata 예시와 mock 제한

`RobotisSW/mock_test`는 shape가32×16, prefix가0~8이어도 새 LOAD가 거절됩니다. 실제 `experiment_cfg/tt_rtc_training_manifest.json`은 `training_time_rtc: true`라는 boolean과 `trained: false`, `complete: false`, `scope: "mock-inference-only"`를 선언합니다. source loader가 요구하는 nested 학습 계약이 없고, TT-RTC 학습이 완료된 가중치도 아닙니다. CPU 검증은 이 metadata를 읽어서 거절하는 것까지 확인했으며 모델 가중치를 로드하거나 checkpoint를 수정하지 않았습니다. 설정 숫자가 맞는다는 사실은 학습 성능이나 로봇 배포 적합성의 증거가 아닙니다.

실제0~8 학습이 완료된 checkpoint의 `tt_rtc_manifest.json` 예시는 다음과 같습니다. 아래는 형식 설명이며, 미학습 checkpoint의 flag를 바꿔 통과시키는 용도가 아닙니다. 실제 로드한 모델/processor의 horizon, 차원, denoising 설정도 이 선언과 대조합니다.

```json
{
  "schema": "cyclo.training-time-rtc/v1",
  "training_time_rtc": {
    "trained": true,
    "action_horizon": 32,
    "action_dimension": 16,
    "action_hz": 15.0,
    "min_delay_steps": 0,
    "max_delay_steps": 8,
    "min_postfix_steps": 24,
    "model_action_horizon": 40,
    "model_action_dimension": 132,
    "processor_action_horizon": 40,
    "processor_action_dimension": 132,
    "num_inference_timesteps": 4,
    "num_timestep_buckets": 1000,
    "delay_sampling": {
      "type": "uniform_integer",
      "min_inclusive": 0,
      "max_inclusive": 8
    },
    "prefix_input": "ground_truth_clean_action",
    "loss_region": "postfix_only",
    "per_action_timestep": true,
    "flow_convention": {
      "noise_endpoint": 0.0,
      "clean_endpoint": 1.0,
      "velocity_target": "action_minus_noise"
    }
  }
}
```

외부 학습 기록 `experiment_cfg/tt_rtc_training_manifest.json`도 지원합니다. 이때 `training_time_rtc`는 동일한 object여야 하며 `state_timestep: "sampled_flow_time"`, 학습 config의 logical horizon/dim/delay 값, processor의 logical delta_indices가 서로 일치해야 합니다. RLT를 함께 켜려면 RLT bundle 자격 검증에 더해 manifest의 `rlt`에 `chunk_length: 10`, `reference_horizon: 32`, `reference_slice: "[d:d+10]"`를 선언해야 합니다.
