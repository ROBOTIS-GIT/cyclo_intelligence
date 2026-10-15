# Qwen Reasoning 모듈

Qwen3.5 비전-언어 모델에 텍스트와 이미지 한 장을 전달하고 판단을 텍스트로 받는 독립 Docker 서비스입니다. Operate와 Record에서 원본 0.8B·2B·4B 또는 학습된 LoRA를 선택할 수 있습니다. 현재 PC의 AMD64 NVIDIA GPU를 대상으로 합니다.

```text
이미지/상황 설명 → HTTP API → qwen_engine → 판단 텍스트
```

LeRobot·GR00T처럼 `policy/` 아래에 의존성, 엔진, Dockerfile을 모으고, 소스는 읽기 전용 마운트로 연결합니다. 가중치는 공용 Hugging Face 캐시에 보관합니다. 기존 `common/runtime`은 관절 action chunk를 출력하므로 Qwen은 별도 HTTP 계약을 사용합니다. 단일 서버 프로세스만 필요해 Docker가 프로세스를 관리하며 s6의 제어 루프는 사용하지 않습니다.

상황 설명과 미션 판단의 `reason`은 영어로 직접 생성합니다. 별도의 한국어 번역 단계는 없습니다. Collect에서는 **Load Map → Start reasoning**으로 미션 판단을 미리 보고, Operate에서는 **Load Map → Localize → Run autonomy**로 판단과 미션 실행을 시작합니다. 자세한 사용법은 [Operate 가이드](docs/OPERATE_KO.md)를 참고하세요.

Operate는 미션 판단에 따라 기존 Mission Run 실행기를 호출합니다. Manual intervention의 언어 지시는 다음 판단 입력에 포함됩니다. 음성 인식과 장기 상황 기억은 아직 포함하지 않습니다. Collect에서 탭을 이동하면 관측 루프가 멈추지만 모델은 GPU에 남습니다. 빨간 Stop은 컨테이너 종료로 GPU 메모리까지 반환하며, 로딩·추론 중에도 사용할 수 있습니다. 종료 완료 전 Run은 비활성화됩니다.

UI 연결은 supervisor API의 `/reasoning/start`, `/reasoning/stop`, `/reasoning/status`, `/reasoning/observe`(Collect 미리보기), `/reasoning/decide`(Operate 판단)가 담당합니다. 기본 카메라는 `/zed/zed_node/left/image_rect_color/compressed`이며 supervisor 환경의 `QWEN_HEAD_CAMERA_TOPIC`으로 변경할 수 있습니다. supervisor API가 ROS CompressedImage 토픽을 한 번 구독하고 최신 JPEG 한 장만 보관합니다. 추론과 Collect 캡처는 이 프레임을 읽으므로 web_video_server에 의존하지 않습니다. 2초보다 오래되거나 같은 타임스탬프로 반복되는 프레임은 신선도를 갱신하지 않으며, 새 이미지가 없으면 카메라 오류를 반환합니다. 요청 완료 후 0.2초 대기한 뒤 다음 이미지를 분석합니다. 요청은 순차 처리하므로 추론이 겹치지 않습니다.

## 1. 시작

```bash
cd ~/sw_workspace/mission_reasoning/cyclo_intelligence
bash docker/container.sh start-qwen --build
curl -s http://127.0.0.1:7110/health
```

처음에는 `state: unloaded`, `ready: false`가 정상입니다. 컨테이너 기동만으로 모델 가중치를 다운로드하거나 GPU에 올리지 않습니다. 이후에는 `--build` 없이 시작할 수 있습니다. Python 코드 수정은 컨테이너를 재시작하면 반영되고, 의존성 수정은 재빌드가 필요합니다.

## 2. 로드하고 추론

```bash
python3 cyclo_brain/policy/qwen/scripts/smoke_qwen.py --load \
  --prompt '휴게실에 사람들이 다시 들어왔다. 청소 미션을 계속해도 될지 간단히 판단해줘.'
```

첫 `/load`는 원본 모델 다운로드와 4bit 로딩 때문에 오래 걸립니다. 4bit는 메모리에 올릴 때 적용되므로 원본 가중치 다운로드 용량까지 4분의 1이 되지는 않습니다. 이후 다운로드는 캐시를 재사용합니다. `/load`가 실패하면 `logs-qwen`으로 원인을 확인하고 다시 요청하세요.

이미지와 함께:

```bash
python3 cyclo_brain/policy/qwen/scripts/smoke_qwen.py \
  --image /absolute/path/lounge.jpg \
  --prompt '보이는 사람과 테이블 상태를 설명하고, 청소를 시작해도 될지 판단해줘. 보이지 않는 사실은 추측하지 마.'
```

기본값은 빠른 최종 응답 모드입니다. `--thinking --max-new-tokens 2048`로 thinking 모드를 시험할 수 있습니다. thinking 토큰도 출력 한도를 사용합니다. `finish_reason: length`이거나 `text`가 비어 있으면 완성된 판단으로 취급하지 마세요.

## 3. API

기본 주소: `http://127.0.0.1:7110` · API 문서: `/docs`

| 요청 | 역할 |
| --- | --- |
| `GET /health` | 서버 생존 및 모델 상태 조회. unloaded 상태도 HTTP 200 |
| `GET /ready` | 모델 로드 완료 시 200, 미로드 시 503 |
| `POST /load` | 환경 변수의 기본 모델 또는 요청한 모델/리비전/선택적 LoRA를 로드. 완료까지 대기 |
| `POST /generate` | 텍스트 + 선택적 JPEG/PNG 한 장 추론 |
| `POST /unload` | 모델과 GPU 메모리 해제 |

`POST /generate` JSON 예:

```json
{"prompt":"현재 상황을 판단해줘.","max_new_tokens":512,"enable_thinking":false}
```

선택 필드: `system_prompt`, `image_base64`(이미지 파일의 원시 base64). 이미지 URL/서버 파일 경로는 받지 않습니다. 응답은 `model`, `text`, `finish_reason`, `input_tokens`, `output_tokens`, `elapsed_seconds`를 포함합니다. 미로드 요청은 503, 중복 추론·로드·해제 요청은 409, 잘못된 입력은 422를 반환합니다. 요청은 대기열에 쌓지 않으며 대화 이력도 저장하지 않습니다. 클라이언트가 연결을 끊어도 이미 시작한 추론이 즉시 취소되지는 않습니다.

API는 기본적으로 호스트의 loopback에만 공개합니다. 현재 Cyclo 컨테이너는 host network를 사용하므로 같은 주소로 접근할 수 있습니다.

## 4. 메모리 해제와 종료

```bash
curl -s -X POST http://127.0.0.1:7110/unload
bash docker/container.sh stop-qwen
# 로그/셸
bash docker/container.sh logs-qwen
bash docker/container.sh enter-qwen
```

`/unload`는 모델 가중치와 미사용 캐시를 반환합니다. CUDA 컨텍스트까지 없애려면 `stop-qwen`으로 컨테이너를 정지하세요. `/health`의 `gpu_memory`는 모델 로드 중 할당량과 예약량(MiB)을 보여줍니다.

GR00T와 함께 사용할 때는 실제 GPU 사용량과 추론 지연을 측정해야 합니다. 기본 4bit도 모델 외에 이미지 처리와 생성용 메모리를 사용합니다.

## 학습 취소

Train 화면에서 샘플 Fine-tune 실행 중에는 해당 아이콘 옆에 정지 아이콘(`Cancel fine-tuning`)이 표시됩니다. 본 학습은 `Train model` 설정 패널의 진행률 옆 `Cancel training`으로 취소합니다. 종료 중에는 `Cancelling`, 실제 학습 컨테이너 종료 후에는 `Cancelled`를 표시하며 새 학습을 시작할 수 있습니다. 원본 이미지와 라벨, 완료된 모델은 유지됩니다. 취소된 미완성 체크포인트는 완료 모델 목록에 등록하지 않습니다.

`POST /api/collection/qwen/jobs/{training|reasoning_training}/cancel`에 현재 작업의 `job_id`를 전달합니다. 오래된 작업 ID는 409로 거부합니다. 같은 이름으로 취소된 학습을 다시 시작하면 별도의 `_retry_N` 출력 폴더를 사용하므로 이전 부분 체크포인트를 재사용하지 않습니다.

## 설정

환경 변수를 지정하고 `start-qwen`을 다시 실행하면 Compose가 변경을 반영합니다.

| 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `QWEN_MODEL_ID` | `Qwen/Qwen3.5-9B` | Hugging Face ID 또는 컨테이너 내부 모델 경로 |
| `QWEN_MODEL_REVISION` | `main` | 모델 버전. 재현 실험에서는 HF 커밋 ID로 고정 |
| `QWEN_QUANTIZATION` | `4bit` | `4bit` 또는 BF16용 `none` |
| `QWEN_MAX_INPUT_TOKENS` | `4096` | 이미지 포함 입력 한도 |
| `QWEN_GPU_DEVICE` | `0` | 호스트 NVIDIA GPU 인덱스 |
| `QWEN_PORT` | `7110` | 호스트 API 포트 |

직접 받은 모델은 `docker/workspace/model/qwen/<model>`에 놓고 `QWEN_MODEL_ID=/workspace/model/qwen/<model>`로 지정합니다. 캐시는 `docker/huggingface`에 저장됩니다. Qwen 서비스는 `reasoning` 프로필에 있어 기본 Compose 전체 기동에는 포함되지 않으며 `up qwen`처럼 명시적으로 선택하면 시작됩니다. Jetson/ARM64 이미지는 아직 제공하지 않습니다. Operate·Record의 모델 목록에는 원본 0.8B·2B·4B와 맵에 맞게 학습된 모델이 표시됩니다.

## 테스트

모델 다운로드 없이 API·동시 요청·입력 검증·출력 처리 테스트:

```bash
cd ~/sw_workspace/mission_reasoning/cyclo_intelligence
docker run --rm --entrypoint python \
  -v "$PWD/cyclo_brain/policy/qwen/tests:/tests:ro" \
  -v "$PWD/cyclo_brain/policy/qwen/qwen_engine:/app/qwen_engine:ro" \
  mission-reasoning/qwen:0.1.0-amd64 -m unittest discover -s /tests -v
```

검증 기록 (RTX 5090, 2026-09-08):

- Docker 이미지 빌드, Qwen 단위 테스트 9개, Docker 회귀 테스트 21개 통과.
- 일반 모드 한국어 응답과 합성 단색 이미지 인식 성공. 간단한 요청은 약 0.8~1.2초였으며 실제 카메라·장문 입력의 성능을 보장하는 수치는 아닙니다.
- 사람 복귀 시나리오에서 `stop` 판단 텍스트 생성 확인. 구조화된 미션 실행 계약이나 실행 검증은 아직 없습니다.
- 가중치 로드 후 PyTorch 할당 약 7,527 MiB, 예약 약 10,210 MiB 확인. CUDA 컨텍스트 및 다른 프로그램 메모리는 별도입니다. 모델 해제 후 API 대기 상태 복귀 확인.
- 같은 시나리오의 thinking 모드는 2,048토큰 안에 최종 답변을 완료하지 못했습니다. 현재는 일반 모드로 시작하고 thinking 모드는 별도 실험용으로 사용하세요.

구현 참고: [공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-9B), [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [PyTorch CUDA 버전](https://pytorch.org/get-started/previous-versions/).


## 4B 비교 실험 (2026-09-09)

RTX 5090에서 동일한 헤드 카메라 이미지와 기존 한국어 2~4문장 프롬프트를
사용했습니다. 두 모델 모두 NF4 4bit, Thinking OFF, 최대 출력 384토큰이며,
모델별 워밍업 1회를 제외한 5회를 비교했습니다.

| 항목 | Qwen3.5-9B | Qwen3.5-4B |
| --- | ---: | ---: |
| 평균 처리 시간 | 2.433초 | 1.912초 |
| 최소~최대 | 1.917~3.486초 | 1.629~2.146초 |
| 평균 출력 토큰 | 78.4 | 64.8 |
| 유효 출력 토큰/초 | 32.22 | 33.89 |
| 추론 후 PyTorch 할당 메모리 | 7,538 MiB | 3,168 MiB |
| 추론 후 PyTorch 예약 메모리 | 10,232 MiB | 4,744 MiB |

평균 지연은 약 21% 줄었지만 출력 길이도 줄었습니다. 토큰/초 개선은 약 5%로,
모델 축소만으로 응답 시간이 절반이 되지는 않았습니다. 메모리는 CUDA 컨텍스트와
다른 프로세스를 제외한 Qwen 내부 측정치입니다. GR00T와 Gazebo는 유지했지만,
GR00T 추론 부하를 통제한 동시 실행 벤치마크는 아닙니다.
4B 답변에 혼합 표기(예: `파lett truck`)와 물체 이름 혼동이 나타났습니다.
한 장의 이미지로 미션 판단 정확도를 검증한 결과는 아닙니다.

현재 실험 컨테이너는 4B로 교체했습니다. UI Stop/Start는 같은 컨테이너 설정을
유지합니다. Compose 기본값은 9B이므로 컨테이너를 다시 생성할 때도 4B를
사용하려면 `cyclo_intelligence` 폴더에서 다음처럼 지정합니다.

```bash
QWEN_MODEL_ID=Qwen/Qwen3.5-4B docker/container.sh start-qwen
```

9B 복귀는 위 명령의 모델 ID를 `Qwen/Qwen3.5-9B`로 변경합니다.
모델 교체 전 UI에서 Stop observation을 누르고, 교체 후 Start observation으로
다시 로드합니다.

[측정 원본 및 출력](benchmarks/2026-09-09-9b-vs-4b.json)


## Collect 미션 판단 미리보기

- Collect와 Operate는 같은 Qwen 서버(현재 로드된 LoRA 포함), 현재 맵의 Design 미션 목록, 영어 JSON 프롬프트와 생성 설정(`do_sample=false`, `max_new_tokens=192`)을 사용합니다.
- Collect는 **Idle / 실행 이력 없음**을 가정합니다. 진행 중인 미션이나 완료 이력을 가진 Operate 세션과는 입력 조건이 다르므로 결과가 달라질 수 있습니다. 활성 에피소드에서 Record instruction으로 기록한 지시는 다음 판단에 포함됩니다.
- 좌측 라이브 카메라는 영상 비율에 맞춰 작게 표시하고, 우측에는 Decision, Mission, 판단 이유만 표시합니다. 모델·어댑터·입력 조건은 저장된 예측 데이터에 보존합니다. `Save analyzed frame`은 상단 캡처·추론 버튼 옆에 있습니다.
- 지도는 필요하지만 미리보기에 로컬라이제이션은 필수가 아닙니다. Collect 응답은 항상 `preview_only=true`, `action_allowed=false`이며 미션 실행·취소를 호출하지 않습니다. Operate의 로컬라이제이션 및 실행 검증은 그대로 유지됩니다.
- 활성 에피소드에서 `Save analyzed frame`을 누르면 해당 이미지·입력·예측을 함께 저장합니다. 이후 Annotation에서 정답 미션과 이유를 검토하고 `Save label`을 눌러야 학습 라벨로 승인됩니다. 지도나 미션 정의가 다른 에피소드에는 분석 결과를 저장할 수 없습니다.
- Qwen 없이 `Capture frame`으로 이미지만 취득하는 기능도 그대로 사용할 수 있습니다.

## Collect 데이터로 LoRA 학습 및 Gazebo 미션 테스트

[학습 방법, 실행 명령, 실제 결과와 확인된 오판](docs/LORA_SMOKE_TEST_KO.md)을 참고하세요.
현재 한 장의 승인 데이터로 연결 테스트를 마쳤으며, 막대기가 없는 장면에서도 시작하는 오판이 있어
모델 정확도는 추가 데이터로 검증해야 합니다.

[Operate 1차 구현 및 사용법](docs/OPERATE_KO.md): Run에서 Qwen 판단과 기존 미션 실행기를 연결하고 Stop에서 함께 중단합니다.

### Frame annotation workflow

See [Annotate → Reasoning workflow](docs/ANNOTATION_WORKFLOW_KO.md) for sampled human labels, named Qwen fine-tuning, and draft generation.

### Record attention preview

Record's Observation preview sends `attention_preview: true` to `/generate`.
The engine hooks the last full-attention LLM layer once during prompt prefill,
extracts the last prompt query's attention over image keys (softmax per head,
then head mean), and returns an `attention_preview` JPEG alongside the decision.
The hook is removed even if generation fails; decode steps are ignored.
The overlay uses the same fixed density scale and TURBO palette as
`reasoning_tuning/video.py` (density = probability × image token count / 4,
50% image / 50% heat). This is a language-conditioned image attention diagnostic,
not a segmentation mask or confidence score. It uses the analyzed frame, not the
newer live stream. Operate requests keep preview disabled by default.

Mission reasoning requests set `json_object: true`. Generation stops after the first complete JSON object, before a synthetic next conversation turn can be produced. General text generation keeps its existing EOS behavior. Schema, active mission ID, freshness and execution-context validation still determine whether a decision can control the robot.
