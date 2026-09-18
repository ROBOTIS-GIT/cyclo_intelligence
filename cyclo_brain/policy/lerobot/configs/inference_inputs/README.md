# Cyclo 추가 전처리 설정

이 YAML은 **모델에 전달하기 전에 Cyclo가 추가로 수행할 작업만** 설정합니다.
모델 내부 처리와 저장된 LeRobot processor는 변경하지 않습니다.
설정은 Clear 후 LOAD할 때 읽습니다. 실행 중 파일을 수정해도 즉시 적용되지 않습니다.
같은 모델 종류의 모든 체크포인트가 YAML 하나를 공유합니다.

## 추가 처리 없음

```yaml
preprocessing: identity
```

`identity`는 추가 공간 변환이나 custom 처리가 없다는 뜻입니다.
토픽 디코딩, robot config의 카메라 회전, RGB float32 / 255 및 BCHW 포장,
device 이동과 저장된 processor 연결은 Python adapter가 계속 담당합니다.
모델 config에서 관측 이력이나 추가 resize를 추측하지 않습니다.
기존 state 순서와 chunk padding/truncation, step 차원 검사도 유지합니다.

## 이미지 변환: 위에서 아래로 실행

```yaml
preprocessing:
  images:
    - center_crop:
        size: [400, 640]
        backend: opencv
    - resize:
        size: [224, 224]
        backend: opencv
        interpolation: bilinear
```

이 예시는 가운데 crop 후 resize합니다. 크기는 `[높이, 너비]`입니다.
`resize`는 비율 보존 없이 지정 크기로 맞춥니다. 비율 보존과 패딩이 필요하면
`letterbox`를 사용합니다. `size: checkpoint`는 해당 카메라의 체크포인트 크기만
참조하며 학습 시 보간법을 알아내지는 않습니다.

지원 연산은 `resize`, `center_crop`, `letterbox`입니다. 이미지 원본보다 큰
crop은 실행 오류입니다. 연산마다 `backend: opencv` 또는 `torch`를 명시합니다.
보간법은 `nearest`, `bilinear`, `bicubic`, `area`이며 Torch의 bilinear/bicubic에만
`antialias: true`를 사용할 수 있습니다. 기본 antialias는 false입니다.
Letterbox에는 `placement: center` 또는 `top_left`, `fill: 0`~`255`를 지정할 수 있습니다.

OpenCV는 uint8 이미지에, Torch는 float32 / 255 tensor에 적용합니다.
OpenCV 다음 Torch는 가능하지만, Torch 다음 OpenCV는 암묵적 양자화가 필요하므로
LOAD에서 거부합니다. 순서를 자동으로 바꾸지 않습니다.

## 카메라별 설정

```yaml
preprocessing:
  images:
    - resize:
        size: checkpoint
        backend: opencv
        interpolation: bilinear
  cameras:
    observation.images.cam_left_wrist: identity
```

`cameras` 키는 체크포인트에 실제 존재하는 이미지 입력 키입니다.
카메라별 설정은 공통 `images`를 **대체**합니다. 뒤에 추가하지 않습니다.
`identity` 또는 빈 목록 `[]`은 그 카메라의 추가 공간 변환 없음입니다.
위 카메라 이름은 예시이며 존재하지 않는 키는 LOAD 오류입니다.

## 모델별 Python 처리

```yaml
preprocessing:
  custom:
    handler: previous_image_features
    options:
      combine: concat
```

위 handler 이름은 **설명용 예시이며 기본 제공 모듈이 아닙니다**.
모델 adapter의 `input_handlers`에 구현을 등록한 뒤에만 사용할 수 있습니다.
등록만 해서는 실행되지 않고 YAML에서 선택해야 합니다. 임의 Python 파일 경로나
import 문은 허용하지 않습니다. 옵션은 선택한 handler가 검증합니다.

Python 구현이 encoder 호출, 입력 연결, history의 의미, 초기값, 메모리 확정 조건,
다음 추론 조건을 정의합니다. 내부 공통 그래프와 실행 피드백을 재사용하며
Worker가 미래의 command를 기다리는 별도 루프를 만들지 않습니다.
Stop/Clear, instruction 및 generation 변경 시 기존 초기화 규칙을 따릅니다.
모델 내부에서 이미 처리하는 기능은 Cyclo에 다시 구현할 필요가 없습니다.

## 기본값과 검증 범위

- Diffusion: 기존 Cyclo OpenCV bilinear/checkpoint 크기 호환값입니다. 학습 기법을 보장하지 않습니다.
- Multi-Task DiT: 기존 테스트 체크포인트의 Torch 224x224 bilinear/antialias 설정을 유지합니다.
- 나머지 현재 기본 파일: `preprocessing: identity`입니다.
- 실행 API와 모델 내부 history는 기존 adapter가 계속 담당합니다.
- `sources`, `nodes`, `outputs`, `"*"`는 사용자 YAML에서 제거했습니다.
  복잡한 입력 그래프는 등록된 Python handler 내부에만 작성합니다.

개발자는 [adapter 가이드](../../lerobot_engine/adapters/README.md)를 참고하세요.
