# Cyclo Additional Preprocessing Configuration

This YAML configures **only the additional operations Cyclo performs before passing inputs to the model**.
It does not change model-internal processing or saved LeRobot processors.
Configuration is read on LOAD after Clear. Editing a file during execution does not apply changes immediately.
All checkpoints of the same model type share one YAML file.

## No Additional Processing

```yaml
preprocessing: identity
```

`identity` means no additional spatial transforms or custom processing.
The Python adapter still handles topic decoding, camera rotation from the robot config,
RGB float32 / 255 conversion, BCHW packing, device transfer, and saved processor integration.
Cyclo does not infer observation history or additional resizing from the model config.
State/action channel order is resolved from the checkpoint's `cyclo_io_mapping.json`.
Without it, only exact legacy dimensions are accepted. Cyclo does not pad or truncate robot channels.

## Image Transforms: Executed Top to Bottom

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

This example applies a center crop followed by resizing. Sizes use `[height, width]`.
`resize` fits the specified size without preserving the aspect ratio. Use `letterbox`
to preserve the aspect ratio and add padding. `size: checkpoint` references only the
checkpoint dimensions for that camera; it does not determine the interpolation used during training.

Supported operations are `resize`, `center_crop`, and `letterbox`. A crop larger than
the source image fails at execution. Each operation must specify `backend: opencv` or `torch`.
Supported interpolation modes are `nearest`, `bilinear`, `bicubic`, and `area`.
`antialias: true` is available only for Torch bilinear/bicubic interpolation; the default is false.
Letterbox supports `placement: center` or `top_left` and `fill` values from `0` to `255`.

OpenCV operates on uint8 images; Torch operates on float32 / 255 tensors.
OpenCV followed by Torch is supported. Torch followed by OpenCV is rejected at LOAD
because it would require implicit quantization. Operations are never reordered automatically.

## Per-Camera Configuration

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

Keys under `cameras` must be image input keys present in the checkpoint.
Per-camera settings **replace** the shared `images` settings; they are not appended.
`identity` or an empty list `[]` means no additional spatial transforms for that camera.
The camera name above is an example. Unknown keys cause a LOAD error.

## Model-Specific Python Processing

```yaml
preprocessing:
  custom:
    handler: previous_image_features
    options:
      combine: concat
```

The handler name above is **an illustrative example, not a built-in module**.
Its implementation must first be registered in the model adapter's `input_handlers`.
Registration alone does not execute it; it must also be selected in YAML. Arbitrary
Python file paths and import statements are not allowed. The selected handler validates its options.

The Python implementation defines encoder calls, input connections, history semantics,
initial values, memory commit conditions, and prerequisites for the next inference request.
It reuses the shared internal graph and execution feedback without adding a Worker loop
that waits for future commands. Existing reset rules apply on Stop/Clear and instruction
or generation changes. Processing already handled inside the model need not be reimplemented in Cyclo.

## Defaults and Validation Scope

- Diffusion: preserves the existing Cyclo OpenCV bilinear/checkpoint-size compatibility defaults. These do not guarantee a match with training preprocessing.
- Multi-Task DiT: retains the Torch 224x224 bilinear/antialias settings of the existing test checkpoint.
- Other current default files use `preprocessing: identity`.
- Existing adapters continue to handle execution APIs and model-internal history.
- `sources`, `nodes`, `outputs`, and `"*"` have been removed from user-facing YAML.
  Complex input graphs are defined only inside registered Python handlers.

Developers should refer to the [adapter guide](../../lerobot_engine/adapters/README.md).
