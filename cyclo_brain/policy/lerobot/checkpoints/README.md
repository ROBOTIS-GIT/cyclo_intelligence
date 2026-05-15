# LeRobot Checkpoints

This folder is bind-mounted into the LeRobot container at:

```text
/policy_checkpoints/lerobot
```

Put user-trained LeRobot policy checkpoints here when you want to test them
without copying files into the container manually.

Example host layout:

```text
cyclo_brain/policy/lerobot/checkpoints/my_act_model/
└── checkpoints/
    └── last/
        └── pretrained_model/
            ├── config.json
            ├── model.safetensors
            ├── policy_preprocessor.json
            └── policy_postprocessor.json
```

Use this container path when calling `InferenceCommand.LOAD`:

```text
/policy_checkpoints/lerobot/my_act_model/checkpoints/last
```

`model.safetensors` and other model artifacts are intentionally ignored by git.
