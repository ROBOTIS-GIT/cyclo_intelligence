"""Translate the existing LeRobot I/O mapping into a model-independent input spec.

This is the legacy latest-observation contract, not automatic inference of a
checkpoint's temporal or action-feedback requirements.
"""

from inference_context import InputAssembler, InputField, InputSpec, SampleQuery


def latest_input_plan(cameras, state_modalities, image_transform, state_transform, *, max_age_s=None,
                      policy_config=None, model_path=None):
    fields = []
    transforms = {"state": state_transform, "task": lambda values: [values[0] or ""]}
    for camera, key in cameras.items():
        transform = f"image:{camera}"
        fields.append(InputField(key, (SampleQuery(f"camera:{camera}", max_age_s=max_age_s),), transform))
        transforms[transform] = lambda values, cam=camera, target=key: image_transform(cam, target, values[0])
    fields.append(InputField(
        "observation.state",
        tuple(SampleQuery("sensor:odom" if name == "mobile" else f"joint:follower_{name}", max_age_s=max_age_s)
              for name in state_modalities),
        "state",
    ))
    fields.append(InputField("task", (SampleQuery("instruction"),), "task"))
    return InputAssembler(InputSpec(tuple(fields)), transforms)
