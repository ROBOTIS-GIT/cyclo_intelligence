string task_num
string task_name
string task_type
string[] task_instruction
string[] subtask_instruction
string policy_path
string[] tags
bool record_inference_mode
uint16 control_hz
uint16 inference_hz
float64 chunk_align_window_s

# Opt-in: when true, the recorder writes a README.md with the ROBOTIS
# Apache 2.0 license header into the task folder. Default (false) writes
# a minimal attribution-only README and leaves the license unspecified —
# the dataset is the user's, not ROBOTIS', so external users shouldn't
# be auto-stamped with a copyright they don't own. ROBOTIS internal data
# captures tick this on so the license rides through to HF Hub.
bool include_robotis_license

# Inference backend ("lerobot" / "groot" / ...). Empty falls back to
# orchestrator._determine_service_prefix step 2 (config.json type
# detection) for backward compatibility with old UI builds.
string service_type

# Inference output mode. "simulation" runs policy inference and publishes
# preview-only trajectories for the 3D viewer. "robot" permits the policy
# container Main runtime to publish robot command topics.
string inference_mode

# Action chunk request scheduling. "async" requests the next chunk while the
# current buffer is still executing. "sync" waits until the buffer is empty.
string action_request_mode

# Policy acceleration mode. Empty and "pytorch" mean the normal model path.
# "tensorrt_dit" enables GR00T DiT-only TensorRT acceleration when supported.
string acceleration_mode

# Optional TensorRT engine file/directory. Empty lets the backend pick the
# model-local default, e.g. <model_path>/dit_model_bf16.trt for GR00T DiT-only.
string acceleration_engine_path

# Optional PI RLT side policy. Enabling this only preloads the bundle during
# inference LOAD; action_policy_mode decides which candidate is executed.
bool rlt_enabled
string rlt_bundle_path

# Runtime action router selection. "base" executes the frozen VLA reference
# policy and "rlt" executes the RLT Action MLP. The runtime applies changes at
# an action-buffer boundary, never in the middle of a committed chunk.
string action_policy_mode

# Explicit operator acknowledgement for routing an unqualified RLT action to a
# physical robot. This is consumed only by SET_ACTION_POLICY; simulation and
# the base action path do not require the override.
bool rlt_robot_override
