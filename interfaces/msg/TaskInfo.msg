string task_num
string task_name
string task_type
string[] task_instruction
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
