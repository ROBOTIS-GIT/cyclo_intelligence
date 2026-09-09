########################################
# Constants
########################################
# inference_phase
uint8 READY = 0
uint8 LOADING = 1
uint8 INFERENCING = 2
uint8 PAUSED = 3
uint8 SYNCING = 4

string robot_type
uint8 inference_phase
string error

# Complete backend snapshot. Unknown is not equivalent to READY.
bool status_known
string runtime_state
string model_path
string policy_id
bool publish_to_robot
string source_id
uint64 sequence

# Editable settings, independent of the currently loaded policy and recording.
bool has_task_info
uint64 task_info_revision
TaskInfo task_info
