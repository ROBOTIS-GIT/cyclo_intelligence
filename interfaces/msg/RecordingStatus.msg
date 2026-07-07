########################################
# Constants
########################################
# record_phase
uint8 READY = 0
uint8 RECORDING = 1
uint8 SAVING = 2
uint8 PAUSED = 3

TaskInfo task_info
string robot_type
uint8 record_phase
uint16 proceed_time             # [s]
uint16 current_episode_number
uint16 current_scenario_number
string current_task_instruction
uint16 current_subtask_index
uint16 subtask_count
string current_subtask_instruction
string[] subtask_instructions
uint16[] saved_subtask_indices
string[] recording_warnings
string[] camera_monitor_names
string[] camera_monitor_topics
float32[] camera_monitor_rates_hz
float32[] camera_monitor_baseline_hz
float32[] camera_monitor_seconds_since_last
uint8[] camera_monitor_status
float32[] camera_monitor_timestamp_skew_s
uint8[] camera_monitor_timestamp_status
float32 encoding_progress       # [%]
float32 used_storage_size       # [GB]
float32 total_storage_size      # [GB]
float32 used_cpu                # [%]
float32 used_ram_size           # [GB]
float32 total_ram_size          # [GB]
