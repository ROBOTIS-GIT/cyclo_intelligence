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
float32 encoding_progress       # [%]
float32 used_storage_size       # [GB]
float32 total_storage_size      # [GB]
float32 used_cpu                # [%]
float32 used_ram_size           # [GB]
float32 total_ram_size          # [GB]
