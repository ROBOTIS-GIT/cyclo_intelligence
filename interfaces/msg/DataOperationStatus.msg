########################################
# Constants — operation_type
########################################
uint8 OP_RECORDING = 0
uint8 OP_CONVERSION = 1
uint8 OP_HF = 2
uint8 OP_EDIT = 3

########################################
# Constants — status
########################################
uint8 IDLE = 0
uint8 RUNNING = 1
uint8 COMPLETED = 2
uint8 FAILED = 3
uint8 CANCELLED = 4

uint8 operation_type
uint8 status
# Async identifier (empty when not applicable — e.g., global recording state).
string job_id
float32 progress_percentage
# Current stage label, free-form (e.g., "encoding_mp4", "uploading_shard_3").
string stage
# Detail message / error text.
string message
