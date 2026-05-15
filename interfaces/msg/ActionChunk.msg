# Legacy/optional raw action chunk message.
# The current policy runtime uses interfaces/srv/EngineCommand for Main <->
# Engine action_list responses. Keep this message only for tools that still
# consume chunk-shaped previews or old recordings.

# Monotonic id for consumers that need to spot drops or out-of-order arrivals.
uint64 seq_id

# chunk_size × action_dim = len(data). data is row-major.
int32 chunk_size
int32 action_dim
float64[] data
