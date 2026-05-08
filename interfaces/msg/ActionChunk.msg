# Raw action chunk published by Process A (inference_server.py) over Zenoh to
# Process B (control_publisher.py) inside the same policy container. ROS2 srv
# replaced this path in Step 4 (§5.5) because it let obs be read directly by
# Process A instead of round-tripping through orchestrator as a request/reply.

# Monotonic trigger id (forwarded from the run_inference trigger) so Process B
# can spot dropped chunks and log out-of-order arrivals.
uint64 seq_id

# chunk_size × action_dim = len(data). data is row-major.
int32 chunk_size
int32 action_dim
float64[] data
