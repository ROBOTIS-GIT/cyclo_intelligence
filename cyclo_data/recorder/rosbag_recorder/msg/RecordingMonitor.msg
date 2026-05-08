# Per-topic snapshot published while a recording episode is in progress.
# All arrays are parallel and share the same length: index i refers to one
# topic.

# Topic name (full path), e.g. "/robot/camera/cam_left_wrist/image_raw".
string[] topic_names

# Instantaneous rate measured over the last monitor tick (Hz).
float32[] rates_hz

# EMA of healthy rates so far. Frozen while the topic is judged stalled, so
# this still reflects the topic's "normal" speed when comparing.
float32[] baseline_hz

# Wall time elapsed since the last message arrived for this topic.
float32[] seconds_since_last

# Status code per topic:
#   0 = OK     (rate >= baseline * slow_ratio and recently received)
#   1 = SLOW   (rate < baseline * slow_ratio but not yet stalled)
#   2 = STALLED (no message within stall_window or rate < baseline * stall_ratio)
uint8[] status

# Aggregate counters across all topics since the current PREPARE.
uint64 total_received
uint64 total_written
