# Data Conversion Pipeline

A 3-stage pipeline for converting ROSbag2 recorded data into a LeRobot training dataset.

## Overview

```
ROSbag2 (MCAP)
     │
     ▼
┌─────────────────────────────────────┐
│  Stage 0: Merge (optional)          │  Merge multiple datasets into one (symlink)
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Stage 1: ROSbag → MP4              │  RosbagToMp4Converter
│  Image topics → MP4 video extraction│  convert_rosbag2mp4.py
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Stage 2: MP4 → LeRobot v2.1       │  RosbagToLerobotConverter
│  Joint data + MP4 → Parquet/MP4    │  rosbag_to_lerobot_converter.py
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Stage 3: v2.1 → LeRobot v3.0      │  LeRobot built-in converter
│  File structure reorganization      │  (docker exec lerobot_server)
└─────────────────────────────────────┘
     │
     ▼
LeRobot v3.0 Dataset (ready for training)
```

## Directory Structure

```
/workspace/rosbag2/
├── {robot}_{task}/                          # Original + Stage 1 output
│   ├── 0/                                   # Original episode
│   │   ├── 0_0.mcap                         #   ROSbag2 MCAP file
│   │   ├── episode_info.json                #   Episode metadata
│   │   ├── metadata.yaml                    #   ROS2 metadata
│   │   └── robot.urdf                       #   Robot URDF
│   ├── 0_converted/                         # Stage 1 output
│   │   ├── episode.mcap                     #   MCAP with images removed
│   │   ├── cam_head_left.mp4                #   MP4 per camera
│   │   ├── cam_head_right.mp4
│   │   ├── cam_wrist_left.mp4
│   │   ├── cam_wrist_right.mp4
│   │   ├── video_stats.json                 #   Pre-computed video statistics
│   │   ├── episode_info.json                #   Includes dropped frame info
│   │   ├── robot.urdf
│   │   └── meshes/
│   ├── 1/
│   ├── 1_converted/
│   └── ...
│
├── {robot}_{task}_lerobot_v21/              # Stage 2 output
│   ├── data/
│   │   └── chunk-000/
│   │       ├── episode_000000.parquet       #   Parquet per episode
│   │       ├── episode_000001.parquet
│   │       └── ...
│   ├── meta/
│   │   ├── info.json                        #   Dataset metadata
│   │   ├── episodes.jsonl                   #   Episode list
│   │   ├── episodes_stats.jsonl             #   Per-episode statistics
│   │   └── tasks.jsonl                      #   Task list
│   └── videos/
│       └── chunk-000/
│           ├── observation.images.cam_head_left/
│           │   ├── episode_000000.mp4
│           │   └── ...
│           ├── observation.images.cam_head_right/
│           ├── observation.images.cam_wrist_left/
│           └── observation.images.cam_wrist_right/
│
└── {robot}_{task}_lerobot_v30/              # Stage 3 output (final)
    ├── data/
    │   └── chunk-000/
    │       └── file-000.parquet             #   Multiple episodes consolidated
    ├── meta/
    │   ├── info.json
    │   ├── stats.json                       #   Overall statistics
    │   ├── tasks.parquet
    │   └── episodes/
    │       └── chunk-000/
    │           └── file-000.parquet         #   Episode metadata (Parquet)
    └── videos/
        └── {camera_key}/
            └── chunk-000/
                └── file-000.mp4             #   Concatenated episode MP4
```

---

## Stage 0: Merge (Optional)

Merges multiple dataset folders into one. Only runs when "Merge & Convert" mode is selected in the UI.

| Item | Description |
|------|-------------|
| Class | `Mp4ConversionWorker._merge_episodes()` |
| File | `mp4_conversion_worker.py` |
| Input | List of source folders (e.g., `task_A/`, `task_B/`) |
| Output | Merged folder (episodes linked via symlink) |
| Progress | 0% ~ 5% |

### Behavior
1. Create output folder
2. Iterate through source folders and collect numerically named directories (episodes)
3. Reassign episode numbers sequentially starting from 0
4. Link each episode via symlink (no data copying)

### Example
```
source_a/ (ep 0-6) + source_b/ (ep 0-7)
    ↓
merged_output/
  0 -> /workspace/rosbag2/source_a/0   (symlink)
  1 -> /workspace/rosbag2/source_a/1
  ...
  7 -> /workspace/rosbag2/source_b/0
  ...
  14 -> /workspace/rosbag2/source_b/7
```

---

## Stage 1: ROSbag to MP4

Extracts image topics from ROSbag2 MCAP as MP4 videos and generates an MCAP with images removed.

| Item | Description |
|------|-------------|
| Class | `RosbagToMp4Converter` |
| File | `convert_rosbag2mp4.py` |
| Input | Episode directories (`0/`, `1/`, ...) |
| Output | `{ep}_converted/` directories |
| Progress | Single mode: 0% ~ 33% / Merge mode: 5% ~ 35% |

### Processing Steps

#### Step 1: Frame Extraction and Matching
- Reads compressed image topics and camera_info topics from the MCAP file
- Matches image frames and camera_info by timestamp for each of the 4 cameras
- Aligns frame counts across cameras (matches to the minimum frame count)
- Detects dropped frames and interpolates (duplicates previous frame)

#### Step 2: MP4 Encoding
- Encodes raw BGR frames to MP4 using ffmpeg
- Attempts hardware encoder first (h264_nvenc, etc.), falls back to libx264 software encoding on failure
- Parallel encoding per camera using ThreadPoolExecutor (up to 4 workers)

#### Step 2.5: Pre-computing Video Statistics
- Samples 100 frames from in-memory frames
- Computes per-RGB-channel mean, std, min, max (normalized to 0~1)
- Saves to `video_stats.json`
- Allows Stage 2 to reuse statistics without MP4 decoding

#### Step 3: MCAP Regeneration
- Removes image topics from the original MCAP
- Preserves only matched camera_info timestamps
- Applies timestamp smoothing (adjusts 68~71ms intervals to 67~68ms)

#### Step 4: Metadata File Copy and Drop Info Recording
- Copies `episode_info.json`, `metadata.yaml`, `robot.urdf`
- Adds `dropped_frames` information to `episode_info.json`

### Key Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `fps` | 15 | Target frame rate |
| `use_hardware_encoding` | true | Whether to attempt GPU encoding |
| `enable_timestamp_smoothing` | true | Enable timestamp smoothing |
| `trim_start_sec` | 0.5 | Trim from start (seconds) |
| `trim_end_sec` | 0.0 | Trim from end (seconds) |

---

## Stage 2: MP4 to LeRobot v2.1

Converts Stage 1 output (MP4 + MCAP) into LeRobot v2.1 dataset format.

| Item | Description |
|------|-------------|
| Class | `RosbagToLerobotConverter` |
| File | `rosbag_to_lerobot_converter.py` |
| Input | `{ep}_converted/` directories |
| Output | `{dataset}_lerobot_v21/` directory |
| Progress | Single mode: 33% ~ 66% / Merge mode: 35% ~ 68% |

### Processing Steps

#### Episode Parsing (per episode)
1. Read joint_states, cmd_vel, odometry topics from MCAP
2. Sort/merge per-topic joint data according to `joint_order` configuration
3. Separate observation.state (follower) and action (leader)
4. Trim parquet row count to match video frame count (1:1 matching)

#### Dataset Writing (per episode)
1. **Parquet file**: timestamp, frame index, episode index, observation.state, action
2. **MP4 copy**: Copy Stage 1 MP4s to `videos/chunk-000/observation.images.{cam}/` path (no re-encoding)
3. **Episode metadata**: Append episode info to `episodes.jsonl`
4. **Episode statistics**: Append statistics to `episodes_stats.jsonl`

#### Video Statistics Computation
- Loads and uses `video_stats.json` (pre-computed in Stage 1) if available
- Otherwise, samples 100 frames from MP4 and computes directly (fallback)
- Per-channel RGB mean/std/min/max (normalized to 0~1)

#### Final Metadata Files
- `info.json`: Overall dataset information (number of episodes, number of frames, features, splits, etc.)
- `tasks.jsonl`: Task index-to-name mapping

### Parquet Schema
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float32 | Relative time from episode start (seconds) |
| `frame_index` | int64 | Frame index within the episode |
| `episode_index` | int64 | Episode number |
| `index` | int64 | Global index across the entire dataset |
| `task_index` | int64 | Task index |
| `observation.state` | list\<float32\>[N] | Follower joint state |
| `action` | list\<float32\>[N] | Leader joint command |

---

## Stage 3: LeRobot v2.1 to v3.0

Converts from v2.1 to v3.0 format using LeRobot's built-in converter.

| Item | Description |
|------|-------------|
| Execution | subprocess via `docker exec lerobot_server` |
| Script | `lerobot.datasets.v30.convert_dataset_v21_to_v30` |
| Input | `{dataset}_lerobot_v21/` |
| Output | `{dataset}_lerobot_v30/` |
| Progress | Single mode: 66% ~ 100% / Merge mode: 68% ~ 100% |

### Differences Between v2.1 and v3.0

| Item | v2.1 | v3.0 |
|------|------|------|
| Data files | 1 Parquet per episode | Multiple episodes consolidated into a single Parquet |
| Episode metadata | `episodes.jsonl` (JSONL) | `episodes/chunk-000/file-000.parquet` (Parquet) |
| Tasks | `tasks.jsonl` (JSONL) | `tasks.parquet` (Parquet) |
| Statistics | `episodes_stats.jsonl` (per-episode) | `stats.json` (overall consolidated) |
| Videos | MP4 per episode | Concatenated episode MP4 |

### Post-Conversion Folder Cleanup
The LeRobot converter operates in-place:
1. Original v2.1 directory is moved to `{name}_lerobot_v21_old`
2. New v3.0 data is created in the original v2.1 directory location
3. Worker swaps names: v3.0 → `_lerobot_v30`, old → `_lerobot_v21` restored

---

## Orchestration

`Mp4ConversionWorker` (`mp4_conversion_worker.py`) manages the entire pipeline.

### Execution Flow
```
orchestrator.py (ROS2 service)
  └─ CONVERT_MP4 command received
       └─ Mp4ConversionWorker (multiprocessing.Process)
            ├─ Stage 0: _merge_episodes()        [merge mode only]
            ├─ Stage 1: _convert_dataset()        [RosbagToMp4Converter]
            ├─ Stage 2: _convert_to_lerobot_v21() [RosbagToLerobotConverter]
            └─ Stage 3: _convert_to_lerobot_v30() [docker exec lerobot_server]
```

### Progress

| Stage | Single Mode | Merge Mode |
|-------|-------------|------------|
| Stage 0 (Merge) | - | 0% ~ 5% |
| Stage 1 (MP4) | 0% ~ 33% | 5% ~ 35% |
| Stage 2 (v2.1) | 33% ~ 66% | 35% ~ 68% |
| Stage 3 (v3.0) | 66% ~ 100% | 68% ~ 100% |

### Error Handling
- Each Stage stops with an error message on failure
- Error format: `[Stage N/3 {name}] {message}`
- Merge mode: Fails if source folder does not exist or output folder already exists

---

## Files

| File | Role |
|------|------|
| `mp4_conversion_worker.py` | Pipeline orchestration, Stage 0 (Merge) |
| `convert_rosbag2mp4.py` | Stage 1 - RosbagToMp4Converter |
| `rosbag_to_lerobot_converter.py` | Stage 2 - RosbagToLerobotConverter |
| `rosbag_to_lerobot_v30_converter.py` | Stage 3 helper (RosbagToLerobotV30Converter) |
| `bag_reader.py` | MCAP file reading utility |
| `metadata_manager.py` | robot_config.yaml parsing, trim/exclude management |
| `video_metadata_extractor.py` | Video metadata extraction |
| `progress_tracker.py` | Progress tracking |

---

## Docker Containers

| Container | Role |
|-----------|------|
| `orchestrator` | Runs Stage 0, 1, 2 (ROS2 environment) |
| `lerobot_server` | Runs Stage 3 (LeRobot environment, shares `/workspace`) |

Both containers share the `/workspace` volume, so Stage 2 output is directly accessible from Stage 3.
