# Inference Recording Destinations

Inference recording uses the existing rosbag recorder. It is independent of
Try Results and does not add evaluation labels, preview videos, or shortcuts.

## Workflow

- In Real Robot Deploy, Record starts an episode while inference is running.
- Save finalizes the episode; Discard removes the current episode. Both remain
  available after inference Stop. Clear is blocked until Save or Discard succeeds.
- The folder button in the Recording row shows the current path. When no recording,
  Save, or Discard is in progress, use the existing file browser to select a folder,
  or choose Use new to allocate a folder on the next Record. Inference can remain
  running; changing the recording destination does not stop or clear the model.
- The first Record allocates `/workspace/rosbag2/Task_<UTC timestamp>_inference_MCAP`.
  Name collisions receive a numeric suffix. Later episodes reuse that folder.
- Pause, Clear, navigation, refresh, and other browser clients retain the
  backend selection. Policy ID/path or robot changes reset it. Backend restart
  also resets selection, without deleting data. Explicitly selecting a folder
  from the same robot allows episodes from different models in that folder.
- Existing right/left leader triggers use the same session and validation as UI
  commands. They are not new topic-based recording triggers.

## Ownership and Storage

`InferenceRecordingSession` owns the destination. Only the explicit
`SET_INFERENCE_RECORD_FOLDER=26` command changes the selected folder; automatic
settings updates and stale Record payloads cannot replace it. `TaskInfo.task_num`
is the session ID, with an empty ID meaning new-on-next-record. The central
InferenceStatus topic, not service responses, supplies the UI selection.

Selection permits only direct inference folders under the recording root.
Symlinks, robot mismatches, invalid metadata, and unrecognized formats are
rejected. Existing and incomplete episode slots are not overwritten. A failed
save retains recording ownership so Save or Discard can be retried; it never
silently changes destinations. Once source metadata is written, archive retries
reuse it. Video transcoding remains the existing background process.

Each new episode's `episode_info.json`, including the final archive, contains
an `inference` object with `policy_id`, `policy_path`, and `task_instruction`
captured from the active inference session. Older episodes are not rewritten.

## Applying and Verifying

Rebuild the Cyclo `interfaces` and `orchestrator` packages and the UI, then restart
Cyclo processes in a maintenance window. Python-only restart is insufficient for
the new generated service constant. No LeRobot policy/model changes are required.

Automated coverage includes folder validation/allocation, episode numbering,
partial archive failures, frozen model metadata, UI topic ownership, folder
selection, and Stop/Clear guards. Real recorder IO and robot operation require
separate testing; the unit tests use temporary directories and mocked transports.
