# Recording Alongside Inference

Recording and inference share the Operate workspace but have independent
lifecycles. Record and Inference tabs select settings, not execution modes.

## Workflow

- Configure the task number, task name, instruction, and optional subtasks in
  Record. Prepare the session before using leader recording buttons.
- Record starts the same recorder whether inference is unloaded, running,
  paused (including teleop intervention), or cleared.
- Save/Discard and the leader right/left buttons operate the recording session.
  Existing next-subtask behavior is preserved.
- Inference Start, Stop, Clear, model changes, and tab switches do not save,
  discard, or relabel an ongoing recording. Recording commands do not start,
  pause, or unload inference.
- Common camera, robot, and topic views stay mounted across tab switches.
  Shortcuts apply only to the active tab; typing into a field does not run them.

## Ownership and Storage

The Record configuration owns storage. Existing naming is unchanged:

```text
/workspace/rosbag2/Task_<task_num>_<task_name>_MCAP/
  0/
  1/
```

Segmented recordings use the existing segment capture and full-episode archival
pipeline. The inference policy path is not used to choose the recording directory;
new `YYMMDD_HHMM_model` directories are no longer allocated by the workspace.
Existing data is not renamed or deleted.

The backend captures the Record task when recording starts and retains it while
segments are pending. Save/Discard use that captured identity, even after a model
change or another UI edits its draft. Robot/topic changes are rejected while
recording; selecting the same robot does not rebuild the active pipeline.
A failed command does not acknowledge completion or clear the captured task.
Indexed discard requests for a different task are rejected.
Archival acknowledgement is not completion: the task stays owned until the
recorder reports completion, or remains available for retry after an error.
Concurrent commands for the same subsystem receive a busy error instead of
occupying ROS executor threads. A pending recording command does not block
inference commands or conversion submissions.

Recording commands carry a UUID receipt ID. An acknowledgement timeout triggers
`RecordingCommand.GET_RESULT`, never a repeat of the original Save/Discard.
If the result is still unknown, the caller returns an error after bounded waits
and retains pending ownership. The next Record-side request queries that receipt
again, applies a confirmed result once, and asks the operator to review the status
before issuing a new command. Inference remains independent while this happens.
Archive completion is separately queryable, including when its status event is
lost; terminal events carry the originating receipt ID to reject stale events.

The recorder keeps 256 recent receipts in memory, retaining pending commands and
active archives. This is not a durable exactly-once protocol. After a recorder
restart or receipt eviction, an unknown result remains blocked: inspect the
recorded data and recover the session explicitly, rather than retrying a destructive
operation under a new ID. A failed acknowledgement is not proof that no I/O occurred.

Legacy inference recording command IDs remain usable with a prepared Record
task. They do not create a second recording destination. The old
`SET_INFERENCE_RECORD_FOLDER` command returns an error directing clients to Record
settings. Existing command numbers are preserved. `RecordingCommand` adds
`GET_RESULT`, `request_id`, `command_state`, and `archive_state` for receipt recovery.

## Applying and Verifying

Rebuild `interfaces`, `cyclo_data`, and `orchestrator` together, then restart their
processes in a maintenance window. The updated recording service interface is
not wire-compatible with old generated interfaces. Rebuild the UI as well.
Model weights do not change; Runtime/RobotClient improvements must be deployed
with their corresponding processes separately from the recording UI update.

Automated tests cover UI/leader routing, Record folder and metadata ownership,
independent inference Clear/Stop, rejected duplicate starts, failed save/discard
retry, and active robot/topic protection. Native-message integration tests use
temporary files, real DataManager/RecordingService code, and fake recorder I/O.
An isolated ROS service test checks that inference Stop completes while a
recording Save is pending. Browser tests use mocked ROS/API endpoints to check
tab persistence, independent controls, and Record task payloads.
They do not prove real camera recording, model inference, or physical robot safety;
those require a separate operator-controlled validation.
