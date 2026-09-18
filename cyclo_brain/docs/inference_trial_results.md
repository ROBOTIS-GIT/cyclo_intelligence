# Inference Trial Results

The Inference panel includes Success and Fail buttons for manual evaluation
only in Real Robot Deploy mode. Simulation mode does not mount this panel or
fetch its history.
Each click appends one trial to the loaded model's history. If no model is
loaded, the selected Policy Path is used. This does not start, stop, clear,
or otherwise control inference, recording, or saved-pose return.

Expand Try Results to see timestamps, success counts, and success rate. Existing
results can be corrected, or selected trials deleted after confirmation. Like
the original remote implementation, deletion renumbers the remaining trials.
Reopening the panel or refreshing the browser fetches changes made by other
clients; there is no separate refresh button or per-browser polling.
Edits and deletions carry a history revision so an outdated client cannot modify
the wrong row after another client has changed the history. A conflict requires
refreshing and reselecting the intended rows, not an automatic retry.

## Storage

Supervisor API stores CSV files under `/workspace/inference_results`, normally
`docker/workspace/inference_results` on the host. `files.json` maps complete model
paths or repository IDs to timestamped CSV filenames. Keep it alongside the CSVs
when backing up or transferring results. Refreshing the UI or restarting Cyclo
does not erase results. Trailing slashes are ignored, but other path aliases are
not resolved to the same model. Replacing weights at the same path retains that
path's history; use a separate checkpoint path to evaluate a different model.

The CSV columns are `try`, `result`, `created_at`, `updated_at`, and `model_path`.
Times are stored in UTC and displayed in the browser's local timezone. These are
human labels, not automatic task completion checks or measurements of the
physical robot. Trials are grouped by model path, not by robot, instruction,
or inference session. Existing remote result data is not copied automatically.

The existing nginx `/api/` proxy exposes Supervisor's `/try-results` GET, POST,
PATCH, and DELETE endpoints. GET with `download=true` exports CSV. Writes use the
single Supervisor process's lock and atomic file replacement. Multiple
Supervisor processes must not share the same results directory.

## Applying

On a source-mounted installation, restart only `supervisor_api` after any active
container-management operation has completed, then run
`./docker/container.sh build-ui` and refresh the browser. No ROS interface,
Policy Runtime, model Worker, or image rebuild is required for this feature.
The new UI's revision checks require the updated Supervisor API.

The import intentionally excludes remote inference-recording, rosbag recorder,
joystick, URDF, and model-code changes.
