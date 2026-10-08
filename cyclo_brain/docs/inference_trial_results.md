# Inference Trial Results

The Inference panel includes Success and Fail buttons for manual evaluation
only in Real Robot Deploy mode. Simulation mode does not mount this panel or
fetch its history.
Each click appends one trial to the selected Policy Path's history when idle,
or the matching loaded model during inference. A selected/loaded mismatch
disables labeling during inference. This does not start, stop, clear,
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

Supervisor API stores `/workspace/model/.../<selected model>/evaluation_manifest.json`
inside the selected model directory. The JSON is one evaluation object with
top-level `schema_version`, `model`, `context`, and `trials`. It does not track
an evaluation UUID or upload status. Older files with `evaluation_id` or
`uploaded` lose those fields on their next write; existing trials are retained.
Trial IDs are sequential integers starting at 1 and are renumbered after deletion.
New `recorded_at` and `updated_at` timestamps use Korea Standard Time with an
explicit `+09:00` offset. Existing UTC timestamps are preserved; success rate
is computed from trials.
The selected folder must exist
under `/workspace/model`. The local model path is only a lookup hint, not a
portable model version ID. When a readable `training_manifest.json` exists,
only its model ID, run ID, source artifact, and dataset repo ID are copied.
Missing manifest fields are omitted; labels still work without a manifest.
The dataset repo ID describes training provenance, not the evaluated task.
Uploading to a model registry requires an explicit version association.

`evaluation_manifest.json` is mutable and is not listed in a training artifact's
`SHA256SUMS`. Artifact inventory verification will reject the extra file if this
directory is reused directly as an immutable training package. Packaging and
publishing must exclude the evaluation file or handle it as a separate artifact;
do not rewrite the training manifest or `SHA256SUMS` when adding labels.

One model folder can hold one robot evaluation. A different robot type is rejected
instead of mixing results; use a distinct model folder for separate robot evaluations.
Existing single-entry `evaluations` wrappers are converted on the next write,
including UUID trial IDs. Multi-entry wrappers require manual separation and are
never overwritten automatically. Changing
weights at the same path cannot be detected without version metadata, so use a
different path for a different checkpoint. Model paths resolve to their canonical
directory under `/workspace/model`. These results are human labels, not automatic robot
completion checks.

Existing external evaluation JSONs, CSVs, and `files.json` are retained under
`/workspace/inference_results`. A read of a robot's old evaluation imports it
in memory; the next write saves it in the model's `evaluation_manifest.json`.
Old CSV rows have no robot identity and are available through requests without
`robot_type`. New UI labels use the selected robot type and do not silently
assign old CSV rows to it. Back up both locations until migration is complete.

The nginx `/api/` proxy exposes Supervisor's `/try-results` GET, POST, PATCH,
and DELETE endpoints. The response still exposes numbered rows to the UI;
`download=true` exports CSV from the stored JSON. Writes use the single
Supervisor process's lock and atomic file replacement. Multiple Supervisor
processes must not share the same results directory.

## Applying

In a source-mounted Cyclo installation, `/opt/supervisor_api` links to the
mounted repository. Restart `supervisor_api` after active operations finish to
load backend changes; an image rebuild is not required. Run
`./docker/container.sh build-ui` and refresh the browser for frontend changes.
No ROS interface or model Worker changes are required. The new UI's revision
checks require the updated Supervisor API.

The import intentionally excludes remote inference-recording, rosbag recorder,
joystick, URDF, and model-code changes.
