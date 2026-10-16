<!--
Copyright 2026 ROBOTIS CO., LTD.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Seongwoo Kim
-->

# Autonomy UI

These components belong to [Autonomy Canvas](../README.md). Shared map/runtime
composition lives in [mapWorkspace](../../mapWorkspace/README.md); Mission Canvas
has independent stage and scene renderers.

Autonomy Canvas exposes **Record / Train / Operate** as three top-level stages. Previous Annotate, Train, and Learn session links open the unified Train page.
Mission Canvas continues to own Navigation and Mission design/run tools; saved maps,
missions, observation plans, and datasets remain shared through the existing APIs.

| Stage | Responsibility |
| --- | --- |
| **Record** | Map, observation routes, episode capture, live camera, model preview, and teleop |
| **Train** | Episode/frame library, labels and masks, automatic generation, review, and final-model training |
| **Operate** | Autonomous patrol, scene decisions, mission execution and activity |

Train shares one episode library and frame editor. **Reasoning labels** and
**Object masks (optional)** use the same compact labeling controls:

- **Sample Labeling** is the default, without a mode tab. A single toolbar groups
  `Samples [count] [shuffle]`, Fine-tune, Auto-label, and the Review filter.
  Before sampling, all frames are available; after sampling, only the selected
  samples are shown. There is no separate frame filter. Progress beside the
  previous/next frame buttons counts the selected samples (`12/30 labeled`) or
  generated results (`42/239 reviewed`), according to the current mode.
- The **Fine-tune** icon on the sampling row trains on saved, approved manual
  samples across all saved episodes on the selected map. Unlabeled samples and
  approved auto-generated drafts are excluded. No naming/step form is needed.
- The **Auto-label** icon opens a completed-model picker, defaulting to the latest
  compatible model. Its map and target selectors support the selected episode or
  all compatible saved episodes on the chosen map. It fills missing labels and
  preserves training frames, manual labels, and approved results.
- Each model has a delete icon with inline confirmation. Deletion removes the
  checkpoint, never the episode labels; running jobs and loaded models are guarded.
- The **Review** icon shows the number of pending generated frames. If no episode
  is selected or the current episode has no pending drafts, it opens the first saved
  episode with generated drafts on the same map. When no map is selected, the
  first pending episode chooses the session map. Review stays active across episodes;
  saving the last pending frame opens the next queued episode automatically. The
  progress count and pending total cover the whole map session, not only the current
  episode. It stops advancing after the last draft; click Review again to exit.
  Library counts use `qwen_pending_review_count` and `sam_pending_review_count`;
  approved/manual frames and empty SAM results already reviewed are excluded. Click it to
  filter to those frames, and click again to return to sample labeling. Sampling
  controls remain visible but disabled during review. After the final review, the
  editor shows completion and the toggle stays available to exit. Review is locked
  during a job or unsaved edits.
  Empty SAM detections are recorded as generated frames and can be approved empty.
  Failed frames are not presented as successful results; successful partial results
  remain reviewable after the job finishes or fails.

The trash icon beside the frame arrows opens **Delete frame range**. Drag across
its timeline to select an inclusive range, or enter its first and last frame
numbers. Endpoint previews and a selected-frame count are shown before confirmation.
This range always uses the saved episode's complete frame order, even when the
editor is showing sampled frames or pending reviews. Deletion removes the selected
images, reasoning labels, predictions, and mask revisions together; surviving frame
IDs stay unchanged. Sampling selections and generated training files are refreshed.
Recording episodes and episodes used by an active training/auto-label job cannot be
modified. An empty saved episode may remain after all its frames are deleted.

`POST /collection/episodes/{eid}/frames/delete` accepts `{frame_ids: [...]}` and
returns `{episode, deleted_frame_ids}`. The editor and episode library refresh
immediately; no live dataset is deleted merely by opening the range selector.

Fine-tune, Auto-label, and Train model launchers remain enabled without an episode
selection, during pending edits, and while a job runs. Opening them never selects a
frame or starts a duplicate job. With no selected episode, Fine-tune opens a saved-map
picker and Auto-label defaults to map-wide generation. Actual job submissions still
require valid saved data, an available model where needed, and an idle worker.
`useLabelerDataset`, `LabelerDatasetOptions`, and `labelerSampleCounts` keep workflow
map/target selection and approved sample counts separate from the frame editor.

Job progress fills each action icon from bottom to top, with a percentage.
Preparation is indeterminate; failed jobs use an error color and message.
Click a running Fine-tune icon to inspect its model and step progress in a popover;
this does not start another training job.

`POST /collection/{segmentation|qwen}/auto-train` accepts `{map_name}`.
`DELETE /collection/{segmentation|qwen}/models/{model_id}` resolves model metadata
on the server rather than accepting filesystem paths.

The initial automatic step policy is `clamp(10 * instances, 100, 2000)` for SAM
and `clamp(5 * frames, 100, 2000)` for Qwen. Both current workers perform one
instance/frame per optimizer step. These are bounded starting defaults, not an
accuracy guarantee; adjust the policy if worker batching changes.

Models are named `sam_<map>_<index>` or `qwen<size>b_<map>_<index>` using the
configured Qwen base model (`QWEN_ANNOTATION_MODEL_ID`, default Qwen3.5-4B).
A locked `.labeler_sequence.json` in each model root reserves monotonically
increasing indices even after deletion or a failed run.

The top-right **Train model** icon opens a temporary settings drawer. It trains the
final Operate model from all approved, compatible reasoning labels in saved episodes
on the selected dataset map, independently of the current episode/sample/review filter.
Observation, Patrol, and Mission are included by default. Model name, steps, and data
balance are directly available; base model and type scope are under Advanced settings.
Masks remain optional separate annotations, not implicit reasoning-training inputs.

`useReasoningTraining` owns scope, settings, submission, and job polling. Closing the
drawer or switching label tools does not stop the job or remove its header progress.
The drawer shows source and balanced counts, job status, and completed reasoning models.
It uses the existing `qwen/train-reasoning` endpoint and keeps the reasoning checkpoint
storage and `purpose="reasoning"` registry unchanged, so completed models appear in Operate.
There is no separate training-purpose selector or duplicate auto-labeler training page.

## Module ownership

| Module | Responsibility |
| --- | --- |
| `../../mapWorkspace/components/workspace/StageRail.js` | Shared top-level stage navigation; Autonomy supplies its training-status footer |
| `workspace/AutonomyStageChrome.js` / `workspace/AutonomySceneSurface.js` | Autonomy-only layout, map HUD, and thought-bubble presentation |
| `CollectSidebar.js` / `RecordWorkspace.css` | Shared episode/editor session and stage-specific layout |
| `CollectCaptureControls.js` | Compact episode controls and capture mode/interval in the map HUD |
| `RecordReasoningPanel.js` | Model controls, attention image and reasoning above the Record map |
| `../hooks/useReasoningObservation.js` | Preview-only model selection and lifecycle |
| `AnnotationFrameStage.js` / `AnnotateWorkspace.css` | Shared preview and Train labeling layout |
| `FrameRangeDeleteButton.js` | Drag-select frame intervals and confirm bulk frame deletion |
| `CollectDatasetPanel.js` | Train editor and persistent final-training entry point |
| `useReasoningTraining.js` | Final-model scope, submission, and job polling |
| `ReasoningTrainingButton.js` / `ReasoningTrainingDrawer.js` | Final-training status icon and temporary settings panel |
| `CollectReasoningPanel.js` / `QwenWorkflowControls.js` | Reasoning frame editor and Qwen workflow adapter |
| `CollectAnnotationPanel.js` / `SamWorkflowControls.js` | Optional mask frame editor and SAM workflow adapter |
| `LabelerWorkflowControls.js` / `../hooks/useLabelerWorkflow.js` | Shared compact sampling, model jobs, polling and cancellation |
| `Sam2Annotation.js` / `../hooks/useMaskAnnotationSession.js` / `../hooks/useMaskViewport.js` | Mask UI, frame revisions/save, and browser image gestures |
| `../hooks/useCollectDraft.js` / `../hooks/useCollectionSession.js` | Collection facade and single storage mutation authority |
| `../hooks/useEpisodeCapture.js` / `../hooks/collectionEpisodeActions.js` / `../hooks/collectionFrameActions.js` | Capture loop, episode lifecycle, frame labels/deletion |
| `../hooks/useCollectAnnotation.js` | Pinned frame and unsaved label protection |
| `CollectStageHud.js` / `RecordControlPanel.js` | Map toolbar, execution status, and continuously mounted capture controls |
| `RecordAuthoringToolbar.js` / `RecordMapEditOverlay.js` | Map authoring actions and selected-waypoint overlay |
| `RecordObservationEditor.js` / `RecordRouteEditor.js` | Waypoint settings popup and two route editing icons: Save and Set route order |
| `CollectCaptureContext.js` | Capture controls and recording status shared with the map |
| `../hooks/useAutonomyWorkspaceRuntime.js` | Autonomy runtime composition using the shared navigation/BT owner |
| `../hooks/useRecordExecution.js` | Record patrol, waypoint movement, and child-mission coordination |

Record and Train keep the shared sidebar mounted so the selected episode,
frame, and dataset selection survive stage changes. Unsaved edits, active capture,
and navigation prevent unsafe transitions. Leaving Record stops an active Qwen preview
before entering Train; a failed Stop keeps Record open. Camera and teleop
mount only in Record. Background training does not block stage navigation; returning
to Train reads the current server job status.

**Test in Observation** selects an available completed model and returns to Record;
**Start reasoning** remains explicit. Selecting or training a model does not automatically
start inference, generate labels, or execute missions.

## Record

### Record map controller

The map toolbar groups five movement actions — Localize, Previous observation,
Patrol start/pause/resume, Next observation, and Stop — followed by a divider and
Create waypoint, Edit route, Undo, and Redo. The top-right Capture panel remains
visible and mounted while selecting waypoints or editing the map.

- **Create waypoint:** choose On Map or At Robot, complete the waypoint draft, and Save.
- **Edit route:** click a waypoint on the map to open its settings popup. Edit its name,
  dwell time, Include in route state, and linked missions there. Drag the waypoint body
  to move it or the direction handle to rotate it; these changes remain staged with
  the route edits until Save. Map clicks select a waypoint without changing route order.
- **Route order:** only Save and Set observation route order icons appear below the
  toolbar. Set route order starts a fresh staged route. Click waypoints in visit order;
  the map shows their numbers and connecting lines. Clicking the first waypoint again
  after selecting at least two closes the loop and finishes ordering. Toggle the order
  icon off to finish a One-way route. Waypoint settings and dragging are disabled while
  ordering. Existing waypoint settings are retained when reusing a point.
- **Save / discard:** Save applies all staged waypoint and route edits. The saved route
  is unchanged until Save. To discard edits, toggle Edit route off and confirm; there is
  no separate Cancel button. Undo/Redo works on the current edit session.
- **Waypoint selection outside editing:** selecting a saved waypoint exposes Go to
  selected observation. The explicit Go action sends navigation.
- **Capture:** Start episode, Capture/Pause, Save episode and Delete retain four icon
  positions. Capture mode stays visible; Interval appears only for continuous capture.
  Recording type and frame count share one status row. Child mission selection/run
  appears only when the current observation has linked missions.
- **Movement:** Stop cancels movement independently of capture. After movement stops,
  Load Map finishes the active episode and shuts down navigation/localization before
  opening the map picker. Pending waypoint edits must first be saved or discarded.
  Waypoints refresh automatically; Capture has no maintenance menu.

Collection follows actual observation/patrol/mission/return execution. Route or mission
transitions keep the capture session alive. Stationary capture remains supported.
Save episode waits for an in-flight frame and closes the episode without stopping
independent navigation. Delete opens the saved episode list and confirms deletion.

### Annotation and storage

Frame information fields are not manually entered. Episode groups are generated by the
server; Robot comes from the selected UI robot and Environment from collection settings
(default unknown). Reasoning saves only decision, mission_id and reason. SAM mask saves
and mission labels are independent. Historical scenario metadata is retained but is
not a model input or target.
Reasoning labels use only `start`, `wait`, and `unknown`. Choosing a recording route does
not fabricate runtime context. Fine-tuning the Qwen auto-labeler can combine
approved sampled frames from saved episodes on the same map. Unlabeled or draft
sampled frames are skipped; at least two approved frames are required. SAM mask
fine-tuning likewise uses approved sampled frames, excluding any frame with a
pending mask. It requires at least two approved instances, with a 128-frame and
512-instance limit across the combined selection. Both tools retain a
selected-episode option. The removed preview-context, copy-to-map, and training-export
controls are not part of the current Collect UI.

Storage uses `docker/workspace/dataset/reasoning/<map>/<observation|patrol|mission>/raw_episode_NNNN/`. The episode library switches between Observation, Patrol, and Mission. Qwen fine-tuning and final training accept selected types together; models record the types actually trained. Auto labeling respects that supported scope.
contains `episode.json`, `samples.jsonl`, images, and optional masks. Training preparation
is a separate backend/training workflow. See the Qwen collection data specification for
the persisted schema and training format.

## Operate

Load Map → Localize → configure **Observation plan** → select a model → **Run autonomy**.
The backend controller owns the session and repeats observation → task → observation until Stop.
Fixed mode returns to the first configured map spot; Patrol visits spots in order and advances
on `next_observation`; `wait`/`unknown` keep observing. Explicit `timed` points retain timer departure.
Record holds at decision-based points until the operator clicks Next observation. Each spot can limit candidate missions.
ObservationMode is not a required mission name. Qwen remains loaded while navigation/local BTs run.
UI unmount/reload does not stop the server session. Task failure preserves progress and stops
execution for operator recovery instead of replaying a partially completed task.

Implementation and storage: `docker/supervisor_api/AUTONOMY_KO.md`.

Auto Labeling review accepts frames with no masks. Save review records
`observation.mask_review.status = "empty"`, advances to the next frame, and
excludes that frame from unreviewed filters and subsequent mask generation.
Adding a manual mask clears the empty review. Empty reviews do not fabricate
positive mask instances for SAM training.

Both Fine-tuning tools offer **Choose episodes** under Training scope. Check
any saved episodes on the current map with approved sampled labels. Counts and
training requests include only the checked eligible episodes. Changing maps
clears the custom selection; map-wide and single-episode scopes remain available.

### Three recording roles

`useRecordNavigation` owns mission and patrol runners; `useRecordArrival` confirms observation arrival. `useRecordExecutionContext` reports capture-time state. `useCollectRecording` gates recording and ends episodes at terminal route states. Capture type is fixed per episode.

`utils/reasoningDataset.js` mirrors the API `collection_episode_contract.py` eligibility rules. Only matching frames enter reasoning sampling, labeling, auto labeling, exports and training. Original images and optional mask annotation stay available. New recordings use `task_aligned_v1`.

Record mission cycles use `useRecordMissionCycle` for task → pinned observation return and `useRecordReturnCapture` for the recording handoff. The return goal waits until a linked Patrol episode is published. A final settled arrival frame is captured before the return episode ends. A source/return pair shares group_id but keeps task-specific labels separate; Stop aborts delayed goals and Stop capture never restarts the frame loop implicitly.

## Execution state contract

Record and Operate now publish the same version-1 execution view, separate from
camera capture, model lifecycle, and map editing state. `lib/executionState.js`
provides shared labels; presentation code must not turn a label into an action.

| Field | Meaning |
| --- | --- |
| `lifecycle` | Session lifetime: idle, starting, active, stopping, stopped, interrupted, failed |
| `activity` | Patrol navigation, observation posture/observation, mission navigation/action, return navigation/posture, or idle |
| `control` | Running, pausing, paused, resuming, cancelling, or idle |
| `owns_execution` | Execution has not been released; failure alone does not release it |
| `observation_spot_id` / `target_spot_id` | Reached observation and execution target, independent of the editor selection |
| `mission_id` / `step_index` | Executing mission and zero-based step, not the mission dropdown selection |
| `pause_causes` / `return_checkpoint` | Holds and preserved return context |
| `epoch` / `session_id` / `revision` | Server process, session, and publication version; never infer ordering across epochs |
| `source` / `authority` | Origin of the snapshot; see the migration boundary below |

Operate `/autonomy/status` adds `execution` alongside existing compatibility
fields. All controller state writes use one locked commit path. A pending Pause
publishes `pausing`; only the actuator acknowledgment publishes `paused`. Resume
keeps the paused flag until acknowledgment. Manual holds are independent of AI
holds, and Stop failure retains ownership. Reconnecting after a server restart
reports `interrupted`; it does not replay the previous physical task.

Publication `revision` advances for status/results/events. Reasoning
`context_version` changes only at semantic transitions. A log entry must not
invalidate an in-flight inference. The UI rejects lower publication revisions
within the same epoch/session, and hides reasoning results from an older context.

### Current migration boundary

This is the state-contract foundation, not a replacement of the Record runtime.
Operate snapshots have `source=operate`, `authority=server`. Record exposes
`executionState` from `useRecordExecution`, with `source=record`,
`authority=browser`, and null server identity/revision fields. Its navigation,
mission, and return hooks remain responsible for execution. Record capture
continues to be explicitly client-reported; it is not relabeled as authoritative
server evidence. Existing frame/data formats are unchanged.

The next migration can move Record commands to the server execution owner without
changing this presentation contract. Preserve confirmed arrival, remaining dwell,
child-mission return, pause acknowledgment, and the existing capture lifecycle
when replacing the browser runners. Until that migration, do not treat this view
as a shared cross-browser lock or remove either runtime's cancellation checks.


### Operate map visualization

Record and Operate share `../../mapWorkspace/lib/missionRouteOverlay.js` for child mission geometry
and the observation marker palette. Operate keeps the observation route visible,
adds the active child mission as a read-only overlay, and restores the observation
view on return. Waypoint pulses follow running execution, not retained decisions.
`useActiveNavigationPlan` hides cached Nav2 paths after a hold/stop and waits for a
new path when the navigation target changes. Both pages omit costmap layers.

In Operate's Observation plan, Patrol → Route behavior → Closed loop uses the same
map-owned `route_closed` setting as Record. The map draws the last-to-first edge;
the server visits the first spot again after the final dwell until Stop. One-way
keeps observing at the final spot. These settings are saved before execution.
