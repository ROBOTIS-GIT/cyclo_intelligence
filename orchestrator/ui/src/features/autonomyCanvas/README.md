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

# Autonomy Canvas

Autonomy Canvas owns **Record / Train / Operate**. Its implementation lives here,
independently of Mission Canvas's Mapping, Design, and Run tools.

| Location | Responsibility |
| --- | --- |
| `AutonomyCanvasWorkspace.js` | Entry point injecting Autonomy renderers and training status into the shared host |
| `components/workspace/AutonomyStageChrome.js` | Record/Train layout and Operate drawer shell |
| `components/workspace/AutonomySceneSurface.js` | Record/Operate map presentation, observation overlays, and robot thought bubble |
| `components/` | Capture, annotation, SAM/Qwen workflows, training, observation plans, and Operate controls/styles |
| `hooks/useAutonomyWorkspaceRuntime.js` | Composition of Operate status, Record observation management, Record execution, and mission overlays |
| `hooks/` | Recording, dataset editing, patrol/mission/return execution, reasoning preview, and model selection |
| `lib/` / `utils/` | Autonomy execution views, observation/mission overlays, and reasoning/capture data contracts |

The shared [`mapWorkspace`](../mapWorkspace/README.md) owns navigation,
localization, ROS subscriptions, mission-runner integration, and workspace shutdown
wiring. Autonomy hooks use that runtime through explicit inputs and callbacks.
`AutonomyCanvasWorkspace` no longer wraps `MissionCanvasWorkspace`.

Keep Autonomy-only changes in this directory. Do not import Mission Canvas
components or hooks directly. If both canvases require a facility, give it a shared
contract under `mapWorkspace` or an existing neutral API/renderer module.

The shared host still composes both features' hooks with their existing stage gates;
this separation does not introduce runtime lazy loading. Record's browser execution
and Operate's server execution retain their previous ownership and stop semantics.

Session IDs, map/mission routes, datasets, CSS classes, and design tokens remain
compatible. Tests move with the feature they exercise. The repository-wide feature
boundary test is `../mapWorkspace/architecture.test.js`.

See [component and workflow documentation](components/README.md) for the current
Record, labeling/training, and Operate UI behavior.

## Collection and annotation module boundaries

- `useCollectDraft` is the public collection facade. `useCollectionSession` alone owns
  episode state, the mutation lock and stale-response generation. Episode lifecycle
  and frame mutations use `collectionEpisodeActions` and `collectionFrameActions`.
  `useEpisodeCapture` owns the capture loop and uses the same lock; it does not create
  a second storage or execution owner.
- `Sam2Annotation` composes the mask toolbar/image. `useMaskAnnotationSession` owns
  prompt previews, pending instances, revision checks and batch review/save;
  `useMaskViewport` owns pan/zoom/fullscreen, using `maskImageCoordinates`.
- `LabelerWorkflowControls` is the shared SAM/Qwen sample/fine-tune/auto-label UI.
  `useLabelerWorkflow` owns polling, job submission/cancellation and stale poll rejection.
  Family-specific eligibility lives in `lib/labelerWorkflow`. The former hidden manual
  training forms and their unused history/progress components have been removed;
  final reasoning-model training remains in the training drawer.
- `AnnotationFrameNavigation` and `useLabelingJobRefresh` keep the two annotation tools
  consistent. Review scope and saved frame selection continue to belong to their editor.

The compact UI, server endpoints, training settings, dataset layout and model paths
are unchanged. Legacy collection page layouts were removed from `CollectSidebar.css`;
`RecordWorkspace.css` and `AnnotateWorkspace.css` own the current layouts.

Autonomy waypoint editing, capture HUD enablement, Operate overlays and sidebar mapping
are projected by `components/workspace/presentation`. Shared `mapWorkspace/presentation`
builders compose these pure projections with the common scene/chrome contract; they do
not own or duplicate the feature's selection and execution policies.
