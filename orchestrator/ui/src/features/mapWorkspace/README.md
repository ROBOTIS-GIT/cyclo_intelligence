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

# Shared map workspace

`mapWorkspace` is the composition root for the two map-based canvases. It owns
shared runtime wiring; feature-specific presentation and behavior live in
[`missionCanvas`](../missionCanvas/README.md) and
[`autonomyCanvas`](../autonomyCanvas/README.md).

```text
AutonomyStudioPage
  ├─ MissionCanvasWorkspace  ─┐
  └─ AutonomyCanvasWorkspace ─┴─ MapWorkspace
                                  ├─ shared navigation / ROS / BT lifecycle
                                  ├─ feature hooks
                                  └─ MapWorkspaceView
                                       ├─ injected StageChrome
                                       └─ injected SceneSurface
```

The page renders the selected Canvas entry point. Each entry point supplies its
own `StageChrome` and `SceneSurface`; Autonomy also supplies the background-training
rail footer. Neither feature imports the other feature's implementation.

## Ownership

| Location | Responsibility |
| --- | --- |
| `MapWorkspace.js` | Composition only: connects workspace, Design, runtime, and Record models |
| `components/workspace/MapWorkspaceView.js` | Small renderer consuming grouped presentation models |
| `hooks/` | ROS scene, map/mission snapshots, localization, BT lease, stop and workspace-exit coordination |
| `components/` | Shared map chrome, dialogs, stage rail/header, primitives, and mobile teleop |
| `lib/` | Session/stage contract, map/mission geometry, BT files, and shared theme |
| `../../components/navigation/MapViewer.js` | Reusable map renderer used by both canvases |

The host composes both Mission and Autonomy hooks. This refactor establishes
source ownership and separate presentation; it does **not** lazy-load runtimes or
promise that inactive-feature hooks are unmounted. Keep existing `enabled` gates,
refs, effect order, and cleanup behavior when extracting further modules.

There is one shared navigation/localization/BT lifecycle per mounted workspace.
The Mission runner and Record/Operate coordination use this owner and its callbacks;
do not create a second copy in either feature renderer. The Autonomy backend still
owns Operate execution, while Record execution remains browser-controlled. Sharing
the UI host does not merge these execution authorities.

## Dependency rules

- `missionCanvas` and `autonomyCanvas` may use shared `mapWorkspace` contracts.
- They must not directly import one another. `architecture.test.js` checks actual
  source imports, including re-exports and literal dynamic imports.
- The composition root may import both features to assemble the workspace.
- Add Record/Train/Operate behavior in `autonomyCanvas`; add Design/Run/Mapping
  behavior in `missionCanvas`. Extract a shared contract here when both need it.
- Keep low-level navigation/ROS API clients and the map renderer in their existing
  neutral modules. Moving a file must not duplicate listeners or execution owners.

## Compatibility preserved by this move

The session key remains `autonomy_studio_session`, with migration from
`mission_canvas_session`. Existing workspace IDs, persisted `collect`/`learn`
stage IDs, remembered stages, and shutdown-retry markers are retained. `session`
and exit-guard tests cover these contracts.

No API routes, map/mission file formats, observation plans, or dataset paths are
changed by this source reorganization. Shared design tokens remain `--mc-*` under
`.autonomy-studio-page` in `App.css`. Feature styles remain beside their components;
Autonomy CSS import order and the Nunito font URL must be preserved when moving them.
Existing exported names such as `MISSION_*` and `readMissionSession` are compatible
shared contracts, not a reason to place new Autonomy features under Mission Canvas.

## Verification

Run the ownership test, shared session/runtime tests, feature scene/chrome tests,
and the `AutonomyStudioPage` integration suite after cross-feature changes. Finish
with `npm run build` to check assets, CSS imports, and production compilation.

## Controller and presentation boundaries

- `useMapWorkspaceState` owns shared map/navigation UI state and stable pose refs.
- `useWorkspaceMapScene` composes the map documents and ROS scene; render layers
  remain in `components/navigation/mapViewer` and the map document editor in
  `components/navigation/mapEditor`.
- `useWorkspaceMissionRunner`, `useRunMissionStart`, and
  `useWorkspaceNavigationSession` retain one runner, revision-checked start,
  and navigation status ownership. They do not duplicate backend execution.
- `useWorkspaceStageTransition` awaits shutdown before changing stages and
  retains the Record/Train dataset selection. `useWorkspaceLocalize` and
  `useWorkspaceLocalizationEffects` keep runtime restarts from reusing stale poses.
- Design state/history/files/persistence live under Mission Canvas. Record capture,
  masks, model jobs, and map-change handling live under Autonomy Canvas.
- `presentation/` builds dialog, rail, chrome, scene, and HUD props from the
  `workspace`, `design`, `designEditing`, `recordData`, `mapping`, and `runtime`
  contracts. These pure projections must not start jobs or publish robot commands.
  Autonomy-specific projection policy is delegated to Autonomy Canvas helpers.

Controllers receive the domain model they operate on, rather than acquiring a
second copy of its state. Extracted effects depend on the individual stable refs,
setters, and values they use; do not put the freshly assembled model objects in
an effect dependency list. The small composition hooks are intentionally separate
from the stateful controller implementations they connect.

Keep new behavior in its owning module; do not grow the host or projection builders
into another monolithic editor. Most modules should cover one responsibility in
roughly 100–250 lines; composition files can be larger when the explicit ownership
and execution order remain clear.
