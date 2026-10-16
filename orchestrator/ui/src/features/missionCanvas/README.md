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

# Mission Canvas

Mission Canvas owns **Mapping / Map Edit / Navigation / Design / Run**.
Autonomy's Record, Train, and Operate tools live in
[`autonomyCanvas`](../autonomyCanvas/README.md).

| Location | Responsibility |
| --- | --- |
| `MissionCanvasWorkspace.js` | Entry point injecting Mission renderers into the shared map host |
| `components/workspace/MissionStageChrome.js` | Mapping, Design, Navigation, and Run panels |
| `components/workspace/MissionCanvasSceneSurface.js` | Mission-side map/BT visualization and editing HUD |
| `components/mapping`, `components/run`, `components/stages` | Stage-specific controls and presentation |
| `hooks/` | Design documents, waypoint/BT editing, map editing, and Mapping workflow |
| `lib/` | Mission design persistence and authoring-specific helpers |

Shared map/navigation/ROS lifecycle, session migration, dialogs, theme, and route
contracts live in [`mapWorkspace`](../mapWorkspace/README.md). The entry point
supplies feature presentation rather than hosting Autonomy stages itself.

Do not import `autonomyCanvas` directly. Shared behavior belongs in `mapWorkspace`
or existing neutral navigation/API modules, while the shared composition root may
wire both feature hook sets. Keep one runtime owner; moving UI ownership must not
create another mission runner or duplicate navigation subscriptions.

This refactor preserves public UI stage labels, persisted session IDs, mission/route
storage, and API calls. Shared legacy symbol names such as `MISSION_*` remain stable
for compatibility. Source ownership is enforced by
`../mapWorkspace/architecture.test.js`.

## Design document ownership

`useDesignWorkspaceState` owns identity, selections, route edits, loader leases,
and the document ledger. `useDesignWorkspaceModel` adds the catalog, behavior
nodes, and shared document history. `useDesignEditingControllers` composes task
file operations, document persistence, and waypoint interaction. The shared map
host supplies navigation/pose services without becoming the document owner.

- `hooks/taskFiles/`: selection, hydration, file load/save/default commands, and a
  shared operation gate. Every async result is checked against its document lease.
- `lib/documentLedger/`: one private state store with edit, baseline, history,
  query, and save commands. `useDesignMissionDocumentLedger` only adapts that store
  to React. It must remain synchronous for consecutive commands in one event.
- `hooks/waypoint/` and `createWaypointMutationQueue`: per-waypoint ordering and
  stale-result/rollback behavior shared by move and rename.
- `useDesignDocumentPersistence`: document switch/new/save/unsaved guard wiring.
  Saving a local task and saving its mission share ledger revision ownership.
- `btEditor/`: Mission-specific BT document and file adapters. Graph behavior and
  parameter controls belong to the neutral `../btEditor` feature.

Keep one ledger and one file-operation guard for a mounted Design document.
Splitting implementation files must not split transaction ownership or create
independent dirty flags.
