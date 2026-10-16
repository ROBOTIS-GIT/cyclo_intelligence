// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { FiSave } from "react-icons/fi";
import { MdAltRoute } from "react-icons/md";
import { RecordToolButton } from "./RecordAuthoringControls";
import "./RecordMapEditor.css";

export default function RecordRouteEditor({ manager, disabled = false }) {
  const route = manager.route, editor = manager.routeEditor;
  if (!route?.routeMode || !editor?.plan) return null;
  const points = editor.plan.observations || [];
  const locked = disabled || editor.locked || manager.working || manager.blocked;
  const run = action => { void Promise.resolve().then(action).catch(() => {}); };

  return <fieldset className="record-route-bar" disabled={locked} aria-label="Observation route editor">
    <div className="record-route-bar-main">
      <RecordToolButton label="Save observation route" disabled={locked || !points.length}
        onClick={() => run(editor.save)}><FiSave aria-hidden="true" /></RecordToolButton>
      <RecordToolButton label="Set observation route order" active={route.orderMode} disabled={locked}
        title={route.orderMode
          ? "Click waypoints in order. Click the first waypoint again to close the loop. Click here to finish."
          : "Set a new route by clicking waypoints in order."}
        onClick={route.toggleOrderMode}><MdAltRoute aria-hidden="true" /></RecordToolButton>
    </div>
  </fieldset>;
}
