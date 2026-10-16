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

import { useState } from "react";
import { MdAddLocationAlt, MdRoute, MdUndo, MdRedo } from "react-icons/md";
import { WaypointOptionButton } from "../../mapWorkspace/components/primitives";
import { RecordToolButton } from "./RecordAuthoringControls";
import "./RecordAuthoring.css";

export default function RecordAuthoringToolbar({ manager, disabled = false }) {
  const [options, setOptions] = useState(false);
  if (!manager) return null;
  const locked = disabled || manager.blocked || manager.working || !manager.ready;
  const routing = !!manager.route?.routeMode;
  const create = placement => { setOptions(false); placement === "map" ? manager.createOnMap() : manager.createAtRobot(); };
  const toggleEdit = () => {
    setOptions(false);
    if (routing && manager.routeEditor?.dirty) {
      if (window.confirm("Discard unsaved observation route edits?")) {
        void Promise.resolve().then(() => manager.routeEditor.discard()).catch(() => {});
      }
      return;
    }
    manager.route.toggleMode();
  };
  return <div className="record-authoring" aria-label="Observation authoring tools">
    <div className="record-authoring-toolbar" role="group" aria-label="Waypoint tools">
      <div className="relative">
        <RecordToolButton label="Create observation waypoint" active={options || !!manager.draft}
          disabled={locked || routing || manager.planDirty} onClick={() => setOptions(value => !value)}><MdAddLocationAlt size={17} /></RecordToolButton>
        {options && <div className="record-waypoint-options" role="menu" aria-label="Observation creation options">
          <WaypointOptionButton disabled={locked || routing} onClick={() => create("map")}>On Map</WaypointOptionButton>
          <WaypointOptionButton disabled={locked || routing || !manager.canUseRobot} onClick={() => create("robot")}>At Robot</WaypointOptionButton>
        </div>}
      </div>
      <RecordToolButton label="Edit observation route" active={routing} disabled={locked || !!manager.draft}
        onClick={toggleEdit}><MdRoute size={17} /></RecordToolButton>
      <RecordToolButton label="Undo" disabled={locked || !manager.canUndo} onClick={manager.undo}><MdUndo size={17} /></RecordToolButton>
      <RecordToolButton label="Redo" disabled={locked || !manager.canRedo} onClick={manager.redo}><MdRedo size={17} /></RecordToolButton>
    </div>
  </div>;
}
