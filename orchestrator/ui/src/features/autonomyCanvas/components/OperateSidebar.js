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

import OperateConversation from "./OperateConversation";
import { Panel, SessionRow } from "../../mapWorkspace/components/primitives";
import ObservationPlanEditor from "./ObservationPlanEditor";
import OperateModelSelect from "./OperateModelSelect";
import "./OperateSidebar.css";
import { autonomyPhaseLabel, returnSpotLabel } from "./autonomyPresentation";

export default function OperateSidebar({ mapName, mapLoaded, poseReady, running, observation = {}, section = "all" }) {
  const localized = Boolean(mapLoaded && running && poseReady);
  const mission = observation.mission;
  const runner = observation.runner || {};
  const returnSpot = returnSpotLabel(observation);
  const manuallyPaused = observation.pause_causes
    ? Boolean(observation.pause_causes.manual) : observation.paused || observation.phase === "paused";
  const status = !mapLoaded ? "Map required" : localized ? "Localized" : "Localization required";
  return (
    <div className="operate-sidebar grid gap-4" role="region" aria-label="Autonomy operation panels">
      {(section === "all" || section === "situation") && <Panel title="Current situation">
        <div className="operate-sidebar-status"><span className={localized ? "is-localized" : ""} />{status}</div>
        <div className="grid gap-2 mb-3">
          <SessionRow label="Map" value={mapLoaded ? mapName : "Not loaded"} />
          <SessionRow label="Localization" value={localized ? "Ready" : "Not ready"} />
          <SessionRow label="State" value={autonomyPhaseLabel(observation)} />
          <SessionRow label="Observation" value={observation.observation?.label || "None"} />
          {returnSpot && <SessionRow label="Return to" value={returnSpot} />}
          <SessionRow label="Active mission" value={mission?.mission_name || observation.pendingMission || "None"} />
          {mission && <SessionRow label="Progress" value={`${Math.max(0, (runner.currentIndex ?? -1) + 1)} / ${runner.total || 0} · ${runner.phase || runner.status}`} />}
        </div>
        <OperateModelSelect choice={observation.modelChoice} loadedModel={observation.loadedModel} active={observation.active} />
        <ObservationPlanEditor config={observation.config} />
        {!localized && <p className="operate-sidebar-note">Load Map → Localize → Run</p>}
      </Panel>}
      {(section === "all" || section === "activity") && <Panel title="Activity">
        <div className="operate-activity-label">Operation events <span>{observation.active ? autonomyPhaseLabel(observation) : "Autonomy idle"}</span></div>
        <ol className="operate-sidebar-events" aria-label="Operation activity">{(observation.events || []).map((event, index) => <li key={`${event.time}-${index}`}><time>{event.time}</time><div><strong>{event.title}</strong><p>{event.detail}</p></div></li>)}</ol>
      </Panel>}
      {(section === "all" || section === "message") && <Panel title="Conversation">
        <OperateConversation/>
        <div className="operate-execution-controls">
        {(observation.pause || observation.resume) && <button type="button" className="operate-pause-mission"
          disabled={!observation.active || observation.working || observation.phase === "stopping" || !(manuallyPaused ? observation.resume : observation.pause)}
          onClick={manuallyPaused ? observation.resume : observation.pause}>{manuallyPaused ? "Resume autonomy" : "Pause autonomy"}</button>}
        <button type="button" className="operate-stop-mission" disabled={!observation.needsStop || observation.phase === "stopping"} onClick={observation.stop}>Stop autonomy</button>
        </div>
      </Panel>}
    </div>
  );
}
