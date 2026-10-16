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
import { MapViewer } from "../../../../components/navigation/MapViewer";
import { STAGE_COLLECT, STAGE_OPERATE } from "../../../mapWorkspace/lib/stages";
import useActiveNavigationPlan from "../../hooks/useActiveNavigationPlan";
import { OBSERVATION_WAYPOINT_COLORS } from "../../lib/missionRouteOverlay";
import OperateRobotThought from "../OperateRobotThought";
import { createRobotAnchorStore } from "../robotAnchorStore";
import CollectStageHud from "../CollectStageHud";
import OperateStageHud from "./OperateStageHud";

const EMPTY_LIST = Object.freeze([]);

export default function AutonomySceneSurface({ stage, scene, mission, editors, interaction, layers, hud, observation }) {
  const collecting = stage.id === STAGE_COLLECT;
  const operating = stage.id === STAGE_OPERATE;
  const waypoints = collecting ? mission.collectWaypoints : mission.operateWaypoints;
  const route = collecting ? waypoints?.routeView : waypoints?.route;
  const [recordWaypointAnchor, setRecordWaypointAnchor] = useState(null);
  const [robotAnchorStore] = useState(createRobotAnchorStore);
  const activePlan = useActiveNavigationPlan(!!waypoints?.navigationActive, scene.plan, waypoints?.navigationKey);
  const visiblePlan = layers.needsPlan ? activePlan : null;

  return (
    <section className={operating ? "operate-scene-surface" : "min-h-0 overflow-hidden relative"}
      style={{ backgroundColor: "var(--mc-surface)", borderRight: "1px solid var(--mc-border)" }}>
      <div className={operating ? "operate-map-canvas" : "contents"}>
        <MapViewer
          renderProfile="autonomy"
          robotType={scene.robotType}
          map={scene.displayedMap}
          globalCostmap={null}
          localCostmap={null}
          scan={layers.needsScan ? scene.scan : null}
          scanPose={stage.mappingTopicsActive ? scene.mappingPoseSync.scanPose : stage.runTopicsActive ? scene.runPoseSync.scanPose : null}
          pose={stage.designLocalizationActive || stage.navigationTopicsActive ? scene.currentPose : null}
          goalPose={null}
          showGoalPose={false}
          plan={visiblePlan}
          footprint={layers.needsRobotModel ? scene.footprint : null}
          tf={layers.needsTf || layers.needsRobotModel ? scene.displayTf : null}
          spots={waypoints?.spots || EMPTY_LIST}
          selectedSpotId={collecting ? waypoints?.selectedId || "" : ""}
          activeWaypointId={waypoints?.activeWaypointId || ""}
          navigationDestination={operating ? waypoints?.navigationDestination : null}
          missionFollowRobot={mission.followRobot}
          behaviorNodes={EMPTY_LIST}
          selectedBehaviorNodeId=""
          behaviorPreviewNode={null}
          waypointColors={OBSERVATION_WAYPOINT_COLORS}
          missionRouteOverlay={waypoints?.missionOverlay}
          missionRouteOrder={route?.order || EMPTY_LIST}
          missionRouteClosed={!!route?.closed}
          missionRouteMode={collecting && !!waypoints?.orderMode}
          selectedMissionRouteSourceId={collecting ? waypoints?.routeSourceId : mission.routeSourceId}
          mapAnnotations={EMPTY_LIST}
          selectedMapAnnotationId=""
          mapRefined={false}
          editorBrush={null}
          btLayer={null}
          showMap={layers.active.map}
          showGlobalCostmap={false}
          showLocalCostmap={false}
          showScan={layers.needsScan}
          showGlobalPlan={!!visiblePlan}
          showTf={false}
          showRobotModel={false}
          showFootprint={layers.needsRobotModel}
          interactionDisabled={!!interaction.busy}
          interactionMode={interaction.mode}
          editorActive={false}
          editorPaintOnDrag
          editorAreaSelection={false}
          fitContainer
          viewKey={`mission:${mission.mapName}:${scene.displayedMap ? "ready" : "wait"}`}
          waitingLabel={stage.running ? "Waiting for /map" : editors.runDisplayMapEditor.busy ? "Loading map" : "Load a map"}
          onRobotScreenPositionChange={operating ? robotAnchorStore.set : undefined}
          onSelectedSpotScreenPositionChange={collecting ? setRecordWaypointAnchor : undefined}
          onSpotClick={collecting ? waypoints?.onSelect || (waypoints?.clickEnabled ? waypoints.navigate : undefined) : undefined}
          onMissionRouteSpotClick={collecting ? waypoints?.onSelect : undefined}
          onMissionRouteMapClick={collecting ? waypoints?.onRouteMapClick : undefined}
          onSpotPoseChange={collecting && !waypoints?.orderMode ? waypoints?.onPoseChange : undefined}
          onMapTap={collecting ? waypoints?.onClearSelection : undefined}
          onMapClick={collecting ? undefined : interaction.onMapClick}
          onMapPose={interaction.onMapPose}
        />
        {operating && <OperateRobotThought observation={observation} anchorStore={robotAnchorStore} />}
        {operating && <OperateStageHud {...hud.run} />}
        {collecting && <CollectStageHud {...hud.collect} waypointAnchor={recordWaypointAnchor} />}
      </div>
    </section>
  );
}
