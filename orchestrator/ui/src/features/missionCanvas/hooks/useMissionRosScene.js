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

import { useMemo } from "react";
import { useMappingPoseSync } from "../../../hooks/useMappingPoseSync";
import { useNavigationRosTopic } from "../../../hooks/useNavigationRosTopic";
import {
  applyPoseSyncToTf,
  mergeTfMessages,
  poseFromBaseLinkTf,
  tfMessageFromBuffer,
} from "../../../utils/navigationTf";
import { DEFAULT_MAP_NAME } from "../lib/missionNames";
import {
  LAYER_PRESETS,
  LAYER_TOPIC_IDS,
  STAGE_EXTRA_TOPIC_IDS,
  STAGE_LAYER_IDS,
  TOPIC_ORDER,
} from "../lib/layers";
import { hasTopicMessage, messageData, rosStringData } from "../lib/rosTopicPayload";
import {
  STAGE_AUTHORING,
  STAGE_MAPPING,
  STAGE_MAP_EDIT,
  STAGE_NAVIGATE,
  STAGE_RUN,
} from "../lib/stages";

const ROS2_WS_FAST_TOPIC_OPTIONS = { throttleMs: 100 };

const ROS2_WS_ODOM_TOPIC_OPTIONS = { throttleMs: 50, staleMs: 1000 };

// The BT glow only needs about 7fps. Unthrottled status messages re-render the
// whole workspace per message and can starve the map pulse loop while the
// read-only BT split view is open.
const BT_TOPIC_OPTIONS = { staleMs: 3000, throttleMs: 150 };

// These gates are needed before useMissionRosScene: the workspace uses them to
// open its three map-editor hooks and those editors, in turn, provide the static
// map fallbacks consumed by the ROS scene hook. Keeping this calculation pure
// avoids moving the editor effects across the ROS subscription effects.
export function deriveMissionRosSceneGates({
  busy,
  designMapPath,
  layersByStage,
  mapName,
  missionMapLoaded,
  navigationRuntimeMode,
  runBtVisualizationActive,
  runMapName,
  running,
  workspaceStage,
}) {
  const mappingEditorActive = workspaceStage === STAGE_MAP_EDIT;
  const designMapActive = (
    workspaceStage === STAGE_AUTHORING && !!designMapPath && missionMapLoaded
  );
  const robotPoseCaptureActive = workspaceStage === STAGE_AUTHORING && designMapActive;
  const mappingRuntimeActive = running && navigationRuntimeMode === "mapping";
  const runRuntimeActive = running && navigationRuntimeMode === "run";
  const designLocalizationActive = (
    workspaceStage === STAGE_AUTHORING
    && designMapActive
    && running
    && navigationRuntimeMode === "localization"
  );
  const mappingTopicsActive = (
    workspaceStage === STAGE_MAPPING
    && mappingRuntimeActive
    && busy !== "Stop"
    && !mappingEditorActive
  );
  const mappingPoseSubscriptionActive = (
    workspaceStage === STAGE_MAPPING
    && !mappingEditorActive
    && busy !== "Stop"
  );
  // Navigate shares the Run runtime plumbing (map snapshot, localization and
  // live topics); only the mission machinery remains Run-only.
  const runFamilyStage = workspaceStage === STAGE_RUN || workspaceStage === STAGE_NAVIGATE;
  const runTopicsActive = (
    runFamilyStage
    && runRuntimeActive
    && busy !== "Stop"
  );
  const stageNavigationTopicsActive = mappingTopicsActive || runTopicsActive;
  const activeLayers = layersByStage[workspaceStage] || LAYER_PRESETS[workspaceStage];
  const runSessionActive = runFamilyStage;
  const currentMapName = (
    (runSessionActive ? runMapName : mapName).trim() || DEFAULT_MAP_NAME
  );
  const runBtMapLightweight = workspaceStage === STAGE_RUN && runBtVisualizationActive;

  return {
    activeLayers,
    currentMapName,
    designLocalizationActive,
    designMapActive,
    mappingEditorActive,
    mappingPoseSubscriptionActive,
    mappingRuntimeActive,
    mappingTopicsActive,
    robotPoseCaptureActive,
    runBtMapLightweight,
    runFamilyStage,
    runRuntimeActive,
    runSessionActive,
    runTopicsActive,
    running,
    stageNavigationTopicsActive,
    workspaceStage,
  };
}

export function deriveMissionTopicRows({
  activeLayers,
  amclPose,
  btActiveNodesData,
  btNodeIsUp,
  btStatusData,
  designLocalizationActive,
  footprint,
  globalCostmap,
  localCostmap,
  map,
  odometry,
  plan,
  runFamilyStage,
  scan,
  slamPose,
  tf,
  tfStatic,
  workspaceStage,
}) {
  const liveByTopic = {
    "/map": !!map,
    "/scan": !!scan,
    "/pose": !!slamPose,
    "/odom": !!odometry,
    "/amcl_pose": !!amclPose,
    "/tf": !!(tf?.transforms?.length),
    "/tf_static": !!(tfStatic?.transforms?.length),
    "/local_costmap/published_footprint": !!(footprint?.polygon?.points?.length),
    "/global_costmap/costmap": !!globalCostmap,
    "/local_costmap/costmap": !!localCostmap,
    "/plan": !!plan,
    "/bt/status": hasTopicMessage(btStatusData) || btNodeIsUp,
    "/bt/active_nodes": hasTopicMessage(btActiveNodesData) || btNodeIsUp,
  };
  const selectedTopics = new Set(STAGE_EXTRA_TOPIC_IDS[workspaceStage] || []);
  (STAGE_LAYER_IDS[workspaceStage] || []).forEach((layerId) => {
    if (!activeLayers[layerId]) return;
    (LAYER_TOPIC_IDS[layerId] || []).forEach((topic) => selectedTopics.add(topic));
  });
  const robotPoseLayerActive = (
    !!activeLayers.scan || !!activeLayers.robotModel || !!activeLayers.tf
  );
  if (workspaceStage === STAGE_MAPPING && robotPoseLayerActive) {
    selectedTopics.add("/pose");
    selectedTopics.add("/odom");
  }
  if (runFamilyStage && robotPoseLayerActive) {
    selectedTopics.add("/amcl_pose");
  }
  if (workspaceStage === STAGE_AUTHORING && !designLocalizationActive) {
    selectedTopics.delete("/map");
    selectedTopics.delete("/scan");
    selectedTopics.delete("/amcl_pose");
    selectedTopics.delete("/tf");
    selectedTopics.delete("/tf_static");
    selectedTopics.delete("/local_costmap/published_footprint");
  }
  if (designLocalizationActive) {
    ["/scan", "/amcl_pose", "/tf", "/tf_static", "/local_costmap/published_footprint"].forEach((topic) => {
      selectedTopics.add(topic);
    });
  }
  return TOPIC_ORDER.filter((topic) => selectedTopics.has(topic)).map((topic) => ({
    topic,
    isLive: !!liveByTopic[topic],
  }));
}

// Subscribe only to the topics required by the current stage/layer selection,
// normalize their payloads, synchronize SLAM/AMCL poses with odometry and pick
// the live-or-static map shown by MapViewer.
//
// tfBufferRef and tfBufferRevision intentionally remain caller-owned. Runtime
// start/stop handlers clear the same buffer alongside AMCL/current-pose refs,
// while a later workspace effect appends latestTf. Moving those mutation
// effects here would reorder them relative to the existing session lifecycle.
export default function useMissionRosScene({
  sceneGates,
  mapEditorMap,
  designMapEditorMap,
  runDisplayMapEditorMap,
  btNodeIsUp,
  tfBufferRef,
  tfBufferRevision,
}) {
  const {
    activeLayers,
    designLocalizationActive,
    designMapActive,
    mappingEditorActive,
    mappingPoseSubscriptionActive,
    mappingTopicsActive,
    robotPoseCaptureActive,
    runBtMapLightweight,
    runFamilyStage,
    runSessionActive,
    runTopicsActive,
    stageNavigationTopicsActive,
    workspaceStage,
  } = sceneGates;

  const needsGlobalCostmap = (
    stageNavigationTopicsActive && activeLayers.globalCostmap && !runBtMapLightweight
  );
  const needsLocalCostmap = (
    stageNavigationTopicsActive && activeLayers.localCostmap && !runBtMapLightweight
  );
  const needsScan = designLocalizationActive || (
    stageNavigationTopicsActive && activeLayers.scan && !runBtMapLightweight
  );
  const needsPlan = (
    stageNavigationTopicsActive && activeLayers.globalPlan && !runBtMapLightweight
  );
  const needsRobotModel = designLocalizationActive || (
    stageNavigationTopicsActive && activeLayers.robotModel
  );
  // Keep the SLAM/odometry anchor warm for the whole Mapping session. If all
  // pose-dependent layers are toggled off while the robot is stationary,
  // slam_toolbox may not publish another /pose when a layer is re-enabled.
  const needsMappingPose = mappingPoseSubscriptionActive;
  const needsAmclPose = robotPoseCaptureActive || runTopicsActive;
  const needsTf = robotPoseCaptureActive || (
    stageNavigationTopicsActive && !runBtMapLightweight && (
      activeLayers.tf
      || activeLayers.scan
      || activeLayers.robotModel
    )
  );
  const needsMap = (
    stageNavigationTopicsActive || designLocalizationActive
  ) && activeLayers.map;
  const needsBtTopics = workspaceStage === STAGE_RUN;

  const { topicData: mapData } = useNavigationRosTopic(
    needsMap ? "/map" : null,
  );
  const { topicData: globalCostmapData } = useNavigationRosTopic(
    needsGlobalCostmap ? "/global_costmap/costmap" : null,
  );
  const { topicData: localCostmapData } = useNavigationRosTopic(
    needsLocalCostmap ? "/local_costmap/costmap" : null,
  );
  const { topicData: footprintData } = useNavigationRosTopic(
    needsRobotModel ? "/local_costmap/published_footprint" : null,
    ROS2_WS_FAST_TOPIC_OPTIONS,
  );
  const { topicData: scanData } = useNavigationRosTopic(
    needsScan ? "/scan" : null,
    ROS2_WS_FAST_TOPIC_OPTIONS,
  );
  const { topicData: slamPoseData } = useNavigationRosTopic(
    needsMappingPose ? "/pose" : null,
    ROS2_WS_FAST_TOPIC_OPTIONS,
  );
  const { topicData: odometryData } = useNavigationRosTopic(
    needsMappingPose || runTopicsActive ? "/odom" : null,
    ROS2_WS_ODOM_TOPIC_OPTIONS,
  );
  const { topicData: amclData } = useNavigationRosTopic(
    needsAmclPose ? "/amcl_pose" : null,
    ROS2_WS_FAST_TOPIC_OPTIONS,
  );
  const { topicData: planData } = useNavigationRosTopic(
    needsPlan ? "/plan" : null,
  );
  const { topicData: tfData } = useNavigationRosTopic(
    needsTf ? "/tf" : null,
    ROS2_WS_FAST_TOPIC_OPTIONS,
  );
  const { topicData: tfStaticData } = useNavigationRosTopic(
    needsTf ? "/tf_static" : null,
  );
  const { topicData: btStatusData } = useNavigationRosTopic(
    needsBtTopics ? "/bt/status" : null,
    BT_TOPIC_OPTIONS,
  );
  const { topicData: btActiveNodesData } = useNavigationRosTopic(
    needsBtTopics ? "/bt/active_nodes" : null,
    BT_TOPIC_OPTIONS,
  );

  const map = useMemo(() => messageData(mapData), [mapData]);
  const globalCostmap = useMemo(() => messageData(globalCostmapData), [globalCostmapData]);
  const localCostmap = useMemo(() => messageData(localCostmapData), [localCostmapData]);
  const footprint = useMemo(() => messageData(footprintData), [footprintData]);
  const scan = useMemo(() => messageData(scanData), [scanData]);
  const slamPose = useMemo(() => messageData(slamPoseData), [slamPoseData]);
  const odometry = useMemo(() => messageData(odometryData), [odometryData]);
  const amclPose = useMemo(() => messageData(amclData), [amclData]);
  const plan = useMemo(() => messageData(planData), [planData]);
  const tf = useMemo(() => messageData(tfData), [tfData]);
  const tfStatic = useMemo(() => messageData(tfStaticData), [tfStaticData]);
  const btStatusText = useMemo(() => rosStringData(btStatusData), [btStatusData]);
  const btActiveNodesText = useMemo(() => {
    const names = rosStringData(btActiveNodesData)
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean);
    return names.join(", ");
  }, [btActiveNodesData]);
  const latestTf = useMemo(() => mergeTfMessages(tfStatic, tf), [tf, tfStatic]);
  // The revision makes mutations to the caller-owned Map observable here.
  void tfBufferRevision;
  const bufferedTf = tfMessageFromBuffer(tfBufferRef.current) ?? latestTf;
  const fallbackPose = amclPose?.pose?.pose ?? null;
  const tfPose = poseFromBaseLinkTf(bufferedTf);
  const mappingPoseSync = useMappingPoseSync({
    active: mappingPoseSubscriptionActive && !!odometry,
    slamPose,
    odometry,
    scanStamp: scan?.header?.stamp ?? null,
  });
  const runPoseSync = useMappingPoseSync({
    active: runTopicsActive && !!odometry,
    slamPose: amclPose,
    odometry,
    scanStamp: scan?.header?.stamp ?? null,
  });
  const mappingTf = mappingTopicsActive
    ? applyPoseSyncToTf(bufferedTf, mappingPoseSync)
    : bufferedTf;
  const runTf = runTopicsActive
    ? applyPoseSyncToTf(bufferedTf, runPoseSync)
    : bufferedTf;
  const displayTf = runTopicsActive ? runTf : mappingTf;
  // AMCL is authoritative in Run/Navigation. Retain TF as a startup fallback
  // until the first AMCL pose arrives because rosbridge throttling can miss
  // low-rate map -> odom updates while odom -> base_link continues.
  const currentPose = runSessionActive
    ? runPoseSync.pose ?? fallbackPose ?? tfPose
    : mappingTopicsActive
      ? mappingPoseSync.pose ?? tfPose
      : tfPose ?? fallbackPose;
  const displayedMap = mappingEditorActive
    ? mapEditorMap
    : workspaceStage === STAGE_AUTHORING
      ? designMapActive ? designMapEditorMap : null
      : runFamilyStage
        ? (map || runDisplayMapEditorMap)
        : map;
  const designMapAvailable = designMapActive && !!designMapEditorMap;
  const topicRows = useMemo(() => deriveMissionTopicRows({
    activeLayers,
    amclPose,
    btActiveNodesData,
    btNodeIsUp,
    btStatusData,
    designLocalizationActive,
    footprint,
    globalCostmap,
    localCostmap,
    map,
    odometry,
    plan,
    runFamilyStage,
    scan,
    slamPose,
    tf,
    tfStatic,
    workspaceStage,
  }), [
    activeLayers,
    amclPose,
    btActiveNodesData,
    btNodeIsUp,
    btStatusData,
    designLocalizationActive,
    footprint,
    globalCostmap,
    localCostmap,
    map,
    odometry,
    plan,
    runFamilyStage,
    scan,
    slamPose,
    tf,
    tfStatic,
    workspaceStage,
  ]);

  return {
    amclPose,
    btActiveNodesText,
    btStatusText,
    currentPose,
    designMapAvailable,
    displayedMap,
    displayTf,
    footprint,
    globalCostmap,
    latestTf,
    localCostmap,
    mappingPoseSync,
    needsBtTopics,
    needsGlobalCostmap,
    needsLocalCostmap,
    needsPlan,
    needsRobotModel,
    needsScan,
    needsTf,
    plan,
    resetMappingPoseSync: mappingPoseSync.reset,
    runPoseSync,
    scan,
    topicRows,
  };
}
