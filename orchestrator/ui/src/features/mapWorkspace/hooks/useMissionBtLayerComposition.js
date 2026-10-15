// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import MissionBtEditor from "../../../components/navigation/MissionBtEditor";
import MissionBtRunView from "../../../components/navigation/MissionBtRunView";
import { RunnerStatus } from "../../../hooks/missionRunnerCore";
import { defaultLocalBtXml, localBtPathForSpot } from "../lib/missionBtFiles";
import { STAGE_AUTHORING, STAGE_RUN, STAGE_OPERATE } from "../lib/stages";

export default function useMissionBtLayerComposition({
  workspaceStage,
  selectedBtLayerSpot,
  selectedBtLayerPath,
  selectedBtLayerPaths,
  selectedBtLayerDefaultPath,
  missionBtFiles,
  missionBtLoadingPath,
  loadMissionLocalBtXml,
  saveMissionLocalBtXml,
  selectMissionLocalBtXml,
  saveMissionLocalBtXmlAs,
  setMissionLocalBtDefault,
  localBtFileActionsDisabled,
  handleMissionLocalBtXmlChange,
  missionRunner,
  autonomyOperation,
  runVisibleSpots,
  runMissionBtFiles,
  btActiveNodesText,
}) {
  const btActiveNodeNames = String(btActiveNodesText || "")
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
  const waypointBtEditor = selectedBtLayerSpot ? (
    <MissionBtEditor
      title={`${selectedBtLayerSpot.label || selectedBtLayerSpot.id} Waypoint Task`}
      filePath={selectedBtLayerPath}
      fileOptions={selectedBtLayerPaths}
      defaultFilePath={selectedBtLayerDefaultPath}
      xml={missionBtFiles[selectedBtLayerPath] || defaultLocalBtXml(selectedBtLayerSpot)}
      loading={missionBtLoadingPath === selectedBtLayerPath}
      activeNodeNames={[]}
      onLoadXml={loadMissionLocalBtXml}
      onSaveXml={saveMissionLocalBtXml}
      onFilePathChange={selectMissionLocalBtXml}
      onSaveXmlAs={saveMissionLocalBtXmlAs}
      onSetDefaultXml={setMissionLocalBtDefault}
      fileActionsDisabled={localBtFileActionsDisabled}
      onXmlChange={handleMissionLocalBtXmlChange}
    />
  ) : null;
  const waypointBtLayer = (
    workspaceStage === STAGE_AUTHORING
    && selectedBtLayerSpot
  ) ? {
      spot: selectedBtLayerSpot,
      editor: waypointBtEditor,
    }
    : null;

  const runActiveSpot = (
    workspaceStage === STAGE_RUN
    && missionRunner.status === RunnerStatus.RUNNING_BT
    && missionRunner.activeSpotId
  )
    ? runVisibleSpots.find((spot) => spot.id === missionRunner.activeSpotId) || null
    : null;
  const operateRunner = autonomyOperation?.runner;
  const operateActiveSpot = workspaceStage === STAGE_OPERATE
    && autonomyOperation?.active && autonomyOperation?.phase === "executing"
    && operateRunner?.phase === "waypoint-task" && operateRunner?.bt_xml
    ? autonomyOperation.mission_route?.find(spot => spot.id === operateRunner.activeSpotId) || null : null;
  const activeSpot = operateActiveSpot || runActiveSpot;
  const runBtLayer = activeSpot ? {
    spot: activeSpot,
    editor: (
      <MissionBtRunView
        xml={operateActiveSpot ? operateRunner.bt_xml : runMissionBtFiles[localBtPathForSpot(runActiveSpot)] || defaultLocalBtXml(runActiveSpot)}
        activeNodeNames={btActiveNodeNames}
      />
    ),
  } : null;
  const activeBtLayer = waypointBtLayer || runBtLayer;

  return {
    waypointBtLayer,
    runBtLayer,
    activeBtLayer,
    runBtViewActive: !!activeSpot,
  };
}
