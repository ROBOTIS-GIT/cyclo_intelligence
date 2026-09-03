// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import MissionBtEditor from "../../../components/navigation/MissionBtEditor";
import MissionBtRunView from "../../../components/navigation/MissionBtRunView";
import { RunnerStatus } from "../../../hooks/missionRunnerCore";
import { defaultLocalBtXml, localBtPathForSpot } from "../lib/missionBtFiles";
import { STAGE_AUTHORING, STAGE_RUN } from "../lib/stages";

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
  const runBtLayer = runActiveSpot ? {
    spot: runActiveSpot,
    editor: (
      <MissionBtRunView
        xml={runMissionBtFiles[localBtPathForSpot(runActiveSpot)] || defaultLocalBtXml(runActiveSpot)}
        activeNodeNames={btActiveNodeNames}
      />
    ),
  } : null;
  const activeBtLayer = waypointBtLayer || runBtLayer;

  return {
    waypointBtLayer,
    runBtLayer,
    activeBtLayer,
    runBtViewActive: !!runActiveSpot,
  };
}
