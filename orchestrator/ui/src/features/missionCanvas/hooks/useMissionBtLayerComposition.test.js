// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { renderHook } from "@testing-library/react";
import { RunnerStatus } from "../../../hooks/missionRunnerCore";
import { localBtPathForSpot } from "../lib/missionBtFiles";
import { STAGE_AUTHORING, STAGE_NAVIGATE, STAGE_RUN } from "../lib/stages";
import useMissionBtLayerComposition from "./useMissionBtLayerComposition";

const callbacks = {
  loadMissionLocalBtXml: jest.fn(),
  saveMissionLocalBtXml: jest.fn(),
  selectMissionLocalBtXml: jest.fn(),
  saveMissionLocalBtXmlAs: jest.fn(),
  setMissionLocalBtDefault: jest.fn(),
  handleMissionLocalBtXmlChange: jest.fn(),
};

const selectedSpot = { id: "wp-1", label: "Dock", x: 1, y: 2, yaw: 0 };
const runSpot = { id: "wp-2", label: "Inspect", x: 3, y: 4, yaw: 1 };

function renderComposition(overrides = {}) {
  return renderHook((props) => useMissionBtLayerComposition(props), {
    initialProps: {
      workspaceStage: STAGE_AUTHORING,
      selectedBtLayerSpot: selectedSpot,
      selectedBtLayerPath: "tasks/dock.xml",
      selectedBtLayerPaths: ["tasks/dock.xml", "tasks/other.xml"],
      selectedBtLayerDefaultPath: "tasks/dock.xml",
      missionBtFiles: { "tasks/dock.xml": "<root>design</root>" },
      missionBtLoadingPath: "tasks/dock.xml",
      localBtFileActionsDisabled: true,
      missionRunner: { status: RunnerStatus.IDLE, activeSpotId: "" },
      runVisibleSpots: [runSpot],
      runMissionBtFiles: {},
      btActiveNodesText: "Drive",
      ...callbacks,
      ...overrides,
    },
  });
}

test("builds the Design waypoint editor with the existing file action contract", () => {
  const { result } = renderComposition();
  const { waypointBtLayer, runBtLayer, activeBtLayer, runBtViewActive } = result.current;

  expect(waypointBtLayer.spot).toBe(selectedSpot);
  expect(activeBtLayer).toBe(waypointBtLayer);
  expect(runBtLayer).toBeNull();
  expect(runBtViewActive).toBe(false);
  expect(waypointBtLayer.editor.props).toMatchObject({
    title: "Dock Waypoint Task",
    filePath: "tasks/dock.xml",
    fileOptions: ["tasks/dock.xml", "tasks/other.xml"],
    defaultFilePath: "tasks/dock.xml",
    xml: "<root>design</root>",
    loading: true,
    activeNodeNames: [],
    onLoadXml: callbacks.loadMissionLocalBtXml,
    onSaveXml: callbacks.saveMissionLocalBtXml,
    onFilePathChange: callbacks.selectMissionLocalBtXml,
    onSaveXmlAs: callbacks.saveMissionLocalBtXmlAs,
    onSetDefaultXml: callbacks.setMissionLocalBtDefault,
    fileActionsDisabled: true,
    onXmlChange: callbacks.handleMissionLocalBtXmlChange,
  });
});

test("uses the default Design XML and id title when stored content and label are absent", () => {
  const unlabeled = { id: "wp-empty", x: 0, y: 0, yaw: 0 };
  const { result } = renderComposition({
    selectedBtLayerSpot: unlabeled,
    selectedBtLayerPath: "tasks/missing.xml",
    missionBtFiles: {},
    missionBtLoadingPath: "",
  });

  expect(result.current.waypointBtLayer.editor.props.title).toBe("wp-empty Waypoint Task");
  expect(result.current.waypointBtLayer.editor.props.xml).toContain("<root");
  expect(result.current.waypointBtLayer.editor.props.loading).toBe(false);
});

test("gates the waypoint layer to Design even when a waypoint remains selected", () => {
  const { result } = renderComposition({ workspaceStage: STAGE_NAVIGATE });

  expect(result.current).toEqual({
    waypointBtLayer: null,
    runBtLayer: null,
    activeBtLayer: null,
    runBtViewActive: false,
  });
});

test("builds the read-only Run layer only for the active RUNNING_BT waypoint", () => {
  const path = localBtPathForSpot(runSpot);
  const { result } = renderComposition({
    workspaceStage: STAGE_RUN,
    missionRunner: { status: RunnerStatus.RUNNING_BT, activeSpotId: runSpot.id },
    runMissionBtFiles: { [path]: "<root>run</root>" },
  });

  expect(result.current.waypointBtLayer).toBeNull();
  expect(result.current.runBtLayer.spot).toBe(runSpot);
  expect(result.current.activeBtLayer).toBe(result.current.runBtLayer);
  expect(result.current.runBtViewActive).toBe(true);
  expect(result.current.runBtLayer.editor.props).toEqual({
    xml: "<root>run</root>",
    activeNodeNames: ["Drive"],
  });
});

test("parses active Run node names without changing order, case, or duplicates", () => {
  const { result } = renderComposition({
    workspaceStage: STAGE_RUN,
    missionRunner: { status: RunnerStatus.RUNNING_BT, activeSpotId: runSpot.id },
    btActiveNodesText: " Drive, ,Check,Drive ",
  });

  expect(result.current.runBtLayer.editor.props.activeNodeNames).toEqual([
    "Drive", "Check", "Drive",
  ]);
});

test("uses the waypoint default XML when the active Run task has no stored file", () => {
  const { result } = renderComposition({
    workspaceStage: STAGE_RUN,
    missionRunner: { status: RunnerStatus.RUNNING_BT, activeSpotId: runSpot.id },
    runMissionBtFiles: {},
  });

  expect(result.current.runBtLayer.editor.props.xml).toContain("<root");
});

test.each([
  [RunnerStatus.IDLE, runSpot.id],
  [RunnerStatus.RUNNING_BT, "missing"],
])("does not expose a Run layer for status %s and spot %s", (status, activeSpotId) => {
  const { result } = renderComposition({
    workspaceStage: STAGE_RUN,
    missionRunner: { status, activeSpotId },
  });

  expect(result.current.runBtLayer).toBeNull();
  expect(result.current.runBtViewActive).toBe(false);
});
