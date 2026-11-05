// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { render, screen } from "@testing-library/react";
import { STAGE_AUTHORING, STAGE_MAPPING, STAGE_MAP_EDIT, STAGE_NAVIGATE, STAGE_RUN } from "../../lib/stages";
import MissionCanvasSceneSurface from "./MissionCanvasSceneSurface";

const mockMapViewer = jest.fn(() => <div data-testid="map-viewer" />);
jest.mock("../../../../components/navigation/MapViewer", () => ({
  MapViewer: (props) => mockMapViewer(props),
}));
jest.mock("../mapChrome", () => ({
  LayersPopover: () => <div data-testid="layers" />,
}));
jest.mock("../stages/DesignStageHud", () => () => <div data-testid="design-hud" />);
jest.mock("../stages/MapEditStageOverlay", () => () => <div data-testid="map-edit-hud" />);
jest.mock("../stages/MappingStageHud", () => () => <div data-testid="mapping-hud" />);
jest.mock("../stages/NavigationStage", () => ({
  NavigationStageHud: () => <div data-testid="navigate-hud" />,
}));
jest.mock("../stages/RunStage", () => ({
  RunStageHud: () => <div data-testid="run-hud" />,
}));

function props(overrides = {}) {
  const value = {
    stage: {
      id: STAGE_AUTHORING, mappingEditorActive: false, mappingTopicsActive: false,
      runTopicsActive: false, designLocalizationActive: false,
      navigationTopicsActive: false, designMapActive: true, runFamily: false,
      running: false,
    },
    scene: {
      displayedMap: { id: "map" }, globalCostmap: "global", localCostmap: "local",
      scan: "scan", mappingPoseSync: { scanPose: "mapping-pose" },
      runPoseSync: { scanPose: "run-pose" }, currentPose: "pose",
      navGoalPose: "goal", navGoalStatus: "driving", plan: "plan",
      footprint: "footprint", displayTf: "tf",
    },
    mission: {
      overlayActive: true, renderedVisibleSpots: [{ id: "wp" }], selectedSpotId: "wp",
      activeWaypointId: "active", followRobot: false,
      renderedBehaviorNodes: [{ id: "behavior" }], selectedBehaviorNodeId: "behavior",
      behaviorPreviewNode: null, renderedRouteView: { order: ["wp"], closed: false },
      routeMode: false, routeSourceId: "", mapLoaded: true, documentReady: true,
      designMapPath: "factory.pgm", mapName: "factory",
    },
    editors: {
      mapEditor: {
        annotations: ["edit-area"], selectedAnnotationId: "area", map: { id: "edit" },
        tool: "view", brushSize: 3, busy: false, selectedPath: "edit.pgm",
        editAreaAtMapPoint: jest.fn(), editAtMapPoint: jest.fn(),
        placeAnnotationAtMapArea: jest.fn(),
      },
      designMapEditor: { annotations: ["design-area"], busy: false, selectedPath: "factory.pgm" },
      runDisplayMapEditor: { annotations: ["run-area"], busy: false },
    },
    interaction: {
      busy: "", mode: "view", onRouteSpotClick: jest.fn(), onSpotClick: jest.fn(),
      onBehaviorNodeClick: jest.fn(), onRouteMapClick: jest.fn(),
      onSpotPoseChange: jest.fn(), onBehaviorNodePoseChange: jest.fn(),
      onMapClick: jest.fn(), onMapPose: jest.fn(), onBtLayerClose: jest.fn(),
    },
    bt: { waypointLayer: null, runLayer: null, activeLayer: null },
    layers: {
      active: { map: true, mapAreas: true, tf: true }, toggles: [{ id: "map" }],
      needsGlobalCostmap: true, needsLocalCostmap: true, needsScan: true,
      needsPlan: true, needsRobotModel: true, needsTf: true,
    },
    hud: { design: {}, mapping: {}, mapEdit: {}, run: {}, navigate: {} },
  };
  return { ...value, ...overrides };
}

function viewerProps() {
  return mockMapViewer.mock.calls.at(-1)[0];
}

beforeEach(() => mockMapViewer.mockClear());

test("Design ready exposes authoring overlays, edit callbacks, HUD, and layers", () => {
  render(<MissionCanvasSceneSurface {...props()} />);
  const map = viewerProps();
  expect(map.selectedSpotId).toBe("wp");
  expect(map.spots).toEqual([{ id: "wp" }]);
  expect(map.mapAnnotations).toEqual(["design-area"]);
  expect(map.onSpotClick).toBeDefined();
  expect(map.onSpotPoseChange).toBeDefined();
  expect(map.interactionDisabled).toBe(false);
  expect(screen.getByTestId("design-hud")).toBeInTheDocument();
  expect(screen.getByTestId("layers")).toBeInTheDocument();
});

test("Design loading disables interaction and a Task layer hides editing chrome", () => {
  const waypointLayer = { spot: { id: "wp" }, editor: <div /> };
  const input = props({
    mission: { ...props().mission, documentReady: false },
    bt: { waypointLayer, runLayer: null, activeLayer: waypointLayer },
  });
  render(<MissionCanvasSceneSurface {...input} />);
  const map = viewerProps();
  expect(map.interactionDisabled).toBe(true);
  expect(map.onSpotClick).toBeUndefined();
  expect(map.onMapClick).toBeUndefined();
  expect(map.btLayer).toBe(waypointLayer);
  expect(screen.queryByTestId("design-hud")).not.toBeInTheDocument();
  expect(screen.queryByTestId("layers")).not.toBeInTheDocument();
});

test("Mapping preserves live layers and renders only the Mapping HUD", () => {
  const input = props({
    stage: { ...props().stage, id: STAGE_MAPPING, designMapActive: false, mappingTopicsActive: true },
  });
  render(<MissionCanvasSceneSurface {...input} />);
  const map = viewerProps();
  expect(map.scanPose).toBe("mapping-pose");
  expect(map.globalCostmap).toBe("global");
  expect(screen.getByTestId("mapping-hud")).toBeInTheDocument();
  expect(screen.queryByTestId("design-hud")).not.toBeInTheDocument();
});

test("Map Edit suppresses live layers, forces view semantics, and wires editor callbacks", () => {
  const mapEditor = {
    ...props().editors.mapEditor, tool: "label_marker", busy: true,
  };
  const input = props({
    stage: { ...props().stage, id: STAGE_MAP_EDIT, mappingEditorActive: true, designMapActive: false },
    editors: { ...props().editors, mapEditor },
  });
  render(<MissionCanvasSceneSurface {...input} />);
  const map = viewerProps();
  expect(map.globalCostmap).toBeNull();
  expect(map.scan).toBeNull();
  expect(map.showMap).toBe(true);
  expect(map.interactionMode).toBe("view");
  expect(map.interactionDisabled).toBe(true);
  expect(map.editorAreaSelection).toBe(true);
  expect(map.onEditorMapArea).toBe(mapEditor.placeAnnotationAtMapArea);
  expect(screen.getByTestId("map-edit-hud")).toBeInTheDocument();
  expect(screen.queryByTestId("layers")).not.toBeInTheDocument();
});

test("Run uses runtime overlays, active waypoint, Run Task layer, and Run HUD", () => {
  const runLayer = { spot: { id: "active" }, editor: <div /> };
  const input = props({
    stage: { ...props().stage, id: STAGE_RUN, designMapActive: false, runFamily: true, runTopicsActive: true },
    bt: { waypointLayer: null, runLayer, activeLayer: runLayer },
  });
  render(<MissionCanvasSceneSurface {...input} />);
  const map = viewerProps();
  expect(map.activeWaypointId).toBe("active");
  expect(map.scanPose).toBe("run-pose");
  expect(map.btLayer).toBe(runLayer);
  expect(map.onSpotClick).toBeUndefined();
  expect(screen.getByTestId("run-hud")).toBeInTheDocument();
  expect(screen.queryByTestId("layers")).not.toBeInTheDocument();
});

test("Navigate alone exposes goal visualization and Navigate HUD", () => {
  const input = props({
    stage: { ...props().stage, id: STAGE_NAVIGATE, designMapActive: false, runFamily: true, navigationTopicsActive: true },
  });
  render(<MissionCanvasSceneSurface {...input} />);
  const map = viewerProps();
  expect(map.goalPose).toBe("goal");
  expect(map.showGoalPose).toBe(true);
  expect(map.pose).toBe("pose");
  expect(map.selectedSpotId).toBe("");
  expect(screen.getByTestId("navigate-hud")).toBeInTheDocument();
});
