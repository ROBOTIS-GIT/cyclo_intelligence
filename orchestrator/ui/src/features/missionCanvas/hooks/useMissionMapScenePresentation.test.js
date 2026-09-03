import { act, renderHook } from "@testing-library/react";
import useMissionMapScenePresentation, {
  deriveMissionRoutePresentation,
} from "./useMissionMapScenePresentation";
import { LAYER_PRESETS } from "../lib/layers";
import {
  STAGE_AUTHORING,
  STAGE_MAPPING,
  STAGE_MAP_EDIT,
  STAGE_RUN,
} from "../lib/stages";

const grid = (originX) => ({
  info: {
    resolution: 1,
    width: 10,
    height: 10,
    origin: {
      position: { x: originX, y: 0 },
      orientation: { x: 0, y: 0, z: 0, w: 1 },
    },
  },
  data: [],
});

const legacySpot = (id) => ({ id, pose: { x: 5, y: 5, yaw: 0 } });
const emptyRoute = { order: [], closed: false, treeSpots: [], executionSpots: [] };

function ports(overrides = {}) {
  return {
    workspaceStage: STAGE_AUTHORING,
    designMapName: "factory",
    runtimeMapName: "runtime",
    designMapSample: { identity: "factory.pgm", map: grid(100) },
    runMapSample: { identity: "runtime.pgm", map: grid(200) },
    designMapAvailable: true,
    designMapBusy: false,
    designMissionLoadPhase: "idle",
    designMissionLoadError: "",
    displayedMap: grid(0),
    runSessionActive: false,
    designSpots: [legacySpot("design")],
    runSpots: [legacySpot("run")],
    designBehaviorNodes: [{ id: "design-node" }],
    runBehaviorNodes: [{ id: "run-node" }],
    runMissionFlowNodes: [],
    runMissionFlowEdges: [],
    activeLayers: LAYER_PRESETS[STAGE_AUTHORING],
    setLayersByStage: jest.fn(),
    ...overrides,
  };
}

test.each([
  [false, false, "idle", "", true],
  [true, false, "idle", "", false],
  [false, true, "idle", "", false],
  [false, false, "loading", "", false],
  [false, false, "idle", "load failed", false],
])("gates Design readiness for available=%s busy=%s phase=%s error=%s", (
  unavailable, busy, phase, error, ready,
) => {
  const { result } = renderHook(() => useMissionMapScenePresentation(ports({
    designMapAvailable: !unavailable,
    designMapBusy: busy,
    designMissionLoadPhase: phase,
    designMissionLoadError: error,
  })));
  expect(result.current.designDocumentReady).toBe(ready);
  expect(result.current.missionOverlayActive).toBe(ready);
  expect(result.current.designPanelSpots).toHaveLength(ready ? 1 : 0);
  expect(result.current.designPanelBehaviorNodes).toHaveLength(ready ? 1 : 0);
});

test("shows a Run overlay only when the displayed map exists", () => {
  const { result, rerender } = renderHook(({ displayedMap }) => (
    useMissionMapScenePresentation(ports({
      workspaceStage: STAGE_RUN,
      runSessionActive: true,
      displayedMap,
    }))
  ), { initialProps: { displayedMap: null } });
  expect(result.current.missionOverlayActive).toBe(false);
  expect(result.current.renderedVisibleSpots[0].id).toBe("run");
  expect(result.current.renderedBehaviorNodes[0].id).toBe("run-node");
  rerender({ displayedMap: grid(0) });
  expect(result.current.missionOverlayActive).toBe(true);
});

test("does not reuse or relabel a cached map across map identity changes", () => {
  const firstMap = grid(100);
  const { result, rerender } = renderHook((props) => (
    useMissionMapScenePresentation(ports(props))
  ), {
    initialProps: {
      designMapName: "factory",
      designMapSample: { identity: "factory.pgm", map: firstMap },
    },
  });
  expect(result.current.designMissionMap).toBe(firstMap);
  expect(result.current.designVisibleSpots[0].pose.x).toBe(105);

  rerender({
    designMapName: "warehouse",
    // MapEditor can still expose the prior image during the new request.
    designMapSample: { identity: "factory.pgm", map: firstMap },
  });
  expect(result.current.designMissionMap).toBeNull();
  expect(result.current.designVisibleSpots[0].pose.x).toBe(5);

  const replacement = grid(300);
  rerender({
    designMapName: "warehouse",
    designMapSample: { identity: "warehouse.pgm", map: replacement },
  });
  expect(result.current.designMissionMap).toBe(replacement);
  expect(result.current.designVisibleSpots[0].pose.x).toBe(305);
});

test("keeps Design and Run projection and route selectors independent", () => {
  const secondRunSpot = { ...legacySpot("run-2"), metadata: { coordinate_space: "map" } };
  const runNodes = [{ id: "run" }, { id: "run-2" }];
  const runEdges = [{ source: "run", target: "run-2" }];
  const { result } = renderHook(() => useMissionMapScenePresentation(ports({
    workspaceStage: STAGE_RUN,
    runSessionActive: true,
    runSpots: [legacySpot("run"), secondRunSpot],
    runMissionFlowNodes: runNodes,
    runMissionFlowEdges: runEdges,
  })));
  expect(result.current.designVisibleSpots[0].pose.x).toBe(105);
  expect(result.current.runVisibleSpots[0].pose.x).toBe(205);
  expect(result.current.runRouteView.order).toEqual([
    { id: "run", order: 1 },
    { id: "run-2", order: 2 },
  ]);
});

test.each([
  [false, true, "design", 1, true],
  [false, false, "design", 0, false],
  [true, true, "run", 1, true],
  [true, false, "run", 0, false],
])("derives late route presentation for run=%s ready=%s", (
  runSessionActive, designDocumentReady, renderedId, panelCount, panelClosed,
) => {
  const designRouteView = {
    ...emptyRoute,
    id: "design",
    treeSpots: [{ id: "design-spot" }],
    closed: true,
  };
  const runRouteView = { ...emptyRoute, id: "run" };
  const result = deriveMissionRoutePresentation({
    runSessionActive,
    designDocumentReady,
    runRouteView,
    designRouteView,
  });
  expect(result.renderedRouteView.id).toBe(renderedId);
  expect(result.designPanelRouteSpots).toHaveLength(panelCount);
  expect(result.designPanelRouteClosed).toBe(panelClosed);
});

test.each([
  [STAGE_MAPPING, "scan"],
  [STAGE_AUTHORING, "mapAreas"],
])("builds only %s stage layer toggles", (stage, changedLayer) => {
  const setLayersByStage = jest.fn();
  const { result } = renderHook(() => useMissionMapScenePresentation(ports({
    workspaceStage: stage,
    activeLayers: LAYER_PRESETS[stage],
    setLayersByStage,
  })));
  expect(result.current.layerToggles.map(({ id }) => id)).toEqual(
    expect.arrayContaining([changedLayer]),
  );
  act(() => result.current.layerToggles.find(({ id }) => id === changedLayer).onChange(false));
  const previous = {
    [STAGE_MAPPING]: { ...LAYER_PRESETS[STAGE_MAPPING] },
    [STAGE_AUTHORING]: { ...LAYER_PRESETS[STAGE_AUTHORING] },
  };
  const updater = setLayersByStage.mock.calls[0][0];
  const next = updater(previous);
  expect(next[stage][changedLayer]).toBe(false);
  const otherStage = stage === STAGE_MAPPING ? STAGE_AUTHORING : STAGE_MAPPING;
  expect(next[otherStage]).toEqual(previous[otherStage]);
});

test("exposes no layer toggles in Map Edit", () => {
  const { result } = renderHook(() => useMissionMapScenePresentation(ports({
    workspaceStage: STAGE_MAP_EDIT,
    activeLayers: LAYER_PRESETS[STAGE_MAP_EDIT],
  })));
  expect(result.current.layerToggles).toEqual([]);
});
