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

// Structural assertions verify map/bubble ownership and positioning.
/* eslint-disable testing-library/no-container, testing-library/no-node-access */
// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, render, screen } from "@testing-library/react";
import { STAGE_COLLECT, STAGE_AUTHORING, STAGE_OPERATE } from "../../../mapWorkspace/lib/stages";
import AutonomySceneSurface from "./AutonomySceneSurface";
import { autonomyMapView } from "../autonomyMapView";

jest.mock("../LiveHeadCamera", () => () => <img alt="Live head camera"/>);

const mockMapViewer = jest.fn(() => <div data-testid="map-viewer" />);
jest.mock("../../../../components/navigation/MapViewer", () => ({
  MapViewer: (props) => mockMapViewer(props),
}));
jest.mock("../../../mapWorkspace/components/mapChrome", () => ({
  LayersPopover: () => <div data-testid="layers" />,
}));
jest.mock("./OperateStageHud", () => () => <div data-testid="run-hud" />);

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

test("Collect displays map waypoints and supports click navigation without editing", () => {
  const input = props();
  input.stage.id = STAGE_COLLECT;
  input.mission.overlayActive = false;
  input.hud.collect = {};
  const navigate = jest.fn();
  input.mission.collectWaypoints = { clickEnabled: true, spots: [{ id: "mission:wp" }], selectedId: "mission:wp", navigate };
  render(<AutonomySceneSurface {...input} />);
  expect(viewerProps().spots).toEqual([{ id: "mission:wp" }]);
  expect(viewerProps().onSpotClick).toBe(navigate);
  expect(viewerProps().selectedSpotId).toBe("mission:wp");
  expect(viewerProps().onSpotPoseChange).toBeUndefined();
  expect(viewerProps().goalPose).toBeNull();
  expect(viewerProps().globalCostmap).toBeNull();
  expect(viewerProps().localCostmap).toBeNull();
  expect(viewerProps().showGlobalCostmap).toBe(false);
  expect(viewerProps().showLocalCostmap).toBe(false);
});

test("Operate keeps the full map and attaches reasoning to the robot", () => {
  const input = props({
    stage: { ...props().stage, id: STAGE_OPERATE, designMapActive: false, runFamily: true },
    observation: { active: true, phase: "observing", result: { text: "People are seated.", elapsed_seconds: 1.2 } },
    hud: { ...props().hud, run: {} },
  });
  const view = render(<AutonomySceneSurface {...input} />);
  const surface = view.container.querySelector(".operate-scene-surface");
  const header = surface.querySelector(".operate-map-top-panels");
  const map = surface.querySelector(".operate-map-canvas");
  expect(header).not.toBeInTheDocument();
  expect(map).toBeInTheDocument();
  expect(Array.from(surface.children)).toEqual([map]);
  expect(map).toHaveTextContent("People are seated.");
  expect(viewerProps().onRobotScreenPositionChange).toEqual(expect.any(Function));
  expect(screen.queryByTestId("layers")).not.toBeInTheDocument();
  expect(mockMapViewer).toHaveBeenCalled();
  expect(screen.queryByTestId("layers")).not.toBeInTheDocument();
});

test('Record forwards observation pose editing to the same map handler as Design',()=>{
 const input=props();input.stage.id=STAGE_COLLECT;input.hud.collect={};
 const move=jest.fn();input.mission.collectWaypoints={onPoseChange:move};
 const view=render(<AutonomySceneSurface {...input}/>);
 viewerProps().onSpotPoseChange('tv',1,2,0.5);
 expect(move).toHaveBeenCalledWith('tv',1,2,0.5);
 input.mission.collectWaypoints={clickEnabled:true};
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().onSpotPoseChange).toBeUndefined();
});


test("Record empty-map clicks clear observation selection instead of Design selection", () => {
 const input=props(); input.stage.id=STAGE_COLLECT; input.hud.collect={};
 const clear=jest.fn(),select=jest.fn();
 input.mission.collectWaypoints={selectedId:"table",onSelect:select,onClearSelection:clear};
 const view=render(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().onMapClick).toBeUndefined();
 viewerProps().onMapTap();
 expect(clear).toHaveBeenCalledTimes(1);
 expect(select).not.toHaveBeenCalled();
 expect(input.interaction.onMapClick).not.toHaveBeenCalled();
 input.mission.collectWaypoints.selectedId="";
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().selectedSpotId).toBe("");
});


test("Record order mode dispatches sequence clicks without pose dragging",()=>{
 const input=props();input.stage.id=STAGE_COLLECT;input.hud.collect={};
 const select=jest.fn(),move=jest.fn();
 input.mission.collectWaypoints={orderMode:true,onSelect:select,onPoseChange:move,
  routeView:{order:[{id:"table",order:1},{id:"tv",order:2}],closed:false}};
 const view=render(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().missionRouteMode).toBe(true);
 expect(viewerProps().onSpotPoseChange).toBeUndefined();
 expect(viewerProps().missionRouteOrder).toEqual([{id:"table",order:1},{id:"tv",order:2}]);
 viewerProps().onMissionRouteSpotClick("tv");
 expect(select).toHaveBeenCalledWith("tv");
 expect(move).not.toHaveBeenCalled();
 input.mission.collectWaypoints.orderMode=false;
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().missionRouteMode).toBe(false);
 expect(viewerProps().onSpotPoseChange).toBe(move);
});


test("Record clears global plan on Stop and waits for a new path on restart",()=>{
 const input=props();input.stage.id=STAGE_COLLECT;input.hud.collect={};
 input.layers.needsPlan=true;
 input.mission.collectWaypoints={navigationActive:false};
 input.scene.plan={poses:["old path"]};
 const view=render(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBeNull();
 input.mission.collectWaypoints.navigationActive=true;
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBeNull();
 input.scene.plan={poses:["current path"]};
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBe(input.scene.plan);
 expect(viewerProps().showGlobalPlan).toBe(true);
 input.mission.collectWaypoints.navigationActive=false;
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBeNull();
 expect(viewerProps().showGlobalPlan).toBe(false);
 input.scene.plan={poses:["late path after Stop"]};
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBeNull();
 input.mission.collectWaypoints.navigationActive=true;
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBeNull();
 input.scene.plan={poses:["new goal path"]};
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBe(input.scene.plan);
});


test("Record mission overlay appears separately and disappears after completion",()=>{
 const input=props();input.stage.id=STAGE_COLLECT;input.hud.collect={};
 const observations=[{id:"observe"}];const overlay={spots:[{id:"task"}],pathSpots:[],order:[]};
 input.mission.collectWaypoints={spots:observations,missionOverlay:overlay};
 const view=render(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().spots).toBe(observations);
 expect(viewerProps().missionRouteOverlay).toBe(overlay);
 input.mission.collectWaypoints.missionOverlay=null;
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().missionRouteOverlay).toBeNull();
 expect(viewerProps().spots).toBe(observations);
});


test("Operate shares Record colors, observations, child overlay and costmap visibility", () => {
 const input=props(); input.stage.id=STAGE_OPERATE;
 const state={active:true,mode:"mission",phase:"executing",mission:{mission_id:"clean"},
  observation:{spot_id:"observe",pose:{x:0,y:0}}, runner:{phase:"navigation",activeSpotId:"table"},
  mission_route:[{id:"table",pose:{x:2,y:0}}],
  config:{draft:{mode:"patrol",route_closed:true,observations:[{spot_id:"observe"},{spot_id:"tv"}]},
   spots:[{id:"observe",pose:{x:0,y:0}},{id:"tv",pose:{x:3,y:1}}]}};
 input.mission.operateWaypoints=autonomyMapView(state);
 const view=render(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().spots.map(spot=>spot.id)).toEqual(["observe","tv"]);
 expect(viewerProps().waypointColors).toEqual({idle:0x8d7899});
 expect(viewerProps().missionRouteOverlay.activeWaypointId).toBe("table");
 expect(viewerProps().missionRouteClosed).toBe(true);
 expect(viewerProps().globalCostmap).toBeNull();
 expect(viewerProps().localCostmap).toBeNull();
 expect(viewerProps().showGlobalCostmap).toBe(false);
 expect(viewerProps().showLocalCostmap).toBe(false);
 input.mission.operateWaypoints=autonomyMapView({...state,mode:"observation",phase:"observing",runner:{},mission:null});
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().missionRouteOverlay).toBeNull();
 expect(viewerProps().activeWaypointId).toBe("observe");
 expect(viewerProps().spots).toHaveLength(2);
 expect(viewerProps().missionRouteClosed).toBe(true);
});

test("Operate clears global plan at pause/stop and when a different navigation leg starts",()=>{
 const input=props();input.stage.id=STAGE_OPERATE;
 input.mission.operateWaypoints={navigationActive:false};
 const view=render(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBeNull();
 input.mission.operateWaypoints={navigationActive:true,navigationKey:"patrol:tv"};
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBeNull();
 input.scene.plan={poses:["fresh patrol path"]};
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBe(input.scene.plan);
 input.mission.operateWaypoints={navigationActive:true,navigationKey:"mission:table"};
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBeNull();
 input.scene.plan={poses:["fresh mission path"]};
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBe(input.scene.plan);
 input.mission.operateWaypoints={navigationActive:false};
 view.rerender(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().plan).toBeNull();expect(viewerProps().showGlobalPlan).toBe(false);
});

 test.each([STAGE_COLLECT, STAGE_OPERATE])("%s uses only lightweight telemetry layers",stageId=>{
  const input=props();input.stage.id=stageId;input.hud.collect={};
  render(<AutonomySceneSurface {...input}/>);
  const map=viewerProps();
  expect(map.renderProfile).toBe("autonomy");
  expect(map.showMap).toBe(true);expect(map.showScan).toBe(true);
  expect(map.showFootprint).toBe(true);expect(map.footprint).toBe("footprint");
  expect(map.showTf).toBe(false);expect(map.showRobotModel).toBe(false);
  expect(map.mapAnnotations).toEqual([]);expect(map.behaviorNodes).toEqual([]);
  expect(map.btLayer).toBeNull();expect(map.showGoalPose).toBe(false);
  // TF still transforms the lidar/footprint into the map frame.
  expect(map.tf).toBe("tf");
 });
 test("moving the robot thought does not re-render MapViewer or map HUD",()=>{
  const input=props();input.stage.id=STAGE_OPERATE;
  input.observation={active:true,phase:"observing",result:{text:"Waiting."}};
  const view=render(<AutonomySceneSurface {...input}/>);
  const publish=viewerProps().onRobotScreenPositionChange;
  const renders=mockMapViewer.mock.calls.length;
  act(()=>publish({x:600,y:650,width:1200,height:900}));
  expect(view.container.querySelector(".operate-thought-bubble")).not.toHaveClass("is-docked");
  act(()=>publish({x:620,y:650,width:1200,height:900}));
  expect(mockMapViewer).toHaveBeenCalledTimes(renders);
  expect(view.container.querySelector(".operate-thought-bubble").style.left).toBe("380px");
 });

test("passes the selected robot identity to the lightweight map",()=>{
 const input=props();input.stage.id=STAGE_OPERATE;input.scene.robotType="ffw_sg2_rev1";
 render(<AutonomySceneSurface {...input}/>);
 expect(viewerProps().robotType).toBe("ffw_sg2_rev1");
 expect(viewerProps().renderProfile).toBe("autonomy");
});


test("only Operate forwards the live navigation destination to the map", () => {
  const destination = { id: "kitchen", label: "Kitchen", pose: { x: 1, y: 2 }, scope: "observation" };
  const input = props();
  input.stage.id = STAGE_OPERATE;
  input.mission.operateWaypoints = { navigationActive: true, navigationDestination: destination };
  const view = render(<AutonomySceneSurface {...input}/>);
  expect(viewerProps().navigationDestination).toBe(destination);
  input.mission.operateWaypoints = { navigationActive: false, navigationDestination: null };
  view.rerender(<AutonomySceneSurface {...input}/>);
  expect(viewerProps().navigationDestination).toBeNull();
  input.stage.id = STAGE_COLLECT;
  input.hud.collect = {};
  input.mission.collectWaypoints = { navigationActive: true, navigationDestination: destination };
  view.rerender(<AutonomySceneSurface {...input}/>);
  expect(viewerProps().navigationDestination).toBeNull();
});
