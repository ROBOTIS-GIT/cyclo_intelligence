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

import { render, screen } from "@testing-library/react";
import OperateThoughtAction, { thoughtAction } from "./OperateThoughtAction";

const active = (activity, rest = {}) => ({ active:true, phase:"executing", ...rest,
  execution:{lifecycle:"active",activity,control:"running",owns_execution:true,...rest.execution} });

test("shows the actual action with a decorative icon and no proposed task", () => {
  const {container} = render(<OperateThoughtAction observation={active("observing", {
    result:{decision:{decision:"start"},selected_mission:{mission_name:"ServeDish"}},
  })}/>);
  expect(screen.getByRole("status", {name:"Current action"})).toHaveTextContent("Looking around");
  expect(container.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
  expect(screen.queryByText(/Serving food|Mission|Observation|Patrol/)).not.toBeInTheDocument();
});

test("follows physical task admission, navigation, action, and return", () => {
  const state = {mission:{mission_name:"ServeDish"},mission_route:[{id:"table",label:"DiningTable"}],
    observation:{spot_id:"kitchen",label:"KitchenObservation"},runner:{currentIndex:0,activeSpotId:"table",phase:"waypoint-task"}};
  const {rerender} = render(<OperateThoughtAction observation={active("idle",state)}/>);
  expect(screen.getByRole("status")).toHaveTextContent("Getting ready");
  rerender(<OperateThoughtAction observation={active("mission_navigation", {...state,execution:{target_spot_id:"table"}})}/>);
  expect(screen.getByRole("status")).toHaveTextContent("Moving to Dining Table");
  rerender(<OperateThoughtAction observation={active("mission_action",state)}/>);
  expect(screen.getByRole("status")).toHaveTextContent("Serving food");
  rerender(<OperateThoughtAction observation={active("return_navigation", {...state,execution:{target_spot_id:"kitchen"}})}/>);
  expect(screen.getByRole("status")).toHaveTextContent("Returning to Kitchen");
});

test.each(["patrol_navigation", "mission_navigation", "mission_action", "observing"])(
  "pause and errors override retained %s activity", activity => {
    const state = active(activity, {mission:{mission_name:"ServeDish"},execution:{control:"paused"}});
    expect(thoughtAction(state).label).toBe("Paused");
    expect(thoughtAction({...state,error:"Navigation failed"})).toMatchObject({label:"Needs attention",tone:"danger"});
    expect(thoughtAction({...state,execution:{...state.execution,lifecycle:"stopped"}}).label).toBe("Stopped");
  });

test("authoritative control takes precedence over a retained compatibility pause", () => {
  expect(thoughtAction(active("mission_action", {paused:true,mission:{mission_name:"CleanTable"}})).label).toBe("Cleaning the table");
  expect(thoughtAction(active("mission_action", {execution:{control:"pausing"}})).label).toBe("Pausing");
  expect(thoughtAction(active("mission_action", {execution:{control:"cancelling"}})).label).toBe("Stopping");
});

test("resolves the acknowledged destination without borrowing a stale waypoint", () => {
  const state = active("mission_navigation", {execution:{target_spot_id:"new"},
    mission_route:[{id:"old",label:"Old table"},{id:"new",label:"Kitchen"}],
    runner:{currentIndex:0,activeSpotId:"old",label:"Old table"}});
  expect(thoughtAction(state).label).toBe("Moving to Kitchen");
  expect(thoughtAction({...state,mission_route:[]}).label).toBe("Moving");
  expect(thoughtAction({phase:"executing",mission_route:[{label:"Kitchen"}],runner:{phase:"navigation",currentIndex:0}}).label).toBe("Moving to Kitchen");
});

test("legacy runner state can describe movement and work, but a selection cannot", () => {
  expect(thoughtAction({phase:"executing",mission:{mission_name:"ServeDish"},runner:{phase:"mission-admission"}}).label).toBe("Getting ready");
  expect(thoughtAction({phase:"executing",mission:{mission_name:"ServeDish"},runner:{phase:"waypoint-task"}}).label).toBe("Serving food");
  expect(thoughtAction({phase:"positioning",observation:{label:"Kitchen"}}).label).toBe("Moving to Kitchen");
  expect(thoughtAction({phase:"observing",result:{selected_mission:{mission_name:"ServeDish"}}}).label).toBe("Looking around");
});

test("unknown tasks stay concise without exposing internal names", () => {
  expect(thoughtAction(active("mission_action", {mission:{mission_name:"InternalMission42"}})).label).toBe("Working");
  expect(thoughtAction(active("observation_posture")).label).toBe("Getting ready");
  expect(thoughtAction({phase:"error",paused:true}).label).toBe("Needs attention");
  expect(thoughtAction().label).toBe("Ready");
});


test.each(["patrol_navigation", "mission_navigation", "return_navigation"])(
  "%s does not borrow a destination when its snapshot target is absent", activity => {
    const retained = {mission_route:[{id:"stale",label:"Old table"}],
      observation:{spot_id:"stale",label:"Old kitchen"},resume_checkpoint:{return_spot_id:"stale"},
      runner:{currentIndex:0,activeSpotId:"stale",label:"Old table"}};
    for (const target_spot_id of [null, undefined]) {
      expect(thoughtAction(active(activity,{...retained,execution:{target_spot_id,step_index:0}})).label)
        .toBe(activity === "return_navigation" ? "Returning" : "Moving");
    }
  });

test("navigation destinations use the correct waypoint namespace", () => {
  const state = {mission_route:[{id:"shared",label:"Task table"}],
    config:{spots:[{id:"shared",label:"Kitchen"}]},
    runner:{activeSpotId:"shared",label:"Task table"},execution:{target_spot_id:"shared"}};
  expect(thoughtAction(active("patrol_navigation",state)).label).toBe("Moving to Kitchen");
  expect(thoughtAction(active("return_navigation",state)).label).toBe("Returning to Kitchen");
  expect(thoughtAction(active("mission_navigation",state)).label).toBe("Moving to Task table");
  expect(thoughtAction(active("return_navigation", {...state,config:{spots:[]}})).label).toBe("Returning");
  expect(thoughtAction(active("mission_navigation", {...state,mission_route:[],runner:{}})).label).toBe("Moving");
});

test.each(["patrol_navigation", "mission_navigation", "mission_action", "observing"])(
  "unowned or idle snapshots cannot announce retained %s activity", activity => {
    const state = {mission:{mission_name:"ServeDish"}};
    expect(thoughtAction(active(activity,{...state,execution:{owns_execution:false}})).label).toBe("Ready");
    expect(thoughtAction(active(activity,{...state,execution:{control:"idle"}})).label).toBe("Ready");
    expect(thoughtAction(active(activity,{...state,execution:{owns_execution:false,error:"Disconnected"}})).label).toBe("Needs attention");
    expect(thoughtAction(active(activity,{...state,execution:{owns_execution:false,lifecycle:"stopped"}})).label).toBe("Stopped");
  });

test("inactive legacy state cannot announce a retained runner task", () => {
  expect(thoughtAction({active:false,phase:"executing",mission:{mission_name:"ServeDish"},runner:{phase:"waypoint-task"}}).label).toBe("Ready");
});
