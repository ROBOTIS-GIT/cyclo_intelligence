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
import OperateObservation from "./OperateObservation";

test("clears the previous frame and decision when autonomy stops", () => {
  const result = { image_base64: "test", decision: { decision: "start", reason: "Table is empty." }, elapsed_seconds: 1 };
  const { rerender } = render(<OperateObservation observation={{ phase: "watching", result }} />);
  expect(screen.getByAltText("Head camera frame analyzed by Qwen")).toBeInTheDocument();
  for (const phase of ["stopping", "stopped"]) {
    rerender(<OperateObservation observation={{ phase, result }} />);
    expect(screen.queryByAltText("Head camera frame analyzed by Qwen")).not.toBeInTheDocument();
    expect(screen.queryByText("Table is empty.")).not.toBeInTheDocument();
  }
  expect(screen.getByText("Stopped")).toBeInTheDocument();
});

test("keeps supervision visible during a mission pause", () => {
 render(<OperateObservation observation={{active:true,mode:"mission",phase:"paused",paused:true,pause_reason:"ai",
  reasoning_task:"mission_supervision",result:{reasoning_task:"mission_supervision",image_base64:"test",
   decision:{decision:"pause",mission_id:"clean",reason:"A person reaches into the workspace."},
   selected_mission:{mission_name:"CleanTable"},elapsed_seconds:1}}} />);
 expect(screen.getByText("Paused")).toBeInTheDocument();
 expect(screen.getByText("Pause · CleanTable")).toBeInTheDocument();
 expect(screen.getByAltText("Head camera frame analyzed by Qwen")).toBeInTheDocument();
});

test("does not display an old selection result as current mission supervision", () => {
 render(<OperateObservation observation={{active:true,mode:"mission",phase:"executing",result:{
  decision:{decision:"start",reason:"Old selection result"}}}} />);
 expect(screen.queryByText("Old selection result")).not.toBeInTheDocument();
 expect(screen.getByText("Waiting for the next reasoning result.")).toBeInTheDocument();
 expect(screen.getByText("Mission")).toBeInTheDocument();
});

test.each([['error', 'Failed'], ['stopping', 'Stopping'], ['stopped', 'Stopped']])(
 'terminal %s overrides a retained pause and hides stale supervision', (phase, label) => {
  render(<OperateObservation observation={{active:false,mode:'mission',phase,paused:true,
   reasoning_task:'mission_supervision',result:{reasoning_task:'mission_supervision',image_base64:'test',
    decision:{decision:'continue',mission_id:'clean',reason:'Previously clear'}}}} />);
  expect(screen.getByText(label)).toBeInTheDocument();
  expect(screen.queryByText('Paused')).not.toBeInTheDocument();
  expect(screen.queryByText('Previously clear')).not.toBeInTheDocument();
  expect(screen.queryByAltText('Head camera frame analyzed by Qwen')).not.toBeInTheDocument();
 },
);


test("shows mission cancellation, then clears it when return patrol reasoning takes over", () => {
 const result={reasoning_task:"mission_supervision",decision:{decision:"cancel",mission_id:"clean",reason:"The task is no longer needed."},
  context:{active_mission:{mission_id:"clean",mission_name:"CleanTable"}}};
 const {rerender}=render(<OperateObservation observation={{active:true,mode:"mission",phase:"executing",reasoning_task:"mission_supervision",result}} />);
 expect(screen.getByText("Cancel mission · CleanTable")).toBeInTheDocument();
 rerender(<OperateObservation observation={{active:true,mode:"observation",phase:"returning",reasoning_task:"patrol_supervision",result}} />);
 expect(screen.getByText("Returning")).toBeInTheDocument();
 expect(screen.queryByText("Cancel mission · CleanTable")).not.toBeInTheDocument();
});


test.each([
  ["patrol_navigation", "patrol"], ["return_navigation", "patrol"], ["return_posture", "patrol"],
  ["observing", "observation"], ["mission_navigation", "mission"], ["mission_action", "mission"],
])("highlights acknowledged %s mode even when paused", (activity, mode) => {
  const {container} = render(<OperateObservation observation={{active:true, phase:"paused", paused:true,
    execution:{lifecycle:"active",activity,control:"paused"}}}/>);
  expect(container.querySelector('[aria-current="step"]')).toHaveClass(`mode-${mode}`);
  expect(screen.getByRole("status")).toHaveTextContent("Paused");
});

test("start proposal does not mark Mission active or expose model metadata", () => {
  const {container}=render(<OperateObservation observation={{active:true,phase:"observing",
    execution:{lifecycle:"active",activity:"observing",control:"running"}, observation:{label:"KitchenObservation"},
    result:{decision:{decision:"start",reason:"Empty dishes remain."},selected_mission:{mission_name:"CleanTable"},
      model_name:"private_model_name",adapter:"/adapter",elapsed_seconds:3}}}/>);
  expect(container.querySelector('[aria-current="step"]')).toHaveClass("mode-observation");
  expect(screen.getByText("Start mission · CleanTable")).toBeInTheDocument();
  expect(screen.getByText("At KitchenObservation")).toBeInTheDocument();
  expect(screen.queryByText(/private_model_name|LoRA|3.0s/)).not.toBeInTheDocument();
});

test("terminal lifecycle wins over retained mission phase and context", () => {
  const {container}=render(<OperateObservation observation={{phase:"executing",mode:"mission",
    mission:{mission_name:"Old mission"},execution:{lifecycle:"stopped",activity:"mission_action",control:"paused"}}}/>);
  expect(container.querySelector('[aria-current="step"]')).toBeNull();
  expect(screen.getByRole("status")).toHaveTextContent("Stopped");
  expect(screen.queryByText("Old mission")).not.toBeInTheDocument();
});

test("shows admitted mission and current waypoint without using the prediction", () => {
  render(<OperateObservation observation={{phase:"executing",active:true,mode:"mission",
    mission:{mission_name:"CleanTable"},mission_route:[{label:"Table"}],runner:{currentIndex:0},
    execution:{lifecycle:"active",activity:"mission_action",control:"running"}}}/>);
  expect(screen.getByText("CleanTable · Table")).toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent("Running task");
});
