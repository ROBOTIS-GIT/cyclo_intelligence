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

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import OperateSidebar from "./OperateSidebar";
import { autonomyRequest } from "../hooks/autonomyApi";
jest.mock("../hooks/autonomyApi", () => ({ autonomyRequest: jest.fn() }));
beforeEach(() => {
  autonomyRequest.mockReset();
  autonomyRequest.mockResolvedValue({ messages: [], pending: false });
});
test("requires map/localization and shows three panels", () => {
  render(<OperateSidebar mapLoaded={false} poseReady running />);
  for (const title of ["Current situation", "Activity", "Conversation"]) expect(screen.getByText(title)).toBeInTheDocument();
  expect(screen.getByText("Map required")).toBeInTheDocument();
  expect(screen.getByRole("button", {name:"Send message"})).toBeDisabled();
});
test("sends language instruction and exposes autonomy stop", async () => {
  const stop=jest.fn();
  render(<OperateSidebar mapLoaded poseReady running observation={{active:true,mission:{mission_name:"Mission2"},stop,needsStop:true,runner:{currentIndex:0,total:5}}} />);
  fireEvent.change(screen.getByRole("textbox"),{target:{value:"Wait for now"}});
  fireEvent.click(screen.getByRole("button",{name:"Send message"}));
  expect(autonomyRequest).toHaveBeenCalledWith("/autonomy/conversation", "POST", expect.objectContaining({ text: "Wait for now", request_id: expect.any(String) }));
  await waitFor(() => expect(screen.getByRole("textbox")).toHaveValue(""));
  fireEvent.click(screen.getByRole("button",{name:"Stop autonomy"}));
  expect(stop).toHaveBeenCalledTimes(1);
});
test("does not report active supervision after autonomy stops", () => {
  render(<OperateSidebar mapLoaded poseReady running observation={{active:false,phase:"stopped",mission:{mission_name:"Mission2"},runner:{currentIndex:1,total:5}}} />);
  expect(screen.getByText("Autonomy idle")).toBeInTheDocument();
  expect(screen.queryByText("Qwen · Stopped")).not.toBeInTheDocument();
  expect(screen.getByRole("button",{name:"Send message"})).toBeDisabled();
});

test("shows the saved return spot and manual pause/resume without stopping Qwen", () => {
 const pause=jest.fn(),resume=jest.fn();
 const observation={active:true,mode:"mission",phase:"executing",pause,resume,
  resume_checkpoint:{return_spot_id:"dining"},config:{spots:[{id:"dining",label:"Dining Front"}]}};
 const {rerender}=render(<OperateSidebar observation={observation} />);
 expect(screen.getByText("Dining Front")).toBeInTheDocument();
 fireEvent.click(screen.getByRole("button",{name:"Pause autonomy"}));
 expect(pause).toHaveBeenCalledTimes(1);
 rerender(<OperateSidebar observation={{...observation,phase:"paused",paused:true}} />);
 fireEvent.click(screen.getByRole("button",{name:"Resume autonomy"}));
 expect(resume).toHaveBeenCalledTimes(1);
});

test("an AI-only pause lets the operator add a manual hold rather than offering ineffective resume",()=>{
 const pause=jest.fn(),resume=jest.fn();
 render(<OperateSidebar observation={{active:true,mode:"mission",phase:"executing",paused:true,
  pause_causes:{ai:"Person near robot"},pause,resume}} />);
 expect(screen.queryByRole("button",{name:"Resume autonomy"})).not.toBeInTheDocument();
 fireEvent.click(screen.getByRole("button",{name:"Pause autonomy"}));
 expect(pause).toHaveBeenCalledTimes(1);
 expect(resume).not.toHaveBeenCalled();
});


test("observation plan is editable in Operate", () => {
 const edit=jest.fn(), save=jest.fn();
 render(<OperateSidebar observation={{config:{draft:{mode:"fixed",observations:[],excluded_mission_ids:[]},dirty:true,edit,save}}}/>);
 fireEvent.change(screen.getByLabelText("Observation mode"),{target:{value:"patrol"}});
 expect(edit).toHaveBeenCalledWith({mode:"patrol"});
 fireEvent.click(screen.getByLabelText("Save observation plan"));
 expect(save).toHaveBeenCalledTimes(1);
});
