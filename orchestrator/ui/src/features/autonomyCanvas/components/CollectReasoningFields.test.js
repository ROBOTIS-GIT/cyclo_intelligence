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

import { fireEvent, render, screen } from "@testing-library/react";
import CollectReasoningFields from "./CollectReasoningFields";

test("shows a deleted historical mission without offering it as a current choice", () => {
  const selectedFrame = { frame_id: "frame-1", context: {} };
  render(<CollectReasoningFields
    draft={{ active: false, selectedFrame }}
    editor={{
      decision: "start",
      setDecision: jest.fn(),
      missionId: "deleted-id",
      setMissionId: jest.fn(),
      reason: "Legacy decision",
      setReason: jest.fn(),
      saving: false,
      maskBusy: false,
      receipt: "",
      loadedFrame: "frame-1",
      busy: false,
      missions: [{ mission_id: "observe-id", mission_name: "ObservationMode" }],
      deletedSelectedMission: { mission_id: "deleted-id", mission_name: "Mission2" },
      selectedMissionUnavailable: true,
      selectedLabel: null,
      frameMission: null,
      invalidDecisionContext: false,
      missionRequired: true,
    }}
    onSave={jest.fn()}
  />);

  const option = screen.getByRole("option", { name: "Mission2 (deleted)" });
  expect(option).toBeDisabled();
  expect(option.selected).toBe(true);
  expect(screen.getByRole("option", { name: "ObservationMode" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  expect(screen.getByText("Selected mission is no longer available in this frame's candidates. Select an available mission."))
    .toBeInTheDocument();
});

function renderFields(overrides = {}) {
  const setDecision = jest.fn();
  const setMissionId = jest.fn();
  const selectedFrame = { frame_id: "frame-1", context: {} };
  render(<CollectReasoningFields
    draft={{ active: false, selectedFrame }}
    editor={{
      decision: "unknown",
      setDecision,
      missionId: "",
      setMissionId,
      reason: "Visible evidence",
      setReason: jest.fn(),
      saving: false,
      maskBusy: false,
      receipt: "",
      loadedFrame: "frame-1",
      busy: false,
      missions: [{ mission_id: "observe-id", mission_name: "ObservationMode" }],
      deletedSelectedMission: null,
      selectedMissionUnavailable: false,
      selectedLabel: null,
      frameMission: null,
      invalidDecisionContext: false,
      missionRequired: false,
      ...overrides,
    }}
    onSave={jest.fn()}
  />);
  return { setDecision, setMissionId };
}

test("places Expected decision before Mission and disables Mission when none is required", () => {
  renderFields();

  const decision = screen.getByRole("combobox", { name: "Expected decision" });
  const mission = screen.getByRole("combobox", { name: "Mission" });
  expect(decision.compareDocumentPosition(mission) & Node.DOCUMENT_POSITION_FOLLOWING)
    .toBeTruthy();
  expect(mission).toBeDisabled();
  expect(screen.getByRole("option", { name: "No mission required" })).toBeInTheDocument();
});

test("Wait clears a previously selected mission", () => {
  const { setDecision, setMissionId } = renderFields({
    decision: "start",
    missionId: "observe-id",
    missionRequired: true,
  });

  fireEvent.change(screen.getByRole("combobox", { name: "Expected decision" }), {
    target: { value: "wait" },
  });
  expect(setDecision).toHaveBeenCalledWith("wait");
  expect(setMissionId).toHaveBeenCalledWith("");
});

test("Start allows a current Design mission to be selected", () => {
  const { setMissionId } = renderFields({ decision: "start", missionRequired: true });
  const mission = screen.getByRole("combobox", { name: "Mission" });

  expect(mission).toBeEnabled();
  fireEvent.change(mission, { target: { value: "observe-id" } });
  expect(setMissionId).toHaveBeenCalledWith("observe-id");
});

test("offers only Start, Wait, and Unknown decisions", () => {
  renderFields();

  expect(screen.getByRole("option", { name: "Start" })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: "Wait" })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: "Unknown" })).toBeInTheDocument();
  expect(screen.queryByRole("option", { name: "Continue" })).not.toBeInTheDocument();
  expect(screen.queryByRole("option", { name: "Cancel" })).not.toBeInTheDocument();
});

test("reuses approved reasons matching the current decision", () => {
  const setReason = jest.fn();
  renderFields({
    decision: "wait",
    reason: "",
    setReason,
    reasonPresets: [
      {
        decision: "wait",
        mission_id: "",
        reason: "The dish is empty, but a person remains visible.",
        count: 3,
      },
      {
        decision: "unknown",
        mission_id: "",
        reason: "The scene is unclear.",
        count: 1,
      },
    ],
  });

  const preset = screen.getByRole("combobox", { name: "Reason preset" });
  expect(screen.getByRole("option", {
    name: "The dish is empty, but a person remains visible. · 3 labels",
  })).toBeInTheDocument();
  expect(screen.queryByRole("option", { name: "The scene is unclear. · 1 label" }))
    .not.toBeInTheDocument();
  fireEvent.change(preset, {
    target: { value: "The dish is empty, but a person remains visible." },
  });
  expect(setReason).toHaveBeenCalledWith(
    "The dish is empty, but a person remains visible.",
  );
});

test("supervision offers Continue, Pause, and Cancel without a mission selector", () => {
 const { setMissionId } = renderFields({supervision:true,reasoningTask:"mission_supervision",
  decisionChoices:["continue","pause","cancel"],decision:"pause",missionId:"observe-id",missionRequired:true,
  frameMission:{mission_id:"observe-id",status:"running",step:0,action:"navigation"}});
 expect(screen.getByRole("option",{name:"Pause"})).toBeInTheDocument();
 expect(screen.getByRole("option",{name:"Continue"})).toBeInTheDocument();
 expect(screen.getByRole("option",{name:"Cancel and return to patrol"})).toBeInTheDocument();
 expect(screen.queryByRole("option",{name:"Start"})).not.toBeInTheDocument();
 expect(screen.queryByRole("combobox",{name:"Mission"})).not.toBeInTheDocument();
 expect(screen.queryByRole("combobox",{name:"Expected decision"})).not.toBeInTheDocument();
 expect(screen.getByText(/Active mission: ObservationMode/)).toBeInTheDocument();
 fireEvent.change(screen.getByRole("combobox",{name:"Mission action"}),{target:{value:"cancel"}});
 expect(setMissionId).not.toHaveBeenCalled();
});

test("cancel is unavailable on historical return frames", () => {
 renderFields({supervision:true,reasoningTask:"mission_supervision",decisionChoices:["pause","continue"],
  decision:"cancel",unsupportedSavedLabel:true,invalidDecisionContext:true});
 expect(screen.getByRole("option",{name:"cancel (legacy)"})).toBeDisabled();
 expect(screen.getByRole("button",{name:"Save"})).toBeDisabled();
});


test("Mission frames without recorded execution never offer observation decisions", () => {
 const onSave=jest.fn();
 render(<CollectReasoningFields
  draft={{active:false,episode:{episode_type:"mission"},selectedFrame:{frame_id:"before-start",context:{reasoning_task:"mission_selection"}}}}
  editor={{decision:"wait",missions:[],reason:"A person is visible.",missionId:"",loadedFrame:"before-start",
   invalidDecisionContext:false,missionRequired:false}}
  onSave={onSave}/>);
 expect(screen.getByRole("combobox",{name:"Mission action"})).toBeDisabled();
 expect(screen.queryByRole("combobox",{name:"Mission"})).not.toBeInTheDocument();
 expect(screen.queryByRole("combobox",{name:"Expected decision"})).not.toBeInTheDocument();
 expect(screen.queryByRole("option",{name:"Start"})).not.toBeInTheDocument();
 expect(screen.getByRole("button",{name:"Save"})).toBeDisabled();
 fireEvent.submit(screen.getByRole("button",{name:"Save"}).closest("form"));
 expect(onSave).not.toHaveBeenCalled();
});

test("patrol frames show movement actions without selecting a mission", () => {
 renderFields({reasoningTask:"patrol_supervision",decisionChoices:["pause","continue"],decision:"continue"});
 expect(screen.getByRole("combobox",{name:"Movement action"})).toBeEnabled();
 expect(screen.queryByRole("combobox",{name:"Mission"})).not.toBeInTheDocument();
 expect(screen.queryByRole("combobox",{name:"Expected decision"})).not.toBeInTheDocument();
 expect(screen.getByRole("option",{name:"Pause"})).toBeInTheDocument();
 expect(screen.queryByRole("option",{name:"Cancel and return to patrol"})).not.toBeInTheDocument();
});
