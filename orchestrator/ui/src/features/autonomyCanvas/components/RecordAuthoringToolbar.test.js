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

import { act, fireEvent, render, screen } from "@testing-library/react";
import RecordAuthoringToolbar from "./RecordAuthoringToolbar";

function manager() {
  return {ready:true,spots:[{id:"a",label:"Table view"}],selectedId:"a",canUseRobot:true,
    route:{routeMode:false,routeView:{closed:false},toggleMode:jest.fn(),moveSpot:jest.fn(),removeSpot:jest.fn(),clearRoute:jest.fn(),closeLoop:jest.fn(),openLoop:jest.fn()},
    routeEditor:{plan:{observations:[{spot_id:"a",mission_ids:[]}],excluded_mission_ids:[]},missions:[{mission_id:"clean",mission_name:"CleanTable"}],dirty:true,
      save:jest.fn(),discard:jest.fn(),updatePoint:jest.fn()},
    createOnMap:jest.fn(),createAtRobot:jest.fn(),begin:jest.fn(),remove:jest.fn(),reload:jest.fn(),select:jest.fn(),
    save:jest.fn(),cancel:jest.fn(),onMap:jest.fn(),atRobot:jest.fn(),setDraft:jest.fn()};
}
test("creation uses Design's On Map and At Robot options",()=>{
  const m=manager();render(<RecordAuthoringToolbar manager={m}/>);
  fireEvent.click(screen.getByRole("button",{name:"Create observation waypoint"}));
  fireEvent.click(screen.getByRole("button",{name:"On Map"}));expect(m.createOnMap).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button",{name:"Create observation waypoint"}));
  fireEvent.click(screen.getByRole("button",{name:"At Robot"}));expect(m.createAtRobot).toHaveBeenCalledTimes(1);
});
test("editing has one entry point and leaves details on the map",()=>{
 const m=manager();render(<RecordAuthoringToolbar manager={m}/>);
 fireEvent.click(screen.getByRole("button",{name:"Edit observation route"}));
 expect(m.route.toggleMode).toHaveBeenCalledTimes(1);
 expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
 expect(screen.queryByRole("button",{name:"Save observation route"})).not.toBeInTheDocument();
});
test("active runtime disables authoring tools",()=>{
  const m=manager();render(<RecordAuthoringToolbar manager={m} disabled/>);
  expect(screen.getByRole("button",{name:"Create observation waypoint"})).toBeDisabled();
  expect(screen.getByRole("button",{name:"Edit observation route"})).toBeDisabled();
  expect(screen.getByRole("button",{name:"Undo"})).toBeDisabled();
});


test("turning off Edit offers discard without a separate Cancel button", async () => {
 const m=manager();m.route.routeMode=true;
 const confirm=jest.spyOn(window,"confirm").mockReturnValue(false);
 try {
  render(<RecordAuthoringToolbar manager={m}/>);
  fireEvent.click(screen.getByRole("button",{name:"Edit observation route"}));
  expect(m.routeEditor.discard).not.toHaveBeenCalled();
  expect(m.route.toggleMode).not.toHaveBeenCalled();
  confirm.mockReturnValue(true);
  await act(async()=>fireEvent.click(screen.getByRole("button",{name:"Edit observation route"})));
  expect(m.routeEditor.discard).toHaveBeenCalledTimes(1);
 } finally {confirm.mockRestore();}
});
