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
import ObservationPlanEditor from "./ObservationPlanEditor";
const setup = extra => ({
  draft: {mode:"fixed", observations:[{spot_id:"a",mission_ids:[]}], excluded_mission_ids:[]},
  spots: [{id:"a",label:"Observation",map_name:"Home"},
    {id:"extra",label:"Laundry",metadata:{role:"observation"}}, {id:"table",label:"Task table"}],
  missions:[], edit:jest.fn(), ...extra,
});
test("Operate adds saved observations to the shared patrol plan", () => {
  const config = setup(); render(<ObservationPlanEditor config={config} />);
  expect(screen.queryByRole("button",{name:"Delete saved spot"})).not.toBeInTheDocument();
  expect(screen.queryByRole("button",{name:"Add observation at robot"})).not.toBeInTheDocument();
  expect(screen.queryByRole("option",{name:"Task table"})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button",{name:"Add saved observation spot"}));
  expect(config.edit).toHaveBeenCalledWith({ observations: [config.draft.observations[0],
    {spot_id:"extra",departure_policy:"reasoning",dwell_seconds:15,mission_ids:[],arrival_bt:""}] });
});
test("removing from the plan does not delete saved data", () => {
  const config = setup(); render(<ObservationPlanEditor config={config} />);
  fireEvent.click(screen.getByRole("button",{name:"Remove observation 1 from plan"}));
  expect(config.edit).toHaveBeenCalledWith({observations:[]});
});
test("plan editing is disabled during autonomy execution", () => {
  render(<ObservationPlanEditor config={setup({locked:true})} />);
  expect(screen.getByRole("button",{name:"Add saved observation spot"})).toBeDisabled();
});
test("an empty mission scope is displayed as no missions",()=>{
 render(<ObservationPlanEditor config={setup()} />);
 expect(screen.getByText("No missions")).toBeInTheDocument();
});

test("Operate shows and edits the route behavior saved in Record",()=>{
 const config=setup({draft:{mode:"patrol",route_closed:false,observations:[{spot_id:"a",mission_ids:[],dwell_seconds:15}],excluded_mission_ids:[]}});
 render(<ObservationPlanEditor config={config}/>);
 expect(screen.getByRole("combobox",{name:"Observation route behavior"})).toHaveValue("one-way");
 fireEvent.change(screen.getByRole("combobox",{name:"Observation route behavior"}),{target:{value:"closed-loop"}});
 expect(config.edit).toHaveBeenCalledWith({route_closed:true});
});
test("older patrol plans default to closed loop and fixed mode hides the route option",()=>{
 const config=setup();const {rerender}=render(<ObservationPlanEditor config={config}/>);
 expect(screen.queryByRole("combobox",{name:"Observation route behavior"})).not.toBeInTheDocument();
 rerender(<ObservationPlanEditor config={{...config,draft:{...config.draft,mode:"patrol"}}}/>);
 expect(screen.getByRole("combobox",{name:"Observation route behavior"})).toHaveValue("closed-loop");
});
