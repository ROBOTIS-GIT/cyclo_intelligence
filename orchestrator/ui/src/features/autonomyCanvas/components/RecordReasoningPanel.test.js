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

import {fireEvent,render,screen} from "@testing-library/react";
import RecordReasoningPanel from "./RecordReasoningPanel";
jest.mock("./LiveHeadCamera",()=>()=> <img alt="Live head camera"/>);
const model={id:"m",model_name:"House",base_model:"Qwen"};
const observation={available:true,active:false,phase:"watching",needsStop:false,start:jest.fn(),stop:jest.fn(),
 modelChoice:{models:[model],selected:model,select:jest.fn(),refresh:jest.fn()},
 result:{reasoning_task:"patrol_supervision",decision:{decision:"continue",reason:"No person nearby."},attention_preview:{image_base64:"heat",layer:23}}};
test("keeps preview controls and supervision results in the map header",()=>{
 const h=render(<RecordReasoningPanel observation={observation}/>);
 expect(screen.getByText("No person nearby.")).toBeInTheDocument();
 expect(screen.getByText("continue")).toBeInTheDocument();
 expect(screen.getByLabelText("Reasoning model")).toHaveValue("m");
 fireEvent.click(screen.getByRole("button",{name:"Start reasoning"}));expect(observation.start).toHaveBeenCalledTimes(1);
 h.rerender(<RecordReasoningPanel observation={{...observation,active:true,needsStop:true,modelChoice:{...observation.modelChoice,locked:true}}}/>);
 expect(screen.getByRole("button",{name:"Start reasoning"})).toBeDisabled();
 expect(screen.getByLabelText("Reasoning model")).toBeDisabled();
 fireEvent.click(screen.getByRole("button",{name:"Stop reasoning"}));expect(observation.stop).toHaveBeenCalledTimes(1);
});
