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

import {render} from "@testing-library/react";
import ReasoningText,{actionRanges} from "./ReasoningText";
const result=(decision,mission_name)=>({decision:{decision},selected_mission:mission_name?{mission_name}:null});
const highlighted=(text,r)=>actionRanges(text,r).map(({start,end})=>text.slice(start,end));

test.each([
 ["start","CleanTable","Empty dishes remain on the table. I need to clean the table.",["clean the table"]],
 ["start","ServeDish","People are ready. I need to serve dishes to them.",["serve dishes"]],
 ["start","DoLaundry","The basket is full. I need to do laundry.",["do laundry"]],
 ["start","TurnOffTV","The kid is asleep. I need to turn it off.",["turn it off"]],
 ["start","TurnOnTV","I accidentally turned the TV off. I need to turn the TV on for her.",["turn the TV on"]],
 ["next_observation",null,"I will move on to the next observation point.",["move on to the next observation point"]],
 ["continue",null,"I don't see anyone nearby. Good to continue.",["continue"]],
 ["wait",null,"I need to wait until the person leaves.",["wait"]],
 ["wait",null,"Do not start a mission.",["Do not start a mission"]],
 ["pause",null,"A person is in the way. Pause the mission.",["Pause the mission"]],
 ["cancel",null,"Cancel the mission and resume patrol.",["Cancel the mission","resume patrol"]],
 ["start","Inspect_Shelf","I will inspect the shelf.",["inspect the shelf"]],
])("highlights %s / %s without changing the reason",(decision,mission,text,expected)=>{
 const r=result(decision,mission);
 expect(highlighted(text,r)).toEqual(expected);
 const {container}=render(<ReasoningText text={text} result={r}/>);
 expect(container.textContent).toBe(text);
 expect([...container.querySelectorAll('strong')].map(el=>el.textContent)).toEqual(expected);
});

test("negated actions, other decisions and evidence remain plain",()=>{
 expect(highlighted("Do not clean the table. There is no need to clean the table.",result("start","CleanTable"))).toEqual([]);
 expect(highlighted("It is not safe to continue.",result("continue"))).toEqual([]);
 expect(highlighted("Wait; do not clean the table.",result("wait"))).toEqual(["Wait"]);
 expect(highlighted("The dishes are empty.",result("start","CleanTable"))).toEqual([]);
 expect(highlighted("CleanTable failed",null)).toEqual([]);
});

test("mission names are literal text, not regex instructions",()=>{
 expect(highlighted("Start Inspect (A).",result("start","Inspect (A)"))).toEqual(["Inspect (A)"]);
 // Word-boundary matching of the normalized action, with a literal plus sign.
 expect(highlighted("Clean+Table",result("start","Clean+Table"))).toEqual(["Clean+Table"]);
 expect(highlighted("CleanTable",result("start","Clean+Table"))).toEqual([]);
});
