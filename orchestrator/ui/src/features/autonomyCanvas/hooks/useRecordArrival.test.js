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

import { act, renderHook } from "@testing-library/react";
import useRecordArrival from "./useRecordArrival";
const spot={id:"table",pose:{x:1,y:2,yaw:0}};
const pose={position:{x:1,y:2},orientation:{w:1,z:0}};
beforeEach(()=>jest.useFakeTimers()); afterEach(()=>jest.useRealTimers());
test("observation requires localized, aligned and settled pose; movement invalidates immediately",()=>{
 const props={enabled:true,localized:false,moving:false,pose,spot};
 const {result,rerender}=renderHook(useRecordArrival,{initialProps:props});
 act(()=>jest.advanceTimersByTime(1000)); expect(result.current).toBe(false);
 rerender({...props,localized:true});
 act(()=>jest.advanceTimersByTime(749)); expect(result.current).toBe(false);
 act(()=>jest.advanceTimersByTime(1)); expect(result.current).toBe(true);
 rerender({...props,localized:true,moving:true}); expect(result.current).toBe(false);
 rerender({...props,localized:true,pose:{...pose,position:{x:1.5,y:2}}});
 act(()=>jest.advanceTimersByTime(1000)); expect(result.current).toBe(false);
 rerender({...props,localized:true,pose:{...pose,orientation:{w:0,z:1}}});
 act(()=>jest.advanceTimersByTime(1000)); expect(result.current).toBe(false);
});
test("pose drift resets settling even within arrival tolerance",()=>{
 const props={enabled:true,localized:true,moving:false,pose,spot};
 const {result,rerender}=renderHook(useRecordArrival,{initialProps:props});
 act(()=>jest.advanceTimersByTime(500));
 rerender({...props,pose:{...pose,position:{x:1.08,y:2}}});
 act(()=>jest.advanceTimersByTime(500)); expect(result.current).toBe(false);
 act(()=>jest.advanceTimersByTime(250)); expect(result.current).toBe(true);
});
