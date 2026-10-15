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
