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

import {act, renderHook} from "@testing-library/react";
import useLabelingFrameScope from "./useLabelingFrameScope";
const frames=[{frame_id:"a"},{frame_id:"b"},{frame_id:"c"}];
const props={episodeId:"one",sampleIds:["a","c"],frames,isReviewed:id=>id==="a",generatedIds:["b"]};
test("saved samples do not restrict default browsing and progress",()=>{
 const h=renderHook(useLabelingFrameScope,{initialProps:props});
 expect(h.result.current.frameScope).toBe("all");
 expect(h.result.current.visibleFrames).toEqual(frames);
 expect(h.result.current.progress).toBe("1/3 labeled");
 act(()=>h.result.current.setFrameScope("samples"));
 expect(h.result.current.visibleFrames).toEqual([frames[0],frames[2]]);
 expect(h.result.current.progress).toBe("1/2 labeled");
 act(()=>h.result.current.setFrameScope("all"));
 expect(h.result.current.visibleFrames).toEqual(frames);
});
test("opening another episode defaults to all frames while map review can follow episodes",()=>{
 const h=renderHook(useLabelingFrameScope,{initialProps:props});
 act(()=>h.result.current.setFrameScope("samples"));
 h.rerender({...props,episodeId:"two"});
 expect(h.result.current.frameScope).toBe("all");
 h.rerender(props);
 expect(h.result.current.frameScope).toBe("all");
 act(()=>h.result.current.setFrameScope("review",null));
 h.rerender({...props,episodeId:"three"});
 expect(h.result.current.frameScope).toBe("review");
 expect(h.result.current.visibleFrames).toEqual([frames[1]]);
});
