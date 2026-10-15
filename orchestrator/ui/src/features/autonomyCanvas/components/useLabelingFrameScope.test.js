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
