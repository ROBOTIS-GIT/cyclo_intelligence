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
import useCollectDraft from "./useCollectDraft";

let episode, deleteError, listError;
const frame = frame_id => ({frame_id});
beforeEach(() => {
  deleteError = null; listError = false;
  episode = {id:"episode", ended_at:1, map_name:"house", observations:["a","b","c","d","e"].map(frame),
    annotations:[{frame_ids:["d"], revision:3}], regions:[]};
  global.fetch = jest.fn(async (url, options) => {
    if(url.endsWith("/frames/delete")) {
      if(deleteError) return {ok:false,status:409,json:async()=>({detail:deleteError})};
      const ids = JSON.parse(options.body).frame_ids;
      episode = {...episode, observations:episode.observations.filter(f=>!ids.includes(f.frame_id)),
        annotations:episode.annotations.filter(a=>!ids.includes(a.frame_ids[0]))};
      return {ok:true,json:async()=>({episode,deleted_frame_ids:ids})};
    }
    if(url.endsWith("/episodes") && listError) throw new Error("Offline");
    return {ok:true,json:async()=>url.endsWith("/episodes") ? {episodes:[episode]} :
      url.includes("reason-presets?") ? {presets:[]} : episode};
  });
});
async function setup(selected="c") {
  const hook = renderHook(()=>useCollectDraft({enabled:true,mapName:"house"}));
  await act(async()=>{await hook.result.current.load("episode");});
  act(()=>hook.result.current.selectFrame(episode.observations.find(f=>f.frame_id===selected)));
  return hook;
}
test("bulk deletion refreshes counts and selects the next surviving frame with its label revision", async()=>{
  const {result}=await setup();
  await act(async()=>{expect(await result.current.deleteFrames(["b","c"])).toBe(true);});
  expect(fetch).toHaveBeenCalledWith("/api/collection/episodes/episode/frames/delete", expect.objectContaining({
    method:"POST", body:JSON.stringify({frame_ids:["b","c"]}),
  }));
  expect(result.current.episode.observations.map(f=>f.frame_id)).toEqual(["a","d","e"]);
  expect(result.current.selectedFrame).toMatchObject({frame_id:"d",labelRevision:3});
  expect(result.current.episodes[0].observations).toHaveLength(3);
  expect(result.current.busy).toBe(false);
});
test("keeps the selected frame when it survives", async()=>{
  const {result}=await setup("d");
  await act(async()=>{await result.current.deleteFrames(["a","b"]);});
  expect(result.current.selectedFrame).toMatchObject({frame_id:"d",labelRevision:3});
});
test("selects the preceding frame when deleting the tail, then clears an empty episode", async()=>{
  const {result}=await setup("e");
  await act(async()=>{await result.current.deleteFrames(["d","e"]);});
  expect(result.current.selectedFrame.frame_id).toBe("c");
  await act(async()=>{await result.current.deleteFrames(["a","b","c"]);});
  expect(result.current.selectedFrame).toBeNull();
  expect(result.current.episode.observations).toEqual([]);
});
test("propagates deletion failures without changing the episode or selected frame", async()=>{
  const {result}=await setup();
  deleteError="This episode is being used for training.";
  await act(async()=>{await expect(result.current.deleteFrames(["c"])).rejects.toThrow(deleteError);});
  expect(result.current.episode.observations).toHaveLength(5);
  expect(result.current.selectedFrame.frame_id).toBe("c");
  expect(result.current.error).toBe(deleteError);
  expect(result.current.busy).toBe(false);
});
test("a refresh failure after a committed deletion does not report the delete as failed", async()=>{
  const {result}=await setup();
  listError=true;
  await act(async()=>{expect(await result.current.deleteFrames(["c"])).toBe(true);});
  expect(result.current.episode.observations.map(f=>f.frame_id)).toEqual(["a","b","d","e"]);
  expect(result.current.error).toContain("Frames deleted.");
});
test("rejects deletion while an episode is recording", async()=>{
  episode.ended_at=null;
  const {result}=await setup();
  await act(async()=>{await expect(result.current.deleteFrames(["c"])).rejects.toThrow("Save the episode");});
  expect(fetch.mock.calls.some(([url])=>url.endsWith("/frames/delete"))).toBe(false);
});

test("a late pre-deletion episode response cannot restore deleted frames", async()=>{
  const {result}=await setup();
  const stale={...episode,observations:[...episode.observations]}, impl=fetch.getMockImplementation();
  let resolveOld, oldRefresh;
  fetch.mockImplementation((url,options)=>url.endsWith('/episodes/episode')
    ? new Promise(resolve=>{resolveOld=()=>resolve({ok:true,json:async()=>stale});}) : impl(url,options));
  act(()=>{oldRefresh=result.current.refresh();});
  await act(async()=>{await result.current.deleteFrames(['c']);});
  await act(async()=>{resolveOld();await oldRefresh;});
  expect(result.current.episode.observations.map(f=>f.frame_id)).toEqual(['a','b','d','e']);
});

test("a late initial library request cannot overwrite counts refreshed by deletion", async()=>{
  const stale={...episode,observations:[...episode.observations]}, impl=fetch.getMockImplementation();
  let resolveOld, first=true;
  fetch.mockImplementation((url,options)=>{
    if(url.endsWith('/episodes') && first) {
      first=false;
      return new Promise(resolve=>{resolveOld=()=>resolve({ok:true,json:async()=>({episodes:[stale]})});});
    }
    return impl(url,options);
  });
  const {result}=await setup();
  await act(async()=>{await result.current.deleteFrames(['b','c']);});
  await act(async()=>{resolveOld();});
  expect(result.current.episodes[0].observations.map(f=>f.frame_id)).toEqual(['a','d','e']);
});

test("committed cleanup warnings preserve the updated dataset and remain visible", async()=>{
  const {result}=await setup(), impl=fetch.getMockImplementation();
  fetch.mockImplementation(async(url,options)=>{
    const response=await impl(url,options);
    if(!url.endsWith('/frames/delete'))return response;
    const data=await response.json();
    return {ok:true,json:async()=>({...data,cleanup_pending:true,warning:'Frames deleted, but an unused mask file could not be removed.'})};
  });
  await act(async()=>{expect(await result.current.deleteFrames(['c'])).toBe(true);});
  expect(result.current.episode.observations).toHaveLength(4);
  expect(result.current.error).toContain('unused mask file');
});
