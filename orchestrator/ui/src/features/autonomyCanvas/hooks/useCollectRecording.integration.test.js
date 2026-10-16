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
import useCollectDraft from "./useCollectDraft";
import useCollectRecording from "./useCollectRecording";

const session = (name, running = false) => ({ mapName: "house", type: running ? "mission" : "observation", ready: true,
  missionId: name, missionName: name, observationSpotId: "table", returnSpotId: "table", planRevision: 3,
  patrolSpotIds: ["table", "laundry"] });
let episodes, captures;
const update = episode => { episodes = episodes.map(item => item.id === episode.id ? episode : item); return episode; };
beforeEach(() => {
  jest.useFakeTimers(); episodes = []; captures = [];
  global.fetch = jest.fn(async (url, options) => {
    const body = options?.body ? JSON.parse(options.body) : null;
    let data;
    if (url.includes("reason-presets")) data = { presets: [] };
    else if (url.endsWith("/episodes") && !body) data = { episodes };
    else if (url.endsWith("/episodes") && body) {
      data = { id: `ep${episodes.length + 1}`, group_id: `g${episodes.length + 1}`, ended_at: null, observations: [], annotations: [], ...body };
      episodes.push(data);
    } else {
      const [, id, operation] = url.match(/\/episodes\/([^/]+)(?:\/([^/]+))?$/) || [];
      const source = episodes.find(item => item.id === id);
      if (!source) throw new Error(`Unknown test episode: ${url}`);
      if (operation === "capture") {
        const frame = { frame_id: `frame${captures.length + 1}`, capture: { received_at_ms: Date.now() }, context: body.context };
        const episode = update({ ...source, observations: [...source.observations, frame] });
        captures.push({ episode_id: id, type: episode.episode_type, frame });
        data = { episode, saved_frame: frame };
      } else if (operation === "segment" || operation === "return") {
        const fields = operation === "return" ? { episode_type: "patrol", patrol_spot_ids: [source.return_observation_spot_id], capture_segment: "mission_return" } : { ...body, capture_segment: "execution_change" };
        data = { ...source, ...fields, id: `ep${episodes.length + 1}`, previous_episode_id: id, ended_at: null, observations: [], annotations: [] };
        update({ ...source, ended_at: Date.now(), next_episode_id: data.id });
        episodes.push(data);
      } else if (operation === "end") data = update({ ...source, ended_at: Date.now() });
      else data = source;
    }
    return { ok: true, json: async () => data };
  });
});
afterEach(() => { delete global.fetch; jest.useRealTimers(); });
const setup = initialScene => renderHook(({ scene }) => {
  const draft = useCollectDraft({ enabled: true, mapName: "house", getExecutionContext: () => ({
    recording_execution: scene.ready ? { type: scene.type, mission_id: scene.type === "mission" ? scene.missionId : null, status: "running" } : null,
  }) });
  const recording = useCollectRecording({ draft, session: scene });
  return { draft, recording };
}, { initialProps: { scene: initialScene } });
const ticks = async count => {
  for (let i = 0; i < count; i++) await act(async () => jest.advanceTimersByTime(1000));
};

test("continuous capture can save several frames, finish, then collect another mission again", async () => {
  const h = setup(session("CleanTable", true));
  await act(async () => h.result.current.recording.startEpisode());
  let first; await act(async () => { first = h.result.current.recording.startCapture(); });
  await ticks(5);
  expect(captures.length).toBeGreaterThanOrEqual(4);
  await act(async () => h.result.current.recording.saveEpisode());
  await act(async () => { jest.advanceTimersByTime(1000); await first; });
  const firstCount = captures.length;
  h.rerender({ scene: session("ServeDish", true) });
  await act(async () => h.result.current.recording.startEpisode());
  let second; await act(async () => { second = h.result.current.recording.startCapture(); });
  await ticks(5);
  expect(captures.length - firstCount).toBeGreaterThanOrEqual(4);
  expect(captures.slice(firstCount).every(c => c.episode_id === "ep2" && c.frame.context.recording_execution.mission_id === "ServeDish")).toBe(true);
  act(() => h.result.current.recording.stopCapture());
  await act(async () => { jest.advanceTimersByTime(1000); await second; });
  h.unmount();
});

test("real recorder retains its capture loop and group through mission detour and patrol return", async () => {
  const patrol = { ...session(null), type: "patrol" };
  const states = [patrol, session("CleanTable"), session("CleanTable", true),
    { ...patrol, returning: true, missionId: "CleanTable", returnSpotId: "table", patrolSpotIds: ["table"] },
    session("CleanTable"), patrol];
  const h = setup(patrol);
  await act(async () => h.result.current.recording.startEpisode());
  let loop; await act(async () => { loop = h.result.current.recording.startCapture(); });
  for (const scene of states) {
    h.rerender({ scene });
    await ticks(3);
    expect(h.result.current.draft.recording).toBe(true);
    expect(h.result.current.draft.episode.episode_type).toBe(scene.type);
    expect(h.result.current.draft.episode.observations.length).toBeGreaterThan(0);
    expect(h.result.current.draft.error).toBe("");
    expect(h.result.current.recording.recordError).toBe("");
  }
  expect(episodes.map(episode => episode.episode_type)).toEqual(["patrol", "observation", "mission", "patrol", "observation", "patrol"]);
  expect(new Set(episodes.map(episode => episode.group_id)).size).toBe(1);
  expect(episodes.slice(0, -1).every(episode => episode.ended_at)).toBe(true);
  expect(captures.every(capture => capture.type === capture.frame.context.recording_execution.type)).toBe(true);
  expect(episodes[3].capture_segment).toBe("mission_return");
  await act(async () => h.result.current.recording.saveEpisode());
  await act(async () => { jest.advanceTimersByTime(1000); await loop; });
  expect(h.result.current.draft.recording).toBe(false);
  expect(episodes.every(episode => episode.ended_at)).toBe(true);
  h.unmount();
});


test("failed DoLaundry waits without ending capture and resumes on the next execution",async()=>{
 const mission=session("DoLaundry",true),h=setup(mission);
 await act(async()=>h.result.current.recording.startEpisode());
 let loop;await act(async()=>{loop=h.result.current.recording.startCapture();});
 await ticks(3);const before=captures.length;
 h.rerender({scene:{...mission,type:null,ready:false,waitingReason:"Waypoint Task failed at LaundryBasket"}});
 await ticks(10);
 expect(captures).toHaveLength(before);
 expect(h.result.current.draft.recording).toBe(true);
 expect(h.result.current.draft.active).toBe(true);
 expect(h.result.current.recording.awaitingTarget).toBe(true);
 expect(episodes.every(e=>!e.ended_at)).toBe(true);
 h.rerender({scene:mission});await ticks(3);
 expect(captures.length).toBeGreaterThan(before);
 expect(h.result.current.draft.recording).toBe(true);
 h.rerender({scene:{...session(null),type:"patrol"}});await ticks(3);
 expect(captures.at(-1).type).toBe("patrol");
 expect(h.result.current.draft.recording).toBe(true);
 act(()=>h.result.current.recording.stopCapture());
 await act(async()=>{jest.advanceTimersByTime(1000);await loop;});h.unmount();
});
