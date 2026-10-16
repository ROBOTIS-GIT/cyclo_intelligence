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

import {act,renderHook} from '@testing-library/react';
import useCollectAnnotation from './useCollectAnnotation';
test('uses current mission names while preserving the selected label mission ID',()=>{
 const frame={frame_id:'frame'};
 const draft={episodes:[],selectedFrame:frame,episode:{id:'ep',ended_at:1,observations:[frame],
  catalog:{missions:[{mission_id:'same-id',mission_name:'Mission1'}]},
  display_missions:[{mission_id:'same-id',mission_name:'CleanTable'}],
  annotations:[{frame_ids:['frame'],revision:1,target:{decision:'start',mission_id:'same-id',reason:'Table needs cleaning'}}]}};
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'warehouse'}));
 expect(result.current.missions).toEqual([{mission_id:'same-id',mission_name:'CleanTable'}]);
 expect(result.current.missionId).toBe('same-id');
});

test('defaults a new mission-selection label to wait while preserving saved decisions',()=>{
 const frame={frame_id:'frame'};
 const draft={episodes:[],selectedFrame:frame,episode:{id:'ep',ended_at:1,observations:[frame],annotations:[]}};
 const {result,rerender}=renderHook(({draft})=>useCollectAnnotation({draft,mapName:'Home'}),{initialProps:{draft}});
 expect(result.current.decision).toBe('wait');
 expect(result.current.missionId).toBe('');
 expect(result.current.missionRequired).toBe(false);
 expect(result.current.labelDirty).toBe(false);
 const saved={...draft,episode:{...draft.episode,annotations:[{frame_ids:['frame'],revision:1,
  target:{decision:'start',mission_id:'clean',reason:'The table is ready to clear.'}}]}};
 rerender({draft:saved});
 expect(result.current.decision).toBe('start');
 expect(result.current.missionId).toBe('clean');
});

test('uses the live Design catalog for mission choices',()=>{
 const frame={frame_id:'frame'};
 const draft={episodes:[],selectedFrame:frame,episode:{id:'ep',ended_at:1,observations:[frame],
  catalog:{missions:[{mission_id:'old-id',mission_name:'Mission1'}]},
  current_catalog:{missions:[
   {mission_id:'old-id',mission_name:'CleanTable'},
   {mission_id:'observe-id',mission_name:'ObservationMode'},
  ]},annotations:[]}};
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'reasoning'}));
 expect(result.current.missions.map((mission)=>mission.mission_name)).toEqual([
  'CleanTable','ObservationMode',
 ]);
});

test('preserves a deleted historical mission as a display-only selection',()=>{
 const frame={frame_id:'frame'};
 const draft={episodes:[],selectedFrame:frame,episode:{id:'ep',ended_at:1,observations:[frame],
  catalog:{missions:[{mission_id:'deleted-id',mission_name:'Mission2'}]},
  current_catalog:{missions:[{mission_id:'active-id',mission_name:'ObservationMode'}]},
  annotations:[{frame_ids:['frame'],revision:1,
   catalog:{missions:[{mission_id:'deleted-id',mission_name:'ArchivedObservation'}]},
   target:{decision:'start',mission_id:'deleted-id',reason:'legacy'}}]}};
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'reasoning'}));
 expect(result.current.missionId).toBe('deleted-id');
 expect(result.current.deletedSelectedMission).toEqual({
  mission_id:'deleted-id',mission_name:'ArchivedObservation',
 });
 expect(result.current.selectedMissionUnavailable).toBe(true);
 expect(result.current.missions).toEqual([{mission_id:'active-id',mission_name:'ObservationMode'}]);
});

test.each(['wait','unknown'])('normalizes a %s label by clearing its stale mission ID',decision=>{
 const frame={frame_id:'frame'};
 const draft={episodes:[],selectedFrame:frame,episode:{id:'ep',ended_at:1,observations:[frame],
  catalog:{missions:[{mission_id:'legacy-id',mission_name:'LegacyMission'}]},
  annotations:[{frame_ids:['frame'],revision:1,
   target:{decision,mission_id:'legacy-id',reason:'No mission action is required.'}}]}};
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'reasoning'}));

 expect(result.current.decision).toBe(decision);
 expect(result.current.missionId).toBe('');
 expect(result.current.missionRequired).toBe(false);
 expect(result.current.labelDirty).toBe(false);
});

test.each(['continue','cancel'])('marks a legacy %s label for relabeling',decision=>{
 const frame={frame_id:'frame',context:{active_mission:{mission_id:'legacy-id',status:'running'}}};
 const draft={episodes:[],selectedFrame:frame,episode:{id:'ep',ended_at:1,observations:[frame],
  catalog:{missions:[{mission_id:'legacy-id',mission_name:'LegacyMission'}]},
  annotations:[{frame_ids:['frame'],revision:1,review_status:'approved',
   target:{decision,mission_id:'legacy-id',reason:'Legacy execution supervision.'}}]}};
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'reasoning'}));

 expect(result.current.decision).toBe(decision);
 expect(result.current.unsupportedSavedLabel).toBe(true);
 expect(result.current.invalidDecisionContext).toBe(true);
 expect(result.current.labelDirty).toBe(false);
 expect(result.current.labeled('frame')).toBe(false);
});

test('groups approved reason presets by decision, mission, and text',()=>{
 const frames=['one','two','three'].map(frame_id=>({frame_id}));
 const annotations=[
  {frame_ids:['one'],revision:1,review_status:'approved',
   target:{decision:'wait',mission_id:null,reason:'A person remains visible.'}},
  {frame_ids:['two'],revision:1,review_status:'approved',
   target:{decision:'wait',mission_id:null,reason:'A person remains visible.'}},
  {frame_ids:['three'],revision:1,review_status:'draft',
   target:{decision:'wait',mission_id:null,reason:'Draft reason.'}},
 ];
 const draft={episodes:[],selectedFrame:frames[0],episode:{id:'ep',ended_at:1,
  observations:frames,annotations}};
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'reasoning'}));

 expect(result.current.reasonPresets).toEqual([{
  reasoning_task:'mission_selection',decision:'wait',mission_id:'',reason:'A person remains visible.',count:2,
 }]);
});

function supervisionDraft(context = {}, target = null) {
 const frame={frame_id:'supervised-frame',context:{reasoning_task:'mission_supervision',
  active_mission:{mission_id:'clean',status:'running',step:1,action:'collect_dishes'},...context}};
 return {episodes:[],active:false,selectedFrame:frame,annotate:jest.fn().mockResolvedValue(true),
  episode:{id:'ep',episode_type:'mission',ended_at:1,observations:[frame],current_catalog:{missions:[{mission_id:'clean',mission_name:'CleanTable'}]},
   annotations:target?[{frame_ids:[frame.frame_id],revision:1,review_status:'approved',target}]:[]}};
}

test('supervision defaults to continue and saves only the captured active mission',async()=>{
 const draft=supervisionDraft();
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'Home'}));
 expect(result.current.decision).toBe('continue');
 expect(result.current.decisionChoices).toEqual(['continue','pause','cancel']);
 expect(result.current.missionId).toBe('clean');
 expect(result.current.invalidDecisionContext).toBe(false);
 act(()=>{result.current.setDecision('pause');result.current.setReason('A person reaches toward the robot.');result.current.setLoadedFrame('supervised-frame');});
 await act(async()=>result.current.save('approved'));
 expect(draft.annotate).toHaveBeenCalledWith(expect.objectContaining({decision:'pause',mission_id:'clean'}));
 act(()=>result.current.setMissionId('other-mission'));
 expect(result.current.invalidDecisionContext).toBe(true);
});

test('supervision without captured active mission cannot be saved',()=>{
 const draft=supervisionDraft({active_mission:null});
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'Home'}));
 act(()=>result.current.setDecision('continue'));
 expect(result.current.invalidDecisionContext).toBe(true);
});

test('approved pause labels are recognized only with their recorded supervision task',()=>{
 const draft=supervisionDraft({}, {decision:'pause',mission_id:'clean',reason:'Person near arm.'});
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'Home'}));
 expect(result.current.labeled('supervised-frame')).toBe(true);
 expect(result.current.labelDirty).toBe(false);
 expect(result.current.reasonPresets[0]).toMatchObject({reasoning_task:'mission_supervision',decision:'pause',mission_id:'clean'});
});

test('an explicit empty candidate list never expands to all map missions',()=>{
 const draft=supervisionDraft({reasoning_task:'mission_selection',active_mission:null,candidate_mission_ids:[]});
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'Home'}));
 expect(result.current.missions).toEqual([]);
 act(()=>{result.current.setDecision('start');result.current.setMissionId('clean');});
 expect(result.current.invalidDecisionContext).toBe(true);
});


test('cancel saves the active mission ID and remains an approved reusable supervision label',async()=>{
 const draft=supervisionDraft({}, {decision:'cancel',mission_id:'clean',reason:'The task is no longer needed.'});
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'Home'}));
 expect(result.current.labeled('supervised-frame')).toBe(true);
 expect(result.current.missionRequired).toBe(true);
 expect(result.current.invalidDecisionContext).toBe(false);
 expect(result.current.reasonPresets[0]).toMatchObject({reasoning_task:'mission_supervision',decision:'cancel',mission_id:'clean'});
 act(()=>result.current.setLoadedFrame('supervised-frame'));
 await act(async()=>result.current.save('approved'));
 expect(draft.annotate).toHaveBeenCalledWith(expect.objectContaining({decision:'cancel',mission_id:'clean'}));
});

test('a captured return trip does not offer or save cancel',async()=>{
 const draft=supervisionDraft({active_mission:{mission_id:'clean',status:'returning'}});
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'Home'}));
 expect(result.current.decisionChoices).toEqual(['pause','continue']);
 act(()=>{result.current.setDecision('cancel');result.current.setReason('No task needed.');result.current.setLoadedFrame('supervised-frame');});
 expect(result.current.invalidDecisionContext).toBe(true);
 await act(async()=>result.current.save('approved'));
 expect(draft.annotate).not.toHaveBeenCalled();
});


test('a manually paused return cannot offer or save cancel',async()=>{
 const draft=supervisionDraft({active_mission:{mission_id:'clean',status:'paused',action:'return-navigation'}});
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'Home'}));
 expect(result.current.decisionChoices).toEqual(['pause','continue']);
 act(()=>{result.current.setDecision('cancel');result.current.setReason('No task needed.');result.current.setLoadedFrame('supervised-frame');});
 expect(result.current.invalidDecisionContext).toBe(true);
 await act(async()=>result.current.save('approved'));
 expect(draft.annotate).not.toHaveBeenCalled();
});


test('plan-bound frames keep recorded candidate identities while displaying current names',()=>{
 const frame={frame_id:'f',context:{reasoning_task:'mission_selection',candidate_mission_ids:['clean']},catalog:{missions:[{mission_id:'clean',mission_name:'OldName'}]}};
 const draft={episodes:[],selectedFrame:frame,episode:{id:'ep',ended_at:1,observation_binding:'observation_plan_v1',observations:[frame],annotations:[],
  current_catalog:{missions:[{mission_id:'clean',mission_name:'CleanTable'},{mission_id:'new',mission_name:'NewMission'}]}}};
 const {result}=renderHook(()=>useCollectAnnotation({draft,mapName:'house'}));
 expect(result.current.missions).toEqual([{mission_id:'clean',mission_name:'CleanTable'}]);
});
