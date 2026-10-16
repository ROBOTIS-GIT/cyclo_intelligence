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

import { frameMatchesEpisode, approvedReasoningCount } from './reasoningDataset';
const common = {map_name:'house',capture_contract:'task_aligned_v1'};
const context = {source:'server_at_capture',runtime_source:'record_runner_reported',map_name:'house',localized:true};
test('observation capture must match the arrived localized spot',()=>{
 const ep={...common,episode_type:'observation',observation_spot_id:'table'};
 const frame={context:{...context,reasoning_task:'mission_selection',observation:{phase:'observing',at_spot_id:'table'}}};
 expect(frameMatchesEpisode(ep,frame)).toBe(true);
 expect(frameMatchesEpisode(ep,{context:{...frame.context,localized:false}})).toBe(false);
 expect(frameMatchesEpisode(ep,{context:{...frame.context,observation:{phase:'moving',target_spot_id:'table'}}})).toBe(false);
});
test('patrol requires a route member and cannot masquerade as a mission',()=>{
 const ep={...common,episode_type:'patrol',patrol_spot_ids:['table','laundry']};
 const frame={context:{...context,reasoning_task:'patrol_supervision',observation:{phase:'moving',from_spot_id:'table',target_spot_id:'laundry'}}};
 expect(frameMatchesEpisode(ep,frame)).toBe(true);
 expect(frameMatchesEpisode(ep,{context:{...frame.context,active_mission:{mission_id:'clean',status:'running'}}})).toBe(false);
 expect(frameMatchesEpisode(ep,{context:{...frame.context,observation:{phase:'moving',target_spot_id:'tv'}}})).toBe(false);
});
test('mission navigation and paused frames stay Mission; preparation and return do not',()=>{
 const ep={...common,episode_type:'mission',recording_mission_id:'clean'};
 const frame={context:{...context,reasoning_task:'mission_supervision',active_mission:{mission_id:'clean',status:'running',action:'navigation'}}};
 expect(frameMatchesEpisode(ep,frame)).toBe(true);
 expect(frameMatchesEpisode(ep,{context:{...frame.context,active_mission:{mission_id:'clean',status:'paused'}}})).toBe(true);
 expect(frameMatchesEpisode(ep,{context:{...frame.context,active_mission:{mission_id:'other',status:'running'}}})).toBe(false);
 expect(frameMatchesEpisode(ep,{context:{...frame.context,active_mission:{mission_id:'clean',status:'paused',action:'return-navigation'}}})).toBe(false);
 expect(frameMatchesEpisode(ep,{context:{...frame.context,reasoning_task:'patrol_supervision',active_mission:null}})).toBe(false);
 expect(frameMatchesEpisode(ep,{context:{...frame.context,runtime_source:'unverified'}})).toBe(false);
});
test('wrong-role legacy images remain present but are not counted as training labels',()=>{
 const ep={episode_type:'mission',observations:[{frame_id:'before'},{frame_id:'during',context:{reasoning_task:'mission_supervision',active_mission:{mission_id:'clean',status:'running'}}}],annotations:[{frame_ids:['before'],review_status:'approved'},{frame_ids:['during'],review_status:'approved'}]};
 expect(approvedReasoningCount(ep)).toBe(1);
 expect(ep.observations).toHaveLength(2);
});


test('patrol approaching frames accepted by the API stay available for review',()=>{
 const ep={...common,episode_type:'patrol',patrol_spot_ids:['table','laundry']};
 const frame={context:{...context,reasoning_task:'patrol_supervision',observation:{phase:'approaching',from_spot_id:'table',target_spot_id:'laundry'}}};
 expect(frameMatchesEpisode(ep,frame)).toBe(true);
 expect(frameMatchesEpisode({...ep,capture_segment:'mission_return'},frame)).toBe(false);
});
