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

import {act,fireEvent,render,screen,waitFor} from '@testing-library/react';
import QwenWorkflowControls from './QwenWorkflowControls';
import {collectionRequest} from '../../../utils/collectionApi';
jest.mock('../../../utils/collectionApi',()=>({collectionRequest:jest.fn()}));
const frames=[{frame_id:'f1'},{frame_id:'f2'},{frame_id:'f3'}];
const episode={id:'ep',map_name:'warehouse',ended_at:1,observations:frames,annotations:[],qwen_sample:{schema_version:'user_count_v1',frame_ids:['f1','f2']}};
const status={training:{status:'idle'},batch:{status:'idle'},models:[]};
beforeEach(()=>collectionRequest.mockReset().mockResolvedValue(status));

const savedQwenEpisode={...episode,annotations:['f1','f2'].map(id=>({frame_ids:[id],review_status:'approved'}))};
const oldQwenModel={id:'old-model',map_name:'warehouse',model_name:'Older Qwen',sample_count:2,created_at:1};

test('unified Qwen surfaces a failed background job outside hidden workflow settings',async()=>{
 collectionRequest.mockResolvedValue({...status,training:{id:'failed-training',status:'failed',error:'Adapter training failed'}});
 render(<QwenWorkflowControls unified stage="training" episode={savedQwenEpisode} episodes={[savedQwenEpisode]}
   renderLayout={({activity,feedback})=><section aria-label="Current activity">{activity}{feedback}</section>}/>);
 expect(await screen.findByRole('alert')).toHaveTextContent('Adapter training failed');
});


test('clears a failed background job so fine-tuning can be started again',async()=>{
 const failed={...status,training:{id:'stopped',status:'failed',error:'Stopped training'}};
 collectionRequest.mockImplementation(path=>Promise.resolve(path==='qwen/jobs/training/reset'?status:failed));
 render(<QwenWorkflowControls unified stage="training" episode={savedQwenEpisode} episodes={[savedQwenEpisode]}
   renderLayout={({feedback})=><section>{feedback}</section>}/>);
 fireEvent.click(await screen.findByRole('button',{name:'Clear Qwen training error'}));
 await waitFor(()=>expect(screen.queryByRole('alert')).not.toBeInTheDocument());
 expect(collectionRequest).toHaveBeenCalledWith('qwen/jobs/training/reset',{});
});


test('cancels only the displayed fine-tune and permits starting again without clearing an error',async()=>{
 const job={id:'a'.repeat(32),status:'training',step:5,steps:200,model_name:'Current Qwen'};
 let finishCancel;
 collectionRequest.mockImplementation(path=>path==='qwen/jobs/training/cancel'
   ?new Promise(resolve=>{finishCancel=resolve;}) : Promise.resolve({...status,training:job}));
 render(<QwenWorkflowControls unified episode={savedQwenEpisode} episodes={[savedQwenEpisode]}
   renderLayout={({actions,feedback})=><>{actions}{feedback}</>}/>);
 const cancel=await screen.findByRole('button',{name:'Cancel fine-tuning'});
 fireEvent.click(cancel);expect(cancel).toBeDisabled();fireEvent.click(cancel);
 expect(collectionRequest.mock.calls.filter(([path])=>path.endsWith('/cancel'))).toHaveLength(1);
 expect(collectionRequest).toHaveBeenCalledWith('qwen/jobs/training/cancel',{job_id:job.id});
 await act(async()=>finishCancel({...status,training:{...job,status:'cancelled'}}));
 expect(screen.getByText('Fine-tuning cancelled')).toBeVisible();
 expect(screen.queryByRole('alert')).not.toBeInTheDocument();
 expect(screen.getByRole('button',{name:'Fine-tune'})).toBeEnabled();
 fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/auto-train',{map_name:'warehouse'}));
});


const globalLayout=({actions,feedback})=><>{actions}{feedback}</>;
test('Qwen tunes saved map samples without opening an episode',async()=>{
 const a={id:'a',map_name:'alpha',ended_at:1,qwen_sample_approved_count:2};
 const b={...a,id:'b',map_name:'beta'};
 render(<QwenWorkflowControls unified episode={null} episodes={[a,b]} renderLayout={globalLayout}/>);
 fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Start fine-tuning'})).toBeEnabled());
 fireEvent.change(screen.getByLabelText('Fine-tuning map'),{target:{value:'beta'}});
 fireEvent.click(screen.getByRole('button',{name:'Start fine-tuning'}));
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/auto-train',{map_name:'beta'}));
});

test('Qwen map auto-labeling includes compatible types without an editor episode',async()=>{
 const obs={id:'obs',map_name:'house',ended_at:1,episode_type:'observation'};
 const mission={...obs,id:'task',episode_type:'mission'};
 const patrol={...obs,id:'walk',episode_type:'patrol'};
 const model={id:'trained',model_name:'Mission-only',map_name:'house',episode_types:['mission']};
 collectionRequest.mockImplementation(path=>Promise.resolve(path==='qwen/workflow'?{...status,models:[model]}:{status:'queued'}));
 render(<QwenWorkflowControls unified episode={null} episodes={[obs,mission,patrol,{...obs,id:'active',ended_at:null}]}
   renderLayout={globalLayout}/>);
 fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
 await screen.findByRole('button',{name:'Mission-only'});
 expect(screen.getByLabelText('Auto-label target')).toHaveValue('map');
 fireEvent.click(screen.getByRole('button',{name:'Auto-label map'}));
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/auto-label',{
   model_id:'trained',episode_ids:['task'],replace_unreviewed:false}));
});

test('switching an auto-label map keeps its picker open and leaves the editor episode unchanged',async()=>{
 const episodeOnOtherMap={...savedQwenEpisode,id:'second',map_name:'beta'};
 const model={id:'beta-model',map_name:'beta',model_name:'Beta model',episode_types:['observation']};
 collectionRequest.mockImplementation(path=>Promise.resolve(path==='qwen/workflow'?{...status,models:[oldQwenModel,model]}:{status:'queued'}));
 render(<QwenWorkflowControls unified episode={savedQwenEpisode} episodes={[savedQwenEpisode,episodeOnOtherMap]}
   renderLayout={globalLayout}/>);
 fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
 await screen.findByRole('button',{name:'Older Qwen'});
 fireEvent.change(screen.getByLabelText('Auto-label map'),{target:{value:'beta'}});
 expect(screen.getByRole('region',{name:'Auto-label models'})).toBeVisible();
 expect(screen.getByLabelText('Auto-label target')).toHaveValue('map');
 fireEvent.click(screen.getByRole('button',{name:'Auto-label map'}));
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/auto-label',{
   model_id:'beta-model',episode_ids:['second'],replace_unreviewed:false}));
});
