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

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SamWorkflowControls from './SamWorkflowControls';
import { collectionRequest } from '../../../utils/collectionApi';
jest.mock('../../../utils/collectionApi',()=>({collectionRequest:jest.fn()}));
const state={training:{status:'idle'},batch:{status:'idle'},labels:[{map_name:'warehouse',object_label:'stick',frame_count:3}],models:[{id:'abc',map_name:'warehouse',model_name:'Table objects v1',object_labels:['stick'],sample_count:3,strategy:'sam_multiclass_instances_v1'}]};
beforeEach(()=>collectionRequest.mockReset().mockResolvedValue(state));
test('cancels SAM fine-tuning without stopping the annotation service',async()=>{
 const episode={id:'ep',map_name:'warehouse',ended_at:1,observations:[],regions:[]};
 const job={id:'b'.repeat(32),status:'training',step:10,steps:200};
 collectionRequest.mockImplementation(path=>Promise.resolve(path==='segmentation/training/cancel'
   ?{...state,training:{...job,status:'cancelling'}} : {...state,training:job}));
 render(<SamWorkflowControls unified episode={episode} episodes={[episode]}
   renderLayout={({actions})=>actions}/>);
 fireEvent.click(await screen.findByRole('button',{name:'Cancel fine-tuning'}));
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('segmentation/training/cancel',{job_id:job.id}));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Cancel fine-tuning'})).toBeDisabled());
 expect(collectionRequest.mock.calls.some(([path])=>/workflow\/start|stop/.test(path))).toBe(false);
});


test('SAM fine-tunes a map without opening one of its episodes',async()=>{
 const ep={id:'saved',map_name:'warehouse',ended_at:1,sam_sample_trainable_instance_count:2};
 render(<SamWorkflowControls unified episode={null} episodes={[ep]} renderLayout={({actions})=>actions}/>);
 fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Start fine-tuning'})).toBeEnabled());
 fireEvent.click(screen.getByRole('button',{name:'Start fine-tuning'}));
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('segmentation/auto-train',{map_name:'warehouse'}));
});

test('SAM auto-labels saved map episodes without opening a frame',async()=>{
 const ep={id:'saved',map_name:'warehouse',ended_at:1};
 render(<SamWorkflowControls unified episode={null} episodes={[ep,{...ep,id:'second'},
   {...ep,id:'foreign',map_name:'other'},{...ep,id:'active',ended_at:null}]}
   renderLayout={({actions})=>actions}/>);
 fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
 fireEvent.change(screen.getByLabelText('Auto-label map'),{target:{value:'warehouse'}});
 await screen.findByRole('button',{name:'Table objects v1'});
 fireEvent.click(screen.getByRole('button',{name:'Auto-label map'}));
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('segmentation/auto-label',{
   model_id:'abc',episode_ids:['saved','second'],replace_unreviewed:false}));
});
