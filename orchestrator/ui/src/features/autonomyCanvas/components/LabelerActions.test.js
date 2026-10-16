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

import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import LabelerActions from './LabelerActions';
import LabelingWorkflowBar from './LabelingWorkflowBar';

const model={id:'one',model_name:'sam_house_1'};
const props={models:[model],selected:model,episodeId:'ep',onSelect:jest.fn(),onTrain:jest.fn(),onGenerate:jest.fn(),onDelete:jest.fn(),canTrain:true,canGenerate:true};

beforeEach(()=>jest.clearAllMocks());

test('one click trains, auto-label opens only model selection, deletion is explicit',async()=>{
 render(<LabelerActions {...props}/>);
 fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));
 expect(props.onTrain).toHaveBeenCalledTimes(1);
 expect(screen.queryByRole('region',{name:'Fine-tuning details'})).not.toBeInTheDocument();
 const autoLabel=screen.getByRole('button',{name:'Auto-label'});
 expect(autoLabel).toHaveAttribute('aria-expanded','false');
 fireEvent.click(autoLabel);
 expect(autoLabel).toHaveAttribute('aria-expanded','true');
 expect(screen.getByRole('button',{name:'sam_house_1',exact:true})).toHaveAttribute('aria-pressed','true');
 fireEvent.click(screen.getByRole('button',{name:'Auto-label episode'}));
 expect(props.onGenerate).toHaveBeenCalledTimes(1);
 expect(autoLabel).toHaveAttribute('aria-expanded','false');
 fireEvent.click(autoLabel);
 fireEvent.click(screen.getByRole('button',{name:'Delete sam_house_1'}));
 expect(props.onDelete).not.toHaveBeenCalled();
 fireEvent.click(screen.getByRole('button',{name:'Delete',exact:true}));
 expect(props.onDelete).toHaveBeenCalledWith('one');
 await waitFor(()=>expect(screen.queryByRole('button',{name:'Delete',exact:true})).not.toBeInTheDocument());
});

test('a running fine-tune exposes its progress and details without starting another job',()=>{
 render(<LabelerActions {...props} disabled trainingJob={{status:'training',steps:200,step:50,model_name:'sam_house_1'}}/>);
 expect(screen.getByRole('progressbar',{name:'Fine-tune progress'})).toHaveAttribute('aria-valuenow','25');
 const fineTune=screen.getByRole('button',{name:'Fine-tune'});
 expect(fineTune).toBeEnabled();
 fireEvent.click(fineTune);
 expect(screen.getByRole('region',{name:'Fine-tuning details'})).toBeVisible();
 expect(props.onTrain).not.toHaveBeenCalled();
 fireEvent.keyDown(document,{key:'Escape'});
 expect(screen.queryByRole('region',{name:'Fine-tuning details'})).not.toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
 expect(screen.getByRole('button',{name:'Delete sam_house_1'})).toBeDisabled();
});

test('Review enters only when pending generated labels are available',()=>{
 const change=jest.fn();
 const view=render(<LabelingWorkflowBar frameScope="samples" sampleCount={3} pendingReviewCount={0} onFrameScopeChange={change}/>);
 expect(screen.queryByRole('button',{name:'Sample Labeling'})).not.toBeInTheDocument();
 expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
 expect(screen.getByRole('button',{name:'Review'})).toBeDisabled();
 view.rerender(<LabelingWorkflowBar frameScope="samples" sampleCount={3} pendingReviewCount={2} onFrameScopeChange={change}/>);
 expect(screen.getByLabelText('Pending reviews')).toHaveTextContent('2');
 fireEvent.click(screen.getByRole('button',{name:'Review'}));
 expect(change).toHaveBeenCalledWith('review');
});

test.each([[3,'all'],[0,'all']])('Review can return to %s saved sample frames after its last pending review', (sampleCount,scope)=>{
 const change=jest.fn();
 render(<LabelingWorkflowBar frameScope="review" sampleCount={sampleCount} pendingReviewCount={0} onFrameScopeChange={change}/>);
 const review=screen.getByRole('button',{name:'Review'});
 expect(review).toHaveAttribute('aria-pressed','true');
 expect(review).toBeEnabled();
 fireEvent.click(review);
 expect(change).toHaveBeenCalledWith(scope);
});

test.each(['samples','review'])('a global lock prevents changing the %s scope',frameScope=>{
 const change=jest.fn();
 render(<LabelingWorkflowBar frameScope={frameScope} sampleCount={3} pendingReviewCount={2} disabled onFrameScopeChange={change}/>);
 const review=screen.getByRole('button',{name:'Review'});
 expect(review).toBeDisabled();
 fireEvent.click(review);
 expect(change).not.toHaveBeenCalled();
});


test('running job cancellation stays available while training locks the editor',()=>{
 const onCancelTrain=jest.fn();
 const view=render(<LabelerActions {...props} disabled onCancelTrain={onCancelTrain}
   trainingJob={{status:'training',steps:200,step:50}}/>);
 const cancel=screen.getByRole('button',{name:'Cancel fine-tuning'});
 expect(cancel).toBeEnabled();fireEvent.click(cancel);expect(onCancelTrain).toHaveBeenCalledTimes(1);
 view.rerender(<LabelerActions {...props} disabled onCancelTrain={onCancelTrain}
   trainingJob={{status:'cancelling',steps:200,step:50}}/>);
 expect(screen.getByRole('button',{name:'Cancel fine-tuning'})).toBeDisabled();
 view.rerender(<LabelerActions {...props} onCancelTrain={onCancelTrain}
   trainingJob={{status:'cancelled',steps:200,step:50}}/>);
 expect(screen.queryByRole('button',{name:'Cancel fine-tuning'})).not.toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));expect(props.onTrain).toHaveBeenCalledTimes(1);
});

test('a failed cancellation can be retried without releasing the job lock',()=>{
 const onCancelTrain=jest.fn();
 render(<LabelerActions {...props} disabled onCancelTrain={onCancelTrain}
   trainingJob={{status:'cancelling',error:'Docker unavailable'}}/>);
 expect(screen.getByRole('button',{name:'Cancel fine-tuning'})).toBeEnabled();
});
