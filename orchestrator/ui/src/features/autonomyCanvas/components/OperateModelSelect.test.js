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

import { fireEvent, render, screen } from '@testing-library/react';
import OperateModelSelect from './OperateModelSelect';
test('selects by readable name and locks changes during operation',()=>{
 const model={id:'tuned',model_name:'Warehouse Qwen',base_model:'Qwen/Qwen3.5-4B',sample_count:30};
 const choice={models:[model],selected:null,select:jest.fn(),refresh:jest.fn()};
 const h=render(<OperateModelSelect choice={choice} />);
 fireEvent.change(screen.getByLabelText('Reasoning model'),{target:{value:'tuned'}});
 expect(choice.select).toHaveBeenCalledWith('tuned');
 h.rerender(<OperateModelSelect choice={{...choice,selected:model,locked:true}} active loadedModel={{selected_model_id:'tuned'}} />);
 expect(screen.getByLabelText('Reasoning model')).toBeDisabled();
 expect(screen.getByLabelText('Refresh reasoning models')).toBeDisabled();
 expect(screen.getByRole('status')).toHaveTextContent('Loaded · Qwen/Qwen3.5-4B · 30 samples');
});

test('marks a catalog-incompatible model but keeps explicit selection available',()=>{
 const model={id:'stale',model_name:'Shelf Qwen',base_model:'Qwen/Qwen3.5-4B',sample_count:20,catalog_compatible:false};
 const choice={models:[model],selected:model,select:jest.fn(),refresh:jest.fn()};
 render(<OperateModelSelect choice={choice} />);

 expect(screen.getByRole('option',{name:'Shelf Qwen (Design changed)'})).toBeInTheDocument();
 expect(screen.getByLabelText('Reasoning model')).toBeEnabled();
 expect(screen.getByRole('status')).toHaveTextContent('Selected · Design changed since training. Retrain for the current missions.');
});


test('offers the untuned 0.8B vision-language model for preview',()=>{
 const model={id:'base:Qwen3.5-0.8B',model_name:'Qwen3.5-0.8B',base_model:'Qwen/Qwen3.5-0.8B',sample_count:0,source:'base'};
 const choice={models:[model],selected:model,select:jest.fn(),refresh:jest.fn()};
 render(<OperateModelSelect choice={choice}/>);
 expect(screen.getByRole('option',{name:'Qwen3.5-0.8B (Base · Untuned)'})).toBeInTheDocument();
 expect(screen.getByRole('status')).toHaveTextContent('Selected · Qwen/Qwen3.5-0.8B · Untuned base model');
});

test('offers the untuned 2B vision-language model for preview',()=>{
 const model={id:'base:Qwen3.5-2B',model_name:'Qwen3.5-2B',base_model:'Qwen/Qwen3.5-2B',sample_count:0,source:'base'};
 const choice={models:[model],selected:model,select:jest.fn(),refresh:jest.fn()};
 render(<OperateModelSelect choice={choice}/>);
 expect(screen.getByRole('option',{name:'Qwen3.5-2B (Base · Untuned)'})).toBeInTheDocument();
 expect(screen.getByRole('status')).toHaveTextContent('Selected · Qwen/Qwen3.5-2B · Untuned base model');
});
