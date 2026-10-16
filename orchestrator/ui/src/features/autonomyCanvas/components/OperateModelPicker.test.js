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
import OperateModelPicker from './OperateModelPicker';

test('icon opens model choices, selection closes them, and Escape restores focus', () => {
  const choice = {models:[{id:'base',model_name:'Base'},{id:'tuned',model_name:'Tuned'}],select:jest.fn()};
  render(<OperateModelPicker choice={choice}/>);
  const trigger = screen.getByRole('button',{name:'Select reasoning model'});
  expect(screen.queryByRole('group',{name:'Reasoning models'})).not.toBeInTheDocument();
  fireEvent.click(trigger);
  fireEvent.click(screen.getByRole('button',{name:'Tuned'}));
  expect(choice.select).toHaveBeenCalledWith('tuned');
  expect(trigger).toHaveAttribute('aria-expanded','false');
  fireEvent.click(trigger);
  fireEvent.keyDown(document,{key:'Escape'});
  expect(trigger).toHaveFocus();
  expect(trigger).toHaveAttribute('aria-expanded','false');
});

test('active inference closes choices and locks model switching', () => {
  const choice = {models:[{id:'base',model_name:'Base'}],select:jest.fn()};
  const view=render(<OperateModelPicker choice={choice}/>);
  fireEvent.click(screen.getByRole('button',{name:'Select reasoning model'}));
  view.rerender(<OperateModelPicker choice={choice} active/>);
  expect(screen.getByRole('button',{name:'Select reasoning model'})).toBeDisabled();
  expect(screen.queryByRole('group',{name:'Reasoning models'})).not.toBeInTheDocument();
});
