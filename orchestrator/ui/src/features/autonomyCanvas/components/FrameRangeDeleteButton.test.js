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

import { act, fireEvent, render, screen } from '@testing-library/react';
import FrameRangeDeleteButton, { frameIndexAt } from './FrameRangeDeleteButton';

const episode = {id:'episode-a',ended_at:1,storage_path:'house/raw_episode_0004',observations:Array.from({length:10},(_,i)=>({frame_id:`frame-${i}`}))};
function setup(props={}) {
  const onDelete = props.onDelete || jest.fn(async()=>true);
  const view = render(<FrameRangeDeleteButton episode={episode} {...props} onDelete={onDelete}/>);
  fireEvent.click(screen.getByRole('button',{name:'Delete frame range'}));
  const track = screen.getByRole('slider',{name:'Frame range'});
  track.getBoundingClientRect=()=>({left:10,width:100,right:110});
  track.setPointerCapture=jest.fn();
  track.hasPointerCapture=()=>true;
  track.releasePointerCapture=jest.fn();
  return {view,onDelete,track};
}
function pointer(target,type,clientX,pointerId=1) {
  const event = new Event(type,{bubbles:true});
  Object.assign(event,{clientX,pointerId,button:0});
  fireEvent(target,event);
}
function drag(track,from,to) {pointer(track,'pointerdown',from);pointer(track,'pointermove',to);pointer(track,'pointerup',to);}
function expectRange(from,to) {
  expect(screen.getByLabelText('From')).toHaveValue(from);
  expect(screen.getByLabelText('To')).toHaveValue(to);
}

test('forward drag selects an inclusive saved-frame range without deleting',()=>{
  const {track,onDelete}=setup();
  drag(track,31,71);
  expectRange(3,7);
  expect(screen.getByText('5 selected')).toBeInTheDocument();
  expect(screen.getByAltText('From selected frame 3')).toHaveAttribute('src','/api/collection/episodes/episode-a/images/frame-2');
  expect(screen.getAllByRole('img')).toHaveLength(2);
  expect(onDelete).not.toHaveBeenCalled();
});

test('reverse drag and movement outside the track clamp to episode boundaries',()=>{
  const {track}=setup();
  drag(track,91,21); expectRange(2,9);
  drag(track,41,-100); expectRange(1,4);
  drag(track,41,900); expectRange(4,10);
  expect(track.setPointerCapture).toHaveBeenCalledWith(1);
  expect(track.releasePointerCapture).toHaveBeenCalledWith(1);
});

test('pointer coordinates are bounded and zero-width tracks are safe',()=>{
  expect(frameIndexAt(-5,{left:10,width:100},10)).toBe(0);
  expect(frameIndexAt(110,{left:10,width:100},10)).toBe(9);
  expect(frameIndexAt(50,{left:10,width:0},10)).toBe(0);
  expect(frameIndexAt(50,{left:10,width:100},0)).toBe(0);
});

test('numeric endpoints and keyboard selection work without pointer input',()=>{
  const {track}=setup();
  fireEvent.change(screen.getByLabelText('From'),{target:{value:'3'}});
  fireEvent.change(screen.getByLabelText('To'),{target:{value:'8'}});
  expectRange(3,8);
  fireEvent.keyDown(track,{key:'ArrowLeft'}); expectRange(3,7);
  fireEvent.keyDown(track,{key:'End'}); expectRange(3,10);
  fireEvent.keyDown(track,{key:'Home'}); expectRange(1,3);
});

test('delete confirms exact count then sends frame IDs in saved order',async()=>{
  const {track,onDelete}=setup();
  drag(track,61,31);
  fireEvent.click(screen.getByRole('button',{name:'Delete 4 frames'}));
  expect(screen.getByText('Delete 4 frames and their images, labels, and masks? This cannot be undone.')).toBeInTheDocument();
  expect(onDelete).not.toHaveBeenCalled();
  await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Delete permanently'})));
  expect(onDelete).toHaveBeenCalledWith(['frame-2','frame-3','frame-4','frame-5']);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

test('keeping frames, cancellation and Escape leave data unchanged',()=>{
  const {track,onDelete}=setup();
  drag(track,11,41);
  fireEvent.click(screen.getByRole('button',{name:'Delete 4 frames'}));
  fireEvent.click(screen.getByRole('button',{name:'Keep frames'}));
  expectRange(1,4);
  fireEvent.click(screen.getByRole('button',{name:'Cancel'}));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Delete frame range'}));
  fireEvent.keyDown(screen.getByRole('dialog'),{key:'Escape'});
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(onDelete).not.toHaveBeenCalled();
});

test('delete failure preserves selection and displays the error',async()=>{
  const {track}=setup({onDelete:jest.fn(async()=>{throw new Error('Frame is in use.');})});
  drag(track,21,51);
  fireEvent.click(screen.getByRole('button',{name:'Delete 4 frames'}));
  await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Delete permanently'})));
  expectRange(2,5);
  expect(screen.getByRole('alert')).toHaveTextContent('Frame is in use.');
  expect(screen.getByRole('button',{name:'Delete 4 frames'})).toBeEnabled();
});

test('false delete response is an error, never a successful close',async()=>{
  const {track}=setup({onDelete:jest.fn(async()=>false)});
  drag(track,11,11);
  fireEvent.click(screen.getByRole('button',{name:'Delete 1 frame'}));
  await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Delete permanently'})));
  expect(screen.getByRole('dialog')).toBeInTheDocument();
  expect(screen.getByRole('alert')).toHaveTextContent('Could not delete');
});

test('pending deletion locks selection, prevents duplicate submission and cannot close',async()=>{
  let finish;
  const onDelete=jest.fn(()=>new Promise(resolve=>{finish=resolve;}));
  const {track}=setup({onDelete});
  drag(track,11,31);
  fireEvent.click(screen.getByRole('button',{name:'Delete 3 frames'}));
  fireEvent.click(screen.getByRole('button',{name:'Delete permanently'}));
  expect(screen.getByRole('button',{name:'Deleting…'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Close frame selection'})).toBeDisabled();
  expect(screen.getByLabelText('From')).toBeDisabled();
  fireEvent.keyDown(screen.getByRole('dialog'),{key:'Escape'});
  expect(screen.getByRole('dialog')).toBeInTheDocument();
  expect(onDelete).toHaveBeenCalledTimes(1);
  await act(async()=>finish(true));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

test('episode switch closes and resets the range; sampled IDs never restrict deletion',()=>{
  const sampled={...episode,qwen_sample:{frame_ids:['frame-0']},sam_sample:{frame_ids:['frame-1']}};
  const {view,track,onDelete}=setup({episode:sampled});
  drag(track,61,81); expectRange(6,8);
  view.rerender(<FrameRangeDeleteButton episode={{...episode,id:'episode-b'}} onDelete={onDelete}/>);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Delete frame range'}));
  expect(screen.getByText('0 selected')).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Delete 0 frames'})).toBeDisabled();
});

test('empty and disabled episodes cannot open deletion',()=>{
  const {rerender}=render(<FrameRangeDeleteButton episode={{...episode,observations:[]}}/>);
  expect(screen.getByRole('button',{name:'Delete frame range'})).toBeDisabled();
  rerender(<FrameRangeDeleteButton episode={episode} disabled/>);
  expect(screen.getByRole('button',{name:'Delete frame range'})).toBeDisabled();
});


test('dragging defers image requests until endpoints are selected',()=>{
  const {track}=setup();
  pointer(track,'pointerdown',21);
  pointer(track,'pointermove',71);
  expect(screen.queryAllByRole('img')).toHaveLength(0);
  pointer(track,'pointerup',71);
  expect(screen.getAllByRole('img')).toHaveLength(2);
  expectRange(2,7);
});


test('fractional and out-of-range numeric edits keep integer endpoint indices',()=>{
  setup();
  fireEvent.change(screen.getByLabelText('From'),{target:{value:'2.9'}});
  fireEvent.change(screen.getByLabelText('To'),{target:{value:'7.8'}});
  expectRange(2,7);
  expect(screen.getByAltText('From selected frame 2')).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('From'),{target:{value:'-3'}});
  fireEvent.change(screen.getByLabelText('To'),{target:{value:'99'}});
  expectRange(1,10);
});

test('active or unsaved episodes cannot open deletion',()=>{
  const {rerender}=render(<FrameRangeDeleteButton episode={{...episode,ended_at:null}}/>);
  expect(screen.getByRole('button',{name:'Delete frame range'})).toBeDisabled();
  rerender(<FrameRangeDeleteButton episode={{...episode,id:null}}/>);
  expect(screen.getByRole('button',{name:'Delete frame range'})).toBeDisabled();
});
