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

import { createRef } from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import Sam2Annotation, { imagePoint } from './Sam2Annotation';
const frame={frame_id:'frame1',image:{width:320,height:240},capture:{received_at_ms:1000}};
const props={episode:{id:'ep1',catalog:{missions:[]}},frame,missionId:'mission2',onMissionChange:jest.fn(),onFrameLoaded:jest.fn()};
beforeEach(()=>{
  global.fetch=jest.fn(async(url,options)=>({ok:true,json:async()=>url.endsWith('/regions/batch')?{regions:JSON.parse(options.body).items.map(item=>({...item,revision:item.expected_revision+1,review_status:'approved'}))}:url.endsWith('/segmentation/preview')?{frame_id:'frame1',preview_id:'preview1',overlay_base64:'AA==',area:20,elapsed_seconds:.1}:options?.method==='POST'?{object_id:'saved',object_label:'stick',frame_id:'frame1',revision:1,review_status:'approved'}:{regions:[]}}));
});
afterEach(()=>{delete global.fetch;});

test('maps clicks into original pixels and ignores letterbox margins',()=>{
  const rect={left:10,top:20,width:640,height:240};
  expect(imagePoint(330,140,rect,320,240)).toEqual({x:160,y:120});
  expect(imagePoint(20,140,rect,320,240)).toBeNull();
  expect(imagePoint(330,140,{...rect,width:0},320,240)).toBeNull();
});

test('click creates preview only; loaded mask and explicit save approve without reviewer fields',async()=>{
  const editorRef=createRef();
  const view=render(<Sam2Annotation {...props} ref={editorRef}/>);
  const canvas=screen.getByRole('img',{name:'Saved frame for SAM2 annotation'});
  canvas.getBoundingClientRect=()=>({left:0,top:0,width:320,height:240});
  fireEvent.load(view.container.querySelector('image'));
  fireEvent.click(canvas,{clientX:160,clientY:120});
  await waitFor(()=>expect(view.container.querySelectorAll('image')).toHaveLength(2));
  expect(fetch.mock.calls.filter(([url,options])=>url.endsWith('/regions/batch')&&options?.method==='POST')).toHaveLength(0);
  expect(screen.queryByRole('button',{name:'Save',exact:true})).not.toBeInTheDocument();
  await act(async()=>expect(await editorRef.current.save()).toBe(false));
  fireEvent.change(screen.getByRole('combobox',{name:'Object name'}),{target:{value:'stick'}});
  fireEvent.load(view.container.querySelectorAll('image')[1]);
  expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  expect(screen.queryByRole('textbox',{name:'Mask annotator'})).not.toBeInTheDocument();
  await act(async()=>expect(await editorRef.current.save()).toBe(true));
  await screen.findByText(/Masks saved/);
  await act(async()=>expect(await editorRef.current.save()).toBe(true));
  const post=fetch.mock.calls.find(([url,options])=>url.endsWith('/regions/batch')&&options?.method==='POST');
  expect(JSON.parse(post[1].body).items[0]).toMatchObject({frame_id:'frame1',preview_id:'preview1',reviewed:true,object_label:'stick',expected_revision:0,annotator:'collect-ui',mission_id:'mission2'});
});

test('new point blocks saving until new overlay loads and keeps foreground/background prompts',async()=>{
  const editorRef=createRef();
  const view=render(<Sam2Annotation {...props} ref={editorRef}/>);
  const canvas=screen.getByRole('img',{name:'Saved frame for SAM2 annotation'});
  canvas.getBoundingClientRect=()=>({left:0,top:0,width:320,height:240});
  fireEvent.load(view.container.querySelector('image'));fireEvent.click(canvas,{clientX:100,clientY:100});
  await waitFor(()=>expect(view.container.querySelectorAll('image')).toHaveLength(2));
  fireEvent.load(view.container.querySelectorAll('image')[1]);
  fireEvent.click(screen.getByRole('button',{name:'Exclude point'}));
  fireEvent.click(canvas,{clientX:200,clientY:100});
  await waitFor(()=>expect(fetch.mock.calls.filter(([url])=>url.endsWith('/segmentation/preview'))).toHaveLength(2));
  await act(async()=>expect(await editorRef.current.save()).toBe(false));
  const req=fetch.mock.calls.filter(([url])=>url.endsWith('/segmentation/preview'))[1];
  expect(JSON.parse(req[1].body).points).toEqual([{x:100,y:100,label:1},{x:200,y:100,label:0}]);
});

test('switching away aborts preview and does not turn a late response into a saved mask',async()=>{
  let finish;fetch.mockImplementation(url=>url.endsWith('/segmentation/preview')?new Promise(resolve=>{finish=resolve;}):Promise.resolve({ok:true,json:async()=>({regions:[]})}));
  const editorRef=createRef();
  const view=render(<Sam2Annotation {...props} ref={editorRef}/>);
  const canvas=screen.getByRole('img',{name:'Saved frame for SAM2 annotation'});canvas.getBoundingClientRect=()=>({left:0,top:0,width:320,height:240});
  fireEvent.load(view.container.querySelector('image'));fireEvent.click(canvas,{clientX:100,clientY:100});
  const call=fetch.mock.calls.find(([url])=>url.endsWith('/segmentation/preview'));
  view.unmount();expect(call[1].signal.aborted).toBe(true);
  await act(async()=>finish({ok:true,json:async()=>({frame_id:'frame1',preview_id:'late'})}));
  expect(fetch.mock.calls.filter(([url,opts])=>url.endsWith('/regions')&&opts?.method==='POST')).toHaveLength(0);
});


test('zoom and pan map to original pixels; dragging and view changes do not request masks',async()=>{
  const originalPointerEvent=window.PointerEvent;
  window.PointerEvent=MouseEvent;
  try {
    const view=render(<Sam2Annotation {...props}/>);
    const canvas=screen.getByRole('img',{name:'Saved frame for SAM2 annotation'});
    canvas.getBoundingClientRect=()=>({left:0,top:0,width:320,height:240});
    fireEvent.load(view.container.querySelector('image'));
    fireEvent.click(screen.getByRole('button',{name:'Zoom in'}));
    expect(screen.getByLabelText('Image zoom')).toHaveTextContent('150%');
    fireEvent.pointerDown(canvas,{button:0,clientX:160,clientY:120});
    fireEvent.pointerMove(canvas,{clientX:208,clientY:120});
    fireEvent.pointerUp(canvas,{clientX:208,clientY:120});
    fireEvent.click(canvas,{clientX:208,clientY:120});
    expect(fetch.mock.calls.filter(([url])=>url.endsWith('/segmentation/preview'))).toHaveLength(0);
    fireEvent.click(canvas,{clientX:96,clientY:120});
    await waitFor(()=>expect(view.container.querySelectorAll('image')).toHaveLength(2));
    const call=fetch.mock.calls.find(([url])=>url.endsWith('/segmentation/preview'));
    expect(JSON.parse(call[1].body).points).toEqual([{x:85,y:120,label:1}]);
    fireEvent.click(screen.getByRole('button',{name:'Fit image'}));
    expect(canvas).toHaveAttribute('viewBox','0 0 320 240');
    expect(screen.getByRole('button',{name:'Zoom out'})).toBeDisabled();
    expect(view.container.querySelectorAll('image')).toHaveLength(2);
    expect(fetch.mock.calls.filter(([url])=>url.endsWith('/segmentation/preview'))).toHaveLength(1);
    for(let i=0;i<10;i++)fireEvent.click(screen.getByRole('button',{name:'Zoom in'}));
    expect(screen.getByLabelText('Image zoom')).toHaveTextContent('800%');
    expect(screen.getByRole('button',{name:'Zoom in'})).toBeDisabled();
  } finally {window.PointerEvent=originalPointerEvent;}
});

test('expanded view preserves the editor and follows native fullscreen exit',async()=>{
  const view=render(<Sam2Annotation {...props}/>);
  const editor=view.container.querySelector('.sam2-image-editor');
  editor.requestFullscreen=jest.fn(async()=>{
    Object.defineProperty(document,'fullscreenElement',{configurable:true,value:editor});
    document.dispatchEvent(new Event('fullscreenchange'));
  });
  try {
    fireEvent.click(screen.getByRole('button',{name:'Zoom in'}));
    fireEvent.click(screen.getByRole('button',{name:'Expand view'}));
    await screen.findByRole('button',{name:'Exit expanded view'});
    act(()=>{
      Object.defineProperty(document,'fullscreenElement',{configurable:true,value:null});
      document.dispatchEvent(new Event('fullscreenchange'));
    });
    expect(screen.getByRole('button',{name:'Expand view'})).toBeInTheDocument();
    expect(screen.getByLabelText('Image zoom')).toHaveTextContent('150%');
  }finally{delete document.fullscreenElement;}
});


test('saving without a mask requires no segmentation request', async()=>{
  const ref=createRef(); render(<Sam2Annotation {...props} ref={ref}/>);
  await act(async()=>expect(await ref.current.save()).toBe(true));
  expect(fetch.mock.calls.some(([,options])=>options?.method==='POST')).toBe(false);
});

test('mask storage failure is returned to the unified save caller', async()=>{
  const ref=createRef();const view=render(<Sam2Annotation {...props} ref={ref}/>);
  const canvas=screen.getByRole('img',{name:'Saved frame for SAM2 annotation'});
  canvas.getBoundingClientRect=()=>({left:0,top:0,width:320,height:240});
  fireEvent.load(view.container.querySelector('image'));
  fireEvent.click(canvas,{clientX:100,clientY:100});
  await waitFor(()=>expect(view.container.querySelectorAll('image')).toHaveLength(2));
  fireEvent.load(view.container.querySelectorAll('image')[1]);
  fireEvent.change(screen.getByRole('combobox',{name:'Object name'}),{target:{value:'stick'}});
  fetch.mockImplementation(async()=>({ok:false,json:async()=>({detail:'Disk full'})}));
  await act(async()=>expect(await ref.current.save()).toBe(false));
  expect(screen.getByRole('alert')).toHaveTextContent('Disk full');
});

test('same-label instances are separately selectable and new instance clears selection', async()=>{
 const regions=[1,2].map(n=>({object_id:`auto-${n}`,object_label:'stick',frame_id:'frame1',revision:1,review_status:'draft',points:[]}));
 fetch.mockImplementation(async()=>({ok:true,json:async()=>({regions})}));
 render(<Sam2Annotation episode={{id:'ep'}} frame={{frame_id:'frame1',image:{width:640,height:480}}} standaloneSave reviewMode/>);
 const second=await screen.findByRole('button',{name:'stick #2'});
 fireEvent.click(second);
 expect(second).toHaveAttribute('aria-pressed','true');
 expect(screen.getByRole('button',{name:'stick #1'})).toHaveAttribute('aria-pressed','false');
 fireEvent.click(screen.getByRole('button',{name:'New instance'}));
 expect(second).toHaveAttribute('aria-pressed','false');
 expect(screen.getByRole('combobox',{name:'Object name'})).toHaveValue('');
});

test('two objects stay visible and Save commits both in one batch',async()=>{
 const view=render(<Sam2Annotation {...props} standaloneSave/>);
 const canvas=screen.getByRole('img',{name:'Saved frame for SAM2 annotation'});
 canvas.getBoundingClientRect=()=>({left:0,top:0,width:320,height:240});
 fireEvent.load(view.container.querySelector('image'));
 fireEvent.change(screen.getByRole('combobox',{name:'Object name'}),{target:{value:'stick'}});
 fireEvent.click(canvas,{clientX:60,clientY:80});
 await waitFor(()=>expect(view.container.querySelectorAll('image')).toHaveLength(2));
 fireEvent.load(view.container.querySelectorAll('image')[1]);
 fireEvent.click(screen.getByRole('button',{name:'New instance',exact:true}));
 expect(fetch.mock.calls.filter(([url,opts])=>url.endsWith('/regions/batch')&&opts?.method==='POST')).toHaveLength(0);
 expect(screen.getByRole('button',{name:/stick #1 · Unsaved/})).toBeInTheDocument();
 fireEvent.change(screen.getByRole('combobox',{name:'Object name'}),{target:{value:'box'}});
 fireEvent.click(canvas,{clientX:200,clientY:100});
 await waitFor(()=>expect(view.container.querySelectorAll('image')).toHaveLength(3));
 fireEvent.load(view.container.querySelectorAll('image')[2]);
 fireEvent.click(screen.getByRole('button',{name:'Save',exact:true}));
 await screen.findByText('Masks saved · 2 instances');
 const calls=fetch.mock.calls.filter(([url,opts])=>url.endsWith('/regions/batch')&&opts?.method==='POST');
 expect(calls).toHaveLength(1);
 const items=JSON.parse(calls[0][1].body).items;
 expect(items.map(item=>item.object_label)).toEqual(['stick','box']);
 expect(new Set(items.map(item=>item.object_id)).size).toBe(2);
});

test('discarding a staged object removes it without saving',async()=>{
 const view=render(<Sam2Annotation {...props} standaloneSave/>);
 const canvas=screen.getByRole('img',{name:'Saved frame for SAM2 annotation'});
 canvas.getBoundingClientRect=()=>({left:0,top:0,width:320,height:240});
 fireEvent.load(view.container.querySelector('image'));
 fireEvent.change(screen.getByRole('combobox',{name:'Object name'}),{target:{value:'stick'}});
 fireEvent.click(canvas,{clientX:60,clientY:80});
 await waitFor(()=>expect(view.container.querySelectorAll('image')).toHaveLength(2));
 fireEvent.load(view.container.querySelectorAll('image')[1]);
 fireEvent.click(screen.getByRole('button',{name:'New instance',exact:true}));
 fireEvent.click(screen.getByRole('button',{name:'Remove instance stick 1'}));
 expect(screen.queryByRole('button',{name:/stick #1 · Unsaved/})).not.toBeInTheDocument();
 expect(screen.getByRole('button',{name:'Save',exact:true})).toBeDisabled();
 expect(fetch.mock.calls.some(([url])=>url.endsWith('/regions/batch'))).toBe(false);
});

test('Ctrl wheel zooms the image at the cursor and cancels browser zoom only with Ctrl',()=>{
 render(<Sam2Annotation {...props}/>);
 const canvas=screen.getByRole('img',{name:'Saved frame for SAM2 annotation'});
 canvas.getBoundingClientRect=()=>({left:0,top:0,width:320,height:240});
 const plain=new WheelEvent('wheel',{deltaY:-100,clientX:160,clientY:120,bubbles:true,cancelable:true});
 act(()=>canvas.dispatchEvent(plain));
 expect(plain.defaultPrevented).toBe(false);
 expect(screen.getByLabelText('Image zoom')).toHaveTextContent('100%');
 const ctrl=new WheelEvent('wheel',{ctrlKey:true,deltaY:-100,clientX:160,clientY:120,bubbles:true,cancelable:true});
 act(()=>canvas.dispatchEvent(ctrl));
 expect(ctrl.defaultPrevented).toBe(true);
 expect(screen.getByLabelText('Image zoom')).toHaveTextContent('120%');
 const [x,y,w,h]=canvas.getAttribute('viewBox').split(' ').map(Number);
 expect(x+w/2).toBeCloseTo(160);expect(y+h/2).toBeCloseTo(120);
 fireEvent.wheel(canvas,{ctrlKey:true,deltaY:100,clientX:160,clientY:120});
 expect(screen.getByLabelText('Image zoom')).toHaveTextContent('100%');
});

function reviewFixture() {
 const regions=['person','dish','empty dish'].map((object_label,i)=>({
   object_id:`auto-${i}`,object_label,frame_id:'frame1',revision:1,review_status:'draft',points:[],
 }));
 regions.push({...regions[0],object_id:'rejected',review_status:'rejected'});
 regions.push({...regions[0],object_id:'other-frame',frame_id:'frame2'});
 const fallback=fetch.getMockImplementation();
 fetch.mockImplementation(async(url,options)=>{
   if(url.endsWith('/regions')) return {ok:true,json:async()=>({regions})};
   if(url.endsWith('/regions/batch')) {
     const body=JSON.parse(options.body);
     const saved=[...body.items.map(item=>({...item,revision:item.expected_revision+1,review_status:'approved'})),
       ...body.removed.map(item=>({...regions.find(r=>r.object_id===item.object_id),revision:item.expected_revision+1,review_status:'rejected'}))];
     return {ok:true,json:async()=>({regions:saved})};
   }
   return fallback(url,options);
 });
 return regions;
}
const loadMasks = view => view.container.querySelectorAll('image').forEach(image=>fireEvent.load(image));
const reviewPosts = () => fetch.mock.calls.filter(([url])=>url.endsWith('/regions/batch'));

test('frame review displays every class and approves them together only after all overlays load',async()=>{
 reviewFixture(); const onMaskSaved=jest.fn();
 const view=render(<Sam2Annotation {...props} standaloneSave reviewMode onMaskSaved={onMaskSaved}/>);
 await waitFor(()=>expect(screen.getByRole('button',{name:'person #1'})).toHaveAttribute('aria-pressed','true'));
 const images=view.container.querySelectorAll('image');
 expect(images).toHaveLength(4); // Original frame plus all three classes.
 expect([...view.container.querySelectorAll('svg text')].map(text=>text.textContent).sort()).toEqual(['dish','empty dish','person']);
 expect(screen.getByRole('list',{name:'Frame instances'}).children).toHaveLength(3);
 expect(view.container.querySelector('image[href*="/regions/rejected/"]')).toBeNull();
 expect(view.container.querySelector('image[href*="/regions/other-frame/"]')).toBeNull();
 expect(screen.queryByRole('button',{name:'Approve mask'})).not.toBeInTheDocument();
 expect(screen.queryByRole('button',{name:'Reject mask'})).not.toBeInTheDocument();
 const save=screen.getByRole('button',{name:'Save review'});
 expect(save).toBeDisabled();
 [images[0],images[1],images[3]].forEach(image=>fireEvent.load(image));
 expect(save).toBeDisabled();
 fireEvent.load(images[2]); expect(save).toBeEnabled();
 fireEvent.click(save);
 await screen.findByText('Frame review saved.');
 expect(reviewPosts()).toHaveLength(1);
 const body=JSON.parse(reviewPosts()[0][1].body);
 expect(body.frame_id).toBe('frame1');
 expect(body.items.map(item=>item.object_label)).toEqual(['person','dish','empty dish']);
 expect(body.items.every(item=>item.preview_id===null&&item.reviewed&&item.expected_revision===1)).toBe(true);
 expect(body.removed).toEqual([]);
 expect(onMaskSaved).toHaveBeenCalledWith({advance:true});
});

test('frame review saves corrected labels, exclusions and untouched drafts without reapproving saved masks',async()=>{
 const regions=reviewFixture();
 regions.push({...regions[0],object_id:'human',object_label:'table',review_status:'approved'});
 const onMaskSaved=jest.fn();
 const view=render(<Sam2Annotation {...props} standaloneSave reviewMode onMaskSaved={onMaskSaved}/>);
 await screen.findByRole('button',{name:'dish #2'});
 loadMasks(view);
 fireEvent.click(screen.getByRole('button',{name:'dish #2'}));
 loadMasks(view);
 fireEvent.change(screen.getByRole('combobox',{name:'Object name'}),{target:{value:'plate'}});
 fireEvent.click(screen.getByRole('button',{name:'Remove instance empty dish 3'}));
 expect(reviewPosts()).toHaveLength(0); // Exclusions are staged until Save review.
 fireEvent.click(screen.getByRole('button',{name:'Save review'}));
 await screen.findByText('Frame review saved.');
 const body=JSON.parse(reviewPosts()[0][1].body);
 expect(body.items.map(item=>item.object_label).sort()).toEqual(['person','plate']);
 expect(body.removed).toEqual([{object_id:'auto-2',expected_revision:1}]);
 expect(body.items.some(item=>item.object_id==='human')).toBe(false);
 expect(onMaskSaved).toHaveBeenCalledWith({advance:true});
});

test('rejecting every mask can be saved as one review and a failed save preserves exclusions for retry',async()=>{
 reviewFixture(); const onMaskSaved=jest.fn();
 const view=render(<Sam2Annotation {...props} standaloneSave reviewMode onMaskSaved={onMaskSaved}/>);
 await screen.findByRole('button',{name:'person #1'});
 loadMasks(view);
 screen.getAllByRole('button',{name:/Remove instance/}).forEach(button=>fireEvent.click(button));
 expect(view.container.querySelectorAll('image')).toHaveLength(1);
 const fallback=fetch.getMockImplementation();
 fetch.mockImplementationOnce(async()=>({ok:false,json:async()=>({detail:'Disk full'})}));
 fireEvent.click(screen.getByRole('button',{name:'Save review'}));
 await screen.findByText('Disk full');
 expect(onMaskSaved).not.toHaveBeenCalled();
 expect(screen.getByRole('button',{name:'Save review'})).toBeEnabled();
 expect(screen.queryByText('Unsaved changes')).not.toBeInTheDocument();
 expect(screen.queryByRole('button',{name:'Discard changes'})).not.toBeInTheDocument();
 fetch.mockImplementation(fallback);
 fireEvent.click(screen.getByRole('button',{name:'Save review'}));
 await screen.findByText('Frame review saved.');
 const body=JSON.parse(reviewPosts()[1][1].body);
 expect(body.items).toEqual([]);
 expect(body.removed.map(item=>item.object_id)).toEqual(['auto-0','auto-1','auto-2']);
 expect(onMaskSaved).toHaveBeenCalledTimes(1);
 expect(onMaskSaved).toHaveBeenCalledWith({advance:true});
});


test('an empty frame can be reviewed after image and mask list load',async()=>{
 const onMaskSaved=jest.fn();
 const view=render(<Sam2Annotation {...props} standaloneSave reviewMode onMaskSaved={onMaskSaved}/>);
 const button=screen.getByRole('button',{name:'Save review'});
 expect(button).toBeDisabled();
 fireEvent.load(view.container.querySelector('image'));
 await waitFor(()=>expect(button).toBeEnabled());
 fireEvent.click(button);
 await waitFor(()=>expect(onMaskSaved).toHaveBeenCalledWith({advance:true}));
 const call=fetch.mock.calls.find(([url,options])=>url.endsWith('/regions/batch')&&options?.method==='POST');
 expect(JSON.parse(call[1].body)).toEqual({frame_id:'frame1',review_empty:true,items:[],removed:[]});
});
test('regeneration replaces the selected overlay instead of retaining the original draft',async()=>{
 jest.useFakeTimers();
 try {
  let regions=[{object_id:'old',frame_id:'frame1',object_label:'person',revision:1,review_status:'draft',points:[]}];
  global.fetch=jest.fn(async()=>({ok:true,json:async()=>({regions})}));
  const view=render(<Sam2Annotation {...props} reviewMode/>);
  await act(async()=>{await Promise.resolve();await Promise.resolve();});
  expect(view.container.querySelector('image[href*="/old/overlay"]')).not.toBeNull();
  regions=[{...regions[0],object_id:'new'}];
  await act(async()=>{jest.advanceTimersByTime(3000);});
  expect(view.container.querySelector('image[href*="/old/overlay"]')).toBeNull();
  expect(view.container.querySelector('image[href*="/new/overlay"]')).not.toBeNull();
  regions=[];
  await act(async()=>{jest.advanceTimersByTime(3000);});
  expect(view.container.querySelector('image[href*="/new/overlay"]')).toBeNull();
  view.unmount();
 } finally {jest.useRealTimers();}
});


test('preview validation errors are shown with the failing field and retry stays available',async()=>{
  const original=fetch.getMockImplementation();
  fetch.mockImplementation((url,options)=>url.endsWith('/segmentation/preview')
    ? Promise.resolve({ok:false,json:async()=>({detail:[{loc:['body','points',0,'x'],msg:'Invalid coordinate'}]})})
    : original(url,options));
  const view=render(<Sam2Annotation {...props}/>);
  const canvas=screen.getByRole('img',{name:'Saved frame for SAM2 annotation'});
  canvas.getBoundingClientRect=()=>({left:0,top:0,width:320,height:240});
  fireEvent.load(view.container.querySelector('image'));
  fireEvent.click(screen.getByRole('button',{name:'Include point'}));
  expect(fetch.mock.calls.some(([url])=>url.endsWith('/segmentation/preview'))).toBe(false);
  fireEvent.click(canvas,{clientX:160,clientY:120});
  expect(await screen.findByRole('alert')).toHaveTextContent('points.0.x: Invalid coordinate');
  expect(screen.getByRole('button',{name:'Retry mask'})).toBeEnabled();
  expect(fetch.mock.calls.some(([url])=>url.endsWith('/segment'))).toBe(false);
});
