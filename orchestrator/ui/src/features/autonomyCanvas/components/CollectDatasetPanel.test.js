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

import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { useState } from 'react';
import CollectDatasetPanel from './CollectDatasetPanel';
import useCollectAnnotation from '../hooks/useCollectAnnotation';
import { collectionRequest } from '../../../utils/collectionApi';

jest.mock('../../../utils/collectionApi', () => ({ collectionRequest: jest.fn() }));
jest.mock('./Sam2Annotation', () => {
  const React = require('react');
  return React.forwardRef(({ frame, onFrameLoaded, onDirtyChange, onMissionChange, onMaskSaved, renderLayout }, ref) => {
    React.useEffect(() => { onFrameLoaded(frame.frame_id); }, [frame.frame_id, onFrameLoaded]);
    const content = <div aria-label="Mask editor">
      <span>{frame.frame_id}</span>
      <button onClick={() => onDirtyChange(true)}>Draft mask</button>
      <button onClick={() => { onDirtyChange(false); onMissionChange('mask-mission'); }}>Discard mask</button>
      <button onClick={() => onMaskSaved({advance:true})}>Save mask</button>
    </div>;
    return renderLayout ? renderLayout({imageEditor:content, objectControls:<div>Object settings</div>}) : content;
  });
});
const idle = { training: {status:'idle'}, batch: {status:'idle'}, models: [] };
const frames = ['first','middle','last'].map(frame_id => ({frame_id}));
function fixture(overrides = {}) {
  return { active:false, busy:false, episodes:[], selectedFrame:frames[1], selectFrame:jest.fn(),
    refresh:jest.fn().mockResolvedValue(), refreshEpisodes:jest.fn().mockResolvedValue(), annotate:jest.fn().mockResolvedValue(true),
    episode:{id:'ep', ended_at:1, map_name:'warehouse', observations:frames, annotations:[], regions:[], catalog:{missions:[]},
      qwen_sample:{schema_version:'user_count_v1',frame_ids:['first','last']},
      sam_sample:{schema_version:'user_count_v1',frame_ids:['first','last']}}, ...overrides };
}
function Workspace({draft, ...props}) {
  const editor = useCollectAnnotation({draft});
  const [maskDirty, setMaskDirty] = useState(false);
  return <CollectDatasetPanel {...{draft, editor, maskDirty, setMaskDirty}} {...props}/>;
}
beforeEach(() => collectionRequest.mockReset().mockResolvedValue(idle));

test.each(['mission', 'masks'])('Annotate keeps the %s editor visible with all workflow steps available', async tool => {
  const draft=fixture();
  render(<Workspace draft={draft} selection={{tool}}/>);
  expect(screen.getByRole('button',{name:tool==='mission'?'Reasoning labels':'Object masks (optional)'})).toHaveAttribute('aria-pressed','true');
  expect(screen.queryByRole('button',{name:'Sample Labeling'})).not.toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'Manual',exact:true})).not.toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'Auto Labeling',exact:true})).not.toBeInTheDocument();
  expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
  for (const name of ['Fine-tune', 'Auto-label', 'Review']) {
    expect(screen.getByRole('button',{name})).toBeVisible();
  }
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
  expect(screen.getByLabelText('Sample count')).toBeVisible();
  expect(screen.queryByLabelText('Model name')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Target')).not.toBeInTheDocument();
  expect(within(screen.getByRole('region',{name:'Frame preview'})).getByRole('button',{name:'Next frame'})).toBeVisible();
  if (tool==='mission') {
    expect(within(screen.getByRole('region',{name:'Frame labeling'})).getByLabelText('Reason')).toBeVisible();
  } else {
    expect(screen.getByLabelText('Mask editor')).toBeVisible();
    expect(screen.getByText('Object settings').closest('.annotate-frame-stage')).not.toBeNull();
    expect(screen.getByLabelText('SAM workflow').closest('.annotate-mask-workspace')).not.toBeNull();
  }
  expect(draft.selectFrame).not.toHaveBeenCalled();
  expect(screen.getByLabelText('Current frame')).toHaveTextContent('2 / 3');
  expect(screen.getByLabelText('Sample count')).toBeVisible();
  expect(screen.getByRole('button',{name:'Review'})).toBeDisabled();
  fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
  expect(screen.getByRole('region',{name:'Auto-label models'})).toBeVisible();
  expect(screen.getByLabelText('Auto-label target')).toHaveValue('episode');
  expect(screen.getByRole('button',{name:'Next frame'})).toBeVisible();
  await waitFor(() => expect(collectionRequest).toHaveBeenCalledWith(tool==='mission'?'qwen/workflow':'segmentation/workflow'));
});

test.each(['annotate', 'train'])('Learn keeps the labeling workspace for the legacy %s page', async workspacePage => {
  render(<Workspace draft={fixture()} workspacePage={workspacePage}
    selection={{tool:'mission',trainingPurpose:'reasoning',trainingTool:'masks'}}/>);
  expect(screen.getByRole('region',{name:'Dataset editor'})).toBeVisible();
  expect(screen.getByRole('button',{name:'Reasoning labels'})).toHaveAttribute('aria-pressed','true');
  expect(screen.getByLabelText('Reason')).toBeVisible();
  expect(screen.getByLabelText('Sample count')).toBeVisible();
  expect(screen.getByRole('button',{name:'Train model'})).toBeVisible();
  expect(screen.queryByLabelText('Training purpose')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Model',{exact:true})).not.toBeInTheDocument();
  expect(screen.queryByRole('dialog',{name:'Train reasoning model'})).not.toBeInTheDocument();
  await waitFor(() => expect(collectionRequest).toHaveBeenCalledWith('qwen/workflow'));
});

test('unsaved mission labels lock tools and frame filters while workflow settings remain inspectable', async () => {
  render(<Workspace draft={fixture()}/>);
  fireEvent.load(screen.getByAltText('Saved frame for reasoning annotation'));
  fireEvent.change(screen.getByLabelText('Reason'),{target:{value:'Unsaved scene evidence'}});
  expect(screen.getByRole('button',{name:'Object masks (optional)'})).toBeDisabled();
  expect(screen.getByLabelText('Sample count')).toBeDisabled();
  expect(screen.getByRole('button',{name:'Next frame'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Review'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Fine-tune'})).toBeEnabled();
  fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));
  expect(screen.getByRole('button',{name:'Start fine-tuning'})).toBeDisabled();
  fireEvent.click(screen.getByRole('button',{name:'Discard changes'}));
  expect(screen.getByRole('button',{name:'Object masks (optional)'})).toBeEnabled();
  expect(screen.getByLabelText('Sample count')).toBeEnabled();
  await waitFor(() => expect(collectionRequest).toHaveBeenCalled());
});

test('unsaved masks lock frame changes and keep read-only workflow settings available', async () => {
  render(<Workspace draft={fixture()}/>);
  fireEvent.click(screen.getByRole('button',{name:'Object masks (optional)'}));
  fireEvent.click(screen.getByRole('button',{name:'Draft mask'}));
  expect(screen.getByRole('button',{name:'Reasoning labels'})).toBeDisabled();
  expect(screen.getByLabelText('Sample count')).toBeDisabled();
  expect(screen.getByRole('button',{name:'Next frame'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Review'})).toBeDisabled();
  fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
  expect(screen.getByRole('button',{name:'Auto-label episode'})).toBeDisabled();
  fireEvent.click(screen.getByRole('button',{name:'Discard mask'}));
  expect(screen.getByRole('button',{name:'Reasoning labels'})).toBeEnabled();
  expect(screen.getByLabelText('Sample count')).toBeEnabled();
  fireEvent.click(screen.getByRole('button',{name:'Reasoning labels'}));
  expect(screen.getByLabelText('Mission')).toHaveValue('');
  await waitFor(() => expect(collectionRequest).toHaveBeenCalled());
});

test.each(['mission','masks'])('%s sampling locks only the request and returns navigation to the sample', async tool => {
  let finishSample;
  const draft=fixture(), onBusyChange=jest.fn();
  const route=tool==='mission'?'qwen':'segmentation';
  collectionRequest.mockImplementation(path => path.endsWith('/sample') ? new Promise(resolve => {finishSample=resolve;}) : Promise.resolve(idle));
  render(<Workspace draft={draft} selection={{tool}} onBusyChange={onBusyChange}/>);
  fireEvent.change(screen.getByLabelText('Sample count'),{target:{value:'2'}});
  fireEvent.click(screen.getByRole('button',{name:'Resample frames'}));
  expect(collectionRequest).toHaveBeenCalledWith(`episodes/ep/${route}/sample`,{count:2});
  expect(screen.getByRole('button',{name:tool==='mission'?'Object masks (optional)':'Reasoning labels'})).toBeDisabled();
  expect(screen.getByLabelText('Sample count')).toBeDisabled();
  expect(onBusyChange).toHaveBeenLastCalledWith(true);
  await act(async () => finishSample(draft.episode));
  expect(draft.refresh).toHaveBeenCalled();
  expect(draft.refreshEpisodes).toHaveBeenCalled();
  expect(draft.selectFrame).toHaveBeenCalledWith(frames[0]);
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
  expect(onBusyChange).toHaveBeenLastCalledWith(false);
  expect(screen.getByLabelText('Sample count')).toBeEnabled();
});

test('background reasoning training leaves labeling and the training drawer available', async () => {
  const onBusyChange=jest.fn();
  collectionRequest.mockResolvedValue({...idle,reasoning_training:{status:'training',step:10,steps:200}});
  render(<Workspace draft={fixture()} onBusyChange={onBusyChange}/>);
  await screen.findByRole('progressbar',{name:'Reasoning training progress'});
  expect(screen.getByRole('button',{name:'Train model'})).toBeEnabled();
  expect(screen.getByRole('button',{name:'Object masks (optional)'})).toBeEnabled();
  expect(screen.getByRole('button',{name:'Next frame'})).toBeEnabled();
  expect(onBusyChange).toHaveBeenLastCalledWith(false);
});

test('saving a training mask advances only through the sample', async () => {
  const draft=fixture({selectedFrame:frames[0]});
  render(<Workspace draft={draft}/>);
  fireEvent.click(screen.getByRole('button',{name:'Object masks (optional)'}));
  fireEvent.click(screen.getByRole('button',{name:'Show sampled frames'}));
  fireEvent.click(screen.getByRole('button',{name:'Save mask'}));
  await waitFor(() => expect(draft.selectFrame).toHaveBeenCalledWith(frames[2]));
  expect(draft.refresh.mock.invocationCallOrder[0]).toBeLessThan(draft.refreshEpisodes.mock.invocationCallOrder[0]);
});

test.each(['mission','masks'])('legacy automatic stage does not trap %s in a separate labeling mode', async tool => {
  const draft=fixture({selectedFrame:frames[0]});
  render(<Workspace draft={draft} selection={{tool,stages:{[tool]:'automatic'}}}/>);
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
  expect(screen.getByLabelText('Sample count')).toBeVisible();
  expect(screen.queryByLabelText('Training steps')).not.toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'Auto Labeling',exact:true})).not.toBeInTheDocument();
  await waitFor(() => expect(collectionRequest).toHaveBeenCalled());
});

test.each(['mission','masks'])('%s defaults to all frames when no sample is saved', async tool => {
  const episode={...fixture().episode,qwen_sample:null,sam_sample:null};
  render(<Workspace draft={fixture({episode})} selection={{tool}}/>);
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
  expect(screen.queryByText(/^Frame \d+ \/ \d+$/)).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Frame filter')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Saved frame')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('0/3 labeled');
  await waitFor(() => expect(collectionRequest).toHaveBeenCalled());
});

test('Learn restores running reasoning progress after returning from Operate', async () => {
  collectionRequest.mockResolvedValue({...idle,reasoning_training:{status:'training',model_name:'ongoing',step:37,steps:200}});
  const draft=fixture();
  function LearnAndOperate() {
    const [page,setPage]=useState('learn');
    return <>
      <button onClick={() => setPage(page==='learn'?'operate':'learn')}>Switch page</button>
      {page==='learn' && <Workspace draft={draft}/>}
    </>;
  }
  render(<LearnAndOperate/>);
  expect(await screen.findByRole('progressbar',{name:'Reasoning training progress'})).toHaveAttribute('aria-valuenow','19');
  fireEvent.click(screen.getByRole('button',{name:'Switch page'}));
  expect(screen.queryByRole('region',{name:'Dataset editor'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Switch page'}));
  expect(await screen.findByRole('progressbar',{name:'Reasoning training progress'})).toHaveAttribute('aria-valuenow','19');
  expect(screen.getByLabelText('Reason')).toBeVisible();
});

test('Learn uses the selected auto-label model while preserving the frame editor', async () => {
  const models=['older','newer'].map((id,i)=>({id,map_name:'warehouse',model_name:id,sample_count:2,created_at:i}));
  collectionRequest.mockResolvedValue({...idle,models});
  const draft=fixture({selectedFrame:frames[0]});
  render(<Workspace draft={draft} selection={{tool:'mission',autoLabelModels:{mission:'older'}}}/>);
  fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
  await waitFor(()=>expect(screen.getByRole('button',{name:'older',exact:true})).toHaveAttribute('aria-pressed','true'));
  expect(screen.getByLabelText('Reason')).toBeVisible();
  expect(screen.queryByLabelText('Training steps')).not.toBeInTheDocument();
});

test.each(['annotate','train'])('Qwen fine-tuning from the legacy %s page uses the current map', async workspacePage => {
  const episode={...fixture().episode,annotations:['first','last'].map(id=>({frame_ids:[id],review_status:'approved'}))};
  collectionRequest.mockImplementation(async path=>path==='qwen/auto-train'?{status:'queued'}:idle);
  render(<Workspace workspacePage={workspacePage} draft={fixture({episode,episodes:[episode]})}/>);
  await act(async()=>{});
  fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/auto-train',{map_name:'warehouse'}));
  expect(screen.queryByLabelText('Training steps')).not.toBeInTheDocument();
  expect(collectionRequest.mock.calls.some(([path])=>path.endsWith('/sample'))).toBe(false);
});

test.each(['annotate','train'])('SAM fine-tuning from the legacy %s page uses the current map', async workspacePage => {
  const episode={...fixture().episode,regions:['first','last'].map(frame_id=>({frame_id,object_label:'dish',review_status:'approved'}))};
  collectionRequest.mockImplementation(async path=>path==='segmentation/auto-train'?{status:'queued'}:idle);
  render(<Workspace workspacePage={workspacePage} draft={fixture({episode,episodes:[episode]})}/>);
  fireEvent.click(screen.getByRole('button',{name:'Object masks (optional)'}));
  await act(async()=>{});
  fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('segmentation/auto-train',{map_name:'warehouse'}));
  expect(screen.queryByLabelText('Model name')).not.toBeInTheDocument();
  expect(collectionRequest.mock.calls.some(([path])=>path.endsWith('/sample'))).toBe(false);
});


test('automatic mask settings survive switching the keyed frame editor', async () => {
  collectionRequest.mockResolvedValue({...idle,models:[{id:'sam',model_name:'sam_warehouse_1',map_name:'warehouse',strategy:'sam_multiclass_instances_v1'}]});
  const draft=fixture({selectedFrame:frames[0]});
  const view=render(<Workspace draft={draft}/>);
  fireEvent.click(screen.getByRole('button',{name:'Object masks (optional)'}));
  fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
  await waitFor(() => expect(collectionRequest).toHaveBeenCalledWith('segmentation/workflow'));
  fireEvent.click(await screen.findByRole('button',{name:'sam_warehouse_1',exact:true}));
  view.rerender(<Workspace draft={{...draft, selectedFrame:frames[2]}}/>);
  expect(screen.getByRole('button',{name:'sam_warehouse_1',exact:true})).toHaveAttribute('aria-pressed','true');
  expect(within(screen.getByRole('region',{name:'Frame preview'})).getByText('last')).toBeInTheDocument();
});


test('object masks restore a missing frame selection using the current SAM sample', async () => {
  const draft=fixture({selectedFrame:null});
  render(<Workspace draft={draft} selection={{tool:'masks',stages:{masks:'training'}}}/>);
  await waitFor(() => expect(draft.selectFrame).toHaveBeenCalledWith(frames[0]));
  expect(draft.selectFrame.mock.calls.every(([frame]) => frame !== null)).toBe(true);
});


test('SAM sampling in Learn refreshes episode and library data before finishing', async () => {
  const draft=fixture();
  collectionRequest.mockImplementation(async path=>path.endsWith('/segmentation/sample')?draft.episode:idle);
  render(<Workspace workspacePage="train" draft={draft} selection={{tool:'masks'}}/>);
  fireEvent.change(screen.getByLabelText('Sample count'),{target:{value:'2'}});
  fireEvent.click(screen.getByRole('button',{name:'Resample frames'}));
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('episodes/ep/segmentation/sample',{count:2}));
  await waitFor(()=>expect(draft.refreshEpisodes).toHaveBeenCalled());
  expect(draft.refresh).toHaveBeenCalled();
  expect(screen.getByLabelText('Mask editor')).toBeVisible();
});

test.each(['mission','masks'])('%s shows mode-specific progress beside frame navigation', async tool => {
  const episode={...fixture().episode,
    annotations:[{frame_ids:['first'],review_status:'approved',target:{decision:'wait',reason:'Sample'}},{frame_ids:['middle'],review_status:'draft',annotator:'qwen:auto',target:{decision:'wait',reason:'Auto'}}],
    regions:[{frame_id:'first',review_status:'approved'},{frame_id:'middle',review_status:'draft',annotator:'sam-auto'}]};
  render(<Workspace draft={fixture({episode,selectedFrame:frames[0]})} selection={{tool}}/>);
  expect(screen.queryByLabelText('Frame filter')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Saved frame')).not.toBeInTheDocument();
  const progress=screen.getByLabelText('Labeling progress');
  expect(progress).toHaveTextContent('1/3 labeled');
  expect(progress.parentElement).toBe(screen.getByRole('button',{name:'Next frame'}).parentElement);
  fireEvent.click(screen.getByRole('button',{name:'Review'}));
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('0/1 reviewed');
  await waitFor(() => expect(collectionRequest).toHaveBeenCalled());
});

test.each(['mission','masks'])('%s generation refreshes drafts and Review reaches frames outside the sample', async tool => {
  jest.useFakeTimers();
  let view;
  try {
    const route=tool==='mission'?'qwen':'segmentation';
    const model={id:'trained',map_name:'warehouse',model_name:'Trained labels',sample_count:2,
      strategy:'sam_multiclass_instances_v1'};
    const initial={...fixture().episode,
      annotations:['first','last'].map(id=>({frame_ids:[id],review_status:'approved',target:{decision:'wait',reason:'Reviewed sample'}})),
      regions:['first','last'].map(frame_id=>({frame_id,review_status:'approved'}))};
    const generated={...initial,
      annotations:[...initial.annotations,{frame_ids:['middle'],review_status:'draft',annotator:'qwen:auto',target:{decision:'wait',reason:'Generated scene'}}],
      regions:[...initial.regions,{frame_id:'middle',review_status:'draft',annotator:'sam-auto'}]};
    const completed={id:'generation',status:'completed',processed:1,total:1,created:1};
    let generatedRequested=false;
    collectionRequest.mockImplementation(async path=>{
      if(path===`${route}/auto-label`) {
        generatedRequested=true;
        return {id:'generation',status:'queued'};
      }
      return {...idle,models:[model],batch:generatedRequested?completed:idle.batch};
    });
    const refresh=jest.fn(), refreshEpisodes=jest.fn().mockResolvedValue();
    function GeneratedWorkspace() {
      const [episode,setEpisode]=useState(initial);
      const [selectedFrame,selectFrame]=useState(frames[0]);
      refresh.mockImplementation(async()=>{setEpisode(generated);return generated;});
      return <Workspace selection={{tool}} draft={fixture({episode,selectedFrame,selectFrame,refresh,refreshEpisodes})}/>;
    }
    await act(async()=>{view=render(<GeneratedWorkspace/>);});
    expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
    fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
    await act(async()=>{fireEvent.click(screen.getByRole('button',{name:'Auto-label episode'}));});
    expect(collectionRequest).toHaveBeenCalledWith(`${route}/auto-label`,expect.objectContaining({model_id:'trained',episode_ids:['ep']}));
    await act(async()=>{jest.advanceTimersByTime(3000);});
    expect(refresh).toHaveBeenCalled();
    expect(refreshEpisodes).toHaveBeenCalled();
    expect(screen.queryByText(/^Frame \d+ \/ \d+$/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'Review'}));
    expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','true');
    expect(screen.queryByText(/^Frame \d+ \/ \d+$/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Target')).not.toBeInTheDocument();
    if(tool==='mission') expect(screen.getByLabelText('Reason')).toHaveValue('Generated scene');
    else expect(within(screen.getByLabelText('Mask editor')).getByText('middle')).toBeVisible();
  } finally {
    view?.unmount();
    jest.useRealTimers();
  }
});

test.each(['mission', 'masks'])('Annotate opens without a selected episode for %s and can load then clear it', async tool => {
  const empty=fixture({episode:null,selectedFrame:null});
  const selection={tool};
  const view=render(<Workspace draft={empty} selection={selection}/>);
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
  expect(screen.getByLabelText('Sample count')).toBeDisabled();
  expect(screen.getByRole('button',{name:'Sample frames'})).toBeDisabled();
  view.rerender(<Workspace draft={fixture()} selection={selection}/>);
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
  view.rerender(<Workspace draft={empty} selection={selection}/>);
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalled());
});


test.each([false, true])('reasoning label progress updates after saving, with sampling=%s', async sampled => {
  const observations=Array.from({length:20},(_,index)=>({frame_id:`frame-${index}`}));
  const episode={...fixture().episode,observations,annotations:[],
    qwen_sample:sampled ? {schema_version:'user_count_v1',frame_ids:observations.map(f=>f.frame_id)} : undefined};
  const draft=fixture({episode,selectedFrame:observations[0]});
  const view=render(<Workspace draft={draft}/>);
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('0/20 labeled');
  view.rerender(<Workspace draft={{...draft,episode:{...episode,annotations:[
    {frame_ids:[observations[0].frame_id],review_status:'approved',target:{decision:'wait',reason:'People are eating.'}}
  ]}}}/>);
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('1/20 labeled');
  expect(screen.getByLabelText('Labeling progress').parentElement).toBe(screen.getByRole('button',{name:'Next frame'}).parentElement);
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalled());
});

test.each(['mission','masks'])('%s Review preserves sampling controls and permits return after the last review', async tool => {
  const episode={...fixture().episode,
    annotations:[{frame_ids:['middle'],review_status:'draft',annotator:'qwen:auto',target:{decision:'wait',reason:'Generated scene'}}],
    regions:[{frame_id:'middle',review_status:'draft',annotator:'sam-auto'}]};
  function ReviewWorkspace({currentEpisode}) {
    const [selectedFrame,selectFrame]=useState(frames[0]);
    return <Workspace selection={{tool}} draft={fixture({episode:currentEpisode,selectedFrame,selectFrame})}/>;
  }
  const view=render(<ReviewWorkspace currentEpisode={episode}/>);
  const review=screen.getByRole('button',{name:'Review'});
  const count=screen.getByLabelText('Sample count');
  fireEvent.change(count,{target:{value:'1'}});
  expect(screen.getByLabelText('Pending reviews')).toHaveTextContent('1');
  fireEvent.click(review);
  expect(review).toHaveAttribute('aria-pressed','true');
  expect(count).toBeVisible();
  expect(count).toBeDisabled();
  expect(screen.getByRole('button',{name:'Resample frames'})).toBeVisible();
  expect(screen.getByRole('button',{name:'Resample frames'})).toBeDisabled();
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('0/1 reviewed');
  const activeEditor=screen.getByLabelText(tool==='mission'?'Reason':'Mask editor');
  openReasoningTraining();
  fireEvent.keyDown(screen.getByRole('dialog',{name:'Train reasoning model'}),{key:'Escape'});
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.getByLabelText(tool==='mission'?'Reason':'Mask editor')).toBe(activeEditor);
  expect(review).toHaveAttribute('aria-pressed','true');
  expect(count).toHaveValue(1);
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('0/1 reviewed');
  fireEvent.click(review);
  expect(review).toHaveAttribute('aria-pressed','false');
  expect(count).toHaveValue(1);
  expect(count).toBeEnabled();
  expect(screen.getByRole('button',{name:'Resample frames'})).toBeEnabled();
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('0/3 labeled');

  fireEvent.click(review);
  const completed={...episode,
    annotations:episode.annotations.map(annotation=>({...annotation,review_status:'approved'})),
    regions:episode.regions.map(region=>({...region,review_status:'approved'}))};
  view.rerender(<ReviewWorkspace currentEpisode={completed}/>);
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('1/1 reviewed');
  expect(review).toHaveAttribute('aria-pressed','true');
  expect(review).toBeEnabled();
  expect(count).toBeDisabled();
  fireEvent.click(review);
  expect(review).toHaveAttribute('aria-pressed','false');
  expect(review).toBeDisabled();
  expect(count).toHaveValue(1);
  expect(count).toBeEnabled();
  expect(screen.getByRole('button',{name:'Resample frames'})).toBeEnabled();
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('1/3 labeled');
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalled());
});

function preparedReasoningEpisodes() {
  const observation={...fixture().episode,episode_type:'observation',
    qwen_sample:{schema_version:'user_count_v1',frame_ids:['first']},
    annotations:[
      {frame_ids:['first'],review_status:'approved',target:{decision:'wait',reason:'Reviewed sample'}},
      {frame_ids:['middle'],review_status:'approved',target:{decision:'wait',reason:'Reviewed outside sample'}},
      {frame_ids:['last'],review_status:'draft',target:{decision:'wait',reason:'Needs review'}}]};
  const patrol={...observation,id:'patrol',episode_type:'patrol',reasoning_approved_count:3};
  const mission={...observation,id:'mission',episode_type:'mission',reasoning_approved_count:4};
  const foreign={...observation,id:'foreign',map_name:'another-map',reasoning_approved_count:5};
  const active={...observation,id:'active',ended_at:null,reasoning_approved_count:10};
  const unreviewed={...observation,id:'unreviewed',reasoning_approved_count:0};
  return {observation,patrol,mission,foreign,active,unreviewed,
    episodes:[observation,patrol,mission,foreign,active,unreviewed]};
}
function openReasoningTraining() {
  fireEvent.click(screen.getByRole('button',{name:'Train model'}));
  return within(screen.getByRole('dialog',{name:'Train reasoning model'}));
}

test('Train model opens default reasoning settings with advanced controls collapsed', async () => {
  const {observation,episodes}=preparedReasoningEpisodes();
  render(<Workspace draft={fixture({episode:observation,episodes,selectedFrame:frames[0]})}/>);
  const reason=screen.getByLabelText('Reason');
  const drawer=openReasoningTraining();
  expect(drawer.getByLabelText('Training map')).toHaveValue('warehouse');
  expect(drawer.getByLabelText('Model name')).toBeVisible();
  expect(drawer.getByLabelText('Training steps')).toHaveValue(2000);
  expect(drawer.getByRole('checkbox',{name:'Balance decisions per type'})).not.toBeChecked();
  expect(drawer.getByText('9 approved labels · 3 saved episodes')).toBeVisible();
  const advanced=drawer.getByText('Advanced settings').closest('details');
  expect(advanced).not.toHaveAttribute('open');
  expect(drawer.getByLabelText('Base model')).not.toBeVisible();
  fireEvent.click(drawer.getByText('Advanced settings'));
  expect(advanced).toHaveAttribute('open');
  expect(drawer.getByLabelText('Base model')).toHaveValue('Qwen/Qwen3.5-4B');
  for (const type of ['Observation 2 labels','Patrol 3 labels','Mission 4 labels']) {
    expect(drawer.getByRole('checkbox',{name:type})).toBeChecked();
  }
  expect(screen.getByLabelText('Reason')).toBe(reason);
  fireEvent.click(drawer.getByRole('button',{name:'Close training settings'}));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Reason')).toBe(reason);
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/workflow'));
});

test.each(['mission','masks'])('training settings preserve the mounted %s editor and unsaved work', async tool => {
  render(<Workspace draft={fixture({selectedFrame:frames[0]})} selection={{tool}}/>);
  fireEvent.change(screen.getByLabelText('Sample count'),{target:{value:'1'}});
  let editor;
  if(tool==='mission') {
    editor=screen.getByLabelText('Reason');
    fireEvent.load(screen.getByAltText('Saved frame for reasoning annotation'));
    fireEvent.change(editor,{target:{value:'Keep this unsaved reasoning'}});
  } else {
    editor=screen.getByLabelText('Mask editor');
    fireEvent.click(screen.getByRole('button',{name:'Draft mask'}));
  }
  const drawer=openReasoningTraining();
  expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeDisabled();
  expect(drawer.getByLabelText('Training map')).toBeDisabled();
  fireEvent.click(drawer.getByRole('button',{name:'Close training settings'}));
  expect(screen.getByLabelText(tool==='mission'?'Reason':'Mask editor')).toBe(editor);
  if(tool==='mission') expect(editor).toHaveValue('Keep this unsaved reasoning');
  expect(screen.getByLabelText('Sample count')).toHaveValue(1);
  expect(screen.getByRole('button',{name:'Next frame'})).toBeDisabled();
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/workflow'));
});

test('final reasoning training submits all approved types on the current map beyond the labeling sample', async () => {
  const {observation,episodes}=preparedReasoningEpisodes();
  collectionRequest.mockImplementation(async path=>path==='qwen/train-reasoning'?{status:'queued',steps:350,step:0}:idle);
  render(<Workspace draft={fixture({episode:observation,episodes,selectedFrame:frames[0]})}/>);
  const drawer=openReasoningTraining();
  expect(drawer.getByText('9 approved labels · 3 saved episodes')).toBeInTheDocument();
  fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'  Warehouse reasoning  '}});
  fireEvent.change(drawer.getByLabelText('Training steps'),{target:{value:'350'}});
  await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
  fireEvent.click(drawer.getByRole('button',{name:'Train reasoning model'}));
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/train-reasoning',{
    map_name:'warehouse',episode_ids:['ep','patrol','mission'],model_name:'Warehouse reasoning',
    base_model:'Qwen/Qwen3.5-4B',steps:350,batch_size:8}));
  expect(collectionRequest.mock.calls.some(([path])=>path.endsWith('/sample'))).toBe(false);
  expect(collectionRequest.mock.calls.some(([path])=>path==='qwen/train'||path==='qwen/auto-train'||path.startsWith('segmentation/train'))).toBe(false);
});

test('training map changes limit reasoning training to that map and survive closing settings', async () => {
  const {observation,episodes}=preparedReasoningEpisodes();
  collectionRequest.mockImplementation(async path=>path==='qwen/train-reasoning'?{status:'queued'}:idle);
  render(<Workspace draft={fixture({episode:observation,episodes})}/>);
  let drawer=openReasoningTraining();
  fireEvent.change(drawer.getByLabelText('Training map'),{target:{value:'another-map'}});
  expect(drawer.getByText('5 approved labels · 1 saved episodes')).toBeInTheDocument();
  fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'Other map reasoning'}});
  fireEvent.click(drawer.getByRole('button',{name:'Close training settings'}));
  drawer=openReasoningTraining();
  expect(drawer.getByLabelText('Training map')).toHaveValue('another-map');
  expect(drawer.getByLabelText('Model name')).toHaveValue('Other map reasoning');
  await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
  fireEvent.click(drawer.getByRole('button',{name:'Train reasoning model'}));
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/train-reasoning',{
    map_name:'another-map',episode_ids:['foreign'],model_name:'Other map reasoning',
    base_model:'Qwen/Qwen3.5-4B',steps:2000,batch_size:8}));
});

test.each([null,{...fixture().episode,map_name:'not-saved',ended_at:null}])('training defaults to a saved map without a saved selected episode: %s', async episode => {
  const saved={...fixture().episode,id:'saved',map_name:'alpha',reasoning_approved_count:3};
  render(<Workspace draft={fixture({episode,selectedFrame:null,episodes:[
    {...saved,id:'later',map_name:'zeta'},saved,
    {...saved,id:'active',map_name:'a-active',ended_at:null}]})}/>);
  const drawer=openReasoningTraining();
  expect(drawer.getByLabelText('Training map')).toHaveValue('alpha');
  expect(drawer.getByText('3 approved labels · 1 saved episodes')).toBeInTheDocument();
  expect(drawer.queryByRole('option',{name:'a-active'})).not.toBeInTheDocument();
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/workflow'));
});

test('training includes a saved selected episode missing from the episode library', async () => {
  const {observation,patrol}=preparedReasoningEpisodes();
  render(<Workspace draft={fixture({episode:observation,episodes:[patrol]})}/>);
  const drawer=openReasoningTraining();
  expect(drawer.getByText('5 approved labels · 2 saved episodes')).toBeInTheDocument();
  fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'Freshly saved'}});
  await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
  fireEvent.click(drawer.getByRole('button',{name:'Train reasoning model'}));
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/train-reasoning',expect.objectContaining({
    episode_ids:expect.arrayContaining(['ep','patrol'])})));
  const [,payload]=collectionRequest.mock.calls.find(([path])=>path==='qwen/train-reasoning');
  expect(payload.episode_ids).toHaveLength(2);
});

test('advanced settings scope reasoning types and the base model in the submitted job', async () => {
  const {observation,episodes}=preparedReasoningEpisodes();
  render(<Workspace draft={fixture({episode:observation,episodes})}/>);
  const drawer=openReasoningTraining();
  fireEvent.click(drawer.getByText('Advanced settings'));
  fireEvent.click(drawer.getByRole('checkbox',{name:'Observation 2 labels'}));
  expect(drawer.getByText('7 approved labels · 2 saved episodes')).toBeInTheDocument();
  fireEvent.change(drawer.getByLabelText('Base model'),{target:{value:'Qwen/Qwen3.5-0.8B'}});
  fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'Supervision only'}});
  await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
  fireEvent.click(drawer.getByRole('button',{name:'Train reasoning model'}));
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/train-reasoning',{
    map_name:'warehouse',episode_ids:['patrol','mission'],model_name:'Supervision only',
    base_model:'Qwen/Qwen3.5-0.8B',steps:2000,batch_size:8}));
});

test('balanced reasoning training previews the same scope and submits the balance option', async () => {
  const {observation,episodes}=preparedReasoningEpisodes();
  collectionRequest.mockImplementation(async path=>path==='qwen/preview-reasoning'?{
    before:{decisions:{start:2,wait:7}},after:{total:4,decisions:{start:2,wait:2}}
  }:path==='qwen/train-reasoning'?{status:'queued'}:idle);
  render(<Workspace draft={fixture({episode:observation,episodes})}/>);
  const drawer=openReasoningTraining();
  fireEvent.click(drawer.getByRole('checkbox',{name:'Balance decisions per type'}));
  await drawer.findByText('Training: 2 start · 2 wait');
  expect(collectionRequest).toHaveBeenCalledWith('qwen/preview-reasoning',{
    map_name:'warehouse',episode_ids:['ep','patrol','mission'],balance_decisions:true});
  fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'Balanced reasoning'}});
  await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
  fireEvent.click(drawer.getByRole('button',{name:'Train reasoning model'}));
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/train-reasoning',{
    map_name:'warehouse',episode_ids:['ep','patrol','mission'],model_name:'Balanced reasoning',
    base_model:'Qwen/Qwen3.5-4B',steps:2000,batch_size:8,balance_decisions:true}));
});

test('only completed final reasoning models on the training map can be tested in Observation', async () => {
  const {observation,episodes}=preparedReasoningEpisodes();
  const reasoning={id:'reasoning',map_name:'warehouse',model_name:'Warehouse reasoning',sample_count:9,created_at:2,purpose:'reasoning',status:'completed'};
  const annotation={...reasoning,id:'auto',model_name:'Draft generator',purpose:'auto_labeling'};
  const pending={...reasoning,id:'pending',model_name:'Pending reasoning',status:'training'};
  const foreign={...reasoning,id:'foreign',map_name:'another-map',model_name:'Foreign reasoning'};
  collectionRequest.mockResolvedValue({...idle,models:[annotation,reasoning,pending,foreign]});
  const onTestModel=jest.fn();
  render(<Workspace draft={fixture({episode:observation,episodes})} onTestModel={onTestModel}/>);
  const drawer=openReasoningTraining();
  await drawer.findByRole('option',{name:'Warehouse reasoning · 9 labels'});
  expect(drawer.queryByRole('option',{name:/Draft generator|Pending reasoning|Foreign reasoning/})).not.toBeInTheDocument();
  fireEvent.click(drawer.getByRole('button',{name:'Test in Observation'}));
  expect(onTestModel).toHaveBeenCalledWith(reasoning);
  expect(collectionRequest.mock.calls.some(([path])=>path==='qwen/train-reasoning')).toBe(false);
});

test('closing the drawer during submission keeps the request and background polling alive', async () => {
  jest.useFakeTimers();
  let view;
  try {
    const {observation,episodes}=preparedReasoningEpisodes();
    const queued={id:'reasoning-job',map_name:'warehouse',model_name:'Background reasoning',status:'queued',steps:200,step:0};
    let workflow=idle, finishTraining;
    collectionRequest.mockImplementation(path=>path==='qwen/train-reasoning'
      ? new Promise(resolve=>{finishTraining=resolve;}) : Promise.resolve(workflow));
    await act(async()=>{view=render(<Workspace draft={fixture({episode:observation,episodes})} onTestModel={jest.fn()}/>);});
    let drawer=openReasoningTraining();
    fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'Background reasoning'}});
    await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
    fireEvent.click(drawer.getByRole('button',{name:'Train reasoning model'}));
    fireEvent.click(drawer.getByRole('button',{name:'Close training settings'}));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    await act(async()=>{finishTraining(queued);});
    expect(screen.getByRole('progressbar',{name:'Reasoning training progress'})).toHaveAttribute('aria-valuenow','0');
    const reads=collectionRequest.mock.calls.filter(([path])=>path==='qwen/workflow').length;
    workflow={...idle,reasoning_training:{...queued,status:'training',step:37}};
    await act(async()=>{jest.advanceTimersByTime(3000);});
    expect(collectionRequest.mock.calls.filter(([path])=>path==='qwen/workflow').length).toBeGreaterThan(reads);
    expect(screen.getByRole('progressbar',{name:'Reasoning training progress'})).toHaveAttribute('aria-valuenow','19');
    drawer=openReasoningTraining();
    expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeDisabled();
    fireEvent.click(drawer.getByRole('button',{name:'Close training settings'}));
    const completedModel={id:'finished-reasoning',map_name:'warehouse',model_name:'Background reasoning',purpose:'reasoning',sample_count:9,status:'completed',created_at:10};
    workflow={...idle,reasoning_training:{...queued,status:'completed',step:200},models:[completedModel]};
    await act(async()=>{jest.advanceTimersByTime(3000);});
    drawer=openReasoningTraining();
    expect(drawer.getByRole('option',{name:'Background reasoning · 9 labels'})).toBeInTheDocument();
    expect(drawer.getByRole('button',{name:'Test in Observation'})).toBeEnabled();
    expect(collectionRequest.mock.calls.filter(([path])=>path==='qwen/train-reasoning')).toHaveLength(1);
    expect(collectionRequest.mock.calls.some(([path])=>/cancel|stop/.test(path))).toBe(false);
  } finally {
    view?.unmount();
    jest.useRealTimers();
  }
});


test('training counts use fresh selected episode labels instead of stale library totals', async () => {
  const {observation,patrol}=preparedReasoningEpisodes();
  const episodes=[{...observation,reasoning_approved_count:99},patrol];
  const draft=fixture({episode:observation,episodes,selectedFrame:frames[0]});
  const view=render(<Workspace draft={draft}/>);
  const drawer=openReasoningTraining();
  expect(drawer.getByText('5 approved labels · 2 saved episodes')).toBeInTheDocument();
  view.rerender(<Workspace draft={{...draft,episode:{...observation,
    annotations:observation.annotations.map(label=>({...label,review_status:'approved'}))}}}/>);
  expect(drawer.getByText('6 approved labels · 2 saved episodes')).toBeInTheDocument();
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/preview-reasoning',{
    map_name:'warehouse',episode_ids:['ep','patrol'],balance_decisions:false}));
});


test.each(['mission', 'masks'])('Train %s range deletion includes unsampled frames', async tool => {
  const draft=fixture({deleteFrames:jest.fn().mockResolvedValue(true)});
  render(<Workspace draft={draft} selection={{tool}}/>);
  const remove=screen.getByRole('button',{name:'Delete frame range'});
  await waitFor(()=>expect(remove).toBeEnabled());
  fireEvent.click(remove);
  const dialog=within(screen.getByRole('dialog',{name:'Delete frame range'}));
  expect(dialog.getByText('Episode · 3 frames')).toBeVisible();
  fireEvent.change(dialog.getByLabelText('From'),{target:{value:'2'}});
  fireEvent.change(dialog.getByLabelText('To'),{target:{value:'3'}});
  fireEvent.click(dialog.getByRole('button',{name:'Delete 2 frames'}));
  fireEvent.click(dialog.getByRole('button',{name:'Delete permanently'}));
  await waitFor(()=>expect(draft.deleteFrames).toHaveBeenCalledWith(['middle','last']));
  await waitFor(()=>expect(screen.queryByRole('dialog',{name:'Delete frame range'})).not.toBeInTheDocument());
});

test('range deletion is unavailable during final model training', async()=>{
  collectionRequest.mockResolvedValue({...idle,reasoning_training:{status:'running',episode_ids:['ep']}});
  render(<Workspace draft={fixture({deleteFrames:jest.fn()})} selection={{tool:'masks'}}/>);
  await waitFor(()=>expect(screen.getByRole('button',{name:'Delete frame range'})).toBeDisabled());
});


test('final reasoning training can be cancelled from its drawer and trained again',async()=>{
 const {observation,episodes}=preparedReasoningEpisodes();
 const job={id:'c'.repeat(32),model_name:'House reasoning',status:'training',steps:200,step:20};
 let workflow={...idle,reasoning_training:job};
 collectionRequest.mockImplementation(path=>{
   if(path==='qwen/jobs/reasoning_training/cancel')workflow={...idle,reasoning_training:{...job,status:'cancelled'}};
   return Promise.resolve(workflow);
 });
 render(<Workspace draft={fixture({episode:observation,episodes})}/>);
 const drawer=openReasoningTraining();
 const cancel=await drawer.findByRole('button',{name:'Cancel training'});
 expect(cancel).toBeEnabled();fireEvent.click(cancel);
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/jobs/reasoning_training/cancel',{job_id:job.id}));
 await waitFor(()=>expect(drawer.getByRole('status')).toHaveTextContent('cancelled'));
 expect(drawer.queryByRole('button',{name:'Cancel training'})).not.toBeInTheDocument();
 fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'Retry reasoning'}});
 expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled();
});


test.each(['mission','masks'])('model actions remain accessible without an episode for %s',async tool=>{
 const saved={...fixture().episode,id:'saved',qwen_sample_approved_count:2,sam_sample_trainable_instance_count:2,reasoning_approved_count:3};
 const draft=fixture({episode:null,selectedFrame:null,episodes:[saved]});
 render(<Workspace draft={draft} selection={{tool}}/>);
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith(tool==='mission'?'qwen/workflow':'segmentation/workflow'));
 for(const name of ['Fine-tune','Auto-label','Train model'])expect(screen.getByRole('button',{name})).toBeEnabled();
 fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));
 expect(screen.getByLabelText('Fine-tuning map')).toHaveValue('warehouse');
 expect(screen.getByRole('button',{name:'Start fine-tuning'})).toBeEnabled();
 fireEvent.click(screen.getByRole('button',{name:'Close training details'}));
 fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
 expect(screen.getByLabelText('Auto-label map')).toHaveValue('warehouse');
 expect(screen.getByLabelText('Auto-label target')).toHaveValue('map');
 fireEvent.click(screen.getByRole('button',{name:'Close model picker'}));
 const drawer=openReasoningTraining();
 fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'From all reviewed data'}});
 await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
 fireEvent.click(drawer.getByRole('button',{name:'Train reasoning model'}));
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/train-reasoning',{
   map_name:'warehouse',episode_ids:['saved'],model_name:'From all reviewed data',base_model:'Qwen/Qwen3.5-4B',steps:2000,batch_size:8}));
 expect(draft.selectFrame).not.toHaveBeenCalled();
});

test('an empty library still opens every model workflow without starting invalid jobs',async()=>{
 render(<Workspace draft={fixture({episode:null,selectedFrame:null,episodes:[]})}/>);
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/workflow'));
 fireEvent.click(screen.getByRole('button',{name:'Fine-tune'}));
 expect(screen.getByRole('button',{name:'Start fine-tuning'})).toBeDisabled();
 expect(screen.getByLabelText('Fine-tuning map')).toHaveValue('');
 fireEvent.click(screen.getByRole('button',{name:'Close training details'}));
 fireEvent.click(screen.getByRole('button',{name:'Auto-label'}));
 expect(screen.getByRole('button',{name:'Auto-label map'})).toBeDisabled();
 fireEvent.click(screen.getByRole('button',{name:'Close model picker'}));
 expect(openReasoningTraining().getByRole('button',{name:'Train reasoning model'})).toBeDisabled();
 expect(collectionRequest.mock.calls.every(([path])=>path==='qwen/workflow')).toBe(true);
});


test.each(['mission','masks'].flatMap(tool=>[false,true].map(selected=>[tool,selected])))('Review opens generated %s drafts with a reviewed episode selected=%s', async (tool,selected)=>{
  const completed={...fixture().episode,id:'done',
    annotations:frames.map(frame=>({frame_ids:[frame.frame_id],review_status:'approved',target:{decision:'wait',mission_id:null,reason:'Reviewed scene'}}))};
  const generated={...fixture().episode,id:'generated',
    qwen_pending_review_count:2,sam_pending_review_count:2,
    annotations:['middle','last'].map(id=>({frame_ids:[id],review_status:'draft',annotator:'qwen:auto',target:{decision:'wait',reason:'Generated scene'}})),
    regions:['middle','last'].map(frame_id=>({object_id:frame_id,frame_id,review_status:'draft',annotator:'sam-auto'}))};
  const load=jest.fn();
  function ReviewWorkspace(){
    const [episode,setEpisode]=useState(selected?completed:null),[selectedFrame,selectFrame]=useState(null);
    load.mockImplementation(async id=>{setEpisode(generated);selectFrame(null);return true;});
    return <Workspace selection={{tool}} draft={fixture({episode,selectedFrame,selectFrame,load,episodes:[completed,generated]})}/>;
  }
  render(<ReviewWorkspace/>);
  await waitFor(()=>expect(screen.getByRole('button',{name:'Review'})).toBeEnabled());
  fireEvent.click(screen.getByRole('button',{name:'Review'}));
  await waitFor(()=>expect(load).toHaveBeenCalledWith('generated'));
  await waitFor(()=>expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','true'));
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('0/2 reviewed');
  await waitFor(()=>expect(screen.getByLabelText('Current frame')).toHaveTextContent('1 / 2'));
  expect(screen.getByLabelText('Pending reviews')).toHaveTextContent('2');
});

test('Review stays within the current dataset map when its episode is already reviewed',async()=>{
  const episode={...fixture().episode,annotations:frames.map(frame=>({frame_ids:[frame.frame_id],review_status:'approved',target:{decision:'wait',mission_id:null,reason:'Reviewed scene'}}))};
  render(<Workspace draft={fixture({episode,episodes:[episode,{id:'foreign',map_name:'other',ended_at:1,qwen_pending_review_count:12}]})}/>);
  expect(screen.getByRole('button',{name:'Review'})).toBeDisabled();
  await act(async()=>{});
});


test.each(['mission','masks'])('map %s review continues through episodes and ends only after all drafts are saved',async tool=>{
  const make=(id,map_name='warehouse')=>({...fixture().episode,id,map_name,
    observations:[{frame_id:id}],qwen_sample:null,sam_sample:null,
    qwen_pending_review_count:1,sam_pending_review_count:1,
    annotations:[{frame_ids:[id],revision:1,review_status:'draft',annotator:'qwen:auto',target:{decision:'wait',mission_id:null,reason:'Generated scene'}}],
    regions:[{object_id:id,frame_id:id,review_status:'draft',annotator:'sam-auto'}]});
  const initial=[make('first-episode'),make('second-episode'),make('foreign','another-map')];
  const load=jest.fn(),annotate=jest.fn();
  function MapReview(){
    const [episodes,setEpisodes]=useState(initial),[id,setId]=useState('first-episode'),[selectedFrame,selectFrame]=useState(null);
    const episode=episodes.find(ep=>ep.id===id);
    function approve(){
      const updated={...episode,qwen_pending_review_count:0,sam_pending_review_count:0,
        annotations:episode.annotations.map(label=>({...label,review_status:'approved',revision:2})),
        regions:episode.regions.map(region=>({...region,review_status:'approved'}))};
      setEpisodes(previous=>previous.map(ep=>ep.id===id?updated:ep));return updated;
    }
    load.mockImplementation(async next=>{setId(next);selectFrame(null);return true;});
    annotate.mockImplementation(async()=>{approve();return true;});
    return <Workspace selection={{tool}} draft={fixture({episode,episodes,selectedFrame,selectFrame,load,annotate,
      refresh:async()=>tool==='masks'?approve():episode})}/>;
  }
  render(<MapReview/>);
  await waitFor(()=>expect(screen.getByRole('button',{name:'Review'})).toBeEnabled());
  fireEvent.click(screen.getByRole('button',{name:'Review'}));
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('0/2 reviewed');
  async function save(){
    if(tool==='mission'){
      fireEvent.load(screen.getByAltText('Saved frame for reasoning annotation'));
      await waitFor(()=>expect(screen.getByRole('button',{name:'Save'})).toBeEnabled());
      fireEvent.click(screen.getByRole('button',{name:'Save'}));
    } else fireEvent.click(screen.getByRole('button',{name:'Save mask'}));
  }
  await save();
  await waitFor(()=>expect(load).toHaveBeenCalledWith('second-episode'));
  await waitFor(()=>expect(screen.getByLabelText('Current frame')).toHaveTextContent('1 / 1'));
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','true');
  expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('1/2 reviewed');
  await save();
  await waitFor(()=>expect(screen.getByLabelText('Labeling progress')).toHaveTextContent('2/2 reviewed'));
  expect(screen.getByLabelText('Pending reviews')).toHaveTextContent('0');
  expect(load).toHaveBeenCalledTimes(1);
  expect(load).not.toHaveBeenCalledWith('foreign');
  fireEvent.click(screen.getByRole('button',{name:'Review'}));
  expect(screen.getByRole('button',{name:'Review'})).toHaveAttribute('aria-pressed','false');
});


test.each([1, 100, 5000])('reasoning training submits %i steps without a fixed range', async steps => {
  const {observation,episodes}=preparedReasoningEpisodes();
  collectionRequest.mockImplementation(async path=>path==='qwen/train-reasoning'?{status:'queued'}:idle);
  render(<Workspace draft={fixture({episode:observation,episodes})}/>);
  const drawer=openReasoningTraining();
  fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'Free step count'}});
  const input=drawer.getByLabelText('Training steps');
  expect(input).not.toHaveAttribute('max');
  fireEvent.change(input,{target:{value:'0'}});
  await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeDisabled());
  fireEvent.change(input,{target:{value:String(steps)}});
  await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
  fireEvent.click(drawer.getByRole('button',{name:'Train reasoning model'}));
  await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/train-reasoning',expect.objectContaining({steps})));
});


test.each(['cancelled','completed','failed'])('training settings hide previous %s job and show only the new submission',async status=>{
 const {observation,episodes}=preparedReasoningEpisodes();
 const old={id:'old-job',status,model_name:'house_first_test',steps:2000,step:0,phase:'Old log'};
 let resolve;
 collectionRequest.mockImplementation(path=>path==='qwen/train-reasoning'
   ?new Promise(done=>{resolve=done;}) : Promise.resolve({...idle,reasoning_training:old}));
 render(<Workspace draft={fixture({episode:observation,episodes})}/>);
 const drawer=openReasoningTraining();
 fireEvent.change(drawer.getByLabelText('Model name'),{target:{value:'New training'}});
 await waitFor(()=>expect(drawer.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
 expect(drawer.queryByText('house_first_test')).not.toBeInTheDocument();
 fireEvent.click(drawer.getByRole('button',{name:'Train reasoning model'}));
 expect(drawer.getByRole('status')).toHaveTextContent('Preparing training data');
 expect(drawer.queryByText('house_first_test')).not.toBeInTheDocument();
 await act(async()=>resolve({id:'new-job',status:'queued',model_name:'New training',steps:200,step:0}));
 expect(drawer.getByRole('status')).toHaveTextContent('queued');
 expect(drawer.queryByText('house_first_test')).not.toBeInTheDocument();
});


test('saved reasoning models can be deleted after confirming the selected model',async()=>{
 const {observation,episodes}=preparedReasoningEpisodes();
 const model={id:'saved-model',map_name:'warehouse',model_name:'My trained model',sample_count:9,purpose:'reasoning'};
 let models=[model];
 collectionRequest.mockImplementation(async path=>{
   if(path==='qwen/models/saved-model'){models=[];return {deleted:model.id};}
   return {...idle,models};
 });
 render(<Workspace draft={fixture({episode:observation,episodes})}/>);
 const drawer=openReasoningTraining();
 const trash=await drawer.findByRole('button',{name:'Delete reasoning model'});
 expect(trash).toBeEnabled();fireEvent.click(trash);
 let confirm=drawer.getByRole('group',{name:'Delete reasoning model confirmation'});
 fireEvent.click(within(confirm).getByRole('button',{name:'Cancel'}));
 expect(collectionRequest.mock.calls.some(([, ,method])=>method==='DELETE')).toBe(false);
 fireEvent.click(trash);
 confirm=drawer.getByRole('group',{name:'Delete reasoning model confirmation'});
 fireEvent.click(within(confirm).getByRole('button',{name:'Delete'}));
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith('qwen/models/saved-model',undefined,'DELETE'));
 await waitFor(()=>expect(drawer.queryByRole('option',{name:'My trained model · 9 labels'})).not.toBeInTheDocument());
 expect(drawer.queryByRole('group',{name:'Delete reasoning model confirmation'})).not.toBeInTheDocument();
});

test('a loaded reasoning model deletion error keeps the model available',async()=>{
 const {observation,episodes}=preparedReasoningEpisodes();
 const model={id:'loaded-model',map_name:'warehouse',model_name:'Loaded model',sample_count:9,purpose:'reasoning'};
 collectionRequest.mockImplementation(async path=>{
   if(path==='qwen/models/loaded-model')throw new Error('Stop observation and unload this model before deleting it');
   return {...idle,models:[model]};
 });
 render(<Workspace draft={fixture({episode:observation,episodes})}/>);
 const drawer=openReasoningTraining();fireEvent.click(await drawer.findByRole('button',{name:'Delete reasoning model'}));
 fireEvent.click(within(drawer.getByRole('group',{name:'Delete reasoning model confirmation'})).getByRole('button',{name:'Delete'}));
 expect(await drawer.findByRole('alert')).toHaveTextContent('Stop observation and unload');
 expect(drawer.getByRole('option',{name:'Loaded model · 9 labels'})).toBeInTheDocument();
});


test.each(['queued','training'])('model training can be cancelled from the toolbar while %s without opening the drawer',async status=>{
 const {observation,episodes}=preparedReasoningEpisodes();
 const job={id:'active-training',status,model_name:'Current model',steps:4000,step:10};
 let workflow={...idle,reasoning_training:job},resolve;
 collectionRequest.mockImplementation(path=>{
   if(path==='qwen/jobs/reasoning_training/cancel')return new Promise(done=>{resolve=()=>{workflow={...idle,reasoning_training:{...job,status:'cancelled'}};done(workflow);};});
   return Promise.resolve(workflow);
 });
 render(<Workspace draft={fixture({episode:observation,episodes})}/>);
 const cancel=await screen.findByRole('button',{name:'Cancel model training'});
 expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
 fireEvent.click(cancel);
 expect(cancel).toBeDisabled();
 expect(collectionRequest).toHaveBeenCalledWith('qwen/jobs/reasoning_training/cancel',{job_id:job.id});
 await act(async()=>resolve());
 expect(screen.queryByRole('button',{name:'Cancel model training'})).not.toBeInTheDocument();
 expect(screen.getByRole('button',{name:'Train model'})).toBeEnabled();
});

test.each([
 ['reasoning_training','mission'], ['training','mission'], ['training','masks'],
])('%s permits episode/frame label inspection in %s',async(kind,tool)=>{
 const state={...idle,[kind]:{id:'active-training',status:'training',step:20,steps:4000}};
 collectionRequest.mockResolvedValue(state);
 const draft=fixture({selectedFrame:frames[1]}),onBusyChange=jest.fn();
 render(<Workspace draft={draft} selection={{tool}} onBusyChange={onBusyChange}/>);
 await waitFor(()=>expect(collectionRequest).toHaveBeenCalledWith(tool==='mission'?'qwen/workflow':'segmentation/workflow'));
 await act(async()=>Promise.resolve());
 expect(screen.getByRole('button',{name:'Previous frame'})).toBeEnabled();
 expect(screen.getByRole('button',{name:'Next frame'})).toBeEnabled();
 expect(screen.getByRole('button',{name:tool==='mission'?'Object masks (optional)':'Reasoning labels'})).toBeEnabled();
 expect(screen.getByRole('button',{name:'Show sampled frames'})).toBeEnabled();
 fireEvent.click(screen.getByRole('button',{name:'Next frame'}));
 expect(draft.selectFrame).toHaveBeenCalledWith(frames[2]);
 expect(onBusyChange).toHaveBeenLastCalledWith(false);
});

test('background training start request does not lock browsing before the response',async()=>{
 let resolveStart;
 collectionRequest.mockImplementation((path)=>path==='qwen/train-reasoning'?new Promise(resolve=>{resolveStart=resolve;}):Promise.resolve(idle));
 const ep={...fixture().episode,reasoning_approved_count:3};
 const draft=fixture({episode:ep,selectedFrame:frames[1],episodes:[ep]}),onBusyChange=jest.fn();
 render(<Workspace draft={draft} onBusyChange={onBusyChange}/>);
 openReasoningTraining();
 fireEvent.change(screen.getByLabelText('Model name'),{target:{value:'New model'}});
 await waitFor(()=>expect(screen.getByRole('button',{name:'Train reasoning model'})).toBeEnabled());
 fireEvent.click(screen.getByRole('button',{name:'Train reasoning model'}));
 await waitFor(()=>expect(resolveStart).toBeDefined());
 expect(screen.getByRole('button',{name:'Next frame'})).toBeEnabled();
 expect(screen.getByRole('button',{name:'Object masks (optional)'})).toBeEnabled();
 expect(onBusyChange).toHaveBeenLastCalledWith(false);
 await act(async()=>resolveStart({id:'new',status:'training',step:0,steps:200}));
});
