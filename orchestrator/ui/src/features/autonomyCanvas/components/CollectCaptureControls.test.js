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

import { useState } from "react";
import { act, render, screen, fireEvent, within } from '@testing-library/react';
import CollectSidebar from './CollectSidebar';
import CollectCaptureControls from './CollectCaptureControls';
import CollectStageHud from './CollectStageHud';
jest.mock('./LiveHeadCamera', () => () => null);
jest.mock('../../mapWorkspace/components/MappingTeleopPanel', () => () => null);

function setup(active = false) {
  const draft = {episodes:[],episode:active ? {id:'ep',map_name:'warehouse',observations:[],annotations:[]} : null,
    active,busy:false,recording:false,start:jest.fn(),startRecording:jest.fn(),stopRecording:jest.fn(),finish:jest.fn(),selectFrame:jest.fn()};
  const props={draft,mapName:'warehouse',mapLoaded:true,mapCanvas:<CollectStageHud missionNames={['Mission1']} selectedMission=""/>};
  const view=render(<CollectSidebar {...props}/>);
  return {draft,props,view};
}

test('Mission and Start episode share the map HUD without requiring a selected route', () => {
  const {draft}=setup();
  const mission=screen.getByLabelText('Collect map mission');
  const start=screen.getByRole('button',{name:'Start episode'});
  expect(mission.closest('.record-control-panel')).toContainElement(start);
  expect(start).toBeEnabled();
  fireEvent.click(start);
  expect(draft.start).toHaveBeenCalledWith(expect.objectContaining({episode_type:'observation',recording_mission_name:null}));
  expect(draft.startRecording).not.toHaveBeenCalled();
});

test('only an actually running mission supplies a mission capture identity', () => {
  const {draft,props,view}=setup();
  view.rerender(<CollectSidebar {...props} recordMission="Mission1" recordSession={{type:"mission",ready:true,missionId:"m1",missionName:"Mission1",observationSpotId:"a",returnSpotId:"a",planRevision:2}}/>);
  fireEvent.click(screen.getByRole('button',{name:'Start episode'}));
  expect(draft.start).toHaveBeenCalledWith(expect.objectContaining({episode_type:'mission',recording_mission_name:'Mission1'}));
});

test('continuous capture works while stationary and saving is available during recording', async () => {
  const {draft,props,view}=setup(true);
  const capture=screen.getByRole('button',{name:'Capture'});
  expect(capture).toBeEnabled();
  fireEvent.click(capture);
  expect(draft.startRecording).toHaveBeenCalledWith(1000,expect.any(Function),expect.any(Function));
  draft.recording=true;
  view.rerender(<CollectSidebar {...props}/>);
  const save=screen.getByRole('button',{name:'Save episode'});
  expect(save).toBeEnabled();
  await act(async()=>fireEvent.click(save));
  expect(draft.finish).toHaveBeenCalledTimes(1);
});

test('single capture appends frames without automatically ending the episode', async () => {
  const {draft}=setup(true);
  draft.capture=jest.fn();
  fireEvent.change(screen.getByLabelText('Capture mode'),{target:{value:'single'}});
  await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Capture'})));
  await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Capture'})));
  expect(draft.capture).toHaveBeenCalledTimes(2);
  expect(draft.finish).not.toHaveBeenCalled();
});

test('capture controls keep four action positions throughout the episode lifecycle', () => {
  const {draft, props, view} = setup();
  const panel = within(screen.getByRole('group', {name:'Episode actions'}));
  const buttons = panel.getAllByRole('button');
  expect(buttons.map(button => button.getAttribute('aria-label'))).toEqual([
    'Start episode', 'Capture', 'Save episode', 'Delete episode',
  ]);
  const expectState = (enabled, captureName = 'Capture') => {
    const current = panel.getAllByRole('button');
    expect(current).toHaveLength(4);
    current.forEach((button, i) => {
      expect(button).toBe(buttons[i]);
      if (enabled.includes(i)) expect(button).toBeEnabled();
      else expect(button).toBeDisabled();
    });
    expect(buttons[1]).toHaveAccessibleName(captureName);
  };
  expectState([0, 3]);
  draft.active = true;
  draft.episode = {id:'ep',map_name:'warehouse',observations:[],annotations:[]};
  view.rerender(<CollectSidebar {...props}/>);
  expectState([1, 2, 3]);
  draft.recording = true;
  view.rerender(<CollectSidebar {...props}/>);
  expectState([1, 2], 'Stop capture');
  draft.recording = false;
  draft.active = false;
  draft.episode.ended_at = '2026-09-22';
  draft.episode.observations = [{frame_id:'f1'}];
  view.rerender(<CollectSidebar {...props}/>);
  expectState([0, 3]);
  expect(screen.queryByRole('button', {name:'Annotate',exact:true})).not.toBeInTheDocument();
  expect(screen.queryByRole('button', {name:'New episode',exact:true})).not.toBeInTheDocument();
  expect(screen.getByLabelText('Capture mode')).toBeInTheDocument();
  expect(screen.getByLabelText('Capture interval')).toHaveValue('1000');
  fireEvent.click(buttons[0]);
  expect(draft.start).toHaveBeenCalledTimes(1);
});


test('all unfinished map episodes are selectable without changing robot execution', () => {
  const draft = {episode:null,active:false,busy:false,episodes:[
    {id:'return',map_name:'house',episode_type:'patrol',capture_segment:'mission_return',storage_path:'house/patrol/raw_episode_0002'},
    {id:'patrol',map_name:'house',episode_type:'patrol',storage_path:'house/patrol/raw_episode_0003'},
    {id:'other',map_name:'warehouse',episode_type:'patrol',capture_segment:'mission_return'},
  ]};
  const recording = {captureMode:'continuous',intervalMs:1000,captureFields:{current:{}},resumeEpisode:jest.fn()};
  const props = {draft,recording,mapName:'house',mapLoaded:true,recordSession:{type:'mission'}};
  const view = render(<CollectCaptureControls {...props}/>);
  expect(screen.getByRole('option', {name:'Return · raw_episode_0002'})).toBeInTheDocument();
  expect(screen.getByRole('option', {name:'raw_episode_0003'})).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Resume episode'), {target:{value:'return'}});
  expect(recording.resumeEpisode).toHaveBeenCalledWith('return');
  view.rerender(<CollectCaptureControls {...props} recordSession={{type:'patrol'}}/>);
  expect(screen.getByRole('option', {name:'Return · raw_episode_0002'})).toBeInTheDocument();
  expect(screen.getByRole('option', {name:'raw_episode_0003'})).toBeInTheDocument();
});

test('Episode appears first and capture preparation does not gate child missions',async()=>{
 const run=jest.fn(),onStart=jest.fn();
 function View(){
  const [episode,setEpisode]=useState(null),[recording,setRecording]=useState(false);
  const draft={episode,episodes:[],active:!!episode&&!episode.ended_at,busy:false,recording,
   start:async fields=>{setEpisode({id:'prepared',...fields,observations:[]});return true;},
   startRecording:jest.fn(()=>setRecording(true)),stopRecording:()=>setRecording(false),
   finish:async()=>{setEpisode(value=>({...value,ended_at:1000}));return true;}};
  const recordSession={type:'observation',ready:true,observationSpotId:'table',planRevision:1,
   captureTargets:{mission:{type:'mission',ready:true,missionId:'clean',missionName:'CleanTable',observationSpotId:'table',returnSpotId:'table',planRevision:1}}};
  return <CollectSidebar draft={draft} mapName="house" mapLoaded recordSession={recordSession}
   mapCanvas={<CollectStageHud localized canStartMission missionNames={['CleanTable']} selectedMission="CleanTable" onRunMission={run} onStart={onStart}/>}/>;
 }
 render(<View/>);
 const episode=screen.getByRole('region',{name:'Episode capture'});
 expect(episode.closest('.record-control-panel')).toContainElement(screen.getByRole('button',{name:'Run child mission'}));
 const runButton=screen.getByRole('button',{name:'Run child mission'}),start=screen.getByRole('button',{name:'Start episode'}),capture=screen.getByRole('button',{name:'Capture'});
 expect(runButton).toBeEnabled();expect(capture).toBeDisabled();
 expect(start).toBeEnabled();
 await act(async()=>fireEvent.click(start));
 expect(capture).toBeEnabled();expect(runButton).toBeEnabled();
 await act(async()=>fireEvent.click(capture));
 expect(runButton).toBeEnabled();expect(screen.getByRole('button',{name:'Start patrol'})).toBeEnabled();
 expect(screen.queryByLabelText('Capture type')).not.toBeInTheDocument();
 await act(async()=>fireEvent.click(runButton));
 expect(run).toHaveBeenCalledTimes(1);expect(onStart).not.toHaveBeenCalled();
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Save episode'})));
 expect(runButton).toBeEnabled();
});

test('recording follows execution without a Capture type selector and Patrol needs no selected mission',async()=>{
 const onStart=jest.fn(async()=>true),onRunMission=jest.fn();
 function View(){
  const [episode,setEpisode]=useState(null),[recording,setRecording]=useState(false);
  const draft={episode,episodes:[],active:!!episode,busy:false,recording,
   start:async fields=>{setEpisode({id:'prepared',...fields,observations:[]});return true;},
   startRecording:()=>setRecording(true),stopRecording:()=>setRecording(false)};
  const recordSession={type:'observation',ready:true,observationSpotId:'table',planRevision:1,
   captureTargets:{patrol:{type:'patrol',ready:true,patrolSpotIds:['table','laundry'],planRevision:1}}};
  return <CollectSidebar draft={draft} mapName="house" mapLoaded recordSession={recordSession}
   mapCanvas={<CollectStageHud localized canStartMission missionNames={['CleanTable']} onRunMission={onRunMission} onStart={onStart}/>}/>;
 }
 render(<View/>);
 const select=screen.getByLabelText('Collect map mission');
 expect(select).toBeEnabled();
 expect(screen.queryByLabelText('Capture type')).not.toBeInTheDocument();
 expect(screen.getByLabelText('Recording type')).toHaveTextContent('Observation');
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Start episode'})));
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Capture'})));
 expect(select).toBeEnabled();expect(screen.getByRole('button',{name:'Run child mission'})).toBeDisabled();
 expect(screen.getByRole('button',{name:'Start patrol'})).toBeEnabled();
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Start patrol'})));
 expect(onStart).toHaveBeenCalledTimes(1);expect(onRunMission).not.toHaveBeenCalled();
});


test('read-only recording type follows the verified execution role',()=>{
 const draft={episodes:[],active:true,episode:{id:'ep',episode_type:'patrol',observations:[]}};
 const recording={captureMode:'continuous',intervalMs:1000,targetReady:true};
 const view=render(<CollectCaptureControls draft={draft} recording={{...recording,captureType:'patrol'}} mapName="house" mapLoaded/>);
 expect(screen.getByLabelText('Recording type')).toHaveTextContent('Patrol');
 view.rerender(<CollectCaptureControls draft={draft} recording={{...recording,captureType:'mission'}} mapName="house" mapLoaded/>);
 expect(screen.getByLabelText('Recording type')).toHaveTextContent('Mission');
 view.rerender(<CollectCaptureControls draft={draft} recording={{...recording,captureType:null,awaitingTarget:true}} recordSession={{type:'mission',ready:false}} mapName="house" mapLoaded/>);
 expect(screen.getByLabelText('Recording type')).toHaveTextContent('Awaiting execution');
 expect(screen.queryByRole('combobox',{name:'Capture type'})).not.toBeInTheDocument();
});


test('failed mission shows why capture is waiting and keeps Stop and Save available',()=>{
 const {draft,props,view}=setup(true);draft.recording=true;
 view.rerender(<CollectSidebar {...props} recordSession={{mapName:'warehouse',ready:false,type:null,
   waitingReason:'Waypoint Task failed at LaundryBasket'}}/>);
 expect(screen.getByText(/Capture waiting .*Waypoint Task failed at LaundryBasket/)).toBeInTheDocument();
 expect(screen.getByRole('button',{name:'Stop capture'})).toBeEnabled();
 expect(screen.getByRole('button',{name:'Save episode'})).toBeEnabled();
});


test('capture mode and interval are always visible and retain their values', () => {
  const {props, view} = setup(true);
  expect(screen.queryByRole('button', {name:'Capture settings'})).not.toBeInTheDocument();
  expect(screen.getByLabelText('Capture mode')).toBeInTheDocument();
  expect(screen.getByLabelText('Capture interval')).toHaveValue('1000');
  fireEvent.change(screen.getByLabelText('Capture interval'), {target:{value:'2000'}});
  expect(screen.getByLabelText('Capture interval')).toHaveValue('2000');
  fireEvent.change(screen.getByLabelText('Capture mode'), {target:{value:'single'}});
  expect(screen.queryByLabelText('Capture summary')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Capture interval')).not.toBeInTheDocument();
  view.rerender(<CollectSidebar {...props}/>);
  expect(screen.getByLabelText('Capture mode')).toHaveValue('single');
  fireEvent.change(screen.getByLabelText('Capture mode'), {target:{value:'continuous'}});
  expect(screen.getByLabelText('Capture interval')).toHaveValue('2000');
});

test('visible settings are locked during capture while Stop remains available', () => {
  const {draft, props, view} = setup(true);
  draft.recording = true;
  view.rerender(<CollectSidebar {...props}/>);
  const capture = screen.getByRole('button', {name:'Stop capture'});
  expect(screen.getByLabelText('Capture mode')).toBeDisabled();
  expect(screen.getByLabelText('Capture interval')).toBeDisabled();
  expect(screen.getByRole('button', {name:'Stop capture'})).toBe(capture);
  expect(capture).toBeEnabled();
  expect(capture).toHaveAttribute('aria-pressed', 'true');
  expect(draft.startRecording).not.toHaveBeenCalled();
  expect(draft.stopRecording).not.toHaveBeenCalled();
  fireEvent.click(capture);
  expect(draft.stopRecording).toHaveBeenCalledTimes(1);
  expect(draft.finish).not.toHaveBeenCalled();
});

test('continuous capture toggles the same button through start, pause, and restart', async () => {
  const draft = {episodes:[],episode:{id:'ep',observations:[]},active:true,busy:false,recording:false};
  const recording = {captureMode:'continuous',intervalMs:1000,targetReady:true,
    startCapture:jest.fn(),stopCapture:jest.fn(),saveEpisode:jest.fn()};
  const props = {draft,recording,mapName:'warehouse',mapLoaded:true};
  const view = render(<CollectCaptureControls {...props}/>);
  const capture = screen.getByRole('button', {name:'Capture'});
  await act(async () => fireEvent.click(capture));
  expect(recording.startCapture).toHaveBeenCalledTimes(1);
  view.rerender(<CollectCaptureControls {...props} draft={{...draft,recording:true,busy:true}}
    recording={{...recording,captureTransition:true}} disabled mapLoaded={false}/>);
  expect(screen.getByRole('button', {name:'Stop capture'})).toBe(capture);
  expect(capture).toBeEnabled();
  expect(capture).toHaveAttribute('title', 'Pause capture');
  fireEvent.click(capture);
  expect(recording.stopCapture).toHaveBeenCalledTimes(1);
  expect(recording.saveEpisode).not.toHaveBeenCalled();
  view.rerender(<CollectCaptureControls {...props}/>);
  expect(screen.getByRole('button', {name:'Capture'})).toBe(capture);
  await act(async () => fireEvent.click(capture));
  expect(recording.startCapture).toHaveBeenCalledTimes(2);
});

test('an in-flight single frame keeps the combined stop action available while busy or disabled', async () => {
  let finishFrame;
  const frame = new Promise(resolve => { finishFrame = resolve; });
  const draft = {episodes:[],episode:{id:'ep',observations:[]},active:true,busy:false,recording:false};
  const recording = {captureMode:'single',intervalMs:1000,targetReady:true,
    captureSingle:jest.fn(() => frame),stopCapture:jest.fn()};
  const props = {draft,recording,mapName:'warehouse',mapLoaded:true};
  const view = render(<CollectCaptureControls {...props}/>);
  const capture = screen.getByRole('button', {name:'Capture'});
  fireEvent.click(capture);
  expect(recording.captureSingle).toHaveBeenCalledTimes(1);
  view.rerender(<CollectCaptureControls {...props} draft={{...draft,busy:true}} disabled mapLoaded={false}/>);
  expect(screen.getByRole('button', {name:'Stop capture'})).toBe(capture);
  expect(capture).toBeEnabled();
  fireEvent.click(capture);
  expect(recording.stopCapture).toHaveBeenCalledTimes(1);
  expect(recording.captureSingle).toHaveBeenCalledTimes(1);
  await act(async () => finishFrame());
  view.rerender(<CollectCaptureControls {...props}/>);
  expect(screen.getByRole('button', {name:'Capture'})).toBe(capture);
  expect(capture).toBeEnabled();
});

test('a queued single frame can still be stopped when capture is disabled', () => {
  const draft = {episodes:[],episode:{id:'ep',observations:[]},active:true,busy:true,recording:false};
  const recording = {captureMode:'single',intervalMs:1000,singlePending:true,
    captureSingle:jest.fn(),stopCapture:jest.fn()};
  render(<CollectCaptureControls draft={draft} recording={recording} mapName="warehouse" disabled/>);
  const capture = screen.getByRole('button', {name:'Stop capture'});
  expect(capture).toBeEnabled();
  fireEvent.click(capture);
  expect(recording.stopCapture).toHaveBeenCalledTimes(1);
  expect(recording.captureSingle).not.toHaveBeenCalled();
});


test('map selection updates leave live capture and settings mounted', async()=>{
 const {draft,props,view}=setup(true);
 fireEvent.change(screen.getByLabelText('Capture interval'),{target:{value:'2000'}});
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Capture'})));
 draft.recording=true;view.rerender(<CollectSidebar {...props}/>);
 const pause=screen.getByRole('button',{name:'Stop capture'});
 view.rerender(<CollectSidebar {...props} mapCanvas={<CollectStageHud observationName="Laundry"/>}/>);
 expect(screen.getByRole('button',{name:'Stop capture'})).toBe(pause);
 expect(screen.getByRole('toolbar',{name:'Movement controls'})).toBeInTheDocument();
 expect(draft.stopRecording).not.toHaveBeenCalled();
 view.rerender(<CollectSidebar {...props} mapCanvas={<CollectStageHud observationName="TV"/>}/>);
 expect(screen.getByRole('button',{name:'Stop capture'})).toBe(pause);
 expect(screen.getByLabelText('Capture interval')).toHaveValue('2000');
 expect(draft.startRecording).toHaveBeenCalledTimes(1);
 expect(draft.finish).not.toHaveBeenCalled();
});
