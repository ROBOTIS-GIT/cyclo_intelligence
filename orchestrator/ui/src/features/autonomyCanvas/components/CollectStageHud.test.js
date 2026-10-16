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

import { act, render, screen, fireEvent, within } from '@testing-library/react';
import CollectStageHud from './CollectStageHud';
import { CollectCaptureContext } from './CollectCaptureContext';
const ReadyHud = props => <CollectCaptureContext.Provider value={{}}><CollectStageHud {...props}/></CollectCaptureContext.Provider>;

test('prepared capture admits Start which runs Patrol only',async()=>{
 const onStart=jest.fn(async()=>true),onRunMission=jest.fn();
 render(<ReadyHud localized onStart={onStart} onRunMission={onRunMission}/>);
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Start patrol'})));
 expect(onStart).toHaveBeenCalledTimes(1);expect(onRunMission).not.toHaveBeenCalled();
 expect(screen.getByRole('button',{name:'Stop',exact:true})).toBeDisabled();
});
test('Stop only stops execution; unloading uses a separate explicit control',async()=>{
 const onStopRoute=jest.fn(async()=>true),onShutdown=jest.fn(async()=>true);
 const props={localized:true,navigationActive:true,onStopRoute,onShutdown};
 const view=render(<ReadyHud {...props} routeActive/>);
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Stop',exact:true})));
 expect(onStopRoute).toHaveBeenCalledTimes(1);expect(onShutdown).not.toHaveBeenCalled();
 view.rerender(<ReadyHud {...props}/>);
 expect(screen.getByRole('button',{name:'Stop',exact:true})).toBeDisabled();
 expect(screen.queryByRole('button',{name:'More record options'})).not.toBeInTheDocument();
});
test('movement controls are fixed across Route and Capture with localization gates',()=>{
 const view=render(<ReadyHud/>);
 for(const name of ['Start patrol','Previous observation','Next observation','Stop'])expect(screen.getByRole('button',{name,exact:true})).toBeDisabled();
 const controls=screen.getByRole('toolbar',{name:'Movement controls'});
 expect(within(controls).getAllByRole('button').map(button=>button.getAttribute('aria-label'))).toEqual(['Localize','Previous observation','Start patrol','Next observation','Stop']);
 const start=screen.getByRole('button',{name:'Start patrol'});
 expect(screen.getByRole('button',{name:'Start patrol'})).toBe(start);
 view.rerender(<ReadyHud localized/>);
 expect(start).toBeEnabled();
});
test('child mission launch is separate and available only at its observation',async()=>{
 const run=jest.fn(async()=>true),onMissionChange=jest.fn();
 const props={missionNames:['CleanTable'],selectedMission:'CleanTable',onMissionChange,onRunMission:run};
 const view=render(<ReadyHud {...props}/>);
 expect(screen.getByRole('button',{name:'Run child mission'})).toBeDisabled();
 view.rerender(<ReadyHud {...props} canStartMission/>);
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Run child mission'})));
 expect(run).toHaveBeenCalledTimes(1);
});
test('a pending motion command is not submitted twice',async()=>{
 let resolve;const onStart=jest.fn(()=>new Promise(done=>{resolve=done;}));
 render(<ReadyHud localized onStart={onStart}/>);
 fireEvent.click(screen.getByRole('button',{name:'Start patrol'}));
 fireEvent.click(screen.getByRole('button',{name:'Start patrol'}));
 expect(onStart).toHaveBeenCalledTimes(1);
 await act(async()=>resolve(true));
});


test('Stop remains usable during a pending Start and after localization is lost',async()=>{
 let resolve;const onStart=jest.fn(()=>new Promise(done=>{resolve=done;})),onStopRoute=jest.fn(async()=>true);
 const view=render(<ReadyHud localized onStart={onStart} onStopRoute={onStopRoute}/>);
 fireEvent.click(screen.getByRole('button',{name:'Start patrol'}));
 view.rerender(<ReadyHud routeActive onStart={onStart} onStopRoute={onStopRoute}/>);
 expect(screen.getByRole('button',{name:'Stop',exact:true})).toBeEnabled();
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Stop',exact:true})));
 expect(onStopRoute).toHaveBeenCalledTimes(1);
 await act(async()=>resolve(false));
 expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

test('patrol and child missions are independent of capture preparation',()=>{
 const props={localized:true,canStartMission:true,selectedMission:'CleanTable',missionNames:['CleanTable']};
 const view=render(<CollectStageHud {...props}/>);
 expect(screen.getByRole('button',{name:'Start patrol'})).toBeEnabled();
 expect(screen.getByRole('button',{name:'Run child mission'})).toBeEnabled();
 view.rerender(<ReadyHud {...props}/>);
 expect(screen.getByRole('button',{name:'Start patrol'})).toBeEnabled();
 expect(screen.getByRole('button',{name:'Run child mission'})).toBeEnabled();
});

test('patrol start icon toggles Pause and Resume without controlling capture', async () => {
 const onStart=jest.fn(async()=>true),onPausePatrol=jest.fn(async()=>true),onResumePatrol=jest.fn(async()=>true);
 const props={localized:true,onStart,onPausePatrol,onResumePatrol};
 const view=render(<ReadyHud {...props} patrolRunning routeActive/>);
 const pause=screen.getByRole('button',{name:'Pause patrol'});
 expect(screen.getByRole('button',{name:'Previous observation'}).nextElementSibling).toBe(pause);
 await act(async()=>fireEvent.click(pause));expect(onPausePatrol).toHaveBeenCalledTimes(1);
 view.rerender(<ReadyHud {...props} patrolPaused routeActive/>);
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Resume patrol'})));
 expect(onResumePatrol).toHaveBeenCalledTimes(1);expect(onStart).not.toHaveBeenCalled();
 expect(screen.getByRole('button',{name:'Stop',exact:true})).toBeEnabled();
});


test('Next observation icon sits between patrol and Stop and follows execution availability',async()=>{
 const onSkipObservation=jest.fn(async()=>true);
 const view=render(<ReadyHud localized patrolRunning routeActive onSkipObservation={onSkipObservation}/>);
 const next=screen.getByRole('button',{name:'Next observation'});
 expect(next).toBeDisabled();expect(next.previousElementSibling).toBe(screen.getByRole('button',{name:'Pause patrol'}));
 expect(next.nextElementSibling).toBe(screen.getByRole('button',{name:'Stop',exact:true}));
 view.rerender(<ReadyHud localized patrolRunning routeActive canSkipObservation onSkipObservation={onSkipObservation}/>);
 await act(async()=>fireEvent.click(next));expect(onSkipObservation).toHaveBeenCalledTimes(1);
 view.rerender(<ReadyHud canSkipObservation onSkipObservation={onSkipObservation}/>);
 expect(next).toBeDisabled();
});


test('Previous observation dispatches independently of Next',async()=>{
 const previous=jest.fn(async()=>true);
 render(<ReadyHud localized routeActive canPreviousObservation onPreviousObservation={previous}/>);
 expect(screen.getByRole('button',{name:'Next observation'})).toBeDisabled();
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Previous observation'})));
 expect(previous).toHaveBeenCalledTimes(1);
});

function observationManager(changes={}) {
 return {ready:true, spots:[{id:'table',label:'Table'}],selectedId:'table',
  configuration:{plan:{observations:[{spot_id:'table',dwell_seconds:120}],route_closed:false}},
  route:{routeMode:false,routeView:{closed:false},resetEditing:jest.fn(),toggleMode:jest.fn()},
  routeEditor:{dirty:false},select:jest.fn(), ...changes};
}
test('map authoring stays beside movement controls and Capture has no mode tabs',()=>{
 const manager=observationManager();
 render(<ReadyHud observations={manager}/>);
 expect(screen.queryByRole('tab',{name:'Route'})).not.toBeInTheDocument();
 expect(screen.queryByRole('tab',{name:'Capture'})).not.toBeInTheDocument();
 expect(screen.getByRole('group',{name:'Waypoint tools'}).closest('.record-map-tools'))
  .toContainElement(screen.getByRole('toolbar',{name:'Movement controls'}));
 expect(screen.getByRole('button',{name:'Create observation waypoint'})).toBeEnabled();
 fireEvent.click(screen.getByRole('button',{name:'Edit observation route'}));
 expect(manager.route.toggleMode).toHaveBeenCalledTimes(1);
});
test('Go to selected observation is explicit and remains independent from authoring locks',async()=>{
 const go=jest.fn(async()=>true),manager=observationManager();
 const ui=render(<ReadyHud observations={manager} waypointAnchor={{x:400,y:300,width:1000,height:800}} onGoToObservation={go} authoringDisabled/>);
 expect(screen.getByRole('button',{name:'Go to selected observation'})).toBeDisabled();
 ui.rerender(<ReadyHud observations={manager} waypointAnchor={{x:400,y:300,width:1000,height:800}} onGoToObservation={go} authoringDisabled localized/>);
 expect(screen.getByRole('button',{name:'Edit observation route'})).toBeDisabled();
 await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Go to selected observation'})));
 expect(go).toHaveBeenCalledWith('table');
});

test('selected waypoint actions stay hidden during a child mission or return',()=>{
 const manager=observationManager();
 const props={observations:manager,waypointAnchor:{x:400,y:300,width:1000,height:800},localized:true};
 const ui=render(<ReadyHud {...props}/>);
 expect(screen.getByRole('button',{name:'Go to selected observation'})).toBeInTheDocument();
 ui.rerender(<ReadyHud {...props} observationSelectionDisabled/>);
 expect(screen.queryByRole('button',{name:'Go to selected observation'})).not.toBeInTheDocument();
});



test('order mode keeps only two route controls and hides waypoint settings',()=>{
 const manager=observationManager({route:{routeMode:true,orderMode:true,routeView:{closed:false}},
  routeEditor:{plan:{observations:[{spot_id:'table',dwell_seconds:120}]},dirty:true,error:'Save failed'}});
 render(<ReadyHud observations={manager} waypointAnchor={{x:400,y:300,width:1000,height:800}}/>);
 expect(screen.getByRole('button',{name:'Set observation route order'})).toHaveAttribute('aria-pressed','true');
 expect(screen.getByRole('button',{name:'Save observation route'})).toBeEnabled();
 expect(screen.queryByRole('region',{name:'Observation waypoint settings'})).not.toBeInTheDocument();
 expect(screen.queryByRole('button',{name:'Go to selected observation'})).not.toBeInTheDocument();
 expect(screen.getByRole('alert')).toHaveTextContent('Save failed');
});
