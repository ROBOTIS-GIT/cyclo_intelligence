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

import {render,screen} from '@testing-library/react';
import CollectSidebar from './CollectSidebar';
import CollectStageHud from './CollectStageHud';
jest.mock('./LiveHeadCamera',()=>()=> <img alt="Live head camera"/>);
jest.mock('../../mapWorkspace/components/MappingTeleopPanel',()=>()=> <div>Mobile Teleop</div>);
jest.mock('./CollectDatasetPanel',()=>()=> <div>Dataset editor</div>);
jest.mock('./CollectEpisodeLibrary',()=>()=> <div>Episode library</div>);

test('Record puts reasoning above the full-width map; Dataset preserves the map mount',()=>{
 const draft={episodes:[],episode:null,selectedFrame:null,active:false,busy:false};
 const observation={available:true,mapName:'warehouse',phase:'watching',result:{
  attention_preview:{image_base64:'heat',layer:23},decision:{decision:'wait',reason:'No task object is visible.'},selected_mission:null}};
 const props={draft,observation,mapName:'warehouse',mapCanvas:<div data-testid="record-map"><CollectStageHud/></div>};
 const view=render(<CollectSidebar {...props} mode="live"/>);
 expect(screen.getByAltText('Attention over the analyzed camera frame')).toBeInTheDocument();
 expect(screen.getAllByAltText('Live head camera')).toHaveLength(1);
 expect(screen.getByRole('region',{name:'Record reasoning'}).compareDocumentPosition(screen.getByTestId('record-map')) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
 expect(screen.queryByText('Mobile Teleop')).not.toBeInTheDocument();
 expect(screen.getByRole('button',{name:'Start reasoning'})).toBeEnabled();
 expect(screen.getByText('No task object is visible.')).toBeInTheDocument();
 expect(screen.queryByText('Dataset editor')).not.toBeInTheDocument();
 view.rerender(<CollectSidebar {...props} mode="dataset"/>);
 expect(screen.queryByAltText('Live head camera')).not.toBeInTheDocument();
 expect(screen.queryByRole('button',{name:'Start reasoning'})).not.toBeInTheDocument();
 expect(screen.getByTestId('record-map')).not.toBeVisible();
 expect(screen.getByText('Dataset editor')).toBeInTheDocument();
 expect(screen.getByText('Episode library')).toBeInTheDocument();
 view.rerender(<CollectSidebar {...props} mode="live"/>);
 expect(screen.getByTestId('record-map')).toBeVisible();
 expect(screen.getByAltText('Attention over the analyzed camera frame')).toBeInTheDocument();
 expect(screen.getAllByAltText('Live head camera')).toHaveLength(1);
});


test('continuous capture does not toggle the stage rail lock for every saved frame', () => {
 const draft={episodes:[],episode:null,selectedFrame:null,active:false,busy:false,recording:true};
 const onEditorBusyChange=jest.fn();
 const props={draft,mode:'live',onEditorBusyChange};
 const view=render(<CollectSidebar {...props}/>);
 expect(onEditorBusyChange).toHaveBeenLastCalledWith(true);
 onEditorBusyChange.mockClear();
 for(const busy of [true,false,true,false]) {
   view.rerender(<CollectSidebar {...props} draft={{...draft,busy}}/>);
 }
 expect(onEditorBusyChange).not.toHaveBeenCalled();
 view.rerender(<CollectSidebar {...props} draft={{...draft,recording:false}}/>);
 expect(onEditorBusyChange).toHaveBeenLastCalledWith(false);
});
