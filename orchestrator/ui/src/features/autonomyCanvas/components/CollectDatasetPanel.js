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

import {useEffect,useState} from "react";
import CollectAnnotationPanel from "./CollectAnnotationPanel";
import CollectReasoningPanel from "./CollectReasoningPanel";
import useReasoningTraining from "./useReasoningTraining";
import ReasoningTrainingButton from "./ReasoningTrainingButton";
import ReasoningTrainingDrawer from "./ReasoningTrainingDrawer";
import "./CollectDatasetPanel.css";

export default function CollectDatasetPanel({draft,editor,maskDirty=false,setMaskDirty,onTestModel,onBusyChange,selection,onSelectionChange}) {
  const [localSelection,setLocalSelection]=useState({tool:"mission"});
  const current=selection||localSelection;
  const {tool="mission"}=current;
  const updateSelection=onSelectionChange||setLocalSelection;
  const [working,setWorking]=useState(false),[trainingOpen,setTrainingOpen]=useState(false);
  const locked=draft.active||draft.busy||editor.busy||editor.saving||editor.maskBusy||editor.labelDirty||maskDirty||working;
  const training=useReasoningTraining({episode:draft.episode,episodes:draft.episodes,disabled:locked});
  useEffect(()=>{
    onBusyChange?.(working);
    return()=>onBusyChange?.(false);
  },[working,onBusyChange]);
  const labelEditor={...editor,busy:editor.busy,frameDeletionBlocked:training.running||training.working};
  return <section className="collect-dataset-panel annotate-dataset-panel" aria-label="Dataset editor">
    <div className="collect-dataset-toolbar">
      <div className="collect-dataset-tools" role="group" aria-label="Dataset tools">
        <button type="button" aria-pressed={tool==="mission"} disabled={locked} onClick={()=>updateSelection(previous=>({...previous,tool:"mission"}))}>Reasoning labels</button>
        <button type="button" aria-pressed={tool==="masks"} disabled={locked} onClick={()=>updateSelection(previous=>({...previous,tool:"masks"}))}>Object masks <span>(optional)</span></button>
      </div>
      <ReasoningTrainingButton training={training} open={trainingOpen} onClick={()=>setTrainingOpen(value=>!value)}/>
    </div>
    {tool==="mission" ? <CollectReasoningPanel draft={draft} editor={labelEditor}
      selectedModelId={current.autoLabelModels?.mission} onBusyChange={setWorking}/> :
      <CollectAnnotationPanel {...{draft,maskDirty,setMaskDirty}} editor={labelEditor}
        selectedModelId={current.autoLabelModels?.masks} onBusyChange={setWorking}/>}
    <ReasoningTrainingDrawer open={trainingOpen} onClose={()=>setTrainingOpen(false)} training={training} onTestModel={onTestModel}/>
  </section>;
}
