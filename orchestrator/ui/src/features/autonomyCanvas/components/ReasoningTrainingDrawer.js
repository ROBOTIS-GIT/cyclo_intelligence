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

import {useEffect,useRef,useState} from "react";
import {MdClose,MdDeleteOutline} from "react-icons/md";
import {jobRunning} from "./trainingJobStatus";
import TrainingCancelButton from "./TrainingCancelButton";
import TrainingBalance from "./TrainingBalance";
import ReasoningTypeScope from "./ReasoningTypeScope";
import {EPISODE_TYPES, TYPE_LABELS} from "../utils/reasoningDataset";
import "./ReasoningTrainingDrawer.css";

export default function ReasoningTrainingDrawer({open,onClose,training:t,onTestModel}) {
  const panel=useRef(null);
  const [deleteId,setDeleteId]=useState(null);
  useEffect(()=>{setDeleteId(null);},[open,t.selectedModel?.id]);
  const [sessionJobId,setSessionJobId]=useState(null);
  useEffect(()=>{if(!open)setSessionJobId(null);},[open]);
  const activeJobId = t.job?.id;
  const jobIsRunning = jobRunning(t.job);
  useEffect(() => {
    if (open && !t.starting && jobIsRunning) setSessionJobId(activeJobId);
  }, [open, t.starting, activeJobId, jobIsRunning]);
  const job=!t.starting && (jobIsRunning || (sessionJobId && sessionJobId===activeJobId)) ? t.job : null;
  useEffect(()=>{
    if(!open)return;
    const previous=document.activeElement;
    panel.current?.querySelector('button')?.focus();
    return()=>previous?.focus?.();
  },[open]);
  if(!open)return null;
  function keyDown(event) {
    if(event.key==="Escape"){event.stopPropagation();onClose();return;}
    if(event.key!=="Tab")return;
    const focusable=[...panel.current.querySelectorAll('button:not(:disabled),input:not(:disabled),select:not(:disabled),summary')]
      .filter(node=>!node.closest('details:not([open])')||node.tagName==="SUMMARY");
    const first=focusable[0],last=focusable[focusable.length-1];
    if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
    if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
  }
  return <div className="learn-training-overlay" onClick={event=>{if(event.target===event.currentTarget)onClose();}}>
    <section ref={panel} className="learn-training-drawer" role="dialog" aria-modal="true" aria-label="Train reasoning model" onKeyDown={keyDown}>
      <header><strong>Train reasoning model</strong><div className="learn-training-header-actions">
        <TrainingCancelButton label="Cancel training" showLabel job={job} onCancel={t.cancel} pending={t.working}/>
        <button type="button" aria-label="Close training settings" onClick={onClose}><MdClose size={20}/></button>
      </div></header>
      <div className="learn-training-body">
        <label>Training map<select value={t.mapName} disabled={t.locked||!t.maps.length} onChange={e=>t.setMap(e.target.value)}>
          {!t.maps.length&&<option value="">No saved dataset map</option>}
          {t.maps.map(map=><option key={map} value={map}>{map}</option>)}
        </select></label>
        <section className="learn-training-data" aria-label="Training dataset">
          <strong>{t.count} approved labels · {t.episodeIds.length} saved episodes</strong>
          <div className="learn-training-counts">{EPISODE_TYPES.map(type=><span key={type}>{TYPE_LABELS[type]} <b>{t.types.includes(type)?t.typeCounts[type]||0:0}</b></span>)}</div>
          <small>Approved reasoning labels across this map.</small>
        </section>
        <div className="learn-training-fields">
          <label>Model name<input value={t.name} maxLength={80} disabled={t.locked} onChange={e=>t.setName(e.target.value)} placeholder="e.g. House autonomy v1"/></label>
          <label>Training steps<input type="number" min="1" step="1" value={t.steps} disabled={t.locked} onChange={e=>t.setSteps(e.target.value)}/></label>
        </div>
        <TrainingBalance compact mapName={t.mapName} episodeIds={t.episodeIds} count={t.count}
          enabled={t.balance} onChange={t.setBalance} disabled={t.locked} onReady={t.setBalanceReady}/>
        <details className="learn-training-advanced"><summary>Advanced settings</summary>
          <label>Base model<select value={t.baseModel} disabled={t.locked} onChange={e=>t.setBaseModel(e.target.value)}>
            <option value="Qwen/Qwen3.5-4B">Qwen3.5-4B</option><option value="Qwen/Qwen3.5-2B">Qwen3.5-2B</option><option value="Qwen/Qwen3.5-0.8B">Qwen3.5-0.8B</option>
          </select></label>
          <ReasoningTypeScope value={t.types} onChange={t.setTypes} counts={t.typeCounts} disabled={t.locked}/>
        </details>
        <button className="learn-training-submit" type="button" disabled={Boolean(t.block)} title={t.block||"Train with approved labels"} onClick={t.train}>Train reasoning model</button>
        {t.error&&<p role="alert">{t.error}</p>}
        {t.starting&&<section className="learn-training-job" aria-label="Reasoning training preparation">
          <strong>{t.name.trim()}</strong><span role="status">Preparing training data</span>
          <progress aria-label="Preparing training data"/>
        </section>}
        {job?.status&&job.status!=="idle"&&<section className="learn-training-job" aria-label="Reasoning training job">
          <div className="learn-training-job-heading"><strong>{job.model_name}</strong></div>
          <span role="status">{job.status.replaceAll("_"," ")}{job.steps?` · ${job.step||0}/${job.steps} steps`:""}</span>
          {job.steps>0&&<progress aria-label="Training steps completed" max={job.steps} value={Math.min(job.step||0,job.steps)}/>}
          {job.phase&&<small>{job.phase}</small>}
          {job.error&&<p role="alert">{job.error}</p>}
        </section>}
        {!!t.models.length&&<section className="learn-trained-models" aria-label="Completed reasoning models">
          <label>Reasoning model<div className="learn-model-picker">
            <select value={t.selectedModel?.id||""} disabled={t.locked} onChange={e=>t.setSelectedModelId(e.target.value)}>
              {t.models.map(model=><option key={model.id} value={model.id}>{model.model_name} · {model.sample_count} labels</option>)}
            </select>
            <button type="button" className="learn-model-delete" aria-label="Delete reasoning model" title="Delete selected reasoning model"
              disabled={t.locked||!t.selectedModel} onClick={()=>setDeleteId(t.selectedModel.id)}><MdDeleteOutline size={19}/></button>
          </div></label>
          {deleteId===t.selectedModel?.id&&<div className="learn-model-delete-confirm" role="group" aria-label="Delete reasoning model confirmation">
            <span>Delete {t.selectedModel.model_name}?</span>
            <button type="button" disabled={t.locked} onClick={async()=>{if(await t.deleteModel())setDeleteId(null);}}>Delete</button>
            <button type="button" disabled={t.working} onClick={()=>setDeleteId(null)}>Cancel</button>
          </div>}
          <small>Available in Operate.</small>
          {onTestModel&&<button type="button" disabled={t.locked||!t.selectedModel} onClick={()=>onTestModel(t.selectedModel)}>Test in Observation</button>}
        </section>}
      </div>
    </section>
  </div>;
}
