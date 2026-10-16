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

import {useEffect, useRef, useState} from "react";
import LabelerDatasetOptions from "./LabelerDatasetOptions";
import TrainingCancelButton from "./TrainingCancelButton";
import {MdAutoFixHigh, MdModelTraining, MdDeleteOutline, MdClose, MdPlayArrow} from "react-icons/md";

import {jobRunning} from "./trainingJobStatus";
export {jobRunning} from "./trainingJobStatus";

function JobButton({label, icon:Icon, job, training=false, disabled, onClick, expanded}) {
  const running=jobRunning(job), failed=job?.status==="failed" || job?.status==="completed_with_errors";
  const total=training?job?.steps:job?.total, done=training?job?.step:job?.processed;
  const preparing=running && !done;
  const percent=Math.min(100, Math.max(0, Math.round((done||0)/Math.max(1,total||0)*100)));
  const detail=job?.model_name ? `${job.model_name} · ${job.status}${running?` · ${percent}%`:""}` : "";
  return <button type="button" className={`labeler-job-button${preparing?" is-preparing":""}${failed?" is-failed":""}`}
    aria-label={label} aria-busy={running} aria-expanded={expanded} title={`${label}${detail?` — ${detail}`:""}`}
    disabled={disabled} onClick={onClick}>
    <span className="labeler-job-fill" style={{height:`${percent}%`}} aria-hidden="true"/>
    <Icon size={19}/>
    {running && <span className="labeler-job-percent" role="progressbar" aria-label={`${label} progress`}
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={preparing?undefined:percent}>{preparing?"…":`${percent}%`}</span>}
  </button>;
}

export default function LabelerActions({models, selected, onSelect, onTrain, onGenerate, onDelete, trainingJob, batchJob, onCancelTrain, cancelling=false, disabled, canTrain, canGenerate, episodeId, maps=[], mapName, onMapChange, trainingHint, trainingSummary, generationTarget="episode", onTargetChange}) {
  const [open,setOpen]=useState(null), [confirm,setConfirm]=useState(null);
  const root=useRef(null);
  useEffect(()=>{setConfirm(null);},[episodeId]);
  useEffect(()=>{setConfirm(null);},[selected?.id]);
  useEffect(()=>{
    if(!open)return;
    const outside=e=>{if(!root.current?.contains(e.target))setOpen(null);};
    const escape=e=>{if(e.key==="Escape")setOpen(null);};
    document.addEventListener("pointerdown",outside);document.addEventListener("keydown",escape);
    return()=>{document.removeEventListener("pointerdown",outside);document.removeEventListener("keydown",escape);};
  },[open]);
  const trainingRunning=jobRunning(trainingJob);
  const trainingFailed=["failed","completed_with_errors"].includes(trainingJob?.status);
  const inspectTraining=trainingRunning || trainingFailed;
  return <div className="labeler-actions" ref={root}>
    <JobButton label="Fine-tune" icon={MdModelTraining} training job={trainingJob}
      expanded={open==="training"}
      onClick={()=>inspectTraining||disabled||!canTrain||!episodeId ? setOpen(value=>value==="training" ? null : "training") : onTrain()}/>
    <TrainingCancelButton job={trainingJob} onCancel={onCancelTrain} pending={cancelling}/>
    <JobButton label="Auto-label" icon={MdAutoFixHigh} job={batchJob}
      expanded={open==="auto"} onClick={()=>setOpen(value=>value==="auto" ? null : "auto")}/>
    {open==="training" && <section className="labeler-model-popover labeler-training-details" aria-label="Fine-tuning details">
      <header><strong>Fine-tuning</strong><button type="button" aria-label="Close training details" onClick={()=>setOpen(null)}><MdClose/></button></header>
      {inspectTraining&&<>
        {trainingJob?.model_name && <strong>{trainingJob.model_name}</strong>}
        <span>{trainingJob?.status?.replaceAll("_"," ")}{trainingJob?.steps ? ` · ${trainingJob.step||0}/${trainingJob.steps} steps` : ""}</span>
        <progress aria-label="Training steps" max={trainingJob?.steps||1} value={trainingJob?.steps ? trainingJob.step||0 : undefined}/>
        {trainingJob?.phase && <small>{trainingJob.phase}</small>}
        {trainingJob?.error && <p role="alert">{trainingJob.error}</p>}
      </>}
      {!trainingRunning&&<>
        {onMapChange&&<LabelerDatasetOptions kind="Fine-tuning" {...{maps,mapName,onMapChange,disabled}}/>}
        {trainingSummary&&<small>{trainingSummary}</small>}
        <button type="button" disabled={disabled||!canTrain} onClick={()=>{onTrain();setOpen(null);}}>{trainingFailed?"Retry fine-tuning":"Start fine-tuning"}</button>
        {trainingHint&&<small role="status">{trainingHint}</small>}
      </>}
    </section>}
    {open==="auto" && <section className="labeler-model-popover" aria-label="Auto-label models">
      <header><strong>Auto-label</strong><button type="button" title="Close" aria-label="Close model picker" onClick={()=>setOpen(null)}><MdClose/></button></header>
      {onMapChange&&<LabelerDatasetOptions kind="Auto-label" {...{maps,mapName,onMapChange,disabled}}
        target={generationTarget} onTargetChange={onTargetChange} hasSelectedEpisode={!!episodeId}/>}
      <div className="labeler-model-list">
        {!models.length && <small>No trained model for this map</small>}
        {models.map(model=><div className="labeler-model-row" key={model.id}>
          <button type="button" disabled={disabled} aria-pressed={selected?.id===model.id} onClick={()=>onSelect(model.id)}>{model.model_name}</button>
          <button type="button" disabled={disabled} title={`Delete ${model.model_name}`} aria-label={`Delete ${model.model_name}`} onClick={()=>setConfirm(model.id)}><MdDeleteOutline size={18}/></button>
          {confirm===model.id && <div className="labeler-delete-confirm"><span>Delete this model?</span>
            <button type="button" disabled={disabled} onClick={async()=>{await onDelete(model.id);setConfirm(null);}}>Delete</button>
            <button type="button" onClick={()=>setConfirm(null)}>Cancel</button></div>}
        </div>)}
      </div>
      <button type="button" className="labeler-generate" disabled={disabled||!selected||!canGenerate} onClick={()=>{onGenerate();setOpen(null);}}><MdPlayArrow size={19}/>{generationTarget==="map"?"Auto-label map":"Auto-label episode"}</button>
    </section>}
  </div>;
}
