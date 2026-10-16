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
import {collectionRequest} from "../../../utils/collectionApi";
import {approvedReasoningCount, episodeCategory, EPISODE_TYPES} from "../utils/reasoningDataset";
import {jobRunning} from "./LabelerActions";

// Final Operate-model training is independent of the current label tool and frame filter.
export default function useReasoningTraining({episode, episodes=[], disabled=false}) {
  const [state,setState]=useState(null), [error,setError]=useState(""), [connectionError,setConnectionError]=useState("");
  const [starting,setStarting]=useState(false);
  const [working,setWorking]=useState(false), requestVersion=useRef(0), submitting=useRef(false);
  const [map,setMap]=useState(episode?.map_name||"");
  const [name,setName]=useState(""), [steps,setSteps]=useState("2000");
  const [baseModel,setBaseModel]=useState("Qwen/Qwen3.5-4B"), [types,setTypes]=useState(EPISODE_TYPES);
  const [balance,setBalance]=useState(false), [balanceReady,setBalanceReady]=useState(false);
  const [selectedModelId,setSelectedModelId]=useState("");
  useEffect(()=>{if(episode?.map_name)setMap(episode.map_name);},[episode?.map_name]);
  useEffect(()=>{
    let alive=true, timer;
    async function poll() {
      const version=requestVersion.current;
      try {
        const next=await collectionRequest("qwen/workflow");
        if(alive && version===requestVersion.current){setState(next);setConnectionError("");}
      } catch(err) {if(alive)setConnectionError(err.message);}
      if(alive)timer=setTimeout(poll,3000);
    }
    poll();
    return()=>{alive=false;clearTimeout(timer);};
  },[]);
  const saved=episodes.map(ep=>ep.id===episode?.id ? episode : ep);
  if(episode?.ended_at && !saved.some(ep=>ep.id===episode.id))saved.push(episode);
  const ended=saved.filter(ep=>ep.ended_at&&ep.map_name);
  const maps=[...new Set(ended.map(ep=>ep.map_name))].sort();
  const mapName=maps.includes(map) ? map : maps[0]||"";
  const eligible=ended.filter(ep=>ep.map_name===mapName && approvedReasoningCount(ep)>0);
  const prepared=eligible.filter(ep=>types.includes(episodeCategory(ep)));
  const episodeIds=prepared.map(ep=>ep.id);
  const count=prepared.reduce((sum,ep)=>sum+approvedReasoningCount(ep),0);
  const typeCounts=Object.fromEntries(EPISODE_TYPES.map(type=>[type,eligible.filter(ep=>episodeCategory(ep)===type)
    .reduce((sum,ep)=>sum+approvedReasoningCount(ep),0)]));
  const job=state?.reasoning_training;
  const running=[state?.training,state?.batch,job].some(jobRunning);
  const locked=disabled||working||running;
  const models=(state?.models||[]).filter(model=>model.purpose==="reasoning" && model.map_name===mapName &&
    (!model.status||model.status==="completed")).sort((a,b)=>(b.created_at||0)-(a.created_at||0));
  const selectedModel=models.find(model=>model.id===selectedModelId)||models[0];
  const stepCount=Number(steps);
  const validSteps=steps.trim()!=="" && Number.isSafeInteger(stepCount) && stepCount>=1;
  const duplicate=(state?.models||[]).some(model=>model.map_name===mapName && model.model_name?.toLowerCase()===name.trim().toLowerCase());
  const block=locked ? "Finish the current edit or job before training." : !state || connectionError ? "Waiting for training service."
    : !mapName || count<2 ? "At least 2 approved reasoning labels are required."
    : !name.trim() ? "Enter a model name." : duplicate ? "This model name already exists on this map."
    : !validSteps ? "Enter a positive whole number of training steps." : balance && !balanceReady ? "Waiting for data balance preview." : "";
  async function train() {
    if(block || submitting.current)return;
    submitting.current=true;requestVersion.current+=1;setWorking(true);setStarting(true);setError("");
    try {
      const result=await collectionRequest("qwen/train-reasoning",{map_name:mapName,episode_ids:episodeIds,
        model_name:name.trim(),base_model:baseModel,steps:stepCount,batch_size:8,...(balance?{balance_decisions:true}:{})});
      requestVersion.current+=1;
      setState(previous=>({...previous,reasoning_training:result}));
    } catch(err) {setError(err.message);}
    finally {submitting.current=false;setWorking(false);setStarting(false);}
  }
  async function deleteModel() {
    if(locked || !selectedModel || submitting.current)return false;
    submitting.current=true;requestVersion.current+=1;setWorking(true);setError("");
    try {
      await collectionRequest(`qwen/models/${selectedModel.id}`,undefined,"DELETE");
      requestVersion.current+=1;
      setState(await collectionRequest("qwen/workflow"));
      setSelectedModelId("");
      return true;
    } catch(err) {setError(err.message);return false;}
    finally {submitting.current=false;setWorking(false);}
  }
  async function cancel() {
    if(!jobRunning(job)||!job?.id||submitting.current)return;
    submitting.current=true;requestVersion.current+=1;setWorking(true);setError("");
    try {
      const next=await collectionRequest("qwen/jobs/reasoning_training/cancel",{job_id:job.id});
      requestVersion.current+=1;setState(next);
    } catch(err) {setError(err.message);}
    finally {submitting.current=false;setWorking(false);}
  }
  return {mapName,maps,setMap,name,setName,steps,setSteps,baseModel,setBaseModel,types,setTypes,
    balance,setBalance,balanceReady,setBalanceReady,models,selectedModel,setSelectedModelId,
    count,episodeIds,typeCounts,job,locked,working,starting,running,block,train,cancel,deleteModel,error: error||connectionError};
}
