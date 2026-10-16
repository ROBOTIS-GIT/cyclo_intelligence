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
import {collectionRequest} from "../../../utils/collectionApi";

export default function TrainingBalance({mapName,episodeIds,count,enabled,onChange,disabled,onReady,compact=false}) {
  const [result,setResult]=useState(null), [error,setError]=useState("");
  const ids=JSON.stringify(episodeIds);
  useEffect(()=>{
    let alive=true;
    setResult(null);setError("");onReady(false);
    if(mapName&&JSON.parse(ids).length){
      collectionRequest("qwen/preview-reasoning",{map_name:mapName,episode_ids:JSON.parse(ids),balance_decisions:enabled})
        .then(value=>{if(alive){setResult(value);onReady(Boolean(value.after&&!value.error));}})
        .catch(err=>{if(alive)setError(err.message);});
    }
    return()=>{alive=false;};
  },[mapName,ids,count,enabled,onReady]);
  return <div>
    <label style={{display:"flex",alignItems:"center",gap:6}}>
      <input style={{width:"auto",margin:0}} type="checkbox" checked={enabled} disabled={disabled}
        onChange={event=>{onReady(false);onChange(event.target.checked);}}/>Balance decisions per type
    </label>
    {compact ? <details className="training-balance-details"><summary>Label distribution</summary>
      <BalanceDistribution result={result}/>
    </details> : <BalanceDistribution result={result}/>}
    {result?.after&&<small>{result.after.total} training labels · {result.excluded || 0} excluded</small>}
    {result?.warnings?.map(warning=><div key={warning}><small>{warning}</small></div>)}
    {(error||result?.error)&&<small role="alert">{error||result.error}</small>}
  </div>;
}

function BalanceDistribution({result}) {
  return result?.before?.tasks ? <div className="reasoning-balance-summary">
    {Object.entries(result.before.tasks).map(([task,before])=><div key={task}>
      <strong>{TASK_LABELS[task]||task}</strong>
      <div><small>Available: {decisionCounts(before)}</small></div>
      <div><small>Training: {decisionCounts(result.after?.tasks?.[task])}</small></div>
    </div>)}
  </div> : <>
    {result?.before&&<small>Available: {decisionCounts(result.before)}</small>}
    {result?.after&&<div><small>Training: {decisionCounts(result.after)}</small></div>}
  </>;
}

const TASK_LABELS = {mission_selection:"Observation",patrol_supervision:"Patrol",mission_supervision:"Mission"};
function decisionCounts(summary) {
  const decisions = summary?.decisions || {};
  return Object.entries(decisions).map(([decision,count]) => `${count} ${decision}`).join(" · ") || "0 labels";
}
