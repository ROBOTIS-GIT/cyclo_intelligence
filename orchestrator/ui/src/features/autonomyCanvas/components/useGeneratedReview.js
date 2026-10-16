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

// A review session follows generated drafts across saved episodes on one map.
export default function useGeneratedReview({draft,kind,scope,disabled=false}) {
  const [busy,setBusy]=useState(false), [error,setError]=useState("");
  const [session,setSession]=useState(null);
  const loading=useRef(false), exhausted=useRef(new Set()), latest=useRef(null);
  const map=session?.map||draft.episode?.map_name||draft.mapName;
  const field=`${kind}_pending_review_count`;
  const episodes=(draft.episodes||[]).filter(episode=>episode.ended_at
    && (!map||episode.map_name===map) && !exhausted.current.has(episode.id)
    && (!session||session.ids.includes(episode.id)));
  const remaining=episodes.reduce((total,episode)=>total+(episode.id===draft.episode?.id
    ?scope.pendingReviewCount:episode[field]||0),0)
    + (!episodes.some(episode=>episode.id===draft.episode?.id)
       && (!session||session.ids.includes(draft.episode?.id)) ? scope.pendingReviewCount : 0);
  const pendingReviewCount=remaining;
  async function openEpisode(id) {
    if(!draft.load||loading.current)return false;
    loading.current=true;setBusy(true);
    try {
      if(await draft.load(id)===false)throw new Error("Could not open the next review episode. Exit Review and retry.");
      scope.setFrameScope("review",null);
      return true;
    } catch(err){setError(err.message);return false;}
    finally{loading.current=false;setBusy(false);}
  }
  async function changeScope(next) {
    setError("");
    if(next!=="review") {
      exhausted.current.clear();setSession(null);scope.setFrameScope(next);return;
    }
    if(loading.current||disabled)return;
    exhausted.current.clear();
    const saved=(draft.episodes||[]).filter(episode=>episode.ended_at
      && (!map||episode.map_name===map) && (episode.id===draft.episode?.id?scope.pendingReviewCount:episode[field])>0);
    if(scope.pendingReviewCount&&!saved.some(episode=>episode.id===draft.episode?.id))saved.unshift(draft.episode);
    const first=scope.pendingReviewCount?draft.episode:saved[0];
    if(!first)return;
    const queue=saved.filter(episode=>episode.map_name===first.map_name);
    const total=queue.reduce((count,episode)=>count+(episode.id===draft.episode?.id?scope.pendingReviewCount:episode[field]||0),0);
    setSession({map:first.map_name,ids:queue.map(episode=>episode.id),total});
    if(first.id===draft.episode?.id)scope.setFrameScope("review",null);
    else if(!await openEpisode(first.id))setSession(null);
  }
  latest.current={draft,scope,episodes,openEpisode};
  useEffect(()=>{
    if(!session||busy||loading.current||disabled||error)return;
    const current=latest.current;
    if(current.scope.frameScope!=="review"){setSession(null);return;}
    if(current.draft.episode?.map_name!==session.map) {
      setSession(null);exhausted.current.clear();current.scope.setFrameScope("all");return;
    }
    if(current.scope.pendingReviewCount)return;
    exhausted.current.add(current.draft.episode.id);
    const next=session.ids.find(id=>!exhausted.current.has(id)
      && current.episodes.some(episode=>episode.id===id&&episode[field]>0));
    if(next)current.openEpisode(next);
  },[session,busy,disabled,error,field,scope.frameScope,scope.pendingReviewCount,draft.episode?.id,draft.episode?.map_name,draft.episodes]);
  const total=session?Math.max(session.total,remaining):0;
  return {busy,error,pendingReviewCount,changeScope,
    progress:session&&scope.frameScope==="review"?`${total-remaining}/${total} reviewed`:null};
}
