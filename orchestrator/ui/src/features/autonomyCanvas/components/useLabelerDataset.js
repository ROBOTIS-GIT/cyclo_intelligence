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

// A model workflow owns its map independently of the frame currently being edited.
export default function useLabelerDataset({episode,episodes=[],models=[]}) {
  const [chosenMap,setMap]=useState(episode?.map_name||"");
  useEffect(()=>{if(episode?.map_name)setMap(episode.map_name);},[episode?.map_name]);
  const saved=episodes.filter(ep=>ep.ended_at).map(ep=>ep.id===episode?.id?{...ep,...episode}:ep);
  if(episode?.ended_at&&!saved.some(ep=>ep.id===episode.id))saved.push(episode);
  const maps=[...new Set([...saved.map(ep=>ep.map_name),...models.map(model=>model.map_name)].filter(Boolean))].sort();
  const mapName=maps.includes(chosenMap)?chosenMap:[...new Set(saved.map(ep=>ep.map_name).filter(Boolean))].sort()[0]||maps[0]||"";
  return {maps,mapName,setMap,saved,mapEpisodes:saved.filter(ep=>ep.map_name===mapName),
    selectedEpisode:episode?.ended_at&&episode.map_name===mapName?episode:null};
}
