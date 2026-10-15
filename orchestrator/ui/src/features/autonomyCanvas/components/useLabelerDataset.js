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
