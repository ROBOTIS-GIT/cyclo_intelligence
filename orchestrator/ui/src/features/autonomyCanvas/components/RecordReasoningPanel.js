import OperateModelSelect from "./OperateModelSelect";
import OperateObservation from "./OperateObservation";
import CollectHeadCamera from "./CollectHeadCamera";
import CollectAttentionPreview from "./CollectAttentionPreview";
import {resultTask} from "./autonomyPresentation";
import "./RecordReasoningPanel.css";

export default function RecordReasoningPanel({observation={}, disabled=false}) {
  const preview={...observation,reasoning_task:resultTask(observation.result)};
  return <section className="record-reasoning-panel" aria-label="Record reasoning">
    <div className="record-reasoning-controls">
      <OperateModelSelect choice={observation.modelChoice} loadedModel={observation.loadedModel} active={observation.active}/>
      <div className="record-reasoning-buttons">
        <button type="button" onClick={observation.start}
          disabled={!observation.available||disabled||observation.active||observation.phase==="stopping"}>Start reasoning</button>
        <button type="button" onClick={observation.stop}
          disabled={!observation.needsStop||observation.phase==="stopping"}>Stop reasoning</button>
      </div>
    </div>
    <OperateObservation preview observation={preview} imageContent={<div className="record-reasoning-images">
      <figure className="record-reasoning-live" aria-label="Live camera">
        <figcaption>Live</figcaption>
        <CollectHeadCamera topic={observation.result?.camera_topic}/>
      </figure>
      <CollectAttentionPreview result={observation.result}/>
    </div>}/>
  </section>;
}
