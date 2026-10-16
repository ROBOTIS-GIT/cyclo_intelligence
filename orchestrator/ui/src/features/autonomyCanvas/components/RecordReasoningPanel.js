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
