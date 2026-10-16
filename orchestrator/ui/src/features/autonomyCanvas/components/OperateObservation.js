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

import { FiCamera } from "react-icons/fi";
import "./OperateObservation.css";
import OperateModeSummary from "./OperateModeSummary";
import { decisionLabel, operateReasoningView } from "./operateReasoningView";
import { autonomyPhaseLabel, displayedReasoningResult, resultMissionName } from "./autonomyPresentation";

export default function OperateObservation({ observation = {}, imageContent, preview = false }) {
  const result = displayedReasoningResult(observation);
  const missionName = resultMissionName(result);
  const paused = preview ? observation.paused : operateReasoningView(observation).status === "Paused";
  return <section className="operate-map-observation" aria-label="Head camera and reasoning">
        <div className="operate-observation">
          {imageContent || (result?.image_base64 ? <img className="operate-head-image" src={`data:image/jpeg;base64,${result.image_base64}`} alt="Head camera frame analyzed by Qwen" /> : <><FiCamera size={21} /><strong>Head camera</strong><span>Run autonomy to start reasoning.</span></>)}
        </div>
        <div className="operate-map-reasoning">
        <>{preview ? <div className="operate-qwen-heading"><strong>Qwen · {autonomyPhaseLabel(observation)}</strong></div> : <OperateModeSummary observation={observation}/>}</>
        {observation.error && <p role="alert" className="operate-qwen-error">{observation.error}</p>}
        {paused && observation.pause_reason && <p className="operate-sidebar-note">Pause reason: {observation.pause_reason === "ai" ? "AI" : observation.pause_reason}</p>}
        {!result && observation.active && <p className="operate-sidebar-note">Waiting for the next reasoning result.</p>}
        {result && <>
          {result.decision && <div className={`operate-map-decision${preview ? "" : " operate-decision-card"}`} data-decision={result.decision.decision}><span>AI decision</span><strong>{preview ? result.decision.decision : decisionLabel(result.decision.decision)}{missionName !== "None" ? ` · ${missionName}` : ""}</strong></div>}
          <p className="operate-qwen-text" aria-live="polite">{result.decision?.reason || result.text || "No valid decision returned."}</p>
          {result.block_reason && <p className="operate-sidebar-note">{result.block_reason}</p>}
          {preview && <p className="operate-sidebar-note">{result.model_name || result.model}{result.adapter ? " · LoRA" : ""} · {Number(result.elapsed_seconds).toFixed(1)}s</p>}
        </>}
        </div>
  </section>;
}
