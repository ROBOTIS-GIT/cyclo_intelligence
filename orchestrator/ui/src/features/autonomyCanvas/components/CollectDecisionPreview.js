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

import { resultMissionName, resultTask } from "./autonomyPresentation";
import React from "react";

// Presentation only: this component never loads, starts, or cancels a mission.
export default function CollectDecisionPreview({ result, phase, error }) {
  const decision = result?.decision;
  return <div className="collect-observation-description" aria-label="Mission decision preview">
    {error ? <p role="alert" className="operate-qwen-error">{error}</p> : decision ? <>
      <dl className="collect-decision-fields">
        <div><dt>{resultTask(result) !== "mission_selection" ? "Supervision" : "Decision"}</dt><dd>{decision.decision}</dd></div>
        <div><dt>Mission</dt><dd>{resultMissionName(result)}</dd></div>
      </dl>
      <p className="operate-qwen-text">{decision.reason}</p>
    </> : <p className="operate-sidebar-note">{result ? "No valid mission decision for this frame." : phase}</p>}
  </div>;
}
