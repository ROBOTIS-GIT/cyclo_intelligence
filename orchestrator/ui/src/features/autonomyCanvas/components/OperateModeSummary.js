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

import { FiEye, FiNavigation, FiPlay, FiArrowRight } from "react-icons/fi";
import { operateReasoningView } from "./operateReasoningView";

const modes = [{id: "patrol", label: "Patrol", Icon: FiNavigation},
  {id: "observation", label: "Observation", Icon: FiEye}, {id: "mission", label: "Mission", Icon: FiPlay}];

export default function OperateModeSummary({ observation }) {
  const view = operateReasoningView(observation);
  return <div className="operate-mode-summary">
    <div className="operate-mode-heading">
      <ol className="operate-mode-flow" aria-label="Autonomy modes">
        {modes.map(({id, label, Icon}, index) => <li key={id} aria-current={view.mode === id ? "step" : undefined}
          className={`operate-mode-item mode-${id}${view.mode === id ? " is-current" : ""}`}>
          {index > 0 && <FiArrowRight className="operate-mode-arrow" aria-hidden="true"/>}
          <span><Icon aria-hidden="true"/><strong>{label}</strong></span>
        </li>)}
      </ol>
      <span className={`operate-mode-status tone-${view.tone}`} role="status">{view.status}</span>
    </div>
    <div className="operate-mode-location">{view.detail || " "}</div>
  </div>;
}
