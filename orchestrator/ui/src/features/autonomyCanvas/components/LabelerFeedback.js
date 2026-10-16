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

import { MdClose } from "react-icons/md";
import { jobRunning } from "./trainingJobStatus";

export default function LabelerFeedback({ family, workflow }) {
  const { state, error, connectionError, working, running, dismissError } = workflow;
  return <>
    {state?.training?.status === "cancelled" && <small role="status">Fine-tuning cancelled</small>}
    {connectionError && <p role="alert">{connectionError}</p>}
    {error && <p role="alert">{error}</p>}
    {["training", "batch", "reasoning_training"].filter(kind => state?.[kind]?.error && !jobRunning(state[kind])).map(kind =>
      <div key={state[kind].id || kind} className="collect-actions">
        <p role="alert">{family === "qwen" ? state[kind].error.split("\n")[0].slice(0, 200) : state[kind].error}</p>
        {family === "qwen" && <button type="button" title="Clear error" aria-label={`Clear Qwen ${kind.replaceAll("_", " ")} error`}
          disabled={working || running} onClick={() => dismissError(kind)}><MdClose size={18}/></button>}
      </div>)}
  </>;
}
