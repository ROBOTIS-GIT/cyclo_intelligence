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

import TrainingCancelButton from "./TrainingCancelButton";
import {MdSchool} from "react-icons/md";
import {jobRunning} from "./LabelerActions";

export default function ReasoningTrainingButton({training,open,onClick}) {
  const job=training.job, running=jobRunning(job);
  const failed=["failed","completed_with_errors"].includes(job?.status)||Boolean(training.error);
  const percent=Math.min(100,Math.round((job?.step||0)/Math.max(1,job?.steps||0)*100));
  return <div className="learn-training-launcher">
    <span className="learn-training-map" title={training.mapName}>{training.mapName}</span>
    <button type="button" className={`learn-training-button${failed?" is-failed":""}`}
      aria-label="Train model" aria-expanded={open} aria-haspopup="dialog"
      title={running?`${job.model_name||"Reasoning model"} · ${job.step||0}/${job.steps||"…"} steps`:failed?"View training error":"Train reasoning model"}
      onClick={onClick}>
      {running && <span className="learn-training-fill" style={{height:`${percent}%`}} aria-hidden="true"/>}
      <MdSchool size={20}/>
      {running && <span className="learn-training-percent" role="progressbar" aria-label="Reasoning training progress"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={job.steps ? percent : undefined}>{job.steps?`${percent}%`:"…"}</span>}
    </button>
    <TrainingCancelButton label="Cancel model training" job={job} onCancel={training.cancel} pending={training.working}/>
  </div>;
}
