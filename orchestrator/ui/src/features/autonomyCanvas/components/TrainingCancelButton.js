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

import {MdStop} from "react-icons/md";
import {jobRunning} from "./trainingJobStatus";

export default function TrainingCancelButton({job, onCancel, pending=false, label="Cancel fine-tuning",showLabel=false}) {
  if(!onCancel || !jobRunning(job))return null;
  const cancelling=pending || (job.status==="cancelling" && !job.error);
  return <button type="button" className="training-cancel-button" aria-label={label}
    title={cancelling?"Cancelling…":label} aria-busy={cancelling} disabled={cancelling} onClick={onCancel}>
    <MdStop size={19}/>{showLabel&&<span>{cancelling?"Cancelling…":label}</span>}
  </button>;
}
