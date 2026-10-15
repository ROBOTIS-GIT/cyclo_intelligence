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

// Mission-level language guidance; execution is handled separately by the runtime.
export const MISSION_CONDITION_FIELDS = [
  { key: "start_condition", label: "Start condition", placeholder: "Describe when this mission may start." },
  { key: "completion_condition", label: "Completion condition", placeholder: "Describe the observable result that confirms completion." },
  { key: "stop_condition", label: "Stop condition", placeholder: "Describe when this mission should be interrupted." },
  { key: "post_stop_action", label: "Post-stop action", placeholder: "Describe what should happen after interruption." },
];
