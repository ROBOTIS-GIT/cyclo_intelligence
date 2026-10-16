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

import { EPISODE_TYPES, TYPE_LABELS } from "../utils/reasoningDataset";

export default function ReasoningTypeScope({value,onChange,counts={},disabled}) {
  return <fieldset className="reasoning-type-scope" disabled={disabled}>
    <legend>Training types</legend>
    {EPISODE_TYPES.map(type=><label key={type}>
      <input type="checkbox" checked={value.includes(type)} onChange={event=>onChange(previous=>event.target.checked
        ? EPISODE_TYPES.filter(item=>item===type||previous.includes(item)) : previous.filter(item=>item!==type))}/>
      <span>{TYPE_LABELS[type]}</span><small>{counts[type]||0} labels</small>
    </label>)}
  </fieldset>;
}
