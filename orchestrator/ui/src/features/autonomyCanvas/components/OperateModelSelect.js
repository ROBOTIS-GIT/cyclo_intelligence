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

import ReasoningModelDelete from "./ReasoningModelDelete";
import { FiRefreshCw } from "react-icons/fi";

export default function OperateModelSelect({ choice, loadedModel, active }) {
  if (!choice) return null;
  const selected = choice.selected;
  const applied = active && loadedModel?.selected_model_id === selected?.id;
  const designChanged = selected?.catalog_compatible === false;
  return <div className="operate-model-select">
    <label htmlFor="operate-reasoning-model">Reasoning model</label>
    <div className="operate-model-controls">
      <select id="operate-reasoning-model" title={selected?.model_name || "Select a reasoning model"} value={selected?.id || ""}
        disabled={choice.locked || choice.loading || !choice.models.length}
        onChange={event => choice.select(event.target.value)}>
        <option value="">{choice.loading ? "Loading models…" : "Select a reasoning model"}</option>
        {choice.models.map(model => <option key={model.id} value={model.id}>
          {model.model_name}{model.source === "base" ? " (Base · Untuned)" : model.source === "legacy" ? " (Legacy)" : ""}{model.catalog_compatible === false ? " (Design changed)" : ""}
        </option>)}
      </select>
      <button type="button" aria-label="Refresh reasoning models" title="Refresh models"
        disabled={choice.locked || choice.loading} onClick={choice.refresh}><FiRefreshCw /></button>
      <ReasoningModelDelete key={selected?.id || "none"} model={selected} disabled={choice.locked || choice.loading || active} onDeleted={choice.refresh}/>
    </div>
    {choice.error ? <p role="alert" className="operate-qwen-error">{choice.error}</p> :
      <p className="operate-model-status" role="status">{selected
        ? designChanged
          ? `${applied ? "Loaded" : "Selected"} · Design changed since training. Retrain for the current missions.`
          : selected.source === "base" ? `${applied ? "Loaded" : "Selected"} · ${selected.base_model} · Untuned base model`
          : `${applied ? "Loaded" : "Selected"} · ${selected.base_model} · ${selected.sample_count ?? "—"} samples`
        : choice.loading ? "Loading available models…" : choice.models.length
          ? "Select a model, then run autonomy." : "No reasoning models for this map."}</p>}
  </div>;
}
