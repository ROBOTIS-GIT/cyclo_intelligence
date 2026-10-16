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

import {useState} from "react";
import {MdDeleteOutline} from "react-icons/md";
import {collectionRequest} from "../../../utils/collectionApi";

export default function ReasoningModelDelete({model, disabled, onDeleted}) {
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  const removable=model && !model.id.startsWith("base:") && !model.id.startsWith("legacy:");
  async function remove() {
    if(!removable || disabled || busy)return;
    if(!window.confirm(`Delete “${model.model_name}”?\nThis permanently removes the saved model files. Dataset labels are kept.`))return;
    setBusy(true);setError("");
    try {
      await collectionRequest(`qwen/models/${encodeURIComponent(model.id)}`,undefined,"DELETE");
      await onDeleted?.();
    } catch(err) {setError(err.message);}
    finally {setBusy(false);}
  }
  return <>
    <button type="button" aria-label="Delete selected reasoning model"
      title={removable?"Delete selected reasoning model":"Select a trained model to delete"}
      disabled={!removable||disabled||busy} onClick={remove}><MdDeleteOutline/></button>
    {error&&<span role="alert" style={{fontSize:11,color:"var(--mc-danger, #bd4545)"}}>{error}</span>}
  </>;
}
