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

export default function CollectAttentionPreview({result}) {
  const attention=result?.attention_preview;
  const image=attention?.image_base64 || result?.image_base64;
  return <figure className="collect-attention-preview" aria-label="Reasoning attention">
    <figcaption title={attention ? `LLM layer ${attention.layer} · Last prompt query to image · Fixed scale` : undefined}>Attention</figcaption>
    {image ? <img src={`data:image/jpeg;base64,${image}`}
      alt={attention?.image_base64 ? "Attention over the analyzed camera frame" : "Analyzed camera frame"}/>
      : <div className="collect-attention-placeholder">Waiting for reasoning</div>}
    {result?.attention_error && <span role="status" title={result.attention_error}>Attention unavailable for this result.</span>}
  </figure>;
}
