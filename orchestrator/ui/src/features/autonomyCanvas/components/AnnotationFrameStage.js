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

// Shared frame stage for reasoning labels and object masks.
export default function AnnotationFrameStage({ episode, controls, tools, footer, children, compact = false }) {
  const name = episode?.storage_path?.split("/").filter(Boolean).pop() || episode?.scenario || "Select an episode";
  return <section className={`annotate-frame-stage${compact ? " annotate-frame-stage-compact" : ""}`} aria-label="Frame preview">
    <header className="annotate-frame-heading">
      <div><strong>{name}</strong><span>{episode?.map_name || "Saved dataset"}</span></div>
      {compact && <div className="annotate-frame-navigation">{controls}</div>}
    </header>
    {!compact && <div className="annotate-frame-navigation">{controls}</div>}
    {tools && <div className="annotate-frame-tools">{tools}</div>}
    <div className="annotate-frame-viewport">{children}</div>
    {footer && <div className="annotate-frame-footer">{footer}</div>}
  </section>;
}
