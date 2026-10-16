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

import { MdMyLocation, MdPlayArrow, MdStop } from "react-icons/md";

// Visual contract only: each feature owns command eligibility and lifecycle.
export default function RuntimeControlsHud({ localize, start, stop, leading }) {
  return (
    <div className="absolute top-5 left-5 z-10 flex items-center gap-2 p-2"
      style={{ borderRadius: 14, backgroundColor: "color-mix(in srgb, var(--mc-surface) 88%, transparent)", border: "1px solid var(--mc-border)", boxShadow: "var(--mc-shadow)", backdropFilter: "blur(8px)" }}>
      {leading}
      <button type="button" onClick={localize.onClick} disabled={localize.disabled}
        aria-label="Localize" aria-pressed={localize.pressed || undefined}
        title="Bring navigation up and set the robot pose"
        className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
        style={{ borderRadius: 9, border: `1px solid ${localize.pressed ? "var(--mc-accent)" : "var(--mc-border-strong)"}`,
          backgroundColor: localize.pressed ? "var(--mc-accent-soft)" : "var(--mc-surface)", color: "var(--mc-text)" }}>
        <MdMyLocation size={17} aria-hidden="true" />
      </button>
      <button type="button" onClick={start.onClick} disabled={start.disabled}
        aria-label={start.label} aria-pressed={start.pressed || undefined} title={start.title}
        className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
        style={{ borderRadius: 9, border: "none", backgroundColor: "var(--mc-success)", color: "var(--mc-accent-fg)" }}>
        <MdPlayArrow size={19} aria-hidden="true" />
      </button>
      <button type="button" onClick={stop.onClick} disabled={stop.disabled}
        aria-label="Stop" aria-pressed={stop.pressed || undefined} title={stop.title}
        className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
        style={{ borderRadius: 9, border: "1px solid var(--mc-danger-border)",
          backgroundColor: stop.pressed ? "var(--mc-danger)" : "var(--mc-surface)",
          color: stop.pressed ? "var(--mc-accent-fg)" : "var(--mc-danger)" }}>
        <MdStop size={18} aria-hidden="true" />
      </button>
    </div>
  );
}
