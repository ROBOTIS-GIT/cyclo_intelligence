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

import { MdAccountTree, MdRoute } from "react-icons/md";
import { MISSION_BORDER, MISSION_CARD_RADIUS, MISSION_TEXT } from "../../features/missionCanvas/lib/theme";

export function WorkspaceChoiceCard({
  ariaLabel,
  examples,
  icon,
  onClick,
  title,
}) {
  return (
    <button
      type="button"
      aria-label={ariaLabel}
      onClick={onClick}
      className="group flex min-h-[230px] w-full flex-col border bg-[var(--mc-surface)] p-6 text-left transition-colors hover:bg-[var(--mc-surface-2)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--mc-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--mc-bg)]"
      style={{
        borderColor: MISSION_BORDER,
        borderRadius: MISSION_CARD_RADIUS,
        boxShadow: "var(--mc-shadow)",
      }}
    >
      <div className="flex items-start">
        <span
          className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[var(--mc-surface-hover)] transition-colors group-hover:bg-[var(--mc-accent-soft)]"
          style={{ color: "var(--mc-accent)" }}
        >
          {icon}
        </span>
      </div>
      <h3 className="mt-7 text-[20px] font-bold tracking-tight" style={{ color: MISSION_TEXT }}>
        {title}
      </h3>
      <p className="mt-auto pt-5 text-[12px] font-mono font-semibold tracking-[0.08em]" style={{ color: "var(--mc-text-subtle)" }}>
        {examples}
      </p>
    </button>
  );
}

// Rendered inside the Autonomy Studio shell, which owns the page root and the
// app bar; this is only the choice between the two workspaces.
export default function AutonomyStudioWorkspaceChooser({ onChooseMissionCanvas, onChooseTaskBuilder }) {
  return (
    <main className="flex flex-1 items-center justify-center overflow-auto px-8 py-10">
        <section className="w-full max-w-[900px]">
          <div className="text-[10px] font-mono font-bold tracking-[0.18em]" style={{ color: "var(--mc-accent)" }}>
            AUTONOMY STUDIO
          </div>
          <h2 className="mt-3 text-[28px] font-bold tracking-[-0.025em]" style={{ color: MISSION_TEXT }}>
            Choose a workspace
          </h2>
          <div className="mt-8 grid grid-cols-1 gap-5 md:grid-cols-2">
            <WorkspaceChoiceCard
              ariaLabel="Open Action Canvas"
              title="Action Canvas"
              examples="STATIONARY TASKS"
              icon={<MdAccountTree size={23} aria-hidden="true" />}
              onClick={onChooseTaskBuilder}
            />
            <WorkspaceChoiceCard
              ariaLabel="Open Mission Canvas"
              title="Mission Canvas"
              examples="MOBILE MISSIONS"
              icon={<MdRoute size={23} aria-hidden="true" />}
              onClick={onChooseMissionCanvas}
            />
          </div>
        </section>
      </main>
  );
}
