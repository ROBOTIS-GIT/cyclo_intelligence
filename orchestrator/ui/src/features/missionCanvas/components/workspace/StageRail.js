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

import {
  STAGE_MAP_EDIT,
  WORKSPACE_NAV_GROUPS,
  WORKSPACE_STAGES,
} from "../../lib/stages";
import {
  MISSION_BORDER,
  MISSION_RAIL_BG,
  MISSION_STAGE_EMPTY,
  MISSION_TEXT,
  MISSION_TEXT_MUTED,
} from "../../lib/theme";
import StageIcon from "../StageIcon";

export default function StageRail({
  busy,
  mappingRuntimeActive,
  onSelectStage,
  workspaceStage,
}) {
  return (
    <aside
      className="shrink-0 flex flex-col p-4 border-r"
      style={{ width: 210, backgroundColor: MISSION_RAIL_BG, borderColor: MISSION_BORDER }}
    >
      <nav className="grid gap-1" role="tablist" aria-label="Mission Canvas stages">
        {WORKSPACE_NAV_GROUPS.map((group, groupIndex) => (
          <div key={group.caption} className={`grid gap-1 ${groupIndex === 0 ? "" : "mt-4"}`}>
            <div
              className="px-1 pb-1.5 text-[11px] font-mono font-semibold tracking-[0.14em]"
              style={{ color: MISSION_TEXT_MUTED }}
            >
              {group.caption}
            </div>
            {group.stageIds.map((stageId) => {
              const stage = WORKSPACE_STAGES.find((item) => item.id === stageId);
              const selected = workspaceStage === stage.id;
              // SLAM may rewrite the saved PGM while it is being edited.
              // Run/Navigation sessions use the ordinary stage-exit stop,
              // so Map Edit must stay clickable to initiate that shutdown.
              const editLocked = stage.id === STAGE_MAP_EDIT && !selected && mappingRuntimeActive;
              return (
                <button
                  key={stage.id}
                  type="button"
                  role="tab"
                  aria-selected={selected}
                  disabled={!!busy || editLocked}
                  title={editLocked ? "Stop mapping before editing saved maps" : undefined}
                  onClick={() => onSelectStage(stage.id)}
                  className="flex items-center gap-3 px-3 py-2.5 text-[13px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50"
                  style={{
                    borderRadius: 10,
                    color: selected ? MISSION_TEXT : MISSION_TEXT_MUTED,
                    backgroundColor: selected ? MISSION_STAGE_EMPTY : "transparent",
                    border: `1px solid ${selected ? MISSION_BORDER : "transparent"}`,
                    boxShadow: selected ? "var(--mc-shadow)" : "none",
                  }}
                >
                  <StageIcon id={stage.id} active={selected} />
                  {stage.label}
                </button>
              );
            })}
          </div>
        ))}
      </nav>
    </aside>
  );
}
