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

import { MdArrowBack } from "react-icons/md";
import { MISSION_BORDER, MISSION_SURFACE, MISSION_TEXT, MISSION_TEXT_MUTED } from "../../features/missionCanvas/lib/theme";

export default function AutonomyStudioAppBar({
  onBack,
  backLabel = "Back to Home",
  backTitle = "Back to Cyclo Intelligence Home",
  blockReason = "",
}) {
  return (
    <header
      className="shrink-0 h-14 flex items-center gap-3 px-4 border-b"
      style={{ borderColor: MISSION_BORDER, backgroundColor: MISSION_SURFACE }}
    >
      <button
        type="button"
        onClick={onBack}
        aria-label={backLabel}
        aria-describedby={blockReason ? "autonomy-studio-back-status" : undefined}
        title={blockReason || backTitle}
        className="h-10 w-10 shrink-0 inline-flex items-center justify-center rounded-[10px] transition-colors hover:bg-[var(--mc-surface-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--mc-accent)]"
        style={{ color: MISSION_TEXT_MUTED }}
      >
        <MdArrowBack size={19} aria-hidden="true" />
      </button>
      <div className="flex min-w-0 items-center gap-2.5">
        <div
          aria-hidden="true"
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px]"
          style={{ backgroundColor: MISSION_TEXT }}
        >
          <svg data-testid="autonomy-studio-brand-icon" aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--mc-bg)" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
            <rect x="4" y="8" width="16" height="12" rx="3" />
            <path d="M12 8V4" />
            <circle cx="12" cy="3" r="1.4" fill="var(--mc-accent)" stroke="none" />
            <path d="M9 13h.01M15 13h.01" />
          </svg>
        </div>
        <h1
          className="min-w-0 truncate text-[15px] font-bold tracking-tight"
          aria-label="Autonomy Studio"
          style={{ color: MISSION_TEXT }}
        >
          Autonomy Studio
        </h1>
      </div>
      {blockReason && (
        <div
          id="autonomy-studio-back-status"
          role="status"
          aria-live="polite"
          title={blockReason}
          className="ml-auto min-w-0 max-w-[50vw] truncate text-right text-[11px] font-medium"
          style={{ color: "var(--mc-text-subtle)" }}
        >
          {blockReason}
        </div>
      )}
    </header>
  );
}
