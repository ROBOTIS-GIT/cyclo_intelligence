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
// Author: Howon Kim, Seongwoo Kim

"use client";

export const TOOL_ICONS = {
  view: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z" /><circle cx="12" cy="12" r="2.6" />
        </svg>,
  erase_black: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="m7 21-4-4 11-11 4 4-11 11H7z" /><path d="m14 6 4 4" />
        </svg>,
  draw_black: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19c-4 1.5-8-1-8-5 0-6 8-11 8-11s8 5 8 11c0 4-4 6.5-8 5z" />
        </svg>,
  draw_unknown: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="9" />
            <path d="M9.6 9.2a2.5 2.5 0 1 1 3.5 2.4c-.8.4-1.1 1-1.1 1.9" />
            <path d="M12 16.8h.01" />
        </svg>,
  label_marker: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20.6 13.2 13.2 20.6a2 2 0 0 1-2.8 0l-7-7A2 2 0 0 1 2.8 12V4.4A1.6 1.6 0 0 1 4.4 2.8H12a2 2 0 0 1 1.4.6l7.2 7a2 2 0 0 1 0 2.8z" />
            <circle cx="7.5" cy="7.5" r="1.2" />
        </svg>,
  extend_area: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19c-4 1.5-8-1-8-5 0-4.5 5-8.5 7-10" />
            <path d="M17 3v8" /><path d="M13 7h8" />
        </svg>,
  erase_area: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 5h16" /><path d="M9 5V3h6v2" /><path d="M7 9l1 11h8l1-11" /><path d="M10 12v5" /><path d="M14 12v5" />
        </svg>
};
export function SegGroup({
  children,
  ariaLabel
}) {
  return <div role="group" aria-label={ariaLabel} className="flex items-center gap-0.5 p-[3px]" style={{
    borderRadius: 12,
    border: "1px solid var(--mc-border)",
    backgroundColor: "var(--mc-surface-hover)"
  }}>
            {children}
        </div>;
}
export function SegButton({
  selected,
  disabled,
  onClick,
  title,
  ariaLabel,
  children,
  narrow = false
}) {
  return <button type="button" disabled={disabled} aria-pressed={selected} onClick={onClick} title={title} aria-label={ariaLabel} className={`h-[30px] ${narrow ? "min-w-[42px] px-2 justify-center" : "px-3"} inline-flex items-center gap-1.5 whitespace-nowrap text-[12.5px] font-semibold transition-colors disabled:opacity-50`} style={{
    borderRadius: 9,
    border: "none",
    color: selected ? "var(--mc-bg)" : "var(--mc-text-muted)",
    backgroundColor: selected ? "var(--mc-text)" : "transparent"
  }}>
            {children}
        </button>;
}
export function EditorSection({
  label,
  children
}) {
  return <section className="inline-flex flex-wrap items-center gap-2 px-2 py-1" style={{
    borderRadius: 12,
    border: "1px solid var(--mc-border)",
    backgroundColor: "var(--mc-surface)"
  }}>
            <span className="text-[9.5px] font-mono tracking-[0.1em]" style={{
      color: "var(--mc-text-subtle)"
    }}>
                {label}
            </span>
            {children}
        </section>;
}
