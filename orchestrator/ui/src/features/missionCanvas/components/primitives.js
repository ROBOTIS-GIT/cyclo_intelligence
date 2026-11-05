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
  MISSION_BUTTON_BORDER,
  MISSION_CARD_RADIUS,
  MISSION_LIVE,
  MISSION_PANEL_BORDER,
  MISSION_STAGE_EMPTY,
  MISSION_SURFACE_STRONG,
  MISSION_SWITCH_OFF,
  MISSION_TEXT,
  MISSION_TEXT_MUTED,
} from "../lib/theme";

export function Panel({ title, children, className = "", compact = false }) {
  return (
    <div
      className={`border rounded-md min-h-0 min-w-0 ${compact ? "p-3" : "p-4"} ${className}`}
      style={{
        color: MISSION_TEXT,
        borderColor: MISSION_PANEL_BORDER,
        backgroundColor: MISSION_STAGE_EMPTY,
        borderRadius: MISSION_CARD_RADIUS,
        boxShadow: "var(--mc-shadow)",
      }}
    >
      {title && (
        <div className={`text-[13.5px] font-bold ${compact ? "mb-2" : "mb-3"}`}>
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

export function LayerToggle({ label, checked, compact = false, onChange }) {
  const trackW = compact ? 36 : 40;
  const trackH = compact ? 20 : 22;
  const knob = trackH - 6;
  return (
    <div className={`${compact ? "min-h-6" : "min-h-7"} flex items-center justify-between gap-3 text-xs font-medium select-none`} style={{ color: MISSION_TEXT }}>
      <span className="truncate" style={{ color: checked ? MISSION_TEXT : MISSION_TEXT_MUTED }}>{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className="inline-flex relative shrink-0 cursor-pointer transition-colors active:translate-y-px"
        style={{ width: trackW, height: trackH, borderRadius: 999, border: "none", backgroundColor: checked ? MISSION_LIVE : MISSION_SWITCH_OFF }}
      >
        <span
          aria-hidden="true"
          className="block absolute rounded-full transition-transform duration-150 ease-out"
          style={{
            width: knob, height: knob, top: 3, left: 3, backgroundColor: "#fff",
            boxShadow: "0 1px 2px rgba(28,26,23,0.3)",
            transform: checked ? `translateX(${trackW - knob - 6}px)` : "translateX(0)",
          }}
        />
      </button>
    </div>
  );
}

export function SessionRow({ label, value, stacked = false }) {
  if (stacked) {
    return (
      <div className="grid gap-0.5 text-xs min-w-0">
        <span style={{ color: MISSION_TEXT_MUTED }}>{label}</span>
        <span className="font-mono truncate">{value}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between gap-2 text-xs min-w-0">
      <span style={{ color: MISSION_TEXT_MUTED }}>{label}</span>
      <span className="font-mono truncate text-right">{value}</span>
    </div>
  );
}

export function ActionButton({
  children,
  active = false,
  disabled = false,
  className = "",
  onClick,
  title,
  type = "button",
  variant = "primary",
}) {
  const styles = {
    primary: { color: "var(--mc-accent-fg)", backgroundColor: "var(--mc-accent)", borderColor: "var(--mc-accent)", boxShadow: "var(--mc-shadow)" },
    secondary: { color: MISSION_TEXT, backgroundColor: MISSION_STAGE_EMPTY, borderColor: MISSION_BUTTON_BORDER, boxShadow: "var(--mc-shadow)" },
    danger: { color: "var(--mc-danger)", backgroundColor: MISSION_STAGE_EMPTY, borderColor: "var(--mc-danger-border)", boxShadow: "var(--mc-shadow)" },
  };
  const activeStyles = active
    ? {
      color: variant === "danger" ? "var(--mc-danger)" : MISSION_TEXT,
      backgroundColor: variant === "danger" ? "var(--mc-accent-soft)" : MISSION_SURFACE_STRONG,
      borderColor: variant === "danger" ? "var(--mc-danger-border)" : MISSION_BUTTON_BORDER,
      boxShadow: "none",
    }
    : {};

  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      title={title}
      aria-pressed={active ? true : undefined}
      className={[
        "h-9 px-4 border rounded-md inline-flex items-center justify-center text-center whitespace-nowrap text-[13px] font-semibold transition-all active:translate-y-px",
        active ? "disabled:opacity-90" : "disabled:opacity-50",
        "disabled:active:translate-y-0",
        className,
      ].join(" ")}
      style={{ borderRadius: 10, ...styles[variant], ...activeStyles }}
    >
      {children}
    </button>
  );
}

export function WaypointOptionButton({ active = false, disabled = false, onClick, children }) {
  return (
    <button
      type="button"
      disabled={disabled}
      aria-pressed={active ? true : undefined}
      onClick={onClick}
      className="h-10 min-w-[92px] px-4 inline-flex items-center justify-center text-center whitespace-nowrap text-[13px] font-semibold transition-all active:translate-y-px disabled:opacity-50 disabled:active:translate-y-0"
      style={{
        borderRadius: 10,
        border: `1px solid ${active ? "var(--mc-border-strong)" : MISSION_BUTTON_BORDER}`,
        backgroundColor: active ? MISSION_SURFACE_STRONG : MISSION_STAGE_EMPTY,
        color: MISSION_TEXT,
        boxShadow: active ? "none" : "var(--mc-shadow)",
      }}
    >
      {children}
    </button>
  );
}

// Icon toggle for the Map Edit HUD — the Design HUD's Edit Route idiom
// (accent-soft fill + accent border while the tool is active).
export function MapEditToolButton({ label, active = false, disabled = false, onClick, children }) {
  return (
    <button
      type="button"
      disabled={disabled}
      aria-pressed={active}
      aria-label={label}
      title={label}
      onClick={onClick}
      className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
      style={{
        borderRadius: 9,
        border: `1px solid ${active ? "var(--mc-accent)" : "var(--mc-border-strong)"}`,
        backgroundColor: active ? "var(--mc-accent-soft)" : "var(--mc-surface)",
        color: "var(--mc-text)",
      }}
    >
      {children}
    </button>
  );
}
