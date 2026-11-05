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

// Mission Canvas design tokens live in App.css scoped to `.autonomy-studio-page`
// (see `--mc-*`, light + dark). These constants map the legacy names onto those
// CSS variables so inline styles stay theme-aware.
export const MISSION_BORDER = "var(--mc-border)";

export const MISSION_BUTTON_BORDER = "var(--mc-border-strong)";

export const MISSION_PANEL_BORDER = "var(--mc-border)";

export const MISSION_STAGE_FILL = "var(--mc-bg)";

export const MISSION_STAGE_EMPTY = "var(--mc-surface)";

export const MISSION_SURFACE = "var(--mc-surface-2)";

export const MISSION_SURFACE_STRONG = "var(--mc-surface-hover)";

export const MISSION_TEXT = "var(--mc-text)";

export const MISSION_TEXT_MUTED = "var(--mc-text-muted)";

// Warm-minimal literals: the layer switch on/off backgrounds use the light-mode
// hex equivalents of var(--mc-success)/var(--mc-border-strong) rather than the
// tokens themselves. The switch reads identically in light mode, and literal
// colors stay assertable in jsdom (which does not resolve var() in toHaveStyle).
export const MISSION_LIVE = "#5b8266";

export const MISSION_SWITCH_OFF = "#dcd7ca";

// Left-rail + card language (warm-minimal Console shell).
export const MISSION_RAIL_BG = "var(--mc-surface-hover)";

export const MISSION_CARD_RADIUS = 16;

export const MISSION_GLASS = "color-mix(in srgb, var(--mc-surface) 88%, transparent)";

// Editor brush-ring colors per tool (matches the warm OCC/MARKER palettes):
// ink/cream for obstacles, paper for clearing, sage for extend, danger for erase.
export const EDITOR_BRUSH_RING_COLORS = {
  draw_black: "#2A2620",
  erase_black: "#FAF8F3",
  draw_unknown: "#8C8677",
  extend_area: "#5B8266",
  erase_area: "#C14E34",
};
