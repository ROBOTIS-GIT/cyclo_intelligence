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
// Author: Howon Kim

"use client";
// Constants
const SUCCESS_STATUS_VALUES = ["running", "available"];
const DEFAULT_LABELS = {
    up: "Running",
    down: "Stopped",
};
const SUCCESS_COLORS = {
    background: "rgba(137, 209, 133, 0.2)",
    border: "rgba(137, 209, 133, 0.3)",
    foreground: "var(--vscode-successForeground)",
};
const ERROR_COLORS = {
    background: "rgba(244, 135, 113, 0.2)",
    border: "rgba(244, 135, 113, 0.3)",
    foreground: "var(--vscode-errorForeground)",
};
// Utility functions
function isStatusUp(status) {
    if (typeof status === "boolean") {
        return status;
    }
    const normalized = status.toLowerCase();
    return SUCCESS_STATUS_VALUES.some((val) => normalized === val);
}
function getStatusLabel(isUp, customLabel) {
    return customLabel || (isUp ? DEFAULT_LABELS.up : DEFAULT_LABELS.down);
}
function getStatusStyles(isUp) {
    const colors = isUp ? SUCCESS_COLORS : ERROR_COLORS;
    return {
        backgroundColor: colors.background,
        color: colors.foreground,
        border: `1px solid ${colors.border}`,
    };
}
function getIndicatorStyle(isUp) {
    return {
        backgroundColor: isUp ? SUCCESS_COLORS.foreground : ERROR_COLORS.foreground,
    };
}
// Component
export default function StatusBadge({ status, label, className = "", dotOnly = false, }) {
    const isUp = isStatusUp(status);
    const indicatorStyle = getIndicatorStyle(isUp);
    if (dotOnly) {
        return (<span className={`inline-flex items-center ${className}`} title={getStatusLabel(isUp, label)}>
        <span className="w-2 h-2 rounded-full" style={indicatorStyle}/>
      </span>);
    }
    const displayLabel = getStatusLabel(isUp, label);
    const badgeStyles = getStatusStyles(isUp);
    return (<span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${className}`} style={badgeStyles}>
      <span className="w-1.5 h-1.5 rounded-full mr-1.5" style={indicatorStyle}/>
      {displayLabel}
    </span>);
}

