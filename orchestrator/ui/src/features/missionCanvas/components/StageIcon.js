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

export default function StageIcon({ id, active }) {
  const stroke = active ? "var(--mc-accent)" : "currentColor";
  const common = {
    width: 17,
    height: 17,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke,
    strokeWidth: 1.7,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
    className: "shrink-0",
  };
  if (id === "mapping") {
    return (
      <svg {...common}>
        <path d="M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2z" />
        <path d="M9 4v14M15 6v14" />
      </svg>
    );
  }
  if (id === "map_edit") {
    return (
      <svg {...common}>
        <path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 3 21.5l1-4.5L17 3z" />
      </svg>
    );
  }
  if (id === "authoring") {
    return (
      <svg {...common}>
        <circle cx="6" cy="6" r="2.5" />
        <circle cx="18" cy="18" r="2.5" />
        <path d="M8.5 6H14a4 4 0 0 1 4 4v5.5" />
      </svg>
    );
  }
  if (id === "navigate") {
    // Compass needle: point-to-point driving.
    return (
      <svg {...common}>
        <path d="M12 3 19 20l-7-4-7 4 7-17z" />
      </svg>
    );
  }
  return (
    <svg {...common}>
      <polygon points="6 4 20 12 6 20 6 4" />
    </svg>
  );
}
