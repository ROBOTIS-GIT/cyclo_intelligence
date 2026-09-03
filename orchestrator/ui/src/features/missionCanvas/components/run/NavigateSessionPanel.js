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

import { Panel, SessionRow } from "../primitives";

export default function NavigateSessionPanel({ mapName, poseReady, goalStatus }) {
  const goalLabel = goalStatus === "driving"
    ? "Driving"
    : goalStatus === "reached"
      ? "Reached"
      : goalStatus === "failed"
        ? "Failed"
        : "—";
  return (
    <Panel title="Navigate Session" compact className="grid gap-1 content-start overflow-auto">
      <div className="grid gap-1">
        <SessionRow label="Map" value={mapName} />
        <SessionRow label="Pose" value={poseReady ? "Localized" : "Not localized"} />
        <SessionRow label="Goal" value={goalLabel} />
      </div>
    </Panel>
  );
}
