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

import CollectSidebar from "../CollectSidebar";
import OperateDrawer from "../OperateDrawer";
import StageHeader from "../../../mapWorkspace/components/workspace/StageHeader";
import { STAGE_COLLECT, STAGE_LEARN, normalizeWorkspaceStage } from "../../../mapWorkspace/lib/stages";

export default function AutonomyStageChrome({ stage: requestedStage, header, sidebar, children }) {
  const stage = normalizeWorkspaceStage(requestedStage);
  return (
    <div className="flex-1 min-w-0 flex flex-col">
      <StageHeader {...header} workspaceStage={stage} />
      {[STAGE_COLLECT, STAGE_LEARN].includes(stage) ? (
        <CollectSidebar {...sidebar.collect} mode={stage === STAGE_COLLECT ? "live" : "dataset"} mapCanvas={children} />
      ) : (
        <div className="operate-experience">{children}<OperateDrawer {...sidebar.operate} /></div>
      )}
    </div>
  );
}
