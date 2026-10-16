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

import DesignStageSidebar from "../stages/DesignStageSidebar";
import MappingStageSidebar from "../stages/MappingStageSidebar";
import { NavigationStageSidebar } from "../stages/NavigationStage";
import { RunStageSidebar } from "../stages/RunStage";
import {
  STAGE_AUTHORING,
  STAGE_MAPPING,
  STAGE_NAVIGATE,
  normalizeWorkspaceStage,
} from "../../../mapWorkspace/lib/stages";
import StageHeader from "../../../mapWorkspace/components/workspace/StageHeader";

export default function MissionStageChrome({
  stage: requestedStage,
  header,
  map,
  sidebar,
  children,
}) {
  const stage = normalizeWorkspaceStage(requestedStage);
  const waypointBtLayer = map.waypointBtLayer;
  const mappingEditorActive = map.mappingEditorActive;

  return (
    <div className="flex-1 min-w-0 flex flex-col">
      <StageHeader {...header} workspaceStage={stage} />
      <div className={`flex-1 min-h-0 grid grid-cols-1 ${waypointBtLayer || mappingEditorActive ? "" : "xl:grid-cols-[minmax(460px,1fr)_380px]"}`}>
        {children}

        {stage === STAGE_AUTHORING ? (!waypointBtLayer ? (
          <DesignStageSidebar {...sidebar.design} />
        ) : null) : mappingEditorActive ? null : (
          <aside className="min-h-0 grid gap-4 overflow-auto p-4 content-start">
            {stage === STAGE_MAPPING ? (
              <MappingStageSidebar {...sidebar.mapping} />
            ) : stage === STAGE_NAVIGATE ? (
              <NavigationStageSidebar {...sidebar.navigation} />
            ) : (
              <RunStageSidebar {...sidebar.run} />
            )}
          </aside>
        )}
      </div>
    </div>
  );
}
