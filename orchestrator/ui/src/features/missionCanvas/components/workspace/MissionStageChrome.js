// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import DesignStageSidebar from "../stages/DesignStageSidebar";
import MappingStageSidebar from "../stages/MappingStageSidebar";
import { NavigationStageSidebar } from "../stages/NavigationStage";
import { RunStageSidebar } from "../stages/RunStage";
import {
  STAGE_AUTHORING,
  STAGE_MAPPING,
  STAGE_NAVIGATE,
} from "../../lib/stages";
import StageHeader from "./StageHeader";

export default function MissionStageChrome({
  stage,
  header,
  map,
  sidebar,
  children,
}) {
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
