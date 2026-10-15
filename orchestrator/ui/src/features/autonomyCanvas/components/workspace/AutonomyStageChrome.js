// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

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
