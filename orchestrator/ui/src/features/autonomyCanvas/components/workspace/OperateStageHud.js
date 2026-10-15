// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import RuntimeControlsHud from "../../../mapWorkspace/components/RuntimeControlsHud";

export default function OperateStageHud({ observation, busy, btNodeBusy, runMapBusy, missionMapLoaded,
  runShutdownPending, interactionMode, onLocalize, runPoseInitialized, missionRunnerActive,
  onStopNavigation, missionRunnerStopping, running }) {
  return <RuntimeControlsHud
    localize={{
      onClick: onLocalize,
      disabled: !!busy || missionRunnerActive || runMapBusy || !missionMapLoaded || runShutdownPending,
      pressed: interactionMode === "initial" || busy === "Localize" || busy === "Set robot pose",
    }}
    start={{
      onClick: observation?.start,
      disabled: !!busy || !!btNodeBusy || runMapBusy || !missionMapLoaded || !runPoseInitialized || !running
        || observation?.canStart === false || observation?.active || observation?.phase === "stopping" || missionRunnerActive || runShutdownPending,
      pressed: busy === "Run mission" || missionRunnerActive,
      label: "Run autonomy", title: "Start reasoning and mission execution",
    }}
    stop={{
      onClick: onStopNavigation,
      disabled: !!busy || missionRunnerStopping || observation?.phase === "stopping"
        || (!running && !missionRunnerActive && !runShutdownPending && !observation?.needsStop),
      pressed: busy === "Stop" || runShutdownPending,
      title: "Stop Qwen container and navigation",
    }}
  />;
}
