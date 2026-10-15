import { STAGE_LEARN } from "../lib/stages";


// Presentational mapping only; execution state stays in its owning controller.
export default function buildWorkspaceRail({ runtime, recordData, workspace }) {
  const {
    railFooter, workspaceKind, missionRunnerActive, collectWaypoints, mappingRuntimeActive, handleSelectStageTab
  } = runtime;
  const {
    collectEditorBusy, collectDraft
  } = recordData;
  const {
    dataStageBusy, busy, workspaceStage
  } = workspace;
  return {
    footer: railFooter,
    workspaceKind: workspaceKind,
    interactionLocked: collectEditorBusy || dataStageBusy,
    disabledStages: (collectDraft.active || collectDraft.recording || collectDraft.busy || missionRunnerActive || collectWaypoints.moving) ? [STAGE_LEARN] : [],
    busy: busy,
    mappingRuntimeActive: mappingRuntimeActive,
    onSelectStage: handleSelectStageTab,
    workspaceStage: workspaceStage
  };
}
