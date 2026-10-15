// The shared shell renders feature projections without owning document or runtime state.
import MissionCanvasDialogLayer from "./MissionCanvasDialogLayer";
import StageRail from "./StageRail";
import buildWorkspaceDialogs from "../../presentation/buildWorkspaceDialogs";
import buildWorkspaceRail from "../../presentation/buildWorkspaceRail";
import buildWorkspaceChrome from "../../presentation/buildWorkspaceChrome";
import buildWorkspaceHud from "../../presentation/buildWorkspaceHud";
import buildWorkspaceScene from "../../presentation/buildWorkspaceScene";

export default function MapWorkspaceView({ SceneSurface, StageChrome, model }) {
  const dialogs = buildWorkspaceDialogs(model);
  const rail = buildWorkspaceRail(model);
  const chrome = buildWorkspaceChrome(model);
  const scene = buildWorkspaceScene(model);
  const { hud } = buildWorkspaceHud(model);
  return (
    <>
      <MissionCanvasDialogLayer {...dialogs} />
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <StageRail {...rail} />
        <StageChrome {...chrome}>
          <SceneSurface {...scene} hud={hud} />
        </StageChrome>
      </div>
    </>
  );
}
