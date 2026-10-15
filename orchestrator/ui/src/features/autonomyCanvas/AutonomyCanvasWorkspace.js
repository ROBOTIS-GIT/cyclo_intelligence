import MapWorkspace from "../mapWorkspace/MapWorkspace";
import { WORKSPACE_AUTONOMY } from "../mapWorkspace/lib/stages";
import AutonomyStageChrome from "./components/workspace/AutonomyStageChrome";
import AutonomySceneSurface from "./components/workspace/AutonomySceneSurface";
import BackgroundTrainingStatus from "./components/BackgroundTrainingStatus";

export default function AutonomyCanvasWorkspace(props) {
  return (
    <MapWorkspace
      {...props}
      workspaceKind={WORKSPACE_AUTONOMY}
      StageChrome={AutonomyStageChrome}
      SceneSurface={AutonomySceneSurface}
      railFooter={<BackgroundTrainingStatus />}
    />
  );
}
