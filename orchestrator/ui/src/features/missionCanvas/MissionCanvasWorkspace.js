import MapWorkspace from "../mapWorkspace/MapWorkspace";
import { WORKSPACE_MISSION } from "../mapWorkspace/lib/stages";
import MissionStageChrome from "./components/workspace/MissionStageChrome";
import MissionCanvasSceneSurface from "./components/workspace/MissionCanvasSceneSurface";

export default function MissionCanvasWorkspace(props) {
  return (
    <MapWorkspace
      {...props}
      workspaceKind={WORKSPACE_MISSION}
      StageChrome={MissionStageChrome}
      SceneSurface={MissionCanvasSceneSurface}
    />
  );
}
