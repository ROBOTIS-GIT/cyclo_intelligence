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
