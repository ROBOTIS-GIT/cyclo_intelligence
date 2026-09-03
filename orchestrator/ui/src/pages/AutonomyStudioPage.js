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

"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import ActionCanvasWorkspace from "../components/navigation/ActionCanvasWorkspace";
import AutonomyStudioAppBar from "../components/autonomyStudio/AutonomyStudioAppBar";
import AutonomyStudioWorkspaceChooser from "../components/autonomyStudio/WorkspaceChooser";
import MissionCanvasWorkspace from "../features/missionCanvas/MissionCanvasWorkspace";
import useStaleRunShutdownRecovery from "../features/missionCanvas/hooks/useStaleRunShutdownRecovery";
import { WORKSPACE_ACTION_CANVAS, WORKSPACE_MISSION } from "../features/missionCanvas/lib/stages";
import {
  initialWorkspaceKind,
  readMissionSession,
  saveMissionSession,
} from "../features/missionCanvas/lib/session";
import { MISSION_STAGE_FILL, MISSION_TEXT } from "../features/missionCanvas/lib/theme";

const IDLE_EXIT_STATE = { active: false, busy: false };

// Autonomy Studio is light-only. The app's theme provider toggles a `dark`
// class on <html> (Tailwind dark: variants and the App.css overrides key on
// it), so the class is suspended while the studio is mounted — and kept off
// if the provider re-applies it — then restored on leave for the other pages.
function useLightOnlyTheme() {
  useLayoutEffect(() => {
    if (typeof document === "undefined") return undefined;
    const root = document.documentElement;
    let restoreDark = false;
    const suspendDark = () => {
      if (!root.classList.contains("dark")) return;
      restoreDark = true;
      root.classList.remove("dark");
    };
    suspendDark();
    const observer = new MutationObserver(suspendDark);
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => {
      observer.disconnect();
      if (restoreDark && root.getAttribute("data-theme") !== "light") {
        root.classList.add("dark");
      }
    };
  }, []);
}

function actionCanvasExitBlockReason({ active, busy }) {
  if (busy) return "Wait for the current operation to finish before going back";
  if (active) return "Stop the active runtime before going back";
  return "";
}

// Hosts one workspace under the shared app bar. The Action Canvas needs no map
// runtime, so it mounts on its own; the Mission Canvas owns its runtime and the
// unsaved-Design guard, so leaving it is requested through the workspace.
function AutonomyStudioWorkspace({
  workspaceKindOverride = null,
  dialogHost,
  onBackToWorkspaceChooser,
}) {
  // The kind is fixed for the lifetime of a mount; the shell remounts (keyed)
  // when a different workspace is chosen.
  const [workspaceKind] = useState(() => (
    workspaceKindOverride === WORKSPACE_MISSION
      || workspaceKindOverride === WORKSPACE_ACTION_CANVAS
      ? workspaceKindOverride
      : initialWorkspaceKind(readMissionSession())
  ));
  const actionCanvas = workspaceKind === WORKSPACE_ACTION_CANVAS;
  const [actionCanvasExitState, setActionCanvasExitState] = useState(IDLE_EXIT_STATE);
  const [missionExitBlockReason, setMissionExitBlockReason] = useState("");
  const missionExitRef = useRef(null);

  useEffect(() => {
    saveMissionSession({ workspaceKind });
  }, [workspaceKind]);
  useStaleRunShutdownRecovery(actionCanvas);

  const handleActionCanvasExitStateChange = useCallback((nextState) => {
    const normalized = {
      active: nextState?.active === true,
      busy: nextState?.busy === true,
    };
    setActionCanvasExitState((current) => (
      current.active === normalized.active && current.busy === normalized.busy
        ? current
        : normalized
    ));
  }, []);
  const handleMissionExitStateChange = useCallback((nextState) => {
    setMissionExitBlockReason(
      typeof nextState?.blockReason === "string" ? nextState.blockReason : "",
    );
  }, []);

  const blockReason = actionCanvas
    ? actionCanvasExitBlockReason(actionCanvasExitState)
    : missionExitBlockReason;

  const handleBack = () => {
    // The app bar already shows the reason; the click is simply not honored.
    if (blockReason) return;
    if (actionCanvas) {
      onBackToWorkspaceChooser();
      return;
    }
    const handle = missionExitRef.current;
    if (handle && typeof handle.requestExit === "function") {
      handle.requestExit(onBackToWorkspaceChooser);
    } else {
      onBackToWorkspaceChooser();
    }
  };

  return (
    <>
      <AutonomyStudioAppBar
        onBack={handleBack}
        backLabel="Back to workspace chooser"
        backTitle="Back to workspace selection"
        blockReason={blockReason}
      />
      {actionCanvas ? (
        <main className="flex-1 min-w-0 min-h-0" aria-label="Action Canvas workspace">
          <ActionCanvasWorkspace
            isActive
            title="Action Canvas"
            variant="autonomy-studio"
            onExitStateChange={handleActionCanvasExitStateChange}
          />
        </main>
      ) : (
        <MissionCanvasWorkspace
          dialogHost={dialogHost}
          onExitStateChange={handleMissionExitStateChange}
          exitHandleRef={missionExitRef}
        />
      )}
    </>
  );
}

export default function AutonomyStudioPage({
  onBackHome = null,
  showWorkspaceChooser = false,
}) {
  const [chooserOpen, setChooserOpen] = useState(showWorkspaceChooser);
  const [chosenWorkspace, setChosenWorkspace] = useState("");
  // Dialog portal host: dialogs opened from inside a glass HUD must escape
  // its backdrop-filter (which hijacks position:fixed) but stay inside
  // .autonomy-studio-page, where the --mc-* tokens are scoped.
  const pageRootRef = useRef(null);
  useLightOnlyTheme();

  useEffect(() => {
    if (showWorkspaceChooser) setChooserOpen(true);
  }, [showWorkspaceChooser]);

  const chooseWorkspace = (workspaceKind) => {
    setChosenWorkspace(workspaceKind);
    setChooserOpen(false);
  };

  return (
    <div
      ref={pageRootRef}
      className="autonomy-studio-page h-full min-h-[560px] flex flex-col overflow-hidden"
      style={{ backgroundColor: MISSION_STAGE_FILL, color: MISSION_TEXT }}
    >
      {chooserOpen ? (
        <>
          <AutonomyStudioAppBar onBack={onBackHome} />
          <AutonomyStudioWorkspaceChooser
            onChooseTaskBuilder={() => chooseWorkspace(WORKSPACE_ACTION_CANVAS)}
            onChooseMissionCanvas={() => chooseWorkspace(WORKSPACE_MISSION)}
          />
        </>
      ) : (
        <AutonomyStudioWorkspace
          key={chosenWorkspace || "restored-workspace"}
          workspaceKindOverride={chosenWorkspace || null}
          dialogHost={pageRootRef}
          onBackToWorkspaceChooser={() => setChooserOpen(true)}
        />
      )}
    </div>
  );
}
