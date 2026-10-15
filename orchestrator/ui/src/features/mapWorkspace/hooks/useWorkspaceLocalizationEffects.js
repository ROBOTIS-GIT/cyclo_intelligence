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

import { useCallback, useEffect } from "react";
import { requestNoMotionUpdate } from "../../../utils/navigationApi";
import { updateTfBuffer } from "../../../utils/navigationTf";
import { STAGE_AUTHORING } from "../lib/stages";

export default function useWorkspaceLocalizationEffects({
  workspace, design, currentPose, amclPose, runRuntimeActive, running, designLocalizationActive, latestTf, NOMOTION_UPDATE_INTERVAL_MS
}) {
  const {
    currentPoseRef, amclPoseRef, invalidateRunPoseRequestRef, setRunPoseInitialized, prevRunningRef,
    workspaceStage, setMissionMapLoaded, nomotionUpdateBusyRef, tfBufferRef, setTfBufferRevision,
  } = workspace;
  const { designPoseInitialized } = design;
  useEffect(() => {
    currentPoseRef.current = currentPose;
  }, [currentPose, currentPoseRef]);

  useEffect(() => {
    amclPoseRef.current = amclPose;
  }, [amclPose, amclPoseRef]);

  useEffect(() => {
    if (!runRuntimeActive) {
      invalidateRunPoseRequestRef.current();
      setRunPoseInitialized(false);
    }
  }, [invalidateRunPoseRequestRef, runRuntimeActive, setRunPoseInitialized]);

  useEffect(() => {
    if (
      prevRunningRef.current
      && !running
      && workspaceStage !== STAGE_AUTHORING
    ) {
      setMissionMapLoaded(false);
    }
    prevRunningRef.current = running;
  }, [prevRunningRef, running, setMissionMapLoaded, workspaceStage]);

  useEffect(() => {
    if (!designLocalizationActive || !designPoseInitialized) {
      return undefined;
    }
    let cancelled = false;
    const tick = async () => {
      if (cancelled || nomotionUpdateBusyRef.current) return;
      nomotionUpdateBusyRef.current = true;
      try {
        await requestNoMotionUpdate();
      } catch (error) {
        console.warn("No-motion AMCL update failed:", error);
      } finally {
        nomotionUpdateBusyRef.current = false;
      }
    };
    void tick();
    const interval = window.setInterval(tick, NOMOTION_UPDATE_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [
    NOMOTION_UPDATE_INTERVAL_MS, designLocalizationActive, designPoseInitialized,
    nomotionUpdateBusyRef,
  ]);

  useEffect(() => {
    if (updateTfBuffer(tfBufferRef.current, latestTf)) {
      setTfBufferRevision((value) => value + 1);
    }
  }, [latestTf, setTfBufferRevision, tfBufferRef]);

  const clearLocalizationPoseCache = useCallback(() => {
    tfBufferRef.current.clear();
    currentPoseRef.current = null;
    amclPoseRef.current = null;
    setTfBufferRevision((value) => value + 1);
  }, [amclPoseRef, currentPoseRef, setTfBufferRevision, tfBufferRef]);

  return {
    clearLocalizationPoseCache,
  };
}
