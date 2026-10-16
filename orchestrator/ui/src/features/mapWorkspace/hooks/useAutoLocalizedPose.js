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

import { useCallback } from "react";
import { requestNoMotionUpdate } from "../../../utils/navigationApi";

const AUTO_LOCALIZE_MAX_UPDATES = 10;

const AUTO_LOCALIZE_MIN_UPDATES = 3;

const AUTO_LOCALIZE_UPDATE_DELAY_MS = 700;

const AUTO_LOCALIZE_XY_COVARIANCE_MAX = 0.6;

const AUTO_LOCALIZE_YAW_COVARIANCE_MAX = 0.5;

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function amclPoseLooksLocalized(amclPose) {
  const covariance = amclPose?.pose?.covariance;
  if (!Array.isArray(covariance) || covariance.length < 36) return false;
  const xVariance = Number(covariance[0]);
  const yVariance = Number(covariance[7]);
  const yawVariance = Number(covariance[35]);
  if (![xVariance, yVariance, yawVariance].every(Number.isFinite)) return false;
  return (
    Math.max(xVariance, yVariance) <= AUTO_LOCALIZE_XY_COVARIANCE_MAX &&
    yawVariance <= AUTO_LOCALIZE_YAW_COVARIANCE_MAX
  );
}

// Shared convergence polling for Design At Robot and Run localization.
// Keep the latest available pose as the existing fallback after all updates.
export default function useAutoLocalizedPose(amclPoseRef) {
  return useCallback(async () => {
    let latestPose = null;
    for (let attempt = 0; attempt < AUTO_LOCALIZE_MAX_UPDATES; attempt += 1) {
      await requestNoMotionUpdate();
      await delay(AUTO_LOCALIZE_UPDATE_DELAY_MS);
      const amclPoseMessage = amclPoseRef.current;
      const pose = amclPoseMessage?.pose?.pose;
      if (pose?.position) {
        latestPose = pose;
        if (
          attempt + 1 >= AUTO_LOCALIZE_MIN_UPDATES &&
          amclPoseLooksLocalized(amclPoseMessage)
        ) {
          return pose;
        }
      }
    }
    if (latestPose?.position) return latestPose;
    throw new Error("Robot pose unavailable after automatic localization");
  }, [amclPoseRef]);
}
