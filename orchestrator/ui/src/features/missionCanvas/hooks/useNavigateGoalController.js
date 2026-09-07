// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useRef, useState } from "react";
import { orientationFromYaw } from "../../../utils/navigationTf";
import { rosTimestampNow } from "../../../utils/rosTime";

export default function useNavigateGoalController({
  sendGoalAndWait,
  onDisarm,
  onMessage,
}) {
  const [goalPose, setGoalPose] = useState(null);
  const [goalStatus, setGoalStatus] = useState("idle");
  const requestSequenceRef = useRef(0);

  const invalidateGoal = useCallback(() => {
    requestSequenceRef.current += 1;
    setGoalPose(null);
    setGoalStatus("idle");
  }, []);

  const sendGoal = useCallback(async (x, y, yaw) => {
    const sequence = requestSequenceRef.current + 1;
    requestSequenceRef.current = sequence;
    const poseStamped = {
      header: { frame_id: "map", stamp: rosTimestampNow() },
      pose: {
        position: { x, y, z: 0 },
        orientation: orientationFromYaw(yaw),
      },
    };
    setGoalPose({ pose: poseStamped.pose });
    setGoalStatus("driving");
    onDisarm();
    onMessage(`Goal ${x.toFixed(2)}, ${y.toFixed(2)}`);
    try {
      const result = await sendGoalAndWait({ pose: poseStamped });
      if (requestSequenceRef.current !== sequence) return;
      const status = String(result?.status || "").toUpperCase();
      if (result?.ok === false || (status && status !== "SUCCEEDED")) {
        setGoalStatus("failed");
        onMessage(result?.message || `Navigation goal ${status || "failed"}`);
        return;
      }
      setGoalStatus("reached");
      onMessage("Goal reached");
    } catch (error) {
      if (requestSequenceRef.current !== sequence) return;
      setGoalStatus("failed");
      onMessage(error instanceof Error ? error.message : "Navigation goal failed");
    }
  }, [onDisarm, onMessage, sendGoalAndWait]);

  return {
    goalPose,
    goalStatus,
    driving: goalStatus === "driving",
    sendGoal,
    invalidateGoal,
  };
}
