// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
import { JointTeleopClient } from "./jointTeleop";

// SG2 follower launch remap and prismatic joint limits from ffw_body.urdf.xacro.
// +Z raises the lift; positions are joint displacement, not height above ground.
export const LIFT_TOPIC = "/leader/joystick_controller_right/joint_trajectory";
export const LIFT_JOINTS = ["lift_joint"];
export const LIFT_LIMITS = [[-0.5, 0.0]];
export const LIFT_DIRECTIONS = { up: 1, down: -1 };
const STEP_METRES = 0.003;
const LIMIT_MARGIN_METRES = 0.005;

export function liftPositions(message) {
  if (!Array.isArray(message?.name) || !Array.isArray(message?.position)) return null;
  const index = message.name.indexOf("lift_joint");
  if (index < 0 || message.name.lastIndexOf("lift_joint") !== index) return null;
  return Number.isFinite(message.position[index]) ? [message.position[index]] : null;
}

export function liftTrajectory(positions, direction) {
  let target = positions[0];
  if (direction) {
    target = LIFT_DIRECTIONS[direction] > 0
      ? Math.max(target, Math.min(-LIMIT_MARGIN_METRES, target + STEP_METRES))
      : Math.min(target, Math.max(-0.5 + LIMIT_MARGIN_METRES, target - STEP_METRES));
  }
  return {
    joint_names: LIFT_JOINTS,
    points: [{ positions: [target], velocities: [0], time_from_start: { sec: 0, nanosec: 300000000 } }],
  };
}

export class LiftTeleopClient extends JointTeleopClient {
  constructor(ros, now) {
    super(ros, {
      id: "lift", label: "Lift", topic: LIFT_TOPIC, controller: "lift_controller",
      limits: LIFT_LIMITS, directions: LIFT_DIRECTIONS, axisLabels: ["Lift"],
      positions: liftPositions, trajectory: liftTrajectory,
    }, now);
  }
}
