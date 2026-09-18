# Saved Initial Pose

The Inference panel offers **Save** and **Return** in Robot mode. Save captures
fresh joint positions by joint name. Return publishes a target
trajectory once per joint group and zero base velocity. Clicking Return sends the
command without a confirmation dialog. Stop interrupts return with a current-pose hold.
During return, the same button changes to a red Stop button. It changes back
only when the backend status reports that returning has ended, including after
a successful stop. A failed stop keeps the Stop button available for retry.
The inline duration field next to Return defaults to 5 seconds and accepts
1 through 60 seconds. Blur or Enter saves the setting silently. Return also
accepts a valid unsaved value and saves it before requesting motion;
failed or stale setting requests never proceed to motion. The duration applies to both
JointTrajectory timing and the return interlock. It is independent of Slow Start.

This is separate from automatic **Initial Pose Sync**, which remains available
when starting a policy. No model or LeRobot Worker is required to save a pose.
Stop inference before saving or returning. Inference Stop (PAUSE) already
discards its execution plan and invalidates in-flight predictions. Return does
not unload the model or change inference settings or the paused session, even
when return is interrupted or fails. After return ends, Start resumes the paused
session using fresh inference rather than replaying the old plan. Resume remains
blocked while return or its recovery hold is pending. Fault and heartbeat safety
handling still apply independently.

## Ownership and Lifetime

- Policy Runtime owns one saved pose per robot type in memory.
- Reloading the browser retains it; restarting Policy Runtime clears it.
- Return duration is also session-scoped and stored per robot type. Other UIs
  receive it through the status topic. Duration cannot change during a return.
- A runtime instance ID prevents reuse across sessions. It is not a robot serial
  number: changing the physical robot behind the same topics requires restarting
  Runtime and saving again.
- `/policy/pose_command` accepts STATUS, SAVE, RETURN, STOP, and SET_DURATION.
- RETURN carries the duration selected in the UI; if another client has
  changed it, the backend rejects the stale request without publishing.
- Runtime publishes `/policy/pose_status` at 4 Hz. Browsers subscribe rather than
  repeatedly querying the service. Silent status disables motion controls.
- Only this topic updates saved-pose UI state. Service responses acknowledge
  requests; they cannot overwrite a newer status. Responses from an old robot,
  Runtime instance, connection, or unmounted panel are ignored. A disconnected
  status retains pending return and Stop controls, but disables Save and Return.
- Saved poses contain only configured JointTrajectory joints, not a mobile-base
  position. They use radians or meters according to the robot URDF.

## Safety and Limits

Save and Return require fresh, finite joint states with the configured joint names.
Measured positions are not clamped or rejected using URDF joint limits. Concurrent
inference starts, pending holds, and Worker maintenance are interlocked.
The Runtime monitors joint freshness and Orchestrator heartbeat during return.
A failed hold stays pending and blocks new execution until a retry succeeds.
Each zero Twist and joint-group hold is attempted independently. Missing, stale,
or non-finite joint state blocks only that group's hold; valid groups are still
stopped. All failures are reported together, and any failure prevents overall
stop success. Automatic Initial Pose Sync uses the same recovery path after a
partial publication failure. The UI does not proceed to Clear after a failed or
cancelled pose Stop.

After a successful publication, returning status remains active for the selected duration
to prevent concurrent motion requests and keep Stop available. It then clears
without comparing measured positions to the target, retrying the trajectory,
or issuing a final hold. This indicates duration expiry, not confirmed arrival
or controller acknowledgement. Failed stops and partial-publication recovery
remain pending beyond this duration until hold succeeds.

This is not collision checking or motion planning. A measured position alone
does not guarantee a safe return path. Controllers must enforce their motion
limits and honor JointTrajectory timing. Other teleoperation publishers must not command the
same controllers during return. Emergency stop and controller watchdogs remain
necessary; software cannot guarantee a hold after power or transport loss.

## Robot Description Snapshot

The seven URDF files under `shared/shared/robot_configs/urdf/` were copied from
the working tree on `robotis-ai@192.168.10.193` on 2026-09-18. They are not a
fresh upstream import. Upstream synchronization is deferred; robot configuration
joint order and topics remain unchanged.

Runtime reads top-level URDF joints for position units only; it does not
launch the embedded ros2_control or Gazebo configuration. The UI keeps its
existing mesh resolver. RealSense mesh references to the source container's
absolute paths still use the viewer's fallback geometry, since that description
package is not bundled. Visual camera meshes are not used for pose validation.

## Applying

Build/recreate Cyclo once (`./docker/container.sh start --build`) for the initial
UI, RobotPoseCommand interface, and source-mounted image layout. Subsequent
Runtime Python edits require only a launch restart; UI edits use `build-ui`, and
interface changes require a ROS workspace rebuild. URDF assets are mounted from
the host checkout. See [source-mounted containers](../../docker/README.md).
This feature does not change the LeRobot Worker protocol or require its image
to be rebuilt. Test with a clear workspace and an emergency stop available.
Automated tests do not move a robot.

The configurable duration adds request/response fields to RobotPoseCommand.
On an already source-mounted installation, Stop/Clear, stop the Cyclo launch,
run `cb` inside Cyclo, and relaunch from a shell sourcing the updated install.
Update the UI with `./docker/container.sh build-ui` and refresh open browsers.
Runtime and rosbridge must both restart with the new service definition;
restarting only the Worker or rebuilding only the UI is insufficient.
