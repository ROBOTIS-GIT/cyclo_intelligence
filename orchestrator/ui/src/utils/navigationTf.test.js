import {
  createMappingPoseSynchronizer,
  poseForScanFrame,
  poseForScanFrameAtBasePose,
  poseForTfAxesFrame,
  replaceTfFramePose,
} from './navigationTf';

const pose2d = (x, y = 0, yaw = 0) => ({
  position: { x, y, z: 0 },
  orientation: { x: 0, y: 0, z: Math.sin(yaw / 2), w: Math.cos(yaw / 2) },
});

const odometryMessage = (sec, nanosec, pose) => ({
  header: { stamp: { sec, nanosec }, frame_id: 'odom' },
  child_frame_id: 'base_link',
  pose: { pose },
});

const slamPoseMessage = (sec, nanosec, pose) => ({
  header: { stamp: { sec, nanosec }, frame_id: 'map' },
  pose: { pose },
});

test('uses the authoritative robot pose for a base_link laser scan', () => {
  const staleTfPose = {
    position: { x: 52, y: 53, z: 0 },
    orientation: { x: 0, y: 0, z: 0, w: 1 },
  };
  const amclPose = {
    position: { x: 1.25, y: -0.5, z: 0 },
    orientation: { x: 0, y: 0, z: 0.25, w: 0.97 },
  };

  expect(poseForScanFrame(
    'base_link',
    new Map([['base_link', staleTfPose]]),
    amclPose,
  )).toBe(amclPose);
});

test('retains the TF pose for a laser frame with a sensor offset', () => {
  const laserTfPose = {
    position: { x: 1.5, y: -0.4, z: 0.2 },
    orientation: { x: 0, y: 0, z: 0.25, w: 0.97 },
  };
  const robotPose = {
    position: { x: 1.25, y: -0.5, z: 0 },
    orientation: { x: 0, y: 0, z: 0.25, w: 0.97 },
  };

  expect(poseForScanFrame(
    'base_scan',
    new Map([['base_scan', laserTfPose]]),
    robotPose,
  )).toBe(laserTfPose);
});

test('applies an offset laser frame to the synchronized scan-time base pose', () => {
  const bufferedBasePose = pose2d(100, 0, 0);
  const bufferedLaserPose = pose2d(100.3, 0.2, 0);
  const scanTimeBasePose = pose2d(1, 2, Math.PI / 2);

  const scanPose = poseForScanFrameAtBasePose(
    'base_scan',
    new Map([
      ['base_link', bufferedBasePose],
      ['base_scan', bufferedLaserPose],
    ]),
    scanTimeBasePose,
  );

  expect(scanPose.position.x).toBeCloseTo(0.8);
  expect(scanPose.position.y).toBeCloseTo(2.3);
  expect(scanPose.position.z).toBeCloseTo(0);
});

test('aligns only the displayed base_link TF axis with the robot pose', () => {
  const staleBaseLinkPose = {
    position: { x: 52, y: 53, z: 0 },
    orientation: { x: 0, y: 0, z: 0, w: 1 },
  };
  const odomPose = {
    position: { x: 50, y: 50, z: 0 },
    orientation: { x: 0, y: 0, z: 0, w: 1 },
  };
  const amclPose = {
    position: { x: 1.25, y: -0.5, z: 0 },
    orientation: { x: 0, y: 0, z: 0.25, w: 0.97 },
  };

  expect(poseForTfAxesFrame('/base_link', staleBaseLinkPose, amclPose)).toBe(amclPose);
  expect(poseForTfAxesFrame('base_link', staleBaseLinkPose, null)).toBe(staleBaseLinkPose);
  expect(poseForTfAxesFrame('odom', odomPose, amclPose)).toBe(odomPose);
});

test('anchors Mapping in the SLAM map frame and propagates current odometry', () => {
  const synchronizer = createMappingPoseSynchronizer();
  synchronizer.addOdometry(odometryMessage(10, 0, pose2d(2)));
  synchronizer.addSlamPose(slamPoseMessage(10, 0, pose2d(10)));
  synchronizer.addOdometry(odometryMessage(11, 0, pose2d(3)));

  const snapshot = synchronizer.snapshot({ sec: 10, nanosec: 0 });
  expect(snapshot.mapToOdomPose.position.x).toBeCloseTo(8);
  expect(snapshot.pose.position.x).toBeCloseTo(11);
  expect(snapshot.scanPose.position.x).toBeCloseTo(10);
  expect(snapshot.odomPose.position.x).toBeCloseTo(3);
  expect(snapshot.odomStamp).toEqual({ sec: 11, nanosec: 0 });
});

test('matches Mapping odometry stamps at nanosecond precision', () => {
  const synchronizer = createMappingPoseSynchronizer();
  synchronizer.addOdometry(odometryMessage(10, 200_000_000, pose2d(100)));
  synchronizer.addOdometry(odometryMessage(10, 100_000_000, pose2d(1)));
  synchronizer.addSlamPose(slamPoseMessage(10, 100_000_000, pose2d(5)));

  const snapshot = synchronizer.snapshot({ sec: 10, nanosec: 100_000_000 });
  expect(snapshot.scanPose.position.x).toBeCloseTo(5);
  expect(snapshot.pose.position.x).toBeCloseTo(104);
});

test('composes Mapping translation through a rotated map to odom anchor', () => {
  const synchronizer = createMappingPoseSynchronizer();
  synchronizer.addOdometry(odometryMessage(10, 0, pose2d(1, 0, 0)));
  synchronizer.addSlamPose(slamPoseMessage(10, 0, pose2d(0, 0, Math.PI / 2)));
  synchronizer.addOdometry(odometryMessage(11, 0, pose2d(2, 0, 0)));

  const snapshot = synchronizer.snapshot({ sec: 10, nanosec: 0 });
  expect(snapshot.mapToOdomPose.position.x).toBeCloseTo(0);
  expect(snapshot.mapToOdomPose.position.y).toBeCloseTo(-1);
  expect(snapshot.pose.position.x).toBeCloseTo(0);
  expect(snapshot.pose.position.y).toBeCloseTo(1);
});

test('waits for matching odometry instead of pairing a stale SLAM pose', () => {
  const synchronizer = createMappingPoseSynchronizer({ maxSkewSeconds: 0.1 });
  synchronizer.addSlamPose(slamPoseMessage(10, 0, pose2d(5)));
  synchronizer.addOdometry(odometryMessage(11, 0, pose2d(1)));

  expect(synchronizer.snapshot({ sec: 10, nanosec: 0 })).toMatchObject({
    pose: null,
    scanPose: null,
    mapToOdomPose: null,
  });
});

test('waits for odometry to reach a SLAM timestamp before choosing the nearest sample', () => {
  const synchronizer = createMappingPoseSynchronizer({ maxSkewSeconds: 0.25 });
  synchronizer.addOdometry(odometryMessage(9, 800_000_000, pose2d(1)));
  synchronizer.addSlamPose(slamPoseMessage(10, 0, pose2d(5)));

  expect(synchronizer.snapshot({ sec: 10, nanosec: 0 }).pose).toBeNull();

  synchronizer.addOdometry(odometryMessage(10, 0, pose2d(2)));
  const snapshot = synchronizer.snapshot({ sec: 10, nanosec: 0 });
  expect(snapshot.mapToOdomPose.position.x).toBeCloseTo(3);
  expect(snapshot.pose.position.x).toBeCloseTo(5);
  expect(snapshot.scanPose.position.x).toBeCloseTo(5);
});

test('ignores Mapping poses whose frames do not match map, odom, and base_link', () => {
  const synchronizer = createMappingPoseSynchronizer();
  synchronizer.addOdometry({
    ...odometryMessage(10, 0, pose2d(2)),
    header: { stamp: { sec: 10, nanosec: 0 }, frame_id: 'map' },
  });
  synchronizer.addSlamPose({
    ...slamPoseMessage(10, 0, pose2d(5)),
    header: { stamp: { sec: 10, nanosec: 0 }, frame_id: 'odom' },
  });

  expect(synchronizer.snapshot({ sec: 10, nanosec: 0 })).toMatchObject({
    pose: null,
    scanPose: null,
    mapToOdomPose: null,
  });
});

test('does not reuse a volatile SLAM pose after a session reset', () => {
  const synchronizer = createMappingPoseSynchronizer();
  const odom = odometryMessage(10, 0, pose2d(2));
  const slam = slamPoseMessage(10, 0, pose2d(5));
  synchronizer.addOdometry(odom);
  synchronizer.addSlamPose(slam);
  expect(synchronizer.snapshot().pose).not.toBeNull();

  synchronizer.clear({ preserveSeenKeys: true });
  expect(synchronizer.addOdometry(odom)).toBe(false);
  expect(synchronizer.addSlamPose(slam)).toBe(false);
  expect(synchronizer.snapshot()).toMatchObject({ pose: null, mapToOdomPose: null });

  synchronizer.addOdometry(odometryMessage(11, 0, pose2d(3)));
  synchronizer.addSlamPose(slamPoseMessage(11, 0, pose2d(6)));
  expect(synchronizer.snapshot().pose.position.x).toBeCloseTo(6);
});

test('replaces only the map to odom TF edge with the synchronized correction', () => {
  const tf = {
    transforms: [
      {
        header: { frame_id: 'map' },
        child_frame_id: 'odom',
        transform: { translation: { x: 99, y: 0, z: 0 } },
      },
      {
        header: { frame_id: 'odom' },
        child_frame_id: 'base_link',
        transform: { translation: { x: 2, y: 0, z: 0 } },
      },
    ],
  };

  const corrected = replaceTfFramePose(tf, 'map', 'odom', pose2d(8), { sec: 10, nanosec: 0 });
  expect(corrected.transforms).toHaveLength(2);
  expect(corrected.transforms.find((item) => item.child_frame_id === 'odom')).toMatchObject({
    header: { frame_id: 'map', stamp: { sec: 10, nanosec: 0 } },
    transform: { translation: { x: 8 } },
  });
  expect(corrected.transforms.find((item) => item.child_frame_id === 'base_link')).toBe(
    tf.transforms[1]
  );
});
