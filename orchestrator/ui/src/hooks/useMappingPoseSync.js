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

import { useCallback, useEffect, useRef, useState } from 'react';
import { createMappingPoseSynchronizer } from '../utils/navigationTf';

/**
 * Synchronize slam_toolbox's scan-matched /pose with continuous /odom.
 *
 * /pose provides the authoritative map-frame correction, while /odom carries
 * smooth motion between accepted SLAM scans. A short odometry history also
 * lets MapViewer project each LaserScan near its own timestamp.
 */
export function useMappingPoseSync({ active, slamPose, odometry, scanStamp }) {
  const synchronizerRef = useRef(null);
  if (!synchronizerRef.current) {
    synchronizerRef.current = createMappingPoseSynchronizer();
  }
  const [, setRevision] = useState(0);

  const reset = useCallback(() => {
    // Topic hooks can still hold the last volatile /pose while a runtime is
    // stopped. Preserve seen stamps so that value cannot silently re-anchor a
    // new Mapping session; only a newly published pose may establish it.
    synchronizerRef.current.clear({ preserveSeenKeys: true });
    setRevision((value) => value + 1);
  }, []);

  useEffect(() => {
    const synchronizer = synchronizerRef.current;
    if (!active) {
      if (synchronizer.clear({ preserveSeenKeys: true })) {
        setRevision((value) => value + 1);
      }
      return;
    }
    let changed = false;
    if (odometry) changed = synchronizer.addOdometry(odometry) || changed;
    if (slamPose) changed = synchronizer.addSlamPose(slamPose) || changed;
    if (changed) setRevision((value) => value + 1);
  }, [active, odometry, slamPose]);

  return {
    ...synchronizerRef.current.snapshot(scanStamp),
    reset,
  };
}
