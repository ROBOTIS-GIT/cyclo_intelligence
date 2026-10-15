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

import { useEffect } from 'react';
import { setBtStatus, setActiveNodeNames } from '../actionCanvasSlice';

export default function useBtRuntimeSubscription({ rosbridgeUrl, isActive, dispatch }) {
  // ── BT status / active-nodes subscription ────────────────────────────────
  useEffect(() => {
    if (!rosbridgeUrl || !isActive) return;

    let disposed = false;
    let statusTopic = null;
    let activeNodesTopic = null;

    const setupSubscription = async () => {
      try {
        const ROSLIB = (await import('roslib')).default;
        if (disposed) return;
        const { default: rosConnectionManager } = await import('../../../utils/rosConnectionManager');
        if (disposed) return;
        const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
        if (disposed) return;

        statusTopic = new ROSLIB.Topic({
          ros,
          name: '/bt/status',
          messageType: 'std_msgs/msg/String',
        });
        statusTopic.subscribe((msg) => {
          if (disposed) return;
          dispatch(setBtStatus(msg.data));
          if (msg.data !== 'running') dispatch(setActiveNodeNames([]));
        });

        activeNodesTopic = new ROSLIB.Topic({
          ros,
          name: '/bt/active_nodes',
          messageType: 'std_msgs/msg/String',
        });
        activeNodesTopic.subscribe((msg) => {
          if (disposed) return;
          const names = msg.data ? msg.data.split(',') : [];
          dispatch(setActiveNodeNames(names));
        });
      } catch (err) {
        if (!disposed) console.debug('BT status subscription not available:', err.message);
      }
    };

    setupSubscription();
    return () => {
      // Invalidate pending setup and queued topic callbacks before unsubscribing.
      // The connection manager owns the shared connection; never close it here.
      disposed = true;
      if (statusTopic) statusTopic.unsubscribe();
      if (activeNodesTopic) activeNodesTopic.unsubscribe();
    };
  }, [rosbridgeUrl, isActive, dispatch]);

}
