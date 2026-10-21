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

import React, { useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import ROSLIB from 'roslib';
import rosConnectionManager from '../../utils/rosConnectionManager';
import { captureSg2Pose } from '../../utils/sg2PoseCapture';

// Request a new feedback message on each click; never reuse a cached pose.
export default function PoseCaptureButton({ nodeType, params, onCapture, onSnapshot }) {
  const url = useSelector((state) => state.ros?.rosbridgeUrl);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const cancelRef = useRef(null);
  useEffect(() => {
    setBusy(false);
    setError('');
    return () => cancelRef.current?.();
  }, [url]);

  const capture = async () => {
    if (cancelRef.current || !url) return;
    setBusy(true);
    setError('');
    let active = true;
    let subscription;
    const finish = () => {
      active = false;
      clearTimeout(timer);
      subscription?.unsubscribe();
      cancelRef.current = null;
    };
    const fail = (message) => {
      if (!active) return;
      finish();
      setBusy(false);
      setError(message);
    };
    const timer = setTimeout(() => fail('No fresh /joint_states received within 3 seconds.'), 3000);
    cancelRef.current = finish;
    try {
      const ros = await rosConnectionManager.getConnection(url);
      if (!active) return;
      if (!ros?.isConnected) throw new Error('ROS connection is not available.');
      subscription = new ROSLIB.Topic({
        ros, name: '/joint_states', messageType: 'sensor_msgs/msg/JointState',
        queue_length: 1, throttle_rate: 0,
      });
      subscription.subscribe((message) => {
        if (!active) return;
        try {
          if (!ros.isConnected) throw new Error('ROS connection is not available.');
          const patch = captureSg2Pose(nodeType, params, message);
          finish();
          setBusy(false);
          onSnapshot?.({ name: [...message.name], position: [...message.position] });
          onCapture(patch);
        } catch (err) {
          fail(err.message);
        }
      });
    } catch (err) {
      fail(err.message || 'Could not capture the robot pose.');
    }
  };

  return (
    <div className="px-4 py-3 border-b border-[var(--mc-border,#e5e7eb)]">
      <button type="button" onClick={capture} disabled={busy || !url}
        className="w-full px-3 py-2 rounded-lg bg-[var(--mc-accent,#2563eb)] text-white text-sm disabled:opacity-50">
        {busy ? 'Capturing…' : 'Capture current pose'}
      </button>
      <p className="mt-1 text-xs text-[var(--mc-text-muted)]">
        {nodeType === 'ArmStateGate' ? 'Select target joints to capture.' : 'Captures enabled joint groups.'}
        {' '}Arms, grippers and head: rad. Lift: m.
      </p>
      {!url && <p className="text-xs" role="status">Connect to ROS to capture a pose.</p>}
      {error && <p className="mt-1 text-xs text-red-600" role="alert">{error}</p>}
    </div>
  );
}
