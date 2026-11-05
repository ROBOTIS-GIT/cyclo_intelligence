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

import { useCallback, useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import ROSLIB from 'roslib';
import rosConnectionManager from '../utils/rosConnectionManager';

const TOPIC_TYPES = {
  '/map': 'nav_msgs/msg/OccupancyGrid',
  '/global_costmap/costmap': 'nav_msgs/msg/OccupancyGrid',
  '/local_costmap/costmap': 'nav_msgs/msg/OccupancyGrid',
  '/local_costmap/published_footprint': 'geometry_msgs/msg/PolygonStamped',
  '/scan': 'sensor_msgs/msg/LaserScan',
  '/pose': 'geometry_msgs/msg/PoseWithCovarianceStamped',
  '/odom': 'nav_msgs/msg/Odometry',
  '/amcl_pose': 'geometry_msgs/msg/PoseWithCovarianceStamped',
  '/plan': 'nav_msgs/msg/Path',
  '/tf': 'tf2_msgs/msg/TFMessage',
  '/tf_static': 'tf2_msgs/msg/TFMessage',
  '/bt/status': 'std_msgs/msg/String',
  '/bt/active_nodes': 'std_msgs/msg/String',
};

const SERVER_GRID_TOPICS = new Set([
  '/map',
  '/global_costmap/costmap',
]);

export function wrapNavigationRosMessage(message) {
  return { available: true, data: message };
}

export function navigationGridWebSocketUrl(topic, location = window.location) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${location.host}/api/navigation/topics/ws?topic=${encodeURIComponent(topic)}`;
}

export function applyNavigationGridEnvelope(previous, incoming) {
  if (!incoming?.available || !incoming.update) return incoming;
  const grid = previous?.available ? previous.data : null;
  const info = grid?.info;
  const update = incoming.update;
  const gridWidth = Number(info?.width || 0);
  const gridHeight = Number(info?.height || 0);
  const x = Number(update.x);
  const y = Number(update.y);
  const width = Number(update.width);
  const height = Number(update.height);
  if (
    !grid
    || !Array.isArray(grid.data)
    || !Number.isInteger(x)
    || !Number.isInteger(y)
    || !Number.isInteger(width)
    || !Number.isInteger(height)
    || x < 0
    || y < 0
    || width <= 0
    || height <= 0
    || x + width > gridWidth
    || y + height > gridHeight
    || !Array.isArray(update.data)
    || update.data.length !== width * height
  ) {
    return previous;
  }

  const data = grid.data.slice();
  for (let row = 0; row < height; row += 1) {
    const sourceStart = row * width;
    const targetStart = (y + row) * gridWidth + x;
    for (let column = 0; column < width; column += 1) {
      data[targetStart + column] = update.data[sourceStart + column];
    }
  }
  const updateHeader = update.header && Object.keys(update.header).length
    ? update.header
    : grid.header;
  return {
    available: true,
    data: {
      ...grid,
      header: updateHeader,
      data,
      updateRegion: {
        x,
        y,
        width,
        height,
      },
    },
  };
}

/** Subscribe to a Navigation ROS topic through the app-wide rosbridge connection. */
export function useNavigationRosTopic(topic, options = {}) {
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const [topicData, setTopicData] = useState(null);
  const [status, setStatus] = useState('disconnected');

  useEffect(() => {
    const usesServerGridSocket = SERVER_GRID_TOPICS.has(topic);
    const staleMs = Math.max(0, Number(options.staleMs || 0));
    if (!topic || (!usesServerGridSocket && !rosbridgeUrl)) {
      setTopicData(null);
      setStatus('disconnected');
      return undefined;
    }

    let mounted = true;
    let subscription = null;
    let staleTimer = null;
    const resetStaleTimer = () => {
      if (staleTimer) window.clearTimeout(staleTimer);
      if (!staleMs) return;
      staleTimer = window.setTimeout(() => {
        if (mounted) setTopicData(null);
      }, staleMs);
    };

    if (usesServerGridSocket) {
      setStatus('connecting');
      const socket = new WebSocket(navigationGridWebSocketUrl(topic));
      socket.onopen = () => {
        if (mounted) setStatus('connected');
      };
      socket.onmessage = (event) => {
        if (!mounted) return;
        try {
          const incoming = JSON.parse(event.data);
          setTopicData((previous) => applyNavigationGridEnvelope(previous, incoming));
          resetStaleTimer();
        } catch (error) {
          console.error(`Failed to decode Navigation grid ${topic}:`, error);
          setStatus('error');
        }
      };
      socket.onerror = () => {
        if (mounted) setStatus('error');
      };
      socket.onclose = () => {
        if (mounted) setStatus('disconnected');
      };
      return () => {
        mounted = false;
        if (staleTimer) window.clearTimeout(staleTimer);
        socket.close();
      };
    }

    const subscribe = async () => {
      setStatus('connecting');
      try {
        const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
        if (!mounted) return;
        const messageType = TOPIC_TYPES[topic];
        if (!messageType) throw new Error(`Unknown Navigation topic: ${topic}`);
        subscription = new ROSLIB.Topic({
          ros,
          name: topic,
          messageType,
          throttle_rate: Math.max(0, Number(options.throttleMs || 0)),
          queue_length: 1,
          compression: 'none',
        });
        subscription.subscribe((message) => {
          // Preserve the page's transport envelope. OccupancyGrid itself has
          // a `data` array, so passing the raw message would be mistaken for
          // an envelope and discard its header/info fields.
          if (mounted) {
            setTopicData(wrapNavigationRosMessage(message));
            resetStaleTimer();
          }
        });
        setStatus('connected');
      } catch (error) {
        if (mounted) {
          console.error(`Failed to subscribe to ${topic}:`, error);
          setStatus('error');
        }
      }
    };

    subscribe();
    return () => {
      mounted = false;
      if (staleTimer) window.clearTimeout(staleTimer);
      if (subscription) subscription.unsubscribe();
    };
  }, [options.staleMs, options.throttleMs, rosbridgeUrl, topic]);

  return { status, topicData };
}

/** Publish a ROS message through the same singleton rosbridge connection. */
export function useNavigationRosPublisher() {
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);

  return useCallback(async (topic, messageType, data) => {
    const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
    if (!ros || !ros.isConnected) {
      throw new Error('ROS connection is not available');
    }
    const publisher = new ROSLIB.Topic({ ros, name: topic, messageType });
    publisher.publish(new ROSLIB.Message(data));
    window.setTimeout(() => publisher.unadvertise(), 250);
  }, [rosbridgeUrl]);
}
