// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

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
  '/amcl_pose': 'geometry_msgs/msg/PoseWithCovarianceStamped',
  '/plan': 'nav_msgs/msg/Path',
  '/goal_pose': 'geometry_msgs/msg/PoseStamped',
  '/tf': 'tf2_msgs/msg/TFMessage',
  '/tf_static': 'tf2_msgs/msg/TFMessage',
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

/** Subscribe to a Navigation ROS topic through the app-wide rosbridge connection. */
export function useNavigationRosTopic(topic, options = {}) {
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const [topicData, setTopicData] = useState(null);
  const [status, setStatus] = useState('disconnected');

  useEffect(() => {
    const usesServerGridSocket = SERVER_GRID_TOPICS.has(topic);
    if (!topic || (!usesServerGridSocket && !rosbridgeUrl)) {
      setTopicData(null);
      setStatus('disconnected');
      return undefined;
    }

    let mounted = true;
    let subscription = null;

    if (usesServerGridSocket) {
      setStatus('connecting');
      const socket = new WebSocket(navigationGridWebSocketUrl(topic));
      socket.onopen = () => {
        if (mounted) setStatus('connected');
      };
      socket.onmessage = (event) => {
        if (!mounted) return;
        try {
          setTopicData(JSON.parse(event.data));
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
          if (mounted) setTopicData(wrapNavigationRosMessage(message));
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
      if (subscription) subscription.unsubscribe();
    };
  }, [options.throttleMs, rosbridgeUrl, topic]);

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
