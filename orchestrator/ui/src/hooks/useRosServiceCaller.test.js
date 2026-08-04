import {
  getRecordCommandServiceTimeoutMs,
  transformReplayDataResult,
} from './useRosServiceCaller';

describe('getRecordCommandServiceTimeoutMs', () => {
  test('does not time out recording save commands', () => {
    expect(getRecordCommandServiceTimeoutMs('stop_segment')).toBe(0);
    expect(getRecordCommandServiceTimeoutMs('finish_episode')).toBe(0);
    expect(getRecordCommandServiceTimeoutMs('stop_inference_record')).toBe(0);
  });

  test('keeps shorter defaults for non-save commands', () => {
    expect(getRecordCommandServiceTimeoutMs('refresh_topics')).toBe(10000);
    expect(getRecordCommandServiceTimeoutMs('start_inference')).toBe(30000);
  });

  test('allows callers to override the service timeout', () => {
    expect(getRecordCommandServiceTimeoutMs('stop_segment', {
      serviceTimeoutMs: 45000,
    })).toBe(45000);
  });
});

describe('transformReplayDataResult', () => {
  test('preserves replay robot metadata for the 3D viewer', () => {
    const result = transformReplayDataResult(
      {
        success: true,
        robot_type: 'ffw_sh5_rev1',
        urdf_path: '/workspace/robot_configs/urdf/ffw_sh5_follower.urdf',
        end_effector_links: ['tool0'],
      },
      '/workspace/rosbag2/sh5/0'
    );

    expect(result.robot_type).toBe('ffw_sh5_rev1');
    expect(result.urdf_path).toBe(
      '/workspace/robot_configs/urdf/ffw_sh5_follower.urdf'
    );
    expect(result.end_effector_links).toEqual(['tool0']);
    expect(result.bag_path).toBe('/workspace/rosbag2/sh5/0');
  });
});
