import { RecordPhase } from '../constants/taskPhases';
import {
  buildCameraMonitorTopics,
  isMonitorOnlyStatusMessage,
  shouldAnnounceRecordingStart,
} from './recordingStatusMonitor';

describe('recordingStatusMonitor helpers', () => {
  test('builds camera monitor rows from structured backend status', () => {
    const rows = buildCameraMonitorTopics({
      camera_monitor_names: ['cam_left_head'],
      camera_monitor_topics: ['/zed/zed_node/left/image_rect_color/compressed'],
      camera_monitor_rates_hz: [30],
      camera_monitor_baseline_hz: [30],
      camera_monitor_seconds_since_last: [0.1],
      camera_monitor_status: [2],
      camera_monitor_timestamp_status: [2],
      camera_monitor_timestamp_skew_s: [1.25],
    });

    expect(rows).toEqual([
      expect.objectContaining({
        name: '/zed/zed_node/left/image_rect_color/compressed',
        cameraName: 'cam_left_head',
        status: 2,
        statusLabel: 'Timestamp skew',
        timestampStatus: 2,
        timestampSkewS: 1.25,
      }),
    ]);
  });

  test('decodes rosbridge base64 uint8 arrays', () => {
    const rows = buildCameraMonitorTopics({
      camera_monitor_names: ['cam_left_head'],
      camera_monitor_topics: ['/camera/image/compressed'],
      camera_monitor_status: btoa(String.fromCharCode(2)),
      camera_monitor_timestamp_status: btoa(String.fromCharCode(0)),
    });

    expect(rows[0].status).toBe(2);
    expect(rows[0].statusLabel).toBe('Stalled');
  });

  test('detects monitor-only fake messages so they do not perturb record phase', () => {
    const msg = {
      record_phase: RecordPhase.READY,
      robot_type: 'ffw_sg2_rev1',
      task_info: {},
      camera_monitor_topics: ['/camera/image/compressed'],
      camera_monitor_status: [2],
      camera_monitor_timestamp_status: [2],
      camera_monitor_timestamp_skew_s: [1.25],
    };

    expect(isMonitorOnlyStatusMessage(msg, buildCameraMonitorTopics(msg))).toBe(true);
  });

  test('does not treat the first recording status as a start transition', () => {
    expect(shouldAnnounceRecordingStart({
      hasSeenRecordPhase: false,
      previousPhase: null,
      currentPhase: RecordPhase.RECORDING,
      proceedTime: 0,
    })).toBe(false);
  });

  test('announces only fresh ready to recording transitions', () => {
    expect(shouldAnnounceRecordingStart({
      hasSeenRecordPhase: true,
      previousPhase: RecordPhase.READY,
      currentPhase: RecordPhase.RECORDING,
      proceedTime: 0,
    })).toBe(true);

    expect(shouldAnnounceRecordingStart({
      hasSeenRecordPhase: true,
      previousPhase: RecordPhase.READY,
      currentPhase: RecordPhase.RECORDING,
      proceedTime: 26,
    })).toBe(false);
  });
});
