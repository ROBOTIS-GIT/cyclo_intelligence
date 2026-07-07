import { configureStore } from '@reduxjs/toolkit';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import RecordTopicMonitor from './RecordTopicMonitor';
import taskReducer from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

const renderMonitor = (recordingMonitor) => {
  useRosServiceCaller.mockReturnValue({ sendRecordCommand: jest.fn() });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: {
      tasks: taskReducer,
    },
    preloadedState: {
      tasks: {
        ...initialTasks,
        recordingMonitor,
      },
    },
  });

  render(
    <Provider store={store}>
      <RecordTopicMonitor />
    </Provider>
  );
};

describe('RecordTopicMonitor', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders camera monitor rows even without rosbag monitor rows', () => {
    renderMonitor({
      topics: [],
      cameraTopics: [
        {
          name: '/zed/zed_node/left/image_rect_color/compressed',
          cameraName: 'cam_left_head',
          rateHz: 29.8,
          baselineHz: 30,
          secondsSinceLast: 0.1,
          status: 0,
          timestampStatus: 0,
          timestampSkewS: -0.01,
          source: 'camera',
        },
      ],
      totalReceived: 0,
      totalWritten: 0,
    });

    expect(screen.getByText('Topic Monitor')).toBeInTheDocument();
    expect(screen.getByText('.../image_rect_color/compressed')).toBeInTheDocument();
    expect(screen.getByText('29.8')).toBeInTheDocument();
    expect(screen.getByText('All 1 OK')).toBeInTheDocument();
  });

  test('labels camera timestamp skew in the status tooltip', () => {
    renderMonitor({
      topics: [],
      cameraTopics: [
        {
          name: '/camera/image/compressed',
          cameraName: 'cam_right_wrist',
          rateHz: 30,
          baselineHz: 30,
          secondsSinceLast: 0,
          status: 2,
          timestampStatus: 2,
          timestampSkewS: 1.25,
          source: 'camera',
        },
      ],
      totalReceived: 0,
      totalWritten: 0,
    });

    expect(screen.getByTitle(/Timestamp skew/)).toBeInTheDocument();
    expect(screen.getByTitle(/skew 1.250s/)).toBeInTheDocument();
  });
});
