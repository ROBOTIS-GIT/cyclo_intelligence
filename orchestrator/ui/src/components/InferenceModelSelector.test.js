import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferenceModelSelector from './InferenceModelSelector';
import taskReducer from '../features/tasks/taskSlice';

describe('InferenceModelSelector RLT compatibility', () => {
  test('disables RLT when leaving N1.7 while retaining the selected bundle', () => {
    const initialTasks = taskReducer(undefined, { type: '@@INIT' });
    const store = configureStore({
      reducer: { tasks: taskReducer },
      preloadedState: {
        tasks: {
          ...initialTasks,
          inferenceTaskInfo: {
            ...initialTasks.inferenceTaskInfo,
            serviceType: 'groot',
            policyType: 'n17',
            rltEnabled: true,
            rltBundlePath: '/workspace/checkpoint/rlt/showroom_groot_bundle',
            rltRobotOverride: true,
          },
        },
      },
    });

    render(
      <Provider store={store}>
        <InferenceModelSelector />
      </Provider>
    );

    fireEvent.change(screen.getByRole('combobox'), {
      target: { value: 'lerobot:act' },
    });

    const info = store.getState().tasks.inferenceTaskInfo;
    expect(info.serviceType).toBe('lerobot');
    expect(info.policyType).toBe('act');
    expect(info.rltEnabled).toBe(false);
    expect(info.rltRobotOverride).toBe(false);
    expect(info.rltBundlePath).toBe(
      '/workspace/checkpoint/rlt/showroom_groot_bundle'
    );
  });
});
