import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import RobotTypeSelector from './RobotTypeSelector';
import taskReducer from '../features/tasks/taskSlice';
import uiReducer from '../features/ui/uiSlice';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));
jest.mock('react-hot-toast', () => ({
  __esModule: true, default: { error: jest.fn(), success: jest.fn() },
}));

test.each([RecordPhase.RECORDING, RecordPhase.SAVING])(
  'inference Clear does not unlock robot changes during recording phase %s', async (recordPhase) => {
    const initial = taskReducer(undefined, { type: '@@INIT' });
    const store = configureStore({ reducer: { tasks: taskReducer, ui: uiReducer }, preloadedState: { tasks: {
      ...initial, robotType: 'test_robot',
      recordStatus: { ...initial.recordStatus, recordPhase },
      inferenceStatus: { ...initial.inferenceStatus, inferencePhase: InferencePhase.READY, runtimeState: 'unloaded' },
    } } });
    const setRobotType = jest.fn();
    useRosServiceCaller.mockReturnValue({
      getRobotTypeList: jest.fn().mockResolvedValue({ robot_types: ['test_robot'] }), setRobotType,
    });
    render(<Provider store={store}><RobotTypeSelector /></Provider>);
    expect(await screen.findByRole('option', { name: 'test_robot' })).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toBeDisabled();
    const button = screen.getByRole('button', { name: 'Set Robot Type' });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(setRobotType).not.toHaveBeenCalled();
  },
);
