import { configureStore } from '@reduxjs/toolkit';
import { render, screen, within, act } from '@testing-library/react';
import { Provider } from 'react-redux';
import taskReducer, { setRecordStatus } from '../features/tasks/taskSlice';
import RecordControlPanel from './RecordControlPanel';

function setup(status) {
  const store = configureStore({ reducer: { tasks: taskReducer } });
  store.dispatch(setRecordStatus(status));
  render(<Provider store={store}><RecordControlPanel /></Provider>);
  return store;
}

test('episode is a bordered labeled badge and preserves the backend number on updates', () => {
  const store = setup({ currentEpisodeNumber: 0 });
  const group = screen.getByRole('group', { name: 'Episode' });
  expect(group).toHaveClass('border', 'border-blue-400', 'bg-blue-50', 'dark:text-blue-200');
  expect(within(group).getByText('Episode')).toBeVisible();
  expect(within(group).getByText('0')).toHaveClass('font-bold', 'text-base', 'border-l');
  act(() => store.dispatch(setRecordStatus({ currentEpisodeNumber: 1234 })));
  expect(within(group).getByText('1234')).toBeVisible();
});

test('subtask counter and instruction have an outer border and an internal divider', () => {
  const store = setup({ subtaskCount: 2, currentSubtaskIndex: 0, currentSubtaskInstruction: '1' });
  const group = screen.getByRole('group', { name: 'Current subtask' });
  expect(group).toHaveClass('border', 'border-gray-400');
  expect(within(group).getByText('ST 1/2')).toHaveClass('shrink-0', 'bg-gray-100');
  expect(within(group).getByText('1')).toHaveClass('border-l');
  act(() => store.dispatch(setRecordStatus({ currentSubtaskIndex: 1, currentSubtaskInstruction: 'Place object' })));
  expect(within(group).getByText('ST 2/2')).toBeVisible();
  expect(within(group).getByTitle('Place object')).toBeVisible();
});

test('long instructions retain a tooltip and can shrink without hiding the counter', () => {
  const instruction = 'Move the object to the destination '.repeat(15).trim();
  setup({ subtaskCount: 2, currentSubtaskInstruction: instruction });
  expect(screen.getByTitle(instruction)).toHaveClass('min-w-0', 'truncate');
  expect(screen.getByText('ST 1/2')).toBeVisible();
});

test('missing instructions leave only the counter; zero subtasks hide the group', () => {
  const store = setup({ subtaskCount: 2, currentSubtaskInstruction: '' });
  expect(screen.getByRole('group', { name: 'Current subtask' })).toHaveTextContent(/^ST 1\/2$/);
  act(() => store.dispatch(setRecordStatus({ subtaskCount: 0 })));
  expect(screen.queryByRole('group', { name: 'Current subtask' })).not.toBeInTheDocument();
});
