import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import DatasetConvertSection from './DatasetConvertSection';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';

jest.mock('../../../hooks/useRosServiceCaller');
jest.mock('../../../components/FileBrowserModal', () => ({ isOpen, onFileSelect }) => (
  isOpen && <button onClick={() => onFileSelect({ full_path: '/workspace/rosbag2/nested/Task_A/' })}>Pick Task A</button>
));

function setup() {
  const send = jest.fn(() => new Promise(() => {}));
  useRosServiceCaller.mockReturnValue({ sendRecordCommand: send });
  const store = configureStore({ reducer: { editDataset: () => ({ conversionStatus: { status: 'idle' } }) } });
  render(<Provider store={store}><DatasetConvertSection /></Provider>);
  fireEvent.click(screen.getByRole('button', { name: 'Browse for task folder' }));
  fireEvent.click(screen.getByText('Pick Task A'));
  return send;
}

test('prefills the source basename and submits the edited stem separately from the source', () => {
  const send = setup();
  const output = screen.getByLabelText('Output dataset name');
  expect(output.value).toBe('Task_A_lerobot');
  fireEvent.change(output, { target: { value: 'Peanut' } });
  fireEvent.click(screen.getByLabelText('LeRobot v2.1'));
  expect(screen.queryByText('/workspace/lerobot/Peanut_v21')).toBeNull();
  expect(screen.getByText('/workspace/lerobot/Peanut_v30')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Convert Dataset' }));
  expect(send).toHaveBeenCalledWith('convert_mp4', expect.objectContaining({
    conversionSource: 'nested/Task_A', conversionOutputName: 'Peanut', convertV21: false, convertV30: true,
  }));
});

test('invalid names disable conversion and changing source restores its default name', () => {
  setup();
  const output = screen.getByLabelText('Output dataset name');
  const button = screen.getByRole('button', { name: 'Convert Dataset' });
  for (const value of ['', '../tmp', 'x/y', ' ']) {
    fireEvent.change(output, { target: { value } });
    expect(button.disabled).toBe(true);
  }
  fireEvent.change(screen.getByPlaceholderText(/e.g. Task_/), { target: { value: 'NewTask' } });
  expect(output.value).toBe('NewTask_lerobot');
  expect(button.disabled).toBe(false);
});
