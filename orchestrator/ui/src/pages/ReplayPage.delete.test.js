import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import ReplayPage from './ReplayPage';
import replay from '../features/replay/replaySlice';
import layout from '../features/layout/layoutSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller');
jest.mock('../hooks/useMcapFramePlayer', () => ({ useMcapFramePlayer: () => ({}) }));
jest.mock('../components/replay/Viewer3DPanel', () => () => null);
jest.mock('../components/replay/JointDataPanel', () => () => null);
jest.mock('../components/layout/ReplayLayoutContainer', () => () => null);
jest.mock('../components/FileBrowserModal', () => ({ onFileSelect }) => (
  <button onClick={() => onFileSelect({ full_path: '/workspace/rosbag2/task' })}>Open task</button>
));

const task = '/workspace/rosbag2/task';
const listing = names => ({
  success: true, rosbags: names.map(name => ({ name, path: `${task}/${name}` })),
});
const list = jest.fn();
const edit = jest.fn();
const load = jest.fn();

async function openTask(names) {
  list.mockResolvedValueOnce(listing(names));
  const store = configureStore({ reducer: { replay, layout, ros: () => ({}) } });
  const view = render(<Provider store={store}><ReplayPage isActive /></Provider>);
  fireEvent.click(screen.getByText('Select Task'));
  fireEvent.click(screen.getByText('Open task'));
  await waitFor(() => expect(store.getState().replay.isLoaded).toBe(true));
  return view;
}

beforeEach(() => {
  jest.resetAllMocks();
  jest.spyOn(window, 'confirm').mockReturnValue(true);
  load.mockImplementation(async path => ({ success: true, bag_path: path }));
  useRosServiceCaller.mockReturnValue({ getReplayData: load, getRosbagList: list, sendEditDatasetCommand: edit });
});
afterEach(() => jest.restoreAllMocks());

test('deletes the reviewed directory number, prevents repeat clicks, and opens its successor', async () => {
  await openTask(['9', '0', '4']);
  fireEvent.click(screen.getByText('Next'));
  const button = await screen.findByRole('button', { name: 'Delete Episode 4' });
  let finish;
  edit.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  list.mockResolvedValueOnce(listing(['0', '1']));
  fireEvent.click(button);
  fireEvent.click(button);
  expect(edit).toHaveBeenCalledTimes(1);
  expect(edit).toHaveBeenCalledWith('delete', {
    deleteTaskDir: task, deleteEpisodeNums: [4], deleteCompact: true,
  });
  expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining(task + '/4'));
  await act(async () => finish({ success: true }));
  expect(await screen.findByRole('button', { name: 'Delete Episode 1' })).toBeTruthy();
  expect(load).toHaveBeenLastCalledWith(task + '/1');
});

test('cancel and refused deletion retain the episode; deleting the last leaves an empty task', async () => {
  await openTask(['0']);
  const button = screen.getByRole('button', { name: 'Delete Episode 0' });
  window.confirm.mockReturnValueOnce(false);
  fireEvent.click(button);
  expect(edit).not.toHaveBeenCalled();
  edit.mockResolvedValueOnce({ success: false, message: 'Recording active' });
  fireEvent.click(button);
  await waitFor(() => expect(button.disabled).toBe(false));
  expect(list).toHaveBeenCalledTimes(1);
  edit.mockResolvedValueOnce({ success: true });
  list.mockResolvedValueOnce(listing([]));
  fireEvent.click(button);
  expect(await screen.findByText('No episodes to review in this task.')).toBeTruthy();
  expect(screen.queryByRole('button', { name: /Delete Episode/ })).toBeNull();
});

test('reloads previously viewed videos when surviving episodes reuse their paths', async () => {
  const previous = [global.fetch, URL.createObjectURL, URL.revokeObjectURL];
  let blobCount = 0;
  URL.createObjectURL = jest.fn(() => 'blob:' + ++blobCount);
  URL.revokeObjectURL = jest.fn();
  global.fetch = jest.fn(async () => ({
    ok: true, headers: new Headers(), body: { getReader: () => ({ read: async () => ({ done: true }) }) },
  }));
  load.mockImplementation(async path => ({ success: true, bag_path: path, video_files: ['cam.mp4'] }));
  let view;
  try {
    view = await openTask(['0', '1', '2']);
    await waitFor(() => expect(blobCount).toBe(1));
    fireEvent.click(screen.getByText('Next'));
    await waitFor(() => expect(blobCount).toBe(2));
    fireEvent.click(screen.getByText('Prev'));
    const button = await screen.findByRole('button', { name: 'Delete Episode 0' });
    edit.mockResolvedValueOnce({ success: true });
    list.mockResolvedValueOnce(listing(['0', '1']));
    fireEvent.click(button);
    await waitFor(() => expect(blobCount).toBe(3));
    fireEvent.click(screen.getByText('Next'));
    await waitFor(() => expect(blobCount).toBe(4));
    const urls = global.fetch.mock.calls.map(([url]) => url);
    expect(urls.map(url => url.split('?')[0])).toEqual([0, 1, 0, 1].map(n => `/files${task}/${n}/cam.mp4`));
    expect(urls[0]).not.toBe(urls[2]);
    expect(urls[1]).not.toBe(urls[3]);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:1');
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:2');
  } finally {
    // Release the page's remaining object URLs before restoring jsdom globals.
    view?.unmount();
    [global.fetch, URL.createObjectURL, URL.revokeObjectURL] = previous;
  }
});

test.each([['65536'], ['custom'], ['1', '01']])('does not offer deletion for unmatched or ambiguous index %s', async (...names) => {
  await openTask(names);
  expect(screen.queryByRole('button', { name: /Delete Episode/ })).toBeNull();
});
