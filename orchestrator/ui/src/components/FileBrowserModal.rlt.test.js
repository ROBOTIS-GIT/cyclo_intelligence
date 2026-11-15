import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import FileBrowserModal from './FileBrowserModal';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));

const root = '/workspace/checkpoint/rlt/stage2';
const files = ['manifest.json'];
const folders = ['artifacts', 'training_state', 'replay_cache'];

test.each(['actor', 'critic', 'rl_training'])('navigates %s and selects only a completed bundle', async (phase) => {
  const browseFile = jest.fn(async (_action, path) => ({
    success: true,
    current_path: path,
    parent_path: root,
    items: path === root
      ? [{ name: phase, full_path: `${root}/${phase}`, is_directory: true, has_target_file: false }]
      : [
        { name: 'latest', full_path: `${root}/${phase}/latest`, is_directory: true, has_target_file: true },
        { name: 'incomplete', full_path: `${root}/${phase}/incomplete`, is_directory: true, has_target_file: false },
      ],
  }));
  useRosServiceCaller.mockReturnValue({ browseFile });
  const select = jest.fn();
  render(<FileBrowserModal isOpen initialPath={root} onClose={jest.fn()} onFileSelect={select}
    title="RLT bundles" selectButtonText="Use Bundle" allowDirectorySelect allowFileSelect={false}
    targetFileName={files} targetFolderName={folders} requireTargetForSelection />);
  fireEvent.click(await screen.findByText(phase));
  await screen.findByText('latest');
  expect(screen.getAllByRole('button', { name: 'Select' })).toHaveLength(1);
  expect(screen.getByRole('button', { name: 'Use Bundle' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Select' }));
  fireEvent.click(screen.getByRole('button', { name: 'Use Bundle' }));
  expect(select).toHaveBeenCalledWith(expect.objectContaining({ full_path: `${root}/${phase}/latest` }));
  expect(browseFile).toHaveBeenLastCalledWith('browse', `${root}/${phase}`, '', files, folders);
});
