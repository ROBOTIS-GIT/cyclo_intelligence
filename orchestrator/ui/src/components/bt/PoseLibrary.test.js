// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import PoseLibrary from './PoseLibrary';
import { listBtTrees, readBtTree, saveBtTree } from '../../features/actionCanvas/btTreesApi';
import { SG2_GROUPS } from '../../utils/sg2PoseCapture';
import { parseSg2Pose, serializeSg2Pose } from '../../utils/sg2PoseFile';

const mockPose = { name: Object.values(SG2_GROUPS).flat(), position: Array(19).fill(0.5) };
jest.mock('../../features/actionCanvas/btTreesApi', () => ({ listBtTrees: jest.fn(), readBtTree: jest.fn(), saveBtTree: jest.fn() }));
jest.mock('./PoseCaptureButton', () => function MockCapture({ onSnapshot, onCapture }) {
  return <button onClick={() => { onSnapshot(mockPose); onCapture({ left_target_positions: '0.5' }); }}>Capture current pose</button>;
});

beforeEach(() => {
  listBtTrees.mockResolvedValue({ directory: '/workspace/bt/trees', trees: [{ name: 'sg2_pose_ready.xml' }, { name: 'unrelated.xml' }] });
  readBtTree.mockResolvedValue({ name: 'sg2_pose_ready.xml', content: serializeSg2Pose(mockPose, 'ready') });
  saveBtTree.mockResolvedValue({ path: '/workspace/bt/trees/sg2_pose_ready.xml' });
});

test('saves the captured snapshot to a named server file without overwriting', async () => {
  const view = render(<PoseLibrary nodeType="ArmStateGate" params={{}} onApply={jest.fn()} />);
  expect(screen.getByText('Save pose')).toBeDisabled();
  fireEvent.click(screen.getByText('Capture current pose'));
  view.rerender(<PoseLibrary nodeType="ArmStateGate" params={{ left_target_positions: '9' }} onApply={jest.fn()} />);
  fireEvent.change(screen.getByLabelText('Pose name'), { target: { value: 'ready' } });
  fireEvent.click(screen.getByText('Save pose'));
  expect(await screen.findByText('Saved: /workspace/bt/trees/sg2_pose_ready.xml')).toBeInTheDocument();
  const request = saveBtTree.mock.calls[0][0];
  expect(request.filename).toBe('sg2_pose_ready.xml');
  expect(request.overwrite).toBe(false);
  expect(parseSg2Pose(request.content)).toEqual(mockPose);
});

test('loads a file into the selected block in one parameter update', async () => {
  const onApply = jest.fn();
  render(<PoseLibrary nodeType="ArmStateGate" params={{ right_target_joints: 'arm_r_joint2' }} onApply={onApply} />);
  fireEvent.click(screen.getByText('Load pose'));
  await screen.findByLabelText('Saved pose');
  expect(screen.queryByText('unrelated.xml')).not.toBeInTheDocument();
  fireEvent.click(screen.getByText('Apply pose to block'));
  await waitFor(() => expect(onApply).toHaveBeenCalledWith({ right_target_positions: '0.5' }));
});

test('a delayed file read cannot overwrite parameters changed while loading', async () => {
  let resolve;
  readBtTree.mockReturnValue(new Promise((done) => { resolve = done; }));
  const onApply = jest.fn();
  const view = render(<PoseLibrary nodeType="ArmStateGate" params={{}} onApply={onApply} />);
  fireEvent.click(screen.getByText('Load pose'));
  await screen.findByLabelText('Saved pose');
  fireEvent.click(screen.getByText('Apply pose to block'));
  view.rerender(<PoseLibrary nodeType="ArmStateGate" params={{ left_target_joints: 'arm_l_joint1' }} onApply={onApply} />);
  await act(async () => resolve({ content: serializeSg2Pose(mockPose, 'ready') }));
  expect(screen.getByRole('alert')).toHaveTextContent('parameters changed');
  expect(onApply).not.toHaveBeenCalled();
});

test('save failures keep the captured pose available for retry', async () => {
  saveBtTree.mockRejectedValue(new Error('Tree already exists'));
  render(<PoseLibrary nodeType="ArmStateGate" params={{}} onApply={jest.fn()} />);
  fireEvent.click(screen.getByText('Capture current pose'));
  fireEvent.change(screen.getByLabelText('Pose name'), { target: { value: 'ready' } });
  fireEvent.click(screen.getByText('Save pose'));
  expect(await screen.findByRole('alert')).toHaveTextContent('already exists');
  expect(screen.getByText('Save pose')).toBeEnabled();
});
