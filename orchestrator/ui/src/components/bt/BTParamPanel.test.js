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
// Author: Seongwoo Kim

import { fireEvent, render, screen } from '@testing-library/react';
import BTParamPanel from './BTParamPanel';

jest.mock('react-redux', () => ({
  useDispatch: () => jest.fn(),
  useSelector: (selector) => selector({ tasks: { robotType: 'ffw_sg2' } }),
}));

jest.mock('../FileBrowserModal', () => () => null);

const node = {
  id: 'bt_1',
  data: {
    label: 'JointControl_1',
    nodeType: 'JointControl',
    params: {
      enable_head: 'false',
      enable_arms: 'true',
      enable_lift: 'false',
      duration: '2.0',
    },
  },
};

function fieldControl(name) {
  return screen.getByText(name, { selector: 'label' }).parentElement.querySelector(
    'input, select, textarea',
  );
}

test('commits parameter drafts while typing instead of waiting for blur', () => {
  const onParamChange = jest.fn();
  render(
    <BTParamPanel
      nodes={[node]}
      selectedNodeId={node.id}
      onParamChange={onParamChange}
      onNameChange={jest.fn()}
    />,
  );

  fireEvent.change(screen.getByDisplayValue('2.0'), { target: { value: '3.5' } });

  expect(onParamChange).toHaveBeenCalledTimes(1);
  expect(onParamChange).toHaveBeenCalledWith('bt_1', 'duration', '3.5');
});

test('commits a valid node name while typing instead of waiting for blur', () => {
  const onNameChange = jest.fn();
  render(
    <BTParamPanel
      nodes={[node]}
      selectedNodeId={node.id}
      onParamChange={jest.fn()}
      onNameChange={onNameChange}
    />,
  );

  fireEvent.change(screen.getByDisplayValue('JointControl_1'), {
    target: { value: 'CloseGripper' },
  });

  expect(onNameChange).toHaveBeenCalledTimes(1);
  expect(onNameChange).toHaveBeenCalledWith('bt_1', 'CloseGripper');
});

test('restores the pre-edit node name when Escape is pressed', () => {
  const onNameChange = jest.fn();
  render(
    <BTParamPanel
      nodes={[node]}
      selectedNodeId={node.id}
      onParamChange={jest.fn()}
      onNameChange={onNameChange}
    />,
  );

  const input = screen.getByDisplayValue('JointControl_1');
  fireEvent.focus(input);
  fireEvent.change(input, { target: { value: 'TemporaryName' } });
  fireEvent.keyDown(input, { key: 'Escape' });

  expect(input).toHaveValue('JointControl_1');
  expect(onNameChange).toHaveBeenLastCalledWith('bt_1', 'JointControl_1');
});

test('JointControl renders per-joint chips that keep positions aligned', () => {
  const onParamChange = jest.fn();
  const jointNode = {
    id: 'bt_2',
    data: {
      label: 'JointControl_2',
      nodeType: 'JointControl',
      params: {
        enable_arms: 'true',
        left_joint_names: 'arm_l_joint1, arm_l_joint2',
        left_positions: '0.1, 0.2',
        right_joint_names: '',
        right_positions: '',
        duration: '2.0',
      },
    },
  };
  render(
    <BTParamPanel
      nodes={[jointNode]}
      selectedNodeId={jointNode.id}
      onParamChange={onParamChange}
      onNameChange={jest.fn()}
    />,
  );

  // Each side renders the SG2 joint chips (left list + right list).
  expect(screen.getAllByRole('button', { name: 'arm_l_joint3' })).toHaveLength(1);
  expect(screen.getAllByRole('button', { name: 'arm_r_joint1' })).toHaveLength(1);

  // Deselecting a joint drops its position from the paired CSV.
  fireEvent.click(screen.getByRole('button', { name: 'arm_l_joint1' }));
  expect(onParamChange).toHaveBeenCalledWith('bt_2', 'left_joint_names', 'arm_l_joint2');
  expect(onParamChange).toHaveBeenCalledWith('bt_2', 'left_positions', '0.2');

  // Selecting a new joint appends it in canonical order with a 0.0 target.
  fireEvent.click(screen.getByRole('button', { name: 'arm_l_joint3' }));
  expect(onParamChange).toHaveBeenCalledWith(
    'bt_2', 'left_joint_names', 'arm_l_joint2, arm_l_joint3',
  );
  expect(onParamChange).toHaveBeenCalledWith('bt_2', 'left_positions', '0.2, 0.0');
});

test('JointControl joint chips are disabled while arms are disabled', () => {
  const jointNode = {
    id: 'bt_3',
    data: {
      label: 'JointControl_3',
      nodeType: 'JointControl',
      params: {
        enable_arms: 'false',
        left_joint_names: 'arm_l_joint1',
        left_positions: '0.0',
      },
    },
  };
  render(
    <BTParamPanel
      nodes={[jointNode]}
      selectedNodeId={jointNode.id}
      onParamChange={jest.fn()}
      onNameChange={jest.fn()}
    />,
  );

  expect(screen.getByRole('button', { name: 'arm_l_joint1' })).toBeDisabled();
});

test('legacy JointControl nodes without joint_names still get the chips', () => {
  const onParamChange = jest.fn();
  const legacyNode = {
    id: 'bt_4',
    data: {
      label: 'JointControl_4',
      nodeType: 'JointControl',
      params: {
        enable_arms: 'true',
        left_positions: '0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8',
        right_positions: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
        duration: '2.0',
      },
    },
  };
  render(
    <BTParamPanel
      nodes={[legacyNode]}
      selectedNodeId={legacyNode.id}
      onParamChange={onParamChange}
      onNameChange={jest.fn()}
    />,
  );

  // The synthesized selection defaults to the full joint list (what the
  // engine does when names are omitted), so deselecting one joint keeps
  // the other joints' existing positions aligned.
  fireEvent.click(screen.getByRole('button', { name: 'gripper_l_joint1' }));
  expect(onParamChange).toHaveBeenCalledWith(
    'bt_4',
    'left_joint_names',
    'arm_l_joint1, arm_l_joint2, arm_l_joint3, arm_l_joint4, '
    + 'arm_l_joint5, arm_l_joint6, arm_l_joint7',
  );
  expect(onParamChange).toHaveBeenCalledWith(
    'bt_4', 'left_positions', '0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7',
  );
});

test('treats legacy SendCommand nodes without a target as inference commands', () => {
  const legacyCommand = {
    id: 'bt_5',
    data: {
      label: 'SendCommand_1',
      nodeType: 'SendCommand',
      params: {
        command: 'STOP',
        model: 'lerobot:act',
        policy_path: '/workspace/model/lerobot/example',
      },
    },
  };

  render(
    <BTParamPanel
      nodes={[legacyCommand]}
      selectedNodeId={legacyCommand.id}
      onParamChange={jest.fn()}
      onNameChange={jest.fn()}
    />,
  );

  expect(fieldControl('target')).toHaveValue('INFERENCE');
  expect(fieldControl('command')).toHaveValue('STOP');
  expect(Array.from(fieldControl('command').options, (option) => option.value)).toEqual([
    'LOAD', 'RESUME', 'STOP', 'CLEAR',
  ]);
  expect(fieldControl('model')).toBeDisabled();
  expect(fieldControl('policy_path')).toBeDisabled();
});

test('switches SendCommand to Docker controls and enables only target, command, and model', () => {
  const onParamChange = jest.fn();
  const dockerCommand = {
    id: 'bt_6',
    data: {
      label: 'SendCommand_2',
      nodeType: 'SendCommand',
      params: {
        target: 'INFERENCE',
        command: 'LOAD',
        model: 'groot:n17',
        policy_path: '/workspace/model/groot/example',
        task_instruction: 'Pick up the object',
        inference_mode: 'robot',
        action_request_mode: 'sync',
        inference_hz: '10',
        control_hz: '100',
        chunk_align_window_s: '0.3',
        acceleration_mode: 'pytorch',
        acceleration_engine_path: '',
      },
    },
  };

  render(
    <BTParamPanel
      nodes={[dockerCommand]}
      selectedNodeId={dockerCommand.id}
      onParamChange={onParamChange}
      onNameChange={jest.fn()}
    />,
  );

  fireEvent.change(fieldControl('target'), { target: { value: 'DOCKER' } });

  expect(onParamChange).toHaveBeenCalledWith('bt_6', 'target', 'DOCKER');
  expect(onParamChange).toHaveBeenCalledWith('bt_6', 'command', 'START');
  expect(fieldControl('target')).toBeEnabled();
  expect(fieldControl('command')).toBeEnabled();
  expect(fieldControl('model')).toBeEnabled();
  expect(fieldControl('command')).toHaveValue('START');
  expect(Array.from(fieldControl('command').options, (option) => option.value)).toEqual([
    'START', 'STOP', 'RESTART',
  ]);
  [
    'policy_path',
    'task_instruction',
    'inference_mode',
    'action_request_mode',
    'inference_hz',
    'control_hz',
    'chunk_align_window_s',
    'acceleration_mode',
    'acceleration_engine_path',
  ].forEach((key) => expect(fieldControl(key)).toBeDisabled());
});
