/*
 * Copyright 2025 ROBOTIS CO., LTD.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Dongyun Kim
 */

import reducer, { resetReplayState, setReplayData } from './replaySlice';

describe('replaySlice', () => {
  test('keeps replay robot metadata for the 3D viewer', () => {
    const state = reducer(
      undefined,
      setReplayData({
        robot_type: 'ffw_sh5_rev1',
        urdf_path: '/workspace/robot_configs/urdf/ffw_sh5_follower.urdf',
        end_effector_links: ['tool0'],
      })
    );

    expect(state.robotType).toBe('ffw_sh5_rev1');
    expect(state.urdfPath).toBe('/workspace/robot_configs/urdf/ffw_sh5_follower.urdf');
    expect(state.endEffectorLinks).toEqual(['tool0']);
  });

  test('resets replay robot metadata with the rest of replay state', () => {
    const loaded = reducer(
      undefined,
      setReplayData({
        robot_type: 'ffw_sh5_rev1',
        urdf_path: '/workspace/robot_configs/urdf/ffw_sh5_follower.urdf',
        end_effector_links: ['tool0'],
      })
    );

    const reset = reducer(loaded, resetReplayState());

    expect(reset.robotType).toBe('');
    expect(reset.urdfPath).toBe('');
    expect(reset.endEffectorLinks).toEqual([]);
  });
});
