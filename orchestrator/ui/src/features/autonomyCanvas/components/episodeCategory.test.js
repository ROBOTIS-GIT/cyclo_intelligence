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

import { episodeCategory } from './episodeCategory';

test('episode category comes from metadata or immediate parent folder', () => {
  expect(episodeCategory({episode_type:'mission',storage_path:'/data/map/observation/raw_episode_0001'})).toBe('mission');
  expect(episodeCategory({storage_path:'/data/mission/observation/raw_episode_0001'})).toBe('observation');
  expect(episodeCategory({storage_path:'/data/map/mission/raw_episode_0002'})).toBe('mission');
  expect(episodeCategory({recording_mission_name:'ObservationMode'})).toBe('observation');
});
