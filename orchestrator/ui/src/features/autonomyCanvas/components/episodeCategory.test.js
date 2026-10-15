import { episodeCategory } from './episodeCategory';

test('episode category comes from metadata or immediate parent folder', () => {
  expect(episodeCategory({episode_type:'mission',storage_path:'/data/map/observation/raw_episode_0001'})).toBe('mission');
  expect(episodeCategory({storage_path:'/data/mission/observation/raw_episode_0001'})).toBe('observation');
  expect(episodeCategory({storage_path:'/data/map/mission/raw_episode_0002'})).toBe('mission');
  expect(episodeCategory({recording_mission_name:'ObservationMode'})).toBe('observation');
});
