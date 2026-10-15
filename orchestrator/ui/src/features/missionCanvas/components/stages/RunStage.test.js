import { fireEvent, render, screen } from '@testing-library/react';
import { RunStageHud } from './RunStage';

jest.mock('../../../mapWorkspace/components/mapChrome', () => ({ TopicStatusPanel: () => null }));
jest.mock('../run/RunSessionPanel', () => () => null);

test('Run requires the loaded mission snapshot and invokes the mission runner', () => {
  const onRunMission = jest.fn();
  const props = {missionMapLoaded: true, runPoseInitialized: true, onRunMission};
  const view = render(<RunStageHud {...props} runMapSnapshotInvalid />);
  expect(screen.getByRole('button', {name: 'Run Mission'})).toBeDisabled();
  view.rerender(<RunStageHud {...props} />);
  fireEvent.click(screen.getByRole('button', {name: 'Run Mission'}));
  expect(onRunMission).toHaveBeenCalledTimes(1);
});
