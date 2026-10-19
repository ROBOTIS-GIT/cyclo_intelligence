import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ReturnWaypointFields from './ReturnWaypointFields';
import { getPgmFiles } from '../../../utils/navigationApi';
import { getNavigationMission, getNavigationMissions, subscribeNavigationMissionChanges } from '../../../utils/navigationMissionsApi';
jest.mock('../../../utils/navigationApi');
jest.mock('../../../utils/navigationMissionsApi');
beforeEach(() => {
  jest.clearAllMocks();
  subscribeNavigationMissionChanges.mockReturnValue(() => {});
  getPgmFiles.mockResolvedValue({ files: [{path: '/maps/house.pgm'}] });
  getNavigationMissions.mockResolvedValue({missions: ['DockTest']});
  getNavigationMission.mockResolvedValue({waypoints: [{id: 'point-1', label: 'InitPosition'}]});
});
test('shows ordered cascading fields, stores waypoint ID, clears dependent selections', async () => {
  const change = jest.fn();
  const view = render(<ReturnWaypointFields params={{}} onChange={change} />);
  await screen.findByRole('option', {name: 'house'});
  expect(screen.getAllByRole('combobox').map(e => e.id)).toEqual(['return-map_name', 'return-mission_name', 'return-waypoint']);
  expect(screen.getByLabelText('mission_name')).toBeDisabled();
  fireEvent.change(screen.getByLabelText('map_name'), {target: {value:'house'}});
  expect(change.mock.calls).toEqual([['map_name','house'],['mission_name',''],['waypoint','']]);
  view.rerender(<ReturnWaypointFields params={{map_name:'house', mission_name:'DockTest'}} onChange={change} />);
  await screen.findByRole('option', {name: 'InitPosition'});
  await waitFor(() => expect(screen.getByLabelText('waypoint')).not.toBeDisabled());
  fireEvent.change(screen.getByLabelText('waypoint'), {target: {value:'point-1'}});
  expect(change).toHaveBeenLastCalledWith('waypoint','point-1');
});
