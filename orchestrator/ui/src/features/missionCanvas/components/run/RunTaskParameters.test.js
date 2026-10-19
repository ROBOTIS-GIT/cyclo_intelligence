import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RunTaskParameters from './RunTaskParameters';
import { getNavigationMission, getNavigationMissionBtFile, saveNavigationMissionBtFile } from '../../../../utils/navigationMissionsApi';
jest.mock('../../../../utils/navigationMissionsApi');
jest.mock('../../../../hooks/useBTNodeCatalog', () => ({useBTNodeCatalog: () => ({catalog: []})}));
jest.mock('../../../btEditor/parameters/BtParameterPanel', () => props => props.selectedNodeId ? <button onClick={() => props.onParamChange(props.selectedNodeId,'duration','5')}>Change duration</button> : null);
beforeEach(() => {
 jest.clearAllMocks();
 getNavigationMission.mockResolvedValue({waypoints:[{id:'a',label:'DockingStart',local_bt:'locals/a/main.xml'}]});
 getNavigationMissionBtFile.mockResolvedValue({exists:true, revision:8, content:'<root><BehaviorTree ID="MainTree"><Wait name="Hold" duration="2"/></BehaviorTree></root>'});
 saveNavigationMissionBtFile.mockResolvedValue({revision:9});
});
test('saves parameters with revision and owner then reloads Run snapshot', async () => {
 const reload=jest.fn().mockResolvedValue({});
 render(<RunTaskParameters mapName="house" missionName="Dock" disabled={false} onSaved={reload}/>);
 fireEvent.click(screen.getByLabelText('Edit waypoint task parameters'));
 fireEvent.click(await screen.findByRole('button',{name:/Hold/}));
 fireEvent.click(screen.getByText('Change duration'));
 fireEvent.click(screen.getByRole('button',{name:'Save'}));
 await screen.findByText('Saved. Ready for the next run.');
 expect(saveNavigationMissionBtFile).toHaveBeenCalledWith('house','locals/a/main.xml',expect.stringContaining('duration="5"'),'Dock',{waypointId:'a',expectedRevision:8});
 expect(reload).toHaveBeenCalledTimes(1);
});
test('blocks editing while running and reports revision conflicts without reload', async () => {
 const reload=jest.fn();
 const view=render(<RunTaskParameters mapName="house" missionName="Dock" disabled onSaved={reload}/>);
 expect(screen.getByLabelText('Edit waypoint task parameters')).toBeDisabled();
 view.rerender(<RunTaskParameters mapName="house" missionName="Dock" disabled={false} onSaved={reload}/>);
 fireEvent.click(screen.getByLabelText('Edit waypoint task parameters'));
 fireEvent.click(await screen.findByRole('button',{name:/Hold/}));
 fireEvent.click(screen.getByText('Change duration'));
 saveNavigationMissionBtFile.mockRejectedValue(new Error('Mission revision conflict'));
 fireEvent.click(screen.getByRole('button',{name:'Save'}));
 await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Mission revision conflict'));
 expect(reload).not.toHaveBeenCalled();
});
