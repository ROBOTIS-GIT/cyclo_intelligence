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

import { StrictMode } from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import AutonomyStudioPage from './AutonomyStudioPage';
import {
  cancelNavigateToPoseGoal,
  configureDesignLocalizationAmcl,
  deletePgmMap,
  getMapAnnotations,
  getPgmFiles,
  getPgmImage,
  getServiceStatus,
  requestNoMotionUpdate,
  saveNavigationMap,
  saveMapAnnotations,
  savePgmImage,
  sendInitialPoseEstimate,
  sendNavigateToPoseGoalAndWait,
  sendNavigateThroughPosesGoalsAndWait,
  startNavigation,
  stopNavigation,
} from '../utils/navigationApi';
import {
  createNavigationSpot,
  deleteNavigationSpot,
  getNavigationSpots,
  updateNavigationSpot,
} from '../utils/navigationSpotsApi';
import {
  deleteNavigationMission,
  deleteNavigationMissionBtFile,
  duplicateNavigationMission,
  getNavigationMission,
  getNavigationMissionBtFile,
  getNavigationMissions,
  renameNavigationMission,
  saveNavigationMission,
  saveNavigationMissionBtFile,
  setNavigationMissionDefaultBtFile,
} from '../utils/navigationMissionsApi';

const mockMapViewer = jest.fn(() => <div>Mission Canvas Map</div>);
const mockActionCanvasWorkspace = jest.fn(() => (
  <div data-testid="action-canvas-workspace">Action Canvas Workspace</div>
));
const mockPublishRosTopic = jest.fn();
const mockCallService = jest.fn();
const mockTopicDataByName = {};

function amclPoseMessage(x, y, yaw, covarianceValue = 0.05) {
  return {
    pose: {
      pose: {
        position: { x, y, z: 0 },
        orientation: { x: 0, y: 0, z: Math.sin(yaw / 2), w: Math.cos(yaw / 2) },
      },
      covariance: [
        covarianceValue, 0, 0, 0, 0, 0,
        0, covarianceValue, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, covarianceValue,
      ],
    },
  };
}

function stringTopicMessage(data) {
  return { available: true, data: { data } };
}

function topicRow(topic) {
  return screen.getByText(topic).parentElement;
}

function mockJsonResponse(data, ok = true, status = ok ? 200 : 500) {
  return {
    ok,
    status,
    text: () => Promise.resolve(JSON.stringify(data)),
  };
}

async function loadRunMapFromDialog(path = 'map.pgm') {
  getPgmFiles.mockResolvedValue({ files: [{ path, name: path }] });
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'Run mission map file' });
  await waitFor(() => expect(mapSelect).toHaveValue(path));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());
}

async function openMappingEditorAndSelect(path = 'factory.pgm') {
  fireEvent.click(screen.getByRole('tab', { name: 'Map Edit' }));
  await waitFor(() => expect(getPgmFiles).toHaveBeenCalled());
  // Load Map stays disabled until the PGM listing lands (mapEditor.busy).
  await waitFor(() => expect(screen.getByRole('button', { name: 'Load Map' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'PGM map' });
  fireEvent.change(mapSelect, { target: { value: path } });
  await waitFor(() => expect(mapSelect).toHaveValue(path));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.queryByRole('combobox', { name: 'PGM map' })).not.toBeInTheDocument());
  // The HUD tools stay disabled until the PGM image itself has loaded.
  await waitFor(() => expect(screen.getByRole('button', { name: 'View' })).toBeEnabled());
}

jest.mock('../components/navigation/MapViewer', () => ({
  MapViewer: (props) => mockMapViewer(props),
}));

jest.mock('../components/navigation/ActionCanvasWorkspace', () => ({
  __esModule: true,
  default: (props) => mockActionCanvasWorkspace(props),
}));

jest.mock('../utils/navigationApi', () => ({
  configureDesignLocalizationAmcl: jest.fn().mockResolvedValue({ ok: true }),
  deletePgmMap: jest.fn().mockResolvedValue({ deleted: true, removed_missions: 0 }),
  getMapAnnotations: jest.fn().mockResolvedValue({ path: 'map.pgm', annotations: [] }),
  getPgmFiles: jest.fn().mockResolvedValue({ files: [] }),
  getPgmImage: jest.fn().mockResolvedValue({
    path: 'map.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  }),
  getServiceStatus: jest.fn().mockResolvedValue({ is_up: false }),
  requestNoMotionUpdate: jest.fn().mockResolvedValue({ ok: true }),
  saveNavigationMap: jest.fn().mockResolvedValue({ ok: true }),
  saveMapAnnotations: jest.fn().mockImplementation((path, annotations) => Promise.resolve({ path, annotations, saved: true })),
  savePgmImage: jest.fn().mockResolvedValue({ path: 'map.pgm', saved: true }),
  sendInitialPoseEstimate: jest.fn().mockResolvedValue({ ok: true }),
  sendNavigateToPoseGoalAndWait: jest.fn().mockResolvedValue({ ok: true, status: 'SUCCEEDED', message: 'Goal succeeded' }),
  sendNavigateThroughPosesGoalsAndWait: jest.fn().mockResolvedValue({ ok: true, status: 'SUCCEEDED', message: 'Goals succeeded' }),
  cancelNavigateToPoseGoal: jest.fn().mockResolvedValue({ ok: true }),
  startNavigation: jest.fn().mockResolvedValue({ ok: true }),
  stopNavigation: jest.fn().mockResolvedValue({ ok: true }),
}));

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: () => ({ callService: mockCallService }),
}));

jest.mock('../utils/navigationSpotsApi', () => ({
  createNavigationSpot: jest.fn().mockResolvedValue({
    id: 'spot_a',
    map_name: 'map',
    label: 'Waypoint A',
    pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
    linked_bt_tree: '',
    metadata: {},
  }),
  deleteNavigationSpot: jest.fn().mockResolvedValue({ ok: true }),
  getNavigationSpots: jest.fn().mockResolvedValue({ map_name: 'map', spots: [] }),
  updateNavigationSpot: jest.fn().mockResolvedValue({
    id: 'spot_a',
    map_name: 'map',
    label: 'Waypoint A',
    pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
    linked_bt_tree: '',
    metadata: {},
  }),
}));

jest.mock('../utils/navigationMissionsApi', () => ({
  deleteNavigationMission: jest.fn().mockResolvedValue({
    map_name: 'map',
    mission_name: 'default',
    deleted: true,
  }),
  renameNavigationMission: jest.fn().mockResolvedValue({
    exists: true,
    map_name: 'map',
    mission_name: 'renamed',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }),
  duplicateNavigationMission: jest.fn().mockResolvedValue({
    exists: true,
    map_name: 'map',
    mission_name: 'default-copy',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }),
  deleteNavigationMissionBtFile: jest.fn().mockResolvedValue({
    path: 'locals/waypoint_a.xml',
    content: '',
    exists: false,
  }),
  getNavigationMission: jest.fn().mockResolvedValue({
    exists: false,
    map_name: 'map',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }),
  getNavigationMissions: jest.fn().mockResolvedValue({
    map_name: 'map',
    missions: ['default'],
  }),
  getNavigationMissionBtFile: jest.fn().mockResolvedValue({
    path: 'global.xml',
    content: '',
    exists: false,
  }),
  saveNavigationMission: jest.fn().mockResolvedValue({
    exists: true,
    map_name: 'map',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }),
  saveNavigationMissionBtFile: jest.fn().mockResolvedValue({
    path: 'global.xml',
    content: '',
    exists: true,
  }),
  setNavigationMissionDefaultBtFile: jest.fn().mockResolvedValue({
    exists: true,
    map_name: 'map',
    mission_name: 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }),
}));

jest.mock('../hooks/useNavigationRosTopic', () => ({
  useNavigationRosTopic: (topic) => ({
    status: topic && mockTopicDataByName[topic] ? 'connected' : 'disconnected',
    topicData: topic ? mockTopicDataByName[topic] || null : null,
  }),
  useNavigationRosPublisher: () => mockPublishRosTopic,
}));

beforeEach(() => {
  jest.clearAllMocks();
  Object.keys(mockTopicDataByName).forEach((topic) => {
    delete mockTopicDataByName[topic];
  });
  window.localStorage.clear();
  window.sessionStorage.clear();
  global.fetch = jest.fn().mockResolvedValue(mockJsonResponse({
    name: 'bt_node',
    state: 'down',
    raw: 'down',
  }));
  mockPublishRosTopic.mockResolvedValue(undefined);
  mockCallService.mockResolvedValue({ success: true });
  createNavigationSpot.mockResolvedValue({
    id: 'spot_a',
    map_name: 'factory',
    label: 'Waypoint A',
    pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
    linked_bt_tree: '',
    metadata: {},
  });
  updateNavigationSpot.mockImplementation((spotId, patch) => Promise.resolve({
    id: spotId,
    map_name: patch.map_name || 'factory',
    label: patch.label || 'Waypoint A',
    pose: patch.pose || { frame_id: 'map', x: 1, y: 2, yaw: 0 },
    linked_bt_tree: '',
    metadata: {},
  }));
  getPgmFiles.mockResolvedValue({ files: [] });
  getPgmImage.mockResolvedValue({
    path: 'map.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });
  getMapAnnotations.mockResolvedValue({ path: 'map.pgm', annotations: [] });
  getServiceStatus.mockResolvedValue({ is_up: false });
  configureDesignLocalizationAmcl.mockResolvedValue({ ok: true });
  requestNoMotionUpdate.mockResolvedValue({ ok: true });
  getNavigationSpots.mockResolvedValue({ map_name: 'map', spots: [] });
  getNavigationMission.mockResolvedValue({
    exists: false,
    map_name: 'map',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  });
  getNavigationMissions.mockResolvedValue({
    map_name: 'map',
    missions: ['default'],
  });
  getNavigationMissionBtFile.mockResolvedValue({
    path: 'global.xml',
    content: '',
    exists: false,
  });
  saveNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'map',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  });
  saveNavigationMissionBtFile.mockResolvedValue({
    path: 'global.xml',
    content: '',
    exists: true,
  });
  setNavigationMissionDefaultBtFile.mockResolvedValue({
    exists: true,
    map_name: 'map',
    mission_name: 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  });
  deleteNavigationMissionBtFile.mockResolvedValue({
    path: 'locals/waypoint_a.xml',
    content: '',
    exists: false,
  });
  saveNavigationMap.mockResolvedValue({ ok: true, message: 'Saved map' });
  saveMapAnnotations.mockImplementation((path, annotations) => Promise.resolve({ path, annotations, saved: true }));
  savePgmImage.mockResolvedValue({ path: 'map.pgm', saved: true });
  sendInitialPoseEstimate.mockResolvedValue({ ok: true });
  startNavigation.mockResolvedValue({ ok: true });
  stopNavigation.mockResolvedValue({ ok: true });
  mockMapViewer.mockImplementation(() => <div>Mission Canvas Map</div>);
  mockActionCanvasWorkspace.mockImplementation(() => (
    <div data-testid="action-canvas-workspace">Action Canvas Workspace</div>
  ));
});

afterEach(() => {
  // Restore spies (e.g. the Math.random spy in the area-color test) so they
  // cannot leak into later tests. Module mocks (jest.fn) are unaffected.
  jest.restoreAllMocks();
});

test('renders Autonomy Studio with the Mission Canvas workspace', async () => {
  render(<AutonomyStudioPage />);

  expect(screen.getByRole('heading', { name: 'Autonomy Studio' })).toBeInTheDocument();
  expect(screen.getByText('Mission Canvas Map')).toBeInTheDocument();
  expect(screen.getByRole('tab', { name: 'Mapping' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByRole('tab', { name: 'Design' })).toBeInTheDocument();
  expect(screen.getByRole('tab', { name: 'Run' })).toBeInTheDocument();
  expect(screen.getByText('Status: idle')).toBeInTheDocument();
  expect(screen.queryByLabelText('Map name')).not.toBeInTheDocument();
  expect(screen.getByText('Mapping Session')).toBeInTheDocument();
  expect(screen.getByText('Live mapping')).toBeInTheDocument();
  expect(screen.getByText('Not saved')).toBeInTheDocument();
  expect(screen.getByText('Clean')).toBeInTheDocument();
  expect(screen.queryByText('PID:')).not.toBeInTheDocument();
  const startMappingButton = screen.getByRole('button', { name: 'Start Mapping' });
  const stopButton = screen.getByRole('button', { name: 'Stop' });
  const saveMapButton = screen.getByRole('button', { name: 'Save Map' });
  // The HUD puts Save Map before Stop: saving is the primary action while
  // recording; Stop ends the session.
  expect(Boolean(startMappingButton.compareDocumentPosition(saveMapButton) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
  expect(Boolean(saveMapButton.compareDocumentPosition(stopButton) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
  // Map editing is its own top-level stage now.
  expect(screen.getByRole('tab', { name: 'Map Edit' })).toBeInTheDocument();
  expect(screen.getByText('Mobile Teleop')).toBeInTheDocument();
  expect(screen.getByRole('group', { name: 'Mobile Teleop' })).toBeInTheDocument();
  expect(screen.getByText('/cmd_vel')).toBeInTheDocument();
  expect(screen.getAllByText('Inactive').length).toBeGreaterThan(0);
  expect(screen.getByRole('button', { name: 'Activate' })).toBeEnabled();
  // Layers is now a glass popover over the map (not a docked panel), so only its
  // presence + the switch structure below are asserted.
  expect(screen.getByText('Layers')).toBeInTheDocument();
  expect(screen.getByText('Topics')).toBeInTheDocument();
  expect(screen.getByText('/map')).toBeInTheDocument();
  expect(screen.getByText('/scan')).toBeInTheDocument();
  expect(screen.getByText('/pose')).toBeInTheDocument();
  expect(screen.getByText('/odom')).toBeInTheDocument();
  expect(screen.getByText('/tf')).toBeInTheDocument();
  expect(screen.queryByText('/amcl_pose')).not.toBeInTheDocument();
  expect(screen.queryByText('/global_costmap/costmap')).not.toBeInTheDocument();
  expect(screen.queryByText('/bt/status')).not.toBeInTheDocument();
  await waitFor(() => expect(getServiceStatus).toHaveBeenCalled());
  expect(getNavigationSpots).not.toHaveBeenCalled();
});

test('shows a lightweight Autonomy Studio workspace chooser on a fresh entry', () => {
  const onBackHome = jest.fn();

  render(
    <AutonomyStudioPage
      onBackHome={onBackHome}
      showWorkspaceChooser
    />,
  );

  expect(screen.getByRole('heading', { name: 'Autonomy Studio', level: 1 }))
    .toBeInTheDocument();
  expect(screen.getByText('Choose a workspace')).toBeInTheDocument();
  expect(screen.queryByText('Does your robot need to move on a map?'))
    .not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open Action Canvas' }))
    .toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open Mission Canvas' }))
    .toBeInTheDocument();
  expect(screen.queryByText('NO MAP REQUIRED')).not.toBeInTheDocument();
  expect(screen.queryByText('MAP REQUIRED')).not.toBeInTheDocument();
  expect(screen.queryByText('Build and run robot tasks without a map.'))
    .not.toBeInTheDocument();
  expect(screen.queryByText('Plan navigation and run waypoint tasks on a map.'))
    .not.toBeInTheDocument();

  // The chooser is intentionally lightweight. ROS/map work starts only after
  // the user selects the map-bound Mission Canvas workspace.
  expect(mockMapViewer).not.toHaveBeenCalled();
  expect(mockActionCanvasWorkspace).not.toHaveBeenCalled();
  expect(getServiceStatus).not.toHaveBeenCalled();
  expect(getPgmFiles).not.toHaveBeenCalled();
  expect(getPgmImage).not.toHaveBeenCalled();
  expect(getNavigationSpots).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Back to Home' }));
  expect(onBackHome).toHaveBeenCalledTimes(1);
});

test('opens the mapless Action Canvas from the Autonomy Studio chooser', async () => {
  render(<AutonomyStudioPage showWorkspaceChooser />);

  fireEvent.click(screen.getByRole('button', { name: 'Open Action Canvas' }));

  expect(screen.getByTestId('action-canvas-workspace')).toBeInTheDocument();
  expect(screen.queryByText('Choose a workspace')).not.toBeInTheDocument();
  expect(mockActionCanvasWorkspace).toHaveBeenLastCalledWith(expect.objectContaining({
    isActive: true,
    title: 'Action Canvas',
    variant: 'autonomy-studio',
  }));
  expect(mockActionCanvasWorkspace.mock.calls.at(-1)[0]).not.toHaveProperty('subtitle');
  expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
  expect(screen.queryByRole('tab', { name: 'Mapping' })).not.toBeInTheDocument();
  expect(screen.queryByRole('tab', { name: 'Action Canvas' })).not.toBeInTheDocument();
  expect(screen.queryByText('NAVIGATION')).not.toBeInTheDocument();
  expect(screen.queryByText('MISSION')).not.toBeInTheDocument();
  expect(mockMapViewer).not.toHaveBeenCalled();
  expect(getServiceStatus).not.toHaveBeenCalled();
  expect(getPgmFiles).not.toHaveBeenCalled();
  expect(getPgmImage).not.toHaveBeenCalled();
  expect(getNavigationSpots).not.toHaveBeenCalled();
  expect(screen.getByRole('button', { name: 'Back to workspace chooser' }))
    .toBeInTheDocument();
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')).workspaceKind,
  ).toBe('action_canvas'));
});

test('opens the Mapping stage from the Autonomy Studio chooser card', async () => {
  render(<AutonomyStudioPage showWorkspaceChooser />);

  fireEvent.click(screen.getByRole('button', { name: 'Open Mission Canvas' }));

  expect(screen.queryByText('Choose a workspace')).not.toBeInTheDocument();
  expect(screen.getByRole('tab', { name: 'Mapping' }))
    .toHaveAttribute('aria-selected', 'true');
  const stageNavigation = screen.getByRole('tablist', { name: 'Mission Canvas stages' });
  expect(within(stageNavigation).getByText('NAVIGATION')).toBeInTheDocument();
  expect(within(stageNavigation).getByText('MISSION')).toBeInTheDocument();
  expect(within(stageNavigation).queryByText('AUTOMATION')).not.toBeInTheDocument();
  expect(within(stageNavigation).queryByRole('tab', { name: 'Action Canvas' }))
    .not.toBeInTheDocument();
  expect(screen.getByText('Mission Canvas Map')).toBeInTheDocument();
  expect(screen.queryByTestId('action-canvas-workspace')).not.toBeInTheDocument();
  await waitFor(() => {
    expect(getServiceStatus).toHaveBeenCalled();
    expect(getPgmFiles).toHaveBeenCalled();
    expect(
      JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')).workspaceKind,
    ).toBe('mission');
  });
});

test('places the Back icon, robot brand, and Autonomy Studio title in the workspace header', async () => {
  render(<AutonomyStudioPage onBackHome={jest.fn()} />);
  await waitFor(() => {
    expect(getServiceStatus).toHaveBeenCalled();
    expect(getPgmFiles).toHaveBeenCalled();
  });

  const backToChooserButton = screen.getByRole('button', { name: 'Back to workspace chooser' });
  const topHeader = backToChooserButton.closest('header');

  expect(topHeader).not.toBeNull();
  expect(backToChooserButton.closest('aside')).toBeNull();
  expect(backToChooserButton).toHaveAccessibleName('Back to workspace chooser');
  expect(backToChooserButton.querySelector('svg')).toHaveAttribute('aria-hidden', 'true');
  expect(within(backToChooserButton).queryByText('Back to workspace chooser')).not.toBeInTheDocument();

  const robotBrandIcon = within(topHeader).getByTestId('autonomy-studio-brand-icon');
  const autonomyStudioTitle = within(topHeader).getByRole('heading', {
    name: 'Autonomy Studio',
  });
  expect(robotBrandIcon.tagName.toLowerCase()).toBe('svg');
  expect(robotBrandIcon).toHaveAttribute('aria-hidden', 'true');
  expect(robotBrandIcon.closest('header')).toBe(topHeader);
  expect(
    Boolean(
      backToChooserButton.compareDocumentPosition(robotBrandIcon)
      & Node.DOCUMENT_POSITION_FOLLOWING
    )
  ).toBe(true);
  expect(
    Boolean(
      robotBrandIcon.compareDocumentPosition(autonomyStudioTitle)
      & Node.DOCUMENT_POSITION_FOLLOWING
    )
  ).toBe(true);
});

test('returns a clean direct workspace to the chooser before returning Home', async () => {
  const onBackHome = jest.fn();

  render(<AutonomyStudioPage onBackHome={onBackHome} />);
  await waitFor(() => {
    expect(getServiceStatus).toHaveBeenCalled();
    expect(getPgmFiles).toHaveBeenCalled();
  });

  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));

  expect(screen.getByText('Choose a workspace')).toBeInTheDocument();
  expect(onBackHome).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Back to Home' }));
  expect(onBackHome).toHaveBeenCalledTimes(1);
});

test('guards returning to the chooser with the existing unsaved Design dialog', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const onBackHome = jest.fn();
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });

  render(<AutonomyStudioPage onBackHome={onBackHome} />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'On Map' }));
  await act(async () => {
    await latestMapViewerProps().onMapPose(1, 2, 0.25);
  });
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));

  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));
  expect(screen.getByText('Unsaved changes')).toBeInTheDocument();
  expect(onBackHome).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(onBackHome).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));
  fireEvent.click(screen.getByRole('button', { name: 'Discard' }));
  expect(screen.getByText('Choose a workspace')).toBeInTheDocument();
  expect(onBackHome).not.toHaveBeenCalled();
});

test('keeps unsaved Map Edit changes in place when the chooser is requested', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const onBackHome = jest.fn();
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: '/g==',
  });

  render(<AutonomyStudioPage onBackHome={onBackHome} />);
  await openMappingEditorAndSelect();
  fireEvent.click(screen.getByRole('button', { name: 'Map Edit' }));
  fireEvent.click(screen.getByRole('button', { name: 'Add Obstacle' }));
  await act(async () => {
    latestMapViewerProps().onEditorMapPoint(0, 0);
  });
  await waitFor(() => expect(screen.getByText('· unsaved')).toBeInTheDocument());

  const backToChooserButton = screen.getByRole('button', { name: 'Back to workspace chooser' });
  fireEvent.click(backToChooserButton);

  expect(onBackHome).not.toHaveBeenCalled();
  const exitStatus = screen.getByRole('status');
  expect(exitStatus).toHaveTextContent('Save the current map edits');
  expect(exitStatus).toHaveAttribute('aria-live', 'polite');
  expect(exitStatus).toHaveAttribute('id', expect.stringMatching(/\S+/));
  expect(backToChooserButton).toHaveAttribute('aria-describedby', exitStatus.id);
});

test.each([
  ['Mapping', 'mapping', 'map', false],
  ['Run', 'run', 'nav', true],
])('blocks returning to the chooser while the %s runtime is active', async (
  _label,
  workspaceStage,
  statusMode,
  runRuntimeOwned,
) => {
  const onBackHome = jest.fn();
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'mission',
    workspaceStage,
    navigationRuntimeMode: workspaceStage,
    runRuntimeOwned,
    runShutdownPending: false,
  }));
  getServiceStatus.mockResolvedValue({ is_up: true, mode: statusMode });

  render(<AutonomyStudioPage onBackHome={onBackHome} />);
  await waitFor(() => expect(getServiceStatus).toHaveBeenCalled());
  if (workspaceStage === 'mapping') {
    await waitFor(() => expect(screen.getByText('Status: running')).toBeInTheDocument());
  } else {
    await waitFor(() => expect(screen.getByRole('button', { name: 'Load Map' })).toBeDisabled());
  }

  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));

  expect(onBackHome).not.toHaveBeenCalled();
  expect(screen.getByRole('status')).toHaveTextContent('Stop the active runtime');
  expect(screen.queryByText('Choose a workspace')).not.toBeInTheDocument();
});

test('keeps a restored runtime mode guarded while status confirmation is pending', async () => {
  const onBackHome = jest.fn();
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'mission',
    workspaceStage: 'mapping',
    navigationRuntimeMode: 'mapping',
    runRuntimeOwned: false,
    runShutdownPending: false,
  }));
  getServiceStatus.mockReturnValue(new Promise(() => {}));

  render(<AutonomyStudioPage onBackHome={onBackHome} />);
  await waitFor(() => expect(getServiceStatus).toHaveBeenCalled());

  // The first status response can lag behind a successful runtime start. Keep
  // the session-owned mode authoritative during that gap even though the HUD
  // does not yet have an `is_up` confirmation.
  expect(screen.getByText('Status: idle')).toBeInTheDocument();
  expect(screen.queryByRole('tab', { name: 'Action Canvas' })).not.toBeInTheDocument();
  const backToChooserButton = screen.getByRole('button', { name: 'Back to workspace chooser' });
  expect(backToChooserButton).toHaveAttribute('title', expect.stringContaining('Stop'));

  fireEvent.click(backToChooserButton);

  expect(onBackHome).not.toHaveBeenCalled();
  expect(screen.getByRole('status')).toHaveTextContent('Stop the active runtime');
  expect(screen.queryByText('Choose a workspace')).not.toBeInTheDocument();
});

test('waits for an in-flight Mission operation before returning to the chooser', async () => {
  const onBackHome = jest.fn();
  let finishStartMapping;
  startNavigation.mockReturnValue(new Promise((resolve) => {
    finishStartMapping = resolve;
  }));

  render(<AutonomyStudioPage onBackHome={onBackHome} />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Start Mapping' })).toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Start Mapping' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Back to workspace chooser' })).toHaveAttribute(
    'title',
    expect.stringContaining('Wait for the current operation'),
  ));
  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));

  expect(onBackHome).not.toHaveBeenCalled();
  expect(screen.getByRole('status')).toHaveTextContent('Wait for the current operation');
  expect(screen.queryByText('Choose a workspace')).not.toBeInTheDocument();

  await act(async () => {
    finishStartMapping({ ok: true, message: 'started' });
    await Promise.resolve();
  });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Start Mapping' })).toBeEnabled());
});

test('uses Action Canvas exit state to guard returning to the chooser', async () => {
  const onBackHome = jest.fn();

  render(<AutonomyStudioPage onBackHome={onBackHome} showWorkspaceChooser />);
  fireEvent.click(screen.getByRole('button', { name: 'Open Action Canvas' }));
  const exitStateChange = mockActionCanvasWorkspace.mock.calls.at(-1)[0].onExitStateChange;

  act(() => exitStateChange({ active: true, busy: false }));
  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));
  expect(onBackHome).not.toHaveBeenCalled();
  expect(screen.getByRole('status')).toHaveTextContent('Stop the active runtime');
  expect(screen.queryByText('Choose a workspace')).not.toBeInTheDocument();

  act(() => exitStateChange({ active: false, busy: true }));
  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));
  expect(onBackHome).not.toHaveBeenCalled();
  expect(screen.getByRole('status')).toHaveTextContent('Wait for the current operation');
  expect(screen.queryByText('Choose a workspace')).not.toBeInTheDocument();

  act(() => exitStateChange({ active: false, busy: false }));
  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));
  expect(screen.getByText('Choose a workspace')).toBeInTheDocument();
  expect(onBackHome).not.toHaveBeenCalled();
});

test('shows only grouped navigation and mission stages inside Mission Canvas', () => {
  render(<AutonomyStudioPage />);

  const tablist = screen.getByRole('tablist', { name: 'Mission Canvas stages' });
  expect(within(tablist).getByText('NAVIGATION')).toBeInTheDocument();
  expect(within(tablist).getByText('MISSION')).toBeInTheDocument();
  expect(within(tablist).getByRole('tab', { name: 'Mapping' })).toHaveAttribute('aria-selected', 'true');
  expect(within(tablist).getByRole('tab', { name: 'Map Edit' })).toBeInTheDocument();
  expect(within(tablist).getByRole('tab', { name: 'Navigation' })).toBeInTheDocument();
  expect(within(tablist).getByRole('tab', { name: 'Design' })).toBeInTheDocument();
  expect(within(tablist).getByRole('tab', { name: 'Run' })).toBeInTheDocument();
  expect(screen.getByText('Mission Canvas Map')).toBeInTheDocument();
  expect(within(tablist).queryByRole('tab', { name: 'Action Canvas' })).not.toBeInTheDocument();
  expect(within(tablist).queryByRole('button', { name: 'Mission Canvas' })).not.toBeInTheDocument();
  expect(within(tablist).queryByText('AUTOMATION')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Open Action Canvas' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Open Mission Canvas' })).not.toBeInTheDocument();
});

test('restores the standalone workspace from the Mission Canvas session', () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'action_canvas',
    workspaceStage: 'authoring',
  }));

  render(<AutonomyStudioPage />);

  expect(screen.getByTestId('action-canvas-workspace')).toBeInTheDocument();
  expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
  expect(screen.queryByRole('tab', { name: 'Design' })).not.toBeInTheDocument();
  expect(screen.queryByText('Mission Canvas Map')).not.toBeInTheDocument();
  expect(getPgmFiles).not.toHaveBeenCalled();
  expect(getPgmImage).not.toHaveBeenCalled();
  expect(getNavigationSpots).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));
  expect(screen.getByText('Choose a workspace')).toBeInTheDocument();
});

test('defaults legacy Mission Canvas sessions without workspaceKind to Mission', () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceStage: 'authoring',
    mapName: 'factory',
  }));

  render(<AutonomyStudioPage />);

  expect(screen.getByRole('tab', { name: 'Design' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.queryByRole('tab', { name: 'Action Canvas' })).not.toBeInTheDocument();
  expect(screen.getByText('Mission Canvas Map')).toBeInTheDocument();
  expect(screen.queryByTestId('action-canvas-workspace')).not.toBeInTheDocument();
});

test('updates mapping topics when layer toggles change', async () => {
  render(<AutonomyStudioPage />);

  expect(screen.getByText('/scan')).toBeInTheDocument();
  expect(screen.getByText('/pose')).toBeInTheDocument();
  expect(screen.getByText('/odom')).toBeInTheDocument();
  expect(screen.queryByText('/amcl_pose')).not.toBeInTheDocument();
  expect(screen.getByText('/tf')).toBeInTheDocument();
  expect(screen.getByText('/tf_static')).toBeInTheDocument();
  expect(screen.getByText('/local_costmap/published_footprint')).toBeInTheDocument();
  const lidarSwitch = screen.getByRole('switch', { name: 'Lidar' });
  expect(lidarSwitch).toHaveAttribute('aria-checked', 'true');

  fireEvent.click(lidarSwitch);
  expect(lidarSwitch).toHaveAttribute('aria-checked', 'false');
  expect(screen.queryByText('/scan')).not.toBeInTheDocument();
  expect(screen.getByText('/pose')).toBeInTheDocument();
  expect(screen.getByText('/odom')).toBeInTheDocument();
  expect(screen.queryByText('/amcl_pose')).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('switch', { name: 'Robot Footprint' }));
  expect(screen.queryByText('/amcl_pose')).not.toBeInTheDocument();
  expect(screen.queryByText('/local_costmap/published_footprint')).not.toBeInTheDocument();
  expect(screen.getByText('/pose')).toBeInTheDocument();
  expect(screen.getByText('/odom')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('switch', { name: 'TF' }));
  expect(screen.queryByText('/tf')).not.toBeInTheDocument();
  expect(screen.queryByText('/tf_static')).not.toBeInTheDocument();
  expect(screen.queryByText('/pose')).not.toBeInTheDocument();
  expect(screen.queryByText('/odom')).not.toBeInTheDocument();

  fireEvent.click(lidarSwitch);
  expect(screen.getByText('/scan')).toBeInTheDocument();
  expect(screen.getByText('/pose')).toBeInTheDocument();
  expect(screen.getByText('/odom')).toBeInTheDocument();
  expect(screen.queryByText('/amcl_pose')).not.toBeInTheDocument();

  await waitFor(() => expect(getServiceStatus).toHaveBeenCalled());
  expect(getNavigationSpots).not.toHaveBeenCalled();
});

test('shows waypoint authoring panels without Design BT runtime controls', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));

  expect(screen.queryByText('Mission Flow')).not.toBeInTheDocument();
  expect(screen.queryByText('Properties')).not.toBeInTheDocument();
  expect(screen.queryByText('BT Runtime')).not.toBeInTheDocument();
  expect(screen.queryByText(/BT Node (Unknown|Inactive|Active)/)).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Activate BT' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Deactivate BT' })).not.toBeInTheDocument();
  expect(screen.getByText('Waypoints')).toBeInTheDocument();
  expect(screen.queryByText('Behavior Nodes')).not.toBeInTheDocument();
  expect(screen.queryByText('No behavior nodes placed yet.')).not.toBeInTheDocument();
  expect(screen.queryByText('Waypoints / Local BT')).not.toBeInTheDocument();
  expect(screen.getByText('Mission Route')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Load Map' })).toBeInTheDocument();
  // Mission actions live in the rail session card and appear once a map loads.
  expect(screen.queryByRole('button', { name: 'Save Mission' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Edit On Map' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Add Selected' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Select Waypoints' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Move Waypoints' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeDisabled();
  expect(screen.queryByRole('menu', { name: 'Waypoint creation options' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'At Robot' })).not.toBeInTheDocument();
  expect(screen.queryByText('Select a waypoint or behavior node on the map.')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Delete Waypoint/ })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Delete Node/ })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Create BT' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Edit Task' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Start Mapping' })).not.toBeInTheDocument();
  expect(screen.queryByText('/map')).not.toBeInTheDocument();
  expect(screen.queryByText('/bt/status')).not.toBeInTheDocument();
  expect(screen.queryByText('/bt/active_nodes')).not.toBeInTheDocument();
  expect(screen.queryByText('/scan')).not.toBeInTheDocument();
  expect(screen.queryByText('/global_costmap/costmap')).not.toBeInTheDocument();
  await waitFor(() => expect(getServiceStatus).toHaveBeenCalled());
  expect(getNavigationSpots).not.toHaveBeenCalled();
  expect(getPgmImage).not.toHaveBeenCalled();
  expect(latestMapViewerProps().map).toBeNull();
  expect(latestMapViewerProps().waitingLabel).toBe('Load a map');
  // Design shows the raw grid — waypoints are placed against real pixels, so
  // the beautified floor-plan rendering is reserved for the Run stage.
  expect(latestMapViewerProps().mapRefined).toBe(false);
  expect(latestMapViewerProps().missionRouteOrder).toEqual([]);
});

test('keeps Waypoints empty on first Design entry until a map and mission are loaded', async () => {
  getNavigationSpots.mockResolvedValue({
    map_name: 'map',
    spots: [{
      id: 'persisted_waypoint',
      map_name: 'map',
      label: 'Persisted Waypoint',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      linked_bt_tree: '',
      metadata: {},
    }],
  });
  window.sessionStorage.setItem('cyclo_intelligence.robot_type', 'ffw_sg2');
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceStage: 'authoring',
    mapName: 'map',
    missionName: 'Mission1',
    designMapPath: 'map.pgm',
  }));

  render(<AutonomyStudioPage />);
  expect(screen.getByRole('tab', { name: 'Design' })).toHaveAttribute('aria-selected', 'true');
  await waitFor(() => expect(getServiceStatus).toHaveBeenCalled());

  const waypointsPanel = screen.getByText('Waypoints').parentElement.parentElement;
  expect(within(waypointsPanel).getByText('0')).toBeInTheDocument();
  expect(within(waypointsPanel).getByText('No waypoints for this map yet.')).toBeInTheDocument();
  expect(within(waypointsPanel).queryByText('Persisted Waypoint')).not.toBeInTheDocument();
  expect(getNavigationSpots).not.toHaveBeenCalled();
});

test('does not start or stop the BT runtime while entering Design', async () => {
  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));

  await waitFor(() => expect(getServiceStatus).toHaveBeenCalled());
  expect(screen.queryByText('BT Runtime')).not.toBeInTheDocument();
  expect(global.fetch.mock.calls.some(([url, options]) => (
    String(url).includes('/api/services/bt_node/')
    && options?.method === 'POST'
  ))).toBe(false);
});

test('does not surface BT execution topics in Design', async () => {
  mockTopicDataByName['/bt/status'] = stringTopicMessage('running');
  mockTopicDataByName['/bt/active_nodes'] = stringTopicMessage('MoveBase, Wait');
  global.fetch.mockResolvedValue(mockJsonResponse({
    name: 'bt_node',
    state: 'up',
    raw: 'up',
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));

  expect(screen.queryByText('/bt/status')).not.toBeInTheDocument();
  expect(screen.queryByText('/bt/active_nodes')).not.toBeInTheDocument();
  expect(screen.queryByText('BT Node Active')).not.toBeInTheDocument();
  expect(screen.queryByText('Running')).not.toBeInTheDocument();
  expect(screen.queryByText('MoveBase, Wait')).not.toBeInTheDocument();
  await waitFor(() => expect(getServiceStatus).toHaveBeenCalled());
});

test('loads a saved map into the design stage', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });
  getMapAnnotations.mockResolvedValue({
    path: 'factory.pgm',
    annotations: [{
      id: 'area_dock',
      label: 'Dock',
      color: '#3B241F',
      pose: { frame_id: 'map', x: 0.5, y: 0.5, yaw: 0 },
      region: {
        seed_cell: { x: 0, y: 0 },
        bounds: { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
        cell_count: 1,
        width: 1,
        height: 1,
      },
    }],
  });
  getNavigationMissions.mockResolvedValue({
    map_name: 'factory',
    missions: ['chestnut', 'default'],
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));

  const mapSelect = await screen.findByRole('combobox', { name: 'Design mission map file' });
  await waitFor(() => expect(mapSelect).toHaveValue('factory.pgm'));
  const missionSelect = screen.getByRole('combobox', { name: 'Design mission file' });
  // No phantom "default": the first server-listed mission is preselected.
  await waitFor(() => expect(missionSelect).toHaveValue('chestnut'));

  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(getNavigationMission).toHaveBeenCalledWith('factory', 'chestnut'));
  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('factory.pgm'));
  await waitFor(() => expect(getMapAnnotations).toHaveBeenCalledWith('factory.pgm'));
  await waitFor(() => expect(latestMapViewerProps().map).toMatchObject({
    info: { width: 1, height: 1 },
  }));
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({
      label: 'Dock',
      color: '#3B241F',
      region: expect.objectContaining({
        bounds: { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
      }),
    }),
  ]));
  const designAreasSwitch = screen.getByRole('switch', { name: 'Map areas' });
  expect(designAreasSwitch).toHaveAttribute('aria-checked', 'true');
  fireEvent.click(designAreasSwitch);
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([]));
  fireEvent.click(designAreasSwitch);
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toHaveLength(1));
  const activeMissionSelect = screen.getByRole('combobox', { name: 'Active mission' });
  expect(activeMissionSelect).toHaveValue('chestnut');
  fireEvent.change(activeMissionSelect, { target: { value: 'default' } });
  await waitFor(() => expect(getNavigationMission).toHaveBeenCalledWith('factory', ''));
  await waitFor(() => expect(activeMissionSelect).toHaveValue('default'));
  expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  expect(screen.getByRole('menu', { name: 'Waypoint creation options' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'On Map' })).toBeEnabled();
  expect(screen.queryByRole('button', { name: 'Set Robot Pose' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'At Robot' })).toBeEnabled();
  await waitFor(() => expect(getNavigationSpots).toHaveBeenCalledWith('factory'));
});

test('prefers the active design mission when opening Run', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'floor.pgm', name: 'floor.pgm' }],
  });
  getNavigationMissions.mockResolvedValue({
    map_name: 'floor',
    missions: ['mission', 'High_Table_Recycling'],
  });
  getNavigationMission.mockImplementation((mapName, missionName) => Promise.resolve({
    exists: true,
    map_name: mapName,
    mission_name: missionName || 'mission',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const designMissionSelect = await screen.findByRole('combobox', { name: 'Design mission file' });
  fireEvent.change(designMissionSelect, { target: { value: 'High_Table_Recycling' } });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('High_Table_Recycling'));

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const runMissionSelect = await screen.findByRole('combobox', { name: 'Run mission file' });
  await waitFor(() => expect(runMissionSelect).toHaveValue('High_Table_Recycling'));
});

test('offers a fresh default mission for a map with no missions', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions.mockResolvedValue({ map_name: 'factory', missions: [] });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const missionSelect = await screen.findByRole('combobox', { name: 'Design mission file' });
  await waitFor(() => expect(missionSelect).toHaveValue('default'));
  expect(screen.getByRole('option', { name: 'default (new)' })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(getNavigationMission).toHaveBeenCalledWith('factory', ''));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' })).toHaveValue('default'));
});

// Only a not-yet-saved mission prompts for its name; existing missions save in
// place (rename/duplicate are their own rail actions).
test('names a not-yet-saved mission through the save dialog', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions
    .mockResolvedValueOnce({ map_name: 'factory', missions: [] })
    .mockResolvedValue({ map_name: 'factory', missions: ['evening_route'] });
  getNavigationMission.mockImplementation((mapName, missionName) => Promise.resolve({
    exists: false,
    revision: missionName === 'evening_route' ? 7 : 0,
    map_name: mapName,
    mission_name: missionName || 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' })).toHaveValue('default'));

  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  const nameInput = screen.getByRole('textbox', { name: 'Save mission name' });
  expect(nameInput).toHaveValue('default');
  fireEvent.change(nameInput, { target: { value: 'evening_route' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save', exact: true }));

  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalledWith(
    'factory',
    expect.objectContaining({ expected_revision: 7 }),
    'evening_route',
  ));
  expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'global.xml',
    expect.any(String),
    'evening_route',
    { expectedRevision: 7 },
  );
  // The catalog refetched and the rail now tracks the new mission.
  const railSelect = screen.getByRole('combobox', { name: 'Active mission' });
  await waitFor(() => expect(railSelect).toHaveValue('evening_route'));
});

test('keeps a durable Save As in the active catalog when post-save refresh fails', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions
    .mockResolvedValueOnce({ map_name: 'factory', missions: [] })
    .mockRejectedValueOnce(new Error('catalog temporarily unavailable'));
  getNavigationMission.mockImplementation((mapName, missionName) => Promise.resolve({
    exists: false,
    revision: missionName === 'durable_route' ? 7 : 0,
    map_name: mapName,
    mission_name: missionName || 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('default'));

  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Save mission name' }), {
    target: { value: 'durable_route' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Save', exact: true }));

  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalledWith(
    'factory',
    expect.any(Object),
    'durable_route',
  ));
  const activeMission = screen.getByRole('combobox', { name: 'Active mission' });
  await waitFor(() => expect(activeMission).toHaveValue('durable_route'));
  expect(within(activeMission).getByRole('option', { name: 'durable_route' }))
    .toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save Mission' }))
    .toBeEnabled());
  expect(screen.queryByRole('textbox', { name: 'Save mission name' }))
    .not.toBeInTheDocument();
});

test('does not overwrite an existing catalog mission when naming an unsaved mission', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions.mockResolvedValue({
    map_name: 'factory',
    missions: ['default', 'stored_route'],
  });
  getNavigationMission.mockResolvedValue({
    exists: true,
    revision: 0,
    map_name: 'factory',
    mission_name: 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('default'));

  fireEvent.click(screen.getByRole('button', { name: 'New Mission' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('untitled'));
  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  fireEvent.click(screen.getByRole('button', { name: 'stored_route' }));

  expect(screen.getByText('A mission named "stored_route" already exists.')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Overwrite' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Save', exact: true })).toBeDisabled();

  fireEvent.click(screen.getByRole('button', { name: 'Save', exact: true }));
  expect(saveNavigationMissionBtFile).not.toHaveBeenCalled();
  expect(saveNavigationMission).not.toHaveBeenCalled();
});

test('starts a fresh mission and guards unsaved changes', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    global_bt: 'global.xml',
    waypoints: [
      { id: 'wp1', label: 'Kitchen', pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 }, local_bt: 'locals/wp1.xml', metadata: {} },
    ],
    metadata: {},
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));

  // Clean session: New Mission resets the canvas immediately.
  fireEvent.click(screen.getByRole('button', { name: 'New Mission' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toEqual([]));
  expect(deleteNavigationSpot).not.toHaveBeenCalled();
  const railSelect = screen.getByRole('combobox', { name: 'Active mission' });
  expect(railSelect).toHaveValue('untitled');
  expect(screen.getByRole('option', { name: 'untitled (unsaved)' })).toBeInTheDocument();

  // Dirty the session with a new waypoint, then New Mission must be guarded.
  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'On Map' }));
  await act(async () => {
    await latestMapViewerProps().onMapPose(1, 2, 0.25);
  });
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));

  fireEvent.click(screen.getByRole('button', { name: 'New Mission' }));
  expect(screen.getByText('Unsaved changes')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(latestMapViewerProps().spots).toHaveLength(1);

  fireEvent.click(screen.getByRole('button', { name: 'New Mission' }));
  fireEvent.click(screen.getByRole('button', { name: 'Discard' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toEqual([]));
  // The second fresh document can reuse the same unsaved "untitled" name;
  // history still belongs to the discarded document and must be gone.
  expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled();
});

test('temporarily replaces Design Load Map with unsaved confirmation', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [
      { path: 'factory.pgm', name: 'factory.pgm' },
      { path: 'warehouse.pgm', name: 'warehouse.pgm' },
    ],
  });
  getNavigationMissions.mockImplementation((mapName) => Promise.resolve({
    map_name: mapName,
    missions: ['default'],
  }));
  getNavigationMission.mockImplementation((mapName) => Promise.resolve({
    exists: true,
    revision: 0,
    map_name: mapName,
    mission_name: 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const initialMapSelect = await screen.findByRole('combobox', {
    name: 'Design mission map file',
  });
  await waitFor(() => expect(initialMapSelect).toHaveValue('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().map).not.toBeNull());

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'On Map' }));
  await act(async () => {
    await latestMapViewerProps().onMapPose(1, 2, 0.25);
  });
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));

  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const pendingMapSelect = await screen.findByRole('combobox', {
    name: 'Design mission map file',
  });
  fireEvent.change(pendingMapSelect, { target: { value: 'warehouse.pgm' } });
  await waitFor(() => expect(pendingMapSelect).toHaveValue('warehouse.pgm'));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Load' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  const unsavedDialog = screen.getByRole('dialog', { name: 'Unsaved changes' });
  expect(screen.getAllByRole('dialog')).toEqual([unsavedDialog]);
  expect(screen.queryByRole('dialog', { name: 'Load Map' })).not.toBeInTheDocument();

  fireEvent.click(within(unsavedDialog).getByRole('button', { name: 'Cancel' }));
  expect(screen.queryByRole('dialog', { name: 'Unsaved changes' })).not.toBeInTheDocument();
  const restoredLoadDialog = screen.getByRole('dialog', { name: 'Load Map' });
  expect(screen.getAllByRole('dialog')).toEqual([restoredLoadDialog]);
  expect(within(restoredLoadDialog).getByRole('combobox', {
    name: 'Design mission map file',
  })).toHaveValue('warehouse.pgm');

  fireEvent.click(within(restoredLoadDialog).getByRole('button', { name: 'Load' }));
  const repeatedUnsavedDialog = screen.getByRole('dialog', { name: 'Unsaved changes' });
  expect(screen.getAllByRole('dialog')).toEqual([repeatedUnsavedDialog]);
  expect(screen.queryByRole('dialog', { name: 'Load Map' })).not.toBeInTheDocument();
  fireEvent.click(within(repeatedUnsavedDialog).getByRole('button', { name: 'Discard' }));

  await waitFor(() => expect(getNavigationMission).toHaveBeenCalledWith('warehouse', ''));
  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('warehouse.pgm'));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

test('saves dirty Design changes before continuing the pending map load exactly once', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [
      { path: 'factory.pgm', name: 'factory.pgm' },
      { path: 'warehouse.pgm', name: 'warehouse.pgm' },
    ],
  });
  getNavigationMissions.mockImplementation((mapName) => Promise.resolve({
    map_name: mapName,
    missions: ['default'],
  }));
  getNavigationMission.mockImplementation((mapName) => Promise.resolve({
    exists: true,
    revision: 0,
    map_name: mapName,
    mission_name: 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().map).not.toBeNull());

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'On Map' }));
  await act(async () => {
    await latestMapViewerProps().onMapPose(1, 2, 0.25);
  });

  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.change(mapSelect, { target: { value: 'warehouse.pgm' } });
  await waitFor(() => expect(mapSelect).toHaveValue('warehouse.pgm'));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Load' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  const unsavedDialog = screen.getByRole('dialog', { name: 'Unsaved changes' });
  fireEvent.click(within(unsavedDialog).getByRole('button', { name: 'Save & continue' }));

  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(
    getNavigationMission.mock.calls.filter(([mapName]) => mapName === 'warehouse'),
  ).toHaveLength(1));
  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('warehouse.pgm'));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

test('keeps the unsaved guard and pending map selection when save and continue fails', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [
      { path: 'factory.pgm', name: 'factory.pgm' },
      { path: 'warehouse.pgm', name: 'warehouse.pgm' },
    ],
  });
  getNavigationMissions.mockImplementation((mapName) => Promise.resolve({
    map_name: mapName,
    missions: ['default'],
  }));
  getNavigationMission.mockImplementation((mapName) => Promise.resolve({
    exists: true,
    revision: 0,
    map_name: mapName,
    mission_name: 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }));
  saveNavigationMissionBtFile.mockRejectedValueOnce(new Error('save failed'));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().map).not.toBeNull());

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'On Map' }));
  await act(async () => {
    await latestMapViewerProps().onMapPose(1, 2, 0.25);
  });

  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.change(mapSelect, { target: { value: 'warehouse.pgm' } });
  await waitFor(() => expect(mapSelect).toHaveValue('warehouse.pgm'));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Load' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  fireEvent.click(within(
    screen.getByRole('dialog', { name: 'Unsaved changes' }),
  ).getByRole('button', { name: 'Save & continue' }));

  await waitFor(() => expect(saveNavigationMissionBtFile).toHaveBeenCalled());
  const retainedUnsavedDialog = screen.getByRole('dialog', { name: 'Unsaved changes' });
  await waitFor(() => expect(
    within(retainedUnsavedDialog).getByRole('button', { name: 'Cancel' }),
  ).toBeEnabled());
  expect(screen.queryByRole('dialog', { name: 'Load Map' })).not.toBeInTheDocument();
  expect(getNavigationMission.mock.calls.filter(([mapName]) => mapName === 'warehouse')).toHaveLength(0);

  fireEvent.click(within(retainedUnsavedDialog).getByRole('button', { name: 'Cancel' }));
  expect(within(
    screen.getByRole('dialog', { name: 'Load Map' }),
  ).getByRole('combobox', { name: 'Design mission map file' })).toHaveValue('warehouse.pgm');
});

test('atomically switches a clean active Design mission and resets transient authoring state', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  let resolveBeta;
  const betaManifest = new Promise((resolve) => { resolveBeta = resolve; });
  const manifest = (name) => ({
    exists: true,
    revision: 1,
    map_name: 'factory',
    mission_name: name,
    global_bt: 'global.xml',
    waypoints: [{
      id: `${name}-wp`,
      label: `${name} waypoint`,
      pose: { frame_id: 'map', x: name === 'alpha' ? 1 : 2, y: 0, yaw: 0 },
      local_bt: `locals/${name}.xml`,
      metadata: {},
    }],
    metadata: {},
  });
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getNavigationMissions.mockResolvedValue({ map_name: 'factory', missions: ['alpha', 'beta'] });
  getNavigationMission.mockImplementation((_mapName, missionName) => (
    missionName === 'beta' ? betaManifest : Promise.resolve(manifest('alpha'))
  ));
  getNavigationMissionBtFile.mockImplementation((_mapName, path, missionName) => Promise.resolve({
    path,
    exists: true,
    revision: 1,
    content: `<root><BehaviorTree ID="${missionName}:${path}"/></root>`,
  }));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots[0]?.id).toBe('alpha-wp'));

  fireEvent.click(screen.getByRole('button', { name: 'alpha waypoint' }));
  fireEvent.change(screen.getByRole('combobox', { name: 'Active mission' }), {
    target: { value: 'beta' },
  });
  // The authoring view is gated while the replacement snapshot assembles;
  // neither the old nor a partially loaded new document remains interactive.
  expect(latestMapViewerProps().spots).toEqual([]);

  await act(async () => {
    resolveBeta(manifest('beta'));
    await betaManifest;
  });
  await waitFor(() => expect(latestMapViewerProps().spots[0]?.id).toBe('beta-wp'));
  expect(screen.getByRole('combobox', { name: 'Active mission' })).toHaveValue('beta');
  expect(latestMapViewerProps().selectedSpotId).toBe('');
  expect(latestMapViewerProps().btLayer).toBeNull();
  expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled();
  expect(getNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory', 'locals/beta.xml', 'beta',
  );
});

test('guards a dirty active Design mission switch through Cancel and Discard', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const manifest = (name) => ({
    exists: true,
    revision: 0,
    map_name: 'factory',
    mission_name: name,
    global_bt: 'global.xml',
    waypoints: name === 'beta' ? [{
      id: 'beta-wp',
      label: 'Beta waypoint',
      pose: { frame_id: 'map', x: 2, y: 0, yaw: 0 },
      local_bt: 'locals/beta.xml',
      metadata: {},
    }] : [],
    metadata: {},
  });
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getNavigationMissions.mockResolvedValue({ map_name: 'factory', missions: ['alpha', 'beta'] });
  getNavigationMission.mockImplementation((_mapName, missionName) => Promise.resolve(
    manifest(missionName === 'beta' ? 'beta' : 'alpha'),
  ));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('alpha'));

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'On Map' }));
  await act(async () => { await latestMapViewerProps().onMapPose(1, 2, 0); });
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));

  fireEvent.change(screen.getByRole('combobox', { name: 'Active mission' }), {
    target: { value: 'beta' },
  });
  fireEvent.click(within(
    screen.getByRole('dialog', { name: 'Unsaved changes' }),
  ).getByRole('button', { name: 'Cancel' }));
  expect(screen.getByRole('combobox', { name: 'Active mission' })).toHaveValue('alpha');
  expect(latestMapViewerProps().spots[0].id).toBe('spot_a');

  fireEvent.change(screen.getByRole('combobox', { name: 'Active mission' }), {
    target: { value: 'beta' },
  });
  fireEvent.click(within(
    screen.getByRole('dialog', { name: 'Unsaved changes' }),
  ).getByRole('button', { name: 'Discard' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('beta'));
  await waitFor(() => expect(latestMapViewerProps().spots[0]?.id).toBe('beta-wp'));
});

test('keeps the previous Design content on mission load failure and retries atomically', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const manifest = (name) => ({
    exists: true,
    revision: 2,
    map_name: 'factory',
    mission_name: name,
    global_bt: 'global.xml',
    waypoints: [{
      id: `${name}-wp`,
      label: `${name} waypoint`,
      pose: { frame_id: 'map', x: name === 'alpha' ? 1 : 2, y: 0, yaw: 0 },
      local_bt: `locals/${name}.xml`,
      metadata: {},
    }],
    metadata: {},
  });
  let failBeta = true;
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getNavigationMissions.mockResolvedValue({ map_name: 'factory', missions: ['alpha', 'beta'] });
  getNavigationMission.mockImplementation((_mapName, missionName) => Promise.resolve(
    manifest(missionName === 'beta' ? 'beta' : 'alpha'),
  ));
  getNavigationMissionBtFile.mockImplementation((_mapName, path, missionName) => {
    if (missionName === 'beta' && path === 'locals/beta.xml' && failBeta) {
      return Promise.reject(new Error('beta task failed'));
    }
    return Promise.resolve({
      path,
      exists: true,
      revision: 2,
      content: `<root><BehaviorTree ID="${missionName}:${path}"/></root>`,
    });
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots[0]?.id).toBe('alpha-wp'));

  fireEvent.change(screen.getByRole('combobox', { name: 'Active mission' }), {
    target: { value: 'beta' },
  });
  await waitFor(() => expect(getNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory', 'locals/beta.xml', 'beta',
  ));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save Mission' })).toBeDisabled());
  expect(screen.getByRole('combobox', { name: 'Active mission' })).toHaveValue('beta');
  expect(latestMapViewerProps().spots).toEqual([]);
  expect(screen.getByRole('button', { name: 'Save Mission' })).toBeDisabled();

  failBeta = false;
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const missionSelect = await screen.findByRole('combobox', { name: 'Design mission file' });
  await waitFor(() => expect(missionSelect).toHaveValue('beta'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots[0]?.id).toBe('beta-wp'));
  expect(screen.getByRole('button', { name: 'Save Mission' })).toHaveAttribute(
    'title', 'Save Mission',
  );
});

test('renames the active mission from the rail', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions
    .mockResolvedValueOnce({ map_name: 'factory', missions: ['picnic'] })
    .mockResolvedValue({ map_name: 'factory', missions: ['evening'] });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' })).toHaveValue('picnic'));

  fireEvent.click(screen.getByRole('button', { name: 'Rename mission' }));
  const renameInput = screen.getByRole('textbox', { name: 'Rename mission name' });
  expect(renameInput).toHaveValue('picnic');
  fireEvent.change(renameInput, { target: { value: 'evening' } });
  fireEvent.click(screen.getByRole('button', { name: 'Rename', exact: true }));

  await waitFor(() => expect(renameNavigationMission).toHaveBeenCalledWith(
    'factory',
    'picnic',
    'evening',
    { expectedRevision: 0 },
  ));
  await waitFor(() => (
    expect(screen.getByRole('combobox', { name: 'Active mission' })).toHaveValue('evening')
  ));
});

test('duplicates and deletes the active mission from the rail', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions
    .mockResolvedValueOnce({ map_name: 'factory', missions: ['chestnut', 'default'] })
    .mockResolvedValueOnce({ map_name: 'factory', missions: ['chestnut', 'chestnut-copy', 'default'] })
    .mockResolvedValue({ map_name: 'factory', missions: ['chestnut-copy', 'default'] });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' })).toHaveValue('chestnut'));

  fireEvent.click(screen.getByRole('button', { name: 'Duplicate mission' }));
  const dupInput = screen.getByRole('textbox', { name: 'Duplicate mission name' });
  expect(dupInput).toHaveValue('chestnut-copy');
  fireEvent.click(screen.getByRole('button', { name: 'Duplicate', exact: true }));

  await waitFor(() => expect(duplicateNavigationMission).toHaveBeenCalledWith(
    'factory',
    'chestnut',
    'chestnut-copy',
    { expectedRevision: 0 },
  ));
  // Active mission unchanged after duplicating.
  expect(screen.getByRole('combobox', { name: 'Active mission' })).toHaveValue('chestnut');

  fireEvent.click(screen.getByRole('button', { name: 'Delete mission' }));
  expect(screen.getByText(/Delete mission "chestnut"\?/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Delete', exact: true }));

  await waitFor(() => expect(deleteNavigationMission).toHaveBeenCalledWith(
    'factory',
    'chestnut',
    { expectedRevision: 0 },
  ));
  // Switches to the first remaining mission from the refreshed catalog.
  await waitFor(() => expect(getNavigationMission).toHaveBeenCalledWith('factory', 'chestnut-copy'));
});

test('edits mission manifest waypoints without legacy spot persistence', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });
  getNavigationMission.mockImplementation((mapName) => Promise.resolve(
    mapName === 'factory'
      ? {
        exists: true,
        map_name: 'factory',
        global_bt: 'global.xml',
        waypoints: [{
          id: 'mission_pickup',
          label: 'Mission Pickup',
          pose: { frame_id: 'map', x: 3.5, y: -1.25, yaw: 1.57 },
          local_bt: 'locals/mission_pickup.xml',
          metadata: { role: 'pickup' },
        }],
        metadata: {},
      }
      : {
        exists: false,
        map_name: mapName,
        global_bt: 'global.xml',
        waypoints: [],
        metadata: {},
      },
  ));
  getNavigationSpots.mockImplementation((mapName) => Promise.resolve({
    map_name: mapName,
    spots: mapName === 'factory' ? [{
      id: 'legacy_spot',
      map_name: 'factory',
      label: 'Legacy Waypoint',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      linked_bt_tree: 'legacy.xml',
      metadata: {},
    }] : [],
  }));
  getNavigationMissionBtFile.mockImplementation((mapName, path) => Promise.resolve({
    path,
    exists: true,
    content: `<root><BehaviorTree ID="${mapName}:${path}"/></root>`,
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  expect(latestMapViewerProps().spots[0]).toMatchObject({
    id: 'mission_pickup',
    map_name: 'factory',
    label: 'Mission Pickup',
    linked_bt_tree: 'locals/mission_pickup.xml',
    pose: {
      frame_id: 'map',
      x: 3.5,
      y: -1.25,
      yaw: 1.57,
    },
    metadata: {
      role: 'pickup',
      source: 'mission_manifest',
      coordinate_space: 'map',
      local_bt: 'locals/mission_pickup.xml',
    },
  });
  expect(screen.getByRole('button', { name: 'Mission Pickup' })).toBeInTheDocument();
  expect(latestMapViewerProps().missionRouteOrder).toEqual([]);
  expect(screen.queryByText('legacy.xml')).not.toBeInTheDocument();
  expect(getNavigationSpots.mock.calls.some(([mapName]) => mapName === 'factory')).toBe(false);
  expect(getNavigationMissionBtFile).toHaveBeenCalledWith('factory', 'global.xml', '');
  expect(getNavigationMissionBtFile).toHaveBeenCalledWith('factory', 'locals/mission_pickup.xml', '');

  expect(screen.queryByRole('button', { name: 'Select Waypoints' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Move Waypoints' })).not.toBeInTheDocument();
  expect(latestMapViewerProps().onSpotPoseChange).toEqual(expect.any(Function));
  await act(async () => {
    await latestMapViewerProps().onSpotPoseChange('mission_pickup', 4, 5, 0.25);
  });

  expect(updateNavigationSpot).not.toHaveBeenCalled();
  await waitFor(() => expect(latestMapViewerProps().spots[0].pose).toMatchObject({
    x: 4,
    y: 5,
    yaw: 0.25,
  }));
  expect(latestMapViewerProps().onSpotPoseChange).toEqual(expect.any(Function));

  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalledWith(
    'factory',
    expect.objectContaining({
      waypoints: [expect.objectContaining({
        id: 'mission_pickup',
        pose: expect.objectContaining({ x: 4, y: 5, yaw: 0.25 }),
      })],
    }),
    '',
  ));

  fireEvent.doubleClick(screen.getByRole('button', { name: 'Mission Pickup' }));
  const waypointNameInput = screen.getByRole('textbox', { name: 'Waypoint name' });
  fireEvent.change(waypointNameInput, { target: { value: 'Mission Dropoff' } });
  fireEvent.keyDown(waypointNameInput, { key: 'Enter' });

  await waitFor(() => expect(screen.getByRole('button', { name: 'Mission Dropoff' })).toBeInTheDocument());
  expect(updateNavigationSpot).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: /Delete Waypoint Mission Dropoff/ }));
  await waitFor(() => expect(screen.queryByRole('button', { name: 'Mission Dropoff' })).not.toBeInTheDocument());
  expect(deleteNavigationSpot).not.toHaveBeenCalled();
});

test('wires legacy behavior nodes through move history and mission save', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const behaviorNode = {
    id: 'behavior_4_sequence',
    map_name: 'factory',
    tag: 'Sequence',
    label: 'Sequence',
    category: 'control',
    pose: { frame_id: 'map', x: 1, y: 2, yaw: 0.4 },
    metadata: { source: 'mission_canvas' },
  };
  window.localStorage.setItem('mission_canvas_designs', JSON.stringify({
    factory: { behaviorNodes: [behaviorNode] },
  }));
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions.mockResolvedValue({ map_name: 'factory', missions: ['default'] });
  getNavigationMission.mockResolvedValue({
    exists: false,
    map_name: 'factory',
    mission_name: 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(latestMapViewerProps().behaviorNodes).toEqual([behaviorNode]));
  act(() => latestMapViewerProps().onBehaviorNodeClick(behaviorNode.id));
  act(() => latestMapViewerProps().onBehaviorNodePoseChange(behaviorNode.id, 5, 6));
  await waitFor(() => expect(latestMapViewerProps().behaviorNodes[0].pose).toMatchObject({
    x: 5, y: 6, yaw: 0.4,
  }));

  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(latestMapViewerProps().behaviorNodes[0].pose.x).toBe(1));
  fireEvent.click(screen.getByRole('button', { name: 'Redo' }));
  await waitFor(() => expect(latestMapViewerProps().behaviorNodes[0].pose.x).toBe(5));

  const deleteNodeButton = screen.getByRole('button', { name: 'Delete Node Sequence' });
  expect(deleteNodeButton).toBeInTheDocument();
  fireEvent.click(deleteNodeButton);
  await waitFor(() => expect(latestMapViewerProps().behaviorNodes).toEqual([]));
  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(latestMapViewerProps().behaviorNodes[0].pose.x).toBe(5));

  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  await waitFor(() => {
    const stored = JSON.parse(window.localStorage.getItem('mission_canvas_designs'));
    expect(stored.factory.behaviorNodes[0].pose).toMatchObject({ x: 5, y: 6, yaw: 0.4 });
  });
});

test('hides loaded design waypoints after returning to mapping stage', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });
  getNavigationSpots.mockImplementation((mapName) => Promise.resolve({
    map_name: mapName,
    spots: mapName === 'factory' ? [{
      id: 'spot_factory',
      map_name: 'factory',
      label: 'Waypoint Factory',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      linked_bt_tree: '',
      metadata: {},
    }] : [],
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));

  fireEvent.click(screen.getByRole('tab', { name: 'Mapping' }));

  await waitFor(() => expect(latestMapViewerProps().spots).toEqual([]));
  expect(latestMapViewerProps().selectedSpotId).toBe('');
  expect(latestMapViewerProps().map).toBeNull();
});

test('renders legacy pixel-coordinate waypoints in loaded map coordinates', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 100,
    height: 100,
    resolution: 0.05,
    origin: {
      position: { x: -1, y: -2, z: 0 },
      orientation: { x: 0, y: 0, z: 0, w: 1 },
    },
    maxval: 255,
    pixels_base64: 'AA==',
  });
  getNavigationSpots.mockResolvedValue({
    map_name: 'factory',
    spots: [{
      id: 'legacy_spot',
      map_name: 'factory',
      label: 'Legacy Waypoint',
      pose: { frame_id: 'map', x: 50, y: 20, yaw: 0.5 },
      linked_bt_tree: '',
      metadata: {},
    }],
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(latestMapViewerProps().map).toMatchObject({
    info: { resolution: 0.05 },
  }));
  await waitFor(() => {
    expect(latestMapViewerProps().spots).toHaveLength(1);
    expect(latestMapViewerProps().spots[0].pose.x).toBeCloseTo(1.5);
    expect(latestMapViewerProps().spots[0].pose.y).toBeCloseTo(-1);
    expect(latestMapViewerProps().spots[0].metadata.coordinate_space).toBe('legacy_cell_display');
  });
});

test('shows waypoint actions in Waypoints after placing a waypoint', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().map).toMatchObject({
    info: { width: 1, height: 1 },
  }));

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'On Map' }));

  expect(screen.getByRole('button', { name: 'Create Waypoint' })).toHaveAttribute('aria-pressed', 'true');

  await act(async () => {
    await latestMapViewerProps().onMapPose(1, 2, 0.25);
  });

  await waitFor(() => expect(createNavigationSpot).toHaveBeenCalledWith({
    map_name: 'factory',
    label: 'Waypoint 1',
    pose: { frame_id: 'map', x: 1, y: 2, yaw: 0.25 },
    metadata: { source: 'mission_canvas', coordinate_space: 'map' },
  }));
  expect(screen.getByRole('button', { name: 'Waypoint A' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Create Waypoint' })).not.toHaveAttribute('aria-pressed', 'true');
  expect(latestMapViewerProps().interactionMode).toBe('view');
  expect(screen.queryByRole('button', { name: 'Create BT' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Edit Task for Waypoint A' })).toBeEnabled();
  expect(screen.getByRole('button', { name: /Delete Waypoint Waypoint A/ })).toBeEnabled();
  expect(screen.queryByRole('button', { name: 'Select Waypoints' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Move Waypoints' })).not.toBeInTheDocument();
  expect(latestMapViewerProps().onSpotPoseChange).toEqual(expect.any(Function));

  await act(async () => {
    await latestMapViewerProps().onSpotPoseChange('spot_a', 4, 5, 0.25);
  });

  await waitFor(() => expect(updateNavigationSpot).toHaveBeenCalledWith('spot_a', {
    map_name: 'factory',
    pose: { frame_id: 'map', x: 4, y: 5, yaw: 0.25 },
    metadata: {
      coordinate_space: 'map',
      linked_bt_tree: 'locals/spot_a/main.xml',
      local_bt: 'locals/spot_a/main.xml',
      local_bt_files: ['locals/spot_a/main.xml'],
    },
  }));
  await waitFor(() => expect(latestMapViewerProps().spots[0].pose).toMatchObject({
    x: 4,
    y: 5,
    yaw: 0.25,
  }));
  expect(latestMapViewerProps().onSpotPoseChange).toEqual(expect.any(Function));

  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  // "default" already exists in the catalog, so Save writes in place — no dialog.
  expect(screen.queryByRole('textbox', { name: 'Save mission name' })).not.toBeInTheDocument();

  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalledWith(
    'factory',
    expect.objectContaining({
      global_bt: 'global.xml',
      waypoints: [
        expect.objectContaining({
          id: 'spot_a',
          label: 'Waypoint A',
          local_bt: 'locals/spot_a/main.xml',
          pose: expect.objectContaining({
            frame_id: 'map',
            x: 4,
            y: 5,
            yaw: 0.25,
          }),
        }),
      ],
      metadata: expect.objectContaining({
        source: 'mission_canvas',
        mission_flow: expect.objectContaining({
          nodes: [expect.objectContaining({ id: 'spot_a' })],
          edges: [],
        }),
      }),
    }),
    '',
  ));
  await waitFor(() => expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'global.xml',
    expect.stringContaining('<Sequence name="GlobalMission"/>'),
    '',
    { expectedRevision: 0 },
  ));
  expect(saveNavigationMissionBtFile).not.toHaveBeenCalledWith(
    'factory',
    'compiled.xml',
    expect.any(String),
  );
  expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/spot_a/main.xml',
    expect.stringContaining('<BehaviorTree ID="MainTree"/>'),
    '',
    { expectedRevision: 0 },
  );

  await act(async () => {
    latestMapViewerProps().onMapClick(0, 0);
  });
  await waitFor(() => expect(latestMapViewerProps().selectedSpotId).toBe(''));
  expect(latestMapViewerProps().btLayer).toBeNull();
  expect(screen.queryByRole('button', { name: 'Create BT' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Edit Task for Waypoint A' })).toBeEnabled();

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'On Map' }));
  expect(screen.getByRole('button', { name: 'Create Waypoint' })).toHaveAttribute('aria-pressed', 'true');

  fireEvent.doubleClick(screen.getByRole('button', { name: 'Waypoint A' }));
  const waypointNameInput = screen.getByRole('textbox', { name: 'Waypoint name' });
  expect(waypointNameInput).toHaveValue('Waypoint A');
  fireEvent.change(waypointNameInput, { target: { value: 'Pickup A' } });
  fireEvent.keyDown(waypointNameInput, { key: 'Enter' });

  await waitFor(() => expect(updateNavigationSpot).toHaveBeenCalledWith('spot_a', {
    map_name: 'factory',
    label: 'Pickup A',
  }));
  expect(screen.getByRole('button', { name: 'Pickup A' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  await waitFor(() => expect(saveNavigationMission).toHaveBeenLastCalledWith(
    'factory',
    expect.objectContaining({
      waypoints: [
        expect.objectContaining({
          label: 'Pickup A',
          local_bt: 'locals/spot_a/main.xml',
          local_bt_files: ['locals/spot_a/main.xml'],
        }),
      ],
    }),
    '',
  ));
  expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/spot_a/main.xml',
    expect.any(String),
    '',
    { waypointId: 'spot_a', expectedRevision: 1 },
  );
  expect(deleteNavigationMissionBtFile).not.toHaveBeenCalledWith(
    'factory',
    'locals/spot_a/main.xml',
    '',
    expect.anything(),
  );

  fireEvent.click(screen.getByRole('button', { name: /Delete Waypoint Pickup A/ }));
  await waitFor(() => expect(deleteNavigationSpot).toHaveBeenCalledWith('spot_a', 'factory'));
  await waitFor(() => expect(screen.queryByRole('button', { name: 'Pickup A' })).not.toBeInTheDocument());
  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  await waitFor(() => expect(deleteNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/spot_a/main.xml',
    '',
    { expectedRevision: expect.any(Number) },
  ));
});

test('undoes and redoes Design waypoint edits from buttons and shortcuts', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().map).not.toBeNull());

  const undo = screen.getByRole('button', { name: 'Undo' });
  const redo = screen.getByRole('button', { name: 'Redo' });
  expect(undo).toBeDisabled();
  expect(redo).toBeDisabled();

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'On Map' }));
  await act(async () => {
    await latestMapViewerProps().onMapPose(1, 2, 0.25);
  });
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  expect(undo).toBeEnabled();
  expect(redo).toBeDisabled();

  fireEvent.click(undo);
  await waitFor(() => expect(latestMapViewerProps().spots).toEqual([]));
  expect(redo).toBeEnabled();

  fireEvent.click(redo);
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));

  fireEvent.keyDown(document, { key: 'z', ctrlKey: true });
  await waitFor(() => expect(latestMapViewerProps().spots).toEqual([]));

  fireEvent.keyDown(document, { key: 'y', ctrlKey: true });
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
});

test('opens waypoint BT from the rail while map selection stays runtime-independent', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });
  getNavigationSpots.mockImplementation((mapName) => Promise.resolve({
    map_name: mapName,
    spots: mapName === 'factory' ? [{
      id: 'spot_factory',
      map_name: 'factory',
      label: 'Waypoint Factory',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0.25 },
      linked_bt_tree: 'factory_waypoint.xml',
      metadata: {},
    }] : [],
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  expect(latestMapViewerProps().onSpotPoseChange).toEqual(expect.any(Function));
  expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Edit Task for Waypoint Factory' })).toBeEnabled();

  act(() => {
    latestMapViewerProps().onSpotClick('spot_factory');
  });
  await waitFor(() => expect(latestMapViewerProps().selectedSpotId).toBe('spot_factory'));
  expect(latestMapViewerProps().btLayer).toBeNull();

  act(() => {
    latestMapViewerProps().onMapClick(0, 0);
  });
  await waitFor(() => expect(latestMapViewerProps().selectedSpotId).toBe(''));
  expect(latestMapViewerProps().btLayer).toBeNull();

  const mapBeforeBt = latestMapViewerProps().map;
  const viewKeyBeforeBt = latestMapViewerProps().viewKey;
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Waypoint Factory' }));

  await waitFor(() => expect(latestMapViewerProps().btLayer).toMatchObject({
    spot: {
      id: 'spot_factory',
      label: 'Waypoint Factory',
      linked_bt_tree: 'factory_waypoint.xml',
    },
  }));
  expect(latestMapViewerProps().btLayer).not.toHaveProperty('fullCanvas');
  expect(latestMapViewerProps().btLayer).not.toHaveProperty('focusMap');
  expect(latestMapViewerProps().btLayer.editor.props.fileActionsDisabled).toBe(false);
  expect(latestMapViewerProps().btLayer.editor.props.title)
    .toBe('Waypoint Factory Waypoint Task');
  expect(screen.getByText('· Waypoint Task')).toBeInTheDocument();
  expect(latestMapViewerProps().onSpotClick).toBeUndefined();
  expect(latestMapViewerProps().onSpotPoseChange).toBeUndefined();
  expect(latestMapViewerProps().onMapClick).toBeUndefined();
  expect(screen.getByRole('button', { name: /Back to Map/ })).toBeEnabled();
  expect(global.fetch.mock.calls.some(([url, options]) => (
    String(url).includes('/api/services/bt_node/start')
    && options?.method === 'POST'
  ))).toBe(false);

  const draftXml = [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait name="DraftStep" duration="1.0"/></BehaviorTree>',
    '</root>',
  ].join('\n');
  await act(async () => {
    const editor = latestMapViewerProps().btLayer.editor;
    await editor.props.onSaveXml(editor.props.filePath, draftXml);
  });
  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalledWith(
    'factory',
    expect.objectContaining({
      waypoints: [expect.objectContaining({ id: 'spot_factory' })],
    }),
    '',
  ));
  expect(saveNavigationMissionBtFile.mock.calls.some(([
    mapName,
    path,
    content,
  ]) => (
    mapName === 'factory'
    && path.startsWith('locals/')
    && content.includes('DraftStep')
  ))).toBe(true);
  expect(screen.queryByRole('dialog', { name: 'Waypoint BT' })).not.toBeInTheDocument();
  expect(screen.getAllByText('Waypoint Factory').length).toBeGreaterThan(0);
  expect(screen.queryByText('factory_waypoint.xml')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Create BT' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Edit Task' })).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: /Back to Map/ }));

  await waitFor(() => expect(latestMapViewerProps().btLayer).toBeNull());
  expect(latestMapViewerProps().selectedSpotId).toBe('spot_factory');
  expect(latestMapViewerProps().map).toBe(mapBeforeBt);
  expect(latestMapViewerProps().viewKey).toBe(viewKeyBeforeBt);
  expect(screen.getByRole('button', { name: 'Edit Task for Waypoint Factory' })).toBeEnabled();
});

test('loads and saves each waypoint XML through its mission-local storage path', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const xmlFor = (name) => [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    `  <BehaviorTree ID="MainTree"><Wait name="${name}" duration="1.0"/></BehaviorTree>`,
    '</root>',
  ].join('\n');
  const stored = {
    'global.xml': '<root BTCPP_format="4" main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"/></root>',
    'locals/a.xml': xmlFor('StoredA'),
    'locals/b.xml': xmlFor('StoredB'),
  };
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getNavigationMissions.mockResolvedValue({ map_name: 'factory', missions: ['inspection'] });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    mission_name: 'inspection',
    revision: 7,
    global_bt: 'global.xml',
    waypoints: [
      {
        id: 'wp_a',
        label: 'A',
        pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 },
        local_bt: 'locals/a.xml',
        metadata: {},
      },
      {
        id: 'wp_b',
        label: 'B',
        pose: { frame_id: 'map', x: 2, y: 0, yaw: 0 },
        local_bt: 'locals/b.xml',
        metadata: {},
      },
    ],
    metadata: {},
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: stored[path],
    exists: true,
  }));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Design mission file' }))
    .toHaveValue('inspection'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(2));

  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for A' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.editor).toBeTruthy());
  expect(latestMapViewerProps().btLayer.editor.props.fileActionsDisabled).toBe(false);
  getNavigationMissionBtFile.mockClear();
  saveNavigationMissionBtFile.mockClear();
  saveNavigationMission.mockClear();
  stored['locals/a.xml'] = xmlFor('ReloadedA');
  let finishReload;
  getNavigationMissionBtFile.mockReturnValueOnce(new Promise((resolve) => {
    finishReload = resolve;
  }));
  let reloadPromise;
  act(() => {
    reloadPromise = latestMapViewerProps().btLayer.editor.props.onLoadXml('locals/a.xml');
  });
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.fileActionsDisabled)
    .toBe(true));
  act(() => latestMapViewerProps().onBtLayerClose());
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toBeDisabled());
  await act(async () => {
    finishReload({
      path: 'locals/a.xml',
      content: stored['locals/a.xml'],
      exists: true,
    });
    await reloadPromise;
  });
  expect(getNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/a.xml',
    'inspection',
  );
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for A' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.xml)
    .toContain('ReloadedA'));

  let finishStaleReload;
  getNavigationMissionBtFile.mockReturnValueOnce(new Promise((resolve) => {
    finishStaleReload = resolve;
  }));
  let staleReloadPromise;
  act(() => {
    staleReloadPromise = latestMapViewerProps().btLayer.editor.props.onLoadXml('locals/a.xml');
  });
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.fileActionsDisabled)
    .toBe(true));
  const editedDuringLoad = xmlFor('EditedDuringLoad');
  act(() => latestMapViewerProps().btLayer.editor.props.onXmlChange(
    'locals/a.xml',
    editedDuringLoad,
  ));
  await act(async () => {
    finishStaleReload({
      path: 'locals/a.xml',
      content: xmlFor('LateDiskA'),
      exists: true,
    });
    await expect(staleReloadPromise).rejects.toThrow(
      'Waypoint Task changed while its saved file was loading',
    );
  });
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.xml)
    .toBe(editedDuringLoad));

  const editedA = xmlFor('EditedA');
  await act(async () => {
    await latestMapViewerProps().btLayer.editor.props.onSaveXml('locals/a.xml', editedA);
  });
  expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/a.xml',
    editedA,
    'inspection',
    { waypointId: 'wp_a', expectedRevision: 7 },
  );

  fireEvent.click(screen.getByRole('button', { name: /Back to Map/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for B' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.filePath)
    .toBe('locals/b.xml'));
  const editedB = xmlFor('EditedB');
  await act(async () => {
    await latestMapViewerProps().btLayer.editor.props.onSaveXml('locals/b.xml', editedB);
  });
  expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/b.xml',
    editedB,
    'inspection',
    { waypointId: 'wp_b', expectedRevision: 7 },
  );
  expect(saveNavigationMission).not.toHaveBeenCalled();
  act(() => latestMapViewerProps().onBtLayerClose());
  fireEvent.click(await screen.findByRole('button', { name: 'New Mission' }));
  expect(screen.queryByText('· unsaved')).not.toBeInTheDocument();
  await waitFor(() => expect(latestMapViewerProps().spots).toEqual([]));
});

test('keeps a waypoint XML library separate from its changeable runtime default', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const xmlFor = (name) => [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    `  <BehaviorTree ID="MainTree"><Wait name="${name}" duration="1.0"/></BehaviorTree>`,
    '</root>',
  ].join('\n');
  const stored = {
    'global.xml': xmlFor('Global'),
    'locals/a.xml': xmlFor('DefaultA'),
    'locals/a_alt.xml': xmlFor('AlternateA'),
  };
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getNavigationMissions.mockResolvedValue({ map_name: 'factory', missions: ['inspection'] });
  let serverRevision = 7;
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    mission_name: 'inspection',
    revision: serverRevision,
    global_bt: 'global.xml',
    waypoints: [{
      id: 'wp_a',
      label: 'A',
      pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 },
      local_bt: 'locals/a.xml',
      local_bt_files: ['locals/a.xml', 'locals/a_alt.xml'],
      metadata: {},
    }],
    metadata: {},
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: stored[path],
    exists: true,
    revision: serverRevision,
  }));
  saveNavigationMissionBtFile.mockImplementation((
    _mapName,
    path,
    content,
    _missionName,
    options,
  ) => {
    expect(options.expectedRevision).toBe(serverRevision);
    serverRevision += 1;
    return Promise.resolve({ path, content, exists: true, revision: serverRevision });
  });
  setNavigationMissionDefaultBtFile.mockImplementation((
    _mapName,
    _waypointId,
    _path,
    _missionName,
    options,
  ) => {
    expect(options.expectedRevision).toBe(serverRevision);
    serverRevision += 1;
    return Promise.resolve({ exists: true, revision: serverRevision });
  });
  saveNavigationMission.mockImplementation((_mapName, payload) => {
    expect(payload.expected_revision).toBe(serverRevision);
    serverRevision += 1;
    return Promise.resolve({ exists: true, revision: serverRevision });
  });
  deleteNavigationMissionBtFile.mockImplementation((
    _mapName,
    path,
    _missionName,
    options,
  ) => {
    expect(options.expectedRevision).toBe(serverRevision);
    serverRevision += 1;
    return Promise.resolve({ path, content: '', exists: false, revision: serverRevision });
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Design mission file' }))
    .toHaveValue('inspection'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for A' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.editor).toBeTruthy());

  expect(latestMapViewerProps().btLayer.editor.props).toMatchObject({
    filePath: 'locals/a.xml',
    defaultFilePath: 'locals/a.xml',
    fileOptions: ['locals/a.xml', 'locals/a_alt.xml'],
  });

  await act(async () => {
    await latestMapViewerProps().btLayer.editor.props.onLoadXml('locals/a_alt.xml');
    latestMapViewerProps().btLayer.editor.props.onFilePathChange('locals/a_alt.xml');
  });
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.filePath)
    .toBe('locals/a_alt.xml'));
  expect(latestMapViewerProps().btLayer.editor.props.defaultFilePath).toBe('locals/a.xml');

  const editedAlternate = xmlFor('EditedAlternate');
  await act(async () => {
    await latestMapViewerProps().btLayer.editor.props.onSaveXml(
      'locals/a_alt.xml',
      editedAlternate,
    );
  });
  expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/a_alt.xml',
    editedAlternate,
    'inspection',
    { waypointId: 'wp_a', expectedRevision: 7 },
  );

  const thirdXml = xmlFor('ThirdA');
  let saveAsResponse;
  await act(async () => {
    saveAsResponse = await latestMapViewerProps().btLayer.editor.props.onSaveXmlAs(
      'locals/a_alt.xml',
      'third',
      thirdXml,
    );
  });
  expect(saveAsResponse).toMatchObject({
    path: 'locals/wp_a/third.xml',
    selected: true,
  });
  expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/wp_a/third.xml',
    thirdXml,
    'inspection',
    { waypointId: 'wp_a', expectedRevision: 8 },
  );
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props).toMatchObject({
    filePath: 'locals/wp_a/third.xml',
    defaultFilePath: 'locals/a.xml',
    fileOptions: ['locals/a.xml', 'locals/a_alt.xml', 'locals/wp_a/third.xml'],
  }));

  await act(async () => {
    await latestMapViewerProps().btLayer.editor.props.onSetDefaultXml('locals/wp_a/third.xml');
  });
  expect(setNavigationMissionDefaultBtFile).toHaveBeenCalledWith(
    'factory',
    'wp_a',
    'locals/wp_a/third.xml',
    'inspection',
    { expectedRevision: 9 },
  );
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.defaultFilePath)
    .toBe('locals/wp_a/third.xml'));

  act(() => latestMapViewerProps().onBtLayerClose());
  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  await waitFor(() => expect(saveNavigationMission).toHaveBeenLastCalledWith(
    'factory',
    expect.objectContaining({
      waypoints: [expect.objectContaining({
        id: 'wp_a',
        local_bt: 'locals/wp_a/third.xml',
        local_bt_files: [
          'locals/wp_a/third.xml',
          'locals/wp_a/a.xml',
          'locals/wp_a/a_alt.xml',
        ],
      })],
    }),
    'inspection',
  ));
  await waitFor(() => {
    ['locals/a.xml', 'locals/a_alt.xml'].forEach((path) => {
      expect(deleteNavigationMissionBtFile).toHaveBeenCalledWith(
        'factory',
        path,
        'inspection',
        { expectedRevision: expect.any(Number) },
      );
    });
  });

  fireEvent.click(screen.getByRole('button', { name: /Delete Waypoint A/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  await waitFor(() => {
    ['locals/wp_a/a.xml', 'locals/wp_a/a_alt.xml', 'locals/wp_a/third.xml'].forEach((path) => {
      expect(deleteNavigationMissionBtFile).toHaveBeenCalledWith(
        'factory',
        path,
        'inspection',
        { expectedRevision: expect.any(Number) },
      );
    });
  });
});

test('saves the latest local BT snapshot after closing the editor', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const originalXml = [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait name="Before" duration="1.0"/></BehaviorTree>',
    '</root>',
  ].join('\n');
  const editedXml = originalXml.replace('Before', 'After').replace('1.0', '3.5');
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    global_bt: 'global.xml',
    waypoints: [{
      id: 'wp1',
      label: 'Pickup',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      local_bt: 'locals/wp1.xml',
      metadata: { local_bt: 'locals/wp1.xml' },
    }],
    metadata: {},
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: path === 'locals/wp1.xml' ? originalXml : '<root/>',
    exists: true,
  }));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.editor).toBeTruthy());

  const editor = latestMapViewerProps().btLayer.editor;
  act(() => {
    editor.props.onXmlChange('locals/wp1.xml', editedXml);
    latestMapViewerProps().onBtLayerClose();
  });
  fireEvent.click(await screen.findByRole('button', { name: 'Save Mission' }));

  await waitFor(() => expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/wp1/main.xml',
    editedXml,
    '',
    { expectedRevision: 0 },
  ));
  expect(Math.max(...saveNavigationMissionBtFile.mock.invocationCallOrder))
    .toBeLessThan(saveNavigationMission.mock.invocationCallOrder[0]);
});

test('saves the local BT restored by Design undo', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const originalXml = [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait name="Before" duration="1.0"/></BehaviorTree>',
    '</root>',
  ].join('\n');
  const editedXml = originalXml.replace('Before', 'After');
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    global_bt: 'global.xml',
    waypoints: [{
      id: 'wp1',
      label: 'Pickup',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      local_bt: 'locals/wp1.xml',
      metadata: { local_bt: 'locals/wp1.xml' },
    }],
    metadata: {},
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: path === 'locals/wp1.xml' ? originalXml : '<root/>',
    exists: true,
  }));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.editor).toBeTruthy());
  act(() => {
    latestMapViewerProps().btLayer.editor.props.onXmlChange('locals/wp1.xml', editedXml);
    latestMapViewerProps().onBtLayerClose();
  });

  const undo = screen.getByRole('button', { name: 'Undo' });
  await waitFor(() => expect(undo).toBeEnabled());
  fireEvent.click(undo);
  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));

  await waitFor(() => expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/wp1/main.xml',
    originalXml,
    '',
    { expectedRevision: 0 },
  ));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled());
  expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled();
});

test('blocks saving when any local BT file fails to load', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    global_bt: 'global.xml',
    waypoints: [{
      id: 'wp1',
      label: 'Pickup',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      local_bt: 'locals/wp1.xml',
      metadata: { local_bt: 'locals/wp1.xml' },
    }],
    metadata: {},
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => {
    if (path === 'locals/wp1.xml') {
      return Promise.reject(new Error('Local BT unavailable'));
    }
    return Promise.resolve({ path, content: '<root/>', exists: true });
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(screen.getByRole('button', { name: 'Save Mission' })).toBeDisabled());
  expect(latestMapViewerProps().spots).toEqual([]);
  const saveButton = screen.getByRole('button', { name: 'Save Mission' });
  expect(saveButton).toBeDisabled();
  fireEvent.click(saveButton);
  expect(saveNavigationMission).not.toHaveBeenCalled();
  expect(saveNavigationMissionBtFile).not.toHaveBeenCalled();
});

test('keeps edits made while a mission save is in flight', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const originalXml = [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait name="Before" duration="1.0"/></BehaviorTree>',
    '</root>',
  ].join('\n');
  const savedXml = originalXml.replace('Before', 'SavedSnapshot');
  const newerXml = originalXml.replace('Before', 'EditedDuringSave');
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    global_bt: 'global.xml',
    waypoints: [{
      id: 'wp1',
      label: 'Pickup',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      local_bt: 'legacy.xml',
      metadata: { local_bt: 'legacy.xml' },
    }],
    metadata: {},
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: path === 'legacy.xml' ? originalXml : '<root/>',
    exists: true,
  }));
  let finishLocalSave;
  let canonicalPath = '';
  saveNavigationMissionBtFile.mockImplementation((_mapName, path) => {
    if (path === 'global.xml') return Promise.resolve({ path, exists: true });
    canonicalPath = path;
    return new Promise((resolve) => { finishLocalSave = resolve; });
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.editor).toBeTruthy());

  act(() => {
    latestMapViewerProps().btLayer.editor.props.onXmlChange('legacy.xml', savedXml);
    latestMapViewerProps().onBtLayerClose();
  });
  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  await waitFor(() => expect(finishLocalSave).toEqual(expect.any(Function)));
  expect(canonicalPath).toBe('locals/wp1/main.xml');
  expect(saveNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    canonicalPath,
    savedXml,
    '',
    { expectedRevision: 0 },
  );
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.editor).toBeTruthy());
  act(() => latestMapViewerProps().btLayer.editor.props.onXmlChange(
    'legacy.xml',
    newerXml,
  ));
  await act(async () => {
    finishLocalSave({ path: canonicalPath, exists: true });
    await Promise.resolve();
  });

  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.filePath)
    .toBe(canonicalPath));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.xml).toBe(newerXml));
  act(() => latestMapViewerProps().onBtLayerClose());

  // Saving rebases history at the durable snapshot. The edit made while the
  // request was in flight remains one Undo/Redo step instead of being lost or
  // returning to the pre-canonical local-BT path generation.
  await waitFor(() => expect(screen.getByRole('button', { name: 'Undo' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.filePath)
    .toBe(canonicalPath));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.xml).toBe(savedXml));
  act(() => latestMapViewerProps().onBtLayerClose());
  await waitFor(() => expect(screen.getByRole('button', { name: 'Redo' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Redo' }));
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.xml).toBe(newerXml));
  act(() => latestMapViewerProps().onBtLayerClose());

  const newMissionButton = await screen.findByRole('button', { name: 'New Mission' });
  await waitFor(() => expect(newMissionButton).toBeEnabled());
  fireEvent.click(newMissionButton);
  expect(await screen.findByText('Unsaved changes')).toBeInTheDocument();
});

test('rebases Save As history across the new mission identity', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const originalXml = [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait name="Before" duration="1.0"/></BehaviorTree>',
    '</root>',
  ].join('\n');
  const savedXml = originalXml.replace('Before', 'SavedAsSnapshot');
  const newerXml = originalXml.replace('Before', 'EditedDuringSaveAs');
  const targetMissionName = 'saved_as_route';
  const canonicalPath = 'locals/wp1/main.xml';
  let globalSaveOptions;
  let localSaveRequest;
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getNavigationMissions
    .mockResolvedValueOnce({ map_name: 'factory', missions: [] })
    .mockResolvedValue({ map_name: 'factory', missions: [targetMissionName] });
  getNavigationMission.mockImplementation((mapName, missionName) => Promise.resolve({
    exists: false,
    revision: missionName === targetMissionName ? 7 : 0,
    map_name: mapName,
    mission_name: missionName || 'default',
    global_bt: 'global.xml',
    waypoints: [],
    metadata: {},
  }));
  getNavigationSpots.mockResolvedValue({
    map_name: 'factory',
    spots: [{
      id: 'wp1',
      map_name: 'factory',
      label: 'Pickup',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      linked_bt_tree: 'legacy.xml',
      metadata: { local_bt: 'legacy.xml' },
    }],
  });
  let finishLocalSave;
  saveNavigationMissionBtFile.mockImplementation((
    _mapName,
    path,
    content,
    missionName,
    options,
  ) => {
    if (path === 'global.xml') {
      globalSaveOptions = options;
      return Promise.resolve({ path, content, exists: true, revision: 8 });
    }
    localSaveRequest = { path, content, missionName, options };
    return new Promise((resolve) => { finishLocalSave = resolve; });
  });
  saveNavigationMission.mockImplementation((_mapName, payload, missionName) => {
    expect(payload.expected_revision).toBe(9);
    expect(missionName).toBe(targetMissionName);
    return Promise.resolve({ exists: true, revision: 10 });
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));

  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.editor).toBeTruthy());
  act(() => {
    latestMapViewerProps().btLayer.editor.props.onXmlChange('legacy.xml', savedXml);
    latestMapViewerProps().onBtLayerClose();
  });

  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  const nameInput = screen.getByRole('textbox', { name: 'Save mission name' });
  fireEvent.change(nameInput, { target: { value: targetMissionName } });
  fireEvent.click(screen.getByRole('button', { name: 'Save', exact: true }));
  await waitFor(() => expect(finishLocalSave).toEqual(expect.any(Function)));
  expect(globalSaveOptions).toEqual({ expectedRevision: 7 });
  expect(localSaveRequest).toEqual({
    path: canonicalPath,
    content: savedXml,
    missionName: targetMissionName,
    options: { expectedRevision: 8 },
  });

  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.editor).toBeTruthy());
  act(() => latestMapViewerProps().btLayer.editor.props.onXmlChange(
    'legacy.xml',
    newerXml,
  ));

  await act(async () => {
    finishLocalSave({
      path: canonicalPath,
      content: savedXml,
      exists: true,
      revision: 9,
    });
    await Promise.resolve();
  });

  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.filePath)
    .toBe(canonicalPath));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.xml).toBe(newerXml));
  act(() => latestMapViewerProps().onBtLayerClose());
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue(targetMissionName));

  await waitFor(() => expect(screen.getByRole('button', { name: 'Undo' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue(targetMissionName);
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.filePath)
    .toBe(canonicalPath));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.xml).toBe(savedXml));
  act(() => latestMapViewerProps().onBtLayerClose());

  await waitFor(() => expect(screen.getByRole('button', { name: 'Redo' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Redo' }));
  expect(screen.getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue(targetMissionName);
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.xml).toBe(newerXml));
  act(() => latestMapViewerProps().onBtLayerClose());
});

test('retries a partially completed mission save from the latest server revision', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const originalXml = [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait name="Before" duration="1.0"/></BehaviorTree>',
    '</root>',
  ].join('\n');
  const editedXml = originalXml.replace('Before', 'After');
  let globalUploads = 0;
  let localUploads = 0;
  const globalExpectedRevisions = [];
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    mission_name: 'default',
    revision: 5,
    global_bt: 'global.xml',
    waypoints: [{
      id: 'wp1',
      label: 'Pickup',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      local_bt: 'locals/wp1.xml',
      local_bt_files: ['locals/wp1.xml'],
      metadata: {},
    }],
    metadata: {},
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: path === 'locals/wp1.xml' ? originalXml : '<root/>',
    exists: true,
    revision: 5,
  }));
  saveNavigationMissionBtFile.mockImplementation((
    _mapName,
    path,
    content,
    _missionName,
    options,
  ) => {
    if (path === 'global.xml') {
      globalUploads += 1;
      globalExpectedRevisions.push(options.expectedRevision);
      return Promise.resolve({ path, content, exists: true, revision: 6 });
    }
    localUploads += 1;
    expect(options).toEqual({ expectedRevision: 6 });
    if (localUploads === 1) return Promise.reject(new Error('Local upload interrupted'));
    return Promise.resolve({ path, content, exists: true, revision: 7 });
  });
  saveNavigationMission.mockImplementation((_mapName, payload) => {
    expect(payload.expected_revision).toBe(7);
    return Promise.resolve({ exists: true, revision: 8 });
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Pickup' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.editor).toBeTruthy());
  act(() => {
    latestMapViewerProps().btLayer.editor.props.onXmlChange('locals/wp1.xml', editedXml);
    latestMapViewerProps().onBtLayerClose();
  });

  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save Mission' })).toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save Mission' })).toBeEnabled());
  expect(globalUploads).toBe(2);
  expect(globalExpectedRevisions).toEqual([5, 6]);
  expect(localUploads).toBe(2);
  expect(saveNavigationMission).toHaveBeenCalledTimes(1);
});

test('locks stage navigation while At Robot localization is starting', async () => {
  let resolveNavigationStart;
  let resolveAmclConfiguration;
  startNavigation.mockReturnValue(new Promise((resolve) => {
    resolveNavigationStart = resolve;
  }));
  configureDesignLocalizationAmcl.mockReturnValue(new Promise((resolve) => {
    resolveAmclConfiguration = resolve;
  }));
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Create Waypoint' }))
    .toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'At Robot' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('localize', 'factory'));

  const designTab = screen.getByRole('tab', { name: 'Design' });
  const mappingTab = screen.getByRole('tab', { name: 'Mapping' });
  const runTab = screen.getByRole('tab', { name: 'Run' });
  expect(mappingTab).toBeDisabled();
  expect(runTab).toBeDisabled();
  fireEvent.click(mappingTab);
  fireEvent.click(runTab);
  expect(designTab).toHaveAttribute('aria-selected', 'true');

  await act(async () => {
    resolveNavigationStart({ ok: true, message: 'started' });
    await Promise.resolve();
  });
  await waitFor(() => expect(configureDesignLocalizationAmcl).toHaveBeenCalledTimes(1));
  expect(mappingTab).toBeDisabled();
  expect(runTab).toBeDisabled();
  fireEvent.click(runTab);
  expect(designTab).toHaveAttribute('aria-selected', 'true');

  await act(async () => {
    resolveAmclConfiguration({ ok: true });
    await Promise.resolve();
  });
  await waitFor(() => expect(mappingTab).toBeEnabled());
  expect(runTab).toBeEnabled();
  expect(designTab).toHaveAttribute('aria-selected', 'true');
});

test('creates a waypoint at robot with automatic localization from the waypoint menu', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getServiceStatus
    .mockResolvedValueOnce({ is_up: false })
    .mockResolvedValue({ is_up: true });
  mockTopicDataByName['/tf'] = {
    transforms: [{
      header: { frame_id: 'map' },
      child_frame_id: 'base_link',
      transform: {
        translation: { x: 1.25, y: -0.5, z: 0 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
      },
    }],
  };
  mockTopicDataByName['/amcl_pose'] = amclPoseMessage(9, 9, 0);
  sendInitialPoseEstimate.mockImplementationOnce(async () => {
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(1.25, -0.5, 0.75);
    return { ok: true };
  });
  stopNavigation.mockImplementationOnce(async () => {
    getServiceStatus.mockResolvedValue({ is_up: false });
    return { ok: true };
  });
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  expect(screen.queryByRole('button', { name: 'Set Robot Pose' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'At Robot' }));

  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('localize', 'factory'));
  await waitFor(() => expect(configureDesignLocalizationAmcl).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  await waitFor(() => expect(latestMapViewerProps().showScan).toBe(true));
  await act(async () => {
    latestMapViewerProps().onMapPose(1.25, -0.5, 0.75);
  });
  await waitFor(() => expect(sendInitialPoseEstimate).toHaveBeenCalledWith({
    x: 1.25,
    y: -0.5,
    yaw: 0.75,
    frameId: 'map',
    mapName: 'factory',
  }));
  await waitFor(() => expect(requestNoMotionUpdate).toHaveBeenCalledTimes(3), { timeout: 4000 });
  await waitFor(() => expect(createNavigationSpot).toHaveBeenCalled(), { timeout: 5000 });
  const [payload] = createNavigationSpot.mock.calls[0];
  expect(payload.map_name).toBe('factory');
  expect(payload.label).toBe('Waypoint 1');
  expect(payload.pose.x).toBeCloseTo(1.25);
  expect(payload.pose.y).toBeCloseTo(-0.5);
  expect(payload.pose.yaw).toBeCloseTo(0.75);
  expect(payload.metadata).toEqual({ source: 'mission_canvas', coordinate_space: 'map' });
  await waitFor(() => expect(stopNavigation).toHaveBeenCalled());
  expect(mockPublishRosTopic).not.toHaveBeenCalledWith(
    '/initialpose',
    expect.any(String),
    expect.any(Object),
  );
  await waitFor(() => expect(latestMapViewerProps().showScan).toBe(false));
  await waitFor(() => expect(getServiceStatus.mock.calls.length).toBeGreaterThan(2));
  expect(latestMapViewerProps().map).not.toBeNull();
  expect(latestMapViewerProps().showRobotModel).toBe(false);

  fireEvent.click(screen.getByRole('tab', { name: 'Mapping' }));
  const startMappingButton = screen.getByRole('button', { name: 'Start Mapping' });
  await waitFor(() => expect(latestMapViewerProps().showScan).toBe(false));
  expect(latestMapViewerProps().showRobotModel).toBe(false);
  expect(latestMapViewerProps().pose).toBeNull();
  expect(startMappingButton).toBeEnabled();
  expect(startMappingButton).not.toHaveAttribute('aria-pressed');
  expect(screen.getByRole('button', { name: 'Stop' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Save Map' })).toBeDisabled();
});

test('finalizes a failed At Robot waypoint create and allows retry', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getServiceStatus
    .mockResolvedValueOnce({ is_up: false })
    .mockResolvedValue({ is_up: true });
  mockTopicDataByName['/amcl_pose'] = amclPoseMessage(0, 0, 0);
  sendInitialPoseEstimate.mockImplementation(async ({ x, y, yaw }) => {
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(x, y, yaw);
    return { ok: true };
  });
  createNavigationSpot
    .mockRejectedValueOnce(new Error('spot create failed'))
    .mockResolvedValueOnce({
      id: 'spot_retry',
      map_name: 'factory',
      label: 'Waypoint 1',
      pose: { frame_id: 'map', x: 2, y: 3, yaw: 0.5 },
      linked_bt_tree: '',
      metadata: {},
    });
  stopNavigation.mockResolvedValue({ ok: true });
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'At Robot' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  await act(async () => latestMapViewerProps().onMapPose(1, 1, 0));
  await waitFor(() => expect(createNavigationSpot).toHaveBeenCalledTimes(1), { timeout: 5000 });
  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(latestMapViewerProps().showScan).toBe(false));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'At Robot' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  await act(async () => latestMapViewerProps().onMapPose(2, 3, 0.5));
  await waitFor(() => expect(createNavigationSpot).toHaveBeenCalledTimes(2), { timeout: 5000 });
  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(2));
  expect(createNavigationSpot.mock.calls[1][0].pose).toMatchObject({ x: 2, y: 3 });
});

test.each([
  ['On Map', false],
  ['At Robot', true],
])('keeps a newly created %s waypoint BT independent from a stored Start BT', async (
  creationMode,
  createAtRobot,
) => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const startBtPath = 'locals/waypoint_1/main.xml';
  const startOnlyXml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait name="StartOnly" duration="7.0"/></BehaviorTree>',
    '</root>',
    '',
  ].join('\n');
  const globalXml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"/>',
    '</root>',
    '',
  ].join('\n');

  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });
  getNavigationMissions.mockResolvedValue({
    map_name: 'factory',
    missions: ['default'],
  });
  getNavigationMission.mockResolvedValue({
    exists: true,
    revision: 0,
    map_name: 'factory',
    mission_name: 'default',
    global_bt: 'global.xml',
    waypoints: [{
      id: 'Waypoint_1_ee992021',
      label: 'Start',
      pose: { frame_id: 'map', x: 0, y: 0, yaw: 0 },
      local_bt: startBtPath,
      local_bt_files: [startBtPath],
      metadata: {
        local_bt: startBtPath,
        local_bt_files: [startBtPath],
      },
    }],
    metadata: {},
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: path === startBtPath ? startOnlyXml : globalXml,
    exists: path === startBtPath || path === 'global.xml',
    revision: 0,
  }));
  createNavigationSpot.mockImplementation((payload) => Promise.resolve({
    id: 'Waypoint_2_d58de4b1',
    map_name: payload.map_name,
    label: payload.label,
    pose: payload.pose,
    linked_bt_tree: payload.linked_bt_tree || '',
    local_bt_files: payload.local_bt_files || [],
    metadata: payload.metadata || {},
  }));

  if (createAtRobot) {
    getServiceStatus
      .mockResolvedValueOnce({ is_up: false })
      .mockResolvedValue({ is_up: true, mode: 'localize' });
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(9, 9, 0);
    sendInitialPoseEstimate.mockImplementationOnce(async () => {
      mockTopicDataByName['/amcl_pose'] = amclPoseMessage(2.5, -1.25, 0.4);
      return { ok: true };
    });
    stopNavigation.mockImplementationOnce(async () => {
      getServiceStatus.mockResolvedValue({ is_up: false });
      return { ok: true };
    });
  }

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  await waitFor(() => expect(getNavigationMissionBtFile)
    .toHaveBeenCalledWith('factory', startBtPath, ''));

  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: creationMode }));
  if (createAtRobot) {
    await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  }
  await act(async () => {
    await latestMapViewerProps().onMapPose(2.5, -1.25, 0.4);
  });
  await waitFor(() => expect(createNavigationSpot).toHaveBeenCalled(), { timeout: 6000 });
  expect(createNavigationSpot).toHaveBeenCalledWith(expect.objectContaining({
    label: 'Waypoint 2',
  }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(2));

  const newSpot = latestMapViewerProps().spots.find(
    ({ id }) => id === 'Waypoint_2_d58de4b1',
  );
  expect(newSpot).toBeDefined();
  const newBtPath = newSpot.linked_bt_tree || newSpot.metadata?.local_bt;
  expect(newBtPath).toBe('locals/waypoint_2/main.xml');
  expect(newBtPath).not.toBe(startBtPath);
  expect(newSpot.local_bt_files || newSpot.metadata?.local_bt_files)
    .toEqual([newBtPath]);

  fireEvent.click(screen.getByRole('button', { name: 'Edit Task for Waypoint 2' }));
  await waitFor(() => expect(latestMapViewerProps().btLayer?.spot?.id)
    .toBe('Waypoint_2_d58de4b1'));
  await waitFor(() => expect(latestMapViewerProps().btLayer.editor.props.filePath)
    .toBe(newBtPath));
  const newWaypointXml = latestMapViewerProps().btLayer.editor.props.xml;
  expect(newWaypointXml).toContain('<BehaviorTree ID="MainTree"/>');
  expect(newWaypointXml).not.toContain('StartOnly');
  act(() => latestMapViewerProps().onBtLayerClose());

  const saveButton = screen.getByRole('button', { name: 'Save Mission' });
  await waitFor(() => expect(saveButton).toBeEnabled());
  fireEvent.click(saveButton);
  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalled());

  const savedPayload = saveNavigationMission.mock.calls[
    saveNavigationMission.mock.calls.length - 1
  ][1];
  const savedStart = savedPayload.waypoints.find(({ id }) => id === 'Waypoint_1_ee992021');
  const savedNew = savedPayload.waypoints.find(({ id }) => id === 'Waypoint_2_d58de4b1');
  expect(savedStart.local_bt).toBe(startBtPath);
  expect(savedNew.local_bt).toBe(newBtPath);
  expect(savedNew.local_bt).not.toBe(savedStart.local_bt);

  const startUpload = saveNavigationMissionBtFile.mock.calls.find(
    ([, path]) => path === startBtPath,
  );
  const newUpload = saveNavigationMissionBtFile.mock.calls.find(
    ([, path]) => path === newBtPath,
  );
  expect(startUpload?.[2]).toBe(startOnlyXml);
  expect(newUpload?.[2]).toContain('<BehaviorTree ID="MainTree"/>');
  expect(newUpload?.[2]).not.toContain('StartOnly');
}, 15000);

test('clears stale robot pose before a second at-robot waypoint attempt', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  let spotSerial = 0;
  getServiceStatus
    .mockResolvedValueOnce({ is_up: false })
    .mockResolvedValue({ is_up: true });
  createNavigationSpot.mockImplementation((payload) => {
    spotSerial += 1;
    return Promise.resolve({
      id: `spot_${spotSerial}`,
      map_name: payload.map_name,
      label: payload.label,
      pose: payload.pose,
      linked_bt_tree: '',
      metadata: payload.metadata,
    });
  });
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });
  mockTopicDataByName['/amcl_pose'] = amclPoseMessage(1, 1, 0);
  sendInitialPoseEstimate
    .mockImplementationOnce(async () => {
      mockTopicDataByName['/amcl_pose'] = amclPoseMessage(1, 1, 0.1);
      return { ok: true };
    })
    .mockImplementationOnce(async () => {
      mockTopicDataByName['/amcl_pose'] = amclPoseMessage(4.5, 5.25, 0.6);
      return { ok: true };
    });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'At Robot' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  expect(configureDesignLocalizationAmcl).toHaveBeenCalledTimes(1);
  await act(async () => {
    latestMapViewerProps().onMapPose(1, 1, 0.1);
  });
  await waitFor(() => expect(createNavigationSpot).toHaveBeenCalledTimes(1), { timeout: 5000 });
  expect(createNavigationSpot.mock.calls[0][0].pose.x).toBeCloseTo(1);
  expect(createNavigationSpot.mock.calls[0][0].pose.y).toBeCloseTo(1);
  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeEnabled());

  mockTopicDataByName['/amcl_pose'] = amclPoseMessage(1, 1, 0.1);
  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  fireEvent.click(screen.getByRole('button', { name: 'At Robot' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  expect(configureDesignLocalizationAmcl).toHaveBeenCalledTimes(2);
  await act(async () => {
    latestMapViewerProps().onMapPose(4, 5, 0.5);
  });

  await waitFor(() => expect(createNavigationSpot).toHaveBeenCalledTimes(2), { timeout: 6000 });
  const [secondPayload] = createNavigationSpot.mock.calls[1];
  expect(secondPayload.label).toBe('Waypoint 2');
  expect(secondPayload.pose.x).toBeCloseTo(4.5);
  expect(secondPayload.pose.y).toBeCloseTo(5.25);
  expect(secondPayload.pose.yaw).toBeCloseTo(0.6);
  expect(secondPayload.pose.x).not.toBeCloseTo(createNavigationSpot.mock.calls[0][0].pose.x);
  expect(stopNavigation).toHaveBeenCalledTimes(2);
}, 10000);

test('syncs design localization state from navigation status mode', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getServiceStatus.mockResolvedValue({
    is_up: true,
    mode: 'localize',
    pid: 123,
    raw: 'up (pid 123 pgid 123) 7 seconds',
  });
  mockTopicDataByName['/tf'] = {
    transforms: [{
      header: { frame_id: 'map' },
      child_frame_id: 'base_link',
      transform: {
        translation: { x: 2.5, y: -1.25, z: 0 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
      },
    }],
  };
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });

  render(<AutonomyStudioPage />);

  // The map must be loaded in-session (it is not auto-restored on refresh).
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(latestMapViewerProps().showScan).toBe(true));
  expect(latestMapViewerProps().showRobotModel).toBe(true);
  expect(latestMapViewerProps().pose).not.toBeNull();
});

test('creates a waypoint at the current robot pose from the design toolbar', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const robotYaw = 0.5;
  getServiceStatus.mockResolvedValue({ is_up: true });
  mockTopicDataByName['/tf'] = {
    transforms: [{
      header: { frame_id: 'map' },
      child_frame_id: 'base_link',
      transform: {
        translation: { x: 2.5, y: -1.25, z: 0 },
        rotation: { x: 0, y: 0, z: Math.sin(robotYaw / 2), w: Math.cos(robotYaw / 2) },
      },
    }],
  };
  mockTopicDataByName['/amcl_pose'] = amclPoseMessage(8, 8, 0);
  sendInitialPoseEstimate.mockImplementationOnce(async () => {
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(2.5, -1.25, robotYaw);
    return { ok: true };
  });
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(screen.getByRole('button', { name: 'Create Waypoint' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Create Waypoint' }));
  await waitFor(() => expect(screen.getByRole('menu', { name: 'Waypoint creation options' })).toBeInTheDocument());
  await waitFor(() => expect(screen.getByRole('button', { name: 'At Robot' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'At Robot' }));

  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('localize', 'factory'));
  await waitFor(() => expect(configureDesignLocalizationAmcl).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  await act(async () => {
    latestMapViewerProps().onMapPose(2.25, -1, robotYaw);
  });
  await waitFor(() => expect(sendInitialPoseEstimate).toHaveBeenCalledWith({
    x: 2.25,
    y: -1,
    yaw: robotYaw,
    frameId: 'map',
    mapName: 'factory',
  }));
  await waitFor(() => expect(requestNoMotionUpdate).toHaveBeenCalledTimes(3), { timeout: 4000 });
  await waitFor(() => expect(createNavigationSpot).toHaveBeenCalled(), { timeout: 5000 });
  const [payload] = createNavigationSpot.mock.calls[0];
  expect(payload.map_name).toBe('factory');
  expect(payload.label).toBe('Waypoint 1');
  expect(payload.pose.frame_id).toBe('map');
  expect(payload.pose.x).toBeCloseTo(2.5);
  expect(payload.pose.y).toBeCloseTo(-1.25);
  expect(payload.pose.yaw).toBeCloseTo(robotYaw);
  expect(payload.metadata).toEqual({ source: 'mission_canvas', coordinate_space: 'map' });
  await waitFor(() => expect(stopNavigation).toHaveBeenCalled());
});

function mockPersistedRouteMission({ closed = false } = {}) {
  const waypoints = [
    {
      id: 'spot_a',
      label: 'Waypoint A',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      local_bt: 'locals/spot_a/main.xml',
      metadata: {},
    },
    {
      id: 'spot_b',
      label: 'Waypoint B',
      pose: { frame_id: 'map', x: 3, y: 4, yaw: 0.5 },
      local_bt: 'locals/spot_b/main.xml',
      metadata: {},
    },
    {
      id: 'spot_c',
      label: 'Waypoint C',
      pose: { frame_id: 'map', x: 5, y: 6, yaw: 0.75 },
      local_bt: 'locals/spot_c/main.xml',
      metadata: {},
    },
    {
      id: 'spot_d',
      label: 'Waypoint D',
      pose: { frame_id: 'map', x: 7, y: 8, yaw: 1 },
      local_bt: 'locals/spot_d/main.xml',
      metadata: {},
    },
  ];
  const edges = [
    { id: 'route_a_b', source: 'spot_a', target: 'spot_b' },
    { id: 'route_b_c', source: 'spot_b', target: 'spot_c' },
    ...(closed ? [{ id: 'route_c_a', source: 'spot_c', target: 'spot_a' }] : []),
  ];
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'map.pgm', name: 'map.pgm' }],
  });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'map',
    mission_name: 'default',
    global_bt: 'global.xml',
    waypoints,
    metadata: {
      mission_flow: {
        nodes: waypoints.map((waypoint, index) => ({
          id: waypoint.id,
          position: { x: 80 + index * 220, y: 72 },
        })),
        edges,
      },
    },
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: '<root BTCPP_format="4" main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"/></root>',
    exists: true,
  }));
}

async function loadPersistedRouteDesign() {
  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const loadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  await within(loadDialog).findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(within(loadDialog).getByRole('button', { name: 'Load' }));
}

test('only appends an unused waypoint while route edit mode is active', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  mockPersistedRouteMission();

  await loadPersistedRouteDesign();
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
  ]));
  expect(latestMapViewerProps().missionRouteClosed).toBe(false);
  expect(latestMapViewerProps().missionRouteMode).toBe(false);
  expect(latestMapViewerProps().spots).toHaveLength(4);
  expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Add Waypoint D to route' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /^Insert waypoint/ })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /^Insert .* here$/ })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Clear Route' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Move Waypoint A down' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Remove Waypoint A from route' })).not.toBeInTheDocument();

  // A callback captured by an older MapViewer render must not bypass the edit
  // toggle after route editing has been turned off.
  act(() => {
    latestMapViewerProps().onMissionRouteSpotClick('spot_d');
  });
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
  ]);
  expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();

  fireEvent.click(screen.getByRole('button', { name: 'Edit On Map' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteMode).toBe(true));
  expect(screen.queryByRole('button', { name: 'Add Waypoint D to route' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /^Insert waypoint/ })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Clear Route' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Move Waypoint A down' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Remove Waypoint A from route' })).toBeInTheDocument();

  // One map click appends an unused waypoint. There is no insertion-slot UI.
  act(() => {
    latestMapViewerProps().onMissionRouteSpotClick('spot_d');
  });
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
    { id: 'spot_d', order: 4 },
  ]));
  expect(latestMapViewerProps().missionRouteClosed).toBe(false);
  expect(latestMapViewerProps().missionRouteMode).toBe(true);
  expect(screen.getByRole('button', { name: 'Undo' })).toBeEnabled();

  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
  ]));
  expect(screen.getByRole('button', { name: 'Redo' })).toBeEnabled();

  fireEvent.click(screen.getByRole('button', { name: 'Edit On Map' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteMode).toBe(false));
  expect(screen.queryByRole('button', { name: 'Clear Route' })).not.toBeInTheDocument();
  act(() => {
    latestMapViewerProps().onMissionRouteSpotClick('spot_d');
  });
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
  ]);
});

test('appends an unused waypoint to a closed route without opening its loop', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  mockPersistedRouteMission({ closed: true });

  await loadPersistedRouteDesign();
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
  ]));
  expect(latestMapViewerProps().missionRouteClosed).toBe(true);
  expect(latestMapViewerProps().missionRouteMode).toBe(false);
  expect(screen.getByText('Return to Waypoint A')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Open loop' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Clear Route' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /^Insert waypoint/ })).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Edit On Map' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteMode).toBe(true));
  expect(screen.queryByRole('button', { name: 'Add Waypoint D to route' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /^Insert waypoint/ })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open loop' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Clear Route' })).toBeInTheDocument();

  // D is appended before the closing edge, preserving A -> B -> C -> D -> A.
  act(() => {
    latestMapViewerProps().onMissionRouteSpotClick('spot_d');
  });
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
    { id: 'spot_d', order: 4 },
  ]));
  expect(latestMapViewerProps().missionRouteClosed).toBe(true);
  expect(latestMapViewerProps().missionRouteMode).toBe(true);
  expect(screen.getByText('Return to Waypoint A')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
  ]));
  expect(latestMapViewerProps().missionRouteClosed).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: 'Redo' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
    { id: 'spot_d', order: 4 },
  ]));
  expect(latestMapViewerProps().missionRouteClosed).toBe(true);
});

test('locks route edits while deleting a legacy waypoint and stitches the captured route', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'map.pgm', name: 'map.pgm' }],
  });
  getNavigationSpots.mockResolvedValue({
    map_name: 'map',
    spots: [
      {
        id: 'spot_a',
        map_name: 'map',
        label: 'Waypoint A',
        pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
        linked_bt_tree: 'waypoint_a.xml',
        metadata: {},
      },
      {
        id: 'spot_b',
        map_name: 'map',
        label: 'Waypoint B',
        pose: { frame_id: 'map', x: 3, y: 4, yaw: 0.5 },
        linked_bt_tree: 'waypoint_b.xml',
        metadata: {},
      },
      {
        id: 'spot_c',
        map_name: 'map',
        label: 'Waypoint C',
        pose: { frame_id: 'map', x: 5, y: 6, yaw: 0.75 },
        linked_bt_tree: 'waypoint_c.xml',
        metadata: {},
      },
      {
        id: 'spot_d',
        map_name: 'map',
        label: 'Waypoint D',
        pose: { frame_id: 'map', x: 7, y: 8, yaw: 1 },
        linked_bt_tree: 'waypoint_d.xml',
        metadata: {},
      },
    ],
  });
  let resolveDelete;
  const pendingDelete = new Promise((resolve) => {
    resolveDelete = resolve;
  });
  deleteNavigationSpot.mockReturnValueOnce(pendingDelete);

  await loadPersistedRouteDesign();
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(4));
  fireEvent.click(screen.getByRole('button', { name: 'Edit On Map' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteMode).toBe(true));
  act(() => { latestMapViewerProps().onMissionRouteSpotClick('spot_a'); });
  act(() => { latestMapViewerProps().onMissionRouteSpotClick('spot_b'); });
  act(() => { latestMapViewerProps().onMissionRouteSpotClick('spot_c'); });
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
  ]));

  fireEvent.click(screen.getByRole('button', { name: 'Delete Waypoint Waypoint B' }));
  await waitFor(() => expect(deleteNavigationSpot).toHaveBeenCalledWith('spot_b', 'map'));

  // The backend deletion is still pending. Every route mutation affordance is
  // locked, and even a stale map callback cannot append D to the captured route.
  expect(screen.getByRole('button', { name: 'Edit On Map' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: /^Insert waypoint/ })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Remove Waypoint A from route' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Move Waypoint A down' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Clear Route' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Delete Waypoint Waypoint C' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();
  act(() => {
    latestMapViewerProps().onMissionRouteSpotClick('spot_d');
  });
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
  ]);

  await act(async () => {
    resolveDelete({ ok: true });
    await pendingDelete;
  });
  await waitFor(() => expect(latestMapViewerProps().spots.map((spot) => spot.id)).toEqual([
    'spot_a', 'spot_c', 'spot_d',
  ]));
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_c', order: 2 },
  ]);
  expect(latestMapViewerProps().missionRouteClosed).toBe(false);
  expect(screen.getByRole('button', { name: 'Edit On Map' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Remove Waypoint A from route' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Clear Route' })).toBeEnabled();
});

test('edits the mission route directly on the map', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'map.pgm', name: 'map.pgm' }],
  });
  getNavigationSpots.mockImplementation((mapName) => Promise.resolve({
    map_name: mapName,
    spots: mapName === 'map' ? [
      {
        id: 'spot_a',
        map_name: 'map',
        label: 'Waypoint A',
        pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
        linked_bt_tree: 'waypoint_a.xml',
        metadata: {},
      },
      {
        id: 'spot_b',
        map_name: 'map',
        label: 'Waypoint B',
        pose: { frame_id: 'map', x: 3, y: 4, yaw: 0.5 },
        linked_bt_tree: 'waypoint_b.xml',
        metadata: {},
      },
      {
        id: 'spot_c',
        map_name: 'map',
        label: 'Waypoint C',
        pose: { frame_id: 'map', x: 5, y: 6, yaw: 0.75 },
        linked_bt_tree: 'waypoint_c.xml',
        metadata: {},
      },
      {
        id: 'spot_d',
        map_name: 'map',
        label: 'Waypoint D',
        pose: { frame_id: 'map', x: 7, y: 8, yaw: 1 },
        linked_bt_tree: 'waypoint_d.xml',
        metadata: {},
      },
    ] : [],
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().map).not.toBeNull());
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(4));

  expect(screen.getAllByText('Waypoint A').length).toBeGreaterThan(0);
  expect(screen.getAllByText('Waypoint B').length).toBeGreaterThan(0);
  expect(screen.getAllByText('Waypoint C').length).toBeGreaterThan(0);
  expect(screen.getAllByText('Waypoint D').length).toBeGreaterThan(0);
  expect(latestMapViewerProps().missionRouteOrder).toEqual([]);
  expect(latestMapViewerProps().missionRouteClosed).toBe(false);

  fireEvent.click(screen.getByRole('button', { name: 'Edit On Map' }));

  await waitFor(() => expect(latestMapViewerProps().missionRouteMode).toBe(true));
  expect(latestMapViewerProps().onSpotPoseChange).toBeUndefined();
  expect(screen.queryByRole('button', { name: 'Clear Route' })).not.toBeInTheDocument();

  act(() => {
    latestMapViewerProps().onMissionRouteSpotClick('spot_b');
  });
  await waitFor(() => expect(latestMapViewerProps().selectedMissionRouteSourceId).toBe('spot_b'));

  act(() => {
    latestMapViewerProps().onMissionRouteSpotClick('spot_a');
  });
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_b', order: 1 },
    { id: 'spot_a', order: 2 },
  ]);
  expect(latestMapViewerProps().missionRouteClosed).toBe(false);

  act(() => {
    latestMapViewerProps().onMissionRouteSpotClick('spot_b');
  });
  await waitFor(() => expect(latestMapViewerProps().selectedMissionRouteSourceId).toBe(''));
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_b', order: 1 },
    { id: 'spot_a', order: 2 },
  ]);
  expect(latestMapViewerProps().missionRouteClosed).toBe(true);
  expect(screen.getByText('closed loop')).toBeInTheDocument();
  expect(screen.getByText('Return to Waypoint B')).toBeInTheDocument();

  // Reordering a closed route must rotate the loop without dropping its
  // last -> first edge.
  fireEvent.click(screen.getByRole('button', { name: 'Move Waypoint B down' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
  ]));
  expect(latestMapViewerProps().missionRouteClosed).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: 'Move Waypoint B up' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_b', order: 1 },
    { id: 'spot_a', order: 2 },
  ]));
  expect(latestMapViewerProps().missionRouteClosed).toBe(true);

  fireEvent.click(screen.getByRole('button', { name: 'Save Mission' }));

  await waitFor(() => expect(saveNavigationMission).toHaveBeenCalledWith(
    'map',
    expect.objectContaining({
      waypoints: expect.arrayContaining([
        expect.objectContaining({ id: 'spot_a' }),
        expect.objectContaining({ id: 'spot_b' }),
        expect.objectContaining({ id: 'spot_c' }),
        expect.objectContaining({ id: 'spot_d' }),
      ]),
      metadata: expect.objectContaining({
        mission_flow: expect.objectContaining({
          edges: [
            expect.objectContaining({ source: 'spot_b', target: 'spot_a' }),
            expect.objectContaining({ source: 'spot_a', target: 'spot_b' }),
          ],
        }),
      }),
    }),
    '',
  ));
  await waitFor(() => {
    const globalSave = saveNavigationMissionBtFile.mock.calls.find(([mapName, path]) => (
      mapName === 'map' && path === 'global.xml'
    ));
    expect(globalSave).toBeTruthy();
    const globalXml = globalSave[2];
    expect(globalXml.match(/<MissionStep/g)).toHaveLength(3);
    expect(globalXml).toMatch(/waypoint_id="spot_b"[\s\S]*waypoint_id="spot_a"[\s\S]*waypoint_id="spot_b"/);
  });
  expect(screen.queryByRole('button', { name: 'Create BT' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Edit Task' })).not.toBeInTheDocument();

  // The route row only removes from the route; actual spot deletion lives in
  // the Waypoints panel alone.
  expect(screen.getByRole('button', { name: 'Remove Waypoint A from route' })).toBeInTheDocument();
  expect(screen.getAllByRole('button', { name: 'Delete Waypoint Waypoint A' })).toHaveLength(1);
  fireEvent.click(screen.getByRole('button', { name: 'Delete Waypoint Waypoint A' }));

  await waitFor(() => expect(deleteNavigationSpot).toHaveBeenCalledWith('spot_a', 'map'));
  await waitFor(() => expect(latestMapViewerProps().spots.map((spot) => spot.id)).toEqual([
    'spot_b',
    'spot_c',
    'spot_d',
  ]));
  expect(latestMapViewerProps().missionRouteOrder).toEqual([]);
  expect(latestMapViewerProps().missionRouteClosed).toBe(false);
});

test('opens a closed loop and clears the route without deleting waypoints', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'map.pgm', name: 'map.pgm' }],
  });
  getNavigationSpots.mockImplementation((mapName) => Promise.resolve({
    map_name: mapName,
    spots: mapName === 'map' ? [
      { id: 'spot_a', map_name: 'map', label: 'Waypoint A', pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 }, linked_bt_tree: 'waypoint_a.xml', metadata: {} },
      { id: 'spot_b', map_name: 'map', label: 'Waypoint B', pose: { frame_id: 'map', x: 3, y: 4, yaw: 0.5 }, linked_bt_tree: 'waypoint_b.xml', metadata: {} },
      { id: 'spot_c', map_name: 'map', label: 'Waypoint C', pose: { frame_id: 'map', x: 5, y: 6, yaw: 0.75 }, linked_bt_tree: 'waypoint_c.xml', metadata: {} },
    ] : [],
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Design mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(3));

  fireEvent.click(screen.getByRole('button', { name: 'Edit On Map' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteMode).toBe(true));

  // Build A -> B and close the loop back to A.
  act(() => { latestMapViewerProps().onMissionRouteSpotClick('spot_a'); });
  act(() => { latestMapViewerProps().onMissionRouteSpotClick('spot_b'); });
  act(() => { latestMapViewerProps().onMissionRouteSpotClick('spot_a'); });
  await waitFor(() => expect(latestMapViewerProps().missionRouteClosed).toBe(true));

  // The closure row's remove button re-opens the loop...
  fireEvent.click(screen.getByRole('button', { name: 'Open loop' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteClosed).toBe(false));
  expect(screen.queryByText(/Return to /)).not.toBeInTheDocument();
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
  ]);

  // ...and the open route is editable again: extend it to Waypoint C.
  act(() => { latestMapViewerProps().onMissionRouteSpotClick('spot_b'); });
  act(() => { latestMapViewerProps().onMissionRouteSpotClick('spot_c'); });
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_b', order: 2 },
    { id: 'spot_c', order: 3 },
  ]));

  // The route row's × takes the waypoint out of the route and stitches its
  // neighbors — without deleting the spot.
  fireEvent.click(screen.getByRole('button', { name: 'Remove Waypoint B from route' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'spot_a', order: 1 },
    { id: 'spot_c', order: 2 },
  ]));
  expect(deleteNavigationSpot).not.toHaveBeenCalled();
  expect(latestMapViewerProps().spots).toHaveLength(3);

  // Clear Route discards every edge but keeps the waypoints.
  fireEvent.click(screen.getByRole('button', { name: 'Clear Route' }));
  await waitFor(() => expect(latestMapViewerProps().missionRouteOrder).toEqual([]));
  expect(latestMapViewerProps().spots).toHaveLength(3);
  expect(screen.queryByRole('button', { name: 'Clear Route' })).not.toBeInTheDocument();
});

test('starts mapping mode from Mission Canvas', async () => {
  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('button', { name: 'Start Mapping' }));

  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('map', 'map'));
});

test('locks mapping controls while mapping is running', async () => {
  getServiceStatus
    .mockResolvedValueOnce({ is_up: true, mode: 'map' })
    .mockResolvedValueOnce({ is_up: false });

  render(<AutonomyStudioPage />);

  await waitFor(() => expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled());
  expect(screen.getByRole('button', { name: 'Save Map' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Start Mapping' })).toBeDisabled();
  // Editing a saved PGM while SLAM may rewrite it would clobber one side.
  expect(screen.getByRole('tab', { name: 'Map Edit' })).toBeDisabled();

  fireEvent.click(screen.getByRole('button', { name: 'Stop' }));

  await waitFor(() => expect(stopNavigation).toHaveBeenCalled());
  await waitFor(() => expect(screen.getByRole('button', { name: 'Start Mapping' })).toBeEnabled());
  expect(screen.getByRole('button', { name: 'Save Map' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Stop' })).toBeDisabled();
  expect(screen.getByRole('tab', { name: 'Map Edit' })).toBeEnabled();
});

test('publishes keyboard teleop commands without mapping runtime', async () => {
  render(<AutonomyStudioPage />);

  const teleop = screen.getByRole('group', { name: 'Mobile Teleop' });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Activate' })).toBeEnabled());
  expect(teleop).toHaveAttribute('tabindex', '-1');

  fireEvent.keyDown(window, { key: 'w' });
  expect(mockPublishRosTopic).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Activate' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Deactivate' })).toBeInTheDocument());
  expect(teleop).toHaveAttribute('tabindex', '0');

  fireEvent.keyDown(window, { key: 'w' });

  await waitFor(() => expect(mockPublishRosTopic).toHaveBeenCalledWith(
    '/cmd_vel',
    'geometry_msgs/msg/Twist',
    {
      linear: { x: 0.4, y: 0, z: 0 },
      angular: { x: 0, y: 0, z: 0 },
    },
  ));

  fireEvent.keyUp(window, { key: 'w' });

  await waitFor(() => expect(mockPublishRosTopic).toHaveBeenCalledWith(
    '/cmd_vel',
    'geometry_msgs/msg/Twist',
    {
      linear: { x: 0, y: 0, z: 0 },
      angular: { x: 0, y: 0, z: 0 },
    },
  ));

  fireEvent.click(screen.getByRole('button', { name: 'Deactivate' }));
  expect(screen.getByRole('button', { name: 'Activate' })).toBeInTheDocument();
});

test('asks for a map name before saving from Mission Canvas', async () => {
  getServiceStatus.mockResolvedValueOnce({ is_up: true, mode: 'map' });

  render(<AutonomyStudioPage />);

  await waitFor(() => expect(screen.getByRole('button', { name: 'Save Map' })).toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Save Map' }));
  fireEvent.change(screen.getByLabelText('Save map name'), {
    target: { value: 'factory' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));

  await waitFor(() => expect(saveNavigationMap).toHaveBeenCalledWith('factory'));
  expect(getPgmImage).not.toHaveBeenCalled();
  expect(screen.queryByDisplayValue('factory.pgm')).not.toBeInTheDocument();
  expect(screen.getByText('Live mapping')).toBeInTheDocument();
});

test('deletes a saved map from the Mapping HUD behind a warning confirm popup', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions.mockResolvedValue({ map_name: 'factory', missions: ['default', 'patrol'] });

  render(<AutonomyStudioPage />);

  // The trash icon opens a saved-map list popover; picking a map opens the
  // warning popup (and closes the list).
  fireEvent.click(screen.getByRole('button', { name: 'Delete Map' }));
  expect(screen.getByRole('menu', { name: 'Saved maps' })).toBeInTheDocument();
  fireEvent.click(await screen.findByRole('button', { name: 'Delete map factory.pgm' }));
  expect(screen.queryByRole('menu', { name: 'Saved maps' })).not.toBeInTheDocument();

  // The popup spells out the cascade before asking.
  expect(await screen.findByText('Delete this map?')).toBeInTheDocument();
  await waitFor(() => expect(
    screen.getByText('This map, its areas, and 2 missions will be deleted permanently.'),
  ).toBeInTheDocument());
  expect(deletePgmMap).not.toHaveBeenCalled();

  // No closes without deleting.
  fireEvent.click(screen.getByRole('button', { name: 'No' }));
  expect(screen.queryByText('Delete this map?')).not.toBeInTheDocument();
  expect(deletePgmMap).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Delete Map' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Delete map factory.pgm' }));
  fireEvent.click(screen.getByRole('button', { name: 'Yes' }));

  await waitFor(() => expect(deletePgmMap).toHaveBeenCalledWith('factory.pgm'));
  await waitFor(() => expect(screen.queryByText('Delete this map?')).not.toBeInTheDocument());

  // The list refreshes: reopening the popover shows the empty state.
  fireEvent.click(screen.getByRole('button', { name: 'Delete Map' }));
  expect(await screen.findByText('No saved maps yet.')).toBeInTheDocument();
  expect(screen.queryByText('factory.pgm')).not.toBeInTheDocument();
});

test('waits for an explicit saved-map selection in the mapping fix editor', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Map Edit' }));

  // The header Load Map stays disabled until the PGM listing lands, so the
  // dialog can never open on an empty pending path.
  expect(screen.getByRole('button', { name: 'Load Map' })).toBeDisabled();

  await waitFor(() => expect(getPgmFiles).toHaveBeenCalled());
  // Entering the stage loads nothing: the picker lives in the header dialog.
  expect(screen.queryByRole('combobox', { name: 'PGM map' })).not.toBeInTheDocument();
  // The HUD tools stay dimmed until a map is loaded, like the other stages.
  expect(screen.getByRole('button', { name: 'View' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Map Edit' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Add Label' })).toBeDisabled();
  expect(getPgmImage).not.toHaveBeenCalled();
  expect(latestMapViewerProps().map).toBeNull();
  expect(latestMapViewerProps().waitingLabel).toBe('Load a map');

  await waitFor(() => expect(screen.getByRole('button', { name: 'Load Map' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'PGM map' });
  // The dialog preselects the first saved map, but nothing loads before Load.
  expect(mapSelect).toHaveValue('factory.pgm');
  expect(getPgmImage).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('factory.pgm'));
  await waitFor(() => expect(screen.queryByRole('combobox', { name: 'PGM map' })).not.toBeInTheDocument());
  // The panel names the loaded map; no session/topics side panels.
  expect(screen.queryByText('Mapping Session')).not.toBeInTheDocument();
  expect(screen.queryByText('Topics')).not.toBeInTheDocument();
  expect(screen.getAllByText('factory.pgm').length).toBeGreaterThan(0);
  expect(latestMapViewerProps().showScan).toBe(false);
  expect(latestMapViewerProps().showMap).toBe(true);
  expect(latestMapViewerProps().waitingLabel).toBe('Load a map');
  // The editor shows the raw grid; floor-plan refinement is viewer-only.
  expect(latestMapViewerProps().mapRefined).toBe(false);
});

test('cancels the Map Edit load dialog without loading anything', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Map Edit' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Load Map' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'PGM map' });

  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  await waitFor(() => expect(screen.queryByRole('combobox', { name: 'PGM map' })).not.toBeInTheDocument());
  expect(getPgmImage).not.toHaveBeenCalled();
  expect(screen.queryByText('factory.pgm')).not.toBeInTheDocument();
});

test('switches between the Mapping and Map Edit stages via the rail tabs', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });

  render(<AutonomyStudioPage />);

  // Mapping is the default stage: SLAM actions, teleop, and layers visible.
  expect(screen.getByRole('tab', { name: 'Mapping' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByRole('tab', { name: 'Map Edit' })).toHaveAttribute('aria-selected', 'false');
  expect(screen.getByRole('button', { name: 'Start Mapping' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Save Map' })).toBeInTheDocument();
  expect(screen.getByRole('group', { name: 'Mobile Teleop' })).toBeInTheDocument();
  expect(screen.getByRole('switch', { name: 'Lidar' })).toBeInTheDocument();

  await openMappingEditorAndSelect();

  await waitFor(() => expect(getPgmFiles).toHaveBeenCalled());
  expect(screen.getByRole('tab', { name: 'Map Edit' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByRole('tab', { name: 'Mapping' })).toHaveAttribute('aria-selected', 'false');
  // Mapping-stage chrome is gone: SLAM actions, teleop, and the layers popover.
  expect(screen.queryByRole('button', { name: 'Start Mapping' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Stop' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Save Map' })).not.toBeInTheDocument();
  expect(screen.queryByRole('group', { name: 'Mobile Teleop' })).not.toBeInTheDocument();
  expect(screen.queryByRole('switch', { name: 'Lidar' })).not.toBeInTheDocument();
  // The whole side column is gone — the editor gets the full width.
  expect(screen.queryByText('Mapping Session')).not.toBeInTheDocument();
  expect(screen.queryByText('Topics')).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: 'Mapping' }));

  expect(screen.getByRole('tab', { name: 'Mapping' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByRole('tab', { name: 'Map Edit' })).toHaveAttribute('aria-selected', 'false');
  expect(screen.getByRole('button', { name: 'Start Mapping' })).toBeEnabled();
  expect(screen.getByText('Live mapping')).toBeInTheDocument();
});

test('keeps edit mode when the mapping runtime comes up', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: 'AA==',
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();
  await waitFor(() => expect(getPgmFiles).toHaveBeenCalled());

  // The runtime comes up behind the editor (e.g. started elsewhere): the
  // editor keeps the view, and Record is one click away from the Stop button.
  getServiceStatus.mockResolvedValue({ is_up: true, mode: 'map' });
  fireEvent(document, new Event('visibilitychange'));

  await waitFor(() => expect(screen.getByText('Status: running')).toBeInTheDocument());
  expect(screen.getByRole('tab', { name: 'Map Edit' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.queryByRole('button', { name: 'Stop' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Save Map' })).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('tab', { name: 'Mapping' }));

  await waitFor(() => expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled());
  expect(screen.getByRole('button', { name: 'Save Map' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Start Mapping' })).toBeDisabled();
});

test('edits and saves loaded map pixels from the fix editor', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: '/g==',
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();

  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('factory.pgm'));
  // A clean load arms nothing: Undo/Redo/Save all disabled, no unsaved badge.
  expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
  expect(screen.queryByText('· unsaved')).not.toBeInTheDocument();
  // Pixel tools + brush sizes live in the Map Edit popover.
  fireEvent.click(screen.getByRole('button', { name: 'Map Edit' }));
  fireEvent.click(screen.getByRole('button', { name: 'Brush size XL' }));
  expect(screen.getByRole('button', { name: 'Brush size XL' })).toHaveAttribute('aria-pressed', 'true');

  fireEvent.click(screen.getByRole('button', { name: 'Add Obstacle' }));
  await waitFor(() => expect(latestMapViewerProps().editorActive).toBe(true));

  await act(async () => {
    latestMapViewerProps().onEditorMapPoint(0, 0);
  });

  expect(screen.getByText('· unsaved')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Undo' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled();

  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Redo' })).toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Redo' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled());

  // The HUD tooltips advertise Ctrl+Z / Ctrl+Shift+Z — they must work here too.
  fireEvent.keyDown(document, { key: 'z', ctrlKey: true });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Redo' })).toBeEnabled());
  fireEvent.keyDown(document, { key: 'z', ctrlKey: true, shiftKey: true });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled());

  await waitFor(() => expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));

  await waitFor(() => expect(savePgmImage).toHaveBeenCalledWith(
    'factory.pgm',
    1,
    1,
    255,
    'AA==',
  ));
  // A successful save clears the dirty state: badge gone, Save re-disabled.
  await waitFor(() => expect(screen.queryByText('· unsaved')).not.toBeInTheDocument());
  expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
});

test('keeps unsaved Map Edit pixels in place by blocking return to the chooser', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: '/g==',
  });

  render(<AutonomyStudioPage />);
  await openMappingEditorAndSelect();
  fireEvent.click(screen.getByRole('button', { name: 'Map Edit' }));
  fireEvent.click(screen.getByRole('button', { name: 'Add Obstacle' }));
  await act(async () => {
    latestMapViewerProps().onEditorMapPoint(0, 0);
  });

  expect(screen.queryByRole('tab', { name: 'Action Canvas' })).not.toBeInTheDocument();
  const backToChooserButton = screen.getByRole('button', { name: 'Back to workspace chooser' });
  expect(backToChooserButton).toHaveAttribute('title', expect.stringContaining('Save the current map edits'));
  fireEvent.click(backToChooserButton);
  expect(screen.getByText('· unsaved')).toBeInTheDocument();
  expect(screen.queryByText('Choose a workspace')).not.toBeInTheDocument();
  expect(screen.queryByTestId('action-canvas-workspace')).not.toBeInTheDocument();
});

test('marks unknown map pixels from the fix editor', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: '/g==',
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();
  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('factory.pgm'));

  fireEvent.click(screen.getByRole('button', { name: 'Map Edit' }));
  fireEvent.click(screen.getByRole('button', { name: 'Mark Unknown' }));
  await waitFor(() => expect(latestMapViewerProps().editorActive).toBe(true));
  await act(async () => {
    latestMapViewerProps().onEditorMapPoint(0, 0);
  });

  await waitFor(() => expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));

  // 0xCD == 205, the map_server "unknown" gray.
  await waitFor(() => expect(savePgmImage).toHaveBeenCalledWith('factory.pgm', 1, 1, 255, 'zQ=='));
});

test('paints continuous map pixel segments while dragging', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 3,
    height: 1,
    maxval: 255,
    pixels_base64: '/v7+',
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();

  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Map Edit' }));
  fireEvent.click(screen.getByRole('button', { name: 'Add Obstacle' }));
  await waitFor(() => expect(latestMapViewerProps().editorActive).toBe(true));

  await act(async () => {
    latestMapViewerProps().onEditorMapPoint(0.5, 0.5, 'start');
    latestMapViewerProps().onEditorMapPoint(2.5, 0.5, 'move');
    latestMapViewerProps().onEditorMapPoint(0, 0, 'end');
  });

  await waitFor(() => expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));

  await waitFor(() => expect(savePgmImage).toHaveBeenCalledWith(
    'factory.pgm',
    3,
    1,
    255,
    'AAAA',
  ));
});

test('marks free-space areas with automatic color and undo/redo support', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  jest.spyOn(Math, 'random').mockReturnValue(0);
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: '/g==',
  });
  getMapAnnotations.mockResolvedValue({
    path: 'factory.pgm',
    annotations: [],
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();

  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('factory.pgm'));
  await waitFor(() => expect(getMapAnnotations).toHaveBeenCalledWith('factory.pgm'));

  fireEvent.click(screen.getByRole('button', { name: 'Add Label' }));
  fireEvent.click(screen.getByRole('button', { name: 'Area' }));
  // Area drags a rectangle — no brush row in this mode.
  expect(screen.queryByRole('button', { name: 'Brush size XL' })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Area name'), {
    target: { value: 'Dock' },
  });
  await waitFor(() => expect(latestMapViewerProps().editorActive).toBe(true));
  // The Area tool now creates areas from a rectangle drag selection.
  await waitFor(() => expect(latestMapViewerProps().editorAreaSelection).toBe(true));

  await act(async () => {
    latestMapViewerProps().onEditorMapArea(0, 0, 0, 0);
  });

  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({
      label: 'Dock',
      color: '#CBB9A4',
      pose: expect.objectContaining({ x: 0.5, y: 0.5 }),
      region: expect.objectContaining({
        seed_cell: { x: 0, y: 0 },
        bounds: { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
        cell_count: 1,
      }),
    }),
  ]));
  expect(latestMapViewerProps().editorPaintOnDrag).toBe(true);
  // New areas are auto-selected in the chip list.
  expect(screen.getByRole('button', { name: 'Dock' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByText('· unsaved')).toBeInTheDocument();
  expect(saveMapAnnotations).not.toHaveBeenCalled();

  const areaSaveCalls = saveMapAnnotations.mock.calls.length;
  await act(async () => {
    latestMapViewerProps().onEditorMapArea(0, 0, 0, 0);
  });
  expect(saveMapAnnotations).toHaveBeenCalledTimes(areaSaveCalls);

  await waitFor(() => expect(screen.getByRole('button', { name: 'Undo' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([]));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Redo' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Redo' }));
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({ label: 'Dock', color: '#CBB9A4' }),
  ]));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));
  await waitFor(() => expect(saveMapAnnotations).toHaveBeenCalledWith(
    'factory.pgm',
    [expect.objectContaining({
      label: 'Dock',
      color: '#CBB9A4',
      region: expect.objectContaining({ cell_count: 1 }),
    })],
  ));
  expect(savePgmImage).not.toHaveBeenCalled();
});

test('auto-numbers and auto-selects areas created by rectangle drag', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  // 3x1 free map: three white pixels.
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 3,
    height: 1,
    maxval: 255,
    pixels_base64: '/v7+',
  });
  getMapAnnotations.mockResolvedValue({ path: 'factory.pgm', annotations: [] });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();
  await waitFor(() => expect(getMapAnnotations).toHaveBeenCalledWith('factory.pgm'));

  fireEvent.click(screen.getByRole('button', { name: 'Add Label' }));
  fireEvent.click(screen.getByRole('button', { name: 'Area' }));
  expect(screen.getByLabelText('Area name')).toHaveAttribute('placeholder', 'Area 1');
  await waitFor(() => expect(latestMapViewerProps().editorAreaSelection).toBe(true));

  await act(async () => {
    latestMapViewerProps().onEditorMapArea(0.5, 0.5, 0.5, 0.5);
  });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Area 1' })).toHaveAttribute('aria-pressed', 'true'));
  expect(screen.getByLabelText('Area name')).toHaveAttribute('placeholder', 'Area 2');

  await act(async () => {
    latestMapViewerProps().onEditorMapArea(1.5, 0.5, 2.5, 0.5);
  });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Area 2' })).toHaveAttribute('aria-pressed', 'true'));
  expect(screen.getByRole('button', { name: 'Area 1' })).toHaveAttribute('aria-pressed', 'false');
  expect(latestMapViewerProps().mapAnnotations).toHaveLength(2);
});

test('keeps the saved free-space Area footprint aligned with the full drag selection', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 3,
    height: 2,
    maxval: 255,
    pixels_base64: '/v7+/v7+',
  });
  getMapAnnotations.mockResolvedValue({ path: 'factory.pgm', annotations: [] });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();
  await waitFor(() => expect(getMapAnnotations).toHaveBeenCalledWith('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Add Label' }));
  fireEvent.click(screen.getByRole('button', { name: 'Area' }));

  await act(async () => {
    latestMapViewerProps().onEditorMapArea(0.05, 0.05, 2.95, 1.95);
  });

  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({
      pose: expect.objectContaining({ x: 1.5, y: 1 }),
      region: expect.objectContaining({
        bounds: { x_min: 0, y_min: 0, x_max: 2, y_max: 1 },
        cell_count: 6,
        width: 3,
        height: 2,
      }),
    }),
  ]));
});

test('removes an area from the chip list', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: '/g==',
  });
  getMapAnnotations.mockResolvedValue({
    path: 'factory.pgm',
    annotations: [{
      id: 'area_dock',
      label: 'Dock',
      color: '#3B241F',
      pose: { frame_id: 'map', x: 0.5, y: 0.5, yaw: 0 },
      region: {
        seed_cell: { x: 0, y: 0 },
        bounds: { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
        cell_count: 1,
        width: 1,
        height: 1,
      },
    }],
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();

  await waitFor(() => expect(getMapAnnotations).toHaveBeenCalledWith('factory.pgm'));
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({ label: 'Dock', color: '#3B241F' }),
  ]));

  fireEvent.click(screen.getByRole('button', { name: 'Add Label' }));
  fireEvent.click(screen.getByRole('button', { name: 'Area' }));
  // Whole-area delete now lives in the popover's area list (two-click confirm).
  fireEvent.click(screen.getByRole('button', { name: 'Delete area Dock' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm delete area Dock' }));

  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([]));
  expect(screen.getByText('· unsaved')).toBeInTheDocument();
  expect(saveMapAnnotations).not.toHaveBeenCalled();

  await waitFor(() => expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));

  await waitFor(() => expect(saveMapAnnotations).toHaveBeenCalledWith('factory.pgm', []));
  expect(savePgmImage).not.toHaveBeenCalled();
});

test('caps the Area list at three rows and isolates scrolled delete actions', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 4,
    height: 1,
    maxval: 255,
    pixels_base64: '/v7+/g==',
  });
  getMapAnnotations.mockResolvedValue({
    path: 'factory.pgm',
    annotations: [0, 1, 2, 3].map((x) => ({
      id: `area_${x + 1}`,
      label: `Area ${x + 1}`,
      color: '#3B241F',
      pose: { frame_id: 'map', x: x + 0.5, y: 0.5, yaw: 0 },
      region: {
        seed_cell: { x, y: 0 },
        bounds: { x_min: x, y_min: 0, x_max: x, y_max: 0 },
        cells: [{ x, y: 0 }],
        cell_count: 1,
        width: 4,
        height: 1,
      },
    })),
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toHaveLength(4));
  fireEvent.click(screen.getByRole('button', { name: 'Add Label' }));
  fireEvent.click(screen.getByRole('button', { name: 'Area' }));

  const areas = screen.getByRole('group', { name: 'Map areas' });

  const documentClick = jest.fn();
  document.addEventListener('click', documentClick);
  fireEvent.click(screen.getByRole('button', { name: 'Delete area Area 4' }));
  expect(documentClick).not.toHaveBeenCalled();
  expect(screen.getByRole('button', { name: 'Area 4' })).toBeInTheDocument();
  expect(screen.getByRole('menu', { name: 'Map labeling tools' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Confirm delete area Area 4' }));
  document.removeEventListener('click', documentClick);

  await waitFor(() => expect(latestMapViewerProps().mapAnnotations.map(({ label }) => label)).toEqual([
    'Area 1', 'Area 2', 'Area 3',
  ]));
  expect(screen.queryByRole('button', { name: 'Area 4' })).not.toBeInTheDocument();
  expect(screen.getByRole('menu', { name: 'Map labeling tools' })).toBeInTheDocument();
});

test('renames a map area from the chip list', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 1,
    height: 1,
    maxval: 255,
    pixels_base64: '/g==',
  });
  getMapAnnotations.mockResolvedValue({
    path: 'factory.pgm',
    annotations: [{
      id: 'area_dock',
      label: 'Dock',
      color: '#3B241F',
      pose: { frame_id: 'map', x: 0.5, y: 0.5, yaw: 0 },
      region: {
        seed_cell: { x: 0, y: 0 },
        bounds: { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
        cell_count: 1,
        width: 1,
        height: 1,
      },
    }],
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();

  await waitFor(() => expect(getMapAnnotations).toHaveBeenCalledWith('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Add Label' }));
  fireEvent.click(screen.getByRole('button', { name: 'Area' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Dock' })).toBeInTheDocument());

  fireEvent.click(screen.getByRole('button', { name: 'Dock' }));
  expect(screen.getByRole('button', { name: 'Dock' })).toHaveAttribute('aria-pressed', 'true');

  fireEvent.doubleClick(screen.getByRole('button', { name: 'Dock' }));
  const renameInput = screen.getByLabelText('Rename area Dock');
  fireEvent.change(renameInput, { target: { value: 'Dock Bay' } });
  fireEvent.keyDown(renameInput, { key: 'Enter' });

  await waitFor(() => expect(screen.getByRole('button', { name: 'Dock Bay' })).toBeInTheDocument());
  expect(screen.getByText('· unsaved')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Dock' })).toBeInTheDocument());
  fireEvent.click(screen.getByRole('button', { name: 'Redo' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Dock Bay' })).toBeInTheDocument());

  await waitFor(() => expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));

  await waitFor(() => expect(saveMapAnnotations).toHaveBeenCalledWith('factory.pgm', [
    expect.objectContaining({ label: 'Dock Bay' }),
  ]));
  expect(savePgmImage).not.toHaveBeenCalled();
});

test('freezes visible area cells when deleting an overlapping area', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 3,
    height: 1,
    maxval: 255,
    pixels_base64: '/v7+',
  });
  getMapAnnotations.mockResolvedValue({
    path: 'factory.pgm',
    annotations: [
      {
        id: 'area_front',
        label: 'Front',
        color: '#3B241F',
        pose: { frame_id: 'map', x: 0.5, y: 0.5, yaw: 0 },
        region: {
          seed_cell: { x: 0, y: 0 },
          bounds: { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
          cell_count: 1,
          width: 3,
          height: 1,
        },
      },
      {
        id: 'area_back',
        label: 'Back',
        color: '#6D1F2A',
        pose: { frame_id: 'map', x: 1.5, y: 0.5, yaw: 0 },
        region: {
          seed_cell: { x: 1, y: 0 },
          bounds: { x_min: 0, y_min: 0, x_max: 2, y_max: 0 },
          cell_count: 3,
          width: 3,
          height: 1,
        },
      },
    ],
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({ label: 'Front' }),
    expect.objectContaining({ label: 'Back' }),
  ]));

  fireEvent.click(screen.getByRole('button', { name: 'Add Label' }));
  fireEvent.click(screen.getByRole('button', { name: 'Area' }));
  // Whole-area delete now lives in the popover's area list (two-click confirm).
  fireEvent.click(screen.getByRole('button', { name: 'Delete area Front' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm delete area Front' }));

  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({
      label: 'Back',
      region: expect.objectContaining({
        cells: [{ x: 1, y: 0 }, { x: 2, y: 0 }],
        cell_count: 2,
      }),
    }),
  ]));
});

test('extends the selected area with the extend brush', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 3,
    height: 1,
    maxval: 255,
    pixels_base64: '/v7+',
  });
  getMapAnnotations.mockResolvedValue({
    path: 'factory.pgm',
    annotations: [{
      id: 'area_dock',
      label: 'Dock',
      color: '#3B241F',
      pose: { frame_id: 'map', x: 0.5, y: 0.5, yaw: 0 },
      region: {
        seed_cell: { x: 0, y: 0 },
        bounds: { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
        cells: [{ x: 0, y: 0 }],
        cell_count: 1,
        width: 3,
        height: 1,
      },
    }],
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();
  await waitFor(() => expect(getMapAnnotations).toHaveBeenCalledWith('factory.pgm'));

  fireEvent.click(screen.getByRole('button', { name: 'Add Label' }));
  // Brush sizes belong to the stroke tools only (Extend/Erase), never Area.
  expect(screen.queryByRole('button', { name: 'Brush size XL' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Extend' }));
  expect(screen.getByRole('button', { name: 'Brush size XL' })).toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole('button', { name: 'Dock' })).toBeInTheDocument());
  await waitFor(() => expect(latestMapViewerProps().editorActive).toBe(true));
  expect(latestMapViewerProps().editorAreaSelection).toBe(false);
  // Brush tools surface a pointer-following ring spec to the viewer.
  expect(latestMapViewerProps().editorBrush).toEqual(
    expect.objectContaining({ sizeCells: 1, color: '#5B8266' }),
  );

  // Without a selected area the extend brush is a guided no-op.
  await act(async () => {
    latestMapViewerProps().onEditorMapPoint(1.5, 0.5, 'start');
    latestMapViewerProps().onEditorMapPoint(0, 0, 'end');
  });
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({ region: expect.objectContaining({ cell_count: 1 }) }),
  ]));

  fireEvent.click(screen.getByRole('button', { name: 'Dock' }));
  expect(screen.getByRole('button', { name: 'Dock' })).toHaveAttribute('aria-pressed', 'true');
  await waitFor(() => expect(latestMapViewerProps().selectedMapAnnotationId).toBe('area_dock'));

  await act(async () => {
    latestMapViewerProps().onEditorMapPoint(0.5, 0.5, 'start');
    latestMapViewerProps().onEditorMapPoint(1.5, 0.5, 'move');
    latestMapViewerProps().onEditorMapPoint(0, 0, 'end');
  });

  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({
      label: 'Dock',
      region: expect.objectContaining({
        cells: [{ x: 0, y: 0 }, { x: 1, y: 0 }],
        cell_count: 2,
      }),
    }),
  ]));
  expect(screen.getByText('· unsaved')).toBeInTheDocument();

  // The whole stroke is a single undo entry.
  fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({ region: expect.objectContaining({ cell_count: 1 }) }),
  ]));
});

test('erases map area pixels with brush drag', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getPgmImage.mockResolvedValue({
    path: 'factory.pgm',
    width: 3,
    height: 1,
    maxval: 255,
    pixels_base64: '/v7+',
  });
  getMapAnnotations.mockResolvedValue({
    path: 'factory.pgm',
    annotations: [{
      id: 'area_dock',
      label: 'Dock',
      color: '#3B241F',
      pose: { frame_id: 'map', x: 1.5, y: 0.5, yaw: 0 },
      region: {
        seed_cell: { x: 1, y: 0 },
        bounds: { x_min: 0, y_min: 0, x_max: 2, y_max: 0 },
        cells: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 }],
        cell_count: 3,
        width: 3,
        height: 1,
      },
    }],
  });

  render(<AutonomyStudioPage />);

  await openMappingEditorAndSelect();

  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({ label: 'Dock' }),
  ]));

  fireEvent.click(screen.getByRole('button', { name: 'Add Label' }));
  fireEvent.click(screen.getByRole('button', { name: 'Erase Area' }));
  await act(async () => {
    latestMapViewerProps().onEditorMapPoint(0.5, 0.5, 'start');
    latestMapViewerProps().onEditorMapPoint(1.5, 0.5, 'move');
    latestMapViewerProps().onEditorMapPoint(0, 0, 'end');
  });

  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({
      label: 'Dock',
      region: expect.objectContaining({
        cells: [{ x: 2, y: 0 }],
        cell_count: 1,
      }),
    }),
  ]));

  fireEvent.click(screen.getByRole('button', { name: 'Save' }));
  await waitFor(() => expect(saveMapAnnotations).toHaveBeenCalledWith(
    'factory.pgm',
    [expect.objectContaining({
      region: expect.objectContaining({ cells: [{ x: 2, y: 0 }] }),
    })],
  ));
});

test('enables live robot and lidar layers while navigation runtime is active', async () => {
  getServiceStatus.mockResolvedValueOnce({ is_up: true, mode: 'map' });

  render(<AutonomyStudioPage />);

  await waitFor(() => {
    expect(mockMapViewer.mock.calls.some(([props]) => (
      props.showScan === true &&
      props.showRobotModel === true
    ))).toBe(true);
  });
});

test('anchors Mapping robot, lidar, and TF to the scan-matched SLAM pose', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getServiceStatus.mockResolvedValue({ is_up: true, mode: 'map' });
  mockTopicDataByName['/tf'] = {
    transforms: [
      {
        header: { frame_id: 'map', stamp: { sec: 10, nanosec: 0 } },
        child_frame_id: 'odom',
        transform: {
          translation: { x: 50, y: 50, z: 0 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
        },
      },
      {
        header: { frame_id: 'odom', stamp: { sec: 10, nanosec: 0 } },
        child_frame_id: 'base_link',
        transform: {
          translation: { x: 99, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
        },
      },
    ],
  };
  mockTopicDataByName['/scan'] = {
    header: { frame_id: 'base_link', stamp: { sec: 10, nanosec: 0 } },
    ranges: [1],
    range_min: 0.02,
    range_max: 20,
    angle_min: 0,
    angle_increment: 0,
  };
  mockTopicDataByName['/pose'] = {
    header: { frame_id: 'map', stamp: { sec: 10, nanosec: 0 } },
    pose: { pose: { position: { x: 1.25, y: -0.5, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } } },
  };
  mockTopicDataByName['/odom'] = {
    header: { frame_id: 'odom', stamp: { sec: 10, nanosec: 0 } },
    child_frame_id: 'base_link',
    pose: { pose: { position: { x: 2, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } } },
  };

  render(<AutonomyStudioPage />);

  await waitFor(() => expect(latestMapViewerProps().pose).toMatchObject({
    position: { x: 1.25, y: -0.5 },
  }));
  expect(latestMapViewerProps().scanPose).toMatchObject({
    position: { x: 1.25, y: -0.5 },
  });
  expect(latestMapViewerProps().tf.transforms.find((transform) => (
    transform.header.frame_id === 'map' && transform.child_frame_id === 'odom'
  ))).toMatchObject({
    transform: { translation: { x: -0.75, y: -0.5 } },
  });
  expect(latestMapViewerProps().tf.transforms.find((transform) => (
    transform.header.frame_id === 'odom' && transform.child_frame_id === 'base_link'
  ))).toMatchObject({
    transform: { translation: { x: 2, y: 0 } },
  });
  expect(screen.getByText('/pose')).toBeInTheDocument();
  expect(screen.getByText('/odom')).toBeInTheDocument();
  expect(screen.queryByText('/amcl_pose')).not.toBeInTheDocument();
});

test('enables navigation runtime layers in the run stage', async () => {
  getServiceStatus.mockResolvedValueOnce({ is_up: true, mode: 'run' });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));

  expect(screen.getByText('Run Session')).toBeInTheDocument();
  expect(screen.getByText('Runtime')).toBeInTheDocument();
  await waitFor(() => expect(screen.getByText('Running')).toBeInTheDocument());
  expect(screen.queryByText('PID:')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Load Map' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Run Mission' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Navigation' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Run BT' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Save Map' })).not.toBeInTheDocument();
  expect(screen.getByText('/global_costmap/costmap')).toBeInTheDocument();
  expect(screen.getByText('/local_costmap/costmap')).toBeInTheDocument();
  expect(screen.getByText('/plan')).toBeInTheDocument();
  expect(screen.queryByText('/goal_pose')).not.toBeInTheDocument();
  expect(screen.queryByRole('switch', { name: 'Goal pose' })).not.toBeInTheDocument();
  expect(screen.getByText('/bt/status')).toBeInTheDocument();
  expect(screen.queryByText('/tf')).not.toBeInTheDocument();
  await waitFor(() => {
    expect(mockMapViewer.mock.calls.some(([props]) => (
      props.showGlobalCostmap === true &&
      props.showLocalCostmap === true &&
      props.showGlobalPlan === true &&
      props.showGoalPose === false
    ))).toBe(true);
  });
  const globalCostmapSwitch = screen.getByRole('switch', { name: 'Global costmap' });
  expect(globalCostmapSwitch).toHaveAttribute('aria-checked', 'true');

  fireEvent.click(globalCostmapSwitch);
  expect(globalCostmapSwitch).toHaveAttribute('aria-checked', 'false');
  expect(screen.queryByText('/global_costmap/costmap')).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('switch', { name: 'TF' }));
  expect(screen.getByText('/tf')).toBeInTheDocument();
  expect(screen.getByText('/tf_static')).toBeInTheDocument();

  expect(mockMapViewer.mock.calls.some(([props]) => (
    props.showGlobalCostmap === true &&
    props.showLocalCostmap === true &&
    props.showGlobalPlan === true &&
    props.showGoalPose === false
  ))).toBe(true);
});

test('aligns the Run pose and odom-frame overlays to AMCL instead of stale TF', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getServiceStatus.mockResolvedValue({ is_up: true, mode: 'run' });
  mockTopicDataByName['/tf'] = {
    transforms: [
      {
        header: { frame_id: 'map' },
        child_frame_id: 'odom',
        transform: {
          translation: { x: 50, y: 50, z: 0 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
        },
      },
      {
        header: { frame_id: 'odom' },
        child_frame_id: 'base_link',
        transform: {
          translation: { x: 2, y: 3, z: 0 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
        },
      },
    ],
  };
  mockTopicDataByName['/amcl_pose'] = {
    ...amclPoseMessage(1.25, -0.5, 0),
    header: { frame_id: 'map', stamp: { sec: 10, nanosec: 0 } },
  };
  mockTopicDataByName['/odom'] = {
    header: { frame_id: 'odom', stamp: { sec: 10, nanosec: 0 } },
    child_frame_id: 'base_link',
    pose: {
      pose: {
        position: { x: 2, y: 3, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      },
    },
  };

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));

  await waitFor(() => expect(latestMapViewerProps().pose).toMatchObject({
    position: { x: 1.25, y: -0.5 },
  }));

  fireEvent.click(screen.getByRole('switch', { name: 'Lidar' }));
  fireEvent.click(screen.getByRole('switch', { name: 'Robot Footprint' }));
  fireEvent.click(screen.getByRole('switch', { name: 'TF' }));

  await waitFor(() => expect(latestMapViewerProps().pose).toMatchObject({
    position: { x: 1.25, y: -0.5 },
  }));
  await waitFor(() => expect(latestMapViewerProps().tf.transforms.find((transform) => (
    transform.header.frame_id === 'map' && transform.child_frame_id === 'odom'
  ))).toMatchObject({
    transform: { translation: { x: -0.75, y: -3.5 } },
  }));
  expect(latestMapViewerProps().tf.transforms.find((transform) => (
    transform.header.frame_id === 'odom' && transform.child_frame_id === 'base_link'
  ))).toMatchObject({
    transform: { translation: { x: 2, y: 3 } },
  });
  expect(topicRow('/amcl_pose')).toHaveTextContent('live');
});

test('loads a saved map for the run stage', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getMapAnnotations.mockResolvedValue({
    path: 'factory.pgm',
    annotations: [{
      id: 'area_dock',
      label: 'Dock',
      color: '#3B241F',
      pose: { frame_id: 'map', x: 0.5, y: 0.5, yaw: 0 },
      region: {
        seed_cell: { x: 0, y: 0 },
        bounds: { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
        cell_count: 1,
        width: 1,
        height: 1,
      },
    }],
  });
  getNavigationMission.mockImplementation((mapName) => Promise.resolve(
    mapName === 'factory'
      ? {
        exists: true,
        map_name: 'factory',
        revision: 3,
        global_bt: 'global.xml',
        waypoints: [{
          id: 'run_waypoint',
          label: 'Run Waypoint',
          pose: { frame_id: 'map', x: 1, y: 2, yaw: 0.5 },
          local_bt: 'locals/run_waypoint.xml',
          local_bt_files: [
            'locals/run_waypoint.xml',
            'locals/run_waypoint_alternate.xml',
          ],
          metadata: {},
        }],
        metadata: {},
      }
      : {
        exists: false,
        map_name: mapName,
        global_bt: 'global.xml',
        waypoints: [],
        metadata: {},
      },
  ));
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => {
    if (path === 'locals/run_waypoint_alternate.xml') {
      return Promise.reject(new Error('Unused alternate is unavailable'));
    }
    return Promise.resolve({
      path,
      content: `<root><BehaviorTree ID="${path}"/></root>`,
      exists: true,
      revision: 3,
    });
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));

  const mapSelect = await screen.findByRole('combobox', { name: 'Run mission map file' });
  await waitFor(() => expect(mapSelect).toHaveValue('factory.pgm'));

  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  // Map name now appears both in the left-rail mission summary and the Run Session panel.
  await waitFor(() => expect(screen.getAllByText('factory').length).toBeGreaterThan(0));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(1));
  expect(latestMapViewerProps().spots[0]).toMatchObject({
    id: 'run_waypoint',
    label: 'Run Waypoint',
    linked_bt_tree: 'locals/run_waypoint.xml',
  });
  expect(getNavigationMissionBtFile).toHaveBeenCalledWith(
    'factory',
    'locals/run_waypoint.xml',
    '',
  );
  expect(getNavigationMissionBtFile).not.toHaveBeenCalledWith(
    'factory',
    'locals/run_waypoint_alternate.xml',
    '',
  );
  expect(getNavigationSpots.mock.calls.some(([mapName]) => mapName === 'factory')).toBe(false);
  // Labeled map areas from Edit Map render in Run too.
  await waitFor(() => expect(getMapAnnotations).toHaveBeenCalledWith('factory.pgm'));
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([
    expect.objectContaining({ label: 'Dock', color: '#3B241F' }),
  ]));
  // ...and viewers who don't want them can hide the layer.
  const areasSwitch = screen.getByRole('switch', { name: 'Map areas' });
  expect(areasSwitch).toHaveAttribute('aria-checked', 'true');
  fireEvent.click(areasSwitch);
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toEqual([]));
  fireEvent.click(areasSwitch);
  await waitFor(() => expect(latestMapViewerProps().mapAnnotations).toHaveLength(1));
  // Run waypoints are display-only: no drag or selection handlers at all.
  expect(latestMapViewerProps().onSpotPoseChange).toBeUndefined();
  expect(latestMapViewerProps().onSpotClick).toBeUndefined();
  expect(latestMapViewerProps().selectedSpotId).toBe('');

  // Localize brings the nav stack up (Run Mission runs the route afterwards).
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));

  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'factory'));
});

test('rejects a Run snapshot assembled from different mission revisions', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions.mockResolvedValue({ map_name: 'factory', missions: ['default'] });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    mission_name: 'default',
    revision: 7,
    global_bt: 'global.xml',
    waypoints: [{
      id: 'run_waypoint',
      label: 'Run Waypoint',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0.5 },
      local_bt: 'locals/run_waypoint.xml',
      local_bt_files: ['locals/run_waypoint.xml'],
      metadata: {},
    }],
    metadata: {},
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: `<root><BehaviorTree ID="${path}"/></root>`,
    exists: true,
    revision: path === 'global.xml' ? 7 : 8,
  }));

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Run mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled());
  expect(latestMapViewerProps().spots).toEqual([]);
  expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled();
});

test('allows changing the Run mission while navigation stays active between runs', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getServiceStatus.mockResolvedValue({ is_up: true, mode: 'run' });
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions.mockResolvedValue({
    map_name: 'factory',
    missions: ['morning', 'evening'],
  });
  getNavigationMission.mockImplementation((mapName, missionName) => Promise.resolve({
    exists: true,
    map_name: mapName,
    mission_name: missionName,
    global_bt: 'global.xml',
    waypoints: [{
      id: `${missionName}_waypoint`,
      label: `${missionName} waypoint`,
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      local_bt: `locals/${missionName}.xml`,
      metadata: {},
    }],
    metadata: {},
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Run mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  const missionSelect = await screen.findByRole('combobox', { name: 'Active mission' });
  await waitFor(() => expect(missionSelect).toHaveValue('morning'));
  expect((await screen.findAllByText('Running')).length).toBeGreaterThan(0);
  expect(missionSelect).toBeEnabled();

  fireEvent.change(missionSelect, { target: { value: 'evening' } });

  await waitFor(() => expect(getNavigationMission).toHaveBeenCalledWith('factory', 'evening'));
  await waitFor(() => expect(missionSelect).toHaveValue('evening'));
  await waitFor(() => expect(latestMapViewerProps().spots[0]).toMatchObject({
    id: 'evening_waypoint',
    label: 'evening waypoint',
  }));
});

test('keeps the previous Run snapshot when a mission switch fails', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  let rejectEveningLoad;
  getServiceStatus.mockResolvedValue({ is_up: true, mode: 'run' });
  mockTopicDataByName['/amcl_pose'] = amclPoseMessage(0, 0, 0);
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions.mockResolvedValue({
    map_name: 'factory',
    missions: ['morning', 'evening'],
  });
  getNavigationMission.mockImplementation((mapName, missionName) => {
    if (missionName === 'evening') {
      return new Promise((_resolve, reject) => {
        rejectEveningLoad = reject;
      });
    }
    return Promise.resolve({
      exists: true,
      map_name: mapName,
      mission_name: missionName,
      global_bt: 'global.xml',
      waypoints: [{
        id: 'morning_waypoint',
        label: 'morning waypoint',
        pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
        local_bt: 'locals/morning.xml',
        metadata: {},
      }],
      metadata: {},
    });
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  await screen.findByRole('combobox', { name: 'Run mission map file' });
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));

  const missionSelect = await screen.findByRole('combobox', { name: 'Active mission' });
  await waitFor(() => expect(missionSelect).toHaveValue('morning'));
  await waitFor(() => expect(latestMapViewerProps().spots[0]).toMatchObject({
    id: 'morning_waypoint',
  }));
  fireEvent.change(missionSelect, { target: { value: 'evening' } });
  await waitFor(() => expect(getNavigationMission).toHaveBeenCalledWith('factory', 'evening'));
  expect(missionSelect).toHaveValue('morning');
  expect(missionSelect).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Run Mission' })).toBeDisabled();

  await act(async () => {
    rejectEveningLoad(new Error('Evening mission unavailable'));
  });
  await waitFor(() => expect(missionSelect).toHaveValue('morning'));
  expect(missionSelect).toHaveValue('morning');
  expect(latestMapViewerProps().spots[0]).toMatchObject({ id: 'morning_waypoint' });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());
});

test('stops an owned Run runtime across a full page refresh', async () => {
  let navigationUp = false;
  getServiceStatus.mockImplementation(() => Promise.resolve(
    navigationUp ? { is_up: true, mode: 'run' } : { is_up: false },
  ));
  startNavigation.mockImplementation(() => {
    navigationUp = true;
    return Promise.resolve({ ok: true, message: 'started' });
  });
  stopNavigation.mockImplementation(() => {
    navigationUp = false;
    return Promise.resolve({ ok: true, message: 'stopped' });
  });

  const currentDocument = render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  await loadRunMapFromDialog();
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));

  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'map'));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toHaveAttribute('aria-pressed', 'true'));
  stopNavigation.mockClear();

  fireEvent(window, new Event('pagehide'));
  expect(stopNavigation).toHaveBeenCalledTimes(1);
  expect(stopNavigation).toHaveBeenCalledWith({ keepalive: true });
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session'))).toEqual(
    expect.objectContaining({
      navigationRuntimeMode: 'idle',
      runRuntimeOwned: true,
      runShutdownPending: true,
      runShutdownRequestedAt: expect.any(Number),
    }),
  );

  // Guard duplicate delivery so it cannot create a burst of shutdown calls.
  fireEvent(window, new Event('pagehide'));
  expect(stopNavigation).toHaveBeenCalledTimes(1);

  currentDocument.unmount();
  stopNavigation.mockClear();
  render(<StrictMode><AutonomyStudioPage /></StrictMode>);

  await waitFor(() => expect(
    stopNavigation.mock.calls.some((args) => args.length === 0),
  ).toBe(true));
  expect(stopNavigation.mock.calls.filter((args) => args.length === 0)).toHaveLength(1);
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')),
  ).toEqual(expect.objectContaining({
    navigationRuntimeMode: 'idle',
    runRuntimeOwned: false,
    runShutdownPending: false,
  })));
  await waitFor(() => expect(screen.getByText('Status: idle')).toBeInTheDocument());
});

test('stops Run even when the page refreshes while navigation is starting', async () => {
  let resolveStart;
  startNavigation.mockReturnValue(new Promise((resolve) => {
    resolveStart = resolve;
  }));

  const currentDocument = render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  await loadRunMapFromDialog();
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'map'));

  fireEvent(window, new Event('pagehide'));
  expect(stopNavigation).toHaveBeenCalledWith({ keepalive: true });
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session'))).toEqual(
    expect.objectContaining({
      navigationRuntimeMode: 'idle',
      runRuntimeOwned: true,
      runShutdownPending: true,
    }),
  );

  await act(async () => {
    resolveStart({ ok: true, message: 'started' });
    await Promise.resolve();
  });
  expect(stopNavigation.mock.calls.filter(
    ([options]) => options?.keepalive === true,
  )).toHaveLength(2);
  // The late start completion must not overwrite the page-exit fallback.
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session'))).toEqual(
    expect.objectContaining({
      navigationRuntimeMode: 'idle',
      runRuntimeOwned: true,
      runShutdownPending: true,
    }),
  );
  currentDocument.unmount();
});

test('retries the page-exit stop after a partially failed navigation start', async () => {
  let rejectStart;
  startNavigation.mockReturnValue(new Promise((resolve, reject) => {
    rejectStart = reject;
  }));

  const currentDocument = render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  await loadRunMapFromDialog();
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'map'));
  fireEvent(window, new Event('pagehide'));

  await act(async () => {
    rejectStart(new Error('start timed out'));
    await Promise.resolve();
  });

  expect(stopNavigation.mock.calls.filter(
    ([options]) => options?.keepalive === true,
  )).toHaveLength(2);
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session'))).toEqual(
    expect.objectContaining({
      navigationRuntimeMode: 'idle',
      runShutdownPending: true,
    }),
  );
  currentDocument.unmount();
});

test('does not stop a Mapping runtime when the page exits', async () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceStage: 'mapping',
    navigationRuntimeMode: 'mapping',
    runRuntimeOwned: true,
    runShutdownPending: false,
  }));
  getServiceStatus.mockResolvedValue({ is_up: true, mode: 'map' });

  render(<AutonomyStudioPage />);
  await waitFor(() => expect(screen.getByText('Status: running')).toBeInTheDocument());
  fireEvent(window, new Event('pagehide'));

  expect(stopNavigation).not.toHaveBeenCalled();
});

test('does not apply an expired Run shutdown marker to a later runtime', async () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceStage: 'run',
    navigationRuntimeMode: 'idle',
    runRuntimeOwned: true,
    runShutdownPending: true,
    runShutdownRequestedAt: Date.now() - 120_000,
  }));
  getServiceStatus.mockResolvedValue({ is_up: true, mode: 'run' });

  render(<AutonomyStudioPage />);
  await waitFor(() => expect(screen.getByText('Status: running')).toBeInTheDocument());
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')),
  ).toEqual(expect.objectContaining({
    runRuntimeOwned: false,
    runShutdownPending: false,
  })));
  fireEvent(window, new Event('pagehide'));

  expect(stopNavigation).not.toHaveBeenCalled();
});

test('lists the mission route waypoints in the run session panel', async () => {
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMission.mockImplementation((mapName) => Promise.resolve(
    mapName === 'factory'
      ? {
        exists: true,
        map_name: 'factory',
        global_bt: 'global.xml',
        waypoints: [
          { id: 'wp1', label: 'Kitchen', pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 }, local_bt: 'locals/wp1.xml', metadata: {} },
          { id: 'wp2', label: 'Living Room', pose: { frame_id: 'map', x: 4, y: 0, yaw: 0 }, local_bt: 'locals/wp2.xml', metadata: {} },
        ],
        metadata: {
          mission_flow: {
            nodes: [{ id: 'wp1', position: { x: 80, y: 72 } }, { id: 'wp2', position: { x: 300, y: 72 } }],
            edges: [
              { id: 'e1', source: 'wp1', target: 'wp2' },
              { id: 'e2', source: 'wp2', target: 'wp1' },
            ],
          },
        },
      }
      : { exists: false, map_name: mapName, global_bt: 'global.xml', waypoints: [], metadata: {} },
  ));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'Run mission map file' });
  await waitFor(() => expect(mapSelect).toHaveValue('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));


  // The Run Session panel reflects the loaded route as an ordered checklist.
  const waypointList = await screen.findByRole('list', { name: 'Mission waypoints' });
  expect(within(waypointList).getByText('Kitchen')).toBeInTheDocument();
  expect(within(waypointList).getByText('Living Room')).toBeInTheDocument();
  expect(within(waypointList).getByText('Return to Kitchen')).toBeInTheDocument();
});

test('clears the loaded Run map and mission snapshot when navigation stops', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  let navigationUp = false;
  getServiceStatus.mockImplementation(() => Promise.resolve(
    navigationUp ? { is_up: true, mode: 'nav' } : { is_up: false, mode: 'idle' },
  ));
  startNavigation.mockImplementation(() => {
    navigationUp = true;
    return Promise.resolve({ ok: true, message: 'started' });
  });
  let resolveStopNavigation;
  stopNavigation.mockImplementation(() => {
    navigationUp = false;
    return new Promise((resolve) => {
      resolveStopNavigation = () => resolve({ ok: true, message: 'stopped' });
    });
  });
  getPgmFiles.mockResolvedValue({
    files: [
      { path: 'map.pgm', name: 'map.pgm' },
      { path: 'factory.pgm', name: 'factory.pgm' },
    ],
  });
  getNavigationMissions.mockResolvedValue({
    map_name: 'factory',
    missions: ['inspection'],
  });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    mission_name: 'inspection',
    global_bt: 'global.xml',
    waypoints: [
      { id: 'wp1', label: 'Kitchen', pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 }, local_bt: 'locals/wp1.xml', metadata: {} },
      { id: 'wp2', label: 'Living Room', pose: { frame_id: 'map', x: 4, y: 0, yaw: 0 }, local_bt: 'locals/wp2.xml', metadata: {} },
    ],
    metadata: {
      mission_flow: {
        nodes: [
          { id: 'wp1', position: { x: 80, y: 72 } },
          { id: 'wp2', position: { x: 300, y: 72 } },
        ],
        edges: [{ id: 'e1', source: 'wp1', target: 'wp2' }],
      },
    },
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const initialLoadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  const initialMapSelect = within(initialLoadDialog).getByRole('combobox', {
    name: 'Run mission map file',
  });
  await waitFor(() => expect(initialMapSelect).toHaveValue('map.pgm'));
  fireEvent.change(initialMapSelect, { target: { value: 'factory.pgm' } });
  await waitFor(() => expect(initialMapSelect).toHaveValue('factory.pgm'));
  await waitFor(() => expect(within(initialLoadDialog).getByRole('combobox', {
    name: 'Run mission file',
  })).toHaveValue('inspection'));
  const initialLoadButton = within(initialLoadDialog).getByRole('button', { name: 'Load' });
  await waitFor(() => expect(initialLoadButton).toBeEnabled());
  fireEvent.click(initialLoadButton);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());

  const runSessionPanel = screen.getByText('Run Session').parentElement;
  expect(within(runSessionPanel).getByText('Selected map').parentElement)
    .toHaveTextContent('factory');
  expect(within(runSessionPanel).getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('inspection');
  await waitFor(() => expect(latestMapViewerProps().map).not.toBeNull());
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(2));
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'wp1', order: 1 },
    { id: 'wp2', order: 2 },
  ]);
  expect(screen.getByRole('list', { name: 'Mission waypoints' })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'factory'));
  const stopButton = screen.getByRole('button', { name: 'Stop' });
  await waitFor(() => expect(stopButton).toBeEnabled());
  fireEvent.click(stopButton);

  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  // The supervisor shutdown can take noticeably longer than the UI status/map
  // transition. Reset the Run snapshot as part of the click itself instead of
  // leaving the old map, mission and progress visible until this request
  // eventually settles (or forever when it fails).
  expect(resolveStopNavigation).toEqual(expect.any(Function));
  expect(within(runSessionPanel).getByText('Selected map').parentElement)
    .toHaveTextContent('Not selected');
  expect(within(runSessionPanel).getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('');
  expect(screen.queryByRole('list', { name: 'Mission waypoints' })).not.toBeInTheDocument();

  await act(async () => {
    resolveStopNavigation();
    await Promise.resolve();
  });
  await waitFor(() => expect(latestMapViewerProps().map).toBeNull());
  await waitFor(() => expect(latestMapViewerProps().spots).toEqual([]));
  expect(latestMapViewerProps().missionRouteOrder).toEqual([]);
  expect(screen.queryByRole('list', { name: 'Mission waypoints' })).not.toBeInTheDocument();
  expect(within(runSessionPanel).getByText('Selected map').parentElement)
    .toHaveTextContent('Not selected');
  expect(within(runSessionPanel).getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('');
  expect(within(runSessionPanel).getByRole('combobox', { name: 'Active mission' }))
    .toBeDisabled();
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')).runMissionName,
  ).toBe(''));
  expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Run Mission' })).toBeDisabled();

  // Reset the executable snapshot, but retain the last file choice so the
  // operator can reload the same Run map without finding it again.
  const pgmCallsBeforeReload = getPgmFiles.mock.calls.length;
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const reloadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  await waitFor(() => expect(getPgmFiles.mock.calls.length)
    .toBeGreaterThan(pgmCallsBeforeReload));
  await waitFor(() => expect(within(reloadDialog).getByRole('button', { name: 'Load' }))
    .toBeEnabled());
  expect(within(reloadDialog).getByRole('combobox', {
    name: 'Run mission map file',
  })).toHaveValue('factory.pgm');
});

test('clears the Run snapshot on Stop intent while retaining ownership after a lost response', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  let navigationUp = false;
  let rejectStopNavigation;
  getServiceStatus.mockImplementation(() => Promise.resolve(
    navigationUp ? { is_up: true, mode: 'nav' } : { is_up: false, mode: 'idle' },
  ));
  startNavigation.mockImplementation(() => {
    navigationUp = true;
    return Promise.resolve({ ok: true, message: 'started' });
  });
  stopNavigation.mockImplementation(() => new Promise((_resolve, reject) => {
    rejectStopNavigation = reject;
  }));
  getNavigationMissions.mockResolvedValue({
    map_name: 'factory',
    missions: ['inspection'],
  });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    mission_name: 'inspection',
    global_bt: 'global.xml',
    waypoints: [
      { id: 'wp1', label: 'Kitchen', pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 }, local_bt: 'locals/wp1.xml', metadata: {} },
      { id: 'wp2', label: 'Living Room', pose: { frame_id: 'map', x: 4, y: 0, yaw: 0 }, local_bt: 'locals/wp2.xml', metadata: {} },
    ],
    metadata: {
      mission_flow: {
        nodes: [
          { id: 'wp1', position: { x: 80, y: 72 } },
          { id: 'wp2', position: { x: 300, y: 72 } },
        ],
        edges: [{ id: 'e1', source: 'wp1', target: 'wp2' }],
      },
    },
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  await loadRunMapFromDialog('factory.pgm');
  const runSessionPanel = screen.getByText('Run Session').parentElement;
  await screen.findByRole('list', { name: 'Mission waypoints' });
  await waitFor(() => expect(latestMapViewerProps().map).not.toBeNull());

  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'factory'));
  const stopButton = screen.getByRole('button', { name: 'Stop' });
  await waitFor(() => expect(stopButton).toBeEnabled());
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')),
  ).toEqual(expect.objectContaining({
    navigationRuntimeMode: 'run',
    runRuntimeOwned: true,
  })));

  fireEvent.click(stopButton);

  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  expect(rejectStopNavigation).toEqual(expect.any(Function));
  expect(latestMapViewerProps().map).toBeNull();
  expect(latestMapViewerProps().spots).toEqual([]);
  expect(latestMapViewerProps().missionRouteOrder).toEqual([]);
  expect(within(runSessionPanel).getByText('Selected map').parentElement)
    .toHaveTextContent('Not selected');
  expect(within(runSessionPanel).getByRole('combobox', { name: 'Active mission' }))
    .toHaveValue('');
  expect(within(runSessionPanel).getByRole('combobox', { name: 'Active mission' }))
    .toBeDisabled();
  expect(screen.queryByRole('list', { name: 'Mission waypoints' })).not.toBeInTheDocument();
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')))
    .toEqual(expect.objectContaining({
      navigationRuntimeMode: 'run',
      runMissionName: '',
      runRuntimeOwned: true,
      runShutdownPending: false,
    }));

  await act(async () => {
    rejectStopNavigation(new Error('Stop response lost'));
    await Promise.resolve();
  });

  await waitFor(() => expect(screen.getByText('Status: running')).toBeInTheDocument());
  await waitFor(() => expect(stopButton).toBeEnabled());
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')))
    .toEqual(expect.objectContaining({
      navigationRuntimeMode: 'run',
      runMissionName: '',
      runRuntimeOwned: true,
      runShutdownPending: false,
    }));
  expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Run Mission' })).toBeDisabled();
});

test('hides run waypoints with the map after leaving and returning to Run', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMission.mockImplementation((mapName) => Promise.resolve(
    mapName === 'factory'
      ? {
        exists: true,
        map_name: 'factory',
        global_bt: 'global.xml',
        waypoints: [
          { id: 'wp1', label: 'Kitchen', pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 }, local_bt: 'locals/wp1.xml', metadata: {} },
          { id: 'wp2', label: 'Living Room', pose: { frame_id: 'map', x: 4, y: 0, yaw: 0 }, local_bt: 'locals/wp2.xml', metadata: {} },
        ],
        metadata: {},
      }
      : { exists: false, map_name: mapName, global_bt: 'global.xml', waypoints: [], metadata: {} },
  ));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  // The BT lifecycle is run-owned (auto activate/release) — no manual panel here.
  expect(screen.queryByText('BT Runtime')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Activate BT' })).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'Run mission map file' });
  await waitFor(() => expect(mapSelect).toHaveValue('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(2));
  // Run shows the same raw map the mission was designed on (no beautification).
  expect(latestMapViewerProps().mapRefined).toBe(false);

  // Leave to Design and come back: the ephemeral map is dropped, and the
  // waypoints must vanish with it (without a map they would render at raw
  // scale — huge and overlapping).
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));
  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toEqual([]));
  expect(latestMapViewerProps().map).toBeNull();
  expect(latestMapViewerProps().missionRouteOrder).toEqual([]);
});

test('stops Run and clears its pending pose gesture before switching to Design', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  let navigationUp = false;
  getServiceStatus.mockImplementation(() => Promise.resolve(
    navigationUp ? { is_up: true, mode: 'nav' } : { is_up: false, mode: 'idle' },
  ));
  startNavigation.mockImplementation(() => {
    navigationUp = true;
    return Promise.resolve({ ok: true, message: 'started' });
  });
  stopNavigation.mockImplementation(() => {
    navigationUp = false;
    return Promise.resolve({ ok: true, message: 'stopped' });
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  await loadRunMapFromDialog('factory.pgm');
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'factory'));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  expect(navigationUp).toBe(true);

  sendInitialPoseEstimate.mockClear();
  createNavigationSpot.mockClear();
  stopNavigation.mockClear();
  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));

  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('view'));
  await waitFor(() => (
    expect(screen.getByRole('tab', { name: 'Design' })).toHaveAttribute('aria-selected', 'true')
  ));
  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  await act(async () => {
    latestMapViewerProps().onMapClick(1, 2);
    await latestMapViewerProps().onMapPose(1, 2, 0.25);
  });

  expect(sendInitialPoseEstimate).not.toHaveBeenCalled();
  expect(createNavigationSpot).not.toHaveBeenCalled();
  expect(navigationUp).toBe(false);
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')))
    .toEqual(expect.objectContaining({
      workspaceStage: 'authoring',
      navigationRuntimeMode: 'idle',
      runRuntimeOwned: false,
      runShutdownPending: false,
    }));
  expect(screen.queryByText('Stop the active navigation session before using At Robot'))
    .not.toBeInTheDocument();
});

test('gates the mission run on an initial robot pose', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  sendNavigateThroughPosesGoalsAndWait.mockResolvedValue({
    ok: true,
    status: 'SUCCEEDED',
    message: 'Goals succeeded',
  });
  sendNavigateToPoseGoalAndWait.mockResolvedValue({
    ok: true,
    status: 'SUCCEEDED',
    message: 'Goal succeeded',
  });
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMission.mockImplementation((mapName) => Promise.resolve(
    mapName === 'factory'
      ? {
        exists: true,
        map_name: 'factory',
        global_bt: 'global.xml',
        waypoints: [
          { id: 'wp1', label: 'Kitchen', pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 }, local_bt: 'locals/wp1.xml', metadata: {} },
          { id: 'wp2', label: 'Living Room', pose: { frame_id: 'map', x: 4, y: 0, yaw: 0 }, local_bt: 'locals/wp2.xml', metadata: {} },
        ],
        metadata: {
          mission_flow: {
            nodes: [{ id: 'wp1', position: { x: 80, y: 72 } }, { id: 'wp2', position: { x: 300, y: 72 } }],
            edges: [
              { id: 'e1', source: 'wp1', target: 'wp2' },
              { id: 'e2', source: 'wp2', target: 'wp1' },
            ],
          },
        },
      }
      : { exists: false, map_name: mapName, global_bt: 'global.xml', waypoints: [], metadata: {} },
  ));
  // A fresh AMCL message arrives once the initial pose is published.
  sendInitialPoseEstimate.mockImplementation(async () => {
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(0.6, 0.4, 0.1);
    return { ok: true };
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'Run mission map file' });
  await waitFor(() => expect(mapSelect).toHaveValue('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());

  // Run Mission is disabled until the robot is localized.
  getServiceStatus.mockResolvedValue({ is_up: true, mode: 'nav' });
  expect(screen.getByRole('button', { name: 'Run Mission' })).toBeDisabled();
  expect(sendNavigateToPoseGoalAndWait).not.toHaveBeenCalled();

  // Localize brings the nav stack up and enters the pose-set gesture.
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'factory'));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));

  // Operator sets the robot's real pose on the map.
  await act(async () => {
    latestMapViewerProps().onMapPose(0.6, 0.4, 0.1);
  });
  await waitFor(() => expect(sendInitialPoseEstimate).toHaveBeenCalledWith({
    x: 0.6,
    y: 0.4,
    yaw: 0.1,
    frameId: 'map',
  }));
  await waitFor(() => expect(screen.getByText('Ready')).toBeInTheDocument(), { timeout: 6000 });
  expect(sendNavigateToPoseGoalAndWait).not.toHaveBeenCalled();

  // Both saved waypoints have empty local BTs. Run batches the outward route,
  // then sends the closing wp2 -> wp1 leg as the final navigation goal.
  expect(latestMapViewerProps().missionRouteClosed).toBe(true);
  expect(screen.getByText('Return to Kitchen')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Run Mission' }));
  await waitFor(
    () => expect(sendNavigateThroughPosesGoalsAndWait).toHaveBeenCalled(),
    { timeout: 8000 },
  );
  const goals = sendNavigateThroughPosesGoalsAndWait.mock.calls[0][0];
  expect(goals.poses).toHaveLength(2);
  expect(goals.poses[0].pose.position).toMatchObject({ x: 1, y: 0 });
  await waitFor(() => expect(sendNavigateToPoseGoalAndWait).toHaveBeenCalled());
  const [closingRequest] = sendNavigateToPoseGoalAndWait.mock.calls.at(-1);
  expect(closingRequest.pose.pose.position).toMatchObject({ x: 1, y: 0 });
}, 20000);

test('run mission activates the BT node on demand and releases it on stop', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const filledBt = [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait duration="0.1"/></BehaviorTree>',
    '</root>',
  ].join('\n');
  // Supervisor mock with real node state: status reflects start/stop calls.
  let btUp = false;
  let resolveBtStop;
  global.fetch.mockImplementation((url) => {
    const target = String(url);
    if (target.includes('/services/bt_node/start')) {
      btUp = true;
      return Promise.resolve(mockJsonResponse({ ok: true }));
    }
    if (target.includes('/services/bt_node/stop')) {
      btUp = false;
      return new Promise((resolve) => {
        resolveBtStop = () => resolve(mockJsonResponse({ ok: true }));
      });
    }
    return Promise.resolve(mockJsonResponse({ name: 'bt_node', state: btUp ? 'up' : 'down', raw: '' }));
  });
  const navigationRequests = [];
  sendNavigateToPoseGoalAndWait.mockImplementation((goal, signal) => new Promise((resolve, reject) => {
    navigationRequests.push({ resolve, reject });
    signal.addEventListener('abort', () => {
      reject(new DOMException('Aborted', 'AbortError'));
    }, { once: true });
  }));
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMission.mockImplementation((mapName) => Promise.resolve(
    mapName === 'factory'
      ? {
        exists: true,
        map_name: 'factory',
        global_bt: 'global.xml',
        waypoints: [
          { id: 'wp1', label: 'Kitchen', pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 }, local_bt: 'locals/wp1.xml', metadata: {} },
          { id: 'wp2', label: 'Living Room', pose: { frame_id: 'map', x: 4, y: 0, yaw: 0 }, local_bt: 'locals/wp2.xml', metadata: {} },
        ],
        metadata: {
          mission_flow: {
            nodes: [{ id: 'wp1', position: { x: 80, y: 72 } }, { id: 'wp2', position: { x: 300, y: 72 } }],
            edges: [{ id: 'e1', source: 'wp1', target: 'wp2' }],
          },
        },
      }
      : { exists: false, map_name: mapName, global_bt: 'global.xml', waypoints: [], metadata: {} },
  ));
  // wp1 carries a real behavior, so the mission needs the BT node.
  getNavigationMissionBtFile.mockImplementation((mapName, path) => Promise.resolve(
    path === 'locals/wp1.xml'
      ? { path, content: filledBt, exists: true }
      : { path, content: '', exists: false },
  ));
  sendInitialPoseEstimate.mockImplementation(async () => {
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(0.6, 0.4, 0.1);
    return { ok: true };
  });
  mockTopicDataByName['/bt/status'] = stringTopicMessage('running');
  mockTopicDataByName['/bt/active_nodes'] = stringTopicMessage('Wait');

  // s6 can already report up while the ROS services are still being created.
  // Hold the read-only catalog probe to prove navigation cannot start in that
  // window; once it resolves, /bt/load_and_run is known to be registered too.
  let resolveBtReady;
  mockCallService.mockImplementation((serviceName) => {
    if (serviceName === '/bt/nodes/catalog') {
      return new Promise((resolve) => { resolveBtReady = resolve; });
    }
    return Promise.resolve({ success: true });
  });

  const { rerender } = render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'Run mission map file' });
  await waitFor(() => expect(mapSelect).toHaveValue('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());

  getServiceStatus.mockResolvedValue({ is_up: true, mode: 'nav' });
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'factory'));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  await act(async () => {
    latestMapViewerProps().onMapPose(0.6, 0.4, 0.1);
  });
  await waitFor(() => expect(screen.getByText('Ready')).toBeInTheDocument(), { timeout: 6000 });

  // Run Mission with the node down: activation happens before navigation.
  expect(global.fetch).not.toHaveBeenCalledWith('/api/services/bt_node/start', expect.anything());
  fireEvent.click(screen.getByRole('button', { name: 'Run Mission' }));
  await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
    '/api/services/bt_node/start',
    expect.objectContaining({ method: 'POST' }),
  ));
  await waitFor(() => expect(mockCallService).toHaveBeenCalledWith(
    '/bt/nodes/catalog',
    'interfaces/srv/GetNodeCatalog',
    {},
    1000,
  ));
  expect(sendNavigateToPoseGoalAndWait).not.toHaveBeenCalled();
  await act(async () => {
    resolveBtReady({ success: true, catalog_json: '[]' });
    await Promise.resolve();
  });
  await waitFor(() => expect(sendNavigateToPoseGoalAndWait).toHaveBeenCalled());
  await waitFor(() => expect(latestMapViewerProps().activeWaypointId).toBe('wp1'));

  // Arriving at wp1 opens the read-only BT canvas. While it is visible, keep
  // the base map, current pose/footprint, waypoint and route context, but
  // suspend the high-frequency UI-only overlays behind ReactFlow.
  fireEvent.click(screen.getByRole('switch', { name: 'TF' }));
  await waitFor(() => expect(latestMapViewerProps().showTf).toBe(true));
  await act(async () => {
    navigationRequests[0].resolve({ ok: true, status: 'SUCCEEDED', message: 'Goal succeeded' });
    await Promise.resolve();
  });
  await waitFor(() => expect(latestMapViewerProps().btLayer).not.toBeNull());
  expect(screen.getByText('Task running')).toBeInTheDocument();
  expect(screen.getAllByRole('button', { name: 'Stop' })).toHaveLength(1);
  await waitFor(() => expect(latestMapViewerProps()).toMatchObject({
    showMap: true,
    showGlobalCostmap: false,
    showLocalCostmap: false,
    showScan: false,
    showGlobalPlan: false,
    showTf: false,
    showRobotModel: true,
    activeWaypointId: 'wp1',
  }));
  expect(latestMapViewerProps().spots).toHaveLength(2);
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'wp1', order: 1 },
    { id: 'wp2', order: 2 },
  ]);
  // Closing the BT view restores the selected layers before the runner starts
  // navigating to the next waypoint.
  mockTopicDataByName['/bt/status'] = stringTopicMessage('completed');
  rerender(<AutonomyStudioPage />);
  await waitFor(() => expect(sendNavigateToPoseGoalAndWait).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(latestMapViewerProps().btLayer).toBeNull());
  await waitFor(() => expect(latestMapViewerProps()).toMatchObject({
    showGlobalCostmap: true,
    showLocalCostmap: true,
    showScan: true,
    showGlobalPlan: true,
    showTf: true,
    showRobotModel: true,
  }));

  // Stopping the run releases the node and clears the active waypoint before
  // the same map is loaded again. A stale currentIndex used to recreate an
  // orange pulsing marker as soon as the map layer returned.
  getServiceStatus.mockResolvedValue({ is_up: false, mode: 'idle' });
  const stopButton = screen.getByRole('button', { name: 'Stop' });
  fireEvent.click(stopButton);
  await waitFor(() => expect(latestMapViewerProps().activeWaypointId).toBe(''));
  await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
    '/api/services/bt_node/stop',
    expect.objectContaining({ method: 'POST' }),
  ));
  // Navigation shutdown can finish before the owned Task Engine release.
  // The already-cancelled runner must not make Stop clickable again during
  // that final cleanup window.
  await waitFor(() => expect(stopButton).not.toHaveAttribute('aria-pressed', 'true'));
  expect(stopButton).toBeDisabled();
  expect(global.fetch.mock.calls.filter(([url, options]) => (
    String(url) === '/api/services/bt_node/stop'
    && options?.method === 'POST'
  ))).toHaveLength(1);
  await act(async () => {
    resolveBtStop();
    await Promise.resolve();
  });
  const reloadMapButton = screen.getByRole('button', { name: 'Load Map' });
  await waitFor(() => expect(reloadMapButton).toBeEnabled());
  fireEvent.click(reloadMapButton);
  const reloadMapSelect = await screen.findByRole('combobox', { name: 'Run mission map file' });
  await waitFor(() => expect(reloadMapSelect).toHaveValue('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(2));
  expect(latestMapViewerProps().activeWaypointId).toBe('');
}, 20000);

test.each([
  ['is stopped by the operator', false],
  ['completes', true],
])('does not stop a pre-existing BT node when the mission %s', async (_ending, completesNaturally) => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  const filledBt = [
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"><Wait duration="0.1"/></BehaviorTree>',
    '</root>',
  ].join('\n');

  // The idle process belongs to another workspace and remains up throughout
  // this Mission Run. Mission may borrow its ROS services, but must not stop
  // the supervisor process when the run ends.
  global.fetch.mockImplementation((url) => {
    const target = String(url);
    if (target.endsWith('/services/bt_node/status')) {
      return Promise.resolve(mockJsonResponse({ name: 'bt_node', state: 'up', raw: 'up' }));
    }
    return Promise.resolve(mockJsonResponse({ ok: true }));
  });

  if (completesNaturally) {
    sendNavigateToPoseGoalAndWait.mockResolvedValue({
      ok: true,
      status: 'SUCCEEDED',
      message: 'Goal succeeded',
    });
  } else {
    sendNavigateToPoseGoalAndWait.mockImplementation((_goal, signal) => new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => {
        reject(new DOMException('Aborted', 'AbortError'));
      }, { once: true });
    }));
  }
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMission.mockImplementation((mapName) => Promise.resolve(
    mapName === 'factory'
      ? {
        exists: true,
        map_name: 'factory',
        global_bt: 'global.xml',
        waypoints: [
          { id: 'wp1', label: 'Kitchen', pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 }, local_bt: 'locals/wp1.xml', metadata: {} },
          { id: 'wp2', label: 'Living Room', pose: { frame_id: 'map', x: 4, y: 0, yaw: 0 }, local_bt: 'locals/wp2.xml', metadata: {} },
        ],
        metadata: {
          mission_flow: {
            nodes: [{ id: 'wp1', position: { x: 80, y: 72 } }, { id: 'wp2', position: { x: 300, y: 72 } }],
            edges: [{ id: 'e1', source: 'wp1', target: 'wp2' }],
          },
        },
      }
      : { exists: false, map_name: mapName, global_bt: 'global.xml', waypoints: [], metadata: {} },
  ));
  getNavigationMissionBtFile.mockImplementation((mapName, path) => Promise.resolve(
    path === 'locals/wp1.xml'
      ? { path, content: filledBt, exists: true }
      : { path, content: '', exists: false },
  ));
  sendInitialPoseEstimate.mockImplementation(async () => {
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(0.6, 0.4, 0.1);
    return { ok: true };
  });
  mockTopicDataByName['/bt/status'] = stringTopicMessage('stopped');
  let resolveBtLoad;
  mockCallService.mockImplementation((serviceName) => {
    if (serviceName === '/bt/load_and_run' && completesNaturally) {
      return new Promise((resolve) => { resolveBtLoad = resolve; });
    }
    return Promise.resolve({ success: true, catalog_json: '[]' });
  });

  const { rerender } = render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'Run mission map file' });
  await waitFor(() => expect(mapSelect).toHaveValue('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());

  let navigationUp = true;
  getServiceStatus.mockImplementation(() => Promise.resolve(
    navigationUp ? { is_up: true, mode: 'nav' } : { is_up: false, mode: 'idle' },
  ));
  stopNavigation.mockImplementation(() => {
    navigationUp = false;
    return Promise.resolve({ ok: true, message: 'stopped' });
  });
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'factory'));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  await act(async () => {
    latestMapViewerProps().onMapPose(0.6, 0.4, 0.1);
  });
  await waitFor(() => expect(screen.getByText('Ready')).toBeInTheDocument(), { timeout: 6000 });

  fireEvent.click(screen.getByRole('button', { name: 'Run Mission' }));
  await waitFor(() => expect(mockCallService).toHaveBeenCalledWith(
    '/bt/nodes/catalog',
    'interfaces/srv/GetNodeCatalog',
    {},
    1000,
  ));
  await waitFor(() => expect(sendNavigateToPoseGoalAndWait).toHaveBeenCalledTimes(1));
  expect(global.fetch.mock.calls.some(([url, options]) => (
    String(url) === '/api/services/bt_node/start'
    && options?.method === 'POST'
  ))).toBe(false);

  if (completesNaturally) {
    await waitFor(() => expect(mockCallService).toHaveBeenCalledWith(
      '/bt/load_and_run',
      'interfaces/srv/LoadAndRunTree',
      { tree_xml: filledBt },
      30000,
    ));
    await act(async () => {
      mockTopicDataByName['/bt/status'] = stringTopicMessage('running');
      rerender(<AutonomyStudioPage />);
      resolveBtLoad({ success: true });
      await Promise.resolve();
    });
    await act(async () => {
      mockTopicDataByName['/bt/status'] = stringTopicMessage('completed');
      rerender(<AutonomyStudioPage />);
    });
    await waitFor(() => expect(sendNavigateToPoseGoalAndWait).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run Mission' })).toBeEnabled());
  } else {
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
    await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Stop' })).toBeDisabled());
  }

  expect(global.fetch.mock.calls.filter(([url, options]) => (
    String(url) === '/api/services/bt_node/stop'
    && options?.method === 'POST'
  ))).toHaveLength(0);
}, 20000);

test('keeps Run localization active while the BT node is up', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  // A running mission can keep the BT node up while localization still uses
  // the map pose-set gesture.
  global.fetch.mockResolvedValue(mockJsonResponse({ name: 'bt_node', state: 'up', raw: 'up' }));
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMission.mockImplementation((mapName) => Promise.resolve(
    mapName === 'factory'
      ? {
        exists: true,
        map_name: 'factory',
        global_bt: 'global.xml',
        waypoints: [
          { id: 'wp1', label: 'Kitchen', pose: { frame_id: 'map', x: 1, y: 0, yaw: 0 }, local_bt: 'locals/wp1.xml', metadata: {} },
        ],
        metadata: {},
      }
      : { exists: false, map_name: mapName, global_bt: 'global.xml', waypoints: [], metadata: {} },
  ));

  render(<AutonomyStudioPage />);

  // Run observes the active BT service internally without exposing manual
  // lifecycle controls in Design.
  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const mapSelect = await screen.findByRole('combobox', { name: 'Run mission map file' });
  await waitFor(() => expect(mapSelect).toHaveValue('factory.pgm'));
  fireEvent.click(screen.getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(getMapAnnotations).toHaveBeenCalledWith('factory.pgm'));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  // Settle so the guard effect (which would reset "initial" in the buggy code)
  // has run; the pose-set mode must remain active.
  await act(async () => { await Promise.resolve(); });
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
}, 15000);


test('invalidates a Run mission snapshot when Navigate loads a different map', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [
      { path: 'map-a.pgm', name: 'map-a.pgm' },
      { path: 'map-b.pgm', name: 'map-b.pgm' },
    ],
  });
  getNavigationMissions.mockImplementation((mapName) => Promise.resolve({
    map_name: mapName,
    missions: mapName === 'map-a' ? ['mission-a'] : [],
  }));
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'map-a',
    mission_name: 'mission-a',
    revision: 11,
    global_bt: 'global.xml',
    waypoints: [
      {
        id: 'map_a_wp_1',
        label: 'Map A Start',
        pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
        local_bt: 'locals/map_a_wp_1/main.xml',
        metadata: {},
      },
      {
        id: 'map_a_wp_2',
        label: 'Map A Finish',
        pose: { frame_id: 'map', x: 3, y: 4, yaw: 0 },
        local_bt: 'locals/map_a_wp_2/main.xml',
        metadata: {},
      },
    ],
    metadata: {
      mission_flow: {
        nodes: [
          { id: 'map_a_wp_1', position: { x: 80, y: 72 } },
          { id: 'map_a_wp_2', position: { x: 300, y: 72 } },
        ],
        edges: [{ id: 'map_a_route', source: 'map_a_wp_1', target: 'map_a_wp_2' }],
      },
    },
  });
  getNavigationMissionBtFile.mockImplementation((_mapName, path) => Promise.resolve({
    path,
    content: `<root><BehaviorTree ID="${path}"/></root>`,
    exists: true,
    revision: 11,
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const runLoadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  expect(within(runLoadDialog).getByRole('combobox', {
    name: 'Run mission map file',
  })).toHaveValue('map-a.pgm');
  expect(await within(runLoadDialog).findByRole('combobox', {
    name: 'Run mission file',
  })).toHaveValue('mission-a');
  fireEvent.click(within(runLoadDialog).getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(latestMapViewerProps().spots).toHaveLength(2));
  expect(latestMapViewerProps().missionRouteOrder).toEqual([
    { id: 'map_a_wp_1', order: 1 },
    { id: 'map_a_wp_2', order: 2 },
  ]);
  expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled();
  const missionCatalogCalls = getNavigationMissions.mock.calls.length;
  const missionLoadCalls = getNavigationMission.mock.calls.length;
  const missionBtCalls = getNavigationMissionBtFile.mock.calls.length;

  fireEvent.click(screen.getByRole('tab', { name: 'Navigation' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const navigateLoadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  const navigateMapSelect = within(navigateLoadDialog).getByRole('combobox', {
    name: 'Navigation map file',
  });
  expect(within(navigateLoadDialog).getAllByRole('combobox')).toHaveLength(1);
  fireEvent.change(navigateMapSelect, { target: { value: 'map-b.pgm' } });
  fireEvent.click(within(navigateLoadDialog).getByRole('button', { name: 'Load' }));

  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('map-b.pgm'));
  expect(getNavigationMissions).toHaveBeenCalledTimes(missionCatalogCalls);
  expect(getNavigationMission).toHaveBeenCalledTimes(missionLoadCalls);
  expect(getNavigationMissionBtFile).toHaveBeenCalledTimes(missionBtCalls);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  await waitFor(() => expect(latestMapViewerProps().spots).toEqual([]));
  expect(latestMapViewerProps().missionRouteOrder).toEqual([]);
  expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Run Mission' })).toBeDisabled();
  expect(screen.getByRole('combobox', { name: 'Active mission' })).toBeDisabled();
});


test('loads a Navigate map without mission state while Run keeps its mission selector', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({
    files: [{ path: 'factory.pgm', name: 'factory.pgm' }],
  });
  getNavigationMissions.mockResolvedValue({
    map_name: 'factory',
    missions: ['inspection'],
  });
  getNavigationMission.mockResolvedValue({
    exists: true,
    map_name: 'factory',
    mission_name: 'inspection',
    global_bt: 'global.xml',
    waypoints: [{
      id: 'mission_only_waypoint',
      label: 'Mission Only',
      pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
      local_bt: 'locals/mission_only/main.xml',
      local_bt_files: ['locals/mission_only/main.xml'],
      metadata: {},
    }],
    metadata: {},
  });

  render(<AutonomyStudioPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Navigation' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));

  const navigateLoadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  const navigateSelectors = within(navigateLoadDialog).getAllByRole('combobox');
  expect(navigateSelectors).toHaveLength(1);
  expect(navigateSelectors[0]).toHaveValue('factory.pgm');
  expect(within(navigateLoadDialog).queryByRole('combobox', {
    name: 'Run mission file',
  })).not.toBeInTheDocument();
  expect(getNavigationMissions).not.toHaveBeenCalled();
  expect(getNavigationMission).not.toHaveBeenCalled();
  expect(getNavigationMissionBtFile).not.toHaveBeenCalled();

  fireEvent.click(within(navigateLoadDialog).getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('factory.pgm'));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());
  expect(getNavigationMissions).not.toHaveBeenCalled();
  expect(getNavigationMission).not.toHaveBeenCalled();
  expect(getNavigationMissionBtFile).not.toHaveBeenCalled();
  expect(latestMapViewerProps().spots).toEqual([]);
  expect(latestMapViewerProps().missionRouteOrder).toEqual([]);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const runLoadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  expect(within(runLoadDialog).getByRole('combobox', {
    name: 'Run mission map file',
  })).toHaveValue('factory.pgm');
  expect(await within(runLoadDialog).findByRole('combobox', {
    name: 'Run mission file',
  })).toHaveValue('inspection');
  expect(getNavigationMissions).toHaveBeenCalledWith('factory');
});


test('drives to a clicked goal from the Navigation stage', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  let navigationUp = false;
  getServiceStatus.mockImplementation(() => Promise.resolve(
    navigationUp ? { is_up: true, mode: 'nav' } : { is_up: false, mode: 'idle' },
  ));
  startNavigation.mockImplementation(() => {
    navigationUp = true;
    return Promise.resolve({ ok: true, message: 'started' });
  });
  stopNavigation.mockImplementation(() => {
    navigationUp = false;
    return Promise.resolve({ ok: true, message: 'stopped' });
  });
  mockTopicDataByName['/amcl_pose'] = amclPoseMessage(0, 0, 0);
  sendInitialPoseEstimate.mockImplementationOnce(async () => {
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(1, 2, 0.5);
    return { ok: true };
  });
  let resolveGoal;
  sendNavigateToPoseGoalAndWait.mockReturnValue(new Promise((resolve) => {
    resolveGoal = resolve;
  }));
  let resolveCancel;
  cancelNavigateToPoseGoal.mockReturnValueOnce(new Promise((resolve) => {
    resolveCancel = resolve;
  }));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Navigation' }));
  expect(screen.getByRole('button', { name: 'Set Goal' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled();

  // Navigate loads only a map, and the stage stays Navigate.
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const navigateLoadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  expect(within(navigateLoadDialog).getAllByRole('combobox')).toHaveLength(1);
  fireEvent.click(within(navigateLoadDialog).getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());
  expect(screen.getByRole('tab', { name: 'Navigation' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByRole('tab', { name: 'Run' })).toHaveAttribute('aria-selected', 'false');

  // Localize brings nav up and enters pose-set mode; clicking the map
  // converges AMCL and unlocks Set Goal.
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'factory'));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  await act(async () => {
    latestMapViewerProps().onMapPose(1, 2, 0.5);
  });
  await waitFor(
    () => expect(screen.getByRole('button', { name: 'Set Goal' })).toBeEnabled(),
    { timeout: 5000 },
  );

  // An accidental Localize re-click must not invalidate the pose: with the
  // runtime already up the first click re-arms pose-set mode, the second
  // disarms it, and Set Goal stays usable throughout.
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  expect(screen.getByRole('button', { name: 'Set Goal' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('view'));
  expect(screen.getByRole('button', { name: 'Set Goal' })).toBeEnabled();

  // Arm goal mode and click the target: nav2 receives a NavigateToPose goal.
  fireEvent.click(screen.getByRole('button', { name: 'Set Goal' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('goal'));
  await act(async () => {
    latestMapViewerProps().onMapPose(3, 4, 1.0);
  });
  await waitFor(() => expect(sendNavigateToPoseGoalAndWait).toHaveBeenCalled());
  const [goalPayload] = sendNavigateToPoseGoalAndWait.mock.calls[0];
  expect(goalPayload.pose.pose.position).toEqual({ x: 3, y: 4, z: 0 });
  expect(latestMapViewerProps().goalPose).not.toBeNull();
  expect(latestMapViewerProps().showGoalPose).toBe(true);
  expect(screen.getByText('Driving')).toBeInTheDocument();
  // Sending is one-shot: the map returns to view mode (scroll zoom works)
  // and the camera follows the robot for the duration of the drive.
  expect(latestMapViewerProps().interactionMode).toBe('view');
  expect(latestMapViewerProps().missionFollowRobot).toBe(true);

  // Even a same-map reload starts a fresh direct-goal lifecycle. Simulate the
  // supervisor status reaching idle before the old goal wait settles, reload,
  // and ensure that late completion cannot revive the old target.
  navigationUp = false;
  fireEvent(document, new Event('visibilitychange'));
  await waitFor(
    () => expect(screen.getByRole('button', { name: 'Load Map' })).toBeEnabled(),
    { timeout: 4000 },
  );
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const reloadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  fireEvent.click(within(reloadDialog).getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(latestMapViewerProps().goalPose).toBeNull());
  expect(screen.queryByText('Driving')).not.toBeInTheDocument();
  await act(async () => {
    resolveGoal({ ok: true, status: 'SUCCEEDED' });
  });
  expect(latestMapViewerProps().goalPose).toBeNull();
  expect(screen.queryByText('Goal reached')).not.toBeInTheDocument();

  // The transport can still be up despite a stale idle status sample. Restore
  // that status so the existing single-Stop shutdown contract remains tested.
  navigationUp = true;
  fireEvent(document, new Event('visibilitychange'));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled());

  // A single Stop cancels the active goal and shuts the navigation runtime
  // down. A second click must not be necessary.
  fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
  await waitFor(() => expect(cancelNavigateToPoseGoal).toHaveBeenCalled());
  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Set Goal' })).toBeDisabled();

  await act(async () => {
    resolveCancel({ ok: true });
  });
  await waitFor(() => expect(screen.queryByText('Driving')).not.toBeInTheDocument());
  expect(latestMapViewerProps().missionFollowRobot).toBe(false);
  expect(latestMapViewerProps().goalPose).toBeNull();
  expect(latestMapViewerProps().showGoalPose).toBe(false);
  expect(navigationUp).toBe(false);
  // Full shutdown unloads the live navigation map, so a fresh map load is
  // required before Localize can start another session.
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled());
  expect(screen.getByRole('button', { name: 'Stop' })).toBeDisabled();

  expect(screen.queryByText('Driving')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Stop' })).toBeDisabled();
}, 15000);


test('stops Navigation and clears its localization before switching to Map Edit', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  let navigationUp = false;
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getServiceStatus.mockImplementation(() => Promise.resolve(
    navigationUp ? { is_up: true, mode: 'nav' } : { is_up: false, mode: 'idle' },
  ));
  startNavigation.mockImplementation(() => {
    navigationUp = true;
    return Promise.resolve({ ok: true, message: 'started' });
  });
  stopNavigation.mockImplementation(() => {
    navigationUp = false;
    return Promise.resolve({ ok: true, message: 'stopped' });
  });
  mockTopicDataByName['/amcl_pose'] = amclPoseMessage(0, 0, 0);
  sendInitialPoseEstimate.mockImplementation(async () => {
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(1, 2, 0.5);
    return { ok: true };
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Navigation' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const loadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  fireEvent.click(within(loadDialog).getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());

  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  await act(async () => {
    latestMapViewerProps().onMapPose(1, 2, 0.5);
  });
  await waitFor(
    () => expect(screen.getByRole('button', { name: 'Set Goal' })).toBeEnabled(),
    { timeout: 5000 },
  );
  expect(navigationUp).toBe(true);

  stopNavigation.mockClear();
  expect(screen.getByRole('tab', { name: 'Map Edit' })).toBeEnabled();
  fireEvent.click(screen.getByRole('tab', { name: 'Map Edit' }));

  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(screen.getByRole('tab', { name: 'Map Edit' }))
    .toHaveAttribute('aria-selected', 'true'));
  expect(navigationUp).toBe(false);
  expect(latestMapViewerProps().interactionMode).toBe('view');
  expect(latestMapViewerProps().goalPose).toBeNull();
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')),
  ).toEqual(expect.objectContaining({
    workspaceStage: 'map_edit',
    navigationRuntimeMode: 'idle',
    runRuntimeOwned: false,
    runShutdownPending: false,
  })));

  await waitFor(() => expect(screen.getByRole('tab', { name: 'Navigation' })).toBeEnabled());
  fireEvent.click(screen.getByRole('tab', { name: 'Navigation' }));
  expect(screen.getByRole('button', { name: 'Set Goal' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Localize' })).toBeDisabled();
  expect(stopNavigation).toHaveBeenCalledTimes(1);
});


test('keeps Run selected when automatic stage-exit shutdown fails', async () => {
  let navigationUp = false;
  getServiceStatus.mockImplementation(() => Promise.resolve(
    navigationUp ? { is_up: true, mode: 'nav' } : { is_up: false, mode: 'idle' },
  ));
  startNavigation.mockImplementation(() => {
    navigationUp = true;
    return Promise.resolve({ ok: true, message: 'started' });
  });
  stopNavigation.mockRejectedValue(new Error('navigation stop failed'));

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Run' }));
  await loadRunMapFromDialog('factory.pgm');
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(startNavigation).toHaveBeenCalledWith('nav', 'factory'));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled());
  stopNavigation.mockClear();

  fireEvent.click(screen.getByRole('tab', { name: 'Design' }));

  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(screen.getByRole('tab', { name: 'Run' }))
    .toHaveAttribute('aria-selected', 'true'));
  expect(screen.getByRole('tab', { name: 'Design' }))
    .toHaveAttribute('aria-selected', 'false');
  expect(navigationUp).toBe(true);
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')),
  ).toEqual(expect.objectContaining({
    workspaceStage: 'run',
    navigationRuntimeMode: 'run',
    runRuntimeOwned: true,
    runShutdownPending: false,
  })));
});


test('reports a non-succeeded navigate goal as failed', async () => {
  const latestMapViewerProps = () => (
    mockMapViewer.mock.calls[mockMapViewer.mock.calls.length - 1][0]
  );
  getPgmFiles.mockResolvedValue({ files: [{ path: 'factory.pgm', name: 'factory.pgm' }] });
  getServiceStatus
    .mockResolvedValueOnce({ is_up: false })
    .mockResolvedValue({ is_up: true, mode: 'nav' });
  mockTopicDataByName['/amcl_pose'] = amclPoseMessage(0, 0, 0);
  sendInitialPoseEstimate.mockImplementationOnce(async () => {
    mockTopicDataByName['/amcl_pose'] = amclPoseMessage(1, 2, 0.5);
    return { ok: true };
  });
  // nav2 reports non-success in-band: HTTP 200 with ok:false + a status.
  sendNavigateToPoseGoalAndWait.mockResolvedValue({
    ok: false, status: 'ABORTED', message: 'Goal aborted by nav2',
  });

  render(<AutonomyStudioPage />);

  fireEvent.click(screen.getByRole('tab', { name: 'Navigation' }));
  fireEvent.click(screen.getByRole('button', { name: 'Load Map' }));
  const navigateLoadDialog = await screen.findByRole('dialog', { name: 'Load Map' });
  expect(within(navigateLoadDialog).getAllByRole('combobox')).toHaveLength(1);
  fireEvent.click(within(navigateLoadDialog).getByRole('button', { name: 'Load' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Localize' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Localize' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('initial'));
  await act(async () => {
    latestMapViewerProps().onMapPose(1, 2, 0.5);
  });
  await waitFor(
    () => expect(screen.getByRole('button', { name: 'Set Goal' })).toBeEnabled(),
    { timeout: 5000 },
  );

  fireEvent.click(screen.getByRole('button', { name: 'Set Goal' }));
  await waitFor(() => expect(latestMapViewerProps().interactionMode).toBe('goal'));
  await act(async () => {
    latestMapViewerProps().onMapPose(3, 4, 1.0);
  });

  await waitFor(() => expect(screen.getByText('Failed')).toBeInTheDocument());
  // The intended target stays visible so the operator can see what failed.
  expect(latestMapViewerProps().showGoalPose).toBe(true);
});


test('restores a session saved with the legacy standalone_bt workspace kind', () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'standalone_bt',
    workspaceStage: 'authoring',
  }));

  render(<AutonomyStudioPage />);

  expect(screen.getByTestId('action-canvas-workspace')).toBeInTheDocument();
  expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Open Action Canvas' })).not.toBeInTheDocument();
});

test('confirms a pending Run shutdown when an Action Canvas session is restored', async () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'action_canvas',
    workspaceStage: 'run',
    navigationRuntimeMode: 'idle',
    runRuntimeOwned: true,
    runShutdownPending: true,
    runShutdownRequestedAt: Date.now(),
  }));

  render(<AutonomyStudioPage />);

  expect(screen.getByTestId('action-canvas-workspace')).toBeInTheDocument();
  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')),
  ).toEqual(expect.objectContaining({
    workspaceKind: 'action_canvas',
    navigationRuntimeMode: 'idle',
    runRuntimeOwned: false,
    runShutdownPending: false,
    runShutdownRequestedAt: null,
  })));
  expect(getServiceStatus).not.toHaveBeenCalled();
});

test('stops an owned Run runtime when the page exits from the Action Canvas', () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'action_canvas',
    workspaceStage: 'run',
    navigationRuntimeMode: 'run',
    runRuntimeOwned: true,
    runShutdownPending: false,
  }));

  render(<AutonomyStudioPage />);
  expect(screen.getByTestId('action-canvas-workspace')).toBeInTheDocument();
  fireEvent(window, new Event('pagehide'));

  expect(stopNavigation).toHaveBeenCalledWith(expect.objectContaining({ keepalive: true }));
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session'))).toEqual(
    expect.objectContaining({
      workspaceKind: 'action_canvas',
      navigationRuntimeMode: 'idle',
      runRuntimeOwned: true,
      runShutdownPending: true,
    }),
  );
});

test('does not stop a Run runtime the Action Canvas session does not own on page exit', () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'action_canvas',
    navigationRuntimeMode: 'idle',
    runRuntimeOwned: false,
    runShutdownPending: false,
  }));

  render(<AutonomyStudioPage />);
  fireEvent(window, new Event('pagehide'));

  expect(stopNavigation).not.toHaveBeenCalled();
});

test('does not apply an expired Run shutdown marker to a later runtime from the Action Canvas', async () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'action_canvas',
    workspaceStage: 'run',
    navigationRuntimeMode: 'idle',
    runRuntimeOwned: true,
    runShutdownPending: true,
    runShutdownRequestedAt: Date.now() - 120_000,
  }));

  render(<AutonomyStudioPage />);
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')),
  ).toEqual(expect.objectContaining({
    runRuntimeOwned: false,
    runShutdownPending: false,
    runShutdownRequestedAt: null,
  })));
  fireEvent(window, new Event('pagehide'));

  expect(stopNavigation).not.toHaveBeenCalled();
});

test('keeps the retry marker when the Action Canvas shutdown confirmation fails', async () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'action_canvas',
    workspaceStage: 'run',
    navigationRuntimeMode: 'run',
  }));
  stopNavigation.mockImplementationOnce(() => Promise.reject(new Error('supervisor down')));

  render(<AutonomyStudioPage />);
  await waitFor(() => expect(stopNavigation).toHaveBeenCalledTimes(1));
  await act(async () => { await Promise.resolve(); });

  // A pre-ownership Run session gains ownership and a fresh marker on mount, so
  // the failed confirmation is retried on page exit.
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session'))).toEqual(
    expect.objectContaining({ runRuntimeOwned: true, runShutdownPending: true }),
  );
  fireEvent(window, new Event('pagehide'));
  expect(stopNavigation).toHaveBeenCalledTimes(2);
  expect(stopNavigation).toHaveBeenLastCalledWith(expect.objectContaining({ keepalive: true }));
});

test('sends the Action Canvas page-exit stop once even if pagehide is delivered twice', () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'action_canvas',
    navigationRuntimeMode: 'run',
    runRuntimeOwned: true,
    runShutdownPending: false,
  }));

  render(<AutonomyStudioPage />);
  fireEvent(window, new Event('pagehide'));
  const requestedAt = JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')).runShutdownRequestedAt;
  fireEvent(window, new Event('pagehide'));

  expect(stopNavigation).toHaveBeenCalledTimes(1);
  expect(JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')).runShutdownRequestedAt)
    .toBe(requestedAt);
});

test('confirms a pending Run shutdown once under StrictMode from the Action Canvas', async () => {
  window.sessionStorage.setItem('autonomy_studio_session', JSON.stringify({
    workspaceKind: 'action_canvas',
    workspaceStage: 'run',
    navigationRuntimeMode: 'idle',
    runRuntimeOwned: true,
    runShutdownPending: true,
    runShutdownRequestedAt: Date.now(),
  }));

  render(<StrictMode><AutonomyStudioPage /></StrictMode>);
  await waitFor(() => expect(
    JSON.parse(window.sessionStorage.getItem('autonomy_studio_session')).runShutdownPending,
  ).toBe(false));

  expect(stopNavigation).toHaveBeenCalledTimes(1);
});

test('is light-only: suspends the global dark theme class while mounted and restores it on leave', async () => {
  const root = document.documentElement;
  root.classList.add('dark');
  root.setAttribute('data-theme', 'dark');
  try {
    const view = render(<AutonomyStudioPage />);
    expect(root.classList.contains('dark')).toBe(false);

    // The theme provider re-applying its class (e.g. on its own mount effect)
    // must not turn the studio dark.
    act(() => { root.classList.add('dark'); });
    await waitFor(() => expect(root.classList.contains('dark')).toBe(false));

    view.unmount();
    expect(root.classList.contains('dark')).toBe(true);
  } finally {
    root.classList.remove('dark');
    root.removeAttribute('data-theme');
  }
});
