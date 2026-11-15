import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import AutonomyStudioPage from './AutonomyStudioPage';
import uiReducer, { CURRENT_PAGE_STORAGE_KEY, resolveInitialPageState } from '../features/ui/uiSlice';
import { MISSION_SESSION_STORAGE_KEY } from '../features/mapWorkspace/lib/session';
import { shouldApplyServerTaskInfoToPage } from '../utils/taskInfoSync';
import PageType from '../constants/pageType';

let mockRlProps;
jest.mock('../features/offlineRL/components/RLWorkflowLayout', () => (props) => {
  mockRlProps = props;
  return <div>RL environment and training</div>;
});
jest.mock('../components/navigation/ActionCanvasWorkspace', () => () => <div>Action workspace</div>);
jest.mock('../features/missionCanvas/MissionCanvasWorkspace', () => () => <div>Mission workspace</div>);
jest.mock('../features/autonomyCanvas/AutonomyCanvasWorkspace', () => () => <div>Autonomy workspace</div>);
jest.mock('../features/mapWorkspace/hooks/useStaleRunShutdownRecovery', () => () => {});

function renderStudio(props = {}) {
  const store = configureStore({ reducer: { ui: uiReducer } });
  const view = render(<Provider store={store}><AutonomyStudioPage {...props} /></Provider>);
  return { ...view, store };
}

beforeEach(() => { window.sessionStorage.clear(); mockRlProps = null; });

test('opens RL Canvas beneath the shared Studio app bar and returns to the chooser', () => {
  const { store } = renderStudio({ showWorkspaceChooser: true });
  expect(screen.getByRole('button', { name: 'Open Action Canvas' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open Mission Canvas' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open Autonomy Canvas' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Open RL Canvas' }));
  expect(screen.getByRole('heading', { name: 'Autonomy Studio' })).toBeInTheDocument();
  expect(screen.getByRole('main', { name: 'RL Canvas workspace' })).toBeInTheDocument();
  expect(mockRlProps.embedded).toBe(true);
  expect(store.getState().ui.activeStudioWorkspace).toBe('rl_canvas');
  expect(JSON.parse(sessionStorage.getItem(MISSION_SESSION_STORAGE_KEY)).workspaceKind).toBe('rl_canvas');
  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));
  expect(screen.getByRole('button', { name: 'Open RL Canvas' })).toBeInTheDocument();
  expect(store.getState().ui.activeStudioWorkspace).toBe('');
});

test('restores RL Canvas and prevents leaving while inference owns the robot', () => {
  sessionStorage.setItem(MISSION_SESSION_STORAGE_KEY, JSON.stringify({ workspaceKind: 'rl_canvas' }));
  renderStudio();
  expect(screen.getByText('RL environment and training')).toBeInTheDocument();
  act(() => { mockRlProps.onExitStateChange({ active: true, busy: false }); });
  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));
  expect(screen.getByRole('status')).toHaveTextContent('Stop the active runtime');
  expect(screen.queryByRole('button', { name: 'Open RL Canvas' })).not.toBeInTheDocument();
  act(() => { mockRlProps.onExitStateChange({ active: false, busy: false }); });
  fireEvent.click(screen.getByRole('button', { name: 'Back to workspace chooser' }));
  expect(screen.getByRole('button', { name: 'Open RL Canvas' })).toBeInTheDocument();
});

test('migrates saved Playground sessions into RL Canvas', () => {
  sessionStorage.setItem(CURRENT_PAGE_STORAGE_KEY, 'offline_rl');
  expect(resolveInitialPageState()).toEqual({ currentPage: PageType.AUTONOMY_STUDIO, restoredPageFromSession: true });
  expect(JSON.parse(sessionStorage.getItem(MISSION_SESSION_STORAGE_KEY)).workspaceKind).toBe('rl_canvas');
});

test('accepts task-info echoes only for the inference workspace inside Studio', () => {
  expect(shouldApplyServerTaskInfoToPage({ currentPage: PageType.AUTONOMY_STUDIO, workspaceKind: 'rl_canvas' })).toBe(true);
  expect(shouldApplyServerTaskInfoToPage({ currentPage: PageType.AUTONOMY_STUDIO, workspaceKind: 'action_canvas' })).toBe(false);
  expect(shouldApplyServerTaskInfoToPage({ currentPage: PageType.AUTONOMY_STUDIO, workspaceKind: 'autonomy' })).toBe(false);
});
