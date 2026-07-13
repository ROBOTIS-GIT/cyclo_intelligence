import { render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import NavigationPage from './NavigationPage';
import store from '../store/store';
import { getServiceStatus } from '../utils/navigationApi';

jest.mock('../components/navigation/MapViewer', () => ({
  MapViewer: () => <div>Navigation Map Viewer</div>,
}));

jest.mock('../utils/navigationApi', () => ({
  cancelNavigateToPoseGoal: jest.fn().mockResolvedValue({ ok: true }),
  getServiceStatus: jest.fn().mockResolvedValue({ is_up: false }),
  saveNavigationMap: jest.fn().mockResolvedValue({ ok: true }),
  sendNavigateToPoseGoal: jest.fn().mockResolvedValue({ ok: true }),
  startNavigation: jest.fn().mockResolvedValue({ ok: true }),
  stopNavigation: jest.fn().mockResolvedValue({ ok: true }),
}));

jest.mock('../hooks/useNavigationRosTopic', () => ({
  useNavigationRosPublisher: () => jest.fn(),
  useNavigationRosTopic: () => ({ status: 'disconnected', topicData: null }),
}));

test('renders the Navigation page shell', async () => {
  render(
    <Provider store={store}>
      <NavigationPage />
    </Provider>
  );

  expect(screen.getByRole('heading', { name: 'Navigation' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Mapping' })).toBeInTheDocument();
  expect(screen.getByText('Navigation Map Viewer')).toBeInTheDocument();
  await waitFor(() => expect(getServiceStatus).toHaveBeenCalled());
});
