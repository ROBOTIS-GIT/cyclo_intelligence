import { act, render, screen, waitFor, within } from '@testing-library/react';
import { Provider } from 'react-redux';
import toast from 'react-hot-toast';
import App from './App';
import { ThemeProvider } from './contexts/ThemeContext';
import store from './store/store';
import PageType from './constants/pageType';
import { CURRENT_PAGE_STORAGE_KEY, moveToPage } from './features/ui/uiSlice';
import { selectRobotType } from './features/tasks/taskSlice';

const mockAutonomyStudioPage = jest.fn();

jest.mock('react-hot-toast', () => {
  const React = require('react');
  return {
    __esModule: true,
    default: {
      dismiss: jest.fn(),
      error: jest.fn(),
    },
    Toaster: () => React.createElement('div', { 'data-testid': 'toaster' }),
  };
});

jest.mock('./components/ThemeToggle', () => {
  const React = require('react');
  return function MockThemeToggle() {
    return React.createElement('button', { type: 'button' }, 'Theme');
  };
});

jest.mock('./pages/HomePage', () => {
  const React = require('react');
  return function MockHomePage() {
    return React.createElement('div', null, 'Home Page');
  };
});

jest.mock('./pages/RecordPage', () => {
  const React = require('react');
  return function MockRecordPage() {
    return React.createElement('div', null, 'Record Page');
  };
});

jest.mock('./pages/InferencePage', () => {
  const React = require('react');
  return function MockInferencePage() {
    return React.createElement('div', null, 'Inference Page');
  };
});

jest.mock('./pages/TrainingPage', () => {
  const React = require('react');
  return function MockTrainingPage() {
    return React.createElement('div', null, 'Training Page');
  };
});

jest.mock('./pages/EditDatasetPage', () => {
  const React = require('react');
  return function MockEditDatasetPage() {
    return React.createElement('div', null, 'Edit Dataset Page');
  };
});

jest.mock('./pages/ReplayPage', () => {
  const React = require('react');
  return function MockReplayPage() {
    return React.createElement('div', null, 'Replay Page');
  };
});

jest.mock('./pages/AutonomyStudioPage', () => {
  const React = require('react');
  return function MockAutonomyStudioPage(props) {
    mockAutonomyStudioPage(props);
    return React.createElement(
      'div',
      { 'data-testid': 'autonomy-studio-page' },
      React.createElement('span', null, 'Mission Canvas Page'),
      React.createElement(
        'button',
        { type: 'button', onClick: props.onBackHome },
        'Back to Home'
      )
    );
  };
});

jest.mock('./hooks/useRosTopicSubscription', () => ({
  useRosTopicSubscription: () => ({
    initializeSubscriptions: jest.fn(),
  }),
}));

jest.mock('./utils/rosConnectionManager', () => ({
  __esModule: true,
  default: {
    setOnConnected: jest.fn(),
    disconnect: jest.fn(),
  },
}));

test('renders the Cyclo Intelligence shell navigation for regular pages', () => {
  render(
    <Provider store={store}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </Provider>
  );

  expect(screen.getByRole('button', { name: 'Cyclo Intelligence' }))
    .toBeInTheDocument();
  expect(screen.getByLabelText('Cyclo Intelligence navigation'))
    .toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Inference/i }))
    .toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Autonomy Studio' }))
    .toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'BT Manager' }))
    .not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Nav' }))
    .not.toBeInTheDocument();
  expect(screen.getByText('Home Page')).toBeInTheDocument();
});

test('orders the Cyclo Intelligence navigation into workflow sections', () => {
  render(
    <Provider store={store}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </Provider>
  );

  const navigation = within(screen.getByLabelText('Cyclo Intelligence navigation'));
  const pageButtons = [
    'Home',
    'Record',
    'Data Tools',
    'Training Guide',
    'Inference',
    'Autonomy Studio',
  ].map((name) => navigation.getByRole('button', { name }));

  pageButtons.slice(1).forEach((button, index) => {
    expect(pageButtons[index].compareDocumentPosition(button))
      .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  const separator = navigation.getByRole('separator', { name: 'Navigation sections' });
  expect(navigation.getByRole('button', { name: 'Inference' }).nextElementSibling)
    .toBe(separator);
  expect(separator.nextElementSibling)
    .toBe(navigation.getByRole('button', { name: 'Autonomy Studio' }));
  expect(navigation.queryByRole('button', { name: 'Nav' }))
    .not.toBeInTheDocument();
  expect(navigation.queryByRole('button', { name: 'BT Manager' }))
    .not.toBeInTheDocument();
});

test.each([
  [
    'no robot type is selected',
    '',
    'Please select a robot type first in the Home page',
  ],
  [
    'an unsupported robot type is selected',
    'ffw_sh5_rev1',
    'Autonomy Studio currently supports only ffw_sg2_rev1. Current robot type: ffw_sh5_rev1',
  ],
])('blocks Autonomy Studio when %s', async (_scenario, robotType, message) => {
  window.sessionStorage.clear();
  mockAutonomyStudioPage.mockClear();
  toast.error.mockClear();
  act(() => {
    store.dispatch(moveToPage(PageType.HOME));
    store.dispatch(selectRobotType(robotType));
  });

  const view = render(
    <Provider store={store}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </Provider>
  );

  act(() => {
    screen.getByRole('button', { name: 'Autonomy Studio' }).click();
  });

  await waitFor(() => {
    expect(toast.error).toHaveBeenCalled();
  });
  expect(toast.error.mock.calls.at(-1)[0]).toBe(message);
  expect(store.getState().ui.currentPage).toBe(PageType.HOME);
  expect(screen.getByText('Home Page')).toBeInTheDocument();
  expect(screen.getByLabelText('Cyclo Intelligence navigation'))
    .toBeInTheDocument();
  expect(screen.queryByTestId('autonomy-studio-page')).not.toBeInTheDocument();
  expect(mockAutonomyStudioPage).not.toHaveBeenCalled();

  view.unmount();
  act(() => {
    store.dispatch(selectRobotType(''));
  });
  window.sessionStorage.clear();
});

test('opens the Autonomy Studio workspace chooser from the canonical navigation entry point', async () => {
  window.sessionStorage.clear();
  mockAutonomyStudioPage.mockClear();
  toast.error.mockClear();
  act(() => {
    store.dispatch(moveToPage(PageType.HOME));
    store.dispatch(selectRobotType('ffw_sg2_rev1'));
  });

  const view = render(
    <Provider store={store}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </Provider>
  );

  act(() => {
    screen.getByRole('button', { name: 'Autonomy Studio' }).click();
  });

  expect(await screen.findByTestId('autonomy-studio-page'))
    .toHaveTextContent('Mission Canvas Page');
  expect(screen.queryByLabelText('Cyclo Intelligence navigation'))
    .not.toBeInTheDocument();
  expect(mockAutonomyStudioPage).toHaveBeenLastCalledWith(
    expect.objectContaining({
      onBackHome: expect.any(Function),
      showWorkspaceChooser: true,
    })
  );
  expect(toast.error).not.toHaveBeenCalled();
  await waitFor(() => {
    expect(window.sessionStorage.getItem(CURRENT_PAGE_STORAGE_KEY))
      .toBe(PageType.AUTONOMY_STUDIO);
    expect(store.getState().ui.currentPage).toBe(PageType.AUTONOMY_STUDIO);
  });

  act(() => {
    screen.getByRole('button', { name: 'Back to Home' }).click();
  });

  expect(await screen.findByText('Home Page')).toBeInTheDocument();
  expect(screen.getByLabelText('Cyclo Intelligence navigation'))
    .toBeInTheDocument();
  await waitFor(() => {
    expect(window.sessionStorage.getItem(CURRENT_PAGE_STORAGE_KEY))
      .toBe(PageType.HOME);
    expect(store.getState().ui.currentPage).toBe(PageType.HOME);
  });
  view.unmount();
  act(() => {
    store.dispatch(moveToPage(PageType.HOME));
    store.dispatch(selectRobotType(''));
  });
  window.sessionStorage.clear();
});
