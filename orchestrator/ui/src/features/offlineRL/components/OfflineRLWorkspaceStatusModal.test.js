import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import taskReducer from '../../tasks/taskSlice';
import OfflineRLWorkspaceStatusModal from './OfflineRLWorkspaceStatusModal';

jest.mock('../../../components/InlineSystemStatus', () => {
  return function MockInlineSystemStatus() {
    return <div data-testid="inline-system-status">CPU RAM Storage</div>;
  };
});

jest.mock('../../../components/RecordTopicMonitor', () => {
  return function MockRecordTopicMonitor({ showWhenEmpty }) {
    return (
      <div
        data-testid="record-topic-monitor"
        data-show-when-empty={String(showWhenEmpty)}
      >
        Topic Monitor
        <button type="button">Refresh</button>
      </div>
    );
  };
});

const renderModal = (props = {}) => {
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        robotType: 'ffw_sg2_rev1',
      },
    },
  });
  const onClose = props.onClose || jest.fn();
  const view = render(
    <Provider store={store}>
      <OfflineRLWorkspaceStatusModal
        isOpen={props.isOpen ?? true}
        onClose={onClose}
        workspaceMode={props.workspaceMode || 'inference'}
        settingsContent={props.settingsContent || (
          <div data-testid="workspace-settings-content">Settings content</div>
        )}
      />
    </Provider>
  );
  return { ...view, onClose };
};

describe('OfflineRLWorkspaceStatusModal', () => {
  test('does not render while closed', () => {
    renderModal({ isOpen: false });

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('reuses robot, system, and topic status with empty-topic refresh visible', () => {
    renderModal();

    expect(screen.getByRole('dialog', {
      name: 'Inference Workspace Status',
    })).toBeInTheDocument();
    expect(screen.getByRole('dialog')
      .querySelector('[data-robot-lab-icon="true"]')).toBeInTheDocument();
    expect(screen.getByText('ffw_sg2_rev1')).toBeInTheDocument();
    expect(screen.getByTestId('inline-system-status')).toHaveTextContent(
      'CPU RAM Storage'
    );
    expect(screen.getByTestId('record-topic-monitor'))
      .toHaveAttribute('data-show-when-empty', 'true');
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Robot Status' }))
      .toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Settings' }))
      .toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('button', {
      name: 'Back to Offline RL workspace',
    })).toHaveFocus();
  });

  test('shows mode-specific settings in the Settings tab', () => {
    renderModal({
      workspaceMode: 'recording',
      settingsContent: (
        <div data-testid="recording-settings-content">Recording controls</div>
      ),
    });

    expect(screen.getByRole('dialog', {
      name: 'Recording Workspace Status',
    })).toBeInTheDocument();
    expect(screen.getByTestId('recording-settings-content')).not.toBeVisible();

    fireEvent.click(screen.getByRole('tab', { name: 'Settings' }));

    expect(screen.getByRole('tab', { name: 'Settings' }))
      .toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText('Recording Settings')).toBeInTheDocument();
    expect(screen.getByTestId('recording-settings-content'))
      .toHaveTextContent('Recording controls');
    expect(screen.getByTestId('record-topic-monitor')).not.toBeVisible();
  });

  test('closes with Back and Escape', () => {
    const { onClose } = renderModal();

    fireEvent.click(screen.getByRole('button', {
      name: 'Back to Offline RL workspace',
    }));
    fireEvent.keyDown(window, { key: 'Escape' });

    expect(onClose).toHaveBeenCalledTimes(2);
  });

  test('closes only when the backdrop itself is pressed', () => {
    const { onClose } = renderModal();
    const backdrop = screen.getByTestId('offline-rl-workspace-status-backdrop');

    fireEvent.mouseDown(screen.getByRole('dialog'));
    expect(onClose).not.toHaveBeenCalled();

    fireEvent.mouseDown(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
