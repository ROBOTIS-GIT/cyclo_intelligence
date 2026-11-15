import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import useRLTDataValidation from './useRLTDataValidation';
import { validateRLTStage2Data, getRLTStage2DataValidation } from '../../utils/offlineRlApi';

jest.mock('../../utils/offlineRlApi', () => ({
  validateRLTStage2Data: jest.fn(),
  getRLTStage2DataValidation: jest.fn(),
}));

function Probe({ dataset, enabled = true }) {
  const state = useRLTDataValidation({ dataset_paths: [dataset] }, enabled);
  return (
    <div>
      <button type="button" onClick={state.run}>validate</button>
      <span data-testid="status">{state.result?.status || 'idle'}</span>
      <span data-testid="ready">{String(state.ready)}</span>
      <span data-testid="busy">{String(state.busy)}</span>
      <span data-testid="attempted">{String(state.attempted)}</span>
      <span data-testid="message">{state.result?.message || ''}</span>
    </div>
  );
}

describe('useRLTDataValidation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('starts validation once and exposes a ready result', async () => {
    validateRLTStage2Data.mockResolvedValue({ status: 'ready', message: 'ready' });
    render(<Probe dataset="/workspace/lerobot/a" />);

    fireEvent.click(screen.getByRole('button', { name: 'validate' }));
    expect(screen.getByTestId('attempted')).toHaveTextContent('true');
    expect(screen.getByTestId('busy')).toHaveTextContent('true');
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('ready'));
    expect(screen.getByTestId('ready')).toHaveTextContent('true');
    expect(validateRLTStage2Data).toHaveBeenCalledWith({
      dataset_paths: ['/workspace/lerobot/a'],
    });
  });

  test('does not run while disabled or while a request is pending', async () => {
    let resolve;
    validateRLTStage2Data.mockImplementation(() => new Promise((done) => { resolve = done; }));
    const { rerender } = render(<Probe dataset="/workspace/lerobot/a" enabled={false} />);
    fireEvent.click(screen.getByRole('button', { name: 'validate' }));
    expect(validateRLTStage2Data).not.toHaveBeenCalled();
    rerender(<Probe dataset="/workspace/lerobot/a" />);
    fireEvent.click(screen.getByRole('button', { name: 'validate' }));
    fireEvent.click(screen.getByRole('button', { name: 'validate' }));
    expect(validateRLTStage2Data).toHaveBeenCalledTimes(1);
    await act(async () => { resolve({ status: 'ready' }); });
  });

  test('polls a running validation until it completes', async () => {
    jest.useFakeTimers();
    validateRLTStage2Data.mockResolvedValue({ status: 'running', validation_id: 'validation-1' });
    getRLTStage2DataValidation.mockResolvedValue({ status: 'ready', validation_id: 'validation-1' });
    render(<Probe dataset="/workspace/lerobot/a" />);
    fireEvent.click(screen.getByRole('button', { name: 'validate' }));
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('running'));
    await act(async () => { jest.advanceTimersByTime(1500); await Promise.resolve(); });
    expect(getRLTStage2DataValidation).toHaveBeenCalledWith('validation-1');
    expect(screen.getByTestId('status')).toHaveTextContent('ready');
    jest.useRealTimers();
  });

  test('blocks on API errors and discards stale dataset results', async () => {
    validateRLTStage2Data.mockRejectedValueOnce(new Error('invalid replay'));
    const { rerender } = render(<Probe dataset="/workspace/lerobot/a" />);
    fireEvent.click(screen.getByRole('button', { name: 'validate' }));
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('blocked'));
    expect(screen.getByTestId('message')).toHaveTextContent('invalid replay');
    rerender(<Probe dataset="/workspace/lerobot/b" />);
    expect(screen.getByTestId('status')).toHaveTextContent('idle');
    expect(screen.getByTestId('ready')).toHaveTextContent('false');
  });
});
