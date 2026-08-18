import { fireEvent, render, screen } from '@testing-library/react';
import OfflineRLPage from './OfflineRLPage';

jest.mock('../features/editDataset/components/DatasetConvertSection', () => {
  return function MockDatasetConvertSection({ isEditable, onBusyChange }) {
    return (
      <div>
        Convert Dataset Section: {isEditable ? 'editable' : 'locked'}
        <button type="button" onClick={() => onBusyChange(true)}>
          Simulate dataset operation
        </button>
        <button type="button" onClick={() => onBusyChange(false)}>
          Finish dataset operation
        </button>
      </div>
    );
  };
});

jest.mock('../features/offlineRL/components/OfflineRLTrainingSection', () => {
  return function MockOfflineRLTrainingSection({ isActive, onRunningChange }) {
    return (
      <div>
        Offline RL Training Section: {isActive ? 'active' : 'locked'}
        <button type="button" onClick={() => onRunningChange(false)}>
          Report status ready
        </button>
        <button type="button" onClick={() => onRunningChange(true)}>
          Simulate training
        </button>
      </div>
    );
  };
});

test('places dataset conversion before offline RL training', () => {
  const { container } = render(<OfflineRLPage />);

  expect(screen.getByText(/Convert Dataset Section/)).toBeInTheDocument();
  expect(screen.getByText(/Offline RL Training Section/)).toBeInTheDocument();
  expect(container.textContent.indexOf('Convert Dataset Section')).toBeLessThan(
    container.textContent.indexOf('Offline RL Training Section')
  );
});

test('keeps conversion locked until status is ready and while training', () => {
  render(<OfflineRLPage />);

  expect(screen.getByText('Convert Dataset Section: locked')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Report status ready' }));
  expect(screen.getByText('Convert Dataset Section: editable')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Simulate training' }));
  expect(screen.getByText('Convert Dataset Section: locked')).toBeInTheDocument();
});

test('locks training while a conversion or data deletion is active', () => {
  render(<OfflineRLPage />);

  fireEvent.click(screen.getByRole('button', { name: 'Report status ready' }));
  expect(screen.getByText('Offline RL Training Section: active')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Simulate dataset operation' }));
  expect(screen.getByText('Offline RL Training Section: locked')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Finish dataset operation' }));
  expect(screen.getByText('Offline RL Training Section: active')).toBeInTheDocument();
});
