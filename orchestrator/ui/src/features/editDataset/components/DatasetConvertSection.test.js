import { configureStore } from '@reduxjs/toolkit';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import editDatasetReducer from '../editDatasetSlice';
import taskReducer from '../../tasks/taskSlice';
import DatasetConvertSection from './DatasetConvertSection';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';

jest.mock('../../../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('../../../components/FileBrowserModal', () => () => null);

const renderSection = () => {
  const store = configureStore({
    reducer: { tasks: taskReducer, editDataset: editDatasetReducer },
  });
  return render(
    <Provider store={store}>
      <DatasetConvertSection />
    </Provider>
  );
};

test('keeps dataset conversion controls without exposing status cleanup', () => {
  useRosServiceCaller.mockReturnValue({
    sendRecordCommand: jest.fn(),
  });

  renderSection();

  expect(screen.getByPlaceholderText(/Task_1_1_MCAP/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Convert Dataset' }))
    .toBeInTheDocument();
  expect(screen.queryByText('Data Status & Cleanup')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Delete Oldest' }))
    .not.toBeInTheDocument();
});
