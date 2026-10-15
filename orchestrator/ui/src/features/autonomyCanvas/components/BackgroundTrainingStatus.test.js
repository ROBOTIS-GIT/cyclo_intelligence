import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import BackgroundTrainingStatus from "./BackgroundTrainingStatus";
import StageRail from '../../mapWorkspace/components/workspace/StageRail';
import { collectionRequest } from '../../../utils/collectionApi';

jest.mock('../../../utils/collectionApi', () => ({ collectionRequest: jest.fn() }));

test('Qwen training remains visible while Operate stays selectable', async () => {
  collectionRequest.mockImplementation(async path => path === 'qwen/workflow'
    ? { reasoning_training: { id:'job1', status:'training', model_name:'Home 2B', step:210, steps:1000 } }
    : { training: { status:'idle' } });
  const onSelectStage = jest.fn();
  render(<StageRail workspaceKind="autonomy" workspaceStage="learn" onSelectStage={onSelectStage} footer={<BackgroundTrainingStatus />} />);
  const status = await screen.findByRole('status', { name:'Background training' });
  expect(status).toHaveTextContent('Qwen · Home 2B');
  expect(status).toHaveTextContent('210/1000');
  expect(screen.getByRole('progressbar', { name:'Qwen training progress' })).toHaveAttribute('value', '210');
  const operate = screen.getByRole('tab', { name:'Operate' });
  expect(operate).toBeEnabled();
  fireEvent.click(operate);
  expect(onSelectStage).toHaveBeenCalled();
  await waitFor(() => expect(collectionRequest).toHaveBeenCalledWith('segmentation/workflow'));
});
