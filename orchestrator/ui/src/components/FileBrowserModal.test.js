import { fireEvent, render, screen } from '@testing-library/react';
import FileBrowserModal from './FileBrowserModal';

jest.mock('./FileBrowser', () => ({ initialPath, onDirectorySelect, onPathChange }) => (
  <div data-testid="browser" data-path={initialPath}>
    <button onClick={() => onDirectorySelect({ name:'checkpoint', full_path:initialPath+'/checkpoint', is_directory:true })}>Choose checkpoint</button>
    <button onClick={() => onPathChange(initialPath+'/nested')}>Enter nested folder</button>
  </div>
));

test('workspace shortcuts navigate and clear a previous selection before confirming the new checkpoint', () => {
  const onFileSelect = jest.fn();
  render(<FileBrowserModal isOpen initialPath="/workspace/model/lerobot" allowDirectorySelect allowFileSelect={false}
    onClose={jest.fn()} onFileSelect={onFileSelect}
    shortcuts={[{ label:'hero_act_inference', path:'/workspace/model/lerobot/hero_act_inference' }]} />);
  fireEvent.click(screen.getByRole('button', { name:'Choose checkpoint' }));
  expect(screen.getByText('Selected:')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name:'hero_act_inference' }));
  expect(screen.getByTestId('browser')).toHaveAttribute('data-path','/workspace/model/lerobot/hero_act_inference');
  expect(screen.queryByText('Selected:')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name:'Choose checkpoint' }));
  fireEvent.click(screen.getByRole('button', { name:'Select' }));
  expect(onFileSelect).toHaveBeenCalledWith({ name:'checkpoint', full_path:'/workspace/model/lerobot/hero_act_inference/checkpoint', is_directory:true });
});

test('reopening resets navigation to the requested initial path', () => {
  const props = { initialPath:'/workspace/model/groot', allowDirectorySelect:true, onClose:jest.fn(), onFileSelect:jest.fn() };
  const { rerender } = render(<FileBrowserModal {...props} isOpen />);
  fireEvent.click(screen.getByRole('button', { name:'Enter nested folder' }));
  expect(screen.getByText('/workspace/model/groot/nested')).toBeInTheDocument();
  rerender(<FileBrowserModal {...props} isOpen={false} />);
  rerender(<FileBrowserModal {...props} isOpen />);
  expect(screen.getByTestId('browser')).toHaveAttribute('data-path','/workspace/model/groot');
  expect(screen.queryByText('/workspace/model/groot/nested')).not.toBeInTheDocument();
});
