import { fireEvent, render, screen } from '@testing-library/react';
import OfflineRLDataConversionGuideModal from './OfflineRLDataConversionGuideModal';

const renderGuide = (props = {}) => render(
  <OfflineRLDataConversionGuideModal
    open
    onBack={jest.fn()}
    {...props}
  />
);

describe('OfflineRLDataConversionGuideModal', () => {
  afterEach(() => {
    document.body.style.overflow = '';
  });

  test('does not render while closed', () => {
    renderGuide({ open: false });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('moves conversion explanations into a focused viewport guide', () => {
    const { container } = renderGuide();
    const backdrop = screen.getByTestId('offline-rl-data-conversion-guide-backdrop');

    expect(screen.getByRole('dialog', { name: 'Data Conversion Guide' }))
      .toBeInTheDocument();
    expect(backdrop.parentElement).toBe(document.body);
    expect(container).not.toContainElement(backdrop);
    expect(screen.getByText('Data Epoch').closest('article')).toHaveTextContent(
      'automatically reserves the next monotonically increasing output folder'
    );
    expect(screen.getByText('Conversion outputs').closest('article'))
      .toHaveTextContent('LeRobot v3.0 output can be selected for training');
    expect(screen.getByText('Verified cleanup').closest('article'))
      .toHaveTextContent('removed only after every selected LeRobot output passes verification');
    expect(screen.getByText('Episode inspection').closest('article'))
      .toHaveTextContent('video and synchronized joint traces');
    expect(document.body.style.overflow).toBe('hidden');
    expect(screen.getByRole('button', { name: 'Back to Replay Buffer' })).toHaveFocus();
  });

  test('closes with Back, Escape, and the backdrop while restoring focus', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const onBack = jest.fn();
    const { rerender } = renderGuide({ onBack });

    fireEvent.mouseDown(screen.getByRole('dialog'));
    fireEvent.mouseDown(screen.getByTestId('offline-rl-data-conversion-guide-backdrop'));
    fireEvent.keyDown(window, { key: 'Escape' });
    fireEvent.click(screen.getByRole('button', { name: 'Back to Replay Buffer' }));
    expect(onBack).toHaveBeenCalledTimes(3);

    rerender(<OfflineRLDataConversionGuideModal open={false} onBack={onBack} />);
    expect(opener).toHaveFocus();
    expect(document.body.style.overflow).toBe('');
    opener.remove();
  });
});
