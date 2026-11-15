import { fireEvent, render, screen } from '@testing-library/react';
import OfflineRLTrainingGuideModal from './OfflineRLTrainingGuideModal';

const renderGuide = (props = {}) => render(
  <OfflineRLTrainingGuideModal
    open
    onBack={jest.fn()}
    {...props}
  />
);

describe('OfflineRLTrainingGuideModal', () => {
  afterEach(() => {
    document.body.style.overflow = '';
  });

  test('does not render while closed', () => {
    renderGuide({ open: false });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('opens the GR00T guide without other model or tuning tabs', () => {
    const { container } = renderGuide();
    const backdrop = screen.getByTestId('offline-rl-training-guide-backdrop');
    expect(backdrop.parentElement).toBe(document.body);
    expect(container).not.toContainElement(backdrop);
    expect(screen.getAllByRole('tab')).toHaveLength(1);
    expect(screen.getByRole('tab', { name: 'GR00T' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tabpanel')).toHaveTextContent('RL Token Stage 1');
    expect(screen.getByRole('tabpanel')).toHaveTextContent('RLT Stage 2');
    expect(screen.getByRole('tabpanel')).toHaveTextContent('Frozen Token Feature MSE');
    expect(screen.getByRole('button', { name: 'Back to Training' })).toHaveFocus();
    expect(document.body.style.overflow).toBe('hidden');
  });

  test('keeps keyboard navigation within the GR00T guide', () => {
    renderGuide();
    const tab = screen.getByRole('tab', { name: 'GR00T' });
    tab.focus();
    for (const key of ['ArrowDown', 'ArrowUp', 'Home', 'End']) {
      fireEvent.keyDown(tab, { key });
      expect(tab).toHaveFocus();
      expect(tab).toHaveAttribute('aria-selected', 'true');
    }
  });

  test('closes with Back, Escape, and the backdrop while restoring focus', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const onBack = jest.fn();
    const { rerender } = renderGuide({ onBack });

    fireEvent.mouseDown(screen.getByRole('dialog'));
    fireEvent.mouseDown(screen.getByTestId('offline-rl-training-guide-backdrop'));
    fireEvent.keyDown(window, { key: 'Escape' });
    fireEvent.click(screen.getByRole('button', { name: 'Back to Training' }));
    expect(onBack).toHaveBeenCalledTimes(3);

    rerender(<OfflineRLTrainingGuideModal open={false} onBack={onBack} />);
    expect(opener).toHaveFocus();
    expect(document.body.style.overflow).toBe('');
    opener.remove();
  });
});
