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

  test('renders as a viewport portal with the Quick Start guide selected', () => {
    const { container } = renderGuide();
    const backdrop = screen.getByTestId('offline-rl-training-guide-backdrop');

    expect(screen.getByRole('dialog', { name: 'Training Guide' })).toBeInTheDocument();
    expect(backdrop.parentElement).toBe(document.body);
    expect(container).not.toContainElement(backdrop);
    expect(screen.getByRole('tab', { name: 'Quick Start' }))
      .toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tablist', { name: 'Training Guide sections' }))
      .toHaveAttribute('aria-orientation', 'vertical');
    expect(screen.getAllByRole('tab').map((tab) => tab.textContent)).toEqual([
      'Quick StartEnd-to-end workflow',
      'ACTIL · Critic · TD3',
      'Diffusion PolicyIL · Critic · Flow-SDE PPO',
      'GR00TRLT integration',
      'Pi0.5Architecture preview',
    ]);
    expect(screen.getByText('Training workflow')).toBeInTheDocument();
    expect(screen.getByTestId('training-guide-quick-start')).toHaveTextContent('Compatibility');
    expect(document.body.style.overflow).toBe('hidden');
    expect(screen.getByRole('button', { name: 'Back to Training' })).toHaveFocus();
  });

  test('shows model-specific setup, training, validation, and limitations', () => {
    renderGuide();

    fireEvent.click(screen.getByRole('tab', { name: 'ACT' }));
    const actGuide = screen.getByTestId('training-guide-act');
    expect(actGuide).toHaveTextContent('Critic Warm-up');
    expect(actGuide).toHaveTextContent('TD3-BC');
    expect(actGuide).toHaveTextContent('success/fail episode outcomes');
    expect(actGuide).toHaveTextContent(
      'resume checkpoint → previous round → policy warm-up → random initialization'
    );

    fireEvent.click(screen.getByRole('tab', { name: 'Diffusion Policy' }));
    const diffusionGuide = screen.getByTestId('training-guide-diffusion');
    expect(diffusionGuide).toHaveTextContent('Offline Value Critic');
    expect(diffusionGuide).toHaveTextContent('Flow-SDE PPO');
    expect(diffusionGuide).toHaveTextContent('16 × 22D');
    expect(diffusionGuide).toHaveTextContent('action-step ACK');
    expect(diffusionGuide).toHaveTextContent('Critic training is not embedded in PPO');
    expect(diffusionGuide).toHaveTextContent(
      'compatible completed critic bundle is reused automatically'
    );
    expect(diffusionGuide).toHaveTextContent(
      'policy checkpoint and task instruction match'
    );

    fireEvent.click(screen.getByRole('tab', { name: 'GR00T' }));
    const grootGuide = screen.getByTestId('training-guide-groot');
    expect(grootGuide).toHaveTextContent('10 × 19 @ 15 Hz');
    expect(grootGuide).toHaveTextContent('not connected to Start Training');
    expect(screen.getByRole('link', { name: /RLinf: RL on GR00T Models/i }))
      .toHaveAttribute(
        'href',
        'https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html'
      );

    fireEvent.click(screen.getByRole('tab', { name: 'Pi0.5' }));
    const piGuide = screen.getByTestId('training-guide-pi05');
    expect(piGuide).toHaveTextContent('No Pi0.5 trainer');
    expect(piGuide).toHaveTextContent('Start Training remains disabled');
  });

  test('supports keyboard navigation in the model tab rail', () => {
    renderGuide();
    const quickStartTab = screen.getByRole('tab', { name: 'Quick Start' });
    quickStartTab.focus();

    fireEvent.keyDown(quickStartTab, { key: 'ArrowDown' });
    const actTab = screen.getByRole('tab', { name: 'ACT' });
    expect(actTab).toHaveFocus();
    expect(actTab).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('training-guide-act')).toBeInTheDocument();

    fireEvent.keyDown(actTab, { key: 'End' });
    const piTab = screen.getByRole('tab', { name: 'Pi0.5' });
    expect(piTab).toHaveFocus();
    expect(piTab).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(piTab, { key: 'Home' });
    expect(quickStartTab).toHaveFocus();
    expect(quickStartTab).toHaveAttribute('aria-selected', 'true');
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
