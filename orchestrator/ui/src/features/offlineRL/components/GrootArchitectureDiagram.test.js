import { render, screen } from '@testing-library/react';
import GrootArchitectureDiagram from './GrootArchitectureDiagram';

describe('GrootArchitectureDiagram', () => {
  test('shows the official N1.7 default frozen and trainable boundaries as locked', () => {
    render(<GrootArchitectureDiagram />);

    for (const name of ['Visual encoder', 'Language model']) {
      const node = screen.getByRole('button', { name: new RegExp(`${name}: Frozen`, 'i') });
      expect(node).toHaveAttribute('aria-pressed', 'false');
      expect(node).toBeDisabled();
    }

    for (const name of ['VL adapter', 'State/action projectors', 'Flow-matching DiT']) {
      const node = screen.getByRole('button', { name: new RegExp(`${name}: Trainable`, 'i') });
      expect(node).toHaveAttribute('aria-pressed', 'true');
      expect(node).toBeDisabled();
    }

    expect(screen.getByText(/Official fine-tuning defaults/i)).toBeInTheDocument();
    expect(screen.getByText('Locked')).toBeInTheDocument();
  });
});
