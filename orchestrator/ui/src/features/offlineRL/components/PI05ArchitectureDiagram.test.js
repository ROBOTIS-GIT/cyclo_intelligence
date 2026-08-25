import { fireEvent, render, screen } from '@testing-library/react';
import PI05ArchitectureDiagram from './PI05ArchitectureDiagram';

const expectState = (name, state) => {
  expect(screen.getByRole('group', { name: `${name}: ${state}` })).toBeInTheDocument();
};

describe('PI05ArchitectureDiagram', () => {
  test('derives only the three freeze modes supported by PI05Config', () => {
    render(<PI05ArchitectureDiagram />);

    expect(screen.getByRole('button', { name: 'Full fine-tune' }))
      .toHaveAttribute('aria-pressed', 'true');
    expectState('SigLIP vision encoder', 'Trainable');
    expectState('PaliGemma VLM', 'Trainable');
    expectState('Action/time projections', 'Trainable');
    expectState('Gemma action expert', 'Trainable');

    fireEvent.click(screen.getByRole('button', { name: 'Frozen vision' }));
    expectState('SigLIP vision encoder', 'Frozen');
    expectState('PaliGemma VLM', 'Trainable');

    fireEvent.click(screen.getByRole('button', { name: 'Expert only' }));
    expectState('SigLIP vision encoder', 'Frozen');
    expectState('PaliGemma VLM', 'Frozen');
    expectState('Action/time projections', 'Trainable');
    expectState('Gemma action expert', 'Trainable');
  });

  test('locks mode previews while the surrounding workflow is locked', () => {
    render(<PI05ArchitectureDiagram disabled />);

    for (const name of ['Full fine-tune', 'Frozen vision', 'Expert only']) {
      expect(screen.getByRole('button', { name })).toBeDisabled();
    }
  });
});
