import { fireEvent, render, screen } from '@testing-library/react';
import ACTArchitectureDiagram, {
  DEFAULT_ACT_TRAINABLE_GROUPS,
} from './ACTArchitectureDiagram';

describe('ACTArchitectureDiagram', () => {
  test('renders the four real ACT groups as trainable by default', () => {
    render(
      <ACTArchitectureDiagram
        trainableGroups={DEFAULT_ACT_TRAINABLE_GROUPS}
        onChange={jest.fn()}
      />
    );

    expect(screen.getAllByRole('button')).toHaveLength(4);
    for (const name of [
      'Visual backbone',
      'CVAE encoder',
      'Transformer encoder',
      'Action decoder',
    ]) {
      expect(screen.getByRole('button', { name: new RegExp(name, 'i') }))
        .toHaveAttribute('aria-pressed', 'true');
    }
    expect(screen.getAllByText('Fire · Trainable')).toHaveLength(4);
  });

  test('reports the ordered next config when a network block is toggled', () => {
    const onChange = jest.fn();
    render(
      <ACTArchitectureDiagram
        trainableGroups={DEFAULT_ACT_TRAINABLE_GROUPS}
        onChange={onChange}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /CVAE encoder/i }));

    expect(onChange).toHaveBeenCalledWith([
      'visual_backbone',
      'transformer_encoder',
      'action_decoder',
    ]);
  });

  test('stretches the transformer and action rows to the full diagram width', () => {
    const { container } = render(
      <ACTArchitectureDiagram
        trainableGroups={DEFAULT_ACT_TRAINABLE_GROUPS}
        onChange={jest.fn()}
      />
    );

    expect(container.querySelector('[data-trainable-group="transformer_encoder"]'))
      .toHaveClass('w-full');
    expect(container.querySelector('[data-trainable-group="action_decoder"]'))
      .toHaveClass('w-full');
  });

  test('uses the available architecture height with enlarged labels', () => {
    render(
      <ACTArchitectureDiagram
        trainableGroups={DEFAULT_ACT_TRAINABLE_GROUPS}
        onChange={jest.fn()}
      />
    );

    expect(screen.getByTestId('act-architecture-diagram'))
      .toHaveClass('h-full', 'flex-col');
    expect(screen.getByTestId('act-architecture-flow'))
      .toHaveClass('min-h-0', 'flex-1');
    expect(screen.getByText('Visual backbone'))
      .toHaveClass('text-[12px]');
    expect(screen.getByText('3 cameras → ResNet features'))
      .toHaveClass('text-[10px]');
  });

  test('exposes frozen state and prevents changes while disabled', () => {
    const onChange = jest.fn();
    render(
      <ACTArchitectureDiagram
        trainableGroups={['action_decoder']}
        onChange={onChange}
        disabled
      />
    );

    const visual = screen.getByRole('button', { name: /Visual backbone.*Frozen/i });
    const action = screen.getByRole('button', { name: /Action decoder.*Trainable/i });
    expect(visual).toHaveAttribute('aria-pressed', 'false');
    expect(visual).toHaveTextContent('Frozen');
    expect(action).toHaveAttribute('aria-pressed', 'true');
    expect(action).toHaveTextContent('Fire · Trainable');
    expect(visual).toBeDisabled();
    expect(action).toBeDisabled();

    fireEvent.click(action);
    expect(onChange).not.toHaveBeenCalled();
  });
});
