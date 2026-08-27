import { fireEvent, render, screen } from '@testing-library/react';
import ACTArchitectureDiagram, {
  DEFAULT_ACT_TRAINABLE_GROUPS,
} from './ACTArchitectureDiagram';

describe('ACTArchitectureDiagram', () => {
  test('renders three controls while preserving the four real ACT groups', () => {
    render(
      <ACTArchitectureDiagram
        trainableGroups={DEFAULT_ACT_TRAINABLE_GROUPS}
        onChange={jest.fn()}
      />
    );

    expect(DEFAULT_ACT_TRAINABLE_GROUPS).toEqual([
      'visual_backbone',
      'cvae_encoder',
      'transformer_encoder',
      'action_decoder',
    ]);
    expect(screen.getAllByRole('button')).toHaveLength(3);
    for (const name of [
      'Visual backbone',
      'CVAE encoder',
      'Action Module',
    ]) {
      expect(screen.getByRole('button', { name: new RegExp(name, 'i') }))
        .toHaveAttribute('aria-pressed', 'true');
    }
    expect(screen.getAllByText('Fire · Trainable')).toHaveLength(3);
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

  test('renders one full-width action module for transformer and decoder', () => {
    const { container } = render(
      <ACTArchitectureDiagram
        trainableGroups={DEFAULT_ACT_TRAINABLE_GROUPS}
        onChange={jest.fn()}
      />
    );

    expect(container.querySelector('[data-trainable-group="action_module"]'))
      .toHaveClass('w-full');
    expect(container.querySelector('[data-trainable-group="transformer_encoder"]'))
      .not.toBeInTheDocument();
    expect(container.querySelector('[data-trainable-group="action_decoder"]'))
      .not.toBeInTheDocument();
  });

  test('distinguishes policy inputs, network roles, and action-chunk output', () => {
    const { container } = render(
      <ACTArchitectureDiagram
        trainableGroups={DEFAULT_ACT_TRAINABLE_GROUPS}
        onChange={jest.fn()}
      />
    );

    expect(screen.getByText('3 camera images')).toBeInTheDocument();
    expect(screen.getByText('Robot state')).toBeInTheDocument();
    expect(screen.getByTestId('act-policy-inputs')).toHaveClass('grid-cols-2');
    expect(container.querySelector('[data-trainable-group="visual_backbone"]'))
      .toHaveClass('bg-[#f0edfa]', 'border-[#c7bde6]');
    expect(container.querySelector('[data-trainable-group="action_module"]'))
      .toHaveClass('bg-[#edf4ec]', 'border-[#acc2ae]');
    expect(screen.getByTestId('act-policy-output'))
      .toHaveClass('bg-[#e9edfa]', 'border-[#9faacf]');
    expect(screen.getByText('30 steps')).toBeInTheDocument();
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
        trainableGroups={['transformer_encoder', 'action_decoder']}
        onChange={onChange}
        disabled
      />
    );

    const visual = screen.getByRole('button', { name: /Visual backbone.*Frozen/i });
    const action = screen.getByRole('button', { name: /Action Module.*Trainable/i });
    expect(visual).toHaveAttribute('aria-pressed', 'false');
    expect(visual).toHaveTextContent('Frozen');
    expect(action).toHaveAttribute('aria-pressed', 'true');
    expect(action).toHaveTextContent('Fire · Trainable');
    expect(visual).toBeDisabled();
    expect(action).toBeDisabled();

    fireEvent.click(action);
    expect(onChange).not.toHaveBeenCalled();
  });

  test('toggles the transformer encoder and action decoder atomically', () => {
    const onChange = jest.fn();
    render(
      <ACTArchitectureDiagram
        trainableGroups={DEFAULT_ACT_TRAINABLE_GROUPS}
        onChange={onChange}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /Action Module.*Trainable/i }));

    expect(onChange).toHaveBeenCalledWith([
      'visual_backbone',
      'cvae_encoder',
    ]);
  });

  test('exposes a legacy partial selection as mixed and normalizes it on click', () => {
    const onChange = jest.fn();
    render(
      <ACTArchitectureDiagram
        trainableGroups={[
          'visual_backbone',
          'cvae_encoder',
          'action_decoder',
        ]}
        onChange={onChange}
      />
    );

    const action = screen.getByRole('button', { name: /Action Module.*Mixed/i });
    expect(action).toHaveAttribute('aria-pressed', 'mixed');
    expect(action).toHaveTextContent('Mixed');

    fireEvent.click(action);
    expect(onChange).toHaveBeenCalledWith(DEFAULT_ACT_TRAINABLE_GROUPS);
  });

  test('locks an objective-incompatible block without disabling the remaining ACT policy', () => {
    const onChange = jest.fn();
    render(
      <ACTArchitectureDiagram
        trainableGroups={[
          'visual_backbone',
          'transformer_encoder',
          'action_decoder',
        ]}
        lockedGroups={['cvae_encoder']}
        onChange={onChange}
      />
    );

    const cvae = screen.getByRole('button', { name: /CVAE encoder.*locked for pure TD3/i });
    const visual = screen.getByRole('button', { name: /Visual backbone.*Trainable/i });
    expect(cvae).toBeDisabled();
    expect(cvae).toHaveTextContent('Frozen · TD3');
    expect(visual).not.toBeDisabled();

    fireEvent.click(cvae);
    expect(onChange).not.toHaveBeenCalled();
  });
});
