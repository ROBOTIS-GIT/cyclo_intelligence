import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import PolicyParameterFields from './PolicyParameterFields';

const model = {
  parameters: [
    {
      key: 'mode', label: 'Mode', control: 'select',
      binding: 'policy_parameters.mode', options: ['safe', 'fast'],
      visible_when: {},
    },
    {
      key: 'gain', label: 'Gain', control: 'number',
      binding: 'policy_parameters.gain', min: 0, max: 1, step: 0.1,
      visible_when: { mode: 'fast' },
    },
    {
      key: 'enabled', label: 'Enabled', control: 'toggle',
      binding: 'policy_parameters.enabled', visible_when: {},
    },
  ],
};

test('renders conditional schema controls and updates policy parameter state', () => {
  const onPolicyParametersChange = jest.fn();
  const info = { policyParameters: { mode: 'fast', gain: 0.5, enabled: false } };

  render(
    <PolicyParameterFields
      model={model}
      info={info}
      labelClassName="label"
      onTaskInfoChange={jest.fn()}
      onPolicyParametersChange={onPolicyParametersChange}
    />
  );

  fireEvent.change(screen.getByRole('spinbutton', { name: 'Gain' }), {
    target: { value: '0.7' },
  });
  expect(onPolicyParametersChange).toHaveBeenCalledWith({
    mode: 'fast', gain: 0.7, enabled: false,
  });
  fireEvent.click(screen.getByRole('checkbox', { name: 'Enabled' }));
  expect(onPolicyParametersChange).toHaveBeenCalledWith({
    mode: 'fast', gain: 0.5, enabled: true,
  });
});

test('hides conditional controls and removes cleared optional values', () => {
  const onPolicyParametersChange = jest.fn();
  const { rerender } = render(
    <PolicyParameterFields
      model={model}
      info={{ policyParameters: { mode: 'safe', gain: 0.5 } }}
      labelClassName="label"
      onTaskInfoChange={jest.fn()}
      onPolicyParametersChange={onPolicyParametersChange}
    />
  );
  expect(screen.queryByRole('spinbutton', { name: 'Gain' })).not.toBeInTheDocument();

  rerender(
    <PolicyParameterFields
      model={model}
      info={{ policyParameters: { mode: 'fast', gain: 0.5 } }}
      labelClassName="label"
      onTaskInfoChange={jest.fn()}
      onPolicyParametersChange={onPolicyParametersChange}
    />
  );
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Gain' }), {
    target: { value: '' },
  });
  expect(onPolicyParametersChange).toHaveBeenCalledWith({ mode: 'fast' });
});

test('does not render fields owned by a specialized runtime control', () => {
  const specializedModel = {
    parameters: [
      {
        key: 'engine_path', label: 'Engine Path', control: 'path',
        binding: 'task_info.accelerationEnginePath', visible_when: {},
      },
    ],
  };

  render(
    <PolicyParameterFields
      model={specializedModel}
      info={{ accelerationEnginePath: '/workspace/model/engine.trt' }}
      labelClassName="label"
      onTaskInfoChange={jest.fn()}
      onPolicyParametersChange={jest.fn()}
      excludedBindings={['task_info.accelerationEnginePath']}
    />
  );

  expect(screen.queryByRole('textbox', { name: 'Engine Path' }))
    .not.toBeInTheDocument();
});
