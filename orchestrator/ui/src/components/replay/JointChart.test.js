import { buildJointChartData } from './JointChart';

describe('buildJointChartData', () => {
  it('fills state and action values at union timestamps', () => {
    const stateData = [
      { time: 0.0, state_joint1: 1 },
      { time: 0.1, state_joint1: 2 },
    ];
    const actionData = [
      { time: 0.05, action_joint1: 10 },
    ];

    const result = buildJointChartData(stateData, actionData, 'joint1', true);

    expect(result).toEqual([
      { time: 0.0, state_joint1: 1, action_joint1: 10 },
      { time: 0.05, state_joint1: 1, action_joint1: 10 },
      { time: 0.1, state_joint1: 2, action_joint1: 10 },
    ]);
  });

  it('does not include action values when a joint has no matching action', () => {
    const stateData = [
      { time: 0.0, state_joint1: 1 },
    ];
    const actionData = [
      { time: 0.0, action_other_joint: 10 },
    ];

    const result = buildJointChartData(stateData, actionData, 'joint1', false);

    expect(result).toEqual([
      { time: 0.0, state_joint1: 1 },
    ]);
  });
});
