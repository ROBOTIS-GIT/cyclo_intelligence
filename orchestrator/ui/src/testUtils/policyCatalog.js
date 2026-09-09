export const testPolicyCatalog = {
  schema_version: 1,
  runtimes: [
    {
      id: 'lerobot',
      label: 'LeRobot',
      compose_service: 'lerobot',
      service_prefix: 'lerobot',
      checkpoint_root: '/workspace/model/lerobot',
      capabilities: {
        action_request_modes: ['async', 'sync'],
        requires_hf_token: false,
        operations: [],
      },
      models: [
        {
          id: 'act',
          policy_id: 'lerobot:act',
          label: 'ACT',
          aliases: ['act'],
          requires_instruction: false,
          parameters: [],
        },
        ...[
          ['pi0', 'Pi0'],
          ['pi0_fast', 'Pi0-FAST'],
          ['eo1', 'EO1'],
          ['evo1', 'Evo1'],
          ['wall_x', 'WALL-X'],
          ['groot', 'GR00T N1.7 (LeRobot)'],
        ].map(([id, label]) => ({
          id,
          policy_id: `lerobot:${id}`,
          label,
          aliases: id === 'groot' ? [] : [id],
          requires_instruction: true,
          parameters: [],
        })),
      ],
    },
    {
      id: 'groot',
      label: 'GR00T',
      compose_service: 'groot',
      service_prefix: 'groot',
      checkpoint_root: '/workspace/model/groot',
      capabilities: {
        action_request_modes: ['async', 'sync'],
        requires_hf_token: true,
        operations: ['groot_trt'],
      },
      models: [
        {
          id: 'n17',
          policy_id: 'groot:n17',
          label: 'N1.7',
          aliases: ['groot', 'n17', 'n1.7'],
          requires_instruction: true,
          parameters: [
            {
              key: 'acceleration_mode',
              label: 'Acceleration',
              control: 'select',
              binding: 'task_info.accelerationMode',
              default: 'pytorch',
              required: false,
              options: ['pytorch', 'tensorrt_dit'],
              visible_when: {},
            },
            {
              key: 'acceleration_engine_path',
              label: 'TensorRT Engine Path',
              control: 'path',
              binding: 'task_info.accelerationEnginePath',
              default: '',
              required: false,
              options: [],
              visible_when: { acceleration_mode: 'tensorrt_dit' },
            },
          ],
        },
      ],
    },
  ],
};

export const testRuntime = (runtimeId) => (
  testPolicyCatalog.runtimes.find((runtime) => runtime.id === runtimeId)
);
