import { getPolicyBackendReadiness } from './usePolicyBackendStatus';

describe('getPolicyBackendReadiness', () => {
  it('blocks inference start when the backend container image is stale', () => {
    const readiness = getPolicyBackendReadiness({
      image_pulled: true,
      image_status: 'stale',
      container_state: 'exited',
      raw_state: 'stale_image',
      services: [],
    });

    expect(readiness).toEqual({
      ready: false,
      state: 'update_required',
      message: 'Policy Docker image changed. Update container before starting.',
    });
  });

  it('blocks inference start when the backend workspace mount is stale', () => {
    const readiness = getPolicyBackendReadiness({
      image_pulled: true,
      container_state: 'exited',
      raw_state: 'workspace_mount_mismatch',
      services: [],
    });

    expect(readiness).toEqual({
      ready: false,
      state: 'update_required',
      message: 'Policy Docker container changed. Update container before starting.',
    });
  });

  it('treats the common main and engine runtime services as ready', () => {
    const readiness = getPolicyBackendReadiness({
      image_pulled: true,
      image_status: 'current',
      container_state: 'running',
      services: [
        {
          name: 'main-runtime',
          state: 'up',
          uptime_s: 60,
        },
        {
          name: 'engine-process',
          state: 'up',
          uptime_s: 60,
        },
      ],
    });

    expect(readiness).toEqual({
      ready: true,
      state: 'ready',
      message: 'Backend ready',
    });
  });

  it('treats the RLDX policy server container as ready without s6 services', () => {
    const readiness = getPolicyBackendReadiness({
      name: 'rldx-server',
      image_pulled: true,
      image_status: 'current',
      container_state: 'running',
      services: [],
    });

    expect(readiness).toEqual({
      ready: true,
      state: 'ready',
      message: 'Backend ready',
    });
  });
});
