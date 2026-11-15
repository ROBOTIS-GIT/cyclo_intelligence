import {
  mapWorkspacePathToContainer,
  mapWorkspacePathToHost,
} from './workspacePath';

describe('workspace path mapping', () => {
  test('keeps container paths canonical', () => {
    expect(mapWorkspacePathToContainer('/workspace/lerobot/task/'))
      .toBe('/workspace/lerobot/task');
  });

  test('maps the repository workspace symlink path into the container', () => {
    expect(mapWorkspacePathToContainer(
      '/home/robotis/cyclo_intelligence/docker/workspace/lerobot/task'
    )).toBe('/workspace/lerobot/task');
  });

  test('maps the resolved host mount when mount metadata is available', () => {
    const mount = {
      container_root: '/workspace',
      host_root: '/mnt/ssd/cyclo_intelligence/workspace',
      host_available: true,
    };
    expect(mapWorkspacePathToContainer(
      '/mnt/ssd/cyclo_intelligence/workspace/lerobot/task',
      mount
    )).toBe('/workspace/lerobot/task');
    expect(mapWorkspacePathToHost('/workspace/lerobot/task', mount))
      .toBe('/mnt/ssd/cyclo_intelligence/workspace/lerobot/task');
  });

  test('does not rewrite unrelated absolute paths', () => {
    expect(mapWorkspacePathToContainer('/opt/data/task')).toBe('/opt/data/task');
  });
});
