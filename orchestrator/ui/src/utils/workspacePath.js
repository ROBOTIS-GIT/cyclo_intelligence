export const DEFAULT_WORKSPACE_MOUNT = {
  container_root: '/workspace',
  host_root: '',
  host_available: false,
};

const stripTrailingSlash = (value) => (value || '').trim().replace(/\/+$/, '');

const WORKSPACE_PATH_MARKERS = [
  '/docker/workspace',
  '/cyclo_intelligence/workspace',
];

export const mapWorkspacePathToContainer = (workspacePath, workspaceMount) => {
  const normalizedPath = stripTrailingSlash(workspacePath);
  const containerRoot = stripTrailingSlash(
    workspaceMount?.container_root || DEFAULT_WORKSPACE_MOUNT.container_root
  );
  const hostRoot = stripTrailingSlash(workspaceMount?.host_root || '');

  if (!normalizedPath || !containerRoot) return normalizedPath;
  if (
    normalizedPath === containerRoot ||
    normalizedPath.startsWith(`${containerRoot}/`)
  ) {
    return normalizedPath;
  }

  if (
    hostRoot &&
    (normalizedPath === hostRoot || normalizedPath.startsWith(`${hostRoot}/`))
  ) {
    const relativePath = normalizedPath.slice(hostRoot.length).replace(/^\/+/, '');
    return relativePath ? `${containerRoot}/${relativePath}` : containerRoot;
  }

  for (const marker of WORKSPACE_PATH_MARKERS) {
    const markerIndex = normalizedPath.lastIndexOf(marker);
    if (markerIndex < 0) continue;
    const markerEnd = markerIndex + marker.length;
    const nextCharacter = normalizedPath[markerEnd];
    if (nextCharacter && nextCharacter !== '/') continue;
    const relativePath = normalizedPath.slice(markerEnd).replace(/^\/+/, '');
    return relativePath ? `${containerRoot}/${relativePath}` : containerRoot;
  }

  return normalizedPath;
};

export const mapWorkspacePathToHost = (containerPath, workspaceMount) => {
  const normalizedPath = stripTrailingSlash(containerPath);
  const containerRoot = stripTrailingSlash(
    workspaceMount?.container_root || DEFAULT_WORKSPACE_MOUNT.container_root
  );
  const hostRoot = stripTrailingSlash(workspaceMount?.host_root || '');

  if (!normalizedPath || !containerRoot || !hostRoot) return '';
  if (normalizedPath !== containerRoot && !normalizedPath.startsWith(`${containerRoot}/`)) {
    return '';
  }

  const relativePath = normalizedPath.slice(containerRoot.length).replace(/^\/+/, '');
  return relativePath ? `${hostRoot}/${relativePath}` : hostRoot;
};
