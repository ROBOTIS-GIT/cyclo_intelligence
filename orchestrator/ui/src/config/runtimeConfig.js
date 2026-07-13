const runtimeConfig = (
  typeof window !== 'undefined' && window.__CYCLO_CONFIG__
) ? window.__CYCLO_CONFIG__ : {};

function getPort(name, defaultValue) {
  const port = Number(runtimeConfig[name]);
  return Number.isInteger(port) && port > 0 ? port : defaultValue;
}

export const CYCLO_UI_PORT = getPort('uiPort', 7080);
export const CYCLO_ROSBRIDGE_PORT = getPort('rosbridgePort', 7090);
export const CYCLO_VIDEO_SERVER_PORT = getPort('videoServerPort', 7082);
export const CYCLO_WEB_VIDEO_SERVER_PORT = getPort('webVideoServerPort', 7085);
export const CYCLO_SUPERVISOR_API_PORT = getPort('supervisorApiPort', 7100);
