const storageKey = (robotType) => `cyclo_intelligence.camera_preferences.v1.${encodeURIComponent(robotType)}`;

export function readCameraPreferences(robotType) {
  if (!robotType) return null;
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey(robotType)));
    if (!Array.isArray(saved?.topics) || !saved.topics.every((topic) => typeof topic === 'string' && topic.trim())) return null;
    const rotations = Object.fromEntries(Object.entries(saved.rotations || {}).filter(
      ([, angle]) => typeof angle === 'number' && Number.isFinite(angle) && angle >= 0 && angle < 360
    ));
    return { topics: [...new Set(saved.topics)], rotations };
  } catch {
    return null;
  }
}

export function writeCameraPreferences(robotType, topics, rotations) {
  if (!robotType || topics === null) return;
  try {
    localStorage.setItem(storageKey(robotType), JSON.stringify({ topics, rotations }));
  } catch {
    // Camera controls must remain usable when browser storage is unavailable.
  }
}
