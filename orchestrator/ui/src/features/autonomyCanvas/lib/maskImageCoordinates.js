// Same letterboxing as SVG preserveAspectRatio="xMidYMid meet".
export function imagePoint(clientX, clientY, rect, width, height, view = { zoom: 1, x: 0, y: 0 }) {
  const scale = Math.min(rect.width / width, rect.height / height);
  if (!Number.isFinite(scale) || scale <= 0) return null;
  const x = (clientX - rect.left - (rect.width - width * scale) / 2) / scale;
  const y = (clientY - rect.top - (rect.height - height * scale) / 2) / scale;
  return x >= 0 && y >= 0 && x < width && y < height ? { x: Math.floor(view.x + x / view.zoom), y: Math.floor(view.y + y / view.zoom) } : null;
}

