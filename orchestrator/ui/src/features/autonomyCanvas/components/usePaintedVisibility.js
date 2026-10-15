import { useEffect, useState } from "react";

// Commit an invisible frame before fading in, including cached camera images
// that finish loading before the browser's first paint.
export default function usePaintedVisibility(enabled) {
  const [painted, setPainted] = useState(false);
  useEffect(() => {
    if (!enabled) { setPainted(false); return undefined; }
    let nextFrame;
    const firstFrame = requestAnimationFrame(() => {
      nextFrame = requestAnimationFrame(() => setPainted(true));
    });
    return () => {
      cancelAnimationFrame(firstFrame);
      if (nextFrame !== undefined) cancelAnimationFrame(nextFrame);
    };
  }, [enabled]);
  return enabled && painted;
}
