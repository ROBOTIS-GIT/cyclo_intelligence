import { useState } from "react";
import LiveHeadCamera from "./LiveHeadCamera";

// Mounted only in the active workspace so Record/Reasoning never duplicate the stream.
export default function CollectHeadCamera({ topic, className = "" }) {
  const [aspect, setAspect] = useState(16 / 9);
  return <div className={`operate-observation collect-camera ${className}`}
    style={{ "--collect-camera-aspect": aspect }}>
    <LiveHeadCamera topic={topic} onAspectRatio={setAspect} />
  </div>;
}
