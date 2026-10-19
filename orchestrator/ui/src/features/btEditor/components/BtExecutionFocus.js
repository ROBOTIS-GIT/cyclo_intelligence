import { useEffect, useMemo, useState } from "react";
import { Panel, useNodesInitialized, useReactFlow } from "@xyflow/react";
import { MdCenterFocusStrong } from "react-icons/md";

// Ancestors also glow: focus only the deepest visible running nodes.
export function executionFocusIds(nodes, edges) {
  const active = new Set(nodes.filter(n => !n.hidden && n.data?.isActive).map(n => n.id));
  const children = new Map();
  edges.forEach(e => children.set(e.source, [...(children.get(e.source) || []), e.target]));
  return [...active].filter(id => {
    const seen = new Set([id]);
    const queue = [...(children.get(id) || [])];
    while (queue.length) {
      const next = queue.shift();
      if (seen.has(next)) continue;
      seen.add(next);
      if (active.has(next)) return false;
      queue.push(...(children.get(next) || []));
    }
    return true;
  }).sort();
}

export default function BtExecutionFocus({ nodes, edges }) {
  const { fitView } = useReactFlow();
  const ready = useNodesInitialized();
  const [follow, setFollow] = useState(true);
  const key = useMemo(() => JSON.stringify(executionFocusIds(nodes, edges)), [nodes, edges]);
  useEffect(() => {
    const ids = JSON.parse(key);
    if (!follow || !ready || !ids.length) return undefined;
    // Avoid chasing transient nodes, and leave the last view on completion.
    const timer = setTimeout(() => {
      fitView({ nodes: ids.map(id => ({ id })), padding: 0.35,
        minZoom: 0.9, maxZoom: 1.15, duration: 250 });
    }, 180);
    return () => clearTimeout(timer);
  }, [key, ready, follow, fitView]);
  return <Panel position="top-right">
    <button type="button" aria-label="Follow running node" aria-pressed={follow}
      title={follow ? "Following running node — click to pause" : "Follow running node"}
      onClick={() => setFollow(value => !value)}
      style={{ width: 30, height: 30, display: "grid", placeItems: "center",
        border: "1px solid var(--mc-border, #ddd)", borderRadius: 4,
        background: follow ? "var(--mc-accent, #a4b7a5)" : "var(--mc-bg, white)",
        color: "var(--mc-text, #333)", cursor: "pointer" }}>
      <MdCenterFocusStrong size={18} />
    </button>
  </Panel>;
}
