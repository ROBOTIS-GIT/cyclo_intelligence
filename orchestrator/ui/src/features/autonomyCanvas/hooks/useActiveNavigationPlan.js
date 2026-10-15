import { useRef } from "react";

// Nav2 retains /plan after cancellation. A new leg must receive a fresh path.
export default function useActiveNavigationPlan(active, plan, legKey = "navigation") {
  const cache = useRef({ key: null, stale: plan });
  const key = active ? legKey : null;
  if (!active || cache.current.key !== key) cache.current = { key, stale: plan };
  return active && plan !== cache.current.stale ? plan : null;
}
