import { useEffect, useState } from "react";
import ReasoningText from "./ReasoningText";

const identity = (text, result) => JSON.stringify([text, result?.decision?.decision, result?.selected_mission?.mission_name]);

// Retain the last thought while awaiting the next result; crossfade complete
// snapshots so the outgoing sentence keeps its own action emphasis.
export default function AnimatedReason({ text = "", result, immediate = false, active = true }) {
  const key = identity(text, result);
  const [current, setCurrent] = useState({ text, result, key });
  const [previous, setPrevious] = useState(null);
  useEffect(() => {
    if (!text || key === current.key) return;
    setPrevious(!immediate && current.text ? current : null);
    setCurrent({ text, result, key });
  }, [text, result, key, current, immediate]);
  useEffect(() => {
    if (!previous) return;
    const timer = setTimeout(() => setPrevious(null), 320);
    return () => clearTimeout(timer);
  }, [previous]);
  if (!current.text) return null;
  return <div className="operate-thought-reason-stack" aria-busy={active && !text}>
    {previous && <p key={`old-${previous.key}`} className="operate-thought-reason is-outgoing" aria-hidden="true">
      <ReasoningText text={previous.text} result={previous.result}/>
    </p>}
    <p key={current.key} className={`operate-thought-reason is-current${immediate ? " is-immediate" : ""}`}
      aria-live={active ? "polite" : "off"}>
      <ReasoningText text={current.text} result={current.result}/>
    </p>
  </div>;
}
