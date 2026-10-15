import { useEffect, useRef, useState } from "react";
import { FiActivity, FiMessageSquare, FiSliders, FiX } from "react-icons/fi";
import OperateSidebar from "./OperateSidebar";
import "./OperateExperience.css";

const panels = [
  { id: "situation", label: "Status and settings", Icon: FiSliders },
  { id: "activity", label: "Activity", Icon: FiActivity },
  { id: "message", label: "Talk to robot", Icon: FiMessageSquare },
];
export default function OperateDrawer(props) {
  const [open, setOpen] = useState(false);
  const [panel, setPanel] = useState("situation");
  const triggers = useRef({});
  const close = () => { setOpen(false); triggers.current[panel]?.focus(); };
  useEffect(() => {
    if (!open) return;
    const escape = event => { if (event.key === "Escape") { setOpen(false); triggers.current[panel]?.focus(); } };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [open, panel]);
  return <>
    {[["situation", "activity"], ["message"]].map((ids, index) => <nav key={index}
      className={`operate-tools${index ? " operate-chat-trigger" : ""}`} aria-label={index ? "Conversation controls" : "Robot controls"}>
      {panels.filter(item => ids.includes(item.id)).map(({ id, label, Icon }) => <button key={id} type="button" title={label} aria-label={label}
        aria-expanded={open && panel === id} aria-controls="operate-detail-drawer"
        ref={el => { triggers.current[id] = el; }}
        onClick={() => { setPanel(id); setOpen(!(open && panel === id)); }}>
        <Icon size={19} aria-hidden="true"/>
      </button>)}
    </nav>)}
    <aside id="operate-detail-drawer" className="operate-detail-drawer" hidden={!open} aria-label={panels.find(item => item.id === panel).label}>
      <div className="operate-drawer-heading"><strong>{panels.find(item => item.id === panel).label}</strong>
        <button type="button" aria-label="Close details" title="Close details" onClick={close}><FiX aria-hidden="true"/></button></div>
      <div className="operate-drawer-body"><OperateSidebar {...props} section={panel}/></div>
    </aside>
  </>;
}
