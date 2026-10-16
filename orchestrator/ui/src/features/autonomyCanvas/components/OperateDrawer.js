// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { useEffect, useRef, useState } from "react";
import { FiMessageSquare, FiX } from "react-icons/fi";
import OperateSidebar from "./OperateSidebar";
import "./OperateExperience.css";

const panels = [
  { id: "message", label: "Talk to robot", Icon: FiMessageSquare },
];
export default function OperateDrawer(props) {
  const [open, setOpen] = useState(false);
  const [panel, setPanel] = useState("message");
  const triggers = useRef({});
  const close = () => { setOpen(false); triggers.current[panel]?.focus(); };
  useEffect(() => {
    if (!open) return;
    const escape = event => { if (event.key === "Escape") { setOpen(false); triggers.current[panel]?.focus(); } };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [open, panel]);
  return <>
    {[["message"]].map((ids, index) => <nav key={index}
      className="operate-tools operate-chat-trigger" aria-label="Conversation controls">
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
