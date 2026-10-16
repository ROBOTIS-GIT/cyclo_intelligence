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

import { useEffect, useId, useRef, useState } from "react";
import { FiCheck, FiCpu } from "react-icons/fi";
import "./OperateModelPicker.css";

export default function OperateModelPicker({ choice, active }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);
  const trigger = useRef(null);
  const menuId = useId();
  const locked = active || choice?.locked || choice?.loading;
  useEffect(() => {
    if (!open) return;
    const outside = event => { if (!root.current?.contains(event.target)) setOpen(false); };
    const escape = event => {
      if (event.key === "Escape") { setOpen(false); trigger.current?.focus(); }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  useEffect(() => { if (locked) setOpen(false); }, [locked]);
  if (!choice) return null;
  return <div className="operate-model-picker" ref={root}>
    <button ref={trigger} type="button" className="operate-model-trigger"
      aria-label="Select reasoning model" title={choice.selected?.model_name || "Select reasoning model"}
      aria-expanded={open} aria-controls={menuId} disabled={Boolean(locked)}
      onClick={() => setOpen(value => !value)}><FiCpu size={17} aria-hidden="true"/></button>
    {open && <div id={menuId} className="operate-model-popup" role="group" aria-label="Reasoning models">
      {choice.models.map(model => <button type="button" key={model.id}
        className="operate-model-option" aria-pressed={choice.selected?.id === model.id}
        onClick={() => { choice.select(model.id); setOpen(false); trigger.current?.focus(); }}>
        <span>{model.model_name}{model.source === "base" ? " (Base)" : ""}</span>
        {choice.selected?.id === model.id && <FiCheck aria-hidden="true"/>}
      </button>)}
      {choice.error && <p role="alert">{choice.error}</p>}
      {!choice.models.length && <p>No reasoning models for this map.</p>}
    </div>}
  </div>;
}
