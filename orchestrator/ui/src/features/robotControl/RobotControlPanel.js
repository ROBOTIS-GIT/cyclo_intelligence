// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useSelector } from 'react-redux';
import { MdTune } from 'react-icons/md';
import useRobotControl from './useRobotControl';
import { modeLabel } from './controlProtocol';
import './RobotControlPanel.css';

export default function RobotControlPanel() {
  const robotType = useSelector((state) => state.tasks.robotType);
  return robotType ? <RobotControls key={robotType} robotType={robotType} /> : null;
}

export function RobotControls({ robotType, modeTopic }) {
  const control = useRobotControl(robotType, modeTopic);
  const [expanded, setExpanded] = useState(false);
  const [left, setLeft] = useState('');
  const [right, setRight] = useState('');
  const [dirty, setDirty] = useState({ left: false, right: false });
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const toggle = useRef(null);
  const popup = useRef(null);
  const disabled = !control.connected || Boolean(control.pending);
  const sourceLabel = control.source === 'teleop' ? 'Teleop' : control.source === 'external' ? 'External' : 'Unknown';
  const joystickLabel = control.mode ? modeLabel(control.mode) : control.modeAvailable === false ? 'Offline' : 'Waiting';

  useLayoutEffect(() => {
    if (!expanded) return undefined;
    const place = () => {
      const rect = toggle.current.getBoundingClientRect();
      const width = Math.min(320, window.innerWidth - 24);
      const height = popup.current?.offsetHeight || 220;
      setPosition({
        left: Math.max(12, Math.min(rect.right - width, window.innerWidth - width - 12)),
        top: Math.max(12, rect.bottom + height + 12 < window.innerHeight ? rect.bottom + 6 : rect.top - height - 6),
      });
    };
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => { window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true); };
  }, [expanded, control.error, control.canSetPreset]);

  useEffect(() => {
    if (!expanded) return undefined;
    const dismiss = (event) => {
      if (!toggle.current?.contains(event.target) && !popup.current?.contains(event.target)) setExpanded(false);
    };
    const escape = (event) => {
      if (event.key === 'Escape') { setExpanded(false); toggle.current?.focus(); }
    };
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', dismiss); document.removeEventListener('keydown', escape); };
  }, [expanded]);

  const apply = async (event) => {
    event.preventDefault();
    const target = dirty.left && dirty.right ? 'both' : dirty.left ? 'left' : 'right';
    if (await control.setPreset(target, left, right)) setDirty({ left: false, right: false });
  };

  return (
    <div className="robot-control-anchor">
      <button ref={toggle} type="button" className="robot-control-toggle" aria-label="Robot controls"
        aria-expanded={expanded} aria-controls="robot-control-popup" onClick={() => setExpanded(!expanded)}>
        <MdTune size={16} aria-hidden="true" /> Robot
        <span className={`robot-source-dot is-${control.source || 'unknown'}`} aria-hidden="true" />
      </button>
      {expanded && createPortal(
        <div ref={popup} id="robot-control-popup" role="dialog" aria-label="Robot controls"
          className="robot-control-popup" style={position}>
          <div className="robot-control-status-row">
            <button type="button" className={`robot-control-status is-${control.source || 'unknown'}`}
              disabled={disabled || !control.canSetSource || !control.source}
              title={`Switch to ${control.source === 'teleop' ? 'External' : 'Teleop'}`}
              onClick={() => control.setSource(control.source === 'teleop' ? 'external' : 'teleop')}>
              Source: <strong>{sourceLabel}</strong>
            </button>
            <button type="button" className={`robot-control-status is-${control.mode || 'unknown'}`}
              disabled title="Switch Head / Base with both leader buttons">
              Joystick: <strong>{joystickLabel}</strong>
            </button>
          </div>
          {control.canSetPreset && (
            <form className="robot-control-presets" onSubmit={apply}>
              <PresetInput side="Left" value={left} disabled={disabled} onChange={(value) => { setLeft(value); setDirty((old) => ({ ...old, left: true })); }} />
              <PresetInput side="Right" value={right} disabled={disabled} onChange={(value) => { setRight(value); setDirty((old) => ({ ...old, right: true })); }} />
              <button className="robot-control-apply" type="submit" disabled={disabled || (!dirty.left && !dirty.right)}>Apply</button>
            </form>
          )}
          {control.pending && <p className="robot-control-hint" role="status">Applying…</p>}
          {!control.connected && <p className="robot-control-hint">Disconnected</p>}
          {control.error && <p className="robot-control-error" role="alert">{control.error}</p>}
        </div>, document.body,
      )}
    </div>
  );
}

function PresetInput({ side, value, onChange, disabled }) {
  const [custom, setCustom] = useState(false);
  return (
    <label className="robot-control-preset-input">{side} preset
      <select aria-label={`${side} preset`} value={custom ? 'custom' : value} disabled={disabled}
        onChange={(event) => { const next = event.target.value; setCustom(next === 'custom'); if (next !== 'custom') onChange(next); }}>
        <option value="" disabled>Select…</option>
        <option value="1">1 · Envelope</option><option value="2">2 · Pinch</option><option value="3">3 · Three-finger</option>
        <option value="custom">Custom ID</option>
      </select>
      {custom && <input aria-label={`${side} custom preset ID`} type="number" min="1" max="65535" step="1" required
        disabled={disabled} value={value} onChange={(event) => onChange(event.target.value)} />}
    </label>
  );
}
