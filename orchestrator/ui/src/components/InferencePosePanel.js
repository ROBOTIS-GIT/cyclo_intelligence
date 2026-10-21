// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
import React, { useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import rosConnectionManager from '../utils/rosConnectionManager';
import { InferencePoseClient, POSE_GROUPS } from '../utils/inferencePoseControl';

const INITIAL = { positions: {}, reasons: {}, failures: {}, holding: {}, busy: false, moving: null };

export default function InferencePosePanel({ isActive = true, externalBusy = false, onBusyChange }) {
  const robotType = useSelector((state) => state.tasks.robotType);
  const phase = useSelector((state) => state.tasks.inferenceStatus.inferencePhase);
  const recordPhase = useSelector((state) => state.tasks.recordStatus?.recordPhase ?? RecordPhase.READY);
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const supported = /^ffw_sg2(?:_rev1)?$/i.test(robotType);
  const permitted = isActive && !externalBusy && phase === InferencePhase.READY && recordPhase === RecordPhase.READY;
  const permittedRef = useRef(permitted); permittedRef.current = permitted;
  const clientRef = useRef(null);
  const onBusyRef = useRef(onBusyChange); onBusyRef.current = onBusyChange;
  const [state, setState] = useState(INITIAL);
  const [selected, setSelected] = useState({ left: 0, right: 0, head: 0, lift: 0 });

  useEffect(() => {
    const update = () => {
      const ros = rosConnectionManager.url === rosbridgeUrl ? rosConnectionManager.getCurrentConnection() : null;
      if (ros && clientRef.current?.ros !== ros) {
        const pendingMove = clientRef.current?.moving;
        clientRef.current?.dispose();
        clientRef.current = ros && supported ? new InferencePoseClient(ros) : null;
        // A finite trajectory may continue at the controller after ROS disconnects.
        // Preserve the Start lock until its duration and feedback check finish.
        if (clientRef.current && pendingMove) clientRef.current.moving = pendingMove;
      }
      const client = clientRef.current;
      client?.setEnabled(permittedRef.current && !document.hidden);
      client?.tick();
      const next = client?.snapshot() || INITIAL;
      setState(next); onBusyRef.current?.(next.busy);
    };
    update();
    const timer = window.setInterval(update, 100);
    return () => {
      window.clearInterval(timer); clientRef.current?.dispose(); clientRef.current = null;
      onBusyRef.current?.(false);
    };
  }, [rosbridgeUrl, supported]);

  const command = (group, hold = false) => {
    const client = clientRef.current;
    client?.setEnabled(permittedRef.current && !document.hidden);
    if (!supported || !permittedRef.current || !client) return;
    client.command(group, selected[group], hold);
    const next = client.snapshot(); setState(next); onBusyRef.current?.(next.busy);
  };
  const reason = !supported ? 'Select SG2 (rev1) to prepare poses'
    : phase !== InferencePhase.READY ? 'Clear inference before preparing poses'
      : recordPhase !== RecordPhase.READY ? 'Finish recording before preparing poses'
        : externalBusy ? 'Wait for the gripper movement to finish' : '';
  const stopShortcut = (event) => {
    if (event.key === ' ' || event.code === 'Space' || event.key === 'Enter') event.stopPropagation();
  };

  return (
    <div className="w-full p-4 space-y-4" onKeyDown={stopShortcut} onKeyUp={stopShortcut}>
      <div>
        <h2 className="text-lg font-semibold text-gray-800">Pose preparation</h2>
        <p className="mt-1 text-sm text-gray-600">Moves the real robot in either deploy mode. Only the selected group moves; its controller holds the final pose until the next command.</p>
        <p className="mt-1 text-xs text-gray-600">Arm presets include the gripper: all supplied presets open it to 0.0.</p>
      </div>
      {reason && <p role="status" className="text-sm text-amber-700">{reason}</p>}
      {Object.entries(POSE_GROUPS).map(([group, config]) => {
        const blocker = reason || state.reasons[group] || (!clientRef.current ? 'ROS disconnected' : '');
        const disabled = !permitted || !!blocker;
        const moving = state.moving?.group === group;
        const heldIndex = config.poses.findIndex((pose) => pose.every((value, i) => value === state.holding[group]?.[i]));
        return (
          <section key={group} aria-label={`${config.label} pose preparation`} className="rounded-xl border border-gray-200 bg-gray-50 p-3">
            <h3 className="font-semibold text-gray-800">{config.label}</h3>
            <label className="mt-2 block text-xs text-gray-600" htmlFor={`pose-${group}`}>Preset</label>
            <select id={`pose-${group}`} aria-label={`${config.label} preset`} value={selected[group]}
              disabled={disabled || state.busy} className="mt-1 w-full rounded-md border border-gray-300 bg-white px-2 py-2 text-sm disabled:opacity-40"
              onChange={(event) => setSelected((previous) => ({ ...previous, [group]: Number(event.target.value) }))}>
              {config.poses.map((_, index) => <option key={index} value={index}>{config.label} {group === 'lift' ? 'height' : 'pose'} {index + 1}</option>)}
            </select>
            <p className="mt-2 break-words font-mono text-[11px] text-gray-600" aria-label={`${config.label} target`}>
              [{config.poses[selected[group]].map((value) => Number(value.toFixed(6))).join(', ')}] {group === 'lift' ? 'm' : 'rad'}
            </p>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <button type="button" aria-label={`Move ${group} to selected pose`} disabled={disabled || state.busy}
                className="rounded-lg bg-blue-600 px-2 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={() => command(group)}>Move &amp; hold</button>
              <button type="button" aria-label={`Hold current ${group} pose`} disabled={disabled || (state.busy && !moving)}
                className="rounded-lg border border-blue-200 bg-white px-2 py-2 text-sm font-medium text-blue-700 hover:bg-blue-50 disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={() => command(group, true)}>Hold current</button>
            </div>
            <p aria-label={`${config.label} measured pose`} className="mt-2 break-words font-mono text-[11px] text-gray-500">
              Current: {state.positions[group] ? `[${state.positions[group].map((value) => value.toFixed(3)).join(', ')}]` : '—'}
            </p>
            <p role="status" aria-live="polite" className={`mt-2 text-xs ${state.failures[group] ? 'text-red-700' : 'text-gray-600'}`}>
              {blocker || state.failures[group] || (moving
                ? `${state.moving.hold ? 'Holding current pose' : 'Moving to pose ' + (state.moving.poseIndex + 1)}… ${state.moving.duration.toFixed(1)} s`
                : state.holding[group] ? `Holding ${heldIndex >= 0 ? 'pose ' + (heldIndex + 1) : 'current pose'}` : 'Ready')}
            </p>
          </section>
        );
      })}
    </div>
  );
}
