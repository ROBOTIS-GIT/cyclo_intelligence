// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
import React, { useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import rosConnectionManager from '../utils/rosConnectionManager';
import { GripperControlClient, GRIPPER_OPEN, GRIPPER_CLOSED } from '../utils/gripperControl';

const INITIAL = { positions: { left: null, right: null },
  reasons: { left: 'ROS disconnected', right: 'ROS disconnected' }, busy: false, moving: null };

export default function InferenceGripperPanel({ isActive = true, externalBusy = false, onBusyChange }) {
  const robotType = useSelector((state) => state.tasks.robotType);
  const phase = useSelector((state) => state.tasks.inferenceStatus.inferencePhase);
  const recordPhase = useSelector((state) => state.tasks.recordStatus?.recordPhase ?? RecordPhase.READY);
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const supported = /^ffw_sg2(?:_rev1)?$/i.test(robotType);
  const permitted = isActive && !externalBusy && phase === InferencePhase.READY && recordPhase === RecordPhase.READY;
  const permittedRef = useRef(permitted);
  permittedRef.current = permitted;
  const clientRef = useRef(null);
  const onBusyRef = useRef(onBusyChange);
  onBusyRef.current = onBusyChange;
  const [state, setState] = useState(INITIAL);

  useEffect(() => {
    const update = () => {
      const ros = rosConnectionManager.url === rosbridgeUrl
        ? rosConnectionManager.getCurrentConnection() : null;
      if (clientRef.current?.ros !== ros) {
        clientRef.current?.dispose();
        clientRef.current = ros && supported ? new GripperControlClient(ros) : null;
      }
      const client = clientRef.current;
      client?.setEnabled(permittedRef.current && !document.hidden);
      client?.tick();
      const next = client?.snapshot() || INITIAL;
      setState(next);
      onBusyRef.current?.(next.busy);
    };
    update();
    const timer = window.setInterval(update, 100);
    return () => {
      window.clearInterval(timer);
      clientRef.current?.dispose();
      clientRef.current = null;
      onBusyRef.current?.(false);
    };
  }, [rosbridgeUrl, supported]);

  const command = (side, target) => {
    const client = clientRef.current;
    client?.setEnabled(permittedRef.current && !document.hidden);
    if (!supported || !permittedRef.current || !client?.command(side, target)) return;
    const next = client.snapshot();
    setState(next);
    onBusyRef.current?.(next.busy);
  };
  const globalReason = !supported ? 'Select SG2 (rev1) to control its grippers'
    : phase !== InferencePhase.READY ? 'Clear inference before using manual gripper controls'
      : recordPhase !== RecordPhase.READY ? 'Finish recording before using manual gripper controls'
        : externalBusy ? 'Wait for pose movement to finish' : '';
  const stopShortcut = (event) => {
    // Space/Enter on a gripper button must not reach the global Start shortcut.
    if (event.key === ' ' || event.code === 'Space' || event.key === 'Enter') event.stopPropagation();
  };

  return (
    <div className="w-full p-5 space-y-4" onKeyDown={stopShortcut} onKeyUp={stopShortcut}>
      <div>
        <h2 className="text-lg font-semibold text-gray-800">Gripper preparation</h2>
        <p className="mt-1 text-sm text-gray-600">
          Load the paper towel roll before starting inference. These buttons move the real robot in either deploy mode.
        </p>
      </div>
      {globalReason && <p role="status" className="text-sm text-amber-700">{globalReason}</p>}
      {['right', 'left'].map((side) => {
        const label = side === 'right' ? 'Right' : 'Left';
        const position = state.positions[side];
        const reason = globalReason || state.reasons[side];
        const disabled = !permitted || !!reason || state.busy;
        return (
          <section key={side} aria-label={`${label} gripper`} className="rounded-xl border border-gray-200 bg-gray-50 p-4">
            <div className="flex items-center justify-between gap-2">
              <h3 className="font-semibold text-gray-800">{label} gripper</h3>
              <span aria-label={`${label} gripper position`} className="text-sm tabular-nums text-gray-600">
                {Number.isFinite(position) ? `${position.toFixed(3)} rad` : '—'}
              </span>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <button type="button" aria-label={`Open ${side} gripper`} disabled={disabled}
                className="rounded-lg border border-blue-200 bg-white px-3 py-3 font-medium text-blue-700 hover:bg-blue-50 disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={() => command(side, GRIPPER_OPEN)}>Open · 0.0</button>
              <button type="button" aria-label={`Close ${side} gripper`} disabled={disabled}
                className="rounded-lg bg-blue-600 px-3 py-3 font-medium text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={() => command(side, GRIPPER_CLOSED)}>Close · 1.1</button>
            </div>
            {reason && !globalReason && <p role="status" className="mt-2 text-xs text-gray-600">{reason}</p>}
          </section>
        );
      })}
      <p role="status" aria-live="polite" className="text-sm text-gray-600">
        {state.moving
          ? `${state.moving.target === GRIPPER_CLOSED ? 'Closing' : 'Opening'} ${state.moving.side} gripper… Wait before starting inference.`
          : 'Leader connection is not required. Only the selected gripper moves.'}
      </p>
    </div>
  );
}
