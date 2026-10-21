// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import rosConnectionManager from "../../../../utils/rosConnectionManager";
import { HeadTeleopClient } from "../../lib/headTeleop";
import { MISSION_TEXT_MUTED } from "../../../mapWorkspace/lib/theme";
import { ActionButton, Panel, SessionRow } from "../../../mapWorkspace/components/primitives";

const KEYS = { i: "up", k: "down", j: "left", l: "right" };
const BUTTONS = [
  ["up", "↑", "I", "col-start-2"],
  ["left", "←", "J", "col-start-1"],
  ["right", "→", "L", "col-start-3"],
  ["down", "↓", "K", "col-start-2"],
];
const INITIAL = { positions: null, reason: "ROS disconnected", active: false, direction: null };
const angle = (value) => Number.isFinite(value) ? `${value.toFixed(3)} rad` : "—";

export default function HeadTeleopPanel({ disabled = false }) {
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const robotType = useSelector((state) => state.tasks?.robotType || "");
  const supported = /^ffw_sg2(?:_rev1)?$/i.test(robotType);
  const clientRef = useRef(null);
  const [state, setState] = useState(INITIAL);

  useEffect(() => {
    const update = () => {
      const ros = rosConnectionManager.url === rosbridgeUrl
        ? rosConnectionManager.getCurrentConnection?.() : null;
      if (clientRef.current?.ros !== ros) {
        clientRef.current?.dispose();
        clientRef.current = ros && supported ? new HeadTeleopClient(ros) : null;
      }
      const client = clientRef.current;
      if (disabled || document.hidden) client?.deactivate();
      client?.tick();
      setState(client ? client.snapshot() : INITIAL);
    };
    const deactivate = () => {
      clientRef.current?.deactivate();
      setState(clientRef.current?.snapshot() || INITIAL);
    };
    update();
    const timer = window.setInterval(update, 100);
    window.addEventListener("blur", deactivate);
    document.addEventListener("visibilitychange", deactivate);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("blur", deactivate);
      document.removeEventListener("visibilitychange", deactivate);
      clientRef.current?.dispose();
      clientRef.current = null;
    };
  }, [disabled, rosbridgeUrl, supported]);

  const reason = !supported ? "Select SG2 (rev1) to control this head"
    : disabled ? "Unavailable while another operation is running" : state.reason;
  const controlsDisabled = !!reason || !state.active;
  const act = (method, direction) => {
    if (method === "start" && controlsDisabled) return;
    clientRef.current?.[method](direction);
    setState(clientRef.current?.snapshot() || INITIAL);
  };

  return (
    <Panel title="Head Teleop" className="grid gap-3 content-start">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs" style={{ color: MISSION_TEXT_MUTED }}>
          {state.active ? "Active" : "Inactive"}
        </span>
        <ActionButton variant="secondary" active={state.active} disabled={!!reason}
          onClick={() => act(state.active ? "deactivate" : "activate")}>
          {state.active ? "Deactivate head" : "Activate head"}
        </ActionButton>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <SessionRow label="Pitch · joint 1" value={angle(state.positions?.[0])} stacked />
        <SessionRow label="Yaw · joint 2" value={angle(state.positions?.[1])} stacked />
      </div>
      {reason && <p role="status" className="text-xs m-0" style={{ color: MISSION_TEXT_MUTED }}>{reason}</p>}
      <div role="group" aria-label="Head Teleop" tabIndex={controlsDisabled ? -1 : 0}
        className="grid grid-cols-3 gap-2 justify-self-center rounded-md focus-visible:outline"
        onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) act("stop"); }}
        onKeyDown={(event) => {
          const direction = KEYS[event.key.toLowerCase()];
          if (direction || event.code === "Space" || event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            if (!event.repeat) act(direction ? "start" : "stop", direction);
          }
        }}
        onKeyUp={(event) => {
          const direction = KEYS[event.key.toLowerCase()];
          if (direction) {
            event.preventDefault();
            event.stopPropagation();
            if (clientRef.current?.direction === direction) act("stop");
          }
        }}>
        {BUTTONS.map(([direction, arrow, key, placement]) => (
          <button key={direction} type="button" aria-label={`Head ${direction}`}
            title={`Look ${direction} (hold ${key})`} disabled={controlsDisabled}
            aria-pressed={state.direction === direction}
            className={`h-11 w-14 rounded-xl border text-lg font-bold disabled:opacity-45 ${placement}`}
            style={{ touchAction: "none", borderColor: "var(--mc-border-strong)",
              backgroundColor: state.direction === direction ? "var(--mc-accent-soft)" : "var(--mc-surface-2)" }}
            onPointerDown={(event) => {
              if (event.button !== 0) return;
              event.preventDefault();
              event.currentTarget.focus();
              event.currentTarget.setPointerCapture?.(event.pointerId);
              act("start", direction);
            }}
            onPointerUp={() => act("stop")}
            onPointerCancel={() => act("stop")}
            onLostPointerCapture={() => act("stop")}
            onPointerLeave={() => act("stop")}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.code === "Space") {
                event.preventDefault(); event.stopPropagation();
                if (!event.repeat) act("start", direction);
              }
            }}
            onKeyUp={(event) => {
              if (event.key === "Enter" || event.code === "Space") {
                event.preventDefault(); event.stopPropagation(); act("stop");
              }
            }}>
            {arrow}<span className="ml-1 text-[10px]">{key}</span>
          </button>
        ))}
      </div>
      <ActionButton variant="secondary" disabled={controlsDisabled} onClick={() => act("stop")}>Stop head</ActionButton>
      <p className="text-xs m-0" style={{ color: MISSION_TEXT_MUTED }}>
        Hold a direction to move slowly. Release to stop.<br />
        Focus the controls to use I / J / K / L.<br />
        Pitch −0.232…0.830 · Yaw −0.350…0.350 rad
      </p>
    </Panel>
  );
}
