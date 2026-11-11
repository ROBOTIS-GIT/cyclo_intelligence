import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import toast from 'react-hot-toast';
import { MdExpandLess, MdExpandMore, MdHome, MdSave, MdStop } from 'react-icons/md';
import { InferencePhase } from '../constants/taskPhases';
import { selectInferenceTaskInfo } from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import './SavedPoseControlPanel.css';

export default function SavedPoseControlPanel() {
  const robotType = useSelector((state) => state.tasks.robotType);
  const rosbridgeUrl = useSelector((state) => state.ros?.rosbridgeUrl);
  const info = useSelector(selectInferenceTaskInfo);
  const phase = useSelector((state) => state.tasks.inferenceStatus.inferencePhase);
  const status = useSelector((state) => state.tasks.robotPoseStatus);
  const returning = Boolean(status?.returning);
  const { sendRobotPoseCommand } = useRosServiceCaller();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [durationDraft, setDurationDraft] = useState(null);
  const [expanded, setExpanded] = useState(false);
  const editingDuration = useRef(false);
  const returnButton = useRef(null);
  const pendingRef = useRef(false);
  const revision = useRef(0);
  const isRobot = info.inferenceMode === 'robot';
  const enabled = Boolean(robotType) && (isRobot || returning);
  const deviceId = status?.device_id;
  const duration = status?.robot_type === robotType ? (status.duration_s ?? 5) : 5;

  useEffect(() => {
    if (!editingDuration.current) setDurationDraft(null);
  }, [duration]);

  useEffect(() => {
    let cancelled = false;
    revision.current += 1;
    pendingRef.current = false;
    setPending(false);
    setError('');
    setDurationDraft(null);
    setExpanded(false);
    editingDuration.current = false;
    if (!enabled) return undefined;
    const load = async () => {
      const version = revision.current;
      try {
        const result = await sendRobotPoseCommand(0, robotType);
        if (result && !cancelled && version === revision.current) {
          setError(result.success ? '' : result.message || 'Pose status unavailable.');
        }
      } catch (failure) {
        if (!cancelled && version === revision.current) {
          setError(failure.message);
        }
      }
    };
    load();
    return () => { cancelled = true; revision.current += 1; };
  }, [enabled, robotType, sendRobotPoseCommand, deviceId, rosbridgeUrl]);

  const send = useCallback(async (command, durationS) => {
    if (pendingRef.current) return;
    pendingRef.current = true;
    setPending(true);
    const version = ++revision.current;
    try {
      const result = await (durationS === undefined
        ? sendRobotPoseCommand(command, robotType)
        : sendRobotPoseCommand(command, robotType, durationS));
      if (!result || version !== revision.current) return;
      if (!result.success) throw new Error(result.message || 'Pose command failed.');
      setError('');
      if (command !== 4) toast.success(result.message);
      return true;
    } catch (failure) {
      if (version === revision.current) { setError(failure.message); toast.error(failure.message); }
    } finally {
      if (version === revision.current) { pendingRef.current = false; setPending(false); }
    }
  }, [robotType, sendRobotPoseCommand]);

  const commitDuration = async () => {
    editingDuration.current = false;
    if (durationDraft === null || pendingRef.current) return;
    const value = Number(durationDraft);
    if (!Number.isFinite(value) || value < 1 || value > 60) {
      setError('Return duration must be between 1 and 60 seconds.');
      setDurationDraft(null);
      return;
    }
    if (value === duration) { setDurationDraft(null); return; }
    const request = send(4, value);
    const version = revision.current;
    if (!(await request) && version === revision.current) setDurationDraft(null);
  };

  const requestedDuration = Number(durationDraft ?? duration);
  const validDuration = Number.isFinite(requestedDuration) && requestedDuration >= 1 && requestedDuration <= 60;
  const returnToPose = async () => {
    if (pendingRef.current || !validDuration) return;
    editingDuration.current = false;
    // Persist a valid draft before motion, without waiting for the status topic
    // to echo it. RETURN still checks this expected duration on the backend.
    if (requestedDuration !== duration && !(await send(4, requestedDuration))) return;
    await send(2, requestedDuration);
  };

  if (!isRobot && !returning) return null;
  const validStatus = status?.available && status.robot_type === robotType;
  const unavailable = !validStatus || !status.connected || pending || returning;
  const inactive = [InferencePhase.READY, InferencePhase.PAUSED].includes(phase);
  const displayError = error || status?.error;
  const buttonClass = 'flex h-8 min-w-0 items-center justify-center gap-1 whitespace-nowrap rounded-md text-xs font-semibold text-white disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-400';
  return (
    <section aria-label="Saved Initial Pose" className="mb-2.5">
      <div className="flex items-center">
        <span className="w-28 flex-shrink-0 text-sm font-medium text-gray-600">Initial Pose</span>
        <div className="saved-pose-actions grid min-w-0 flex-1 grid-cols-[minmax(0,1fr)_3rem_minmax(0,1.3fr)_2rem] gap-1">
          <button type="button" aria-label="Save Initial Pose" disabled={unavailable || !inactive}
            onClick={() => send(1)} className={`${buttonClass} bg-emerald-600 hover:bg-emerald-700`}><MdSave size={16} className="saved-pose-action-icon shrink-0" /><span>Save</span></button>
          <label title="Return duration (seconds)" className="flex h-8 min-w-0 items-center gap-0.5 text-xs text-gray-500">
            <input type="number" aria-label="Return duration (seconds)" min="1" max="60" step="0.5"
              value={durationDraft ?? duration} disabled={!validStatus || pending || returning || !inactive}
              onFocus={() => { editingDuration.current = true; }}
              onChange={(event) => setDurationDraft(event.target.value)}
              onBlur={(event) => {
                editingDuration.current = false;
                if (event.relatedTarget !== returnButton.current) commitDuration();
              }}
              onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }}
              className="h-8 w-full min-w-0 appearance-none rounded-md border border-gray-300 px-1 text-xs text-gray-700 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none disabled:bg-gray-100 disabled:text-gray-400" />
            <span className="shrink-0">s</span>
          </label>
          <button ref={returnButton} type="button"
            aria-label={returning ? 'Stop pose return' : 'Return to Saved Pose'}
            disabled={returning ? pending : unavailable || !status?.saved || !inactive || !validDuration}
            onClick={returning ? () => send(3) : returnToPose}
            className={`${buttonClass} ${returning ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-500 hover:bg-blue-600'}`}>
            {returning ? <><MdStop size={16} className="saved-pose-action-icon shrink-0" /><span>Stop</span></> : <><MdHome size={16} className="saved-pose-action-icon shrink-0" /><span>Return</span></>}
          </button>
          <button type="button" aria-label="Toggle saved joint values"
            aria-expanded={expanded} aria-controls="saved-joint-values"
            title={expanded ? 'Hide saved joint values' : 'Show saved joint values'}
            disabled={!validStatus || !status?.saved}
            onClick={() => setExpanded(value => !value)}
            className="flex h-8 w-8 items-center justify-center rounded-md text-gray-500 hover:bg-gray-100 disabled:cursor-not-allowed disabled:text-gray-300">
            {expanded ? <MdExpandLess size={20} /> : <MdExpandMore size={20} />}
          </button>
        </div>
      </div>
      {displayError && (
        <div aria-live="polite" className="ml-28 mt-1 text-[11px] leading-snug text-gray-500">
          {displayError}
        </div>
      )}
      {expanded && validStatus && status.saved && (
          <div id="saved-joint-values" className="mt-2 max-h-48 overflow-auto rounded-md border border-gray-200 text-gray-500">
            <table className="w-full text-[10px]">
              <thead className="sticky top-0 bg-gray-50"><tr>
                <th className="px-2 py-1 text-left font-medium">Joint</th>
                <th className="px-2 py-1 text-right font-medium">Position</th>
              </tr></thead>
              <tbody>{status.joint_names.map((name, index) => (
                <tr key={name} className="border-t border-gray-100">
                  <td className="px-2 py-1 font-mono">{name}</td>
                  <td className="px-2 py-1 text-right font-mono whitespace-nowrap">{status.positions[index].toFixed(4)} {status.units[index]}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
      )}
    </section>
  );
}
