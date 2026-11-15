import React, { useEffect, useRef, useState } from 'react';

const COMMAND = { STATUS: 8, AUTO_ON: 9, AUTO_OFF: 10, APPLY: 11, TRAIN_ON: 12, TRAIN_OFF: 13, SAVE: 15 };

async function requestStatus(callService, bundlePath, command, extra = {}) {
  const response = await callService('/groot/inference_command',
    'interfaces/srv/InferenceCommand', { command, rlt_bundle_path: bundlePath, ...extra });
  if (!response.success) throw new Error(response.message || 'Policy update failed');
  const result = JSON.parse(response.message);
  if (typeof result.auto_apply !== 'boolean' || typeof result.can_apply !== 'boolean') {
    throw new Error('Unsupported policy update response');
  }
  return result;
}

export default function RltPolicyUpdateControl({ callService, available, bundlePath }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [maxUpdates, setMaxUpdates] = useState('1000');
  useEffect(() => { setMaxUpdates(String(status?.max_updates || 1000)); }, [bundlePath, status?.max_updates]);
  const sequence = useRef({ value: 0 }).current;
  const mutating = useRef(false);

  useEffect(() => {
    let active = true;
    let timer;
    setStatus(null);
    setError('');
    setBusy(false);
    mutating.current = false;
    const poll = async () => {
      if (!active) return;
      if (!mutating.current) {
        const ticket = ++sequence.value;
        try {
          const result = await requestStatus(callService, bundlePath, COMMAND.STATUS);
          if (active && ticket === sequence.value) { setStatus(result); setError(''); }
        } catch (failure) {
          if (active && ticket === sequence.value) { setStatus(null); setError(failure.message); }
        }
      }
      if (active) timer = setTimeout(poll, 2000);
    };
    if (available && bundlePath && typeof callService === 'function') poll();
    return () => { active = false; ++sequence.value; clearTimeout(timer); };
  }, [available, bundlePath, callService, sequence]);

  const command = async (value) => {
    const ticket = ++sequence.value;
    mutating.current = true;
    setBusy(true);
    try {
      const result = await requestStatus(callService, bundlePath, value,
        value === COMMAND.TRAIN_ON ? { rlt_max_updates: Number(maxUpdates) } : {});
      if (ticket === sequence.value) { setStatus(result); setError(''); }
    } catch (failure) {
      if (ticket === sequence.value) { setStatus(null); setError(failure.message); }
    } finally {
      if (ticket === sequence.value) { mutating.current = false; setBusy(false); }
    }
  };
  const ready = available && status && !busy;
  const budgetSupported = Number.isInteger(status?.max_updates);
  const validLimit = Number.isInteger(Number(maxUpdates)) && Number(maxUpdates) >= 1 && Number(maxUpdates) <= 4294967295;
  const style = 'rounded-lg border border-[#d9d2c5] px-3 py-1.5 text-xs font-semibold disabled:opacity-50 disabled:cursor-not-allowed';
  return (
    <div className="mt-2 border-t border-[#e2dcd1] pt-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-[#6e675c]">MLP Policy Update</span>
        <button type="button" role="switch" aria-label="Async RL"
          aria-checked={Boolean(status?.async_enabled)} disabled={!ready || (!status?.async_enabled && (!budgetSupported || !validLimit))}
          onClick={() => command(status.async_enabled ? COMMAND.TRAIN_OFF : COMMAND.TRAIN_ON)}
          className={`${style} ${status?.async_enabled ? 'bg-[#71806b] text-white' : 'bg-[#fffefa] text-[#6e675c]'}`}
          title="Train on selected LeRobot datasets; an empty selection waits for data. Stage an MLP every 10 actor updates. Save Bundle persists training separately from Apply Policy.">
          Async RL {status?.async_enabled ? 'ON' : 'OFF'}
        </button>
        <label className="flex items-center gap-2 text-xs text-[#6e675c]"
          title="Critic updates per ON run. At the limit, finish the actor/target phase and pause. ON again starts a new budget without resetting weights or optimizer.">
          Max updates
          <input type="number" min="1" max="4294967295" step="1" aria-label="Async RL max updates"
            value={maxUpdates} onChange={(event) => setMaxUpdates(event.target.value)}
            disabled={!ready || status?.async_enabled || !budgetSupported}
            className="w-24 rounded-lg border border-[#d9d2c5] bg-[#fffefa] px-2 py-1.5 disabled:opacity-50" />
        </label>
        <button type="button" role="switch" aria-label="Auto Apply"
          aria-checked={Boolean(status?.auto_apply)} disabled={!ready}
          onClick={() => command(status.auto_apply ? COMMAND.AUTO_OFF : COMMAND.AUTO_ON)}
          className={`${style} ${status?.auto_apply ? 'bg-[#71806b] text-white' : 'bg-[#fffefa] text-[#6e675c]'}`}
          title="Apply new MLP snapshots at the next RLT request; this does not start training">
          Auto Apply {status?.auto_apply ? 'ON' : 'OFF'}
        </button>
        <button type="button" className={`${style} bg-[#fffefa] text-[#6e675c]`}
          disabled={!ready || status.auto_apply || !status.can_apply || status.pending_version != null}
          onClick={() => command(COMMAND.APPLY)}>Apply Policy</button>
        <button type="button" className={`${style} bg-[#fffefa] text-[#6e675c]`}
          disabled={!ready || status.saving || !status.last_update}
          title="Save the learner, critic, optimizer and replay to a new bundle. Does not switch the inference policy."
          onClick={() => command(COMMAND.SAVE)}>{status?.saving ? 'Saving…' : 'Save Bundle'}</button>
      </div>
      <div className="mt-1 text-xs text-[#81786b]" role="status" title={error}>
        {error ? `Unavailable: ${error}` : status
          ? status.training_error || status.replay_error || (status.preparing ? 'Preparing Async RL…' : `Staged v${status.training_version} · Inference v${status.inference_version}${status.pending_version != null ? ` · Pending v${status.pending_version}` : ''}`)
          : 'Load GR00T with an RLT bundle to use policy updates'}
      </div>
      {status && !error && (
        <div className="mt-1 text-xs text-[#81786b]">
          {!budgetSupported && <div role="alert">Update the GR00T runtime and interfaces to enable bounded Async RL.</div>}
          {budgetSupported && <div>
            Run: {status.updates_this_run || 0} / {status.max_updates} critic updates
            {' · '}Total Critic {status.total_critic_updates ?? status.last_update?.completed_critic_updates ?? '—'}
            {' / '}Actor {status.total_actor_updates ?? status.last_update?.completed_actor_updates ?? '—'}
            {status.limit_reached && ' · Limit reached — paused'}
          </div>}
          <div title={status.replay_source?.paths?.join('\n')}>
            Data: selected LeRobot datasets
            {status.replay_source?.episodes != null && ` · ${status.replay_source.episodes} episodes`}
            {status.replay_source?.transitions != null && ` · ${status.replay_source.transitions} transitions`}
            {status.replay_source?.batch_size != null && ` · batch ${status.replay_source.batch_size}`}
          </div>
          <div>{status.waiting_data ? 'Waiting for selected data. ' : ''}Use Save Bundle to keep training across restarts.</div>
          {status.save_error && <div role="alert">Save failed: {status.save_error}</div>}
          {status.saved_bundle_path && <div className="break-all select-text" title="Select this RLT Bundle Path after stopping and reloading to resume training.">
            Saved · Critic {status.saved_critic_updates} / Actor {status.saved_actor_updates}<br />{status.saved_bundle_path}
          </div>}
          {status.can_apply && (
            <div>{status.auto_apply || status.pending_version != null
              ? 'Waiting for the next RLT request to apply. VLA requests do not apply the MLP.'
              : 'Candidate ready; enable Auto Apply or select Apply Policy, then use RLT Action.'}</div>
          )}
        </div>
      )}
    </div>
  );
}
