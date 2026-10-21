import React, { useEffect, useRef, useState } from 'react';

export default function CheckpointInstructionControl({ info, readonly, autoApply, languageConditioned,
  running = false, inferencing = false, transitioning = false, onSelect, onComplete, onProfile }) {
  const [result, setResult] = useState(null);
  const path = (info.policyPath || '').trim();
  const key = JSON.stringify([info.serviceType, path]);
  const current = info.taskInstruction?.[0] || '';
  const latest = useRef();
  latest.current = { key, current, readonly, autoApply, onSelect, onProfile, languageConditioned, appliedKey: info.checkpointInstructionKey };

  useEffect(() => {
    if (!path || !['groot', 'lerobot'].includes(info.serviceType)) return undefined;
    const controller = new AbortController();
    const instructionAtRequest = latest.current.current;
    setResult({ key, status: 'loading' });
    const timer = setTimeout(async () => {
      try {
        const params = new URLSearchParams({ model_path: path });
        const response = await fetch(`/api/backends/${info.serviceType}/inference/instructions?${params}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error('Checkpoint instructions unavailable');
        const data = await response.json();
        if (controller.signal.aborted || latest.current.key !== key) return;
        setResult({ key, status: 'ready', data });
        latest.current.onProfile?.({ key, data });
        const live = latest.current;
        // A delayed metadata reply may never replace a user's edit or change a
        // running policy's prompt. Remounting also preserves the selected phase.
        if (data.available && data.default_instruction && data.conditioning !== 'unknown'
          && live.languageConditioned && live.autoApply && live.current === instructionAtRequest
          && live.appliedKey !== key) {
          const known = [...(data.task_instructions || []), ...(data.subtask_instruction_choices || []).flat()];
          const canPreserve = data.conditioning === 'subtask'
            ? (data.subtask_instruction_choices || []).flat().includes(live.current)
            : known.includes(live.current);
          live.onSelect(canPreserve ? live.current : data.default_instruction);
        }
      } catch (error) {
        if (!controller.signal.aborted && latest.current.key === key) {
          setResult({ key, status: 'unavailable' });
          latest.current.onProfile?.({ key, data: null });
        }
      }
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [key, path, info.serviceType]);

  if (!path || !['groot', 'lerobot'].includes(info.serviceType)) return null;
  const matching = result?.key === key ? result : null;
  const data = matching?.data;
  const tasks = data?.task_instructions || [];
  const stages = data?.subtask_instruction_choices || [];
  const phasePrompts = stages.flat();
  const choices = data?.instruction_choices || (data?.conditioning === 'subtask' ? phasePrompts : tasks);
  const selected = choices.includes(current) ? current : '';
  const phased = info.serviceType === 'groot' && data?.conditioning === 'subtask' && stages.length > 1;
  const stageIndex = stages.findIndex((prompts) => prompts.includes(current));
  const nextInstruction = stageIndex >= 0 ? stages[stageIndex + 1]?.[0] : null;
  return (
    <div className="mb-3 p-2 border border-gray-200 rounded-md text-xs" aria-label="Checkpoint instructions">
      <div className="font-medium text-gray-700 mb-1">Training Instructions</div>
      {!matching || matching.status === 'loading' ? (
        <p role="status" className="text-gray-500">Reading checkpoint instructions…</p>
      ) : !data?.available ? (
        <p role="status" className="text-gray-500">Training instruction metadata unavailable. Enter the instruction manually.</p>
      ) : (
        <>
          {tasks.length > 0 && <div className="mb-2">
            <div className="text-gray-500">Overall instruction</div>
            {tasks.map((text) => <p key={text} className="mt-1 whitespace-pre-wrap">{text}</p>)}
          </div>}
          {stages.length > 0 && <div className="mb-2">
            <div className="text-gray-500">Subtasks ({stages.length})</div>
            <ol className="mt-1 space-y-1">
              {stages.map((prompts, index) => <li key={index}>
                <span className="font-medium">Subtask {index + 1}: </span>{prompts.join(' / ')}
              </li>)}
            </ol>
          </div>}
          {languageConditioned && choices.length > 0 && <label className="block">
            <span className="text-gray-500">{data.conditioning === 'subtask' ? 'Current subtask' : 'Trained instruction'}</span>
            <select aria-label="Checkpoint instruction selection" value={selected}
              disabled={readonly || (phased && running) || transitioning}
              onChange={(event) => event.target.value && onSelect(event.target.value)}
              className="mt-1 w-full border border-gray-300 rounded-md p-1 disabled:bg-gray-100">
              <option value="" disabled>Custom instruction</option>
              {choices.map((prompt) => <option key={prompt} value={prompt}>
                {data.conditioning === 'subtask' ? `Subtask ${stages.findIndex((stage) => stage.includes(prompt)) + 1}: ` : ''}{prompt}
              </option>)}
            </select>
          </label>}
          {languageConditioned && phased && (
            <div className="mt-2">
              {nextInstruction ? <>
                <p className="text-gray-500 mb-1">Next: {nextInstruction}</p>
                <button type="button" disabled={!inferencing || transitioning || !onComplete}
                  onClick={() => onComplete(nextInstruction)}
                  className="px-2 py-1 rounded bg-blue-500 text-white disabled:bg-gray-300">
                  {transitioning ? 'Switching subtask…' : 'Complete subtask & continue'}
                </button>
                <p className="mt-1 text-gray-500">Confirm completion before continuing. Pending actions are discarded before inference resumes with the next instruction.</p>
              </> : <p className="text-gray-500">{stageIndex >= 0
                ? 'Final subtask. Stop inference when this step is complete.'
                : 'Select a trained subtask instruction before starting inference.'}</p>}
            </div>
          )}
          {data.dataset?.repo_id && <p className="mt-2 text-gray-500 break-all">Dataset: {data.dataset.repo_id}</p>}
        </>
      )}
    </div>
  );
}
