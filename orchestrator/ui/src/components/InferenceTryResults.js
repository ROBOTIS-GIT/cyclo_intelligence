import React, { useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import { MdDeleteOutline, MdExpandLess, MdExpandMore } from 'react-icons/md';
import { selectInferenceTaskInfo } from '../features/tasks/taskSlice';

const emptyHistory = () => ({ rows: [], file: '', revision: '' });

async function readResponse(response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Trial history request failed.');
  }
  return data;
}

export default function InferenceTryResults() {
  const info = useSelector(selectInferenceTaskInfo);
  const loaded = useSelector(state => state.tasks.inferenceStatus.loadedModelPath);
  const model = (loaded || info.policyPath || '').trim().replace(/\/+$/, '');
  const [data, setData] = useState(emptyHistory);
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState([]);
  const generation = useRef(0);
  const pending = useRef(false);

  useEffect(() => { setExpanded(false); }, [model]);
  useEffect(() => {
    const id = ++generation.current;
    const controller = new AbortController();
    pending.current = false;
    setBusy(false);
    setReady(false);
    setData(emptyHistory());
    setSelected([]);
    setError('');
    if (model) {
      fetch(`/api/try-results?model=${encodeURIComponent(model)}`, {
        signal: controller.signal, cache: 'no-store',
      }).then(readResponse).then(history => {
        if (generation.current === id) { setData(history); setReady(true); }
      }).catch(failure => {
        if (failure.name !== 'AbortError' && generation.current === id) setError(failure.message);
      });
    }
    return () => { generation.current += 1; controller.abort(); };
  }, [model]);

  async function save(result, attempt, remove = false) {
    if (pending.current || !ready) return;
    if (remove && (!selected.length || !window.confirm(
      `Delete ${selected.length} selected trials?\n${model}\nThis cannot be undone.`
    ))) return;
    const id = generation.current;
    pending.current = true;
    setBusy(true);
    setError('');
    const body = remove
      ? { model, tries: selected, revision: data.revision }
      : { model, result, ...(attempt ? { try: attempt, revision: data.revision } : {}) };
    try {
      const history = await fetch('/api/try-results', {
        method: remove ? 'DELETE' : attempt ? 'PATCH' : 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      }).then(readResponse);
      if (generation.current === id) {
        setData(history);
        setSelected([]);
      }
    } catch (failure) {
      if (generation.current === id) setError(failure.message);
    } finally {
      if (generation.current === id) { pending.current = false; setBusy(false); }
    }
  }

  const successes = data.rows.filter(row => row.result === 'success').length;
  const resultButton = 'h-8 min-w-0 rounded-md px-2 text-xs font-semibold disabled:bg-gray-200 disabled:text-gray-400';
  const iconButton = 'flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-gray-500 hover:bg-gray-100 disabled:text-gray-300';
  return (
    <section aria-label="Try Results" className="mb-2 text-sm">
      <div className="flex items-center">
        <span className="w-28 shrink-0 font-medium text-gray-600">Try Results</span>
        <div className="flex min-w-0 flex-1 items-center gap-1">
        <button type="button" disabled={!ready || busy}
          className={`${resultButton} bg-green-100 text-green-700 hover:bg-green-200`}
          onClick={() => save('success')}>Success</button>
        <button type="button" disabled={!ready || busy}
          className={`${resultButton} bg-red-100 text-red-700 hover:bg-red-200`}
          onClick={() => save('fail')}>Fail</button>
        <button type="button" aria-label="Toggle try results" aria-expanded={expanded}
          aria-controls="try-results-history" title={expanded ? 'Hide try results' : 'Show try results'}
          className={`${iconButton} ml-auto`} onClick={() => setExpanded(value => !value)}>
          {expanded ? <MdExpandLess size={20} /> : <MdExpandMore size={20} />}
        </button>
        </div>
      </div>
      {error && <p role="alert" className="mt-1 break-words text-xs text-red-600">{error}</p>}
      {expanded && (
        <div id="try-results-history">
          {ready && (
            <div className="my-2 flex items-center justify-between gap-2 text-xs text-gray-600">
              <span>{data.rows.length} trials · {successes} success / {data.rows.length - successes} fail · {data.rows.length ? Math.round(successes / data.rows.length * 100) : 0}%</span>
              <button type="button" aria-label="Delete selected trials" title="Delete selected trials"
                disabled={busy || !selected.length} className={iconButton}
                onClick={() => save(undefined, undefined, true)}><MdDeleteOutline size={18} /></button>
            </div>
          )}
          {!!data.rows.length && (
            <div className="max-h-36 overflow-auto rounded-md border border-gray-200">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 text-gray-500"><tr>
                  <th className="w-6 p-1"><input type="checkbox" aria-label="Select all trials"
                    disabled={busy || !ready} checked={selected.length === data.rows.length}
                    onChange={event => setSelected(event.target.checked ? data.rows.map(row => row.try) : [])} /></th>
                  <th className="p-1 text-left">Try</th><th className="p-1 text-left">Time</th><th className="p-1 text-left">Result</th>
                </tr></thead>
                <tbody>{[...data.rows].reverse().map(row => (
                  <tr key={row.try} className="border-t border-gray-100">
                    <td className="p-1"><input type="checkbox" aria-label={`Select try ${row.try}`}
                      disabled={busy || !ready} checked={selected.includes(row.try)}
                      onChange={event => setSelected(values => event.target.checked
                        ? [...values, row.try] : values.filter(value => value !== row.try))} /></td>
                    <td className="p-1">#{row.try}</td>
                    <td className="p-1" title={row.created_at}>{new Date(row.created_at).toLocaleString()}</td>
                    <td className="p-1"><select aria-label={`Try ${row.try} result`} disabled={busy || !ready}
                      value={row.result} onChange={event => save(event.target.value, row.try)}
                      className={`rounded border border-gray-200 p-1 ${row.result === 'success' ? 'text-green-700' : 'text-red-600'}`}>
                      <option value="success">Success</option><option value="fail">Fail</option>
                    </select></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
