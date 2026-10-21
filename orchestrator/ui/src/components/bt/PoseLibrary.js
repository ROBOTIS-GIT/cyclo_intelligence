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

import React, { useEffect, useRef, useState } from 'react';
import PoseCaptureButton from './PoseCaptureButton';
import { listBtTrees, readBtTree, saveBtTree } from '../../features/actionCanvas/btTreesApi';
import { applySg2PoseFile, parseSg2Pose, poseFilename, POSE_FILE_PREFIX, serializeSg2Pose } from '../../utils/sg2PoseFile';

export default function PoseLibrary({ nodeType, params, onApply }) {
  const [snapshot, setSnapshot] = useState(null);
  const [name, setName] = useState('');
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState('');
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const mounted = useRef(false);
  const signature = JSON.stringify(params);
  const currentSignature = useRef(signature);
  currentSignature.current = signature;
  const inFlight = useRef(false);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);

  const run = async (action) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError('');
    setStatus('');
    try { await action(); }
    catch (err) { if (mounted.current) setError(err.message || 'Pose file operation failed.'); }
    finally { inFlight.current = false; if (mounted.current) setBusy(false); }
  };
  const save = () => run(async () => {
    const content = serializeSg2Pose(snapshot, name);
    const result = await saveBtTree({ filename: poseFilename(name), content, overwrite: false });
    if (mounted.current) setStatus(`Saved: ${result.path}`);
  });
  const browse = () => run(async () => {
    const result = await listBtTrees();
    if (!mounted.current) return;
    const poses = (result.trees || []).filter((file) => file.name.startsWith(POSE_FILE_PREFIX));
    setFiles(poses);
    setSelected(poses[0]?.name || '');
    setOpen(true);
    setStatus(poses.length ? `Pose folder: ${result.directory}` : 'No saved poses yet. Capture a pose and save it first.');
  });
  const load = () => run(async () => {
    const result = await readBtTree(selected);
    if (!mounted.current) return;
    if (currentSignature.current !== signature) throw new Error('Block parameters changed. Load the pose again.');
    const message = parseSg2Pose(result.content);
    const patch = applySg2PoseFile(nodeType, params, message);
    onApply(patch);
    setSnapshot(message);
    setStatus(`Applied: ${result.name}`);
    setOpen(false);
  });
  const buttonClass = 'px-2 py-1.5 border border-[var(--mc-border-strong,#d1d5db)] rounded-lg text-xs disabled:opacity-50';
  return (
    <div className="max-h-[45%] overflow-y-auto shrink-0">
      <PoseCaptureButton key={signature} nodeType={nodeType} params={params}
        onSnapshot={(message) => { setSnapshot(message); setStatus('Pose captured. Enter a name to save it.'); setError(''); }}
        onCapture={onApply} />
      <div className="px-4 py-2 space-y-2 border-b border-[var(--mc-border,#e5e7eb)]">
        <input aria-label="Pose name" placeholder="Pose name" value={name}
          onChange={(event) => setName(event.target.value)}
          className="w-full px-2 py-1.5 border rounded-lg text-sm bg-[var(--mc-surface,#fff)] text-[var(--mc-text)]" />
        <div className="flex gap-2">
          <button type="button" className={buttonClass} disabled={busy || !snapshot || !name.trim()} onClick={save}>Save pose</button>
          <button type="button" className={buttonClass} disabled={busy} onClick={browse}>Load pose</button>
        </div>
        <p className="text-xs text-[var(--mc-text-muted)]">Saves the full captured SG2 pose as BT XML. Loading applies selected joints and enabled groups; an empty selection uses both arms.</p>
        {open && <div className="space-y-2">
          <select aria-label="Saved pose" className="w-full text-sm bg-[var(--mc-surface,#fff)] text-[var(--mc-text)]"
            value={selected} onChange={(event) => setSelected(event.target.value)} disabled={busy}>
            {files.map((file) => <option key={file.name} value={file.name}>{file.name}</option>)}
          </select>
          <button type="button" className={buttonClass} disabled={busy || !selected} onClick={load}>Apply pose to block</button>
        </div>}
        {busy && <p role="status" className="text-xs">Working…</p>}
        {status && <p role="status" className="text-xs break-all">{status}</p>}
        {error && <p role="alert" className="text-xs text-red-600">{error}</p>}
      </div>
    </div>
  );
}
