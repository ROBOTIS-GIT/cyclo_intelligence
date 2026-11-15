import React, { useEffect, useRef, useState } from 'react';
import { shallowEqual, useSelector } from 'react-redux';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { selectInferenceTaskInfo } from '../features/tasks/taskSlice';
import { selectOfflineRLDatasetSelections } from '../features/offlineRL/offlineRLSlice';
import { supportsRltInference } from '../constants/policyCapabilities';

export function RltDatasetSelectionSync({ callService, bundlePath, paths, enabled }) {
  const currentPaths = useRef(paths);
  currentPaths.current = paths;
  const [message, setMessage] = useState('');
  useEffect(() => {
    let active = true;
    let timer;
    setMessage(enabled && bundlePath ? 'Async RL: checking selected data…' : '');
    const request = async (command, extra = {}) => {
      const response = await callService('/groot/inference_command',
        'interfaces/srv/InferenceCommand', { command, rlt_bundle_path: bundlePath, ...extra });
      if (!response.success) throw new Error(response.message);
      return JSON.parse(response.message);
    };
    const poll = async () => {
      try {
        let status = await request(8);
        if (!active) return;
        const selected = [...new Set(currentPaths.current)];
        if (JSON.stringify(status.selected_paths) !== JSON.stringify(selected)) {
          status = await request(14, { rlt_dataset_paths: selected });
        }
        if (active) setMessage(status.replay_error
          ? `Async RL: ${status.replay_error} (previous replay retained)`
          : status.preparing ? 'Async RL: preparing selected data…'
          : `Async RL: ${status.replay_source?.transitions || 0} active transitions · ${selected.length} datasets selected`);
      } catch (error) {
        if (active) setMessage(`Async RL data: ${error.message}`);
      }
      if (active) timer = setTimeout(poll, 1000);
    };
    if (enabled && bundlePath) poll();
    return () => { active = false; clearTimeout(timer); };
  }, [enabled, bundlePath, callService]);
  return enabled && message ? <div className="text-xs text-[#81786b]" role="status">{message}</div> : null;
}

function SelectionTransport(props) {
  const { callService } = useRosServiceCaller();
  return <RltDatasetSelectionSync {...props} callService={callService} enabled />;
}

export default function ConnectedRltDatasetSelectionSync() {
  const info = useSelector(selectInferenceTaskInfo);
  const selections = useSelector(selectOfflineRLDatasetSelections, shallowEqual);
  const supported = supportsRltInference(info.serviceType, info.policyType);
  const bundlePath = String(info.rltBundlePath || '').trim();
  const prerequisite = !supported ? 'select GR00T N1.7 in Inference Settings'
    : !info.rltEnabled ? 'enable RLT in Inference Settings'
    : !bundlePath ? 'select an RLT Bundle Path in Inference Settings' : '';
  if (prerequisite) return supported || selections.length
    ? <div className="text-xs text-[#81786b]" role="status">Async RL data: waiting — {prerequisite}</div>
    : null;
  return <SelectionTransport
    bundlePath={bundlePath}
    paths={selections.map((item) => String(item.path || '').trim()).filter(Boolean)} />;
}
