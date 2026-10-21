import React, { useEffect, useState } from 'react';
import { defaultInferenceChunkSize, inferenceChunkSizeError } from '../utils/inferenceChunks';

export default function InferenceChunkControl({ info, onChange, readonly }) {
  const [checkpoint, setCheckpoint] = useState(null);
  const defaultSize = defaultInferenceChunkSize(info.serviceType, info.policyType);
  useEffect(() => {
    setCheckpoint(null);
    if (!defaultSize || !info.policyPath?.trim()) return undefined;
    const controller = new AbortController();
    const params = new URLSearchParams({ model_path: info.policyPath.trim() });
    fetch(`/api/backends/${info.serviceType}/inference/options?${params}`, { signal: controller.signal })
      .then(async (r) => { if (r.ok) return r.json(); return null; })
      .then((data) => { if (!controller.signal.aborted) setCheckpoint(data); })
      .catch(() => {});
    return () => controller.abort();
  }, [defaultSize, info.policyPath, info.serviceType]);
  if (!defaultSize) return null;
  const error = inferenceChunkSizeError(info);
  const used = checkpoint?.max_chunk_size && !error
    ? Math.min(Number(info.actionChunkSize), checkpoint.max_chunk_size) : null;
  return (
    <div className="mb-2.5">
      <div className="flex items-center">
        <label htmlFor="inference-chunk-size" className="text-sm text-gray-600 w-28 flex-shrink-0 font-medium">Chunk Size</label>
        <input id="inference-chunk-size" type="number" min="1" max="65535" step="1"
          list="inference-chunk-presets" aria-label="Chunk Size" aria-invalid={Boolean(error)}
          value={info.actionChunkSize ?? defaultSize} disabled={readonly}
          className="flex-1 w-full h-8 px-2 border border-gray-300 rounded-md disabled:bg-gray-100"
          onChange={(e) => onChange('actionChunkSize', e.target.value === '' ? '' : Number(e.target.value))} />
        <datalist id="inference-chunk-presets">
          {[1, 4, 8, 16, 24, 30, 50, 100].map((n) => <option key={n} value={n} />)}
        </datalist>
      </div>
      <p className="mt-1 text-xs text-gray-500" aria-live="polite">
        {error || (used
          ? `Checkpoint horizon: ${checkpoint.max_chunk_size}. Used: ${used}.`
          : `Default: ${defaultSize}. Limited to the checkpoint's trained horizon.`)}
        {!error && checkpoint?.temporal_ensemble && ` Temporal ensemble executes 1 action per inference using ${used || checkpoint.max_chunk_size} prediction steps.`}
      </p>
    </div>
  );
}
