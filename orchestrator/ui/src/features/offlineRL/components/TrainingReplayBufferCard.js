// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';

export const DEFAULT_TRAINING_REPLAY_CAPACITY = 200;

const nonNegativeInteger = (...values) => {
  const value = values.find((candidate) => (
    candidate !== null && candidate !== undefined && candidate !== ''
  ));
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? Math.floor(number) : null;
};

const outcomeFromEpisode = (episode) => {
  const outcome = String(episode?.outcome || '').trim().toLowerCase();
  if (['success', 'successful'].includes(outcome)) return 'success';
  if (['failure', 'fail', 'failed'].includes(outcome)) return 'failure';
  if (episode?.episode_success === true || episode?.success === true) return 'success';
  if (episode?.episode_success === false || episode?.success === false) return 'failure';
  return 'unlabeled';
};

const datasetName = (dataset, path) => {
  const explicit = String(dataset?.name || dataset?.dataset_name || '').trim();
  if (explicit) return explicit;
  const segments = path.split('/').filter(Boolean);
  return segments[segments.length - 1] || 'Unnamed dataset';
};

/**
 * Accepts both an inventory summary and the smaller datasetSelections shape.
 * Missing episode metadata remains explicitly unknown; it is never fabricated.
 */
export function normalizeTrainingReplayDatasets(values = []) {
  const seen = new Set();
  return (Array.isArray(values) ? values : []).flatMap((dataset) => {
    const path = String(dataset?.path || dataset?.dataset_path || '').trim();
    if (!path || seen.has(path)) return [];
    seen.add(path);

    const episodes = Array.isArray(dataset?.episodes) ? dataset.episodes : [];
    const episodeCounts = episodes.reduce((counts, episode) => {
      counts[outcomeFromEpisode(episode)] += 1;
      return counts;
    }, { success: 0, failure: 0, unlabeled: 0 });
    const hasEpisodeRows = episodes.length > 0;
    const successCount = nonNegativeInteger(
      dataset?.successCount,
      dataset?.success_count,
      hasEpisodeRows ? episodeCounts.success : null
    );
    const failureCount = nonNegativeInteger(
      dataset?.failureCount,
      dataset?.failure_count,
      hasEpisodeRows ? episodeCounts.failure : null
    );
    const explicitUnlabeledCount = nonNegativeInteger(
      dataset?.unlabeledCount,
      dataset?.unlabeled_count,
      hasEpisodeRows ? episodeCounts.unlabeled : null
    );
    const providedTotal = nonNegativeInteger(
      dataset?.totalEpisodes,
      dataset?.total_episodes,
      dataset?.episodeCount,
      dataset?.episode_count
    );
    const inferredUnlabeledCount = (
      providedTotal !== null && successCount !== null && failureCount !== null
    ) ? Math.max(0, providedTotal - successCount - failureCount) : null;
    const unlabeledCount = explicitUnlabeledCount ?? inferredUnlabeledCount;
    const knownOutcomes = [successCount, failureCount, unlabeledCount]
      .every((count) => count !== null);
    const outcomeTotal = knownOutcomes
      ? successCount + failureCount + unlabeledCount
      : null;
    const totalEpisodes = providedTotal ?? outcomeTotal;
    const metadataKnown = (
      totalEpisodes !== null &&
      knownOutcomes &&
      totalEpisodes === outcomeTotal
    );

    return [{
      path,
      name: datasetName(dataset, path),
      version: String(dataset?.version || dataset?.codebase_version || ''),
      dataEpoch: dataset?.dataEpoch ?? dataset?.data_epoch ?? null,
      metadataKnown,
      totalEpisodes,
      successCount,
      failureCount,
      unlabeledCount,
    }];
  });
}

const formatEpisodes = (count) => (
  `${count} episode${count === 1 ? '' : 's'}`
);

const OUTCOME_PRESENTATION = {
  success: {
    label: 'Success',
    color: '#6f9277',
    softColor: '#e6eee5',
    textColor: '#45634d',
  },
  failure: {
    label: 'Failure',
    color: '#d87969',
    softColor: '#f8e5e0',
    textColor: '#9b4f43',
  },
  unlabeled: {
    label: 'Unlabeled',
    color: '#c4ad87',
    softColor: '#eee5d5',
    textColor: '#7a684e',
  },
};

function OutcomeDetail({ outcome, datasets, onInspectDataset, detailRef }) {
  const presentation = OUTCOME_PRESENTATION[outcome];
  const countKey = `${outcome}Count`;
  const matching = datasets.filter((dataset) => Number(dataset[countKey]) > 0);
  const total = matching.reduce((sum, dataset) => sum + dataset[countKey], 0);

  return (
    <div
      ref={detailRef}
      role="status"
      aria-live="polite"
      className="absolute left-1/2 top-1/2 z-20 w-52 -translate-x-1/2 -translate-y-1/2 rounded-xl border border-[#d7d0c4] bg-white p-3 text-left shadow-[0_12px_30px_rgba(76,66,52,0.16)]"
      data-testid="training-replay-outcome-detail"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-[11px] font-semibold text-[#3e3932]">
          <span
            className="h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: presentation.color }}
            aria-hidden="true"
          />
          {presentation.label} data
        </span>
        <span className="text-[10px] font-semibold" style={{ color: presentation.textColor }}>
          {formatEpisodes(total)}
        </span>
      </div>
      <div className="mt-2 max-h-28 space-y-1 overflow-y-auto pr-0.5">
        {matching.map((dataset) => {
          const content = (
            <>
              <span className="min-w-0 flex-1 truncate" title={dataset.path}>
                {dataset.name}
              </span>
              <span className="shrink-0 font-semibold text-[#625b50]">
                {formatEpisodes(dataset[countKey])}
              </span>
            </>
          );
          return onInspectDataset ? (
            <button
              key={dataset.path}
              type="button"
              onClick={() => onInspectDataset(dataset, outcome)}
              className="flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-[9px] text-[#71695e] hover:bg-[#f4f0e8] focus:outline-none focus:ring-1 focus:ring-[#8da391]"
            >
              {content}
            </button>
          ) : (
            <div
              key={dataset.path}
              className="flex items-center gap-2 rounded-md px-1.5 py-1 text-[9px] text-[#71695e]"
            >
              {content}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function TrainingReplayBufferCylinder({
  datasets = [],
  capacityEpisodes = DEFAULT_TRAINING_REPLAY_CAPACITY,
  onInspectDataset = null,
}) {
  const normalizedDatasets = useMemo(
    () => normalizeTrainingReplayDatasets(datasets),
    [datasets]
  );
  const [activeOutcome, setActiveOutcome] = useState(null);
  const [pinnedOutcome, setPinnedOutcome] = useState(null);
  const rootRef = useRef(null);
  const detailRef = useRef(null);
  const capacity = Math.max(1, nonNegativeInteger(capacityEpisodes) || 1);
  const knownDatasets = normalizedDatasets.filter((dataset) => dataset.metadataKnown);
  const totals = knownDatasets.reduce((result, dataset) => ({
    success: result.success + dataset.successCount,
    failure: result.failure + dataset.failureCount,
    unlabeled: result.unlabeled + dataset.unlabeledCount,
    episodes: result.episodes + dataset.totalEpisodes,
  }), {
    success: 0, failure: 0, unlabeled: 0, episodes: 0,
  });
  const fillPercent = Math.min(100, (totals.episodes / capacity) * 100);
  const displayedOutcome = pinnedOutcome || activeOutcome;
  const unknownDatasetCount = normalizedDatasets.length - knownDatasets.length;

  const visiblePercent = (count) => (
    totals.episodes > 0 ? (count / totals.episodes) * fillPercent : 0
  );
  const percentages = {
    success: visiblePercent(totals.success),
    failure: visiblePercent(totals.failure),
    unlabeled: visiblePercent(totals.unlabeled),
  };
  const offsets = {
    success: 0,
    failure: percentages.success,
    unlabeled: percentages.success + percentages.failure,
  };

  const showOutcome = (outcome) => {
    if (!pinnedOutcome) setActiveOutcome(outcome);
  };
  const hideOutcome = (outcome) => {
    if (!pinnedOutcome && activeOutcome === outcome) setActiveOutcome(null);
  };
  const toggleOutcome = (outcome) => {
    if (pinnedOutcome === outcome) {
      setPinnedOutcome(null);
      setActiveOutcome(null);
      return;
    }
    setPinnedOutcome(outcome);
    setActiveOutcome(outcome);
  };

  useEffect(() => {
    if (!pinnedOutcome) return undefined;
    const handleOutsideMouseDown = (event) => {
      const target = event.target;
      if (detailRef.current?.contains(target)) return;
      const outcomeTrigger = target?.closest?.('[data-training-replay-outcome]');
      if (outcomeTrigger && rootRef.current?.contains(outcomeTrigger)) return;
      setPinnedOutcome(null);
      setActiveOutcome(null);
    };
    document.addEventListener('mousedown', handleOutsideMouseDown);
    return () => document.removeEventListener('mousedown', handleOutsideMouseDown);
  }, [pinnedOutcome]);

  return (
    <div
      ref={rootRef}
      className="relative flex min-w-0 items-center justify-center gap-5"
      data-testid="training-replay-buffer-cylinder"
      onKeyDown={(event) => {
        if (event.key === 'Escape') {
          setPinnedOutcome(null);
          setActiveOutcome(null);
        }
      }}
    >
      <div className="min-w-[124px] text-right">
        <div className="text-[28px] font-semibold tracking-[-0.04em] text-[#38342e]">
          {Math.round(fillPercent)}%
        </div>
        <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-[#999083]">
          buffer filled
        </div>
        <div className="mt-2 text-[11px] font-medium text-[#6f675c]">
          {totals.episodes} / {capacity} episodes
        </div>
        {unknownDatasetCount > 0 && (
          <div className="mt-1 max-w-[128px] text-[9px] leading-3 text-[#a06c4e]">
            {unknownDatasetCount} dataset{unknownDatasetCount === 1 ? '' : 's'} awaiting episode metadata
          </div>
        )}
      </div>

      <div className="relative h-48 w-40 shrink-0" aria-label="Training replay buffer fill">
        <div
          className="absolute inset-x-1 bottom-4 top-4 overflow-hidden rounded-b-[42%] border-x border-b border-[#bfb6a8] bg-[#f6f1e7]/80 shadow-[inset_8px_0_14px_rgba(75,66,51,0.05),inset_-8px_0_14px_rgba(75,66,51,0.06)]"
          data-testid="training-replay-buffer-body"
        >
          {['success', 'failure', 'unlabeled'].map((outcome) => {
            const count = totals[outcome];
            if (!count) return null;
            const presentation = OUTCOME_PRESENTATION[outcome];
            return (
              <button
                key={outcome}
                type="button"
                aria-label={`Inspect ${outcome} datasets: ${formatEpisodes(count)}`}
                aria-pressed={pinnedOutcome === outcome}
                data-training-replay-outcome={outcome}
                onMouseEnter={() => showOutcome(outcome)}
                onMouseLeave={() => hideOutcome(outcome)}
                onFocus={() => showOutcome(outcome)}
                onBlur={() => hideOutcome(outcome)}
                onClick={() => toggleOutcome(outcome)}
                className="absolute inset-x-0 cursor-pointer border-0 p-0 outline-none transition-[filter] hover:brightness-105 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-white/80"
                style={{
                  bottom: `${offsets[outcome]}%`,
                  height: `${percentages[outcome]}%`,
                  background: `linear-gradient(90deg, ${presentation.color} 0%, ${presentation.softColor} 52%, ${presentation.color} 100%)`,
                }}
              />
            );
          })}
          <div className="pointer-events-none absolute inset-y-0 left-[15%] w-[12%] bg-white/10 blur-sm" />
        </div>
        <div className="pointer-events-none absolute inset-x-1 top-0 h-8 rounded-[50%] border border-[#bfb6a8] bg-[#fbf8f1]/90 shadow-[inset_0_-4px_9px_rgba(75,66,51,0.08)]" />
        <div className="pointer-events-none absolute inset-x-1 bottom-0 h-8 rounded-[50%] border border-[#bfb6a8] bg-transparent shadow-[0_8px_14px_rgba(75,66,51,0.10)]" />
        {displayedOutcome && totals[displayedOutcome] > 0 && (
          <OutcomeDetail
            outcome={displayedOutcome}
            datasets={knownDatasets}
            onInspectDataset={onInspectDataset}
            detailRef={detailRef}
          />
        )}
      </div>
    </div>
  );
}

export default function TrainingReplayBufferCard({
  datasets = [],
  capacityEpisodes = DEFAULT_TRAINING_REPLAY_CAPACITY,
  onInspectDataset = null,
  className = '',
}) {
  const normalizedDatasets = useMemo(
    () => normalizeTrainingReplayDatasets(datasets),
    [datasets]
  );
  const totals = normalizedDatasets.reduce((result, dataset) => ({
    success: result.success + (dataset.successCount || 0),
    failure: result.failure + (dataset.failureCount || 0),
    unlabeled: result.unlabeled + (dataset.unlabeledCount || 0),
  }), { success: 0, failure: 0, unlabeled: 0 });

  return (
    <section
      className={clsx(
        'min-w-0 rounded-2xl border border-[#d8d1c5] bg-white p-4 shadow-[0_8px_24px_rgba(75,66,51,0.07)]',
        className
      )}
      aria-labelledby="training-replay-buffer-title"
      data-testid="training-replay-buffer-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-[#9a9286]">
            Dataset
          </div>
          <h3 id="training-replay-buffer-title" className="mt-0.5 text-[14px] font-semibold text-[#38342e]">
            Replay Buffer
          </h3>
          <p className="mt-1 text-[10px] text-[#8b8378]">
            Deployed LeRobot data · hover or click a color
          </p>
        </div>
        <span className="rounded-full bg-[#eeeae2] px-2.5 py-1 text-[9px] font-semibold text-[#71695f]">
          {normalizedDatasets.length} dataset{normalizedDatasets.length === 1 ? '' : 's'}
        </span>
      </div>

      <div className="mt-2 flex h-[190px] min-h-0 items-center justify-center pr-20">
        <TrainingReplayBufferCylinder
          datasets={datasets}
          capacityEpisodes={capacityEpisodes}
          onInspectDataset={onInspectDataset}
        />
      </div>

      <div className="mt-2 flex flex-wrap items-center justify-center gap-x-4 gap-y-2 border-t border-[#eee9df] pt-3">
        {Object.entries(OUTCOME_PRESENTATION).map(([outcome, presentation]) => (
          <span key={outcome} className="flex items-center gap-2 text-[11px] font-medium leading-3 text-[#756d62]">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: presentation.color }}
              aria-hidden="true"
            />
            {presentation.label} {totals[outcome]}
          </span>
        ))}
        <span className="flex items-center gap-2 text-[11px] font-medium leading-3 text-[#756d62]">
          <span className="h-2.5 w-2.5 rounded-full border border-[#d8d1c5] bg-[#f6f1e7]" aria-hidden="true" />
          Empty
        </span>
      </div>
    </section>
  );
}
