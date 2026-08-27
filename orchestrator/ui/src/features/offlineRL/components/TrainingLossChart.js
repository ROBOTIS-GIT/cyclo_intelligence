// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useRef, useState } from 'react';
import { MdOpenInFull } from 'react-icons/md';
import TrainingMetricsModal from './TrainingMetricsModal';

const finiteNumber = value => (
  typeof value === 'number' && Number.isFinite(value) ? value : null
);

const latestLoss = (history) => {
  if (!Array.isArray(history)) return null;
  return history.reduce((latest, point) => {
    const step = finiteNumber(point?.step);
    const loss = finiteNumber(point?.loss);
    if (step === null || loss === null) return latest;
    if (!latest || step >= latest.step) return { step, loss };
    return latest;
  }, null)?.loss ?? null;
};

const formatLoss = (value) => {
  if (value === null) return '—';
  const magnitude = Math.abs(value);
  if (magnitude > 0 && (magnitude < 0.001 || magnitude >= 1000)) {
    return value.toExponential(1);
  }
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  return value.toFixed(2);
};

const formatDuration = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const rounded = Math.round(seconds);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const remainingSeconds = rounded % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
  if (minutes > 0) return `${minutes}m ${String(remainingSeconds).padStart(2, '0')}s`;
  return `${remainingSeconds}s`;
};

const statusStyle = (status) => {
  switch (String(status || '').toLowerCase()) {
    case 'running':
      return 'border-[#aebead] bg-[#e7eee6] text-[#56705c]';
    case 'completed':
      return 'border-[#9faf9f] bg-[#dfe9df] text-[#405c47]';
    case 'failed':
      return 'border-[#d9aaa1] bg-[#f7e8e4] text-[#9c5145]';
    case 'stopped':
      return 'border-[#dfbd9f] bg-[#f6eadf] text-[#9b653e]';
    default:
      return 'border-[#d8d1c5] bg-[#f3f0e9] text-[#756e63]';
  }
};

const metricStyle = (tone) => {
  switch (tone) {
    case 'critic':
      return {
        card: 'border-[#ead4c5] bg-[#fff7f1]',
        label: 'text-[#a06c49]',
        value: 'text-[#6c4934]',
      };
    case 'actor':
      return {
        card: 'border-[#cfdbce] bg-[#f2f7f1]',
        label: 'text-[#66806b]',
        value: 'text-[#405c47]',
      };
    default:
      return {
        card: 'border-[#d7dbe5] bg-[#f4f6fa]',
        label: 'text-[#758098]',
        value: 'text-[#4b5874]',
      };
  }
};

/** Compact progress summary shared by ACT and the other policy workflows. */
export default function TrainingLossChart({
  actorLossHistory = [],
  criticLossHistory = [],
  metrics = null,
  percentage = 0,
  status = 'idle',
  displayStatus = '',
  durationSeconds = null,
  etaSeconds = null,
  showEta = false,
  detailLabel = '',
  progressLabel = 'Training loss progress',
  expandable = false,
  rlMetricHistory = [],
}) {
  const [metricsOpen, setMetricsOpen] = useState(false);
  const expandButtonRef = useRef(null);
  const safePercentage = Number.isFinite(percentage)
    ? Math.max(0, Math.min(100, percentage))
    : 0;
  const normalizedStatus = String(status || 'idle').trim() || 'idle';
  const visibleStatus = String(displayStatus || normalizedStatus).trim();
  const isEta = showEta || Number.isFinite(etaSeconds);
  const timingLabel = isEta
    ? `ETA ${formatDuration(etaSeconds)}`
    : formatDuration(durationSeconds);
  const actorLoss = latestLoss(actorLossHistory);
  const criticLoss = latestLoss(criticLossHistory);
  const visibleMetrics = Array.isArray(metrics) && metrics.length
    ? metrics
    : [
      {
        label: 'Critic loss',
        value: formatLoss(criticLoss),
        tone: 'critic',
        ariaLabel: 'Latest critic loss',
      },
      {
        label: 'Actor loss',
        value: formatLoss(actorLoss),
        tone: 'actor',
        ariaLabel: 'Latest actor loss',
      },
    ];

  return (
    <section
      className="min-w-0 rounded-xl border border-[#ddd6ca] bg-white p-3 shadow-[0_1px_2px_rgba(65,57,46,0.04)]"
      data-testid="training-loss-chart"
    >
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-[12px] font-semibold text-[#39352e]">Training loss</h3>
        <div className="ml-auto flex flex-wrap items-center justify-end gap-2 text-[10px]">
          <span
            className={`rounded-full border px-2 py-1 font-semibold capitalize ${statusStyle(normalizedStatus)}`}
          >
            {visibleStatus}
          </span>
          <span className="text-[#756e63]" aria-label="Training percentage">
            {safePercentage.toFixed(1)}%
          </span>
          <span
            className="text-[#8d8579]"
            aria-label={isEta ? 'Training ETA' : 'Training duration'}
          >
            {timingLabel}
          </span>
          {detailLabel && (
            <span className="text-[#6f675c]" aria-label="Training update detail">
              {detailLabel}
            </span>
          )}
        </div>
        {expandable && (
          <button
            ref={expandButtonRef}
            type="button"
            onClick={() => setMetricsOpen(true)}
            aria-label="Expand training metrics"
            title="Expand training metrics"
            className="grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-[#d8d1c5] bg-[#fbfaf6] text-[#6f675c] transition-colors hover:border-[#aebead] hover:bg-[#e7eee6] hover:text-[#4f6b55] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-1"
          >
            <MdOpenInFull size={13} aria-hidden="true" />
          </button>
        )}
      </div>

      <div
        className="mt-2 h-1.5 overflow-hidden rounded-full bg-[#ebe6dd]"
        role="progressbar"
        aria-label={progressLabel}
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={safePercentage}
      >
        <div
          className="h-full rounded-full bg-[#485984] transition-[width]"
          style={{ width: `${safePercentage}%` }}
        />
      </div>

      <div
        className={`mt-2 grid gap-2 text-[10px] ${visibleMetrics.length >= 3 ? 'grid-cols-3' : 'grid-cols-2'}`}
        data-testid="training-progress-metrics"
      >
        {visibleMetrics.map((metric) => {
          const styles = metricStyle(metric?.tone);
          const label = String(metric?.label || 'Metric');
          const value = metric?.value === null || metric?.value === undefined
            ? '—'
            : String(metric.value);
          return (
            <div
              key={metric?.key || label}
              className={`min-w-0 rounded-lg border px-2.5 py-2 ${styles.card}`}
            >
              <span className={styles.label}>{label}</span>
              <strong
                className={`float-right max-w-[58%] truncate font-mono ${styles.value}`}
                aria-label={metric?.ariaLabel || `Latest ${label.toLowerCase()}`}
                title={value}
              >
                {value}
              </strong>
            </div>
          );
        })}
      </div>

      <TrainingMetricsModal
        open={metricsOpen}
        onBack={() => setMetricsOpen(false)}
        history={rlMetricHistory}
        returnFocusRef={expandButtonRef}
      />
    </section>
  );
}
