// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useEffect, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import { MdArrowBack, MdShowChart } from 'react-icons/md';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const finiteNumber = value => (
  typeof value === 'number' && Number.isFinite(value) ? value : null
);

export const normalizeRlMetricHistory = (history) => {
  if (!Array.isArray(history)) return [];

  const byEpoch = new Map();
  history.forEach((point) => {
    const rlEpoch = finiteNumber(point?.rl_epoch);
    if (rlEpoch === null || !Number.isInteger(rlEpoch) || rlEpoch < 1) return;

    const actorLoss = finiteNumber(point?.actor_loss_mean);
    const criticLoss = finiteNumber(point?.critic_loss_mean);
    const averageReward = finiteNumber(point?.replay_average_reward);
    if (actorLoss === null && criticLoss === null && averageReward === null) return;

    const previous = byEpoch.get(rlEpoch);
    byEpoch.set(rlEpoch, {
      rl_epoch: rlEpoch,
      actor_loss_mean: actorLoss ?? previous?.actor_loss_mean ?? null,
      critic_loss_mean: criticLoss ?? previous?.critic_loss_mean ?? null,
      replay_average_reward: averageReward ?? previous?.replay_average_reward ?? null,
    });
  });

  return Array.from(byEpoch.values()).sort((left, right) => (
    left.rl_epoch - right.rl_epoch
  ));
};

const tooltipFormatter = (value, name) => {
  if (!Number.isFinite(value)) return ['—', name];
  return [Number(value).toFixed(4), name];
};

const xDomainFor = (history) => {
  if (history.length !== 1) return ['dataMin', 'dataMax'];
  const epoch = history[0].rl_epoch;
  return [Math.max(1, epoch - 1), epoch + 1];
};

export default function TrainingMetricsModal({
  open,
  onBack,
  history = [],
  returnFocusRef,
}) {
  const dialogRef = useRef(null);
  const backButtonRef = useRef(null);
  const normalizedHistory = useMemo(
    () => normalizeRlMetricHistory(history),
    [history]
  );

  useEffect(() => {
    if (!open || typeof document === 'undefined') return undefined;
    const previousOverflow = document.body.style.overflow;
    const focusTarget = returnFocusRef?.current;
    document.body.style.overflow = 'hidden';
    backButtonRef.current?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
      focusTarget?.focus?.();
    };
  }, [open, returnFocusRef]);

  useEffect(() => {
    if (!open || typeof window === 'undefined') return undefined;
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onBack?.();
        return;
      }
      if (event.key !== 'Tab') return;

      const focusable = Array.from(dialogRef.current?.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), '
        + 'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      ) || []);
      if (focusable.length === 0) {
        event.preventDefault();
        dialogRef.current?.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onBack, open]);

  if (!open || typeof document === 'undefined') return null;

  const handleBackdropMouseDown = (event) => {
    if (event.target === event.currentTarget) onBack?.();
  };
  const epochCountLabel = `${normalizedHistory.length} RL epoch${normalizedHistory.length === 1 ? '' : 's'}`;

  return createPortal(
    <div
      className="fixed inset-0 z-[140] flex items-center justify-center bg-black/50 p-3"
      data-testid="training-metrics-backdrop"
      onMouseDown={handleBackdropMouseDown}
    >
      <section
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="training-metrics-title"
        aria-describedby="training-metrics-description"
        tabIndex={-1}
        className="flex h-[min(720px,82vh)] w-[min(1180px,92vw)] flex-col overflow-hidden rounded-2xl border border-[#d9d2c5] bg-[#f7f4ed] shadow-2xl"
      >
        <header className="flex shrink-0 items-center gap-3 border-b border-[#ded8cc] bg-[#fbfaf6] px-5 py-4">
          <button
            ref={backButtonRef}
            type="button"
            onClick={() => onBack?.()}
            aria-label="Back to Training"
            className="flex h-9 items-center gap-1.5 rounded-lg border border-[#d9d2c5] bg-white px-3 text-xs font-semibold text-[#514b42] transition-colors hover:bg-[#f1ede4] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-2"
          >
            <MdArrowBack size={16} aria-hidden="true" />
            Back
          </button>
          <span className="grid h-9 w-9 place-items-center rounded-lg border border-[#ddd5c7] bg-[#f1ede4] text-[#555046]">
            <MdShowChart size={19} aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#989083]">
              Reinforcement Learning
            </p>
            <h2 id="training-metrics-title" className="truncate text-sm font-semibold text-[#292720]">
              Training Metrics
            </h2>
            <p id="training-metrics-description" className="sr-only">
              Mean actor loss, critic loss, and replay-buffer reward by RL epoch.
            </p>
          </div>
          <span className="ml-auto rounded-full border border-[#d8d1c5] bg-white px-2.5 py-1 text-[10px] font-semibold text-[#756e63]">
            {epochCountLabel}
          </span>
        </header>

        <div className="min-h-0 flex-1 p-5">
          {normalizedHistory.length === 0 ? (
            <div
              className="flex h-full items-center justify-center rounded-xl border border-dashed border-[#d8d1c5] bg-white text-sm text-[#81796e]"
              data-testid="training-metrics-empty"
            >
              No RL epoch metrics yet.
            </div>
          ) : (
            <div
              className="h-full min-h-[320px] rounded-xl border border-[#ddd6ca] bg-white px-3 pb-3 pt-5"
              data-testid="training-metrics-chart"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={normalizedHistory}
                  margin={{ top: 12, right: 28, left: 24, bottom: 28 }}
                >
                  <CartesianGrid stroke="#e8e2d8" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="rl_epoch"
                    type="number"
                    domain={xDomainFor(normalizedHistory)}
                    allowDecimals={false}
                    tick={{ fill: '#756e63', fontSize: 10 }}
                    stroke="#a8a094"
                    label={{ value: 'RL epoch', position: 'insideBottom', offset: -18 }}
                  />
                  <YAxis
                    yAxisId="loss"
                    tick={{ fill: '#756e63', fontSize: 10 }}
                    stroke="#a8a094"
                    width={64}
                    label={{ value: 'Average loss', angle: -90, position: 'insideLeft' }}
                  />
                  <YAxis
                    yAxisId="reward"
                    orientation="right"
                    tick={{ fill: '#66758d', fontSize: 10 }}
                    stroke="#8fa0ba"
                    width={64}
                    label={{ value: 'Average reward', angle: 90, position: 'insideRight' }}
                  />
                  <Tooltip formatter={tooltipFormatter} labelFormatter={value => `RL epoch ${value}`} />
                  <Legend verticalAlign="top" height={34} />
                  <Line
                    yAxisId="loss"
                    type="monotone"
                    dataKey="actor_loss_mean"
                    name="Actor loss"
                    stroke="#5f8065"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                    connectNulls
                    isAnimationActive={false}
                  />
                  <Line
                    yAxisId="loss"
                    type="monotone"
                    dataKey="critic_loss_mean"
                    name="Critic loss"
                    stroke="#b5764e"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                    connectNulls
                    isAnimationActive={false}
                  />
                  <Line
                    yAxisId="reward"
                    type="monotone"
                    dataKey="replay_average_reward"
                    name="Replay average reward"
                    stroke="#64779a"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                    connectNulls
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </section>
    </div>,
    document.body
  );
}
