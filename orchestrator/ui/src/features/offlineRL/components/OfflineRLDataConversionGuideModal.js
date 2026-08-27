// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import {
  MdArrowBack,
  MdCheckCircle,
  MdOutlineDataset,
  MdPlayCircleOutline,
  MdStorage,
  MdSwapHoriz,
} from 'react-icons/md';

const GUIDE_ITEMS = [
  {
    icon: MdStorage,
    title: 'Data Epoch',
    body: 'A Data Epoch is one collection and conversion unit. Conversion automatically reserves the next monotonically increasing output folder, such as data_epoch_0003, without overwriting an earlier epoch.',
  },
  {
    icon: MdSwapHoriz,
    title: 'Conversion outputs',
    body: 'LeRobot v3.0 output can be selected for training. LeRobot v2.1 remains available for compatibility and dataset preview.',
  },
  {
    icon: MdCheckCircle,
    title: 'Verified cleanup',
    body: 'Source MCAP episode folders are removed only after every selected LeRobot output passes verification. A failed or cancelled conversion keeps the source data.',
  },
  {
    icon: MdPlayCircleOutline,
    title: 'Episode inspection',
    body: 'Episode lists read metadata only. Open View when video and synchronized joint traces are needed for inspection.',
  },
];

export default function OfflineRLDataConversionGuideModal({ open, onBack }) {
  const dialogRef = useRef(null);
  const backButtonRef = useRef(null);
  const previouslyFocusedRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    previouslyFocusedRef.current = document.activeElement;
    backButtonRef.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = previousOverflow;
      previouslyFocusedRef.current?.focus?.();
      previouslyFocusedRef.current = null;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
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
      if (focusable.length === 0) return;
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

  return createPortal(
    <div
      className="fixed inset-0 z-[130] flex items-center justify-center bg-black/50 p-3"
      data-testid="offline-rl-data-conversion-guide-backdrop"
      onMouseDown={handleBackdropMouseDown}
    >
      <section
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="offline-rl-data-conversion-guide-title"
        aria-describedby="offline-rl-data-conversion-guide-description"
        className="flex max-h-[84vh] w-[min(820px,92vw)] flex-col overflow-hidden rounded-2xl border border-[#d9d2c5] bg-[#f7f4ed] shadow-2xl"
      >
        <header className="flex shrink-0 items-center gap-3 border-b border-[#ded8cc] bg-[#fbfaf6] px-5 py-4">
          <button
            ref={backButtonRef}
            type="button"
            onClick={() => onBack?.()}
            aria-label="Back to Replay Buffer"
            className="flex h-9 items-center gap-1.5 rounded-lg border border-[#d9d2c5] bg-white px-3 text-xs font-semibold text-[#514b42] transition-colors hover:bg-[#f1ede4] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-2"
          >
            <MdArrowBack size={16} aria-hidden="true" />
            Back
          </button>
          <span className="grid h-9 w-9 place-items-center rounded-lg border border-[#ddd5c7] bg-[#f1ede4] text-[#555046]">
            <MdOutlineDataset size={19} aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#989083]">Data Workflow</p>
            <h2 id="offline-rl-data-conversion-guide-title" className="truncate text-sm font-semibold text-[#292720]">Data Conversion Guide</h2>
            <p id="offline-rl-data-conversion-guide-description" className="sr-only">
              Data Epoch, conversion output, cleanup, and episode inspection guidance.
            </p>
          </div>
        </header>

        <div className="min-h-0 overflow-y-auto overscroll-contain p-5">
          <div className="mb-4 rounded-xl border border-[#d9d2c5] bg-white px-4 py-3">
            <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#91897d]">Workflow</div>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] font-semibold text-[#514b42]">
              <span className="rounded-lg bg-[#f1ede4] px-3 py-2">Collect MCAP</span>
              <MdSwapHoriz size={16} className="text-[#879b83]" aria-hidden="true" />
              <span className="rounded-lg bg-[#f1ede4] px-3 py-2">Convert &amp; verify</span>
              <MdSwapHoriz size={16} className="text-[#879b83]" aria-hidden="true" />
              <span className="rounded-lg bg-[#e5eee3] px-3 py-2 text-[#4f6b55]">Select training epochs</span>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            {GUIDE_ITEMS.map(({ icon: Icon, title, body }) => (
              <article key={title} className="rounded-xl border border-[#ded8cc] bg-white p-4">
                <div className="flex items-center gap-2.5">
                  <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-[#e8eee6] text-[#58705d]">
                    <Icon size={16} aria-hidden="true" />
                  </span>
                  <h3 className="text-[11px] font-semibold text-[#403b34]">{title}</h3>
                </div>
                <p className="mt-2 text-[10px] leading-5 text-[#716a60]">{body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </div>,
    document.body
  );
}
