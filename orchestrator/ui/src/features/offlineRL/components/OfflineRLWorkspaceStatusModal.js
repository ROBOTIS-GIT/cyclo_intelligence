// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useEffect, useRef } from 'react';
import { useSelector } from 'react-redux';
import { MdArrowBack, MdMemory } from 'react-icons/md';
import InlineSystemStatus from '../../../components/InlineSystemStatus';
import RecordTopicMonitor from '../../../components/RecordTopicMonitor';

export default function OfflineRLWorkspaceStatusModal({ isOpen, onClose }) {
  const robotType = useSelector((state) => state.tasks.robotType);
  const backButtonRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return undefined;

    const previouslyFocused = document.activeElement;
    backButtonRef.current?.focus();

    const handleKeyDown = (event) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onClose();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      previouslyFocused?.focus?.();
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleBackdropMouseDown = (event) => {
    if (event.target === event.currentTarget) onClose();
  };

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-black/45 p-4"
      data-testid="offline-rl-workspace-status-backdrop"
      onMouseDown={handleBackdropMouseDown}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="offline-rl-workspace-status-title"
        className="flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-[#d9d2c5] bg-[#f7f4ed] shadow-2xl"
      >
        <header className="flex shrink-0 items-center gap-3 border-b border-[#ded8cc] bg-[#fbfaf6] px-5 py-4">
          <button
            ref={backButtonRef}
            type="button"
            onClick={onClose}
            aria-label="Back to Offline RL workspace"
            className="flex h-9 items-center gap-1.5 rounded-lg border border-[#d9d2c5] bg-white px-3 text-xs font-semibold text-[#514b42] transition-colors hover:bg-[#f1ede4] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-2"
          >
            <MdArrowBack size={16} aria-hidden="true" />
            Back
          </button>
          <span className="grid h-9 w-9 place-items-center rounded-lg border border-[#ddd5c7] bg-[#f1ede4] text-[#555046]">
            <MdMemory size={18} aria-hidden="true" />
          </span>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#989083]">
              Environment
            </p>
            <h2
              id="offline-rl-workspace-status-title"
              className="text-sm font-semibold text-[#292720]"
            >
              Inference Workspace Status
            </h2>
          </div>
        </header>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
          <div className="rounded-xl border border-[#ded8cc] bg-[#fbfaf6] p-4">
            <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[#91897d]">
              Robot
            </div>
            <div className="inline-flex items-center rounded-full border border-[#d8d1c5] bg-white px-3 py-1.5 shadow-sm">
              <span className="text-xs font-medium text-[#716a5f]">Robot Type</span>
              <span className="ml-2 rounded-full bg-[#e4ece2] px-2.5 py-0.5 text-xs font-semibold text-[#58705d]">
                {robotType || 'Not selected'}
              </span>
            </div>
          </div>

          <div className="rounded-xl border border-[#ded8cc] bg-[#fbfaf6] p-4">
            <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[#91897d]">
              System Status
            </div>
            <div className="overflow-x-auto pb-1">
              <InlineSystemStatus />
            </div>
          </div>

          <div className="min-h-[280px] rounded-xl border border-[#ded8cc] bg-[#fbfaf6] p-4">
            <RecordTopicMonitor showWhenEmpty />
          </div>
        </div>
      </section>
    </div>
  );
}
