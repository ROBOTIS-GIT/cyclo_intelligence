// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { MdArrowBack, MdDeleteOutline, MdVideocam } from 'react-icons/md';
import JointDataPanel from '../../../components/replay/JointDataPanel';
import { prepareChartData } from '../../../utils/chartUtils';

const asFiniteNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const segmentStart = (item) => Math.max(0, asFiniteNumber(item?.fromS) ?? 0);

const segmentEnd = (item) => {
  const start = segmentStart(item);
  const end = asFiniteNumber(item?.toS);
  return end !== null && end > start ? end : null;
};

const segmentDuration = (item) => {
  const end = segmentEnd(item);
  return end === null ? null : end - segmentStart(item);
};

const formatEpisodeLabel = (index) => {
  if (index === null || index === undefined || index === '') return 'Episode';
  const numeric = Number(index);
  if (Number.isInteger(numeric) && numeric >= 0) {
    return `episode_${String(numeric).padStart(3, '0')}`;
  }
  const value = String(index);
  return value.startsWith('episode_') ? value : `episode_${value}`;
};

const formatOutcome = (outcome) => {
  if (!outcome) return 'Unlabeled';
  const normalized = String(outcome).trim().toLowerCase();
  if (normalized === 'success') return 'Success';
  if (normalized === 'fail' || normalized === 'failure') return 'Fail';
  return String(outcome);
};

const formatDuration = (seconds) => {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return '—';
  }
  return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} s`;
};

const safeArray = (value) => (Array.isArray(value) ? value : []);

/** Keep the Episode viewer independent from MCAP/LeRobot response casing. */
export function normalizeEpisodeJointData(value = {}) {
  return {
    jointTimestamps: safeArray(value.joint_timestamps ?? value.jointTimestamps),
    jointNames: safeArray(value.joint_names ?? value.jointNames).map(String),
    jointPositions: safeArray(value.joint_positions ?? value.jointPositions),
    actionTimestamps: safeArray(value.action_timestamps ?? value.actionTimestamps),
    actionNames: safeArray(value.action_names ?? value.actionNames).map(String),
    actionValues: safeArray(value.action_values ?? value.actionValues),
    duration: Math.max(0, asFiniteNumber(value.duration) ?? 0),
  };
}

/**
 * Preview one recorded episode without coupling the UI to an MCAP or LeRobot API.
 *
 * onDelete owns the actual deletion and must not show a second confirmation. A
 * resolved value other than false is treated as success; false or a rejection
 * keeps this dialog open.
 */
export default function OfflineRLEpisodeMediaModal({
  open,
  sourceLabel,
  episode,
  media = [],
  loading = false,
  error = '',
  jointData = null,
  jointLoading = false,
  jointError = '',
  onBack,
  onDelete,
  deletePending = false,
  deleteDisabled = false,
}) {
  const backButtonRef = useRef(null);
  const dialogRef = useRef(null);
  const previouslyFocusedRef = useRef(null);
  const videoRefs = useRef([]);
  const programmaticPlayRef = useRef(new Set());
  const programmaticPauseRef = useRef(new Set());
  const leaderIndexRef = useRef(0);
  const [localDeletePending, setLocalDeletePending] = useState(false);
  const [localDeleteError, setLocalDeleteError] = useState('');
  const [playbackTime, setPlaybackTime] = useState(0);
  const [expandedJoints, setExpandedJoints] = useState(() => new Set());
  const deleting = deletePending || localDeletePending;

  const safeMedia = useMemo(
    () => (Array.isArray(media) ? media.filter((item) => item?.url) : []),
    [media]
  );
  const episodeLabel = formatEpisodeLabel(episode?.index);
  const outcomeLabel = formatOutcome(episode?.outcome);
  const fps = safeMedia
    .map((item) => asFiniteNumber(item.fps))
    .find((value) => value !== null && value > 0) ?? null;
  const explicitDurations = safeMedia
    .map(segmentDuration)
    .filter((value) => value !== null);
  const mediaDuration = explicitDurations.length > 0
    ? Math.min(...explicitDurations)
    : (asFiniteNumber(episode?.frames) !== null && fps
      ? asFiniteNumber(episode.frames) / fps
      : null);
  const normalizedJointData = useMemo(
    () => normalizeEpisodeJointData(jointData || {}),
    [jointData]
  );
  const allJointNames = useMemo(() => Array.from(new Set([
    ...normalizedJointData.jointNames,
    ...normalizedJointData.actionNames,
  ])), [normalizedJointData.actionNames, normalizedJointData.jointNames]);
  const stateChartData = useMemo(() => prepareChartData(
    normalizedJointData.jointTimestamps,
    normalizedJointData.jointNames,
    normalizedJointData.jointPositions,
    'state_',
    1000
  ), [
    normalizedJointData.jointNames,
    normalizedJointData.jointPositions,
    normalizedJointData.jointTimestamps,
  ]);
  const actionChartData = useMemo(() => prepareChartData(
    normalizedJointData.actionTimestamps,
    normalizedJointData.actionNames,
    normalizedJointData.actionValues,
    'action_',
    1000
  ), [
    normalizedJointData.actionNames,
    normalizedJointData.actionTimestamps,
    normalizedJointData.actionValues,
  ]);
  const seriesDuration = useMemo(() => Math.max(
    normalizedJointData.duration,
    ...normalizedJointData.jointTimestamps.map((value) => asFiniteNumber(value) ?? 0),
    ...normalizedJointData.actionTimestamps.map((value) => asFiniteNumber(value) ?? 0)
  ), [normalizedJointData]);
  const duration = Math.max(mediaDuration ?? 0, seriesDuration) || null;
  const hasJointData = allJointNames.length > 0 && (
    stateChartData.length > 0 || actionChartData.length > 0
  );
  const hasActionData = normalizedJointData.actionTimestamps.length > 0 &&
    normalizedJointData.actionNames.length > 0;
  const taskText = Array.isArray(episode?.tasks)
    ? episode.tasks.filter(Boolean).join(' · ')
    : episode?.tasks;

  useEffect(() => {
    if (!open) return undefined;
    previouslyFocusedRef.current = document.activeElement;
    backButtonRef.current?.focus();

    return () => {
      previouslyFocusedRef.current?.focus?.();
      previouslyFocusedRef.current = null;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        if (deleting) return;
        event.preventDefault();
        onBack?.();
        return;
      }
      if (event.key !== 'Tab') return;

      const focusable = Array.from(dialogRef.current?.querySelectorAll(
        'button:not([disabled]), video[controls], [href], input:not([disabled]), '
        + 'select:not([disabled]), textarea:not([disabled]), '
        + '[tabindex]:not([tabindex="-1"])'
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
  }, [deleting, onBack, open]);

  useEffect(() => {
    if (!open) {
      setLocalDeletePending(false);
      setLocalDeleteError('');
      setPlaybackTime(0);
      setExpandedJoints(new Set());
      videoRefs.current = [];
    }
  }, [open]);

  useEffect(() => {
    setPlaybackTime(0);
    setExpandedJoints(new Set());
  }, [episode?.index]);

  if (!open) return null;

  const relativeTime = (index) => {
    const video = videoRefs.current[index];
    if (!video) return 0;
    return Math.max(0, video.currentTime - segmentStart(safeMedia[index]));
  };

  const seekVideo = (index, elapsed) => {
    const video = videoRefs.current[index];
    const item = safeMedia[index];
    if (!video || !item) return;
    const start = segmentStart(item);
    const end = segmentEnd(item);
    const target = Math.max(start, end === null ? start + elapsed : Math.min(end, start + elapsed));
    if (Math.abs(video.currentTime - target) > 0.04) video.currentTime = target;
  };

  const playVideo = (video, index) => {
    if (!video || !video.paused) return;
    programmaticPlayRef.current.add(index);
    try {
      const playResult = video.play();
      Promise.resolve(playResult).catch(() => {}).finally(() => {
        programmaticPlayRef.current.delete(index);
      });
    } catch (_error) {
      programmaticPlayRef.current.delete(index);
    }
  };

  const pauseAll = (sourceIndex = -1) => {
    videoRefs.current.forEach((video, index) => {
      if (!video || index === sourceIndex || video.paused) return;
      programmaticPauseRef.current.add(index);
      video.pause();
    });
  };

  const syncAt = (sourceIndex, shouldPlay) => {
    const elapsed = relativeTime(sourceIndex);
    videoRefs.current.forEach((video, index) => {
      if (!video || index === sourceIndex) return;
      seekVideo(index, elapsed);
      if (shouldPlay) playVideo(video, index);
    });
  };

  const handleLoadedMetadata = (index) => {
    seekVideo(index, 0);
  };

  const handlePlay = (index) => {
    if (programmaticPlayRef.current.delete(index)) return;
    const item = safeMedia[index];
    const video = videoRefs.current[index];
    const start = segmentStart(item);
    const end = segmentEnd(item);
    if (video.currentTime < start - 0.01) video.currentTime = start;
    if (end !== null && video.currentTime >= end - 0.01) video.currentTime = start;
    leaderIndexRef.current = index;
    syncAt(index, true);
  };

  const handlePause = (index) => {
    if (programmaticPauseRef.current.delete(index)) return;
    pauseAll(index);
  };

  const handleSeeked = (index) => {
    const video = videoRefs.current[index];
    const start = segmentStart(safeMedia[index]);
    const end = segmentEnd(safeMedia[index]);
    if (video && video.currentTime < start) video.currentTime = start;
    if (video && end !== null && video.currentTime > end) video.currentTime = end;
    if (index !== leaderIndexRef.current) leaderIndexRef.current = index;
    setPlaybackTime(Math.round(relativeTime(index) * 10) / 10);
    syncAt(index, !video?.paused);
  };

  const handleTimeUpdate = (index) => {
    if (index !== leaderIndexRef.current) return;
    const video = videoRefs.current[index];
    const start = segmentStart(safeMedia[index]);
    const end = segmentEnd(safeMedia[index]);
    if (video.currentTime < start) {
      video.currentTime = start;
      return;
    }
    if (end !== null && video.currentTime >= end - 0.01) {
      video.currentTime = end;
      setPlaybackTime(Math.round((end - start) * 10) / 10);
      pauseAll(index);
      if (!video.paused) video.pause();
      return;
    }

    const elapsed = relativeTime(index);
    const roundedElapsed = Math.round(elapsed * 10) / 10;
    setPlaybackTime((current) => (
      current === roundedElapsed ? current : roundedElapsed
    ));
    videoRefs.current.forEach((other, otherIndex) => {
      if (!other || otherIndex === index) return;
      const expected = segmentStart(safeMedia[otherIndex]) + elapsed;
      if (Math.abs(other.currentTime - expected) > 0.15) seekVideo(otherIndex, elapsed);
    });
  };

  const handleRateChange = (index) => {
    const rate = videoRefs.current[index]?.playbackRate;
    if (!rate) return;
    videoRefs.current.forEach((video, otherIndex) => {
      if (video && otherIndex !== index && video.playbackRate !== rate) {
        video.playbackRate = rate;
      }
    });
  };

  const handleChartSeek = (time) => {
    const requestedTime = asFiniteNumber(time);
    if (requestedTime === null) return;
    const elapsed = Math.max(0, duration === null
      ? requestedTime
      : Math.min(duration, requestedTime));
    const leader = videoRefs.current[leaderIndexRef.current];
    const shouldPlay = Boolean(leader && !leader.paused);
    videoRefs.current.forEach((_video, index) => seekVideo(index, elapsed));
    setPlaybackTime(Math.round(elapsed * 10) / 10);
    if (shouldPlay) {
      videoRefs.current.forEach((video, index) => playVideo(video, index));
    }
  };

  const toggleJoint = (jointName) => {
    setExpandedJoints((current) => {
      const next = new Set(current);
      if (next.has(jointName)) next.delete(jointName);
      else next.add(jointName);
      return next;
    });
  };

  const handleDelete = async () => {
    if (!onDelete || deleting || deleteDisabled) return;
    const confirmed = window.confirm(
      `Delete ${episodeLabel}? This removes the selected episode from its dataset.`
    );
    if (!confirmed) return;

    setLocalDeleteError('');
    setLocalDeletePending(true);
    try {
      const result = await onDelete(episode);
      if (result === false) {
        setLocalDeleteError('The episode could not be deleted.');
        return;
      }
      onBack?.();
    } catch (deleteError) {
      setLocalDeleteError(deleteError?.message || 'The episode could not be deleted.');
    } finally {
      setLocalDeletePending(false);
    }
  };

  const handleBackdropMouseDown = (event) => {
    if (event.target === event.currentTarget && !deleting) onBack?.();
  };

  return createPortal(
    <div
      className="fixed inset-0 z-[120] flex items-center justify-center bg-black/50 p-1.5 sm:p-2"
      data-testid="offline-rl-episode-media-backdrop"
      onMouseDown={handleBackdropMouseDown}
    >
      <section
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="offline-rl-episode-media-title"
        aria-describedby="offline-rl-episode-media-summary"
        className="flex h-[86vh] w-[98vw] max-w-none flex-col overflow-hidden rounded-2xl border border-[#d9d2c5] bg-[#f7f4ed] shadow-2xl"
      >
        <header className="flex shrink-0 items-center justify-between gap-4 border-b border-[#ded8cc] bg-[#fbfaf6] px-5 py-3">
          <div className="flex min-w-0 items-center gap-3">
            <button
              ref={backButtonRef}
              type="button"
              onClick={() => onBack?.()}
              disabled={deleting}
              aria-label="Back to episode list"
              className="flex h-9 shrink-0 items-center gap-1.5 rounded-lg border border-[#d9d2c5] bg-white px-3 text-xs font-semibold text-[#514b42] transition-colors hover:bg-[#f1ede4] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <MdArrowBack size={16} aria-hidden="true" />
              Back
            </button>
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-[#ddd5c7] bg-[#f1ede4] text-[#555046]">
              <MdVideocam size={19} aria-hidden="true" />
            </span>
            <div className="min-w-0">
              <p className="truncate text-[10px] font-semibold uppercase tracking-[0.15em] text-[#989083]">
                {sourceLabel || 'Dataset'}
              </p>
              <h2
                id="offline-rl-episode-media-title"
                className="truncate text-sm font-semibold text-[#292720]"
              >
                {episodeLabel}
              </h2>
            </div>
          </div>

          {onDelete && (
            <button
              type="button"
              onClick={handleDelete}
              disabled={deleting || deleteDisabled}
              className="flex h-9 shrink-0 items-center gap-1.5 rounded-lg border border-[#c98f88] bg-[#fff4f2] px-3 text-xs font-semibold text-[#a13e35] transition-colors hover:bg-[#fbe5e1] focus:outline-none focus:ring-2 focus:ring-[#bd6258] focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <MdDeleteOutline size={16} aria-hidden="true" />
              {deleting ? 'Deleting…' : 'Delete'}
            </button>
          )}
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-3 lg:p-4">
          <div
            id="offline-rl-episode-media-summary"
            className="mb-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-5"
          >
            {[
              ['Source', sourceLabel || '—'],
              ['Outcome', outcomeLabel],
              ['FPS', fps ? String(fps) : '—'],
              ['Duration', formatDuration(duration)],
              ['Frames', asFiniteNumber(episode?.frames) ?? '—'],
            ].map(([label, value]) => (
              <div key={label} className="rounded-xl border border-[#ded8cc] bg-[#fbfaf6] px-3 py-2">
                <div className="text-[9px] font-semibold uppercase tracking-[0.13em] text-[#91897d]">
                  {label}
                </div>
                <div className="mt-1 truncate text-xs font-semibold text-[#3b3832]">
                  {value}
                </div>
              </div>
            ))}
          </div>

          {taskText && (
            <div className="mb-3 rounded-xl border border-[#ded8cc] bg-[#fbfaf6] px-4 py-2">
              <div className="text-[9px] font-semibold uppercase tracking-[0.13em] text-[#91897d]">
                Task
              </div>
              <p className="mt-1 text-xs text-[#514b42]">{taskText}</p>
            </div>
          )}

          {(error || localDeleteError) && (
            <div role="alert" className="mb-4 rounded-xl border border-[#d8aaa4] bg-[#fff4f2] px-4 py-3 text-xs font-medium text-[#923c34]">
              {localDeleteError || error}
            </div>
          )}

          {loading ? (
            <div role="status" className="grid min-h-[300px] place-items-center rounded-xl border border-[#ded8cc] bg-[#fbfaf6] text-xs font-medium text-[#716a5f]">
              Loading episode media…
            </div>
          ) : safeMedia.length === 0 ? (
            <div className="grid min-h-[300px] place-items-center rounded-xl border border-dashed border-[#d7d0c4] bg-[#fbfaf6] px-6 text-center text-xs text-[#827b70]">
              No playable camera video is available for this episode.
            </div>
          ) : (
            <div className="grid gap-3 lg:grid-cols-3 lg:gap-4">
              {safeMedia.slice(0, 3).map((item, index) => (
                <article key={item.key || item.url || index} className="overflow-hidden rounded-xl border border-[#d8d1c5] bg-[#fbfaf6] shadow-sm">
                  <div className="flex items-center justify-between gap-3 border-b border-[#e1dbd0] px-3 py-2">
                    <h3 className="truncate text-xs font-semibold text-[#3e3a33]">
                      {item.label || `Camera ${index + 1}`}
                    </h3>
                    <span className="shrink-0 text-[10px] font-medium text-[#8c8579]">
                      {formatDuration(segmentDuration(item))}
                    </span>
                  </div>
                  <div className="aspect-[4/3] bg-[#25241f]">
                    <video
                      ref={(node) => { videoRefs.current[index] = node; }}
                      src={item.url}
                      aria-label={`${item.label || `Camera ${index + 1}`} episode video`}
                      className="h-full w-full object-contain"
                      controls
                      playsInline
                      preload="metadata"
                      onLoadedMetadata={() => handleLoadedMetadata(index)}
                      onPlay={() => handlePlay(index)}
                      onPause={() => handlePause(index)}
                      onSeeked={() => handleSeeked(index)}
                      onTimeUpdate={() => handleTimeUpdate(index)}
                      onRateChange={() => handleRateChange(index)}
                    >
                      Video playback is not supported by this browser.
                    </video>
                  </div>
                </article>
              ))}
            </div>
          )}

          <section
            className="mt-4 h-[460px] overflow-hidden rounded-xl border border-[#d8d1c5] bg-[#fbfaf6] p-3 shadow-sm"
            aria-label="Episode joint data"
          >
            {jointLoading ? (
              <div role="status" className="grid h-full place-items-center text-xs font-medium text-[#716a5f]">
                Loading joint data…
              </div>
            ) : jointError ? (
              <div role="alert" className="grid h-full place-items-center px-6 text-center text-xs font-medium text-[#923c34]">
                {jointError}
              </div>
            ) : (
              <JointDataPanel
                allJointNames={allJointNames}
                stateChartData={stateChartData}
                actionChartData={actionChartData}
                currentTime={playbackTime}
                duration={duration || 0}
                expandedJoints={expandedJoints}
                toggleJoint={toggleJoint}
                expandAllJoints={() => setExpandedJoints(new Set(allJointNames))}
                collapseAllJoints={() => setExpandedJoints(new Set())}
                hasActionData={hasActionData}
                actionNames={normalizedJointData.actionNames}
                handleChartSeek={handleChartSeek}
              />
            )}
            {!jointLoading && !jointError && !hasJointData && (
              <span className="sr-only">No episode joint samples were loaded.</span>
            )}
          </section>
        </div>
      </section>
    </div>,
    document.body
  );
}
