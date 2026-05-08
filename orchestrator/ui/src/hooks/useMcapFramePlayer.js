/* global BigInt */
// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...
// Author: Claude (AI-assisted)

/**
 * MCAP Frame Player Hook — High-performance MCAP image playback.
 *
 * Architecture: Producer-Consumer + Lazy-decode pipeline
 *  1. MCAP Worker (producer): HTTP fetch + zstd/lz4 decompress + CDR parse → raw JPEG bytes
 *  2. Drain loop (consumer): polls Worker.getNextBatch() → inserts into FrameQueue incrementally
 *  3. FrameQueue: stores raw JPEG ArrayBuffers (~50KB each, not decoded ImageBitmaps)
 *  4. Display: on-demand createImageBitmap() for only the 4 displayed frames per tick
 *
 * Key design decisions:
 *  - currentTimeRef is the source of truth during playback (rAF loop owns it)
 *  - Redux currentTime is updated at ~15fps for UI (progress bar, time display)
 *  - Drain loop uses signal-based wait (Worker wakes consumer when frames arrive)
 *  - Seek is non-blocking with token-based cancellation and error recovery
 */

import { useRef, useState, useCallback, useEffect } from "react";
import { useDispatch } from "react-redux";
import { setCurrentTime, setIsPlaying } from "../features/replay/replaySlice";
import { getMcapWorker, disposeMcapWorker } from "../workers/mcapReaderProxy";
import { FrameQueue } from "../utils/FrameQueue";

const MAX_QUEUE_FRAMES = 3000; // raw JPEG buffers are small, can afford more
const MAX_BITMAP_CACHE = 32; // decoded ImageBitmap LRU cache (current + next per topic)
const DRAIN_BATCH_SIZE = 120; // frames per getNextBatch() poll (~2s of 4cam × 15fps)
const DRAIN_SIGNAL_TIMEOUT_MS = 100; // max wait for Worker signal before re-polling
const DISPATCH_EVERY_N_TICKS = 4; // Redux dispatch throttle: 60fps / 4 = 15fps UI update

export function useMcapFramePlayer({
  mcapUrl,
  duration = 0,
  currentTime = 0,
  isPlaying = false,
  playbackSpeed = 1,
  loopStart = null,
  loopEnd = null,
  isActive = false,
}) {
  const dispatch = useDispatch();

  const [readerState, setReaderState] = useState({
    imageTopics: [],
    startTimeNs: BigInt(0),
    endTimeNs: BigInt(0),
    isLoading: false,
    isReady: false,
    error: null,
  });

  // Refs for animation loop (avoid stale closure)
  const currentTimeRef = useRef(currentTime);
  const durationRef = useRef(duration);
  const playbackSpeedRef = useRef(playbackSpeed);
  const loopStartRef = useRef(loopStart);
  const loopEndRef = useRef(loopEnd);
  const readerStateRef = useRef(readerState);
  const isPlayingRef = useRef(isPlaying);
  // Only sync currentTime from prop when NOT playing.
  // During playback, the rAF loop owns currentTimeRef (updates it every tick).
  // Redux currentTime is a throttled shadow — may lag 1-3 ticks behind the ref.
  if (!isPlaying) {
    currentTimeRef.current = currentTime;
  }
  durationRef.current = duration;
  playbackSpeedRef.current = playbackSpeed;
  loopStartRef.current = loopStart;
  loopEndRef.current = loopEnd;
  readerStateRef.current = readerState;
  isPlayingRef.current = isPlaying;

  const canvasRefs = useRef([]);
  const webglPanelRefs = useRef([]); // WebGLVideoPanel imperative handles
  const animFrameRef = useRef(null);
  const lastTickTimeRef = useRef(null);

  // FrameQueue: stores raw JPEG bytes (NOT decoded ImageBitmaps)
  const frameQueueRef = useRef(new FrameQueue(MAX_QUEUE_FRAMES));
  // Last displayed logTime per topic (avoid re-rendering same frame)
  const lastDisplayedNsRef = useRef([]);

  // Decoded bitmap LRU cache: "topicIdx:logTime" → ImageBitmap
  const bitmapCacheRef = useRef(new Map());
  // Set of keys currently being decoded (prevent duplicate decodes)
  const decodingRef = useRef(new Set());

  // Worker proxy
  const mcapWorkerRef = useRef(null);

  // Drain loop state (producer-consumer)
  const drainActiveRef = useRef(false);

  // Redux dispatch throttle counter
  const dispatchCounterRef = useRef(0);

  // Seeking flag — freezes time advancement in rAF loop during async seek
  const seekingRef = useRef(false);
  // Seek token — monotonically increasing counter to handle rapid consecutive seeks
  const seekTokenRef = useRef(0);

  // ---------------------------------------------------------------------------
  // Canvas ref management
  // ---------------------------------------------------------------------------

  const setCanvasRef = useCallback((index, element) => {
    canvasRefs.current[index] = element;
  }, []);

  const setWebGLPanelRef = useCallback((index, handle) => {
    webglPanelRefs.current[index] = handle;
  }, []);

  const getCanvas = useCallback((index) => {
    return canvasRefs.current[index];
  }, []);

  // ---------------------------------------------------------------------------
  // Render bitmap to canvas (Canvas 2D) or WebGL panel
  // ---------------------------------------------------------------------------

  const renderToCanvas = useCallback((canvasIndex, bitmap) => {
    if (!bitmap) return;

    // Try WebGL panel first (Phase 5)
    const webglPanel = webglPanelRefs.current[canvasIndex];
    if (webglPanel && webglPanel.drawFrame) {
      webglPanel.drawFrame(bitmap);
      return;
    }

    // Fallback: Canvas 2D
    const canvas = canvasRefs.current[canvasIndex];
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }
    ctx.drawImage(bitmap, 0, 0);
  }, []);

  // ---------------------------------------------------------------------------
  // Bitmap cache management
  // ---------------------------------------------------------------------------

  const closeBitmapCache = useCallback(() => {
    const cache = bitmapCacheRef.current;
    for (const bmp of cache.values()) {
      if (bmp && bmp.close) bmp.close();
    }
    cache.clear();
    decodingRef.current.clear();
  }, []);

  const cacheBitmap = useCallback((key, bitmap) => {
    const cache = bitmapCacheRef.current;
    cache.set(key, bitmap);
    // Evict oldest if over limit
    if (cache.size > MAX_BITMAP_CACHE) {
      const firstKey = cache.keys().next().value;
      const old = cache.get(firstKey);
      if (old && old.close) old.close();
      cache.delete(firstKey);
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Initialize MCAP reader (Worker)
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!isActive || !mcapUrl) return;
    let cancelled = false;

    async function init() {
      setReaderState((prev) => ({ ...prev, isLoading: true, error: null }));
      console.log("[MCAP] Initializing worker for:", mcapUrl);

      try {
        const { remote: mcapRemote } = getMcapWorker();
        mcapWorkerRef.current = mcapRemote;

        const result = await mcapRemote.initialize(mcapUrl);
        if (cancelled) return;

        const imageTopics = result.imageTopics;
        const startTimeNs = BigInt(result.startTimeNs);
        const endTimeNs = BigInt(result.endTimeNs);

        console.log("[MCAP] Worker initialized:", {
          topics: imageTopics.map(
            (t) => `${t.cameraName}(${t.frameCount}f, ${t.fps.toFixed(1)}fps)`
          ),
          duration: result.duration?.toFixed(1),
        });

        // Initialize FrameQueue timelines
        frameQueueRef.current.initTimelines(imageTopics.length);
        lastDisplayedNsRef.current = new Array(imageTopics.length).fill(null);

        setReaderState({
          imageTopics,
          startTimeNs,
          endTimeNs,
          isLoading: false,
          isReady: true,
          error: null,
        });
      } catch (err) {
        if (!cancelled) {
          console.error("[MCAP] Worker init failed:", err);
          setReaderState((prev) => ({
            ...prev,
            isLoading: false,
            error: err.message,
          }));
        }
      }
    }

    init();

    const queue = frameQueueRef.current;
    return () => {
      cancelled = true;
      const mcap = mcapWorkerRef.current;
      if (mcap) {
        mcap.stopProducer().catch(() => {});
        mcap.cleanup().catch(() => {});
      }
      disposeMcapWorker();
      mcapWorkerRef.current = null;
      queue.clear();
      closeBitmapCache();
      lastDisplayedNsRef.current = [];
    };
  }, [mcapUrl, isActive, closeBitmapCache]);

  // ---------------------------------------------------------------------------
  // Fetch initial frames at a specific time (for seek/init)
  // Decodes on main thread — only 4 frames, fast enough
  // ---------------------------------------------------------------------------

  const fetchInitialFrames = useCallback(
    async (timeSec) => {
      const { imageTopics, startTimeNs } = readerStateRef.current;
      const mcap = mcapWorkerRef.current;
      if (!mcap || !imageTopics.length) return;

      const targetNs = startTimeNs + BigInt(Math.round(timeSec * 1e9));

      const t0 = performance.now();

      try {
        const result = await mcap.readFramesAtTime(targetNs.toString());
        const { frames, buffers } = result;
        if (!frames.length) return;

        const queue = frameQueueRef.current;

        // Store raw bytes in queue + decode all 4 frames in parallel
        const decodePromises = [];
        for (let i = 0; i < frames.length; i++) {
          const { topicIdx, logTime, format } = frames[i];
          const jpegBuffer = buffers[i];
          const logTimeBI = BigInt(logTime);

          queue.put(topicIdx, logTimeBI, { jpegBuffer, format });

          const mimeType =
            format && format.includes("png") ? "image/png" : "image/jpeg";
          const blob = new Blob([jpegBuffer], { type: mimeType });

          decodePromises.push(
            createImageBitmap(blob)
              .then((bitmap) => {
                const key = `${topicIdx}:${logTimeBI}`;
                cacheBitmap(key, bitmap);
                lastDisplayedNsRef.current[topicIdx] = logTimeBI;
                renderToCanvas(topicIdx, bitmap);
              })
              .catch((err) => {
                console.error(
                  `[MCAP] Initial decode failed: topic=${topicIdx}, size=${jpegBuffer.byteLength}`,
                  err
                );
              })
          );
        }

        await Promise.all(decodePromises);

        console.log(
          `[MCAP] Initial frames in ${(performance.now() - t0).toFixed(0)}ms, ` +
            `${frames.length}/${imageTopics.length} topics`
        );
      } catch (e) {
        console.error("[MCAP] fetchInitialFrames error:", e);
      }
    },
    [renderToCanvas, cacheBitmap]
  );

  // ---------------------------------------------------------------------------
  // Producer-Consumer drain loop
  // Uses signal-based wait: Worker.getNextBatch(count, waitMs) blocks in Worker
  // until frames are available or timeout expires. No blind 50ms polling.
  // ---------------------------------------------------------------------------

  const stopDraining = useCallback(() => {
    drainActiveRef.current = false;
  }, []);

  const startDraining = useCallback(() => {
    if (drainActiveRef.current) return;
    drainActiveRef.current = true;

    let totalDrained = 0;
    const t0 = performance.now();

    const drain = async () => {
      while (drainActiveRef.current) {
        const mcap = mcapWorkerRef.current;
        if (!mcap) break;

        try {
          // getNextBatch waits up to DRAIN_SIGNAL_TIMEOUT_MS in Worker
          // if queue is empty — wakes instantly when Producer adds frames
          const result = await mcap.getNextBatch(
            DRAIN_BATCH_SIZE,
            DRAIN_SIGNAL_TIMEOUT_MS
          );
          if (!drainActiveRef.current) break;

          const { frames, buffers, producerDone } = result;

          if (frames.length > 0) {
            const queue = frameQueueRef.current;
            for (let i = 0; i < frames.length; i++) {
              const { topicIdx, logTime, format } = frames[i];
              queue.put(topicIdx, BigInt(logTime), {
                jpegBuffer: buffers[i],
                format,
              });
            }
            totalDrained += frames.length;

            // Log first batch arrival (latency indicator)
            if (totalDrained === frames.length) {
              console.log(
                `[MCAP] First ${frames.length} frames in ${(performance.now() - t0).toFixed(0)}ms`
              );
            }

            // Evict old frames (keep 5s behind current time)
            const { startTimeNs } = readerStateRef.current;
            const currentNs =
              startTimeNs +
              BigInt(Math.round(currentTimeRef.current * 1e9));
            const cutoff =
              currentNs > BigInt(5_000_000_000)
                ? currentNs - BigInt(5_000_000_000)
                : BigInt(0);
            queue.evictBefore(cutoff);
          }

          if (producerDone) {
            console.log(
              `[MCAP] Producer done: ${totalDrained} frames in ${(performance.now() - t0).toFixed(0)}ms`
            );
            break;
          }

          // No separate idle wait needed — getNextBatch already waited in Worker
        } catch (e) {
          if (drainActiveRef.current) {
            console.error("[MCAP] Drain error:", e);
          }
          break;
        }
      }
      drainActiveRef.current = false;
    };

    drain();
  }, []);

  // ---------------------------------------------------------------------------
  // Look-ahead decode — pre-decode the NEXT frame per topic so it's cached
  // when displayFramesAtTime needs it (eliminates 1-2 frame decode latency)
  // ---------------------------------------------------------------------------

  const predecodeNext = useCallback(
    (queue, topicIdx, afterLogTime) => {
      const next = queue.getNext(topicIdx, afterLogTime);
      if (!next) return;

      const nextKey = `${topicIdx}:${next.logTime}`;
      if (bitmapCacheRef.current.has(nextKey)) return;
      if (decodingRef.current.has(nextKey)) return;
      decodingRef.current.add(nextKey);

      const { jpegBuffer, format } = next.data;
      const mimeType =
        format && format.includes("png") ? "image/png" : "image/jpeg";
      const blob = new Blob([jpegBuffer], { type: mimeType });

      createImageBitmap(blob)
        .then((bitmap) => {
          decodingRef.current.delete(nextKey);
          cacheBitmap(nextKey, bitmap);
        })
        .catch(() => {
          decodingRef.current.delete(nextKey);
        });
    },
    [cacheBitmap]
  );

  // ---------------------------------------------------------------------------
  // Display frames from FrameQueue — on-demand decode with look-ahead.
  // Decodes the ~4 displayed frames + pre-decodes next 4 per rAF tick.
  // ---------------------------------------------------------------------------

  const displayFramesAtTime = useCallback(
    (timeSec) => {
      const { imageTopics, startTimeNs } = readerStateRef.current;
      if (!imageTopics.length) return;

      const targetNs = startTimeNs + BigInt(Math.round(timeSec * 1e9));
      const queue = frameQueueRef.current;
      const bitmapCache = bitmapCacheRef.current;

      for (let i = 0; i < imageTopics.length; i++) {
        const nearest = queue.getNearest(i, targetNs);
        if (!nearest) continue;

        const currentLogTime = nearest.logTime;

        // Skip if same frame already displayed
        if (lastDisplayedNsRef.current[i] === currentLogTime) {
          // Already displayed — but pre-decode the NEXT frame
          predecodeNext(queue, i, currentLogTime);
          continue;
        }

        const cacheKey = `${i}:${currentLogTime}`;

        // Check bitmap decode cache first
        const cachedBitmap = bitmapCache.get(cacheKey);
        if (cachedBitmap) {
          lastDisplayedNsRef.current[i] = currentLogTime;
          renderToCanvas(i, cachedBitmap);
          // Pre-decode next frame so it's ready when needed
          predecodeNext(queue, i, currentLogTime);
          continue;
        }

        // Skip if already being decoded
        if (decodingRef.current.has(cacheKey)) continue;
        decodingRef.current.add(cacheKey);

        // On-demand async decode (non-blocking, ~2-5ms per frame)
        const { jpegBuffer, format } = nearest.data;
        const logTime = currentLogTime;
        const topicIdx = i;
        const mimeType =
          format && format.includes("png") ? "image/png" : "image/jpeg";
        const blob = new Blob([jpegBuffer], { type: mimeType });

        createImageBitmap(blob)
          .then((bitmap) => {
            decodingRef.current.delete(cacheKey);
            cacheBitmap(cacheKey, bitmap);
            lastDisplayedNsRef.current[topicIdx] = logTime;
            renderToCanvas(topicIdx, bitmap);
          })
          .catch((err) => {
            decodingRef.current.delete(cacheKey);
            console.error(
              `[MCAP] Decode failed: topic=${topicIdx}, size=${jpegBuffer.byteLength}`,
              err
            );
          });
      }

    },
    [renderToCanvas, cacheBitmap, predecodeNext]
  );

  // ---------------------------------------------------------------------------
  // Playback controls
  // ---------------------------------------------------------------------------

  const playAll = useCallback(() => {
    if (!readerState.isReady) return;
    lastTickTimeRef.current = null;
    dispatchCounterRef.current = 0;

    // Start producer in worker + drain loop on main thread
    const { startTimeNs } = readerStateRef.current;
    const fromNs =
      startTimeNs + BigInt(Math.round(currentTimeRef.current * 1e9));
    const mcap = mcapWorkerRef.current;
    if (mcap) {
      mcap.startProducer(fromNs.toString()); // fire-and-forget (runs in Worker)
    }
    startDraining();

    dispatch(setIsPlaying(true));
  }, [readerState.isReady, dispatch, startDraining]);

  const pauseAll = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    lastTickTimeRef.current = null;
    stopDraining();

    const mcap = mcapWorkerRef.current;
    if (mcap) mcap.stopProducer().catch(() => {});

    dispatch(setIsPlaying(false));
  }, [dispatch, stopDraining]);

  const togglePlayPause = useCallback(() => {
    if (isPlaying) pauseAll();
    else playAll();
  }, [isPlaying, pauseAll, playAll]);

  const clearCache = useCallback(() => {
    stopDraining();
    frameQueueRef.current.clear();
    closeBitmapCache();
    lastDisplayedNsRef.current = lastDisplayedNsRef.current.map(() => null);

    const mcap = mcapWorkerRef.current;
    if (mcap) mcap.stopProducer().catch(() => {});
  }, [closeBitmapCache, stopDraining]);

  // ---------------------------------------------------------------------------
  // Seek — non-blocking, token-based cancellation, error-safe
  // ---------------------------------------------------------------------------

  const syncToTime = useCallback(
    (time) => {
      const clamped = Math.max(0, Math.min(duration, time));
      const myToken = ++seekTokenRef.current;

      // IMMEDIATELY freeze time at exact seek position (synchronous)
      seekingRef.current = true;
      currentTimeRef.current = clamped;
      dispatch(setCurrentTime(clamped));

      // Stop drain loop synchronously
      stopDraining();

      const mcap = mcapWorkerRef.current;
      if (!mcap) {
        seekingRef.current = false;
        return;
      }

      // Fire-and-forget: async work runs in background via .then() chains
      // Seek token ensures rapid consecutive seeks don't cause race conditions
      mcap
        .stopProducer()
        .then(() => {
          if (seekTokenRef.current !== myToken) return; // superseded

          frameQueueRef.current.clear();
          closeBitmapCache();
          lastDisplayedNsRef.current = lastDisplayedNsRef.current.map(
            () => null
          );

          return fetchInitialFrames(clamped).then(() => {
            if (seekTokenRef.current !== myToken) return; // superseded

            // Unfreeze — time is exactly at clamped, no drift
            currentTimeRef.current = clamped;
            dispatch(setCurrentTime(clamped));
            lastTickTimeRef.current = null;
            seekingRef.current = false;

            // If still playing, restart producer + drain
            if (isPlayingRef.current) {
              const { startTimeNs } = readerStateRef.current;
              const fromNs =
                startTimeNs + BigInt(Math.round(clamped * 1e9));
              mcap.startProducer(fromNs.toString());
              startDraining();
            }
          });
        })
        .catch((err) => {
          // Ensure seekingRef is always reset even on error
          console.error("[MCAP] Seek error:", err);
          seekingRef.current = false;
        });
    },
    [
      duration,
      dispatch,
      stopDraining,
      startDraining,
      closeBitmapCache,
      fetchInitialFrames,
    ]
  );

  const restart = useCallback(() => {
    dispatch(setCurrentTime(0));
    clearCache();
    fetchInitialFrames(0);
    lastTickTimeRef.current = null;
    dispatch(setIsPlaying(true));
  }, [dispatch, clearCache, fetchInitialFrames]);

  const stepFrame = useCallback(
    (direction) => {
      if (isPlaying) pauseAll();
      const currentFps = readerState.imageTopics[0]?.fps || 15;
      const frameTime = 1 / currentFps;
      const delta = direction === "forward" ? frameTime : -frameTime;
      const newTime = Math.max(0, Math.min(duration, currentTime + delta));
      dispatch(setCurrentTime(newTime));
      fetchInitialFrames(newTime);
    },
    [
      isPlaying,
      pauseAll,
      readerState.imageTopics,
      duration,
      currentTime,
      dispatch,
      fetchInitialFrames,
    ]
  );

  const seekRelative = useCallback(
    (seconds) => {
      const newTime = Math.max(0, Math.min(duration, currentTime + seconds));
      syncToTime(newTime);
    },
    [duration, currentTime, syncToTime]
  );

  const handleProgressClick = useCallback(
    (e, containerElement) => {
      if (!containerElement) return;
      const rect = containerElement.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = x / rect.width;
      const newTime = percentage * duration;
      syncToTime(newTime);
    },
    [duration, syncToTime]
  );

  // ---------------------------------------------------------------------------
  // Stable ref for animation loop
  // ---------------------------------------------------------------------------

  const displayFramesRef = useRef(displayFramesAtTime);
  displayFramesRef.current = displayFramesAtTime;

  // ---------------------------------------------------------------------------
  // Playback animation loop
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!isPlaying || !readerState.isReady) {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
      return;
    }

    lastTickTimeRef.current = null;
    dispatchCounterRef.current = 0;

    const tick = (timestamp) => {
      // During async seek: freeze time, keep rAF alive so playback resumes
      // when seek completes. Cost is negligible (~0.01ms per idle tick).
      if (seekingRef.current) {
        lastTickTimeRef.current = null;
        animFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      if (!lastTickTimeRef.current) {
        lastTickTimeRef.current = timestamp;
        animFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      const elapsedMs = timestamp - lastTickTimeRef.current;
      lastTickTimeRef.current = timestamp;

      const speed = playbackSpeedRef.current;
      const elapsedSec = (elapsedMs / 1000) * speed;
      const newTime = currentTimeRef.current + elapsedSec;
      const dur = durationRef.current;
      const ls = loopStartRef.current;
      const le = loopEndRef.current;

      if (ls !== null && le !== null && newTime >= le) {
        currentTimeRef.current = ls;
        dispatch(setCurrentTime(ls)); // always dispatch on loop boundary
        displayFramesRef.current(ls);
        animFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      if (newTime >= dur) {
        currentTimeRef.current = dur;
        dispatch(setCurrentTime(dur)); // always dispatch on end
        dispatch(setIsPlaying(false));
        displayFramesRef.current(dur);
        return;
      }

      // Update ref BEFORE dispatch — ensures next tick uses fresh value
      currentTimeRef.current = newTime;

      // Throttle Redux dispatch to ~15fps (every Nth tick)
      // Reduces React reconciliation overhead while keeping UI smooth
      if (++dispatchCounterRef.current % DISPATCH_EVERY_N_TICKS === 0) {
        dispatch(setCurrentTime(newTime));
      }

      displayFramesRef.current(newTime);

      animFrameRef.current = requestAnimationFrame(tick);
    };

    animFrameRef.current = requestAnimationFrame(tick);

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
    };
  }, [isPlaying, readerState.isReady, dispatch]);

  // ---------------------------------------------------------------------------
  // Display initial frame when ready
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (readerState.isReady && !isPlaying) {
      fetchInitialFrames(currentTime);
    }
  }, [readerState.isReady]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------------------
  // Cleanup on unmount
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const queue = frameQueueRef.current;
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      drainActiveRef.current = false;

      const mcap = mcapWorkerRef.current;
      if (mcap) {
        mcap.stopProducer().catch(() => {});
        mcap.cleanup().catch(() => {});
      }

      queue.clear();
      // Close all cached bitmaps
      const cache = bitmapCacheRef.current;
      for (const bmp of cache.values()) {
        if (bmp && bmp.close) bmp.close();
      }
      cache.clear();
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Return interface
  // ---------------------------------------------------------------------------

  return {
    imageTopics: readerState.imageTopics,
    isLoading: readerState.isLoading,
    isReady: readerState.isReady,
    mcapError: readerState.error,

    canvasRefs,
    setCanvasRef,
    setWebGLPanelRef,
    getCanvas,

    togglePlayPause,
    playAll,
    pauseAll,
    restart,
    stepFrame,
    seekRelative,
    syncToTime,
    handleProgressClick,

    setPlaybackSpeed: (speed) => {
      playbackSpeedRef.current = speed;
    },
    handleTimeUpdate: () => {},
    handleEnded: () => {},
  };
}

export default useMcapFramePlayer;
