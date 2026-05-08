/* global BigInt */
/* eslint-disable no-restricted-globals */
/**
 * MCAP Reader Web Worker — All MCAP I/O, decompression, and CDR parsing
 * runs off the main thread. Communicates via Comlink.
 *
 * Pipeline: HTTP Range fetch → zstd/lz4 decompress → CDR parse → JPEG bytes
 * All heavy work stays here; only JPEG ArrayBuffers are transferred to main.
 */

import * as Comlink from "comlink";
import { McapIndexedReader } from "@mcap/core";
import { decompress as decompressZstdJS } from "fzstd";
import lz4js from "lz4js";

// ---------------------------------------------------------------------------
// WASM decompression (Phase 2) — Custom minimal Emscripten loader
//
// Loads pre-built Emscripten .wasm from public/wasm/ via fetch().
// No Node.js dependencies, no CRA polyfills needed.
// Falls back to fzstd/lz4js on failure.
// ---------------------------------------------------------------------------

let wasmZstdDecompress = null;
let wasmLz4Decompress = null;
let wasmLoadAttempted = false;

/**
 * Minimal Emscripten runtime — provides only the 2 imports Emscripten needs:
 *  a.a = emscripten_resize_heap
 *  a.b = emscripten_memcpy_big
 */
function loadEmscriptenModule(wasmBytes) {
  let HEAPU8;
  let memory;

  const importObject = {
    a: {
      // emscripten_resize_heap
      a: (requestedSize) => {
        try {
          memory.grow((requestedSize - HEAPU8.length + 65535) >>> 16);
          HEAPU8 = new Uint8Array(memory.buffer);
          return 1;
        } catch {
          return 0;
        }
      },
      // emscripten_memcpy_big
      b: (dest, src, num) => {
        HEAPU8.copyWithin(dest, src, src + num);
      },
    },
  };

  const mod = new WebAssembly.Module(wasmBytes);
  const instance = new WebAssembly.Instance(mod, importObject);

  memory = instance.exports.c; // 'c' is the memory export
  HEAPU8 = new Uint8Array(memory.buffer);

  // Call __wasm_call_ctors ('d' export)
  if (instance.exports.d) instance.exports.d();

  return { instance, HEAPU8: () => new Uint8Array(memory.buffer), exports: instance.exports };
}

/**
 * Build a decompress function from an Emscripten module.
 * Handles malloc/free + heap copy to return a clean Uint8Array.
 */
function makeDecompressor(mod, mallocKey, freeKey, decompressKey) {
  const malloc = mod.exports[mallocKey];
  const free = mod.exports[freeKey];
  const decompress = mod.exports[decompressKey];

  return (srcBuffer, destSize) => {
    const srcSize = srcBuffer.byteLength;
    const srcPtr = malloc(srcSize);
    const destPtr = malloc(destSize);

    // Copy source into WASM heap
    mod.HEAPU8().set(
      new Uint8Array(srcBuffer.buffer || srcBuffer, srcBuffer.byteOffset || 0, srcSize),
      srcPtr,
    );

    const resultSize = decompress(destPtr, destSize, srcPtr, srcSize);

    try {
      if (resultSize < 0) throw new Error("WASM decompression error");
      const output = new Uint8Array(resultSize);
      output.set(mod.HEAPU8().subarray(destPtr, destPtr + resultSize));
      return output;
    } finally {
      free(srcPtr);
      free(destPtr);
    }
  };
}

/**
 * LZ4 needs a persistent decompression context.
 */
function makeLz4Decompressor(mod, mallocKey, freeKey, createCtxKey, decompressFrameKey) {
  const malloc = mod.exports[mallocKey];
  const free = mod.exports[freeKey];
  const createCtx = mod.exports[createCtxKey];
  const decompressFrame = mod.exports[decompressFrameKey];
  let ctx = null;

  return (srcBuffer, destSize) => {
    if (!ctx) ctx = createCtx();
    const srcSize = srcBuffer.byteLength;
    const srcPtr = malloc(srcSize);
    const destPtr = malloc(destSize);

    mod.HEAPU8().set(
      new Uint8Array(srcBuffer.buffer || srcBuffer, srcBuffer.byteOffset || 0, srcSize),
      srcPtr,
    );

    const resultSize = decompressFrame(ctx, destPtr, destSize, srcPtr, srcSize);

    try {
      if (resultSize < 0) throw new Error("WASM LZ4 decompression error");
      const output = new Uint8Array(resultSize);
      output.set(mod.HEAPU8().subarray(destPtr, destPtr + resultSize));
      return output;
    } finally {
      free(srcPtr);
      free(destPtr);
    }
  };
}

async function tryLoadWasm() {
  if (wasmLoadAttempted) return;
  wasmLoadAttempted = true;

  // Load both WASM modules in parallel
  const results = await Promise.allSettled([
    fetch("/wasm/zstd-dec.wasm")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.arrayBuffer();
      })
      .then((bytes) => {
        // ZSTD exports: f=malloc, g=free, j=decompress
        const mod = loadEmscriptenModule(bytes);
        wasmZstdDecompress = makeDecompressor(mod, "f", "g", "j");
        console.log("[Worker] WASM zstd loaded successfully");
      }),
    fetch("/wasm/lz4-dec.wasm")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.arrayBuffer();
      })
      .then((bytes) => {
        // LZ4 exports: e=malloc, f=free, i=createDecompressionContext, h=decompressFrame
        const mod = loadEmscriptenModule(bytes);
        wasmLz4Decompress = makeLz4Decompressor(mod, "e", "f", "i", "h");
        console.log("[Worker] WASM lz4 loaded successfully");
      }),
  ]);

  for (const r of results) {
    if (r.status === "rejected") {
      console.warn("[Worker] WASM load failed (using JS fallback):", r.reason?.message);
    }
  }
}

// ---------------------------------------------------------------------------
// IReadable — HTTP Range requests
// ---------------------------------------------------------------------------

class HttpRangeReadable {
  constructor(url) {
    this._url = url;
    this._fileSize = null;
  }

  async size() {
    if (this._fileSize != null) return this._fileSize;
    const res = await fetch(this._url, { method: "HEAD" });
    if (!res.ok) throw new Error(`HEAD failed: HTTP ${res.status}`);
    const cl = res.headers.get("Content-Length");
    if (!cl) throw new Error("No Content-Length header");
    this._fileSize = BigInt(cl);
    return this._fileSize;
  }

  async read(offset, size) {
    const start = Number(offset);
    const end = start + Number(size) - 1;
    const res = await fetch(this._url, {
      headers: { Range: `bytes=${start}-${end}` },
    });
    if (!res.ok && res.status !== 206) {
      throw new Error(`Range request failed: HTTP ${res.status}`);
    }
    return new Uint8Array(await res.arrayBuffer());
  }
}

// ---------------------------------------------------------------------------
// Decompression handlers — WASM with JS fallback
// ---------------------------------------------------------------------------

function getDecompressHandlers() {
  return {
    zstd: (buffer, _decompressedSize) => {
      if (wasmZstdDecompress) {
        try {
          return wasmZstdDecompress(buffer, Number(_decompressedSize));
        } catch {
          // fallback to JS
        }
      }
      return decompressZstdJS(buffer);
    },
    lz4: (buffer, decompressedSize) => {
      if (wasmLz4Decompress) {
        try {
          return wasmLz4Decompress(buffer, Number(decompressedSize));
        } catch {
          // fallback to JS
        }
      }
      const decompressed = new Uint8Array(Number(decompressedSize));
      lz4js.decompressBlock(buffer, decompressed, 0, buffer.length, 0);
      return decompressed;
    },
    "": (buffer) => buffer,
  };
}

// ---------------------------------------------------------------------------
// CDR parser — CompressedImage
// ---------------------------------------------------------------------------

function parseCompressedImage(rawData) {
  const view = new DataView(rawData.buffer, rawData.byteOffset, rawData.byteLength);
  let offset = 0;

  // CDR encapsulation header
  offset += 4;

  const stampSec = view.getInt32(offset, true);
  offset += 4;
  const stampNanosec = view.getUint32(offset, true);
  offset += 4;

  // Header.frame_id (CDR string)
  const frameIdLen = view.getUint32(offset, true);
  offset += 4;
  offset += frameIdLen;
  offset = (offset + 3) & ~3;

  // format (CDR string)
  const formatLen = view.getUint32(offset, true);
  offset += 4;
  const formatBytes = rawData.slice(offset, offset + formatLen - 1);
  const format = new TextDecoder().decode(formatBytes);
  offset += formatLen;
  offset = (offset + 3) & ~3;

  // data (CDR sequence<uint8>)
  const dataLen = view.getUint32(offset, true);
  offset += 4;
  const data = rawData.slice(offset, offset + dataLen);

  return { format, data, stampSec, stampNanosec };
}

// ---------------------------------------------------------------------------
// Camera name extraction
// ---------------------------------------------------------------------------

function extractCameraName(topic) {
  const parts = topic.split("/").filter(Boolean);
  for (const part of parts) {
    if (part.startsWith("cam_")) return part;
  }
  const imgIdx = parts.findIndex((p) => p === "image_raw" || p === "compressed");
  if (imgIdx > 0) return parts[imgIdx - 1];
  return topic.replace(/\//g, "_").replace(/^_/, "");
}

// ---------------------------------------------------------------------------
// Worker state
// ---------------------------------------------------------------------------

let reader = null;
let imageTopics = [];
let startTimeNs = BigInt(0);
let endTimeNs = BigInt(0);
let mcapDuration = 0;
let channelMap = new Map(); // channelId → topicIdx

// Producer state (Phase 3) — token-based cancellation to prevent race conditions
let producerRunning = false;
let producerToken = 0; // incremented on each start/stop to invalidate stale producers

// Frame queue for producer-consumer (Phase 3)
const frameQueue = [];
const MAX_QUEUE_SIZE = 3000; // ~10s at 4 cameras × 15fps × ~5
let queueResolve = null; // Wake up consumer when frames available

// ---------------------------------------------------------------------------
// API exposed via Comlink
// ---------------------------------------------------------------------------

const api = {
  /**
   * Initialize the MCAP reader. Downloads index, builds frame metadata.
   * @param {string} mcapUrl
   * @returns {{ imageTopics, startTimeNs, endTimeNs, duration }}
   */
  async initialize(mcapUrl) {
    // Try loading WASM decompressors (non-blocking, fallback to JS)
    await tryLoadWasm();

    const readable = new HttpRangeReadable(mcapUrl);
    reader = await McapIndexedReader.Initialize({
      readable,
      decompressHandlers: getDecompressHandlers(),
    });

    imageTopics = [];
    for (const channel of reader.channelsById.values()) {
      const schema = reader.schemasById.get(channel.schemaId);
      if (schema && schema.name.includes("CompressedImage")) {
        imageTopics.push({
          topic: channel.topic,
          channelId: channel.id,
          cameraName: extractCameraName(channel.topic),
          frameCount: 0,
          fps: 0,
        });
      }
    }

    // Build channelId → topicIdx map
    channelMap = new Map();
    for (let i = 0; i < imageTopics.length; i++) {
      channelMap.set(imageTopics[i].channelId, i);
    }

    // Time range from chunk indexes
    let globalStartNs = null;
    let globalEndNs = null;
    for (const chunk of reader.chunkIndexes) {
      if (globalStartNs === null || chunk.messageStartTime < globalStartNs)
        globalStartNs = chunk.messageStartTime;
      if (globalEndNs === null || chunk.messageEndTime > globalEndNs)
        globalEndNs = chunk.messageEndTime;
    }

    startTimeNs = globalStartNs ?? BigInt(0);
    endTimeNs = globalEndNs ?? BigInt(0);
    mcapDuration = Number(endTimeNs - startTimeNs) / 1e9;

    // Frame counts from statistics
    if (reader.statistics) {
      for (const info of imageTopics) {
        const count = reader.statistics.channelMessageCounts?.get(info.channelId);
        if (count != null) {
          info.frameCount = Number(count);
          info.fps = mcapDuration > 0 ? info.frameCount / mcapDuration : 0;
        }
      }
    }

    // Return serializable data (BigInt → string for Comlink)
    return {
      imageTopics,
      startTimeNs: startTimeNs.toString(),
      endTimeNs: endTimeNs.toString(),
      duration: mcapDuration,
    };
  },

  /**
   * Read nearest frame for all topics at a target time.
   * Returns array of { topicIdx, logTime (string), jpeg (ArrayBuffer) }.
   * JPEG ArrayBuffers are transferred (zero-copy).
   */
  async readFramesAtTime(targetNsStr, windowNsStr) {
    if (!reader) throw new Error("Reader not initialized");

    const targetNs = BigInt(targetNsStr);
    const windowNs = windowNsStr ? BigInt(windowNsStr) : BigInt(500_000_000);
    const topics = imageTopics.map((t) => t.topic);

    const results = new Map();
    const bestDiff = new Map();

    const searchStart = targetNs > windowNs ? targetNs - windowNs : BigInt(0);
    const searchEnd = targetNs + windowNs;

    for await (const message of reader.readMessages({
      topics,
      startTime: searchStart,
      endTime: searchEnd,
      validateCrcs: false,
    })) {
      const channel = reader.channelsById.get(message.channelId);
      if (!channel) continue;

      const diff = message.logTime > targetNs
        ? message.logTime - targetNs
        : targetNs - message.logTime;

      const prevDiff = bestDiff.get(channel.topic);
      if (prevDiff == null || diff < prevDiff) {
        results.set(channel.topic, message);
        bestDiff.set(channel.topic, diff);
      }
    }

    // Parse and transfer JPEG bytes
    const frames = [];
    const transferables = [];

    for (const [topic, message] of results) {
      const topicIdx = imageTopics.findIndex((t) => t.topic === topic);
      if (topicIdx < 0) continue;

      try {
        const parsed = parseCompressedImage(new Uint8Array(message.data));
        if (parsed.data && parsed.data.length > 0) {
          const buf = parsed.data.buffer.slice(
            parsed.data.byteOffset,
            parsed.data.byteOffset + parsed.data.byteLength
          );
          frames.push({
            topicIdx,
            logTime: message.logTime.toString(),
            format: parsed.format,
          });
          transferables.push(buf);
        }
      } catch {
        // skip malformed frames
      }
    }

    return Comlink.transfer({ frames, buffers: transferables }, transferables);
  },

  /**
   * Read all frames in a time range (batch pre-buffer).
   * Returns array of { topicIdx, logTime, jpeg } with transferred buffers.
   */
  async readFrameBatch(startNsStr, endNsStr) {
    if (!reader) throw new Error("Reader not initialized");

    const batchStartNs = BigInt(startNsStr);
    const batchEndNs = BigInt(endNsStr);
    const topics = imageTopics.map((t) => t.topic);

    const frames = [];
    const transferables = [];

    for await (const message of reader.readMessages({
      topics,
      startTime: batchStartNs,
      endTime: batchEndNs,
      validateCrcs: false,
    })) {
      const topicIdx = channelMap.get(message.channelId);
      if (topicIdx == null) continue;

      try {
        const parsed = parseCompressedImage(new Uint8Array(message.data));
        if (parsed.data && parsed.data.length > 0) {
          const buf = parsed.data.buffer.slice(
            parsed.data.byteOffset,
            parsed.data.byteOffset + parsed.data.byteLength
          );
          frames.push({
            topicIdx,
            logTime: message.logTime.toString(),
            format: parsed.format,
          });
          transferables.push(buf);
        }
      } catch {
        // skip malformed
      }
    }

    return Comlink.transfer({ frames, buffers: transferables }, transferables);
  },

  // -----------------------------------------------------------------------
  // Phase 3: Producer-Consumer streaming
  // -----------------------------------------------------------------------

  /**
   * Start the producer loop. Reads ahead from startNsStr, filling the
   * internal queue. Consumer calls getNextBatch() to drain.
   *
   * Uses token-based cancellation: each start/stop increments producerToken.
   * A producer loop exits when its token no longer matches the current token.
   */
  async startProducer(startNsStr) {
    // Invalidate any old producer by incrementing token
    const myToken = ++producerToken;
    producerRunning = true;

    // Clear queue
    frameQueue.length = 0;

    const fromNs = BigInt(startNsStr);
    const topics = imageTopics.map((t) => t.topic);
    let msgCount = 0;

    try {
      for await (const message of reader.readMessages({
        topics,
        startTime: fromNs,
        endTime: endTimeNs,
        validateCrcs: false,
      })) {
        if (myToken !== producerToken) break;

        const topicIdx = channelMap.get(message.channelId);
        if (topicIdx == null) continue;

        try {
          const parsed = parseCompressedImage(new Uint8Array(message.data));
          if (parsed.data && parsed.data.length > 0) {
            const buf = parsed.data.buffer.slice(
              parsed.data.byteOffset,
              parsed.data.byteOffset + parsed.data.byteLength
            );

            frameQueue.push({
              topicIdx,
              logTime: message.logTime.toString(),
              format: parsed.format,
              buffer: buf,
            });

            // Wake up consumer if waiting (signal-based drain)
            if (queueResolve) {
              queueResolve();
              queueResolve = null;
            }

            // Yield every 30 messages so Comlink can process getNextBatch()
            // Without this, the for-await microtask chain blocks the event loop
            // until the entire chunk is processed (~300 messages).
            // Note: SharedArrayBuffer + Atomics would eliminate this yield,
            // but requires COOP/COEP headers on nginx.
            if (++msgCount % 30 === 0) {
              await new Promise((r) => setTimeout(r, 0));
              if (myToken !== producerToken) break;
            }

            // Back-pressure: wait if queue is full
            while (frameQueue.length >= MAX_QUEUE_SIZE && myToken === producerToken) {
              await new Promise((r) => setTimeout(r, 50));
            }
            if (myToken !== producerToken) break;
          }
        } catch {
          // skip
        }
      }
    } catch (e) {
      if (myToken === producerToken) {
        console.error("[Worker] Producer error:", e);
      }
    } finally {
      if (myToken === producerToken) {
        producerRunning = false;
        // Wake consumer one last time so it knows producer is done
        if (queueResolve) {
          queueResolve();
          queueResolve = null;
        }
      }
    }
  },

  /**
   * Stop the producer loop.
   */
  stopProducer() {
    producerToken++; // invalidate current producer
    producerRunning = false;
    frameQueue.length = 0;
  },

  /**
   * Get next batch of frames from the queue.
   * If queue is empty and producer is running, waits up to `waitMs` for
   * the producer to signal that frames are available (signal-based drain).
   * Returns up to `count` frames with transferred buffers.
   */
  async getNextBatch(count = 60, waitMs = 0) {
    // Signal-based wait: if queue is empty, wait for producer to wake us
    if (frameQueue.length === 0 && waitMs > 0 && producerRunning) {
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, waitMs);
        queueResolve = () => {
          clearTimeout(timer);
          resolve();
        };
      });
    }

    const batch = frameQueue.splice(0, Math.min(count, frameQueue.length));
    if (batch.length === 0) {
      return { frames: [], buffers: [], producerDone: !producerRunning };
    }

    const frames = [];
    const transferables = [];
    for (const item of batch) {
      frames.push({
        topicIdx: item.topicIdx,
        logTime: item.logTime,
        format: item.format,
      });
      transferables.push(item.buffer);
    }

    return Comlink.transfer(
      { frames, buffers: transferables, producerDone: !producerRunning },
      transferables
    );
  },

  /**
   * Get queue status for debugging.
   */
  getQueueStatus() {
    return {
      queueSize: frameQueue.length,
      producerRunning,
      maxQueueSize: MAX_QUEUE_SIZE,
    };
  },

  /**
   * Clean up resources.
   */
  cleanup() {
    producerToken++; // invalidate any running producer
    frameQueue.length = 0;
    reader = null;
    imageTopics = [];
    channelMap.clear();
  },
};

Comlink.expose(api);
