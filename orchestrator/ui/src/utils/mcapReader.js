/* global BigInt */
/**
 * MCAP Direct Reader — Lichtblick-style browser-side MCAP reading.
 *
 * Uses @mcap/core McapIndexedReader with HTTP Range requests to read
 * MCAP files directly in the browser without server-side processing.
 *
 * Key optimization: batch reading ALL image topics in a single
 * readMessages() call so each chunk is downloaded/decompressed once.
 */

import { McapIndexedReader } from "@mcap/core";
import { decompress as decompressZstd } from "fzstd";
import lz4 from "lz4js";

// ---------------------------------------------------------------------------
// IReadable implementation — HTTP Range requests
// ---------------------------------------------------------------------------

class HttpRangeReadable {
  constructor(url) {
    this._url = url;
    this._fileSize = null;
  }

  async size() {
    if (this._fileSize != null) {
      return this._fileSize;
    }
    console.log("[MCAP HTTP] HEAD request:", this._url);
    const res = await fetch(this._url, { method: "HEAD" });
    if (!res.ok) {
      throw new Error(`Failed to get file size: HTTP ${res.status} for ${this._url}`);
    }
    const contentLength = res.headers.get("Content-Length");
    if (!contentLength) {
      throw new Error("Server did not return Content-Length header");
    }
    this._fileSize = BigInt(contentLength);
    console.log("[MCAP HTTP] File size:", Number(this._fileSize), "bytes");
    return this._fileSize;
  }

  async read(offset, size) {
    const start = Number(offset);
    const end = start + Number(size) - 1;
    const res = await fetch(this._url, {
      headers: { Range: `bytes=${start}-${end}` },
    });
    if (!res.ok && res.status !== 206) {
      throw new Error(`Range request failed: HTTP ${res.status} for bytes=${start}-${end}`);
    }
    return new Uint8Array(await res.arrayBuffer());
  }
}

// ---------------------------------------------------------------------------
// Decompression handlers (pure JS — CRA compatible)
// ---------------------------------------------------------------------------

const decompressHandlers = {
  zstd: (buffer, _decompressedSize) => {
    return decompressZstd(buffer);
  },
  lz4: (buffer, decompressedSize) => {
    const decompressed = new Uint8Array(Number(decompressedSize));
    lz4.decompressBlock(buffer, decompressed, 0, buffer.length, 0);
    return decompressed;
  },
  "": (buffer) => buffer,
};

// ---------------------------------------------------------------------------
// MCAP reader initialization
// ---------------------------------------------------------------------------

export async function openMcapReader(mcapUrl) {
  const readable = new HttpRangeReadable(mcapUrl);
  const reader = await McapIndexedReader.Initialize({
    readable,
    decompressHandlers,
  });
  console.log("[MCAP] Reader initialized:", {
    channels: reader.channelsById.size,
    schemas: reader.schemasById.size,
    chunkIndexes: reader.chunkIndexes?.length ?? 0,
    hasStatistics: !!reader.statistics,
  });
  return reader;
}

// ---------------------------------------------------------------------------
// CDR deserialization — CompressedImage only
// ---------------------------------------------------------------------------

export function parseCompressedImage(rawData) {
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
// Frame index builder
// ---------------------------------------------------------------------------

function extractCameraName(topic) {
  const parts = topic.split("/").filter(Boolean);
  for (const part of parts) {
    if (part.startsWith("cam_")) {
      return part;
    }
  }
  const imgIdx = parts.findIndex(
    (p) => p === "image_raw" || p === "compressed"
  );
  if (imgIdx > 0) {
    return parts[imgIdx - 1];
  }
  return topic.replace(/\//g, "_").replace(/^_/, "");
}

export async function buildFrameIndex(reader) {
  const imageTopics = [];

  for (const channel of reader.channelsById.values()) {
    const schema = reader.schemasById.get(channel.schemaId);
    if (schema && schema.name.includes("CompressedImage")) {
      const cameraName = extractCameraName(channel.topic);
      imageTopics.push({
        topic: channel.topic,
        channelId: channel.id,
        cameraName,
        frameCount: 0,
        fps: 0,
      });
    }
  }

  if (imageTopics.length === 0) {
    return { imageTopics, startTimeNs: BigInt(0), endTimeNs: BigInt(0), duration: 0 };
  }

  let globalStartNs = null;
  let globalEndNs = null;
  for (const chunk of reader.chunkIndexes) {
    if (globalStartNs === null || chunk.messageStartTime < globalStartNs)
      globalStartNs = chunk.messageStartTime;
    if (globalEndNs === null || chunk.messageEndTime > globalEndNs)
      globalEndNs = chunk.messageEndTime;
  }

  const startTimeNs = globalStartNs ?? BigInt(0);
  const endTimeNs = globalEndNs ?? BigInt(0);
  const duration = Number(endTimeNs - startTimeNs) / 1e9;

  if (reader.statistics) {
    for (const info of imageTopics) {
      const count = reader.statistics.channelMessageCounts?.get(info.channelId);
      if (count != null) {
        info.frameCount = Number(count);
        info.fps = duration > 0 ? info.frameCount / duration : 0;
      }
    }
  }

  return { imageTopics, startTimeNs, endTimeNs, duration };
}

// ---------------------------------------------------------------------------
// Batch frame reading — ALL topics in ONE readMessages call
// ---------------------------------------------------------------------------

/**
 * Read closest frame for ALL topics in a single readMessages() call.
 * Each MCAP chunk is downloaded and decompressed only ONCE regardless
 * of how many topics are requested — critical for performance.
 *
 * @returns {Map<string, {logTime: bigint, data: Uint8Array}>} topic → message
 */
export async function readAllFramesAtTime(reader, topics, targetNs, windowNs = BigInt(500_000_000)) {
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
    const topic = channel.topic;

    const diff = message.logTime > targetNs
      ? message.logTime - targetNs
      : targetNs - message.logTime;

    const prevDiff = bestDiff.get(topic);
    if (prevDiff == null || diff < prevDiff) {
      results.set(topic, message);
      bestDiff.set(topic, diff);
    }
  }

  return results;
}

/**
 * Read ALL frames for ALL topics in a time range (batch pre-buffer).
 * Downloads and decompresses each chunk once, extracting all image messages.
 *
 * @param {McapIndexedReader} reader
 * @param {string[]} topics
 * @param {bigint} startNs
 * @param {bigint} endNs
 * @param {function} onFrame - callback(channelId, logTime, data) called per frame
 * @returns {Promise<number>} total frames yielded
 */
export async function readFrameBatch(reader, topics, startNs, endNs, onFrame) {
  let count = 0;

  for await (const message of reader.readMessages({
    topics,
    startTime: startNs,
    endTime: endNs,
    validateCrcs: false,
  })) {
    onFrame(message.channelId, message.logTime, message.data);
    count++;
  }

  return count;
}
