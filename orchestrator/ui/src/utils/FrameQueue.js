/**
 * FrameQueue — Ring buffer for raw JPEG frame data.
 *
 * Stores frames indexed by topicIdx + logTime for nearest-neighbor lookup.
 * Memory-bounded: evicts oldest frames when exceeding limits.
 *
 * Data stored: { jpegBuffer: ArrayBuffer, format: string }
 * Raw JPEG buffers are ~50KB each (vs ~2MB for decoded ImageBitmap),
 * allowing much larger buffers with less memory.
 */

export class FrameQueue {
  /**
   * @param {number} maxFrames - Maximum frames to keep (across all topics)
   */
  constructor(maxFrames = 2000) {
    this.maxFrames = maxFrames;
    // Map<"topicIdx:logTime", { jpegBuffer, format }>
    this._cache = new Map();
    // Per-topic sorted timestamp arrays: topicIdx → BigInt[]
    this._timelines = [];
  }

  /**
   * Initialize timelines for N topics.
   */
  initTimelines(topicCount) {
    this._timelines = Array.from({ length: topicCount }, () => []);
  }

  /**
   * Insert a frame into the queue.
   * @param {number} topicIdx
   * @param {bigint} logTime
   * @param {{ jpegBuffer: ArrayBuffer, format: string }} data
   */
  put(topicIdx, logTime, data) {
    const key = `${topicIdx}:${logTime}`;
    if (this._cache.has(key)) return;
    this._cache.set(key, data);
    this._insertTimeline(topicIdx, logTime);
  }

  /**
   * Get nearest frame for a topic at a target time.
   * @param {number} topicIdx
   * @param {bigint} targetNs
   * @returns {{ data: { jpegBuffer, format }, logTime: bigint } | null}
   */
  getNearest(topicIdx, targetNs) {
    const timeline = this._timelines[topicIdx];
    if (!timeline || timeline.length === 0) return null;

    const idx = this._binarySearchNearest(timeline, targetNs);
    if (idx < 0) return null;

    const logTime = timeline[idx];
    const key = `${topicIdx}:${logTime}`;
    const data = this._cache.get(key);
    return data ? { data, logTime } : null;
  }

  /**
   * Get the next frame AFTER afterNs for a topic (for look-ahead decode).
   * @param {number} topicIdx
   * @param {bigint} afterNs
   * @returns {{ data: { jpegBuffer, format }, logTime: bigint } | null}
   */
  getNext(topicIdx, afterNs) {
    const timeline = this._timelines[topicIdx];
    if (!timeline || timeline.length === 0) return null;

    // Binary search: first timestamp strictly > afterNs
    let lo = 0;
    let hi = timeline.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (timeline[mid] <= afterNs) lo = mid + 1;
      else hi = mid;
    }
    if (lo >= timeline.length) return null;

    const logTime = timeline[lo];
    const key = `${topicIdx}:${logTime}`;
    const data = this._cache.get(key);
    return data ? { data, logTime } : null;
  }

  /**
   * Get buffer-ahead seconds from a given time.
   */
  getBufferAhead(topicIdx, currentNs) {
    const timeline = this._timelines[topicIdx];
    if (!timeline || timeline.length === 0) return 0;
    const lastNs = timeline[timeline.length - 1];
    if (lastNs <= currentNs) return 0;
    return Number(lastNs - currentNs) / 1e9;
  }

  /**
   * Evict frames before cutoffNs across all topics.
   */
  evictBefore(cutoffNs) {
    for (let i = 0; i < this._timelines.length; i++) {
      const timeline = this._timelines[i];
      let removeCount = 0;
      while (removeCount < timeline.length && timeline[removeCount] < cutoffNs) {
        const key = `${i}:${timeline[removeCount]}`;
        this._cache.delete(key);
        removeCount++;
      }
      if (removeCount > 0) timeline.splice(0, removeCount);
    }
  }

  /**
   * Evict all frames.
   */
  clear() {
    this._cache.clear();
    for (let i = 0; i < this._timelines.length; i++) {
      this._timelines[i] = [];
    }
  }

  get size() {
    return this._cache.size;
  }

  get timelines() {
    return this._timelines;
  }

  // -- internal --

  _insertTimeline(topicIdx, logTime) {
    const timeline = this._timelines[topicIdx];
    if (!timeline) return;

    let lo = 0;
    let hi = timeline.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (timeline[mid] < logTime) lo = mid + 1;
      else hi = mid;
    }
    if (lo < timeline.length && timeline[lo] === logTime) return;
    timeline.splice(lo, 0, logTime);
  }

  _binarySearchNearest(sortedArr, targetNs) {
    if (!sortedArr.length) return -1;
    let lo = 0;
    let hi = sortedArr.length - 1;

    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sortedArr[mid] < targetNs) lo = mid + 1;
      else hi = mid;
    }

    if (lo === 0) return 0;
    if (lo >= sortedArr.length) return sortedArr.length - 1;

    const diffHi = sortedArr[lo] - targetNs;
    const diffLo = targetNs - sortedArr[lo - 1];
    return diffLo <= diffHi ? lo - 1 : lo;
  }
}
