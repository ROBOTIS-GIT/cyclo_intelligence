/* eslint-disable no-restricted-globals */
/**
 * Image Decoder Worker (Phase 4)
 *
 * Decodes JPEG/PNG bytes into ImageBitmap off the main thread.
 * ImageBitmap is Transferable → zero-copy to main thread.
 *
 * Pipeline: JPEG ArrayBuffer → createImageBitmap() → ImageBitmap transfer
 */

import * as Comlink from "comlink";

const api = {
  /**
   * Decode a batch of JPEG/PNG buffers into ImageBitmaps.
   *
   * @param {Array<{ topicIdx: number, logTime: string, format: string }>} frames
   * @param {ArrayBuffer[]} buffers - Corresponding JPEG/PNG data
   * @returns {{ bitmaps: ImageBitmap[], frames: Array }} with transferred bitmaps
   */
  async decodeBatch(frames, buffers) {
    const promises = [];
    const validIndices = [];

    for (let i = 0; i < frames.length; i++) {
      const buf = buffers[i];
      if (!buf || buf.byteLength === 0) continue;

      const format = frames[i].format;
      const mimeType = format && format.includes("png") ? "image/png" : "image/jpeg";
      const blob = new Blob([buf], { type: mimeType });

      validIndices.push(i);
      promises.push(createImageBitmap(blob));
    }

    const results = await Promise.allSettled(promises);

    const outFrames = [];
    const outBitmaps = [];
    const transferables = [];

    for (let j = 0; j < results.length; j++) {
      if (results[j].status === "fulfilled") {
        const bitmap = results[j].value;
        const idx = validIndices[j];
        outFrames.push(frames[idx]);
        outBitmaps.push(bitmap);
        transferables.push(bitmap);
      }
    }

    return Comlink.transfer(
      { frames: outFrames, bitmaps: outBitmaps },
      transferables
    );
  },

  /**
   * Decode a single frame.
   */
  async decodeSingle(buffer, format) {
    const mimeType = format && format.includes("png") ? "image/png" : "image/jpeg";
    const blob = new Blob([buffer], { type: mimeType });
    const bitmap = await createImageBitmap(blob);
    return Comlink.transfer(bitmap, [bitmap]);
  },
};

Comlink.expose(api);
