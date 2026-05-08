/**
 * Image Decoder Worker Proxy — Main thread interface.
 * Singleton pattern, same as mcapReaderProxy.
 */

import * as Comlink from "comlink";

let worker = null;
let remote = null;

export function getImageDecoderWorker() {
  if (!worker) {
    worker = new Worker(
      new URL("./imageDecoder.worker.js", import.meta.url),
      { type: "module" }
    );
    remote = Comlink.wrap(worker);
  }
  return { remote, dispose: disposeImageDecoderWorker };
}

export function disposeImageDecoderWorker() {
  if (remote) {
    remote[Comlink.releaseProxy]();
    remote = null;
  }
  if (worker) {
    worker.terminate();
    worker = null;
  }
}
