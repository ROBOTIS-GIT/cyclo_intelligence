/**
 * MCAP Reader Worker Proxy — Main thread interface to the MCAP Worker.
 *
 * CRA5 native Worker support: `new Worker(new URL(...), import.meta.url)`
 * Comlink.wrap() creates a proxy that makes Worker calls look like local async functions.
 */

import * as Comlink from "comlink";

let worker = null;
let remote = null;

/**
 * Get or create the MCAP Worker proxy (singleton).
 * @returns {{ remote: Object, dispose: Function }}
 */
export function getMcapWorker() {
  if (!worker) {
    worker = new Worker(
      new URL("./mcapReader.worker.js", import.meta.url),
      { type: "module" }
    );
    remote = Comlink.wrap(worker);
  }
  return { remote, dispose: disposeMcapWorker };
}

/**
 * Terminate the Worker and clean up.
 */
export function disposeMcapWorker() {
  if (remote) {
    remote[Comlink.releaseProxy]();
    remote = null;
  }
  if (worker) {
    worker.terminate();
    worker = null;
  }
}
