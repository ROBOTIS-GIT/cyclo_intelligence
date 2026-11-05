// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { formatTaskDisplayMessage } from "../../../utils/taskTerminology";
import { getBtNodeServiceStatus, setBtNodeServiceActive } from "../lib/btNodeApi";

const BT_NODE_STATUS_POLL_MS = 5000;

// s6 reports "up" as soon as the launcher process exists, before ROS has
// necessarily registered /bt/load_and_run. Mission execution therefore waits
// for a read-only service created by the same node before using its mutating
// service.
const BT_NODE_ACTIVATION_POLL_MS = 500;

const BT_NODE_ACTIVATION_POLL_ATTEMPTS = 10;

const BT_NODE_READY_PROBE_TIMEOUT_MS = 1000;

function taskDisplayMessage(value) {
  return formatTaskDisplayMessage(value, "Waypoint Task");
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

/**
 * Own the Mission Run lease on bt_node without taking ownership away from
 * another workspace.
 *
 * A node that is already up is borrowed. A node started by this hook is owned
 * until releaseMissionBt resolves. Deliberately do not release on unmount:
 * runtime shutdown is coordinated by the mission runner and workspace exit
 * paths, and an unmount cleanup could race a later owner.
 */
export default function useMissionBtNodeLease({
  callService,
  needsBtTopics,
  onMessage,
}) {
  const btStatusRef = useRef("stopped");
  const btNodeReleaseRef = useRef(Promise.resolve());
  const missionBtNodeOwnedRef = useRef(false);
  const [btNodeStatus, setBtNodeStatus] = useState({
    state: "unknown",
    raw: "not checked",
  });
  const [btNodeBusy, setBtNodeBusy] = useState("");
  const setBtStatusText = useCallback((value) => {
    btStatusRef.current = value;
  }, []);

  const refreshBtNodeStatus = useCallback(async ({ quiet = false } = {}) => {
    try {
      const nextStatus = await getBtNodeServiceStatus();
      setBtNodeStatus(nextStatus);
      return nextStatus;
    } catch (error) {
      const nextStatus = {
        state: "unknown",
        raw: error instanceof Error ? error.message : "status failed",
      };
      setBtNodeStatus(nextStatus);
      if (!quiet) {
        onMessage(error instanceof Error
          ? `Task Engine status check failed: ${taskDisplayMessage(error.message)}`
          : "Task Engine status check failed");
      }
      return nextStatus;
    }
  }, [onMessage]);

  // Run borrows an already-up BT node, or owns the process only when it starts
  // one on demand. Cleanup must never stop a node owned by the standalone
  // workspace (or another external client).
  const ensureMissionBtActive = useCallback(async () => {
    // A completed/failed mission releases bt_node asynchronously. Serialize a
    // quick retry behind that shutdown so an old stop cannot kill the new node.
    await btNodeReleaseRef.current;
    const readState = async () => {
      try {
        const status = await getBtNodeServiceStatus();
        setBtNodeStatus(status);
        return status.state;
      } catch {
        return "unknown";
      }
    };
    const servicesAreReady = async () => {
      try {
        // /bt/nodes/catalog is created after /bt/load_and_run in bt_node's
        // constructor. A successful response is therefore a readiness barrier
        // for the mutating service used immediately after navigation arrives.
        const result = await callService(
          "/bt/nodes/catalog",
          "interfaces/srv/GetNodeCatalog",
          {},
          BT_NODE_READY_PROBE_TIMEOUT_MS,
        );
        return result?.success !== false;
      } catch {
        return false;
      }
    };

    setBtNodeBusy("activate");
    try {
      const initialState = await readState();
      missionBtNodeOwnedRef.current = false;
      if (initialState === "up") {
        const executionStatus = String(btStatusRef.current || "").trim().toLowerCase();
        if (executionStatus === "running" || executionStatus === "stopping") {
          onMessage(
            "Task Engine is already running another task. Stop it before running this mission.",
          );
          return false;
        }
        if (!["stopped", "completed", "failed"].includes(executionStatus)) {
          // Failing closed is intentional: /bt/load_and_run replaces the
          // current tree, so an unknown status must not be treated as idle.
          onMessage(
            "Unable to verify that the Task Engine is idle. Wait for its status and try again.",
          );
          return false;
        }
      }
      if (initialState !== "up" && initialState !== "down") {
        onMessage(
          "Unable to verify the Task Engine state. Try again after its status is available.",
        );
        return false;
      }
      if (initialState === "down") {
        await setBtNodeServiceActive(true);
        missionBtNodeOwnedRef.current = true;
      }
      for (let attempt = 0; attempt < BT_NODE_ACTIVATION_POLL_ATTEMPTS; attempt += 1) {
        if ((await readState()) === "up" && await servicesAreReady()) return true;
        await delay(BT_NODE_ACTIVATION_POLL_MS);
      }
      return false;
    } catch {
      return false;
    } finally {
      setBtNodeBusy("");
    }
  }, [callService, onMessage]);

  const releaseMissionBt = useCallback(async () => {
    const release = (async () => {
      if (!missionBtNodeOwnedRef.current) {
        try {
          setBtNodeStatus(await getBtNodeServiceStatus());
        } catch {
          setBtNodeStatus({ state: "unknown", raw: "status failed" });
        }
        return;
      }
      // Clear ownership before the request so duplicate cleanup paths cannot
      // stop a process that a later run has already borrowed or restarted.
      missionBtNodeOwnedRef.current = false;
      setBtNodeBusy("deactivate");
      try {
        await setBtNodeServiceActive(false);
      } catch {
        // Best-effort — the node may already be down.
      }
      try {
        setBtNodeStatus(await getBtNodeServiceStatus());
      } catch {
        setBtNodeStatus({ state: "unknown", raw: "status failed" });
      } finally {
        setBtNodeBusy("");
      }
    })();
    btNodeReleaseRef.current = release;
    await release;
  }, []);

  useEffect(() => {
    if (!needsBtTopics || document.visibilityState === "hidden") return undefined;
    void refreshBtNodeStatus({ quiet: true });
    const interval = window.setInterval(() => {
      void refreshBtNodeStatus({ quiet: true });
    }, BT_NODE_STATUS_POLL_MS);
    return () => {
      window.clearInterval(interval);
    };
  }, [needsBtTopics, refreshBtNodeStatus]);

  return {
    btNodeBusy,
    btNodeIsUp: btNodeStatus.state === "up",
    btNodeStatus,
    btStatusRef,
    ensureMissionBtActive,
    refreshBtNodeStatus,
    releaseMissionBt,
    setBtStatusText,
  };
}
