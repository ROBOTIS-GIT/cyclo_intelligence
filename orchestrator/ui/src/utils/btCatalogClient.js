// Copyright 2026 ROBOTIS CO., LTD.
import { getBtNodeReadiness } from "../features/mapWorkspace/lib/btNodeApi";

// A process can be up before its ROS services exist. Discovery must succeed
// before a catalog request; this helper never starts the Task Engine.
export async function readBtNodeCatalog(callService, { attempts = 10, intervalMs = 500 } = {}) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const status = await getBtNodeReadiness();
    if (status.ready === true) {
      return callService("/bt/nodes/catalog", "interfaces/srv/GetNodeCatalog", {});
    }
    if (attempt + 1 < attempts) {
      await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
  }
  throw new Error("Task Engine ROS services are not ready. Wait for startup, then retry.");
}
