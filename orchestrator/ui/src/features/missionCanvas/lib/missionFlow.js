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

import { localBtPathForSpot } from "./missionBtFiles";

export function missionFlowEdgeId(source, target) {
  return `mission_flow_${source}_${target}`;
}

export function missionFlowEdgesForRouteOrder(orderedIds, closed = false) {
  const ids = orderedIds.filter((id, index) => id && orderedIds.indexOf(id) === index);
  const edges = ids.slice(0, -1).map((source, index) => {
    const target = ids[index + 1];
    return {
      id: missionFlowEdgeId(source, target),
      source,
      target,
      type: "smoothstep",
      animated: false,
    };
  });
  if (closed && ids.length > 1) {
    const source = ids[ids.length - 1];
    const target = ids[0];
    edges.push({
      id: missionFlowEdgeId(source, target),
      source,
      target,
      type: "smoothstep",
      animated: false,
    });
  }
  return edges;
}

export function missionFlowNodeForSpot(spot, index, position) {
  return {
    id: spot.id,
    type: "missionWaypoint",
    position: position ?? { x: 80 + index * 220, y: 72 },
    data: {
      label: spot.label || spot.id,
      localBt: localBtPathForSpot(spot),
    },
  };
}

export function normalizeMissionFlow(spots, storedFlow = null) {
  const storedNodes = Array.isArray(storedFlow?.nodes) ? storedFlow.nodes : [];
  const storedEdges = Array.isArray(storedFlow?.edges) ? storedFlow.edges : [];
  const validSpotIds = new Set(spots.map((spot) => spot.id));
  const storedById = new Map(storedNodes.map((node) => [node.id, node]));
  const nodes = spots.map((spot, index) => {
    const storedNode = storedById.get(spot.id);
    const storedPosition = storedNode?.position;
    const position = storedPosition &&
      Number.isFinite(Number(storedPosition.x)) &&
      Number.isFinite(Number(storedPosition.y))
      ? { x: Number(storedPosition.x), y: Number(storedPosition.y) }
      : undefined;
    return missionFlowNodeForSpot(spot, index, position);
  });
  const edges = storedEdges
    .filter((edge) => validSpotIds.has(edge.source) && validSpotIds.has(edge.target))
    .map((edge) => ({
      id: edge.id || missionFlowEdgeId(edge.source, edge.target),
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
      animated: false,
    }));
  return { nodes, edges };
}

export function syncMissionFlowNodesWithSpots(nodes, spots) {
  const validSpotIds = new Set(spots.map((spot) => spot.id));
  const byId = new Map(nodes.filter((node) => validSpotIds.has(node.id)).map((node) => [node.id, node]));
  const maxX = nodes.reduce((max, node) => Math.max(max, Number(node.position?.x ?? 0)), 80);
  let added = 0;
  return spots.map((spot, index) => {
    const existing = byId.get(spot.id);
    if (existing) {
      return {
        ...existing,
        type: "missionWaypoint",
        data: {
          ...(existing.data ?? {}),
          label: spot.label || spot.id,
          localBt: localBtPathForSpot(spot),
        },
      };
    }
    const node = missionFlowNodeForSpot(spot, index, { x: maxX + 220 * (added + 1), y: 72 });
    added += 1;
    return node;
  });
}

export function filterMissionFlowEdges(edges, spots) {
  const validSpotIds = new Set(spots.map((spot) => spot.id));
  return edges.filter((edge) => validSpotIds.has(edge.source) && validSpotIds.has(edge.target));
}

export function serializeMissionFlow(nodes, edges) {
  return {
    nodes: nodes.map((node) => ({
      id: node.id,
      position: {
        x: Number(node.position?.x ?? 0),
        y: Number(node.position?.y ?? 0),
      },
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
    })),
  };
}

export function orderedSpotIdsFromMissionFlow(spots, nodes, edges, { includeClosingTarget = false } = {}) {
  const spotById = new Map(spots.map((spot) => [spot.id, spot]));
  const validEdges = edges.filter((edge) => (
    spotById.has(edge.source) && spotById.has(edge.target)
  ));
  if (validEdges.length === 0) return [];
  const connectedIds = new Set();
  validEdges.forEach((edge) => {
    connectedIds.add(edge.source);
    connectedIds.add(edge.target);
  });
  const flowNodes = nodes.filter((node) => connectedIds.has(node.id));
  const nodeById = new Map(flowNodes.map((node) => [node.id, node]));
  spots.forEach((spot, index) => {
    if (!connectedIds.has(spot.id) || nodeById.has(spot.id)) return;
    nodeById.set(spot.id, missionFlowNodeForSpot(spot, index));
  });
  const ids = [...nodeById.keys()];
  const incoming = new Map(ids.map((id) => [id, 0]));
  const outgoing = new Map(ids.map((id) => [id, []]));
  validEdges.forEach((edge) => {
    if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) return;
    outgoing.get(edge.source).push(edge);
    incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1);
  });
  const byPosition = (a, b) => {
    const nodeA = nodeById.get(a);
    const nodeB = nodeById.get(b);
    const xDiff = Number(nodeA?.position?.x ?? 0) - Number(nodeB?.position?.x ?? 0);
    if (Math.abs(xDiff) > 0.001) return xDiff;
    return Number(nodeA?.position?.y ?? 0) - Number(nodeB?.position?.y ?? 0);
  };
  outgoing.forEach((targets) => targets.sort((a, b) => byPosition(a.target, b.target)));
  const startId = ids.find((id) => (incoming.get(id) || 0) === 0) || validEdges[0].source;
  const visitedEdges = new Set();
  const orderedIds = [];
  let currentId = startId;
  while (currentId && nodeById.has(currentId)) {
    orderedIds.push(currentId);
    const nextEdge = (outgoing.get(currentId) || [])[0];
    if (!nextEdge) break;
    const edgeKey = nextEdge.id || missionFlowEdgeId(nextEdge.source, nextEdge.target);
    if (visitedEdges.has(edgeKey)) break;
    visitedEdges.add(edgeKey);
    currentId = nextEdge.target;
    if (currentId === startId) {
      if (includeClosingTarget) orderedIds.push(currentId);
      break;
    }
  }
  return orderedIds;
}

export function orderedSpotsFromMissionFlow(spots, nodes, edges) {
  const spotById = new Map(spots.map((spot) => [spot.id, spot]));
  return orderedSpotIdsFromMissionFlow(spots, nodes, edges)
    .map((id) => spotById.get(id))
    .filter(Boolean);
}

export function missionStepSpotsFromMissionFlow(spots, nodes, edges) {
  const spotById = new Map(spots.map((spot) => [spot.id, spot]));
  return orderedSpotIdsFromMissionFlow(spots, nodes, edges, { includeClosingTarget: true })
    .map((id) => spotById.get(id))
    .filter(Boolean);
}
