// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useMemo, useRef, useState } from "react";
import {
  behaviorNodeDefinition,
  behaviorNodeId,
  behaviorNodeSerialFromNodes,
} from "../lib/designStore";
import { spotPoseFromMapPose } from "../lib/missionSpots";

export default function useDesignBehaviorNodeController({
  designMapName,
  runtimeMapName,
  markDirty,
  onMessage,
  captureDocumentLease,
  isDocumentLeaseCurrent,
  getCurrentIdentity,
  onPrepareSelect,
  onPlaced,
}) {
  const [nodes, setNodes] = useState([]);
  const [selectedNodeId, setSelectedNodeId] = useState("");
  const [pendingTag, setPendingTag] = useState("");
  const serialRef = useRef(0);
  const renderLease = captureDocumentLease();
  const renderIdentity = { ...getCurrentIdentity() };
  const documentCurrent = useCallback(() => {
    const current = getCurrentIdentity();
    return (
      isDocumentLeaseCurrent(renderLease)
      && current.mapName === renderIdentity.mapName
      && current.missionName === renderIdentity.missionName
    );
  }, [getCurrentIdentity, isDocumentLeaseCurrent, renderIdentity.mapName,
    renderIdentity.missionName, renderLease]);

  const designNodes = useMemo(
    () => nodes.filter((node) => node.map_name === designMapName),
    [designMapName, nodes],
  );
  const runNodes = useMemo(
    () => nodes.filter((node) => node.map_name === runtimeMapName),
    [nodes, runtimeMapName],
  );
  const previewNode = useMemo(
    () => (pendingTag ? behaviorNodeDefinition(pendingTag) : null),
    [pendingTag],
  );

  const advanceSerial = useCallback((nextNodes) => {
    serialRef.current = Math.max(
      serialRef.current,
      behaviorNodeSerialFromNodes(nextNodes),
    );
  }, []);

  const mergeMapPatch = useCallback((targetMapName, patch) => {
    if (!Array.isArray(patch)) return;
    setNodes((current) => [
      ...current.filter((node) => node.map_name !== targetMapName),
      ...patch,
    ]);
    advanceSerial(patch);
  }, [advanceSerial]);

  const getHistorySlice = useCallback(() => ({
    behaviorNodes: nodes,
    selectedBehaviorNodeId: selectedNodeId,
  }), [nodes, selectedNodeId]);

  const restoreHistorySlice = useCallback((slice) => {
    const restored = Array.isArray(slice?.behaviorNodes) ? slice.behaviorNodes : [];
    setNodes(restored);
    setSelectedNodeId(restored.some((node) => node.id === slice?.selectedBehaviorNodeId)
      ? slice.selectedBehaviorNodeId
      : "");
    advanceSerial(restored);
  }, [advanceSerial]);

  const clearSelection = useCallback(() => setSelectedNodeId(""), []);
  const clearPendingPlacement = useCallback(() => setPendingTag(""), []);

  const selectNode = useCallback((nodeId) => {
    if (!documentCurrent() || !designNodes.some((node) => node.id === nodeId)) return;
    onPrepareSelect?.(nodeId);
    setSelectedNodeId(nodeId);
    setPendingTag("");
  }, [designNodes, documentCurrent, onPrepareSelect]);

  const beginPlacement = useCallback((tag) => {
    if (!documentCurrent() || !behaviorNodeDefinition(tag)) return false;
    setPendingTag(tag);
    return true;
  }, [documentCurrent]);

  const placePendingAtPose = useCallback((x, y, yaw) => {
    if (!documentCurrent() || !pendingTag) return null;
    const definition = behaviorNodeDefinition(pendingTag);
    if (!definition) return null;
    serialRef.current += 1;
    const node = {
      id: behaviorNodeId(pendingTag, serialRef.current),
      map_name: designMapName,
      tag: pendingTag,
      label: pendingTag,
      category: definition.category || "action",
      pose: spotPoseFromMapPose(x, y, yaw),
      metadata: { source: "mission_canvas" },
    };
    markDirty();
    setNodes((current) => [...current, node]);
    setSelectedNodeId(node.id);
    setPendingTag("");
    onPlaced?.(node);
    onMessage(`Placed ${node.tag}`);
    return node;
  }, [designMapName, documentCurrent, markDirty, onMessage, onPlaced, pendingTag]);

  const moveNode = useCallback((nodeId, x, y, yaw) => {
    if (!documentCurrent()) return;
    const node = designNodes.find((item) => item.id === nodeId);
    if (!node) return;
    markDirty();
    setNodes((current) => current.map((item) => (
      item.map_name === designMapName && item.id === nodeId
        ? { ...item, pose: spotPoseFromMapPose(x, y, yaw ?? item.pose?.yaw ?? 0) }
        : item
    )));
    onMessage(`Moved ${node.tag || "node"}`);
  }, [designMapName, designNodes, documentCurrent, markDirty, onMessage]);

  const deleteNode = useCallback((node) => {
    if (
      !documentCurrent()
      || !node
      || node.map_name !== designMapName
      || !designNodes.some((item) => item.id === node.id)
    ) return;
    markDirty();
    setNodes((current) => current.filter((item) => !(
      item.map_name === designMapName && item.id === node.id
    )));
    setSelectedNodeId((current) => (current === node.id ? "" : current));
    onMessage(`Deleted ${node.tag}`);
  }, [designMapName, designNodes, documentCurrent, markDirty, onMessage]);

  return {
    nodes,
    designNodes,
    runNodes,
    selectedNodeId,
    pendingTag,
    previewNode,
    mergeMapPatch,
    getHistorySlice,
    restoreHistorySlice,
    clearSelection,
    clearPendingPlacement,
    selectNode,
    beginPlacement,
    placePendingAtPose,
    moveNode,
    deleteNode,
  };
}
