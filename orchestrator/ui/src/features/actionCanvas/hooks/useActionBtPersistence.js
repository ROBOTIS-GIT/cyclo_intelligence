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

import { useCallback, useEffect, useRef } from 'react';
import { parseBTXml } from '../../../utils/btTreeParser';
import { serializeFromGraph } from '../../../utils/btXmlSerializer';
import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';
import { setTreeXml } from '../actionCanvasSlice';

export default function useActionBtPersistence({ graph, treeXml, dispatch }) {
  const { nodes, edges, nodeDataMap, nodesRef, edgesRef, nodeDataMapRef,
    setNodes, setEdges, setNodeDataMap, setParseError } = graph;
  const graphHydratedRef = useRef(!treeXml);
  // ── Initial load from Redux treeXml (e.g. on page mount) ─────────────────
  useEffect(() => {
    if (!treeXml) {
      graphHydratedRef.current = true;
      setNodes([]);
      setEdges([]);
      setNodeDataMap(new Map());
      setParseError(null);
      return;
    }
    try {
      const { nodes: n, edges: e, nodeDataMap: ndm } = parseBTXml(treeXml);
      graphHydratedRef.current = n.length === 0;
      setNodes(n);
      setEdges(e);
      setNodeDataMap(ndm);
      setParseError(null);
    } catch (err) {
      graphHydratedRef.current = false;
      setParseError(formatTaskDisplayMessage(err.message));
      setNodes([]);
      setEdges([]);
      setNodeDataMap(new Map());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount to restore Redux-persisted tree

  useEffect(() => {
    if (!graphHydratedRef.current && nodes.length > 0) {
      graphHydratedRef.current = true;
    }
  }, [nodes.length]);

  // ── Persist working tree to Redux so it survives page switches ────────────
  // The graph state (nodes/edges/nodeDataMap) lives in local useState, which
  // is torn down on unmount. Without this, navigating away and back to the BT
  // Manager wipes the user's in-progress tree. We serialise on a small debounce
  // so a flurry of edits doesn't dispatch on every keystroke, and again on
  // unmount to catch whatever's still in the debounce window.
  useEffect(() => {
    if (!graphHydratedRef.current) return undefined;
    const t = setTimeout(() => {
      try {
        dispatch(setTreeXml(
          nodes.length === 0 ? '' : serializeFromGraph(nodes, edges, nodeDataMap),
        ));
      } catch {
        // Partial graphs (e.g. mid-drag, disconnected nodes) can throw here.
        // Drop the snapshot rather than nuking the previously-good treeXml.
      }
    }, 400);
    return () => clearTimeout(t);
  }, [nodes, edges, nodeDataMap, dispatch]);

  // These are graph data refs, not DOM nodes. Read at cleanup time so the
  // most recent edit survives unmount even inside the debounce window.
  const serializeLatestGraph = useCallback(() => {
    const n = nodesRef.current;
    return n.length === 0 ? '' : serializeFromGraph(n, edgesRef.current, nodeDataMapRef.current);
  }, [nodesRef, edgesRef, nodeDataMapRef]);

  useEffect(() => () => {
    if (!graphHydratedRef.current) return;
    try {
      dispatch(setTreeXml(serializeLatestGraph()));
    } catch {
      // A partial graph must not overwrite the last valid persisted draft.
    }
  }, [dispatch, serializeLatestGraph]);
}
