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

import { useRef, useState, useEffect, useLayoutEffect } from 'react';
import { parseBTXml } from '../../../utils/btTreeParser';
import { serializeFromGraph } from '../../../utils/btXmlSerializer';
import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';

// Parent-owned XML is emitted synchronously, only after the selected path is
// hydrated. Keep this guard in one place to prevent edits leaking across files.
export default function useMissionBtDocument({ graph, history, filePath, xml, onXmlChange, setSelectedNodeId }) {
  const { nodes, edges, nodeDataMap, parseError, setNodes, setEdges, setNodeDataMap, setParseError } = graph;
  const { resetHistory, coalescedEditRef } = history;
  const [hydratedPath, setHydratedPath] = useState('');
  const lastEmittedXmlRef = useRef(null);
  const lastEmittedPathRef = useRef('');
  const onXmlChangeRef = useRef(onXmlChange);
  onXmlChangeRef.current = onXmlChange;
  useEffect(() => {
    if (
      filePath === lastEmittedPathRef.current &&
      xml === lastEmittedXmlRef.current
    ) {
      return;
    }
    try {
      const parsed = parseBTXml(xml || "");
      // The prop already is parent-owned state. Treat its normalized graph as
      // the baseline instead of emitting the component's initial empty graph
      // (or marking formatting-only normalization as a user edit).
      lastEmittedXmlRef.current = serializeFromGraph(
        parsed.nodes || [],
        parsed.edges || [],
        parsed.nodeDataMap || new Map(),
      );
      lastEmittedPathRef.current = filePath;
      setNodes(parsed.nodes || []);
      setEdges(parsed.edges || []);
      setNodeDataMap(parsed.nodeDataMap || new Map());
      setSelectedNodeId(null);
      setParseError(null);
      setHydratedPath(filePath);
      coalescedEditRef.current = { key: "", lastChangeAt: 0 };
      resetHistory();
    } catch (error) {
      setNodes([]);
      setEdges([]);
      setNodeDataMap(new Map());
      setSelectedNodeId(null);
      setParseError(error instanceof Error
        ? formatTaskDisplayMessage(error.message, "Waypoint Task")
        : "Could not read this waypoint task file");
      setHydratedPath(filePath);
      coalescedEditRef.current = { key: "", lastChangeAt: 0 };
    }
  }, [
    coalescedEditRef,
    filePath,
    resetHistory,
    setEdges,
    setNodeDataMap,
    setNodes,
    setParseError,
    setSelectedNodeId,
    xml,
  ]);

  // Persist tree edits to the parent immediately (not debounced): a debounce
  // whose timer was reset by every parent re-render never fired, and switching
  // waypoints cleared the pending timer, silently dropping the whole tree.
  // serializeFromGraph omits node positions, so drags produce identical XML and
  // are skipped by the guard below — only real structural edits re-emit. filePath
  // is intentionally excluded from the deps: on a waypoint switch it changes one
  // render before `nodes` reloads, and emitting in that gap would write the old
  // tree to the new path.
  useLayoutEffect(() => {
    if (parseError || hydratedPath !== filePath) return;
    const emit = onXmlChangeRef.current;
    if (typeof emit !== "function") return;
    let serialized;
    try {
      serialized = serializeFromGraph(nodes, edges, nodeDataMap);
    } catch {
      return; // partial graph mid-edit; wait for the next change
    }
    if (filePath === lastEmittedPathRef.current && serialized === lastEmittedXmlRef.current) {
      return;
    }
    lastEmittedXmlRef.current = serialized;
    lastEmittedPathRef.current = filePath;
    emit(filePath, serialized);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [edges, hydratedPath, nodeDataMap, nodes, parseError]);

  return hydratedPath;
}
