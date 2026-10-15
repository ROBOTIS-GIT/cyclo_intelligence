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

import { useCallback, useState } from 'react';
import toast from 'react-hot-toast';
import { parseBTXml } from '../../../utils/btTreeParser';
import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';
import { setTreeFileName, setTreeXml, setSelectedNodeId } from '../actionCanvasSlice';
import { readBtTree, saveBtTree } from '../btTreesApi';

export default function useActionBtFiles({ graph, dispatch, disarmClearTree,
  resetHistory, getSerializedXml }) {
  const { setNodes, setEdges, setNodeDataMap, setParseError } = graph;
  const [showTreeList, setShowTreeList] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveFileName, setSaveFileName] = useState('');
  const [saveConflict, setSaveConflict] = useState(null);
  const handleServerFileSelect = useCallback(async (item) => {
    const requestedName = item?.name || String(item?.full_path || '').split('/').pop();
    if (!requestedName) return;
    try {
      const tree = await readBtTree(requestedName);
      const xmlContent = tree.content || '';
      const fileName = tree.name || requestedName;

      const { nodes: n, edges: e, nodeDataMap: ndm } = parseBTXml(xmlContent);
      setNodes(n);
      setEdges(e);
      setNodeDataMap(ndm);
      setParseError(null);

      disarmClearTree();
      resetHistory();
      dispatch(setSelectedNodeId(null));
      dispatch(setTreeXml(xmlContent));
      dispatch(setTreeFileName(fileName));
      toast.success(`Opened: ${fileName}`);
    } catch (err) {
      toast.error(`Failed to load file: ${formatTaskDisplayMessage(err.message)}`);
    }
  }, [setNodes, setEdges, setNodeDataMap, setParseError, disarmClearTree, resetHistory, dispatch]);

  // ── Save As ───────────────────────────────────────────────────────────────
  const handleSaveAs = useCallback(async ({ overwrite = false } = {}) => {
    const name = saveFileName.trim();
    if (!name) return;

    const content = getSerializedXml();
    if (!content) return;

    try {
      const data = await saveBtTree({ filename: name, content, overwrite });
      toast.success(formatTaskDisplayMessage(data.message));
      setShowSaveDialog(false);
      setSaveFileName('');
      setSaveConflict(null);
    } catch (err) {
      const conflict = err.status === 409 || err.detail?.code === 'file_exists';
      if (conflict) {
        const detail = err.detail && typeof err.detail === 'object' ? err.detail : {};
        setSaveConflict({ ...detail, message: err.message });
        toast.error(formatTaskDisplayMessage(err.message) || 'File already exists');
        return;
      }
      toast.error(`Save failed: ${formatTaskDisplayMessage(err.message)}`);
    }
  }, [saveFileName, getSerializedXml]);

  return { showTreeList, setShowTreeList, showSaveDialog, setShowSaveDialog,
    saveFileName, setSaveFileName, saveConflict, setSaveConflict,
    handleServerFileSelect, handleSaveAs };
}
