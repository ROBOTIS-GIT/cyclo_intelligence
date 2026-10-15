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

import { useCallback, useState, useRef, useEffect, useMemo } from 'react';
import toast from 'react-hot-toast';
import { serializeFromGraph } from '../../../utils/btXmlSerializer';
import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';

// I/O uses explicit parent callbacks. Request identity prevents old file
// responses from closing dialogs or selecting files in the next waypoint.
export default function useMissionBtFiles({ graph, filePath, fileOptions,
  onLoadXml, onSaveXml, onFilePathChange, onSaveXmlAs, onSetDefaultXml, disarmClearTree }) {
  const { nodesRef, edgesRef, nodeDataMapRef } = graph;
  const [fileAction, setFileAction] = useState('');
  const [showLoadDialog, setShowLoadDialog] = useState(false);
  const [pendingLoadPath, setPendingLoadPath] = useState(filePath || '');
  const [showSaveAsDialog, setShowSaveAsDialog] = useState(false);
  const [saveAsName, setSaveAsName] = useState('');
  const fileActionRequestRef = useRef(0);
  useEffect(() => {
    fileActionRequestRef.current += 1;
    disarmClearTree();
    setFileAction("");
    setShowLoadDialog(false);
    setPendingLoadPath(filePath || "");
    setShowSaveAsDialog(false);
    setSaveAsName("");
  }, [disarmClearTree, filePath]);

  const availableFileOptions = useMemo(() => (
    [filePath, ...fileOptions]
      .map((path) => String(path || "").trim())
      .filter((path, index, paths) => path && paths.indexOf(path) === index)
  ), [fileOptions, filePath]);

  const serializeCurrentXml = useCallback(() => serializeFromGraph(
    nodesRef.current,
    edgesRef.current,
    nodeDataMapRef.current,
  ), [edgesRef, nodeDataMapRef, nodesRef]);

  const handleLoadXml = useCallback(async (targetPath) => {
    if (!targetPath || typeof onLoadXml !== "function") return;
    const requestId = fileActionRequestRef.current + 1;
    fileActionRequestRef.current = requestId;
    setFileAction("load");
    try {
      await onLoadXml(targetPath);
      if (fileActionRequestRef.current !== requestId) return;
      setShowLoadDialog(false);
      toast.success(`Opened: ${targetPath}`);
      if (typeof onFilePathChange === "function") onFilePathChange(targetPath);
    } catch (error) {
      if (fileActionRequestRef.current !== requestId) return;
      toast.error(`Failed to load ${targetPath}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
    } finally {
      if (fileActionRequestRef.current === requestId) setFileAction("");
    }
  }, [onFilePathChange, onLoadXml]);

  const handleSaveXml = useCallback(async () => {
    if (!filePath || typeof onSaveXml !== "function") return;
    let serialized;
    try {
      serialized = serializeCurrentXml();
    } catch (error) {
      toast.error(`Failed to serialize ${filePath}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
      return;
    }
    const requestId = fileActionRequestRef.current + 1;
    fileActionRequestRef.current = requestId;
    setFileAction("save");
    try {
      await onSaveXml(filePath, serialized);
      if (fileActionRequestRef.current !== requestId) return;
      toast.success(`Saved: ${filePath}`);
    } catch (error) {
      if (fileActionRequestRef.current !== requestId) return;
      toast.error(`Failed to save ${filePath}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
    } finally {
      if (fileActionRequestRef.current === requestId) setFileAction("");
    }
  }, [filePath, onSaveXml, serializeCurrentXml]);

  const handleSaveXmlAs = useCallback(async (event) => {
    event.preventDefault();
    const targetName = saveAsName.trim();
    if (!filePath || !targetName || typeof onSaveXmlAs !== "function") return;
    let serialized;
    try {
      serialized = serializeCurrentXml();
    } catch (error) {
      toast.error(`Failed to serialize ${filePath}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
      return;
    }
    const requestId = fileActionRequestRef.current + 1;
    fileActionRequestRef.current = requestId;
    setFileAction("save-as");
    try {
      const response = await onSaveXmlAs(filePath, targetName, serialized);
      if (fileActionRequestRef.current !== requestId) return;
      const nextPath = String(response?.path || "").trim();
      setShowSaveAsDialog(false);
      setSaveAsName("");
      toast.success(`Saved as: ${nextPath || targetName}`);
      if (nextPath && response?.selected !== true && typeof onFilePathChange === "function") {
        onFilePathChange(nextPath);
      }
    } catch (error) {
      if (fileActionRequestRef.current !== requestId) return;
      toast.error(`Failed to save as ${targetName}: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
    } finally {
      if (fileActionRequestRef.current === requestId) setFileAction("");
    }
  }, [filePath, onFilePathChange, onSaveXmlAs, saveAsName, serializeCurrentXml]);

  const handleSetDefaultXml = useCallback(async () => {
    if (!filePath || typeof onSetDefaultXml !== "function") return;
    const requestId = fileActionRequestRef.current + 1;
    fileActionRequestRef.current = requestId;
    setFileAction("set-default");
    try {
      await onSetDefaultXml(filePath);
      if (fileActionRequestRef.current !== requestId) return;
      toast.success(`Used for Run: ${filePath}`);
    } catch (error) {
      if (fileActionRequestRef.current !== requestId) return;
      toast.error(`Failed to use ${filePath} for Run: ${formatTaskDisplayMessage(error instanceof Error ? error.message : error, "Waypoint Task")}`);
    } finally {
      if (fileActionRequestRef.current === requestId) setFileAction("");
    }
  }, [filePath, onSetDefaultXml]);

  return { fileAction, showLoadDialog, setShowLoadDialog, pendingLoadPath,
    setPendingLoadPath, showSaveAsDialog, setShowSaveAsDialog, saveAsName, setSaveAsName,
    availableFileOptions, handleLoadXml, handleSaveXml, handleSaveXmlAs, handleSetDefaultXml };
}
