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
// Author: Howon Kim, Seongwoo Kim

"use client";

import { useRef, useState, useMemo, useCallback } from 'react';
import { DEFAULT_BRUSH_SIZE_CELLS } from './constants';
import { useMapDocumentLoading } from './useMapDocumentLoading';
import { pgmPixelsToGrid } from './pgmPixels';
import { useMapPixelPainting } from './useMapPixelPainting';
import { useMapAreaPainting } from './useMapAreaPainting';
import { useMapAnnotationCommands } from './useMapAnnotationCommands';
import { useMapEditHistory } from './useMapEditHistory';
import { useMapDocumentSave } from './useMapDocumentSave';
import { normalizeBrushSize } from './pgmCoordinates';
// One document and one chronological history own both pixels and area edits.
// Drawing and persistence hooks receive that state instead of creating copies.
export function useMapEditor({
  open,
  mapName,
  onMessage,
  reloadToken = 0,
  autoSelect = true
}) {
  const pixelsRef = useRef(null);
  const annotationsRef = useRef([]);
  const lastPaintPixelRef = useRef(null);
  const areaPaintRef = useRef(null);
  const areaEraseRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [selectedPath, setSelectedPath] = useState("");
  const [image, setImage] = useState(null);
  const [pixels, setPixels] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [annotationLabel, setAnnotationLabel] = useState("");
  const [undoStack, setUndoStack] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const [dirty, setDirty] = useState(false);
  const [annotationsDirty, setAnnotationsDirty] = useState(false);
  const [tool, setTool] = useState("view");
  const [brushSize, setBrushSize] = useState(DEFAULT_BRUSH_SIZE_CELLS);
  const [busy, setBusy] = useState(false);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState("");
  useMapDocumentLoading({
    selectedAnnotationId,
    annotations,
    setSelectedAnnotationId,
    open,
    autoSelect,
    setSelectedPath,
    setBusy,
    setFiles,
    mapName,
    reloadToken,
    onMessage,
    selectedPath,
    setImage,
    setPixels,
    pixelsRef,
    lastPaintPixelRef,
    areaPaintRef,
    areaEraseRef,
    setUndoStack,
    setRedoStack,
    setDirty,
    setAnnotationsDirty,
    annotationsRef,
    setAnnotations
  });
  const map = useMemo(() => {
    if (!image || !pixels) return null;
    return pgmPixelsToGrid(image, pixels);
  }, [image, pixels]);
  const {
    editAtMapPoint
  } = useMapPixelPainting({
    lastPaintPixelRef,
    areaPaintRef,
    areaEraseRef,
    brushSize,
    open,
    selectedPath,
    tool,
    image,
    busy,
    pixelsRef,
    onMessage,
    setUndoStack,
    setRedoStack,
    setPixels,
    setDirty
  });
  const applyAnnotationsSnapshot = useCallback((nextAnnotations, successMessage) => {
    // Callers pass already-normalized annotations (loads/saves normalize at the
    // boundary). Skipping re-normalization here keeps unchanged entries
    // reference-identical — which lets the viewer reuse their rasterized
    // textures — and avoids re-sorting every region's cells per brush move.
    annotationsRef.current = nextAnnotations;
    setAnnotations(nextAnnotations);
    setAnnotationsDirty(true);
    onMessage(successMessage);
  }, [onMessage]);
  const persistAnnotations = useCallback((nextAnnotations, successMessage) => {
    if (!selectedPath || busy) return;
    const currentAnnotations = annotationsRef.current;
    setUndoStack(stack => [...stack, {
      type: "annotations",
      annotations: currentAnnotations
    }]);
    setRedoStack([]);
    applyAnnotationsSnapshot(nextAnnotations, successMessage);
  }, [applyAnnotationsSnapshot, busy, selectedPath]);
  const recordAnnotationUndo = useCallback(session => {
    if (!session || session.undoPushed) return;
    const currentAnnotations = annotationsRef.current;
    setUndoStack(stack => [...stack, {
      type: "annotations",
      annotations: currentAnnotations
    }]);
    setRedoStack([]);
    session.undoPushed = true;
  }, []);
  const {
    editAreaAtMapPoint
  } = useMapAreaPainting({
    pixelsRef,
    open,
    image,
    busy,
    annotationsRef,
    selectedAnnotationId,
    onMessage,
    areaPaintRef,
    brushSize,
    recordAnnotationUndo,
    applyAnnotationsSnapshot,
    tool,
    areaEraseRef
  });
  const {
    placeAnnotationAtMapArea,
    deleteAnnotationById,
    renameAnnotation,
    clearAnnotations
  } = useMapAnnotationCommands({
    open,
    image,
    busy,
    tool,
    onMessage,
    pixelsRef,
    annotationsRef,
    annotationLabel,
    persistAnnotations,
    setSelectedAnnotationId
  });
  const {
    undo,
    redo
  } = useMapEditHistory({
    pixelsRef,
    setUndoStack,
    setRedoStack,
    setPixels,
    setDirty,
    onMessage,
    annotationsRef,
    applyAnnotationsSnapshot
  });
  const {
    save
  } = useMapDocumentSave({
    image,
    pixels,
    busy,
    dirty,
    annotationsDirty,
    setBusy,
    setDirty,
    annotationsRef,
    setAnnotations,
    setAnnotationsDirty,
    setUndoStack,
    setRedoStack,
    onMessage
  });
  const hasUnsavedChanges = dirty || annotationsDirty;
  return {
    files,
    selectedPath,
    setSelectedPath,
    image,
    map,
    annotations,
    annotationLabel,
    setAnnotationLabel,
    tool,
    setTool,
    brushSize,
    setBrushSize: value => setBrushSize(normalizeBrushSize(value)),
    busy,
    dirty: hasUnsavedChanges,
    pixelsDirty: dirty,
    annotationsDirty,
    canUndo: undoStack.length > 0,
    canRedo: redoStack.length > 0,
    undo,
    redo,
    save,
    editAtMapPoint,
    editAreaAtMapPoint,
    placeAnnotationAtMapArea,
    clearAnnotations,
    selectedAnnotationId,
    setSelectedAnnotationId,
    deleteAnnotationById,
    renameAnnotation
  };
}
