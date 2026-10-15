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

import { useEffect } from 'react';
import { getPgmFiles, getPgmImage, getMapAnnotations } from '../../../utils/navigationApi';
import { preferredPgmFile, mapNameFromPgmPath } from './pgmCoordinates';
import { decodePgmPixels } from './pgmPixels';
import { normalizeAnnotation } from './annotationModel';
export function useMapDocumentLoading({
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
}) {
  useEffect(() => {
    if (selectedAnnotationId && !annotations.some(annotation => annotation.id === selectedAnnotationId)) {
      setSelectedAnnotationId("");
    }
  }, [annotations, selectedAnnotationId, setSelectedAnnotationId]);
  useEffect(() => {
    if (!open) {
      if (!autoSelect) setSelectedPath("");
      return;
    }
    let cancelled = false;
    setBusy(true);
    getPgmFiles().then(response => {
      if (cancelled) return;
      setFiles(response.files);
      const preferred = preferredPgmFile(response.files, mapName);
      setSelectedPath(current => {
        var _a;
        if (!autoSelect) {
          return response.files.some(file => file.path === current) ? current : "";
        }
        if (preferred && (reloadToken || mapNameFromPgmPath(current) !== mapName)) {
          return preferred.path;
        }
        return current || ((preferred === null) || (preferred === void 0) ? void 0 : preferred.path) || (((_a = response.files[0]) === null) || (_a === void 0) ? void 0 : _a.path) || "";
      });
      if (!response.files.length) onMessage("No PGM files found");
    }).catch(error => {
      if (!cancelled) onMessage(error instanceof Error ? error.message : "Failed to list PGM files");
    }).finally(() => {
      if (!cancelled) setBusy(false);
    });
    return () => {
      cancelled = true;
    };
  }, [autoSelect, mapName, onMessage, open, reloadToken, setBusy, setFiles, setSelectedPath]);
  useEffect(() => {
    if (!open || !selectedPath) {
      setImage(null);
      setPixels(null);
      pixelsRef.current = null;
      lastPaintPixelRef.current = null;
      areaPaintRef.current = null;
      areaEraseRef.current = null;
      setUndoStack([]);
      setRedoStack([]);
      setDirty(false);
      setAnnotationsDirty(false);
      return;
    }
    let cancelled = false;
    setBusy(true);
    getPgmImage(selectedPath).then(response => {
      if (cancelled) return;
      setImage(response);
      const decodedPixels = decodePgmPixels(response);
      pixelsRef.current = decodedPixels;
      lastPaintPixelRef.current = null;
      areaPaintRef.current = null;
      areaEraseRef.current = null;
      setPixels(decodedPixels);
      setUndoStack([]);
      setRedoStack([]);
      setDirty(false);
      setAnnotationsDirty(false);
      onMessage(`Loaded ${response.path}`);
    }).catch(error => {
      if (!cancelled) onMessage(error instanceof Error ? error.message : "Failed to load PGM file");
    }).finally(() => {
      if (!cancelled) setBusy(false);
    });
    return () => {
      cancelled = true;
    };
  }, [areaEraseRef, areaPaintRef, lastPaintPixelRef, onMessage, open, pixelsRef, selectedPath, setAnnotationsDirty, setBusy, setDirty, setImage, setPixels, setRedoStack, setUndoStack]);
  useEffect(() => {
    annotationsRef.current = annotations;
  }, [annotations, annotationsRef]);
  useEffect(() => {
    if (!open || !selectedPath) {
      annotationsRef.current = [];
      setAnnotations([]);
      setAnnotationsDirty(false);
      return;
    }
    let cancelled = false;
    getMapAnnotations(selectedPath).then(response => {
      if (cancelled) return;
      const next = Array.isArray((response === null) || (response === void 0) ? void 0 : response.annotations) ? response.annotations.map(normalizeAnnotation) : [];
      annotationsRef.current = next;
      setAnnotations(next);
      setAnnotationsDirty(false);
    }).catch(error => {
      if (cancelled) return;
      annotationsRef.current = [];
      setAnnotations([]);
      setAnnotationsDirty(false);
      onMessage(error instanceof Error ? error.message : "Failed to load map areas");
    });
    return () => {
      cancelled = true;
    };
  }, [annotationsRef, onMessage, open, selectedPath, setAnnotations, setAnnotationsDirty]);
}
