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

import { useEffect, useCallback } from 'react';
import { PIXEL_TOOL_IDS } from './constants';
import { mapPointToPgmPixel } from './pgmCoordinates';
import { paintPgmPixelSegment, paintPgmPixels } from './pgmPixels';
export function useMapPixelPainting({
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
}) {
  useEffect(() => {
    lastPaintPixelRef.current = null;
    areaPaintRef.current = null;
    areaEraseRef.current = null;
  }, [areaEraseRef, areaPaintRef, brushSize, lastPaintPixelRef, open, selectedPath, tool]);
  const editAtMapPoint = useCallback((x, y, phase = "paint") => {
    if (phase === "end") {
      lastPaintPixelRef.current = null;
      return;
    }
    if (!open || !image || busy || !PIXEL_TOOL_IDS.has(tool)) return;
    const pixel = mapPointToPgmPixel(image, x, y);
    if (!pixel) {
      lastPaintPixelRef.current = null;
      return;
    }
    const {
      pixelX,
      pixelY
    } = pixel;
    const currentPixels = pixelsRef.current;
    if (!currentPixels) return;
    const previousPixel = phase === "move" ? lastPaintPixelRef.current : null;
    const nextPixels = previousPixel ? paintPgmPixelSegment(currentPixels, image.width, image.height, previousPixel, pixel, tool, brushSize) : paintPgmPixels(currentPixels, image.width, image.height, pixelX, pixelY, tool, brushSize);
    let editedPixels = 0;
    for (let index = 0; index < currentPixels.length; index += 1) {
      if (currentPixels[index] !== nextPixels[index]) editedPixels += 1;
    }
    if (editedPixels === 0) {
      lastPaintPixelRef.current = pixel;
      onMessage("No pixels changed");
      return;
    }
    lastPaintPixelRef.current = pixel;
    pixelsRef.current = nextPixels;
    setUndoStack(stack => [...stack, {
      type: "pixels",
      pixels: currentPixels
    }]);
    setRedoStack([]);
    setPixels(nextPixels);
    setDirty(true);
    const action = tool === "erase_black" ? "Removed" : tool === "draw_unknown" ? "Marked" : "Added";
    onMessage(`${action} ${editedPixels} pixels locally`);
  }, [brushSize, busy, image, lastPaintPixelRef, onMessage, open, pixelsRef, setDirty, setPixels, setRedoStack, setUndoStack, tool]);
  return {
    editAtMapPoint
  };
}
