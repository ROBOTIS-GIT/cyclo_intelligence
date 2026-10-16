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

import { act, renderHook, waitFor } from '@testing-library/react';
import { useMapEditor } from './MapEditor';
import { getMapAnnotations, getPgmFiles, getPgmImage, saveMapAnnotations, savePgmImage } from '../../utils/navigationApi';

jest.mock('../../utils/navigationApi', () => ({
  getMapAnnotations: jest.fn(), getPgmFiles: jest.fn(), getPgmImage: jest.fn(),
  saveMapAnnotations: jest.fn(), savePgmImage: jest.fn(),
}));

const image = (name = 'Home', width = 4) => ({
  path: `/maps/${name}.pgm`, width, height: 4, maxval: 255, resolution: 1,
  origin: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
  pixels_base64: btoa(String.fromCharCode(...new Array(width * 4).fill(254))),
});
const message = jest.fn();

beforeEach(() => {
  jest.clearAllMocks();
  getPgmFiles.mockResolvedValue({ files: [{ path: '/maps/Home.pgm' }, { path: '/maps/Other.pgm' }] });
  getPgmImage.mockImplementation(p => Promise.resolve(image(p.includes('Other') ? 'Other' : 'Home')));
  getMapAnnotations.mockResolvedValue({ annotations: [] });
  savePgmImage.mockResolvedValue({ path: '/maps/Home.pgm' });
  saveMapAnnotations.mockImplementation((p, annotations) => Promise.resolve({ annotations }));
});

async function openEditor() {
  const view = renderHook(() => useMapEditor({ open: true, mapName: 'Home', onMessage: message }));
  await waitFor(() => expect(view.result.current.map?.data).toHaveLength(16));
  await waitFor(() => expect(view.result.current.busy).toBe(false));
  return view;
}

test('map pixels and area edits share one chronological undo/redo history', async () => {
  const { result } = await openEditor();
  act(() => result.current.setTool('draw_black'));
  act(() => result.current.editAtMapPoint(0.5, 0.5));
  expect(result.current.map.data[0]).toBe(100);
  act(() => result.current.setTool('label_marker'));
  act(() => result.current.placeAnnotationAtMapArea(1.5, 1.5, 2.5, 2.5));
  expect(result.current.annotations).toHaveLength(1);
  act(() => result.current.undo());
  expect(result.current.annotations).toHaveLength(0);
  expect(result.current.map.data[0]).toBe(100);
  act(() => result.current.undo());
  expect(result.current.map.data[0]).toBe(0);
  act(() => result.current.redo());
  expect(result.current.map.data[0]).toBe(100);
  act(() => result.current.redo());
  expect(result.current.annotations).toHaveLength(1);
});

test('saving commits the pixel document and areas and clears history', async () => {
  const { result } = await openEditor();
  act(() => result.current.setTool('draw_black'));
  act(() => result.current.editAtMapPoint(0.5, 0.5));
  act(() => result.current.setTool('label_marker'));
  act(() => result.current.setAnnotationLabel('Dining'));
  act(() => result.current.placeAnnotationAtMapArea(1.5, 1.5, 2.5, 2.5));
  act(() => result.current.save());
  await waitFor(() => expect(result.current.dirty).toBe(false));
  expect(savePgmImage).toHaveBeenCalledWith('/maps/Home.pgm', 4, 4, 255, expect.any(String));
  expect(saveMapAnnotations).toHaveBeenCalledWith('/maps/Home.pgm', [expect.objectContaining({ label: 'Dining' })]);
  expect(result.current.canUndo).toBe(false);
  expect(result.current.canRedo).toBe(false);
});

test('discarded map load responses cannot overwrite a newly selected document', async () => {
  let resolveHome;
  getPgmImage.mockImplementation(p => p.includes('Home') ? new Promise(resolve => { resolveHome = resolve; }) : Promise.resolve(image('Other', 5)));
  const { result } = renderHook(() => useMapEditor({ open: true, mapName: 'Home', onMessage: message }));
  await waitFor(() => expect(getPgmImage).toHaveBeenCalledWith('/maps/Home.pgm'));
  act(() => result.current.setSelectedPath('/maps/Other.pgm'));
  await waitFor(() => expect(result.current.map?.data).toHaveLength(20));
  await act(async () => resolveHome(image('Home')));
  expect(result.current.image.path).toBe('/maps/Other.pgm');
  expect(result.current.map.data).toHaveLength(20);
});

test('area brush extensions remain one undoable stroke and save failures preserve unsaved changes', async () => {
  const { result } = await openEditor();
  act(() => result.current.setTool('label_marker'));
  act(() => result.current.placeAnnotationAtMapArea(0.5, 0.5, 0.5, 0.5));
  const original = result.current.annotations[0];
  act(() => result.current.setTool('extend_area'));
  act(() => result.current.editAreaAtMapPoint(1.5, 0.5, 'start'));
  act(() => result.current.editAreaAtMapPoint(2.5, 0.5, 'move'));
  act(() => result.current.editAreaAtMapPoint(0, 0, 'end'));
  expect(result.current.annotations[0].region.cell_count).toBeGreaterThan(original.region.cell_count);
  act(() => result.current.undo());
  expect(result.current.annotations[0]).toEqual(original);
  saveMapAnnotations.mockRejectedValue(new Error('offline'));
  act(() => result.current.save());
  await waitFor(() => expect(message).toHaveBeenCalledWith('offline'));
  expect(result.current.annotationsDirty).toBe(true);
  expect(result.current.busy).toBe(false);
});
