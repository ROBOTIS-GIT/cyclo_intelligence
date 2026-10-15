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

import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import {
  MapViewer,
  mapAreaPreviewCellIndices,
  mapRenderIntervalMs,
  updateGlobalCostmapTexture,
} from './MapViewer';

const mockControls = [];
const mockPointPositions = [];
const mockPoints = [];
const mockRenderers = [];
const mockDataTextures = [];
const mockCanvasFillStyles = [];
const mockMeshes = [];
const mockSprites = [];
const mockCanvasContext = {
  arc: jest.fn(),
  beginPath: jest.fn(),
  clearRect: jest.fn(),
  closePath: jest.fn(),
  fill: jest.fn(),
  fillRect: jest.fn(),
  fillText: jest.fn(),
  lineTo: jest.fn(),
  measureText: (text) => ({ width: String(text).length * 28 }),
  moveTo: jest.fn(),
  roundRect: jest.fn(),
  stroke: jest.fn(),
};
let canvasContextSpy;

jest.mock('three', () => {
  const actual = jest.requireActual('three');

  class WebGLRenderer {
    constructor() {
      this.domElement = global.document.createElement('canvas');
      mockRenderers.push(this);
    }

    setPixelRatio(value) { this.pixelRatio = value; }

    setClearColor() {}

    setSize() {}

    render() {}

    dispose() {}
  }

  class Points extends actual.Points {
    constructor(geometry, material) {
      super(geometry, material);
      mockPointPositions.push(Array.from(geometry.attributes.position.array));
      mockPoints.push(this);
    }
  }

  class Mesh extends actual.Mesh {
    constructor(...args) {
      super(...args);
      mockMeshes.push(this);
    }
  }

  class DataTexture extends actual.DataTexture {
    constructor(...args) {
      super(...args);
      mockDataTextures.push(this);
    }
  }

  class Sprite extends actual.Sprite {
    constructor(...args) {
      super(...args);
      mockSprites.push(this);
    }
  }

  return { ...actual, DataTexture, Mesh, Points, Sprite, WebGLRenderer };
});

jest.mock('three/examples/jsm/controls/OrbitControls.js', () => ({
  OrbitControls: class OrbitControls {
    constructor(camera) {
      this.object = camera;
      this.listeners = {};
      mockControls.push(this);
      this.target = {
        x: 0,
        y: 0,
        z: 0,
        copy(value) {
          this.x = value.x;
          this.y = value.y;
          this.z = value.z;
          return this;
        },
      };
    }

    addEventListener(type, listener) { this.listeners[type] = listener; }

    removeEventListener(type) { delete this.listeners[type]; }

    dispatchEvent(event) { this.listeners[event.type]?.(event); }

    update() {}

    dispose() {}
  },
}));

const waypointBtLayer = {
  spot: {
    id: 'waypoint-a',
    label: 'Waypoint A',
    pose: { x: 1, y: 2, yaw: 0 },
    linked_bt_tree: 'locals/waypoint-a.xml',
  },
  editor: <div>Waypoint BT editor</div>,
};

beforeAll(() => {
  canvasContextSpy = jest
    .spyOn(HTMLCanvasElement.prototype, 'getContext')
    .mockReturnValue(mockCanvasContext);
});

afterAll(() => {
  canvasContextSpy.mockRestore();
});

beforeEach(() => {
  canvasContextSpy.mockReturnValue(mockCanvasContext);
  mockCanvasContext.fill.mockImplementation(function captureFillStyle() {
    mockCanvasFillStyles.push(this.fillStyle);
  });
  mockControls.length = 0;
  mockPointPositions.length = 0;
  mockPoints.length = 0;
  mockRenderers.length = 0;
  mockDataTextures.length = 0;
  mockCanvasFillStyles.length = 0;
  mockMeshes.length = 0;
  mockSprites.length = 0;
});

test('closes the waypoint BT split when its map context is clicked', () => {
  const onBtLayerClose = jest.fn();

  render(
    <MapViewer
      btLayer={waypointBtLayer}
      onBtLayerClose={onBtLayerClose}
      showMap={false}
    />,
  );

  const mapContext = screen.getByRole('button', {
    name: 'Back to Map from waypoint context',
  });

  fireEvent.click(mapContext);

  expect(onBtLayerClose).toHaveBeenCalledTimes(1);
});

test('uses adaptive map render intervals for active, idle and hidden states', () => {
  expect(mapRenderIntervalMs({ active: true })).toBe(16);
  expect(mapRenderIntervalMs({ active: false })).toBe(100);
  expect(mapRenderIntervalMs({ hidden: true, active: true })).toBe(500);
});

test('anchors waypoint labels above their markers in screen space', async () => {
  render(
    <MapViewer
      spots={[{
        id: 'waypoint-a',
        label: 'Waypoint A',
        pose: { x: 1, y: 2, yaw: 0 },
      }]}
      selectedSpotId="waypoint-a"
      showMap={false}
    />,
  );

  const label = await waitFor(() => {
    const sprite = mockSprites.find((item) => item.userData.spotId === 'waypoint-a');
    expect(sprite).toBeDefined();
    return sprite;
  });

  expect(label.position.x).toBe(0);
  expect(label.position.y).toBe(0);
  expect(label.position.z).toBe(0.04);
  expect(label.center.x).toBe(0.5);
  expect(label.material.depthWrite).toBe(false);
  expect(mockCanvasFillStyles).toContain('rgba(244,229,220,0.5)');
  // Sprite.center is evaluated in camera/screen space. This effective offset
  // remains 8.9 — the existing marker-to-label distance — at every map roll.
  expect((0.5 - label.center.y) * label.scale.y).toBeCloseTo(8.9);
});

test('uses waypoint labels as selection targets without hiding rotation handles', async () => {
  render(
    <MapViewer
      spots={[{
        id: 'waypoint-a',
        label: 'A very long waypoint name',
        pose: { x: 1, y: 2, yaw: 0 },
      }]}
      showMap={false}
    />,
  );

  const { label, rotateHitArea } = await waitFor(() => {
    const nextLabel = mockSprites.find((item) => item.userData.spotId === 'waypoint-a');
    const nextHitArea = mockMeshes.find((item) => item.userData.waypointRotateHitArea);
    expect(nextLabel).toBeDefined();
    expect(nextHitArea).toBeDefined();
    return { label: nextLabel, rotateHitArea: nextHitArea };
  });

  const THREE = jest.requireActual('three');
  expect(label.raycast).toBe(THREE.Sprite.prototype.raycast);
  expect(label.userData).toMatchObject({
    spotId: 'waypoint-a',
    dragAction: 'move',
  });
  expect(mockCanvasFillStyles).toContain('rgba(243,241,234,0.5)');

  rotateHitArea.geometry.computeBoundingSphere();
  expect(rotateHitArea.geometry.boundingSphere.radius).toBeCloseTo(1.4);
  expect(rotateHitArea.position.x).toBeCloseTo(4.5);
  expect(rotateHitArea.material.opacity).toBe(0);
  expect(rotateHitArea.material.colorWrite).toBe(false);
  expect(rotateHitArea.position.z).toBeGreaterThan(label.position.z);
  expect(rotateHitArea.userData).toMatchObject({
    spotId: 'waypoint-a',
    dragAction: 'rotate',
  });
});

test('covers the complete waypoint body with an enlarged pointer target', async () => {
  render(
    <MapViewer
      spots={[{
        id: 'waypoint-a',
        label: 'Waypoint A',
        pose: { x: 1, y: 2, yaw: 0 },
      }]}
      showMap={false}
    />,
  );

  const bodyHitArea = await waitFor(() => {
    const target = mockMeshes.find((item) => item.userData.waypointBodyHitArea);
    expect(target).toBeDefined();
    return target;
  });

  bodyHitArea.geometry.computeBoundingSphere();
  expect(bodyHitArea.geometry.boundingSphere.radius).toBeCloseTo(3.72);
  expect(bodyHitArea.material.opacity).toBe(0);
  expect(bodyHitArea.material.colorWrite).toBe(false);
  expect(bodyHitArea.userData).toMatchObject({
    spotId: 'waypoint-a',
    dragAction: 'move',
  });
});

test('keeps the route order badge clear at the waypoint screen-space bottom-left', async () => {
  const map = {
    info: {
      width: 20,
      height: 20,
      resolution: 0.05,
      origin: {
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      },
    },
    data: Array(400).fill(0),
  };

  render(
    <MapViewer
      map={map}
      spots={[{
        id: 'waypoint-a',
        label: 'Waypoint A',
        pose: { x: 1, y: 2, yaw: 0 },
      }]}
      missionRouteOrder={[{ id: 'waypoint-a', order: 1 }]}
      showMap={false}
    />,
  );

  const badge = await waitFor(() => {
    const sprite = mockSprites.find((item) => item.userData.missionRouteBadge);
    expect(sprite).toBeDefined();
    return sprite;
  });

  const screenOffsetX = (0.5 - badge.center.x) * badge.scale.x;
  const screenOffsetY = (0.5 - badge.center.y) * badge.scale.y;
  const expectedClearRadius = (4.4 * 0.05) + (0.34 / 2) + 0.04;

  expect(badge.position.x).toBe(1);
  expect(badge.position.y).toBe(2);
  expect(screenOffsetX).toBeLessThan(0);
  expect(screenOffsetY).toBeLessThan(0);
  expect(Math.hypot(screenOffsetX, screenOffsetY)).toBeCloseTo(expectedClearRadius);
});

test('previews only the free, unclaimed cells that an Area drag will save', () => {
  const grid = {
    info: {
      width: 4,
      height: 2,
      resolution: 1,
      origin: {
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      },
    },
    data: [0, 100, -1, 0, 0, 0, 0, 0],
  };

  expect(mapAreaPreviewCellIndices(
    grid,
    { startX: 0.05, startY: 0.05, endX: 3.95, endY: 1.95 },
    new Set([4]),
  )).toEqual([0, 3, 5, 6, 7]);
});

test('keeps the left waypoint context passive when no close action is provided', () => {
  render(<MapViewer btLayer={waypointBtLayer} showMap={false} />);

  expect(screen.queryByRole('button', {
    name: 'Back to Map from waypoint context',
  })).not.toBeInTheDocument();
  expect(screen.getByRole('region', { name: 'Waypoint Task focus canvas' }))
    .toHaveTextContent('Waypoint BT editor');
});

test('uses task terminology for an empty waypoint focus canvas', () => {
  render(
    <MapViewer
      btLayer={{
        spot: { ...waypointBtLayer.spot, linked_bt_tree: '' },
        editor: null,
      }}
      showMap={false}
    />,
  );

  const focusCanvas = screen.getByRole('region', { name: 'Waypoint Task focus canvas' });
  expect(focusCanvas).toHaveTextContent('New Task');
  expect(focusCanvas).not.toHaveTextContent('New BT');
});

test('reprojects an offset laser frame when its synchronized scan pose improves', async () => {
  const scan = {
    header: { frame_id: 'base_scan', stamp: { sec: 10, nanosec: 0 } },
    ranges: [1],
    range_min: 0.02,
    range_max: 20,
    angle_min: 0,
    angle_increment: 0,
  };
  const tf = {
    transforms: [
      {
        header: { frame_id: 'map' },
        child_frame_id: 'base_link',
        transform: {
          translation: { x: 100, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
        },
      },
      {
        header: { frame_id: 'base_link' },
        child_frame_id: 'base_scan',
        transform: {
          translation: { x: 0.2, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
        },
      },
    ],
  };
  const { rerender } = render(
    <MapViewer
      scan={scan}
      scanPose={{ position: { x: 1, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } }}
      tf={tf}
      showMap={false}
      showScan
    />,
  );

  await waitFor(() => expect(mockPointPositions.at(-1)?.[0]).toBeCloseTo(2.2));

  rerender(
    <MapViewer
      scan={scan}
      scanPose={{ position: { x: 3, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } }}
      tf={tf}
      showMap={false}
      showScan
    />,
  );

  await waitFor(() => expect(mockPointPositions.at(-1)?.[0]).toBeCloseTo(4.2));
});

test('decimates lidar display rays while preserving both field-of-view edges', async () => {
  render(
    <MapViewer
      scan={{
        header: { frame_id: 'base_link' },
        ranges: [1, 1, 1, 1, 1, 1],
        range_min: 0.02,
        range_max: 20,
        angle_min: 0,
        angle_increment: Math.PI / 4,
      }}
      pose={{
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      }}
      showMap={false}
      showScan
    />,
  );

  // Six source rays become indices 0, 2, 4 and the final edge ray 5.
  await waitFor(() => expect(mockPointPositions.at(-1)).toHaveLength(12));
  const positions = mockPointPositions.at(-1);
  expect(positions.slice(0, 2)).toEqual([1, 0]);
  expect(positions[9]).toBeCloseTo(Math.cos(5 * Math.PI / 4));
  expect(positions[10]).toBeCloseTo(Math.sin(5 * Math.PI / 4));
});

test('updates only the dirty rows of an existing global costmap texture', () => {
  const THREE = jest.requireActual('three');
  const texture = new THREE.DataTexture(
    new Uint8Array(4 * 5 * 4),
    4,
    5,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  const grid = {
    info: {
      width: 4,
      height: 5,
      resolution: 0.05,
      origin: {
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      },
    },
    data: [
      0, 0, 0, 0,
      0, 50, 100, 0,
      0, 25, 75, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ],
  };

  expect(updateGlobalCostmapTexture(
    texture,
    grid,
    { x: 1, y: 1, width: 2, height: 2 },
  )).toBe(true);

  expect(texture.updateRanges).toEqual([
    { start: 52, count: 8 },
    { start: 36, count: 8 },
  ]);
  // Grid (2,1)=100 maps to the vertically flipped texture: gray 70, alpha 110.
  expect(Array.from(texture.image.data.slice(52, 56))).toEqual([70, 70, 70, 110]);
  // Pixels outside the dirty rectangle remain untouched.
  expect(Array.from(texture.image.data.slice(0, 4))).toEqual([0, 0, 0, 0]);
});

test('uses one full texture upload when a costmap delta covers a large area', () => {
  const THREE = jest.requireActual('three');
  const texture = new THREE.DataTexture(
    new Uint8Array(4 * 4 * 4),
    4,
    4,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  texture.userData.globalCostmapFullUploadPending = false;
  const grid = {
    info: {
      width: 4,
      height: 4,
      resolution: 0.05,
      origin: {
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      },
    },
    data: Array(16).fill(100),
  };

  expect(updateGlobalCostmapTexture(
    texture,
    grid,
    { x: 0, y: 0, width: 4, height: 4 },
  )).toBe(true);

  // An empty update-range list makes Three.js issue one full texSubImage2D
  // instead of one call for every row in a broad rectangle.
  expect(texture.updateRanges).toEqual([]);
  expect(texture.userData.globalCostmapFullUploadPending).toBe(true);
  expect(Array.from(texture.image.data.slice(0, 4))).toEqual([70, 70, 70, 110]);
});

test('reuses the global costmap texture for deltas and rebuilds on a full resync', async () => {
  const info = {
    width: 2,
    height: 2,
    resolution: 0.05,
    origin: {
      position: { x: 0, y: 0, z: 0 },
      orientation: { x: 0, y: 0, z: 0, w: 1 },
    },
  };
  const { rerender } = render(
    <MapViewer
      globalCostmap={{ info, data: [0, 0, 0, 0] }}
      showGlobalCostmap
      showMap={false}
    />,
  );

  await waitFor(() => expect(mockDataTextures).toHaveLength(1));
  const firstTexture = mockDataTextures[0];

  rerender(
    <MapViewer
      globalCostmap={{
        info,
        data: [0, 100, 0, 0],
        updateRegion: { x: 1, y: 0, width: 1, height: 1 },
      }}
      showGlobalCostmap
      showMap={false}
    />,
  );

  await waitFor(() => expect(firstTexture.updateRanges).toHaveLength(1));
  expect(mockDataTextures).toHaveLength(1);

  rerender(
    <MapViewer
      globalCostmap={{ info, data: [0, 100, 0, 0] }}
      showGlobalCostmap
      showMap={false}
    />,
  );

  await waitFor(() => expect(mockDataTextures).toHaveLength(2));
});

 test('keeps BT focus stable across status polls and resumes robot following when closed', () => {
  jest.useFakeTimers();
  const map = {
    info: { width: 20, height: 20, resolution: 0.05,
      origin: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } } },
    data: Array(400).fill(0),
  };
  const props = { map, showMap: false, missionFollowRobot: true,
    pose: { position: { x: 10, y: 10, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } } };
  const { rerender, unmount } = render(<MapViewer {...props} btLayer={waypointBtLayer} />);
  try {
    const controls = mockControls.at(-1);
    const focused = { x: controls.target.x, y: controls.target.y };
    act(() => jest.advanceTimersByTime(1000));
    expect(controls.target).toMatchObject(focused);
    // A manual pan must not be reset by a fresh server snapshot.
    controls.target.x += 1;
    const panned = controls.target.x;
    rerender(<MapViewer {...props} map={{ ...map }} btLayer={{
      ...waypointBtLayer, spot: { ...waypointBtLayer.spot, pose: { ...waypointBtLayer.spot.pose } },
    }} />);
    act(() => jest.advanceTimersByTime(1000));
    expect(controls.target.x).toBe(panned);
    rerender(<MapViewer {...props} btLayer={null} />);
    const beforeFollow = controls.target.x;
    act(() => jest.advanceTimersByTime(1000));
    expect(controls.target.x).toBeGreaterThan(beforeFollow);
    rerender(<MapViewer {...props} btLayer={waypointBtLayer} />);
    expect(controls.target).toMatchObject(focused);
  } finally {
    unmount();
    jest.useRealTimers();
  }
});


test('reports selected waypoint anchors after pose, camera, and viewport changes', () => {
  let frame;
  let now = 0;
  let width = 800;
  const raf = jest.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
    frame = callback;
    return 1;
  });
  const cancel = jest.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
  const clock = jest.spyOn(performance, 'now').mockImplementation(() => now);
  const clientWidth = jest.spyOn(HTMLElement.prototype, 'clientWidth', 'get')
    .mockImplementation(() => width);
  const clientHeight = jest.spyOn(HTMLElement.prototype, 'clientHeight', 'get')
    .mockReturnValue(600);
  const onPosition = jest.fn();
  const spot = { id: 'anchor', label: 'Anchor', pose: { x: 0, y: 0, yaw: 0 } };
  const props = {
    spots: [spot], selectedSpotId: 'anchor', showMap: false,
    onSelectedSpotScreenPositionChange: onPosition,
  };
  const view = render(<MapViewer {...props} />);
  const nextFrame = () => act(() => {
    now += 1000;
    frame();
  });
  try {
    expect(onPosition).toHaveBeenLastCalledWith({ x: 400, y: 300, width: 800, height: 600 });
    onPosition.mockClear();
    nextFrame();
    expect(onPosition).not.toHaveBeenCalled();

    mockControls[0].object.position.x = 1;
    nextFrame();
    const afterPan = onPosition.mock.calls.at(-1)[0];
    expect(afterPan.x).toBeLessThan(400);
    mockControls[0].object.zoom = 2;
    mockControls[0].object.updateProjectionMatrix();
    nextFrame();
    expect(onPosition.mock.calls.at(-1)[0].x).toBeLessThan(afterPan.x);

    view.rerender(<MapViewer {...props} spots={[{ ...spot, pose: { x: 1, y: 0, yaw: 0 } }]} />);
    expect(onPosition).toHaveBeenLastCalledWith({ x: 400, y: 300, width: 800, height: 600 });
    width = 1000;
    fireEvent(window, new Event('resize'));
    nextFrame();
    expect(onPosition).toHaveBeenLastCalledWith({ x: 500, y: 300, width: 1000, height: 600 });

    view.rerender(<MapViewer {...props} selectedSpotId="missing" />);
    expect(onPosition).toHaveBeenLastCalledWith(null);
  } finally {
    view.unmount();
    raf.mockRestore();
    cancel.mockRestore();
    clock.mockRestore();
    clientWidth.mockRestore();
    clientHeight.mockRestore();
  }
});


function mapPointer(canvas, type, options = {}) {
  const { pointerId = 1, isPrimary = true, ...mouseOptions } = options;
  const event = new MouseEvent(type, { bubbles: true, button: 0, clientX: 100, clientY: 100, ...mouseOptions });
  Object.defineProperties(event, {
    pointerId: { value: pointerId },
    isPrimary: { value: isPrimary },
  });
  fireEvent(canvas, event);
}

test('background tap waits for pointer-up and survives callback rerenders', () => {
  const first = jest.fn(), latest = jest.fn();
  const props = { interactionMode: 'view', showMap: false };
  const view = render(<MapViewer {...props} onMapTap={first} />);
  const canvas = view.container.querySelector('canvas');
  mapPointer(canvas, 'pointerdown');
  expect(first).not.toHaveBeenCalled();
  view.rerender(<MapViewer {...props} onMapTap={latest} />);
  mapPointer(canvas, 'pointerup', { clientX: 108 });
  expect(first).not.toHaveBeenCalled();
  expect(latest).toHaveBeenCalledTimes(1);
});

test('background tap rejects drags, canceled gestures, other buttons, and non-primary pointers', () => {
  const onMapTap = jest.fn();
  const view = render(<MapViewer interactionMode="view" showMap={false} onMapTap={onMapTap} />);
  const canvas = view.container.querySelector('canvas');
  canvas.hasPointerCapture = () => false;
  canvas.setPointerCapture = () => {};
  mapPointer(canvas, 'pointerdown');
  mapPointer(canvas, 'pointermove', { clientX: 120 });
  mapPointer(canvas, 'pointerup'); // Returning to the start is still a drag.
  mapPointer(canvas, 'pointerdown');
  mapPointer(canvas, 'pointerup', { clientX: 109 });
  mapPointer(canvas, 'pointerdown');
  mapPointer(canvas, 'pointercancel');
  mapPointer(canvas, 'pointerup');
  mapPointer(canvas, 'pointerdown');
  mapPointer(canvas, 'pointerleave');
  mapPointer(canvas, 'pointerup');
  mapPointer(canvas, 'pointerdown', { button: 1 });
  mapPointer(canvas, 'pointerup', { button: 1 });
  mapPointer(canvas, 'pointerdown', { button: 2 });
  mapPointer(canvas, 'pointerup', { button: 2 });
  mapPointer(canvas, 'pointerdown', { isPrimary: false });
  mapPointer(canvas, 'pointerup', { isPrimary: false });
  expect(onMapTap).not.toHaveBeenCalled();
});

test('background tap excludes waypoint selection and disabled or non-view interactions', () => {
  const onMapTap = jest.fn(), onSpotClick = jest.fn();
  const props = { interactionMode: 'view', showMap: false, onMapTap, onSpotClick,
    spots: [{ id: 'selected', pose: { x: 0, y: 0, yaw: 0 } }] };
  const view = render(<MapViewer {...props} />);
  const canvas = view.container.querySelector('canvas');
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 200, height: 200 });
  mockControls[0].object.updateMatrixWorld();
  mapPointer(canvas, 'pointerdown');
  mapPointer(canvas, 'pointerup');
  expect(onSpotClick).toHaveBeenCalledWith('selected');
  expect(onMapTap).not.toHaveBeenCalled();
  view.rerender(<MapViewer {...props} onSpotClick={undefined} />);
  mapPointer(canvas, 'pointerdown');
  mapPointer(canvas, 'pointerup');
  expect(onMapTap).not.toHaveBeenCalled();
  view.rerender(<MapViewer {...props} interactionDisabled />);
  mapPointer(canvas, 'pointerdown');
  mapPointer(canvas, 'pointerup');
  view.rerender(<MapViewer {...props} interactionMode="spot" />);
  canvas.hasPointerCapture = () => false;
  mapPointer(canvas, 'pointerdown');
  mapPointer(canvas, 'pointerup');
  expect(onMapTap).not.toHaveBeenCalled();
});

test('legacy map click still fires on pointer-down independently of background tap', () => {
  const onMapClick = jest.fn(), onMapTap = jest.fn();
  const map = { info: { width: 20, height: 20, resolution: 1,
    origin: { position: { x: -10, y: -10, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } } } };
  const view = render(<MapViewer map={map} interactionMode="view" showMap={false}
    onMapClick={onMapClick} onMapTap={onMapTap} />);
  const canvas = view.container.querySelector('canvas');
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 200, height: 200 });
  mockControls[0].object.updateMatrixWorld();
  mapPointer(canvas, 'pointerdown');
  expect(onMapClick).toHaveBeenCalledTimes(1);
  expect(onMapTap).not.toHaveBeenCalled();
  mapPointer(canvas, 'pointermove', { clientX: 120 });
  mapPointer(canvas, 'pointerup', { clientX: 120 });
  expect(onMapClick).toHaveBeenCalledTimes(1);
  expect(onMapTap).not.toHaveBeenCalled();
});


test('pulses a child mission waypoint even when no observation waypoint is active', () => {
 let frame, now=0;
 const raf=jest.spyOn(window,'requestAnimationFrame').mockImplementation(callback=>{frame=callback;return 1;});
 const cancel=jest.spyOn(window,'cancelAnimationFrame').mockImplementation(()=>{});
 const clock=jest.spyOn(performance,'now').mockImplementation(()=>now);
 const spot={id:'table',label:'Table',pose:{x:1,y:1,yaw:0}};
 const overlay={spots:[spot],pathSpots:[spot],order:[],activeWaypointId:'table'};
 const view=render(<MapViewer showMap={false} missionRouteOverlay={overlay}/>);
 try {
  const pulse=mockMeshes.find(mesh=>mesh.userData.pulse);
  expect(pulse).toBeDefined();
  const initial=pulse.material.opacity;
  act(()=>{now=1000;frame();});
  expect(pulse.material.opacity).not.toBe(initial);
  const updated=pulse.material.opacity;
  act(()=>{now=1200;frame();});
  expect(pulse.material.opacity).not.toBe(updated);
 } finally { view.unmount();raf.mockRestore();cancel.mockRestore();clock.mockRestore(); }
});


test('autonomy limits motion rendering while retaining gesture and hidden-tab cadence', () => {
  expect(mapRenderIntervalMs({ active: true, renderProfile: 'autonomy' })).toBe(33);
  expect(mapRenderIntervalMs({ active: true, interactive: true, renderProfile: 'autonomy' })).toBe(16);
  expect(mapRenderIntervalMs({ renderProfile: 'autonomy' })).toBe(100);
  expect(mapRenderIntervalMs({ hidden: true, active: true, interactive: true, renderProfile: 'autonomy' })).toBe(500);
});

test('changes autonomy pixel density without recreating the renderer or camera', () => {
  const originalDpr = Object.getOwnPropertyDescriptor(window, 'devicePixelRatio');
  Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 2 });
  const view = render(<MapViewer showMap={false} />);
  try {
    const renderer = mockRenderers[0];
    const camera = mockControls[0].object;
    expect(renderer.pixelRatio).toBe(1.5);
    view.rerender(<MapViewer showMap={false} renderProfile="autonomy" />);
    expect(renderer.pixelRatio).toBe(1);
    expect(mockRenderers).toHaveLength(1);
    expect(mockControls[0].object).toBe(camera);
    view.rerender(<MapViewer showMap={false} />);
    expect(renderer.pixelRatio).toBe(1.5);
  } finally {
    view.unmount();
    Object.defineProperty(window, 'devicePixelRatio', originalDpr);
  }
});

test('retains scan geometry and waypoint labels through pose-only updates', () => {
  const scan = {
    header: { frame_id: 'base_link' },
    ranges: [1, 2, 3], range_min: 0.02, range_max: 20,
    angle_min: 0, angle_increment: 0,
  };
  const pose = { position: { x: 0, y: 0, z: 0 }, orientation: { w: 1 } };
  const spots = [{ id: 'waypoint-a', label: 'Waypoint A', pose: { x: 1, y: 2, yaw: 0 } }];
  const props = { scan, pose, spots, showMap: false, showScan: true, renderProfile: 'autonomy' };
  const view = render(<MapViewer {...props} />);
  const points = mockPoints[0];
  const geometryDispose = jest.spyOn(points.geometry, 'dispose');
  const materialDispose = jest.spyOn(points.material, 'dispose');
  const label = mockSprites.find((item) => item.userData.spotId === 'waypoint-a');
  const labelDispose = jest.spyOn(label.material, 'dispose');
  const pointPositions = Array.from(points.geometry.attributes.position.array);

  view.rerender(<MapViewer {...props} pose={{ ...pose, position: { x: 5, y: 0, z: 0 } }} />);
  expect(mockPoints).toHaveLength(1);
  expect(points.parent).not.toBeNull();
  expect(Array.from(points.geometry.attributes.position.array)).toEqual(pointPositions);
  expect(geometryDispose).not.toHaveBeenCalled();
  expect(materialDispose).not.toHaveBeenCalled();
  expect(labelDispose).not.toHaveBeenCalled();
  expect(mockSprites.filter((item) => item.userData.spotId === 'waypoint-a')).toEqual([label]);

  view.rerender(<MapViewer {...props} scan={{ ...scan, ranges: [4, 5, 6] }} />);
  expect(mockPoints).toHaveLength(2);
  expect(mockPointPositions.at(-1)[0]).toBeCloseTo(4);
  expect(geometryDispose).toHaveBeenCalledTimes(1);
  expect(materialDispose).toHaveBeenCalledTimes(1);
  const replacement = mockPoints.at(-1);
  const replacementDispose = jest.spyOn(replacement.geometry, 'dispose');
  view.rerender(<MapViewer {...props} showScan={false} />);
  expect(replacement.parent).toBeNull();
  expect(replacementDispose).toHaveBeenCalledTimes(1);
  view.rerender(<MapViewer {...props} />);
  expect(mockPoints.at(-1).parent).not.toBeNull();
  const finalGeometryDispose = jest.spyOn(mockPoints.at(-1).geometry, 'dispose');
  view.unmount();
  expect(finalGeometryDispose).toHaveBeenCalledTimes(1);
});

test('can display and hide the footprint independently from the robot-model layer', () => {
  const footprint = { header: { frame_id: 'map' }, polygon: { points: [
    { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 },
  ] } };
  const props = { footprint, showMap: false, showRobotModel: false };
  const view = render(<MapViewer {...props} showFootprint />);
  const fill = mockMeshes.find((mesh) => mesh.material.color.getHex() === 0x38bdf8);
  expect(fill).toBeDefined();
  expect(fill.parent.parent).not.toBeNull();
  const dispose = jest.spyOn(fill.geometry, 'dispose');
  view.rerender(<MapViewer {...props} showFootprint={false} />);
  expect(fill.parent.parent).toBeNull();
  expect(dispose).toHaveBeenCalledTimes(1);
  view.rerender(<MapViewer {...props} showRobotModel />);
  expect(mockMeshes.filter((mesh) => mesh.material.color.getHex() === 0x38bdf8)).toHaveLength(2);
});


test('keeps autonomy waypoint pulses at 30fps outside gestures and release damping', () => {
  let frame;
  let now = 0;
  const raf = jest.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
    frame = callback;
    return 1;
  });
  const cancel = jest.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
  const clock = jest.spyOn(performance, 'now').mockImplementation(() => now);
  const spot = { id: 'table', label: 'Table', pose: { x: 1, y: 1, yaw: 0 } };
  const overlay = { spots: [spot], pathSpots: [spot], order: [], activeWaypointId: 'table' };
  const view = render(<MapViewer showMap={false} renderProfile="autonomy" missionRouteOverlay={overlay} />);
  const renderer = jest.spyOn(mockRenderers[0], 'render');
  const nextFrame = (time) => act(() => { now = time; frame(); });
  try {
    nextFrame(1000);
    expect(renderer).toHaveBeenCalledTimes(1);
    nextFrame(1016);
    expect(renderer).toHaveBeenCalledTimes(1);
    nextFrame(1033);
    expect(renderer).toHaveBeenCalledTimes(2);

    act(() => mockControls[0].dispatchEvent({ type: 'start' }));
    nextFrame(1049);
    expect(renderer).toHaveBeenCalledTimes(3);
    act(() => mockControls[0].dispatchEvent({ type: 'end' }));
    nextFrame(1065);
    expect(renderer).toHaveBeenCalledTimes(4);

    nextFrame(1500);
    expect(renderer).toHaveBeenCalledTimes(5);
    nextFrame(1516);
    expect(renderer).toHaveBeenCalledTimes(5);
    nextFrame(1533);
    expect(renderer).toHaveBeenCalledTimes(6);
  } finally {
    view.unmount();
    raf.mockRestore();
    cancel.mockRestore();
    clock.mockRestore();
    renderer.mockRestore();
  }
});

test('retains the SG2 top-view mesh through localization and TF updates and disposes it on removal', () => {
  const pose = { position: { x: 1, y: 2 }, orientation: { w: 1, z: 0 } };
  const footprint = { header: { frame_id: 'base_link' }, polygon: { points: Array.from({ length: 16 }, (_, i) => ({
    x: 0.42 * Math.cos(i * Math.PI / 8), y: 0.42 * Math.sin(i * Math.PI / 8),
  })) } };
  const props = { pose, footprint, showMap: false, showFootprint: true, renderProfile: 'autonomy', robotType: 'ffw_sg2_rev1' };
  const view = render(<MapViewer {...props} />);
  const mesh = mockMeshes.find((item) => item.name === 'sg2-top-down-silhouette');
  expect(mesh).toBeDefined();
  const marker = mesh.parent;
  expect(marker.position.x).toBe(1);
  expect(marker.position.y).toBe(2);
  expect(mockMeshes.some((item) => item.geometry.type === 'CircleGeometry')).toBe(false);
  expect(mockMeshes.some((item) => item.material.color.getHex() === 0x38bdf8)).toBe(false);
  const geometryDispose = jest.spyOn(mesh.geometry, 'dispose');
  const materialDispose = jest.spyOn(mesh.material, 'dispose');
  const nextPose = { position: { x: 4, y: -3 }, orientation: { w: Math.SQRT1_2, z: Math.SQRT1_2 } };
  view.rerender(<MapViewer {...props} pose={nextPose} tf={{ transforms: [] }} />);
  expect(marker.position.x).toBe(4);
  expect(marker.position.y).toBe(-3);
  expect(marker.rotation.z).toBeCloseTo(Math.PI / 2);
  expect(mockMeshes.filter((item) => item.name === mesh.name)).toEqual([mesh]);
  expect(geometryDispose).not.toHaveBeenCalled();
  expect(materialDispose).not.toHaveBeenCalled();
  view.rerender(<MapViewer {...props} pose={null} />);
  expect(marker.visible).toBe(false);
  view.rerender(<MapViewer {...props} />);
  expect(marker.visible).toBe(true);
  expect(geometryDispose).not.toHaveBeenCalled();
  view.unmount();
  expect(geometryDispose).toHaveBeenCalledTimes(1);
  expect(materialDispose).toHaveBeenCalledTimes(1);
  expect(marker.parent).toBeNull();
});

test.each([
  { renderProfile: 'default', robotType: 'ffw_sg2_rev1', showFootprint: true },
  { renderProfile: 'autonomy', robotType: 'ffw_f1', showFootprint: true },
  { renderProfile: 'autonomy', robotType: '', showFootprint: true },
  { renderProfile: 'autonomy', robotType: 'ffw_sg2_rev1', showFootprint: false },
])('returns to the generic pose marker when SG2 display is inapplicable: %j', (fallback) => {
  const pose = { position: { x: 1, y: 2 }, orientation: { w: 1, z: 0 } };
  const props = { pose, showMap: false, showFootprint: true, renderProfile: 'autonomy', robotType: 'ffw_sg2_rev1' };
  const view = render(<MapViewer {...props} />);
  const mesh = mockMeshes.find((item) => item.name === 'sg2-top-down-silhouette');
  const marker = mesh.parent;
  const dispose = jest.spyOn(mesh.geometry, 'dispose');
  view.rerender(<MapViewer {...props} {...fallback} />);
  expect(marker.parent).toBeNull();
  expect(dispose).toHaveBeenCalledTimes(1);
  expect(mockMeshes.some((item) => item.geometry.type === 'CircleGeometry' && item.parent?.parent)).toBe(true);
  view.rerender(<MapViewer {...props} />);
  const markers = mockMeshes.filter((item) => item.name === 'sg2-top-down-silhouette');
  expect(markers).toHaveLength(2);
  expect(markers[1].parent.parent).not.toBeNull();
});


test('navigation destination flag replaces only its waypoint label and is disposed on arrival', () => {
  const target = { id: 'kitchen', label: 'Kitchen', pose: { x: 1, y: 2 }, scope: 'observation' };
  const spots = [target, { id: 'tv', label: 'TV', pose: { x: 3, y: 4 } }];
  const props = { spots, showMap: false, renderProfile: 'autonomy', selectedSpotId: 'tv' };
  const view = render(<MapViewer {...props} navigationDestination={target} />);
  const flag = mockSprites.find(sprite => sprite.userData.destinationId === target.id);
  expect(flag).toBeDefined();
  expect(flag.position.toArray()).toEqual([1, 2, 0.05]);
  expect(flag.center.y).toBeLessThan(0);
  expect(flag.material.depthTest).toBe(false);
  expect(flag.raycast).not.toBe(require('three').Sprite.prototype.raycast);
  expect(mockSprites.filter(sprite => sprite.parent?.parent && sprite.userData.waypointHitLabel)
    .map(sprite => sprite.userData.spotId)).toEqual(['tv']);
  const disposeTexture = jest.spyOn(flag.material.map, 'dispose');
  const disposeMaterial = jest.spyOn(flag.material, 'dispose');
  view.rerender(<MapViewer {...props} navigationDestination={null} />);
  expect(flag.parent).toBeNull();
  expect(disposeTexture).toHaveBeenCalledTimes(1);
  expect(disposeMaterial).toHaveBeenCalledTimes(1);
  expect(mockSprites.filter(sprite => sprite.parent?.parent && sprite.userData.waypointHitLabel)
    .map(sprite => sprite.userData.spotId)).toEqual(['kitchen', 'tv']);
});

test('mission destination with colliding observation ID keeps the observation label and follows the task pose', () => {
  const observation = { id: 'table', label: 'Table view', pose: { x: 1, y: 1 } };
  const task = { id: 'table', label: 'Kitchen', pose: { x: 5, y: 6 }, scope: 'mission' };
  const overlay = { spots: [task], pathSpots: [observation, task], order: [], activeWaypointId: task.id };
  render(<MapViewer showMap={false} spots={[observation]} missionRouteOverlay={overlay} navigationDestination={task} />);
  const flags = mockSprites.filter(sprite => sprite.userData.destinationId);
  expect(flags).toHaveLength(1);
  expect(flags[0].position.toArray()).toEqual([5, 6, 0.05]);
  expect(mockSprites.filter(sprite => sprite.userData.waypointHitLabel)).toHaveLength(1);
  expect(mockSprites.filter(sprite => sprite.parent)).toHaveLength(2);
});


test('destination sprite survives live robot updates and releases its resources on unmount', () => {
  const destination = { id: 'kitchen', label: 'Kitchen', pose: { x: 1, y: 2 }, scope: 'observation' };
  const spots = [destination];
  const props = { spots, navigationDestination: destination, showMap: false, renderProfile: 'autonomy' };
  const pose = x => ({ position: { x, y: 0 }, orientation: { w: 1, z: 0 } });
  const view = render(<MapViewer {...props} pose={pose(0)} />);
  const flag = mockSprites.find(sprite => sprite.userData.destinationId === destination.id);
  const texture = flag.material.map;
  const disposeTexture = jest.spyOn(texture, 'dispose');
  const disposeMaterial = jest.spyOn(flag.material, 'dispose');
  view.rerender(<MapViewer {...props} pose={pose(1)} />);
  expect(mockSprites.filter(sprite => sprite.userData.destinationId === destination.id)).toEqual([flag]);
  expect(flag.material.map).toBe(texture);
  expect(disposeTexture).not.toHaveBeenCalled();
  expect(disposeMaterial).not.toHaveBeenCalled();
  view.unmount();
  expect(disposeTexture).toHaveBeenCalledTimes(1);
  expect(disposeMaterial).toHaveBeenCalledTimes(1);
});
