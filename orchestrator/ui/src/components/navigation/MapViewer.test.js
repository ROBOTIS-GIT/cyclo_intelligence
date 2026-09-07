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

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import {
  MapViewer,
  mapAreaPreviewCellIndices,
  mapRenderIntervalMs,
  updateGlobalCostmapTexture,
} from './MapViewer';

const mockPointPositions = [];
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
    }

    setPixelRatio() {}

    setClearColor() {}

    setSize() {}

    render() {}

    dispose() {}
  }

  class Points extends actual.Points {
    constructor(geometry, material) {
      super(geometry, material);
      mockPointPositions.push(Array.from(geometry.attributes.position.array));
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
    constructor() {
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
  mockPointPositions.length = 0;
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
