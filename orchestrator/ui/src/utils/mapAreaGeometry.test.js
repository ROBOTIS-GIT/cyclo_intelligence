import {
  mapAreaSelectionBounds,
  mapPointToAreaGridCell,
} from './mapAreaGeometry';

const identityOrigin = {
  position: { x: 0, y: 0, z: 0 },
  orientation: { x: 0, y: 0, z: 0, w: 1 },
};

test('uses identical inclusive cells for PGM and OccupancyGrid area selections', () => {
  const image = {
    width: 3,
    height: 2,
    resolution: 1,
    origin: identityOrigin,
  };
  const selection = { startX: 0.05, startY: 0.05, endX: 2.95, endY: 1.95 };
  const expected = { x_min: 0, y_min: 0, x_max: 2, y_max: 1 };

  expect(mapAreaSelectionBounds(image, selection)).toEqual(expected);
  expect(mapAreaSelectionBounds({ info: image }, selection)).toEqual(expected);
  expect(mapAreaSelectionBounds(image, {
    startX: selection.endX,
    startY: selection.endY,
    endX: selection.startX,
    endY: selection.startY,
  })).toEqual(expected);
});

test('converts points in a rotated map origin before validating the cell', () => {
  const source = {
    width: 3,
    height: 2,
    resolution: 1,
    origin: {
      position: { x: 10, y: 20, z: 0 },
      orientation: {
        x: 0,
        y: 0,
        z: Math.sin(Math.PI / 4),
        w: Math.cos(Math.PI / 4),
      },
    },
  };

  // Local grid point (1.25, 0.25) rotated +90 degrees around the origin.
  expect(mapPointToAreaGridCell(source, 9.75, 21.25)).toEqual({ x: 1, y: 0 });
  expect(mapPointToAreaGridCell(source, 10.25, 23.25)).toBeNull();
});
