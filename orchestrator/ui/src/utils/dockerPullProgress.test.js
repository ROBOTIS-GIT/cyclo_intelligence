import {
  createDockerPullProgressTracker,
  parseDockerPullSseBlock,
} from './dockerPullProgress';

test('parses docker pull SSE data blocks', () => {
  expect(parseDockerPullSseBlock(
    'data: {"id":"layer-a","status":"Downloading"}'
  )).toEqual({
    event: 'message',
    data: { id: 'layer-a', status: 'Downloading' },
  });
});

test('parses docker pull SSE event names', () => {
  expect(parseDockerPullSseBlock(
    'event: done\ndata: {"ok":true,"image":"robotis/groot-zenoh:1.3.1-arm64"}'
  )).toEqual({
    event: 'done',
    data: { ok: true, image: 'robotis/groot-zenoh:1.3.1-arm64' },
  });
});

test('reports active layer progress without pretending it is total image progress', () => {
  const tracker = createDockerPullProgressTracker();

  expect(tracker.update({
    id: 'layer-a',
    status: 'Downloading',
    progressDetail: { current: 50, total: 100 },
  })).toMatchObject({
    percent: null,
    layerPercent: 50,
    message: 'layer-a: Downloading 50%',
    detail: '0/1 layers ready',
  });

  expect(tracker.update({
    id: 'layer-b',
    status: 'Pulling fs layer',
  })).toMatchObject({
    percent: null,
    layerPercent: null,
    detail: '0/2 layers ready',
  });

  expect(tracker.update({
    id: 'layer-a',
    status: 'Downloading',
    progressDetail: { current: 100, total: 100 },
  })).toMatchObject({
    percent: null,
    layerPercent: 100,
    detail: '0/2 layers ready',
  });

  expect(tracker.update({
    id: 'layer-a',
    status: 'Pull complete',
  })).toMatchObject({
    percent: null,
    layerPercent: null,
    detail: '1/2 layers ready',
  });

  expect(tracker.complete()).toMatchObject({
    percent: 100,
    layerPercent: null,
    message: 'Image pull complete',
    detail: '2/2 layers ready',
  });
});
