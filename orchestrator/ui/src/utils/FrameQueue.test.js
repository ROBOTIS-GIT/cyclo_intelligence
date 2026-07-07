import { FrameQueue } from './FrameQueue';

describe('FrameQueue', () => {
  it('evicts the oldest frames when maxFrames is exceeded', () => {
    const queue = new FrameQueue(3);
    queue.initTimelines(1);

    queue.put(0, 1n, { jpegBuffer: new ArrayBuffer(1), format: 'jpeg' });
    queue.put(0, 2n, { jpegBuffer: new ArrayBuffer(1), format: 'jpeg' });
    queue.put(0, 3n, { jpegBuffer: new ArrayBuffer(1), format: 'jpeg' });
    queue.put(0, 4n, { jpegBuffer: new ArrayBuffer(1), format: 'jpeg' });

    expect(queue.size).toBe(3);
    expect(queue.timelines[0]).toEqual([2n, 3n, 4n]);
  });

  it('removes evicted frames from their topic timeline', () => {
    const queue = new FrameQueue(2);
    queue.initTimelines(2);

    queue.put(0, 10n, { jpegBuffer: new ArrayBuffer(1), format: 'jpeg' });
    queue.put(1, 20n, { jpegBuffer: new ArrayBuffer(1), format: 'jpeg' });
    queue.put(0, 30n, { jpegBuffer: new ArrayBuffer(1), format: 'jpeg' });

    expect(queue.size).toBe(2);
    expect(queue.timelines[0]).toEqual([30n]);
    expect(queue.timelines[1]).toEqual([20n]);
  });
});
