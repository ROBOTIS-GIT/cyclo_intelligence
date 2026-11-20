let announceRecording, disableRecordingAudio, enableRecordingAudio, preloadRecordingAudio, recordingSoundState;

let sources;
let ctx;
const originalFetch = global.fetch;
const originalContext = window.AudioContext;
beforeEach(() => {
  jest.resetModules();
  ({ announceRecording, disableRecordingAudio, enableRecordingAudio, preloadRecordingAudio, recordingSoundState } =
    require('./recordingAudio'));
  sources = [];
  global.fetch = jest.fn(async url => ({
    ok: true, arrayBuffer: async () => ({ slice: () => url }),
  }));
  ctx = {
    state: 'suspended',
    resume: jest.fn(async () => { ctx.state = 'running'; }),
    suspend: jest.fn(async () => { ctx.state = 'suspended'; }),
    close: jest.fn(async () => { ctx.state = 'closed'; }),
    decodeAudioData: async bytes => bytes,
    createBufferSource: () => {
      const node = { connect() {}, disconnect() {}, start() {} };
      sources.push(node);
      return node;
    },
  };
  window.AudioContext = jest.fn(() => ctx);
});
afterEach(() => {
  disableRecordingAudio();
  global.fetch = originalFetch;
  window.AudioContext = originalContext;
});

test('unlock confirms audibly, reuses preloaded audio and plays events in order', async () => {
  await preloadRecordingAudio();
  expect(global.fetch).toHaveBeenCalledTimes(7);
  announceRecording('start');
  expect(sources).toHaveLength(0);
  await enableRecordingAudio();
  expect(recordingSoundState()).toBe('on');
  for (const key of ['start', 'cameraWarning', 'cameraWarning', 'finish']) announceRecording(key);
  expect(sources).toHaveLength(1);
  for (const buffer of ['enabled.mp3', 'start.mp3', 'camera-warning.mp3', 'finish.mp3']) {
    const current = sources[sources.length - 1];
    expect(current.buffer).toBe(buffer);
    current.onended();
  }
  expect(sources).toHaveLength(4);
  expect(global.fetch).toHaveBeenCalledTimes(7);
});

test('mute drops the queue; late ended callbacks cannot resume it', async () => {
  await enableRecordingAudio();
  announceRecording('finish');
  disableRecordingAudio();
  sources[0].onended();
  announceRecording('cancel');
  expect(sources).toHaveLength(1);
  expect(ctx.close).toHaveBeenCalledTimes(1);
});

test('blocked playback stays off and can be enabled again', async () => {
  ctx.resume.mockRejectedValueOnce(new Error('blocked'));
  await expect(enableRecordingAudio()).rejects.toThrow('blocked');
  expect(recordingSoundState()).toBe('off');
  expect(sources).toHaveLength(0);
  await enableRecordingAudio();
  expect(sources).toHaveLength(1);
});

test('idle output suspends and the next event wakes it without a second click', async () => {
  await enableRecordingAudio();
  sources[0].onended();
  await Promise.resolve();
  ctx.onstatechange();
  expect(ctx.state).toBe('suspended');
  expect(recordingSoundState()).toBe('on');
  announceRecording('start');
  await Promise.resolve();
  expect(ctx.state).toBe('running');
  expect(sources[1].buffer).toBe('start.mp3');
});

test.each([false, true])('cue arriving during suspend is preserved unless muted (%s)', async muted => {
  let finishSuspend;
  ctx.suspend.mockImplementationOnce(() => new Promise(resolve => {
    finishSuspend = () => { ctx.state = 'suspended'; resolve(); };
  }));
  await enableRecordingAudio();
  sources[0].onended();
  announceRecording('finish');
  if (muted) disableRecordingAudio();
  finishSuspend();
  await Promise.resolve();
  await Promise.resolve();
  expect(sources.map(s => s.buffer)).toEqual(muted ? ['enabled.mp3'] : ['enabled.mp3', 'finish.mp3']);
});
