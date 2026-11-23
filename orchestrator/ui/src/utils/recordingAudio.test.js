let announceRecording, disableRecordingAudio, enableRecordingAudio, initializeRecordingAudio, recordingSoundState;

let sources;
let ctx;
const originalFetch = global.fetch;
const originalContext = window.AudioContext;
beforeEach(() => {
  localStorage.clear();
  jest.resetModules();
  ({ announceRecording, disableRecordingAudio, enableRecordingAudio, initializeRecordingAudio, recordingSoundState } =
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

const settle = () => new Promise(resolve => setTimeout(resolve, 0));

test('sound defaults on, opens silently, releases idle output and plays the first recording cue', async () => {
  expect(recordingSoundState()).toBe('on');
  const cleanup = initializeRecordingAudio();
  await settle();
  expect(sources).toHaveLength(0);
  expect(ctx.state).toBe('suspended');
  announceRecording('start');
  await settle();
  expect(sources.map(s => s.buffer)).toEqual(['start.mp3']);
  cleanup();
});

test.each(['click', 'keydown'])('a normal %s unlocks blocked autoplay without announcing stale events', async event => {
  ctx.resume.mockRejectedValueOnce(new DOMException('Gesture required', 'NotAllowedError'));
  const cleanup = initializeRecordingAudio();
  await settle();
  expect(recordingSoundState()).toBe('on');
  announceRecording('start');
  window.dispatchEvent(new Event(event));
  await settle();
  expect(sources).toHaveLength(0);
  announceRecording('finish');
  await settle();
  expect(sources.map(s => s.buffer)).toEqual(['finish.mp3']);
  cleanup();
});

test('mute before the first interaction survives a reload and never wakes the old context', async () => {
  ctx.resume.mockImplementationOnce(() => new Promise(() => {}));
  const cleanup = initializeRecordingAudio();
  disableRecordingAudio();
  window.dispatchEvent(new Event('click'));
  expect(ctx.resume).toHaveBeenCalledTimes(1);
  cleanup();
  jest.resetModules();
  const reloaded = require('./recordingAudio');
  expect(reloaded.recordingSoundState()).toBe('off');
  window.AudioContext.mockClear();
  reloaded.initializeRecordingAudio();
  expect(window.AudioContext).not.toHaveBeenCalled();
  await reloaded.enableRecordingAudio();
  expect(sources.map(s => s.buffer)).toEqual(['enabled.mp3']);
  jest.resetModules();
  expect(require('./recordingAudio').recordingSoundState()).toBe('on');
  reloaded.disableRecordingAudio();
});
afterEach(() => {
  disableRecordingAudio();
  global.fetch = originalFetch;
  window.AudioContext = originalContext;
});

test('unlock confirms audibly, reuses preloaded audio and plays events in order', async () => {
  initializeRecordingAudio();
  await settle();
  disableRecordingAudio();
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
