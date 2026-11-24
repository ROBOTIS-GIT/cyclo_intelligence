import enabled from '../assets/recording-audio/enabled.mp3';
import start from '../assets/recording-audio/start.mp3';
import finish from '../assets/recording-audio/finish.mp3';
import cancel from '../assets/recording-audio/cancel.mp3';
import reviewOn from '../assets/recording-audio/review-on.mp3';
import reviewOff from '../assets/recording-audio/review-off.mp3';
import cameraWarning from '../assets/recording-audio/camera-warning.mp3';

const clips = { enabled, start, finish, cancel, review_on: reviewOn, review_off: reviewOff, cameraWarning };
let preload;
function preloadRecordingAudio() {
  if (!preload) {
    preload = Promise.all(Object.entries(clips).map(async ([key, url]) => {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Could not load recording sounds');
      return [key, await response.arrayBuffer()];
    })).catch(error => { preload = null; throw error; });
  }
  return preload;
}

let context;
let buffers;
let playing;
let cancelUnlock;
let changingOutput = false;
const queue = [];
const listeners = new Set();
const muteKey = 'cyclo_intelligence.recordingAudio.muted';
function initialState() {
  try { return window.localStorage.getItem(muteKey) === 'true' ? 'off' : 'on'; }
  catch { return 'on'; }
}
function rememberMute(muted) {
  try { window.localStorage.setItem(muteKey, String(muted)); }
  catch { /* The preference still works in memory. */ }
}
let state = initialState();
export const recordingSoundState = () => state;
export function subscribeRecordingSound(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
function setState(value) {
  state = value;
  listeners.forEach(listener => listener());
}

function playNext() {
  if (state !== 'on' || !buffers || playing || changingOutput) return;
  const ctx = context;
  const suspend = !queue.length && ctx.state === 'running';
  if (suspend || (queue.length && ctx.state === 'suspended')) {
    // A cue can arrive while suspend() is in flight. Recheck the queue once
    // the transition completes, and never wake an old, muted context.
    changingOutput = true;
    (suspend ? ctx.suspend() : ctx.resume()).then(() => {
      if (context !== ctx) return;
      changingOutput = false;
      if (!suspend && ctx.state !== 'running') stopOutput();
      else if (queue.length) playNext();
    }).catch(() => { if (context === ctx) stopOutput(); });
    return;
  }
  if (!queue.length || ctx.state !== 'running') return;
  const key = queue.shift();
  const source = context.createBufferSource();
  source.buffer = buffers.get(key);
  source.connect(context.destination);
  playing = { key, source };
  source.onended = () => {
    source.disconnect();
    if (playing?.source !== source) return;
    playing = null;
    playNext();
  };
  source.start();
}

export function announceRecording(key) {
  if (state !== 'on' || !Object.hasOwn(clips, key)) return;
  // Keep cues while decoding, but never replay events from before autoplay was allowed.
  if (!buffers && context?.state !== 'running') return;
  if (key === 'cameraWarning' && (playing?.key === key || queue.includes(key))) return;
  // Don't narrate a long backlog after rapid changes.
  if (queue.length === 3) queue.shift();
  queue.push(key);
  playNext();
}

export function disableRecordingAudio() {
  rememberMute(true);
  stopOutput();
}

function stopOutput() {
  releaseOutput();
  setState('off');
}

function releaseOutput() {
  cancelUnlock?.();
  queue.length = 0;
  changingOutput = false;
  playing = null;
  buffers = null;
  const old = context;
  context = null;
  if (old) {
    old.onstatechange = null;
    if (old.state !== 'closed') old.close().catch(() => {});
  }
}

function resumeOnStartup(ctx) {
  return new Promise((resolve, reject) => {
    const finish = (error) => {
      window.removeEventListener('click', resume, true);
      window.removeEventListener('keydown', resume, true);
      if (cancelUnlock === cancel) cancelUnlock = null;
      if (error) reject(error);
      else resolve();
    };
    const cancel = () => finish();
    // Autoplay policies may reject immediately or leave resume() pending.
    // Retry inside a gesture handler, before any asynchronous work.
    const resume = () => {
      ctx.resume().then(() => {
        if (ctx.state === 'running') finish();
      }).catch(error => { if (error.name !== 'NotAllowedError') finish(error); });
    };
    cancelUnlock = cancel;
    window.addEventListener('click', resume, true);
    window.addEventListener('keydown', resume, true);
    resume();
  });
}

async function openOutput(confirm) {
  if (context) return;
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  let ctx;
  try {
    ctx = new AudioContext();
    context = ctx;
    const [, decoded] = await Promise.all([confirm ? ctx.resume() : resumeOnStartup(ctx), preloadRecordingAudio().then(entries =>
      Promise.all(entries.map(async ([key, bytes]) => [key, await ctx.decodeAudioData(bytes.slice(0))])))]);
    if (context !== ctx) return;
    if (ctx.state !== 'running') throw new Error('Audio playback is blocked');
    buffers = new Map(decoded);
    ctx.onstatechange = () => {
      if (context === ctx && (ctx.state === 'closed' || ctx.state === 'interrupted'
        || (ctx.state === 'suspended' && playing))) stopOutput();
    };
    if (confirm) announceRecording('enabled');
    else playNext();
  } catch (error) {
    if (!ctx || context === ctx) stopOutput();
    throw error;
  }
}

export function enableRecordingAudio() {
  rememberMute(false);
  setState('on');
  return openOutput(true);
}

export function initializeRecordingAudio() {
  if (state === 'on') openOutput(false).catch(() => {});
  return releaseOutput;
}
