import enabled from '../assets/recording-audio/enabled.mp3';
import start from '../assets/recording-audio/start.mp3';
import finish from '../assets/recording-audio/finish.mp3';
import cancel from '../assets/recording-audio/cancel.mp3';
import reviewOn from '../assets/recording-audio/review-on.mp3';
import reviewOff from '../assets/recording-audio/review-off.mp3';
import cameraWarning from '../assets/recording-audio/camera-warning.mp3';

const clips = { enabled, start, finish, cancel, review_on: reviewOn, review_off: reviewOff, cameraWarning };
let preload;
export function preloadRecordingAudio() {
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
let changingOutput = false;
const queue = [];
const listeners = new Set();
let state = 'off';
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
  if (state !== 'on' || playing || changingOutput) return;
  const ctx = context;
  const suspend = !queue.length && ctx.state === 'running';
  if (suspend || (queue.length && ctx.state === 'suspended')) {
    // A cue can arrive while suspend() is in flight. Recheck the queue once
    // the transition completes, and never wake an old, muted context.
    changingOutput = true;
    (suspend ? ctx.suspend() : ctx.resume()).then(() => {
      if (context !== ctx) return;
      changingOutput = false;
      if (!suspend && ctx.state !== 'running') disableRecordingAudio();
      else if (queue.length) playNext();
    }).catch(() => { if (context === ctx) disableRecordingAudio(); });
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
  if (state !== 'on' || !buffers.has(key)) return;
  if (key === 'cameraWarning' && (playing?.key === key || queue.includes(key))) return;
  // Don't narrate a long backlog after rapid changes.
  if (queue.length === 3) queue.shift();
  queue.push(key);
  playNext();
}

export function disableRecordingAudio() {
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
  setState('off');
}

export async function enableRecordingAudio() {
  if (state !== 'off') return;
  // Resume directly in the button's click handler, before any async loading.
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioContext();
  context = ctx;
  setState('loading');
  try {
    const [, decoded] = await Promise.all([ctx.resume(), preloadRecordingAudio().then(entries =>
      Promise.all(entries.map(async ([key, bytes]) => [key, await ctx.decodeAudioData(bytes.slice(0))])))]);
    if (context !== ctx) return;
    if (ctx.state !== 'running') throw new Error('Audio playback is blocked');
    buffers = new Map(decoded);
    ctx.onstatechange = () => {
      if (context === ctx && (ctx.state === 'closed' || ctx.state === 'interrupted'
        || (ctx.state === 'suspended' && playing))) disableRecordingAudio();
    };
    setState('on');
    announceRecording('enabled');
  } catch (error) {
    if (context === ctx) disableRecordingAudio();
    throw error;
  }
}
