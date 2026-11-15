import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ResizableWorkspacePanel from './ResizableWorkspacePanel';

beforeEach(() => window.localStorage.clear());

function setup(side = 'left', label = 'Replay Buffer') {
  const result = render(
    <div><ResizableWorkspacePanel side={side} label={label} data-testid="panel">
      <button>Keep mounted</button>
    </ResizableWorkspacePanel></div>
  );
  const panel = screen.getByTestId('panel');
  panel.getBoundingClientRect = () => ({ width: Number(panel.style.getPropertyValue('--pg-panel-width').replace('px', '')) || 600 });
  Object.defineProperty(panel.parentElement, 'clientWidth', { configurable: true, value: 1400 });
  return { ...result, panel, handle: screen.getByRole('separator'), button: screen.getByText('Keep mounted') };
}

test('resizes beyond half the workspace, persists per panel and resets', () => {
  const { panel, handle, button, unmount } = setup();
  for (let i = 0; i < 10; i += 1) fireEvent.keyDown(handle, { key: 'ArrowRight' });
  expect(panel.style.getPropertyValue('--pg-panel-width')).toBe('840px');
  expect(window.localStorage.getItem('playground.panel-width.left')).toBe('840');
  expect(window.localStorage.getItem('playground.panel-width.right')).toBeNull();
  expect(screen.getByText('Keep mounted')).toBe(button);
  unmount();
  const next = setup();
  expect(next.panel.style.getPropertyValue('--pg-panel-width')).toBe('840px');
  fireEvent.doubleClick(next.handle);
  expect(next.panel.style.getPropertyValue('--pg-panel-width')).toBe('');
  expect(window.localStorage.getItem('playground.panel-width.left')).toBeNull();
});

test('right panel expands toward the left and stays within the available area', () => {
  const { panel, handle } = setup('right', 'Training');
  for (let i = 0; i < 60; i += 1) fireEvent.keyDown(handle, { key: 'ArrowLeft' });
  expect(panel.style.getPropertyValue('--pg-panel-width')).toBe('1368px');
  for (let i = 0; i < 60; i += 1) fireEvent.keyDown(handle, { key: 'ArrowRight' });
  expect(panel.style.getPropertyValue('--pg-panel-width')).toBe('360px');
});

test('pointer drag commits width only on release and cancellation restores it', () => {
  const OriginalPointerEvent = window.PointerEvent;
  window.PointerEvent = MouseEvent;
  try {
    const { panel, handle } = setup();
    handle.setPointerCapture = jest.fn();
    handle.hasPointerCapture = () => false;
    fireEvent.pointerDown(handle, { button: 0, clientX: 600 });
    fireEvent.pointerMove(handle, { clientX: 1000 });
    expect(panel.style.getPropertyValue('--pg-panel-width')).toBe('1000px');
    expect(window.localStorage.getItem('playground.panel-width.left')).toBeNull();
    fireEvent.pointerUp(handle, { clientX: 1000 });
    expect(window.localStorage.getItem('playground.panel-width.left')).toBe('1000');
    fireEvent.pointerDown(handle, { button: 0, clientX: 1000 });
    fireEvent.pointerMove(handle, { clientX: 700 });
    fireEvent.pointerCancel(handle);
    expect(panel.style.getPropertyValue('--pg-panel-width')).toBe('1000px');
  } finally {
    window.PointerEvent = OriginalPointerEvent;
  }
});

test('ignores invalid saved widths and continues when storage writes fail', () => {
  window.localStorage.setItem('playground.panel-width.left', 'NaN');
  const { panel, handle } = setup();
  expect(panel.style.getPropertyValue('--pg-panel-width')).toBe('');
  const spy = jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('disabled'); });
  try {
    fireEvent.keyDown(handle, { key: 'ArrowRight' });
    expect(panel.style.getPropertyValue('--pg-panel-width')).toBe('624px');
  } finally {
    spy.mockRestore();
  }
});
