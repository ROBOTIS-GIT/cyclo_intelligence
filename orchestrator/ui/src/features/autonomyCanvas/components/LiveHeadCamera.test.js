import { act, fireEvent, render, screen } from '@testing-library/react';
import LiveHeadCamera from './LiveHeadCamera';
jest.mock('react-redux', () => ({useSelector: fn => fn({ros:{rosHost:'robot.local'}})}));
beforeEach(() => { jest.useFakeTimers(); });
afterEach(() => { jest.useRealTimers(); jest.restoreAllMocks(); });

test('keeps one compressed stream across rerenders and closes it when hidden or unmounted', () => {
  const hidden = jest.spyOn(document, 'hidden', 'get').mockReturnValue(false);
  const view = render(<LiveHeadCamera />);
  const first = screen.getByAltText('Live head camera');
  expect(first.src).toContain('robot.local:7085/stream?type=ros_compressed');
  view.rerender(<LiveHeadCamera />);
  expect(screen.getByAltText('Live head camera')).toBe(first);
  hidden.mockReturnValue(true);
  fireEvent(document, new Event('visibilitychange'));
  expect(first).not.toHaveAttribute('src');
  expect(screen.queryByAltText('Live head camera')).not.toBeInTheDocument();
  hidden.mockReturnValue(false);
  fireEvent(document, new Event('visibilitychange'));
  const resumed = screen.getByAltText('Live head camera');
  view.unmount();
  expect(resumed).not.toHaveAttribute('src');
  act(() => jest.runOnlyPendingTimers());
  expect(screen.queryByAltText('Live head camera')).not.toBeInTheDocument();
});

test('limits failed stream reconnects and offers an explicit retry', () => {
  render(<LiveHeadCamera />);
  for(let i=0;i<4;i++) {
    fireEvent.error(screen.getByAltText('Live head camera'));
    act(() => jest.runOnlyPendingTimers());
  }
  expect(screen.getByText('Camera unavailable')).toBeInTheDocument();
  expect(screen.queryByAltText('Live head camera')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Reconnect camera'}));
  expect(screen.getAllByAltText('Live head camera')).toHaveLength(1);
});
