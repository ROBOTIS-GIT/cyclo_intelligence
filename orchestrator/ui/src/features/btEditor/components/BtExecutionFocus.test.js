import React from 'react';
import { act, fireEvent, render } from '@testing-library/react';
import BtExecutionFocus, { executionFocusIds } from './BtExecutionFocus';
const mockFit = jest.fn();
jest.mock('@xyflow/react', () => ({
  useReactFlow: () => ({ fitView: mockFit }),
  useNodesInitialized: () => true,
  Panel: ({children}) => <div>{children}</div>,
}));
const node = (id, active=true, hidden=false) => ({id, hidden, data:{isActive:active}});
const edges = [{source:'root', target:'a'}, {source:'root', target:'b'}];
test('focuses leaves rather than the full highlighted branch; collapsed branch remains visible', () => {
  expect(executionFocusIds([node('root'), node('a'), node('b',false)],edges)).toEqual(['a']);
  expect(executionFocusIds([node('root'), node('a',true,true)],edges)).toEqual(['root']);
});
test('follows node changes, ignores repeated updates, and can pause tracking', () => {
  jest.useFakeTimers(); mockFit.mockClear();
  const v=render(<BtExecutionFocus nodes={[node('root'),node('a')]} edges={edges}/>);
  act(()=>jest.advanceTimersByTime(200));
  expect(mockFit).toHaveBeenLastCalledWith(expect.objectContaining({nodes:[{id:'a'}],minZoom:.9}));
  v.rerender(<BtExecutionFocus nodes={[node('root'),node('a')]} edges={edges}/>);
  act(()=>jest.advanceTimersByTime(200)); expect(mockFit).toHaveBeenCalledTimes(1);
  fireEvent.click(v.getByRole('button'));
  v.rerender(<BtExecutionFocus nodes={[node('root'),node('b')]} edges={edges}/>);
  act(()=>jest.advanceTimersByTime(200)); expect(mockFit).toHaveBeenCalledTimes(1);
  fireEvent.click(v.getByRole('button'));
  act(()=>jest.advanceTimersByTime(200)); expect(mockFit).toHaveBeenLastCalledWith(expect.objectContaining({nodes:[{id:'b'}]}));
  v.rerender(<BtExecutionFocus nodes={[node('root',false),node('b',false)]} edges={edges}/>);
  act(()=>jest.advanceTimersByTime(200)); expect(mockFit).toHaveBeenCalledTimes(2);
  v.unmount();jest.useRealTimers();
});
