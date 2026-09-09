import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ReplayBufferVessel from './ReplayBufferVessel';

test('shows an empty vessel without fabricating plates', () => {
  render(<ReplayBufferVessel counts={{}} />);
  expect(screen.getByRole('img')).toHaveAttribute('data-capacity-empty', '200');
  expect(screen.queryAllByTestId('replay-cylinder-disc')).toHaveLength(0);
});

test('bounds geometry for large collections and preserves minority outcomes', () => {
  render(<ReplayBufferVessel counts={{ success: 10000, failure: 1, unlabeled: 1 }} />);
  expect(screen.getAllByTestId('replay-cylinder-disc')).toHaveLength(36);
  expect(screen.getByRole('img')).toHaveAttribute('data-capacity-percent', '100');
  const outcomes = screen.getAllByTestId('replay-cylinder-disc').map((disc) => disc.dataset.outcome);
  expect(outcomes).toEqual(expect.arrayContaining(['success', 'failure', 'unlabeled']));
});

test('uses unique gradient IDs and keyboard-accessible outcome controls', () => {
  const onClick = jest.fn();
  const { container } = render(<>
    <ReplayBufferVessel counts={{ success: 1 }} />
    <ReplayBufferVessel counts={{ success: 1 }} outcomeProps={() => ({ onClick, 'aria-label': 'Success data' })} />
  </>);
  const ids = [...container.querySelectorAll('[id]')].map((node) => node.id);
  expect(new Set(ids).size).toBe(ids.length);
  fireEvent.keyDown(screen.getByRole('button', { name: 'Success data' }), { key: 'Enter' });
  expect(onClick).toHaveBeenCalledTimes(1);
});
