// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import toast from 'react-hot-toast';

import TreeListModal from './TreeListModal';

const mockListBtTrees = jest.fn();

jest.mock('../btTreesApi', () => ({
  listBtTrees: () => mockListBtTrees(),
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { error: jest.fn() },
}));

beforeEach(() => {
  mockListBtTrees.mockReset();
  toast.error.mockClear();
});

test('presents saved XML documents as tasks without exposing BT terminology', async () => {
  const onClose = jest.fn();
  const onSelect = jest.fn();
  mockListBtTrees.mockResolvedValue({
    directory: '/tasks',
    trees: [{ name: 'pick-and-place.xml', path: '/tasks/pick-and-place.xml', modified_at: 1 }],
  });

  render(
    <TreeListModal
      isOpen
      onClose={onClose}
      onSelect={onSelect}
      variant="autonomy-studio"
    />,
  );

  expect(screen.getByRole('heading', { name: 'Open Task' })).toBeInTheDocument();
  const task = await screen.findByRole('button', { name: 'pick-and-place.xml' });
  expect(screen.queryByText(/BT|behavior tree|tree XML/i)).not.toBeInTheDocument();

  fireEvent.click(task);
  expect(onSelect).toHaveBeenCalledWith({
    name: 'pick-and-place.xml',
    full_path: '/tasks/pick-and-place.xml',
  });
  expect(onClose).toHaveBeenCalled();
});

test('uses task language for empty and failed library states', async () => {
  mockListBtTrees.mockResolvedValueOnce({ directory: '/tasks', trees: [] });
  const view = render(
    <TreeListModal isOpen onClose={jest.fn()} onSelect={jest.fn()} />,
  );

  expect(await screen.findByText('No saved tasks found')).toBeInTheDocument();

  mockListBtTrees.mockRejectedValueOnce(new Error('offline'));
  view.rerender(
    <TreeListModal isOpen={false} onClose={jest.fn()} onSelect={jest.fn()} />,
  );
  view.rerender(
    <TreeListModal isOpen onClose={jest.fn()} onSelect={jest.fn()} />,
  );

  expect(await screen.findByText('Failed to load tasks')).toBeInTheDocument();
  await waitFor(() => {
    expect(toast.error).toHaveBeenCalledWith('Failed to load tasks: offline');
  });
});
