import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import {
  PolicyCatalogProvider,
  usePolicyCatalog,
} from './PolicyCatalogContext';
import { testPolicyCatalog } from '../testUtils/policyCatalog';

const Probe = () => {
  const { catalog, status, error, retry } = usePolicyCatalog();
  return (
    <div>
      <span>{status}</span>
      <span>{catalog?.runtimes?.length || 0}</span>
      <span>{error}</span>
      <button type="button" onClick={retry}>Retry</button>
    </div>
  );
};

const response = (payload, ok = true) => ({
  ok,
  status: ok ? 200 : 500,
  json: async () => payload,
});

describe('PolicyCatalogProvider', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('loads and validates the supervisor catalog', async () => {
    global.fetch = jest.fn().mockResolvedValue(response(testPolicyCatalog));

    render(<PolicyCatalogProvider><Probe /></PolicyCatalogProvider>);

    await waitFor(() => expect(screen.getByText('ready')).toBeInTheDocument());
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledWith('/api/policies/catalog');
  });

  test('fails closed and retries after a catalog error', async () => {
    global.fetch = jest.fn()
      .mockRejectedValueOnce(new Error('catalog offline'))
      .mockResolvedValueOnce(response(testPolicyCatalog));

    render(<PolicyCatalogProvider><Probe /></PolicyCatalogProvider>);

    await waitFor(() => expect(screen.getByText('error')).toBeInTheDocument());
    expect(screen.getByText('catalog offline')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(screen.getByText('ready')).toBeInTheDocument());
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
});
