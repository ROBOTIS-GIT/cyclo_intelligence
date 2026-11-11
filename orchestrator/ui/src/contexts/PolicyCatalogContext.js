// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const PolicyCatalogContext = createContext(null);

const validateCatalog = (catalog) => {
  if (!catalog || catalog.schema_version !== 1 || !Array.isArray(catalog.runtimes)) {
    throw new Error('Invalid policy catalog response');
  }
  if (catalog.runtimes.length === 0) {
    throw new Error('Policy catalog has no runtimes');
  }
  catalog.runtimes.forEach((runtime) => {
    if (!runtime.id || !runtime.label || !Array.isArray(runtime.models) || runtime.models.length === 0) {
      throw new Error('Policy catalog contains an invalid runtime');
    }
  });
  return catalog;
};

export const policyOptions = (catalog) => (
  (catalog?.runtimes || []).flatMap((runtime) => (
    runtime.models.map((model) => ({
      ...model,
      runtime,
      value: model.policy_id,
      serviceType: runtime.id,
      policyType: model.id,
    }))
  ))
);

export const findPolicy = (
  catalog,
  policyId,
  serviceType = '',
  policyType = ''
) => {
  const options = policyOptions(catalog);
  const requested = String(policyId || '').trim();
  if (requested) {
    return options.find((option) => (
      option.policy_id === requested || option.aliases?.includes(requested)
    )) || null;
  }
  const runtimeId = String(serviceType || '').trim();
  const modelId = String(policyType || '').trim();
  const composite = runtimeId && modelId ? `${runtimeId}:${modelId}` : '';
  const legacyMatch = options.find((option) => (
    (composite && option.policy_id === composite) ||
    (modelId && option.runtime.id === runtimeId && option.aliases?.includes(modelId))
  ));
  if (legacyMatch) return legacyMatch;
  const runtimeOptions = options.filter((option) => option.runtime.id === runtimeId);
  return runtimeOptions.length === 1 ? runtimeOptions[0] : null;
};

export const parameterDefaults = (model, bindingPrefix) => Object.fromEntries(
  (model?.parameters || [])
    .filter((parameter) => parameter.binding.startsWith(bindingPrefix))
    .filter((parameter) => Object.prototype.hasOwnProperty.call(parameter, 'default'))
    .map((parameter) => [parameter.key, parameter.default])
);

export function PolicyCatalogProvider({ children, initialCatalog = null }) {
  const validatedInitialCatalog = useMemo(
    () => (initialCatalog ? validateCatalog(initialCatalog) : null),
    [initialCatalog]
  );
  const [catalog, setCatalog] = useState(validatedInitialCatalog);
  const [status, setStatus] = useState(validatedInitialCatalog ? 'ready' : 'loading');
  const [error, setError] = useState('');
  const [generation, setGeneration] = useState(0);

  const retry = useCallback(() => {
    setStatus('loading');
    setError('');
    setGeneration((value) => value + 1);
  }, []);

  useEffect(() => {
    if (validatedInitialCatalog) {
      setCatalog(validatedInitialCatalog);
      setStatus('ready');
      setError('');
      return undefined;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch('/api/policies/catalog');
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || `Policy catalog request failed (${response.status})`);
        }
        const validated = validateCatalog(payload);
        if (!cancelled) {
          setCatalog(validated);
          setStatus('ready');
          setError('');
        }
      } catch (loadError) {
        if (!cancelled) {
          setCatalog(null);
          setStatus('error');
          setError(loadError?.message || 'Failed to load policy catalog');
        }
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [generation, validatedInitialCatalog]);

  const value = useMemo(() => ({ catalog, status, error, retry }), [catalog, status, error, retry]);
  return (
    <PolicyCatalogContext.Provider value={value}>
      {children}
    </PolicyCatalogContext.Provider>
  );
}

export const usePolicyCatalog = () => {
  const value = useContext(PolicyCatalogContext);
  if (!value) {
    throw new Error('usePolicyCatalog must be used inside PolicyCatalogProvider');
  }
  return value;
};
