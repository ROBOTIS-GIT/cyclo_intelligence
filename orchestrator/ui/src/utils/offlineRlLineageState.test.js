import {
  DEFAULT_OFFLINE_RL_LINEAGE_STATE,
  OFFLINE_RL_LINEAGE_STORAGE_KEY,
  advanceOfflineRLLineage,
  createOfflineRLLineage,
  persistOfflineRLLineageState,
  resolveOfflineRLLineageState,
} from './offlineRlLineageState';

const createStorage = () => {
  const values = new Map();
  return {
    getItem: jest.fn((key) => values.get(key) ?? null),
    setItem: jest.fn((key, value) => values.set(key, value)),
  };
};

describe('offline RL lineage state', () => {
  test('resolves a valid stored lineage', () => {
    const storage = createStorage();
    const lineage = {
      policyEpoch: 3,
      policyPath: '/workspace/model/act/epoch_0003/pretrained_model',
      forceFresh: false,
      lineageId: 'lineage-a',
    };
    storage.setItem(OFFLINE_RL_LINEAGE_STORAGE_KEY, JSON.stringify(lineage));

    expect(resolveOfflineRLLineageState(storage)).toEqual(lineage);
  });

  test.each([
    ['missing storage value', null],
    ['corrupt JSON', '{bad-json'],
    ['negative epoch', JSON.stringify({
      policyEpoch: -1,
      policyPath: '/workspace/model/act',
      forceFresh: false,
      lineageId: 'lineage-a',
    })],
    ['fractional epoch', JSON.stringify({
      policyEpoch: 1.5,
      policyPath: '/workspace/model/act',
      forceFresh: false,
      lineageId: 'lineage-a',
    })],
    ['wrong field type', JSON.stringify({
      policyEpoch: 1,
      policyPath: '/workspace/model/act',
      forceFresh: 'false',
      lineageId: 'lineage-a',
    })],
  ])('uses the epoch-zero non-fresh default for %s', (_label, storedValue) => {
    const storage = createStorage();
    if (storedValue !== null) {
      storage.setItem(OFFLINE_RL_LINEAGE_STORAGE_KEY, storedValue);
    }

    expect(resolveOfflineRLLineageState(storage)).toEqual(
      DEFAULT_OFFLINE_RL_LINEAGE_STATE
    );
  });

  test('persists a valid state and returns the persisted value', () => {
    const storage = createStorage();
    const lineage = {
      policyEpoch: 2,
      policyPath: '/workspace/model/act/round_0002/pretrained_model',
      forceFresh: false,
      lineageId: 'lineage-b',
    };

    expect(persistOfflineRLLineageState(lineage, storage)).toEqual(lineage);
    expect(JSON.parse(
      storage.getItem(OFFLINE_RL_LINEAGE_STORAGE_KEY)
    )).toEqual(lineage);
  });

  test('creates a fresh epoch-zero lineage while retaining the baseline policy', () => {
    const storage = createStorage();

    const lineage = createOfflineRLLineage(
      '/workspace/model/act/base/pretrained_model',
      { lineageId: 'lineage-new', storage }
    );

    expect(lineage).toEqual({
      policyEpoch: 0,
      policyPath: '/workspace/model/act/base/pretrained_model',
      forceFresh: true,
      lineageId: 'lineage-new',
    });
    expect(resolveOfflineRLLineageState(storage)).toEqual(lineage);
  });

  test('advances to an absolute policy epoch and clears force-fresh', () => {
    const storage = createStorage();
    const current = {
      policyEpoch: 2,
      policyPath: '/workspace/model/act/round_0002/pretrained_model',
      forceFresh: true,
      lineageId: 'lineage-c',
    };

    const advanced = advanceOfflineRLLineage(current, {
      policyEpoch: 5,
      policyPath: '/workspace/model/act/round_0005/pretrained_model',
      storage,
    });

    expect(advanced).toEqual({
      policyEpoch: 5,
      policyPath: '/workspace/model/act/round_0005/pretrained_model',
      forceFresh: false,
      lineageId: 'lineage-c',
    });
    expect(resolveOfflineRLLineageState(storage)).toEqual(advanced);
  });

  test('falls back without throwing when storage access is blocked', () => {
    const storage = {
      getItem: jest.fn(() => {
        throw new Error('blocked');
      }),
      setItem: jest.fn(() => {
        throw new Error('blocked');
      }),
    };

    expect(resolveOfflineRLLineageState(storage)).toEqual(
      DEFAULT_OFFLINE_RL_LINEAGE_STATE
    );
    expect(() => persistOfflineRLLineageState({
      policyEpoch: 0,
      policyPath: '',
      forceFresh: false,
      lineageId: '',
    }, storage)).not.toThrow();
  });
});
