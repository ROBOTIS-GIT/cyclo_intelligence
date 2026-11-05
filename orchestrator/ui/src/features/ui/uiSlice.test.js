import PageType from '../../constants/pageType';
import {
  CURRENT_PAGE_STORAGE_KEY,
  LEGACY_BT_MANAGER_PAGE,
  LEGACY_MISSION_CANVAS_PAGE,
  LEGACY_MISSION_CANVAS_SESSION_STORAGE_KEY,
  LEGACY_NAVIGATION_PAGE,
  AUTONOMY_STUDIO_SESSION_STORAGE_KEY,
  persistCurrentPage,
  resolveInitialPageState,
} from './uiSlice';

const makeStorage = (initial = {}) => {
  const values = { ...initial };
  return {
    getItem: jest.fn((key) => (
      Object.prototype.hasOwnProperty.call(values, key) ? values[key] : null
    )),
    setItem: jest.fn((key, value) => {
      values[key] = value;
    }),
    values,
  };
};

describe('uiSlice page session state', () => {
  test('restores a valid page from tab session storage', () => {
    const storage = makeStorage({
      [CURRENT_PAGE_STORAGE_KEY]: PageType.INFERENCE,
    });

    expect(resolveInitialPageState(storage)).toEqual({
      currentPage: PageType.INFERENCE,
      restoredPageFromSession: true,
    });
  });

  test('falls back to Home when stored page is missing or invalid', () => {
    expect(resolveInitialPageState(makeStorage())).toEqual({
      currentPage: PageType.HOME,
      restoredPageFromSession: false,
    });
    expect(resolveInitialPageState(makeStorage({
      [CURRENT_PAGE_STORAGE_KEY]: 'unknown',
    }))).toEqual({
      currentPage: PageType.HOME,
      restoredPageFromSession: false,
    });
  });

  test('persists only valid pages', () => {
    const storage = makeStorage();

    persistCurrentPage(PageType.RECORD, storage);
    persistCurrentPage('unknown', storage);
    persistCurrentPage(LEGACY_BT_MANAGER_PAGE, storage);

    expect(storage.setItem).toHaveBeenCalledTimes(1);
    expect(storage.values[CURRENT_PAGE_STORAGE_KEY]).toBe(PageType.RECORD);
  });

  test('migrates the legacy bt_manager page into the mapless Mission Canvas workspace', () => {
    const storage = makeStorage({
      [CURRENT_PAGE_STORAGE_KEY]: LEGACY_BT_MANAGER_PAGE,
      [AUTONOMY_STUDIO_SESSION_STORAGE_KEY]: JSON.stringify({
        workspaceKind: 'mission',
        workspaceStage: 'authoring',
        mapName: 'factory',
      }),
    });

    expect(PageType).not.toHaveProperty('BT_MANAGER');
    expect(LEGACY_BT_MANAGER_PAGE).toBe('bt_manager');
    expect(resolveInitialPageState(storage)).toEqual({
      currentPage: PageType.AUTONOMY_STUDIO,
      restoredPageFromSession: true,
    });
    expect(storage.values[CURRENT_PAGE_STORAGE_KEY]).toBe(PageType.AUTONOMY_STUDIO);
    expect(JSON.parse(storage.values[AUTONOMY_STUDIO_SESSION_STORAGE_KEY]))
      .toEqual(expect.objectContaining({
        workspaceKind: 'action_canvas',
        workspaceStage: 'authoring',
        mapName: 'factory',
      }));
  });

  test('migrates legacy bt_manager even when its Mission Canvas session is malformed', () => {
    const storage = makeStorage({
      [CURRENT_PAGE_STORAGE_KEY]: LEGACY_BT_MANAGER_PAGE,
      [AUTONOMY_STUDIO_SESSION_STORAGE_KEY]: '{broken json',
    });

    expect(resolveInitialPageState(storage)).toEqual({
      currentPage: PageType.AUTONOMY_STUDIO,
      restoredPageFromSession: true,
    });
    expect(JSON.parse(storage.values[AUTONOMY_STUDIO_SESSION_STORAGE_KEY]))
      .toEqual(expect.objectContaining({
        workspaceKind: 'action_canvas',
      }));
  });

  test('migrates the legacy navigation page into the Mission Canvas Navigate workspace', () => {
    const storage = makeStorage({
      [CURRENT_PAGE_STORAGE_KEY]: LEGACY_NAVIGATION_PAGE,
      [AUTONOMY_STUDIO_SESSION_STORAGE_KEY]: JSON.stringify({
        workspaceKind: 'action_canvas',
        workspaceStage: 'authoring',
        mapName: 'factory',
      }),
    });

    expect(PageType).not.toHaveProperty('NAVIGATION');
    expect(LEGACY_NAVIGATION_PAGE).toBe('navigation');
    expect(resolveInitialPageState(storage)).toEqual({
      currentPage: PageType.AUTONOMY_STUDIO,
      restoredPageFromSession: true,
    });
    expect(storage.values[CURRENT_PAGE_STORAGE_KEY]).toBe(PageType.AUTONOMY_STUDIO);
    expect(JSON.parse(storage.values[AUTONOMY_STUDIO_SESSION_STORAGE_KEY]))
      .toEqual(expect.objectContaining({
        workspaceKind: 'mission',
        workspaceStage: 'navigate',
        mapName: 'factory',
      }));
  });
});

test('migrates the legacy mission_canvas page id to Autonomy Studio and keeps its session', () => {
  const storage = makeStorage({
    [CURRENT_PAGE_STORAGE_KEY]: LEGACY_MISSION_CANVAS_PAGE,
    [LEGACY_MISSION_CANVAS_SESSION_STORAGE_KEY]: JSON.stringify({ workspaceStage: 'run' }),
  });

  expect(LEGACY_MISSION_CANVAS_PAGE).toBe('mission_canvas');
  expect(resolveInitialPageState(storage)).toEqual({
    currentPage: PageType.AUTONOMY_STUDIO,
    restoredPageFromSession: true,
  });
  expect(storage.values[CURRENT_PAGE_STORAGE_KEY]).toBe(PageType.AUTONOMY_STUDIO);
  // The old session blob is left untouched for the page's own fallback read.
  expect(JSON.parse(storage.values[LEGACY_MISSION_CANVAS_SESSION_STORAGE_KEY]))
    .toEqual({ workspaceStage: 'run' });
});
