import PageType from '../../constants/pageType';
import {
  CURRENT_PAGE_STORAGE_KEY,
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

    expect(storage.setItem).toHaveBeenCalledTimes(1);
    expect(storage.values[CURRENT_PAGE_STORAGE_KEY]).toBe(PageType.RECORD);
  });
});
