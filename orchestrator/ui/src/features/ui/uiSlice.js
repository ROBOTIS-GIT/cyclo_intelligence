/*
 * Copyright 2025 ROBOTIS CO., LTD.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Kiwoong Park
 */

import { createSlice } from '@reduxjs/toolkit';

import PageType from '../../constants/pageType';

export const CURRENT_PAGE_STORAGE_KEY = 'cyclo_intelligence.current_page';

const validPages = new Set(Object.values(PageType));

const getSessionStorage = () => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    return window.sessionStorage;
  } catch (_error) {
    return null;
  }
};

export const resolveInitialPageState = (storage = getSessionStorage()) => {
  if (!storage) {
    return {
      currentPage: PageType.HOME,
      restoredPageFromSession: false,
    };
  }
  try {
    const storedPage = storage.getItem(CURRENT_PAGE_STORAGE_KEY);
    if (validPages.has(storedPage)) {
      return {
        currentPage: storedPage,
        restoredPageFromSession: true,
      };
    }
  } catch (_error) {
    // Ignore blocked storage and fall back to Home.
  }
  return {
    currentPage: PageType.HOME,
    restoredPageFromSession: false,
  };
};

export const persistCurrentPage = (page, storage = getSessionStorage()) => {
  if (!storage || !validPages.has(page)) {
    return;
  }
  try {
    storage.setItem(CURRENT_PAGE_STORAGE_KEY, page);
  } catch (_error) {
    // Storage can be disabled in private/browser-restricted contexts.
  }
};

const initialPageState = resolveInitialPageState();

const initialState = {
  isLoading: false,
  error: null,
  currentPage: initialPageState.currentPage,
  restoredPageFromSession: initialPageState.restoredPageFromSession,
  sidebarOpen: false,
  modalOpen: false,
  notifications: [],
  robotTypeList: [],
  isFirstLoad: {
    home: true,
    record: true,
    inference: true,
    btmanager: true,
  },
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
    moveToPage: (state, action) => {
      state.currentPage = action.payload;
    },
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen: (state, action) => {
      state.sidebarOpen = action.payload;
    },
    setModalOpen: (state, action) => {
      state.modalOpen = action.payload;
    },
    addNotification: (state, action) => {
      state.notifications.push({
        id: Date.now(),
        ...action.payload,
      });
    },
    removeNotification: (state, action) => {
      state.notifications = state.notifications.filter(
        (notification) => notification.id !== action.payload
      );
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    setRobotTypeList: (state, action) => {
      state.robotTypeList = action.payload;
    },
    setIsFirstLoadFalse: (state, action) => {
      state.isFirstLoad[action.payload] = false;
    },
    setIsFirstLoadTrue: (state, action) => {
      state.isFirstLoad[action.payload] = true;
    },
  },
});

export const {
  setLoading,
  setError,
  clearError,
  moveToPage,
  toggleSidebar,
  setSidebarOpen,
  setModalOpen,
  addNotification,
  removeNotification,
  clearNotifications,
  setRobotTypeList,
  setIsFirstLoadFalse,
  setIsFirstLoadTrue,
} = uiSlice.actions;

export default uiSlice.reducer;
