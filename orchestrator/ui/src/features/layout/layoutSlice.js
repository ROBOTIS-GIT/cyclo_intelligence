import { createSlice } from '@reduxjs/toolkit';

// Panel IDs
export const PANEL_IDS = {
  CAMERAS: 'cameras',
  VIEWER_3D: 'viewer3d',
  JOINT_DATA: 'jointData',
  SIDEBAR: 'sidebar',
};

// Default layout: 12-column x 3-row grid
// Cameras left strip (col 0-2, row 0-1), 3D Viewer center (col 3-8, row 0-1),
// Sidebar right (col 9-11, row 0-2), Joint Data bottom (col 0-8, row 2)
const DEFAULT_PANELS = {
  [PANEL_IDS.CAMERAS]: {
    id: PANEL_IDS.CAMERAS,
    title: 'Cameras',
    gridColumn: '1 / 5',
    gridRow: '1 / 3',
    visible: true,
    expanded: false,
    // Expanded: takes cols 1-7
    expandedGridColumn: '1 / 7',
    expandedGridRow: '1 / 3',
  },
  [PANEL_IDS.VIEWER_3D]: {
    id: PANEL_IDS.VIEWER_3D,
    title: '3D Viewer',
    gridColumn: '5 / 10',
    gridRow: '1 / 3',
    visible: true,
    expanded: false,
    expandedGridColumn: '1 / 10',
    expandedGridRow: '1 / 3',
  },
  [PANEL_IDS.JOINT_DATA]: {
    id: PANEL_IDS.JOINT_DATA,
    title: 'Joint Data',
    gridColumn: '1 / 10',
    gridRow: '3 / 4',
    visible: true,
    expanded: false,
    expandedGridColumn: '1 / 13',
    expandedGridRow: '1 / 4',
  },
  [PANEL_IDS.SIDEBAR]: {
    id: PANEL_IDS.SIDEBAR,
    title: 'Sidebar',
    gridColumn: '10 / 13',
    gridRow: '1 / 4',
    visible: true,
    expanded: false,
    expandedGridColumn: '7 / 13',
    expandedGridRow: '1 / 4',
  },
};

const STORAGE_KEY = 'replayPanelLayout';

// Load layout from localStorage
function loadFromStorage() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate structure
      if (parsed.panels && typeof parsed.panels === 'object') {
        // Merge with defaults to handle new panels added in updates
        const merged = { ...DEFAULT_PANELS };
        Object.keys(parsed.panels).forEach((key) => {
          if (merged[key]) {
            merged[key] = { ...merged[key], ...parsed.panels[key] };
          }
        });
        return merged;
      }
    }
  } catch {
    // Ignore parse errors
  }
  return null;
}

// Save layout to localStorage
function saveToStorage(panels) {
  try {
    // Only save position/visibility state, not expanded state
    const toSave = {};
    Object.keys(panels).forEach((key) => {
      const p = panels[key];
      toSave[key] = {
        gridColumn: p.gridColumn,
        gridRow: p.gridRow,
        visible: p.visible,
      };
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ panels: toSave }));
  } catch {
    // Ignore storage errors
  }
}

const initialPanels = loadFromStorage() || { ...DEFAULT_PANELS };

const layoutSlice = createSlice({
  name: 'layout',
  initialState: {
    panels: initialPanels,
  },
  reducers: {
    swapPanels: (state, action) => {
      const { sourceId, targetId } = action.payload;
      const source = state.panels[sourceId];
      const target = state.panels[targetId];
      if (!source || !target) return;

      // Swap grid positions
      const tempCol = source.gridColumn;
      const tempRow = source.gridRow;
      const tempExpCol = source.expandedGridColumn;
      const tempExpRow = source.expandedGridRow;

      source.gridColumn = target.gridColumn;
      source.gridRow = target.gridRow;
      source.expandedGridColumn = target.expandedGridColumn;
      source.expandedGridRow = target.expandedGridRow;

      target.gridColumn = tempCol;
      target.gridRow = tempRow;
      target.expandedGridColumn = tempExpCol;
      target.expandedGridRow = tempExpRow;

      saveToStorage(state.panels);
    },

    togglePanelVisibility: (state, action) => {
      const panelId = action.payload;
      const panel = state.panels[panelId];
      if (!panel) return;
      panel.visible = !panel.visible;
      if (!panel.visible) {
        panel.expanded = false;
      }
      saveToStorage(state.panels);
    },

    showPanel: (state, action) => {
      const panelId = action.payload;
      const panel = state.panels[panelId];
      if (!panel) return;
      panel.visible = true;
      saveToStorage(state.panels);
    },

    togglePanelExpanded: (state, action) => {
      const panelId = action.payload;
      const panel = state.panels[panelId];
      if (!panel) return;

      const wasExpanded = panel.expanded;

      // Collapse all first
      Object.values(state.panels).forEach((p) => {
        p.expanded = false;
      });

      // Toggle the target
      panel.expanded = !wasExpanded;
    },

    resetToDefaultLayout: (state) => {
      state.panels = { ...DEFAULT_PANELS };
      // Deep copy to avoid mutation
      Object.keys(DEFAULT_PANELS).forEach((key) => {
        state.panels[key] = { ...DEFAULT_PANELS[key] };
      });
      saveToStorage(state.panels);
    },

  },
});

export const {
  swapPanels,
  togglePanelVisibility,
  showPanel,
  togglePanelExpanded,
  resetToDefaultLayout,
} = layoutSlice.actions;

export default layoutSlice.reducer;
