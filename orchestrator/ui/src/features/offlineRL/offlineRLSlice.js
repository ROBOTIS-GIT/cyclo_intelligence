// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  datasetPath: '',
  datasetVersion: '',
  datasetSelections: [],
  checkpointPath: '',
  replayBufferPath: '',
  conversionDestinationPath: '/workspace/lerobot',
  conversionFps: 15,
  conversionFormats: {
    v21: true,
    v30: true,
  },
  convertedDatasetPaths: {
    v21: '',
    v30: '',
  },
};

const offlineRLSlice = createSlice({
  name: 'offlineRL',
  initialState,
  reducers: {
    setOfflineRLDatasetPath: (state, action) => {
      const path = String(action.payload || '');
      state.datasetPath = path;
      state.datasetVersion = '';
      state.datasetSelections = path ? [{
        path, version: '', dataEpoch: null, dataEpochProvenance: null,
      }] : [];
    },
    setOfflineRLDatasetSelection: (state, action) => {
      const path = String(action.payload?.path || '');
      const version = String(action.payload?.version || '');
      const rawEpoch = action.payload?.dataEpoch == null
        ? Number.NaN
        : Number(action.payload.dataEpoch);
      const dataEpoch = Number.isInteger(rawEpoch) && rawEpoch >= 0 ? rawEpoch : null;
      const dataEpochProvenance = action.payload?.dataEpochProvenance || null;
      state.datasetPath = path;
      state.datasetVersion = version;
      if (path) {
        const existing = state.datasetSelections.findIndex((item) => item.path === path);
        const next = { path, version, dataEpoch, dataEpochProvenance };
        if (existing >= 0) state.datasetSelections[existing] = next;
        else state.datasetSelections.push(next);
      }
    },
    setOfflineRLDatasetPreview: (state, action) => {
      state.datasetPath = String(action.payload?.path || '');
      state.datasetVersion = String(action.payload?.version || '');
    },
    setOfflineRLDatasetSelections: (state, action) => {
      const seen = new Set();
      state.datasetSelections = (Array.isArray(action.payload) ? action.payload : [])
        .map((item) => {
          const path = String(item?.path || '');
          const version = String(item?.version || '');
          const rawEpoch = item?.dataEpoch == null ? Number.NaN : Number(item.dataEpoch);
          return {
            path,
            version,
            dataEpoch: Number.isInteger(rawEpoch) && rawEpoch >= 0 ? rawEpoch : null,
            dataEpochProvenance: item?.dataEpochProvenance || null,
          };
        })
        .filter((item) => {
          if (!item.path || seen.has(item.path)) return false;
          seen.add(item.path);
          return true;
        });
    },
    setOfflineRLCheckpointPath: (state, action) => {
      state.checkpointPath = String(action.payload || '');
    },
    setOfflineRLReplayBufferPath: (state, action) => {
      state.replayBufferPath = String(action.payload || '');
    },
    setOfflineRLConversionDestinationPath: (state, action) => {
      state.conversionDestinationPath = String(action.payload || '');
    },
    setOfflineRLConversionFps: (state, action) => {
      const fps = Number(action.payload);
      state.conversionFps = Number.isFinite(fps) ? fps : 15;
    },
    setOfflineRLConversionFormats: (state, action) => {
      state.conversionFormats = {
        ...state.conversionFormats,
        ...(action.payload || {}),
      };
    },
    setOfflineRLConvertedDatasetPaths: (state, action) => {
      state.convertedDatasetPaths = {
        ...state.convertedDatasetPaths,
        ...(action.payload || {}),
      };
    },
  },
});

export const selectOfflineRLDatasetPath = (state) => (
  state.offlineRL?.datasetPath || ''
);

export const selectOfflineRLDatasetVersion = (state) => (
  state.offlineRL?.datasetVersion || ''
);

export const selectOfflineRLDatasetSelections = (state) => {
  const selections = state.offlineRL?.datasetSelections;
  if (Array.isArray(selections)) return selections;
  const path = state.offlineRL?.datasetPath || '';
  return path ? [{
    path,
    version: state.offlineRL?.datasetVersion || '',
    dataEpoch: null,
    dataEpochProvenance: null,
  }] : [];
};

export const selectOfflineRLDatasetPaths = (state) => (
  selectOfflineRLDatasetSelections(state).map((item) => item.path)
);

export const selectOfflineRLCheckpointPath = (state) => (
  state.offlineRL?.checkpointPath || ''
);

export const selectOfflineRLReplayBufferPath = (state) => (
  state.offlineRL?.replayBufferPath || ''
);

export const selectOfflineRLConversionDestinationPath = (state) => (
  state.offlineRL?.conversionDestinationPath ?? '/workspace/lerobot'
);

export const selectOfflineRLConversionFps = (state) => (
  Number(state.offlineRL?.conversionFps || 15)
);

export const selectOfflineRLConversionFormats = (state) => (
  state.offlineRL?.conversionFormats || { v21: true, v30: true }
);

export const selectOfflineRLConvertedDatasetPaths = (state) => (
  state.offlineRL?.convertedDatasetPaths || { v21: '', v30: '' }
);

export const {
  setOfflineRLDatasetPath,
  setOfflineRLDatasetSelection,
  setOfflineRLDatasetPreview,
  setOfflineRLDatasetSelections,
  setOfflineRLCheckpointPath,
  setOfflineRLReplayBufferPath,
  setOfflineRLConversionDestinationPath,
  setOfflineRLConversionFps,
  setOfflineRLConversionFormats,
  setOfflineRLConvertedDatasetPaths,
} = offlineRLSlice.actions;

export default offlineRLSlice.reducer;
