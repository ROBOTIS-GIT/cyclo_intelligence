// Copyright 2025 ROBOTIS CO., LTD.
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
// Author: Kiwoong Park

import React, { useState, useEffect, useLayoutEffect, useCallback, useRef } from 'react';
import { useSelector, useDispatch, useStore } from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { MdAdd, MdRefresh, MdGridView } from 'react-icons/md';
import { requestSupervisorApi } from '../features/missionCanvas/lib/btNodeApi';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import ImageGridCell from './ImageGridCell';
import ImageTopicSelectModal from './ImageTopicSelectModal';
import { setImageTopicList, setAssignedImageTopics } from '../features/ros/rosSlice';
import { readCameraPreferences, writeCameraPreferences } from '../utils/cameraPreferences';
import './ImageGrid.css';

export const getGridColumnCount = (count, width, height = 0, ratios = []) => {
  const maxColumns = Math.min(Math.max(1, count), Math.max(1, Math.floor((width + 8) / (240 + 8))));
  if (height > 0 && width > 0 && count > 0) {
    let bestColumns = 1;
    let bestArea = -1;
    for (let columns = 1; columns <= maxColumns; columns += 1) {
      const rows = Math.ceil(count / columns);
      const tileHeight = Math.max(200, (height - 8 * (rows - 1)) / rows);
      const imageHeight = tileHeight - 36; // Only the camera toolbar takes up a row.
      const tileWidth = (width - 8 * (columns - 1)) / columns;
      const area = Array.from({ length: count }, (_, i) => {
        const ratio = ratios[i] || 4 / 3;
        const w = count > columns && count % columns === 1 && i === count - 1 ? width : tileWidth;
        const fittedWidth = Math.min(w, imageHeight * ratio);
        return fittedWidth * fittedWidth / ratio;
      }).reduce((sum, value) => sum + value, 0);
      // Compare visible image area, not oversized layouts below the scroll fold.
      const visibleArea = area * Math.min(1, height / (rows * tileHeight + (rows - 1) * 8));
      if (visibleArea > bestArea) {
        bestArea = visibleArea;
        bestColumns = columns;
      }
    }
    return bestColumns;
  }
  const rows = Math.max(1, Math.ceil(count / maxColumns));
  return Math.max(1, Math.ceil(count / rows));
};

export const normalizeRotationDeg = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return ((Math.round(numeric) % 360) + 360) % 360;
};

export const buildTopicRotationMap = (imageTopics, rotationDegList) => {
  const topics = Array.isArray(imageTopics) ? imageTopics : [];
  const rotations = Array.isArray(rotationDegList) ? rotationDegList : [];
  return topics.reduce((acc, topic, idx) => {
    if (topic) {
      acc[topic] = normalizeRotationDeg(rotations[idx] ?? 0);
    }
    return acc;
  }, {});
};

const getRotationForTopic = (topic, topicRotationMap) => (
  normalizeRotationDeg(topicRotationMap?.[topic] ?? 0)
);

const getSavedAssignmentForRobot = (state, robotType) => {
  const normalizedRobotType = String(robotType || '').trim();
  if (!normalizedRobotType) return null;
  const savedRobotType = String(state.ros.assignedImageTopicsRobotType || '').trim();
  if (savedRobotType !== normalizedRobotType) return null;
  const saved = state.ros.assignedImageTopics;
  return Array.isArray(saved) ? [...new Set(saved.filter(Boolean))] : null;
};

export default function ImageGrid({ isActive = true, responsive = false, isRecording = false, headerActions }) {
  const robotType = useSelector((state) => String(state.tasks.robotType || '').trim());
  // A robot switch owns a fresh request/state lifetime, including its streams and modal.
  return <RobotImageGrid key={robotType} robotType={robotType} isActive={isActive} responsive={responsive} isRecording={isRecording} headerActions={headerActions} />;
}

function RobotImageGrid({ robotType, isActive, responsive, isRecording, headerActions }) {
  const dispatch = useDispatch();
  const store = useStore();
  const imageTopicList = useSelector((state) => state.ros.imageTopicList);
  const [modalOpen, setModalOpen] = useState(false);
  const [isLoadingTopics, setIsLoadingTopics] = useState(false);
  const [topicListError, setTopicListError] = useState(null);
  const [savedPreferences] = useState(() => readCameraPreferences(robotType));
  // Read persistence once, not through a selector: subscribing to our own writes
  // previously caused assignment/reset feedback loops on page transitions.
  const [assignedTopics, setAssignedTopics] = useState(() => (
    savedPreferences?.topics ?? getSavedAssignmentForRobot(store.getState(), robotType)
  ));
  const [topicRotationMap, setTopicRotationMap] = useState({});
  // Rotation belongs to the camera, so moving its tile cannot change its orientation.
  const [rotationOverrides, setRotationOverrides] = useState(() => savedPreferences?.rotations || {});
  const [imageSizes, setImageSizes] = useState({});
  const requestId = useRef(0);
  const gridRef = useRef(null);
  const [gridSize, setGridSize] = useState({ width: 0, height: 0 });
  const dragRef = useRef(null);
  const [dropTopic, setDropTopic] = useState(null);
  const [topicLabels, setTopicLabels] = useState({});
  const [tileSizes, setTileSizes] = useState(null);
  const resizeRef = useRef(null);
  const visibleTopics = (assignedTopics || []).filter(Boolean);
  const hiddenTopics = imageTopicList.filter((topic) => !visibleTopics.includes(topic));
  const hasCells = visibleTopics.length > 0;

  useEffect(() => {
    if (!robotType) return undefined;
    const controller = new AbortController();
    requestSupervisorApi(`/robots/${encodeURIComponent(robotType)}/topic-keys`, { signal: controller.signal })
      .then((metadata) => {
        if (controller.signal.aborted) return;
        const labels = {};
        Object.entries(metadata?.topic_roles || {}).forEach(([topic, roles]) => {
          const keys = roles?.observation;
          if (Array.isArray(keys) && keys.length) labels[topic] = keys.join(', ');
        });
        setTopicLabels(labels);
      }).catch(() => {}); // Metadata failure must not interrupt camera streams.
    return () => controller.abort();
  }, [robotType]);

  const snapshotSizes = () => Object.fromEntries(
    [...gridRef.current.querySelectorAll('[data-camera-topic]')].map((tile) => {
      const { width, height } = tile.getBoundingClientRect();
      return [tile.dataset.cameraTopic, { width, height }];
    })
  );
  const resizeTile = (topic, width, height, initial) => {
    const available = gridRef.current.clientWidth;
    setTileSizes((old) => ({ ...(old || initial), [topic]: {
      width: Math.max(Math.min(240, available), width),
      height: Math.max(200, height),
    } }));
  };
  const resizeHandlers = (topic) => ({
    onPointerDown: (event) => {
      if (event.button !== 0) return;
      const sizes = snapshotSizes();
      resizeRef.current = {
        topic, sizes, x: event.clientX, y: event.clientY,
        scrollLeft: gridRef.current.scrollLeft, scrollTop: gridRef.current.scrollTop,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
      event.preventDefault();
    },
    onPointerMove: (event) => {
      const drag = resizeRef.current;
      if (!drag || drag.topic !== topic) return;
      const size = drag.sizes[topic];
      resizeTile(topic,
        size.width + event.clientX - drag.x + gridRef.current.scrollLeft - drag.scrollLeft,
        size.height + event.clientY - drag.y + gridRef.current.scrollTop - drag.scrollTop,
        drag.sizes);
    },
    onPointerUp: (event) => {
      resizeRef.current = null;
      event.currentTarget.releasePointerCapture(event.pointerId);
    },
    onPointerCancel: () => { resizeRef.current = null; },
    onLostPointerCapture: () => { resizeRef.current = null; },
    onKeyDown: (event) => {
      const delta = { ArrowLeft: [-20, 0], ArrowRight: [20, 0], ArrowUp: [0, -20], ArrowDown: [0, 20] }[event.key];
      if (!delta) return;
      event.preventDefault();
      const sizes = snapshotSizes();
      resizeTile(topic, sizes[topic].width + delta[0], sizes[topic].height + delta[1], sizes);
    },
  });

  useLayoutEffect(() => {
    const grid = gridRef.current;
    const measure = ({ width, height }) => setGridSize((old) => (
      old.width === width && old.height === height ? old : { width, height }
    ));
    measure(grid.getBoundingClientRect());
    const observer = new ResizeObserver(([entry]) => measure(entry.contentRect));
    observer.observe(grid);
    return () => observer.disconnect();
  }, []);

  const { getImageTopicList } = useRosServiceCaller();

  const getCellRotation = (topic) => rotationOverrides[topic] ?? getRotationForTopic(topic, topicRotationMap);

  const handleRotateClick = (idx) => {
    const topic = visibleTopics[idx];
    const configured = getRotationForTopic(topic, topicRotationMap);
    setRotationOverrides((prev) => ({
      ...prev,
      [topic]: normalizeRotationDeg((prev[topic] ?? configured) + 90),
    }));
  };

  useEffect(() => {
    writeCameraPreferences(robotType, assignedTopics, rotationOverrides);
  }, [robotType, assignedTopics, rotationOverrides]);

  // Persist topic assignment to Redux so it survives remounts (page swap).
  // Compare against the latest store value via getState() so we never
  // re-trigger this effect from our own dispatch.
  useEffect(() => {
    if (assignedTopics === null || !robotType) return;
    const current = store.getState().ros.assignedImageTopics;
    const currentRobotType = store.getState().ros.assignedImageTopicsRobotType || '';
    const normalizedRobotType = String(robotType || '').trim();
    const same =
      Array.isArray(current) &&
      current.length === assignedTopics.length &&
      assignedTopics.every((t, i) => t === current[i]);
    if (!same || currentRobotType !== normalizedRobotType) {
      dispatch(setAssignedImageTopics({
        robotType: normalizedRobotType,
        topics: assignedTopics,
      }));
    }
  }, [assignedTopics, dispatch, robotType, store]);

  const fetchTopics = useCallback(async () => {
    const id = ++requestId.current;
    setIsLoadingTopics(true);
    setTopicListError(null);
    try {
      const result = await getImageTopicList();
      if (id !== requestId.current) return;
      if (result && result.success) {
        const imageTopics = Array.isArray(result.image_topic_list) ? result.image_topic_list : [];
        setTopicRotationMap(buildTopicRotationMap(
          imageTopics,
          result.rotation_deg_list || []
        ));
        dispatch(setImageTopicList(imageTopics));
        setAssignedTopics((current) => current === null
          ? imageTopics
          : [...new Set(current.filter((topic) => topic && imageTopics.includes(topic)))]);
        setTopicListError(null);
        toast.success(`Loaded ${imageTopics.length} image topics`);
      } else {
        const errorMsg = result?.message || 'Unknown error occurred';
        setTopicListError(`Service error: ${errorMsg}`);
        dispatch(setImageTopicList([]));
        toast.error(`Failed to load image topics: ${errorMsg}`);
      }
    } catch (error) {
      if (id !== requestId.current) return;
      setTopicListError('Failed to load image topic list');
      dispatch(setImageTopicList([]));
      toast.error('Failed to load image topic list');
    } finally {
      if (id === requestId.current) setIsLoadingTopics(false);
    }
  }, [getImageTopicList, dispatch]);

  useEffect(() => {
    fetchTopics();
    // Invalidates automatic loads AND modal refreshes on replacement/unmount.
    return () => { requestId.current += 1; };
  }, [fetchTopics]);

  const handleTopicSelect = (topic) => {
    if (!imageTopicList.includes(topic)) return;
    setTileSizes(null);
    setAssignedTopics((current) => [...new Set([...(current || []).filter(Boolean), topic])]);
    setModalOpen(false);
  };

  const moveCamera = (topic, target) => setAssignedTopics((current) => {
    const next = (current || []).filter(Boolean);
    const from = next.indexOf(topic), to = next.indexOf(target);
    if (from < 0 || to < 0 || from === to) return current;
    next.splice(from, 1);
    next.splice(to, 0, topic);
    return next;
  });

  const dragHandlers = (topic) => ({
    onPointerDown: (event) => {
      if (event.button !== 0) return;
      dragRef.current = { topic, x: event.clientX, y: event.clientY, target: null };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    onPointerMove: (event) => {
      const drag = dragRef.current;
      if (!drag || Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 6) return;
      const target = document.elementFromPoint(event.clientX, event.clientY)?.closest('[data-camera-topic]');
      drag.target = target && gridRef.current.contains(target) ? target.dataset.cameraTopic : null;
      setDropTopic(drag.target);
    },
    onPointerUp: (event) => {
      const drag = dragRef.current;
      if (drag?.target) moveCamera(drag.topic, drag.target);
      dragRef.current = null;
      setDropTopic(null);
      event.currentTarget.releasePointerCapture(event.pointerId);
    },
    onPointerCancel: () => { dragRef.current = null; setDropTopic(null); },
    onKeyDown: (event) => {
      const index = visibleTopics.indexOf(topic);
      const delta = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }[event.key];
      if (delta === undefined) return;
      event.preventDefault();
      moveCamera(topic, visibleTopics[index + delta]);
    },
  });

  const handleImageSize = useCallback((topic, width, height) => {
    if (!width || !height) return;
    setImageSizes((old) => old[topic]?.width === width && old[topic]?.height === height
      ? old : { ...old, [topic]: { width, height } });
  }, []);
  const ratios = visibleTopics.map((topic) => {
    const { width, height } = imageSizes[topic] || { width: 4, height: 3 };
    const r = getCellRotation(topic) * Math.PI / 180;
    return (width * Math.abs(Math.cos(r)) + height * Math.abs(Math.sin(r))) /
      (height * Math.abs(Math.cos(r)) + width * Math.abs(Math.sin(r)));
  });
  const columns = getGridColumnCount(visibleTopics.length, gridSize.width, gridSize.height, ratios);

  return (
    <>
      <div role="region" aria-label="Camera feeds" className={clsx('image-grid-region', { 'image-grid-responsive': responsive, 'image-grid-recording': isRecording })}>
        <div className="image-grid-header border-b border-gray-300 bg-gray-50">
          <div className="image-grid-heading">
          <span className="text-sm font-semibold min-w-0 truncate">Cameras <span className="text-gray-500 font-normal">{visibleTopics.length}/{imageTopicList.length}</span></span>
          <span role="status" aria-label="Camera recording status">
            {isRecording && <span className="image-grid-recording-badge"><span className="image-grid-recording-dot" aria-hidden="true" />Recording</span>}
          </span>
          </div>
          <div className="flex items-center shrink-0">
          {headerActions}
          <button type="button" className="image-grid-tool" title="Reset camera layout" aria-label="Reset camera layout"
            disabled={!tileSizes} onClick={() => setTileSizes(null)}><MdGridView size={20} /></button>
          <button type="button" className="image-grid-tool" title="Show hidden camera" aria-label="Show hidden camera"
            disabled={!hiddenTopics.length} onClick={() => setModalOpen(true)}><MdAdd size={20} /></button>
          </div>
        </div>
        <div ref={gridRef} className="image-grid-scroll">
        {(!hasCells || topicListError) && (
          <div className="image-grid-status" role="status">
            <span>{isLoadingTopics ? 'Loading cameras...' : (topicListError || (imageTopicList.length ? 'All cameras hidden' : 'No cameras available'))}</span>
            <button
              type="button"
              className="image-grid-tool"
              title="Refresh cameras"
              aria-label="Refresh cameras"
              disabled={isLoadingTopics}
              onClick={() => fetchTopics()}
            >
              <MdRefresh size={20} />
            </button>
          </div>
        )}
        {hasCells && (
          <div
            className={clsx('image-grid-layout', { 'image-grid-manual': tileSizes })}
            style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
          >
            {visibleTopics.map((topic, idx) => (
              <ImageGridCell
                key={topic}
                topic={topic}
                label={topicLabels[topic]}
                rotationDegrees={getCellRotation(topic)}
                onRotateClick={handleRotateClick}
                idx={idx}
                onClose={() => {
                  setTileSizes(null);
                  setAssignedTopics((current) => current.filter((value) => value && value !== topic));
                }}
                onImageSize={handleImageSize}
                dragHandleProps={dragHandlers(topic)}
                resizeHandleProps={resizeHandlers(topic)}
                dropTarget={dropTopic === topic}
                style={tileSizes?.[topic] || (visibleTopics.length > columns && visibleTopics.length % columns === 1 && idx === visibleTopics.length - 1 ? { gridColumn: '1 / -1' } : undefined)}
                isActive={isActive}
              />
            ))}
          </div>
        )}
        </div>
      </div>
      {modalOpen && (
        <ImageTopicSelectModal
          topicList={hiddenTopics}
          topicLabels={topicLabels}
          onSelect={handleTopicSelect}
          onClose={() => setModalOpen(false)}
          isLoading={isLoadingTopics}
          onRefresh={fetchTopics}
          errorMessage={topicListError}
        />
      )}
    </>
  );
}
