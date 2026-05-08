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
// Author: Dongyun Kim

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { MdFolder, MdClose, MdRefresh, MdDashboard } from 'react-icons/md';
import toast from 'react-hot-toast';
import clsx from 'clsx';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import FileBrowserModal from '../components/FileBrowserModal';
import {
  lttbDownsample,
  formatTime,
} from '../utils/chartUtils';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import {
  setSelectedBagPath,
  setLoading,
  setReplayData,
  setTaskMarkers,
  setError,
  setCurrentTime,
  setIsPlaying,
  setIsVideoLoaded,
  setVideoLoadProgress,
} from '../features/replay/replaySlice';
import {
  resetToDefaultLayout,
  showPanel,
} from '../features/layout/layoutSlice';
import { useMcapFramePlayer } from '../hooks/useMcapFramePlayer';

// Layout components
import ReplayLayoutContainer from '../components/layout/ReplayLayoutContainer';

// Panel components
import CameraPanel from '../components/replay/CameraPanel';
import Viewer3DPanel from '../components/replay/Viewer3DPanel';
import JointDataPanel from '../components/replay/JointDataPanel';
import SidebarPanel from '../components/replay/SidebarPanel';
import TimelineControls from '../components/replay/TimelineControls';

function ReplayPage({ isActive }) {
  const dispatch = useDispatch();
  const { getReplayData, getRosbagList } = useRosServiceCaller();

  // Redux state
  const {
    selectedBagPath,
    isLoading,
    isLoaded,
    error,
    videoFiles,
    videoNames,
    videoFps,
    bagPath,
    jointTimestamps,
    jointNames,
    jointPositions,
    actionTimestamps,
    actionNames,
    actionValues,
    duration,
    currentTime,
    isPlaying,
    isVideoLoaded,
    // eslint-disable-next-line no-unused-vars
    videoLoadProgress,
    // Extended metadata
    robotType,
    recordingDate,
    fileSizeBytes,
    taskMarkers,
    frameCounts,
    // MCAP direct streaming
    hasRawImages,
    mcapFile,
  } = useSelector((state) => state.replay);

  const rosHost = useSelector((state) => state.ros.rosHost);
  const videoServerPort = useSelector((state) => state.replay.videoServerPort);

  // Layout state
  const layoutPanels = useSelector((state) => state.layout.panels);

  // Local state
  const [showFileBrowser, setShowFileBrowser] = useState(false);
  const [expandedJoints, setExpandedJoints] = useState(new Set());
  const [videoBlobUrls, setVideoBlobUrls] = useState([]);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [isDownloading, setIsDownloading] = useState(false);
  const [expandedVideoIndex, setExpandedVideoIndex] = useState(null);
  const [rosbagList, setRosbagList] = useState([]);
  const [currentBagIndex, setCurrentBagIndex] = useState(-1);
  const [parentFolderPath, setParentFolderPath] = useState('');
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // WebGL rendering mode
  const [useWebGL, setUseWebGL] = useState(true);
  const [videoBrightness, setVideoBrightness] = useState(0);
  const [videoContrast, setVideoContrast] = useState(1);

  // A-B Loop state
  const [loopStart, setLoopStart] = useState(null);
  const [loopEnd, setLoopEnd] = useState(null);

  // MCAP direct streaming mode detection
  const isDirectMcapMode = isLoaded && hasRawImages && videoFiles.length === 0;

  // MCAP frame player hook — serve via the video file server (port 8082),
  // not nginx, because only the video server has Range support + access to
  // the rosbag2 filesystem.
  const mcapUrl = isDirectMcapMode && bagPath && mcapFile && rosHost
    ? `http://${rosHost}:${videoServerPort || 8082}${bagPath}/${mcapFile}`
    : null;

  const mcapPlayer = useMcapFramePlayer({
    mcapUrl,
    duration,
    currentTime,
    isPlaying,
    playbackSpeed,
    loopStart,
    loopEnd,
    isActive: isDirectMcapMode,
  });

  // Task Marker state
  const [showMarkerDialog, setShowMarkerDialog] = useState(false);
  const [pendingMarkerTime, setPendingMarkerTime] = useState(null);
  const [markerInput, setMarkerInput] = useState('');
  const [isSavingMarkers, setIsSavingMarkers] = useState(false);
  const [instructionPalette, setInstructionPalette] = useState(() => {
    try {
      const saved = localStorage.getItem('taskInstructionPalette');
      if (saved) {
        const parsed = JSON.parse(saved);
        return parsed.instructions || [];
      }
    } catch {
      // Ignore parse errors
    }
    return [
      'Pick up the object',
      'Move to target position',
      'Place the object',
      'Open gripper',
      'Close gripper',
    ];
  });
  const [newPaletteInput, setNewPaletteInput] = useState('');
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [trimStart, setTrimStart] = useState(null);
  const [trimEnd, setTrimEnd] = useState(null);
  const [showTrimStartDialog, setShowTrimStartDialog] = useState(false);
  const [trimStartInstruction, setTrimStartInstruction] = useState('');
  const [excludeRegions, setExcludeRegions] = useState([]);
  const [pendingExcludeStart, setPendingExcludeStart] = useState(null);

  // Hidden panels dropdown
  const [showPanelMenu, setShowPanelMenu] = useState(false);

  const PLAYBACK_SPEEDS = [0.5, 1, 1.5, 2, 3];

  // Refs
  const videoRefs = useRef([]);
  const videoCacheRef = useRef(new Map());
  const markersCacheRef = useRef(new Map());
  const isLoadingBagRef = useRef(false); // Guard against auto-save during bag load
  const autoSaveTimerRef = useRef(null);
  const wasPlayingBeforeTrimRef = useRef(false); // Resume playback after S/E dialog

  // Calculate video URL for a given index
  const getVideoUrl = useCallback(
    (index) => {
      if (!bagPath || !videoFiles.length) return null;
      const videoFile = videoFiles[index];
      if (!videoFile) return null;
      return `/files${bagPath}/${videoFile}`;
    },
    [bagPath, videoFiles]
  );

  // Download all videos as blobs for smooth playback (with caching)
  const downloadVideos = useCallback(async () => {
    if (!videoFiles.length || !bagPath) return;

    const cachedUrls = videoCacheRef.current.get(bagPath);
    if (cachedUrls && cachedUrls.length === videoFiles.length) {
      setVideoBlobUrls(cachedUrls);
      dispatch(setIsVideoLoaded(true));
      dispatch(setVideoLoadProgress(100));
      toast.success('Videos loaded from cache');
      return;
    }

    setIsDownloading(true);
    setDownloadProgress(0);

    const totalVideos = videoFiles.length;
    const progressPerVideo = 100 / totalVideos;
    const newBlobUrls = [];

    try {
      for (let i = 0; i < totalVideos; i++) {
        const url = getVideoUrl(i);
        if (!url) continue;

        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to download video ${i + 1}`);
        }

        const contentLength = response.headers.get('content-length');
        const total = parseInt(contentLength, 10) || 0;
        const reader = response.body.getReader();
        const chunks = [];
        let received = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          chunks.push(value);
          received += value.length;

          if (total > 0) {
            const videoProgress = (received / total) * progressPerVideo;
            setDownloadProgress(Math.round(i * progressPerVideo + videoProgress));
          }
        }

        const blob = new Blob(chunks, { type: 'video/mp4' });
        const blobUrl = URL.createObjectURL(blob);
        newBlobUrls.push(blobUrl);

        setDownloadProgress(Math.round((i + 1) * progressPerVideo));
      }

      videoCacheRef.current.set(bagPath, newBlobUrls);

      setVideoBlobUrls(newBlobUrls);
      dispatch(setIsVideoLoaded(true));
      dispatch(setVideoLoadProgress(100));
      toast.success('Videos downloaded successfully');
    } catch (error) {
      console.error('Video download failed:', error);
      toast.error(`Failed to download videos: ${error.message}`);
      dispatch(setError(error.message));
    } finally {
      setIsDownloading(false);
    }
  }, [videoFiles, bagPath, getVideoUrl, dispatch]);

  // Start downloading videos when replay data is loaded
  useEffect(() => {
    if (isLoaded && videoFiles.length > 0 && videoBlobUrls.length === 0 && !isDownloading) {
      downloadVideos();
    }
  }, [isLoaded, videoFiles.length, videoBlobUrls.length, isDownloading, downloadVideos]);

  // Cleanup all cached blob URLs on unmount only
  useEffect(() => {
    const cache = videoCacheRef.current;
    return () => {
      cache.forEach((urls) => {
        urls.forEach((url) => {
          if (url) URL.revokeObjectURL(url);
        });
      });
      cache.clear();
    };
  }, []);

  // Get all unique joint names from both state and action
  const allJointNames = useMemo(() => {
    const names = new Set([...jointNames, ...actionNames]);
    return Array.from(names);
  }, [jointNames, actionNames]);

  const hasActionData = actionTimestamps.length > 0 && actionNames.length > 0;

  // Auto-expand all joints when data loads
  useEffect(() => {
    if (allJointNames.length > 0) {
      setExpandedJoints(new Set(allJointNames));
    }
  }, [allJointNames]);

  // Prepare state chart data
  const stateChartData = useMemo(() => {
    if (!jointTimestamps.length || !jointNames.length || !jointPositions.length) {
      return [];
    }

    const numJoints = jointNames.length;
    const targetPoints = 1000;

    const fullData = jointTimestamps.map((time, i) => {
      const point = { time };
      const startIdx = i * numJoints;
      jointNames.forEach((name, j) => {
        point[`state_${name}`] = jointPositions[startIdx + j] || 0;
      });
      return point;
    });

    if (fullData.length <= targetPoints) {
      return fullData;
    }

    const firstJointKey = `state_${jointNames[0]}`;
    const sampledData = lttbDownsample(fullData, targetPoints, 'time', firstJointKey);
    return sampledData;
  }, [jointTimestamps, jointNames, jointPositions]);

  // Prepare action chart data
  const actionChartData = useMemo(() => {
    if (!actionTimestamps.length || !actionNames.length || !actionValues.length) {
      return [];
    }

    const numActions = actionNames.length;
    const targetPoints = 1000;

    const fullData = actionTimestamps.map((time, i) => {
      const point = { time };
      const startIdx = i * numActions;
      actionNames.forEach((name, j) => {
        point[`action_${name}`] = actionValues[startIdx + j] || 0;
      });
      return point;
    });

    if (fullData.length <= targetPoints) {
      return fullData;
    }

    const firstActionKey = `action_${actionNames[0]}`;
    const sampledData = lttbDownsample(fullData, targetPoints, 'time', firstActionKey);
    return sampledData;
  }, [actionTimestamps, actionNames, actionValues]);

  // Toggle joint expansion by row
  const toggleJoint = useCallback((jointName) => {
    const jointIndex = allJointNames.indexOf(jointName);
    if (jointIndex === -1) return;

    const rowStart = Math.floor(jointIndex / 3) * 3;
    const rowJoints = allJointNames.slice(rowStart, rowStart + 3);

    setExpandedJoints((prev) => {
      const newSet = new Set(prev);
      const isRowExpanded = rowJoints.some((name) => prev.has(name));

      if (isRowExpanded) {
        rowJoints.forEach((name) => newSet.delete(name));
      } else {
        rowJoints.forEach((name) => newSet.add(name));
      }
      return newSet;
    });
  }, [allJointNames]);

  const expandAllJoints = useCallback(() => {
    setExpandedJoints(new Set(allJointNames));
  }, [allJointNames]);

  const collapseAllJoints = useCallback(() => {
    setExpandedJoints(new Set());
  }, []);

  // Handle bag selection
  const handleSelectBag = async (path) => {
    setShowFileBrowser(false);
    isLoadingBagRef.current = true;

    if (bagPath && taskMarkers.length > 0) {
      markersCacheRef.current.set(bagPath, [...taskMarkers]);
    }

    dispatch(setSelectedBagPath(path));
    dispatch(setLoading(true));

    // Clear previous bag's editing state
    setTrimStart(null);
    setTrimEnd(null);
    setLoopStart(null);
    setLoopEnd(null);
    setExcludeRegions([]);
    setPendingExcludeStart(null);

    const parentPath = path.substring(0, path.lastIndexOf('/'));
    if (parentPath && parentPath !== parentFolderPath) {
      setParentFolderPath(parentPath);
      try {
        const listResult = await getRosbagList(parentPath);
        if (listResult.success && listResult.rosbags) {
          setRosbagList(listResult.rosbags);
          const idx = listResult.rosbags.findIndex((bag) => bag.path === path);
          setCurrentBagIndex(idx);
        }
      } catch (err) {
        console.error('Failed to load rosbag list:', err);
      }
    } else {
      const idx = rosbagList.findIndex((bag) => bag.path === path);
      setCurrentBagIndex(idx);
    }

    try {
      const result = await getReplayData(path);
      if (result.success) {
        const cachedMarkers = markersCacheRef.current.get(path);
        if (cachedMarkers && cachedMarkers.length > 0) {
          result.task_markers = cachedMarkers;
          toast('Restored cached markers');
        }
        dispatch(setReplayData(result));
        // Restore trim points from saved data
        if (result.trim_points) {
          const tp = result.trim_points;
          if (tp.start) {
            setTrimStart({ time: tp.start.time, frame: tp.start.frame, instruction: tp.start.instruction || 'Start' });
          }
          if (tp.end) {
            setTrimEnd({ time: tp.end.time, frame: tp.end.frame });
          }
        }
        // Restore exclude regions from saved data
        if (result.exclude_regions && result.exclude_regions.length > 0) {
          setExcludeRegions(result.exclude_regions);
        }
        toast.success('Replay data loaded successfully');
      } else {
        dispatch(setError(result.message));
        toast.error(`Failed to load replay data: ${result.message}`);
      }
    } catch (err) {
      dispatch(setError(err.message));
      toast.error(`Error loading replay data: ${err.message}`);
    } finally {
      // Allow auto-save after load settles
      setTimeout(() => { isLoadingBagRef.current = false; }, 500);
    }
  };

  // Navigate to previous/next rosbag
  const navigateRosbag = useCallback(
    async (direction) => {
      if (rosbagList.length === 0 || currentBagIndex === -1) return;
      if (isDownloading) return;

      const newIndex = direction === 'prev' ? currentBagIndex - 1 : currentBagIndex + 1;
      if (newIndex < 0 || newIndex >= rosbagList.length) return;

      if (bagPath && taskMarkers.length > 0) {
        markersCacheRef.current.set(bagPath, [...taskMarkers]);
      }

      isLoadingBagRef.current = true;
      const newBag = rosbagList[newIndex];
      setCurrentBagIndex(newIndex);
      dispatch(setSelectedBagPath(newBag.path));
      dispatch(setLoading(true));

      setVideoBlobUrls([]);
      setDownloadProgress(0);
      dispatch(setIsVideoLoaded(false));

      setTrimStart(null);
      setTrimEnd(null);
      setLoopStart(null);
      setLoopEnd(null);
      setExcludeRegions([]);
      setPendingExcludeStart(null);

      try {
        const result = await getReplayData(newBag.path);
        if (result.success) {
          const cachedMarkers = markersCacheRef.current.get(newBag.path);
          if (cachedMarkers && cachedMarkers.length > 0) {
            result.task_markers = cachedMarkers;
            toast('Restored cached markers');
          }
          dispatch(setReplayData(result));
          if (result.trim_points) {
            const tp = result.trim_points;
            if (tp.start) {
              setTrimStart({ time: tp.start.time, frame: tp.start.frame, instruction: tp.start.instruction || 'Start' });
            }
            if (tp.end) {
              setTrimEnd({ time: tp.end.time, frame: tp.end.frame });
            }
          }
          if (result.exclude_regions && result.exclude_regions.length > 0) {
            setExcludeRegions(result.exclude_regions);
          }
          toast.success(`Loaded: ${newBag.name}`);
        } else {
          dispatch(setError(result.message));
          toast.error(`Failed to load: ${result.message}`);
        }
      } catch (err) {
        dispatch(setError(err.message));
        toast.error(`Error: ${err.message}`);
      } finally {
        setTimeout(() => { isLoadingBagRef.current = false; }, 500);
      }
    },
    [rosbagList, currentBagIndex, isDownloading, dispatch, getReplayData, bagPath, taskMarkers]
  );

  // Handle video events
  useEffect(() => {
    if (!isVideoLoaded || videoBlobUrls.length === 0) return;

    const videos = videoRefs.current;

    const handleTimeUpdate = () => {
      const video = expandedVideoIndex !== null ? videos[expandedVideoIndex] : videos.find(v => v);
      if (video) {
        dispatch(setCurrentTime(video.currentTime));
      }
    };

    const handleEnded = () => {
      dispatch(setIsPlaying(false));
      dispatch(setCurrentTime(duration));
    };

    videos.forEach((video) => {
      if (video) {
        video.addEventListener('timeupdate', handleTimeUpdate);
        video.addEventListener('ended', handleEnded);
      }
    });

    return () => {
      videos.forEach((video) => {
        if (video) {
          video.removeEventListener('timeupdate', handleTimeUpdate);
          video.removeEventListener('ended', handleEnded);
        }
      });
    };
  }, [isVideoLoaded, videoBlobUrls.length, duration, dispatch, expandedVideoIndex]);

  // A-B Loop: Jump to A when reaching B (video mode only — MCAP handles looping internally)
  useEffect(() => {
    if (isDirectMcapMode) return; // MCAP player's rAF loop handles A-B looping
    if (!isPlaying || loopStart === null || loopEnd === null) return;

    if (currentTime >= loopEnd) {
      videoRefs.current.forEach((v) => {
        if (v) v.currentTime = loopStart;
      });
      dispatch(setCurrentTime(loopStart));
    }
  }, [currentTime, isPlaying, loopStart, loopEnd, isDirectMcapMode, dispatch]);

  // Unified seek-and-play: seeks to time and starts playback (handles both MCAP and video modes)
  const seekAndPlay = useCallback((time) => {
    const clamped = Math.max(0, Math.min(duration, time));
    if (isDirectMcapMode) {
      mcapPlayer.syncToTime(clamped);
      if (!isPlaying) {
        mcapPlayer.playAll();
      }
    } else {
      videoRefs.current.forEach((v) => {
        if (v) { v.currentTime = clamped; v.play().catch(() => {}); }
      });
      dispatch(setCurrentTime(clamped));
      dispatch(setIsPlaying(true));
    }
  }, [duration, isDirectMcapMode, isPlaying, mcapPlayer, dispatch]);

  // Unified seek-to-time: seeks without changing play state (handles both MCAP and video modes)
  const seekToTime = useCallback((time) => {
    const clamped = Math.max(0, Math.min(duration, time));
    if (isDirectMcapMode) {
      mcapPlayer.syncToTime(clamped);
    } else {
      videoRefs.current.forEach((v) => {
        if (v) v.currentTime = clamped;
      });
      dispatch(setCurrentTime(clamped));
    }
  }, [duration, isDirectMcapMode, mcapPlayer, dispatch]);

  // Unified seek-and-pause: pauses playback then seeks (handles both MCAP and video modes)
  const seekAndPause = useCallback((time) => {
    const clamped = Math.max(0, Math.min(duration, time));
    if (isDirectMcapMode) {
      if (isPlaying) mcapPlayer.pauseAll();
      mcapPlayer.syncToTime(clamped);
    } else {
      videoRefs.current.forEach((v) => {
        if (v) { v.pause(); v.currentTime = clamped; }
      });
      dispatch(setCurrentTime(clamped));
      dispatch(setIsPlaying(false));
    }
  }, [duration, isPlaying, isDirectMcapMode, mcapPlayer, dispatch]);

  // Handle play/pause
  const togglePlayPause = useCallback(() => {
    if (isDirectMcapMode) {
      mcapPlayer.togglePlayPause();
      return;
    }

    if (isPlaying) {
      videoRefs.current.forEach((video) => {
        if (video) video.pause();
      });
      dispatch(setIsPlaying(false));
    } else {
      const firstVideo = videoRefs.current[0];

      if (firstVideo && firstVideo.ended) {
        videoRefs.current.forEach((video) => {
          if (video) video.currentTime = 0;
        });
        dispatch(setCurrentTime(0));
      } else {
        const targetTime = firstVideo?.currentTime || 0;
        videoRefs.current.forEach((video) => {
          if (video) video.currentTime = targetTime;
        });
      }

      videoRefs.current.forEach((video) => {
        if (video) video.play().catch(() => {});
      });
      dispatch(setIsPlaying(true));
    }
  }, [isPlaying, isDirectMcapMode, mcapPlayer, dispatch]);

  // Restart playback
  const restartPlayback = useCallback(() => {
    if (isDirectMcapMode) {
      mcapPlayer.restart();
      return;
    }

    videoRefs.current.forEach((video) => {
      if (video) {
        video.currentTime = 0;
        video.play().catch(() => {});
      }
    });
    dispatch(setCurrentTime(0));
    dispatch(setIsPlaying(true));
  }, [isDirectMcapMode, mcapPlayer, dispatch]);

  // Step frame forward or backward
  const stepFrame = useCallback(
    (direction) => {
      if (isDirectMcapMode) {
        mcapPlayer.stepFrame(direction);
        return;
      }

      if (!isVideoLoaded) return;

      if (isPlaying) {
        videoRefs.current.forEach((v) => v?.pause());
        dispatch(setIsPlaying(false));
      }

      const fps = videoFps[0] || 30;
      const frameTime = 1 / fps;
      const delta = direction === 'forward' ? frameTime : -frameTime;
      const newTime = Math.max(0, Math.min(duration, currentTime + delta));

      videoRefs.current.forEach((v) => {
        if (v) v.currentTime = newTime;
      });
      dispatch(setCurrentTime(newTime));
    },
    [isVideoLoaded, isPlaying, isDirectMcapMode, mcapPlayer, videoFps, duration, currentTime, dispatch]
  );

  // Seek relative
  const seekRelative = useCallback(
    (seconds) => {
      if (isDirectMcapMode) {
        mcapPlayer.seekRelative(seconds);
        return;
      }

      if (!isVideoLoaded) return;

      const newTime = Math.max(0, Math.min(duration, currentTime + seconds));
      videoRefs.current.forEach((v) => {
        if (v) v.currentTime = newTime;
      });
      dispatch(setCurrentTime(newTime));
    },
    [isVideoLoaded, isDirectMcapMode, mcapPlayer, duration, currentTime, dispatch]
  );

  // Toggle A-B loop points
  const toggleLoopPoint = useCallback(() => {
    if (!isVideoLoaded) return;

    if (loopStart === null) {
      setLoopStart(currentTime);
      toast(`Loop A set at ${formatTime(currentTime)}`);
    } else if (loopEnd === null) {
      if (currentTime > loopStart) {
        setLoopEnd(currentTime);
        toast(`Loop B set at ${formatTime(currentTime)}`);
      } else {
        setLoopStart(currentTime);
        toast(`Loop A updated to ${formatTime(currentTime)}`);
      }
    } else {
      setLoopStart(currentTime);
      setLoopEnd(null);
      toast(`Loop A set at ${formatTime(currentTime)}`);
    }
  }, [isVideoLoaded, loopStart, loopEnd, currentTime]);

  const clearLoop = useCallback(() => {
    setLoopStart(null);
    setLoopEnd(null);
    toast('Loop cleared');
  }, []);

  // ===== Task Marker Callbacks =====

  const openMarkerDialog = useCallback(() => {
    if (!isVideoLoaded) return;

    if (!trimStart) {
      toast.error('Please set Start point first (press S)');
      return;
    }

    if (isPlaying) {
      videoRefs.current.forEach((v) => v?.pause());
      dispatch(setIsPlaying(false));
    }

    setPendingMarkerTime(currentTime);
    setMarkerInput('');
    setShowMarkerDialog(true);
  }, [isVideoLoaded, isPlaying, currentTime, dispatch, trimStart]);

  const addMarker = useCallback((instruction) => {
    if (pendingMarkerTime === null || !instruction.trim()) return;

    const fps = videoFps[0] || 30;
    const frame = Math.round(pendingMarkerTime * fps);

    const newMarker = {
      frame: frame,
      time: pendingMarkerTime,
      instruction: instruction.trim(),
    };

    const updatedMarkers = [...taskMarkers, newMarker].sort((a, b) => a.frame - b.frame);
    dispatch(setTaskMarkers(updatedMarkers));

    setShowMarkerDialog(false);
    setPendingMarkerTime(null);
    setMarkerInput('');
    toast.success(`Marker added at frame ${frame}: "${instruction.trim()}"`);
  }, [pendingMarkerTime, videoFps, taskMarkers, dispatch]);

  const deleteNearestMarker = useCallback(() => {
    if (!taskMarkers.length) {
      toast('No markers to delete');
      return;
    }

    let nearestIndex = 0;
    let minDiff = Math.abs(taskMarkers[0].time - currentTime);

    taskMarkers.forEach((marker, index) => {
      const diff = Math.abs(marker.time - currentTime);
      if (diff < minDiff) {
        minDiff = diff;
        nearestIndex = index;
      }
    });

    if (minDiff > 2) {
      toast('No marker within 2 seconds');
      return;
    }

    const deletedMarker = taskMarkers[nearestIndex];
    const updatedMarkers = taskMarkers.filter((_, i) => i !== nearestIndex);
    dispatch(setTaskMarkers(updatedMarkers));
    toast(`Deleted: "${deletedMarker.instruction}"`);
  }, [taskMarkers, currentTime, dispatch]);

  const jumpToMarker = useCallback((index) => {
    if (index < 0 || index >= taskMarkers.length) {
      toast(`No marker at position ${index + 1}`);
      return;
    }

    const marker = taskMarkers[index];
    videoRefs.current.forEach((v) => {
      if (v) v.currentTime = marker.time;
    });
    dispatch(setCurrentTime(marker.time));
    toast(`Jumped to: "${marker.instruction}"`);
  }, [taskMarkers, dispatch]);

  const openTrimStartDialog = useCallback(() => {
    if (!isVideoLoaded) return;
    // Remember play state and pause
    wasPlayingBeforeTrimRef.current = isPlaying;
    if (isPlaying) {
      if (isDirectMcapMode) {
        mcapPlayer.pauseAll();
      } else {
        videoRefs.current.forEach((v) => v?.pause());
      }
      dispatch(setIsPlaying(false));
    }
    setTrimStartInstruction(trimStart?.instruction || '');
    setShowTrimStartDialog(true);
  }, [isVideoLoaded, isPlaying, isDirectMcapMode, mcapPlayer, dispatch, trimStart]);

  const applyTrimStart = useCallback((instruction) => {
    const fps = videoFps[0] || 30;
    const frame = Math.round(currentTime * fps);
    setTrimStart({
      time: currentTime,
      frame,
      instruction: instruction.trim() || 'Start',
    });
    setShowTrimStartDialog(false);
    toast(`Start point set at ${formatTime(currentTime)} (frame ${frame})`);
    // Resume playback if it was playing before dialog opened
    if (wasPlayingBeforeTrimRef.current) {
      wasPlayingBeforeTrimRef.current = false;
      seekAndPlay(currentTime);
    }
  }, [currentTime, videoFps, seekAndPlay]);

  const applyTrimEnd = useCallback(() => {
    if (!isVideoLoaded) return;
    // Remember play state and pause briefly
    const wasPlaying = isPlaying;
    if (isPlaying) {
      if (isDirectMcapMode) {
        mcapPlayer.pauseAll();
      } else {
        videoRefs.current.forEach((v) => v?.pause());
      }
      dispatch(setIsPlaying(false));
    }
    const fps = videoFps[0] || 30;
    const frame = Math.round(currentTime * fps);
    setTrimEnd({ time: currentTime, frame });
    toast(`End point set at ${formatTime(currentTime)}`);
    // Resume playback if it was playing
    if (wasPlaying) {
      seekAndPlay(currentTime);
    }
  }, [isVideoLoaded, isPlaying, isDirectMcapMode, mcapPlayer, dispatch, currentTime, videoFps, seekAndPlay]);

  const clearTrimPoints = useCallback(() => {
    setTrimStart(null);
    setTrimEnd(null);
    toast('Trim points cleared');
  }, []);

  const toggleExcludeRegion = useCallback(() => {
    if (!isVideoLoaded) return;

    const fps = videoFps[0] || 30;
    const frame = Math.round(currentTime * fps);

    if (pendingExcludeStart === null) {
      setPendingExcludeStart({ time: currentTime, frame });
      toast(`Exclude start at ${formatTime(currentTime)} - press X again to set end`);
    } else {
      if (currentTime <= pendingExcludeStart.time) {
        toast.error('Exclude end must be after start');
        return;
      }
      const newRegion = {
        start: pendingExcludeStart,
        end: { time: currentTime, frame },
      };
      setExcludeRegions((prev) => [...prev, newRegion].sort((a, b) => a.start.time - b.start.time));
      setPendingExcludeStart(null);
      toast.success(`Excluded: ${formatTime(newRegion.start.time)} - ${formatTime(newRegion.end.time)}`);
    }
  }, [isVideoLoaded, currentTime, videoFps, pendingExcludeStart]);

  const cancelExcludeRegion = useCallback(() => {
    if (pendingExcludeStart) {
      setPendingExcludeStart(null);
      toast('Exclude marking cancelled');
    }
  }, [pendingExcludeStart]);

  const deleteExcludeRegion = useCallback((index) => {
    setExcludeRegions((prev) => prev.filter((_, i) => i !== index));
    toast('Exclude region deleted');
  }, []);

  // Save markers and trim points to server
  const saveMarkers = useCallback(async () => {
    if (!bagPath) {
      toast.error('No bag selected');
      return;
    }

    setIsSavingMarkers(true);
    try {
      const saveData = { task_markers: taskMarkers };

      if (trimStart || trimEnd) {
        saveData.trim_points = {
          start: trimStart ? { time: trimStart.time, frame: trimStart.frame, instruction: trimStart.instruction } : null,
          end: trimEnd ? { time: trimEnd.time, frame: trimEnd.frame } : null,
        };
      }

      if (excludeRegions.length > 0) {
        saveData.exclude_regions = excludeRegions.map((region) => ({
          start: { time: region.start.time, frame: region.start.frame },
          end: { time: region.end.time, frame: region.end.frame },
        }));
      }

      const response = await fetch(
        `/data-api/task-markers${bagPath}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(saveData),
        }
      );

      const result = await response.json();
      if (result.success) {
        toast.success('Data saved to file');
      } else {
        toast.error(result.message || 'Failed to save');
      }
    } catch (error) {
      toast.error(`Save failed: ${error.message}`);
    } finally {
      setIsSavingMarkers(false);
    }
  }, [bagPath, taskMarkers, trimStart, trimEnd, excludeRegions]);

  // Auto-save: debounced save whenever editing data changes
  useEffect(() => {
    // Skip during bag load (state is being set from server, not user edits)
    if (isLoadingBagRef.current) return;
    // Skip if no bag loaded
    if (!bagPath) return;
    // Skip if nothing to save
    const hasData = taskMarkers.length > 0 || trimStart || trimEnd || excludeRegions.length > 0;
    if (!hasData) return;

    // Debounce: save 2 seconds after last change
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
    }
    autoSaveTimerRef.current = setTimeout(async () => {
      try {
        const saveData = { task_markers: taskMarkers };
        if (trimStart || trimEnd) {
          saveData.trim_points = {
            start: trimStart ? { time: trimStart.time, frame: trimStart.frame, instruction: trimStart.instruction } : null,
            end: trimEnd ? { time: trimEnd.time, frame: trimEnd.frame } : null,
          };
        }
        if (excludeRegions.length > 0) {
          saveData.exclude_regions = excludeRegions.map((region) => ({
            start: { time: region.start.time, frame: region.start.frame },
            end: { time: region.end.time, frame: region.end.frame },
          }));
        }
        const response = await fetch(`/data-api/task-markers${bagPath}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(saveData),
        });
        const result = await response.json();
        if (result.success) {
          console.log('[AutoSave] Saved markers/trim/exclude');
        }
      } catch (err) {
        console.error('[AutoSave] Failed:', err);
      }
    }, 2000);

    return () => {
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
      }
    };
  }, [bagPath, taskMarkers, trimStart, trimEnd, excludeRegions]);

  const savePaletteToStorage = useCallback((newPalette) => {
    setInstructionPalette(newPalette);
    localStorage.setItem('taskInstructionPalette', JSON.stringify({
      name: 'Custom Palette',
      instructions: newPalette,
    }));
  }, []);

  const addToPalette = useCallback((instruction) => {
    if (!instruction.trim()) return;
    if (instructionPalette.includes(instruction.trim())) {
      toast('Instruction already in palette');
      return;
    }
    const newPalette = [...instructionPalette, instruction.trim()];
    savePaletteToStorage(newPalette);
    toast.success('Added to palette');
  }, [instructionPalette, savePaletteToStorage]);

  const removeFromPalette = useCallback((index) => {
    const newPalette = instructionPalette.filter((_, i) => i !== index);
    savePaletteToStorage(newPalette);
  }, [instructionPalette, savePaletteToStorage]);

  const importPaletteFromJson = useCallback((event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (data.instructions && Array.isArray(data.instructions)) {
          savePaletteToStorage(data.instructions);
          toast.success(`Imported ${data.instructions.length} instructions`);
        } else {
          toast.error('Invalid JSON format');
        }
      } catch {
        toast.error('Failed to parse JSON file');
      }
    };
    reader.readAsText(file);
    event.target.value = '';
  }, [savePaletteToStorage]);

  const exportPaletteToJson = useCallback(() => {
    if (!instructionPalette.length) {
      toast('No instructions to export');
      return;
    }

    const data = {
      name: 'Instruction Palette',
      instructions: instructionPalette,
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `instruction_palette_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Palette exported');
  }, [instructionPalette]);

  // Get current active task based on time
  const { currentActiveTask, currentActiveTaskIndex } = useMemo(() => {
    if (!taskMarkers.length) return { currentActiveTask: null, currentActiveTaskIndex: -1 };

    let activeMarker = null;
    let activeIndex = -1;
    for (let i = 0; i < taskMarkers.length; i++) {
      if (taskMarkers[i].time <= currentTime) {
        activeMarker = taskMarkers[i];
        activeIndex = i;
      } else {
        break;
      }
    }
    return { currentActiveTask: activeMarker, currentActiveTaskIndex: activeIndex };
  }, [taskMarkers, currentTime]);

  // Keyboard shortcuts
  useKeyboardShortcuts({
    isActive,
    handlers: {
      onNavigatePrev: () => navigateRosbag('prev'),
      onNavigateNext: () => navigateRosbag('next'),
      onStepBackward: () => stepFrame('backward'),
      onStepForward: () => stepFrame('forward'),
      onSeekRelative: seekRelative,
      onTogglePlayPause: togglePlayPause,
      onToggleLoopPoint: toggleLoopPoint,
      onClearLoop: clearLoop,
      onOpenMarkerDialog: openMarkerDialog,
      onDeleteNearestMarker: deleteNearestMarker,
      onJumpToMarker: jumpToMarker,
      onSetTrimStart: openTrimStartDialog,
      onSetTrimEnd: applyTrimEnd,
      onToggleExcludeRegion: toggleExcludeRegion,
      onCancelExclude: cancelExcludeRegion,
      onToggleHelp: () => setShowHelpModal((prev) => !prev),
      onCloseHelp: () => setShowHelpModal(false),
      onCloseTrimDialog: () => setShowTrimStartDialog(false),
      hasPendingExclude: pendingExcludeStart,
      showHelpModal,
      showTrimDialog: showTrimStartDialog,
    },
  });

  // Change playback speed
  const changePlaybackSpeed = useCallback((speed) => {
    setPlaybackSpeed(speed);
    videoRefs.current.forEach((video) => {
      if (video) video.playbackRate = speed;
    });
  }, []);

  // Apply playback speed when videos are loaded
  useEffect(() => {
    videoRefs.current.forEach((video) => {
      if (video) video.playbackRate = playbackSpeed;
    });
  }, [videoBlobUrls, playbackSpeed]);

  // Handle seek for all videos
  const handleSeek = useCallback((e) => {
    if (!isVideoLoaded && !isDirectMcapMode) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = x / rect.width;
    const newTime = percentage * duration;

    if (isDirectMcapMode) {
      mcapPlayer.syncToTime(newTime);
      return;
    }

    videoRefs.current.forEach((video) => {
      if (video) video.currentTime = newTime;
    });
    dispatch(setCurrentTime(newTime));
  }, [isVideoLoaded, isDirectMcapMode, mcapPlayer, duration, dispatch]);

  // Handle chart click to seek
  const handleChartSeek = useCallback((time) => {
    if ((!isVideoLoaded && !isDirectMcapMode) || typeof time !== 'number') return;

    const clampedTime = Math.max(0, Math.min(duration, time));

    if (isDirectMcapMode) {
      mcapPlayer.syncToTime(clampedTime);
      return;
    }

    videoRefs.current.forEach((video) => {
      if (video) video.currentTime = clampedTime;
    });
    dispatch(setCurrentTime(clampedTime));
  }, [isVideoLoaded, isDirectMcapMode, mcapPlayer, duration, dispatch]);

  // Extract short name from video file path
  const getShortVideoName = (filePath) => {
    const fileName = filePath.split('/').pop();
    return fileName
      .replace('.mp4', '')
      .replace('_compressed', '')
      .split('_')
      .slice(-3)
      .join('_');
  };

  // Reset state when leaving page
  useEffect(() => {
    if (!isActive) {
      videoRefs.current.forEach((video) => {
        if (video) video.pause();
      });
      dispatch(setIsPlaying(false));
    }
  }, [isActive, dispatch]);

  // MCAP mode: set isVideoLoaded when MCAP reader is ready
  useEffect(() => {
    if (isDirectMcapMode && mcapPlayer.isReady) {
      dispatch(setIsVideoLoaded(true));
    }
  }, [isDirectMcapMode, mcapPlayer.isReady, dispatch]);

  // Clean up blob URLs when bag changes
  useEffect(() => {
    setVideoBlobUrls([]);
    setDownloadProgress(0);
  }, [selectedBagPath]);

  // Hidden panels list
  const hiddenPanels = Object.values(layoutPanels).filter((p) => !p.visible);

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 flex-shrink-0">
        <h1 className="text-xl font-bold text-gray-800">Replay Viewer</h1>
        <div className="flex items-center gap-2">
          {/* Hidden panel restore dropdown */}
          {hiddenPanels.length > 0 && (
            <div className="relative">
              <button
                onClick={() => setShowPanelMenu(!showPanelMenu)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors text-sm"
              >
                <MdDashboard size={16} />
                Panels ({hiddenPanels.length} hidden)
              </button>
              {showPanelMenu && (
                <div className="absolute right-0 top-full mt-1 bg-white rounded-lg shadow-lg border z-50 py-1 min-w-[160px]">
                  {hiddenPanels.map((panel) => (
                    <button
                      key={panel.id}
                      onClick={() => {
                        dispatch(showPanel(panel.id));
                        setShowPanelMenu(false);
                      }}
                      className="w-full text-left px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
                    >
                      Show {panel.title}
                    </button>
                  ))}
                  <div className="border-t my-1" />
                  <button
                    onClick={() => {
                      dispatch(resetToDefaultLayout());
                      setShowPanelMenu(false);
                    }}
                    className="w-full text-left px-3 py-1.5 text-sm text-blue-600 hover:bg-blue-50 transition-colors"
                  >
                    Reset Layout
                  </button>
                </div>
              )}
            </div>
          )}
          {isLoaded && (
            <button
              onClick={() => dispatch(resetToDefaultLayout())}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors text-sm"
              title="Reset panel layout to default"
            >
              <MdRefresh size={16} />
              Reset Layout
            </button>
          )}
          {isDirectMcapMode && (
            <button
              onClick={() => setUseWebGL(!useWebGL)}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors text-sm',
                useWebGL
                  ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              )}
              title="Toggle WebGL GPU rendering"
            >
              WebGL
            </button>
          )}
          {useWebGL && isDirectMcapMode && (
            <div className="flex items-center gap-2 px-2 py-1 bg-gray-100 rounded-lg">
              <label className="flex items-center gap-1 text-xs text-gray-600">
                <span className="w-8">Bright</span>
                <input
                  type="range" min="-0.5" max="0.5" step="0.05"
                  value={videoBrightness}
                  onChange={(e) => setVideoBrightness(parseFloat(e.target.value))}
                  className="w-14 h-1"
                />
                <span className="w-7 text-right">{videoBrightness.toFixed(2)}</span>
              </label>
              <label className="flex items-center gap-1 text-xs text-gray-600">
                <span className="w-10">Contrast</span>
                <input
                  type="range" min="0.5" max="2" step="0.05"
                  value={videoContrast}
                  onChange={(e) => setVideoContrast(parseFloat(e.target.value))}
                  className="w-14 h-1"
                />
                <span className="w-7 text-right">{videoContrast.toFixed(2)}</span>
              </label>
              <button
                onClick={() => { setVideoBrightness(0); setVideoContrast(1); }}
                className="text-xs text-blue-600 hover:text-blue-800"
              >
                Reset
              </button>
            </div>
          )}
          <button
            onClick={() => setShowFileBrowser(true)}
            className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
          >
            <MdFolder size={18} />
            Select ROSbag
          </button>
        </div>
      </div>

      {/* Main content — Grid Layout */}
      {isLoaded && (videoFiles.length > 0 || isDirectMcapMode) ? (
        <>
          <div className="flex-1 px-4 min-h-0 overflow-hidden">
            <ReplayLayoutContainer
              cameraPanelContent={
                <CameraPanel
                  isDirectMcapMode={isDirectMcapMode}
                  isLoaded={isLoaded}
                  isLoading={isLoading}
                  error={error}
                  videoFiles={videoFiles}
                  videoNames={videoNames}
                  videoBlobUrls={videoBlobUrls}
                  videoRefs={videoRefs}
                  expandedVideoIndex={expandedVideoIndex}
                  setExpandedVideoIndex={setExpandedVideoIndex}
                  mcapPlayer={mcapPlayer}
                  useWebGL={useWebGL}
                  videoBrightness={videoBrightness}
                  videoContrast={videoContrast}
                  isDownloading={isDownloading}
                  downloadProgress={downloadProgress}
                  getShortVideoName={getShortVideoName}
                />
              }
              viewer3DPanelContent={
                <Viewer3DPanel
                  jointTimestamps={jointTimestamps}
                  jointNames={jointNames}
                  jointPositions={jointPositions}
                  actionTimestamps={actionTimestamps}
                  actionNames={actionNames}
                  actionValues={actionValues}
                  currentTime={currentTime}
                />
              }
              jointDataPanelContent={
                <JointDataPanel
                  allJointNames={allJointNames}
                  stateChartData={stateChartData}
                  actionChartData={actionChartData}
                  currentTime={currentTime}
                  duration={duration}
                  expandedJoints={expandedJoints}
                  toggleJoint={toggleJoint}
                  expandAllJoints={expandAllJoints}
                  collapseAllJoints={collapseAllJoints}
                  hasActionData={hasActionData}
                  actionNames={actionNames}
                  handleChartSeek={handleChartSeek}
                />
              }
              sidebarPanelContent={
                <SidebarPanel
                  recordingDate={recordingDate}
                  robotType={robotType}
                  fileSizeBytes={fileSizeBytes}
                  duration={duration}
                  frameCounts={frameCounts}
                  taskMarkers={taskMarkers}
                  currentActiveTask={currentActiveTask}
                  currentActiveTaskIndex={currentActiveTaskIndex}
                  trimStart={trimStart}
                  trimEnd={trimEnd}
                  excludeRegions={excludeRegions}
                  pendingExcludeStart={pendingExcludeStart}
                  isSavingMarkers={isSavingMarkers}
                  instructionPalette={instructionPalette}
                  newPaletteInput={newPaletteInput}
                  setNewPaletteInput={setNewPaletteInput}
                  addToPalette={addToPalette}
                  removeFromPalette={removeFromPalette}
                  importPaletteFromJson={importPaletteFromJson}
                  exportPaletteToJson={exportPaletteToJson}
                  rosbagList={rosbagList}
                  currentBagIndex={currentBagIndex}
                  isDownloading={isDownloading}
                  navigateRosbag={navigateRosbag}
                  handleSelectBag={handleSelectBag}
                  dispatch={dispatch}
                  setTaskMarkers={setTaskMarkers}
                  setTrimStart={setTrimStart}
                  setLoopStart={setLoopStart}
                  setLoopEnd={setLoopEnd}
                  seekAndPlay={seekAndPlay}
                  seekAndPause={seekAndPause}
                  saveMarkers={saveMarkers}
                  deleteExcludeRegion={deleteExcludeRegion}
                  currentTime={currentTime}
                />
              }
            />
          </div>

          {/* Timeline Controls — fixed at bottom */}
          <TimelineControls
            currentTime={currentTime}
            duration={duration}
            isPlaying={isPlaying}
            isVideoLoaded={isVideoLoaded}
            isDirectMcapMode={isDirectMcapMode}
            mcapPlayer={mcapPlayer}
            trimStart={trimStart}
            trimEnd={trimEnd}
            loopStart={loopStart}
            loopEnd={loopEnd}
            excludeRegions={excludeRegions}
            pendingExcludeStart={pendingExcludeStart}
            taskMarkers={taskMarkers}
            currentActiveTask={currentActiveTask}
            currentActiveTaskIndex={currentActiveTaskIndex}
            handleSeek={handleSeek}
            togglePlayPause={togglePlayPause}
            restartPlayback={restartPlayback}
            openMarkerDialog={openMarkerDialog}
            openTrimStartDialog={openTrimStartDialog}
            applyTrimEnd={applyTrimEnd}
            clearTrimPoints={clearTrimPoints}
            clearLoop={clearLoop}
            changePlaybackSpeed={changePlaybackSpeed}
            playbackSpeed={playbackSpeed}
            PLAYBACK_SPEEDS={PLAYBACK_SPEEDS}
            setShowHelpModal={setShowHelpModal}
            videoFiles={videoFiles}
            setLoopStart={setLoopStart}
            setLoopEnd={setLoopEnd}
            seekAndPlay={seekAndPlay}
            seekToTime={seekToTime}
          />
        </>
      ) : (
        <div className="flex-1 flex items-center justify-center text-gray-500 bg-gray-50">
          {isLoading ? (
            <div className="flex flex-col items-center gap-4">
              <MdRefresh className="animate-spin" size={48} />
              <span>Loading replay data...</span>
            </div>
          ) : error ? (
            <div className="text-center text-red-500">
              <p className="font-semibold">Error loading data</p>
              <p className="text-sm">{error}</p>
            </div>
          ) : (
            <div className="text-center">
              <MdFolder size={64} className="mx-auto mb-4 text-gray-400" />
              <p>Select a ROSbag to start viewing</p>
            </div>
          )}
        </div>
      )}

      {/* Bag info */}
      {selectedBagPath && (
        <div className="px-4 py-1 text-xs text-gray-500 flex-shrink-0">
          <span className="font-medium">Selected:</span> {selectedBagPath}
        </div>
      )}

      {/* File browser modal */}
      {showFileBrowser && (
        <FileBrowserModal
          isOpen={showFileBrowser}
          onClose={() => setShowFileBrowser(false)}
          onFileSelect={(item) => handleSelectBag(item.full_path)}
          title="Select ROSbag Directory"
          allowDirectorySelect={true}
          allowFileSelect={false}
          targetFileName="metadata.yaml"
          initialPath="/workspace/rosbag2"
          defaultPath="/workspace/rosbag2"
        />
      )}

      {/* Task Marker add dialog */}
      {showMarkerDialog && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setShowMarkerDialog(false)}
        >
          <div
            className="bg-white rounded-xl shadow-xl w-96 max-w-full mx-4"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                setShowMarkerDialog(false);
              } else if (e.key === 'Enter' && markerInput.trim()) {
                addMarker(markerInput);
              } else if (e.key >= '1' && e.key <= '9') {
                const idx = parseInt(e.key) - 1;
                if (idx < instructionPalette.length) {
                  addMarker(instructionPalette[idx]);
                }
              }
            }}
          >
            <div className="p-4 border-b bg-gray-50 rounded-t-xl">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-800">Add Task Marker</h3>
                <button onClick={() => setShowMarkerDialog(false)} className="text-gray-400 hover:text-gray-600">
                  <MdClose size={20} />
                </button>
              </div>
              <p className="text-sm text-gray-500 mt-1">
                Time: {formatTime(pendingMarkerTime || 0)}
              </p>
            </div>

            <div className="p-4">
              {instructionPalette.length > 0 && (
                <div className="mb-4">
                  <p className="text-xs text-gray-500 mb-2">Quick select (press 1-9):</p>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {instructionPalette.slice(0, 9).map((instruction, idx) => (
                      <button
                        key={`quick-${idx}`}
                        onClick={() => addMarker(instruction)}
                        className="w-full flex items-center gap-2 p-2 text-left hover:bg-purple-50 rounded transition-colors"
                      >
                        <span className="w-5 h-5 flex items-center justify-center bg-purple-100 text-purple-700 text-xs font-bold rounded">
                          {idx + 1}
                        </span>
                        <span className="text-sm text-gray-700 truncate">{instruction}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <p className="text-xs text-gray-500 mb-2">Or enter custom instruction:</p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={markerInput}
                    onChange={(e) => setMarkerInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && markerInput.trim()) {
                        e.preventDefault();
                        addMarker(markerInput);
                      }
                    }}
                    placeholder="Type instruction..."
                    className="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                    autoFocus
                  />
                  <button
                    onClick={() => addMarker(markerInput)}
                    disabled={!markerInput.trim()}
                    className={clsx(
                      'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                      markerInput.trim()
                        ? 'bg-purple-600 text-white hover:bg-purple-700'
                        : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    )}
                  >
                    Add
                  </button>
                </div>
                {markerInput.trim() && !instructionPalette.includes(markerInput.trim()) && (
                  <button
                    onClick={() => addToPalette(markerInput)}
                    className="mt-2 text-xs text-blue-600 hover:text-blue-800"
                  >
                    + Add to palette
                  </button>
                )}
              </div>
            </div>

            <div className="px-4 py-3 border-t bg-gray-50 rounded-b-xl text-xs text-gray-400">
              Press Escape to cancel
            </div>
          </div>
        </div>
      )}

      {/* Keyboard shortcuts help modal */}
      {showHelpModal && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setShowHelpModal(false)}
        >
          <div
            className="bg-white rounded-xl shadow-xl w-[500px] max-w-full mx-4 max-h-[80vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 border-b bg-gray-50 rounded-t-xl">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-800">Keyboard Shortcuts</h3>
                <button onClick={() => setShowHelpModal(false)} className="text-gray-400 hover:text-gray-600">
                  <MdClose size={20} />
                </button>
              </div>
            </div>
            <div className="p-4 overflow-y-auto max-h-[60vh]">
              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Playback</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between"><span className="text-gray-600">Play / Pause</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">Space</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">1 Frame Backward</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">&larr;</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">1 Frame Forward</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">&rarr;</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">5 Seconds Back</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">Shift + &larr;</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">5 Seconds Forward</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">Shift + &rarr;</kbd></div>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">A-B Loop</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between"><span className="text-gray-600">Set A/B Point</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">A</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">Clear Loop</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">Backspace</kbd></div>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Task Markers</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between"><span className="text-gray-600">Add Marker</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">M</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">Delete Nearest Marker</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">D</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">Jump to Marker #N</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">1-9</kbd></div>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Navigation</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between"><span className="text-gray-600">Previous ROSbag</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">&uarr;</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">Next ROSbag</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">&darr;</kbd></div>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Trim Points</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between"><span className="text-gray-600">Set Start Point</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">S</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">Set End Point</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">E</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">Mark Exclude Region</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">X</kbd><span className="text-xs text-gray-400">(2x)</span></div>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Other</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between"><span className="text-gray-600">Show This Help</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">?</kbd></div>
                    <div className="flex justify-between"><span className="text-gray-600">Close Dialog</span><kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs">Esc</kbd></div>
                  </div>
                </div>
              </div>
            </div>
            <div className="px-4 py-3 border-t bg-gray-50 rounded-b-xl text-xs text-gray-400 text-center">
              Press Escape or click outside to close
            </div>
          </div>
        </div>
      )}

      {/* Trim Start dialog */}
      {showTrimStartDialog && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setShowTrimStartDialog(false)}
        >
          <div
            className="bg-white rounded-xl shadow-xl w-96 max-w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 border-b bg-green-50 rounded-t-xl">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-800">Set Start Point</h3>
                <button onClick={() => setShowTrimStartDialog(false)} className="text-gray-400 hover:text-gray-600">
                  <MdClose size={20} />
                </button>
              </div>
              <p className="text-sm text-gray-500 mt-1">
                Time: {formatTime(currentTime)} (Frame: {Math.round(currentTime * (videoFps[0] || 30))})
              </p>
            </div>
            <div className="p-4">
              {instructionPalette.length > 0 && (
                <div className="mb-4">
                  <p className="text-xs text-gray-500 mb-2">Quick select from palette:</p>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {instructionPalette.slice(0, 9).map((instruction, idx) => (
                      <button
                        key={`trim-quick-${idx}`}
                        onClick={() => applyTrimStart(instruction)}
                        className="w-full flex items-center gap-2 p-2 text-left hover:bg-green-50 rounded transition-colors"
                      >
                        <span className="w-5 h-5 flex items-center justify-center bg-green-100 text-green-700 text-xs font-bold rounded">
                          {idx + 1}
                        </span>
                        <span className="text-sm text-gray-700 truncate">{instruction}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div>
                <p className="text-xs text-gray-500 mb-2">Or enter instruction for this segment:</p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={trimStartInstruction}
                    onChange={(e) => setTrimStartInstruction(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        applyTrimStart(trimStartInstruction);
                      }
                    }}
                    placeholder="e.g., Pick up the cube"
                    className="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
                    autoFocus
                  />
                  <button
                    onClick={() => applyTrimStart(trimStartInstruction)}
                    className="px-4 py-2 rounded-lg text-sm font-medium bg-green-600 text-white hover:bg-green-700 transition-colors"
                  >
                    Set
                  </button>
                </div>
              </div>
            </div>
            <div className="px-4 py-3 border-t bg-gray-50 rounded-b-xl text-xs text-gray-400">
              Press Escape to cancel, Enter to confirm
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ReplayPage;
