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
import { MdFolder, MdRefresh, MdDashboard, MdList } from 'react-icons/md';
import toast from 'react-hot-toast';
import clsx from 'clsx';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import FileBrowserModal from '../components/FileBrowserModal';
import { prepareChartData } from '../utils/chartUtils';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import {
  setSelectedBagPath,
  setLoading,
  setReplayData,
  setError,
  setCurrentTime,
  setIsPlaying,
  setIsVideoLoaded,
  setVideoLoadProgress,
  resetReplayState,
} from '../features/replay/replaySlice';
import {
  PANEL_IDS,
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

const rosbagNameCollator = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: 'base',
});

const sortRosbags = (rosbags) => (
  [...(rosbags || [])].sort((a, b) => rosbagNameCollator.compare(a.name || '', b.name || ''))
);

const basename = (path) => {
  if (!path) return '';
  return String(path).split('/').filter(Boolean).pop() || path;
};

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
    videoSegments,
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
    // Extended metadata
    robotType,
    recordingDate,
    fileSizeBytes,
    segments: replaySegments,
    frameCounts,
    // MCAP direct streaming
    hasRawImages,
    mcapFile,
    // Recording format v2 transcode state — non-"done" means the
    // source MP4 isn't ready (still MJPEG or missing) and Chromium
    // can't decode it in <video>.
    transcodingStatus,
    transcodingCamerasFailed,
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
  const [selectedTaskPath, setSelectedTaskPath] = useState('');
  const [episodeDrawerOpen, setEpisodeDrawerOpen] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // WebGL rendering mode
  const [useWebGL, setUseWebGL] = useState(true);
  const [videoBrightness, setVideoBrightness] = useState(0);
  const [videoContrast, setVideoContrast] = useState(1);

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
    loopStart: null,
    loopEnd: null,
    isActive: isDirectMcapMode,
  });

  // Hidden panels dropdown
  const [showPanelMenu, setShowPanelMenu] = useState(false);

  const PLAYBACK_SPEEDS = [0.5, 1, 1.5, 2, 3];

  // Refs
  const videoRefs = useRef([]);
  const videoCacheRef = useRef(new Map());
  const playbackStateRef = useRef({
    currentTime: 0,
    isPlaying: false,
  });
  const lastVideoTimeDispatchRef = useRef({
    time: -1,
    wallTime: 0,
  });
  useEffect(() => {
    playbackStateRef.current = {
      currentTime,
      isPlaying,
    };
  }, [currentTime, isPlaying]);

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

  const segmentVideoMode = useMemo(
    () => Array.isArray(videoSegments) && videoSegments.some(
      (segment) => Array.isArray(segment.video_files) && segment.video_files.length > 0
    ),
    [videoSegments]
  );

  const findVideoSegmentForTime = useCallback(
    (time) => {
      if (!segmentVideoMode) return null;
      const sorted = [...videoSegments]
        .filter((segment) => Array.isArray(segment.frame_duration) && segment.frame_duration.length === 2)
        .sort((a, b) => Number(a.frame_duration[0]) - Number(b.frame_duration[0]));
      if (sorted.length === 0) return null;
      const clampedTime = Math.max(0, Number(time) || 0);
      return sorted.find((segment, index) => {
        const start = Number(segment.frame_duration[0]) || 0;
        const end = Number(segment.frame_duration[1]) || start;
        const isLast = index === sorted.length - 1;
        return clampedTime >= start && (clampedTime < end || isLast);
      }) || sorted[sorted.length - 1];
    },
    [segmentVideoMode, videoSegments]
  );

  const activeVideoSegment = useMemo(
    () => findVideoSegmentForTime(currentTime),
    [findVideoSegmentForTime, currentTime]
  );

  const globalToVideoTime = useCallback(
    (time, segment = findVideoSegmentForTime(time)) => {
      if (!segmentVideoMode || !segment) return Math.max(0, Math.min(duration, time));
      const start = Number(segment.frame_duration?.[0]) || 0;
      const end = Number(segment.frame_duration?.[1]) || start;
      const localDuration = Math.max(0, end - start);
      return Math.max(0, Math.min(localDuration, (Number(time) || 0) - start));
    },
    [duration, findVideoSegmentForTime, segmentVideoMode]
  );

  const videoTimeToGlobal = useCallback(
    (localTime, segment = activeVideoSegment) => {
      if (!segmentVideoMode || !segment) {
        return Math.max(0, Math.min(duration, localTime));
      }
      const start = Number(segment.frame_duration?.[0]) || 0;
      return Math.max(0, Math.min(duration, start + (Number(localTime) || 0)));
    },
    [activeVideoSegment, duration, segmentVideoMode]
  );

  const currentVideoFiles = useMemo(
    () => (segmentVideoMode && activeVideoSegment
      ? (activeVideoSegment.video_files || [])
      : videoFiles),
    [activeVideoSegment, segmentVideoMode, videoFiles]
  );

  const currentVideoNames = useMemo(
    () => (segmentVideoMode && activeVideoSegment
      ? (activeVideoSegment.video_names || [])
      : videoNames),
    [activeVideoSegment, segmentVideoMode, videoNames]
  );

  const currentVideoFps = useMemo(
    () => (segmentVideoMode && activeVideoSegment
      ? (activeVideoSegment.video_fps || [])
      : videoFps),
    [activeVideoSegment, segmentVideoMode, videoFps]
  );

  const getSegmentVideoUrl = useCallback(
    (file) => (bagPath && file ? `/files${bagPath}/${file}` : ''),
    [bagPath]
  );

  const getVideoSegmentKey = useCallback((segment) => {
    if (!segment) return '';
    return [
      segment.index ?? '',
      segment.name ?? '',
      segment.frame_duration?.[0] ?? '',
      segment.frame_duration?.[1] ?? '',
    ].join(':');
  }, []);

  const activeVideoSegmentKey = useMemo(() => {
    if (!segmentVideoMode || !activeVideoSegment) return '';
    return getVideoSegmentKey(activeVideoSegment);
  }, [activeVideoSegment, getVideoSegmentKey, segmentVideoMode]);

  const currentVideoUrls = useMemo(
    () => {
      if (!bagPath) return [];

      return currentVideoFiles.map((file) => (
        getSegmentVideoUrl(file)
      ));
    },
    [
      bagPath,
      currentVideoFiles,
      getSegmentVideoUrl,
    ]
  );

  const segmentVideoSets = useMemo(() => {
    if (!segmentVideoMode || !Array.isArray(videoSegments)) return [];

    return [...videoSegments]
      .filter((segment) => Array.isArray(segment.video_files) && segment.video_files.length > 0)
      .sort((a, b) => Number(a.frame_duration?.[0] || 0) - Number(b.frame_duration?.[0] || 0))
      .map((segment) => {
        const key = getVideoSegmentKey(segment);
        const files = segment.video_files || [];

        return {
          key,
          index: segment.index,
          name: segment.name,
          frameDuration: segment.frame_duration,
          videoFiles: files,
          videoNames: segment.video_names || [],
          urls: files.map((file) => getSegmentVideoUrl(file)),
        };
      });
  }, [
    getSegmentVideoUrl,
    getVideoSegmentKey,
    segmentVideoMode,
    videoSegments,
  ]);

  const isVideoElementForSegment = useCallback(
    (video, segmentKey = activeVideoSegmentKey) => (
      !segmentVideoMode
      || !segmentKey
      || video?.dataset?.segmentKey === segmentKey
    ),
    [activeVideoSegmentKey, segmentVideoMode]
  );

  const getActiveSegmentLocalEnd = useCallback(
    (metadataLocalEnd, segmentKey = activeVideoSegmentKey, preferredIndex = null) => {
      if (!Number.isFinite(metadataLocalEnd)) return metadataLocalEnd;

      if (preferredIndex !== null) {
        const preferredVideo = videoRefs.current[preferredIndex];
        if (isVideoElementForSegment(preferredVideo, segmentKey)) {
          const preferredDuration = Number(preferredVideo?.duration);
          if (Number.isFinite(preferredDuration) && preferredDuration > 0) {
            return Math.min(metadataLocalEnd, preferredDuration);
          }
        }
      }

      let localEnd = metadataLocalEnd;
      videoRefs.current.forEach((video) => {
        if (!isVideoElementForSegment(video, segmentKey)) return;
        const mediaDuration = Number(video?.duration);
        if (Number.isFinite(mediaDuration) && mediaDuration > 0) {
          localEnd = Math.min(localEnd, mediaDuration);
        }
      });

      return localEnd;
    },
    [activeVideoSegmentKey, isVideoElementForSegment]
  );

  const setVideoElementRef = useCallback(
    (index, element) => {
      videoRefs.current[index] = element;
      if (!element || isDirectMcapMode) return;

      const syncElementToPlayback = () => {
        if (videoRefs.current[index] !== element) return;

        const { currentTime: targetTime, isPlaying: shouldPlay } = playbackStateRef.current;
        const segment = findVideoSegmentForTime(targetTime);
        const targetSegmentKey = segmentVideoMode ? getVideoSegmentKey(segment) : '';
        if (!isVideoElementForSegment(element, targetSegmentKey)) return;

        const localTime = globalToVideoTime(targetTime, segment);
        if (Number.isFinite(localTime)) {
          try {
            if (Math.abs(element.currentTime - localTime) > 0.05) {
              element.currentTime = localTime;
            }
          } catch {
            // Metadata can still be loading while the focused camera view mounts.
          }
        }

        element.playbackRate = playbackSpeed;
        if (shouldPlay) {
          element.play().catch(() => {});
        } else {
          element.pause();
        }
      };

      syncElementToPlayback();
      if (element.readyState < 1) {
        element.addEventListener('loadedmetadata', syncElementToPlayback, { once: true });
      }
    },
    [
      findVideoSegmentForTime,
      getVideoSegmentKey,
      globalToVideoTime,
      isDirectMcapMode,
      isVideoElementForSegment,
      playbackSpeed,
      segmentVideoMode,
    ]
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

  // Start downloading videos when replay data is loaded.
  // Skip if transcoding isn't done yet — those MP4 files would still be
  // raw MJPEG and Chromium can't decode them. The overlay shown in the
  // main content area covers the user-facing side.
  const transcodingReady =
    !transcodingStatus || transcodingStatus === 'done' || transcodingStatus === 'not_required';
  useEffect(() => {
    if (
      isLoaded
      && videoFiles.length > 0
      && videoBlobUrls.length === 0
      && !isDownloading
      && transcodingReady
      && !segmentVideoMode
    ) {
      downloadVideos();
    }
  }, [isLoaded, videoFiles.length, videoBlobUrls.length, isDownloading, downloadVideos, transcodingReady, segmentVideoMode]);

  useEffect(() => {
    if (isLoaded && segmentVideoMode && currentVideoUrls.length > 0 && transcodingReady) {
      dispatch(setIsVideoLoaded(true));
      dispatch(setVideoLoadProgress(100));
    }
  }, [isLoaded, segmentVideoMode, currentVideoUrls.length, transcodingReady, dispatch]);

  useEffect(() => {
    if (!segmentVideoMode) return;
    setExpandedVideoIndex((index) => (
      index === null ? null : Math.max(0, Math.min((currentVideoFiles.length || 1) - 1, index))
    ));
  }, [currentVideoFiles.length, segmentVideoMode]);

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
  const chartCurrentTime = useMemo(
    () => Math.round((Number(currentTime) || 0) * 10) / 10,
    [currentTime]
  );

  // Auto-expand all joints when data loads.
  useEffect(() => {
    if (allJointNames.length > 0) {
      setExpandedJoints(new Set(allJointNames));
    }
  }, [allJointNames]);

  // Prepare state chart data
  const stateChartData = useMemo(() => {
    return prepareChartData(
      jointTimestamps,
      jointNames,
      jointPositions,
      'state_',
      1000
    );
  }, [jointTimestamps, jointNames, jointPositions]);

  // Prepare action chart data
  const actionChartData = useMemo(() => {
    return prepareChartData(
      actionTimestamps,
      actionNames,
      actionValues,
      'action_',
      1000
    );
  }, [actionTimestamps, actionNames, actionValues]);

  // Toggle one joint chart at a time.
  const toggleJoint = useCallback((jointName) => {
    setExpandedJoints((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(jointName)) {
        newSet.delete(jointName);
      } else {
        newSet.add(jointName);
      }
      return newSet;
    });
  }, []);

  const expandAllJoints = useCallback(() => {
    setExpandedJoints(new Set(allJointNames));
  }, [allJointNames]);

  const collapseAllJoints = useCallback(() => {
    setExpandedJoints(new Set());
  }, []);

  const loadEpisode = useCallback(
    async (bag, index, options = {}) => {
      if (!bag?.path) return;

      setCurrentBagIndex(index);
      setVideoBlobUrls([]);
      setDownloadProgress(0);
      setExpandedVideoIndex(null);
      dispatch(setSelectedBagPath(bag.path));
      dispatch(setLoading(true));
      dispatch(setIsVideoLoaded(false));

      try {
        const result = await getReplayData(bag.path);
        if (result.success) {
          dispatch(setReplayData(result));
          if (options.showToast !== false) {
            toast.success(`Loaded episode ${bag.name}`);
          }
        } else {
          dispatch(setError(result.message));
          toast.error(`Failed to load episode: ${result.message}`);
        }
      } catch (err) {
        dispatch(setError(err.message));
        toast.error(`Error loading episode: ${err.message}`);
      }
    },
    [dispatch, getReplayData]
  );

  const handleSelectTaskFolder = useCallback(
    async (path) => {
      if (!path) return;

      setShowFileBrowser(false);
      setSelectedTaskPath(path);
      setEpisodeDrawerOpen(true);
      setRosbagList([]);
      setCurrentBagIndex(-1);
      setVideoBlobUrls([]);
      setDownloadProgress(0);
      dispatch(resetReplayState());
      dispatch(setLoading(true));

      try {
        const listResult = await getRosbagList(path);
        if (!listResult.success) {
          dispatch(setError(listResult.message || 'Failed to load task folder'));
          toast.error(listResult.message || 'Failed to load task folder');
          return;
        }

        const rosbags = sortRosbags(listResult.rosbags || []);
        if (rosbags.length === 0) {
          const message = 'No episodes found. Select the task folder that directly contains numbered episode folders with metadata.yaml.';
          dispatch(setError(message));
          toast.error(message);
          return;
        }

        setRosbagList(rosbags);
        await loadEpisode(rosbags[0], 0, { showToast: false });
        toast.success(`Loaded task: ${basename(path)} (${rosbags.length} episodes)`);
      } catch (err) {
        dispatch(setError(err.message));
        toast.error(`Error loading task folder: ${err.message}`);
      }
    },
    [dispatch, getRosbagList, loadEpisode]
  );

  const handleSelectBag = useCallback(
    async (path) => {
      const index = rosbagList.findIndex((bag) => bag.path === path);
      if (index < 0) return;
      await loadEpisode(rosbagList[index], index);
    },
    [loadEpisode, rosbagList]
  );

  // Navigate to previous/next rosbag
  const navigateRosbag = useCallback(
    async (direction) => {
      if (rosbagList.length === 0 || currentBagIndex === -1) return;
      if (isDownloading) return;

      const newIndex = direction === 'prev' ? currentBagIndex - 1 : currentBagIndex + 1;
      if (newIndex < 0 || newIndex >= rosbagList.length) return;

      const newBag = rosbagList[newIndex];
      await loadEpisode(newBag, newIndex);
    },
    [rosbagList, currentBagIndex, isDownloading, loadEpisode]
  );

  // Handle video events
  useEffect(() => {
    const activeVideoUrlCount = segmentVideoMode ? currentVideoUrls.length : videoBlobUrls.length;
    if (!isVideoLoaded || activeVideoUrlCount === 0) return;

    const videos = videoRefs.current.slice();

    const isCurrentActiveVideo = (eventTarget) => {
      const index = videos.indexOf(eventTarget);
      return index >= 0 && videoRefs.current[index] === eventTarget;
    };

    const getDriverVideoIndex = () => {
      if (expandedVideoIndex !== null) return expandedVideoIndex;
      return videoRefs.current.findIndex((video) => isVideoElementForSegment(video));
    };

    const getDriverVideo = () => {
      const driverIndex = getDriverVideoIndex();
      return driverIndex >= 0 ? videoRefs.current[driverIndex] : null;
    };

    const handleTimeUpdate = (event) => {
      if (!isCurrentActiveVideo(event.currentTarget)) return;
      if (!isVideoElementForSegment(event.currentTarget)) return;

      const video = getDriverVideo();
      if (!video || event.currentTarget !== video) return;

      let nextTime = videoTimeToGlobal(video.currentTime);
      const { currentTime: lastKnownTime, isPlaying: currentlyPlaying } = playbackStateRef.current;
      if (currentlyPlaying && nextTime < lastKnownTime - 1 / 60) return;
      if (!currentlyPlaying && Math.abs(nextTime - lastKnownTime) > 0.5) return;

      let isSegmentBoundaryTransition = false;
      if (segmentVideoMode && activeVideoSegment && currentlyPlaying) {
        const segmentEnd = Number(activeVideoSegment.frame_duration?.[1]);
        const metadataLocalEnd = globalToVideoTime(segmentEnd, activeVideoSegment);
        const driverIndex = getDriverVideoIndex();
        const localEnd = getActiveSegmentLocalEnd(
          metadataLocalEnd,
          activeVideoSegmentKey,
          driverIndex >= 0 ? driverIndex : null
        );
        const fps = (
          currentVideoFps[driverIndex]
          || currentVideoFps[0]
          || videoFps[driverIndex]
          || videoFps[0]
          || 30
        );
        const transitionEpsilon = Math.max(1 / fps, 1 / 60);

        if (
          Number.isFinite(segmentEnd)
          && segmentEnd < duration - 0.001
          && Number.isFinite(localEnd)
          && localEnd - video.currentTime <= transitionEpsilon
        ) {
          nextTime = segmentEnd;
          isSegmentBoundaryTransition = true;
        }
      }

      const now = performance.now();
      const last = lastVideoTimeDispatchRef.current;
      if (
        !isSegmentBoundaryTransition
        &&
        currentlyPlaying
        && Math.abs(nextTime - last.time) < 1 / 30
        && now - last.wallTime < 80
      ) {
        return;
      }

      lastVideoTimeDispatchRef.current = {
        time: nextTime,
        wallTime: now,
      };
      playbackStateRef.current = {
        ...playbackStateRef.current,
        currentTime: nextTime,
      };
      dispatch(setCurrentTime(nextTime));
    };

    const handleEnded = (event) => {
      if (!isCurrentActiveVideo(event.currentTarget)) return;
      if (!isVideoElementForSegment(event.currentTarget)) return;

      const video = getDriverVideo();
      if (video && event.currentTarget !== video) return;

      const segmentEnd = segmentVideoMode && activeVideoSegment
        ? Number(activeVideoSegment.frame_duration?.[1]) || duration
        : duration;
      const nextTime = Math.min(duration, segmentEnd);
      const shouldStop = !segmentVideoMode || segmentEnd >= duration - 0.01;
      lastVideoTimeDispatchRef.current = {
        time: nextTime,
        wallTime: performance.now(),
      };
      playbackStateRef.current = {
        ...playbackStateRef.current,
        currentTime: nextTime,
        isPlaying: shouldStop ? false : playbackStateRef.current.isPlaying,
      };
      dispatch(setCurrentTime(nextTime));
      if (shouldStop) {
        dispatch(setIsPlaying(false));
      }
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
  }, [
    isVideoLoaded,
    videoBlobUrls.length,
    currentVideoUrls.length,
    duration,
    dispatch,
    expandedVideoIndex,
    getActiveSegmentLocalEnd,
    globalToVideoTime,
    isVideoElementForSegment,
    segmentVideoMode,
    activeVideoSegment,
    activeVideoSegmentKey,
    currentVideoFps,
    videoTimeToGlobal,
    videoFps,
  ]);

  useEffect(() => {
    if (!isVideoLoaded || isDirectMcapMode || !segmentVideoMode) return;
    const localTime = globalToVideoTime(currentTime, activeVideoSegment);

    videoRefs.current.forEach((video) => {
      if (!video) return;
      if (!isVideoElementForSegment(video)) return;

      if (!isPlaying && Number.isFinite(localTime) && Math.abs(video.currentTime - localTime) > 0.05) {
        try {
          video.currentTime = localTime;
        } catch {
          // Metadata may still be settling just after the active segment switches.
        }
      }

      video.playbackRate = playbackSpeed;
      if (isPlaying) {
        if (video.paused) {
          video.play().catch(() => {});
        }
      } else if (!video.paused) {
        video.pause();
      }
    });
  }, [
    activeVideoSegment,
    currentVideoUrls.length,
    currentTime,
    globalToVideoTime,
    isDirectMcapMode,
    isPlaying,
    isVideoLoaded,
    isVideoElementForSegment,
    playbackSpeed,
    segmentVideoMode,
    expandedVideoIndex,
  ]);

  const syncVideosToGlobalTime = useCallback(
    (time, playback = 'keep') => {
      const segment = findVideoSegmentForTime(time);
      const segmentKey = segmentVideoMode ? getVideoSegmentKey(segment) : '';
      const localTime = globalToVideoTime(time, segment);
      playbackStateRef.current = {
        currentTime: time,
        isPlaying: playback === 'play'
          ? true
          : playback === 'pause'
            ? false
            : playbackStateRef.current.isPlaying,
      };
      videoRefs.current.forEach((video) => {
        if (!video) return;
        if (!isVideoElementForSegment(video, segmentKey)) return;
        video.currentTime = localTime;
        video.playbackRate = playbackSpeed;
        if (playback === 'play') {
          video.play().catch(() => {});
        } else if (playback === 'pause') {
          video.pause();
        }
      });
    },
    [
      findVideoSegmentForTime,
      getVideoSegmentKey,
      globalToVideoTime,
      isVideoElementForSegment,
      playbackSpeed,
      segmentVideoMode,
    ]
  );

  // Unified seek-and-play: seeks to time and starts playback (handles both MCAP and video modes)
  const seekAndPlay = useCallback((time) => {
    const clamped = Math.max(0, Math.min(duration, time));
    if (isDirectMcapMode) {
      mcapPlayer.syncToTime(clamped);
      if (!isPlaying) {
        mcapPlayer.playAll();
      }
    } else {
      syncVideosToGlobalTime(clamped, 'play');
      dispatch(setCurrentTime(clamped));
      dispatch(setIsPlaying(true));
    }
  }, [duration, isDirectMcapMode, isPlaying, mcapPlayer, syncVideosToGlobalTime, dispatch]);

  // Unified seek-to-time: seeks without changing play state (handles both MCAP and video modes)
  const seekToTime = useCallback((time) => {
    const clamped = Math.max(0, Math.min(duration, time));
    if (isDirectMcapMode) {
      mcapPlayer.syncToTime(clamped);
    } else {
      syncVideosToGlobalTime(clamped);
      dispatch(setCurrentTime(clamped));
    }
  }, [duration, isDirectMcapMode, mcapPlayer, syncVideosToGlobalTime, dispatch]);

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
      playbackStateRef.current = {
        ...playbackStateRef.current,
        isPlaying: false,
      };
      dispatch(setIsPlaying(false));
    } else {
      const targetTime = currentTime >= duration ? 0 : currentTime;
      syncVideosToGlobalTime(targetTime, 'play');
      dispatch(setCurrentTime(targetTime));
      dispatch(setIsPlaying(true));
    }
  }, [currentTime, duration, isPlaying, isDirectMcapMode, mcapPlayer, syncVideosToGlobalTime, dispatch]);

  // Restart playback
  const restartPlayback = useCallback(() => {
    if (isDirectMcapMode) {
      mcapPlayer.restart();
      return;
    }

    syncVideosToGlobalTime(0, 'play');
    dispatch(setCurrentTime(0));
    dispatch(setIsPlaying(true));
  }, [isDirectMcapMode, mcapPlayer, syncVideosToGlobalTime, dispatch]);

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
        playbackStateRef.current = {
          ...playbackStateRef.current,
          isPlaying: false,
        };
        dispatch(setIsPlaying(false));
      }

      const fpsIndex = expandedVideoIndex ?? 0;
      const fps = (
        currentVideoFps[fpsIndex]
        || currentVideoFps[0]
        || videoFps[fpsIndex]
        || videoFps[0]
        || 30
      );
      const frameTime = 1 / fps;
      const delta = direction === 'forward' ? frameTime : -frameTime;
      const newTime = Math.max(0, Math.min(duration, currentTime + delta));

      syncVideosToGlobalTime(newTime);
      dispatch(setCurrentTime(newTime));
    },
    [isVideoLoaded, isPlaying, isDirectMcapMode, mcapPlayer, currentVideoFps, videoFps, expandedVideoIndex, duration, currentTime, syncVideosToGlobalTime, dispatch]
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
      syncVideosToGlobalTime(newTime);
      dispatch(setCurrentTime(newTime));
    },
    [isVideoLoaded, isDirectMcapMode, mcapPlayer, duration, currentTime, syncVideosToGlobalTime, dispatch]
  );

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
  }, [videoBlobUrls, currentVideoUrls, playbackSpeed]);

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

    syncVideosToGlobalTime(newTime);
    dispatch(setCurrentTime(newTime));
  }, [isVideoLoaded, isDirectMcapMode, mcapPlayer, duration, syncVideosToGlobalTime, dispatch]);

  // Handle chart click to seek
  const handleChartSeek = useCallback((time) => {
    if ((!isVideoLoaded && !isDirectMcapMode) || typeof time !== 'number') return;

    const clampedTime = Math.max(0, Math.min(duration, time));

    if (isDirectMcapMode) {
      mcapPlayer.syncToTime(clampedTime);
      return;
    }

    syncVideosToGlobalTime(clampedTime);
    dispatch(setCurrentTime(clampedTime));
  }, [isVideoLoaded, isDirectMcapMode, mcapPlayer, duration, syncVideosToGlobalTime, dispatch]);

  // Extract short name from video file path
  const getShortVideoName = useCallback((filePath) => {
    const fileName = filePath.split('/').pop();
    return fileName
      .replace('.mp4', '')
      .replace('_compressed', '')
      .split('_')
      .slice(-3)
      .join('_');
  }, []);

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
    setExpandedVideoIndex(null);
  }, [selectedBagPath]);

  // Hidden panels list
  const hiddenPanels = Object.values(layoutPanels).filter(
    (p) => !p.visible && p.id !== PANEL_IDS.SIDEBAR
  );
  const hasExpandedReplayPanel = Object.values(layoutPanels).some(
    (p) => p.visible && p.expandable !== false && p.expanded
  );

  return (
    <div className="relative flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 flex-shrink-0">
        <div className="min-w-0">
          <h1 className="text-xl font-bold text-gray-800">Replay Viewer</h1>
          {selectedTaskPath && (
            <div className="text-xs text-gray-500 truncate max-w-xl" title={selectedTaskPath}>
              {basename(selectedTaskPath)}
            </div>
          )}
        </div>
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
          {rosbagList.length > 0 && (
            <button
              onClick={() => setEpisodeDrawerOpen((open) => !open)}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors text-sm',
                episodeDrawerOpen
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              )}
              title="Show or hide episodes"
            >
              <MdList size={16} />
              Episodes {currentBagIndex + 1 > 0 ? currentBagIndex + 1 : 0}/{rosbagList.length}
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
            Select Task
          </button>
        </div>
      </div>

      {/* Recording-format-v2 transcode gate: refuse to play until the
          background MJPEG → H.264 pass finishes. Without this guard the
          <video> tag would receive raw MJPEG which Chromium can't
          decode (black screen + no error). */}
      {isLoaded && transcodingStatus && transcodingStatus !== 'done' && transcodingStatus !== 'not_required' ? (
        <div className="flex-1 flex items-center justify-center px-4">
          <div className="max-w-xl w-full bg-gray-900/90 border border-gray-700 rounded-lg p-6 text-center">
            {transcodingStatus === 'pending' || transcodingStatus === 'running' ? (
              <>
                <div className="mx-auto mb-4 w-10 h-10 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                <h3 className="text-white text-lg font-medium mb-2">Transcoding videos…</h3>
                <p className="text-gray-400 text-sm">
                  This episode's camera files are still being re-encoded to H.264.
                  Replay will become available once the background task finishes — usually
                  within a minute per recorded minute on the Jetson. Re-open this bag in a
                  bit, or pick another one in the meantime.
                </p>
              </>
            ) : transcodingStatus === 'failed' ? (
              <>
                <h3 className="text-red-400 text-lg font-medium mb-2">Transcode failed</h3>
                <p className="text-gray-400 text-sm mb-3">
                  The background H.264 transcode didn't finish for this episode, so the
                  source MP4s aren't browser-playable.
                </p>
                {transcodingCamerasFailed && Object.keys(transcodingCamerasFailed).length > 0 && (
                  <ul className="text-left text-xs text-gray-300 bg-black/40 rounded p-2 mb-3">
                    {Object.entries(transcodingCamerasFailed).map(([cam, err]) => (
                      <li key={cam}><span className="text-red-300">{cam}:</span> {String(err).slice(0, 200)}</li>
                    ))}
                  </ul>
                )}
                <p className="text-gray-500 text-xs">
                  Check the cyclo_data logs, fix the cause, and re-trigger the transcode
                  (or delete the episode_info.json's transcoding_status to retry from raw).
                </p>
              </>
            ) : (
              <>
                <h3 className="text-yellow-400 text-lg font-medium mb-2">
                  Unexpected transcode state: {transcodingStatus}
                </h3>
                <p className="text-gray-400 text-sm">
                  This episode's transcoding_status is not a value we know how to handle.
                  Inspect <code>episode_info.json</code> for details.
                </p>
              </>
            )}
          </div>
        </div>
      ) : isLoaded && (currentVideoFiles.length > 0 || isDirectMcapMode) ? (
        <>
          <div className="flex-1 px-4 min-h-0 overflow-hidden">
            <ReplayLayoutContainer
              sidebarMode="drawer"
              cameraPanelContent={
                <CameraPanel
                  isDirectMcapMode={isDirectMcapMode}
                  isLoaded={isLoaded}
                  isLoading={isLoading}
                  error={error}
                  videoFiles={currentVideoFiles}
                  videoNames={currentVideoNames}
                  videoBlobUrls={segmentVideoMode ? currentVideoUrls : videoBlobUrls}
                  segmentVideoSets={segmentVideoSets}
                  activeVideoSegmentKey={activeVideoSegmentKey}
                  videoRefs={videoRefs}
                  setVideoRef={setVideoElementRef}
                  expandedVideoIndex={expandedVideoIndex}
                  setExpandedVideoIndex={setExpandedVideoIndex}
                  mcapPlayer={mcapPlayer}
                  useWebGL={useWebGL}
                  videoBrightness={videoBrightness}
                  videoContrast={videoContrast}
                  isDownloading={segmentVideoMode ? false : isDownloading}
                  downloadProgress={segmentVideoMode ? 100 : downloadProgress}
                  getShortVideoName={getShortVideoName}
                />
              }
              viewer3DPanelContent={
                <Viewer3DPanel
                  robotType={robotType}
                  jointTimestamps={jointTimestamps}
                  jointNames={jointNames}
                  jointPositions={jointPositions}
                  actionTimestamps={actionTimestamps}
                  actionNames={actionNames}
                  actionValues={actionValues}
                  currentTime={currentTime}
                  isPlaying={isPlaying}
                  playbackSpeed={playbackSpeed}
                />
              }
              jointDataPanelContent={
                <JointDataPanel
                  allJointNames={allJointNames}
                  stateChartData={stateChartData}
                  actionChartData={actionChartData}
                  currentTime={chartCurrentTime}
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
            replaySegments={replaySegments}
            handleSeek={handleSeek}
            togglePlayPause={togglePlayPause}
            restartPlayback={restartPlayback}
            changePlaybackSpeed={changePlaybackSpeed}
            playbackSpeed={playbackSpeed}
            PLAYBACK_SPEEDS={PLAYBACK_SPEEDS}
            videoFiles={currentVideoFiles}
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
              <p>Select a task folder to start viewing episodes</p>
            </div>
          )}
        </div>
      )}

      {episodeDrawerOpen && rosbagList.length > 0 && (
        <div
          className={clsx(
            'absolute right-4 top-14 bottom-24 w-[340px] max-w-[calc(100vw-2rem)] rounded-lg border border-gray-200 bg-white shadow-xl overflow-hidden',
            hasExpandedReplayPanel ? 'z-0' : 'z-30'
          )}
        >
          <SidebarPanel
            taskPath={selectedTaskPath}
            recordingDate={recordingDate}
            robotType={robotType}
            fileSizeBytes={fileSizeBytes}
            duration={duration}
            frameCounts={frameCounts}
            replaySegments={replaySegments}
            currentTime={currentTime}
            videoCount={currentVideoFiles.length}
            jointCount={jointNames.length}
            actionCount={actionNames.length}
            rosbagList={rosbagList}
            currentBagIndex={currentBagIndex}
            isBusy={isDownloading || isLoading}
            navigateRosbag={navigateRosbag}
            handleSelectBag={handleSelectBag}
            seekAndPlay={seekAndPlay}
            seekToTime={seekToTime}
            onClose={() => setEpisodeDrawerOpen(false)}
          />
        </div>
      )}

      {/* Bag info */}
      {selectedBagPath && (
        <div className="px-4 py-1 text-xs text-gray-500 flex-shrink-0 flex items-center gap-3 min-w-0">
          {selectedTaskPath && (
            <span className="truncate">
              <span className="font-medium">Task:</span> {selectedTaskPath}
            </span>
          )}
          <span className="truncate">
            <span className="font-medium">Episode:</span> {selectedBagPath}
          </span>
        </div>
      )}

      {/* File browser modal */}
      {showFileBrowser && (
        <FileBrowserModal
          isOpen={showFileBrowser}
          onClose={() => setShowFileBrowser(false)}
          onFileSelect={(item) => handleSelectTaskFolder(item.full_path)}
          title="Select Task Folder"
          selectButtonText="Open Task"
          allowDirectorySelect={true}
          allowFileSelect={false}
          initialPath="/workspace/rosbag2"
          defaultPath="/workspace/rosbag2"
        />
      )}
    </div>
  );
}

export default ReplayPage;
