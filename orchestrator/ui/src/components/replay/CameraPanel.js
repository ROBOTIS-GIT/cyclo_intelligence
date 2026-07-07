import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { MdRefresh } from 'react-icons/md';
import { WebGLVideoPanel } from './WebGLVideoPanel';

const clampFocusIndex = (index, count) => {
  if (count <= 0) return null;
  const numericIndex = Number.isInteger(index) ? index : 0;
  return Math.max(0, Math.min(count - 1, numericIndex));
};

const mosaicGridClass = (count) => {
  if (count <= 1) return 'grid h-full w-full gap-2 p-2';
  if (count === 2) return 'grid h-full w-full gap-2 p-2 md:grid-cols-2';
  if (count <= 4) return 'grid h-full w-full gap-2 p-2 md:grid-cols-2';
  return 'grid h-full w-full gap-2 overflow-y-auto p-2 md:grid-cols-2 xl:grid-cols-3';
};

const VIDEO_SURFACE_CLASS = 'block h-full w-full min-h-0 min-w-0 bg-gray-950 object-contain';
const CANVAS_SURFACE_CLASS = 'block h-full w-full min-h-0 min-w-0 bg-gray-900 object-contain';
const MOSAIC_BUTTON_CLASS = 'relative block min-h-0 min-w-0 overflow-hidden rounded-lg bg-gray-950 p-0 text-left shadow ring-1 ring-black/5 transition hover:ring-2 hover:ring-blue-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400';
const FOCUSED_BUTTON_CLASS = 'relative flex h-full min-h-0 min-w-0 items-center justify-center overflow-hidden rounded-lg bg-gray-950 p-0 text-left shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400';
const THUMB_BUTTON_CLASS = 'relative block min-h-[92px] min-w-0 overflow-hidden rounded-md bg-gray-950 p-0 text-left shadow ring-1 ring-black/5 transition hover:ring-2 hover:ring-blue-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400';
const focusedGridClass = (hasThumbnails) => (
  hasThumbnails
    ? 'grid h-full w-full min-h-0 min-w-0 gap-2 p-2 lg:grid-cols-[minmax(0,1fr)_minmax(180px,24%)]'
    : 'grid h-full w-full min-h-0 min-w-0 gap-2 p-2'
);

function CameraPanel({
  // Mode detection
  isDirectMcapMode,
  isLoaded,
  isLoading,
  error,
  // Video files mode
  videoFiles,
  videoNames,
  videoBlobUrls,
  segmentVideoSets = [],
  activeVideoSegmentKey = '',
  videoRefs,
  setVideoRef,
  expandedVideoIndex,
  setExpandedVideoIndex,
  // MCAP mode
  mcapPlayer,
  useWebGL,
  videoBrightness,
  videoContrast,
  // Download state
  isDownloading,
  downloadProgress,
  // Helpers
  getShortVideoName,
}) {
  const segmentVideoRefs = useRef(new Map());
  const previousActiveSegmentKeyRef = useRef(activeVideoSegmentKey);
  const [transitionCoverSegmentKey, setTransitionCoverSegmentKey] = useState('');
  const hasSegmentVideoSets = !isDirectMcapMode && segmentVideoSets.length > 0;

  const setVideoElement = useCallback((index, element) => {
    if (setVideoRef) {
      setVideoRef(index, element);
    } else if (videoRefs) {
      videoRefs.current[index] = element;
    }
  }, [setVideoRef, videoRefs]);

  const setSegmentVideoElement = useCallback((segmentKey, index, element) => {
    const id = `${segmentKey}@@${index}`;
    if (element) {
      segmentVideoRefs.current.set(id, { segmentKey, index, element });
      if (segmentKey === activeVideoSegmentKey) {
        setVideoElement(index, element);
      } else {
        element.pause();
      }
    } else {
      const existing = segmentVideoRefs.current.get(id);
      segmentVideoRefs.current.delete(id);
      if (
        existing
        && (!videoRefs || videoRefs.current[index] === existing.element)
      ) {
        setVideoElement(index, null);
      }
    }
  }, [activeVideoSegmentKey, setVideoElement, videoRefs]);

  const warmSegmentVideoElement = useCallback((element) => {
    if (!element || !element.src || element.readyState > 0) return;

    const src = element.currentSrc || element.src;
    if (element.dataset.warmedSrc === src) return;

    element.dataset.warmedSrc = src;
    try {
      element.load();
    } catch {
      // Browsers may reject load() while the media pipeline is settling.
    }
  }, []);

  useEffect(() => {
    if (!hasSegmentVideoSets) return;

    segmentVideoRefs.current.forEach(({ element }) => {
      warmSegmentVideoElement(element);
    });
  }, [hasSegmentVideoSets, segmentVideoSets, warmSegmentVideoElement]);

  useEffect(() => {
    if (!hasSegmentVideoSets || !activeVideoSegmentKey) {
      previousActiveSegmentKeyRef.current = activeVideoSegmentKey;
      setTransitionCoverSegmentKey('');
      return undefined;
    }

    const previousKey = previousActiveSegmentKeyRef.current;
    if (previousKey && previousKey !== activeVideoSegmentKey) {
      setTransitionCoverSegmentKey(previousKey);
      const timeoutId = window.setTimeout(() => {
        setTransitionCoverSegmentKey((currentKey) => (
          currentKey === previousKey ? '' : currentKey
        ));
      }, 90);

      previousActiveSegmentKeyRef.current = activeVideoSegmentKey;
      return () => window.clearTimeout(timeoutId);
    }

    previousActiveSegmentKeyRef.current = activeVideoSegmentKey;
    return undefined;
  }, [activeVideoSegmentKey, hasSegmentVideoSets]);

  useLayoutEffect(() => {
    if (!hasSegmentVideoSets) return;

    const activeIndexes = new Set();
    segmentVideoRefs.current.forEach(({ segmentKey, index, element }) => {
      if (!element) return;
      if (segmentKey === activeVideoSegmentKey) {
        setVideoElement(index, element);
        activeIndexes.add(index);
      } else {
        element.pause();
      }
    });

    if (videoRefs) {
      videoRefs.current.forEach((element, index) => {
        if (!element) return;
        if (
          !activeIndexes.has(index)
          || element.dataset?.segmentKey !== activeVideoSegmentKey
        ) {
          setVideoElement(index, null);
        }
      });
    }
  }, [
    activeVideoSegmentKey,
    expandedVideoIndex,
    hasSegmentVideoSets,
    segmentVideoSets,
    setVideoElement,
    videoRefs,
  ]);

  if (!isLoaded || (videoFiles.length === 0 && !isDirectMcapMode)) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500 bg-gray-50 rounded-lg h-full">
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
          <div className="text-center text-gray-400 text-sm">
            No camera data
          </div>
        )}
      </div>
    );
  }

  const renderCameraLabel = (name, extraClassName = '') => (
    <div className={`absolute left-2 top-2 z-10 max-w-[calc(100%-1rem)] truncate rounded bg-black/65 px-2 py-1 text-xs font-medium text-white ${extraClassName}`}>
      {name}
    </div>
  );

  const mcapTopics = mcapPlayer?.imageTopics || [];
  const mcapFocusIndex = clampFocusIndex(expandedVideoIndex, mcapTopics.length);
  const activeSegmentSet = hasSegmentVideoSets
    ? (segmentVideoSets.find((segment) => segment.key === activeVideoSegmentKey) || segmentVideoSets[0])
    : null;
  const activeVideoFiles = activeSegmentSet?.videoFiles || videoFiles;
  const activeVideoNames = activeSegmentSet?.videoNames || videoNames;
  const videoFocusIndex = clampFocusIndex(expandedVideoIndex, activeVideoFiles.length);
  const focusedVideoName = videoFocusIndex !== null
    ? (activeVideoNames[videoFocusIndex] || getShortVideoName(activeVideoFiles[videoFocusIndex]))
    : '';
  const isFocusedView = expandedVideoIndex !== null;

  const renderSegmentVideo = (
    segmentKey,
    index,
    src,
    className,
    preload = 'metadata'
  ) => (
    <video
      ref={(el) => setSegmentVideoElement(segmentKey, index, el)}
      src={src || ''}
      data-segment-key={segmentKey}
      className={className}
      preload={preload}
      playsInline
      muted
    />
  );

  const renderSegmentMosaic = (segmentSet, isActive) => (
    <div className={mosaicGridClass(segmentSet.videoFiles.length)} style={{ minHeight: 0, gridAutoRows: 'minmax(0, 1fr)' }}>
      {segmentSet.videoFiles.map((file, index) => {
        const name = segmentSet.videoNames[index] || getShortVideoName(file);
        return (
          <button
            key={`${segmentSet.key}-${file}-${index}`}
            type="button"
            className={MOSAIC_BUTTON_CLASS}
            onClick={() => setExpandedVideoIndex(index)}
            title={name}
            tabIndex={isActive ? 0 : -1}
          >
            {renderCameraLabel(name)}
            {renderSegmentVideo(
              segmentSet.key,
              index,
              segmentSet.urls[index],
              VIDEO_SURFACE_CLASS,
              'auto'
            )}
          </button>
        );
      })}
    </div>
  );

  const renderSegmentFocused = (segmentSet, isActive) => {
    const focusIndex = clampFocusIndex(expandedVideoIndex, segmentSet.videoFiles.length);
    const focusName = focusIndex !== null
      ? (segmentSet.videoNames[focusIndex] || getShortVideoName(segmentSet.videoFiles[focusIndex]))
      : '';

    return (
      <div
        className={focusedGridClass(segmentSet.videoFiles.length > 1)}
        style={{ minHeight: 0, gridTemplateRows: 'minmax(0, 1fr)' }}
      >
        {focusIndex !== null && (
          <button
            type="button"
            className={FOCUSED_BUTTON_CLASS}
            onClick={() => setExpandedVideoIndex(null)}
            title={focusName}
            tabIndex={isActive ? 0 : -1}
          >
            {renderCameraLabel(focusName)}
            {renderSegmentVideo(
              segmentSet.key,
              focusIndex,
              segmentSet.urls[focusIndex],
              VIDEO_SURFACE_CLASS,
              'auto'
            )}
          </button>
        )}

        {segmentSet.videoFiles.length > 1 && (
          <div className="grid h-full min-h-0 min-w-0 gap-2 overflow-y-auto pr-0.5" style={{ gridAutoRows: 'minmax(92px, 1fr)' }}>
            {segmentSet.videoFiles.map((file, index) => {
              if (index === focusIndex) return null;
              const name = segmentSet.videoNames[index] || getShortVideoName(file);
              return (
                <button
                  key={`${segmentSet.key}-${file}-${index}`}
                  type="button"
                  className={THUMB_BUTTON_CLASS}
                  onClick={() => setExpandedVideoIndex(index)}
                  title={name}
                  tabIndex={isActive ? 0 : -1}
                >
                  {renderCameraLabel(name, 'text-[10px] px-1.5 py-0.5')}
                  {renderSegmentVideo(
                    segmentSet.key,
                    index,
                    segmentSet.urls[index],
                    VIDEO_SURFACE_CLASS,
                    'auto'
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="relative w-full h-full bg-gray-100 flex flex-col" style={{ minHeight: 0, minWidth: 0 }}>
      {/* MCAP Direct Streaming Mode */}
      {isDirectMcapMode ? (
        <div className="flex-1 min-h-0 relative">
          {mcapPlayer.isLoading && (
            <div className="absolute inset-0 flex items-center justify-center z-20 bg-gray-100 bg-opacity-80">
              <div className="text-center">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto mb-2" />
                <p className="text-gray-600 text-sm">Indexing MCAP...</p>
              </div>
            </div>
          )}
          {mcapPlayer.mcapError && (
            <div className="absolute inset-0 flex items-center justify-center z-20">
              <div className="text-center text-red-600">
                <p className="font-medium">Failed to load MCAP</p>
                <p className="text-sm mt-1">{mcapPlayer.mcapError}</p>
              </div>
            </div>
          )}
          {!isFocusedView ? (
            <div className={mosaicGridClass(mcapTopics.length)} style={{ gridAutoRows: 'minmax(0, 1fr)' }}>
              {mcapTopics.map((topicInfo, index) => (
                <button
                  key={topicInfo.topic}
                  type="button"
                  className={MOSAIC_BUTTON_CLASS}
                  onClick={() => setExpandedVideoIndex(index)}
                  title={topicInfo.cameraName}
                >
                  {renderCameraLabel(topicInfo.cameraName)}
                  {useWebGL ? (
                    <WebGLVideoPanel
                      ref={(handle) => mcapPlayer.setWebGLPanelRef(index, handle)}
                      brightness={videoBrightness}
                      contrast={videoContrast}
                      className={CANVAS_SURFACE_CLASS}
                    />
                  ) : (
                    <canvas
                      ref={(el) => mcapPlayer.setCanvasRef(index, el)}
                      className={CANVAS_SURFACE_CLASS}
                      style={{ imageRendering: 'auto' }}
                    />
                  )}
                </button>
              ))}
            </div>
          ) : (
            <div
              className={focusedGridClass(mcapTopics.length > 1)}
              style={{ gridTemplateRows: 'minmax(0, 1fr)' }}
            >
              {mcapFocusIndex !== null && (
                <button
                  type="button"
                  className={FOCUSED_BUTTON_CLASS}
                  onClick={() => setExpandedVideoIndex(null)}
                  title={mcapTopics[mcapFocusIndex].cameraName}
                >
                  {renderCameraLabel(mcapTopics[mcapFocusIndex].cameraName)}
                  {useWebGL ? (
                    <WebGLVideoPanel
                      ref={(handle) => mcapPlayer.setWebGLPanelRef(mcapFocusIndex, handle)}
                      brightness={videoBrightness}
                      contrast={videoContrast}
                      className={CANVAS_SURFACE_CLASS}
                    />
                  ) : (
                    <canvas
                      ref={(el) => mcapPlayer.setCanvasRef(mcapFocusIndex, el)}
                      className={CANVAS_SURFACE_CLASS}
                      style={{ imageRendering: 'auto' }}
                    />
                  )}
                </button>
              )}

              {mcapTopics.length > 1 && (
                <div className="grid h-full min-h-0 min-w-0 gap-2 overflow-y-auto pr-0.5" style={{ gridAutoRows: 'minmax(92px, 1fr)' }}>
                  {mcapTopics.map((topicInfo, index) => {
                    if (index === mcapFocusIndex) return null;
                    return (
                      <button
                        key={topicInfo.topic}
                        type="button"
                        className={THUMB_BUTTON_CLASS}
                        onClick={() => setExpandedVideoIndex(index)}
                        title={topicInfo.cameraName}
                      >
                        {renderCameraLabel(topicInfo.cameraName, 'text-[10px] px-1.5 py-0.5')}
                        {useWebGL ? (
                          <WebGLVideoPanel
                            ref={(handle) => mcapPlayer.setWebGLPanelRef(index, handle)}
                            brightness={videoBrightness}
                            contrast={videoContrast}
                            className={CANVAS_SURFACE_CLASS}
                          />
                        ) : (
                          <canvas
                            ref={(el) => mcapPlayer.setCanvasRef(index, el)}
                            className={CANVAS_SURFACE_CLASS}
                            style={{ imageRendering: 'auto' }}
                          />
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      ) : hasSegmentVideoSets ? (
        <div className="relative flex-1 min-h-0 overflow-hidden">
          {segmentVideoSets.map((segmentSet) => {
            const isActive = segmentSet.key === activeVideoSegmentKey;
            const immediateCoverSegmentKey = (
              previousActiveSegmentKeyRef.current
              && previousActiveSegmentKeyRef.current !== activeVideoSegmentKey
            )
              ? previousActiveSegmentKeyRef.current
              : '';
            const isTransitionCover = (
              segmentSet.key === transitionCoverSegmentKey
              || segmentSet.key === immediateCoverSegmentKey
            );
            const shouldRenderSegment = isActive || isTransitionCover;
            return (
              <div
                key={segmentSet.key}
                className={`absolute inset-0 ${
                  isActive
                    ? 'visible z-10 opacity-100'
                    : isTransitionCover
                      ? 'visible pointer-events-none z-20 opacity-100'
                    : 'invisible pointer-events-none z-0 opacity-0'
                }`}
                aria-hidden={!isActive}
              >
                {shouldRenderSegment && (
                  isFocusedView
                    ? renderSegmentFocused(segmentSet, isActive)
                    : renderSegmentMosaic(segmentSet, isActive)
                )}
              </div>
            );
          })}
        </div>
      ) : (
        !isFocusedView ? (
          <div className={mosaicGridClass(videoFiles.length)} style={{ minHeight: 0, gridAutoRows: 'minmax(0, 1fr)' }}>
            {videoFiles.map((file, index) => {
              const name = videoNames[index] || getShortVideoName(file);
              return (
                <button
                  key={`${file}-${index}`}
                  type="button"
                  className={MOSAIC_BUTTON_CLASS}
                  onClick={() => setExpandedVideoIndex(index)}
                  title={name}
                >
                  {renderCameraLabel(name)}
                  <video
                    ref={(el) => setVideoElement(index, el)}
                    src={videoBlobUrls[index] || ''}
                    className={VIDEO_SURFACE_CLASS}
                    playsInline
                    muted={index > 0}
                  />
                </button>
              );
            })}
          </div>
        ) : (
          <div
            className={focusedGridClass(videoFiles.length > 1)}
            style={{ minHeight: 0, gridTemplateRows: 'minmax(0, 1fr)' }}
          >
            {videoFocusIndex !== null && (
              <button
                type="button"
                className={FOCUSED_BUTTON_CLASS}
                onClick={() => setExpandedVideoIndex(null)}
                title={focusedVideoName}
              >
                {renderCameraLabel(focusedVideoName)}
                <video
                  ref={(el) => setVideoElement(videoFocusIndex, el)}
                  src={videoBlobUrls[videoFocusIndex] || ''}
                  className={VIDEO_SURFACE_CLASS}
                  playsInline
                  muted={videoFocusIndex > 0}
                />
              </button>
            )}

            {videoFiles.length > 1 && (
              <div className="grid h-full min-h-0 min-w-0 gap-2 overflow-y-auto pr-0.5" style={{ gridAutoRows: 'minmax(92px, 1fr)' }}>
                {videoFiles.map((file, index) => {
                  if (index === videoFocusIndex) return null;
                  const name = videoNames[index] || getShortVideoName(file);
                  return (
                    <button
                      key={`${file}-${index}`}
                      type="button"
                      className={THUMB_BUTTON_CLASS}
                      onClick={() => setExpandedVideoIndex(index)}
                      title={name}
                    >
                      {renderCameraLabel(name, 'text-[10px] px-1.5 py-0.5')}
                      <video
                        ref={(el) => setVideoElement(index, el)}
                        src={videoBlobUrls[index] || ''}
                        className={VIDEO_SURFACE_CLASS}
                        playsInline
                        muted={index > 0}
                      />
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )
      )}

      {/* Download overlay */}
      {isDownloading && (
        <div className="absolute inset-0 bg-white bg-opacity-90 flex flex-col items-center justify-center z-20">
          <MdRefresh className="animate-spin text-blue-500 mb-4" size={48} />
          <div className="text-gray-700 mb-4">
            Downloading videos... {downloadProgress}%
          </div>
          <div className="w-48 h-2 bg-gray-300 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${downloadProgress}%` }}
            />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            {Math.floor(downloadProgress / (100 / videoFiles.length))} / {videoFiles.length} videos
          </div>
        </div>
      )}
    </div>
  );
}

export default React.memo(CameraPanel);
