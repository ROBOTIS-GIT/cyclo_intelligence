import React from 'react';
import { MdRefresh } from 'react-icons/md';
import { WebGLVideoPanel } from './WebGLVideoPanel';

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
  videoRefs,
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
          <div className="grid grid-cols-1 gap-2 w-full h-full p-2" style={{ gridAutoRows: 'minmax(0, 1fr)' }}>
            {mcapPlayer.imageTopics.map((topicInfo, index) => (
              <div
                key={topicInfo.topic}
                className="relative bg-white rounded-lg overflow-hidden shadow flex flex-col"
                style={{ minHeight: 0 }}
              >
                <div className="absolute top-1 left-1 z-10 px-1.5 py-0.5 bg-black bg-opacity-60 text-white text-[10px] rounded font-medium">
                  {topicInfo.cameraName}
                </div>
                {useWebGL ? (
                  <WebGLVideoPanel
                    ref={(handle) => mcapPlayer.setWebGLPanelRef(index, handle)}
                    brightness={videoBrightness}
                    contrast={videoContrast}
                    className="w-full h-full object-contain bg-gray-900"
                  />
                ) : (
                  <canvas
                    ref={(el) => mcapPlayer.setCanvasRef(index, el)}
                    className="w-full h-full object-contain bg-gray-900"
                    style={{ imageRendering: 'auto' }}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      ) : expandedVideoIndex !== null ? (
        /* Expanded single video view */
        <div
          className="flex-1 relative bg-white rounded-lg overflow-hidden shadow cursor-pointer flex items-center justify-center min-h-0"
          onClick={() => setExpandedVideoIndex(null)}
          title="Click to collapse"
        >
          <div className="absolute top-2 left-2 z-10 px-2 py-1 bg-black bg-opacity-60 text-white text-sm rounded font-medium">
            {videoNames[expandedVideoIndex] || getShortVideoName(videoFiles[expandedVideoIndex])}
          </div>
          <div className="absolute top-2 right-2 z-10 px-2 py-1 bg-black bg-opacity-60 text-white text-xs rounded">
            Click to collapse
          </div>
          <video
            ref={(el) => (videoRefs.current[expandedVideoIndex] = el)}
            src={videoBlobUrls[expandedVideoIndex] || ''}
            className="max-w-full max-h-full object-contain"
            style={{ width: 'auto', height: '100%' }}
            playsInline
            muted={expandedVideoIndex > 0}
          />
        </div>
      ) : (
        /* Grid view — vertical strip for side panel */
        <div className="flex-1 grid grid-cols-1 gap-2 p-2 overflow-y-auto" style={{ minHeight: 0, gridAutoRows: 'minmax(0, 1fr)' }}>
          {videoFiles.map((file, index) => (
            <div
              key={index}
              className="relative bg-white rounded-lg overflow-hidden shadow flex flex-col cursor-pointer hover:ring-2 hover:ring-blue-400 transition-all min-h-0"
              onClick={() => setExpandedVideoIndex(index)}
              title="Click to expand"
            >
              <div className="absolute top-1 left-1 z-10 px-1.5 py-0.5 bg-black bg-opacity-60 text-white text-[10px] rounded font-medium">
                {videoNames[index] || getShortVideoName(file)}
              </div>
              <video
                ref={(el) => (videoRefs.current[index] = el)}
                src={videoBlobUrls[index] || ''}
                className="w-full h-full object-contain bg-gray-50"
                playsInline
                muted={index > 0}
              />
            </div>
          ))}
        </div>
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
