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

import React, { useCallback, useEffect, useRef } from 'react';
import clsx from 'clsx';
import { MdClose, MdScreenRotation } from 'react-icons/md';
import { useSelector } from 'react-redux';
import { CYCLO_WEB_VIDEO_SERVER_PORT } from '../config/runtimeConfig';

const normalizeRotationDegrees = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return ((Math.round(numeric) % 360) + 360) % 360;
};

const parseAspectRatio = (value) => {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return value;
  }
  const [width, height] = String(value || '').split('/').map(Number);
  if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
    return width / height;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : 1;
};

const formatPercent = (value) => `${Number(value.toFixed(4))}%`;

const classCell = (topic) =>
  clsx(
    'relative',
    'bg-gray-100',
    'rounded-3xl',
    'overflow-hidden',
    'flex',
    'items-center',
    'justify-center',
    'transition-all',
    'duration-300',
    'w-full',
    {
      'border-2 border-dashed border-gray-300 hover:border-gray-400': !topic,
      'bg-white': topic,
    }
  );

const classCloseBtn = clsx(
  'absolute', 'top-2', 'right-2',
  'w-8', 'h-8',
  'bg-black', 'bg-opacity-50', 'text-white',
  'rounded-full', 'flex', 'items-center', 'justify-center',
  'hover:bg-opacity-70', 'z-20'
);

const classRotateBtn = clsx(
  'absolute', 'top-2', 'left-2',
  'w-8', 'h-8',
  'bg-black', 'bg-opacity-50', 'text-white',
  'rounded-full', 'flex', 'items-center', 'justify-center',
  'hover:bg-opacity-70', 'z-20'
);

export default function ImageGridCell({
  topic,
  aspect,
  rotationDegrees = 0,
  onRotateClick,
  idx,
  onClose,
  onPlusClick,
  isActive = true,
  style = {},
}) {
  const normalizedRotationDegrees = normalizeRotationDegrees(rotationDegrees);
  const rotate = normalizedRotationDegrees !== 0;
  const swapsDimensions = normalizedRotationDegrees === 90 || normalizedRotationDegrees === 270;
  const aspectRatio = parseAspectRatio(aspect);
  const rosHost = useSelector((state) => state.ros.rosHost);
  const containerRef = useRef(null);
  const currentImgRef = useRef(null);
  const isCreatingRef = useRef(false);
  const cancelRef = useRef(false);
  const retryTimerRef = useRef(null);
  const retryCountRef = useRef(0);
  const MAX_RETRIES = 5;

  const destroyImage = useCallback(() => {
    // Signal any in-flight createImage waiting on the staggered delay to bail.
    cancelRef.current = true;
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    if (currentImgRef.current) {
      const el = currentImgRef.current;
      const img = el.tagName === 'IMG' ? el : el.querySelector('img');
      if (img) {
        // Detach handlers BEFORE clearing src so the late-fired onerror that
        // src='' triggers can't schedule a retry that revives the stream.
        img.onerror = null;
        img.onload = null;
        img.src = '';
      }
      if (el.parentNode) el.parentNode.removeChild(el);
      currentImgRef.current = null;
    }
  }, []);

  const createImage = useCallback(async () => {
    if (!topic || !topic.trim() || !isActive || !containerRef.current) return;
    if (isCreatingRef.current) return;

    isCreatingRef.current = true;
    destroyImage();
    // Clear cancel flag raised by destroyImage; later destroys will set it again.
    cancelRef.current = false;

    try {
      const staggeredDelay = (idx === 0 || idx === 2) ? 300 : 0;
      if (staggeredDelay > 0) {
        await new Promise((resolve) => setTimeout(resolve, staggeredDelay));
      }

      if (cancelRef.current || !topic || !topic.trim() || !isActive || !containerRef.current) return;

      const img = document.createElement('img');
      const timestamp = Date.now();
      // web_video_server expects base topic (e.g. .../image_raw); use default_transport=compressed to subscribe to CompressedImage
      // Do not encode slashes: server rejects %2F and expects literal /
      const streamTopic = topic.endsWith('/compressed') ? topic.slice(0, -11) : topic;
      img.src = `http://${rosHost}:${CYCLO_WEB_VIDEO_SERVER_PORT}/stream?quality=50&type=ros_compressed&default_transport=compressed&topic=${streamTopic}&t=${timestamp}`;
      img.alt = topic;

      img.onclick = (e) => e.stopPropagation();
      img.onerror = () => {
        // Late-fired error after destroyImage (src='' triggers onerror) must
        // not schedule a retry — cancelRef tells us we're already torn down.
        if (cancelRef.current) return;
        if (retryCountRef.current >= MAX_RETRIES) {
          console.error(`Image stream failed after ${MAX_RETRIES} retries for idx ${idx}, topic: ${topic}`);
          return;
        }
        retryCountRef.current += 1;
        const delay = Math.min(1000 * Math.pow(2, retryCountRef.current - 1), 8000);
        console.warn(`Image stream error for idx ${idx}, retrying in ${delay}ms (${retryCountRef.current}/${MAX_RETRIES})`);
        retryTimerRef.current = setTimeout(() => {
          if (cancelRef.current) return;
          if (isActive && topic && containerRef.current) {
            destroyImage();
            isCreatingRef.current = false;
            createImage();
          }
        }, delay);
      };
      img.onload = () => {
        retryCountRef.current = 0;
      };

      if (rotate) {
        const wrapper = document.createElement('div');
        wrapper.style.position = 'absolute';
        wrapper.style.width = swapsDimensions ? formatPercent(100 / aspectRatio) : '100%';
        wrapper.style.height = swapsDimensions ? formatPercent(100 * aspectRatio) : '100%';
        wrapper.style.top = '50%';
        wrapper.style.left = '50%';
        wrapper.style.transform = `translate(-50%, -50%) rotate(${normalizedRotationDegrees}deg)`;
        wrapper.style.transformOrigin = 'center center';
        wrapper.style.overflow = 'hidden';

        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        img.style.display = 'block';

        wrapper.appendChild(img);

        if (containerRef.current && !cancelRef.current) {
          containerRef.current.appendChild(wrapper);
          currentImgRef.current = wrapper;
        }
      } else {
        img.className = 'w-full h-full object-cover bg-gray-100';

        if (containerRef.current && !cancelRef.current) {
          containerRef.current.appendChild(img);
          currentImgRef.current = img;
        }
      }
    } finally {
      isCreatingRef.current = false;
    }
  }, [
    topic,
    isActive,
    rosHost,
    idx,
    rotate,
    swapsDimensions,
    aspectRatio,
    normalizedRotationDegrees,
    destroyImage,
  ]);

  useEffect(() => {
    retryCountRef.current = 0;
    if (topic && topic.trim() !== '' && isActive) {
      createImage().catch((error) => {
        console.error(`Error creating image stream for idx ${idx}:`, error);
        isCreatingRef.current = false;
      });
    } else {
      destroyImage();
    }

    return () => {
      isCreatingRef.current = false;
      retryCountRef.current = 0;
      destroyImage();
    };
  }, [topic, isActive, rosHost, idx, createImage, destroyImage]);

  useEffect(() => {
    return () => { destroyImage(); };
  }, [idx, destroyImage]);

  const handleClose = (e) => {
    e.stopPropagation();
    destroyImage();
    onClose(idx);
  };

  return (
    <div
      className={classCell(topic)}
      onClick={!topic ? () => onPlusClick(idx) : undefined}
      style={{ cursor: !topic ? 'pointer' : 'default', aspectRatio: aspect, ...style }}
    >
      {topic && topic.trim() !== '' && (
        <>
          <button
            type="button"
            className={classRotateBtn}
            onClick={(e) => { e.stopPropagation(); onRotateClick?.(idx); }}
            title={rotate ? 'View horizontally' : 'View vertically'}
          >
            <MdScreenRotation size={20} />
          </button>
          <button type="button" className={classCloseBtn} onClick={handleClose}>
            <MdClose size={20} />
          </button>
        </>
      )}
      <div ref={containerRef} className="w-full h-full relative overflow-hidden rounded-3xl flex items-center justify-center">
        {(!topic || !isActive) && <div className="text-6xl text-gray-400 font-light">+</div>}
      </div>
    </div>
  );
}
