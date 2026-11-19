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

import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import useDocumentVisible from '../hooks/useDocumentVisible';
import { MdAdd, MdClose, MdDragIndicator, MdRotate90DegreesCw, MdOpenInFull } from 'react-icons/md';
import { useSelector } from 'react-redux';
import { CYCLO_WEB_VIDEO_SERVER_PORT } from '../config/runtimeConfig';
import './ImageGrid.css';

const normalizeRotationDegrees = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return ((Math.round(numeric) % 360) + 360) % 360;
};

export default function ImageGridCell({
  topic,
  label,
  rotationDegrees = 0,
  onRotateClick,
  idx,
  onClose,
  onPlusClick,
  isActive = true,
  style = {},
  onImageSize,
  dragHandleProps,
  resizeHandleProps,
  dropTarget = false,
}) {
  const rotation = normalizeRotationDegrees(rotationDegrees);
  const rosHost = useSelector((state) => state.ros.rosHost);
  const containerRef = useRef(null);
  const documentVisible = useDocumentVisible();
  const [inViewport, setInViewport] = useState(() => typeof IntersectionObserver === 'undefined');
  const streamActive = isActive && documentVisible && inViewport;
  const imageRef = useRef(null);
  const fitImageRef = useRef(() => {});
  const stopStreamRef = useRef(() => {});
  const hasTopic = Boolean(topic && topic.trim());
  const imageSizeCallback = useRef(onImageSize);
  imageSizeCallback.current = onImageSize;

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') return undefined;
    const observer = new IntersectionObserver(([entry]) => {
      setInViewport(entry.isIntersecting);
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Geometry only: neither rotation nor ResizeObserver can restart the stream.
  useLayoutEffect(() => {
    const container = containerRef.current;
    const fitImage = () => {
      const img = imageRef.current;
      if (!img) return;
      const { width, height } = container.getBoundingClientRect();
      const naturalWidth = img.naturalWidth;
      const naturalHeight = img.naturalHeight;
      img.style.transform = `translate(-50%, -50%) rotate(${rotation}deg)`;
      if (!naturalWidth || !naturalHeight || !width || !height) return;
      // Fit the rotated natural bounds, not an assumed camera aspect ratio.
      const radians = rotation * Math.PI / 180;
      const cos = Math.abs(Math.cos(radians));
      const sin = Math.abs(Math.sin(radians));
      const scale = Math.min(
        width / (naturalWidth * cos + naturalHeight * sin),
        height / (naturalWidth * sin + naturalHeight * cos)
      );
      img.style.width = `${naturalWidth * scale}px`;
      img.style.height = `${naturalHeight * scale}px`;
    };
    fitImageRef.current = fitImage;
    fitImage();
    const observer = new ResizeObserver(fitImage);
    observer.observe(container);
    return () => observer.disconnect();
  }, [rotation]);

  useEffect(() => {
    if (!hasTopic || !streamActive) return;
    const container = containerRef.current;
    let cancelled = false;
    let retryTimer;
    let retries = 0;

    const removeImage = () => {
      const img = imageRef.current;
      if (!img) return;
      img.onerror = null;
      img.onload = null;
      img.src = '';
      img.remove();
      imageRef.current = null;
    };
    const stopStream = () => {
      cancelled = true;
      clearTimeout(retryTimer);
      removeImage();
    };
    stopStreamRef.current = stopStream;

    const createImage = () => {
      if (cancelled) return;
      removeImage();
      const img = document.createElement('img');
      img.alt = topic;
      img.className = 'image-grid-stream';
      img.style.objectFit = 'contain';
      img.onload = () => {
        if (cancelled || imageRef.current !== img) return;
        retries = 0;
        clearTimeout(retryTimer);
        retryTimer = undefined;
        imageSizeCallback.current?.(topic, img.naturalWidth, img.naturalHeight);
        fitImageRef.current();
      };
      img.onerror = () => {
        if (cancelled || imageRef.current !== img || retryTimer !== undefined || retries >= 5) return;
        const delay = Math.min(1000 * (2 ** retries), 8000);
        retries += 1;
        retryTimer = setTimeout(() => {
          retryTimer = undefined;
          createImage();
        }, delay);
      };
      imageRef.current = img;
      container.appendChild(img);
      fitImageRef.current();
      // web_video_server requires literal slashes and the base image topic.
      const streamTopic = topic.endsWith('/compressed') ? topic.slice(0, -11) : topic;
      img.src = `http://${rosHost}:${CYCLO_WEB_VIDEO_SERVER_PORT}/stream?quality=50&type=ros_compressed&default_transport=compressed&topic=${streamTopic}&t=${Date.now()}`;
    };
    createImage();
    return stopStream;
  }, [topic, hasTopic, streamActive, rosHost]);

  return (
    <div className={`image-grid-cell${dropTarget ? ' image-grid-drop-target' : ''}${resizeHandleProps ? ' image-grid-resizable' : ''}`} data-cell-idx={idx} data-camera-topic={topic} style={style}>
      <div className="image-grid-toolbar">
        {dragHandleProps ? (
          <button type="button" className="image-grid-drag-handle" aria-label={`Move camera ${idx + 1}`}
            title={`${topic}\nDrag to reorder; arrow keys move the camera`} {...dragHandleProps}>
            <MdDragIndicator size={18} className="shrink-0" />
            <span className="image-grid-topic">{label || topic}</span>
          </button>
        ) : <span className="image-grid-topic" title={topic || ''}>{label || topic || ''}</span>}
        {hasTopic && (
          <>
            <button
              type="button"
              className="image-grid-tool"
              aria-label={`Rotate camera ${idx + 1}`}
              title={`Rotate 90 degrees clockwise (current: ${rotation} degrees)`}
              onClick={() => onRotateClick?.(idx)}
            >
              <MdRotate90DegreesCw size={20} />
            </button>
            <button
              type="button"
              className="image-grid-tool"
              aria-label={`Close camera ${idx + 1}`}
              title="Close camera"
              onClick={() => { stopStreamRef.current(); onClose(idx); }}
            >
              <MdClose size={20} />
            </button>
          </>
        )}
      </div>
      <div ref={containerRef} className="image-grid-viewport">
        {!hasTopic && (
          <button
            type="button"
            className="image-grid-select"
            aria-label={`Select camera ${idx + 1}`}
            title="Select camera"
            onClick={() => onPlusClick(idx)}
          >
            <MdAdd size={36} />
          </button>
        )}
      </div>
      {hasTopic && resizeHandleProps && (
        <button type="button" className="image-grid-resize-handle"
          aria-label={`Resize camera ${idx + 1}`} title="Resize camera; arrow keys adjust size"
          {...resizeHandleProps}><MdOpenInFull size={17} /></button>
      )}
    </div>
  );
}
