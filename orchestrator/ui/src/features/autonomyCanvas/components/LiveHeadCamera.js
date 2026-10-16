// Copyright 2026 ROBOTIS CO., LTD.
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
// Author: Seongwoo Kim

import { useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import { CYCLO_WEB_VIDEO_SERVER_PORT } from "../../../config/runtimeConfig";

// One compressed stream, independent of Qwen's sequential snapshot requests.
export default function LiveHeadCamera({ topic = "/zed/zed_node/left/image_rect_color/compressed", onAspectRatio, onReady, showStatus = true }) {
  const host = useSelector(state => state.ros.rosHost) || window.location.hostname;
  const container = useRef(null);
  const [attempt, setAttempt] = useState(0);
  const [status, setStatus] = useState("Connecting camera…");
  useEffect(() => {
    let img = null;
    let timer = null;
    let retries = 0;
    let disposed = false;
    const disconnect = () => {
      clearTimeout(timer);
      if (img) {
        img.onload = null; img.onerror = null;
        img.removeAttribute("src"); img.remove(); img = null;
      }
    };
    const connect = () => {
      disconnect();
      if (disposed || document.hidden) return;
      setStatus("Connecting camera…");
      onReady?.(false);
      img = document.createElement("img");
      img.alt = "Live head camera";
      img.className = "operate-head-image";
      img.onload = () => {
        if (disposed) return;
        setStatus("");
        onReady?.(true);
        if (img?.naturalWidth > 0 && img?.naturalHeight > 0) {
          onAspectRatio?.(img.naturalWidth / img.naturalHeight);
        }
      };
      img.onerror = () => {
        disconnect();
        if (disposed) return;
        onReady?.(false);
        if (retries >= 3) { setStatus("Camera unavailable"); return; }
        setStatus("Reconnecting camera…");
        timer = setTimeout(connect, 1000 * 2 ** retries++);
      };
      const baseTopic = topic.replace(/\/compressed$/, "");
      const query = `type=ros_compressed&default_transport=compressed&topic=${encodeURIComponent(baseTopic).replace(/%2F/g, "/")}`;
      img.src = `http://${host}:${CYCLO_WEB_VIDEO_SERVER_PORT}/stream?${query}&t=${Date.now()}`;
      container.current?.appendChild(img);
    };
    const visibility = () => {
      if (document.hidden) { disconnect(); onReady?.(false); setStatus("Camera paused while hidden"); }
      else { retries = 0; connect(); }
    };
    connect();
    document.addEventListener("visibilitychange", visibility);
    return () => { disposed = true; disconnect(); document.removeEventListener("visibilitychange", visibility); };
  }, [host, topic, attempt, onAspectRatio, onReady]);
  return <div className="collect-live-camera">
    <div ref={container} className="collect-live-image" />
    {showStatus && status && <div className="collect-live-status" role="status"><span>{status}</span>{status === "Camera unavailable" && <button type="button" onClick={() => setAttempt(n => n + 1)}>Reconnect camera</button>}</div>}
  </div>;
}
