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

import { useState } from "react";
import LiveHeadCamera from "./LiveHeadCamera";

// Mounted only in the active workspace so Record/Reasoning never duplicate the stream.
export default function CollectHeadCamera({ topic, className = "" }) {
  const [aspect, setAspect] = useState(16 / 9);
  return <div className={`operate-observation collect-camera ${className}`}
    style={{ "--collect-camera-aspect": aspect }}>
    <LiveHeadCamera topic={topic} onAspectRatio={setAspect} />
  </div>;
}
