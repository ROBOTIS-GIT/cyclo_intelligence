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

import "./RecordControlPanel.css";

// Capture remains mounted independently of map editing and movement controls.
export default function RecordControlPanel({ capture, footer, error }) {
  return <section className="record-control-panel" aria-label="Record control panel">
    <header className="record-control-header">
      <strong className="record-control-title">Capture</strong>
    </header>
    <div className="record-control-content">{capture}</div>
    {error && <p role="alert" className="record-control-error">{error}</p>}
    <footer className="record-control-footer">{footer}</footer>
  </section>;
}
