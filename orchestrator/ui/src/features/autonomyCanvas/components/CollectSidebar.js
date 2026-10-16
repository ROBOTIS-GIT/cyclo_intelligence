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

import { useEffect, useState } from "react";
import useCollectRecording from "../hooks/useCollectRecording";
import useCollectAnnotation from "../hooks/useCollectAnnotation";
import RecordReasoningPanel from "./RecordReasoningPanel";
import CollectEpisodeLibrary from "./CollectEpisodeLibrary";
import CollectDatasetPanel from "./CollectDatasetPanel";
import CollectCaptureControls from "./CollectCaptureControls";
import { CollectCaptureContext } from "./CollectCaptureContext";
import "./OperateSidebar.css";
import "./CollectSidebar.css";
import "./RecordWorkspace.css";
import "./AnnotateWorkspace.css";

export default function CollectSidebar({
  draft, observation = {}, mapName, mapLoaded, observationDisabled, recordMission = "", recordObservationSpotId = null, captureDisabled = false,
  mode = "live", onEditorBusyChange, onTestModel, mapCanvas,
  datasetSelection, onDatasetSelectionChange, recordSession,
}) {
  const [maskDirty, setMaskDirty] = useState(false);
  const [datasetBusy, setDatasetBusy] = useState(false);
  const recording = useCollectRecording({ draft, session: recordSession });
  const editor = useCollectAnnotation({ draft, mapName: mode === "dataset" ? null : mapName });
  const { busy, saving, maskBusy, labelDirty } = editor;
  // Keep navigation locked across the entire capture session, not individual frame writes.
  const locked = !!draft.recording || busy || saving || maskBusy || maskDirty || labelDirty || datasetBusy || recording.captureTransition;
  useEffect(() => {
    onEditorBusyChange?.(locked);
    return () => onEditorBusyChange?.(false);
  }, [locked, onEditorBusyChange]);
  return <div className={`operate-sidebar collect-sidebar collect-workspace record-workspace record-workspace-${mode}${mode === "dataset" ? " annotate-workspace" : ""}`}
    role="region" aria-label={mode === "live" ? "Autonomy collection panels" : "Autonomy training panels"}>
    <div className="record-workspace-body">
      <div className="record-map-pane" hidden={mode !== "live"}>
        <CollectCaptureContext.Provider value={{
          status: {capturing: draft.recording, frames: draft.episode?.observations?.length || 0},
          controls: <CollectCaptureControls
          {...{ draft, recording, mapName, mapLoaded, recordMission, recordObservationSpotId, recordSession }} disabled={captureDisabled || observationDisabled || saving || maskBusy} /> }}>
          {mode === "live" && <RecordReasoningPanel observation={observation} disabled={observationDisabled}/>}
          <div className="record-map-canvas">{mapCanvas}</div>
        </CollectCaptureContext.Provider>
      </div>
      {mode !== "live" && <>
        <CollectEpisodeLibrary draft={draft} disabled={locked} combined onDelete={recording.discardCapture} />
        <CollectDatasetPanel {...{draft, editor, maskDirty, setMaskDirty, onTestModel}}
          selection={datasetSelection} onSelectionChange={onDatasetSelectionChange} onBusyChange={setDatasetBusy} />
      </>}
    </div>
  </div>;
}
